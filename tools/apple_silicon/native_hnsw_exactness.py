#!/usr/bin/env python3
"""Execute and validate the Apple HNSW campaign-parameter exactness gate."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import signal
import stat
import struct
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from tools.apple_silicon import native_hnsw_d0 as d0

EXACTNESS_SCHEMA_VERSION = 3
EXACTNESS_PROCESS_EVIDENCE_SCHEMA_VERSION = 2
EXACTNESS_RUN_ORDER = (
    ("control", 0),
    ("control", 1),
    ("treatment", 0),
    ("treatment", 1),
)
DATASET_BYTES = 6_291_456
DATASET_SHA256 = "f2e29c0f1a64d48a81e2adba0213c53d3015f459ef9936668a7dabb6a025f32a"
VECTOR_COUNT = 12_288
DIMENSION = 128
M = 32
EF_CONSTRUCTION = 200
CHUNK_SIZE = 8_192
MAX_CHUNKS = 2
WORKERS = 1
GRAPH_MAGIC = b"IFHXGR01"
GRAPH_SCHEMA_VERSION = 1
GRAPH_HEADER_BYTES = 184
WITNESS_MAGIC = b"IFHXWT01"
WITNESS_SCHEMA_VERSION = 2
WITNESS_BYTES = 56
WITNESS_FIELD_NAMES = (
    "incremental_treatment_compiled",
    "incremental_capture_armed",
    "incremental_eligible_branch_entered",
    "incremental_successful_unchanged_observed",
    "incremental_successful_updated_observed",
    "threshold_treatment_compiled",
    "threshold_capture_armed",
    "threshold_eligible_branch_entered",
    "threshold_rejected_lane_observed",
    "threshold_surviving_lane_observed",
)
LEVEL_ZERO_STRIDE = 280
UPPER_LAYER_STRIDE = 132
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
CHILD_DATASET_FD = 10
CHILD_GRAPH_FD = 11
CHILD_SAVE_TO_PTR_FD = 12
CHILD_WITNESS_FD = 13
POSIX_SPAWN_START_SUSPENDED = 0x0080
POSIX_SPAWN_SETSID = 0x0400
POSIX_SPAWN_CLOEXEC_DEFAULT = 0x4000
SPAWN_FLAGS = (
    POSIX_SPAWN_START_SUSPENDED
    | POSIX_SPAWN_SETSID
    | POSIX_SPAWN_CLOEXEC_DEFAULT
)
POLL_SECONDS = 0.01
POST_KILL_WAIT_SECONDS = 5.0
P_PID = 1
WNOHANG = 0x00000001
WEXITED = 0x00000004
WSTOPPED = 0x00000008
WNOWAIT = 0x00000020
CLD_EXITED = 1
CLD_KILLED = 2
CLD_DUMPED = 3
CLD_TRAPPED = 4
CLD_STOPPED = 5
CLD_CONTINUED = 6
TERMINAL_CHILD_CODES = frozenset((CLD_EXITED, CLD_KILLED, CLD_DUMPED))
_SPAWN_LOCK = threading.Lock()


class ExactnessFailure(RuntimeError):
    """Raised when exactness execution or validation fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExactnessFailure(message)


def _validate_expected_witnesses(
    value: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    require(
        isinstance(value, Mapping),
        "expected_witnesses must be an object",
    )
    require(
        all(type(role) is str for role in value),
        "expected_witnesses contains a non-string role",
    )
    require(
        set(value) == {"control", "treatment"},
        "expected_witnesses must contain exactly control and treatment",
    )
    expected_fields = set(WITNESS_FIELD_NAMES)
    normalized: dict[str, dict[str, int]] = {}
    for role in ("control", "treatment"):
        witness = value[role]
        require(
            isinstance(witness, Mapping),
            f"expected_witnesses {role} must be an object",
        )
        require(
            all(type(field) is str for field in witness),
            f"expected_witnesses {role} contains a non-string field",
        )
        require(
            set(witness) == expected_fields,
            f"expected_witnesses {role} fields differ",
        )
        normalized[role] = {}
        for field in WITNESS_FIELD_NAMES:
            bit = witness[field]
            require(
                type(bit) is int and bit in (0, 1),
                f"expected_witnesses {role} {field} must be integer 0 or 1",
            )
            normalized[role][field] = bit
    return normalized


class _ChildOwnership(str, Enum):
    UNSTARTED = "unstarted"
    SPAWN_INDETERMINATE = "spawn-indeterminate"
    NO_CHILD = "no-child"
    OWNED = "owned"
    UNRESOLVED = "unresolved"


class _ChildLifecycle(str, Enum):
    UNOBSERVED = "unobserved"
    STOPPED = "stopped"
    RUNNING = "running"
    TERMINAL_OBSERVED = "terminal-observed"
    REAP_IN_PROGRESS = "reap-in-progress"
    REAPED = "reaped"


class _ChildGroupState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    KILL_SENT = "kill-sent"
    DIRECT_KILL_SENT = "direct-kill-sent"
    QUIESCENT = "quiescent"
    EMPTY = "empty"
    UNRESOLVED = "unresolved"


class _Sigval(ctypes.Union):
    _fields_ = [
        ("sival_int", ctypes.c_int),
        ("sival_ptr", ctypes.c_void_p),
    ]


class _DarwinSiginfo(ctypes.Structure):
    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", _Sigval),
        ("si_band", ctypes.c_long),
        ("_reserved", ctypes.c_ulong * 7),
    ]


@dataclass(frozen=True)
class _WaitObservation:
    pid: int
    code: int
    status: int


@dataclass(frozen=True)
class _CleanupOutcome:
    errors: tuple[str, ...]
    interruption: Optional[BaseException]


@dataclass
class _SpawnedChildState:
    pid_cell: Any
    expected_executable: Optional[dict[str, Any]] = None
    preexisting_child_pids: Optional[tuple[int, ...]] = None
    ownership: _ChildOwnership = _ChildOwnership.UNSTARTED
    spawn_result: Optional[int] = None
    pid: int = -1
    lifecycle: _ChildLifecycle = _ChildLifecycle.UNOBSERVED
    terminal_code: Optional[int] = None
    terminal_status: Optional[int] = None
    terminal_observed_at_monotonic_ns: Optional[int] = None
    group_state: _ChildGroupState = _ChildGroupState.UNVERIFIED
    pre_reap_member_pids: Optional[list[int]] = None
    pre_reap_observed_at_monotonic_ns: Optional[int] = None
    resume_attempted: bool = False
    cleanup_deadline: Optional[float] = None
    reap_status_validated: bool = False
    reap_raw_status: Optional[int] = None
    returncode: Optional[int] = None
    reap_completed_at_monotonic_ns: Optional[int] = None


@dataclass(frozen=True)
class _Graph:
    summary: dict[str, Any]
    levels: tuple[int, ...]
    adjacency: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class _ParsedRun:
    stdout: bytes
    graph: bytes
    save_to_ptr: bytes
    witness: bytes
    verification: dict[str, Any]


@dataclass(frozen=True)
class _CompletedRun:
    record: dict[str, Any]
    parsed: _ParsedRun


class _Reader:
    def __init__(self, data: bytes, label: str) -> None:
        self.data = data
        self.label = label
        self.offset = 0

    def read(self, size: int, field: str) -> bytes:
        require(size >= 0, f"{self.label} {field} has a negative size")
        end = self.offset + size
        require(
            end <= len(self.data),
            f"{self.label} ended while reading {field}",
        )
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def unpack(self, format_text: str, field: str) -> tuple[Any, ...]:
        layout = struct.Struct("<" + format_text)
        return layout.unpack(self.read(layout.size, field))

    def u32(self, field: str) -> int:
        return int(self.unpack("I", field)[0])

    def i32(self, field: str) -> int:
        return int(self.unpack("i", field)[0])

    def u64(self, field: str) -> int:
        return int(self.unpack("Q", field)[0])

    def finish(self) -> None:
        require(
            self.offset == len(self.data),
            f"{self.label} has {len(self.data) - self.offset} trailing bytes",
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _parse_stdout(data: bytes) -> dict[str, Any]:
    require(data.endswith(b"\n"), "Exactness stdout is not newline terminated")
    require(
        data.count(b"\n") == 1 and b"\r" not in data,
        "Exactness stdout is not one canonical line",
    )
    try:
        text = data.decode("ascii")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExactnessFailure(f"Exactness stdout is not canonical JSON: {error}") from error
    require(type(value) is dict, "Exactness stdout JSON is not an object")
    require(
        _canonical_json_bytes(value) == data,
        "Exactness stdout JSON is not canonically serialized",
    )
    expected_top_level = {
        "build",
        "configuration",
        "dataset",
        "graph",
        "save_to_ptr",
        "schema_version",
        "status",
    }
    require(
        set(value) == expected_top_level,
        "Exactness stdout top-level fields differ",
    )
    require(
        type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["status"] == "PASS",
        "Exactness stdout status or schema differs",
    )
    require(
        value["configuration"]
        == {
            "chunk_size": CHUNK_SIZE,
            "dimension": DIMENSION,
            "ef_construction": EF_CONSTRUCTION,
            "m": M,
            "max_chunks": MAX_CHUNKS,
            "optimize": True,
            "vectors": VECTOR_COUNT,
            "workers": WORKERS,
        },
        "Exactness stdout configuration differs",
    )
    require(
        value["dataset"]
        == {"bytes": DATASET_BYTES, "sha256": DATASET_SHA256},
        "Exactness stdout dataset identity differs",
    )
    require(
        value["build"]
        == {
            "end": VECTOR_COUNT,
            "memory_delta_positive": True,
            "start": 0,
            "submitted_tasks": 1,
        },
        "Exactness stdout build result differs",
    )
    return value


def _parse_witness(
    data: bytes,
    *,
    role: str,
    expected_witness: Mapping[str, int],
) -> dict[str, Any]:
    require(role in {"control", "treatment"}, f"Unknown exactness role: {role}")
    require(len(data) == WITNESS_BYTES, f"{role} witness must be 56 bytes")
    reader = _Reader(data, f"{role} witness")
    require(reader.read(8, "magic") == WITNESS_MAGIC, f"{role} witness magic differs")
    require(
        reader.u32("schema") == WITNESS_SCHEMA_VERSION,
        f"{role} witness schema differs",
    )
    require(
        reader.u32("record bytes") == WITNESS_BYTES,
        f"{role} witness byte count differs",
    )
    fields = {name: reader.u32(name) for name in WITNESS_FIELD_NAMES}
    reader.finish()
    require(
        fields == expected_witness,
        f"{role} witness differs from expected_witnesses",
    )
    return {
        "schema_version": WITNESS_SCHEMA_VERSION,
        "bytes": len(data),
        "sha256": _sha256(data),
        **fields,
    }


def _parse_graph(data: bytes) -> _Graph:
    reader = _Reader(data, "canonical graph")
    require(reader.read(8, "magic") == GRAPH_MAGIC, "Canonical graph magic differs")
    schema = reader.u32("schema")
    header_bytes = reader.u32("header bytes")
    vectors = reader.u64("vector count")
    dimension = reader.u64("dimension")
    m = reader.u64("M")
    ef_construction = reader.u64("efConstruction")
    chunk_size = reader.u64("chunk size")
    max_chunks = reader.u64("maximum chunks")
    workers = reader.u64("workers")
    build_start = reader.u64("build start")
    build_end = reader.u64("build end")
    submitted_tasks = reader.u64("submitted tasks")
    index_vectors = reader.u64("index vector count")
    mmax0 = reader.u64("level-zero capacity")
    mmax = reader.u64("upper-layer capacity")
    max_level = reader.i32("maximum level")
    entry_point = reader.i32("entry point")
    vertex_records = reader.u64("vertex records")
    label_bytes = reader.u64("label bytes")
    dataset_digest = reader.read(32, "dataset digest")
    records_offset = reader.u64("records offset")
    require(schema == GRAPH_SCHEMA_VERSION, "Canonical graph schema differs")
    require(
        header_bytes == GRAPH_HEADER_BYTES
        and records_offset == GRAPH_HEADER_BYTES
        and reader.offset == GRAPH_HEADER_BYTES,
        "Canonical graph header size differs",
    )
    require(
        (
            vectors,
            dimension,
            m,
            ef_construction,
            chunk_size,
            max_chunks,
            workers,
            build_start,
            build_end,
            submitted_tasks,
            index_vectors,
            mmax0,
            mmax,
            vertex_records,
            label_bytes,
        )
        == (
            VECTOR_COUNT,
            DIMENSION,
            M,
            EF_CONSTRUCTION,
            CHUNK_SIZE,
            MAX_CHUNKS,
            WORKERS,
            0,
            VECTOR_COUNT,
            1,
            VECTOR_COUNT,
            2 * M,
            M,
            VECTOR_COUNT,
            1,
        ),
        "Canonical graph metadata differs",
    )
    require(
        dataset_digest.hex() == DATASET_SHA256,
        "Canonical graph dataset digest differs",
    )
    require(
        0 <= max_level <= 63 and 0 <= entry_point < VECTOR_COUNT,
        "Canonical graph entry point metadata is invalid",
    )

    levels: list[int] = []
    adjacency: list[tuple[tuple[int, ...], ...]] = []
    directed_edges = 0
    for ordinal in range(VECTOR_COUNT):
        observed_ordinal = reader.u32(f"vertex {ordinal} ordinal")
        label = reader.i32(f"vertex {ordinal} label")
        level = reader.i32(f"vertex {ordinal} level")
        layer_count = reader.u32(f"vertex {ordinal} layer count")
        require(
            observed_ordinal == ordinal and label == ordinal,
            f"Canonical graph vertex {ordinal} identity differs",
        )
        require(
            0 <= level <= max_level and layer_count == level + 1,
            f"Canonical graph vertex {ordinal} level metadata differs",
        )
        vertex_layers: list[tuple[int, ...]] = []
        for layer in range(layer_count):
            observed_layer = reader.u32(f"vertex {ordinal} layer {layer} id")
            degree = reader.u32(f"vertex {ordinal} layer {layer} degree")
            capacity = mmax0 if layer == 0 else mmax
            require(
                observed_layer == layer and degree <= capacity,
                f"Canonical graph vertex {ordinal} layer {layer} metadata differs",
            )
            neighbors = tuple(
                reader.i32(f"vertex {ordinal} layer {layer} neighbor {index}")
                for index in range(degree)
            )
            require(
                all(
                    0 <= neighbor < VECTOR_COUNT and neighbor != ordinal
                    for neighbor in neighbors
                ),
                f"Canonical graph vertex {ordinal} layer {layer} has an invalid edge",
            )
            require(
                len(set(neighbors)) == len(neighbors),
                f"Canonical graph vertex {ordinal} layer {layer} has duplicate edges",
            )
            vertex_layers.append(neighbors)
            directed_edges += degree
        levels.append(level)
        adjacency.append(tuple(vertex_layers))
    reader.finish()
    require(max(levels) == max_level, "Canonical graph maximum level differs")
    require(
        levels[entry_point] == max_level,
        "Canonical graph entry point is not on the maximum level",
    )
    for vertex, vertex_layers in enumerate(adjacency):
        for layer, neighbors in enumerate(vertex_layers):
            require(
                all(levels[neighbor] >= layer for neighbor in neighbors),
                f"Canonical graph vertex {vertex} layer {layer} targets a lower level",
            )
    return _Graph(
        summary={
            "schema_version": schema,
            "bytes": len(data),
            "sha256": _sha256(data),
            "vectors": vectors,
            "dimension": dimension,
            "m": m,
            "ef_construction": ef_construction,
            "mmax0": mmax0,
            "mmax": mmax,
            "max_level": max_level,
            "entry_point": entry_point,
            "directed_edges": directed_edges,
        },
        levels=tuple(levels),
        adjacency=tuple(adjacency),
    )


def _parse_save_to_ptr(
    data: bytes,
    *,
    dataset: bytes,
    graph: _Graph,
) -> dict[str, Any]:
    reader = _Reader(data, "SaveToPtr image")
    m = reader.u64("M")
    ef_construction = reader.u64("efConstruction")
    vectors = reader.u64("vector count")
    dimension = reader.u64("dimension")
    mmax0 = reader.u64("level-zero capacity")
    mmax = reader.u64("upper-layer capacity")
    max_level = reader.i32("maximum level")
    entry_point = reader.i32("entry point")
    require(
        (m, ef_construction, vectors, dimension, mmax0, mmax)
        == (M, EF_CONSTRUCTION, VECTOR_COUNT, DIMENSION, 2 * M, M),
        "SaveToPtr metadata differs",
    )
    require(
        max_level == graph.summary["max_level"]
        and entry_point == graph.summary["entry_point"],
        "SaveToPtr entry point metadata differs from the canonical graph",
    )
    require(
        reader.read(DATASET_BYTES, "vectors") == dataset,
        "SaveToPtr vectors differ from the frozen dataset",
    )
    upper_layer_count = reader.u64("upper-layer count")
    require(
        upper_layer_count == sum(graph.levels),
        "SaveToPtr upper-layer count differs from the canonical graph",
    )
    level_zero = reader.read(
        VECTOR_COUNT * LEVEL_ZERO_STRIDE,
        "level-zero records",
    )
    upper_layers = reader.read(
        upper_layer_count * UPPER_LAYER_STRIDE,
        "upper-layer records",
    )
    labels = reader.read(VECTOR_COUNT * 4, "labels")
    reader.finish()

    upper_index = 0
    for vertex in range(VECTOR_COUNT):
        base = vertex * LEVEL_ZERO_STRIDE
        level = struct.unpack_from("<i", level_zero, base)[0]
        encoded_offset = struct.unpack_from("<Q", level_zero, base + 8)[0]
        degree = struct.unpack_from("<i", level_zero, base + 16)[0]
        require(
            level == graph.levels[vertex],
            f"SaveToPtr vertex {vertex} level differs",
        )
        expected_offset = upper_index * UPPER_LAYER_STRIDE
        require(
            encoded_offset == (expected_offset if level > 0 else 0),
            f"SaveToPtr vertex {vertex} upper-layer offset differs",
        )
        require(
            0 <= degree <= 2 * M,
            f"SaveToPtr vertex {vertex} level-zero degree is invalid",
        )
        neighbors = tuple(
            struct.unpack_from("<i", level_zero, base + 20 + index * 4)[0]
            for index in range(degree)
        )
        require(
            neighbors == graph.adjacency[vertex][0],
            f"SaveToPtr vertex {vertex} level-zero topology differs",
        )
        for layer in range(1, level + 1):
            upper_base = upper_index * UPPER_LAYER_STRIDE
            upper_degree = struct.unpack_from("<i", upper_layers, upper_base)[0]
            require(
                0 <= upper_degree <= M,
                f"SaveToPtr vertex {vertex} layer {layer} degree is invalid",
            )
            upper_neighbors = tuple(
                struct.unpack_from(
                    "<i",
                    upper_layers,
                    upper_base + 4 + index * 4,
                )[0]
                for index in range(upper_degree)
            )
            require(
                upper_neighbors == graph.adjacency[vertex][layer],
                f"SaveToPtr vertex {vertex} layer {layer} topology differs",
            )
            upper_index += 1
        label = struct.unpack_from("<i", labels, vertex * 4)[0]
        require(label == vertex, f"SaveToPtr vertex {vertex} label differs")
    require(
        upper_index == upper_layer_count,
        "SaveToPtr upper-layer traversal count differs",
    )
    return {
        "bytes": len(data),
        "sha256": _sha256(data),
        "vectors": vectors,
        "dimension": dimension,
        "m": m,
        "ef_construction": ef_construction,
        "mmax0": mmax0,
        "mmax": mmax,
        "max_level": max_level,
        "entry_point": entry_point,
        "upper_layer_count": upper_layer_count,
        "level_zero_stride": LEVEL_ZERO_STRIDE,
        "upper_layer_stride": UPPER_LAYER_STRIDE,
        "dataset_sha256": _sha256(dataset),
        "topology_matches_canonical_graph": True,
        "labels_match_ordinals": True,
        "exact_eof": True,
    }


def parse_exactness_artifacts(
    *,
    role: str,
    expected_witnesses: Mapping[str, Mapping[str, int]],
    stdout: bytes,
    stderr: bytes,
    graph: bytes,
    save_to_ptr: bytes,
    witness: bytes,
    dataset: bytes,
) -> dict[str, Any]:
    """Independently parse one producer run and return its compact proof."""
    normalized_witnesses = _validate_expected_witnesses(expected_witnesses)
    require(stderr == b"", f"{role} exactness emitter wrote to stderr")
    require(
        len(dataset) == DATASET_BYTES and _sha256(dataset) == DATASET_SHA256,
        "Exactness dataset identity differs",
    )
    stdout_value = _parse_stdout(stdout)
    parsed_graph = _parse_graph(graph)
    pointer = _parse_save_to_ptr(
        save_to_ptr,
        dataset=dataset,
        graph=parsed_graph,
    )
    witness_record = _parse_witness(
        witness,
        role=role,
        expected_witness=normalized_witnesses[role],
    )
    require(
        stdout_value["graph"]
        == {
            "bytes": parsed_graph.summary["bytes"],
            "entry_point": parsed_graph.summary["entry_point"],
            "header_bytes": GRAPH_HEADER_BYTES,
            "max_level": parsed_graph.summary["max_level"],
            "mmax": M,
            "mmax0": 2 * M,
            "sha256": parsed_graph.summary["sha256"],
        },
        f"{role} stdout graph record differs from the graph artifact",
    )
    require(
        stdout_value["save_to_ptr"]
        == {
            "bytes": pointer["bytes"],
            "roundtrip_graph_equal": True,
            "sha256": pointer["sha256"],
        },
        f"{role} stdout SaveToPtr record differs from the pointer image",
    )
    return {
        "producer_parser": "tools.apple_silicon.native_hnsw_exactness",
        "stdout": {
            "bytes": len(stdout),
            "sha256": _sha256(stdout),
            "canonical_json": True,
            "value": stdout_value,
        },
        "stderr": {"bytes": 0, "sha256": _sha256(stderr)},
        "graph": parsed_graph.summary,
        "save_to_ptr": pointer,
        "witness": witness_record,
    }


def _descriptor_identity(status: os.stat_result) -> dict[str, int]:
    return {
        "device": int(status.st_dev),
        "inode": int(status.st_ino),
        "mode": int(status.st_mode),
        "links": int(status.st_nlink),
        "bytes": int(status.st_size),
        "mtime_ns": int(status.st_mtime_ns),
        "ctime_ns": int(status.st_ctime_ns),
    }


def _same_stable_identity(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return _descriptor_identity(left) == _descriptor_identity(right)


def _same_directory_identity(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return (
        stat.S_ISDIR(left.st_mode)
        and stat.S_ISDIR(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
    )


def _read_stable_fd(
    descriptor: int,
    *,
    initial: os.stat_result,
    label: str,
) -> bytes:
    before = os.fstat(descriptor)
    require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
        f"{label} is not a single-link regular file",
    )
    require(
        before.st_dev == initial.st_dev and before.st_ino == initial.st_ino,
        f"{label} descriptor was replaced",
    )
    require(
        0 <= before.st_size <= MAX_ARTIFACT_BYTES,
        f"{label} has an invalid byte size",
    )
    parts: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        require(chunk != b"", f"{label} ended before its recorded size")
        parts.append(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    require(
        _same_stable_identity(before, after),
        f"{label} changed while being read",
    )
    return b"".join(parts)


def _artifact_record(
    path: Path,
    descriptor: int,
    data: bytes,
    *,
    directory_descriptor: int,
    initial: os.stat_result,
    label: str,
) -> dict[str, Any]:
    descriptor_status = os.fstat(descriptor)
    path_status = os.stat(
        path.name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    require(
        _same_stable_identity(descriptor_status, path_status),
        f"{label} path differs from its held descriptor",
    )
    require(
        descriptor_status.st_dev == initial.st_dev
        and descriptor_status.st_ino == initial.st_ino,
        f"{label} object identity changed",
    )
    require(
        descriptor_status.st_size == len(data),
        f"{label} size differs from its captured bytes",
    )
    return {
        "captured_path": path.name,
        "sha256": _sha256(data),
        "bytes": len(data),
        "identity": _descriptor_identity(descriptor_status),
    }


def _open_output(
    directory_descriptor: int,
    name: str,
) -> tuple[int, os.stat_result]:
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    status = os.fstat(descriptor)
    require(
        stat.S_ISREG(status.st_mode)
        and status.st_nlink == 1
        and status.st_size == 0,
        f"Fresh exactness output is invalid: {name}",
    )
    return descriptor, status


def _duplicate_high(descriptor: int) -> int:
    duplicate = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 64)
    require(duplicate >= 64, "Could not duplicate an exactness descriptor")
    return duplicate


def _configure_spawn_api() -> ctypes.CDLL:
    require(sys.platform == "darwin", "Exactness spawning requires Darwin")
    require(
        ctypes.sizeof(ctypes.c_int) == 4,
        "Exactness spawning requires a 32-bit Darwin pid_t",
    )
    # Keep the GIL across the ownership-publishing call. Exactness runs are
    # sequential, and this also prevents another Python thread from creating a
    # lookalike direct child in the indeterminate-spawn recovery window.
    library = ctypes.PyDLL(None, use_errno=True)
    pointer = ctypes.POINTER(ctypes.c_void_p)
    library.posix_spawnattr_init.argtypes = [pointer]
    library.posix_spawnattr_init.restype = ctypes.c_int
    library.posix_spawnattr_destroy.argtypes = [pointer]
    library.posix_spawnattr_destroy.restype = ctypes.c_int
    library.posix_spawnattr_setflags.argtypes = [pointer, ctypes.c_short]
    library.posix_spawnattr_setflags.restype = ctypes.c_int
    library.posix_spawnattr_getflags.argtypes = [
        pointer,
        ctypes.POINTER(ctypes.c_short),
    ]
    library.posix_spawnattr_getflags.restype = ctypes.c_int
    library.posix_spawn_file_actions_init.argtypes = [pointer]
    library.posix_spawn_file_actions_init.restype = ctypes.c_int
    library.posix_spawn_file_actions_destroy.argtypes = [pointer]
    library.posix_spawn_file_actions_destroy.restype = ctypes.c_int
    library.posix_spawn_file_actions_adddup2.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.posix_spawn_file_actions_adddup2.restype = ctypes.c_int
    library.posix_spawn_file_actions_addclose.argtypes = [pointer, ctypes.c_int]
    library.posix_spawn_file_actions_addclose.restype = ctypes.c_int
    library.posix_spawn.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.posix_spawn.restype = ctypes.c_int
    return library


def _configure_waitid_api() -> ctypes.CDLL:
    require(sys.platform == "darwin", "Exactness waitid requires Darwin")
    require(
        ctypes.sizeof(ctypes.c_int) == 4
        and ctypes.sizeof(ctypes.c_uint) == 4
        and ctypes.sizeof(ctypes.c_void_p) == 8
        and ctypes.sizeof(ctypes.c_long) == 8
        and ctypes.sizeof(_DarwinSiginfo) == 104
        and _DarwinSiginfo.si_pid.offset == 12
        and _DarwinSiginfo.si_status.offset == 20,
        "Exactness waitid requires the Darwin arm64 siginfo_t ABI",
    )
    library = ctypes.CDLL(None, use_errno=True)
    library.waitid.argtypes = [
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_DarwinSiginfo),
        ctypes.c_int,
    ]
    library.waitid.restype = ctypes.c_int
    return library


def _observe_child_nonconsuming(
    pid: int,
    *,
    include_stopped: bool,
) -> Optional[_WaitObservation]:
    require(pid > 0, "Exactness waitid PID must be positive")
    information = _DarwinSiginfo()
    options = WNOHANG | WNOWAIT | WEXITED
    if include_stopped:
        options |= WSTOPPED
    ctypes.set_errno(0)
    result = _configure_waitid_api().waitid(
        P_PID,
        ctypes.c_uint(pid),
        ctypes.byref(information),
        options,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ECHILD:
            raise ChildProcessError(error_number, os.strerror(error_number))
        if error_number == errno.EINTR:
            raise InterruptedError(error_number, os.strerror(error_number))
        raise OSError(error_number, os.strerror(error_number))
    if information.si_pid == 0:
        return None
    require(
        information.si_signo == signal.SIGCHLD
        and information.si_pid == pid
        and information.si_code
        in {
            CLD_EXITED,
            CLD_KILLED,
            CLD_DUMPED,
            CLD_TRAPPED,
            CLD_STOPPED,
            CLD_CONTINUED,
        },
        "waitid returned malformed exactness child information",
    )
    return _WaitObservation(
        pid=int(information.si_pid),
        code=int(information.si_code),
        status=int(information.si_status),
    )


def _spawn_call(result: int, operation: str) -> None:
    if result != 0:
        raise ExactnessFailure(f"{operation} failed: {os.strerror(result)}")


def _string_vector(values: list[str]) -> Any:
    encoded = [os.fsencode(value) for value in values]
    vector = (ctypes.c_char_p * (len(encoded) + 1))()
    for index, value in enumerate(encoded):
        vector[index] = value
    vector[len(encoded)] = None
    return vector


def _spawn_suspended(
    executable: Path,
    arguments: list[str],
    environment: Mapping[str, str],
    descriptor_map: Mapping[int, int],
    child: _SpawnedChildState,
) -> None:
    require(
        child.pid_cell.value == 0
        and child.ownership == _ChildOwnership.UNSTARTED
        and child.preexisting_child_pids is None
        and child.spawn_result is None
        and child.pid < 0
        and child.lifecycle == _ChildLifecycle.UNOBSERVED
        and child.group_state == _ChildGroupState.UNVERIFIED,
        "Exactness spawn state is not fresh",
    )
    require(
        threading.active_count() == 1,
        "Exactness spawning requires a single-threaded supervisor",
    )
    library = _configure_spawn_api()
    attributes = ctypes.c_void_p()
    actions = ctypes.c_void_p()
    attributes_ready = False
    actions_ready = False
    operation_error: Optional[BaseException] = None
    try:
        _spawn_call(
            library.posix_spawnattr_init(ctypes.byref(attributes)),
            "posix_spawnattr_init",
        )
        attributes_ready = True
        _spawn_call(
            library.posix_spawnattr_setflags(
                ctypes.byref(attributes),
                ctypes.c_short(SPAWN_FLAGS),
            ),
            "posix_spawnattr_setflags",
        )
        observed_flags = ctypes.c_short()
        _spawn_call(
            library.posix_spawnattr_getflags(
                ctypes.byref(attributes),
                ctypes.byref(observed_flags),
            ),
            "posix_spawnattr_getflags",
        )
        require(
            observed_flags.value == SPAWN_FLAGS,
            "posix_spawn attributes did not retain the required flags",
        )
        _spawn_call(
            library.posix_spawn_file_actions_init(ctypes.byref(actions)),
            "posix_spawn_file_actions_init",
        )
        actions_ready = True
        for target, source in sorted(descriptor_map.items()):
            _spawn_call(
                library.posix_spawn_file_actions_adddup2(
                    ctypes.byref(actions),
                    source,
                    target,
                ),
                f"posix_spawn_file_actions_adddup2({target})",
            )
        for source in sorted(set(descriptor_map.values())):
            _spawn_call(
                library.posix_spawn_file_actions_addclose(
                    ctypes.byref(actions),
                    source,
                ),
                f"posix_spawn_file_actions_addclose({source})",
            )
        argv = _string_vector([str(executable), *arguments])
        envp = _string_vector(
            [f"{name}={value}" for name, value in sorted(environment.items())]
        )
        blockable_signals = {
            signal_number
            for signal_number in signal.valid_signals()
            if signal_number not in {signal.SIGKILL, signal.SIGSTOP}
        }
        with _SPAWN_LOCK:
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                blockable_signals,
            )
            try:
                child.preexisting_child_pids = tuple(
                    d0.process_child_pids(os.getpid())
                )
                require(
                    child.preexisting_child_pids == (),
                    "Exactness supervisor already owns child processes",
                )
                child.ownership = _ChildOwnership.SPAWN_INDETERMINATE
                result = library.posix_spawn(
                    ctypes.byref(child.pid_cell),
                    os.fsencode(executable),
                    ctypes.byref(actions),
                    ctypes.byref(attributes),
                    argv,
                    envp,
                )
                child.spawn_result = int(result)
                if child.spawn_result == 0 and child.pid_cell.value > 0:
                    child.pid = int(child.pid_cell.value)
                    child.ownership = _ChildOwnership.OWNED
                elif child.spawn_result != 0:
                    child.ownership = _ChildOwnership.NO_CHILD
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        _spawn_call(child.spawn_result, "posix_spawn")
        require(
            child.pid_cell.value > 0,
            "posix_spawn returned a nonpositive PID",
        )
        require(
            child.ownership == _ChildOwnership.OWNED
            and child.pid == child.pid_cell.value,
            "posix_spawn did not publish exactness child ownership",
        )
    except BaseException as error:
        operation_error = error

    destroy_errors: list[str] = []
    destroy_exceptions: list[BaseException] = []
    try:
        if actions_ready:
            try:
                result = library.posix_spawn_file_actions_destroy(
                    ctypes.byref(actions)
                )
                if result != 0:
                    destroy_errors.append(
                        f"posix_spawn_file_actions_destroy failed: {result}"
                    )
            except BaseException as error:
                destroy_exceptions.append(error)
                destroy_errors.append(
                    "posix_spawn_file_actions_destroy raised "
                    f"{type(error).__name__}: {error}"
                )
    finally:
        if attributes_ready:
            try:
                result = library.posix_spawnattr_destroy(
                    ctypes.byref(attributes)
                )
                if result != 0:
                    destroy_errors.append(
                        f"posix_spawnattr_destroy failed: {result}"
                    )
            except BaseException as error:
                destroy_exceptions.append(error)
                destroy_errors.append(
                    "posix_spawnattr_destroy raised "
                    f"{type(error).__name__}: {error}"
                )

    cleanup = _CleanupOutcome((), None)
    if operation_error is not None or destroy_errors:
        cleanup = _cleanup_child_after_failure(child)

    details = [*destroy_errors, *cleanup.errors]
    if cleanup.interruption is not None:
        details.append(
            "cleanup raised "
            f"{type(cleanup.interruption).__name__}: {cleanup.interruption}"
        )
    if operation_error is not None:
        if details:
            _add_exception_note(
                operation_error,
                "Exactness resource cleanup: " + "; ".join(details),
            )
        raise operation_error
    control_exception = next(
        (
            error
            for error in destroy_exceptions
            if not isinstance(error, Exception)
        ),
        cleanup.interruption,
    )
    if control_exception is not None:
        if details:
            _add_exception_note(
                control_exception,
                "Exactness resource cleanup: " + "; ".join(details),
            )
        raise control_exception
    if details:
        raise ExactnessFailure(
            "posix_spawn resource cleanup failed: " + "; ".join(details)
        )
    require(child.pid > 0, "Exactness spawn did not publish child ownership")


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = list(getattr(error, "_infinity_notes", ()))
    notes.append(note)
    setattr(error, "_infinity_notes", tuple(notes))


def _publish_child_observation(
    child: _SpawnedChildState,
    observation: _WaitObservation,
) -> None:
    require(
        child.ownership == _ChildOwnership.OWNED
        and observation.pid == child.pid
        and child.lifecycle
        not in {_ChildLifecycle.REAP_IN_PROGRESS, _ChildLifecycle.REAPED},
        "Exactness child observation is inconsistent with ownership",
    )
    if observation.code in TERMINAL_CHILD_CODES:
        if child.lifecycle == _ChildLifecycle.TERMINAL_OBSERVED:
            require(
                child.terminal_code == observation.code
                and child.terminal_status == observation.status,
                "Exactness child terminal status changed",
            )
            return
        child.terminal_code = observation.code
        child.terminal_status = observation.status
        child.terminal_observed_at_monotonic_ns = time.monotonic_ns()
        child.lifecycle = _ChildLifecycle.TERMINAL_OBSERVED
        return
    require(
        child.lifecycle != _ChildLifecycle.TERMINAL_OBSERVED,
        "Exactness child changed state after terminal observation",
    )
    if observation.code == CLD_STOPPED:
        child.lifecycle = _ChildLifecycle.STOPPED
        return
    if observation.code == CLD_CONTINUED:
        child.lifecycle = _ChildLifecycle.RUNNING
        return
    raise ExactnessFailure(
        f"Exactness child entered unsupported waitid state {observation.code}"
    )


def _wait_for_stopped(
    child: _SpawnedChildState,
    *,
    deadline: float,
) -> int:
    require(
        child.ownership == _ChildOwnership.OWNED
        and child.pid > 0
        and child.lifecycle
        not in {_ChildLifecycle.REAP_IN_PROGRESS, _ChildLifecycle.REAPED},
        "Exactness child is not live",
    )
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Timed out waiting for exactness child suspension")
        observation = _observe_child_nonconsuming(
            child.pid,
            include_stopped=True,
        )
        if observation is not None:
            _publish_child_observation(child, observation)
            require(
                observation.code == CLD_STOPPED,
                "Exactness child terminated before its suspended attestation",
            )
            return observation.status
        time.sleep(min(POLL_SECONDS, remaining))


def _wait_for_exit(
    child: _SpawnedChildState,
    *,
    deadline: float,
) -> int:
    require(
        child.ownership == _ChildOwnership.OWNED
        and child.pid > 0
        and child.lifecycle
        not in {_ChildLifecycle.REAP_IN_PROGRESS, _ChildLifecycle.REAPED},
        "Exactness child is not live",
    )
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Exactness child timed out")
        observation = _observe_child_nonconsuming(
            child.pid,
            include_stopped=False,
        )
        if observation is not None:
            _publish_child_observation(child, observation)
            require(
                observation.code == CLD_EXITED,
                "Exactness child terminated by a signal",
            )
            return observation.status
        time.sleep(min(POLL_SECONDS, remaining))


def _group_snapshot(pid: int) -> list[dict[str, Any]]:
    return d0.process_group_snapshot(
        pid,
        cwd=Path("/"),
        timeout_seconds=1.0,
    )


def _group_member_pids(pid: int) -> list[int]:
    return d0.process_group_member_pids(pid)


def _wait_for_quiescent_group(
    child: _SpawnedChildState,
    *,
    deadline: float,
) -> list[int]:
    require(
        child.ownership == _ChildOwnership.OWNED
        and child.lifecycle
        in {
            _ChildLifecycle.TERMINAL_OBSERVED,
            _ChildLifecycle.REAP_IN_PROGRESS,
        }
        and child.group_state
        in {
            _ChildGroupState.VERIFIED,
            _ChildGroupState.KILL_SENT,
            _ChildGroupState.DIRECT_KILL_SENT,
            _ChildGroupState.QUIESCENT,
        },
        "Exactness child group cannot be checked safely",
    )
    while True:
        members = _group_member_pids(child.pid)
        require(
            members == sorted(set(members)) and all(pid > 0 for pid in members),
            "Exactness process group PID list is malformed",
        )
        if members == [child.pid]:
            child.group_state = _ChildGroupState.QUIESCENT
            child.pre_reap_member_pids = members
            child.pre_reap_observed_at_monotonic_ns = time.monotonic_ns()
            return members
        remaining = deadline - time.monotonic()
        require(
            remaining > 0,
            f"Exactness process group {child.pid} retained live members",
        )
        time.sleep(min(POLL_SECONDS, remaining))


def _expected_child_identity_matches(
    child: _SpawnedChildState,
    process: Mapping[str, Any],
) -> bool:
    expected = child.expected_executable
    return (
        expected is not None
        and process.get("parent_pid") == os.getpid()
        and process.get("pid") == child.pid
        and process.get("process_group") == child.pid
        and process.get("executable_path") == expected.get("path")
        and process.get("executable_device") == expected.get("device")
        and process.get("executable_inode") == expected.get("inode")
        and process.get("executable_size") == expected.get("size")
    )


def _verify_owned_child_group(child: _SpawnedChildState) -> None:
    require(
        child.ownership == _ChildOwnership.OWNED
        and child.pid > 0
        and child.lifecycle != _ChildLifecycle.REAPED,
        "Exactness child group verification lacks ownership",
    )
    if child.group_state in {
        _ChildGroupState.VERIFIED,
        _ChildGroupState.KILL_SENT,
        _ChildGroupState.QUIESCENT,
    }:
        return
    process = d0.capture_process_identity(child.pid)
    require(
        _expected_child_identity_matches(child, process)
        and os.getpgid(child.pid) == child.pid
        and os.getsid(child.pid) == child.pid,
        "Exactness child identity, process group, or session differs",
    )
    child.group_state = _ChildGroupState.VERIFIED
    if child.lifecycle == _ChildLifecycle.UNOBSERVED:
        if process["status"] == "stopped":
            child.lifecycle = _ChildLifecycle.STOPPED
        else:
            child.lifecycle = _ChildLifecycle.RUNNING


def _recover_spawned_child(child: _SpawnedChildState) -> list[str]:
    errors: list[str] = []
    if child.ownership in {
        _ChildOwnership.NO_CHILD,
        _ChildOwnership.OWNED,
        _ChildOwnership.UNRESOLVED,
    }:
        return errors
    if child.ownership == _ChildOwnership.UNSTARTED:
        child.ownership = _ChildOwnership.NO_CHILD
        return errors
    require(
        child.ownership == _ChildOwnership.SPAWN_INDETERMINATE,
        "Exactness child ownership state is invalid",
    )
    candidate = int(child.pid_cell.value)
    if candidate <= 0:
        if child.spawn_result == 0:
            child.ownership = _ChildOwnership.UNRESOLVED
            errors.append(
                "recover spawned child: successful spawn has no positive PID"
            )
        else:
            child.ownership = _ChildOwnership.NO_CHILD
        return errors
    if child.spawn_result is not None:
        if child.spawn_result == 0:
            child.pid = candidate
            child.ownership = _ChildOwnership.OWNED
        else:
            child.ownership = _ChildOwnership.NO_CHILD
        return errors
    try:
        observation = _observe_child_nonconsuming(
            candidate,
            include_stopped=True,
        )
    except ChildProcessError:
        child.ownership = _ChildOwnership.NO_CHILD
        return errors
    except InterruptedError:
        errors.append("recover spawned child: waitid was interrupted")
        return errors
    except Exception as error:
        errors.append(
            f"recover spawned child: {type(error).__name__}: {error}"
        )
        return errors
    child.pid = candidate
    try:
        process = d0.capture_process_identity(candidate)
        identity_matches = (
            child.preexisting_child_pids is not None
            and candidate not in child.preexisting_child_pids
            and _expected_child_identity_matches(child, process)
            and os.getpgid(candidate) == candidate
            and os.getsid(candidate) == candidate
        )
    except Exception as error:
        child.ownership = _ChildOwnership.UNRESOLVED
        errors.append(
            f"recover spawned child identity: {type(error).__name__}: {error}"
        )
        return errors
    if not identity_matches:
        child.ownership = _ChildOwnership.UNRESOLVED
        errors.append("recover spawned child: candidate identity differs")
        return errors
    child.ownership = _ChildOwnership.OWNED
    child.group_state = _ChildGroupState.VERIFIED
    if observation is not None:
        _publish_child_observation(child, observation)
    elif process["status"] == "stopped":
        child.lifecycle = _ChildLifecycle.STOPPED
    else:
        child.lifecycle = _ChildLifecycle.RUNNING
    return errors


def _wait_for_terminal_child(
    child: _SpawnedChildState,
    *,
    deadline: float,
) -> None:
    while child.lifecycle != _ChildLifecycle.TERMINAL_OBSERVED:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Exactness child did not terminate during cleanup")
        observation = _observe_child_nonconsuming(
            child.pid,
            include_stopped=False,
        )
        if observation is not None:
            _publish_child_observation(child, observation)
            continue
        time.sleep(min(POLL_SECONDS, remaining))


def _reap_observed_child(child: _SpawnedChildState) -> None:
    require(
        child.ownership == _ChildOwnership.OWNED
        and child.lifecycle
        in {
            _ChildLifecycle.TERMINAL_OBSERVED,
            _ChildLifecycle.REAP_IN_PROGRESS,
        }
        and child.group_state
        in {
            _ChildGroupState.QUIESCENT,
            _ChildGroupState.EMPTY,
        },
        "Exactness child cannot be reaped before terminal group quiescence",
    )
    terminal_code = child.terminal_code
    terminal_status = child.terminal_status
    child.lifecycle = _ChildLifecycle.REAP_IN_PROGRESS
    try:
        observed_pid, status_code = os.waitpid(child.pid, 0)
    except ChildProcessError:
        child.group_state = _ChildGroupState.EMPTY
        child.lifecycle = _ChildLifecycle.REAPED
        child.reap_status_validated = False
        raise ExactnessFailure(
            "Exactness child was already reaped; its waitpid status "
            "cannot be independently validated"
        )
    require(
        observed_pid == child.pid,
        f"Exactness reap returned an unexpected PID: {observed_pid}",
    )
    if terminal_code == CLD_EXITED:
        require(
            os.WIFEXITED(status_code)
            and os.WEXITSTATUS(status_code) == terminal_status,
            "Exactness reap status differs from waitid exit status",
        )
        returncode = terminal_status
    else:
        require(
            terminal_code in {CLD_KILLED, CLD_DUMPED}
            and os.WIFSIGNALED(status_code)
            and os.WTERMSIG(status_code) == terminal_status,
            "Exactness reap status differs from waitid signal status",
        )
        returncode = -int(terminal_status)
    child.group_state = _ChildGroupState.EMPTY
    child.lifecycle = _ChildLifecycle.REAPED
    child.reap_status_validated = True
    child.reap_raw_status = status_code
    child.returncode = returncode
    child.reap_completed_at_monotonic_ns = time.monotonic_ns()


def _cleanup_child(child: _SpawnedChildState) -> list[str]:
    errors = _recover_spawned_child(child)
    if child.ownership == _ChildOwnership.NO_CHILD:
        return errors
    if child.ownership == _ChildOwnership.UNRESOLVED:
        errors.append("exactness child ownership remains unresolved")
        return errors
    require(
        child.ownership == _ChildOwnership.OWNED and child.pid > 0,
        "Exactness cleanup lacks a valid owned child",
    )
    if child.lifecycle == _ChildLifecycle.REAPED:
        if child.group_state != _ChildGroupState.EMPTY:
            child.group_state = _ChildGroupState.EMPTY
            errors.append(
                "normalized interrupted exactness reap publication"
            )
        return errors
    if child.cleanup_deadline is None:
        child.cleanup_deadline = time.monotonic() + POST_KILL_WAIT_SECONDS
    deadline = child.cleanup_deadline

    if child.lifecycle == _ChildLifecycle.REAP_IN_PROGRESS:
        try:
            _reap_observed_child(child)
        except Exception as error:
            errors.append(f"settle child: {type(error).__name__}: {error}")
        return errors

    if child.group_state == _ChildGroupState.UNVERIFIED:
        try:
            _verify_owned_child_group(child)
        except Exception as error:
            errors.append(
                f"verify process group: {type(error).__name__}: {error}"
            )
            if child.resume_attempted:
                child.group_state = _ChildGroupState.UNRESOLVED
                errors.append(
                    "cannot directly settle an unverified resumed child"
                )
                return errors
            try:
                os.kill(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as kill_error:
                errors.append(
                    "kill direct child: "
                    f"{type(kill_error).__name__}: {kill_error}"
                )
            child.group_state = _ChildGroupState.DIRECT_KILL_SENT

    if child.group_state in {
        _ChildGroupState.VERIFIED,
        _ChildGroupState.KILL_SENT,
    }:
        try:
            os.killpg(child.pid, signal.SIGKILL)
            child.group_state = _ChildGroupState.KILL_SENT
        except ProcessLookupError:
            child.group_state = _ChildGroupState.KILL_SENT
        except Exception as error:
            errors.append(f"kill process group: {type(error).__name__}: {error}")
            return errors
    try:
        _wait_for_terminal_child(child, deadline=deadline)
        _wait_for_quiescent_group(child, deadline=deadline)
        _reap_observed_child(child)
    except Exception as error:
        errors.append(f"settle child: {type(error).__name__}: {error}")
    return errors


def _child_cleanup_complete(child: _SpawnedChildState) -> bool:
    return child.ownership == _ChildOwnership.NO_CHILD or (
        child.ownership == _ChildOwnership.OWNED
        and child.lifecycle == _ChildLifecycle.REAPED
        and child.group_state == _ChildGroupState.EMPTY
    )


def _cleanup_child_after_failure(
    child: _SpawnedChildState,
    *,
    attempts: int = 2,
) -> _CleanupOutcome:
    require(attempts > 0, "Exactness cleanup attempts must be positive")
    errors: list[str] = []
    interruption: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            errors.extend(_cleanup_child(child))
        except BaseException as error:
            if isinstance(error, Exception):
                errors.append(
                    f"cleanup attempt {attempt} raised "
                    f"{type(error).__name__}: {error}"
                )
            else:
                interruption = error
        if _child_cleanup_complete(child):
            return _CleanupOutcome(tuple(errors), interruption)
    if not _child_cleanup_complete(child):
        errors.append(
            f"child cleanup remained incomplete after {attempts} attempts"
        )
    return _CleanupOutcome(tuple(errors), interruption)


def _held_executable_record(
    executable: Path,
    descriptor: int,
) -> dict[str, Any]:
    status = os.fstat(descriptor)
    require(
        stat.S_ISREG(status.st_mode)
        and status.st_nlink == 1
        and status.st_size > 0,
        f"Exactness emitter is not a single-link regular file: {executable}",
    )
    data = _read_stable_fd(
        descriptor,
        initial=status,
        label=f"exactness emitter {executable}",
    )
    path_status = executable.stat()
    require(
        _same_stable_identity(status, path_status),
        f"Exactness emitter path differs from its held descriptor: {executable}",
    )
    return {
        "path": str(executable),
        "sha256": _sha256(data),
        "bytes": len(data),
        "identity": _descriptor_identity(status),
    }


def _attest_suspended_child(
    pid: int,
    *,
    held_executable: dict[str, Any],
) -> dict[str, Any]:
    process = d0.capture_process_identity(pid)
    processes = _group_snapshot(pid)
    require(process["status"] == "stopped", "Exactness child is not stopped")
    require(
        process["parent_pid"] == os.getpid(),
        "Exactness child is not a direct child",
    )
    require(
        process["pid"] == pid
        and process["process_group"] == pid
        and os.getpgid(pid) == pid
        and os.getsid(pid) == pid,
        "Exactness child is not its process, group, and session leader",
    )
    require(process["child_pids"] == [], "Exactness child has descendants")
    require(
        [entry["pid"] for entry in processes] == [pid],
        "Exactness process group topology differs",
    )
    expected_identity = held_executable["identity"]
    require(
        (
            process["executable_device"],
            process["executable_inode"],
            process["executable_size"],
        )
        == (
            expected_identity["device"],
            expected_identity["inode"],
            expected_identity["bytes"],
        ),
        "Exactness mapped Mach-O differs from the held emitter descriptor",
    )
    return {
        "observed_at_monotonic_ns": time.monotonic_ns(),
        "supervisor_pid": os.getpid(),
        "pid": pid,
        "process_group": pid,
        "session": pid,
        "processes": processes,
        "mapped_macho_matches_held_descriptor": True,
        "direct_child": True,
        "no_descendants": True,
    }


def _run_one(
    *,
    sequence: int,
    role: str,
    repetition: int,
    executable: Path,
    dataset_path: Path,
    dataset: bytes,
    output_dir: Path,
    directory_descriptor: int,
    timeout_seconds: float,
    expected_emitter: Mapping[str, Any],
    expected_witnesses: Mapping[str, Mapping[str, int]],
) -> _CompletedRun:
    prefix = f"exactness-{sequence:02d}-{role}-{repetition}"
    names = {
        "graph": f"{prefix}-graph.bin",
        "save_to_ptr": f"{prefix}-save-to-ptr.bin",
        "witness": f"{prefix}-witness.bin",
        "stdout": f"{prefix}.stdout",
        "stderr": f"{prefix}.stderr",
    }
    descriptors: dict[str, int] = {}
    initial: dict[str, os.stat_result] = {}
    source_descriptors: list[int] = []
    emitter_descriptor = -1
    dataset_descriptor = -1
    devnull_descriptor = -1
    child = _SpawnedChildState(pid_cell=ctypes.c_int(0))
    try:
        emitter_descriptor = os.open(
            executable,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        held_emitter = _held_executable_record(executable, emitter_descriptor)
        require(
            held_emitter["path"] == expected_emitter.get("path")
            and held_emitter["sha256"] == expected_emitter.get("sha256")
            and held_emitter["bytes"] == expected_emitter.get("bytes"),
            f"{role} exactness emitter differs from the build record",
        )
        child.expected_executable = {
            "path": held_emitter["path"],
            "device": held_emitter["identity"]["device"],
            "inode": held_emitter["identity"]["inode"],
            "size": held_emitter["identity"]["bytes"],
        }
        dataset_descriptor = os.open(
            dataset_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        dataset_status = os.fstat(dataset_descriptor)
        require(
            stat.S_ISREG(dataset_status.st_mode)
            and dataset_status.st_nlink == 1
            and dataset_status.st_size == DATASET_BYTES,
            "Exactness dataset descriptor is invalid",
        )
        require(
            _sha256(
                _read_stable_fd(
                    dataset_descriptor,
                    initial=dataset_status,
                    label="exactness dataset",
                )
            )
            == DATASET_SHA256,
            "Exactness dataset descriptor digest differs",
        )
        for key, name in names.items():
            descriptors[key], initial[key] = _open_output(
                directory_descriptor,
                name,
            )
        devnull_descriptor = os.open(
            "/dev/null",
            os.O_RDONLY | os.O_CLOEXEC,
        )
        child_sources = {
            0: _duplicate_high(devnull_descriptor),
            1: _duplicate_high(descriptors["stdout"]),
            2: _duplicate_high(descriptors["stderr"]),
            CHILD_DATASET_FD: _duplicate_high(dataset_descriptor),
            CHILD_GRAPH_FD: _duplicate_high(descriptors["graph"]),
            CHILD_SAVE_TO_PTR_FD: _duplicate_high(descriptors["save_to_ptr"]),
            CHILD_WITNESS_FD: _duplicate_high(descriptors["witness"]),
        }
        source_descriptors.extend(child_sources.values())
        arguments = [
            "--dataset-fd",
            str(CHILD_DATASET_FD),
            "--graph-fd",
            str(CHILD_GRAPH_FD),
            "--save-to-ptr-fd",
            str(CHILD_SAVE_TO_PTR_FD),
            "--execution-evidence-fd",
            str(CHILD_WITNESS_FD),
        ]
        started_at_ns = time.time_ns()
        started_at_monotonic_ns = time.monotonic_ns()
        deadline = time.monotonic() + timeout_seconds
        _spawn_suspended(
            executable,
            arguments,
            d0.benchmark_environment(WORKERS),
            child_sources,
            child,
        )
        _wait_for_stopped(child, deadline=deadline)
        suspended = _attest_suspended_child(
            child.pid,
            held_executable=held_emitter,
        )
        child.group_state = _ChildGroupState.VERIFIED
        resumed_at_monotonic_ns = time.monotonic_ns()
        child.resume_attempted = True
        os.kill(child.pid, signal.SIGCONT)
        child.lifecycle = _ChildLifecycle.RUNNING
        exit_code = _wait_for_exit(child, deadline=deadline)
        _wait_for_quiescent_group(
            child,
            deadline=deadline,
        )
        _reap_observed_child(child)
        require(
            child.terminal_code == CLD_EXITED
            and child.terminal_status == 0
            and child.terminal_observed_at_monotonic_ns is not None
            and child.pre_reap_member_pids == [child.pid]
            and child.pre_reap_observed_at_monotonic_ns is not None
            and child.reap_status_validated
            and child.reap_raw_status is not None
            and child.returncode == 0
            and child.reap_completed_at_monotonic_ns is not None,
            "Exactness child reap status was not independently validated",
        )
        ended_at_monotonic_ns = time.monotonic_ns()
        ended_at_ns = time.time_ns()
        require(exit_code == 0, f"{role} exactness emitter exited {exit_code}")
        raw = {
            key: _read_stable_fd(
                descriptors[key],
                initial=initial[key],
                label=f"{role} exactness {key}",
            )
            for key in names
        }
        verification = parse_exactness_artifacts(
            role=role,
            expected_witnesses=expected_witnesses,
            stdout=raw["stdout"],
            stderr=raw["stderr"],
            graph=raw["graph"],
            save_to_ptr=raw["save_to_ptr"],
            witness=raw["witness"],
            dataset=dataset,
        )
        artifact_records = {
            key: _artifact_record(
                output_dir / name,
                descriptors[key],
                raw[key],
                directory_descriptor=directory_descriptor,
                initial=initial[key],
                label=f"{role} exactness {key}",
            )
            for key, name in names.items()
        }
        final_emitter = _held_executable_record(executable, emitter_descriptor)
        require(
            final_emitter == held_emitter,
            f"{role} exactness emitter changed during execution",
        )
        final_dataset = os.fstat(dataset_descriptor)
        require(
            _same_stable_identity(dataset_status, final_dataset),
            "Exactness dataset changed during execution",
        )
        lifecycle = {
            "schema_version": EXACTNESS_PROCESS_EVIDENCE_SCHEMA_VERSION,
            "leader_pid": child.pid,
            "terminal_observation": {
                "observed_at_monotonic_ns": (
                    child.terminal_observed_at_monotonic_ns
                ),
                "pid": child.pid,
                "idtype": P_PID,
                "options": WNOHANG | WEXITED | WNOWAIT,
                "code": child.terminal_code,
                "status": child.terminal_status,
            },
            "pre_reap_quiescence": {
                "observed_at_monotonic_ns": (
                    child.pre_reap_observed_at_monotonic_ns
                ),
                "process_group": child.pid,
                "leader_pid": child.pid,
                "member_pids": child.pre_reap_member_pids,
                "no_members_except_leader": True,
            },
            "reap": {
                "completed_at_monotonic_ns": (
                    child.reap_completed_at_monotonic_ns
                ),
                "requested_pid": child.pid,
                "options": 0,
                "returned_pid": child.pid,
                "raw_status": child.reap_raw_status,
                "status_validated": child.reap_status_validated,
                "returncode": child.returncode,
            },
        }
        return _CompletedRun(
            record={
                "sequence": sequence,
                "role": role,
                "repetition": repetition,
                "status": "pass",
                "command": [str(executable), *arguments],
                "environment": d0.benchmark_environment(WORKERS),
                "started_at_unix_ns": started_at_ns,
                "ended_at_unix_ns": ended_at_ns,
                "started_at_monotonic_ns": started_at_monotonic_ns,
                "resumed_at_monotonic_ns": resumed_at_monotonic_ns,
                "ended_at_monotonic_ns": ended_at_monotonic_ns,
                "emitter": held_emitter,
                "process_evidence": {
                    "schema_version": EXACTNESS_PROCESS_EVIDENCE_SCHEMA_VERSION,
                    "spawn_method": "darwin-posix-spawn-start-suspended-v1",
                    "spawn_flags": {
                        "value": SPAWN_FLAGS,
                        "start_suspended": POSIX_SPAWN_START_SUSPENDED,
                        "setsid": POSIX_SPAWN_SETSID,
                        "cloexec_default": POSIX_SPAWN_CLOEXEC_DEFAULT,
                    },
                    "fixed_descriptors": {
                        "dataset": CHILD_DATASET_FD,
                        "graph": CHILD_GRAPH_FD,
                        "save_to_ptr": CHILD_SAVE_TO_PTR_FD,
                        "witness": CHILD_WITNESS_FD,
                    },
                    "suspended": suspended,
                    "lifecycle": lifecycle,
                },
                "artifacts": artifact_records,
                "producer_verification": verification,
            },
            parsed=_ParsedRun(
                stdout=raw["stdout"],
                graph=raw["graph"],
                save_to_ptr=raw["save_to_ptr"],
                witness=raw["witness"],
                verification=verification,
            ),
        )
    except BaseException as primary_error:
        cleanup = _cleanup_child_after_failure(child)
        if cleanup.errors:
            _add_exception_note(
                primary_error,
                "Exactness cleanup: " + "; ".join(cleanup.errors),
            )
        if cleanup.interruption is not None:
            _add_exception_note(
                primary_error,
                "Exactness cleanup raised "
                f"{type(cleanup.interruption).__name__}: "
                f"{cleanup.interruption}",
            )
        raise
    finally:
        for descriptor in reversed(source_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        for descriptor in reversed(list(descriptors.values())):
            try:
                os.close(descriptor)
            except OSError:
                pass
        for descriptor in (
            devnull_descriptor,
            dataset_descriptor,
            emitter_descriptor,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _validate_gate_comparisons(
    completed: list[_CompletedRun],
    *,
    expected_witnesses: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    normalized_witnesses = _validate_expected_witnesses(expected_witnesses)
    require(
        [(run.record["role"], run.record["repetition"]) for run in completed]
        == list(EXACTNESS_RUN_ORDER),
        "Exactness run order differs",
    )
    by_role = {
        role: [run for run in completed if run.record["role"] == role]
        for role in ("control", "treatment")
    }
    for role, runs in by_role.items():
        require(len(runs) == 2, f"Exactness gate does not have two {role} runs")
        require(
            runs[0].parsed.stdout == runs[1].parsed.stdout
            and runs[0].parsed.graph == runs[1].parsed.graph
            and runs[0].parsed.save_to_ptr == runs[1].parsed.save_to_ptr
            and runs[0].parsed.witness == runs[1].parsed.witness,
            f"{role} exactness artifacts are not byte deterministic",
        )
    control = by_role["control"][0].parsed
    treatment = by_role["treatment"][0].parsed
    require(
        control.stdout == treatment.stdout,
        "Control and treatment exactness stdout differ",
    )
    require(
        control.graph == treatment.graph,
        "Control and treatment canonical graphs differ",
    )
    require(
        control.save_to_ptr == treatment.save_to_ptr,
        "Control and treatment SaveToPtr images differ",
    )
    witnesses_equal = control.witness == treatment.witness
    expected_witnesses_equal = (
        normalized_witnesses["control"]
        == normalized_witnesses["treatment"]
    )
    require(
        witnesses_equal == expected_witnesses_equal,
        "Control and treatment witness relation differs from expected_witnesses",
    )
    return {
        "within_role_byte_determinism": True,
        "cross_role_stdout_byte_equal": True,
        "cross_role_graph_byte_equal": True,
        "cross_role_save_to_ptr_byte_equal": True,
        "cross_role_witness_byte_equal": witnesses_equal,
        "cross_role_witness_relation_matches_expected": True,
        "graph_sha256": _sha256(control.graph),
        "save_to_ptr_sha256": _sha256(control.save_to_ptr),
        "stdout_sha256": _sha256(control.stdout),
        "control_witness_sha256": _sha256(control.witness),
        "treatment_witness_sha256": _sha256(treatment.witness),
    }


def _independent_verifier_record(
    completed: list[_CompletedRun],
    *,
    dataset_path: Path,
    expected_witnesses: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    normalized_witnesses = _validate_expected_witnesses(expected_witnesses)
    roles: dict[str, Any] = {}
    for role in ("control", "treatment"):
        role_runs = [
            run for run in completed if run.record["role"] == role
        ]
        roles[role] = {
            "runs": [
                {
                    "ordinal": ordinal,
                    "lifecycle": run.record["process_evidence"]["lifecycle"],
                    **{
                        artifact: {
                            "path": run.record["artifacts"][artifact][
                                "captured_path"
                            ],
                            "sha256": run.record["artifacts"][artifact]["sha256"],
                            "bytes": run.record["artifacts"][artifact]["bytes"],
                        }
                        for artifact in (
                            "stdout",
                            "stderr",
                            "graph",
                            "save_to_ptr",
                            "witness",
                        )
                    },
                }
                for ordinal, run in enumerate(role_runs, 1)
            ]
        }
    return {
        "schema_version": EXACTNESS_SCHEMA_VERSION,
        "expected_witnesses": normalized_witnesses,
        "dataset": {
            "path": dataset_path.name,
            "sha256": DATASET_SHA256,
            "bytes": DATASET_BYTES,
        },
        "roles": roles,
    }


def run_exactness_gate(
    *,
    dataset_path: Path,
    emitters: Mapping[str, Path],
    output_dir: Path,
    timeout_seconds: float,
    expected_emitters: Mapping[str, Mapping[str, Any]],
    expected_witnesses: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Run the mandatory four-process control/treatment exactness gate."""
    normalized_witnesses = _validate_expected_witnesses(expected_witnesses)
    require(timeout_seconds > 0, "Exactness timeout must be positive")
    require(
        set(emitters) == {"control", "treatment"},
        "Exactness emitters must contain control and treatment",
    )
    require(
        set(expected_emitters) == {"control", "treatment"},
        "Expected exactness emitters must contain both roles",
    )
    output_dir = output_dir.resolve(strict=True)
    dataset_path = dataset_path.resolve(strict=True)
    require(
        dataset_path.parent == output_dir
        and dataset_path.name == d0.DATASET_FILENAME,
        "Exactness dataset must be the frozen campaign artifact",
    )
    canonical_emitters = {
        role: path.resolve(strict=True) for role, path in emitters.items()
    }
    require(
        canonical_emitters["control"] != canonical_emitters["treatment"],
        "Exactness role emitters must use distinct paths",
    )
    directory_descriptor = os.open(
        output_dir,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    directory_status = os.fstat(directory_descriptor)
    try:
        dataset_descriptor = os.open(
            dataset_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            dataset_status = os.fstat(dataset_descriptor)
            dataset = _read_stable_fd(
                dataset_descriptor,
                initial=dataset_status,
                label="exactness gate dataset",
            )
        finally:
            os.close(dataset_descriptor)
        require(
            len(dataset) == DATASET_BYTES and _sha256(dataset) == DATASET_SHA256,
            "Exactness gate dataset identity differs",
        )
        completed: list[_CompletedRun] = []
        for sequence, (role, repetition) in enumerate(EXACTNESS_RUN_ORDER):
            completed.append(
                _run_one(
                    sequence=sequence,
                    role=role,
                    repetition=repetition,
                    executable=canonical_emitters[role],
                    dataset_path=dataset_path,
                    dataset=dataset,
                    output_dir=output_dir,
                    directory_descriptor=directory_descriptor,
                    timeout_seconds=timeout_seconds,
                    expected_emitter=expected_emitters[role],
                    expected_witnesses=normalized_witnesses,
                )
            )
        comparisons = _validate_gate_comparisons(
            completed,
            expected_witnesses=normalized_witnesses,
        )
        require(
            _same_directory_identity(
                directory_status,
                os.fstat(directory_descriptor),
            ),
            "Exactness output directory changed during execution",
        )
        return {
            "schema_version": EXACTNESS_SCHEMA_VERSION,
            "status": "pass",
            "expected_witnesses": normalized_witnesses,
            "execution_order": [
                f"{role}-{repetition}"
                for role, repetition in EXACTNESS_RUN_ORDER
            ],
            "timeout_seconds_per_run": timeout_seconds,
            "dataset": {
                "path": dataset_path.name,
                "sha256": DATASET_SHA256,
                "bytes": DATASET_BYTES,
            },
            "runs": [run.record for run in completed],
            "comparisons": comparisons,
            "independent_verifier_record": _independent_verifier_record(
                completed,
                dataset_path=dataset_path,
                expected_witnesses=normalized_witnesses,
            ),
        }
    finally:
        os.close(directory_descriptor)
