#!/usr/bin/env python3
"""Read-only validation for native HNSW exactness evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "DEFAULT_EXPECTED_ROLES",
    "ExactnessVerificationError",
    "WITNESS_FIELD_NAMES",
    "validate_exactness_gate",
    "validate_exactness_gate_with_producer_semantics",
]


class ExactnessVerificationError(RuntimeError):
    """Raised when exactness evidence is malformed or internally inconsistent."""


SCHEMA_VERSION: Final = 3
LEGACY_SCHEMA_VERSION: Final = 2
EMITTER_STDOUT_SCHEMA_VERSION: Final = 1
PROCESS_EVIDENCE_SCHEMA_VERSION: Final = 2
P_PID: Final = 1
WNOHANG: Final = 0x00000001
WEXITED: Final = 0x00000004
WNOWAIT: Final = 0x00000020
CLD_EXITED: Final = 1
VECTOR_COUNT: Final = 12_288
DIMENSION: Final = 128
M: Final = 32
EF_CONSTRUCTION: Final = 200
CHUNK_SIZE: Final = 8_192
MAX_CHUNKS: Final = 2
WORKER_COUNT: Final = 1
FLOAT32_BYTES: Final = 4
DATASET_BYTES: Final = VECTOR_COUNT * DIMENSION * FLOAT32_BYTES
DATASET_FILENAME: Final = "d0-f32le-n12288-d128-seed0.bin"
DATASET_SHA256: Final = (
    "f2e29c0f1a64d48a81e2adba0213c53d3015f459ef9936668a7dabb6a025f32a"
)
DATASET_DIGEST: Final = bytes.fromhex(DATASET_SHA256)
MMAX0: Final = 2 * M
MMAX: Final = M
MAX_SUPPORTED_LEVEL: Final = 64

GRAPH_MAGIC: Final = b"IFHXGR01"
GRAPH_SCHEMA_VERSION: Final = 1
GRAPH_HEADER_BYTES: Final = 184
GRAPH_LABEL_ENCODING_ORDINAL_I32: Final = 1

WITNESS_MAGIC: Final = b"IFHXWT01"
WITNESS_SCHEMA_VERSION: Final = 2
WITNESS_BYTES: Final = 56
WITNESS_FIELD_NAMES: Final = (
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
LEGACY_WITNESS_SCHEMA_VERSION: Final = 1
LEGACY_WITNESS_BYTES: Final = 36
LEGACY_WITNESS_FIELD_NAMES: Final = (
    "treatment_compiled",
    "capture_armed",
    "eligible_branch_entered",
    "successful_unchanged_observed",
    "successful_updated_observed",
)

LEVEL0_HEADER_BYTES: Final = 24
LEVEL0_RECORD_BYTES: Final = LEVEL0_HEADER_BYTES + MMAX0 * 4
UPPER_RECORD_BYTES: Final = 4 + MMAX * 4
LEVEL0_POINTER_OFFSET: Final = 8
LEVEL0_DEGREE_OFFSET: Final = 16
LEVEL0_NEIGHBORS_OFFSET: Final = 20
LEVEL0_TAIL_PADDING_OFFSET: Final = LEVEL0_NEIGHBORS_OFFSET + MMAX0 * 4

MAX_GRAPH_BYTES: Final = 128 * 1024 * 1024
MAX_SAVE_TO_PTR_BYTES: Final = 128 * 1024 * 1024
MAX_STDOUT_BYTES: Final = 4 * 1024

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_ROLE_RE: Final = re.compile(r"[a-z][a-z0-9_-]*")
_GRAPH_HEADER: Final = struct.Struct("<8sII13Qii2Q32sQ")
_GRAPH_VERTEX: Final = struct.Struct("<IiiI")
_GRAPH_LAYER: Final = struct.Struct("<II")
_WITNESS: Final = struct.Struct("<8sII10I")
_LEGACY_WITNESS: Final = struct.Struct("<8sII5I")
_SAVE_HEADER: Final = struct.Struct("<6Qii")
_LEVEL0_NEIGHBORS: Final = struct.Struct(f"<{MMAX0}i")
_UPPER_NEIGHBORS: Final = struct.Struct(f"<{MMAX}i")

DEFAULT_EXPECTED_ROLES: Mapping[str, int] = MappingProxyType(
    {"control": 0, "treatment": 1}
)


def _fail(message: str) -> None:
    raise ExactnessVerificationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _expect_mapping(value: Any, context: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    _require(
        all(isinstance(key, str) for key in value),
        f"{context} contains a non-string key",
    )
    return value


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{context} keys differ: expected {sorted(expected)}, found {sorted(actual)}",
    )


def _expect_list(value: Any, context: str) -> list[Any]:
    _require(type(value) is list, f"{context} must be an array")
    return value


def _expect_int(
    value: Any,
    context: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    _require(type(value) is int, f"{context} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{context} is below {minimum}")
    if maximum is not None:
        _require(value <= maximum, f"{context} exceeds {maximum}")
    return value


def _expect_string(value: Any, context: str) -> str:
    _require(type(value) is str and value != "", f"{context} must be a nonempty string")
    return value


def _expect_sha256(value: Any, context: str) -> str:
    text = _expect_string(value, context)
    _require(
        _SHA256_RE.fullmatch(text) is not None, f"{context} is not lowercase SHA-256"
    )
    return text


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_relative_path(value: Any, context: str) -> tuple[str, tuple[str, ...]]:
    text = _expect_string(value, f"{context} path")
    _require("\0" not in text, f"{context} path contains a NUL byte")
    _require("\\" not in text, f"{context} path contains a backslash")
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{context} path must be relative")
    _require(
        path.as_posix() == text
        and path.parts
        and all(part not in ("", ".", "..") for part in path.parts),
        f"{context} path is not canonical",
    )
    return text, path.parts


@dataclass
class _EvidenceReader:
    root_fd: int
    root_path: Path
    root_identity: tuple[int, ...]
    used_paths: set[str]
    used_identities: set[tuple[int, int]]

    @classmethod
    def open(
        cls,
        evidence_dir: str | Path,
        *,
        directory_descriptor: int | None = None,
        directory_identity: tuple[int, ...] | None = None,
    ) -> _EvidenceReader:
        _require(
            (directory_descriptor is None) == (directory_identity is None),
            "authenticated directory descriptor and identity must be provided together",
        )
        supplied = Path(os.path.abspath(os.fspath(evidence_dir)))
        _require(
            hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
            "platform lacks required no-follow directory APIs",
        )
        try:
            path_status = os.lstat(supplied)
        except OSError as error:
            _fail(f"cannot inspect evidence directory: {error}")
        _require(
            not stat.S_ISLNK(path_status.st_mode),
            "evidence directory must not be a symlink",
        )
        _require(
            stat.S_ISDIR(path_status.st_mode),
            "evidence directory must be a directory",
        )
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = (
                os.open(supplied, flags)
                if directory_descriptor is None
                else os.dup(directory_descriptor)
            )
        except OSError as error:
            _fail(f"cannot open evidence directory: {error}")
        try:
            descriptor_status = os.fstat(root_fd)
            _require(
                stat.S_ISDIR(descriptor_status.st_mode),
                "evidence directory is not a directory",
            )
            root_identity = _directory_identity(descriptor_status)
            _require(
                _directory_identity(path_status) == root_identity,
                "evidence directory path differs from its opened descriptor",
            )
            if directory_identity is not None:
                _require(
                    isinstance(directory_identity, tuple)
                    and len(directory_identity) == 9
                    and all(type(value) is int for value in directory_identity),
                    "authenticated evidence directory identity is invalid",
                )
                _require(
                    root_identity == directory_identity,
                    "evidence directory differs from the authenticated root",
                )
        except BaseException:
            os.close(root_fd)
            raise
        return cls(
            root_fd=root_fd,
            root_path=supplied,
            root_identity=root_identity,
            used_paths=set(),
            used_identities=set(),
        )

    def close(self) -> None:
        if self.root_fd >= 0:
            descriptor = self.root_fd
            self.root_fd = -1
            try:
                _require(
                    _directory_identity(os.fstat(descriptor))
                    == self.root_identity,
                    "evidence directory descriptor changed during verification",
                )
                _require(
                    _directory_identity(os.lstat(self.root_path))
                    == self.root_identity,
                    "evidence directory path changed during verification",
                )
            except OSError as error:
                _fail(f"cannot revalidate evidence directory: {error}")
            finally:
                os.close(descriptor)

    def _open_relative(self, parts: tuple[str, ...], context: str) -> int:
        directory_fd = os.dup(self.root_fd)
        try:
            directory_flags = (
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            for part in parts[:-1]:
                next_fd = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            return os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except (OSError, ValueError) as error:
            _fail(f"cannot open {context}: {error}")
        finally:
            os.close(directory_fd)

    def read_artifact(
        self,
        value: Any,
        context: str,
        *,
        expected_size: int | None = None,
        maximum_size: int,
    ) -> tuple[bytes, dict[str, Any]]:
        record = _expect_mapping(value, context)
        _expect_exact_keys(record, {"path", "sha256", "bytes"}, context)
        path_text, parts = _canonical_relative_path(record["path"], context)
        digest = _expect_sha256(record["sha256"], f"{context} sha256")
        recorded_size = _expect_int(
            record["bytes"],
            f"{context} bytes",
            minimum=0,
            maximum=maximum_size,
        )
        if expected_size is not None:
            _require(
                recorded_size == expected_size,
                f"{context} recorded size differs from {expected_size}",
            )
        _require(path_text not in self.used_paths, f"{context} reuses an artifact path")

        descriptor = self._open_relative(parts, context)
        try:
            before = os.fstat(descriptor)
            _require(stat.S_ISREG(before.st_mode), f"{context} is not a regular file")
            _require(
                before.st_nlink == 1,
                f"{context} does not have exactly one hard link",
            )
            _require(before.st_size >= 0, f"{context} has a negative size")
            _require(
                before.st_size == recorded_size,
                f"{context} file size differs from its record",
            )
            _require(
                before.st_size <= maximum_size,
                f"{context} exceeds its size limit",
            )
            identity = (before.st_dev, before.st_ino)
            _require(
                identity not in self.used_identities,
                f"{context} aliases another evidence artifact",
            )

            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                try:
                    chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                except InterruptedError:
                    continue
                _require(chunk != b"", f"{context} ended before its recorded size")
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            after = os.fstat(descriptor)
            _require(
                _same_snapshot(before, after),
                f"{context} changed while it was read",
            )
        finally:
            os.close(descriptor)

        actual_digest = hashlib.sha256(content).hexdigest()
        _require(actual_digest == digest, f"{context} SHA-256 differs from its record")
        self.used_paths.add(path_text)
        self.used_identities.add(identity)
        return content, {
            "path": path_text,
            "sha256": actual_digest,
            "bytes": len(content),
        }


class _Cursor:
    def __init__(self, content: bytes, context: str):
        self._content = memoryview(content)
        self._offset = 0
        self._context = context

    @property
    def offset(self) -> int:
        return self._offset

    def take(self, size: int, context: str) -> memoryview:
        _require(size >= 0, f"{self._context} requested a negative byte count")
        _require(
            size <= len(self._content) - self._offset,
            f"{self._context} is truncated in {context}",
        )
        start = self._offset
        self._offset += size
        return self._content[start : self._offset]

    def unpack(self, layout: struct.Struct, context: str) -> tuple[Any, ...]:
        return layout.unpack(self.take(layout.size, context))

    def require_empty(self) -> None:
        _require(
            self._offset == len(self._content),
            f"{self._context} contains trailing bytes",
        )


@dataclass(frozen=True)
class _Topology:
    max_level: int
    entry_point: int
    levels: tuple[int, ...]
    adjacency: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class _GraphImage:
    sha256: str
    bytes: int
    topology: _Topology


@dataclass(frozen=True)
class _SaveToPtrImage:
    sha256: str
    bytes: int
    topology: _Topology


@dataclass
class _ValidatedRun:
    content: dict[str, bytes]
    graph: _GraphImage
    save_to_ptr: _SaveToPtrImage
    witness: tuple[int, ...]
    witness_schema_version: int
    witness_field_names: tuple[str, ...]


def _validate_neighbors(
    neighbors: tuple[int, ...],
    *,
    vertex: int,
    layer: int,
    context: str,
) -> None:
    _require(
        len(set(neighbors)) == len(neighbors),
        f"{context} vertex {vertex} layer {layer} has duplicate neighbors",
    )
    for neighbor in neighbors:
        _require(
            0 <= neighbor < VECTOR_COUNT,
            f"{context} vertex {vertex} layer {layer} has an out-of-range neighbor",
        )
        _require(
            neighbor != vertex,
            f"{context} vertex {vertex} layer {layer} has a self edge",
        )


def _validate_topology(topology: _Topology, context: str) -> None:
    _require(
        len(topology.levels) == VECTOR_COUNT
        and len(topology.adjacency) == VECTOR_COUNT,
        f"{context} has the wrong vertex count",
    )
    observed_max = max(topology.levels)
    _require(
        observed_max == topology.max_level,
        f"{context} maximum level differs from its vertices",
    )
    _require(
        0 <= topology.entry_point < VECTOR_COUNT,
        f"{context} entry point is out of range",
    )
    _require(
        topology.levels[topology.entry_point] == topology.max_level,
        f"{context} entry point is not on the maximum level",
    )
    for vertex, layers in enumerate(topology.adjacency):
        _require(
            len(layers) == topology.levels[vertex] + 1,
            f"{context} vertex {vertex} has the wrong layer count",
        )
        for layer, neighbors in enumerate(layers):
            _validate_neighbors(
                neighbors,
                vertex=vertex,
                layer=layer,
                context=context,
            )
            if layer:
                for neighbor in neighbors:
                    _require(
                        topology.levels[neighbor] >= layer,
                        f"{context} upper-layer edge targets a lower-level vertex",
                    )

    visited = bytearray(VECTOR_COUNT)
    pending = [topology.entry_point]
    visited[topology.entry_point] = 1
    for vertex in pending:
        for neighbor in topology.adjacency[vertex][0]:
            if not visited[neighbor]:
                visited[neighbor] = 1
                pending.append(neighbor)
    _require(
        len(pending) == VECTOR_COUNT,
        f"{context} level-zero graph is not reachable from its entry point",
    )


def _parse_graph(content: bytes, context: str) -> _GraphImage:
    cursor = _Cursor(content, context)
    (
        magic,
        schema_version,
        header_bytes,
        vector_count,
        dimension,
        m,
        ef_construction,
        chunk_size,
        max_chunks,
        workers,
        build_start,
        build_end,
        submitted_tasks,
        index_vector_count,
        mmax0,
        mmax,
        max_level,
        entry_point,
        vertex_records,
        label_encoding,
        dataset_digest,
        body_offset,
    ) = cursor.unpack(_GRAPH_HEADER, "header")
    expected_header = (
        magic == GRAPH_MAGIC
        and schema_version == GRAPH_SCHEMA_VERSION
        and header_bytes == GRAPH_HEADER_BYTES
        and vector_count == VECTOR_COUNT
        and dimension == DIMENSION
        and m == M
        and ef_construction == EF_CONSTRUCTION
        and chunk_size == CHUNK_SIZE
        and max_chunks == MAX_CHUNKS
        and workers == WORKER_COUNT
        and build_start == 0
        and build_end == VECTOR_COUNT
        and submitted_tasks == 1
        and index_vector_count == VECTOR_COUNT
        and mmax0 == MMAX0
        and mmax == MMAX
        and 0 <= max_level <= MAX_SUPPORTED_LEVEL
        and 0 <= entry_point < VECTOR_COUNT
        and vertex_records == VECTOR_COUNT
        and label_encoding == GRAPH_LABEL_ENCODING_ORDINAL_I32
        and dataset_digest == DATASET_DIGEST
        and body_offset == GRAPH_HEADER_BYTES
        and cursor.offset == GRAPH_HEADER_BYTES
    )
    _require(expected_header, f"{context} header differs from the exact format")

    levels: list[int] = []
    adjacency: list[tuple[tuple[int, ...], ...]] = []
    for expected_ordinal in range(VECTOR_COUNT):
        ordinal, label, level, layer_count = cursor.unpack(
            _GRAPH_VERTEX,
            f"vertex {expected_ordinal}",
        )
        _require(
            ordinal == expected_ordinal,
            f"{context} vertex ordinal {ordinal} is out of sequence",
        )
        _require(
            label == expected_ordinal,
            f"{context} vertex {expected_ordinal} label differs from its ordinal",
        )
        _require(
            0 <= level <= max_level,
            f"{context} vertex {expected_ordinal} has an invalid level",
        )
        _require(
            layer_count == level + 1,
            f"{context} vertex {expected_ordinal} has an invalid layer count",
        )
        levels.append(level)
        layers: list[tuple[int, ...]] = []
        for expected_layer in range(level + 1):
            layer, degree = cursor.unpack(
                _GRAPH_LAYER,
                f"vertex {expected_ordinal} layer {expected_layer}",
            )
            capacity = MMAX0 if expected_layer == 0 else MMAX
            _require(
                layer == expected_layer,
                f"{context} vertex {expected_ordinal} layer ordinal differs",
            )
            _require(
                degree <= capacity,
                f"{context} vertex {expected_ordinal} layer {expected_layer} degree exceeds capacity",
            )
            neighbor_bytes = cursor.take(
                degree * 4,
                f"vertex {expected_ordinal} layer {expected_layer} neighbors",
            )
            neighbors = tuple(
                value[0] for value in struct.iter_unpack("<i", neighbor_bytes)
            )
            _validate_neighbors(
                neighbors,
                vertex=expected_ordinal,
                layer=expected_layer,
                context=context,
            )
            layers.append(neighbors)
        adjacency.append(tuple(layers))
    cursor.require_empty()

    topology = _Topology(
        max_level=max_level,
        entry_point=entry_point,
        levels=tuple(levels),
        adjacency=tuple(adjacency),
    )
    _validate_topology(topology, context)
    return _GraphImage(
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
        topology=topology,
    )


def _parse_save_to_ptr(
    content: bytes,
    dataset: bytes,
    context: str,
) -> _SaveToPtrImage:
    cursor = _Cursor(content, context)
    (
        m,
        ef_construction,
        vector_count,
        dimension,
        mmax0,
        mmax,
        max_level,
        entry_point,
    ) = cursor.unpack(_SAVE_HEADER, "metadata")
    _require(
        (
            m,
            ef_construction,
            vector_count,
            dimension,
            mmax0,
            mmax,
        )
        == (M, EF_CONSTRUCTION, VECTOR_COUNT, DIMENSION, MMAX0, MMAX)
        and 0 <= max_level <= MAX_SUPPORTED_LEVEL
        and 0 <= entry_point < VECTOR_COUNT,
        f"{context} metadata differs from the exact plain-L2 format",
    )

    vectors = cursor.take(DATASET_BYTES, "plain-L2 vectors")
    _require(
        vectors == memoryview(dataset),
        f"{context} vectors differ from the frozen dataset",
    )
    (serialized_layer_sum,) = cursor.unpack(
        struct.Struct("<Q"),
        "upper-layer count",
    )
    _require(
        serialized_layer_sum <= VECTOR_COUNT * MAX_SUPPORTED_LEVEL,
        f"{context} upper-layer count is out of range",
    )

    levels: list[int] = []
    level_zero_adjacency: list[tuple[int, ...]] = []
    expected_upper_ordinal = 0
    for vertex in range(VECTOR_COUNT):
        record = cursor.take(LEVEL0_RECORD_BYTES, f"level-zero vertex {vertex}")
        (level,) = struct.unpack_from("<i", record, 0)
        _require(
            bytes(record[4:LEVEL0_POINTER_OFFSET]) == b"\0" * 4,
            f"{context} vertex {vertex} has nonzero level-zero prefix padding",
        )
        (encoded_upper_offset,) = struct.unpack_from(
            "<Q",
            record,
            LEVEL0_POINTER_OFFSET,
        )
        (degree,) = struct.unpack_from("<i", record, LEVEL0_DEGREE_OFFSET)
        neighbors = _LEVEL0_NEIGHBORS.unpack_from(record, LEVEL0_NEIGHBORS_OFFSET)
        _require(
            bytes(record[LEVEL0_TAIL_PADDING_OFFSET:]) == b"\0" * 4,
            f"{context} vertex {vertex} has nonzero level-zero tail padding",
        )
        _require(
            0 <= level <= max_level,
            f"{context} vertex {vertex} has an invalid level",
        )
        _require(
            0 <= degree <= MMAX0,
            f"{context} vertex {vertex} level-zero degree exceeds capacity",
        )
        expected_offset = expected_upper_ordinal * UPPER_RECORD_BYTES
        _require(
            encoded_upper_offset == (expected_offset if level else 0),
            f"{context} vertex {vertex} has an invalid upper-layer offset",
        )
        active_neighbors = tuple(neighbors[:degree])
        _validate_neighbors(
            active_neighbors,
            vertex=vertex,
            layer=0,
            context=context,
        )
        levels.append(level)
        level_zero_adjacency.append(active_neighbors)
        expected_upper_ordinal += level
    _require(
        expected_upper_ordinal == serialized_layer_sum,
        f"{context} upper-layer count differs from vertex levels",
    )

    adjacency: list[tuple[tuple[int, ...], ...]] = []
    observed_upper_ordinal = 0
    for vertex, level in enumerate(levels):
        layers: list[tuple[int, ...]] = [level_zero_adjacency[vertex]]
        for layer in range(1, level + 1):
            record = cursor.take(
                UPPER_RECORD_BYTES,
                f"vertex {vertex} upper layer {layer}",
            )
            (degree,) = struct.unpack_from("<i", record, 0)
            neighbors = _UPPER_NEIGHBORS.unpack_from(record, 4)
            _require(
                0 <= degree <= MMAX,
                f"{context} vertex {vertex} upper-layer degree exceeds capacity",
            )
            active_neighbors = tuple(neighbors[:degree])
            _validate_neighbors(
                active_neighbors,
                vertex=vertex,
                layer=layer,
                context=context,
            )
            layers.append(active_neighbors)
            observed_upper_ordinal += 1
        adjacency.append(tuple(layers))
    _require(
        observed_upper_ordinal == serialized_layer_sum,
        f"{context} did not consume every upper-layer record",
    )

    labels = cursor.take(VECTOR_COUNT * 4, "int32 labels")
    for ordinal, (label,) in enumerate(struct.iter_unpack("<i", labels)):
        _require(
            label == ordinal,
            f"{context} label {ordinal} differs from its ordinal",
        )
    cursor.require_empty()

    topology = _Topology(
        max_level=max_level,
        entry_point=entry_point,
        levels=tuple(levels),
        adjacency=tuple(adjacency),
    )
    _validate_topology(topology, context)
    return _SaveToPtrImage(
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
        topology=topology,
    )


def _reject_json_constant(value: str) -> None:
    _fail(f"stdout JSON contains unsupported constant {value!r}")


def _reject_json_float(value: str) -> None:
    _fail(f"stdout JSON contains unsupported number {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"stdout JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _expected_stdout(
    graph: _GraphImage, save_to_ptr: _SaveToPtrImage
) -> dict[str, Any]:
    return {
        "build": {
            "end": VECTOR_COUNT,
            "memory_delta_positive": True,
            "start": 0,
            "submitted_tasks": 1,
        },
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "dimension": DIMENSION,
            "ef_construction": EF_CONSTRUCTION,
            "m": M,
            "max_chunks": MAX_CHUNKS,
            "optimize": True,
            "vectors": VECTOR_COUNT,
            "workers": WORKER_COUNT,
        },
        "dataset": {
            "bytes": DATASET_BYTES,
            "sha256": DATASET_SHA256,
        },
        "graph": {
            "bytes": graph.bytes,
            "entry_point": graph.topology.entry_point,
            "header_bytes": GRAPH_HEADER_BYTES,
            "max_level": graph.topology.max_level,
            "mmax": MMAX,
            "mmax0": MMAX0,
            "sha256": graph.sha256,
        },
        "save_to_ptr": {
            "bytes": save_to_ptr.bytes,
            "roundtrip_graph_equal": True,
            "sha256": save_to_ptr.sha256,
        },
        "schema_version": EMITTER_STDOUT_SCHEMA_VERSION,
        "status": "PASS",
    }


def _parse_stdout(
    content: bytes,
    graph: _GraphImage,
    save_to_ptr: _SaveToPtrImage,
    context: str,
) -> Mapping[str, Any]:
    _require(
        content.endswith(b"\n") and content.count(b"\n") == 1,
        f"{context} must contain exactly one newline-terminated JSON line",
    )
    try:
        text = content[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"{context} is not canonical ASCII JSON: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except ExactnessVerificationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        _fail(f"{context} is invalid JSON: {error}")
    parsed = _expect_mapping(value, context)
    expected = _expected_stdout(graph, save_to_ptr)
    expected_content = (
        json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    _require(content == expected_content, f"{context} differs from canonical output")
    return parsed


def _witness_format(
    exactness_schema_version: int,
) -> tuple[int, int, tuple[str, ...], struct.Struct]:
    if exactness_schema_version == SCHEMA_VERSION:
        return (
            WITNESS_SCHEMA_VERSION,
            WITNESS_BYTES,
            WITNESS_FIELD_NAMES,
            _WITNESS,
        )
    if exactness_schema_version == LEGACY_SCHEMA_VERSION:
        return (
            LEGACY_WITNESS_SCHEMA_VERSION,
            LEGACY_WITNESS_BYTES,
            LEGACY_WITNESS_FIELD_NAMES,
            _LEGACY_WITNESS,
        )
    _fail("unsupported exactness gate schema version")


def _parse_witness(
    content: bytes,
    context: str,
    *,
    exactness_schema_version: int,
) -> tuple[int, ...]:
    (
        expected_schema_version,
        expected_bytes,
        _,
        layout,
    ) = _witness_format(exactness_schema_version)
    _require(
        len(content) == expected_bytes,
        f"{context} must be {expected_bytes} bytes",
    )
    magic, schema_version, declared_bytes, *flags = layout.unpack(content)
    _require(magic == WITNESS_MAGIC, f"{context} magic differs")
    _require(
        schema_version == expected_schema_version,
        f"{context} schema version differs",
    )
    _require(
        declared_bytes == expected_bytes,
        f"{context} declared size differs",
    )
    _require(
        all(flag in (0, 1) for flag in flags),
        f"{context} contains a non-boolean flag",
    )
    return tuple(flags)


def _validate_expected_roles(value: Mapping[str, int]) -> dict[str, int]:
    roles = _expect_mapping(value, "expected_roles")
    _require(len(roles) == 2, "expected_roles must contain exactly two roles")
    normalized: dict[str, int] = {}
    for role, expected_bit in roles.items():
        _require(
            _ROLE_RE.fullmatch(role) is not None,
            f"expected role {role!r} is not canonical",
        )
        bit = _expect_int(
            expected_bit,
            f"expected role {role} witness bit",
            minimum=0,
            maximum=1,
        )
        normalized[role] = bit
    _require(
        sorted(normalized.values()) == [0, 1],
        "expected_roles must contain one all-zero and one all-one role",
    )
    return normalized


def _validate_expected_witnesses(
    value: Any,
    *,
    context: str = "expected_witnesses",
) -> dict[str, dict[str, int]]:
    roles = _expect_mapping(value, context)
    _require(
        all(type(role) is str for role in roles),
        f"{context} contains a non-string role",
    )
    _require(
        set(roles) == {"control", "treatment"},
        f"{context} must contain exactly control and treatment",
    )
    expected_fields = set(WITNESS_FIELD_NAMES)
    normalized: dict[str, dict[str, int]] = {}
    for role in ("control", "treatment"):
        witness_value = roles[role]
        witness = _expect_mapping(
            witness_value,
            f"{context} {role}",
        )
        _require(
            all(type(field) is str for field in witness),
            f"{context} {role} contains a non-string field",
        )
        _expect_exact_keys(
            witness,
            expected_fields,
            f"{context} {role}",
        )
        normalized[role] = {
            field: _expect_int(
                witness[field],
                f"{context} {role} {field}",
                minimum=0,
                maximum=1,
            )
            for field in WITNESS_FIELD_NAMES
        }
    return normalized


def _validate_lifecycle(value: Any, *, context: str) -> None:
    lifecycle = _expect_mapping(value, context)
    _expect_exact_keys(
        lifecycle,
        {
            "schema_version",
            "leader_pid",
            "terminal_observation",
            "pre_reap_quiescence",
            "reap",
        },
        context,
    )
    _require(
        _expect_int(lifecycle["schema_version"], f"{context} schema version")
        == PROCESS_EVIDENCE_SCHEMA_VERSION,
        f"{context} schema version differs",
    )
    leader_pid = _expect_int(
        lifecycle["leader_pid"],
        f"{context} leader PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )

    terminal_context = f"{context} terminal observation"
    terminal = _expect_mapping(
        lifecycle["terminal_observation"],
        terminal_context,
    )
    _expect_exact_keys(
        terminal,
        {
            "observed_at_monotonic_ns",
            "pid",
            "idtype",
            "options",
            "code",
            "status",
        },
        terminal_context,
    )
    terminal_observed_at = _expect_int(
        terminal["observed_at_monotonic_ns"],
        f"{terminal_context} time",
        minimum=1,
        maximum=(1 << 63) - 1,
    )
    terminal_pid = _expect_int(
        terminal["pid"],
        f"{terminal_context} PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    terminal_idtype = _expect_int(
        terminal["idtype"],
        f"{terminal_context} idtype",
    )
    terminal_options = _expect_int(
        terminal["options"],
        f"{terminal_context} options",
        minimum=0,
    )
    terminal_code = _expect_int(
        terminal["code"],
        f"{terminal_context} code",
    )
    terminal_status = _expect_int(
        terminal["status"],
        f"{terminal_context} status",
    )
    _require(
        terminal_pid == leader_pid,
        f"{terminal_context} PID differs from the leader",
    )
    _require(
        terminal_idtype == P_PID,
        f"{terminal_context} idtype differs",
    )
    _require(
        terminal_options == (WNOHANG | WEXITED | WNOWAIT),
        f"{terminal_context} options differ",
    )
    _require(
        terminal_code == CLD_EXITED and terminal_status == 0,
        f"{terminal_context} does not prove successful exit",
    )

    pre_reap_context = f"{context} pre-reap quiescence"
    pre_reap = _expect_mapping(
        lifecycle["pre_reap_quiescence"],
        pre_reap_context,
    )
    _expect_exact_keys(
        pre_reap,
        {
            "observed_at_monotonic_ns",
            "process_group",
            "leader_pid",
            "member_pids",
            "no_members_except_leader",
        },
        pre_reap_context,
    )
    pre_reap_observed_at = _expect_int(
        pre_reap["observed_at_monotonic_ns"],
        f"{pre_reap_context} time",
        minimum=terminal_observed_at,
        maximum=(1 << 63) - 1,
    )
    process_group = _expect_int(
        pre_reap["process_group"],
        f"{pre_reap_context} process group",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    pre_reap_leader = _expect_int(
        pre_reap["leader_pid"],
        f"{pre_reap_context} leader PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    member_pids = _expect_list(
        pre_reap["member_pids"],
        f"{pre_reap_context} member PIDs",
    )
    for member_index, member_pid in enumerate(member_pids):
        _expect_int(
            member_pid,
            f"{pre_reap_context} member PID {member_index}",
            minimum=1,
            maximum=(1 << 31) - 1,
        )
    _require(
        process_group == leader_pid
        and pre_reap_leader == leader_pid
        and member_pids == [leader_pid]
        and pre_reap["no_members_except_leader"] is True,
        f"{pre_reap_context} does not prove exact leader-only membership",
    )

    reap_context = f"{context} reap"
    reap = _expect_mapping(lifecycle["reap"], reap_context)
    _expect_exact_keys(
        reap,
        {
            "completed_at_monotonic_ns",
            "requested_pid",
            "options",
            "returned_pid",
            "raw_status",
            "status_validated",
            "returncode",
        },
        reap_context,
    )
    _expect_int(
        reap["completed_at_monotonic_ns"],
        f"{reap_context} completion time",
        minimum=pre_reap_observed_at,
        maximum=(1 << 63) - 1,
    )
    requested_pid = _expect_int(
        reap["requested_pid"],
        f"{reap_context} requested PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    reap_options = _expect_int(
        reap["options"],
        f"{reap_context} options",
        minimum=0,
    )
    returned_pid = _expect_int(
        reap["returned_pid"],
        f"{reap_context} returned PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    raw_status = _expect_int(
        reap["raw_status"],
        f"{reap_context} raw status",
        minimum=0,
        maximum=(1 << 32) - 1,
    )
    returncode = _expect_int(
        reap["returncode"],
        f"{reap_context} return code",
    )
    _require(
        requested_pid == leader_pid
        and returned_pid == leader_pid
        and reap_options == 0,
        f"{reap_context} PID or options differ",
    )
    _require(
        os.WIFEXITED(raw_status) and os.WEXITSTATUS(raw_status) == 0,
        f"{reap_context} raw status does not prove successful exit",
    )
    _require(
        reap["status_validated"] is True and returncode == 0,
        f"{reap_context} does not prove successful validated reap",
    )


def _validate_run(
    reader: _EvidenceReader,
    value: Any,
    *,
    role: str,
    ordinal: int,
    exactness_schema_version: int,
    expected_witness: Mapping[str, int],
    dataset: bytes,
) -> _ValidatedRun:
    context = f"{role} exactness run {ordinal}"
    record = _expect_mapping(value, context)
    _expect_exact_keys(
        record,
        {
            "ordinal",
            "lifecycle",
            "stdout",
            "stderr",
            "graph",
            "save_to_ptr",
            "witness",
        },
        context,
    )
    _require(
        _expect_int(record["ordinal"], f"{context} ordinal") == ordinal,
        f"{context} ordinal differs",
    )
    _validate_lifecycle(record["lifecycle"], context=f"{context} lifecycle")

    graph_content, graph_record = reader.read_artifact(
        record["graph"],
        f"{context} graph",
        maximum_size=MAX_GRAPH_BYTES,
    )
    graph = _parse_graph(graph_content, f"{context} graph")
    _require(
        graph.sha256 == graph_record["sha256"] and graph.bytes == graph_record["bytes"],
        f"{context} graph identity differs after parsing",
    )

    save_content, save_record = reader.read_artifact(
        record["save_to_ptr"],
        f"{context} SaveToPtr",
        maximum_size=MAX_SAVE_TO_PTR_BYTES,
    )
    save_to_ptr = _parse_save_to_ptr(
        save_content,
        dataset,
        f"{context} SaveToPtr",
    )
    _require(
        save_to_ptr.sha256 == save_record["sha256"]
        and save_to_ptr.bytes == save_record["bytes"],
        f"{context} SaveToPtr identity differs after parsing",
    )
    _require(
        graph.topology == save_to_ptr.topology,
        f"{context} graph and SaveToPtr topology differ",
    )

    (
        witness_schema_version,
        witness_bytes,
        witness_field_names,
        _,
    ) = _witness_format(exactness_schema_version)
    witness_content, _ = reader.read_artifact(
        record["witness"],
        f"{context} witness",
        expected_size=witness_bytes,
        maximum_size=witness_bytes,
    )
    witness = _parse_witness(
        witness_content,
        f"{context} witness",
        exactness_schema_version=exactness_schema_version,
    )
    _require(
        witness
        == tuple(expected_witness[field] for field in witness_field_names),
        (
            f"{context} witness differs from its role"
            if exactness_schema_version == LEGACY_SCHEMA_VERSION
            else f"{context} witness differs from expected_witnesses"
        ),
    )

    stderr_content, _ = reader.read_artifact(
        record["stderr"],
        f"{context} stderr",
        expected_size=0,
        maximum_size=MAX_STDOUT_BYTES,
    )
    _require(stderr_content == b"", f"{context} stderr is not empty")

    stdout_content, _ = reader.read_artifact(
        record["stdout"],
        f"{context} stdout",
        maximum_size=MAX_STDOUT_BYTES,
    )
    _parse_stdout(
        stdout_content,
        graph,
        save_to_ptr,
        f"{context} stdout",
    )
    return _ValidatedRun(
        content={
            "graph": graph_content,
            "save_to_ptr": save_content,
            "witness": witness_content,
            "stderr": stderr_content,
            "stdout": stdout_content,
        },
        graph=graph,
        save_to_ptr=save_to_ptr,
        witness=witness,
        witness_schema_version=witness_schema_version,
        witness_field_names=witness_field_names,
    )


def _producer_semantics(run: _ValidatedRun) -> dict[str, Any]:
    topology = run.graph.topology
    directed_edges = sum(
        len(neighbors)
        for layers in topology.adjacency
        for neighbors in layers
    )
    witness = dict(zip(run.witness_field_names, run.witness))
    return {
        "producer_parser": "tools.apple_silicon.native_hnsw_exactness",
        "stdout": {
            "bytes": len(run.content["stdout"]),
            "sha256": hashlib.sha256(run.content["stdout"]).hexdigest(),
            "canonical_json": True,
            "value": _expected_stdout(run.graph, run.save_to_ptr),
        },
        "stderr": {
            "bytes": 0,
            "sha256": hashlib.sha256(run.content["stderr"]).hexdigest(),
        },
        "graph": {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "bytes": run.graph.bytes,
            "sha256": run.graph.sha256,
            "vectors": VECTOR_COUNT,
            "dimension": DIMENSION,
            "m": M,
            "ef_construction": EF_CONSTRUCTION,
            "mmax0": MMAX0,
            "mmax": MMAX,
            "max_level": topology.max_level,
            "entry_point": topology.entry_point,
            "directed_edges": directed_edges,
        },
        "save_to_ptr": {
            "bytes": run.save_to_ptr.bytes,
            "sha256": run.save_to_ptr.sha256,
            "vectors": VECTOR_COUNT,
            "dimension": DIMENSION,
            "m": M,
            "ef_construction": EF_CONSTRUCTION,
            "mmax0": MMAX0,
            "mmax": MMAX,
            "max_level": topology.max_level,
            "entry_point": topology.entry_point,
            "upper_layer_count": sum(topology.levels),
            "level_zero_stride": LEVEL0_RECORD_BYTES,
            "upper_layer_stride": UPPER_RECORD_BYTES,
            "dataset_sha256": DATASET_SHA256,
            "topology_matches_canonical_graph": True,
            "labels_match_ordinals": True,
            "exact_eof": True,
        },
        "witness": {
            "schema_version": run.witness_schema_version,
            "bytes": len(run.content["witness"]),
            "sha256": hashlib.sha256(run.content["witness"]).hexdigest(),
            **witness,
        },
    }


def _validate_exactness_gate(
    evidence_dir: str | Path,
    record: Mapping[str, Any],
    expected_roles: Mapping[str, int] | None = None,
    *,
    expected_witnesses: Mapping[str, Mapping[str, int]] | None = None,
    directory_descriptor: int | None = None,
    directory_identity: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Validate two deterministic exactness runs for each causal build role.

    Schema 3 requires ``expected_witnesses`` and authenticates it against the
    ten-field mapping embedded in ``record``. Archived schema 2 records retain
    the original ``expected_roles`` all-zero/all-one five-flag contract.

    The function opens evidence read-only, parses every emitted format without
    producer helpers, and returns a normalized summary on success.
    """

    gate = _expect_mapping(record, "exactness gate")
    schema_version = _expect_int(
        gate.get("schema_version"),
        "exactness gate schema version",
    )
    if schema_version == LEGACY_SCHEMA_VERSION:
        _expect_exact_keys(
            gate,
            {"schema_version", "dataset", "roles"},
            "exactness gate",
        )
        _require(
            expected_witnesses is None,
            "schema 2 exactness does not accept expected_witnesses",
        )
        normalized_roles = _validate_expected_roles(
            DEFAULT_EXPECTED_ROLES
            if expected_roles is None
            else expected_roles
        )
        normalized_witnesses = {
            role: {
                field: bit
                for field in LEGACY_WITNESS_FIELD_NAMES
            }
            for role, bit in normalized_roles.items()
        }
    elif schema_version == SCHEMA_VERSION:
        _expect_exact_keys(
            gate,
            {
                "schema_version",
                "expected_witnesses",
                "dataset",
                "roles",
            },
            "exactness gate",
        )
        _require(
            expected_roles is None,
            "schema 3 exactness does not accept expected_roles",
        )
        _require(
            expected_witnesses is not None,
            "schema 3 exactness requires expected_witnesses",
        )
        normalized_witnesses = _validate_expected_witnesses(
            expected_witnesses,
        )
        recorded_witnesses = _validate_expected_witnesses(
            gate["expected_witnesses"],
            context="exactness gate expected_witnesses",
        )
        _require(
            recorded_witnesses == normalized_witnesses,
            "exactness gate expected_witnesses differ from authenticated expectations",
        )
    else:
        _fail("unsupported exactness gate schema version")

    reader = _EvidenceReader.open(
        evidence_dir,
        directory_descriptor=directory_descriptor,
        directory_identity=directory_identity,
    )
    try:
        dataset, dataset_record = reader.read_artifact(
            gate["dataset"],
            "exactness dataset",
            expected_size=DATASET_BYTES,
            maximum_size=DATASET_BYTES,
        )
        _require(
            dataset_record["path"] == DATASET_FILENAME,
            "exactness dataset path is not frozen",
        )
        _require(
            dataset_record["sha256"] == DATASET_SHA256,
            "exactness dataset SHA-256 is not frozen",
        )

        roles_value = _expect_mapping(gate["roles"], "exactness roles")
        _expect_exact_keys(
            roles_value,
            set(normalized_witnesses),
            "exactness roles",
        )
        validated_roles: dict[str, list[_ValidatedRun]] = {}
        for role, expected_witness in normalized_witnesses.items():
            role_record = _expect_mapping(
                roles_value[role],
                f"{role} exactness role",
            )
            _expect_exact_keys(
                role_record,
                {"runs"},
                f"{role} exactness role",
            )
            runs = _expect_list(role_record["runs"], f"{role} exactness runs")
            _require(
                len(runs) == 2,
                f"{role} must contain exactly two exactness runs",
            )
            validated = [
                _validate_run(
                    reader,
                    run,
                    role=role,
                    ordinal=ordinal,
                    exactness_schema_version=schema_version,
                    expected_witness=expected_witness,
                    dataset=dataset,
                )
                for ordinal, run in enumerate(runs, 1)
            ]
            for artifact in (
                "graph",
                "save_to_ptr",
                "stdout",
                "stderr",
                "witness",
            ):
                _require(
                    validated[0].content[artifact] == validated[1].content[artifact],
                    f"{role} {artifact} is not deterministic across two runs",
                )
            validated_roles[role] = validated

        role_names = list(normalized_witnesses)
        reference = validated_roles[role_names[0]][0]
        for role in role_names[1:]:
            candidate = validated_roles[role][0]
            for artifact in ("graph", "save_to_ptr", "stdout", "stderr"):
                _require(
                    reference.content[artifact] == candidate.content[artifact],
                    f"cross-role {artifact} bytes differ",
                )
            _require(
                reference.graph.topology == candidate.graph.topology
                and reference.save_to_ptr.topology == candidate.save_to_ptr.topology,
                "cross-role exactness semantics differ",
            )
            expected_equal = (
                normalized_witnesses[role_names[0]]
                == normalized_witnesses[role]
            )
            actual_equal = (
                reference.content["witness"]
                == candidate.content["witness"]
            )
            _require(
                actual_equal == expected_equal,
                "cross-role witness relation differs from expected_witnesses",
            )

        summary = {
            "schema_version": schema_version,
            "status": "PASS",
            "dataset": {
                "path": DATASET_FILENAME,
                "sha256": DATASET_SHA256,
                "bytes": DATASET_BYTES,
            },
            "configuration": {
                "vectors": VECTOR_COUNT,
                "dimension": DIMENSION,
                "m": M,
                "ef_construction": EF_CONSTRUCTION,
                "workers": WORKER_COUNT,
            },
            "graph": {
                "sha256": reference.graph.sha256,
                "bytes": reference.graph.bytes,
                "max_level": reference.graph.topology.max_level,
                "entry_point": reference.graph.topology.entry_point,
            },
            "save_to_ptr": {
                "sha256": reference.save_to_ptr.sha256,
                "bytes": reference.save_to_ptr.bytes,
            },
            "roles": {
                role: {
                    "runs": 2,
                    "witness": dict(
                        zip(
                            validated_roles[role][0].witness_field_names,
                            validated_roles[role][0].witness,
                        )
                    ),
                }
                for role in role_names
            },
            "within_role_deterministic": True,
            "cross_role_equal": True,
        }
        if schema_version == SCHEMA_VERSION:
            summary["expected_witnesses"] = normalized_witnesses
            summary["cross_role_witness_byte_equal"] = (
                validated_roles[role_names[0]][0].content["witness"]
                == validated_roles[role_names[1]][0].content["witness"]
            )
        producer_semantics = {
            role: [
                _producer_semantics(run)
                for run in validated_roles[role]
            ]
            for role in role_names
        }
        return summary, producer_semantics
    finally:
        reader.close()


def validate_exactness_gate(
    evidence_dir: str | Path,
    record: Mapping[str, Any],
    expected_roles: Mapping[str, int] | None = None,
    *,
    expected_witnesses: Mapping[str, Mapping[str, int]] | None = None,
    directory_descriptor: int | None = None,
    directory_identity: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Validate exactness evidence and return its normalized summary."""
    summary, _ = _validate_exactness_gate(
        evidence_dir,
        record,
        expected_roles,
        expected_witnesses=expected_witnesses,
        directory_descriptor=directory_descriptor,
        directory_identity=directory_identity,
    )
    return summary


def validate_exactness_gate_with_producer_semantics(
    evidence_dir: str | Path,
    record: Mapping[str, Any],
    expected_roles: Mapping[str, int] | None = None,
    *,
    expected_witnesses: Mapping[str, Mapping[str, int]] | None = None,
    directory_descriptor: int | None = None,
    directory_identity: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Validate exactness evidence and reconstruct each producer parse record."""
    return _validate_exactness_gate(
        evidence_dir,
        record,
        expected_roles,
        expected_witnesses=expected_witnesses,
        directory_descriptor=directory_descriptor,
        directory_identity=directory_identity,
    )
