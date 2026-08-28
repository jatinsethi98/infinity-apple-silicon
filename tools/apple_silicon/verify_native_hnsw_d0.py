#!/usr/bin/env python3
"""Independently verify a completed native Apple Silicon HNSW D0 campaign."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import shlex
import stat
import struct
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

VECTORS = 12_288
DIMENSIONS = 128
M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 32
CHUNK_SIZE = 8192
QUERY_COUNT = 1000
BUILD_GRAIN = 0
MEASURED_PARTICIPANTS = 12
PINNED_FAISS_COMMIT = "20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed"
PINNED_FAISS_BASE_TREE = "75d789cf0458d6ea5dbf4ebb28f16ba3d7d1a078"
PINNED_FAISS_STATS_DISABLED_TREE = "4d72e04389fcfb5ae1af55110216b2cccccae544"
PINNED_FAISS_STATS_PATCH_SHA256 = (
    "41328ad99c945d92c15039cd53a5a09d379adb19143c0e07fa8e8f7ad6bdbb29"
)
PINNED_FAISS_STATS_PATCH_PATHS = (
    "CMakeLists.txt",
    "faiss/CMakeLists.txt",
    "faiss/IndexBinaryHNSW.cpp",
    "faiss/IndexHNSW.cpp",
)
PINNED_LIBOMP_SHA256 = (
    "d672d5e16383cc21b12e80b8fa050b5d07811eefa4835e2c8c0c56762cba00be"
)

D0_SCOPE = "d0-development-only"
EVIDENCE_SCHEMA_VERSION = 11
FULL_CAMPAIGN_MODE = "full-campaign"
PREFLIGHT_ONLY_MODE = "preflight-only"
BUILD_INPUT_CLOSURE_SCHEMA_VERSION = 4
CMAKE_COMPILER_CACHE_KEYS = (
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_ASM_COMPILER",
)
CMAKE_SCAN_DEPS_CACHE_KEYS = (
    "CMAKE_C_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_ASM_COMPILER_CLANG_SCAN_DEPS",
)
AUDIT_SIDECAR_SCHEMA_VERSION = 4
AUDIT_SIDECAR_MAGIC = b"IFD0AUD4"
AUDIT_SIDECAR_SUFFIX = ".audit-v4.bin"
AUDIT_ENGINE_IDENTIFIERS = {"infinity": 1, "faiss": 2}
CAMPAIGN_BINDING_SCHEMA_VERSION = 1
STANDALONE_ROLE_IDS = {"infinity": 3, "faiss": 4}
AUDIT_ROLE_ENGINE_IDENTIFIERS = {
    1: AUDIT_ENGINE_IDENTIFIERS["infinity"],
    2: AUDIT_ENGINE_IDENTIFIERS["infinity"],
    3: AUDIT_ENGINE_IDENTIFIERS["infinity"],
    4: AUDIT_ENGINE_IDENTIFIERS["faiss"],
}
HELDOUT_MANIFEST_SCHEMA_VERSION = 2
HELDOUT_MANIFEST_FILENAME = "heldout-v1.json"
HELDOUT_QUERY_COUNT = 64
HELDOUT_QUERY_SEED = 0x6A09E667F3BCC909
RECALL_POINTS = (
    (10, 32),
    (10, 64),
    (10, 128),
    (10, 256),
    (10, 512),
    (100, 128),
    (100, 256),
    (100, 512),
)
MAXIMUM_AUDIT_SIDECAR_BYTES = 64 * 1024 * 1024
MAXIMUM_PAIRED_RECALL_DEFICIT = 0.005
MAXIMUM_PAIRED_RECALL_DEFICIT_FRACTION = (5, 1000)
QUERY_BENCHMARK_SCHEMA_VERSION = 2
QUERY_K = 10
QUERY_EF_SEARCH = 256
QUERY_RECALL_FLOOR = 0.99
QUERY_RECALL_FLOOR_FRACTION = (99, 100)
QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP = 0.005
QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP_FRACTION = (5, 1000)
QUERY_WARMUP_PASSES = 16
QUERY_MEASURED_PASSES = 160
QUERY_LATENCY_CONCURRENCY = 1
QUERY_THROUGHPUT_CONCURRENCY = 12
QUERY_LATENCY_SAMPLE_COUNT = HELDOUT_QUERY_COUNT * QUERY_MEASURED_PASSES
QUERY_THROUGHPUT_OPERATIONS = HELDOUT_QUERY_COUNT * QUERY_MEASURED_PASSES
QUERY_PERCENTILE_METHOD = "nearest-rank"
QUERY_TRANSACTION_DEFINITION = "one-process-local-top-k-call"
QUERY_PERCENTILE_METHOD_ID = 1
QUERY_TRANSACTION_DEFINITION_ID = 1
ATTESTATION_SCHEMA_VERSION = 3
ATTESTATION_BARRIER_PHASES = (
    "before-work",
    "before-index",
    "after-index",
    "after-work",
)
INDEX_TIMING_BINDING_SCHEMA_VERSION = 1
INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS = 100_000_000
PROCESS_EVIDENCE_SCHEMA_VERSION = 2
CLD_EXITED = 1
CLD_KILLED = 2
CLD_DUMPED = 3
TERMINAL_CHILD_CODES = frozenset((CLD_EXITED, CLD_KILLED, CLD_DUMPED))
PROCESS_STATUS_NAMES = {
    1: "creating",
    2: "running",
    3: "sleeping",
    4: "stopped",
    5: "zombie",
}
PARENT_ENVIRONMENT_POLICY_SCHEMA_VERSION = 1
UNSAFE_PARENT_ENVIRONMENT_NAMES = frozenset(
    {
        "AR",
        "ARFLAGS",
        "AS",
        "ASFLAGS",
        "BASH_ENV",
        "CC",
        "CFLAGS",
        "CLANG_CONFIG_FILE",
        "CLANG_NO_DEFAULT_CONFIG",
        "COMPILER_PATH",
        "CPATH",
        "CPP",
        "CPPFLAGS",
        "CXX",
        "CXXFLAGS",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "DEVELOPER_DIR",
        "ENV",
        "GCC_EXEC_PREFIX",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
        "LD",
        "LDFLAGS",
        "LIBRARY_PATH",
        "LIBS",
        "MACOSX_DEPLOYMENT_TARGET",
        "MAKEFLAGS",
        "MFLAGS",
        "NM",
        "OBJCFLAGS",
        "OBJCXXFLAGS",
        "OBJC_INCLUDE_PATH",
        "OBJCPLUS_INCLUDE_PATH",
        "RANLIB",
        "SDKROOT",
        "SOURCE_DATE_EPOCH",
        "STRIP",
        "TOOLCHAINS",
        "XCODE_DEFAULT_TOOLCHAIN_OVERRIDE",
        "ZERO_AR_DATE",
    }
)
UNSAFE_PARENT_ENVIRONMENT_PREFIXES = (
    "CCC_",
    "CCACHE_",
    "CMAKE_",
    "DISTCC_",
    "DYLD_",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
    "LD_",
    "NINJA_",
    "PYTHON",
    "RC_",
    "SCCACHE_",
)
ATTESTATION_MAXIMUM_IMAGE_COUNT = 4096
ATTESTATION_MAXIMUM_PATH_BYTES = 32 * 1024
DATASET_FILENAME = "d0-f32le-n12288-d128-seed0.bin"
DATASET_SEED = 0
DATASET_SHA256 = "f2e29c0f1a64d48a81e2adba0213c53d3015f459ef9936668a7dabb6a025f32a"
DATASET_FNV1A64 = 8518614624171827427
RUNNER_CAPTURE_FILENAME = "runner-native_hnsw_d0.py"
VERIFIER_CAPTURE_FILENAME = "verifier-native_hnsw_d0.py"
FAISS_BASE_COMMIT_OBJECT_FILENAME = "faiss-base-commit-object.txt"
FAISS_BASE_TREE_LISTING_FILENAME = "faiss-base-tree-listing.txt"
FAISS_DERIVATIVE_DIFF_FILENAME = "faiss-stats-disabled-derivative.patch"
FAISS_CHANGED_PATHS_FILENAME = "faiss-stats-disabled-changed-paths.txt"
FAISS_STATS_PATCH_FILENAME = "faiss-global-hnsw-stats-disabled.patch"
FAISS_STATS_PATCH_SOURCE = (
    "docs/apple_silicon/prototypes/faiss-stats-disabled/"
    "faiss-global-hnsw-stats-disabled.patch"
)
FLOAT32_BYTES = 4
FNV1A64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3
FNV1A64_MASK = (1 << 64) - 1

IDLE_MINIMUM_PERCENT = 95.0
DEVELOPMENT_IDLE_MINIMUM_FLOOR_PERCENT = 25.0
IDLE_WINDOW_SECONDS = 15
IDLE_SAMPLE_INTERVAL_SECONDS = 1
IDLE_SAMPLE_COUNT = 16
IDLE_TO_LAUNCH_MAX_NS = 1_000_000_000
POWER_POLICY = "source-record-only"
BATTERY_MINIMUM_PERCENT = 20
GLOBAL_SWAP_POLICY = "development-record-only"
INDEXING_WINDOW_RSS_LIMITATION = (
    "Whole-process maximum RSS includes post-timing graph and recall audits; "
    "indexing-window peak RSS is unestablished and cannot support a resource claim."
)
QUERY_SCOPE_LIMITATION = (
    "Query QPS/TPS and latency are process-local synthetic diagnostics, "
    "not end-to-end RAG or RPC measurements."
)
SERVER_PORTS = [23871, 23872, 23873, 23874]
MAX_RESPONSE_FILE_BYTES = 1024 * 1024
MAX_RESPONSE_DEPTH = 16
MAX_RESPONSE_TOKENS = 100_000
MACHO_DYNAMIC_PATH_PREFIXES = (
    "@rpath/",
    "@loader_path/",
    "@executable_path/",
)
COMMON_BUILD_SETTINGS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_OSX_ARCHITECTURES",
    "CMAKE_OSX_DEPLOYMENT_TARGET",
    "CMAKE_OSX_SYSROOT",
)
INFINITY_PRODUCTION_BUILD_SETTINGS = (
    "CMAKE_EXPORT_COMPILE_COMMANDS",
    "ENABLE_JEMALLOC",
    "VCPKG_MANIFEST_INSTALL",
    "VCPKG_INSTALLED_DIR",
    "VCPKG_TARGET_TRIPLET",
    "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION",
    "INFINITY_BUILD_TIME_OVERRIDE",
    "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL",
    "INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING",
)

FLOAT_REL_TOLERANCE = 1e-12
FLOAT_ABS_TOLERANCE = 1e-12

PAIR_PLAN = (
    ("C1", "correctness", 1, ("infinity", "faiss")),
    ("W1", "warmup", MEASURED_PARTICIPANTS, ("infinity", "faiss")),
    ("W2", "warmup", MEASURED_PARTICIPANTS, ("faiss", "infinity")),
    ("P1", "measured", MEASURED_PARTICIPANTS, ("infinity", "faiss")),
    ("P2", "measured", MEASURED_PARTICIPANTS, ("faiss", "infinity")),
    ("P3", "measured", MEASURED_PARTICIPANTS, ("faiss", "infinity")),
    ("P4", "measured", MEASURED_PARTICIPANTS, ("infinity", "faiss")),
    ("P5", "measured", MEASURED_PARTICIPANTS, ("infinity", "faiss")),
    ("P6", "measured", MEASURED_PARTICIPANTS, ("faiss", "infinity")),
)

COMMON_DRIVER_FIELDS = frozenset(
    {
        "status",
        "scope",
        "order",
        "data_fnv1a64",
        "data_sha256",
        "data_bytes",
        "vectors",
        "dimensions",
        "M",
        "ef_construction",
        "ef_search",
        "participants",
        "build_grain",
        "query_benchmark_schema",
        "query_unique_queries",
        "query_k",
        "query_ef_search",
        "query_recall_floor",
        "query_maximum_absolute_recall_gap",
        "query_warmup_passes",
        "query_measured_passes",
        "query_latency_concurrency",
        "query_throughput_concurrency",
        "query_percentile_method",
        "query_transaction_definition",
        "query_timed_corpus_sha256",
    }
)
ENGINE_RESULT_SUFFIXES = (
    "valid",
    "threads",
    "index_size",
    "cold_build_ns",
    "insert_call_ns",
    "self_recall_at_1",
    "distance_checksum",
    "query_latency_sample_count",
    "query_latency_validated_operations",
    "query_latency_validated_result_checksum",
    "query_throughput_operations",
    "query_throughput_validated_operations",
    "query_throughput_wall_ns",
    "query_throughput_validated_result_checksum",
    "query_result_checksum",
    "query_per_query_checksum_count",
    "query_per_query_checksums",
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
    "graph_valid",
    "graph_vertex_count",
    "graph_reachable_count",
    "graph_directed_edges",
    "graph_level0_directed_edges",
    "graph_max_level",
    "graph_entry_point",
    "graph_level0_capacity",
    "graph_upper_capacity",
    "graph_levels_sha256",
    "graph_sha256",
    "graph_level_histogram",
    "graph_degree_histograms",
)
EXECUTION_WITNESS_FIELDS = (
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
EXECUTION_WITNESS_MASK = (1 << len(EXECUTION_WITNESS_FIELDS)) - 1
AUDIT_DRIVER_FIELDS = frozenset(
    {
        "audit_sidecar_schema",
        "audit_sidecar_bytes",
        "audit_sidecar_sha256",
        "campaign_nonce",
        "schedule_sequence",
        "role_id",
        "binary_sha256",
        "dataset_sha256",
        "heldout_query_count",
        "heldout_query_seed",
        "heldout_queries_sha256",
        "heldout_truth_sha256",
        "heldout_truth_tie_counts_at_10",
        "heldout_truth_tie_counts_at_100",
    }
)
INFINITY_DRIVER_FIELDS = frozenset(
    {
        "infinity_submitted_tasks",
        "infinity_build_start",
        "infinity_build_end",
    }
)
RUN_RECORD_FIELDS = frozenset(
    {
        "sequence",
        "pair",
        "pair_order",
        "treatment",
        "engine",
        "participants",
        "status",
        "scope",
        "errors",
        "pid",
        "started_at",
        "started_at_unix_ns",
        "ended_at_unix_ns",
        "command",
        "working_directory",
        "binary",
        "environment",
        "idle",
        "idle_to_launch_ns",
        "host_before",
        "host_after",
        "global_swap_growth_bytes",
        "dataset_before",
        "dataset_after",
        "runtime_artifacts_before",
        "runtime_artifacts_after",
        "stdout_path",
        "stderr_path",
        "fields",
        "attestation",
        "attestation_barriers",
        "benchmark_process_evidence",
        "index_timing_binding",
        "process_lifetime_maximum_resident_set_size",
        "process_swaps",
        "audit_sidecar",
        "audit",
    }
)

MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  ([^\n]+)\Z")
SHA256_TEXT = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}\Z")
FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
UNSIGNED_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")
FINITE_DECIMAL = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))" r"(?:[eE][+-]?[0-9]+)?\Z"
)
IDLE_PERCENTAGE = re.compile(r"CPU usage:.*?([0-9.]+)% idle")
MAXIMUM_RSS = re.compile(
    r"^\s*(\d+)\s+maximum resident set size\s*$",
    re.MULTILINE,
)
PROCESS_SWAPS = re.compile(r"^\s*(\d+)\s+swaps\s*$", re.MULTILINE)
HOST_SECTION = re.compile(r"\[([a-z]+)\]\Z")
CONTENT_BLOB_PATH = re.compile(r"provenance/blobs/([0-9a-f]{2})/([0-9a-f]{64})\Z")

_HELDOUT_TRUTH_CACHE: dict[str, dict[str, Any]] = {}
_ACTIVE_VERIFIED_EVIDENCE: VerifiedEvidence | None = None


class VerificationError(RuntimeError):
    """Raised when D0 evidence fails independent verification."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _require_identity(
    expected: tuple[int, ...],
    actual: os.stat_result,
    *,
    context: str,
) -> None:
    require(
        expected == _stat_identity(actual),
        f"{context} changed during verification",
    )


def _read_descriptor(
    descriptor: int,
    *,
    context: str,
    maximum_bytes: int | None = None,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{context} must be a regular file")
    require(before.st_nlink == 1, f"{context} must not be hard-linked")
    if maximum_bytes is not None:
        require(
            before.st_size <= maximum_bytes,
            f"{context} exceeds the permitted byte limit",
        )
    chunks: list[bytes] = []
    remaining = None if maximum_bytes is None else maximum_bytes + 1
    while remaining is None or remaining > 0:
        request = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
        chunk = os.read(descriptor, request)
        if not chunk:
            break
        chunks.append(chunk)
        if remaining is not None:
            remaining -= len(chunk)
    data = b"".join(chunks)
    after = os.fstat(descriptor)
    _require_identity(_stat_identity(before), after, context=context)
    require(
        len(data) == before.st_size,
        f"{context} size changed during verification",
    )
    if maximum_bytes is not None:
        require(
            len(data) <= maximum_bytes,
            f"{context} exceeds the permitted byte limit",
        )
    return data, before


class VerifiedEvidence:
    """Pins one evidence tree and authenticates the bytes used by every parser."""

    def __init__(
        self,
        root: Path,
        *,
        directory_descriptor: int | None = None,
        directory_identity: tuple[int, ...] | None = None,
    ) -> None:
        self.root = Path(root).absolute()
        self._borrowed_descriptor = directory_descriptor
        self._expected_root_identity = directory_identity
        self._descriptor: int | None = None
        self._root_identity: tuple[int, ...] | None = None
        self._manifest_identity: tuple[int, ...] | None = None
        self._manifest_sha256: str | None = None
        self._entries: dict[str, str] = {}
        self._file_identities: dict[str, tuple[int, ...]] = {}
        self._directory_identities: dict[str, tuple[int, ...]] = {}
        self._opened = False

    @property
    def entries(self) -> dict[str, str]:
        require(self._opened, "Verified evidence is not open")
        return dict(self._entries)

    def _open_root(self) -> None:
        try:
            path_stat = os.lstat(self.root)
        except OSError as error:
            raise VerificationError(
                f"Could not lstat evidence directory: {error}"
            ) from error
        require(
            not stat.S_ISLNK(path_stat.st_mode) and stat.S_ISDIR(path_stat.st_mode),
            "Evidence directory must be a non-symlink directory",
        )
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            if self._borrowed_descriptor is None:
                descriptor = os.open(self.root, flags)
            else:
                descriptor = os.dup(self._borrowed_descriptor)
        except OSError as error:
            raise VerificationError(
                f"Could not open evidence directory: {error}"
            ) from error
        try:
            descriptor_stat = os.fstat(descriptor)
            _require_identity(
                _stat_identity(path_stat),
                descriptor_stat,
                context="Evidence directory",
            )
            if self._expected_root_identity is not None:
                require(
                    _stat_identity(descriptor_stat)
                    == self._expected_root_identity,
                    "Evidence directory differs from the authenticated root",
                )
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self._root_identity = _stat_identity(descriptor_stat)

    def _require_root_stable(self) -> None:
        require(
            self._descriptor is not None and self._root_identity is not None,
            "Verified evidence root is closed",
        )
        try:
            _require_identity(
                self._root_identity,
                os.fstat(self._descriptor),
                context="Evidence directory descriptor",
            )
            _require_identity(
                self._root_identity,
                os.lstat(self.root),
                context="Evidence directory path",
            )
        except OSError as error:
            raise VerificationError(
                f"Could not revalidate evidence directory: {error}"
            ) from error

    def _open_relative(self, relative: str) -> int:
        require(self._descriptor is not None, "Verified evidence root is closed")
        pure = PurePosixPath(relative)
        require(
            relative == pure.as_posix()
            and not pure.is_absolute()
            and all(part not in ("", ".", "..") for part in pure.parts),
            f"Evidence path is not canonical: {relative!r}",
        )
        current = os.dup(self._descriptor)
        try:
            for part in pure.parts[:-1]:
                before = os.stat(part, dir_fd=current, follow_symlinks=False)
                require(
                    stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                    f"Evidence path component is not a directory: {part}",
                )
                child = os.open(
                    part,
                    os.O_RDONLY
                    | os.O_CLOEXEC
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW,
                    dir_fd=current,
                )
                try:
                    _require_identity(
                        _stat_identity(before),
                        os.fstat(child),
                        context=f"Evidence directory component {part}",
                    )
                except BaseException:
                    os.close(child)
                    raise
                os.close(current)
                current = child
            before = os.stat(
                pure.parts[-1],
                dir_fd=current,
                follow_symlinks=False,
            )
            require(
                stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                f"Evidence file is not a non-symlink regular file: {relative}",
            )
            require(
                before.st_nlink == 1,
                f"Evidence file must not be hard-linked: {relative}",
            )
            descriptor = os.open(
                pure.parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current,
            )
            try:
                _require_identity(
                    _stat_identity(before),
                    os.fstat(descriptor),
                    context=f"Evidence file {relative}",
                )
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
        except OSError as error:
            raise VerificationError(
                f"Could not open evidence file {relative}: {error}"
            ) from error
        finally:
            os.close(current)

    def _read_relative(
        self,
        relative: str,
        *,
        expected_sha256: str | None,
        expected_identity: tuple[int, ...] | None,
        context: str,
        maximum_bytes: int | None = None,
    ) -> tuple[bytes, tuple[int, ...]]:
        self._require_root_stable()
        descriptor = self._open_relative(relative)
        try:
            data, file_stat = _read_descriptor(
                descriptor,
                context=context,
                maximum_bytes=maximum_bytes,
            )
        except OSError as error:
            raise VerificationError(f"Could not read {context}: {error}") from error
        finally:
            os.close(descriptor)
        identity = _stat_identity(file_stat)
        if expected_identity is not None:
            require(
                identity == expected_identity,
                f"{context} identity differs from the verified manifest",
            )
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None:
            require(
                digest == expected_sha256,
                f"Manifest hash mismatch for {relative}: "
                f"expected {expected_sha256}, found {digest}",
            )
        return data, identity

    def _walk_directory(
        self,
        descriptor: int,
        prefix: PurePosixPath | None = None,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
        files: dict[str, tuple[int, ...]] = {}
        directories: dict[str, tuple[int, ...]] = {}
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise VerificationError(
                f"Could not list evidence directory: {error}"
            ) from error
        for name in names:
            require(
                name not in ("", ".", "..") and "/" not in name,
                f"Evidence entry name is invalid: {name!r}",
            )
            relative_path = (
                PurePosixPath(name)
                if prefix is None
                else prefix.joinpath(name)
            )
            relative = relative_path.as_posix()
            try:
                value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise VerificationError(
                    f"Could not stat evidence entry {relative}: {error}"
                ) from error
            require(
                not stat.S_ISLNK(value.st_mode),
                f"Evidence contains a symbolic link: {relative}",
            )
            if stat.S_ISDIR(value.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_CLOEXEC
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    _require_identity(
                        _stat_identity(value),
                        os.fstat(child),
                        context=f"Evidence directory {relative}",
                    )
                    directories[relative] = _stat_identity(value)
                    child_files, child_directories = self._walk_directory(
                        child,
                        relative_path,
                    )
                    files.update(child_files)
                    directories.update(child_directories)
                finally:
                    os.close(child)
            else:
                require(
                    stat.S_ISREG(value.st_mode),
                    f"Evidence contains a non-regular entry: {relative}",
                )
                require(
                    value.st_nlink == 1,
                    f"Evidence file must not be hard-linked: {relative}",
                )
                files[relative] = _stat_identity(value)
        return files, directories

    def open(self) -> dict[str, str]:
        global _ACTIVE_VERIFIED_EVIDENCE
        require(not self._opened, "Verified evidence is already open")
        require(
            _ACTIVE_VERIFIED_EVIDENCE is None,
            "Nested verified evidence contexts are not supported",
        )
        self._open_root()
        try:
            manifest_bytes, manifest_identity = self._read_relative(
                "MANIFEST.sha256",
                expected_sha256=None,
                expected_identity=None,
                context="MANIFEST.sha256",
            )
            entries = _parse_manifest_entries(self.root, manifest_bytes)
            require(self._descriptor is not None, "Verified evidence root is closed")
            files, directories = self._walk_directory(self._descriptor)
            actual = set(files)
            listed = set(entries)
            actual.discard("MANIFEST.sha256")
            require(
                listed == actual,
                "Manifest file set differs: "
                f"missing={sorted(listed - actual)}, "
                f"unlisted={sorted(actual - listed)}",
            )
            file_identities: dict[str, tuple[int, ...]] = {}
            for relative in sorted(entries):
                _data, identity = self._read_relative(
                    relative,
                    expected_sha256=entries[relative],
                    expected_identity=files[relative],
                    context=relative,
                )
                file_identities[relative] = identity
            self._manifest_identity = manifest_identity
            self._manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self._entries = entries
            self._file_identities = file_identities
            self._directory_identities = directories
            self._opened = True
            _ACTIVE_VERIFIED_EVIDENCE = self
            return dict(entries)
        except BaseException:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            raise

    def contains(self, path: Path) -> bool:
        try:
            Path(path).absolute().relative_to(self.root)
        except ValueError:
            return False
        return True

    def read_bytes(
        self,
        path: Path,
        *,
        context: str,
        maximum_bytes: int | None = None,
    ) -> bytes:
        try:
            relative = Path(path).absolute().relative_to(self.root).as_posix()
        except ValueError as error:
            raise VerificationError(
                f"{context} is outside the verified evidence root"
            ) from error
        require(
            relative in self._entries,
            f"{context} is not listed in MANIFEST.sha256",
        )
        data, _identity = self._read_relative(
            relative,
            expected_sha256=self._entries[relative],
            expected_identity=self._file_identities[relative],
            context=context,
            maximum_bytes=maximum_bytes,
        )
        return data

    def file_stat(self, path: Path, *, context: str) -> os.stat_result:
        try:
            relative = Path(path).absolute().relative_to(self.root).as_posix()
        except ValueError as error:
            raise VerificationError(
                f"{context} is outside the verified evidence root"
            ) from error
        require(
            relative in self._entries,
            f"{context} is not listed in MANIFEST.sha256",
        )
        self._require_root_stable()
        descriptor = self._open_relative(relative)
        try:
            value = os.fstat(descriptor)
            require(
                _stat_identity(value) == self._file_identities[relative],
                f"{context} identity differs from the verified manifest",
            )
            return value
        finally:
            os.close(descriptor)

    def directory_snapshot(
        self,
        path: Path,
        *,
        context: str,
    ) -> tuple[os.stat_result, list[str]]:
        try:
            relative_path = Path(path).absolute().relative_to(self.root)
        except ValueError as error:
            raise VerificationError(
                f"{context} is outside the verified evidence root"
            ) from error
        relative = relative_path.as_posix()
        require(
            relative not in ("", ".")
            and relative in self._directory_identities,
            f"{context} is not a verified evidence directory",
        )
        self._require_root_stable()
        require(self._descriptor is not None, "Verified evidence root is closed")
        descriptor = os.dup(self._descriptor)
        try:
            for part in relative_path.parts:
                before = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                require(
                    stat.S_ISDIR(before.st_mode)
                    and not stat.S_ISLNK(before.st_mode),
                    f"{context} component is not a directory: {part}",
                )
                child = os.open(
                    part,
                    os.O_RDONLY
                    | os.O_CLOEXEC
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    _require_identity(
                        _stat_identity(before),
                        os.fstat(child),
                        context=f"{context} component {part}",
                    )
                except BaseException:
                    os.close(child)
                    raise
                os.close(descriptor)
                descriptor = child
            value = os.fstat(descriptor)
            require(
                _stat_identity(value) == self._directory_identities[relative],
                f"{context} identity differs from the verified tree",
            )
            names = sorted(os.listdir(descriptor))
            _require_identity(
                _stat_identity(value),
                os.fstat(descriptor),
                context=context,
            )
            return value, names
        except OSError as error:
            raise VerificationError(
                f"Could not inspect {context}: {error}"
            ) from error
        finally:
            os.close(descriptor)

    def close(self) -> None:
        global _ACTIVE_VERIFIED_EVIDENCE
        if self._descriptor is None:
            return
        try:
            if self._opened:
                manifest_bytes, _identity = self._read_relative(
                    "MANIFEST.sha256",
                    expected_sha256=self._manifest_sha256,
                    expected_identity=self._manifest_identity,
                    context="MANIFEST.sha256",
                )
                require(
                    _parse_manifest_entries(self.root, manifest_bytes)
                    == self._entries,
                    "MANIFEST.sha256 changed during verification",
                )
                files, directories = self._walk_directory(self._descriptor)
                require(
                    {
                        name: identity
                        for name, identity in files.items()
                        if name != "MANIFEST.sha256"
                    }
                    == self._file_identities,
                    "Evidence file identities changed during verification",
                )
                require(
                    directories == self._directory_identities,
                    "Evidence directory identities changed during verification",
                )
                for relative in sorted(self._entries):
                    self._read_relative(
                        relative,
                        expected_sha256=self._entries[relative],
                        expected_identity=self._file_identities[relative],
                        context=relative,
                    )
        finally:
            if _ACTIVE_VERIFIED_EVIDENCE is self:
                _ACTIVE_VERIFIED_EVIDENCE = None
            self._opened = False
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self) -> dict[str, str]:
        return self.open()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _read_unverified_path(
    path: Path,
    *,
    context: str,
    maximum_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"Could not open {context}: {error}") from error
    try:
        data, _file_stat = _read_descriptor(
            descriptor,
            context=context,
            maximum_bytes=maximum_bytes,
        )
        return data
    except OSError as error:
        raise VerificationError(f"Could not read {context}: {error}") from error
    finally:
        os.close(descriptor)


def read_verified_bytes(
    path: Path,
    context: str,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    if (
        _ACTIVE_VERIFIED_EVIDENCE is not None
        and _ACTIVE_VERIFIED_EVIDENCE.contains(path)
    ):
        return _ACTIVE_VERIFIED_EVIDENCE.read_bytes(
            path,
            context=context,
            maximum_bytes=maximum_bytes,
        )
    return _read_unverified_path(
        path,
        context=context,
        maximum_bytes=maximum_bytes,
    )


def verified_file_stat(path: Path, context: str) -> os.stat_result:
    if (
        _ACTIVE_VERIFIED_EVIDENCE is not None
        and _ACTIVE_VERIFIED_EVIDENCE.contains(path)
    ):
        return _ACTIVE_VERIFIED_EVIDENCE.file_stat(path, context=context)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"Could not open {context}: {error}") from error
    try:
        value = os.fstat(descriptor)
        require(stat.S_ISREG(value.st_mode), f"{context} must be a regular file")
        require(value.st_nlink == 1, f"{context} must not be hard-linked")
        return value
    finally:
        os.close(descriptor)


def verified_directory_snapshot(
    path: Path,
    context: str,
) -> tuple[os.stat_result, list[str]]:
    if (
        _ACTIVE_VERIFIED_EVIDENCE is not None
        and _ACTIVE_VERIFIED_EVIDENCE.contains(path)
    ):
        return _ACTIVE_VERIFIED_EVIDENCE.directory_snapshot(
            path,
            context=context,
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"Could not open {context}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISDIR(before.st_mode), f"{context} must be a directory")
        names = sorted(os.listdir(descriptor))
        _require_identity(
            _stat_identity(before),
            os.fstat(descriptor),
            context=context,
        )
        return before, names
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(read_verified_bytes(path, str(path))).hexdigest()


def _read_text(path: Path, context: str) -> str:
    try:
        return read_verified_bytes(path, context).decode("utf-8")
    except UnicodeError as error:
        raise VerificationError(f"Could not read {context}: {error}") from error


def _reject_json_constant(value: str) -> None:
    raise VerificationError(f"JSON contains non-finite constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def read_json(path: Path, context: str) -> Any:
    text = _read_text(path, context)
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except VerificationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise VerificationError(f"{context} is not strict JSON: {error}") from error


def expect_mapping(value: Any, context: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def expect_list(value: Any, context: str) -> list[Any]:
    require(isinstance(value, list), f"{context} must be a JSON array")
    return value


def expect_exact_keys(
    value: dict[str, Any],
    expected: Iterable[str],
    context: str,
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    require(
        actual_set == expected_set,
        f"{context} keys differ: missing={sorted(expected_set - actual_set)}, "
        f"extra={sorted(actual_set - expected_set)}",
    )


def expect_int(
    value: Any,
    context: str,
    *,
    minimum: int | None = None,
) -> int:
    require(type(value) is int, f"{context} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{context} must be at least {minimum}")
    return value


def expect_number(value: Any, context: str) -> float:
    require(
        type(value) in (int, float),
        f"{context} must be a JSON number",
    )
    result = float(value)
    require(math.isfinite(result), f"{context} must be finite")
    return result


def expect_string(value: Any, context: str) -> str:
    require(isinstance(value, str), f"{context} must be a string")
    return value


def expect_sha256(value: Any, context: str) -> str:
    text = expect_string(value, context)
    require(SHA256_TEXT.fullmatch(text) is not None, f"{context} is not SHA-256")
    return text


def expect_close(actual: Any, expected: float, context: str) -> None:
    actual_number = expect_number(actual, context)
    require(
        math.isclose(
            actual_number,
            expected,
            rel_tol=FLOAT_REL_TOLERANCE,
            abs_tol=FLOAT_ABS_TOLERANCE,
        ),
        f"{context} is {actual_number!r}, recomputed value is {expected!r}",
    )


def relative_evidence_path(
    evidence_dir: Path,
    value: Any,
    context: str,
) -> tuple[str, Path]:
    text = expect_string(value, context)
    pure = PurePosixPath(text)
    require(text != "", f"{context} is empty")
    require(not pure.is_absolute(), f"{context} must be relative")
    require(
        text == pure.as_posix()
        and all(part not in ("", ".", "..") for part in pure.parts),
        f"{context} is not a canonical relative POSIX path: {text!r}",
    )
    require("\\" not in text, f"{context} contains a backslash")
    return text, evidence_dir.joinpath(*pure.parts)


def _parse_manifest_entries(
    evidence_dir: Path,
    manifest_bytes: bytes,
) -> dict[str, str]:
    try:
        text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"MANIFEST.sha256 is not UTF-8: {error}") from error
    require(text != "", "MANIFEST.sha256 is empty")
    require("\r" not in text, "MANIFEST.sha256 must use LF line endings")
    require(text.endswith("\n"), "MANIFEST.sha256 must end with one newline")
    lines = text[:-1].split("\n")
    require(all(lines), "MANIFEST.sha256 contains a blank line")

    entries: dict[str, str] = {}
    listed_order: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        match = MANIFEST_LINE.fullmatch(line)
        require(
            match is not None,
            f"MANIFEST.sha256 line {line_number} is not canonical",
        )
        digest, path_text = match.groups()
        canonical, _ = relative_evidence_path(
            evidence_dir,
            path_text,
            f"MANIFEST.sha256 line {line_number} path",
        )
        require(
            canonical != "MANIFEST.sha256",
            "MANIFEST.sha256 must not list itself",
        )
        require(canonical not in entries, f"Manifest path is duplicated: {canonical}")
        entries[canonical] = digest
        listed_order.append(canonical)

    require(
        listed_order == sorted(listed_order),
        "MANIFEST.sha256 paths are not in canonical sorted order",
    )
    return entries


def verify_manifest(evidence_dir: Path) -> dict[str, str]:
    reader = VerifiedEvidence(evidence_dir)
    try:
        return reader.open()
    finally:
        reader.close()


def captured_content_blob_paths(value: Any) -> set[str]:
    references: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "captured_path" and isinstance(child, str):
                    match = CONTENT_BLOB_PATH.fullmatch(child)
                    if match is not None:
                        require(
                            match.group(1) == match.group(2)[:2],
                            f"Content-addressed path has the wrong shard: {child}",
                        )
                        references.add(child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return references


def verify_no_orphan_content_blobs(
    manifest: dict[str, str],
    *records: Any,
) -> None:
    archived = {path for path in manifest if path.startswith("provenance/blobs/")}
    referenced: set[str] = set()
    for record in records:
        referenced.update(captured_content_blob_paths(record))
    missing = sorted(referenced - archived)
    orphaned = sorted(archived - referenced)
    require(not missing, f"Referenced content-addressed blobs are missing: {missing}")
    require(not orphaned, f"Orphan content-addressed blobs are present: {orphaned}")


def frozen_schedule() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pair, treatment, participants, engines in PAIR_PLAN:
        pair_order = "/".join(engines)
        for engine in engines:
            entries.append(
                {
                    "sequence": len(entries),
                    "pair": pair,
                    "pair_order": pair_order,
                    "treatment": treatment,
                    "engine": engine,
                    "participants": participants,
                }
            )
    return entries


def expected_pair_plan() -> list[dict[str, Any]]:
    return [
        {
            "pair": pair,
            "treatment": treatment,
            "participants": participants,
            "order": "/".join(engines),
        }
        for pair, treatment, participants, engines in PAIR_PLAN
    ]


def validate_schedule(schedule: Any) -> list[dict[str, Any]]:
    entries = expect_list(schedule, "schedule.json")
    expected = frozen_schedule()
    schedule_keys = {
        "sequence",
        "pair",
        "pair_order",
        "treatment",
        "engine",
        "participants",
    }
    for index, entry_value in enumerate(entries):
        entry = expect_mapping(entry_value, f"schedule member {index}")
        expect_exact_keys(entry, schedule_keys, f"schedule member {index}")
        expect_int(entry["sequence"], f"schedule member {index} sequence", minimum=0)
        expect_int(
            entry["participants"],
            f"schedule member {index} participants",
            minimum=1,
        )
        for key in ("pair", "pair_order", "treatment", "engine"):
            expect_string(entry[key], f"schedule member {index} {key}")
    require(
        entries == expected,
        "schedule.json does not match the exact frozen 18-member D0 order",
    )
    measured_orders = [
        entry["pair_order"]
        for entry in entries
        if entry["treatment"] == "measured" and entry["engine"] == "infinity"
    ]
    require(
        measured_orders.count("infinity/faiss") == 3
        and measured_orders.count("faiss/infinity") == 3,
        "Measured pairs are not balanced 3/3 by order",
    )
    return entries


def fnv1a64(data: bytes, initial: int = FNV1A64_OFFSET_BASIS) -> int:
    value = initial
    for byte in data:
        value ^= byte
        value = (value * FNV1A64_PRIME) & FNV1A64_MASK
    return value


def expected_query_protocol() -> dict[str, Any]:
    return {
        "schema_version": QUERY_BENCHMARK_SCHEMA_VERSION,
        "unique_queries": HELDOUT_QUERY_COUNT,
        "k": QUERY_K,
        "ef_search": QUERY_EF_SEARCH,
        "recall_floor": QUERY_RECALL_FLOOR,
        "maximum_absolute_recall_gap": QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP,
        "warmup_passes": QUERY_WARMUP_PASSES,
        "measured_passes": QUERY_MEASURED_PASSES,
        "latency_concurrency": QUERY_LATENCY_CONCURRENCY,
        "throughput_concurrency": QUERY_THROUGHPUT_CONCURRENCY,
        "percentile_method": QUERY_PERCENTILE_METHOD,
        "transaction_definition": QUERY_TRANSACTION_DEFINITION,
    }


def execution_witness_from_bits(bits: int) -> dict[str, int]:
    require(
        type(bits) is int and 0 <= bits <= EXECUTION_WITNESS_MASK,
        "Execution witness contains unknown upper bits",
    )
    return {
        name: (bits >> index) & 1
        for index, name in enumerate(EXECUTION_WITNESS_FIELDS)
    }


def query_result_checksum(
    query_index: int,
    labels: list[int],
    distance_bits: list[int],
) -> int:
    require(
        len(labels) == len(distance_bits) == QUERY_K,
        "Query checksum input has the wrong result count",
    )
    value = fnv1a64(b"infinity-hnsw-d0-query-result-v1")
    value = fnv1a64(struct.pack("<Q", query_index), value)
    value = fnv1a64(struct.pack("<Q", len(labels)), value)
    for label, bits in zip(labels, distance_bits):
        value = fnv1a64(struct.pack("<qI", label, bits), value)
    return value


def query_benchmark_checksum(per_query_checksums: list[int]) -> int:
    require(
        len(per_query_checksums) == HELDOUT_QUERY_COUNT,
        "Query benchmark checksum has the wrong query count",
    )
    value = fnv1a64(b"infinity-hnsw-d0-query-benchmark-v1")
    value = fnv1a64(
        struct.pack("<QQQ", HELDOUT_QUERY_COUNT, QUERY_K, QUERY_EF_SEARCH),
        value,
    )
    for query_index, checksum in enumerate(per_query_checksums):
        value = fnv1a64(struct.pack("<QQ", query_index, checksum), value)
    return value


def nearest_rank(samples: list[int], percentile: float) -> int:
    require(bool(samples), "Cannot calculate a percentile from no samples")
    require(0.0 < percentile <= 1.0, "Nearest-rank percentile is invalid")
    require(
        all(type(sample) is int and sample >= 0 for sample in samples),
        "Latency samples must be nonnegative integers",
    )
    ordered = sorted(samples)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def query_performance_record(
    timed_query_corpus_sha256: str,
    latency_samples_ns: list[int],
    latency_validated_operations: int,
    latency_validated_result_checksum: int,
    throughput_operations: int,
    throughput_validated_operations: int,
    throughput_wall_ns: int,
    throughput_validated_result_checksum: int,
    result_checksum: int,
    per_query_checksums: list[int],
) -> dict[str, Any]:
    expect_sha256(
        timed_query_corpus_sha256,
        "timed query corpus SHA-256",
    )
    require(
        len(latency_samples_ns) == QUERY_LATENCY_SAMPLE_COUNT,
        "Query latency sample count differs from the frozen protocol",
    )
    require(
        latency_validated_operations == QUERY_LATENCY_SAMPLE_COUNT,
        "Query latency validated operation count differs from the frozen protocol",
    )
    require(
        throughput_operations == QUERY_THROUGHPUT_OPERATIONS
        and throughput_validated_operations == throughput_operations,
        "Query throughput operation counts differ from the frozen protocol",
    )
    require(
        len(per_query_checksums) == HELDOUT_QUERY_COUNT
        and all(
            type(value) is int and 0 <= value <= (1 << 64) - 1
            for value in per_query_checksums
        ),
        "Query per-query checksums differ from the frozen protocol",
    )
    require(
        all(
            type(value) is int and 0 <= value <= (1 << 64) - 1
            for value in (
                latency_validated_result_checksum,
                throughput_validated_result_checksum,
                result_checksum,
            )
        )
        and latency_validated_result_checksum == result_checksum
        and throughput_validated_result_checksum == result_checksum
        and query_benchmark_checksum(per_query_checksums) == result_checksum,
        "Query result checksum attestations differ",
    )
    require(throughput_wall_ns > 0, "Query throughput wall duration is not positive")
    qps = throughput_operations * 1e9 / throughput_wall_ns
    return {
        **expected_query_protocol(),
        "timed_query_corpus_sha256": timed_query_corpus_sha256,
        "latency_samples_ns": latency_samples_ns,
        "latency_validated_operations": latency_validated_operations,
        "latency_validated_result_checksum": latency_validated_result_checksum,
        "throughput_operations": throughput_operations,
        "throughput_validated_operations": throughput_validated_operations,
        "throughput_wall_ns": throughput_wall_ns,
        "throughput_validated_result_checksum": (
            throughput_validated_result_checksum
        ),
        "result_checksum": result_checksum,
        "per_query_checksum_count": len(per_query_checksums),
        "per_query_checksums": per_query_checksums,
        "qps": qps,
        "tps": qps,
        "p50_ns": nearest_rank(latency_samples_ns, 0.50),
        "p95_ns": nearest_rank(latency_samples_ns, 0.95),
        "p99_ns": nearest_rank(latency_samples_ns, 0.99),
    }


def fingerprint_file(path: Path) -> dict[str, Any]:
    data = read_verified_bytes(path, "dataset")
    sha = hashlib.sha256()
    fnv = FNV1A64_OFFSET_BASIS
    sha.update(data)
    fnv = fnv1a64(data, fnv)
    return {"sha256": sha.hexdigest(), "fnv1a64": fnv, "bytes": len(data)}


def validate_dataset(
    evidence_dir: Path,
    dataset_value: Any,
) -> dict[str, Any]:
    dataset = expect_mapping(dataset_value, "dataset.json")
    expected_keys = {
        "path",
        "seed",
        "generator",
        "encoding",
        "vectors",
        "dimensions",
        "values",
        "bytes_per_value",
        "mode",
        "sha256",
        "fnv1a64",
        "bytes",
    }
    expect_exact_keys(dataset, expected_keys, "dataset.json")

    expected_values = {
        "path": DATASET_FILENAME,
        "seed": DATASET_SEED,
        "generator": "python-random.Random.random",
        "encoding": "raw-little-endian-float32",
        "vectors": VECTORS,
        "dimensions": DIMENSIONS,
        "values": VECTORS * DIMENSIONS,
        "bytes_per_value": FLOAT32_BYTES,
        "mode": "0o444",
        "bytes": VECTORS * DIMENSIONS * FLOAT32_BYTES,
    }
    for key, expected in expected_values.items():
        if type(expected) is int:
            expect_int(dataset.get(key), f"dataset.json {key}", minimum=0)
        else:
            expect_string(dataset.get(key), f"dataset.json {key}")
        require(
            dataset.get(key) == expected,
            f"dataset.json {key} is {dataset.get(key)!r}, expected {expected!r}",
        )
    expect_sha256(dataset["sha256"], "dataset.json sha256")
    fnv = expect_int(dataset["fnv1a64"], "dataset.json fnv1a64", minimum=0)
    require(fnv <= FNV1A64_MASK, "dataset.json fnv1a64 exceeds uint64")
    require(
        dataset["sha256"] == DATASET_SHA256,
        "dataset.json SHA-256 is not the frozen canonical dataset",
    )
    require(
        fnv == DATASET_FNV1A64,
        "dataset.json FNV-1a is not the frozen canonical dataset",
    )

    path_text, dataset_path = relative_evidence_path(
        evidence_dir,
        dataset["path"],
        "dataset.json path",
    )
    require(path_text == DATASET_FILENAME, "Dataset path is not frozen")
    require(dataset_path.is_file(), "Frozen dataset file is missing")
    mode = stat.S_IMODE(dataset_path.stat().st_mode)
    require(mode & 0o222 == 0, f"Dataset file is writable: mode {mode:#o}")
    actual = fingerprint_file(dataset_path)
    for key in ("sha256", "fnv1a64", "bytes"):
        require(
            actual[key] == dataset[key],
            f"Dataset {key} mismatch: record {dataset[key]!r}, "
            f"actual {actual[key]!r}",
        )
    return dataset


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _float32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & FNV1A64_MASK
    value = state
    value = (
        (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    ) & FNV1A64_MASK
    value = (
        (value ^ (value >> 27)) * 0x94D049BB133111EB
    ) & FNV1A64_MASK
    return state, (value ^ (value >> 31)) & FNV1A64_MASK


def _heldout_queries() -> tuple[tuple[float, ...], str]:
    state = HELDOUT_QUERY_SEED
    values: list[float] = []
    digest = hashlib.sha256()
    digest.update(b"infinity-hnsw-d0-queries-v1")
    digest.update(struct.pack("<QQ", HELDOUT_QUERY_COUNT, DIMENSIONS))
    for _ in range(HELDOUT_QUERY_COUNT * DIMENSIONS):
        state, random_value = _splitmix64(state)
        value = _float32(float(random_value >> 40) * (1.0 / 16_777_216.0))
        values.append(value)
        digest.update(struct.pack("<I", _float32_bits(value)))
    queries = tuple(
        tuple(values[offset : offset + DIMENSIONS])
        for offset in range(0, len(values), DIMENSIONS)
    )
    return queries, digest.hexdigest()


def heldout_truth_for_validated_dataset(
    dataset_path: Path,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Build exhaustive held-out truth once per already validated dataset hash."""

    cached = _HELDOUT_TRUTH_CACHE.get(dataset_sha256)
    if cached is not None:
        return cached

    raw = read_verified_bytes(
        dataset_path,
        "validated dataset for held-out truth",
    )
    require(
        len(raw) == VECTORS * DIMENSIONS * FLOAT32_BYTES,
        "Validated dataset has the wrong size while reconstructing held-out truth",
    )
    require(
        hashlib.sha256(raw).hexdigest() == dataset_sha256,
        "Validated dataset changed before held-out truth reconstruction",
    )

    base = array.array("f")
    require(base.itemsize == FLOAT32_BYTES, "Host float32 representation is unsupported")
    base.frombytes(raw)
    if sys.byteorder != "little":
        base.byteswap()
    require(
        len(base) == VECTORS * DIMENSIONS,
        "Validated dataset float count differs during held-out reconstruction",
    )

    queries, queries_sha256 = _heldout_queries()
    truth_digest = hashlib.sha256()
    truth_digest.update(b"infinity-hnsw-d0-truth-v1")
    truth_digest.update(
        struct.pack("<QQQ", HELDOUT_QUERY_COUNT, VECTORS, DIMENSIONS)
    )
    all_distances: list[array.array[float]] = []
    top_100: list[tuple[int, ...]] = []
    cutoffs_at_10: list[float] = []
    cutoffs_at_100: list[float] = []
    tie_counts_at_10: list[int] = []
    tie_counts_at_100: list[int] = []

    for query_index, query in enumerate(queries):
        distances = array.array("d", [0.0]) * VECTORS
        matches_base_row = False
        for ordinal in range(VECTORS):
            offset = ordinal * DIMENSIONS
            distance = 0.0
            equal = True
            for component in range(DIMENSIONS):
                base_value = float(base[offset + component])
                query_value = query[component]
                difference = query_value - base_value
                distance += difference * difference
                equal = equal and query_value == base_value
            distances[ordinal] = distance
            matches_base_row = matches_base_row or equal
        require(
            not matches_base_row,
            f"Held-out query {query_index} duplicates an indexed row",
        )

        order = sorted(
            range(VECTORS),
            key=lambda ordinal: (distances[ordinal], ordinal),
        )
        query_top_100 = tuple(order[:100])
        cutoff_10 = distances[query_top_100[9]]
        cutoff_100 = distances[query_top_100[99]]
        ties_10 = sum(1 for distance in distances if distance <= cutoff_10)
        ties_100 = sum(1 for distance in distances if distance <= cutoff_100)

        truth_digest.update(
            struct.pack("<QQQ", query_index, ties_10, ties_100)
        )
        for ordinal in query_top_100:
            truth_digest.update(struct.pack("<I", ordinal))

        all_distances.append(distances)
        top_100.append(query_top_100)
        cutoffs_at_10.append(cutoff_10)
        cutoffs_at_100.append(cutoff_100)
        tie_counts_at_10.append(ties_10)
        tie_counts_at_100.append(ties_100)

    result: dict[str, Any] = {
        "queries": queries,
        "queries_sha256": queries_sha256,
        "truth_sha256": truth_digest.hexdigest(),
        "distances": tuple(all_distances),
        "top_100": tuple(top_100),
        "cutoffs_at_10": tuple(cutoffs_at_10),
        "cutoffs_at_100": tuple(cutoffs_at_100),
        "tie_counts_at_10": tuple(tie_counts_at_10),
        "tie_counts_at_100": tuple(tie_counts_at_100),
    }
    _HELDOUT_TRUTH_CACHE[dataset_sha256] = result
    return result


def expected_recall_points_json() -> list[dict[str, int]]:
    return [{"k": k, "ef": ef} for k, ef in RECALL_POINTS]


def validate_heldout_manifest(
    evidence_dir: Path,
    value: Any,
    *,
    dataset: dict[str, Any],
    truth: dict[str, Any],
) -> dict[str, Any]:
    context = HELDOUT_MANIFEST_FILENAME
    manifest = expect_mapping(value, context)
    expect_exact_keys(
        manifest,
        {
            "schema_version",
            "dimension",
            "query_count",
            "seed",
            "generator",
            "recall_points",
            "queries_sha256",
            "truth_sha256",
            "tie_counts_at_10",
            "tie_counts_at_100",
        },
        context,
    )
    expected_scalars = {
        "schema_version": HELDOUT_MANIFEST_SCHEMA_VERSION,
        "dimension": DIMENSIONS,
        "query_count": HELDOUT_QUERY_COUNT,
        "seed": HELDOUT_QUERY_SEED,
        "generator": "splitmix64-high24-float32",
        "queries_sha256": truth["queries_sha256"],
        "truth_sha256": truth["truth_sha256"],
    }
    for name, expected in expected_scalars.items():
        require(
            type(manifest.get(name)) is type(expected)
            and manifest.get(name) == expected,
            f"{context} {name} differs from independent reconstruction",
        )
    require(
        manifest["recall_points"] == expected_recall_points_json(),
        f"{context} recall points differ from the frozen protocol",
    )
    for cutoff in (10, 100):
        name = f"tie_counts_at_{cutoff}"
        counts = expect_list(manifest[name], f"{context} {name}")
        require(
            len(counts) == HELDOUT_QUERY_COUNT
            and all(type(count) is int and cutoff <= count <= VECTORS for count in counts),
            f"{context} {name} is invalid",
        )
        require(
            counts == list(truth[name]),
            f"{context} {name} differs from independent exhaustive truth",
        )
    return manifest


class _AuditSidecarReader:
    def __init__(self, data: bytes, context: str) -> None:
        self.data = data
        self.context = context
        self.offset = 0

    def unpack(self, format_string: str, description: str) -> tuple[Any, ...]:
        parser = struct.Struct("<" + format_string)
        end = self.offset + parser.size
        require(
            end <= len(self.data),
            f"{self.context} is truncated while reading {description}",
        )
        values = parser.unpack_from(self.data, self.offset)
        self.offset = end
        return values

    def finish(self) -> None:
        require(
            self.offset == len(self.data),
            f"{self.context} has trailing bytes after the final recall result",
        )


def read_audit_sidecar(path: Path, context: str) -> bytes:
    try:
        data = read_verified_bytes(
            path,
            context,
            maximum_bytes=MAXIMUM_AUDIT_SIDECAR_BYTES,
        )
    except VerificationError as error:
        raise VerificationError(
            f"{context} is missing or unreadable: {error}"
        ) from error
    require(
        0 < len(data) <= MAXIMUM_AUDIT_SIDECAR_BYTES,
        f"{context} size is outside the permitted bound",
    )
    return data


def _encode_level_histogram(histogram: Sequence[int]) -> str:
    return ",".join(f"{level}:{count}" for level, count in enumerate(histogram))


def _encode_degree_histograms(histograms: Sequence[Sequence[int]]) -> str:
    return ";".join(
        f"L{layer}:" + ",".join(str(count) for count in histogram)
        for layer, histogram in enumerate(histograms)
    )


def parse_audit_sidecar(
    data: bytes,
    *,
    engine: str,
    fields: dict[str, str],
    truth: dict[str, Any],
    context: str,
    expected_campaign_nonce: str,
    expected_schedule_sequence: int,
    expected_role_id: int,
    expected_binary_sha256: str,
    expected_dataset_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        0 < len(data) <= MAXIMUM_AUDIT_SIDECAR_BYTES,
        f"{context} size is outside the permitted bound",
    )
    expected_engine_identifier = AUDIT_ENGINE_IDENTIFIERS.get(engine)
    require(expected_engine_identifier is not None, f"Unknown audit engine {engine!r}")
    require(
        SHA256_TEXT.fullmatch(expected_campaign_nonce) is not None,
        f"{context} expected campaign nonce is not canonical lowercase hex",
    )
    require(
        type(expected_schedule_sequence) is int
        and 0 <= expected_schedule_sequence <= (1 << 64) - 1,
        f"{context} expected schedule sequence does not fit in uint64",
    )
    require(
        type(expected_role_id) is int
        and AUDIT_ROLE_ENGINE_IDENTIFIERS.get(expected_role_id)
        == expected_engine_identifier,
        f"{context} expected role ID is incompatible with engine {engine}",
    )
    for name, value in (
        ("binary SHA-256", expected_binary_sha256),
        ("dataset SHA-256", expected_dataset_sha256),
    ):
        require(
            SHA256_TEXT.fullmatch(value) is not None,
            f"{context} expected {name} is not canonical lowercase hex",
        )
    reader = _AuditSidecarReader(data, context)
    (
        magic,
        schema_version,
        engine_identifier,
        campaign_nonce_raw,
        schedule_sequence,
        role_id,
        execution_witness_bits,
        binary_sha256_raw,
        dataset_sha256_raw,
    ) = reader.unpack("8sII32sQII32s32s", "campaign binding header")
    (
        vector_count,
        dimension,
        graph_m,
        ef_construction,
        max_level,
        entry_point,
        level0_capacity,
        upper_capacity,
        query_count,
        query_seed,
        search_count,
        reserved,
    ) = reader.unpack("QQIIiiIIQQII", "audit body header")
    campaign_nonce = campaign_nonce_raw.hex()
    binary_sha256 = binary_sha256_raw.hex()
    dataset_sha256 = dataset_sha256_raw.hex()
    expected_binding = {
        "campaign_nonce": expected_campaign_nonce,
        "schedule_sequence": expected_schedule_sequence,
        "role_id": expected_role_id,
        "binary_sha256": expected_binary_sha256,
        "dataset_sha256": expected_dataset_sha256,
    }
    actual_binding: dict[str, str | int] = {
        "campaign_nonce": campaign_nonce,
        "schedule_sequence": schedule_sequence,
        "role_id": role_id,
        "binary_sha256": binary_sha256,
        "dataset_sha256": dataset_sha256,
    }
    expected_header = {
        "magic": (magic, AUDIT_SIDECAR_MAGIC),
        "schema": (schema_version, AUDIT_SIDECAR_SCHEMA_VERSION),
        "engine": (engine_identifier, expected_engine_identifier),
    }
    for name, (actual, expected) in expected_header.items():
        require(
            actual == expected,
            f"{context} {name} is {actual!r}, expected {expected!r}",
        )
    require(
        AUDIT_ROLE_ENGINE_IDENTIFIERS.get(role_id) == engine_identifier,
        f"{context} role ID is incompatible with its engine",
    )
    for name, expected in expected_binding.items():
        actual = actual_binding[name]
        require(
            actual == expected,
            f"{context} {name} differs from the expected campaign binding",
        )
        stdout_actual: str | int
        if name in {"schedule_sequence", "role_id"}:
            stdout_actual = parse_unsigned_field(fields, name, context)
        else:
            stdout_actual = fields.get(name, "")
        require(
            stdout_actual == actual,
            f"{context} stdout field {name} differs from the sidecar",
        )
    execution_witness = execution_witness_from_bits(execution_witness_bits)
    (
        query_schema,
        query_unique_queries,
        query_k,
        query_ef_search,
        query_warmup_passes,
        query_measured_passes,
        query_latency_concurrency,
        query_throughput_concurrency,
        query_percentile_method,
        query_transaction_definition,
        query_recall_floor_bits,
        query_recall_gap_bits,
    ) = reader.unpack("IIIIIIIIIIQQ", "query benchmark protocol")
    (timed_query_corpus_sha256_raw,) = reader.unpack(
        "32s",
        "timed query corpus SHA-256",
    )
    (
        query_latency_sample_count,
        query_latency_validated_operations,
        query_latency_validated_result_checksum,
        query_throughput_operations,
        query_throughput_validated_operations,
        query_throughput_wall_ns,
        query_throughput_validated_result_checksum,
        query_result_checksum_value,
        query_per_query_checksum_count,
    ) = reader.unpack("QQQQQQQQQ", "query benchmark measurements")
    require(
        query_latency_sample_count == QUERY_LATENCY_SAMPLE_COUNT
        and query_latency_validated_operations == query_latency_sample_count
        and query_throughput_operations == QUERY_THROUGHPUT_OPERATIONS
        and query_throughput_validated_operations
        == query_throughput_operations
        and query_per_query_checksum_count == HELDOUT_QUERY_COUNT,
        f"{context} query benchmark sample or operation count differs",
    )
    query_per_query_checksums = list(
        reader.unpack(
            "Q" * query_per_query_checksum_count,
            "query benchmark per-query checksums",
        )
    )
    query_latency_samples = list(
        reader.unpack(
            "Q" * query_latency_sample_count,
            "query benchmark latency samples",
        )
    )

    expected_body_header = {
        "vector count": (vector_count, VECTORS),
        "dimension": (dimension, DIMENSIONS),
        "M": (graph_m, M),
        "efConstruction": (ef_construction, EF_CONSTRUCTION),
        "query count": (query_count, HELDOUT_QUERY_COUNT),
        "query seed": (query_seed, HELDOUT_QUERY_SEED),
        "search count": (search_count, len(RECALL_POINTS)),
        "reserved value": (reserved, 0),
    }
    for name, (actual, expected) in expected_body_header.items():
        require(
            actual == expected,
            f"{context} {name} is {actual!r}, expected {expected!r}",
        )
    require(
        0 <= max_level < VECTORS,
        f"{context} graph maximum level is invalid",
    )
    require(
        0 <= entry_point < VECTORS,
        f"{context} graph entry point is out of range",
    )
    require(
        level0_capacity == 2 * M and upper_capacity == M,
        f"{context} graph capacities differ from the frozen HNSW parameters",
    )
    query_recall_floor = struct.unpack(
        "<d", struct.pack("<Q", query_recall_floor_bits)
    )[0]
    query_recall_gap = struct.unpack("<d", struct.pack("<Q", query_recall_gap_bits))[0]
    timed_query_corpus_sha256 = timed_query_corpus_sha256_raw.hex()
    expect_sha256(
        timed_query_corpus_sha256,
        f"{context} timed query corpus SHA-256",
    )
    require(
        timed_query_corpus_sha256 == truth["queries_sha256"],
        f"{context} timed query corpus differs from independent reconstruction",
    )
    require(
        (
            query_schema,
            query_unique_queries,
            query_k,
            query_ef_search,
            query_warmup_passes,
            query_measured_passes,
            query_latency_concurrency,
            query_throughput_concurrency,
            query_percentile_method,
            query_transaction_definition,
        )
        == (
            QUERY_BENCHMARK_SCHEMA_VERSION,
            HELDOUT_QUERY_COUNT,
            QUERY_K,
            QUERY_EF_SEARCH,
            QUERY_WARMUP_PASSES,
            QUERY_MEASURED_PASSES,
            QUERY_LATENCY_CONCURRENCY,
            QUERY_THROUGHPUT_CONCURRENCY,
            QUERY_PERCENTILE_METHOD_ID,
            QUERY_TRANSACTION_DEFINITION_ID,
        )
        and query_recall_floor == QUERY_RECALL_FLOOR
        and query_recall_gap == QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP,
        f"{context} query benchmark protocol differs",
    )
    query_performance = query_performance_record(
        timed_query_corpus_sha256,
        query_latency_samples,
        query_latency_validated_operations,
        query_latency_validated_result_checksum,
        query_throughput_operations,
        query_throughput_validated_operations,
        query_throughput_wall_ns,
        query_throughput_validated_result_checksum,
        query_result_checksum_value,
        query_per_query_checksums,
    )
    query_stdout_integers = {
        "query_benchmark_schema": QUERY_BENCHMARK_SCHEMA_VERSION,
        "query_unique_queries": HELDOUT_QUERY_COUNT,
        "query_k": QUERY_K,
        "query_ef_search": QUERY_EF_SEARCH,
        "query_warmup_passes": QUERY_WARMUP_PASSES,
        "query_measured_passes": QUERY_MEASURED_PASSES,
        "query_latency_concurrency": QUERY_LATENCY_CONCURRENCY,
        "query_throughput_concurrency": QUERY_THROUGHPUT_CONCURRENCY,
        f"{engine}_query_latency_sample_count": QUERY_LATENCY_SAMPLE_COUNT,
        f"{engine}_query_latency_validated_operations": (
            query_latency_validated_operations
        ),
        f"{engine}_query_latency_validated_result_checksum": (
            query_latency_validated_result_checksum
        ),
        f"{engine}_query_throughput_operations": QUERY_THROUGHPUT_OPERATIONS,
        f"{engine}_query_throughput_validated_operations": (
            query_throughput_validated_operations
        ),
        f"{engine}_query_throughput_wall_ns": query_throughput_wall_ns,
        f"{engine}_query_throughput_validated_result_checksum": (
            query_throughput_validated_result_checksum
        ),
        f"{engine}_query_result_checksum": query_result_checksum_value,
        f"{engine}_query_per_query_checksum_count": HELDOUT_QUERY_COUNT,
    }
    for name, expected in query_stdout_integers.items():
        require(
            parse_unsigned_field(fields, name, context) == expected,
            f"{context} stdout field {name} differs from the sidecar",
        )
    require(
        fields.get("query_timed_corpus_sha256")
        == timed_query_corpus_sha256,
        f"{context} stdout timed query corpus differs from the sidecar",
    )
    require(
        parse_uint64_csv(
            fields.get(f"{engine}_query_per_query_checksums"),
            expected_count=HELDOUT_QUERY_COUNT,
            context=f"{context} stdout query per-query checksums",
        )
        == query_per_query_checksums,
        f"{context} stdout per-query checksums differ from the sidecar",
    )
    for name, expected in execution_witness.items():
        require(
            parse_unsigned_field(fields, f"{engine}_{name}", context)
            == expected,
            f"{context} stdout execution witness {name} differs from the sidecar",
        )

    levels: list[int] = []
    labels_seen = bytearray(VECTORS)
    labels: list[int] = []
    adjacency: list[list[list[int]]] = []
    level_histogram = [0] * (max_level + 1)
    degree_histograms = [
        [0] * ((level0_capacity if layer == 0 else upper_capacity) + 1)
        for layer in range(max_level + 1)
    ]
    directed_edges = 0
    level0_directed_edges = 0

    levels_digest = hashlib.sha256()
    levels_digest.update(b"infinity-hnsw-levels-v1")
    levels_digest.update(struct.pack("<Q", VECTORS))
    graph_digest = hashlib.sha256()
    graph_digest.update(b"infinity-hnsw-graph-v1")
    graph_digest.update(
        struct.pack(
            "<QIIQQ",
            VECTORS,
            max_level,
            entry_point,
            level0_capacity,
            upper_capacity,
        )
    )

    for ordinal in range(VECTORS):
        level, label = reader.unpack("iq", f"vertex {ordinal} metadata")
        require(
            0 <= level <= max_level,
            f"{context} vertex {ordinal} level is invalid",
        )
        require(
            0 <= label < VECTORS,
            f"{context} vertex {ordinal} label is out of range",
        )
        require(
            not labels_seen[label],
            f"{context} graph contains duplicate label {label}",
        )
        labels_seen[label] = 1
        levels.append(level)
        labels.append(label)
        level_histogram[level] += 1
        levels_digest.update(struct.pack("<II", ordinal, level))
        graph_digest.update(struct.pack("<IIQ", ordinal, level, label))

        vertex_layers: list[list[int]] = []
        for layer in range(level + 1):
            capacity, degree = reader.unpack(
                "II", f"vertex {ordinal} layer {layer} metadata"
            )
            expected_capacity = level0_capacity if layer == 0 else upper_capacity
            require(
                capacity == expected_capacity,
                f"{context} vertex {ordinal} layer {layer} capacity differs",
            )
            require(
                degree <= capacity,
                f"{context} vertex {ordinal} layer {layer} degree exceeds capacity",
            )
            neighbors: list[int] = []
            neighbors_seen: set[int] = set()
            graph_digest.update(struct.pack("<IQ", layer, degree))
            for neighbor_index in range(degree):
                (neighbor,) = reader.unpack(
                    "i",
                    (
                        f"vertex {ordinal} layer {layer} "
                        f"neighbor {neighbor_index}"
                    ),
                )
                require(
                    0 <= neighbor < VECTORS,
                    f"{context} graph contains out-of-range edge {neighbor}",
                )
                require(
                    neighbor != ordinal,
                    f"{context} graph contains self edge at vertex {ordinal}",
                )
                require(
                    neighbor not in neighbors_seen,
                    (
                        f"{context} graph contains duplicate edge from "
                        f"vertex {ordinal} on layer {layer}"
                    ),
                )
                neighbors_seen.add(neighbor)
                neighbors.append(neighbor)
                graph_digest.update(struct.pack("<I", neighbor))
            degree_histograms[layer][degree] += 1
            directed_edges += degree
            if layer == 0:
                level0_directed_edges += degree
            vertex_layers.append(neighbors)
        adjacency.append(vertex_layers)

    require(
        all(labels_seen),
        f"{context} graph labels do not cover every native label",
    )
    require(
        max(levels) == max_level,
        f"{context} graph maximum level disagrees with its vertices",
    )
    require(
        levels[entry_point] == max_level,
        f"{context} graph entry point is not on the maximum level",
    )
    for ordinal, vertex_layers in enumerate(adjacency):
        for layer, neighbors in enumerate(vertex_layers):
            for neighbor in neighbors:
                require(
                    levels[neighbor] >= layer,
                    (
                        f"{context} upper-layer edge {ordinal}->{neighbor} "
                        f"targets a level-{levels[neighbor]} vertex"
                    ),
                )

    visited = bytearray(VECTORS)
    pending = [entry_point]
    visited[entry_point] = 1
    for ordinal in pending:
        for neighbor in adjacency[ordinal][0]:
            if not visited[neighbor]:
                visited[neighbor] = 1
                pending.append(neighbor)
    reachable_count = len(pending)
    require(
        reachable_count == VECTORS,
        (
            f"{context} level-zero graph is disconnected: "
            f"reached {reachable_count} of {VECTORS}"
        ),
    )

    graph = {
        "valid": True,
        "vertex_count": VECTORS,
        "reachable_count": reachable_count,
        "directed_edges": directed_edges,
        "level0_directed_edges": level0_directed_edges,
        "max_level": max_level,
        "entry_point": entry_point,
        "level0_capacity": level0_capacity,
        "upper_capacity": upper_capacity,
        "levels_sha256": levels_digest.hexdigest(),
        "graph_sha256": graph_digest.hexdigest(),
        "level_histogram": _encode_level_histogram(level_histogram),
        "degree_histograms": _encode_degree_histograms(degree_histograms),
    }

    points: list[dict[str, Any]] = []
    point_details: list[dict[str, str]] = []
    benchmark_query_checksums: list[int] = []
    for point_index, (expected_k, expected_ef) in enumerate(RECALL_POINTS):
        k, ef, result_count = reader.unpack(
            "IIQ", f"recall point {point_index} header"
        )
        require(
            (k, ef) == (expected_k, expected_ef),
            (
                f"{context} recall point {point_index} is {(k, ef)}, "
                f"expected {(expected_k, expected_ef)}"
            ),
        )
        require(
            result_count == HELDOUT_QUERY_COUNT * expected_k,
            f"{context} recall point {point_index} result count differs",
        )
        returned_ids: list[str] = []
        distance_digest = hashlib.sha256()
        eligible_hits = 0
        for query_index in range(HELDOUT_QUERY_COUNT):
            seen: set[int] = set()
            query_ids: list[int] = []
            query_distance_bits: list[int] = []
            previous_distance = -math.inf
            for rank in range(expected_k):
                label, distance_bits = reader.unpack(
                    "qI",
                    (
                        f"recall point {point_index} query {query_index} "
                        f"rank {rank}"
                    ),
                )
                distance = _float32_from_bits(distance_bits)
                require(
                    0 <= label < VECTORS,
                    f"{context} recall result label {label} is out of range",
                )
                require(
                    label not in seen,
                    (
                        f"{context} recall point {point_index} query "
                        f"{query_index} contains duplicate label {label}"
                    ),
                )
                require(
                    math.isfinite(distance),
                    (
                        f"{context} recall point {point_index} query "
                        f"{query_index} contains a non-finite distance"
                    ),
                )
                require(
                    distance >= previous_distance,
                    (
                        f"{context} recall point {point_index} query "
                        f"{query_index} distances are not sorted"
                    ),
                )
                expected_distance = truth["distances"][query_index][label]
                require(
                    math.isclose(
                        distance,
                        expected_distance,
                        rel_tol=2e-5,
                        abs_tol=1e-4,
                    ),
                    (
                        f"{context} recall point {point_index} query "
                        f"{query_index} rank {rank} distance differs from "
                        "independent squared L2"
                    ),
                )
                seen.add(label)
                query_ids.append(label)
                query_distance_bits.append(distance_bits)
                previous_distance = distance
                returned_ids.append(str(label))
                distance_digest.update(struct.pack("<I", distance_bits))
                cutoff = (
                    truth["cutoffs_at_10"][query_index]
                    if expected_k == 10
                    else truth["cutoffs_at_100"][query_index]
                )
                if expected_distance <= cutoff:
                    eligible_hits += 1
            if (expected_k, expected_ef) == (QUERY_K, QUERY_EF_SEARCH):
                benchmark_query_checksums.append(
                    query_result_checksum(
                        query_index,
                        query_ids,
                        query_distance_bits,
                    )
                )
        possible_hits = HELDOUT_QUERY_COUNT * expected_k
        points.append(
            {
                "k": expected_k,
                "ef": expected_ef,
                "eligible_hits": eligible_hits,
                "possible_hits": possible_hits,
                "recall": eligible_hits / float(possible_hits),
            }
        )
        point_details.append(
            {
                "returned_ids": ",".join(returned_ids),
                "returned_distances_sha256": distance_digest.hexdigest(),
            }
        )

    require(
        benchmark_query_checksums == query_per_query_checksums,
        f"{context} timed query checksums differ elementwise from untimed replay",
    )
    require(
        query_benchmark_checksum(benchmark_query_checksums)
        == query_result_checksum_value
        == query_latency_validated_result_checksum
        == query_throughput_validated_result_checksum,
        f"{context} query benchmark checksum differs from recall results",
    )
    reader.finish()
    return (
        {
            "schema_version": AUDIT_SIDECAR_SCHEMA_VERSION,
            "query": query_performance,
            "execution_witness": execution_witness,
            "graph": graph,
            "recall": {
                "query_count": HELDOUT_QUERY_COUNT,
                "query_seed": HELDOUT_QUERY_SEED,
                "queries_sha256": truth["queries_sha256"],
                "truth_sha256": truth["truth_sha256"],
                "points": points,
            },
        },
        {
            "truth_tie_counts_at_10": list(truth["tie_counts_at_10"]),
            "truth_tie_counts_at_100": list(truth["tie_counts_at_100"]),
            "points": point_details,
        },
    )


def expect_normalized_equal(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, dict):
        actual_mapping = expect_mapping(actual, context)
        expect_exact_keys(actual_mapping, expected, context)
        for key, expected_value in expected.items():
            expect_normalized_equal(
                actual_mapping[key],
                expected_value,
                f"{context} {key}",
            )
        return
    if isinstance(expected, list):
        actual_list = expect_list(actual, context)
        require(
            len(actual_list) == len(expected),
            f"{context} length differs from independent reconstruction",
        )
        for index, (actual_value, expected_value) in enumerate(
            zip(actual_list, expected)
        ):
            expect_normalized_equal(
                actual_value,
                expected_value,
                f"{context} item {index}",
            )
        return
    if type(expected) is float:
        expect_close(actual, expected, context)
        return
    require(
        type(actual) is type(expected) and actual == expected,
        (
            f"{context} is {actual!r}, independently reconstructed "
            f"value is {expected!r}"
        ),
    )


def idle_policy(minimum_idle_percent: float) -> str:
    return (
        "default-95-percent"
        if minimum_idle_percent == IDLE_MINIMUM_PERCENT
        else "development-only-override"
    )


def development_idle_limitation(minimum_idle_percent: float) -> str:
    return (
        f"A development-only {minimum_idle_percent:g}% CPU-idle threshold "
        f"replaced the default {IDLE_MINIMUM_PERCENT:g}% threshold; this "
        "campaign is suitable for iteration, not qualification."
    )


DEVELOPMENT_SWAP_LIMITATION = (
    "System-wide swap growth was recorded but not used as a development "
    "acceptance gate; each benchmark process still had to report zero swaps."
)


def parse_cmake_cache(text: str, context: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        name_and_type, value = line.split("=", 1)
        if ":" not in name_and_type:
            continue
        name, _ = name_and_type.split(":", 1)
        require(
            name not in values,
            f"{context} duplicates CMake cache key {name!r} on line {line_number}",
        )
        values[name] = value
    require(values, f"{context} contains no CMake cache entries")
    return values


def configured_compiler_paths(
    cache: Mapping[str, str],
    *,
    context: str,
) -> tuple[str, ...]:
    compilers: list[str] = []
    for key in CMAKE_COMPILER_CACHE_KEYS:
        value = cache.get(key)
        if value is None or value == "" or value.endswith("-NOTFOUND"):
            continue
        require(
            Path(value).is_absolute(),
            f"{context} {key} is not an absolute path",
        )
        if value not in compilers:
            compilers.append(value)
    require(
        cache.get("CMAKE_CXX_COMPILER") in compilers,
        f"{context} has no configured C++ compiler",
    )
    return tuple(compilers)


def configured_scan_deps_paths(
    cache: Mapping[str, str],
    *,
    context: str,
) -> tuple[str, ...]:
    scanners: list[str] = []
    for key in CMAKE_SCAN_DEPS_CACHE_KEYS:
        value = cache.get(key)
        if value is None or value == "" or value.endswith("-NOTFOUND"):
            continue
        require(
            Path(value).is_absolute(),
            f"{context} {key} is not an absolute path",
        )
        if value not in scanners:
            scanners.append(value)
    return tuple(scanners)


def build_setting_names(engine: str) -> tuple[str, ...]:
    require(engine in ("infinity", "faiss"), f"Unknown build engine: {engine}")
    if engine == "infinity":
        return (*COMMON_BUILD_SETTINGS, *INFINITY_PRODUCTION_BUILD_SETTINGS)
    return COMMON_BUILD_SETTINGS


def validate_capture_reference(
    evidence_dir: Path,
    record: dict[str, Any],
    *,
    context: str,
    path_key: str = "captured_path",
    expected_path: str | None = None,
) -> Path:
    path_text, path = relative_evidence_path(
        evidence_dir,
        record.get(path_key),
        f"{context} {path_key}",
    )
    if expected_path is not None:
        require(path_text == expected_path, f"{context} path differs")
    require(path.is_file(), f"{context} captured file is missing")
    recorded_digest = expect_sha256(record.get("sha256"), f"{context} sha256")
    require(
        sha256_file(path) == recorded_digest,
        f"{context} captured-file hash differs",
    )
    return path


def validate_executable_record(
    evidence_dir: Path,
    value: Any,
    context: str,
    *,
    runtime_artifacts: dict[str, str] | None = None,
    require_configured_path_resolution: bool = False,
) -> dict[str, Any]:
    record = expect_mapping(value, context)
    expect_exact_keys(
        record,
        {
            "configured_path",
            "resolved_path",
            "captured_path",
            "sha256",
            "bytes",
            "version_command",
            "version_returncode",
            "version",
        },
        context,
    )
    configured = expect_string(record["configured_path"], f"{context} configured path")
    resolved = expect_string(record["resolved_path"], f"{context} resolved path")
    require(configured != "", f"{context} configured path is empty")
    require(Path(resolved).is_absolute(), f"{context} resolved path is not absolute")
    if require_configured_path_resolution:
        configured_path = Path(configured)
        require(
            configured_path.is_absolute(),
            f"{context} configured path is not absolute",
        )
        try:
            configured_resolved = configured_path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise VerificationError(
                f"{context} configured path cannot be resolved: {error}"
            ) from error
        require(
            str(configured_resolved) == resolved,
            f"{context} configured path resolves to a different executable",
        )
    digest = expect_sha256(record["sha256"], f"{context} sha256")
    size = expect_int(record["bytes"], f"{context} bytes", minimum=1)
    captured_text, captured_path = relative_evidence_path(
        evidence_dir,
        record["captured_path"],
        f"{context} captured path",
    )
    require(
        captured_text == f"provenance/blobs/{digest[:2]}/{digest}",
        f"{context} is not stored at its content-addressed path",
    )
    require(
        captured_path.is_file()
        and captured_path.stat().st_size == size
        and sha256_file(captured_path) == digest,
        f"{context} captured executable differs",
    )
    if runtime_artifacts is not None:
        require(
            runtime_artifacts.get(resolved) == digest,
            f"{context} is absent from runtime artifacts",
        )
    command = expect_list(record["version_command"], f"{context} version command")
    require(
        command
        and command[0] == resolved
        and all(isinstance(item, str) for item in command),
        f"{context} version command does not invoke the resolved executable",
    )
    expect_int(record["version_returncode"], f"{context} version return code")
    require(
        expect_string(record["version"], f"{context} version").strip() != "",
        f"{context} version text is empty",
    )
    return record


def compile_source_paths(value: Any, context: str) -> set[str]:
    entries = expect_list(value, context)
    require(entries, f"{context} is empty")
    paths: set[str] = set()
    for index, entry_value in enumerate(entries):
        entry = expect_mapping(entry_value, f"{context} entry {index}")
        directory = expect_string(
            entry.get("directory"),
            f"{context} entry {index} directory",
        )
        source = expect_string(entry.get("file"), f"{context} entry {index} file")
        require(
            Path(directory).is_absolute(),
            f"{context} entry {index} directory is not absolute",
        )
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = Path(directory) / source_path
        paths.add(
            lexical_absolute_path(
                str(source_path),
                working_directory=Path(directory),
            )
        )
    return paths


def validate_source_records(
    evidence_dir: Path,
    value: Any,
    context: str,
) -> dict[str, dict[str, Any]]:
    records = expect_list(value, context)
    require(records, f"{context} is empty")
    result: dict[str, dict[str, Any]] = {}
    labels: set[str] = set()
    for index, record_value in enumerate(records):
        item_context = f"{context} record {index}"
        record = expect_mapping(record_value, item_context)
        expect_exact_keys(
            record,
            {"path", "absolute_path", "captured_path", "bytes", "sha256"},
            item_context,
        )
        label = expect_string(record["path"], f"{item_context} path")
        absolute = expect_string(
            record["absolute_path"],
            f"{item_context} absolute path",
        )
        require(label != "", f"{item_context} path is empty")
        require(Path(absolute).is_absolute(), f"{item_context} path is not absolute")
        require(label not in labels, f"{context} duplicates source label {label!r}")
        require(
            absolute not in result, f"{context} duplicates source path {absolute!r}"
        )
        labels.add(label)
        size = expect_int(record["bytes"], f"{item_context} bytes", minimum=0)
        digest = expect_sha256(record["sha256"], f"{item_context} sha256")
        captured_text, captured_path = relative_evidence_path(
            evidence_dir,
            record["captured_path"],
            f"{item_context} captured path",
        )
        require(
            captured_text == f"provenance/blobs/{digest[:2]}/{digest}",
            f"{item_context} is not stored at its content-addressed path",
        )
        require(captured_path.is_file(), f"{item_context} captured source is missing")
        require(
            captured_path.stat().st_size == size
            and sha256_file(captured_path) == digest,
            f"{item_context} captured source content differs",
        )
        result[absolute] = record
    require(
        list(result) == sorted(result, key=lambda path: path.encode("utf-8")),
        f"{context} is not in canonical absolute-path order",
    )
    return result


def validate_cross_record_source_identities(
    collections: Iterable[tuple[str, dict[str, dict[str, Any]]]],
) -> None:
    seen: dict[str, tuple[str, tuple[Any, ...]]] = {}
    for collection_name, records in collections:
        for absolute_path, record in records.items():
            identity = (
                record["sha256"],
                record["bytes"],
                record["captured_path"],
            )
            previous = seen.get(absolute_path)
            if previous is None:
                seen[absolute_path] = (collection_name, identity)
                continue
            previous_name, previous_identity = previous
            require(
                identity == previous_identity,
                f"Source {absolute_path!r} has different content in "
                f"{previous_name} and {collection_name}",
            )


def validate_response_file_records(
    evidence_dir: Path,
    value: Any,
    context: str,
) -> dict[tuple[str, str], Path]:
    records = expect_list(value, context)
    result: dict[tuple[str, str], Path] = {}
    order: list[tuple[str, str]] = []
    for index, record_value in enumerate(records):
        item_context = f"{context} record {index}"
        record = expect_mapping(record_value, item_context)
        expect_exact_keys(
            record,
            {
                "path",
                "invocation_working_directory",
                "captured_path",
                "bytes",
                "sha256",
            },
            item_context,
        )
        path = expect_string(record["path"], f"{item_context} path")
        working_directory = expect_string(
            record["invocation_working_directory"],
            f"{item_context} invocation working directory",
        )
        require(Path(path).is_absolute(), f"{item_context} path is not absolute")
        require(
            Path(working_directory).is_absolute(),
            f"{item_context} invocation working directory is not absolute",
        )
        key = (working_directory, path)
        require(key not in result, f"{context} duplicates {key!r}")
        digest = expect_sha256(record["sha256"], f"{item_context} sha256")
        size = expect_int(record["bytes"], f"{item_context} bytes", minimum=0)
        captured_text, captured_path = relative_evidence_path(
            evidence_dir,
            record["captured_path"],
            f"{item_context} captured path",
        )
        require(
            captured_text == f"provenance/blobs/{digest[:2]}/{digest}",
            f"{item_context} is not stored at its content-addressed path",
        )
        require(
            captured_path.is_file()
            and captured_path.stat().st_size == size
            and sha256_file(captured_path) == digest,
            f"{item_context} captured content differs",
        )
        result[key] = captured_path
        order.append(key)
    require(
        order
        == sorted(
            order,
            key=lambda item: (
                item[0].encode("utf-8"),
                item[1].encode("utf-8"),
            ),
        ),
        f"{context} is not in canonical order",
    )
    return result


def compile_entry_arguments(
    value: Any,
    *,
    context: str,
) -> list[str]:
    entry = expect_mapping(value, context)
    has_arguments = "arguments" in entry
    has_command = "command" in entry
    require(
        has_arguments is not has_command,
        f"{context} must contain exactly one command representation",
    )
    if has_arguments:
        arguments = expect_list(entry["arguments"], f"{context} arguments")
        require(
            arguments and all(isinstance(item, str) and item for item in arguments),
            f"{context} arguments are invalid",
        )
        return list(arguments)
    command = expect_string(entry["command"], f"{context} command")
    require(command != "", f"{context} command is empty")
    try:
        arguments = shlex.split(command)
    except ValueError as error:
        raise VerificationError(f"{context} cannot be tokenized: {error}") from error
    require(arguments, f"{context} command is empty")
    return arguments


def expand_response_arguments(
    arguments: list[str],
    *,
    working_directory: Path,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
    context: str,
) -> list[str]:
    require(
        working_directory.is_absolute(),
        f"{context} response-file working directory is not absolute",
    )
    invocation_directory = Path(os.path.abspath(working_directory))
    expanded: list[str] = []

    def visit(tokens: list[str], active: tuple[str, ...]) -> None:
        require(
            len(active) <= MAX_RESPONSE_DEPTH,
            f"{context} exceeds the response-file nesting limit",
        )
        for token in tokens:
            if (
                not token.startswith("@")
                or token == "@"
                or token.startswith(MACHO_DYNAMIC_PATH_PREFIXES)
            ):
                expanded.append(token)
                require(
                    len(expanded) <= MAX_RESPONSE_TOKENS,
                    f"{context} exceeds the expanded response-token limit",
                )
                continue
            response_path = Path(token[1:])
            if not response_path.is_absolute():
                response_path = invocation_directory / response_path
            response_path = Path(
                lexical_absolute_path(
                    str(response_path),
                    working_directory=invocation_directory,
                )
            )
            response_text = str(response_path)
            require(
                response_text not in active,
                f"{context} contains a response-file cycle at {response_text}",
            )
            key = (str(invocation_directory), response_text)
            captured_path = response_files.get(key)
            require(
                captured_path is not None,
                f"{context} response file is not archived: {response_text}",
            )
            data = read_verified_bytes(
                captured_path,
                f"{context} response file {response_text}",
                maximum_bytes=MAX_RESPONSE_FILE_BYTES,
            )
            require(
                len(data) <= MAX_RESPONSE_FILE_BYTES,
                f"{context} response file exceeds the byte limit: {response_text}",
            )
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise VerificationError(
                    f"{context} response file is not UTF-8: {response_text}"
                ) from error
            try:
                nested = shlex.split(text)
            except ValueError as error:
                raise VerificationError(
                    f"{context} response file cannot be tokenized: "
                    f"{response_text}: {error}"
                ) from error
            used_response_files.add(key)
            visit(nested, (*active, response_text))

    visit(arguments, ())
    return expanded


NINJA_DEPS_HEADER = re.compile(
    r"(?P<output>.+): #deps (?P<count>[0-9]+), "
    r"deps mtime (?P<mtime>[0-9]+) \((?P<status>VALID|STALE)\)"
)


def lexical_absolute_path(value: str, *, working_directory: Path) -> str:
    require(value != "" and "\0" not in value and "\n" not in value, "Invalid path")
    path = Path(value)
    if not path.is_absolute():
        path = working_directory / path
    return os.path.abspath(path)


def parse_ninja_deps(
    text: str,
    *,
    build_directory: Path,
    context: str,
) -> dict[str, dict[str, Any]]:
    require("\r" not in text, f"{context} uses non-canonical line endings")
    lines = text.splitlines()
    result: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(lines):
        if lines[index] == "":
            index += 1
            continue
        match = NINJA_DEPS_HEADER.fullmatch(lines[index])
        require(
            match is not None, f"{context} has an invalid header on line {index + 1}"
        )
        output = lexical_absolute_path(
            match.group("output"),
            working_directory=build_directory,
        )
        require(output not in result, f"{context} duplicates output {output}")
        count = int(match.group("count"))
        index += 1
        dependencies: list[str] = []
        for _ in range(count):
            require(
                index < len(lines), f"{context} truncates dependencies for {output}"
            )
            line = lines[index]
            require(
                line.startswith("    ") and line[4:] != "",
                f"{context} has an invalid dependency on line {index + 1}",
            )
            dependencies.append(
                lexical_absolute_path(
                    line[4:],
                    working_directory=build_directory,
                )
            )
            index += 1
        require(
            index == len(lines) or lines[index] == "",
            f"{context} has extra dependencies for {output}",
        )
        result[output] = {
            "output": output,
            "dependency_count": count,
            "deps_mtime": int(match.group("mtime")),
            "status": match.group("status"),
            "dependencies": dependencies,
        }
    require(result, f"{context} is empty")
    return result


def compile_output_lexical_path(
    arguments: list[str],
    *,
    working_directory: Path,
    context: str,
) -> str | None:
    if "-c" not in arguments:
        return None
    output_indices = [
        index for index, argument in enumerate(arguments) if argument == "-o"
    ]
    require(
        len(output_indices) == 1,
        f"{context} must contain exactly one separated -o output",
    )
    output_index = output_indices[0]
    require(output_index + 1 < len(arguments), f"{context} has a dangling -o")
    return lexical_absolute_path(
        arguments[output_index + 1],
        working_directory=working_directory,
    )


def module_output_paths(
    arguments: list[str],
    *,
    working_directory: Path,
    context: str,
) -> list[str]:
    outputs: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-fmodule-output":
            require(
                index + 1 < len(arguments),
                f"{context} has a dangling -fmodule-output",
            )
            value = arguments[index + 1]
            index += 2
        elif argument.startswith("-fmodule-output="):
            value = argument.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        require(value != "", f"{context} has an empty module output")
        outputs.append(
            lexical_absolute_path(value, working_directory=working_directory)
        )
    require(
        len(outputs) == len(set(outputs)),
        f"{context} duplicates a module output",
    )
    return outputs


def compile_direct_inputs(
    entry: dict[str, Any],
    arguments: list[str],
    *,
    working_directory: Path,
    context: str,
) -> list[dict[str, str]]:
    source = Path(expect_string(entry.get("file"), f"{context} source file"))
    if not source.is_absolute():
        source = working_directory / source
    result = [
        {
            "role": "compile-source",
            "path": lexical_absolute_path(
                str(source),
                working_directory=working_directory,
            ),
        }
    ]
    separated = {
        "-include": "forced-include",
        "-include-pch": "forced-precompiled-header",
        "-imacros": "forced-macro-include",
        "-fmodule-map-file": "module-map",
        "-fmodule-file": "module-input",
    }
    joined = {
        "-include=": "forced-include",
        "-include-pch=": "forced-precompiled-header",
        "-imacros=": "forced-macro-include",
        "-fmodule-map-file=": "module-map",
        "-fmodule-file=": "module-input",
    }
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in separated:
            require(index + 1 < len(arguments), f"{context} has dangling {argument}")
            role = separated[argument]
            value = arguments[index + 1]
            index += 2
        else:
            matched = next(
                (
                    (prefix, candidate_role)
                    for prefix, candidate_role in joined.items()
                    if argument.startswith(prefix)
                ),
                None,
            )
            if matched is None:
                index += 1
                continue
            prefix, role = matched
            value = argument[len(prefix) :]
            index += 1
        if role == "module-input" and "=" in value:
            _, value = value.split("=", 1)
        require(value != "", f"{context} has an empty {role} path")
        result.append(
            {
                "role": role,
                "path": lexical_absolute_path(
                    value,
                    working_directory=working_directory,
                ),
            }
        )
    return result


def link_input_kind(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    return {
        ".a": "archive",
        ".bc": "llvm-bitcode",
        ".dylib": "dynamic-library",
        ".o": "object",
        ".so": "shared-library",
        ".tbd": "text-based-stub",
    }.get(suffix)


def framework_search_directories(
    arguments: list[str],
    *,
    working_directory: Path,
    sysroot: Path,
    context: str,
) -> list[Path]:
    directories: list[Path] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-F":
            require(index + 1 < len(arguments), f"{context} has a dangling -F")
            value = arguments[index + 1]
            index += 2
        elif argument.startswith("-F") and argument != "-F":
            value = argument[2:]
            index += 1
        else:
            index += 1
            continue
        directories.append(
            Path(
                lexical_absolute_path(
                    value,
                    working_directory=working_directory,
                )
            )
        )
    directories.append(sysroot / "System" / "Library" / "Frameworks")
    return directories


def library_search_directories(
    arguments: list[str],
    *,
    working_directory: Path,
    sysroot: Path,
    context: str,
) -> list[Path]:
    directories: list[Path] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-L":
            require(index + 1 < len(arguments), f"{context} has a dangling -L")
            value = arguments[index + 1]
            index += 2
        elif argument.startswith("-L") and argument != "-L":
            value = argument[2:]
            index += 1
        else:
            index += 1
            continue
        directories.append(
            Path(
                lexical_absolute_path(
                    value,
                    working_directory=working_directory,
                )
            )
        )
    directories.append(sysroot / "usr" / "lib")
    return directories


def library_search_paths_first(arguments: list[str]) -> bool:
    return any(
        argument == "-search_paths_first"
        or (
            argument.startswith("-Wl,")
            and "-search_paths_first" in argument.split(",")[1:]
        )
        for argument in arguments
    )


def library_input_candidates(
    name: str,
    *,
    search_directories: list[Path],
    search_paths_first: bool,
    context: str,
) -> list[Path]:
    require(
        name != ""
        and not name.startswith(":")
        and "/" not in name
        and "\0" not in name,
        f"{context} has an invalid library name {name!r}",
    )
    dynamic_suffixes = (".tbd", ".dylib", ".so")
    if search_paths_first:
        return [
            directory / f"lib{name}{suffix}"
            for directory in search_directories
            for suffix in (*dynamic_suffixes, ".a")
        ]
    return [
        directory / f"lib{name}{suffix}"
        for suffix in dynamic_suffixes
        for directory in search_directories
    ] + [
        directory / f"lib{name}.a"
        for directory in search_directories
    ]


def resolve_archived_library_input(
    name: str,
    *,
    search_directories: list[Path],
    search_paths_first: bool,
    available_paths: set[str],
    context: str,
) -> str:
    matches = [
        os.path.abspath(candidate)
        for candidate in library_input_candidates(
            name,
            search_directories=search_directories,
            search_paths_first=search_paths_first,
            context=context,
        )
        if os.path.abspath(candidate) in available_paths
    ]
    require(
        len(matches) == 1,
        f"{context} cannot uniquely resolve archived library {name!r}",
    )
    return matches[0]


def resolve_archived_framework_input(
    name: str,
    *,
    search_directories: list[Path],
    available_paths: set[str],
    context: str,
) -> str:
    require(
        name != "" and "/" not in name and "\0" not in name,
        f"{context} has an invalid framework name {name!r}",
    )
    candidates: list[str] = []
    for directory in search_directories:
        framework = directory / f"{name}.framework"
        candidates.extend(
            [
                os.path.abspath(framework / f"{name}.tbd"),
                os.path.abspath(framework / name),
            ]
        )
    matches = [path for path in candidates if path in available_paths]
    require(
        len(matches) == 1,
        f"{context} cannot uniquely resolve archived framework {name}",
    )
    return matches[0]


def derive_link_inputs(
    arguments: list[str],
    *,
    working_directory: Path,
    sysroot: Path,
    available_paths: set[str],
    context: str,
) -> list[dict[str, str]]:
    require(arguments, f"{context} is empty")
    framework_directories = framework_search_directories(
        arguments,
        working_directory=working_directory,
        sysroot=sysroot,
        context=context,
    )
    library_directories = library_search_directories(
        arguments,
        working_directory=working_directory,
        sysroot=sysroot,
        context=context,
    )
    search_paths_first = library_search_paths_first(arguments)
    inputs: list[dict[str, str]] = []
    value_options = {
        "-arch",
        "-compatibility_version",
        "-current_version",
        "-F",
        "-install_name",
        "-isysroot",
        "-L",
        "-o",
        "-rpath",
        "-syslibroot",
    }
    framework_options = {"-framework", "-weak_framework"}
    linker_file_options = {
        "-force_load": "force-load-input",
        "-order_file": "link-order-file",
        "-exported_symbols_list": "exported-symbols-list",
    }
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument in framework_options:
            require(index + 1 < len(arguments), f"{context} has dangling {argument}")
            name = arguments[index + 1]
            inputs.append(
                {
                    "role": "framework-input",
                    "path": resolve_archived_framework_input(
                        name,
                        search_directories=framework_directories,
                        available_paths=available_paths,
                        context=context,
                    ),
                }
            )
            index += 2
            continue
        if argument in linker_file_options:
            require(index + 1 < len(arguments), f"{context} has dangling {argument}")
            inputs.append(
                {
                    "role": linker_file_options[argument],
                    "path": lexical_absolute_path(
                        arguments[index + 1],
                        working_directory=working_directory,
                    ),
                }
            )
            index += 2
            continue
        if argument in value_options:
            require(index + 1 < len(arguments), f"{context} has dangling {argument}")
            index += 2
            continue
        if argument == "-Xlinker":
            raise VerificationError(f"{context} uses unsupported -Xlinker syntax")
        if argument.startswith("-Wl,"):
            pieces = argument.split(",")[1:]
            require(
                pieces and all(piece != "" for piece in pieces),
                f"{context} has an empty -Wl operand",
            )
            file_flags = {
                "-force_load": "force-load-input",
                "-order_file": "link-order-file",
                "-exported_symbols_list": "exported-symbols-list",
                "-unexported_symbols_list": "unexported-symbols-list",
                "-reexported_symbols_list": "reexported-symbols-list",
                "-alias_list": "alias-list",
                "-add_ast_path": "ast-input",
                "-bundle_loader": "bundle-loader-input",
                "-dtrace": "dtrace-script",
                "-reexport_library": "reexport-library-input",
                "-weak_library": "weak-library-input",
                "-lazy_library": "lazy-library-input",
                "-upward_library": "upward-library-input",
                "-segaddr_table": "segment-address-table",
            }
            no_value_flags = {
                "-all_load",
                "-bind_at_load",
                "-dead_strip",
                "-fatal_warnings",
                "-flat_namespace",
                "-headerpad_max_install_names",
                "-no_dead_strip_inits_and_terms",
                "-no_deduplicate",
                "-no_pie",
                "-noall_load",
                "-no_warn_duplicate_libraries",
                "-ObjC",
                "-pie",
                "-search_dylibs_first",
                "-search_paths_first",
                "-twolevel_namespace",
            }
            value_flags = {
                "-compatibility_version",
                "-current_version",
                "-headerpad",
                "-install_name",
                "-macos_version_min",
                "-multiply_defined",
                "-rpath",
                "-sdk_version",
                "-syslibroot",
                "-undefined",
            }
            offset = 0
            while offset < len(pieces):
                piece = pieces[offset]
                if piece == "-filelist":
                    raise VerificationError(
                        f"{context} uses unsupported linker filelist"
                    )
                if piece in no_value_flags:
                    offset += 1
                    continue
                if piece in value_flags:
                    require(
                        offset + 1 < len(pieces),
                        f"{context} has a dangling {piece}",
                    )
                    offset += 2
                    continue
                if piece in framework_options:
                    require(
                        offset + 1 < len(pieces),
                        f"{context} has a dangling {piece}",
                    )
                    inputs.append(
                        {
                            "role": "framework-input",
                            "path": resolve_archived_framework_input(
                                pieces[offset + 1],
                                search_directories=framework_directories,
                                available_paths=available_paths,
                                context=context,
                            ),
                        }
                    )
                    offset += 2
                    continue
                if piece in file_flags:
                    require(
                        offset + 1 < len(pieces) and pieces[offset + 1] != "",
                        f"{context} has a dangling {piece}",
                    )
                    inputs.append(
                        {
                            "role": file_flags[piece],
                            "path": lexical_absolute_path(
                                pieces[offset + 1],
                                working_directory=working_directory,
                            ),
                        }
                    )
                    offset += 2
                    continue
                if piece in {"-sectcreate", "-sectorder"}:
                    require(
                        offset + 3 < len(pieces),
                        f"{context} has a dangling {piece}",
                    )
                    inputs.append(
                        {
                            "role": (
                                "section-content-input"
                                if piece == "-sectcreate"
                                else "section-order-input"
                            ),
                            "path": lexical_absolute_path(
                                pieces[offset + 3],
                                working_directory=working_directory,
                            ),
                        }
                    )
                    offset += 4
                    continue
                if piece == "-l":
                    require(
                        offset + 1 < len(pieces),
                        f"{context} has a dangling -l",
                    )
                    library_name = pieces[offset + 1]
                    offset += 2
                elif piece.startswith("-l"):
                    library_name = piece[2:]
                    offset += 1
                else:
                    library_name = None
                if library_name is not None:
                    path = resolve_archived_library_input(
                        library_name,
                        search_directories=library_directories,
                        search_paths_first=search_paths_first,
                        available_paths=available_paths,
                        context=context,
                    )
                    kind = link_input_kind(path)
                    require(
                        kind is not None,
                        f"{context} resolved an unsupported library {path}",
                    )
                    inputs.append(
                        {
                            "role": f"link-library-{kind}",
                            "path": path,
                        }
                    )
                    continue
                if piece.startswith("-"):
                    raise VerificationError(
                        f"{context} uses unsupported -Wl option {piece!r}"
                    )
                kind = link_input_kind(piece)
                require(
                    kind is not None,
                    f"{context} has unsupported -Wl operand {piece!r}",
                )
                inputs.append(
                    {
                        "role": f"link-{kind}",
                        "path": lexical_absolute_path(
                            piece,
                            working_directory=working_directory,
                        ),
                    }
                )
                offset += 1
            index += 1
            continue
        if argument.startswith(("-F", "-L")):
            index += 1
            continue
        if argument == "-l":
            require(index + 1 < len(arguments), f"{context} has a dangling -l")
            library_name = arguments[index + 1]
            index += 2
        elif argument.startswith("-l"):
            library_name = argument[2:]
            index += 1
        else:
            library_name = None
        if library_name is not None:
            path = resolve_archived_library_input(
                library_name,
                search_directories=library_directories,
                search_paths_first=search_paths_first,
                available_paths=available_paths,
                context=context,
            )
            kind = link_input_kind(path)
            require(
                kind is not None,
                f"{context} resolved an unsupported library {path}",
            )
            inputs.append(
                {
                    "role": f"link-library-{kind}",
                    "path": path,
                }
            )
            continue
        if argument.startswith("-") or argument.startswith(MACHO_DYNAMIC_PATH_PREFIXES):
            index += 1
            continue
        kind = link_input_kind(argument)
        require(kind is not None, f"{context} has unclassified operand {argument!r}")
        inputs.append(
            {
                "role": f"link-{kind}",
                "path": lexical_absolute_path(
                    argument,
                    working_directory=working_directory,
                ),
            }
        )
        index += 1
    require(inputs, f"{context} has no explicit linker inputs")
    return inputs


def closure_file_role(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pcm":
        return "generated-module"
    if suffix == ".o":
        return "object"
    if suffix == ".a":
        return "archive"
    if suffix in (".dylib", ".so"):
        return "dynamic-library"
    if suffix == ".tbd":
        return "text-based-stub"
    return "file"


def ninja_target_for_output(
    output: Path,
    *,
    build_directory: Path,
    context: str,
) -> str:
    output_path = Path(
        lexical_absolute_path(
            str(output),
            working_directory=build_directory,
        )
    )
    build_path = Path(
        lexical_absolute_path(
            str(build_directory),
            working_directory=build_directory,
        )
    )
    try:
        relative = output_path.relative_to(build_path)
    except ValueError as error:
        raise VerificationError(
            f"{context} output is outside its Ninja build directory: {output_path}"
        ) from error
    require(relative.parts, f"{context} output is the build directory")
    require(
        all(part not in ("", ".", "..") for part in relative.parts),
        f"{context} output has an invalid Ninja target path",
    )
    return relative.as_posix()


def parse_ninja_targets(
    text: str,
    *,
    build_directory: Path,
    context: str,
) -> dict[str, str]:
    targets: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(line != "", f"{context} line {line_number} is empty")
        pieces = line.rsplit(": ", 1)
        require(
            len(pieces) == 2 and pieces[0] != "" and pieces[1] != "",
            f"{context} line {line_number} is malformed",
        )
        output = lexical_absolute_path(
            pieces[0],
            working_directory=build_directory,
        )
        previous = targets.get(output)
        require(
            previous is None or previous == pieces[1],
            f"{context} has conflicting rules for target {output}",
        )
        targets[output] = pieces[1]
    require(targets, f"{context} is empty")
    return targets


def parse_ninja_query(
    text: str,
    *,
    build_directory: Path,
    context: str,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line and not line.startswith(" "):
            require(
                line.endswith(":") and line != ":",
                f"{context} line {line_number} has a malformed target",
            )
            if current is not None:
                require(
                    current["rule"] is not None,
                    f"{context} target {current['output']} has no producer rule",
                )
                nodes.append(current)
            current = {
                "output": lexical_absolute_path(
                    line[:-1],
                    working_directory=build_directory,
                ),
                "rule": None,
                "inputs": [],
                "outputs": [],
            }
            section = None
            continue
        require(current is not None, f"{context} line {line_number} precedes a target")
        if line.startswith("  input: "):
            require(section is None, f"{context} target repeats its input section")
            current["rule"] = line[len("  input: ") :]
            require(current["rule"] != "", f"{context} target has an empty rule")
            section = "inputs"
            continue
        if line == "  outputs:":
            require(
                current["rule"] is not None and section == "inputs",
                f"{context} target has an out-of-order outputs section",
            )
            section = "outputs"
            continue
        require(
            line.startswith("    ") and section in {"inputs", "outputs"},
            f"{context} line {line_number} has invalid indentation",
        )
        value = line[4:]
        require(value != "", f"{context} line {line_number} has an empty path")
        if section == "inputs":
            if value.startswith("|| "):
                edge = "order-only"
                value = value[3:]
            elif value.startswith("| "):
                edge = "implicit"
                value = value[2:]
            else:
                edge = "explicit"
            require(value != "", f"{context} line {line_number} has an empty input")
            current["inputs"].append(
                {
                    "edge": edge,
                    "path": lexical_absolute_path(
                        value,
                        working_directory=build_directory,
                    ),
                }
            )
        else:
            if value.startswith("|| "):
                value = value[3:]
            elif value.startswith("| "):
                value = value[2:]
            require(value != "", f"{context} line {line_number} has an empty output")
            current["outputs"].append(
                lexical_absolute_path(
                    value,
                    working_directory=build_directory,
                )
            )
    if current is not None:
        require(
            current["rule"] is not None,
            f"{context} target {current['output']} has no producer rule",
        )
        nodes.append(current)
    require(nodes, f"{context} is empty")
    outputs = [str(node["output"]) for node in nodes]
    require(len(outputs) == len(set(outputs)), f"{context} repeats a queried target")
    return sorted(nodes, key=lambda node: str(node["output"]).encode("utf-8"))


def parse_ninja_clean_plan(
    text: str,
    *,
    build_directory: Path,
    context: str,
    requested_target: str | None = None,
    requested_targets: Iterable[str] | None = None,
) -> list[str]:
    if requested_targets is None:
        require(requested_target is not None, f"{context} has no requested target")
        normalized_targets = [requested_target]
    else:
        require(
            requested_target is None,
            f"{context} mixes single-product and product target arguments",
        )
        normalized_targets = list(requested_targets)
    require(
        normalized_targets
        and all(
            isinstance(target, str) and target != ""
            for target in normalized_targets
        )
        and len(normalized_targets) == len(set(normalized_targets)),
        f"{context} requested targets are invalid",
    )
    require(
        text.endswith("\n") and "\r" not in text,
        f"{context} is not canonical newline-terminated text",
    )
    lines = text.splitlines()
    require(
        len(lines) >= 3 and lines[0] == "Cleaning...",
        f"{context} has an unexpected header",
    )
    count_match = re.fullmatch(r"(0|[1-9][0-9]*) files\.", lines[-1])
    require(count_match is not None, f"{context} has an invalid file count")
    require(
        build_directory.is_absolute()
        and os.path.abspath(str(build_directory)) == str(build_directory),
        f"{context} build directory is not canonical absolute",
    )

    paths: list[str] = []
    seen: set[str] = set()
    targets: list[str] = []
    for line in lines[1:-1]:
        if line.startswith("Target "):
            targets.append(line.removeprefix("Target "))
            continue
        require(
            line.startswith("Remove "),
            f"{context} has an unexpected line",
        )
        path_text = line.removeprefix("Remove ")
        require(
            path_text != ""
            and all(
                ord(character) >= 32 and ord(character) != 127
                for character in path_text
            ),
            f"{context} has an invalid removal path",
        )
        path = Path(path_text)
        require(
            not path.is_absolute()
            and path.as_posix() == path_text
            and all(part not in ("", ".", "..") for part in path.parts),
            f"{context} removal path is not canonical relative: {path_text}",
        )
        absolute_path = str(build_directory / path)
        require(
            absolute_path not in seen,
            f"{context} repeats removal path: {path_text}",
        )
        seen.add(absolute_path)
        paths.append(absolute_path)

    require(targets == normalized_targets, f"{context} target list differs")
    require(
        len(paths) == int(count_match.group(1)),
        f"{context} removal count differs",
    )
    return paths


def graph_proven_link_outputs(
    *,
    nodes: list[dict[str, Any]],
    selected_output: str,
    link_inputs: list[dict[str, str]],
    known_derived_outputs: set[str],
    build_directory: Path,
    context: str,
) -> list[str]:
    by_output = {str(node["output"]): node for node in nodes}
    selected = by_output.get(selected_output)
    require(selected is not None, f"{context} omits the selected output")
    require(selected["rule"] != "phony", f"{context} selected output is phony")
    selected_material_inputs = {
        str(item["path"])
        for item in selected["inputs"]
        if item["edge"] in {"explicit", "implicit"}
    }
    build_prefix = f"{build_directory}/"
    proven: list[str] = []
    for link_input in link_inputs:
        path = str(link_input["path"])
        if not (path == str(build_directory) or path.startswith(build_prefix)):
            continue
        if path not in selected_material_inputs:
            continue
        producer = by_output.get(path)
        if producer is None or producer["rule"] == "phony":
            continue
        material_inputs = {
            str(item["path"])
            for item in producer["inputs"]
            if item["edge"] in {"explicit", "implicit"}
        }
        if not material_inputs or not material_inputs <= known_derived_outputs:
            continue
        proven.append(path)
    return sorted(set(proven), key=lambda value: value.encode("utf-8"))


def normalized_compile_arguments(arguments: list[str], context: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-MD", "-MMD"):
            index += 1
            continue
        if argument in ("-MF", "-MT", "-MQ"):
            require(index + 1 < len(arguments), f"{context} has dangling {argument}")
            index += 2
            continue
        result.append(argument)
        index += 1
    return result


def validate_release_arguments(
    arguments: list[str],
    cache: dict[str, str],
    *,
    context: str,
) -> None:
    optimization = [
        argument
        for argument in arguments
        if re.fullmatch(r"-O(?:0|1|2|3|s|z|g|fast)?", argument)
    ]
    require(
        optimization and optimization[-1] == "-O3",
        f"{context} effective optimization is not -O3",
    )

    ndebug_defined: bool | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        macro: str | None = None
        defined: bool | None = None
        if argument in ("-D", "-U"):
            require(index + 1 < len(arguments), f"{context} has dangling {argument}")
            macro = arguments[index + 1]
            defined = argument == "-D"
            index += 1
        elif argument.startswith("-D"):
            macro = argument[2:]
            defined = True
        elif argument.startswith("-U"):
            macro = argument[2:]
            defined = False
        if macro is not None and macro.split("=", 1)[0] == "NDEBUG":
            ndebug_defined = defined
        index += 1
    require(ndebug_defined is True, f"{context} does not leave NDEBUG defined")

    architecture_values = [
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == "-arch"
    ]
    require(
        architecture_values == ["arm64"],
        f"{context} does not compile for exactly one arm64 architecture",
    )
    sysroot_values = [
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == "-isysroot"
    ]
    require(
        sysroot_values == [cache["CMAKE_OSX_SYSROOT"]],
        f"{context} sysroot differs from CMake",
    )
    deployment_flags = [
        argument
        for argument in arguments
        if argument.startswith("-mmacosx-version-min=")
    ]
    require(
        deployment_flags
        == [f"-mmacosx-version-min={cache['CMAKE_OSX_DEPLOYMENT_TARGET']}"],
        f"{context} deployment target differs from CMake",
    )

    lowered = [argument.lower() for argument in arguments]
    forbidden_prefixes = (
        "-flto",
        "-fprofile",
        "-ffast-math",
        "-funsafe-math",
        "-ffp-model=fast",
    )
    require("-ofast" not in lowered, f"{context} uses forbidden -Ofast")
    require(
        not any(argument.startswith(forbidden_prefixes) for argument in lowered),
        f"{context} uses a forbidden optimization policy",
    )
    for index, argument in enumerate(lowered):
        if argument in ("-march", "-mcpu"):
            require(index + 1 < len(lowered), f"{context} has dangling {argument}")
            value = lowered[index + 1]
        elif argument.startswith(("-march=", "-mcpu=")):
            value = argument.split("=", 1)[1]
        else:
            continue
        require(value != "native", f"{context} uses a native CPU target")
        require(
            re.fullmatch(r"(?:apple-|m[1-9]).*", value) is None,
            f"{context} uses an Apple-specific CPU target",
        )


def validate_release_compile_inputs(
    cache: dict[str, str],
    compile_entries: list[Any],
    *,
    context: str,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
) -> None:
    require(cache.get("CMAKE_BUILD_TYPE") == "Release", f"{context} is not Release")
    require(
        cache.get("CMAKE_OSX_ARCHITECTURES") == "arm64",
        f"{context} is not fixed to arm64",
    )
    require(
        cache.get("CMAKE_OSX_DEPLOYMENT_TARGET") == "14.0",
        f"{context} deployment target is not 14.0",
    )
    require(cache.get("CMAKE_OSX_SYSROOT", "") != "", f"{context} has no sysroot")
    require(compile_entries, f"{context} compile database is empty")
    compilers = set(configured_compiler_paths(cache, context=context))
    for index, entry in enumerate(compile_entries):
        entry_context = f"{context} compile entry {index}"
        raw_arguments = compile_entry_arguments(entry, context=entry_context)
        directory = Path(
            expect_string(
                expect_mapping(entry, entry_context).get("directory"),
                f"{entry_context} directory",
            )
        )
        require(directory.is_absolute(), f"{entry_context} directory is not absolute")
        arguments = expand_response_arguments(
            raw_arguments,
            working_directory=directory,
            response_files=response_files,
            used_response_files=used_response_files,
            context=entry_context,
        )
        require(
            arguments[0] in compilers,
            f"{entry_context} uses a different compiler",
        )
        validate_release_arguments(arguments, cache, context=entry_context)


def validate_required_compile_definition(
    compile_entries: list[Any],
    *,
    name: str,
    value: str,
    context: str,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
) -> None:
    expected = (True, value)
    for index, entry_value in enumerate(compile_entries):
        entry_context = f"{context} compile entry {index}"
        entry = expect_mapping(entry_value, entry_context)
        directory = Path(
            expect_string(entry.get("directory"), f"{entry_context} directory")
        )
        require(directory.is_absolute(), f"{entry_context} directory is not absolute")
        arguments = expand_response_arguments(
            compile_entry_arguments(entry, context=entry_context),
            working_directory=directory,
            response_files=response_files,
            used_response_files=used_response_files,
            context=entry_context,
        )
        operations: list[tuple[bool, str | None]] = []
        argument_index = 0
        while argument_index < len(arguments):
            argument = arguments[argument_index]
            macro: str | None = None
            defined: bool | None = None
            if argument in ("-D", "-U"):
                require(
                    argument_index + 1 < len(arguments),
                    f"{entry_context} has dangling {argument}",
                )
                macro = arguments[argument_index + 1]
                defined = argument == "-D"
                argument_index += 1
            elif argument.startswith("-D"):
                macro = argument[2:]
                defined = True
            elif argument.startswith("-U"):
                macro = argument[2:]
                defined = False
            if macro is not None:
                macro_name, separator, macro_value = macro.partition("=")
                if macro_name == name:
                    operations.append(
                        (
                            bool(defined),
                            macro_value if separator == "=" else None,
                        )
                    )
            argument_index += 1
        require(
            operations == [expected],
            f"{entry_context} must define exactly {name}={value}",
        )


def compile_output_path(
    arguments: list[str],
    *,
    working_directory: Path,
    context: str,
) -> Path | None:
    if "-c" not in arguments:
        return None
    output_indices = [
        index for index, argument in enumerate(arguments) if argument == "-o"
    ]
    require(
        len(output_indices) == 1,
        f"{context} must contain exactly one separated -o output",
    )
    output_index = output_indices[0]
    require(output_index + 1 < len(arguments), f"{context} has a dangling -o")
    output = Path(arguments[output_index + 1])
    if not output.is_absolute():
        output = working_directory / output
    return Path(
        lexical_absolute_path(
            str(output),
            working_directory=working_directory,
        )
    )


def compiler_command_slices(
    tokens: list[str],
    *,
    compiler_tokens: set[str],
    scan_deps_tokens: set[str] | None = None,
    context: str,
) -> list[list[str]]:
    scan_deps_tokens = scan_deps_tokens or set()
    separators = {"&&", "||", ";", ">", ">>"}
    invocations: list[list[str]] = []
    segment_start = 0
    for token_index in range(len(tokens) + 1):
        if token_index != len(tokens) and tokens[token_index] not in separators:
            continue
        segment = tokens[segment_start:token_index]
        segment_start = token_index + 1
        if not segment:
            continue
        driver_positions = [
            index
            for index, token in enumerate(segment)
            if token in compiler_tokens
        ]
        scanner_positions = [
            index
            for index, token in enumerate(segment)
            if token in scan_deps_tokens
        ]
        require(
            len(scanner_positions) <= 1,
            f"{context} contains multiple configured dependency scanners",
        )
        if scanner_positions:
            require(
                len(driver_positions) == 1,
                f"{context} dependency scan does not use exactly one configured compiler",
            )
            driver_position = driver_positions[0]
            driver_arguments = segment[driver_position:]
            compile_positions = [
                index
                for index, argument in enumerate(driver_arguments)
                if argument == "-c"
            ]
            output_positions = [
                index
                for index, argument in enumerate(driver_arguments)
                if argument == "-o"
            ]
            output_is_valid = (
                len(output_positions) == 1
                and output_positions[0] + 1 < len(driver_arguments)
                and driver_arguments[output_positions[0] + 1] != ""
                and not driver_arguments[output_positions[0] + 1].startswith("-")
            )
            require(
                scanner_positions == [0]
                and segment[:driver_position]
                == [segment[0], "-format=p1689", "--"]
                and len(compile_positions) == 1
                and output_is_valid,
                f"{context} has a malformed configured dependency scan",
            )
            continue
        is_compile_action = "-c" in segment and "-o" in segment
        require(
            not is_compile_action or len(driver_positions) == 1,
            f"{context} compile action does not use exactly one configured compiler",
        )
        require(
            len(driver_positions) <= 1,
            f"{context} contains multiple configured compiler tokens",
        )
        if driver_positions:
            require(
                driver_positions == [0],
                f"{context} configured compiler has an unsupported wrapper",
            )
            invocations.append(segment[driver_positions[0] :])
    return invocations


def canonical_compile_entry_union(
    products: Iterable[Mapping[str, Any]],
    *,
    combined_entries: list[Any] | None = None,
    context: str,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
) -> list[Any]:
    def index_entries(
        groups: Iterable[tuple[str, list[Any]]],
        *,
        scope: str,
    ) -> dict[Path, tuple[list[str], Any]]:
        indexed: dict[Path, tuple[list[str], Any]] = {}
        for group_name, entries in groups:
            require(entries, f"{scope} {group_name} is empty")
            for index, entry_value in enumerate(entries):
                entry_context = f"{scope} {group_name} entry {index}"
                entry = expect_mapping(entry_value, entry_context)
                directory = Path(
                    expect_string(
                        entry.get("directory"),
                        f"{entry_context} directory",
                    )
                )
                require(
                    directory.is_absolute(),
                    f"{entry_context} directory is not absolute",
                )
                arguments = expand_response_arguments(
                    compile_entry_arguments(entry, context=entry_context),
                    working_directory=directory,
                    response_files=response_files,
                    used_response_files=used_response_files,
                    context=entry_context,
                )
                output = compile_output_path(
                    arguments,
                    working_directory=directory,
                    context=entry_context,
                )
                require(output is not None, f"{entry_context} is not a compile")
                previous = indexed.get(output)
                require(
                    previous is None or previous[0] == arguments,
                    f"{scope} has conflicting compile entries for {output}",
                )
                indexed.setdefault(output, (arguments, entry_value))
        return indexed

    product_groups: list[tuple[str, list[Any]]] = []
    names: set[str] = set()
    for index, product_value in enumerate(products):
        product_context = f"{context} product {index}"
        product = expect_mapping(product_value, product_context)
        name = expect_string(product.get("name"), f"{product_context} name")
        require(name not in names, f"{product_context} name is duplicated")
        entries = expect_list(
            product.get("compile_entries"),
            f"{product_context} compile entries",
        )
        names.add(name)
        product_groups.append((name, entries))
    require(product_groups, f"{context} has no product compile sets")
    product_union = index_entries(
        product_groups,
        scope=f"{context} product union",
    )
    if combined_entries is not None:
        combined_union = index_entries(
            [("combined capture", combined_entries)],
            scope=f"{context} combined union",
        )
        product_actions = {
            output: arguments
            for output, (arguments, _) in product_union.items()
        }
        combined_actions = {
            output: arguments
            for output, (arguments, _) in combined_union.items()
        }
        require(
            product_actions == combined_actions,
            f"{context} canonical product union differs from combined capture",
        )
    return [
        product_union[output][1]
        for output in sorted(
            product_union,
            key=lambda value: str(value).encode("utf-8"),
        )
    ]


def validate_ninja_commands(
    text: str,
    *,
    cache: dict[str, str],
    build_directory: Path,
    context: str,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
    products: Iterable[Mapping[str, Any]] | None = None,
    compile_entries: list[Any] | None = None,
    expected_output: Path | None = None,
) -> list[str] | list[list[str]]:
    product_mode = products is not None
    if not product_mode:
        require(
            compile_entries is not None and expected_output is not None,
            f"{context} single-product command arguments are incomplete",
        )
        command_products: list[Mapping[str, Any]] = [
            {
                "name": "benchmark",
                "expected_output": expected_output,
                "compile_entries": compile_entries,
            }
        ]
    else:
        require(
            compile_entries is None and expected_output is None,
            f"{context} mixes single-product and ordered-product command arguments",
        )
        command_products = list(products)
    require(command_products, f"{context} has no command products")

    compiler_tokens = set(configured_compiler_paths(cache, context=context))
    scan_deps_tokens = set(configured_scan_deps_paths(cache, context=context))
    expected_product_outputs: list[Path] = []
    product_names: set[str] = set()
    expected_by_output: dict[Path, list[str]] = {}
    expected_exact_by_output: dict[Path, list[str]] = {}
    for product_index, product_value in enumerate(command_products):
        product_context = f"{context} product {product_index}"
        product = expect_mapping(product_value, product_context)
        expect_exact_keys(
            product,
            {"name", "expected_output", "compile_entries"},
            product_context,
        )
        name = expect_string(product["name"], f"{product_context} name")
        require(
            name not in product_names,
            f"{context} duplicates product name {name}",
        )
        product_names.add(name)
        product_output = Path(
            lexical_absolute_path(
                str(product["expected_output"]),
                working_directory=build_directory,
            )
        )
        require(
            product_output not in expected_product_outputs,
            f"{context} duplicates expected output {product_output}",
        )
        expected_product_outputs.append(product_output)
        product_entries = expect_list(
            product["compile_entries"],
            f"{product_context} compile entries",
        )
        require(product_entries, f"{product_context} compile entries are empty")
        product_compile_arguments: dict[Path, list[str]] = {}
        for index, entry in enumerate(product_entries):
            entry_context = f"{product_context} compile entry {index}"
            raw_arguments = compile_entry_arguments(entry, context=entry_context)
            directory = Path(
                expect_string(
                    expect_mapping(entry, entry_context).get("directory"),
                    f"{entry_context} directory",
                )
            )
            require(
                directory.is_absolute(),
                f"{entry_context} directory is not absolute",
            )
            arguments = expand_response_arguments(
                raw_arguments,
                working_directory=directory,
                response_files=response_files,
                used_response_files=used_response_files,
                context=entry_context,
            )
            output = compile_output_path(
                arguments,
                working_directory=directory,
                context=entry_context,
            )
            require(output is not None, f"{entry_context} is not a compile")
            previous_product_arguments = product_compile_arguments.get(output)
            require(
                previous_product_arguments is None
                or previous_product_arguments == arguments,
                f"{product_context} has conflicting duplicate commands for {output}",
            )
            product_compile_arguments.setdefault(output, arguments)
            normalized_arguments = normalized_compile_arguments(
                arguments,
                entry_context,
            )
            previous_exact_arguments = expected_exact_by_output.get(output)
            require(
                previous_exact_arguments is None
                or previous_exact_arguments == arguments,
                f"{context} has conflicting compile commands for {output}",
            )
            expected_exact_by_output.setdefault(output, arguments)
            previous_arguments = expected_by_output.get(output)
            require(
                previous_arguments is None
                or previous_arguments == normalized_arguments,
                f"{context} has conflicting compile commands for {output}",
            )
            expected_by_output.setdefault(output, normalized_arguments)
    observed_outputs: set[Path] = set()
    observed_by_output: dict[Path, list[list[str]]] = {}
    observed_exact_by_output: dict[Path, list[str]] = {}
    matching_link_arguments: dict[Path, list[list[str]]] = {
        output: [] for output in expected_product_outputs
    }
    invocation_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise VerificationError(
                f"{context} line {line_number} cannot be tokenized: {error}"
            ) from error
        for command_tokens in compiler_command_slices(
            tokens,
            compiler_tokens=compiler_tokens,
            scan_deps_tokens=scan_deps_tokens,
            context=f"{context} line {line_number}",
        ):
            invocation_context = f"{context} compiler invocation {invocation_count + 1}"
            arguments = expand_response_arguments(
                command_tokens,
                working_directory=build_directory,
                response_files=response_files,
                used_response_files=used_response_files,
                context=invocation_context,
            )
            invocation_count += 1
            validate_release_arguments(
                arguments,
                cache,
                context=invocation_context,
            )
            output = compile_output_path(
                arguments,
                working_directory=build_directory,
                context=f"{context} compiler invocation {invocation_count}",
            )
            if output is not None:
                previous_arguments = observed_exact_by_output.get(output)
                require(
                    previous_arguments is None or previous_arguments == arguments,
                    f"{context} has conflicting compiler invocations for {output}",
                )
                observed_exact_by_output.setdefault(output, arguments)
                observed_outputs.add(output)
                observed_by_output.setdefault(output, []).append(
                    normalized_compile_arguments(arguments, invocation_context)
                )
            if "-c" not in arguments and "-o" in arguments:
                output_indices = [
                    index
                    for index, argument in enumerate(arguments)
                    if argument == "-o"
                ]
                require(
                    len(output_indices) == 1,
                    f"{context} link invocation must contain exactly one -o",
                )
                output_index = output_indices[0]
                require(
                    output_index + 1 < len(arguments),
                    f"{context} link invocation has a dangling -o",
                )
                observed_output = Path(
                    lexical_absolute_path(
                        arguments[output_index + 1],
                        working_directory=build_directory,
                    )
                )
                if observed_output in matching_link_arguments:
                    matching_link_arguments[observed_output].append(arguments)
    require(invocation_count > 0, f"{context} contains no compiler invocation")
    require(
        observed_outputs == set(expected_by_output),
        f"{context} does not exactly cover the compile database object outputs",
    )
    for output, expected_arguments in expected_by_output.items():
        require(
            expected_arguments in observed_by_output.get(output, []),
            f"{context} has no exact compiler invocation for {output}",
        )
    for output in expected_product_outputs:
        require(
            len(matching_link_arguments[output]) == 1,
            f"{context} must contain exactly one expected link command for {output}",
        )
    ordered_links = [
        matching_link_arguments[output][0] for output in expected_product_outputs
    ]
    return ordered_links if product_mode else ordered_links[0]


def normalize_build_products(
    *,
    build_directory: Path,
    context: str,
    products: Iterable[Mapping[str, Any]] | None = None,
    requested_target: str | None = None,
    expected_output: str | None = None,
    compile_entries: list[Any] | None = None,
    link_arguments: list[str] | None = None,
) -> list[dict[str, Any]]:
    if products is None:
        require(
            requested_target is not None
            and expected_output is not None
            and compile_entries is not None
            and link_arguments is not None,
            f"{context} single-product closure arguments are incomplete",
        )
        raw_products: list[Mapping[str, Any]] = [
            {
                "name": "benchmark",
                "requested_target": requested_target,
                "expected_output": expected_output,
                "compile_entries": compile_entries,
                "link_arguments": link_arguments,
            }
        ]
    else:
        require(
            requested_target is None
            and expected_output is None
            and compile_entries is None
            and link_arguments is None,
            f"{context} mixes single-product and ordered-product arguments",
        )
        raw_products = list(products)
    require(raw_products, f"{context} has no build products")

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    targets: set[str] = set()
    outputs: set[str] = set()
    for index, value in enumerate(raw_products):
        product_context = f"{context} product {index}"
        product = expect_mapping(value, product_context)
        expect_exact_keys(
            product,
            {
                "name",
                "requested_target",
                "expected_output",
                "compile_entries",
                "link_arguments",
            },
            product_context,
        )
        name = expect_string(product["name"], f"{product_context} name")
        require(
            re.fullmatch(r"[a-z][a-z0-9-]*", name) is not None
            and name not in names,
            f"{product_context} name is invalid or duplicated",
        )
        target = expect_string(
            product["requested_target"],
            f"{product_context} requested target",
        )
        require(
            target not in targets,
            f"{context} duplicates requested target {target}",
        )
        raw_output = product["expected_output"]
        require(
            isinstance(raw_output, (str, Path)),
            f"{product_context} expected output is invalid",
        )
        output = lexical_absolute_path(
            str(raw_output),
            working_directory=build_directory,
        )
        require(
            target
            == ninja_target_for_output(
                Path(output),
                build_directory=build_directory,
                context=product_context,
            ),
            f"{product_context} target does not name its physical output",
        )
        require(
            output not in outputs,
            f"{context} duplicates expected output {output}",
        )
        product_entries = expect_list(
            product["compile_entries"],
            f"{product_context} compile entries",
        )
        require(product_entries, f"{product_context} compile entries are empty")
        product_link = expect_list(
            product["link_arguments"],
            f"{product_context} link arguments",
        )
        require(
            product_link
            and all(
                isinstance(argument, str) and argument
                for argument in product_link
            ),
            f"{product_context} link arguments are invalid",
        )
        output_indices = [
            argument_index
            for argument_index, argument in enumerate(product_link)
            if argument == "-o"
        ]
        require(
            "-c" not in product_link
            and len(output_indices) == 1
            and output_indices[0] + 1 < len(product_link),
            f"{product_context} link output is invalid",
        )
        linked_output = lexical_absolute_path(
            product_link[output_indices[0] + 1],
            working_directory=build_directory,
        )
        require(
            linked_output == output,
            f"{product_context} link output differs from its expected output",
        )
        names.add(name)
        targets.add(target)
        outputs.add(output)
        normalized.append(
            {
                "name": name,
                "requested_target": target,
                "expected_output": output,
                "compile_entries": product_entries,
                "link_arguments": product_link,
            }
        )
    return normalized


def validate_build_input_closure(
    evidence_dir: Path,
    reference_value: Any,
    *,
    capture_prefix: str,
    build_directory: Path,
    build_tool_path: str,
    cache: dict[str, str],
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
    context: str,
    products: Iterable[Mapping[str, Any]] | None = None,
    requested_target: str | None = None,
    expected_output: str | None = None,
    compile_entries: list[Any] | None = None,
    link_arguments: list[str] | None = None,
) -> dict[str, Any]:
    normalized_products = normalize_build_products(
        build_directory=build_directory,
        context=context,
        products=products,
        requested_target=requested_target,
        expected_output=expected_output,
        compile_entries=compile_entries,
        link_arguments=link_arguments,
    )
    canonical_compile_entry_union(
        normalized_products,
        context=f"{context} closure",
        response_files=response_files,
        used_response_files=used_response_files,
    )
    reference = expect_mapping(reference_value, f"{context} reference")
    expect_exact_keys(
        reference,
        {
            "closure_schema_version",
            "captured_path",
            "sha256",
            "products",
            "compile_units",
            "dependency_outputs",
            "files",
        },
        f"{context} reference",
    )
    require(
        expect_int(
            reference["closure_schema_version"],
            f"{context} reference schema",
            minimum=1,
        )
        == BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
        f"{context} reference schema is unsupported",
    )
    closure_path = validate_capture_reference(
        evidence_dir,
        {
            "captured_path": reference["captured_path"],
            "sha256": reference["sha256"],
        },
        context=f"{context} record",
        expected_path=f"{capture_prefix}-build-input-closure.json",
    )
    closure = expect_mapping(
        read_json(closure_path, f"{context} record"),
        f"{context} record",
    )
    expect_exact_keys(
        closure,
        {
            "closure_schema_version",
            "products",
            "build_directory",
            "ninja_deps",
            "compile_units",
            "dependency_graph",
            "selected_target_graph",
            "files",
        },
        f"{context} record",
    )
    require(
        expect_int(
            closure["closure_schema_version"],
            f"{context} schema",
            minimum=1,
        )
        == BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
        f"{context} schema is unsupported",
    )
    require(
        closure["build_directory"] == str(build_directory),
        f"{context} build directory differs",
    )

    file_values = expect_list(closure["files"], f"{context} files")
    files: dict[str, dict[str, Any]] = {}
    recorded_order: list[str] = []
    resolved_identities: dict[str, tuple[str, int, str]] = {}
    for index, value in enumerate(file_values):
        item_context = f"{context} file {index}"
        record = expect_mapping(value, item_context)
        expect_exact_keys(
            record,
            {
                "path",
                "resolved_path",
                "type",
                "roles",
                "captured_path",
                "sha256",
                "bytes",
            },
            item_context,
        )
        path = expect_string(record["path"], f"{item_context} path")
        require(
            Path(path).is_absolute() and os.path.abspath(path) == path,
            f"{item_context} path is not canonical absolute",
        )
        require(path not in files, f"{context} duplicates file {path}")
        resolved = expect_string(
            record["resolved_path"],
            f"{item_context} resolved path",
        )
        require(
            Path(resolved).is_absolute(), f"{item_context} resolved path is relative"
        )
        require(
            record["type"] == closure_file_role(path),
            f"{item_context} type differs from its path",
        )
        roles = expect_list(record["roles"], f"{item_context} roles")
        require(
            roles
            and all(isinstance(role, str) and role for role in roles)
            and roles == sorted(set(roles)),
            f"{item_context} roles are not canonical",
        )
        digest = expect_sha256(record["sha256"], f"{item_context} sha256")
        size = expect_int(record["bytes"], f"{item_context} bytes", minimum=0)
        captured_text, captured_path = relative_evidence_path(
            evidence_dir,
            record["captured_path"],
            f"{item_context} captured path",
        )
        require(
            captured_text == f"provenance/blobs/{digest[:2]}/{digest}",
            f"{item_context} is not stored at its content-addressed path",
        )
        require(
            captured_path.is_file()
            and captured_path.stat().st_size == size
            and sha256_file(captured_path) == digest,
            f"{item_context} captured content differs",
        )
        identity = (digest, size, captured_text)
        previous = resolved_identities.get(resolved)
        require(
            previous is None or previous == identity,
            f"{context} resolved path {resolved} has conflicting content",
        )
        resolved_identities[resolved] = identity
        files[path] = record
        recorded_order.append(path)
    require(
        recorded_order
        == sorted(recorded_order, key=lambda value: value.encode("utf-8")),
        f"{context} files are not in canonical path order",
    )
    require(
        expect_int(reference["files"], f"{context} file count", minimum=1)
        == len(files),
        f"{context} file count differs",
    )

    deps_reference = expect_mapping(closure["ninja_deps"], f"{context} Ninja deps")
    expect_exact_keys(
        deps_reference,
        {"command", "captured_path", "sha256", "records"},
        f"{context} Ninja deps",
    )
    expected_deps_command = [
        build_tool_path,
        "-C",
        str(build_directory),
        "-t",
        "deps",
    ]
    require(
        deps_reference["command"] == expected_deps_command,
        f"{context} Ninja deps command differs",
    )
    deps_path = validate_capture_reference(
        evidence_dir,
        {
            "captured_path": deps_reference["captured_path"],
            "sha256": deps_reference["sha256"],
        },
        context=f"{context} Ninja deps",
        expected_path=f"{capture_prefix}-ninja-deps.txt",
    )
    all_deps = parse_ninja_deps(
        _read_text(deps_path, f"{context} Ninja deps"),
        build_directory=build_directory,
        context=f"{context} Ninja deps",
    )
    require(
        expect_int(
            deps_reference["records"], f"{context} Ninja deps records", minimum=1
        )
        == len(all_deps),
        f"{context} Ninja deps record count differs",
    )

    expected_units_by_output: dict[str, dict[str, Any]] = {}
    compile_arguments_by_output: dict[str, list[str]] = {}
    seed_outputs: set[str] = set()
    module_producers: dict[str, str] = {}
    direct_inputs_by_output: dict[str, list[dict[str, str]]] = {}
    product_compile_outputs: dict[str, list[str]] = {}
    for product in normalized_products:
        product_units_by_output: dict[
            str,
            tuple[list[str], dict[str, Any]],
        ] = {}
        for index, entry_value in enumerate(product["compile_entries"]):
            entry_context = (
                f"{context} product {product['name']} compile entry {index}"
            )
            entry = expect_mapping(entry_value, entry_context)
            working_directory = Path(
                expect_string(entry.get("directory"), f"{entry_context} directory")
            )
            require(
                working_directory.is_absolute(),
                f"{entry_context} directory is relative",
            )
            arguments = expand_response_arguments(
                compile_entry_arguments(entry, context=entry_context),
                working_directory=working_directory,
                response_files=response_files,
                used_response_files=used_response_files,
                context=entry_context,
            )
            output = compile_output_lexical_path(
                arguments,
                working_directory=working_directory,
                context=entry_context,
            )
            require(output is not None, f"{entry_context} is not a compilation")
            module_outputs = module_output_paths(
                arguments,
                working_directory=working_directory,
                context=entry_context,
            )
            direct_inputs = compile_direct_inputs(
                entry,
                arguments,
                working_directory=working_directory,
                context=entry_context,
            )
            unit = {
                "output": output,
                "module_outputs": module_outputs,
                "direct_inputs": direct_inputs,
            }
            previous_product_unit = product_units_by_output.get(output)
            require(
                previous_product_unit is None
                or previous_product_unit == (arguments, unit),
                f"{context} product {product['name']} has conflicting "
                f"compile entries for {output}",
            )
            product_units_by_output.setdefault(output, (arguments, unit))
            previous_unit = expected_units_by_output.get(output)
            require(
                previous_unit is None
                or (
                    previous_unit == unit
                    and compile_arguments_by_output[output] == arguments
                ),
                f"{context} has conflicting compile entries for {output}",
            )
            expected_units_by_output.setdefault(output, unit)
            compile_arguments_by_output.setdefault(output, arguments)
            previous_inputs = direct_inputs_by_output.get(output)
            require(
                previous_inputs is None or previous_inputs == direct_inputs,
                f"{context} has conflicting direct inputs for {output}",
            )
            direct_inputs_by_output.setdefault(output, direct_inputs)
            seed_outputs.add(output)
            for module_output in module_outputs:
                previous_producer = module_producers.get(module_output)
                require(
                    previous_producer is None or previous_producer == output,
                    f"{context} has conflicting module output {module_output}",
                )
                module_producers.setdefault(module_output, output)
        product_compile_outputs[product["name"]] = sorted(
            product_units_by_output,
            key=lambda value: value.encode("utf-8"),
        )
    expected_units = sorted(
        expected_units_by_output.values(),
        key=lambda value: value["output"].encode("utf-8"),
    )
    require(
        closure["compile_units"] == expected_units,
        f"{context} compile units differ from archived commands",
    )
    require(
        expect_int(
            reference["compile_units"], f"{context} compile unit count", minimum=1
        )
        == len(expected_units),
        f"{context} compile unit count differs",
    )
    missing_outputs = sorted(seed_outputs - set(all_deps))
    require(
        not missing_outputs,
        f"{context} Ninja deps omit compile/module outputs: {missing_outputs}",
    )

    closure_outputs: set[str] = set()
    pending = sorted(seed_outputs, key=lambda value: value.encode("utf-8"))
    while pending:
        output = pending.pop()
        if output in closure_outputs:
            continue
        record = all_deps[output]
        require(record["status"] == "VALID", f"{context} has stale deps for {output}")
        closure_outputs.add(output)
        for dependency in record["dependencies"]:
            producer = module_producers.get(dependency, dependency)
            if producer in all_deps and producer not in closure_outputs:
                pending.append(producer)
            require(
                Path(dependency).suffix.lower() != ".pcm"
                or dependency in module_producers,
                f"{context} PCM dependency has no selected producer: {dependency}",
            )
    expected_graph = [
        all_deps[output]
        for output in sorted(
            closure_outputs,
            key=lambda value: value.encode("utf-8"),
        )
    ]
    require(
        closure["dependency_graph"] == expected_graph,
        f"{context} dependency graph differs from raw Ninja deps",
    )
    require(
        expect_int(
            reference["dependency_outputs"],
            f"{context} dependency output count",
            minimum=1,
        )
        == len(expected_graph),
        f"{context} dependency output count differs",
    )
    for output, direct_inputs in direct_inputs_by_output.items():
        recorded = set(all_deps[output]["dependencies"])
        for direct_input in direct_inputs:
            if direct_input["role"] == "module-input":
                require(
                    direct_input["path"] in module_producers,
                    f"{context} module input has no selected producer: "
                    f"{direct_input['path']}",
                )
                continue
            if direct_input["role"] == "module-map":
                continue
            require(
                direct_input["path"] in recorded,
                f"{context} direct input {direct_input['path']} is absent from "
                f"Ninja dependencies for {output}",
            )
    expected_products: list[dict[str, Any]] = []
    for product in normalized_products:
        expected_link_inputs = derive_link_inputs(
            product["link_arguments"],
            working_directory=build_directory,
            sysroot=Path(cache["CMAKE_OSX_SYSROOT"]),
            available_paths=set(files),
            context=f"{context} product {product['name']} link command",
        )
        expected_products.append(
            {
                "name": product["name"],
                "requested_target": product["requested_target"],
                "expected_output": product["expected_output"],
                "compile_outputs": product_compile_outputs[product["name"]],
                "link": {
                    "arguments": product["link_arguments"],
                    "inputs": expected_link_inputs,
                },
            }
        )
    require(
        closure["products"] == expected_products,
        f"{context} ordered products differ",
    )
    require(
        expect_int(reference["products"], f"{context} product count", minimum=1)
        == len(expected_products),
        f"{context} product count differs",
    )
    selected_target_graph = expect_mapping(
        closure["selected_target_graph"],
        f"{context} selected target graph",
    )
    expect_exact_keys(
        selected_target_graph,
        {
            "targets",
            "query",
            "clean_plan",
            "selected_outputs",
            "material_outputs",
            "derived_link_outputs",
        },
        f"{context} selected target graph",
    )
    targets_reference = expect_mapping(
        selected_target_graph["targets"],
        f"{context} selected target graph targets",
    )
    expect_exact_keys(
        targets_reference,
        {"command", "captured_path", "sha256", "records"},
        f"{context} selected target graph targets",
    )
    expected_targets_command = [
        build_tool_path,
        "-C",
        str(build_directory),
        "-t",
        "targets",
        "all",
    ]
    require(
        targets_reference["command"] == expected_targets_command,
        f"{context} Ninja targets command differs",
    )
    targets_path = validate_capture_reference(
        evidence_dir,
        {
            "captured_path": targets_reference["captured_path"],
            "sha256": targets_reference["sha256"],
        },
        context=f"{context} Ninja targets",
        expected_path=f"{capture_prefix}-ninja-targets.txt",
    )
    targets = parse_ninja_targets(
        _read_text(targets_path, f"{context} Ninja targets"),
        build_directory=build_directory,
        context=f"{context} Ninja targets",
    )
    require(
        expect_int(
            targets_reference["records"],
            f"{context} Ninja target count",
            minimum=1,
        )
        == len(targets),
        f"{context} Ninja target count differs",
    )
    selected_outputs = [product["expected_output"] for product in expected_products]
    require(
        selected_target_graph["selected_outputs"] == selected_outputs,
        f"{context} selected graph outputs differ",
    )
    for selected_output in selected_outputs:
        require(
            targets.get(selected_output) not in (None, "phony"),
            f"{context} selected physical output has no material Ninja producer: "
            f"{selected_output}",
        )
    build_prefix = f"{build_directory}/"
    candidate_outputs = {
        str(item["path"])
        for product in expected_products
        for item in product["link"]["inputs"]
        if (
            str(item["path"]).startswith(build_prefix)
            and targets.get(str(item["path"])) not in (None, "phony")
        )
    }
    requested_targets = [
        str(product["requested_target"]) for product in expected_products
    ]
    expected_query_targets = list(requested_targets)
    for path in sorted(candidate_outputs, key=lambda value: value.encode("utf-8")):
        target = ninja_target_for_output(
            Path(path),
            build_directory=build_directory,
            context=f"{context} linked output",
        )
        if target not in expected_query_targets:
            expected_query_targets.append(target)
    require(
        len(expected_query_targets) == len(set(expected_query_targets)),
        f"{context} target graph query contains duplicate targets",
    )
    query_reference = expect_mapping(
        selected_target_graph["query"],
        f"{context} selected target graph query",
    )
    expect_exact_keys(
        query_reference,
        {"command", "captured_path", "sha256", "nodes"},
        f"{context} selected target graph query",
    )
    expected_query_command = [
        build_tool_path,
        "-C",
        str(build_directory),
        "-t",
        "query",
        *expected_query_targets,
    ]
    require(
        query_reference["command"] == expected_query_command,
        f"{context} Ninja query command differs",
    )
    query_path = validate_capture_reference(
        evidence_dir,
        {
            "captured_path": query_reference["captured_path"],
            "sha256": query_reference["sha256"],
        },
        context=f"{context} Ninja query",
        expected_path=f"{capture_prefix}-ninja-query.txt",
    )
    nodes = parse_ninja_query(
        _read_text(query_path, f"{context} Ninja query"),
        build_directory=build_directory,
        context=f"{context} Ninja query",
    )
    require(
        query_reference["nodes"] == nodes,
        f"{context} Ninja query nodes differ from the raw capture",
    )
    derived_link_outputs = sorted(
        {
            output
            for product in expected_products
            for output in graph_proven_link_outputs(
                nodes=nodes,
                selected_output=product["expected_output"],
                link_inputs=product["link"]["inputs"],
                known_derived_outputs={
                    *closure_outputs,
                    *module_producers,
                },
                build_directory=build_directory,
                context=f"{context} product {product['name']} target graph",
            )
        },
        key=lambda value: value.encode("utf-8"),
    )
    material_outputs = sorted(
        {
            *selected_outputs,
            *(
                path
                for path in {
                    *closure_outputs,
                    *module_producers,
                }
                if targets.get(path) not in (None, "phony")
            ),
            *candidate_outputs,
        },
        key=lambda value: value.encode("utf-8"),
    )
    require(
        selected_target_graph["material_outputs"] == material_outputs,
        f"{context} material target outputs differ",
    )
    require(
        selected_target_graph["derived_link_outputs"] == derived_link_outputs,
        f"{context} graph-proven link outputs differ",
    )
    clean_plan = expect_mapping(
        selected_target_graph["clean_plan"],
        f"{context} selected target clean plan",
    )
    expect_exact_keys(
        clean_plan,
        {"command", "returncode", "stdout", "stderr", "outputs"},
        f"{context} selected target clean plan",
    )
    expected_clean_command = [
        build_tool_path,
        "-v",
        "-n",
        "-C",
        str(build_directory),
        "-t",
        "clean",
        *requested_targets,
    ]
    require(
        clean_plan["command"] == expected_clean_command,
        f"{context} Ninja target clean command differs",
    )
    require(
        expect_int(
            clean_plan["returncode"],
            f"{context} Ninja target clean return code",
        )
        == 0,
        f"{context} Ninja target clean command failed",
    )
    clean_stdout_reference = expect_mapping(
        clean_plan["stdout"],
        f"{context} Ninja target clean stdout",
    )
    clean_stderr_reference = expect_mapping(
        clean_plan["stderr"],
        f"{context} Ninja target clean stderr",
    )
    expect_exact_keys(
        clean_stdout_reference,
        {"captured_path", "sha256"},
        f"{context} Ninja target clean stdout",
    )
    expect_exact_keys(
        clean_stderr_reference,
        {"captured_path", "sha256"},
        f"{context} Ninja target clean stderr",
    )
    clean_stdout_path = validate_capture_reference(
        evidence_dir,
        clean_stdout_reference,
        context=f"{context} Ninja target clean stdout",
        expected_path=f"{capture_prefix}-ninja-clean-plan.stdout",
    )
    clean_stderr_path = validate_capture_reference(
        evidence_dir,
        clean_stderr_reference,
        context=f"{context} Ninja target clean stderr",
        expected_path=f"{capture_prefix}-ninja-clean-plan.stderr",
    )
    require(
        _read_text(
            clean_stderr_path,
            f"{context} Ninja target clean stderr",
        )
        == "",
        f"{context} Ninja target clean plan emitted stderr",
    )
    clean_outputs = parse_ninja_clean_plan(
        _read_text(
            clean_stdout_path,
            f"{context} Ninja target clean stdout",
        ),
        build_directory=build_directory,
        requested_targets=requested_targets,
        context=f"{context} Ninja target clean plan",
    )
    recorded_clean_outputs = [
        expect_string(
            path,
            f"{context} Ninja target clean output {index}",
        )
        for index, path in enumerate(
            expect_list(
                clean_plan["outputs"],
                f"{context} Ninja target clean outputs",
            )
        )
    ]
    require(
        recorded_clean_outputs == clean_outputs,
        f"{context} Ninja target clean outputs differ from the raw capture",
    )
    require(
        {
            *selected_outputs,
            *material_outputs,
            *derived_link_outputs,
        }
        <= set(clean_outputs),
        f"{context} Ninja target clean plan omits graph-proven outputs",
    )

    expected_roles: dict[str, set[str]] = {}

    def add_role(path: str, role: str) -> None:
        expected_roles.setdefault(path, set()).add(role)

    for record in expected_graph:
        add_role(record["output"], "ninja-output")
        for dependency in record["dependencies"]:
            add_role(dependency, "compiler-dependency")
    for unit in expected_units:
        add_role(unit["output"], "compile-output")
        for module_output in unit["module_outputs"]:
            add_role(module_output, "module-output")
        for direct_input in unit["direct_inputs"]:
            add_role(direct_input["path"], direct_input["role"])
    for _, response_path in response_files:
        add_role(response_path, "response-file")
    for product in expected_products:
        for link_input in product["link"]["inputs"]:
            add_role(link_input["path"], link_input["role"])
    require(
        set(files) == set(expected_roles),
        f"{context} file set differs from reconstructed build inputs",
    )
    for path, roles in expected_roles.items():
        require(
            files[path]["roles"] == sorted(roles),
            f"{context} roles differ for {path}",
        )
    return closure


def validate_compiled_sources_against_closure(
    source_records: dict[str, dict[str, Any]],
    closure: dict[str, Any],
    *,
    context: str,
) -> None:
    closure_sources = {
        expect_string(record["path"], f"{context} closure source path"): record
        for record in expect_list(closure["files"], f"{context} closure files")
        if "compile-source"
        in expect_list(record["roles"], f"{context} closure file roles")
    }
    require(
        set(source_records) == set(closure_sources),
        f"{context} compiled-source records differ from the build closure",
    )
    for path, source in source_records.items():
        closure_source = closure_sources[path]
        require(
            (
                source["sha256"],
                source["bytes"],
                source["captured_path"],
            )
            == (
                closure_source["sha256"],
                closure_source["bytes"],
                closure_source["captured_path"],
            ),
            f"{context} compiled-source content differs from the build closure: "
            f"{path}",
        )


def validate_dependency_entry(
    value: Any,
    *,
    context: str,
    runtime_artifacts: dict[str, str],
) -> dict[str, Any]:
    dependency = expect_mapping(value, context)
    expect_exact_keys(
        dependency,
        {
            "install_names",
            "referenced_by",
            "resolved_path",
            "sha256",
            "bytes",
            "unavailable_reason",
        },
        context,
    )
    install_names = expect_list(dependency["install_names"], f"{context} install names")
    referenced_by = expect_list(dependency["referenced_by"], f"{context} referenced by")
    require(
        install_names
        and referenced_by
        and all(
            isinstance(item, str) and item for item in install_names + referenced_by
        ),
        f"{context} has invalid dependency references",
    )
    resolved = dependency["resolved_path"]
    if resolved is None:
        require(dependency["sha256"] is None, f"{context} unresolved hash is not null")
        require(dependency["bytes"] is None, f"{context} unresolved size is not null")
        require(
            isinstance(dependency["unavailable_reason"], str)
            and dependency["unavailable_reason"],
            f"{context} unresolved reason is missing",
        )
    else:
        path = expect_string(resolved, f"{context} resolved path")
        require(Path(path).is_absolute(), f"{context} resolved path is not absolute")
        if dependency["sha256"] is None:
            require(dependency["bytes"] is None, f"{context} cached size is not null")
            require(
                isinstance(dependency["unavailable_reason"], str)
                and dependency["unavailable_reason"],
                f"{context} cached dependency reason is missing",
            )
        else:
            digest = expect_sha256(dependency["sha256"], f"{context} sha256")
            expect_int(dependency["bytes"], f"{context} bytes", minimum=1)
            require(
                runtime_artifacts.get(path) == digest,
                f"{context} is absent from the runtime artifact set",
            )
            require(
                dependency["unavailable_reason"] is None,
                f"{context} resolved dependency has an unavailable reason",
            )
    return dependency


def parse_otool_libraries(
    text: str,
    *,
    image: str,
    context: str,
) -> list[str]:
    lines = text.splitlines()
    require(lines and lines[0] == f"{image}:", f"{context} header differs")
    install_names: list[str] = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        match = re.fullmatch(r"\s*(\S+)\s+\(.+\)", line)
        require(
            match is not None,
            f"{context} line {line_number} is not an otool library record",
        )
        install_name = match.group(1)
        require(
            install_name not in install_names,
            f"{context} duplicates install name {install_name!r}",
        )
        install_names.append(install_name)
    return install_names


def parse_macho_libraries(path: Path, context: str) -> list[str]:
    data = read_verified_bytes(path, context)
    require(len(data) >= 32, f"{context} is too small to be Mach-O")
    (
        magic,
        cpu_type,
        _cpu_subtype,
        _file_type,
        command_count,
        command_bytes,
        _flags,
        _reserved,
    ) = struct.unpack_from("<IiiIIIII", data, 0)
    require(magic == 0xFEEDFACF, f"{context} is not little-endian Mach-O 64")
    require(cpu_type == 0x0100000C, f"{context} is not arm64 Mach-O")
    command_end = 32 + command_bytes
    require(command_end <= len(data), f"{context} load commands exceed file size")
    dylib_commands = {
        0x0000000C,
        0x0000000D,
        0x00000020,
        0x80000018,
        0x8000001F,
        0x80000023,
    }
    libraries: list[str] = []
    offset = 32
    for index in range(command_count):
        require(offset + 8 <= command_end, f"{context} command {index} is truncated")
        command, command_size = struct.unpack_from("<II", data, offset)
        require(
            command_size >= 8 and offset + command_size <= command_end,
            f"{context} command {index} has invalid size",
        )
        if command in dylib_commands:
            require(command_size >= 24, f"{context} dylib command {index} is short")
            name_offset = struct.unpack_from("<I", data, offset + 8)[0]
            require(
                24 <= name_offset < command_size,
                f"{context} dylib command {index} has invalid name offset",
            )
            name_start = offset + name_offset
            name_end = data.find(b"\0", name_start, offset + command_size)
            require(name_end >= 0, f"{context} dylib command {index} lacks terminator")
            try:
                name = data[name_start:name_end].decode("utf-8")
            except UnicodeDecodeError as error:
                raise VerificationError(
                    f"{context} dylib command {index} is not UTF-8"
                ) from error
            require(
                name and name not in libraries,
                f"{context} has duplicate dylib {name!r}",
            )
            libraries.append(name)
        offset += command_size
    require(offset == command_end, f"{context} load-command size is inconsistent")
    return libraries


def validate_dependency_manifest(
    evidence_dir: Path,
    reference: dict[str, Any],
    *,
    expected_path: str,
    expected_binary_path: str,
    expected_binary_sha256: str,
    runtime_artifacts: dict[str, str],
    runtime_artifact_paths: dict[str, Path],
    context: str,
) -> dict[str, Any]:
    expect_exact_keys(reference, {"captured_path", "sha256", "count"}, context)
    path = validate_capture_reference(
        evidence_dir,
        reference,
        context=context,
        expected_path=expected_path,
    )
    manifest = expect_mapping(read_json(path, context), context)
    expect_exact_keys(
        manifest,
        {"binary", "binary_sha256", "dependencies", "otool_outputs"},
        context,
    )
    require(
        manifest["binary"] == expected_binary_path, f"{context} binary path differs"
    )
    require(
        manifest["binary_sha256"] == expected_binary_sha256,
        f"{context} binary hash differs",
    )
    dependencies = expect_list(manifest["dependencies"], f"{context} dependencies")
    require(
        expect_int(reference["count"], f"{context} count", minimum=0)
        == len(dependencies),
        f"{context} dependency count differs",
    )
    validated_dependencies = [
        validate_dependency_entry(
            dependency,
            context=f"{context} dependency {index}",
            runtime_artifacts=runtime_artifacts,
        )
        for index, dependency in enumerate(dependencies)
    ]
    dependency_order = [
        str(dependency["resolved_path"]) for dependency in validated_dependencies
    ]
    require(
        dependency_order == sorted(dependency_order),
        f"{context} dependencies are not in canonical path order",
    )
    outputs = expect_mapping(manifest["otool_outputs"], f"{context} otool outputs")
    require(
        outputs
        and all(
            isinstance(key, str) and isinstance(value, str) and key and value
            for key, value in outputs.items()
        ),
        f"{context} otool output capture is empty or invalid",
    )
    expected_images = {expected_binary_path}
    expected_images.update(
        expect_string(
            dependency["resolved_path"],
            f"{context} dependency resolved path",
        )
        for dependency in validated_dependencies
        if dependency["sha256"] is not None
    )
    require(
        set(outputs) == expected_images,
        f"{context} otool outputs do not cover the archived image closure",
    )
    observed_edges: set[tuple[str, str]] = set()
    for image, output in outputs.items():
        output_libraries = parse_otool_libraries(
            output,
            image=image,
            context=f"{context} otool output {image}",
        )
        archived_path = runtime_artifact_paths.get(image)
        require(
            archived_path is not None,
            f"{context} has no archived bytes for otool image {image}",
        )
        actual_libraries = parse_macho_libraries(
            archived_path,
            f"{context} archived image {image}",
        )
        require(
            output_libraries == actual_libraries,
            f"{context} otool output differs from archived Mach-O {image}",
        )
        observed_edges.update((image, name) for name in actual_libraries)
    recorded_edges: set[tuple[str, str]] = set()
    for index, dependency in enumerate(validated_dependencies):
        for referenced_by in dependency["referenced_by"]:
            for install_name in dependency["install_names"]:
                edge = (referenced_by, install_name)
                require(
                    edge not in recorded_edges,
                    f"{context} dependency {index} duplicates edge {edge!r}",
                )
                recorded_edges.add(edge)
    require(
        recorded_edges == observed_edges,
        f"{context} dependency records differ from captured otool outputs",
    )
    return manifest


def validate_build_record(
    evidence_dir: Path,
    value: Any,
    *,
    engine: str,
    binary: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    runtime_artifacts: dict[str, str],
    runtime_artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    context = f"preflight {engine} build"
    build = expect_mapping(value, context)
    expect_exact_keys(
        build,
        {
            "build_directory",
            "cmake_home_directory",
            "cmake_cache",
            "compile_commands",
            "settings",
            "compiler",
            "linker",
            "build_tool",
            "ninja_commands",
            "ninja_noop",
            "dependencies",
            "build_input_closure",
            "response_files",
            "compiled_sources",
        },
        context,
    )
    require(
        Path(
            expect_string(build["build_directory"], f"{context} directory")
        ).is_absolute(),
        f"{context} directory is not absolute",
    )
    require(
        Path(
            expect_string(build["cmake_home_directory"], f"{context} source directory")
        ).is_absolute(),
        f"{context} source directory is not absolute",
    )

    cache_reference = expect_mapping(build["cmake_cache"], f"{context} CMake cache")
    expect_exact_keys(
        cache_reference, {"captured_path", "sha256"}, f"{context} CMake cache"
    )
    cache_path = validate_capture_reference(
        evidence_dir,
        cache_reference,
        context=f"{context} CMake cache",
        expected_path=f"{engine}-CMakeCache.txt",
    )
    cache = parse_cmake_cache(
        _read_text(cache_path, f"{context} CMake cache"),
        f"{context} CMake cache",
    )

    compile_reference = expect_mapping(
        build["compile_commands"],
        f"{context} compile database",
    )
    expect_exact_keys(
        compile_reference,
        {"captured_path", "sha256", "entries"},
        f"{context} compile database",
    )
    compile_path = validate_capture_reference(
        evidence_dir,
        compile_reference,
        context=f"{context} compile database",
        expected_path=f"{engine}-compile_commands.json",
    )
    compile_value = read_json(compile_path, f"{context} compile database")
    compile_entries = expect_list(compile_value, f"{context} compile database")
    require(
        expect_int(
            compile_reference["entries"],
            f"{context} compile database entries",
            minimum=1,
        )
        == len(compile_entries),
        f"{context} compile database entry count differs",
    )
    response_files = validate_response_file_records(
        evidence_dir,
        build["response_files"],
        f"{context} response files",
    )
    used_response_files: set[tuple[str, str]] = set()
    validate_release_compile_inputs(
        cache,
        compile_entries,
        context=context,
        response_files=response_files,
        used_response_files=used_response_files,
    )
    require(
        compile_source_paths(compile_entries, f"{context} compile database")
        == set(source_records),
        f"{context} compiled-source records differ from the compile database",
    )
    require(
        expect_int(build["compiled_sources"], f"{context} compiled sources", minimum=1)
        == len(source_records),
        f"{context} compiled-source count differs",
    )

    settings = expect_mapping(build["settings"], f"{context} settings")
    expected_setting_names = set(build_setting_names(engine))
    expect_exact_keys(settings, expected_setting_names, f"{context} settings")
    require(
        settings == {name: cache.get(name) for name in expected_setting_names},
        f"{context} settings differ from its CMake cache",
    )
    if engine == "infinity":
        require(
            Path(binary["path"]).name == "infinity_hnsw_d0_production",
            f"{context} is not the production D0 target",
        )
        expected_production_settings = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "ENABLE_JEMALLOC": "OFF",
            "VCPKG_MANIFEST_INSTALL": "OFF",
            "VCPKG_TARGET_TRIPLET": "arm64-osx",
            "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION": "ON",
            "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL": "OFF",
            "INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING": "ON",
        }
        for name, expected in expected_production_settings.items():
            require(
                settings[name] == expected,
                f"{context} requires {name}={expected}",
            )
        require(
            isinstance(settings["VCPKG_INSTALLED_DIR"], str)
            and Path(settings["VCPKG_INSTALLED_DIR"]).is_absolute(),
            f"{context} has no absolute vcpkg installed directory",
        )
        require(
            isinstance(settings["INFINITY_BUILD_TIME_OVERRIDE"], str)
            and settings["INFINITY_BUILD_TIME_OVERRIDE"] != "",
            f"{context} has no reproducible build-time override",
        )
    compiler = validate_executable_record(
        evidence_dir,
        build["compiler"],
        f"{context} compiler",
        runtime_artifacts=runtime_artifacts,
    )
    linker = validate_executable_record(
        evidence_dir,
        build["linker"],
        f"{context} linker",
        runtime_artifacts=runtime_artifacts,
    )
    build_tool = validate_executable_record(
        evidence_dir,
        build["build_tool"],
        f"{context} build tool",
        runtime_artifacts=runtime_artifacts,
    )
    require(
        compiler["configured_path"] == cache.get("CMAKE_CXX_COMPILER"),
        f"{context} compiler differs from its CMake cache",
    )
    require(
        linker["configured_path"] == cache.get("CMAKE_LINKER"),
        f"{context} linker differs from its CMake cache",
    )
    require(
        build_tool["configured_path"] == cache.get("CMAKE_MAKE_PROGRAM"),
        f"{context} build tool differs from its CMake cache",
    )

    commands_reference = expect_mapping(
        build["ninja_commands"],
        f"{context} Ninja commands",
    )
    expect_exact_keys(
        commands_reference,
        {"captured_path", "sha256"},
        f"{context} Ninja commands",
    )
    commands_path = validate_capture_reference(
        evidence_dir,
        commands_reference,
        context=f"{context} Ninja commands",
        expected_path=f"{engine}-ninja-commands.txt",
    )
    link_arguments = validate_ninja_commands(
        _read_text(commands_path, f"{context} Ninja commands"),
        cache=cache,
        compile_entries=compile_entries,
        build_directory=Path(build["build_directory"]),
        expected_output=Path(binary["path"]),
        context=f"{context} Ninja commands",
        response_files=response_files,
        used_response_files=used_response_files,
    )
    require(
        used_response_files == set(response_files),
        f"{context} contains unreferenced response-file records",
    )
    validated_closure = validate_build_input_closure(
        evidence_dir,
        build["build_input_closure"],
        capture_prefix=engine,
        requested_target=ninja_target_for_output(
            Path(binary["path"]),
            build_directory=Path(build["build_directory"]),
            context=context,
        ),
        expected_output=binary["path"],
        build_directory=Path(build["build_directory"]),
        build_tool_path=build_tool["resolved_path"],
        cache=cache,
        compile_entries=compile_entries,
        link_arguments=link_arguments,
        response_files=response_files,
        used_response_files=used_response_files,
        context=f"{context} build-input closure",
    )
    validate_compiled_sources_against_closure(
        source_records,
        validated_closure,
        context=context,
    )
    noop_reference = expect_mapping(build["ninja_noop"], f"{context} Ninja no-op")
    expect_exact_keys(
        noop_reference,
        {"captured_path", "sha256"},
        f"{context} Ninja no-op",
    )
    noop_path = validate_capture_reference(
        evidence_dir,
        noop_reference,
        context=f"{context} Ninja no-op",
        expected_path=f"{engine}-ninja-noop.txt",
    )
    require(
        "ninja: no work to do." in _read_text(noop_path, f"{context} Ninja no-op"),
        f"{context} build tree was not captured as up to date",
    )
    validated_dependencies = validate_dependency_manifest(
        evidence_dir,
        expect_mapping(build["dependencies"], f"{context} dependencies"),
        expected_path=f"{engine}-dependencies.json",
        expected_binary_path=binary["path"],
        expected_binary_sha256=binary["sha256"],
        runtime_artifacts=runtime_artifacts,
        runtime_artifact_paths=runtime_artifact_paths,
        context=f"{context} dependencies",
    )
    result = dict(build)
    result["_validated_build_input_closure"] = validated_closure
    result["_validated_dependency_manifest"] = validated_dependencies
    return result


def validate_benchmark_input_sources(
    evidence_dir: Path,
    value: Any,
    *,
    runtime_artifacts: dict[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    inputs = expect_mapping(value, "benchmark input sources")
    expect_exact_keys(
        inputs,
        {
            "native_harness",
            "ctpl",
            "simde_headers",
            "libomp_recipe",
            "omp_header",
            "libomp_runtime",
        },
        "benchmark input sources",
    )
    trees: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ("native_harness", "ctpl", "simde_headers"):
        tree = expect_mapping(inputs[name], f"benchmark input {name}")
        expect_exact_keys(tree, {"root", "files"}, f"benchmark input {name}")
        root = expect_string(tree["root"], f"benchmark input {name} root")
        require(
            Path(root).is_absolute(), f"benchmark input {name} root is not absolute"
        )
        records = validate_source_records(
            evidence_dir,
            tree["files"],
            f"benchmark input {name} files",
        )
        for absolute in records:
            try:
                Path(absolute).relative_to(Path(root))
            except ValueError as error:
                raise VerificationError(
                    f"benchmark input {name} source is outside its root: {absolute}"
            ) from error
        trees[name] = records
    ctpl_paths = set(trees["ctpl"])
    require(
        len(ctpl_paths) == 1
        and Path(next(iter(ctpl_paths))).name == "ctpl_stl.h",
        "benchmark input CTPL records do not identify exactly ctpl_stl.h",
    )
    require(
        bool(trees["simde_headers"]),
        "benchmark input SIMDe tree is empty",
    )
    for name in ("libomp_recipe", "omp_header", "libomp_runtime"):
        item = expect_mapping(inputs[name], f"benchmark input {name}")
        expect_exact_keys(
            item,
            {"path", "captured_path", "sha256", "bytes"},
            f"benchmark input {name}",
        )
        path = expect_string(item["path"], f"benchmark input {name} path")
        require(path != "", f"benchmark input {name} path is empty")
        digest = expect_sha256(item["sha256"], f"benchmark input {name} sha256")
        size = expect_int(item["bytes"], f"benchmark input {name} bytes", minimum=1)
        captured_text, captured_path = relative_evidence_path(
            evidence_dir,
            item["captured_path"],
            f"benchmark input {name} captured path",
        )
        require(
            captured_text == f"provenance/blobs/{digest[:2]}/{digest}",
            f"benchmark input {name} is not content addressed",
        )
        require(
            captured_path.is_file()
            and captured_path.stat().st_size == size
            and sha256_file(captured_path) == digest,
            f"benchmark input {name} captured content differs",
        )
        if name == "libomp_runtime":
            require(Path(path).is_absolute(), "libomp runtime path is not absolute")
            require(
                digest == PINNED_LIBOMP_SHA256,
                "Benchmark input libomp runtime is not the pinned runtime",
            )
            require(
                runtime_artifacts.get(path) == digest,
                "Benchmark input libomp runtime is absent from runtime artifacts",
            )
    return trees


def git_object_id(kind: str, data: bytes) -> str:
    header = f"{kind} {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def parse_git_commit_identity(
    data: bytes,
    context: str,
) -> tuple[str, tuple[str, ...]]:
    header, separator, _message = data.partition(b"\n\n")
    require(separator == b"\n\n", f"{context} has no header terminator")
    trees: list[str] = []
    parents: list[str] = []
    for line in header.splitlines():
        if line.startswith(b" "):
            continue
        name, separator, value = line.partition(b" ")
        require(separator == b" " and value, f"{context} has a malformed header")
        if name not in (b"tree", b"parent"):
            continue
        try:
            object_id = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise VerificationError(
                f"{context} has a non-ASCII object ID"
            ) from error
        require(
            GIT_OBJECT_ID.fullmatch(object_id) is not None,
            f"{context} has an invalid {name.decode('ascii')} object ID",
        )
        if name == b"tree":
            trees.append(object_id)
        else:
            parents.append(object_id)
    require(len(trees) == 1, f"{context} must contain exactly one tree")
    return trees[0], tuple(parents)


def parse_changed_paths(text: str, context: str) -> tuple[str, ...]:
    paths: list[str] = []
    for line_number, path in enumerate(text.splitlines(), start=1):
        pure = PurePosixPath(path)
        require(
            path == pure.as_posix()
            and not pure.is_absolute()
            and all(part not in ("", ".", "..") for part in pure.parts),
            f"{context} line {line_number} is not a canonical relative path",
        )
        require(path not in paths, f"{context} duplicates path {path!r}")
        paths.append(path)
    require(paths, f"{context} is empty")
    return tuple(paths)


def parse_git_tree_listing(
    text: str,
    context: str,
) -> dict[str, tuple[str, str, str]]:
    entries: dict[str, tuple[str, str, str]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(
            r"([0-7]{6}) (blob|commit) ([0-9a-f]{40})\t([^\0]+)",
            line,
        )
        require(
            match is not None,
            f"{context} line {line_number} is not a recursive Git tree entry",
        )
        mode, kind, object_id, path = match.groups()
        pure = PurePosixPath(path)
        require(
            path == pure.as_posix()
            and not pure.is_absolute()
            and all(part not in ("", ".", "..") for part in pure.parts),
            f"{context} line {line_number} has a non-canonical path",
        )
        require(path not in entries, f"{context} duplicates path {path!r}")
        entries[path] = (mode, kind, object_id)
    require(entries, f"{context} is empty")
    return entries


def git_tree_id(
    entries: dict[str, tuple[str, str, str]],
    context: str,
) -> str:
    root: dict[str, Any] = {}
    for path, value in entries.items():
        parts = PurePosixPath(path).parts
        node = root
        for part in parts[:-1]:
            existing = node.setdefault(part, {})
            require(
                isinstance(existing, dict),
                f"{context} path prefix {part!r} is also a file",
            )
            node = existing
        require(parts[-1] not in node, f"{context} duplicates path {path!r}")
        node[parts[-1]] = value

    def hash_node(node: dict[str, Any]) -> str:
        serialized_entries: list[tuple[bytes, bytes]] = []
        for name, value in node.items():
            name_bytes = name.encode("utf-8")
            if isinstance(value, dict):
                mode = "40000"
                object_id = hash_node(value)
                sort_key = name_bytes + b"/"
            else:
                mode, _kind, object_id = value
                sort_key = name_bytes + b"\0"
            serialized = (
                mode.encode("ascii")
                + b" "
                + name_bytes
                + b"\0"
                + bytes.fromhex(object_id)
            )
            serialized_entries.append((sort_key, serialized))
        body = b"".join(
            serialized
            for _sort_key, serialized in sorted(
                serialized_entries,
                key=lambda item: item[0],
            )
        )
        return git_object_id("tree", body)

    return hash_node(root)


def validate_git_provenance(
    evidence_dir: Path,
    value: Any,
    *,
    benchmark_repo: str,
    faiss_source_root: str,
    source_records: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    provenance = expect_mapping(value, "preflight source provenance")
    expect_exact_keys(
        provenance,
        {
            "benchmark_input_sources_path",
            "benchmark_input_sources_sha256",
            "compiled_source_hashes_path",
            "compiled_source_hashes_sha256",
            "git_repositories",
            "faiss_audited_reference",
        },
        "preflight source provenance",
    )
    repositories = expect_list(
        provenance["git_repositories"],
        "preflight Git repositories",
    )
    require(repositories, "Preflight Git repository list is empty")
    by_root: dict[str, dict[str, Any]] = {}
    validated_repositories: dict[str, dict[str, Any]] = {}
    for index, repository_value in enumerate(repositories):
        context = f"preflight Git repository {index}"
        repository = expect_mapping(repository_value, context)
        expect_exact_keys(
            repository,
            {
                "root",
                "head",
                "tree",
                "dirty",
                "status_path",
                "status_sha256",
                "diff_path",
                "diff_sha256",
                "commit_object_path",
                "commit_object_sha256",
                "tree_listing_path",
                "tree_listing_sha256",
            },
            context,
        )
        root = expect_string(repository["root"], f"{context} root")
        require(Path(root).is_absolute(), f"{context} root is not absolute")
        require(
            os.path.abspath(root) == root,
            f"{context} root is not a canonical absolute path",
        )
        require(root not in by_root, f"Preflight duplicates Git root {root}")
        by_root[root] = repository
        require(
            GIT_OBJECT_ID.fullmatch(
                expect_string(repository["head"], f"{context} head")
            )
            is not None,
            f"{context} head is not a Git object ID",
        )
        require(
            GIT_OBJECT_ID.fullmatch(
                expect_string(repository["tree"], f"{context} tree")
            )
            is not None,
            f"{context} tree is not a Git object ID",
        )
        status_record = {
            "status_path": repository["status_path"],
            "sha256": repository["status_sha256"],
        }
        status_path = validate_capture_reference(
            evidence_dir,
            status_record,
            context=f"{context} status",
            path_key="status_path",
            expected_path=f"source-repository-{index:02d}-git-status.txt",
        )
        diff_record = {
            "diff_path": repository["diff_path"],
            "sha256": repository["diff_sha256"],
        }
        diff_path = validate_capture_reference(
            evidence_dir,
            diff_record,
            context=f"{context} diff",
            path_key="diff_path",
            expected_path=f"source-repository-{index:02d}-source-diff.patch",
        )
        dirty = _read_text(status_path, f"{context} status") != ""
        require(
            type(repository["dirty"]) is bool, f"{context} dirty flag is not boolean"
        )
        require(
            repository["dirty"] is dirty, f"{context} dirty flag differs from status"
        )
        _read_text(diff_path, f"{context} diff")

        commit_path = validate_capture_reference(
            evidence_dir,
            {
                "commit_object_path": repository["commit_object_path"],
                "sha256": repository["commit_object_sha256"],
            },
            context=f"{context} commit object",
            path_key="commit_object_path",
            expected_path=f"source-repository-{index:02d}-commit-object.txt",
        )
        commit_bytes = read_verified_bytes(
            commit_path,
            f"{context} commit object",
        )
        require(
            git_object_id("commit", commit_bytes) == repository["head"],
            f"{context} commit bytes do not match its head",
        )
        commit_tree, commit_parents = parse_git_commit_identity(
            commit_bytes,
            f"{context} commit object",
        )
        require(
            commit_tree == repository["tree"],
            f"{context} commit tree differs",
        )

        listing_path = validate_capture_reference(
            evidence_dir,
            {
                "tree_listing_path": repository["tree_listing_path"],
                "sha256": repository["tree_listing_sha256"],
            },
            context=f"{context} tree listing",
            path_key="tree_listing_path",
            expected_path=f"source-repository-{index:02d}-tree-listing.txt",
        )
        tree_entries = parse_git_tree_listing(
            _read_text(listing_path, f"{context} tree listing"),
            f"{context} tree listing",
        )
        require(
            git_tree_id(tree_entries, f"{context} tree listing") == repository["tree"],
            f"{context} recursive tree listing does not match its root tree",
        )
        validated_repositories[root] = {
            "record": repository,
            "commit_bytes": commit_bytes,
            "commit_tree": commit_tree,
            "commit_parents": commit_parents,
            "tree_entries": tree_entries,
        }

        if repository["dirty"] is False:
            root_path = Path(root)
            for absolute_path, record in source_records.items():
                try:
                    relative = Path(absolute_path).relative_to(root_path).as_posix()
                except ValueError:
                    continue
                tree_entry = tree_entries.get(relative)
                require(
                    tree_entry is not None and tree_entry[1] == "blob",
                    f"{context} pristine source {relative!r} is absent from HEAD",
                )
                _captured_text, captured_path = relative_evidence_path(
                    evidence_dir,
                    record["captured_path"],
                    f"{context} pristine source {relative}",
                )
                require(
                    git_object_id(
                        "blob",
                        read_verified_bytes(
                            captured_path,
                            f"{context} pristine source {relative}",
                        ),
                    )
                    == tree_entry[2],
                    f"{context} pristine source {relative!r} differs from HEAD",
                )

    reference_context = "preflight FAISS audited reference"
    reference = expect_mapping(
        provenance["faiss_audited_reference"],
        reference_context,
    )
    expect_exact_keys(
        reference,
        {"policy", "repository_root", "base", "derivative", "patch", "feature"},
        reference_context,
    )
    require(
        reference["policy"] == "single-parent-pinned-tree-v1",
        "FAISS audited-reference policy differs",
    )
    repository_root = expect_string(
        reference["repository_root"],
        f"{reference_context} repository root",
    )
    require(
        repository_root == faiss_source_root,
        "FAISS audited root differs from the linked FAISS source root",
    )

    base = expect_mapping(reference["base"], f"{reference_context} base")
    expect_exact_keys(
        base,
        {"commit", "tree", "tag"},
        f"{reference_context} base",
    )
    require(
        base
        == {
            "commit": PINNED_FAISS_COMMIT,
            "tree": PINNED_FAISS_BASE_TREE,
            "tag": "v1.15.0",
        },
        "FAISS audited base identity differs",
    )

    derivative = expect_mapping(
        reference["derivative"],
        f"{reference_context} derivative",
    )
    expect_exact_keys(
        derivative,
        {"head", "tree", "parent", "dirty", "changed_paths"},
        f"{reference_context} derivative",
    )
    derivative_head = expect_string(
        derivative["head"],
        f"{reference_context} derivative head",
    )
    require(
        GIT_OBJECT_ID.fullmatch(derivative_head) is not None,
        "FAISS derivative head is not a Git object ID",
    )
    require(
        derivative_head != PINNED_FAISS_COMMIT,
        "FAISS derivative cannot use the pinned base commit directly",
    )
    require(
        derivative["tree"] == PINNED_FAISS_STATS_DISABLED_TREE,
        "FAISS derivative tree differs from the pinned stats-disabled tree",
    )
    require(
        derivative["parent"] == PINNED_FAISS_COMMIT,
        "FAISS derivative parent differs from the pinned base commit",
    )
    require(
        type(derivative["dirty"]) is bool and derivative["dirty"] is False,
        "FAISS derivative is dirty",
    )
    changed_paths_value = expect_list(
        derivative["changed_paths"],
        f"{reference_context} derivative changed paths",
    )
    require(
        changed_paths_value == list(PINNED_FAISS_STATS_PATCH_PATHS),
        "FAISS derivative changed-path record is not canonical",
    )

    patch = expect_mapping(reference["patch"], f"{reference_context} patch")
    expect_exact_keys(
        patch,
        {"source_path", "captured_path", "sha256"},
        f"{reference_context} patch",
    )
    patch_source = expect_string(
        patch["source_path"],
        f"{reference_context} patch source path",
    )
    expected_patch_source = os.path.abspath(
        os.path.join(benchmark_repo, FAISS_STATS_PATCH_SOURCE)
    )
    require(
        patch_source == expected_patch_source,
        "FAISS patch source path differs from the benchmark repository",
    )
    require(
        patch["sha256"] == PINNED_FAISS_STATS_PATCH_SHA256,
        "FAISS patch recorded hash differs from the pinned patch",
    )
    patch_path = validate_capture_reference(
        evidence_dir,
        patch,
        context=f"{reference_context} patch",
        expected_path=FAISS_STATS_PATCH_FILENAME,
    )
    patch_bytes = read_verified_bytes(
        patch_path,
        f"{reference_context} patch",
    )
    require(
        hashlib.sha256(patch_bytes).hexdigest()
        == PINNED_FAISS_STATS_PATCH_SHA256,
        "FAISS patch bytes differ from the pinned patch",
    )

    feature = expect_mapping(reference["feature"], f"{reference_context} feature")
    expect_exact_keys(
        feature,
        {"cmake_option", "required_value", "compile_definition"},
        f"{reference_context} feature",
    )
    require(
        feature
        == {
            "cmake_option": "FAISS_ENABLE_GLOBAL_HNSW_STATS",
            "required_value": "OFF",
            "compile_definition": "FAISS_DISABLE_GLOBAL_HNSW_STATS=1",
        },
        "FAISS audited feature policy differs",
    )

    faiss_repository = validated_repositories.get(repository_root)
    require(faiss_repository is not None, "FAISS source repository was not captured")
    faiss_record = faiss_repository["record"]
    require(
        faiss_record["head"] == derivative_head
        and faiss_record["tree"] == derivative["tree"],
        "Captured FAISS repository differs from the audited derivative",
    )
    require(
        faiss_record["dirty"] is False,
        "Captured FAISS derivative repository is dirty",
    )
    require(
        faiss_repository["commit_tree"] == PINNED_FAISS_STATS_DISABLED_TREE,
        "FAISS derivative commit has the wrong tree",
    )
    require(
        faiss_repository["commit_parents"] == (PINNED_FAISS_COMMIT,),
        "FAISS derivative must have exactly the pinned base commit as its parent",
    )
    derivative_entries = faiss_repository["tree_entries"]
    require(
        git_tree_id(derivative_entries, "FAISS derivative tree listing")
        == PINNED_FAISS_STATS_DISABLED_TREE,
        "FAISS derivative recursive tree differs from the pinned tree",
    )

    base_commit_path = evidence_dir / FAISS_BASE_COMMIT_OBJECT_FILENAME
    require(base_commit_path.is_file(), "FAISS base commit capture is missing")
    base_commit_bytes = read_verified_bytes(
        base_commit_path,
        "FAISS base commit capture",
    )
    require(
        git_object_id("commit", base_commit_bytes) == PINNED_FAISS_COMMIT,
        "FAISS base commit bytes differ from the pinned commit",
    )
    base_commit_tree, _base_parents = parse_git_commit_identity(
        base_commit_bytes,
        "FAISS base commit capture",
    )
    require(
        base_commit_tree == PINNED_FAISS_BASE_TREE,
        "FAISS base commit has the wrong tree",
    )
    base_listing_path = evidence_dir / FAISS_BASE_TREE_LISTING_FILENAME
    require(base_listing_path.is_file(), "FAISS base tree listing is missing")
    base_entries = parse_git_tree_listing(
        _read_text(base_listing_path, "FAISS base tree listing"),
        "FAISS base tree listing",
    )
    require(
        git_tree_id(base_entries, "FAISS base tree listing")
        == PINNED_FAISS_BASE_TREE,
        "FAISS base recursive tree differs from the pinned tree",
    )

    recomputed_changed_paths = tuple(
        sorted(
            (
                path
                for path in set(base_entries) | set(derivative_entries)
                if base_entries.get(path) != derivative_entries.get(path)
            ),
            key=lambda path: path.encode("utf-8"),
        )
    )
    require(
        recomputed_changed_paths == PINNED_FAISS_STATS_PATCH_PATHS,
        "FAISS derivative tree changes differ from the pinned patch paths",
    )
    changed_path_capture = evidence_dir / FAISS_CHANGED_PATHS_FILENAME
    require(
        changed_path_capture.is_file(),
        "FAISS changed-path capture is missing",
    )
    captured_changed_paths = parse_changed_paths(
        _read_text(changed_path_capture, "FAISS changed-path capture"),
        "FAISS changed-path capture",
    )
    require(
        captured_changed_paths == PINNED_FAISS_STATS_PATCH_PATHS
        and captured_changed_paths == recomputed_changed_paths,
        "FAISS changed-path capture differs from the derivative trees",
    )

    derivative_diff_path = evidence_dir / FAISS_DERIVATIVE_DIFF_FILENAME
    require(derivative_diff_path.is_file(), "FAISS derivative diff capture is missing")
    derivative_diff = read_verified_bytes(
        derivative_diff_path,
        "FAISS derivative diff capture",
    )
    require(
        hashlib.sha256(derivative_diff).hexdigest()
        == PINNED_FAISS_STATS_PATCH_SHA256,
        "FAISS derivative diff hash differs from the pinned patch",
    )
    require(
        derivative_diff == patch_bytes,
        "FAISS derivative diff bytes differ from the pinned patch artifact",
    )

    faiss_compiled_sources = 0
    root_path = Path(repository_root)
    for absolute_path, record in source_records.items():
        try:
            relative = Path(absolute_path).relative_to(root_path).as_posix()
        except ValueError:
            continue
        faiss_compiled_sources += 1
        tree_entry = derivative_entries.get(relative)
        require(
            tree_entry is not None and tree_entry[1] == "blob",
            f"FAISS derivative compiled source {relative!r} is absent from its tree",
        )
        _captured_text, captured_path = relative_evidence_path(
            evidence_dir,
            record["captured_path"],
            f"FAISS derivative compiled source {relative}",
        )
        require(
            git_object_id(
                "blob",
                read_verified_bytes(
                    captured_path,
                    f"FAISS derivative compiled source {relative}",
                ),
            )
            == tree_entry[2],
            f"FAISS derivative compiled source {relative!r} differs from its tree",
        )
    require(
        faiss_compiled_sources > 0,
        "FAISS derivative has no compiled source records",
    )
    return repository_root, PINNED_FAISS_STATS_DISABLED_TREE


def validate_linked_faiss_build(
    evidence_dir: Path,
    value: Any,
    *,
    source_records: dict[str, dict[str, Any]],
    runtime_artifacts: dict[str, str],
    wrapper_build: dict[str, Any],
) -> tuple[str, str]:
    context = "preflight linked FAISS build"
    build = expect_mapping(value, context)
    expect_exact_keys(
        build,
        {
            "library",
            "build_directory",
            "cmake_home_directory",
            "cmake_cache",
            "compile_commands",
            "settings",
            "compiler",
            "linker",
            "build_tool",
            "ninja_commands",
            "ninja_noop",
            "build_input_closure",
            "response_files",
            "compiled_sources",
            "openmp_runtime",
        },
        context,
    )
    library = expect_mapping(build["library"], f"{context} library")
    expect_exact_keys(library, {"path", "sha256", "bytes"}, f"{context} library")
    library_path = expect_string(library["path"], f"{context} library path")
    require(Path(library_path).is_absolute(), f"{context} library path is not absolute")
    library_digest = expect_sha256(library["sha256"], f"{context} library sha256")
    expect_int(library["bytes"], f"{context} library bytes", minimum=1)
    require(
        runtime_artifacts.get(library_path) == library_digest,
        "Linked FAISS library is absent from runtime artifacts",
    )
    for key in ("build_directory", "cmake_home_directory"):
        require(
            Path(expect_string(build[key], f"{context} {key}")).is_absolute(),
            f"{context} {key} is not absolute",
        )

    cache_reference = expect_mapping(build["cmake_cache"], f"{context} CMake cache")
    expect_exact_keys(
        cache_reference, {"captured_path", "sha256"}, f"{context} CMake cache"
    )
    cache_path = validate_capture_reference(
        evidence_dir,
        cache_reference,
        context=f"{context} CMake cache",
        expected_path="faiss-library-CMakeCache.txt",
    )
    cache = parse_cmake_cache(
        _read_text(cache_path, f"{context} CMake cache"),
        f"{context} CMake cache",
    )
    compile_reference = expect_mapping(
        build["compile_commands"],
        f"{context} compile database",
    )
    expect_exact_keys(
        compile_reference,
        {"captured_path", "sha256", "entries"},
        f"{context} compile database",
    )
    compile_path = validate_capture_reference(
        evidence_dir,
        compile_reference,
        context=f"{context} compile database",
        expected_path="faiss-library-compile_commands.json",
    )
    compile_value = read_json(compile_path, f"{context} compile database")
    compile_entries = expect_list(compile_value, f"{context} compile database")
    require(
        expect_int(compile_reference["entries"], f"{context} entries", minimum=1)
        == len(compile_entries),
        f"{context} compile database entry count differs",
    )
    response_files = validate_response_file_records(
        evidence_dir,
        build["response_files"],
        f"{context} response files",
    )
    used_response_files: set[tuple[str, str]] = set()
    validate_release_compile_inputs(
        cache,
        compile_entries,
        context=context,
        response_files=response_files,
        used_response_files=used_response_files,
    )
    validate_required_compile_definition(
        compile_entries,
        name="FAISS_DISABLE_GLOBAL_HNSW_STATS",
        value="1",
        context=context,
        response_files=response_files,
        used_response_files=used_response_files,
    )
    require(
        compile_source_paths(compile_entries, f"{context} compile database")
        == set(source_records),
        f"{context} compiled-source records differ from the compile database",
    )
    require(
        expect_int(build["compiled_sources"], f"{context} compiled sources", minimum=1)
        == len(source_records),
        f"{context} compiled-source count differs",
    )

    settings = expect_mapping(build["settings"], f"{context} settings")
    expected_settings = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_OSX_ARCHITECTURES",
        "CMAKE_OSX_DEPLOYMENT_TARGET",
        "CMAKE_OSX_SYSROOT",
        "CMAKE_CXX_FLAGS_RELEASE",
        "BUILD_SHARED_LIBS",
        "FAISS_ENABLE_GPU",
        "FAISS_ENABLE_PYTHON",
        "FAISS_ENABLE_GLOBAL_HNSW_STATS",
        "OpenMP_CXX_FLAGS",
        "OpenMP_CXX_INCLUDE_DIR",
        "OpenMP_libomp_LIBRARY",
    }
    expect_exact_keys(settings, expected_settings, f"{context} settings")
    require(
        settings == {name: cache.get(name) for name in expected_settings},
        f"{context} settings differ from its CMake cache",
    )
    require(
        settings["FAISS_ENABLE_GLOBAL_HNSW_STATS"] == "OFF",
        "Linked FAISS library requires FAISS_ENABLE_GLOBAL_HNSW_STATS=OFF",
    )
    compiler = validate_executable_record(
        evidence_dir,
        build["compiler"],
        f"{context} compiler",
        runtime_artifacts=runtime_artifacts,
    )
    linker = validate_executable_record(
        evidence_dir,
        build["linker"],
        f"{context} linker",
        runtime_artifacts=runtime_artifacts,
    )
    build_tool = validate_executable_record(
        evidence_dir,
        build["build_tool"],
        f"{context} build tool",
        runtime_artifacts=runtime_artifacts,
    )
    for tool_name, tool in (
        ("compiler", compiler),
        ("linker", linker),
        ("build_tool", build_tool),
    ):
        wrapper_tool = wrapper_build[tool_name]
        require(
            tool["resolved_path"] == wrapper_tool["resolved_path"]
            and tool["sha256"] == wrapper_tool["sha256"],
            f"{context} {tool_name} differs from the FAISS wrapper",
        )

    faiss_link_arguments: list[str] | None = None
    for name, expected_path in (
        ("ninja_commands", "faiss-library-ninja-commands.txt"),
        ("ninja_noop", "faiss-library-ninja-noop.txt"),
    ):
        reference = expect_mapping(build[name], f"{context} {name}")
        expect_exact_keys(reference, {"captured_path", "sha256"}, f"{context} {name}")
        captured = validate_capture_reference(
            evidence_dir,
            reference,
            context=f"{context} {name}",
            expected_path=expected_path,
        )
        text = _read_text(captured, f"{context} {name}")
        require(text.strip() != "", f"{context} {name} capture is empty")
        if name == "ninja_commands":
            faiss_link_arguments = validate_ninja_commands(
                text,
                cache=cache,
                compile_entries=compile_entries,
                build_directory=Path(build["build_directory"]),
                expected_output=Path(library_path),
                context=f"{context} Ninja commands",
                response_files=response_files,
                used_response_files=used_response_files,
            )
        else:
            require(
                "ninja: no work to do." in text,
                "Linked FAISS build tree was not captured as up to date",
            )
    require(
        faiss_link_arguments is not None,
        "Linked FAISS build has no validated link command",
    )
    require(
        used_response_files == set(response_files),
        f"{context} contains unreferenced response-file records",
    )
    validated_closure = validate_build_input_closure(
        evidence_dir,
        build["build_input_closure"],
        capture_prefix="faiss-library",
        requested_target=ninja_target_for_output(
            Path(library_path),
            build_directory=Path(build["build_directory"]),
            context=context,
        ),
        expected_output=library_path,
        build_directory=Path(build["build_directory"]),
        build_tool_path=build_tool["resolved_path"],
        cache=cache,
        compile_entries=compile_entries,
        link_arguments=faiss_link_arguments,
        response_files=response_files,
        used_response_files=used_response_files,
        context=f"{context} build-input closure",
    )
    validate_compiled_sources_against_closure(
        source_records,
        validated_closure,
        context=context,
    )
    openmp = validate_dependency_entry(
        build["openmp_runtime"],
        context=f"{context} OpenMP runtime",
        runtime_artifacts=runtime_artifacts,
    )
    require(
        openmp["sha256"] == PINNED_LIBOMP_SHA256,
        "Linked FAISS OpenMP runtime is not pinned",
    )
    require(
        settings["OpenMP_libomp_LIBRARY"] == openmp["resolved_path"],
        "Linked FAISS OpenMP setting differs from the dependency graph",
    )
    wrapper_dependencies = expect_list(
        wrapper_build["_validated_dependency_manifest"]["dependencies"],
        "preflight FAISS wrapper dependencies",
    )
    wrapper_faiss_libraries = [
        dependency
        for dependency in wrapper_dependencies
        if isinstance(dependency.get("resolved_path"), str)
        and Path(dependency["resolved_path"]).name == "libfaiss.dylib"
    ]
    require(
        len(wrapper_faiss_libraries) == 1,
        "FAISS wrapper must resolve exactly one libfaiss.dylib",
    )
    wrapper_library = wrapper_faiss_libraries[0]
    require(
        wrapper_library["resolved_path"] == library_path
        and wrapper_library["sha256"] == library_digest,
        "FAISS wrapper resolves a different libfaiss.dylib",
    )
    source_root = expect_string(
        build["cmake_home_directory"],
        f"{context} source root",
    )
    require(
        os.path.abspath(source_root) == source_root,
        f"{context} source root is not canonical",
    )
    return source_root, PINNED_FAISS_STATS_DISABLED_TREE


def validate_build_and_source_provenance(
    evidence_dir: Path,
    preflight: dict[str, Any],
    *,
    binaries: dict[str, dict[str, Any]],
    runtime_artifacts: dict[str, str],
    runtime_artifact_paths: dict[str, Path],
) -> None:
    source_provenance = expect_mapping(
        preflight.get("source_provenance"),
        "preflight source provenance",
    )
    benchmark_path_text, benchmark_path = relative_evidence_path(
        evidence_dir,
        source_provenance.get("benchmark_input_sources_path"),
        "benchmark input sources path",
    )
    require(
        benchmark_path_text == "benchmark-input-source-hashes.json",
        "Benchmark input source record path differs",
    )
    require(benchmark_path.is_file(), "Benchmark input source record is missing")
    require(
        sha256_file(benchmark_path)
        == expect_sha256(
            source_provenance.get("benchmark_input_sources_sha256"),
            "benchmark input source record sha256",
        ),
        "Benchmark input source record hash differs",
    )
    source_path_text, source_path = relative_evidence_path(
        evidence_dir,
        source_provenance.get("compiled_source_hashes_path"),
        "compiled source hashes path",
    )
    require(
        source_path_text == "compiled-source-hashes.json",
        "Compiled-source record path differs",
    )
    require(source_path.is_file(), "Compiled-source record is missing")
    require(
        sha256_file(source_path)
        == expect_sha256(
            source_provenance.get("compiled_source_hashes_sha256"),
            "compiled source record sha256",
        ),
        "Compiled-source record hash differs",
    )

    compiled_value = expect_mapping(
        read_json(source_path, "compiled-source-hashes.json"),
        "compiled-source-hashes.json",
    )
    expect_exact_keys(
        compiled_value,
        {"infinity", "faiss", "faiss_library"},
        "compiled-source-hashes.json",
    )
    compiled_records = {
        name: validate_source_records(
            evidence_dir,
            compiled_value[name],
            f"compiled-source-hashes.json {name}",
        )
        for name in ("infinity", "faiss", "faiss_library")
    }
    benchmark_inputs = validate_benchmark_input_sources(
        evidence_dir,
        read_json(benchmark_path, "benchmark-input-source-hashes.json"),
        runtime_artifacts=runtime_artifacts,
    )
    validate_cross_record_source_identities(
        [
            *[
                (f"compiled {name}", records)
                for name, records in compiled_records.items()
            ],
            *[
                (f"benchmark input {name}", records)
                for name, records in benchmark_inputs.items()
            ],
        ]
    )

    builds_value = expect_mapping(preflight.get("builds"), "preflight builds")
    expect_exact_keys(builds_value, {"infinity", "faiss"}, "preflight builds")
    builds = {
        engine: validate_build_record(
            evidence_dir,
            builds_value[engine],
            engine=engine,
            binary=binaries[engine],
            source_records=compiled_records[engine],
            runtime_artifacts=runtime_artifacts,
            runtime_artifact_paths=runtime_artifact_paths,
        )
        for engine in ("infinity", "faiss")
    }
    for tool_name in ("compiler", "linker", "build_tool"):
        require(
            builds["infinity"][tool_name]["resolved_path"]
            == builds["faiss"][tool_name]["resolved_path"]
            and builds["infinity"][tool_name]["sha256"]
            == builds["faiss"][tool_name]["sha256"],
            f"Infinity and FAISS use different {tool_name} identities",
        )
    require(
        {
            name: builds["infinity"]["settings"][name]
            for name in COMMON_BUILD_SETTINGS
        }
        == {
            name: builds["faiss"]["settings"][name]
            for name in COMMON_BUILD_SETTINGS
        },
        "Infinity and FAISS wrapper common build settings differ",
    )
    require(
        os.path.abspath(builds["infinity"]["cmake_home_directory"])
        == builds["infinity"]["cmake_home_directory"]
        == os.path.abspath(preflight["repo"])
        == preflight["repo"],
        "Infinity production build source root differs from the benchmark repo",
    )
    infinity_settings = builds["infinity"]["settings"]
    expected_include_root = (
        Path(infinity_settings["VCPKG_INSTALLED_DIR"])
        / infinity_settings["VCPKG_TARGET_TRIPLET"]
        / "include"
    )
    expected_include_root = Path(os.path.abspath(expected_include_root))
    ctpl_paths = set(benchmark_inputs["ctpl"])
    require(
        ctpl_paths == {str(expected_include_root / "ctpl_stl.h")},
        "Recorded CTPL header does not match the production vcpkg tree",
    )
    expected_simde_root = expected_include_root / "simde"
    require(
        all(
            Path(path).is_relative_to(expected_simde_root)
            for path in benchmark_inputs["simde_headers"]
        ),
        "Recorded SIMDe headers do not match the production vcpkg tree",
    )
    closure_files = {
        record["resolved_path"]: record
        for record in builds["infinity"]["_validated_build_input_closure"]["files"]
    }
    require(
        next(iter(ctpl_paths)) in closure_files,
        "Production build-input closure omits the recorded CTPL header",
    )
    compiled_simde_paths = {
        path
        for path in closure_files
        if Path(path).is_relative_to(expected_simde_root)
    }
    require(
        compiled_simde_paths
        and compiled_simde_paths <= set(benchmark_inputs["simde_headers"]),
        "Production build-input closure SIMDe headers are not fully recorded",
    )
    for path in {next(iter(ctpl_paths)), *compiled_simde_paths}:
        benchmark_record = (
            benchmark_inputs["ctpl"].get(path)
            or benchmark_inputs["simde_headers"].get(path)
        )
        closure_record = closure_files[path]
        require(
            benchmark_record is not None
            and benchmark_record["sha256"] == closure_record["sha256"]
            and benchmark_record["bytes"] == closure_record["bytes"],
            f"Benchmark header identity differs from the build closure: {path}",
        )

    linked_faiss_identity = validate_linked_faiss_build(
        evidence_dir,
        preflight.get("linked_faiss_build"),
        source_records=compiled_records["faiss_library"],
        runtime_artifacts=runtime_artifacts,
        wrapper_build=builds["faiss"],
    )
    native_harness_paths = set(benchmark_inputs["native_harness"])
    for engine in ("infinity", "faiss"):
        harness_sources = {
            path
            for path in compiled_records[engine]
            if "/tools/apple_silicon/native_hnsw_smoke/" in path
        }
        require(
            harness_sources <= native_harness_paths,
            f"{engine} compiled harness sources are missing from benchmark inputs",
        )
    audited_faiss_identity = validate_git_provenance(
        evidence_dir,
        source_provenance,
        benchmark_repo=preflight["repo"],
        faiss_source_root=linked_faiss_identity[0],
        source_records={
            path: record
            for records in compiled_records.values()
            for path, record in records.items()
        },
    )
    require(
        audited_faiss_identity == linked_faiss_identity,
        "Linked FAISS build and audited derivative source state differ",
    )


def validate_runtime_artifact_snapshots(
    evidence_dir: Path,
    value: Any,
    *,
    expected_hashes: dict[str, str],
) -> dict[str, Path]:
    snapshots = expect_list(value, "preflight runtime artifact snapshots")
    require(
        len(snapshots) == len(expected_hashes),
        "Runtime artifact snapshots do not cover the complete artifact set",
    )
    recorded_paths: list[str] = []
    captured_paths: dict[str, Path] = {}
    for index, snapshot_value in enumerate(snapshots):
        context = f"preflight runtime artifact snapshot {index}"
        snapshot = expect_mapping(snapshot_value, context)
        expect_exact_keys(
            snapshot,
            {"original_path", "captured_path", "sha256", "bytes"},
            context,
        )
        original_path = expect_string(
            snapshot["original_path"],
            f"{context} original path",
        )
        require(Path(original_path).is_absolute(), f"{context} path is not absolute")
        require(original_path not in recorded_paths, f"{context} path is duplicated")
        recorded_paths.append(original_path)
        digest = expect_sha256(snapshot["sha256"], f"{context} sha256")
        require(
            expected_hashes.get(original_path) == digest,
            f"{context} identity differs from the runtime artifact map",
        )
        size = expect_int(snapshot["bytes"], f"{context} bytes", minimum=1)
        captured_text, captured_path = relative_evidence_path(
            evidence_dir,
            snapshot["captured_path"],
            f"{context} captured path",
        )
        require(
            captured_text == f"provenance/blobs/{digest[:2]}/{digest}",
            f"{context} is not stored at its content-addressed path",
        )
        require(
            captured_path.is_file()
            and captured_path.stat().st_size == size
            and sha256_file(captured_path) == digest,
            f"{context} captured content differs",
        )
        captured_paths[original_path] = captured_path
    require(
        recorded_paths == sorted(expected_hashes),
        "Runtime artifact snapshots are not in canonical path order",
    )
    return captured_paths


def resolve_invocation_path(value: str, working_directory: str) -> str:
    return lexical_absolute_path(
        value,
        working_directory=Path(working_directory),
    )


def parse_runner_argv(
    argv: list[Any],
    *,
    working_directory: str,
    runner_source_path: str,
) -> dict[str, Any]:
    require(
        argv and all(isinstance(item, str) and item for item in argv),
        "Preflight runner argv is empty or invalid",
    )
    require(
        resolve_invocation_path(argv[0], working_directory) == runner_source_path,
        "Preflight runner argv[0] does not resolve to its source path",
    )
    option_names = {
        "--repo": "repo",
        "--infinity-binary": "infinity_binary",
        "--faiss-binary": "faiss_binary",
        "--output-dir": "output_directory",
        "--development-idle-minimum-percent": "idle_minimum_percent",
        "--idle-timeout-seconds": "idle_timeout_seconds",
        "--member-timeout-seconds": "member_timeout_seconds",
    }
    parsed: dict[str, str] = {}
    preflight_only = False
    index = 1
    while index < len(argv):
        option = argv[index]
        if option == "--preflight-only":
            require(
                not preflight_only,
                "Preflight runner argv duplicates --preflight-only",
            )
            preflight_only = True
            index += 1
            continue
        require(
            option in option_names,
            f"Preflight runner argv contains unknown option {option!r}",
        )
        name = option_names[option]
        require(name not in parsed, f"Preflight runner argv duplicates {option}")
        require(index + 1 < len(argv), f"Preflight runner argv omits {option} value")
        value = argv[index + 1]
        require(
            isinstance(value, str) and value and not value.startswith("--"),
            f"Preflight runner argv has an invalid {option} value",
        )
        parsed[name] = value
        index += 2

    for required in ("infinity_binary", "faiss_binary", "output_directory"):
        require(required in parsed, f"Preflight runner argv omits {required}")
    resolved_repo = resolve_invocation_path(
        parsed.get("repo", working_directory),
        working_directory,
    )
    path_values = {
        "repo": resolved_repo,
        "infinity_binary": resolve_invocation_path(
            parsed["infinity_binary"],
            resolved_repo,
        ),
        "faiss_binary": resolve_invocation_path(
            parsed["faiss_binary"],
            resolved_repo,
        ),
        "output_directory": resolve_invocation_path(
            parsed["output_directory"],
            working_directory,
        ),
    }
    numeric_defaults = {
        "idle_minimum_percent": IDLE_MINIMUM_PERCENT,
        "idle_timeout_seconds": 300.0,
        "member_timeout_seconds": 300.0,
    }
    numeric_values: dict[str, float] = {}
    for name, default in numeric_defaults.items():
        try:
            value = float(parsed[name]) if name in parsed else default
        except ValueError as error:
            raise VerificationError(
                f"Preflight runner argv {name} is not numeric"
            ) from error
        require(math.isfinite(value), f"Preflight runner argv {name} is not finite")
        numeric_values[name] = value
    return {
        **path_values,
        **numeric_values,
        "preflight_only": preflight_only,
    }


def validate_campaign_binding(
    value: Any,
    *,
    dataset_sha256: str,
    schedule_sha256: str,
    binaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    binding = expect_mapping(value, "preflight campaign binding")
    expect_exact_keys(
        binding,
        {
            "schema_version",
            "campaign_nonce",
            "dataset_sha256",
            "schedule_sha256",
            "roles",
        },
        "preflight campaign binding",
    )
    schema_version = expect_int(
        binding["schema_version"],
        "preflight campaign binding schema",
    )
    require(
        schema_version == CAMPAIGN_BINDING_SCHEMA_VERSION,
        "Preflight campaign binding schema differs",
    )
    campaign_nonce = expect_sha256(
        binding["campaign_nonce"],
        "preflight campaign nonce",
    )
    binding_dataset_sha256 = expect_sha256(
        binding["dataset_sha256"],
        "preflight campaign dataset SHA-256",
    )
    binding_schedule_sha256 = expect_sha256(
        binding["schedule_sha256"],
        "preflight campaign schedule SHA-256",
    )
    require(
        binding_dataset_sha256 == dataset_sha256,
        "Preflight campaign dataset SHA-256 differs",
    )
    require(
        binding_schedule_sha256 == schedule_sha256,
        "Preflight campaign schedule SHA-256 differs",
    )
    roles_value = expect_mapping(
        binding["roles"],
        "preflight campaign binding roles",
    )
    expect_exact_keys(
        roles_value,
        set(STANDALONE_ROLE_IDS),
        "preflight campaign binding roles",
    )
    roles: dict[str, dict[str, Any]] = {}
    for engine, role_id in STANDALONE_ROLE_IDS.items():
        role = expect_mapping(
            roles_value[engine],
            f"preflight campaign {engine} role",
        )
        expect_exact_keys(
            role,
            {"role_id", "engine", "binary_sha256"},
            f"preflight campaign {engine} role",
        )
        actual_role_id = expect_int(
            role["role_id"],
            f"preflight campaign {engine} role ID",
        )
        actual_engine = expect_string(
            role["engine"],
            f"preflight campaign {engine} engine",
        )
        actual_binary_sha256 = expect_sha256(
            role["binary_sha256"],
            f"preflight campaign {engine} binary SHA-256",
        )
        require(
            {
                "role_id": actual_role_id,
                "engine": actual_engine,
                "binary_sha256": actual_binary_sha256,
            }
            == {
                "role_id": role_id,
                "engine": engine,
                "binary_sha256": binaries[engine]["sha256"],
            },
            f"Preflight campaign {engine} role differs from binary identity",
        )
        roles[engine] = role
    return {
        "schema_version": schema_version,
        "campaign_nonce": campaign_nonce,
        "dataset_sha256": binding_dataset_sha256,
        "schedule_sha256": binding_schedule_sha256,
        "roles": roles,
    }


def validate_preflight(
    evidence_dir: Path,
    preflight_value: Any,
    *,
    dataset: dict[str, Any],
    dataset_sha256: str,
    schedule_sha256: str,
    heldout_manifest: dict[str, Any],
    heldout_manifest_sha256: str,
    heldout_manifest_bytes: int,
    expected_execution_mode: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    str,
    int,
    str,
    float,
    str,
    dict[str, Any],
]:
    preflight = expect_mapping(preflight_value, "preflight.json")
    expect_exact_keys(
        preflight,
        {
            "evidence_schema_version",
            "execution_mode",
            "captured_at",
            "scope",
            "repo",
            "runner",
            "verifier",
            "host",
            "environment_control",
            "dataset",
            "heldout",
            "schedule",
            "campaign_binding",
            "binaries",
            "runtime_artifact_hashes",
            "runtime_artifact_snapshots",
            "builds",
            "linked_faiss_build",
            "source_provenance",
            "protocol",
        },
        "preflight.json",
    )
    require(
        expect_int(
            preflight.get("evidence_schema_version"),
            "preflight evidence schema version",
            minimum=1,
        )
        == EVIDENCE_SCHEMA_VERSION,
        "Preflight evidence schema version is unsupported; "
        f"expected {EVIDENCE_SCHEMA_VERSION}, found "
        f"{preflight.get('evidence_schema_version')!r}",
    )
    execution_mode = expect_string(
        preflight.get("execution_mode"),
        "preflight execution mode",
    )
    require(
        execution_mode in (FULL_CAMPAIGN_MODE, PREFLIGHT_ONLY_MODE),
        "Preflight execution mode is invalid",
    )
    require(
        execution_mode == expected_execution_mode,
        f"Evidence execution mode is {execution_mode!r}, verifier expects "
        f"{expected_execution_mode!r}",
    )
    require(preflight.get("scope") == D0_SCOPE, "Preflight scope is not D0")
    require(
        Path(
            expect_string(preflight.get("repo"), "preflight repository")
        ).is_absolute(),
        "Preflight repository path is not absolute",
    )
    require(
        expect_string(
            preflight.get("captured_at"), "preflight capture timestamp"
        ).endswith("Z"),
        "Preflight capture timestamp is not UTC",
    )
    runner = expect_mapping(preflight.get("runner"), "preflight runner")
    expect_exact_keys(
        runner,
        {
            "source_path",
            "captured_path",
            "sha256",
            "bytes",
            "argv",
            "invocation_working_directory",
            "python_executable",
            "effective_arguments",
        },
        "preflight runner",
    )
    runner_source_path = expect_string(
        runner["source_path"],
        "preflight runner source path",
    )
    require(
        Path(runner_source_path).is_absolute(),
        "Preflight runner source path is not absolute",
    )
    runner_name, runner_path = relative_evidence_path(
        evidence_dir,
        runner["captured_path"],
        "preflight runner captured path",
    )
    require(
        runner_name == RUNNER_CAPTURE_FILENAME,
        "Preflight runner capture filename differs",
    )
    require(runner_path.is_file(), "Captured campaign runner is missing")
    runner_sha256 = expect_sha256(runner["sha256"], "preflight runner sha256")
    require(
        sha256_file(runner_path) == runner_sha256,
        "Captured campaign runner hash differs from preflight",
    )
    require(
        expect_int(runner["bytes"], "preflight runner bytes", minimum=1)
        == runner_path.stat().st_size,
        "Captured campaign runner size differs from preflight",
    )
    verifier_record = expect_mapping(
        preflight.get("verifier"),
        "preflight verifier",
    )
    expect_exact_keys(
        verifier_record,
        {"source_path", "captured_path", "sha256", "bytes"},
        "preflight verifier",
    )
    verifier_source_path = expect_string(
        verifier_record["source_path"],
        "preflight verifier source path",
    )
    require(
        Path(verifier_source_path).is_absolute(),
        "Preflight verifier source path is not absolute",
    )
    verifier_name, verifier_path = relative_evidence_path(
        evidence_dir,
        verifier_record["captured_path"],
        "preflight verifier captured path",
    )
    require(
        verifier_name == VERIFIER_CAPTURE_FILENAME,
        "Preflight verifier capture filename differs",
    )
    require(verifier_path.is_file(), "Captured campaign verifier is missing")
    verifier_sha256 = expect_sha256(
        verifier_record["sha256"],
        "preflight verifier sha256",
    )
    require(
        sha256_file(verifier_path) == verifier_sha256,
        "Captured campaign verifier hash differs from preflight",
    )
    require(
        expect_int(verifier_record["bytes"], "preflight verifier bytes", minimum=1)
        == verifier_path.stat().st_size,
        "Captured campaign verifier size differs from preflight",
    )
    invocation_working_directory = expect_string(
        runner["invocation_working_directory"],
        "preflight runner invocation working directory",
    )
    require(
        Path(invocation_working_directory).is_absolute(),
        "Preflight runner invocation working directory is not absolute",
    )
    parsed_runner_arguments = parse_runner_argv(
        expect_list(runner["argv"], "preflight runner argv"),
        working_directory=invocation_working_directory,
        runner_source_path=runner_source_path,
    )
    runner_python_value = runner["python_executable"]
    runner_effective = expect_mapping(
        runner["effective_arguments"],
        "preflight runner effective arguments",
    )
    expect_exact_keys(
        runner_effective,
        {
            "repo",
            "infinity_binary",
            "faiss_binary",
            "output_directory",
            "idle_minimum_percent",
            "idle_timeout_seconds",
            "member_timeout_seconds",
            "preflight_only",
        },
        "preflight runner effective arguments",
    )
    require(
        runner_effective == parsed_runner_arguments,
        "Preflight runner effective arguments differ from argv",
    )
    preflight_only = runner_effective["preflight_only"]
    require(
        type(preflight_only) is bool,
        "Preflight runner preflight_only argument is not boolean",
    )
    require(
        preflight_only == (execution_mode == PREFLIGHT_ONLY_MODE),
        "Preflight runner mode differs from the evidence execution mode",
    )
    require(
        runner_effective["repo"] == preflight["repo"],
        "Preflight runner effective repository differs",
    )
    require(
        Path(
            expect_string(
                runner_effective["output_directory"],
                "preflight runner effective output directory",
            )
        ).is_absolute(),
        "Preflight runner effective output directory is not absolute",
    )

    expected_preflight_dataset = {
        **dataset,
        "record_path": "dataset.json",
        "record_sha256": dataset_sha256,
    }
    require(
        preflight.get("dataset") == expected_preflight_dataset,
        "Preflight dataset record is inconsistent with dataset.json",
    )
    expected_preflight_heldout = {
        **heldout_manifest,
        "record_path": HELDOUT_MANIFEST_FILENAME,
        "record_sha256": heldout_manifest_sha256,
    }
    require(
        preflight.get("heldout") == expected_preflight_heldout,
        "Preflight held-out record is inconsistent with heldout-v1.json",
    )
    heldout_path = evidence_dir / HELDOUT_MANIFEST_FILENAME
    require(
        not heldout_path.is_symlink() and heldout_path.is_file(),
        "Preflight held-out manifest must be a regular non-symlink file",
    )
    require(
        heldout_path.stat().st_size == heldout_manifest_bytes,
        "Preflight held-out manifest size changed",
    )

    schedule_record = expect_mapping(
        preflight.get("schedule"),
        "preflight schedule",
    )
    expect_exact_keys(
        schedule_record,
        {"path", "sha256", "members", "pair_plan"},
        "preflight schedule",
    )
    require(
        schedule_record["path"] == "schedule.json", "Preflight schedule path differs"
    )
    require(
        schedule_record["sha256"] == schedule_sha256,
        "Preflight schedule hash differs from schedule.json",
    )
    require(
        expect_int(schedule_record["members"], "preflight schedule members") == 18,
        "Preflight schedule member count is not 18",
    )
    require(
        schedule_record["pair_plan"] == expected_pair_plan(),
        "Preflight pair plan differs from the frozen plan",
    )

    protocol = expect_mapping(preflight.get("protocol"), "preflight protocol")
    expect_exact_keys(
        protocol,
        {
            "idle_minimum_percent",
            "idle_policy",
            "idle_window_seconds",
            "idle_sample_count",
            "idle_to_launch_max_ns",
            "server_ports",
            "power_policy",
            "battery_minimum_percent",
            "global_swap_policy",
            "member_timeout_seconds",
            "idle_timeout_seconds",
            "audit_sidecar_schema_version",
            "attestation_schema_version",
            "attestation_barrier_phases",
            "index_timing_binding_schema_version",
            "index_timing_maximum_boundary_overhead_ns",
            "process_evidence_schema_version",
            "time_wrapper_path",
            "query_performance",
            "heldout_query_count",
            "heldout_query_seed",
            "recall_points",
        },
        "preflight protocol",
    )
    require(
        expect_int(
            protocol["audit_sidecar_schema_version"],
            "preflight audit sidecar schema",
        )
        == AUDIT_SIDECAR_SCHEMA_VERSION,
        "Preflight audit sidecar schema differs",
    )
    require(
        expect_int(
            protocol["attestation_schema_version"],
            "preflight attestation schema",
        )
        == ATTESTATION_SCHEMA_VERSION,
        "Preflight attestation schema differs",
    )
    require(
        protocol["attestation_barrier_phases"]
        == list(ATTESTATION_BARRIER_PHASES),
        "Preflight attestation barrier phases differ",
    )
    require(
        expect_int(
            protocol["index_timing_binding_schema_version"],
            "preflight index timing binding schema",
        )
        == INDEX_TIMING_BINDING_SCHEMA_VERSION,
        "Preflight index timing binding schema differs",
    )
    require(
        expect_int(
            protocol["index_timing_maximum_boundary_overhead_ns"],
            "preflight maximum index timing boundary overhead",
        )
        == INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS,
        "Preflight maximum index timing boundary overhead differs",
    )
    require(
        expect_int(
            protocol["process_evidence_schema_version"],
            "preflight process evidence schema",
        )
        == PROCESS_EVIDENCE_SCHEMA_VERSION,
        "Preflight process evidence schema differs",
    )
    require(
        protocol["time_wrapper_path"] == "/usr/bin/time",
        "Preflight time-wrapper path differs",
    )
    expect_normalized_equal(
        protocol["query_performance"],
        expected_query_protocol(),
        "preflight query performance protocol",
    )
    require(
        expect_int(
            protocol["heldout_query_count"],
            "preflight held-out query count",
        )
        == HELDOUT_QUERY_COUNT,
        "Preflight held-out query count differs",
    )
    require(
        expect_int(
            protocol["heldout_query_seed"],
            "preflight held-out query seed",
        )
        == HELDOUT_QUERY_SEED,
        "Preflight held-out query seed differs",
    )
    require(
        protocol["recall_points"] == expected_recall_points_json(),
        "Preflight recall points differ from the frozen protocol",
    )
    idle_minimum_percent = expect_number(
        protocol["idle_minimum_percent"],
        "idle minimum",
    )
    require(
        DEVELOPMENT_IDLE_MINIMUM_FLOOR_PERCENT
        <= idle_minimum_percent
        <= IDLE_MINIMUM_PERCENT,
        "Preflight idle minimum is outside the permitted development range",
    )
    recorded_idle_policy = expect_string(
        protocol["idle_policy"],
        "preflight idle policy",
    )
    require(
        recorded_idle_policy == idle_policy(idle_minimum_percent),
        "Preflight idle policy disagrees with its threshold",
    )
    require(
        expect_int(protocol["idle_window_seconds"], "idle window")
        == IDLE_WINDOW_SECONDS,
        "Preflight idle window is not 15 seconds",
    )
    require(
        expect_int(protocol["idle_sample_count"], "idle sample count")
        == IDLE_SAMPLE_COUNT,
        "Preflight idle sample count is not 16",
    )
    require(
        expect_int(protocol["idle_to_launch_max_ns"], "idle-to-launch limit")
        == IDLE_TO_LAUNCH_MAX_NS,
        "Preflight idle-to-launch limit is not one second",
    )
    require(protocol["server_ports"] == SERVER_PORTS, "Preflight server ports differ")
    power_policy = expect_string(protocol["power_policy"], "preflight power policy")
    require(
        power_policy == POWER_POLICY,
        "Preflight power policy is invalid",
    )
    require(
        expect_int(
            protocol["battery_minimum_percent"],
            "preflight battery minimum",
        )
        == BATTERY_MINIMUM_PERCENT,
        "Preflight battery minimum disagrees with its power policy",
    )
    recorded_global_swap_policy = expect_string(
        protocol["global_swap_policy"],
        "preflight global swap policy",
    )
    require(
        recorded_global_swap_policy == GLOBAL_SWAP_POLICY,
        "Preflight global swap policy is invalid",
    )
    require(
        expect_number(protocol["member_timeout_seconds"], "member timeout") > 0,
        "Preflight member timeout must be positive",
    )
    require(
        expect_number(protocol["idle_timeout_seconds"], "idle timeout")
        >= IDLE_WINDOW_SECONDS,
        "Preflight idle timeout is shorter than the idle window",
    )

    binaries_value = expect_mapping(preflight.get("binaries"), "preflight binaries")
    expect_exact_keys(binaries_value, {"infinity", "faiss"}, "preflight binaries")
    binaries: dict[str, dict[str, Any]] = {}
    for engine in ("infinity", "faiss"):
        binary = expect_mapping(
            binaries_value[engine],
            f"preflight {engine} binary",
        )
        expect_exact_keys(
            binary,
            {"path", "description", "sha256", "bytes"},
            f"preflight {engine} binary",
        )
        path = expect_string(binary["path"], f"{engine} binary path")
        require(Path(path).is_absolute(), f"{engine} binary path is not absolute")
        description = expect_string(
            binary["description"],
            f"{engine} binary description",
        )
        require(
            "Mach-O 64-bit executable arm64" in description,
            f"{engine} binary is not recorded as native arm64",
        )
        expect_sha256(binary["sha256"], f"{engine} binary sha256")
        expect_int(binary["bytes"], f"{engine} binary bytes", minimum=1)
        binaries[engine] = binary
    require(
        binaries["infinity"]["sha256"] != binaries["faiss"]["sha256"],
        "Infinity and FAISS binary hashes are identical",
    )
    require(
        binaries["infinity"]["path"] != binaries["faiss"]["path"],
        "Infinity and FAISS binary paths are identical",
    )

    runtime_value = expect_mapping(
        preflight.get("runtime_artifact_hashes"),
        "preflight runtime artifact hashes",
    )
    require(runtime_value, "Preflight runtime artifact set is empty")
    runtime_artifacts: dict[str, str] = {}
    for raw_path, raw_digest in runtime_value.items():
        path = expect_string(raw_path, "runtime artifact path")
        require(
            Path(path).is_absolute(), f"Runtime artifact path is not absolute: {path}"
        )
        runtime_artifacts[path] = expect_sha256(
            raw_digest,
            f"runtime artifact {path} sha256",
        )
    for engine in ("infinity", "faiss"):
        require(
            runtime_artifacts.get(binaries[engine]["path"])
            == binaries[engine]["sha256"],
            f"Preflight runtime artifacts omit or alter the {engine} binary",
        )
    require(
        runtime_artifacts.get(runner_source_path) == runner_sha256,
        "Preflight runtime artifacts omit or alter the executing runner",
    )
    require(
        runtime_artifacts.get(verifier_source_path) == verifier_sha256,
        "Preflight runtime artifacts omit or alter the captured verifier",
    )
    validate_executable_record(
        evidence_dir,
        runner_python_value,
        "preflight runner Python executable",
        runtime_artifacts=runtime_artifacts,
    )
    runtime_artifact_paths = validate_runtime_artifact_snapshots(
        evidence_dir,
        preflight.get("runtime_artifact_snapshots"),
        expected_hashes=runtime_artifacts,
    )
    require(
        runner_effective["infinity_binary"] == binaries["infinity"]["path"]
        and runner_effective["faiss_binary"] == binaries["faiss"]["path"],
        "Preflight runner effective binary paths differ",
    )
    require(
        expect_number(
            runner_effective["idle_minimum_percent"],
            "preflight runner effective idle minimum",
        )
        == idle_minimum_percent
        and expect_number(
            runner_effective["idle_timeout_seconds"],
            "preflight runner effective idle timeout",
        )
        == expect_number(protocol["idle_timeout_seconds"], "preflight idle timeout")
        and expect_number(
            runner_effective["member_timeout_seconds"],
            "preflight runner effective member timeout",
        )
        == expect_number(
            protocol["member_timeout_seconds"], "preflight member timeout"
        ),
        "Preflight runner effective timeout or idle arguments differ",
    )
    host = expect_mapping(preflight.get("host"), "preflight host")
    expect_exact_keys(
        host,
        {"machine", "macos", "kernel", "python", "disk"},
        "preflight host",
    )
    require(host.get("machine") == "arm64", "Preflight host is not arm64")
    for key in ("macos", "kernel", "python"):
        require(
            expect_string(host.get(key), f"preflight host {key}") != "",
            f"Preflight host {key} is empty",
        )
    disk = expect_mapping(host.get("disk"), "preflight disk")
    expect_exact_keys(
        disk,
        {
            "total_bytes",
            "used_bytes",
            "free_bytes",
            "minimum_reserve_bytes",
            "declared_worst_case_new_bytes",
            "required_free_bytes",
        },
        "preflight disk",
    )
    disk_values = {
        key: expect_int(value, f"preflight disk {key}", minimum=0)
        for key, value in disk.items()
    }
    require(
        disk_values["required_free_bytes"]
        == disk_values["minimum_reserve_bytes"]
        + disk_values["declared_worst_case_new_bytes"],
        "Preflight required disk space is inconsistent",
    )
    require(
        disk_values["free_bytes"] >= disk_values["required_free_bytes"],
        "Preflight did not have the required free disk space",
    )
    environment = expect_mapping(
        preflight.get("environment_control"),
        "preflight environment control",
    )
    expect_exact_keys(
        environment,
        {
            "parent_performance_environment",
            "unsafe_parent_environment_policy",
            "child_environment_template",
            "inherits_parent_environment",
        },
        "preflight environment control",
    )
    parent_environment = expect_mapping(
        environment["parent_performance_environment"],
        "preflight parent performance environment",
    )
    require(
        all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in parent_environment.items()
        ),
        "Preflight parent performance environment is invalid",
    )
    validate_parent_environment_policy(
        environment["unsafe_parent_environment_policy"],
        context="preflight unsafe parent environment policy",
    )
    require(
        environment["child_environment_template"]
        == expected_environment(MEASURED_PARTICIPANTS),
        "Preflight child environment template differs",
    )
    require(
        environment["inherits_parent_environment"] is False,
        "Benchmark children inherited the parent environment",
    )
    validate_build_and_source_provenance(
        evidence_dir,
        preflight,
        binaries=binaries,
        runtime_artifacts=runtime_artifacts,
        runtime_artifact_paths=runtime_artifact_paths,
    )
    campaign_binding = validate_campaign_binding(
        preflight.get("campaign_binding"),
        dataset_sha256=str(dataset["sha256"]),
        schedule_sha256=schedule_sha256,
        binaries=binaries,
    )
    return (
        binaries,
        runtime_artifacts,
        power_policy,
        BATTERY_MINIMUM_PERCENT,
        recorded_global_swap_policy,
        idle_minimum_percent,
        recorded_idle_policy,
        campaign_binding,
    )


def expected_driver_field_names(engine: str) -> frozenset[str]:
    require(engine in ("infinity", "faiss"), f"Unknown engine {engine!r}")
    names = set(COMMON_DRIVER_FIELDS)
    names.update(f"{engine}_{suffix}" for suffix in ENGINE_RESULT_SUFFIXES)
    names.update(AUDIT_DRIVER_FIELDS)
    for k, ef in RECALL_POINTS:
        names.update(
            {
                f"{engine}_recall_at_{k}_ef_{ef}",
                f"{engine}_returned_ids_ef_{ef}_k_{k}",
                f"{engine}_returned_distances_sha256_ef_{ef}_k_{k}",
            }
        )
    if engine == "infinity":
        names.update(INFINITY_DRIVER_FIELDS)
    return frozenset(names)


def expected_infinity_tasks(participants: int) -> int:
    require(participants > 0, "Infinity participant count must be positive")
    average_bucket_size = (VECTORS - 1) // participants + 1
    bucket_size = max(1024, average_bucket_size)
    return (VECTORS - 1) // bucket_size + 1


def parse_driver_stdout(text: str, engine: str, context: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(line != "", f"{context} line {line_number} is blank")
        require(line == line.strip(), f"{context} line {line_number} has whitespace")
        require(line.count("=") == 1, f"{context} line {line_number} is not key=value")
        name, value = line.split("=", 1)
        require(
            FIELD_NAME.fullmatch(name) is not None,
            f"{context} line {line_number} has invalid key {name!r}",
        )
        require(
            value != "" and value == value.strip(), f"{context} field {name} is invalid"
        )
        require(name not in fields, f"{context} duplicates field {name}")
        fields[name] = value
    expected = expected_driver_field_names(engine)
    require(
        set(fields) == set(expected),
        f"{context} schema differs: missing={sorted(expected - set(fields))}, "
        f"extra={sorted(set(fields) - expected)}",
    )
    return fields


def parse_attested_driver_stdout(
    text: str,
    engine: str,
    context: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    transcript: list[str] = []
    result_lines: list[str] = []
    result_started = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(line != "", f"{context} line {line_number} is blank")
        if line.startswith("attestation_"):
            require(
                not result_started,
                f"{context} attestation appears after result output",
            )
            transcript.append(line)
        else:
            result_started = True
            result_lines.append(line)
    require(transcript, f"{context} lacks an attestation transcript")
    require(result_lines, f"{context} lacks driver result output")

    cursor = 0

    def take(expected: str) -> str:
        nonlocal cursor
        require(
            cursor < len(transcript),
            f"{context} attestation lacks field {expected}",
        )
        line = transcript[cursor]
        cursor += 1
        require(
            line.count("=") == 1,
            f"{context} attestation line {cursor} is not key=value",
        )
        name, value = line.split("=", 1)
        require(
            name == expected,
            f"{context} attestation field {cursor} is {name}, expected {expected}",
        )
        require(
            value != "" and value == value.strip(),
            f"{context} attestation field {expected} is invalid",
        )
        return value

    def unsigned(value: str, name: str, *, positive: bool = False) -> int:
        require(
            UNSIGNED_DECIMAL.fullmatch(value) is not None,
            f"{context} attestation field {name} is not canonical unsigned decimal",
        )
        parsed = int(value)
        if positive:
            require(parsed > 0, f"{context} attestation field {name} is not positive")
        return parsed

    def snapshot(phase: str) -> dict[str, Any]:
        prefix = f"attestation_{phase}_"
        cdhash = take(prefix + "dynamic_cdhash")
        require(
            re.fullmatch(r"[0-9a-f]{40}", cdhash) is not None,
            f"{context} {phase} dynamic CDHash is invalid",
        )
        identities: dict[str, dict[str, int]] = {}
        for identity in ("held_executable", "mapped_executable"):
            values: dict[str, int] = {}
            for output_name, suffix in (
                ("device", "dev"),
                ("inode", "ino"),
                ("size", "size"),
            ):
                name = prefix + identity + "_" + suffix
                values[output_name] = unsigned(take(name), name, positive=True)
            identities[identity] = values
        require(
            take(prefix + "mapped_vnode_matches_held_fd") == "1",
            f"{context} {phase} mapped vnode does not match the held file",
        )
        count_name = prefix + "image_count"
        image_count = unsigned(take(count_name), count_name, positive=True)
        require(
            image_count <= ATTESTATION_MAXIMUM_IMAGE_COUNT,
            f"{context} {phase} image count is too large",
        )
        images: list[dict[str, Any]] = []
        for index in range(image_count):
            image_prefix = prefix + f"image_{index}_"
            path_hex = take(image_prefix + "path_hex")
            require(
                0 < len(path_hex) <= 2 * ATTESTATION_MAXIMUM_PATH_BYTES
                and len(path_hex) % 2 == 0
                and re.fullmatch(r"[0-9a-f]+", path_hex) is not None,
                f"{context} {phase} image {index} path is invalid",
            )
            uuid = take(image_prefix + "uuid")
            require(
                re.fullmatch(r"[0-9a-f]{32}", uuid) is not None,
                f"{context} {phase} image {index} UUID is invalid",
            )
            shared = take(image_prefix + "shared_cache")
            require(
                shared in ("0", "1"),
                f"{context} {phase} image {index} shared-cache flag is invalid",
            )
            images.append(
                {
                    "path_hex": path_hex,
                    "uuid": uuid,
                    "shared_cache": shared == "1",
                }
            )
        require(
            len({(image["path_hex"], image["uuid"]) for image in images})
            == image_count,
            f"{context} {phase} image set contains duplicates",
        )
        require(
            identities["held_executable"] == identities["mapped_executable"],
            f"{context} {phase} mapped executable differs from the held file",
        )
        return {
            "dynamic_cdhash": cdhash,
            **identities,
            "mapped_vnode_matches_held_fd": True,
            "image_count": image_count,
            "images": images,
        }

    require(
        take("attestation_schema") == str(ATTESTATION_SCHEMA_VERSION),
        f"{context} attestation schema differs",
    )
    before = snapshot("before")
    require(
        take("attestation_barrier") == ATTESTATION_BARRIER_PHASES[0],
        f"{context} before-work attestation barrier differs",
    )
    for expected in ATTESTATION_BARRIER_PHASES[1:3]:
        require(
            take("attestation_barrier") == expected,
            f"{context} {expected} attestation barrier differs",
        )
    after = snapshot("after")
    initial_adds = unsigned(
        take("attestation_initial_image_adds"),
        "attestation_initial_image_adds",
    )
    later_adds = unsigned(
        take("attestation_later_image_adds"),
        "attestation_later_image_adds",
    )
    removes = unsigned(
        take("attestation_image_removes"),
        "attestation_image_removes",
    )
    require(
        take("attestation_image_set_unchanged") == "1",
        f"{context} attestation image set is not unchanged",
    )
    require(
        take("attestation_barrier") == ATTESTATION_BARRIER_PHASES[3],
        f"{context} after-work attestation barrier differs",
    )
    require(
        take("attestation_status") == "PASS",
        f"{context} attestation status is not PASS",
    )
    require(
        cursor == len(transcript),
        f"{context} attestation contains trailing fields",
    )
    require(before == after, f"{context} attestation snapshots differ")
    require(
        initial_adds == before["image_count"],
        f"{context} initial image-add count differs from the image count",
    )
    require(
        later_adds == 0 and removes == 0,
        f"{context} attestation observed image-set mutation",
    )
    fields = parse_driver_stdout(
        "\n".join(result_lines) + "\n",
        engine,
        context,
    )
    return (
        fields,
        {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "barriers": list(ATTESTATION_BARRIER_PHASES),
            "before": before,
            "after": after,
            "initial_image_adds": initial_adds,
            "later_image_adds": later_adds,
            "image_removes": removes,
            "image_set_unchanged": True,
            "status": "PASS",
        },
    )


def validate_attestation_barriers(
    value: Any,
    *,
    process_group: int,
    context: str,
) -> None:
    barriers = expect_list(value, f"{context} attestation barriers")
    require(
        len(barriers) == len(ATTESTATION_BARRIER_PHASES),
        f"{context} must contain exactly four attestation barrier observations",
    )
    previous_resume = 0
    for index, expected_phase in enumerate(ATTESTATION_BARRIER_PHASES):
        barrier = expect_mapping(
            barriers[index],
            f"{context} attestation barrier {index}",
        )
        expect_exact_keys(
            barrier,
            {
                "phase",
                "process_group",
                "stopped_pid",
                "observed_at_monotonic_ns",
                "resumed_at_monotonic_ns",
                "processes",
            },
            f"{context} attestation barrier {index}",
        )
        require(
            barrier["phase"] == expected_phase,
            f"{context} attestation barrier {index} phase differs",
        )
        require(
            expect_int(
                barrier["process_group"],
                f"{context} attestation process group",
                minimum=1,
            )
            == process_group,
            f"{context} attestation process group differs from the run PID",
        )
        stopped_pid = expect_int(
            barrier["stopped_pid"],
            f"{context} attestation stopped PID",
            minimum=1,
        )
        observed = expect_int(
            barrier["observed_at_monotonic_ns"],
            f"{context} attestation observed time",
            minimum=1,
        )
        resumed = expect_int(
            barrier["resumed_at_monotonic_ns"],
            f"{context} attestation resumed time",
            minimum=observed,
        )
        require(
            observed >= previous_resume,
            f"{context} attestation barrier observations overlap",
        )
        previous_resume = resumed
        processes = expect_list(
            barrier["processes"],
            f"{context} attestation process snapshot",
        )
        require(processes, f"{context} attestation process snapshot is empty")
        normalized: list[dict[str, Any]] = []
        for process_index, process_value in enumerate(processes):
            process = expect_mapping(
                process_value,
                f"{context} attestation process {process_index}",
            )
            expect_exact_keys(
                process,
                {
                    "pid",
                    "parent_pid",
                    "process_group",
                    "status_code",
                    "status",
                    "start_time_seconds",
                    "start_time_microseconds",
                    "name",
                    "executable_path",
                    "executable_device",
                    "executable_inode",
                    "executable_size",
                    "child_pids",
                },
                f"{context} attestation process {process_index}",
            )
            pid = expect_int(
                process["pid"],
                f"{context} attestation process PID",
                minimum=1,
            )
            group = expect_int(
                process["process_group"],
                f"{context} attestation process group",
                minimum=1,
            )
            parent_pid = expect_int(
                process["parent_pid"],
                f"{context} attestation process parent PID",
                minimum=1,
            )
            status_code = expect_int(
                process["status_code"],
                f"{context} attestation process status code",
                minimum=1,
            )
            status_name = expect_string(
                process["status"],
                f"{context} attestation process status",
            )
            require(
                PROCESS_STATUS_NAMES.get(status_code) == status_name,
                f"{context} attestation process status differs from its code",
            )
            start_seconds = expect_int(
                process["start_time_seconds"],
                f"{context} attestation process start seconds",
                minimum=1,
            )
            start_microseconds = expect_int(
                process["start_time_microseconds"],
                f"{context} attestation process start microseconds",
                minimum=0,
            )
            require(
                start_microseconds < 1_000_000,
                f"{context} attestation process start microseconds are invalid",
            )
            name = expect_string(
                process["name"],
                f"{context} attestation process name",
            )
            require(name != "", f"{context} attestation process name is empty")
            executable_path = expect_string(
                process["executable_path"],
                f"{context} attestation process executable path",
            )
            require(
                Path(executable_path).is_absolute(),
                f"{context} attestation process executable path is not absolute",
            )
            executable_device = expect_int(
                process["executable_device"],
                f"{context} attestation process executable device",
                minimum=1,
            )
            executable_inode = expect_int(
                process["executable_inode"],
                f"{context} attestation process executable inode",
                minimum=1,
            )
            executable_size = expect_int(
                process["executable_size"],
                f"{context} attestation process executable size",
                minimum=1,
            )
            child_values = expect_list(
                process["child_pids"],
                f"{context} attestation process child PIDs",
            )
            child_pids = [
                expect_int(
                    child,
                    f"{context} attestation process child PID",
                    minimum=1,
                )
                for child in child_values
            ]
            require(
                child_pids == sorted(set(child_pids)),
                f"{context} attestation process child PIDs are not canonical",
            )
            require(group == process_group, f"{context} process escaped its group")
            normalized.append(
                {
                    "pid": pid,
                    "parent_pid": parent_pid,
                    "process_group": group,
                    "status_code": status_code,
                    "status": status_name,
                    "start_time_seconds": start_seconds,
                    "start_time_microseconds": start_microseconds,
                    "name": name,
                    "executable_path": executable_path,
                    "executable_device": executable_device,
                    "executable_inode": executable_inode,
                    "executable_size": executable_size,
                    "child_pids": child_pids,
                }
            )
        require(
            [process["pid"] for process in normalized]
            == sorted(process["pid"] for process in normalized),
            f"{context} attestation process snapshot is not PID-sorted",
        )
        require(
            any(process["pid"] == process_group for process in normalized),
            f"{context} attestation process-group leader is missing",
        )
        stopped = [
            process["pid"]
            for process in normalized
            if process["status"] == "stopped"
        ]
        require(
            stopped == [stopped_pid],
            f"{context} attestation stopped-process evidence differs",
        )


def validate_benchmark_process_evidence(
    value: Any,
    *,
    barriers_value: Any,
    process_group: int,
    expected_benchmark_path: str,
    expected_benchmark_bytes: int,
    context: str,
) -> None:
    evidence = expect_mapping(value, f"{context} benchmark process evidence")
    expect_exact_keys(
        evidence,
        {
            "schema_version",
            "supervisor_pid",
            "process_group",
            "wrapper_executable",
            "benchmark_executable",
            "barrier_phases",
            "benchmark_pid",
            "terminal_observation",
            "pre_reap_quiescence",
            "reap",
        },
        f"{context} benchmark process evidence",
    )
    require(
        expect_int(
            evidence["schema_version"],
            f"{context} process evidence schema",
        )
        == PROCESS_EVIDENCE_SCHEMA_VERSION,
        f"{context} process evidence schema differs",
    )
    supervisor_pid = expect_int(
        evidence["supervisor_pid"],
        f"{context} process evidence supervisor PID",
        minimum=1,
    )
    require(
        expect_int(
            evidence["process_group"],
            f"{context} process evidence process group",
            minimum=1,
        )
        == process_group,
        f"{context} process evidence group differs from the run PID",
    )
    require(
        evidence["barrier_phases"] == list(ATTESTATION_BARRIER_PHASES),
        f"{context} process evidence barrier phases differ",
    )
    benchmark_pid = expect_int(
        evidence["benchmark_pid"],
        f"{context} process evidence benchmark PID",
        minimum=1,
    )

    def executable_identity(
        raw: Any,
        label: str,
    ) -> dict[str, Any]:
        identity = expect_mapping(raw, label)
        expect_exact_keys(
            identity,
            {"path", "device", "inode", "size"},
            label,
        )
        path = expect_string(identity["path"], f"{label} path")
        require(Path(path).is_absolute(), f"{label} path is not absolute")
        return {
            "path": path,
            "device": expect_int(
                identity["device"],
                f"{label} device",
                minimum=1,
            ),
            "inode": expect_int(
                identity["inode"],
                f"{label} inode",
                minimum=1,
            ),
            "size": expect_int(
                identity["size"],
                f"{label} size",
                minimum=1,
            ),
        }

    wrapper_executable = executable_identity(
        evidence["wrapper_executable"],
        f"{context} time-wrapper executable",
    )
    benchmark_executable = executable_identity(
        evidence["benchmark_executable"],
        f"{context} benchmark executable",
    )
    require(
        wrapper_executable["path"] == "/usr/bin/time",
        f"{context} process evidence time-wrapper path differs",
    )
    require(
        benchmark_executable["path"] == expected_benchmark_path
        and benchmark_executable["size"] == expected_benchmark_bytes,
        f"{context} process evidence benchmark executable differs from preflight",
    )

    barriers = expect_list(
        barriers_value,
        f"{context} process evidence barriers",
    )
    require(
        len(barriers) == len(ATTESTATION_BARRIER_PHASES),
        f"{context} process evidence requires four barriers",
    )
    stable_wrapper: dict[str, Any] | None = None
    stable_benchmark: dict[str, Any] | None = None
    last_resume = 0
    for index, expected_phase in enumerate(ATTESTATION_BARRIER_PHASES):
        barrier = expect_mapping(
            barriers[index],
            f"{context} process evidence barrier {index}",
        )
        require(
            barrier["phase"] == expected_phase,
            f"{context} process evidence barrier phase differs",
        )
        last_resume = expect_int(
            barrier["resumed_at_monotonic_ns"],
            f"{context} process evidence barrier resume",
            minimum=last_resume,
        )
        processes = expect_list(
            barrier["processes"],
            f"{context} process evidence barrier processes",
        )
        require(
            len(processes) == 2,
            f"{context} process evidence does not contain exactly two processes",
        )
        by_pid: dict[int, dict[str, Any]] = {}
        for process_index, raw_process in enumerate(processes):
            process = expect_mapping(
                raw_process,
                f"{context} process evidence process {process_index}",
            )
            pid = expect_int(
                process.get("pid"),
                f"{context} process evidence PID",
                minimum=1,
            )
            require(
                pid not in by_pid,
                f"{context} process evidence contains duplicate PIDs",
            )
            by_pid[pid] = process
        require(
            len(by_pid) == 2,
            f"{context} process evidence contains duplicate PIDs",
        )
        require(
            set(by_pid) == {process_group, benchmark_pid},
            f"{context} process evidence PID set differs",
        )
        wrapper = by_pid[process_group]
        benchmark = by_pid[benchmark_pid]
        require(
            wrapper["parent_pid"] == supervisor_pid,
            f"{context} time wrapper has an unexpected parent",
        )
        require(
            wrapper["process_group"] == process_group
            and benchmark["process_group"] == process_group,
            f"{context} process escaped the measured group",
        )
        require(
            wrapper["status"] not in {"creating", "stopped", "zombie"},
            f"{context} time wrapper has an invalid status",
        )
        require(
            benchmark["parent_pid"] == process_group
            and benchmark["status"] == "stopped",
            f"{context} benchmark topology or stopped state differs",
        )
        require(
            wrapper["child_pids"] == [benchmark_pid]
            and benchmark["child_pids"] == [],
            f"{context} benchmark descendant topology differs",
        )
        require(
            barrier["stopped_pid"] == benchmark_pid,
            f"{context} stopped PID is not the benchmark child",
        )
        wrapper_actual = {
            "path": wrapper["executable_path"],
            "device": wrapper["executable_device"],
            "inode": wrapper["executable_inode"],
            "size": wrapper["executable_size"],
        }
        benchmark_actual = {
            "path": benchmark["executable_path"],
            "device": benchmark["executable_device"],
            "inode": benchmark["executable_inode"],
            "size": benchmark["executable_size"],
        }
        require(
            wrapper_actual == wrapper_executable
            and benchmark_actual == benchmark_executable,
            f"{context} live executable identity differs from process evidence",
        )
        wrapper_stable = {
            key: item
            for key, item in wrapper.items()
            if key not in {"status_code", "status"}
        }
        benchmark_stable = {
            key: item
            for key, item in benchmark.items()
            if key not in {"status_code", "status"}
        }
        if stable_wrapper is None:
            stable_wrapper = wrapper_stable
            stable_benchmark = benchmark_stable
        else:
            require(
                wrapper_stable == stable_wrapper,
                f"{context} time-wrapper identity changed across barriers",
            )
            require(
                benchmark_stable == stable_benchmark,
                f"{context} benchmark identity changed across barriers",
            )

    terminal = expect_mapping(
        evidence["terminal_observation"],
        f"{context} terminal process evidence",
    )
    expect_exact_keys(
        terminal,
        {"observed_at_monotonic_ns", "pid", "code", "status"},
        f"{context} terminal process evidence",
    )
    terminal_observed_at = expect_int(
        terminal["observed_at_monotonic_ns"],
        f"{context} terminal observation time",
        minimum=last_resume,
    )
    terminal_pid = expect_int(
        terminal["pid"],
        f"{context} terminal observation PID",
        minimum=1,
    )
    terminal_code = expect_int(
        terminal["code"],
        f"{context} terminal waitid code",
        minimum=1,
    )
    terminal_status = expect_int(
        terminal["status"],
        f"{context} terminal waitid status",
        minimum=0,
    )
    require(
        terminal_pid == process_group
        and terminal_code in TERMINAL_CHILD_CODES,
        f"{context} terminal waitid observation differs",
    )

    pre_reap = expect_mapping(
        evidence["pre_reap_quiescence"],
        f"{context} pre-reap process evidence",
    )
    expect_exact_keys(
        pre_reap,
        {
            "observed_at_monotonic_ns",
            "process_group",
            "leader_pid",
            "member_pids",
            "no_members_except_leader",
        },
        f"{context} pre-reap process evidence",
    )
    pre_reap_observed_at = expect_int(
        pre_reap["observed_at_monotonic_ns"],
        f"{context} pre-reap observation time",
        minimum=terminal_observed_at,
    )
    pre_reap_group = expect_int(
        pre_reap["process_group"],
        f"{context} pre-reap process group",
        minimum=1,
    )
    pre_reap_leader = expect_int(
        pre_reap["leader_pid"],
        f"{context} pre-reap leader PID",
        minimum=1,
    )
    member_pids = expect_list(
        pre_reap["member_pids"],
        f"{context} pre-reap member PIDs",
    )
    for member_index, member_pid in enumerate(member_pids):
        expect_int(
            member_pid,
            f"{context} pre-reap member PID {member_index}",
            minimum=1,
        )
    require(
        pre_reap_group == process_group
        and pre_reap_leader == process_group
        and member_pids == [process_group]
        and pre_reap["no_members_except_leader"] is True,
        f"{context} benchmark process group was not quiescent before reap",
    )
    reap = expect_mapping(
        evidence["reap"],
        f"{context} reap process evidence",
    )
    expect_exact_keys(
        reap,
        {
            "completed_at_monotonic_ns",
            "pid",
            "status_validated",
            "returncode",
        },
        f"{context} reap process evidence",
    )
    expect_int(
        reap["completed_at_monotonic_ns"],
        f"{context} reap completion time",
        minimum=pre_reap_observed_at,
    )
    reap_pid = expect_int(
        reap["pid"],
        f"{context} reaped PID",
        minimum=1,
    )
    returncode = expect_int(
        reap["returncode"],
        f"{context} reaped process return code",
    )
    expected_returncode = (
        terminal_status if terminal_code == CLD_EXITED else -terminal_status
    )
    require(
        reap_pid == process_group
        and reap["status_validated"] is True
        and returncode == expected_returncode,
        f"{context} benchmark reap evidence differs from waitid",
    )
    require(
        terminal_code == CLD_EXITED
        and terminal_status == 0
        and returncode == 0,
        f"{context} benchmark did not exit successfully",
    )


def validate_index_timing_binding(
    value: Any,
    *,
    barriers_value: Any,
    fields: dict[str, str],
    engine: str,
    context: str,
) -> None:
    binding = expect_mapping(value, f"{context} index timing binding")
    expect_exact_keys(
        binding,
        {
            "schema_version",
            "clock",
            "outer_start_phase",
            "outer_end_phase",
            "outer_start_monotonic_ns",
            "outer_end_monotonic_ns",
            "outer_index_window_ns",
            "reported_cold_build_ns",
            "boundary_overhead_ns",
            "maximum_boundary_overhead_ns",
        },
        f"{context} index timing binding",
    )
    require(
        expect_int(
            binding["schema_version"],
            f"{context} index timing binding schema",
        )
        == INDEX_TIMING_BINDING_SCHEMA_VERSION,
        f"{context} index timing binding schema differs",
    )
    require(
        expect_string(
            binding["clock"],
            f"{context} index timing binding clock",
        )
        == "python-time-monotonic-ns",
        f"{context} index timing binding clock differs",
    )
    require(
        binding["outer_start_phase"] == "before-index.resumed"
        and binding["outer_end_phase"] == "after-index.observed",
        f"{context} index timing binding phase endpoints differ",
    )

    barriers = expect_list(
        barriers_value,
        f"{context} index timing attestation barriers",
    )
    require(
        len(barriers) == len(ATTESTATION_BARRIER_PHASES),
        f"{context} index timing binding requires four barriers",
    )
    before_index = expect_mapping(
        barriers[1],
        f"{context} before-index timing barrier",
    )
    after_index = expect_mapping(
        barriers[2],
        f"{context} after-index timing barrier",
    )
    require(
        before_index.get("phase") == "before-index"
        and after_index.get("phase") == "after-index",
        f"{context} index timing barriers differ",
    )
    outer_start = expect_int(
        before_index.get("resumed_at_monotonic_ns"),
        f"{context} before-index resume time",
        minimum=1,
    )
    outer_end = expect_int(
        after_index.get("observed_at_monotonic_ns"),
        f"{context} after-index observation time",
        minimum=outer_start,
    )
    outer_ns = outer_end - outer_start
    cold_build_ns = parse_unsigned_field(
        fields,
        f"{engine}_cold_build_ns",
        context,
    )
    require(
        cold_build_ns <= outer_ns,
        f"{context} cold-build duration exceeds the parent-observed index window",
    )
    overhead_ns = outer_ns - cold_build_ns
    require(
        overhead_ns <= INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS,
        f"{context} index timing boundary overhead exceeds the protocol maximum",
    )
    expected = {
        "schema_version": INDEX_TIMING_BINDING_SCHEMA_VERSION,
        "clock": "python-time-monotonic-ns",
        "outer_start_phase": "before-index.resumed",
        "outer_end_phase": "after-index.observed",
        "outer_start_monotonic_ns": outer_start,
        "outer_end_monotonic_ns": outer_end,
        "outer_index_window_ns": outer_ns,
        "reported_cold_build_ns": cold_build_ns,
        "boundary_overhead_ns": overhead_ns,
        "maximum_boundary_overhead_ns": (
            INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
        ),
    }
    require(
        binding == expected,
        f"{context} index timing binding differs from independent reconstruction",
    )


def parse_unsigned_field(fields: dict[str, str], name: str, context: str) -> int:
    value = fields.get(name)
    require(isinstance(value, str), f"{context} field {name} is not a string")
    require(
        UNSIGNED_DECIMAL.fullmatch(value) is not None,
        f"{context} field {name} is not canonical unsigned decimal",
    )
    return int(value)


def parse_uint64_csv(
    value: Any,
    *,
    expected_count: int,
    context: str,
) -> list[int]:
    require(isinstance(value, str), f"{context} is not a string")
    components = value.split(",")
    require(
        len(components) == expected_count,
        f"{context} has the wrong value count",
    )
    result: list[int] = []
    for index, component in enumerate(components):
        require(
            UNSIGNED_DECIMAL.fullmatch(component) is not None,
            f"{context} value {index} is not canonical unsigned decimal",
        )
        parsed = int(component)
        require(
            parsed <= (1 << 64) - 1,
            f"{context} value {index} exceeds uint64",
        )
        result.append(parsed)
    return result


def parse_finite_field(fields: dict[str, str], name: str, context: str) -> float:
    value = fields.get(name)
    require(isinstance(value, str), f"{context} field {name} is not a string")
    require(
        FINITE_DECIMAL.fullmatch(value) is not None,
        f"{context} field {name} is not a decimal number",
    )
    parsed = float(value)
    require(math.isfinite(parsed), f"{context} field {name} is not finite")
    return parsed


def validate_driver_fields(
    fields_value: Any,
    *,
    engine: str,
    participants: int,
    dataset: dict[str, Any],
    context: str,
) -> dict[str, str]:
    fields_mapping = expect_mapping(fields_value, f"{context} fields")
    require(
        all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in fields_mapping.items()
        ),
        f"{context} fields must map strings to strings",
    )
    fields = dict(fields_mapping)
    expected_names = expected_driver_field_names(engine)
    require(
        set(fields) == set(expected_names),
        f"{context} field schema differs: "
        f"missing={sorted(expected_names - set(fields))}, "
        f"extra={sorted(set(fields) - expected_names)}",
    )
    require(fields["status"] == "PASS", f"{context} driver status is not PASS")
    require(fields["scope"] == D0_SCOPE, f"{context} driver scope differs")
    require(fields["order"] == f"{engine}-only", f"{context} driver order differs")

    expected_integers = {
        "data_fnv1a64": dataset["fnv1a64"],
        "data_bytes": dataset["bytes"],
        "vectors": VECTORS,
        "dimensions": DIMENSIONS,
        "M": M,
        "ef_construction": EF_CONSTRUCTION,
        "ef_search": EF_SEARCH,
        "participants": participants,
        "build_grain": BUILD_GRAIN,
        "query_benchmark_schema": QUERY_BENCHMARK_SCHEMA_VERSION,
        "query_unique_queries": HELDOUT_QUERY_COUNT,
        "query_k": QUERY_K,
        "query_ef_search": QUERY_EF_SEARCH,
        "query_warmup_passes": QUERY_WARMUP_PASSES,
        "query_measured_passes": QUERY_MEASURED_PASSES,
        "query_latency_concurrency": QUERY_LATENCY_CONCURRENCY,
        "query_throughput_concurrency": QUERY_THROUGHPUT_CONCURRENCY,
        f"{engine}_valid": 1,
        f"{engine}_threads": participants,
        f"{engine}_index_size": VECTORS,
        f"{engine}_query_latency_sample_count": QUERY_LATENCY_SAMPLE_COUNT,
        f"{engine}_query_latency_validated_operations": (
            QUERY_LATENCY_SAMPLE_COUNT
        ),
        f"{engine}_query_throughput_operations": QUERY_THROUGHPUT_OPERATIONS,
        f"{engine}_query_throughput_validated_operations": (
            QUERY_THROUGHPUT_OPERATIONS
        ),
        f"{engine}_query_per_query_checksum_count": HELDOUT_QUERY_COUNT,
        "audit_sidecar_schema": AUDIT_SIDECAR_SCHEMA_VERSION,
        "heldout_query_count": HELDOUT_QUERY_COUNT,
        "heldout_query_seed": HELDOUT_QUERY_SEED,
        f"{engine}_graph_valid": 1,
        f"{engine}_graph_vertex_count": VECTORS,
        f"{engine}_graph_reachable_count": VECTORS,
        f"{engine}_graph_level0_capacity": 2 * M,
        f"{engine}_graph_upper_capacity": M,
    }
    if engine == "infinity":
        expected_integers.update(
            {
                "infinity_submitted_tasks": expected_infinity_tasks(participants),
                "infinity_build_start": 0,
                "infinity_build_end": VECTORS,
            }
        )
    for name, expected in expected_integers.items():
        actual = parse_unsigned_field(fields, name, context)
        require(
            actual == expected,
            f"{context} field {name} is {actual}, expected {expected}",
        )
    require(
        fields["data_sha256"] == dataset["sha256"]
        and SHA256_TEXT.fullmatch(fields["data_sha256"]) is not None,
        f"{context} data_sha256 differs from the frozen dataset",
    )
    for name in ("campaign_nonce", "binary_sha256", "dataset_sha256"):
        require(
            SHA256_TEXT.fullmatch(fields[name]) is not None,
            f"{context} field {name} is not canonical lowercase 32-byte hex",
        )
    schedule_sequence = parse_unsigned_field(
        fields,
        "schedule_sequence",
        context,
    )
    require(
        schedule_sequence <= (1 << 64) - 1,
        f"{context} schedule_sequence does not fit in uint64",
    )
    role_id = parse_unsigned_field(fields, "role_id", context)
    require(
        AUDIT_ROLE_ENGINE_IDENTIFIERS.get(role_id)
        == AUDIT_ENGINE_IDENTIFIERS[engine],
        f"{context} role_id is incompatible with engine {engine}",
    )
    require(
        fields["dataset_sha256"] == dataset["sha256"],
        f"{context} dataset_sha256 differs from the frozen dataset",
    )
    expect_sha256(
        fields["query_timed_corpus_sha256"],
        f"{context} timed query corpus SHA-256",
    )
    require(
        fields["query_percentile_method"] == QUERY_PERCENTILE_METHOD
        and fields["query_transaction_definition"]
        == QUERY_TRANSACTION_DEFINITION,
        f"{context} query benchmark text protocol differs",
    )
    require(
        parse_finite_field(fields, "query_recall_floor", context)
        == QUERY_RECALL_FLOOR
        and parse_finite_field(
            fields,
            "query_maximum_absolute_recall_gap",
            context,
        )
        == QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP,
        f"{context} query benchmark quality protocol differs",
    )

    cold_build_ns = parse_unsigned_field(
        fields,
        f"{engine}_cold_build_ns",
        context,
    )
    insert_call_ns = parse_unsigned_field(
        fields,
        f"{engine}_insert_call_ns",
        context,
    )
    require(cold_build_ns > 0, f"{context} cold-build duration is not positive")
    require(insert_call_ns > 0, f"{context} insert-call duration is not positive")
    require(
        insert_call_ns <= cold_build_ns,
        f"{context} insert-call duration exceeds cold-build duration",
    )
    require(
        parse_unsigned_field(
            fields,
            f"{engine}_query_throughput_wall_ns",
            context,
        )
        > 0,
        f"{context} query throughput duration is not positive",
    )
    for suffix in (
        "query_latency_validated_result_checksum",
        "query_throughput_validated_result_checksum",
        "query_result_checksum",
    ):
        require(
            parse_unsigned_field(fields, f"{engine}_{suffix}", context)
            <= (1 << 64) - 1,
            f"{context} field {engine}_{suffix} exceeds uint64",
        )
    parse_uint64_csv(
        fields[f"{engine}_query_per_query_checksums"],
        expected_count=HELDOUT_QUERY_COUNT,
        context=f"{context} query per-query checksums",
    )
    for name in EXECUTION_WITNESS_FIELDS:
        require(
            parse_unsigned_field(fields, f"{engine}_{name}", context) in (0, 1),
            f"{context} execution witness {name} is not binary",
        )
    recall = parse_finite_field(fields, f"{engine}_self_recall_at_1", context)
    parse_finite_field(fields, f"{engine}_distance_checksum", context)
    require(0.95 <= recall <= 1.0, f"{context} self-recall is below the D0 gate")
    sidecar_bytes = parse_unsigned_field(fields, "audit_sidecar_bytes", context)
    require(
        0 < sidecar_bytes <= MAXIMUM_AUDIT_SIDECAR_BYTES,
        f"{context} audit sidecar byte count is invalid",
    )
    for name in (
        "audit_sidecar_sha256",
        "heldout_queries_sha256",
        "heldout_truth_sha256",
        f"{engine}_graph_levels_sha256",
        f"{engine}_graph_sha256",
    ):
        expect_sha256(fields[name], f"{context} field {name}")
    parse_unsigned_field(fields, f"{engine}_graph_directed_edges", context)
    parse_unsigned_field(fields, f"{engine}_graph_level0_directed_edges", context)
    parse_unsigned_field(fields, f"{engine}_graph_max_level", context)
    parse_unsigned_field(fields, f"{engine}_graph_entry_point", context)
    for k, ef in RECALL_POINTS:
        point_recall = parse_finite_field(
            fields,
            f"{engine}_recall_at_{k}_ef_{ef}",
            context,
        )
        require(
            0.0 <= point_recall <= 1.0,
            f"{context} Recall@{k}/ef={ef} is outside [0, 1]",
        )
        expect_sha256(
            fields[f"{engine}_returned_distances_sha256_ef_{ef}_k_{k}"],
            f"{context} returned-distance hash at k={k}, ef={ef}",
        )
    return fields


def validate_audit_stdout_fields(
    fields: dict[str, str],
    audit: dict[str, Any],
    details: dict[str, Any],
    *,
    engine: str,
    context: str,
) -> None:
    query = expect_mapping(audit.get("query"), f"{context} query benchmark")
    query_integers = {
        "query_benchmark_schema": QUERY_BENCHMARK_SCHEMA_VERSION,
        "query_unique_queries": HELDOUT_QUERY_COUNT,
        "query_k": QUERY_K,
        "query_ef_search": QUERY_EF_SEARCH,
        "query_warmup_passes": QUERY_WARMUP_PASSES,
        "query_measured_passes": QUERY_MEASURED_PASSES,
        "query_latency_concurrency": QUERY_LATENCY_CONCURRENCY,
        "query_throughput_concurrency": QUERY_THROUGHPUT_CONCURRENCY,
        f"{engine}_query_latency_sample_count": QUERY_LATENCY_SAMPLE_COUNT,
        f"{engine}_query_latency_validated_operations": query[
            "latency_validated_operations"
        ],
        f"{engine}_query_latency_validated_result_checksum": query[
            "latency_validated_result_checksum"
        ],
        f"{engine}_query_throughput_operations": QUERY_THROUGHPUT_OPERATIONS,
        f"{engine}_query_throughput_validated_operations": query[
            "throughput_validated_operations"
        ],
        f"{engine}_query_throughput_wall_ns": query["throughput_wall_ns"],
        f"{engine}_query_throughput_validated_result_checksum": query[
            "throughput_validated_result_checksum"
        ],
        f"{engine}_query_result_checksum": query["result_checksum"],
        f"{engine}_query_per_query_checksum_count": HELDOUT_QUERY_COUNT,
    }
    for name, expected in query_integers.items():
        require(
            parse_unsigned_field(fields, name, context) == expected,
            f"{context} stdout field {name} differs from the query sidecar",
        )
    require(
        fields["query_timed_corpus_sha256"]
        == query["timed_query_corpus_sha256"],
        f"{context} stdout timed query corpus differs from the query sidecar",
    )
    require(
        parse_uint64_csv(
            fields[f"{engine}_query_per_query_checksums"],
            expected_count=HELDOUT_QUERY_COUNT,
            context=f"{context} stdout query per-query checksums",
        )
        == query["per_query_checksums"],
        f"{context} stdout per-query checksums differ from the query sidecar",
    )
    execution_witness = expect_mapping(
        audit.get("execution_witness"),
        f"{context} execution witness",
    )
    expect_exact_keys(
        execution_witness,
        set(EXECUTION_WITNESS_FIELDS),
        f"{context} execution witness",
    )
    for name in EXECUTION_WITNESS_FIELDS:
        expected = expect_int(
            execution_witness[name],
            f"{context} execution witness {name}",
            minimum=0,
        )
        require(
            expected <= 1
            and parse_unsigned_field(fields, f"{engine}_{name}", context)
            == expected,
            f"{context} stdout execution witness {name} differs from the sidecar",
        )

    graph = expect_mapping(audit.get("graph"), f"{context} graph audit")
    graph_integer_fields = {
        "valid": int(graph["valid"]),
        "vertex_count": graph["vertex_count"],
        "reachable_count": graph["reachable_count"],
        "directed_edges": graph["directed_edges"],
        "level0_directed_edges": graph["level0_directed_edges"],
        "max_level": graph["max_level"],
        "entry_point": graph["entry_point"],
        "level0_capacity": graph["level0_capacity"],
        "upper_capacity": graph["upper_capacity"],
    }
    for suffix, expected in graph_integer_fields.items():
        actual = parse_unsigned_field(
            fields,
            f"{engine}_graph_{suffix}",
            context,
        )
        require(
            actual == expected,
            (
                f"{context} stdout graph field {suffix} is {actual}, "
                f"sidecar reconstructs {expected}"
            ),
        )
    for suffix, key in (
        ("levels_sha256", "levels_sha256"),
        ("sha256", "graph_sha256"),
        ("level_histogram", "level_histogram"),
        ("degree_histograms", "degree_histograms"),
    ):
        name = f"{engine}_graph_{suffix}"
        require(
            fields[name] == graph[key],
            f"{context} stdout field {name} differs from the sidecar graph",
        )

    recall = expect_mapping(audit.get("recall"), f"{context} recall audit")
    recall_common = {
        "heldout_query_count": str(recall["query_count"]),
        "heldout_query_seed": str(recall["query_seed"]),
        "heldout_queries_sha256": recall["queries_sha256"],
        "heldout_truth_sha256": recall["truth_sha256"],
        "heldout_truth_tie_counts_at_10": ",".join(
            str(value) for value in details["truth_tie_counts_at_10"]
        ),
        "heldout_truth_tie_counts_at_100": ",".join(
            str(value) for value in details["truth_tie_counts_at_100"]
        ),
    }
    for name, expected in recall_common.items():
        require(
            fields[name] == expected,
            f"{context} stdout field {name} differs from independent truth",
        )

    points = expect_list(recall.get("points"), f"{context} recall points")
    require(
        len(points) == len(RECALL_POINTS),
        f"{context} recall point count differs",
    )
    point_details = expect_list(details.get("points"), f"{context} recall details")
    require(
        len(point_details) == len(RECALL_POINTS),
        f"{context} recall detail count differs",
    )
    for index, ((expected_k, expected_ef), point_value, detail_value) in enumerate(
        zip(RECALL_POINTS, points, point_details)
    ):
        point = expect_mapping(point_value, f"{context} recall point {index}")
        detail = expect_mapping(detail_value, f"{context} recall detail {index}")
        require(
            (point.get("k"), point.get("ef")) == (expected_k, expected_ef),
            f"{context} recall point {index} metadata differs",
        )
        eligible_hits = expect_int(
            point.get("eligible_hits"),
            f"{context} recall point {index} eligible hits",
            minimum=0,
        )
        possible_hits = expect_int(
            point.get("possible_hits"),
            f"{context} recall point {index} possible hits",
            minimum=1,
        )
        require(
            possible_hits == HELDOUT_QUERY_COUNT * expected_k
            and eligible_hits <= possible_hits,
            f"{context} recall point {index} hit counts differ",
        )
        expect_close(
            point["recall"],
            eligible_hits / possible_hits,
            f"{context} recall point {index} ratio",
        )
        recall_name = f"{engine}_recall_at_{expected_k}_ef_{expected_ef}"
        expect_close(
            parse_finite_field(fields, recall_name, context),
            point["recall"],
            f"{context} stdout field {recall_name}",
        )
        ids_name = (
            f"{engine}_returned_ids_ef_{expected_ef}_k_{expected_k}"
        )
        require(
            fields[ids_name] == detail["returned_ids"],
            f"{context} stdout field {ids_name} differs from the sidecar",
        )
        distances_name = (
            f"{engine}_returned_distances_sha256_ef_{expected_ef}_k_{expected_k}"
        )
        require(
            fields[distances_name] == detail["returned_distances_sha256"],
            f"{context} stdout field {distances_name} differs from the sidecar",
        )


def expected_environment(participants: int) -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_NUM_THREADS": str(participants),
        "OMP_PROC_BIND": "FALSE",
        "OMP_THREAD_LIMIT": str(participants),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def expected_parent_environment_policy() -> dict[str, Any]:
    return {
        "schema_version": PARENT_ENVIRONMENT_POLICY_SCHEMA_VERSION,
        "exact_names": sorted(UNSAFE_PARENT_ENVIRONMENT_NAMES),
        "prefixes": list(UNSAFE_PARENT_ENVIRONMENT_PREFIXES),
        "observed_unsafe_names": [],
    }


def validate_parent_environment_policy(
    value: Any,
    *,
    context: str,
) -> None:
    policy = expect_mapping(value, context)
    expect_exact_keys(
        policy,
        {
            "schema_version",
            "exact_names",
            "prefixes",
            "observed_unsafe_names",
        },
        context,
    )
    require(
        policy == expected_parent_environment_policy(),
        f"{context} differs from the frozen rejection policy",
    )


def expected_tool_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "TMPDIR": "/private/tmp",
    }


def parse_idle_percentages(text: str, context: str) -> list[float]:
    values = [float(match.group(1)) for match in IDLE_PERCENTAGE.finditer(text)]
    require(
        all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in values),
        f"{context} contains an invalid CPU-idle percentage",
    )
    require(
        len(values) == IDLE_SAMPLE_COUNT,
        f"{context} contains {len(values)} CPU-idle samples, "
        f"expected {IDLE_SAMPLE_COUNT}",
    )
    return values


def parse_idle_capture(text: str, context: str) -> tuple[str, str]:
    lines = text.splitlines()
    require(lines and lines[0] == "[stdout]", f"{context} lacks stdout section")
    require(
        lines.count("[stdout]") == 1 and lines.count("[stderr]") == 1,
        f"{context} must contain exactly one stdout and stderr section",
    )
    stderr_index = lines.index("[stderr]")
    require(stderr_index > 0, f"{context} stderr section is out of order")
    return (
        "\n".join(lines[1:stderr_index]),
        "\n".join(lines[stderr_index + 1 :]),
    )


def validate_idle(
    evidence_dir: Path,
    idle_value: Any,
    *,
    context: str,
    minimum_idle_percent: float,
) -> dict[str, Any]:
    idle = expect_mapping(idle_value, f"{context} idle")
    require(idle.get("status") == "pass", f"{context} idle status is not pass")
    expect_int(
        idle.get("accepted_at_monotonic_ns"),
        f"{context} accepted monotonic timestamp",
        minimum=1,
    )
    expect_int(
        idle.get("accepted_at_unix_ns"),
        f"{context} accepted Unix timestamp",
        minimum=1,
    )
    require(
        expect_number(
            idle.get("required_minimum_idle_percent"),
            f"{context} required idle minimum",
        )
        == minimum_idle_percent,
        f"{context} required idle minimum differs from preflight",
    )
    require(
        idle.get("required_window_seconds") == IDLE_WINDOW_SECONDS,
        f"{context} required idle window is not 15 seconds",
    )

    attempts = expect_list(idle.get("attempts"), f"{context} idle attempts")
    require(attempts, f"{context} has no idle attempts")
    for index, attempt_value in enumerate(attempts, start=1):
        attempt_context = f"{context} idle attempt {index}"
        attempt = expect_mapping(attempt_value, attempt_context)
        require(attempt.get("attempt") == index, f"{attempt_context} number differs")
        raw_name, raw_path = relative_evidence_path(
            evidence_dir,
            attempt.get("raw_path"),
            f"{attempt_context} raw path",
        )
        require(
            raw_path.is_file(), f"{attempt_context} raw file is missing: {raw_name}"
        )
        raw_stdout, raw_stderr = parse_idle_capture(
            _read_text(raw_path, f"{attempt_context} raw file"),
            f"{attempt_context} raw file",
        )
        require(
            IDLE_PERCENTAGE.search(raw_stderr) is None,
            f"{attempt_context} stderr contains CPU-idle samples",
        )
        raw_samples = parse_idle_percentages(raw_stdout, attempt_context)
        recorded_samples = expect_list(
            attempt.get("idle_percentages"),
            f"{attempt_context} percentages",
        )
        require(
            len(recorded_samples) == IDLE_SAMPLE_COUNT,
            f"{attempt_context} does not record 16 samples",
        )
        for sample_index, (recorded, recomputed) in enumerate(
            zip(recorded_samples, raw_samples),
            start=1,
        ):
            expect_close(
                recorded,
                recomputed,
                f"{attempt_context} sample {sample_index}",
            )
        minimum = min(raw_samples)
        expect_close(
            attempt.get("minimum_idle_percent"),
            minimum,
            f"{attempt_context} minimum",
        )
        require(
            attempt.get("sample_count") == IDLE_SAMPLE_COUNT,
            f"{attempt_context} sample count differs",
        )
        require(
            attempt.get("sample_interval_seconds") == IDLE_SAMPLE_INTERVAL_SECONDS,
            f"{attempt_context} sample interval differs",
        )
        covered_seconds = (IDLE_SAMPLE_COUNT - 1) * IDLE_SAMPLE_INTERVAL_SECONDS
        require(
            attempt.get("covered_seconds") == covered_seconds,
            f"{attempt_context} coverage differs",
        )
        require(
            expect_number(
                attempt.get("command_elapsed_seconds"),
                f"{attempt_context} command elapsed time",
            )
            >= 0,
            f"{attempt_context} command elapsed time is negative",
        )
        passed = (
            minimum >= minimum_idle_percent and covered_seconds >= IDLE_WINDOW_SECONDS
        )
        expected_status = "pass" if passed else "fail"
        require(
            attempt.get("status") == expected_status,
            f"{attempt_context} status disagrees with raw samples",
        )
        if index < len(attempts):
            require(not passed, f"{attempt_context} passed but sampling continued")

    final = expect_mapping(attempts[-1], f"{context} final idle attempt")
    require(final.get("status") == "pass", f"{context} final idle attempt failed")
    require(
        idle.get("accepted_attempt") == final.get("attempt"),
        f"{context} accepted idle attempt is not the final attempt",
    )
    return idle


def expected_command(
    binary_path: str,
    dataset_argument: str,
    participants: int,
    audit_sidecar_name: str,
    *,
    campaign_nonce: str,
    schedule_sequence: int,
    role_id: int,
    binary_sha256: str,
    dataset_sha256: str,
) -> list[str]:
    return [
        "/usr/bin/time",
        "-lp",
        binary_path,
        dataset_argument,
        str(VECTORS),
        str(DIMENSIONS),
        str(M),
        str(EF_CONSTRUCTION),
        str(EF_SEARCH),
        str(CHUNK_SIZE),
        str(QUERY_COUNT),
        str(participants),
        str(BUILD_GRAIN),
        audit_sidecar_name,
        campaign_nonce,
        str(schedule_sequence),
        str(role_id),
        binary_sha256,
        dataset_sha256,
    ]


def validate_dataset_fingerprint(
    value: Any,
    *,
    dataset: dict[str, Any],
    context: str,
) -> None:
    fingerprint = expect_mapping(value, context)
    expect_exact_keys(
        fingerprint,
        {
            "verified_at",
            "path",
            "mode",
            "read_only",
            "sha256",
            "fnv1a64",
            "bytes",
        },
        context,
    )
    require(
        fingerprint.get("path") == dataset["path"],
        f"{context} path differs from dataset.json",
    )
    require(
        expect_string(fingerprint.get("verified_at"), f"{context} timestamp").endswith(
            "Z"
        ),
        f"{context} timestamp is not UTC",
    )
    for key in ("sha256", "fnv1a64", "bytes"):
        require(
            fingerprint.get(key) == dataset[key],
            f"{context} {key} differs from dataset.json",
        )
    require(fingerprint.get("read_only") is True, f"{context} is not marked read-only")
    mode = expect_string(fingerprint.get("mode"), f"{context} mode")
    try:
        parsed_mode = int(mode, 8)
    except ValueError as error:
        raise VerificationError(f"{context} mode is not octal") from error
    require(parsed_mode & 0o222 == 0, f"{context} records a writable mode")


def validate_runtime_artifacts(
    value: Any,
    *,
    expected_hashes: dict[str, str],
    context: str,
) -> None:
    record = expect_mapping(value, context)
    artifacts = expect_list(record.get("artifacts"), f"{context} artifacts")
    require(
        len(artifacts) == len(expected_hashes),
        f"{context} does not cover the complete runtime artifact set",
    )
    expected_items = sorted(expected_hashes.items())
    for index, (artifact_value, (expected_path, expected_sha256)) in enumerate(
        zip(artifacts, expected_items),
        start=1,
    ):
        artifact = expect_mapping(artifact_value, f"{context} artifact {index}")
        expect_exact_keys(
            artifact,
            {"path", "sha256", "bytes"},
            f"{context} artifact {index}",
        )
        require(
            artifact["path"] == expected_path,
            f"{context} artifact {index} path differs from preflight",
        )
        require(
            artifact["sha256"] == expected_sha256,
            f"{context} artifact {index} hash differs from preflight",
        )
        expect_int(
            artifact["bytes"],
            f"{context} artifact {index} bytes",
            minimum=1,
        )


def parse_host_sections(text: str, context: str) -> dict[str, str]:
    expected = ("battery", "custom", "thermal", "swap")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = HOST_SECTION.fullmatch(line)
        if match is not None:
            name = match.group(1)
            require(
                len(sections) < len(expected) and name == expected[len(sections)],
                f"{context} section {name!r} is duplicated, unknown, or out of order",
            )
            sections[name] = []
            current = name
            continue
        require(current is not None, f"{context} has content before its first section")
        sections[current].append(line)
    require(
        tuple(sections) == expected,
        f"{context} sections differ: expected {list(expected)}, got {list(sections)}",
    )
    result = {name: "\n".join(lines).strip() for name, lines in sections.items()}
    require(
        all(result.values()),
        f"{context} contains an empty host-telemetry section",
    )
    return result


def parse_power_source(text: str, context: str) -> str:
    ac = "Now drawing from 'AC Power'" in text
    battery = "Now drawing from 'Battery Power'" in text
    require(ac is not battery, f"{context} does not identify one active power source")
    return "ac" if ac else "battery"


def parse_battery_percent(text: str, context: str) -> int:
    matches = re.findall(r"\b([0-9]{1,3})%;", text)
    require(len(matches) == 1, f"{context} must contain one battery percentage")
    value = int(matches[0])
    require(0 <= value <= 100, f"{context} battery percentage is invalid")
    return value


def active_power_profile(text: str, source: str, context: str) -> str:
    heading = {"ac": "AC Power:", "battery": "Battery Power:"}[source]
    profiles: dict[str, list[str]] = {}
    active_heading: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in ("AC Power:", "Battery Power:"):
            require(
                stripped not in profiles,
                f"{context} duplicates the {stripped} profile",
            )
            profiles[stripped] = []
            active_heading = stripped
        elif active_heading is not None:
            profiles[active_heading].append(line)
    require(
        set(profiles) == {"AC Power:", "Battery Power:"},
        f"{context} does not contain both power profiles",
    )
    return "\n".join(profiles[heading])


def parse_low_power_mode(text: str, source: str, context: str) -> bool:
    profile = active_power_profile(text, source, context)
    matches = re.findall(r"(?m)^\s*lowpowermode\s+([01])\s*$", profile)
    require(
        len(matches) == 1,
        f"{context} active profile must contain one lowpowermode value",
    )
    return matches[0] == "1"


def parse_swap_usage(text: str, context: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("total", "used", "free"):
        matches = re.findall(
            rf"\b{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT]?)\b",
            text,
            re.IGNORECASE,
        )
        require(len(matches) == 1, f"{context} must contain one swap {name} value")
        amount_text, suffix_text = matches[0]
        multiplier = {
            "": 1,
            "K": 1024,
            "M": 1024**2,
            "G": 1024**3,
            "T": 1024**4,
        }[suffix_text.upper()]
        values[f"{name}_bytes"] = round(float(amount_text) * multiplier)
    values["raw"] = text.strip()
    return values


def parse_thermal_nominal(text: str, context: str) -> bool:
    normalized = [
        line.strip().removeprefix("Note: ").strip()
        for line in text.splitlines()
        if line.strip()
    ]
    thermal_lines = [line for line in normalized if "thermal warning" in line.lower()]
    performance_lines = [
        line for line in normalized if "performance warning" in line.lower()
    ]
    require(
        len(thermal_lines) == 1 and len(performance_lines) == 1,
        f"{context} must contain exactly one thermal and performance status",
    )
    return (
        thermal_lines[0] == "No thermal warning level has been recorded"
        and performance_lines[0] == "No performance warning level has been recorded"
    )


def parse_host_capture(path: Path, context: str) -> dict[str, Any]:
    sections = parse_host_sections(_read_text(path, context), context)
    source = parse_power_source(sections["battery"], context)
    thermal = sections["thermal"]
    return {
        "power_source": source,
        "battery_percent": parse_battery_percent(sections["battery"], context),
        "low_power_mode": parse_low_power_mode(
            sections["custom"],
            source,
            context,
        ),
        "thermal_nominal": parse_thermal_nominal(thermal, context),
        "swap": parse_swap_usage(sections["swap"], context),
    }


def validate_host_resources(
    evidence_dir: Path,
    record: dict[str, Any],
    *,
    power_policy: str,
    battery_minimum_percent: int,
    global_swap_policy_name: str,
    context: str,
    prefix: str,
) -> tuple[str, str]:
    before = expect_mapping(record.get("host_before"), f"{context} host_before")
    after = expect_mapping(record.get("host_after"), f"{context} host_after")
    expect_exact_keys(
        before,
        {
            "captured_at",
            "host_raw_path",
            "process_listing_path",
            "power_source",
            "power_policy",
            "ac_power",
            "battery_percent",
            "battery_minimum_percent",
            "low_power_mode",
            "thermal_nominal",
            "swap",
            "ports",
        },
        f"{context} host_before",
    )
    expect_exact_keys(
        after,
        {
            "captured_at",
            "raw_path",
            "power_source",
            "battery_percent",
            "battery_minimum_percent",
            "low_power_mode",
            "thermal_nominal",
            "swap",
        },
        f"{context} host_after",
    )
    require(
        global_swap_policy_name == GLOBAL_SWAP_POLICY,
        f"{context} global swap policy is unsupported",
    )

    before_name, before_path = relative_evidence_path(
        evidence_dir,
        before.get("host_raw_path"),
        f"{context} host_raw_path",
    )
    require(
        before_name == f"{prefix}.host-preflight.txt",
        f"{context} preflight host capture path differs",
    )
    require(before_path.is_file(), f"{context} preflight host capture is missing")
    after_name, after_path = relative_evidence_path(
        evidence_dir,
        after.get("raw_path"),
        f"{context} raw_path",
    )
    require(
        after_name == f"{prefix}.host-postflight.txt",
        f"{context} postflight host capture path differs",
    )
    require(after_path.is_file(), f"{context} postflight host capture is missing")
    process_name, process_path = relative_evidence_path(
        evidence_dir,
        before.get("process_listing_path"),
        f"{context} process_listing_path",
    )
    require(
        process_name == f"{prefix}.processes.txt",
        f"{context} process-listing path differs",
    )
    require(process_path.is_file(), f"{context} process listing is missing")
    require(
        _read_text(process_path, f"{context} process listing").strip() != "",
        f"{context} process listing is empty",
    )

    raw_before = parse_host_capture(
        before_path,
        f"{context} preflight host capture",
    )
    raw_after = parse_host_capture(
        after_path,
        f"{context} postflight host capture",
    )
    source = expect_string(
        before.get("power_source"),
        f"{context} preflight power source",
    )
    require(source in ("ac", "battery"), f"{context} power source is invalid")
    require(
        source == raw_before["power_source"],
        f"{context} preflight power source differs from raw capture",
    )
    require(
        before.get("power_policy") == power_policy,
        f"{context} power policy differs from preflight",
    )
    require(
        before.get("ac_power") is (source == "ac"),
        f"{context} ac_power flag disagrees with its power source",
    )
    before_battery = expect_int(
        before.get("battery_percent"),
        f"{context} preflight battery percentage",
    )
    require(
        before_battery == raw_before["battery_percent"],
        f"{context} preflight battery percentage differs from raw capture",
    )
    require(
        before.get("battery_minimum_percent") == battery_minimum_percent,
        f"{context} preflight battery minimum differs from preflight policy",
    )
    require(
        battery_minimum_percent <= before_battery <= 100,
        f"{context} preflight battery percentage is outside "
        f"{battery_minimum_percent}-100%",
    )
    require(
        before.get("low_power_mode") is raw_before["low_power_mode"],
        f"{context} preflight Low Power Mode differs from raw capture",
    )
    require(before.get("low_power_mode") is False, f"{context} used Low Power Mode")
    require(
        before.get("thermal_nominal") is raw_before["thermal_nominal"],
        f"{context} preflight thermal state differs from raw capture",
    )
    require(
        before.get("thermal_nominal") is True,
        f"{context} preflight thermal state is not nominal",
    )

    after_source = expect_string(
        after.get("power_source"),
        f"{context} postflight power source",
    )
    require(
        after_source in ("ac", "battery"),
        f"{context} postflight power source is invalid",
    )
    require(
        after_source == raw_after["power_source"],
        f"{context} postflight power source differs from raw capture",
    )
    after_battery = expect_int(
        after.get("battery_percent"),
        f"{context} postflight battery percentage",
    )
    require(
        after_battery == raw_after["battery_percent"],
        f"{context} postflight battery percentage differs from raw capture",
    )
    require(
        after.get("battery_minimum_percent") == battery_minimum_percent,
        f"{context} postflight battery minimum differs from preflight policy",
    )
    require(
        battery_minimum_percent <= after_battery <= 100,
        f"{context} postflight battery percentage is outside "
        f"{battery_minimum_percent}-100%",
    )
    require(
        after.get("low_power_mode") is raw_after["low_power_mode"],
        f"{context} postflight Low Power Mode differs from raw capture",
    )
    require(after.get("low_power_mode") is False, f"{context} enabled Low Power Mode")
    require(
        after.get("thermal_nominal") is raw_after["thermal_nominal"],
        f"{context} postflight thermal state differs from raw capture",
    )
    require(
        after.get("thermal_nominal") is True,
        f"{context} postflight thermal state is not nominal",
    )
    before_swap = expect_mapping(before.get("swap"), f"{context} preflight swap")
    after_swap = expect_mapping(after.get("swap"), f"{context} postflight swap")
    expect_exact_keys(
        before_swap,
        {"total_bytes", "used_bytes", "free_bytes", "raw"},
        f"{context} preflight swap",
    )
    expect_exact_keys(
        after_swap,
        {"total_bytes", "used_bytes", "free_bytes", "raw"},
        f"{context} postflight swap",
    )
    require(
        before_swap == raw_before["swap"],
        f"{context} preflight swap differs from raw capture",
    )
    require(
        after_swap == raw_after["swap"],
        f"{context} postflight swap differs from raw capture",
    )
    before_used = expect_int(
        before_swap.get("used_bytes"),
        f"{context} preflight swap usage",
        minimum=0,
    )
    after_used = expect_int(
        after_swap.get("used_bytes"),
        f"{context} postflight swap usage",
        minimum=0,
    )
    swap_growth = after_used - before_used
    require(
        record.get("global_swap_growth_bytes") == swap_growth,
        f"{context} global swap growth differs from host snapshots",
    )
    ports = expect_list(before.get("ports"), f"{context} server-port checks")
    require(
        len(ports) == len(SERVER_PORTS), f"{context} server-port check count differs"
    )
    for port_record, expected_port in zip(ports, SERVER_PORTS):
        port = expect_mapping(port_record, f"{context} port {expected_port}")
        expect_exact_keys(
            port,
            {"host", "port", "connect_errno", "closed"},
            f"{context} port {expected_port}",
        )
        require(
            port.get("host") == "127.0.0.1"
            and port.get("port") == expected_port
            and type(port.get("connect_errno")) is int
            and port.get("closed") is True,
            f"{context} did not prove port {expected_port} closed",
        )
    return source, power_policy


def validate_run_records(
    evidence_dir: Path,
    records_value: Any,
    *,
    schedule: list[dict[str, Any]],
    dataset: dict[str, Any],
    binaries: dict[str, dict[str, Any]],
    campaign_binding: dict[str, Any],
    runtime_artifact_hashes: dict[str, str],
    power_policy: str,
    battery_minimum_percent: int,
    global_swap_policy_name: str,
    idle_minimum_percent: float,
    truth: dict[str, Any],
) -> list[dict[str, Any]]:
    records_raw = expect_list(records_value, "runs.json")
    require(
        len(records_raw) == len(schedule) == 18,
        "runs.json must contain all 18 frozen schedule members",
    )
    records: list[dict[str, Any]] = []
    schedule_keys = (
        "sequence",
        "pair",
        "pair_order",
        "treatment",
        "engine",
        "participants",
    )

    for index, (record_value, member) in enumerate(zip(records_raw, schedule)):
        context = f"run {index:02d}-{member['pair']}-{member['engine']}"
        record = expect_mapping(record_value, context)
        expect_exact_keys(record, RUN_RECORD_FIELDS, context)
        for key in schedule_keys:
            require(
                type(record.get(key)) is type(member[key]),
                f"{context} {key} has the wrong JSON type",
            )
        actual_member = {key: record.get(key) for key in schedule_keys}
        expected_member = {key: member[key] for key in schedule_keys}
        require(
            actual_member == expected_member,
            f"{context} does not match its frozen schedule member",
        )

        prefix = f"{index:02d}-{member['pair']}-{member['engine']}"
        member_record_path = evidence_dir / f"{prefix}.json"
        require(member_record_path.is_file(), f"{context} member JSON is missing")
        require(
            read_json(member_record_path, f"{context} member JSON") == record,
            f"{context} differs between runs.json and its member JSON",
        )

        require(record.get("status") == "pass", f"{context} status is not pass")
        require(record.get("scope") == D0_SCOPE, f"{context} scope differs")
        require(record.get("errors") == [], f"{context} contains recorded errors")
        engine = member["engine"]
        participants = member["participants"]
        role_record = campaign_binding["roles"][engine]
        member_binding: dict[str, str | int] = {
            "campaign_nonce": campaign_binding["campaign_nonce"],
            "schedule_sequence": int(member["sequence"]),
            "role_id": int(role_record["role_id"]),
            "binary_sha256": role_record["binary_sha256"],
            "dataset_sha256": campaign_binding["dataset_sha256"],
        }

        binary = expect_mapping(record.get("binary"), f"{context} binary")
        require(
            binary
            == {
                "path": binaries[engine]["path"],
                "sha256": binaries[engine]["sha256"],
            },
            f"{context} binary identity differs from preflight",
        )

        command = expect_list(record.get("command"), f"{context} command")
        require(len(command) == 19, f"{context} command length differs")
        dataset_argument = expect_string(command[3], f"{context} dataset argument")
        require(
            dataset_argument == dataset["path"],
            f"{context} command does not name the evidence dataset exactly",
        )
        require(
            record.get("working_directory") == ".",
            f"{context} working directory is not the evidence directory",
        )
        require(
            command
            == expected_command(
                binaries[engine]["path"],
                dataset_argument,
                participants,
                f"{prefix}{AUDIT_SIDECAR_SUFFIX}",
                campaign_nonce=str(member_binding["campaign_nonce"]),
                schedule_sequence=int(member_binding["schedule_sequence"]),
                role_id=int(member_binding["role_id"]),
                binary_sha256=str(member_binding["binary_sha256"]),
                dataset_sha256=str(member_binding["dataset_sha256"]),
            ),
            f"{context} command differs from the frozen command",
        )
        require(
            record.get("environment") == expected_environment(participants),
            f"{context} environment differs from the frozen environment",
        )

        validate_dataset_fingerprint(
            record.get("dataset_before"),
            dataset=dataset,
            context=f"{context} dataset_before",
        )
        validate_dataset_fingerprint(
            record.get("dataset_after"),
            dataset=dataset,
            context=f"{context} dataset_after",
        )
        validate_runtime_artifacts(
            record.get("runtime_artifacts_before"),
            expected_hashes=runtime_artifact_hashes,
            context=f"{context} runtime_artifacts_before",
        )
        validate_runtime_artifacts(
            record.get("runtime_artifacts_after"),
            expected_hashes=runtime_artifact_hashes,
            context=f"{context} runtime_artifacts_after",
        )
        validate_host_resources(
            evidence_dir,
            record,
            power_policy=power_policy,
            battery_minimum_percent=battery_minimum_percent,
            global_swap_policy_name=global_swap_policy_name,
            context=context,
            prefix=prefix,
        )

        idle = validate_idle(
            evidence_dir,
            record.get("idle"),
            context=context,
            minimum_idle_percent=idle_minimum_percent,
        )
        idle_to_launch_ns = expect_int(
            record.get("idle_to_launch_ns"),
            f"{context} idle-to-launch delay",
            minimum=0,
        )
        require(
            idle_to_launch_ns <= IDLE_TO_LAUNCH_MAX_NS,
            f"{context} idle-to-launch delay exceeds one second",
        )
        started_ns = expect_int(
            record.get("started_at_unix_ns"),
            f"{context} start timestamp",
            minimum=1,
        )
        ended_ns = expect_int(
            record.get("ended_at_unix_ns"),
            f"{context} end timestamp",
            minimum=started_ns,
        )
        require(ended_ns >= started_ns, f"{context} ends before it starts")
        require(
            started_ns >= idle["accepted_at_unix_ns"],
            f"{context} starts before its idle gate was accepted",
        )

        fields = validate_driver_fields(
            record.get("fields"),
            engine=engine,
            participants=participants,
            dataset=dataset,
            context=context,
        )
        for name, expected in member_binding.items():
            require(
                fields[name] == str(expected),
                f"{context} field {name} differs from preflight binding",
            )
        stdout_name = f"{prefix}.stdout"
        stderr_name = f"{prefix}.stderr"
        require(
            record.get("stdout_path") == stdout_name,
            f"{context} stdout path differs",
        )
        require(
            record.get("stderr_path") == stderr_name,
            f"{context} stderr path differs",
        )
        stdout_path = evidence_dir / stdout_name
        stderr_path = evidence_dir / stderr_name
        require(stdout_path.is_file(), f"{context} stdout file is missing")
        require(stderr_path.is_file(), f"{context} stderr file is missing")
        captured_fields, captured_attestation = parse_attested_driver_stdout(
            _read_text(stdout_path, f"{context} stdout"),
            engine,
            f"{context} stdout",
        )
        require(
            captured_fields == fields,
            f"{context} fields differ from captured stdout",
        )
        expect_normalized_equal(
            record.get("attestation"),
            captured_attestation,
            f"{context} attestation",
        )
        validate_attestation_barriers(
            record.get("attestation_barriers"),
            process_group=expect_int(
                record.get("pid"),
                f"{context} PID",
                minimum=1,
            ),
            context=context,
        )
        validate_benchmark_process_evidence(
            record.get("benchmark_process_evidence"),
            barriers_value=record.get("attestation_barriers"),
            process_group=expect_int(
                record.get("pid"),
                f"{context} PID",
                minimum=1,
            ),
            expected_benchmark_path=binaries[engine]["path"],
            expected_benchmark_bytes=expect_int(
                binaries[engine]["bytes"],
                f"{context} preflight benchmark bytes",
                minimum=1,
            ),
            context=context,
        )
        validate_index_timing_binding(
            record.get("index_timing_binding"),
            barriers_value=record.get("attestation_barriers"),
            fields=fields,
            engine=engine,
            context=context,
        )

        sidecar_record = expect_mapping(
            record.get("audit_sidecar"),
            f"{context} audit_sidecar",
        )
        expect_exact_keys(
            sidecar_record,
            {"path", "sha256", "bytes"},
            f"{context} audit_sidecar",
        )
        sidecar_name, sidecar_path = relative_evidence_path(
            evidence_dir,
            sidecar_record["path"],
            f"{context} audit sidecar path",
        )
        expected_sidecar_name = f"{prefix}{AUDIT_SIDECAR_SUFFIX}"
        require(
            sidecar_name == expected_sidecar_name,
            f"{context} audit sidecar path differs",
        )
        require(
            command[13] == expected_sidecar_name,
            f"{context} command audit sidecar argument differs",
        )
        sidecar_sha256 = expect_sha256(
            sidecar_record["sha256"],
            f"{context} audit sidecar sha256",
        )
        sidecar_bytes = expect_int(
            sidecar_record["bytes"],
            f"{context} audit sidecar bytes",
            minimum=1,
        )
        require(
            sidecar_bytes <= MAXIMUM_AUDIT_SIDECAR_BYTES,
            f"{context} audit sidecar record exceeds the size bound",
        )
        data = read_audit_sidecar(sidecar_path, f"{context} audit sidecar")
        actual_sidecar_sha256 = hashlib.sha256(data).hexdigest()
        stdout_sidecar_bytes = parse_unsigned_field(
            fields,
            "audit_sidecar_bytes",
            context,
        )
        stdout_sidecar_sha256 = fields["audit_sidecar_sha256"]
        require(
            sidecar_bytes == stdout_sidecar_bytes == len(data),
            (
                f"{context} audit sidecar byte counts disagree between "
                "record, stdout, and file"
            ),
        )
        require(
            sidecar_sha256
            == stdout_sidecar_sha256
            == actual_sidecar_sha256,
            (
                f"{context} audit sidecar hashes disagree between "
                "record, stdout, and file"
            ),
        )
        reconstructed_audit, audit_details = parse_audit_sidecar(
            data,
            engine=engine,
            fields=fields,
            truth=truth,
            context=f"{context} audit sidecar",
            expected_campaign_nonce=str(member_binding["campaign_nonce"]),
            expected_schedule_sequence=int(member_binding["schedule_sequence"]),
            expected_role_id=int(member_binding["role_id"]),
            expected_binary_sha256=str(member_binding["binary_sha256"]),
            expected_dataset_sha256=str(member_binding["dataset_sha256"]),
        )
        validate_audit_stdout_fields(
            fields,
            reconstructed_audit,
            audit_details,
            engine=engine,
            context=context,
        )
        expect_normalized_equal(
            record.get("audit"),
            reconstructed_audit,
            f"{context} normalized audit",
        )

        maximum_rss = expect_int(
            record.get("process_lifetime_maximum_resident_set_size"),
            f"{context} process-lifetime maximum resident set size",
            minimum=1,
        )
        stderr_text = _read_text(stderr_path, f"{context} stderr")
        rss_matches = MAXIMUM_RSS.findall(stderr_text)
        require(
            len(rss_matches) == 1,
            f"{context} stderr must contain exactly one maximum RSS",
        )
        require(
            int(rss_matches[0]) == maximum_rss,
            f"{context} maximum RSS differs from captured stderr",
        )
        process_swaps = expect_int(
            record.get("process_swaps"),
            f"{context} process swaps",
            minimum=0,
        )
        swap_matches = PROCESS_SWAPS.findall(stderr_text)
        require(
            len(swap_matches) == 1,
            f"{context} stderr must contain exactly one process swap count",
        )
        require(
            int(swap_matches[0]) == process_swaps,
            f"{context} process swaps differ from captured stderr",
        )
        require(process_swaps == 0, f"{context} process reported swaps")
        records.append(record)
    validate_run_chronology(records)
    return records


def validate_run_chronology(records: Sequence[dict[str, Any]]) -> None:
    previous_end: int | None = None
    for index, record in enumerate(records):
        started_ns = expect_int(
            record.get("started_at_unix_ns"),
            f"run {index} chronology start",
            minimum=1,
        )
        ended_ns = expect_int(
            record.get("ended_at_unix_ns"),
            f"run {index} chronology end",
            minimum=started_ns,
        )
        if previous_end is not None:
            require(
                started_ns >= previous_end,
                f"run {index} overlaps or predates the previous member",
            )
        previous_end = ended_ns


def median(values: Sequence[int | float]) -> float:
    require(bool(values), "Cannot compute a median of no values")
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def geometric_mean(values: Sequence[float]) -> float:
    require(bool(values), "Cannot compute a geometric mean of no values")
    require(
        all(math.isfinite(value) and value > 0 for value in values),
        "Geometric-mean input is invalid",
    )
    return math.exp(sum(math.log(value) for value in values) / len(values))


def relative_mad(values: Sequence[int | float]) -> tuple[float, float]:
    center = median(values)
    require(center > 0, "Cannot compute relative MAD around zero")
    deviation = median([abs(value - center) for value in values])
    return center, deviation / center


def fraction_record(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def recall_point_fraction(
    record: dict[str, Any],
    k: int,
    ef: int,
) -> Fraction:
    engine = str(record["engine"])
    pair = str(record["pair"])
    matches = [
        point
        for point in record["audit"]["recall"]["points"]
        if point["k"] == k and point["ef"] == ef
    ]
    require(
        len(matches) == 1,
        f"{pair} {engine} lacks a unique recall point k={k}, ef={ef}",
    )
    point = expect_mapping(matches[0], f"{pair} {engine} recall k={k}, ef={ef}")
    expect_exact_keys(
        point,
        {"k", "ef", "eligible_hits", "possible_hits", "recall"},
        f"{pair} {engine} recall k={k}, ef={ef}",
    )
    eligible_hits = expect_int(
        point.get("eligible_hits"),
        f"{pair} {engine} eligible hits k={k}, ef={ef}",
        minimum=0,
    )
    possible_hits = expect_int(
        point.get("possible_hits"),
        f"{pair} {engine} possible hits k={k}, ef={ef}",
        minimum=1,
    )
    require(
        eligible_hits <= possible_hits
        and possible_hits == HELDOUT_QUERY_COUNT * k,
        f"{pair} {engine} recall hit counts differ k={k}, ef={ef}",
    )
    recall = Fraction(eligible_hits, possible_hits)
    recorded_recall = point.get("recall")
    require(
        type(recorded_recall) is float
        and math.isfinite(recorded_recall)
        and recorded_recall == float(recall),
        f"{pair} {engine} recall float differs from hit counts k={k}, ef={ef}",
    )
    return recall


def recompute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in records if record["treatment"] == "measured"]
    require(len(measured) == 12, "There must be exactly 12 measured run members")

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for record in measured:
        pair = record["pair"]
        engine = record["engine"]
        pair_members = by_pair.setdefault(pair, {})
        require(engine not in pair_members, f"Duplicate {engine} member in {pair}")
        pair_members[engine] = record

    ingestion_pairs: list[dict[str, Any]] = []
    durations: dict[str, list[int]] = {"infinity": [], "faiss": []}
    query_pairs: list[dict[str, Any]] = []
    query_values: dict[str, dict[str, list[int | float]]] = {
        engine: {
            "throughput_wall_ns": [],
            "qps": [],
            "tps": [],
            "p50_ns": [],
            "p95_ns": [],
            "p99_ns": [],
            "recall_at_10": [],
        }
        for engine in ("infinity", "faiss")
    }
    for pair_name in ("P1", "P2", "P3", "P4", "P5", "P6"):
        members = by_pair.get(pair_name, {})
        require(
            set(members) == {"infinity", "faiss"},
            f"Measured pair {pair_name} is incomplete",
        )
        infinity_fields = members["infinity"]["fields"]
        faiss_fields = members["faiss"]["fields"]
        infinity_ns = parse_unsigned_field(
            infinity_fields,
            "infinity_cold_build_ns",
            pair_name,
        )
        faiss_ns = parse_unsigned_field(
            faiss_fields,
            "faiss_cold_build_ns",
            pair_name,
        )
        order = members["infinity"]["pair_order"]
        require(
            order == members["faiss"]["pair_order"],
            f"Measured pair {pair_name} has inconsistent order",
        )
        durations["infinity"].append(infinity_ns)
        durations["faiss"].append(faiss_ns)
        ingestion_pairs.append(
            {
                "pair": pair_name,
                "order": order,
                "infinity_cold_build_ns": infinity_ns,
                "faiss_cold_build_ns": faiss_ns,
                "infinity_vectors_per_second": VECTORS * 1e9 / infinity_ns,
                "faiss_vectors_per_second": VECTORS * 1e9 / faiss_ns,
                "ratio": faiss_ns / infinity_ns,
            }
        )
        pair_query: dict[str, dict[str, int | float]] = {}
        pair_recall_fractions: dict[str, Fraction] = {}
        for engine in ("infinity", "faiss"):
            raw_query = members[engine]["audit"]["query"]
            samples = [
                expect_int(
                    value,
                    f"{pair_name} {engine} query latency sample",
                    minimum=0,
                )
                for value in expect_list(
                    raw_query["latency_samples_ns"],
                    f"{pair_name} {engine} query latency samples",
                )
            ]
            operations = expect_int(
                raw_query["throughput_operations"],
                f"{pair_name} {engine} query throughput operations",
                minimum=1,
            )
            wall_ns = expect_int(
                raw_query["throughput_wall_ns"],
                f"{pair_name} {engine} query throughput wall time",
                minimum=1,
            )
            require(
                len(samples) == QUERY_LATENCY_SAMPLE_COUNT
                and operations == QUERY_THROUGHPUT_OPERATIONS,
                f"{pair_name} {engine} query evidence count differs",
            )
            recall_fraction = recall_point_fraction(
                members[engine],
                QUERY_K,
                QUERY_EF_SEARCH,
            )
            pair_recall_fractions[engine] = recall_fraction
            qps = operations * 1e9 / wall_ns
            details: dict[str, int | float] = {
                "throughput_wall_ns": wall_ns,
                "qps": qps,
                "tps": qps,
                "p50_ns": nearest_rank(samples, 0.50),
                "p95_ns": nearest_rank(samples, 0.95),
                "p99_ns": nearest_rank(samples, 0.99),
                "recall_at_10": float(recall_fraction),
            }
            require(
                details["p50_ns"] > 0
                and details["p50_ns"] <= details["p95_ns"] <= details["p99_ns"],
                f"{pair_name} {engine} query percentiles are invalid",
            )
            pair_query[engine] = details
            for name, value in details.items():
                query_values[engine][name].append(value)

        infinity_query = pair_query["infinity"]
        faiss_query = pair_query["faiss"]
        absolute_recall_gap_fraction = abs(
            pair_recall_fractions["infinity"]
            - pair_recall_fractions["faiss"]
        )
        absolute_recall_gap = float(absolute_recall_gap_fraction)
        query_pairs.append(
            {
                "pair": pair_name,
                "order": order,
                "infinity": infinity_query,
                "faiss": faiss_query,
                "ratios": {
                    "qps": float(infinity_query["qps"])
                    / float(faiss_query["qps"]),
                    "tps": float(infinity_query["tps"])
                    / float(faiss_query["tps"]),
                    "p50_latency": float(faiss_query["p50_ns"])
                    / float(infinity_query["p50_ns"]),
                    "p95_latency": float(faiss_query["p95_ns"])
                    / float(infinity_query["p95_ns"]),
                    "p99_latency": float(faiss_query["p99_ns"])
                    / float(infinity_query["p99_ns"]),
                },
                "absolute_recall_gap": absolute_recall_gap,
                "absolute_recall_gap_exact": fraction_record(
                    absolute_recall_gap_fraction
                ),
                "recall_floor_met": (
                    pair_recall_fractions["infinity"]
                    >= Fraction(*QUERY_RECALL_FLOOR_FRACTION)
                    and pair_recall_fractions["faiss"]
                    >= Fraction(*QUERY_RECALL_FLOOR_FRACTION)
                ),
                "recall_gap_met": (
                    absolute_recall_gap_fraction
                    <= Fraction(*QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP_FRACTION)
                ),
            }
        )

    ingestion_strata: dict[str, Any] = {}
    for order in ("infinity/faiss", "faiss/infinity"):
        ratios = [
            pair["ratio"] for pair in ingestion_pairs if pair["order"] == order
        ]
        require(
            len(ratios) == 3,
            f"Order stratum {order} must contain exactly three measured pairs",
        )
        ingestion_strata[order] = {
            "ratios": ratios,
            "geometric_mean_ratio": geometric_mean(ratios),
        }
    ingestion_estimate = math.sqrt(
        ingestion_strata["infinity/faiss"]["geometric_mean_ratio"]
        * ingestion_strata["faiss/infinity"]["geometric_mean_ratio"]
    )

    ingestion_engines: dict[str, Any] = {}
    for engine in ("infinity", "faiss"):
        center, relative = relative_mad(durations[engine])
        ingestion_engines[engine] = {
            "wall_time_ns": durations[engine],
            "vectors_per_second": [
                VECTORS * 1e9 / value for value in durations[engine]
            ],
            "median_wall_time_ns": center,
            "median_vectors_per_second": median(
                [VECTORS * 1e9 / value for value in durations[engine]]
            ),
            "relative_mad": relative,
        }
    _, ingestion_ratio_relative_mad = relative_mad(
        [pair["ratio"] for pair in ingestion_pairs]
    )

    query_strata: dict[str, Any] = {}
    for order in ("infinity/faiss", "faiss/infinity"):
        ratios = [
            pair["ratios"]["qps"] for pair in query_pairs if pair["order"] == order
        ]
        require(
            len(ratios) == 3,
            f"Query order stratum {order} must contain three pairs",
        )
        query_strata[order] = {
            "qps_ratios": ratios,
            "geometric_mean_qps_ratio": geometric_mean(ratios),
        }
    query_estimate = math.sqrt(
        query_strata["infinity/faiss"]["geometric_mean_qps_ratio"]
        * query_strata["faiss/infinity"]["geometric_mean_qps_ratio"]
    )
    query_engines: dict[str, Any] = {}
    for engine, values in query_values.items():
        _, qps_relative = relative_mad(values["qps"])
        query_engines[engine] = {
            **values,
            "median_throughput_wall_ns": median(values["throughput_wall_ns"]),
            "median_qps": median(values["qps"]),
            "median_tps": median(values["tps"]),
            "median_p50_ns": median(values["p50_ns"]),
            "median_p95_ns": median(values["p95_ns"]),
            "median_p99_ns": median(values["p99_ns"]),
            "median_recall_at_10": median(values["recall_at_10"]),
            "qps_relative_mad": qps_relative,
        }
    query_qps_ratios = [pair["ratios"]["qps"] for pair in query_pairs]
    _, query_ratio_relative_mad = relative_mad(query_qps_ratios)
    return {
        "ingestion": {
            "pairs": ingestion_pairs,
            "strata": ingestion_strata,
            "stratified_geometric_mean_ratio": ingestion_estimate,
            "ratio_range": [
                min(pair["ratio"] for pair in ingestion_pairs),
                max(pair["ratio"] for pair in ingestion_pairs),
            ],
            "engines": ingestion_engines,
            "paired_ratio_relative_mad": ingestion_ratio_relative_mad,
        },
        "query": {
            "protocol": expected_query_protocol(),
            "pairs": query_pairs,
            "strata": query_strata,
            "stratified_geometric_mean_qps_ratio": query_estimate,
            "qps_ratio_range": [
                min(query_qps_ratios),
                max(query_qps_ratios),
            ],
            "engines": query_engines,
            "paired_qps_ratio_relative_mad": query_ratio_relative_mad,
            "recall_floor_met": all(
                pair["recall_floor_met"] for pair in query_pairs
            ),
            "absolute_recall_gap_met": all(
                pair["recall_gap_met"] for pair in query_pairs
            ),
        },
    }


def recompute_audit_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(records) == 18, "Audit quality requires all 18 run members")
    graph_members = [
        {
            "sequence": record["sequence"],
            "pair": record["pair"],
            "engine": record["engine"],
            **record["audit"]["graph"],
        }
        for record in records
    ]
    all_valid_and_reachable = all(
        member["valid"] is True
        and member["vertex_count"] == VECTORS
        and member["reachable_count"] == VECTORS
        for member in graph_members
    )

    measured = [record for record in records if record["treatment"] == "measured"]
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for record in measured:
        members = by_pair.setdefault(record["pair"], {})
        require(
            record["engine"] not in members,
            f"Duplicate {record['engine']} audit member in pair {record['pair']}",
        )
        members[record["engine"]] = record

    point_quality: list[dict[str, Any]] = []
    for k, ef in RECALL_POINTS:
        infinity_values: list[float] = []
        faiss_values: list[float] = []
        paired: list[dict[str, Any]] = []
        deficit_fractions: list[Fraction] = []
        for pair in ("P1", "P2", "P3", "P4", "P5", "P6"):
            members = by_pair.get(pair, {})
            require(
                set(members) == {"infinity", "faiss"},
                f"Audit pair {pair} is incomplete",
            )
            recall_fractions = {
                engine: recall_point_fraction(members[engine], k, ef)
                for engine in ("infinity", "faiss")
            }
            recalls = {
                engine: float(value)
                for engine, value in recall_fractions.items()
            }
            infinity_values.append(recalls["infinity"])
            faiss_values.append(recalls["faiss"])
            deficit_fraction = (
                recall_fractions["faiss"] - recall_fractions["infinity"]
            )
            deficit_fractions.append(deficit_fraction)
            paired.append(
                {
                    "pair": pair,
                    "infinity": recalls["infinity"],
                    "faiss": recalls["faiss"],
                    "infinity_deficit": float(deficit_fraction),
                    "infinity_deficit_exact": fraction_record(deficit_fraction),
                }
            )
        maximum_deficit_fraction = max(Fraction(0), max(deficit_fractions))
        point_quality.append(
            {
                "k": k,
                "ef": ef,
                "infinity_median": median(infinity_values),
                "faiss_median": median(faiss_values),
                "paired": paired,
                "maximum_infinity_deficit": float(maximum_deficit_fraction),
                "maximum_infinity_deficit_exact": fraction_record(
                    maximum_deficit_fraction
                ),
                "deficit_at_most_0_005": all(
                    deficit
                    <= Fraction(*MAXIMUM_PAIRED_RECALL_DEFICIT_FRACTION)
                    for deficit in deficit_fractions
                ),
            }
        )

    return {
        "graph": {
            "all_valid_and_reachable": all_valid_and_reachable,
            "members": graph_members,
        },
        "recall": {
            "query_count": HELDOUT_QUERY_COUNT,
            "query_seed": HELDOUT_QUERY_SEED,
            "maximum_paired_deficit": MAXIMUM_PAIRED_RECALL_DEFICIT,
            "maximum_paired_deficit_exact": fraction_record(
                Fraction(*MAXIMUM_PAIRED_RECALL_DEFICIT_FRACTION)
            ),
            "all_points_within_deficit": all(
                point["deficit_at_most_0_005"] for point in point_quality
            ),
            "points": point_quality,
        },
    }


def compare_summary(
    summary_value: Any,
    *,
    metrics: dict[str, Any],
    audit_quality: dict[str, Any],
    dataset: dict[str, Any],
    schedule_sha256: str,
    power_sources: list[str],
    power_policies: list[str],
    idle_minimum_percent: float,
    recorded_idle_policy: str,
    global_swap_policy_name: str,
    records: list[dict[str, Any]],
) -> None:
    summary = expect_mapping(summary_value, "summary.json")
    base_summary_keys = {
        "status",
        "scope",
        "claim_eligible",
        "power",
        "idle",
        "resources",
        "quality",
        "workload",
        "ingestion",
        "query",
        "acceptance",
        "limitations",
    }
    expect_exact_keys(summary, base_summary_keys, "summary.json")
    require(
        summary["status"] in ("pass", "fail"),
        "summary.json status is not a completed outcome",
    )
    require(summary["scope"] == D0_SCOPE, "summary.json scope differs")
    require(summary["claim_eligible"] is False, "D0 summary is marked claim-eligible")
    expect_normalized_equal(
        summary["quality"],
        audit_quality,
        "summary quality",
    )
    require(
        summary["power"] == {"sources": power_sources, "policies": power_policies},
        "summary power record differs from the runs",
    )
    require(
        summary["idle"]
        == {
            "minimum_percent": idle_minimum_percent,
            "window_seconds": IDLE_WINDOW_SECONDS,
            "policy": recorded_idle_policy,
        },
        "summary idle protocol differs from preflight",
    )
    maximum_swap_growth = max(
        0,
        max(
            expect_int(
                record.get("global_swap_growth_bytes"),
                "run global swap growth",
            )
            for record in records
        ),
    )
    require(
        summary["resources"]
        == {
            "global_swap_policy": global_swap_policy_name,
            "maximum_observed_global_swap_growth_bytes": maximum_swap_growth,
            "indexing_window_peak_rss_established": False,
        },
        "summary resource protocol differs from the runs",
    )
    limitations = expect_list(summary["limitations"], "summary limitations")
    require(
        limitations and all(isinstance(item, str) and item for item in limitations),
        "summary limitations must be a non-empty string array",
    )
    if recorded_idle_policy == "development-only-override":
        require(
            development_idle_limitation(idle_minimum_percent) in limitations,
            "summary limitations omit the development idle override",
        )
    if global_swap_policy_name == "development-record-only":
        require(
            DEVELOPMENT_SWAP_LIMITATION in limitations,
            "summary limitations omit the development swap policy",
        )
    require(
        INDEXING_WINDOW_RSS_LIMITATION in limitations,
        "summary limitations omit the non-qualifying RSS scope",
    )
    require(
        QUERY_SCOPE_LIMITATION in limitations,
        "summary limitations omit the process-local query scope",
    )

    expected_workload = {
        "vectors": VECTORS,
        "dimensions": DIMENSIONS,
        "M": M,
        "ef_construction": EF_CONSTRUCTION,
        "ef_search": EF_SEARCH,
        "participants": MEASURED_PARTICIPANTS,
        "data_fnv1a64": dataset["fnv1a64"],
        "data_sha256": dataset["sha256"],
        "data_bytes": dataset["bytes"],
        "schedule_sha256": schedule_sha256,
    }
    workload = expect_mapping(summary["workload"], "summary workload")
    expect_exact_keys(workload, expected_workload, "summary workload")
    for key, expected in expected_workload.items():
        require(
            type(workload[key]) is type(expected),
            f"summary workload {key} has the wrong JSON type",
        )
    require(
        workload == expected_workload,
        "summary workload differs from frozen data or schedule",
    )

    expect_normalized_equal(
        summary["ingestion"],
        metrics["ingestion"],
        "summary ingestion",
    )
    expect_normalized_equal(
        summary["query"],
        metrics["query"],
        "summary query",
    )

    acceptance = expect_mapping(summary["acceptance"], "summary acceptance")
    expected_acceptance_keys = {
        "all_runs_valid",
        "ingestion_relative_mad_at_most_0_10",
        "ingestion_paired_ratio_relative_mad_at_most_0_10",
        "query_qps_relative_mad_at_most_0_10",
        "query_paired_qps_ratio_relative_mad_at_most_0_10",
        "query_evidence_complete",
        "query_recall_floor_met",
        "query_absolute_recall_gap_at_most_0_005",
        "idle_protocol_met",
        "idle_to_launch_at_most_1_second",
        "host_resource_protocol_met",
        "dataset_integrity_met",
        "raw_graph_audit_met",
        "heldout_recall_audit_met",
        "paired_recall_deficit_at_most_0_005",
        "exact_balanced_schedule_met",
    }
    expect_exact_keys(acceptance, expected_acceptance_keys, "summary acceptance")
    expected_acceptance = {
        "all_runs_valid": True,
        "ingestion_relative_mad_at_most_0_10": all(
            metrics["ingestion"]["engines"][engine]["relative_mad"] <= 0.10
            for engine in ("infinity", "faiss")
        ),
        "ingestion_paired_ratio_relative_mad_at_most_0_10": (
            metrics["ingestion"]["paired_ratio_relative_mad"] <= 0.10
        ),
        "query_qps_relative_mad_at_most_0_10": all(
            metrics["query"]["engines"][engine]["qps_relative_mad"] <= 0.10
            for engine in ("infinity", "faiss")
        ),
        "query_paired_qps_ratio_relative_mad_at_most_0_10": (
            metrics["query"]["paired_qps_ratio_relative_mad"] <= 0.10
        ),
        "query_evidence_complete": True,
        "query_recall_floor_met": metrics["query"]["recall_floor_met"],
        "query_absolute_recall_gap_at_most_0_005": metrics["query"][
            "absolute_recall_gap_met"
        ],
        "idle_protocol_met": True,
        "idle_to_launch_at_most_1_second": True,
        "host_resource_protocol_met": True,
        "dataset_integrity_met": True,
        "raw_graph_audit_met": audit_quality["graph"][
            "all_valid_and_reachable"
        ],
        "heldout_recall_audit_met": True,
        "paired_recall_deficit_at_most_0_005": audit_quality["recall"][
            "all_points_within_deficit"
        ],
        "exact_balanced_schedule_met": True,
    }
    require(
        acceptance == expected_acceptance,
        "summary acceptance gates differ from independent recomputation",
    )
    expected_status = (
        "pass" if all(expected_acceptance.values()) else "fail"
    )
    require(
        summary["status"] == expected_status,
        "summary status differs from independent acceptance recomputation",
    )


def closure_records_for_orphan_check(
    evidence_dir: Path,
    preflight_value: Any,
) -> list[Any]:
    preflight = expect_mapping(preflight_value, "preflight.json")
    builds = expect_mapping(preflight.get("builds"), "preflight builds")
    values: list[Any] = []
    for context, build_value in (
        ("infinity", builds.get("infinity")),
        ("faiss", builds.get("faiss")),
        ("faiss library", preflight.get("linked_faiss_build")),
    ):
        build = expect_mapping(build_value, f"preflight {context} build")
        reference = expect_mapping(
            build.get("build_input_closure"),
            f"preflight {context} build-input closure",
        )
        _, path = relative_evidence_path(
            evidence_dir,
            reference.get("captured_path"),
            f"preflight {context} build-input closure path",
        )
        values.append(read_json(path, f"preflight {context} build-input closure"))
    return values


def validate_preflight_result(
    evidence_dir: Path,
    value: Any,
    *,
    dataset_sha256: str,
    heldout_manifest_sha256: str,
    preflight_sha256: str,
    schedule_sha256: str,
) -> None:
    result = expect_mapping(value, "preflight-result.json")
    expect_exact_keys(
        result,
        {
            "evidence_schema_version",
            "scope",
            "execution_mode",
            "status",
            "completed_at",
            "benchmark_members_executed",
            "campaign_complete",
            "artifact_sha256",
        },
        "preflight-result.json",
    )
    require(
        expect_int(
            result["evidence_schema_version"],
            "preflight result evidence schema version",
            minimum=1,
        )
        == EVIDENCE_SCHEMA_VERSION,
        "Preflight result schema version is unsupported",
    )
    require(result["scope"] == D0_SCOPE, "Preflight result scope differs")
    require(
        result["execution_mode"] == PREFLIGHT_ONLY_MODE,
        "Preflight result execution mode differs",
    )
    require(result["status"] == "pass", "Preflight result status is not pass")
    require(
        expect_string(
            result["completed_at"], "preflight completion timestamp"
        ).endswith("Z"),
        "Preflight completion timestamp is not UTC",
    )
    require(
        expect_int(
            result["benchmark_members_executed"],
            "preflight benchmark member count",
            minimum=0,
        )
        == 0,
        "Preflight-only evidence claims benchmark members",
    )
    require(
        type(result["campaign_complete"]) is bool
        and result["campaign_complete"] is False,
        "Preflight-only evidence claims a completed campaign",
    )
    artifacts = expect_mapping(
        result["artifact_sha256"],
        "preflight result artifact hashes",
    )
    expect_exact_keys(
        artifacts,
        {
            "dataset.json",
            HELDOUT_MANIFEST_FILENAME,
            "preflight.json",
            "schedule.json",
        },
        "preflight result artifact hashes",
    )
    require(
        artifacts
        == {
            "dataset.json": dataset_sha256,
            HELDOUT_MANIFEST_FILENAME: heldout_manifest_sha256,
            "preflight.json": preflight_sha256,
            "schedule.json": schedule_sha256,
        },
        "Preflight result artifact hashes differ",
    )


def verify_evidence(
    evidence_directory: Path,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    evidence_dir = Path(evidence_directory)
    require(evidence_dir.exists(), f"Evidence directory does not exist: {evidence_dir}")
    require(evidence_dir.is_dir(), f"Evidence path is not a directory: {evidence_dir}")
    require(not evidence_dir.is_symlink(), "Evidence directory must not be a symlink")
    evidence_dir = evidence_dir.resolve()
    with VerifiedEvidence(evidence_dir) as manifest:
        return _verify_open_evidence(
            evidence_dir,
            manifest,
            preflight_only=preflight_only,
        )


def _verify_open_evidence(
    evidence_dir: Path,
    manifest: dict[str, str],
    *,
    preflight_only: bool,
) -> dict[str, Any]:
    require("failure.json" not in manifest, "Evidence contains failure.json")
    common_required = {
        "dataset.json",
        DATASET_FILENAME,
        HELDOUT_MANIFEST_FILENAME,
        "preflight.json",
        "schedule.json",
    }
    missing = sorted(common_required - set(manifest))
    require(not missing, f"Completed evidence is missing required files: {missing}")

    schedule_path = evidence_dir / "schedule.json"
    dataset_path = evidence_dir / "dataset.json"
    heldout_path = evidence_dir / HELDOUT_MANIFEST_FILENAME
    preflight_path = evidence_dir / "preflight.json"

    schedule = validate_schedule(read_json(schedule_path, "schedule.json"))
    preflight_value = read_json(preflight_path, "preflight.json")
    preflight_mapping = expect_mapping(preflight_value, "preflight.json")
    require(
        expect_int(
            preflight_mapping.get("evidence_schema_version"),
            "preflight evidence schema version",
            minimum=1,
        )
        == EVIDENCE_SCHEMA_VERSION,
        "Preflight evidence schema version is unsupported; "
        f"expected {EVIDENCE_SCHEMA_VERSION}, found "
        f"{preflight_mapping.get('evidence_schema_version')!r}",
    )
    recorded_mode = expect_string(
        preflight_mapping.get("execution_mode"),
        "preflight execution mode",
    )
    expected_mode = PREFLIGHT_ONLY_MODE if preflight_only else FULL_CAMPAIGN_MODE
    require(
        recorded_mode == expected_mode,
        f"Evidence execution mode is {recorded_mode!r}, verifier expects "
        f"{expected_mode!r}",
    )
    mode_required = (
        {"preflight-result.json"} if preflight_only else {"runs.json", "summary.json"}
    )
    missing = sorted(mode_required - set(manifest))
    require(not missing, f"Completed evidence is missing required files: {missing}")
    dataset = validate_dataset(
        evidence_dir,
        read_json(dataset_path, "dataset.json"),
    )
    truth = heldout_truth_for_validated_dataset(
        evidence_dir / DATASET_FILENAME,
        dataset["sha256"],
    )
    heldout_manifest = validate_heldout_manifest(
        evidence_dir,
        read_json(heldout_path, HELDOUT_MANIFEST_FILENAME),
        dataset=dataset,
        truth=truth,
    )
    schedule_digest = sha256_file(schedule_path)
    dataset_record_digest = sha256_file(dataset_path)
    heldout_manifest_digest = sha256_file(heldout_path)
    (
        binaries,
        runtime_artifact_hashes,
        power_policy,
        battery_minimum_percent,
        global_swap_policy_name,
        idle_minimum_percent,
        recorded_idle_policy,
        campaign_binding,
    ) = validate_preflight(
        evidence_dir,
        preflight_value,
        dataset=dataset,
        dataset_sha256=dataset_record_digest,
        schedule_sha256=schedule_digest,
        heldout_manifest=heldout_manifest,
        heldout_manifest_sha256=heldout_manifest_digest,
        heldout_manifest_bytes=heldout_path.stat().st_size,
        expected_execution_mode=expected_mode,
    )
    closure_values = closure_records_for_orphan_check(
        evidence_dir,
        preflight_value,
    )
    if preflight_only:
        forbidden_exact = {"runs.json", "summary.json", "README.md"}
        present_forbidden = sorted(forbidden_exact & set(manifest))
        require(
            not present_forbidden,
            f"Preflight-only evidence contains campaign files: {present_forbidden}",
        )
        member_prefixes = tuple(
            f"{member['sequence']:02d}-{member['pair']}-{member['engine']}."
            for member in schedule
        )
        member_files = sorted(
            path for path in manifest if path.startswith(member_prefixes)
        )
        require(
            not member_files,
            f"Preflight-only evidence contains benchmark member files: {member_files}",
        )
        validate_preflight_result(
            evidence_dir,
            read_json(
                evidence_dir / "preflight-result.json",
                "preflight-result.json",
            ),
            dataset_sha256=dataset_record_digest,
            heldout_manifest_sha256=heldout_manifest_digest,
            preflight_sha256=sha256_file(preflight_path),
            schedule_sha256=schedule_digest,
        )
        verify_no_orphan_content_blobs(
            manifest,
            preflight_value,
            read_json(
                evidence_dir / "compiled-source-hashes.json",
                "compiled-source-hashes.json",
            ),
            read_json(
                evidence_dir / "benchmark-input-source-hashes.json",
                "benchmark-input-source-hashes.json",
            ),
            *closure_values,
        )
        return {
            "status": "PREFLIGHT_PASS",
            "execution_mode": PREFLIGHT_ONLY_MODE,
            "campaign_complete": False,
            "benchmark_members_executed": 0,
            "verified_manifest_members": len(manifest),
            "schedule_members": len(schedule),
        }

    require(
        "preflight-result.json" not in manifest,
        "Full campaign evidence contains preflight-result.json",
    )
    runs_path = evidence_dir / "runs.json"
    summary_path = evidence_dir / "summary.json"
    records_value = read_json(runs_path, "runs.json")
    records_list = expect_list(records_value, "runs.json")
    require(
        len(records_list) == 18,
        "runs.json must contain all 18 frozen schedule members",
    )
    for index, (record_value, member) in enumerate(zip(records_list, schedule)):
        record = expect_mapping(record_value, f"runs.json member {index}")
        for key, expected in member.items():
            require(
                record.get(key) == expected,
                f"runs.json member {index} differs from frozen {key}",
            )
        require(record.get("status") == "pass", f"runs.json member {index} failed")

    records = validate_run_records(
        evidence_dir,
        records_value,
        schedule=schedule,
        dataset=dataset,
        binaries=binaries,
        campaign_binding=campaign_binding,
        runtime_artifact_hashes=runtime_artifact_hashes,
        power_policy=power_policy,
        battery_minimum_percent=battery_minimum_percent,
        global_swap_policy_name=global_swap_policy_name,
        idle_minimum_percent=idle_minimum_percent,
        truth=truth,
    )
    metrics = recompute_metrics(records)
    audit_quality = recompute_audit_quality(records)
    power_sources = sorted(
        {
            str(host["power_source"])
            for record in records
            for host in (record["host_before"], record["host_after"])
        }
    )
    power_policies = sorted(
        {str(record["host_before"]["power_policy"]) for record in records}
    )
    compare_summary(
        read_json(summary_path, "summary.json"),
        metrics=metrics,
        audit_quality=audit_quality,
        dataset=dataset,
        schedule_sha256=schedule_digest,
        power_sources=power_sources,
        power_policies=power_policies,
        idle_minimum_percent=idle_minimum_percent,
        recorded_idle_policy=recorded_idle_policy,
        global_swap_policy_name=global_swap_policy_name,
        records=records,
    )
    verify_no_orphan_content_blobs(
        manifest,
        preflight_value,
        read_json(
            evidence_dir / "compiled-source-hashes.json",
            "compiled-source-hashes.json",
        ),
        read_json(
            evidence_dir / "benchmark-input-source-hashes.json",
            "benchmark-input-source-hashes.json",
        ),
        *closure_values,
    )
    return {
        "status": "PASS",
        "verified_manifest_members": len(manifest),
        "schedule_members": len(schedule),
        "measured_pairs": len(metrics["ingestion"]["pairs"]),
        "idle": {
            "minimum_percent": idle_minimum_percent,
            "window_seconds": IDLE_WINDOW_SECONDS,
            "policy": recorded_idle_policy,
        },
        "resources": {
            "global_swap_policy": global_swap_policy_name,
            "maximum_observed_global_swap_growth_bytes": max(
                0,
                max(int(record["global_swap_growth_bytes"]) for record in records),
            ),
            "indexing_window_peak_rss_established": False,
        },
        "metrics": metrics,
        "quality": audit_quality,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify native HNSW D0 campaign or preflight evidence.",
        allow_abbrev=False,
    )
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Require and verify a preflight-only evidence directory",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments.count("--preflight-only") > 1:
        parser.error("--preflight-only must not be repeated")
    return parser.parse_args(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_evidence(
            args.evidence_directory,
            preflight_only=args.preflight_only,
        )
    except VerificationError as error:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(error)},
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
