#!/usr/bin/env python3
"""Run the non-confirmatory Apple Silicon HNSW D0 comparison."""

from __future__ import annotations

import argparse
import array
import base64
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import random
import re
import secrets
import select
import shlex
import shutil
import signal
import socket
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping
from enum import Enum, auto
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

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
PINNED_OMP_HEADER_SHA256 = (
    "5974470842520cea4bc50136e2329bbf4e36ba928d317e86f7def2ba1752d3d4"
)

D0_SCOPE = "d0-development-only"
EVIDENCE_SCHEMA_VERSION = 11
FULL_CAMPAIGN_MODE = "full-campaign"
PREFLIGHT_ONLY_MODE = "preflight-only"
BUILD_INPUT_CLOSURE_SCHEMA_VERSION = 4
NINJA_SETTLEMENT_FAILURE_SCHEMA_VERSION = 1
NINJA_NO_WORK_MARKER = "ninja: no work to do."
NINJA_NO_WORK_MARKER_BYTES = NINJA_NO_WORK_MARKER.encode("utf-8")
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
PROCESS_STATUS_NAMES = {
    1: "creating",
    2: "running",
    3: "sleeping",
    4: "stopped",
    5: "zombie",
}
PROCESS_MAXIMUM_COUNT = 131_072
PROC_PIDTBSDINFO = 3
PROC_PIDTASKALLINFO = 2
PROC_PIDREGIONPATHINFO = 8
PROC_PIDPATHINFO_MAXSIZE = 4096
PROC_REGION_PATH_MAXSIZE = 1024
VM_PROT_READ = 0x01
VM_PROT_EXECUTE = 0x04
P_PID = 1
WNOHANG = 0x00000001
WEXITED = 0x00000004
WNOWAIT = 0x00000020
SA_NOCLDWAIT = 0x0020
CLD_EXITED = 1
CLD_KILLED = 2
CLD_DUMPED = 3
CLD_TRAPPED = 4
CLD_STOPPED = 5
CLD_CONTINUED = 6
TERMINAL_CHILD_CODES = frozenset((CLD_EXITED, CLD_KILLED, CLD_DUMPED))
WAITID_CHILD_CODES = TERMINAL_CHILD_CODES | frozenset(
    (CLD_TRAPPED, CLD_STOPPED, CLD_CONTINUED)
)
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
ATTESTATION_STOP_POLL_SECONDS = 0.01
ATTESTATION_LAUNCH_GATE_SECONDS = 5.0
ATTESTATION_POST_KILL_WAIT_SECONDS = 5.0
ATTESTATION_THREAD_JOIN_SECONDS = 5.0
ATTESTATION_DEFERRED_SIGNALS = (
    signal.SIGINT,
    signal.SIGTERM,
    signal.SIGHUP,
    signal.SIGQUIT,
)
ATTESTATION_MAXIMUM_IMAGE_COUNT = 4096
ATTESTATION_MAXIMUM_PATH_BYTES = 32 * 1024
MAXIMUM_AUDIT_SIDECAR_BYTES = 64 * 1024 * 1024
DATASET_SEED = 0
DATASET_FILENAME = "d0-f32le-n12288-d128-seed0.bin"
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
IDLE_SAMPLE_COUNT = IDLE_WINDOW_SECONDS + 1
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
SERVER_PORTS = (23871, 23872, 23873, 23874)
MINIMUM_DISK_RESERVE_BYTES = 12 * 1024**3
D0_WORST_CASE_NEW_BYTES = 8 * 1024**3
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
FIELD_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
UNSIGNED_DECIMAL_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\Z")
FINITE_DECIMAL_PATTERN = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))" r"(?:[eE][+-]?[0-9]+)?\Z"
)


class _ProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


class _ProcTaskAllInfo(ctypes.Structure):
    _fields_ = [
        ("pbsd", _ProcBsdInfo),
        ("ptinfo", _ProcTaskInfo),
    ]


class _ProcRegionInfo(ctypes.Structure):
    _fields_ = [
        ("pri_protection", ctypes.c_uint32),
        ("pri_max_protection", ctypes.c_uint32),
        ("pri_inheritance", ctypes.c_uint32),
        ("pri_flags", ctypes.c_uint32),
        ("pri_offset", ctypes.c_uint64),
        ("pri_behavior", ctypes.c_uint32),
        ("pri_user_wired_count", ctypes.c_uint32),
        ("pri_user_tag", ctypes.c_uint32),
        ("pri_pages_resident", ctypes.c_uint32),
        ("pri_pages_shared_now_private", ctypes.c_uint32),
        ("pri_pages_swapped_out", ctypes.c_uint32),
        ("pri_pages_dirtied", ctypes.c_uint32),
        ("pri_ref_count", ctypes.c_uint32),
        ("pri_shadow_depth", ctypes.c_uint32),
        ("pri_share_mode", ctypes.c_uint32),
        ("pri_private_pages_resident", ctypes.c_uint32),
        ("pri_shared_pages_resident", ctypes.c_uint32),
        ("pri_obj_id", ctypes.c_uint32),
        ("pri_depth", ctypes.c_uint32),
        ("pri_address", ctypes.c_uint64),
        ("pri_size", ctypes.c_uint64),
    ]


class _VinfoStat(ctypes.Structure):
    _fields_ = [
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_atime", ctypes.c_int64),
        ("vst_atimensec", ctypes.c_int64),
        ("vst_mtime", ctypes.c_int64),
        ("vst_mtimensec", ctypes.c_int64),
        ("vst_ctime", ctypes.c_int64),
        ("vst_ctimensec", ctypes.c_int64),
        ("vst_birthtime", ctypes.c_int64),
        ("vst_birthtimensec", ctypes.c_int64),
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        ("vst_blksize", ctypes.c_int32),
        ("vst_flags", ctypes.c_uint32),
        ("vst_gen", ctypes.c_uint32),
        ("vst_rdev", ctypes.c_uint32),
        ("vst_qspare", ctypes.c_int64 * 2),
    ]


class _Fsid(ctypes.Structure):
    _fields_ = [("val", ctypes.c_int32 * 2)]


class _VnodeInfo(ctypes.Structure):
    _fields_ = [
        ("vi_stat", _VinfoStat),
        ("vi_type", ctypes.c_int),
        ("vi_pad", ctypes.c_int),
        ("vi_fsid", _Fsid),
    ]


class _VnodeInfoPath(ctypes.Structure):
    _fields_ = [
        ("vip_vi", _VnodeInfo),
        ("vip_path", ctypes.c_char * PROC_REGION_PATH_MAXSIZE),
    ]


class _ProcRegionWithPathInfo(ctypes.Structure):
    _fields_ = [
        ("prp_prinfo", _ProcRegionInfo),
        ("prp_vip", _VnodeInfoPath),
    ]


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


class _DarwinSigaction(ctypes.Structure):
    _fields_ = [
        ("handler", ctypes.c_void_p),
        ("mask", ctypes.c_uint32),
        ("flags", ctypes.c_int),
    ]


@dataclass(frozen=True)
class _TerminalObservation:
    pid: int
    code: int
    status: int


class _AttestedLifecycle(Enum):
    OWNED = auto()
    TERMINAL_OBSERVED = auto()
    REAP_IN_PROGRESS = auto()
    REAPED = auto()


class _AttestedGroupState(Enum):
    ACTIVE = auto()
    KILL_SENT = auto()
    QUIESCENT = auto()
    EMPTY = auto()
    UNRESOLVED = auto()


class _AttestedLaunchPhase(Enum):
    REGISTERED = auto()
    FORKED = auto()
    GROUP_VERIFIED = auto()
    EXEC_RELEASED = auto()
    EXEC_CONFIRMED = auto()


@dataclass
class _GatedTextProcess:
    pid: int
    stdout: Any
    stderr: Any
    returncode: int | None = None
    production_owned: bool = True


@dataclass
class _GatedLaunchState:
    phase: _AttestedLaunchPhase = _AttestedLaunchPhase.REGISTERED
    process: _GatedTextProcess | None = None
    command_descriptor: int = -1
    status_descriptor: int = -1


@dataclass
class _AttestedChildState:
    process: Any
    lifecycle: _AttestedLifecycle = _AttestedLifecycle.OWNED
    group_state: _AttestedGroupState = _AttestedGroupState.ACTIVE
    terminal: _TerminalObservation | None = None
    terminal_observed_at_monotonic_ns: int | None = None
    pre_reap_member_pids: list[int] | None = None
    pre_reap_observed_at_monotonic_ns: int | None = None
    reap_status_validated: bool = False
    returncode: int | None = None
    reap_completed_at_monotonic_ns: int | None = None
    cleanup_deadline: float | None = None


@dataclass
class _DeferredSigintState:
    interruption: BaseException
    interruptions: Mapping[int, BaseException] | None = None
    request: tuple[int, BaseException] | None = None

    @property
    def requested(self) -> bool:
        return self.request is not None

    @property
    def signal_number(self) -> int:
        if self.request is not None:
            return self.request[0]
        return signal.SIGINT


def _deferred_signal_exception(state: _DeferredSigintState) -> BaseException:
    if state.request is not None:
        return state.request[1]
    if state.interruptions is not None:
        return state.interruptions[state.signal_number]
    return state.interruption


_LIBPROC: Any | None = None
_WAITID_LIBRARY: Any | None = None
_SIGACTION_LIBRARY: Any | None = None


class D0Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D0Failure(message)


def create_campaign_binding(
    *,
    binaries: dict[str, dict[str, Any]],
    dataset: dict[str, Any],
    schedule_sha256: str,
) -> dict[str, Any]:
    dataset_sha256 = str(dataset.get("sha256", ""))
    require(
        re.fullmatch(r"[0-9a-f]{64}", dataset_sha256) is not None,
        "Campaign dataset SHA-256 is invalid",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", schedule_sha256) is not None,
        "Campaign schedule SHA-256 is invalid",
    )
    roles: dict[str, dict[str, Any]] = {}
    for engine, role_id in STANDALONE_ROLE_IDS.items():
        binary_sha256 = str(binaries.get(engine, {}).get("sha256", ""))
        require(
            re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is not None,
            f"Campaign {engine} binary SHA-256 is invalid",
        )
        roles[engine] = {
            "role_id": role_id,
            "engine": engine,
            "binary_sha256": binary_sha256,
        }
    return {
        "schema_version": CAMPAIGN_BINDING_SCHEMA_VERSION,
        "campaign_nonce": secrets.token_hex(32),
        "dataset_sha256": dataset_sha256,
        "schedule_sha256": schedule_sha256,
        "roles": roles,
    }


def member_campaign_binding(
    entry: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, str | int]:
    engine = str(entry["engine"])
    require(engine in STANDALONE_ROLE_IDS, f"Unknown campaign engine: {engine}")
    require(
        type(campaign.get("schema_version")) is int
        and campaign.get("schema_version") == CAMPAIGN_BINDING_SCHEMA_VERSION,
        "Campaign binding schema differs",
    )
    role_record = campaign.get("roles", {}).get(engine)
    require(isinstance(role_record, dict), f"Campaign binding omits {engine}")
    require(
        role_record
        == {
            "role_id": STANDALONE_ROLE_IDS[engine],
            "engine": engine,
            "binary_sha256": role_record.get("binary_sha256"),
        },
        f"Campaign binding for {engine} differs",
    )
    result: dict[str, str | int] = {
        "campaign_nonce": str(campaign.get("campaign_nonce", "")),
        "schedule_sequence": int(entry["sequence"]),
        "role_id": STANDALONE_ROLE_IDS[engine],
        "binary_sha256": str(role_record["binary_sha256"]),
        "dataset_sha256": str(campaign.get("dataset_sha256", "")),
    }
    for name in ("campaign_nonce", "binary_sha256", "dataset_sha256"):
        require(
            re.fullmatch(r"[0-9a-f]{64}", str(result[name])) is not None,
            f"Campaign member {name} is invalid",
        )
    return result


def validate_driver_campaign_binding(
    fields: dict[str, str],
    binding: dict[str, str | int],
    *,
    context: str,
) -> None:
    for name in (
        "campaign_nonce",
        "schedule_sequence",
        "role_id",
        "binary_sha256",
        "dataset_sha256",
    ):
        require(
            fields.get(name) == str(binding[name]),
            f"{context} driver field {name} differs from preflight binding",
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64(data: bytes, initial: int = FNV1A64_OFFSET_BASIS) -> int:
    value = initial
    for byte in data:
        value ^= byte
        value = (value * FNV1A64_PRIME) & FNV1A64_MASK
    return value


def query_protocol() -> dict[str, Any]:
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
    require(
        all(
            type(checksum) is int and 0 <= checksum <= FNV1A64_MASK
            for checksum in per_query_checksums
        ),
        "Query benchmark checksum contains a value outside uint64",
    )
    value = fnv1a64(b"infinity-hnsw-d0-query-benchmark-v1")
    value = fnv1a64(struct.pack("<QQQ", HELDOUT_QUERY_COUNT, QUERY_K, QUERY_EF_SEARCH), value)
    for query_index, checksum in enumerate(per_query_checksums):
        value = fnv1a64(struct.pack("<QQ", query_index, checksum), value)
    return value


def nearest_rank(samples: list[int], percentile: float) -> int:
    require(bool(samples), "Cannot calculate a percentile from no samples")
    require(
        0.0 < percentile <= 1.0,
        "Nearest-rank percentile must be in (0, 1]",
    )
    require(
        all(type(sample) is int and sample >= 0 for sample in samples),
        "Latency samples must be nonnegative integers",
    )
    ordered = sorted(samples)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def query_performance_record(
    *,
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
    require(
        len(latency_samples_ns) == QUERY_LATENCY_SAMPLE_COUNT,
        "Query latency sample count differs from the frozen protocol",
    )
    require(
        all(
            type(sample) is int and 0 <= sample <= FNV1A64_MASK
            for sample in latency_samples_ns
        ),
        "Query latency sample is outside uint64",
    )
    require(
        type(timed_query_corpus_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", timed_query_corpus_sha256) is not None,
        "Timed query corpus SHA-256 is not canonical lowercase hex",
    )
    scalar_values = (
        latency_validated_operations,
        latency_validated_result_checksum,
        throughput_operations,
        throughput_validated_operations,
        throughput_wall_ns,
        throughput_validated_result_checksum,
        result_checksum,
    )
    require(
        all(
            type(value) is int and 0 <= value <= FNV1A64_MASK
            for value in scalar_values
        ),
        "Query benchmark measurement is outside uint64",
    )
    require(
        latency_validated_operations == QUERY_LATENCY_SAMPLE_COUNT,
        "Query latency validated operation count differs from the frozen protocol",
    )
    require(
        throughput_operations == QUERY_THROUGHPUT_OPERATIONS,
        "Query throughput operation count differs from the frozen protocol",
    )
    require(
        throughput_validated_operations == throughput_operations,
        "Query throughput validated operation count differs from the requested count",
    )
    require(throughput_wall_ns > 0, "Query throughput wall duration is not positive")
    aggregate_checksum = query_benchmark_checksum(per_query_checksums)
    require(
        latency_validated_result_checksum
        == throughput_validated_result_checksum
        == result_checksum
        == aggregate_checksum,
        "Query benchmark phase or aggregate checksum differs",
    )
    qps = throughput_operations * 1e9 / throughput_wall_ns
    return {
        **query_protocol(),
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
    sha = hashlib.sha256()
    fnv = FNV1A64_OFFSET_BASIS
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            sha.update(chunk)
            fnv = fnv1a64(chunk, fnv)
            byte_count += len(chunk)
    return {
        "sha256": sha.hexdigest(),
        "fnv1a64": fnv,
        "bytes": byte_count,
    }


def create_dataset(
    path: Path,
    *,
    vectors: int = VECTORS,
    dimensions: int = DIMENSIONS,
    seed: int = DATASET_SEED,
) -> dict[str, Any]:
    require(vectors > 0, "Dataset vector count must be positive")
    require(dimensions > 0, "Dataset dimension count must be positive")
    require(not path.exists(), f"Dataset already exists: {path}")

    temporary_path = path.with_name(path.name + ".tmp")
    require(
        not temporary_path.exists(), f"Dataset temporary file exists: {temporary_path}"
    )
    generator = random.Random(seed)
    pack_float = struct.Struct("<f").pack
    pending = bytearray()
    try:
        with temporary_path.open("xb") as destination:
            for _ in range(vectors * dimensions):
                pending.extend(pack_float(generator.random()))
                if len(pending) >= 1024 * 1024:
                    destination.write(pending)
                    pending.clear()
            if pending:
                destination.write(pending)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    fingerprint = fingerprint_file(path)
    expected_bytes = vectors * dimensions * FLOAT32_BYTES
    require(
        fingerprint["bytes"] == expected_bytes,
        f"Generated dataset has {fingerprint['bytes']} bytes, expected {expected_bytes}",
    )
    if vectors == VECTORS and dimensions == DIMENSIONS and seed == DATASET_SEED:
        require(
            fingerprint["sha256"] == DATASET_SHA256
            and fingerprint["fnv1a64"] == DATASET_FNV1A64,
            "Generated canonical D0 dataset identity differs",
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    require(mode & 0o222 == 0, f"Dataset is writable after chmod: mode {mode:#o}")
    return {
        "path": path.name,
        "seed": seed,
        "generator": "python-random.Random.random",
        "encoding": "raw-little-endian-float32",
        "vectors": vectors,
        "dimensions": dimensions,
        "values": vectors * dimensions,
        "bytes_per_value": FLOAT32_BYTES,
        "mode": f"{mode:#05o}",
        **fingerprint,
    }


def verify_dataset(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    require(path.is_file(), f"Dataset is missing: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    require(mode & 0o222 == 0, f"Dataset became writable: mode {mode:#o}")
    if (
        expected.get("vectors") == VECTORS
        and expected.get("dimensions") == DIMENSIONS
        and expected.get("seed") == DATASET_SEED
    ):
        require(
            expected.get("sha256") == DATASET_SHA256
            and expected.get("fnv1a64") == DATASET_FNV1A64
            and expected.get("bytes") == VECTORS * DIMENSIONS * FLOAT32_BYTES,
            "Frozen D0 dataset record is not canonical",
        )
    actual = fingerprint_file(path)
    for key in ("sha256", "fnv1a64", "bytes"):
        require(
            actual[key] == expected[key],
            f"Dataset {key} changed: expected {expected[key]}, got {actual[key]}",
        )
    return {
        "verified_at": utc_now(),
        "path": path.name,
        "mode": f"{mode:#05o}",
        "read_only": True,
        **actual,
    }


def run_text(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=tool_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise D0Failure(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr}"
        )
    return completed


def run_bytes(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=tool_environment(),
        capture_output=True,
        text=False,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="backslashreplace")
        raise D0Failure(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{stderr}"
        )
    return completed


def expected_driver_fields(engine: str) -> frozenset[str]:
    require(engine in {"infinity", "faiss"}, f"Unknown engine: {engine}")
    fields = set(COMMON_DRIVER_FIELDS)
    fields.update(AUDIT_DRIVER_FIELDS)
    fields.update(f"{engine}_{suffix}" for suffix in ENGINE_RESULT_SUFFIXES)
    for k, ef in RECALL_POINTS:
        fields.add(f"{engine}_recall_at_{k}_ef_{ef}")
        fields.add(f"{engine}_returned_ids_ef_{ef}_k_{k}")
        fields.add(f"{engine}_returned_distances_sha256_ef_{ef}_k_{k}")
    if engine == "infinity":
        fields.update(INFINITY_DRIVER_FIELDS)
    return frozenset(fields)


def expected_infinity_tasks(participants: int) -> int:
    require(participants > 0, "Infinity participant count must be positive")
    average_bucket_size = (VECTORS - 1) // participants + 1
    bucket_size = max(1024, average_bucket_size)
    return (VECTORS - 1) // bucket_size + 1


def parse_driver_output(text: str, *, engine: str) -> dict[str, str]:
    expected = expected_driver_fields(engine)
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(line != "", f"Driver output line {line_number} is blank")
        require(
            line == line.strip(),
            f"Driver output line {line_number} has surrounding whitespace",
        )
        require(
            line.count("=") == 1,
            f"Driver output line {line_number} is not one key=value field",
        )
        name, value = line.split("=", 1)
        require(
            FIELD_NAME_PATTERN.fullmatch(name) is not None,
            f"Driver output line {line_number} has an invalid key: {name!r}",
        )
        require(value != "", f"Driver field {name} has an empty value")
        require(
            value == value.strip(),
            f"Driver field {name} has surrounding whitespace",
        )
        require(name not in values, f"Driver output contains duplicate field: {name}")
        values[name] = value

    unknown = sorted(set(values) - expected)
    missing = sorted(expected - set(values))
    require(not unknown, f"Driver output contains unknown fields: {unknown}")
    require(not missing, f"Driver output is missing fields: {missing}")
    return values


def _attestation_field(
    lines: list[str],
    cursor: int,
    expected_name: str,
) -> tuple[str, int]:
    require(
        cursor < len(lines),
        f"Attestation transcript is missing field {expected_name}",
    )
    line = lines[cursor]
    require(
        line.count("=") == 1,
        f"Attestation transcript line {cursor + 1} is not one key=value field",
    )
    name, value = line.split("=", 1)
    require(
        name == expected_name,
        f"Attestation transcript field {cursor + 1} is {name}, "
        f"expected {expected_name}",
    )
    require(
        value != "" and value == value.strip(),
        f"Attestation field {expected_name} has an invalid value",
    )
    return value, cursor + 1


def _attestation_unsigned(value: str, name: str) -> int:
    require(
        UNSIGNED_DECIMAL_PATTERN.fullmatch(value) is not None,
        f"Attestation field {name} is not canonical unsigned decimal",
    )
    return int(value)


def _parse_attestation_snapshot(
    lines: list[str],
    cursor: int,
    phase: str,
) -> tuple[dict[str, Any], int]:
    prefix = f"attestation_{phase}_"
    dynamic_cdhash, cursor = _attestation_field(
        lines,
        cursor,
        prefix + "dynamic_cdhash",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", dynamic_cdhash) is not None,
        f"Attestation {phase} dynamic CDHash is invalid",
    )

    identities: dict[str, int] = {}
    for identity in ("held_executable", "mapped_executable"):
        for suffix in ("dev", "ino", "size"):
            name = prefix + identity + "_" + suffix
            value, cursor = _attestation_field(lines, cursor, name)
            parsed = _attestation_unsigned(value, name)
            require(parsed > 0, f"Attestation field {name} must be positive")
            identities[f"{identity}_{suffix}"] = parsed

    vnode_name = prefix + "mapped_vnode_matches_held_fd"
    vnode_value, cursor = _attestation_field(lines, cursor, vnode_name)
    require(vnode_value == "1", f"Attestation field {vnode_name} is not 1")

    count_name = prefix + "image_count"
    count_value, cursor = _attestation_field(lines, cursor, count_name)
    image_count = _attestation_unsigned(count_value, count_name)
    require(
        0 < image_count <= ATTESTATION_MAXIMUM_IMAGE_COUNT,
        f"Attestation {phase} image count is out of range",
    )

    images: list[dict[str, Any]] = []
    for index in range(image_count):
        image_prefix = prefix + f"image_{index}_"
        path_hex, cursor = _attestation_field(
            lines,
            cursor,
            image_prefix + "path_hex",
        )
        require(
            len(path_hex) % 2 == 0
            and 0 < len(path_hex) <= 2 * ATTESTATION_MAXIMUM_PATH_BYTES
            and re.fullmatch(r"[0-9a-f]+", path_hex) is not None,
            f"Attestation {phase} image {index} path hex is invalid",
        )
        uuid, cursor = _attestation_field(
            lines,
            cursor,
            image_prefix + "uuid",
        )
        require(
            re.fullmatch(r"[0-9a-f]{32}", uuid) is not None,
            f"Attestation {phase} image {index} UUID is invalid",
        )
        shared_cache, cursor = _attestation_field(
            lines,
            cursor,
            image_prefix + "shared_cache",
        )
        require(
            shared_cache in {"0", "1"},
            f"Attestation {phase} image {index} shared-cache flag is invalid",
        )
        images.append(
            {
                "path_hex": path_hex,
                "uuid": uuid,
                "shared_cache": shared_cache == "1",
            }
        )
    require(
        len({(image["path_hex"], image["uuid"]) for image in images})
        == len(images),
        f"Attestation {phase} image set contains duplicates",
    )

    held_identity = tuple(
        identities[f"held_executable_{suffix}"]
        for suffix in ("dev", "ino", "size")
    )
    mapped_identity = tuple(
        identities[f"mapped_executable_{suffix}"]
        for suffix in ("dev", "ino", "size")
    )
    require(
        held_identity == mapped_identity,
        f"Attestation {phase} mapped executable differs from the held file",
    )
    return (
        {
            "dynamic_cdhash": dynamic_cdhash,
            "held_executable": {
                "device": held_identity[0],
                "inode": held_identity[1],
                "size": held_identity[2],
            },
            "mapped_executable": {
                "device": mapped_identity[0],
                "inode": mapped_identity[1],
                "size": mapped_identity[2],
            },
            "mapped_vnode_matches_held_fd": True,
            "image_count": image_count,
            "images": images,
        },
        cursor,
    )


def split_attested_driver_output(text: str) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    attestation_lines: list[str] = []
    driver_lines: list[str] = []
    driver_started = False
    for line_number, line in enumerate(lines, start=1):
        require(line != "", f"Driver output line {line_number} is blank")
        if line.startswith("attestation_"):
            require(
                not driver_started,
                "Attestation output appears after driver result output",
            )
            attestation_lines.append(line)
        else:
            driver_started = True
            driver_lines.append(line)

    require(attestation_lines, "Driver output lacks an attestation transcript")
    require(driver_lines, "Driver output lacks result fields")
    cursor = 0
    schema, cursor = _attestation_field(
        attestation_lines,
        cursor,
        "attestation_schema",
    )
    require(
        schema == str(ATTESTATION_SCHEMA_VERSION),
        "Attestation schema version differs",
    )
    before, cursor = _parse_attestation_snapshot(
        attestation_lines,
        cursor,
        "before",
    )
    before_barrier, cursor = _attestation_field(
        attestation_lines,
        cursor,
        "attestation_barrier",
    )
    require(
        before_barrier == ATTESTATION_BARRIER_PHASES[0],
        "Attestation before-work barrier is missing or out of order",
    )
    for expected in ATTESTATION_BARRIER_PHASES[1:3]:
        barrier, cursor = _attestation_field(
            attestation_lines,
            cursor,
            "attestation_barrier",
        )
        require(
            barrier == expected,
            f"Attestation {expected} barrier is missing or out of order",
        )
    after, cursor = _parse_attestation_snapshot(
        attestation_lines,
        cursor,
        "after",
    )

    counters: dict[str, int | bool] = {}
    for suffix in (
        "initial_image_adds",
        "later_image_adds",
        "image_removes",
    ):
        name = "attestation_" + suffix
        value, cursor = _attestation_field(attestation_lines, cursor, name)
        counters[suffix] = _attestation_unsigned(value, name)
    unchanged, cursor = _attestation_field(
        attestation_lines,
        cursor,
        "attestation_image_set_unchanged",
    )
    require(unchanged == "1", "Attestation image-set-unchanged flag is not 1")
    counters["image_set_unchanged"] = True

    after_barrier, cursor = _attestation_field(
        attestation_lines,
        cursor,
        "attestation_barrier",
    )
    require(
        after_barrier == ATTESTATION_BARRIER_PHASES[3],
        "Attestation after-work barrier is missing or out of order",
    )
    status, cursor = _attestation_field(
        attestation_lines,
        cursor,
        "attestation_status",
    )
    require(status == "PASS", "Attestation status is not PASS")
    require(
        cursor == len(attestation_lines),
        "Attestation transcript contains trailing fields",
    )
    require(
        before == after,
        "Attestation before/after snapshots differ",
    )
    require(
        counters["initial_image_adds"] == before["image_count"],
        "Attestation initial image-add count differs from the image count",
    )
    require(
        counters["later_image_adds"] == 0
        and counters["image_removes"] == 0,
        "Attestation observed a dynamic image-set mutation",
    )

    driver_text = "\n".join(driver_lines) + "\n"
    return (
        driver_text,
        {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "barriers": list(ATTESTATION_BARRIER_PHASES),
            "before": before,
            "after": after,
            **counters,
            "status": "PASS",
        },
    )


def parse_unsigned_field(fields: dict[str, str], name: str) -> int:
    value = fields[name]
    require(
        UNSIGNED_DECIMAL_PATTERN.fullmatch(value) is not None,
        f"Driver field {name} is not canonical unsigned decimal: {value!r}",
    )
    return int(value)


def parse_uint64_field(fields: dict[str, str], name: str) -> int:
    value = parse_unsigned_field(fields, name)
    require(
        value <= FNV1A64_MASK,
        f"Driver field {name} does not fit in uint64",
    )
    return value


def parse_finite_field(fields: dict[str, str], name: str) -> float:
    value = fields[name]
    require(
        FINITE_DECIMAL_PATTERN.fullmatch(value) is not None,
        f"Driver field {name} is not a decimal number: {value!r}",
    )
    parsed = float(value)
    require(math.isfinite(parsed), f"Driver field {name} is not finite")
    return parsed


def parse_unsigned_csv(
    value: str,
    *,
    name: str,
    expected_count: int,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[int]:
    parts = value.split(",")
    require(
        len(parts) == expected_count,
        f"Driver field {name} has {len(parts)} entries, expected {expected_count}",
    )
    result: list[int] = []
    for part in parts:
        require(
            UNSIGNED_DECIMAL_PATTERN.fullmatch(part) is not None,
            f"Driver field {name} contains a noncanonical integer",
        )
        parsed = int(part)
        require(parsed >= minimum, f"Driver field {name} contains a small integer")
        if maximum is not None:
            require(
                parsed <= maximum,
                f"Driver field {name} contains an out-of-range integer",
            )
        result.append(parsed)
    return result


def parse_level_histogram(value: str, *, max_level: int) -> list[int]:
    entries = value.split(",")
    require(
        len(entries) == max_level + 1,
        "Driver graph level histogram has the wrong layer count",
    )
    counts: list[int] = []
    for expected_level, entry in enumerate(entries):
        parts = entry.split(":")
        require(
            len(parts) == 2
            and parts[0] == str(expected_level)
            and UNSIGNED_DECIMAL_PATTERN.fullmatch(parts[1]) is not None,
            "Driver graph level histogram is not canonical",
        )
        counts.append(int(parts[1]))
    return counts


def parse_degree_histograms(
    value: str,
    *,
    max_level: int,
    level0_capacity: int,
    upper_capacity: int,
) -> list[list[int]]:
    layers = value.split(";")
    require(
        len(layers) == max_level + 1,
        "Driver graph degree histograms have the wrong layer count",
    )
    histograms: list[list[int]] = []
    for layer, encoded in enumerate(layers):
        prefix = f"L{layer}:"
        require(
            encoded.startswith(prefix),
            "Driver graph degree histogram has a noncanonical layer prefix",
        )
        capacity = level0_capacity if layer == 0 else upper_capacity
        counts = parse_unsigned_csv(
            encoded[len(prefix) :],
            name=f"graph degree histogram layer {layer}",
            expected_count=capacity + 1,
        )
        histograms.append(counts)
    return histograms


def parse_driver_result(
    text: str,
    *,
    engine: str,
    participants: int,
    dataset: dict[str, Any],
    heldout: dict[str, Any] | None = None,
) -> dict[str, str]:
    fields = parse_driver_output(text, engine=engine)
    require(fields["status"] == "PASS", "Driver status is not PASS")
    require(fields["scope"] == D0_SCOPE, "Driver scope mismatch")
    require(fields["order"] == f"{engine}-only", "Driver engine-only order mismatch")

    expected_integers = {
        "data_fnv1a64": int(dataset["fnv1a64"]),
        "data_bytes": int(dataset["bytes"]),
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
        actual = parse_unsigned_field(fields, name)
        require(
            actual == expected, f"Driver field {name} is {actual}, expected {expected}"
        )
    require(
        fields["data_sha256"] == dataset["sha256"]
        and re.fullmatch(r"[0-9a-f]{64}", fields["data_sha256"]) is not None,
        "Driver data_sha256 differs from the frozen dataset",
    )
    for name in ("campaign_nonce", "binary_sha256", "dataset_sha256"):
        require(
            re.fullmatch(r"[0-9a-f]{64}", fields[name]) is not None,
            f"Driver field {name} is not canonical lowercase 32-byte hex",
        )
    schedule_sequence = parse_unsigned_field(fields, "schedule_sequence")
    require(
        schedule_sequence <= (1 << 64) - 1,
        "Driver schedule_sequence does not fit in uint64",
    )
    role_id = parse_unsigned_field(fields, "role_id")
    engine_identifier = AUDIT_ENGINE_IDENTIFIERS[engine]
    require(
        AUDIT_ROLE_ENGINE_IDENTIFIERS.get(role_id) == engine_identifier,
        "Driver role_id is incompatible with the scheduled engine",
    )
    require(
        fields["dataset_sha256"] == dataset["sha256"],
        "Driver dataset_sha256 differs from the frozen dataset",
    )
    require(
        fields["query_percentile_method"] == QUERY_PERCENTILE_METHOD,
        "Driver query percentile method differs from the frozen protocol",
    )
    require(
        fields["query_transaction_definition"] == QUERY_TRANSACTION_DEFINITION,
        "Driver query transaction definition differs from the frozen protocol",
    )
    require(
        parse_finite_field(fields, "query_recall_floor") == QUERY_RECALL_FLOOR,
        "Driver query recall floor differs from the frozen protocol",
    )
    require(
        parse_finite_field(fields, "query_maximum_absolute_recall_gap")
        == QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP,
        "Driver query recall-gap limit differs from the frozen protocol",
    )

    cold_build_ns = parse_unsigned_field(fields, f"{engine}_cold_build_ns")
    insert_call_ns = parse_unsigned_field(fields, f"{engine}_insert_call_ns")
    require(cold_build_ns > 0, "Cold-build duration is not positive")
    require(insert_call_ns > 0, "Insert-call duration is not positive")
    require(
        insert_call_ns <= cold_build_ns,
        "Insert-call duration exceeds cold-build duration",
    )
    throughput_wall_ns = parse_uint64_field(
        fields, f"{engine}_query_throughput_wall_ns"
    )
    require(throughput_wall_ns > 0, "Query throughput wall duration is not positive")
    latency_validated_checksum = parse_uint64_field(
        fields, f"{engine}_query_latency_validated_result_checksum"
    )
    throughput_validated_checksum = parse_uint64_field(
        fields, f"{engine}_query_throughput_validated_result_checksum"
    )
    result_checksum = parse_uint64_field(
        fields, f"{engine}_query_result_checksum"
    )
    per_query_checksums = parse_unsigned_csv(
        fields[f"{engine}_query_per_query_checksums"],
        name=f"{engine}_query_per_query_checksums",
        expected_count=HELDOUT_QUERY_COUNT,
        maximum=FNV1A64_MASK,
    )
    require(
        query_benchmark_checksum(per_query_checksums)
        == latency_validated_checksum
        == throughput_validated_checksum
        == result_checksum,
        "Driver query benchmark phase or aggregate checksum differs",
    )
    for name in EXECUTION_WITNESS_FIELDS:
        require(
            parse_unsigned_field(fields, f"{engine}_{name}") in (0, 1),
            f"Driver execution witness {name} is not binary",
        )

    recall = parse_finite_field(fields, f"{engine}_self_recall_at_1")
    parse_finite_field(fields, f"{engine}_distance_checksum")
    require(0.0 <= recall <= 1.0, "Self recall is outside [0, 1]")
    require(recall >= 0.95, "Self recall is below the D0 smoke threshold")

    sidecar_bytes = parse_unsigned_field(fields, "audit_sidecar_bytes")
    require(
        0 < sidecar_bytes <= MAXIMUM_AUDIT_SIDECAR_BYTES,
        "Audit sidecar byte count is invalid",
    )
    for name in (
        "audit_sidecar_sha256",
        "query_timed_corpus_sha256",
        "heldout_queries_sha256",
        "heldout_truth_sha256",
        f"{engine}_graph_levels_sha256",
        f"{engine}_graph_sha256",
    ):
        require(
            re.fullmatch(r"[0-9a-f]{64}", fields[name]) is not None,
            f"Driver field {name} is not a lowercase SHA-256 digest",
        )
    require(
        fields["query_timed_corpus_sha256"] == fields["heldout_queries_sha256"],
        "Driver timed query corpus differs from its held-out query identity",
    )
    if heldout is not None:
        require(
            fields["query_timed_corpus_sha256"] == heldout["queries_sha256"]
            and fields["heldout_queries_sha256"] == heldout["queries_sha256"]
            and fields["heldout_truth_sha256"] == heldout["truth_sha256"],
            "Driver held-out hashes differ from heldout-v1.json",
        )

    tie_counts_10 = parse_unsigned_csv(
        fields["heldout_truth_tie_counts_at_10"],
        name="heldout_truth_tie_counts_at_10",
        expected_count=HELDOUT_QUERY_COUNT,
        minimum=10,
        maximum=VECTORS,
    )
    tie_counts_100 = parse_unsigned_csv(
        fields["heldout_truth_tie_counts_at_100"],
        name="heldout_truth_tie_counts_at_100",
        expected_count=HELDOUT_QUERY_COUNT,
        minimum=100,
        maximum=VECTORS,
    )
    if heldout is not None:
        require(
            tie_counts_10 == heldout["tie_counts_at_10"]
            and tie_counts_100 == heldout["tie_counts_at_100"],
            "Driver held-out tie counts differ from heldout-v1.json",
        )

    max_level = parse_unsigned_field(fields, f"{engine}_graph_max_level")
    entry_point = parse_unsigned_field(fields, f"{engine}_graph_entry_point")
    directed_edges = parse_unsigned_field(
        fields, f"{engine}_graph_directed_edges"
    )
    level0_edges = parse_unsigned_field(
        fields, f"{engine}_graph_level0_directed_edges"
    )
    require(max_level < VECTORS, "Driver graph maximum level is invalid")
    require(entry_point < VECTORS, "Driver graph entry point is invalid")
    require(
        0 < level0_edges <= directed_edges,
        "Driver graph edge counts are invalid",
    )
    level_histogram = parse_level_histogram(
        fields[f"{engine}_graph_level_histogram"],
        max_level=max_level,
    )
    require(
        sum(level_histogram) == VECTORS and level_histogram[max_level] > 0,
        "Driver graph level histogram does not cover the dataset",
    )
    degree_histograms = parse_degree_histograms(
        fields[f"{engine}_graph_degree_histograms"],
        max_level=max_level,
        level0_capacity=2 * M,
        upper_capacity=M,
    )
    require(
        sum(degree_histograms[0]) == VECTORS,
        "Driver level-zero degree histogram does not cover the graph",
    )

    for k, ef in RECALL_POINTS:
        recall_name = f"{engine}_recall_at_{k}_ef_{ef}"
        point_recall = parse_finite_field(fields, recall_name)
        require(
            0.0 <= point_recall <= 1.0,
            f"Driver field {recall_name} is outside [0, 1]",
        )
        ids_name = f"{engine}_returned_ids_ef_{ef}_k_{k}"
        returned_ids = parse_unsigned_csv(
            fields[ids_name],
            name=ids_name,
            expected_count=HELDOUT_QUERY_COUNT * k,
            maximum=VECTORS - 1,
        )
        for query_index in range(HELDOUT_QUERY_COUNT):
            begin = query_index * k
            query_ids = returned_ids[begin : begin + k]
            require(
                len(set(query_ids)) == k,
                f"Driver field {ids_name} contains duplicate query results",
            )
        distance_hash_name = (
            f"{engine}_returned_distances_sha256_ef_{ef}_k_{k}"
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", fields[distance_hash_name]) is not None,
            f"Driver field {distance_hash_name} is not a lowercase SHA-256",
        )
    return fields


_HELDOUT_TRUTH_CACHE: dict[str, dict[str, Any]] = {}


def splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & FNV1A64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & FNV1A64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & FNV1A64_MASK
    return state, value ^ (value >> 31)


def heldout_truth(
    dataset_path: Path,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    cache_key = str(dataset["sha256"])
    cached = _HELDOUT_TRUTH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    raw = dataset_path.read_bytes()
    require(
        len(raw) == VECTORS * DIMENSIONS * FLOAT32_BYTES,
        "Cannot construct held-out truth from a dataset with the wrong size",
    )
    require(
        hashlib.sha256(raw).hexdigest() == dataset["sha256"],
        "Cannot construct held-out truth from a changed dataset",
    )
    base = array.array("f")
    base.frombytes(raw)
    if sys.byteorder != "little":
        base.byteswap()
    require(
        len(base) == VECTORS * DIMENSIONS,
        "Held-out truth decoded the wrong float count",
    )

    state = HELDOUT_QUERY_SEED
    queries: list[array.array[float]] = []
    query_digest = hashlib.sha256()
    query_digest.update(b"infinity-hnsw-d0-queries-v1")
    query_digest.update(struct.pack("<Q", HELDOUT_QUERY_COUNT))
    query_digest.update(struct.pack("<Q", DIMENSIONS))
    for _ in range(HELDOUT_QUERY_COUNT):
        query = array.array("f")
        for _ in range(DIMENSIONS):
            state, mixed = splitmix64(state)
            numerator = mixed >> 40
            value = struct.unpack(
                "<f",
                struct.pack("<f", numerator * (1.0 / 16_777_216.0)),
            )[0]
            query.append(value)
            query_digest.update(struct.pack("<f", value))
        queries.append(query)

    truth_digest = hashlib.sha256()
    truth_digest.update(b"infinity-hnsw-d0-truth-v1")
    truth_digest.update(struct.pack("<Q", HELDOUT_QUERY_COUNT))
    truth_digest.update(struct.pack("<Q", VECTORS))
    truth_digest.update(struct.pack("<Q", DIMENSIONS))
    all_distances: list[array.array[float]] = []
    cutoffs_at_10: list[float] = []
    cutoffs_at_100: list[float] = []
    tie_counts_at_10: list[int] = []
    tie_counts_at_100: list[int] = []
    top_100: list[list[int]] = []
    for query_index, query in enumerate(queries):
        distances = array.array("d", [0.0]) * VECTORS
        matches_base_row = False
        for vertex in range(VECTORS):
            offset = vertex * DIMENSIONS
            distance = 0.0
            equal = True
            for component in range(DIMENSIONS):
                difference = float(query[component]) - float(base[offset + component])
                distance += difference * difference
                equal = equal and query[component] == base[offset + component]
            distances[vertex] = distance
            matches_base_row = matches_base_row or equal
        require(
            not matches_base_row,
            "Generated held-out query duplicates an indexed vector",
        )
        order = sorted(range(VECTORS), key=lambda vertex: (distances[vertex], vertex))
        cutoff10 = distances[order[9]]
        cutoff100 = distances[order[99]]
        ties10 = sum(distance <= cutoff10 for distance in distances)
        ties100 = sum(distance <= cutoff100 for distance in distances)
        truth_digest.update(struct.pack("<Q", query_index))
        truth_digest.update(struct.pack("<Q", ties10))
        truth_digest.update(struct.pack("<Q", ties100))
        for vertex in order[:100]:
            truth_digest.update(struct.pack("<I", vertex))
        all_distances.append(distances)
        cutoffs_at_10.append(cutoff10)
        cutoffs_at_100.append(cutoff100)
        tie_counts_at_10.append(ties10)
        tie_counts_at_100.append(ties100)
        top_100.append(order[:100])

    result = {
        "schema_version": HELDOUT_MANIFEST_SCHEMA_VERSION,
        "query_count": HELDOUT_QUERY_COUNT,
        "dimension": DIMENSIONS,
        "seed": HELDOUT_QUERY_SEED,
        "generator": "splitmix64-high24-float32",
        "queries_sha256": query_digest.hexdigest(),
        "truth_sha256": truth_digest.hexdigest(),
        "tie_counts_at_10": tie_counts_at_10,
        "tie_counts_at_100": tie_counts_at_100,
        "recall_points": [{"k": k, "ef": ef} for k, ef in RECALL_POINTS],
        "_queries": queries,
        "_distances": all_distances,
        "_cutoffs_at_10": cutoffs_at_10,
        "_cutoffs_at_100": cutoffs_at_100,
        "_top_100": top_100,
    }
    _HELDOUT_TRUTH_CACHE[cache_key] = result
    return result


def heldout_manifest(truth: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in truth.items() if not key.startswith("_")}


class AuditSidecarReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read_bytes(self, size: int, context: str) -> bytes:
        require(size >= 0, f"{context} requested a negative byte count")
        end = self.offset + size
        require(end <= len(self.data), f"Audit sidecar is truncated at {context}")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def unpack(self, format_text: str, context: str) -> tuple[Any, ...]:
        parser = struct.Struct("<" + format_text)
        return parser.unpack(self.read_bytes(parser.size, context))


def audit_sidecar_path(prefix: str) -> str:
    return f"{prefix}{AUDIT_SIDECAR_SUFFIX}"


def read_audit_sidecar(path: Path) -> bytes:
    context = f"Audit sidecar {path.name}"
    try:
        initial = path.lstat()
    except OSError as error:
        raise D0Failure(f"{context} is missing or unreadable: {error}") from error
    require(not stat.S_ISLNK(initial.st_mode), f"{context} must not be a symlink")
    require(stat.S_ISREG(initial.st_mode), f"{context} must be a regular file")
    require(
        0 < initial.st_size <= MAXIMUM_AUDIT_SIDECAR_BYTES,
        f"{context} size is outside the supported range",
    )

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise D0Failure(f"Could not open {context}: {error}") from error

    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{context} is not a regular file")
        require(
            (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
                initial.st_mtime_ns,
                initial.st_ctime_ns,
            )
            == (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ),
            f"{context} path changed before it was read",
        )
        chunks: list[bytes] = []
        total = 0
        while total <= MAXIMUM_AUDIT_SIDECAR_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, MAXIMUM_AUDIT_SIDECAR_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise D0Failure(f"Could not read {context}: {error}") from error
    finally:
        os.close(descriptor)

    require(
        0 < len(data) <= MAXIMUM_AUDIT_SIDECAR_BYTES,
        f"{context} size is outside the supported range",
    )
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{context} changed while it was being read",
    )
    try:
        final = path.lstat()
    except OSError as error:
        raise D0Failure(f"{context} disappeared after reading: {error}") from error
    require(
        not stat.S_ISLNK(final.st_mode)
        and stat.S_ISREG(final.st_mode)
        and (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{context} path changed while it was being read",
    )
    require(
        len(data) == after.st_size,
        f"{context} byte count changed while it was being read",
    )
    return data


def parse_audit_sidecar(
    path: Path,
    *,
    engine: str,
    fields: dict[str, str],
    dataset: dict[str, Any],
    truth: dict[str, Any],
    expected_campaign_nonce: str,
    expected_schedule_sequence: int,
    expected_role_id: int,
    expected_binary_sha256: str,
    expected_dataset_sha256: str,
    data: bytes | None = None,
) -> dict[str, Any]:
    require(engine in {"infinity", "faiss"}, f"Unknown sidecar engine {engine}")
    require(
        re.fullmatch(r"[0-9a-f]{64}", expected_campaign_nonce) is not None,
        "Expected campaign nonce is not canonical lowercase 32-byte hex",
    )
    require(
        type(expected_schedule_sequence) is int
        and 0 <= expected_schedule_sequence <= (1 << 64) - 1,
        "Expected schedule sequence does not fit in uint64",
    )
    require(
        type(expected_role_id) is int
        and AUDIT_ROLE_ENGINE_IDENTIFIERS.get(expected_role_id)
        == AUDIT_ENGINE_IDENTIFIERS[engine],
        "Expected role ID is incompatible with the scheduled engine",
    )
    for name, value in (
        ("binary SHA-256", expected_binary_sha256),
        ("dataset SHA-256", expected_dataset_sha256),
    ):
        require(
            re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Expected {name} is not canonical lowercase hex",
        )
    require(
        expected_dataset_sha256 == dataset["sha256"],
        "Expected dataset SHA-256 differs from the frozen dataset",
    )
    if data is None:
        data = read_audit_sidecar(path)
    require(
        0 < len(data) <= MAXIMUM_AUDIT_SIDECAR_BYTES,
        "Audit sidecar size is outside the supported range",
    )
    digest = hashlib.sha256(data).hexdigest()
    require(
        len(data) == parse_unsigned_field(fields, "audit_sidecar_bytes"),
        "Audit sidecar size differs from driver stdout",
    )
    require(
        digest == fields["audit_sidecar_sha256"],
        "Audit sidecar SHA-256 differs from driver stdout",
    )

    reader = AuditSidecarReader(data)
    require(
        reader.read_bytes(len(AUDIT_SIDECAR_MAGIC), "magic")
        == AUDIT_SIDECAR_MAGIC,
        "Audit sidecar magic is invalid",
    )
    schema, engine_number = reader.unpack("II", "schema and engine")
    campaign_nonce = reader.read_bytes(32, "campaign nonce").hex()
    schedule_sequence, role_id, execution_witness_bits = reader.unpack(
        "QII", "campaign sequence and role"
    )
    binary_sha256 = reader.read_bytes(32, "binary SHA-256").hex()
    dataset_sha256 = reader.read_bytes(32, "dataset SHA-256").hex()
    require(
        schema == AUDIT_SIDECAR_SCHEMA_VERSION,
        "Audit sidecar schema is unsupported",
    )
    require(
        engine_number == AUDIT_ENGINE_IDENTIFIERS[engine],
        "Audit sidecar engine differs from the scheduled engine",
    )
    require(
        AUDIT_ROLE_ENGINE_IDENTIFIERS.get(role_id) == engine_number,
        "Audit sidecar role is incompatible with its engine",
    )
    execution_witness = execution_witness_from_bits(execution_witness_bits)
    binding_values: tuple[tuple[str, str | int, str | int], ...] = (
        ("campaign_nonce", campaign_nonce, expected_campaign_nonce),
        ("schedule_sequence", schedule_sequence, expected_schedule_sequence),
        ("role_id", role_id, expected_role_id),
        ("binary_sha256", binary_sha256, expected_binary_sha256),
        ("dataset_sha256", dataset_sha256, expected_dataset_sha256),
    )
    for name, actual, expected in binding_values:
        require(
            actual == expected,
            f"Audit sidecar {name} differs from the expected campaign binding",
        )
        stdout_value: str | int
        if name in {"schedule_sequence", "role_id"}:
            stdout_value = parse_unsigned_field(fields, name)
        else:
            stdout_value = fields[name]
        require(
            stdout_value == actual,
            f"Driver field {name} differs from the raw audit sidecar",
        )
    require(
        dataset_sha256 == dataset["sha256"],
        "Audit sidecar dataset SHA-256 differs from the frozen dataset",
    )
    vector_count, dimension = reader.unpack("QQ", "workload dimensions")
    sidecar_m, ef_construction = reader.unpack("II", "HNSW parameters")
    max_level, entry_point = reader.unpack("ii", "graph metadata")
    level0_capacity, upper_capacity = reader.unpack("II", "graph capacities")
    query_count, query_seed = reader.unpack("QQ", "held-out query identity")
    search_count, reserved = reader.unpack("II", "search count")
    query_schema, query_unique_queries = reader.unpack(
        "II", "query benchmark schema and query count"
    )
    require(
        query_schema == QUERY_BENCHMARK_SCHEMA_VERSION,
        "Audit sidecar query benchmark schema is unsupported",
    )
    query_k, query_ef_search = reader.unpack("II", "query benchmark search")
    query_warmup_passes, query_measured_passes = reader.unpack(
        "II", "query benchmark passes"
    )
    query_latency_concurrency, query_throughput_concurrency = reader.unpack(
        "II", "query benchmark concurrency"
    )
    query_percentile_method, query_transaction_definition = reader.unpack(
        "II", "query benchmark method identifiers"
    )
    query_recall_floor_bits, query_recall_gap_bits = reader.unpack(
        "QQ", "query benchmark quality gates"
    )
    timed_query_corpus_sha256 = reader.read_bytes(
        32, "timed query corpus SHA-256"
    ).hex()
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
        and query_throughput_operations == QUERY_THROUGHPUT_OPERATIONS,
        "Audit sidecar query benchmark latency or requested operation count differs",
    )
    require(
        query_throughput_validated_operations == query_throughput_operations,
        "Audit sidecar query benchmark throughput validated count differs",
    )
    require(
        query_per_query_checksum_count == HELDOUT_QUERY_COUNT,
        "Audit sidecar query benchmark per-query checksum count differs",
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
    require(
        (vector_count, dimension, sidecar_m, ef_construction)
        == (VECTORS, DIMENSIONS, M, EF_CONSTRUCTION),
        "Audit sidecar workload parameters differ",
    )
    require(
        0 <= max_level < VECTORS and 0 <= entry_point < VECTORS,
        "Audit sidecar graph metadata is invalid",
    )
    require(
        (level0_capacity, upper_capacity) == (2 * M, M),
        "Audit sidecar graph capacities differ",
    )
    require(
        (query_count, query_seed, search_count, reserved)
        == (
            HELDOUT_QUERY_COUNT,
            HELDOUT_QUERY_SEED,
            len(RECALL_POINTS),
            0,
        ),
        "Audit sidecar held-out metadata differs",
    )
    query_recall_floor = struct.unpack(
        "<d", struct.pack("<Q", query_recall_floor_bits)
    )[0]
    query_recall_gap = struct.unpack("<d", struct.pack("<Q", query_recall_gap_bits))[0]
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
            query_latency_sample_count,
            query_latency_validated_operations,
            query_throughput_operations,
            query_throughput_validated_operations,
            query_per_query_checksum_count,
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
            QUERY_LATENCY_SAMPLE_COUNT,
            QUERY_LATENCY_SAMPLE_COUNT,
            QUERY_THROUGHPUT_OPERATIONS,
            QUERY_THROUGHPUT_OPERATIONS,
            HELDOUT_QUERY_COUNT,
        )
        and query_recall_floor == QUERY_RECALL_FLOOR
        and query_recall_gap == QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP,
        "Audit sidecar query benchmark protocol differs",
    )
    query_performance = query_performance_record(
        timed_query_corpus_sha256=timed_query_corpus_sha256,
        latency_samples_ns=query_latency_samples,
        latency_validated_operations=query_latency_validated_operations,
        latency_validated_result_checksum=(
            query_latency_validated_result_checksum
        ),
        throughput_operations=query_throughput_operations,
        throughput_validated_operations=query_throughput_validated_operations,
        throughput_wall_ns=query_throughput_wall_ns,
        throughput_validated_result_checksum=(
            query_throughput_validated_result_checksum
        ),
        result_checksum=query_result_checksum_value,
        per_query_checksums=query_per_query_checksums,
    )
    require(
        timed_query_corpus_sha256 == truth["queries_sha256"],
        "Audit sidecar timed query corpus differs from independent reconstruction",
    )
    require(
        fields["query_timed_corpus_sha256"] == timed_query_corpus_sha256,
        "Driver timed query corpus differs from the raw audit sidecar",
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
        f"{engine}_query_per_query_checksum_count": (
            query_per_query_checksum_count
        ),
    }
    for name, expected in query_stdout_integers.items():
        require(
            parse_unsigned_field(fields, name) == expected,
            f"Driver field {name} differs from the raw audit sidecar",
        )
    stdout_per_query_checksums = parse_unsigned_csv(
        fields[f"{engine}_query_per_query_checksums"],
        name=f"{engine}_query_per_query_checksums",
        expected_count=HELDOUT_QUERY_COUNT,
        maximum=FNV1A64_MASK,
    )
    require(
        stdout_per_query_checksums == query_per_query_checksums,
        "Driver per-query checksums differ from the raw audit sidecar",
    )
    for name, expected in execution_witness.items():
        require(
            parse_unsigned_field(fields, f"{engine}_{name}") == expected,
            f"Driver execution witness {name} differs from the raw audit sidecar",
        )

    levels: list[int] = []
    labels: list[int] = []
    layers_by_vertex: list[list[tuple[int, list[int]]]] = []
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
    graph_digest.update(struct.pack("<Q", VECTORS))
    graph_digest.update(struct.pack("<I", max_level))
    graph_digest.update(struct.pack("<I", entry_point))
    graph_digest.update(struct.pack("<Q", level0_capacity))
    graph_digest.update(struct.pack("<Q", upper_capacity))

    for vertex in range(VECTORS):
        (level,) = reader.unpack("i", f"vertex {vertex} level")
        (label,) = reader.unpack("q", f"vertex {vertex} label")
        require(
            0 <= level <= max_level,
            f"Audit sidecar vertex {vertex} has an invalid level",
        )
        require(
            0 <= label < VECTORS,
            f"Audit sidecar vertex {vertex} has an invalid label",
        )
        levels.append(level)
        labels.append(label)
        level_histogram[level] += 1
        levels_digest.update(struct.pack("<I", vertex))
        levels_digest.update(struct.pack("<I", level))
        graph_digest.update(struct.pack("<I", vertex))
        graph_digest.update(struct.pack("<I", level))
        graph_digest.update(struct.pack("<Q", label))
        vertex_layers: list[tuple[int, list[int]]] = []
        for layer in range(level + 1):
            capacity, degree = reader.unpack(
                "II", f"vertex {vertex} layer {layer} header"
            )
            expected_capacity = level0_capacity if layer == 0 else upper_capacity
            require(
                capacity == expected_capacity and degree <= capacity,
                f"Audit sidecar vertex {vertex} layer {layer} has invalid capacity or degree",
            )
            neighbors = list(
                reader.unpack(
                    "i" * degree,
                    f"vertex {vertex} layer {layer} neighbors",
                )
            )
            require(
                all(0 <= neighbor < VECTORS for neighbor in neighbors),
                f"Audit sidecar vertex {vertex} layer {layer} has an out-of-range edge",
            )
            require(
                vertex not in neighbors,
                f"Audit sidecar vertex {vertex} layer {layer} has a self edge",
            )
            require(
                len(set(neighbors)) == len(neighbors),
                f"Audit sidecar vertex {vertex} layer {layer} has a duplicate edge",
            )
            degree_histograms[layer][degree] += 1
            directed_edges += degree
            if layer == 0:
                level0_directed_edges += degree
            graph_digest.update(struct.pack("<I", layer))
            graph_digest.update(struct.pack("<Q", degree))
            for neighbor in neighbors:
                graph_digest.update(struct.pack("<I", neighbor))
            vertex_layers.append((capacity, neighbors))
        layers_by_vertex.append(vertex_layers)

    require(
        len(set(labels)) == VECTORS,
        "Audit sidecar graph labels are not a permutation",
    )
    require(
        max(levels) == max_level and levels[entry_point] == max_level,
        "Audit sidecar entry point or maximum level is inconsistent",
    )
    for vertex, vertex_layers in enumerate(layers_by_vertex):
        for layer, (_, neighbors) in enumerate(vertex_layers):
            require(
                all(levels[neighbor] >= layer for neighbor in neighbors),
                f"Audit sidecar vertex {vertex} has an ineligible upper-layer edge",
            )
    visited = {entry_point}
    pending = [entry_point]
    for vertex in pending:
        for neighbor in layers_by_vertex[vertex][0][1]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    require(
        len(visited) == VECTORS,
        "Audit sidecar level-zero graph is not directed-reachable",
    )

    encoded_level_histogram = ",".join(
        f"{level}:{count}" for level, count in enumerate(level_histogram)
    )
    encoded_degree_histograms = ";".join(
        f"L{layer}:" + ",".join(str(count) for count in histogram)
        for layer, histogram in enumerate(degree_histograms)
    )
    expected_graph_fields = {
        f"{engine}_graph_vertex_count": VECTORS,
        f"{engine}_graph_reachable_count": len(visited),
        f"{engine}_graph_directed_edges": directed_edges,
        f"{engine}_graph_level0_directed_edges": level0_directed_edges,
        f"{engine}_graph_max_level": max_level,
        f"{engine}_graph_entry_point": entry_point,
        f"{engine}_graph_level0_capacity": level0_capacity,
        f"{engine}_graph_upper_capacity": upper_capacity,
    }
    for name, expected in expected_graph_fields.items():
        require(
            parse_unsigned_field(fields, name) == expected,
            f"Driver field {name} differs from the raw audit sidecar",
        )
    require(
        fields[f"{engine}_graph_levels_sha256"] == levels_digest.hexdigest()
        and fields[f"{engine}_graph_sha256"] == graph_digest.hexdigest()
        and fields[f"{engine}_graph_level_histogram"]
        == encoded_level_histogram
        and fields[f"{engine}_graph_degree_histograms"]
        == encoded_degree_histograms,
        "Driver graph summaries differ from the raw audit sidecar",
    )

    point_records: list[dict[str, Any]] = []
    benchmark_query_checksums: list[int] = []
    for expected_k, expected_ef in RECALL_POINTS:
        k, ef = reader.unpack("II", f"search k={expected_k} ef={expected_ef}")
        (result_count,) = reader.unpack(
            "Q", f"search k={expected_k} ef={expected_ef} result count"
        )
        require(
            (k, ef, result_count)
            == (expected_k, expected_ef, HELDOUT_QUERY_COUNT * expected_k),
            "Audit sidecar search descriptor differs",
        )
        returned_ids: list[int] = []
        distance_digest = hashlib.sha256()
        hits = 0
        for query_index in range(HELDOUT_QUERY_COUNT):
            query_ids: list[int] = []
            query_distance_bits: list[int] = []
            previous_distance = -math.inf
            for rank in range(expected_k):
                label, distance_bits = reader.unpack(
                    "qI",
                    f"search k={expected_k} ef={expected_ef} "
                    f"query {query_index} rank {rank}",
                )
                require(
                    0 <= label < VECTORS,
                    "Audit sidecar search returned an out-of-range label",
                )
                distance = struct.unpack("<f", struct.pack("<I", distance_bits))[0]
                require(
                    math.isfinite(distance) and distance >= previous_distance,
                    "Audit sidecar search distances are nonfinite or unsorted",
                )
                exact_distance = truth["_distances"][query_index][label]
                require(
                    math.isclose(
                        distance,
                        exact_distance,
                        rel_tol=2e-5,
                        abs_tol=1e-4,
                    ),
                    "Audit sidecar search distance disagrees with the frozen dataset",
                )
                previous_distance = distance
                query_ids.append(label)
                query_distance_bits.append(distance_bits)
                returned_ids.append(label)
                distance_digest.update(struct.pack("<I", distance_bits))
                cutoff = (
                    truth["_cutoffs_at_10"][query_index]
                    if expected_k == 10
                    else truth["_cutoffs_at_100"][query_index]
                )
                hits += exact_distance <= cutoff
            require(
                len(set(query_ids)) == expected_k,
                "Audit sidecar search returned duplicate labels",
            )
            if (expected_k, expected_ef) == (QUERY_K, QUERY_EF_SEARCH):
                benchmark_query_checksums.append(
                    query_result_checksum(
                        query_index,
                        query_ids,
                        query_distance_bits,
                    )
                )
        possible_hits = HELDOUT_QUERY_COUNT * expected_k
        recall = hits / float(possible_hits)
        recall_name = f"{engine}_recall_at_{expected_k}_ef_{expected_ef}"
        ids_name = (
            f"{engine}_returned_ids_ef_{expected_ef}_k_{expected_k}"
        )
        distance_hash_name = (
            f"{engine}_returned_distances_sha256_ef_{expected_ef}_k_{expected_k}"
        )
        require(
            parse_finite_field(fields, recall_name) == recall,
            f"Driver field {recall_name} differs from raw returned IDs",
        )
        require(
            fields[ids_name] == ",".join(str(label) for label in returned_ids),
            f"Driver field {ids_name} differs from the raw audit sidecar",
        )
        require(
            fields[distance_hash_name] == distance_digest.hexdigest(),
            f"Driver field {distance_hash_name} differs from raw distances",
        )
        point_records.append(
            {
                "k": expected_k,
                "ef": expected_ef,
                "eligible_hits": hits,
                "possible_hits": possible_hits,
                "recall": recall,
            }
        )

    require(
        benchmark_query_checksums == query_per_query_checksums,
        "Audit sidecar per-query checksums differ from raw recall results",
    )
    replay_checksum = query_benchmark_checksum(benchmark_query_checksums)
    require(
        replay_checksum
        == query_latency_validated_result_checksum
        == query_throughput_validated_result_checksum
        == query_result_checksum_value,
        "Audit sidecar query benchmark phase checksum differs from recall results",
    )
    require(
        reader.offset == len(data),
        "Audit sidecar contains trailing bytes",
    )
    require(
        fields["query_timed_corpus_sha256"] == truth["queries_sha256"]
        and fields["heldout_queries_sha256"] == truth["queries_sha256"]
        and fields["heldout_truth_sha256"] == truth["truth_sha256"],
        "Audit sidecar run uses the wrong held-out query or truth identity",
    )
    return {
        "schema_version": AUDIT_SIDECAR_SCHEMA_VERSION,
        "query": query_performance,
        "execution_witness": execution_witness,
        "graph": {
            "valid": True,
            "vertex_count": VECTORS,
            "reachable_count": len(visited),
            "directed_edges": directed_edges,
            "level0_directed_edges": level0_directed_edges,
            "max_level": max_level,
            "entry_point": entry_point,
            "level0_capacity": level0_capacity,
            "upper_capacity": upper_capacity,
            "levels_sha256": levels_digest.hexdigest(),
            "graph_sha256": graph_digest.hexdigest(),
            "level_histogram": encoded_level_histogram,
            "degree_histograms": encoded_degree_histograms,
        },
        "recall": {
            "query_count": HELDOUT_QUERY_COUNT,
            "query_seed": HELDOUT_QUERY_SEED,
            "queries_sha256": truth["queries_sha256"],
            "truth_sha256": truth["truth_sha256"],
            "points": point_records,
        },
    }


def parse_idle_percentages(text: str) -> list[float]:
    pattern = re.compile(r"CPU usage:.*?([0-9.]+)% idle")
    percentages = [float(match.group(1)) for match in pattern.finditer(text)]
    require(
        all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in percentages),
        "top emitted an invalid CPU-idle percentage",
    )
    require(
        len(percentages) == IDLE_SAMPLE_COUNT,
        f"top emitted {len(percentages)} CPU-idle samples, "
        f"expected {IDLE_SAMPLE_COUNT}",
    )
    return percentages


def idle_policy(minimum_idle_percent: float) -> str:
    return (
        "default-95-percent"
        if minimum_idle_percent == IDLE_MINIMUM_PERCENT
        else "development-only-override"
    )


def validate_idle_minimum_percent(value: float) -> float:
    require(
        math.isfinite(value)
        and DEVELOPMENT_IDLE_MINIMUM_FLOOR_PERCENT <= value <= IDLE_MINIMUM_PERCENT,
        "Development idle minimum must be finite and between "
        f"{DEVELOPMENT_IDLE_MINIMUM_FLOOR_PERCENT:g} and "
        f"{IDLE_MINIMUM_PERCENT:g} percent",
    )
    return value


def wait_for_idle(
    *,
    cwd: Path,
    output_dir: Path,
    prefix: str,
    timeout_seconds: float,
    minimum_idle_percent: float,
) -> dict[str, Any]:
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    while True:
        attempt_number = len(attempts) + 1
        attempt_started = time.monotonic()
        completed = run_text(
            [
                "/usr/bin/top",
                "-l",
                str(IDLE_SAMPLE_COUNT),
                "-s",
                str(IDLE_SAMPLE_INTERVAL_SECONDS),
                "-n",
                "0",
            ],
            cwd=cwd,
            timeout=IDLE_WINDOW_SECONDS + 30,
        )
        attempt_elapsed = time.monotonic() - attempt_started
        raw_name = f"{prefix}.idle-attempt-{attempt_number:02d}.top.txt"
        (output_dir / raw_name).write_text(
            "[stdout]\n" + completed.stdout + "\n[stderr]\n" + completed.stderr,
            encoding="utf-8",
        )

        percentages = parse_idle_percentages(completed.stdout)
        require(
            re.search(
                r"CPU usage:.*?[0-9.]+% idle",
                completed.stderr,
            )
            is None,
            "top emitted CPU-idle samples on stderr",
        )
        covered_seconds = (len(percentages) - 1) * IDLE_SAMPLE_INTERVAL_SECONDS
        passed = (
            covered_seconds >= IDLE_WINDOW_SECONDS
            and min(percentages) >= minimum_idle_percent
        )
        attempt = {
            "attempt": attempt_number,
            "captured_at": utc_now(),
            "raw_path": raw_name,
            "idle_percentages": percentages,
            "minimum_idle_percent": min(percentages),
            "sample_count": len(percentages),
            "sample_interval_seconds": IDLE_SAMPLE_INTERVAL_SECONDS,
            "covered_seconds": covered_seconds,
            "command_elapsed_seconds": attempt_elapsed,
            "status": "pass" if passed else "fail",
        }
        attempts.append(attempt)
        if passed:
            return {
                "status": "pass",
                "accepted_at_monotonic_ns": time.monotonic_ns(),
                "accepted_at_unix_ns": time.time_ns(),
                "required_minimum_idle_percent": minimum_idle_percent,
                "required_window_seconds": IDLE_WINDOW_SECONDS,
                "attempts": attempts,
                "accepted_attempt": attempt_number,
            }
        if time.monotonic() - started >= timeout_seconds:
            return {
                "status": "fail",
                "required_minimum_idle_percent": minimum_idle_percent,
                "required_window_seconds": IDLE_WINDOW_SECONDS,
                "attempts": attempts,
                "accepted_attempt": None,
            }


def idle_record_passes(
    idle: dict[str, Any],
    *,
    expected_minimum_idle_percent: float = IDLE_MINIMUM_PERCENT,
) -> bool:
    attempts = idle.get("attempts", [])
    if not attempts:
        return False
    final = attempts[-1]
    return (
        idle.get("status") == "pass"
        and isinstance(idle.get("accepted_at_monotonic_ns"), int)
        and idle["accepted_at_monotonic_ns"] > 0
        and isinstance(idle.get("accepted_at_unix_ns"), int)
        and idle["accepted_at_unix_ns"] > 0
        and idle.get("required_minimum_idle_percent") == expected_minimum_idle_percent
        and idle.get("required_window_seconds") == IDLE_WINDOW_SECONDS
        and idle.get("accepted_attempt") == final.get("attempt")
        and final.get("status") == "pass"
        and final.get("minimum_idle_percent", -math.inf)
        >= expected_minimum_idle_percent
        and final.get("covered_seconds", 0) >= IDLE_WINDOW_SECONDS
    )


def schedule() -> list[dict[str, Any]]:
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
    validate_schedule(entries)
    return entries


def validate_schedule(entries: list[dict[str, Any]]) -> None:
    expected: list[dict[str, Any]] = []
    for pair, treatment, participants, engines in PAIR_PLAN:
        for engine in engines:
            expected.append(
                {
                    "sequence": len(expected),
                    "pair": pair,
                    "pair_order": "/".join(engines),
                    "treatment": treatment,
                    "engine": engine,
                    "participants": participants,
                }
            )
    require(entries == expected, "Schedule does not match the frozen D0 member order")

    measured_orders = [
        entry["pair_order"]
        for entry in entries
        if entry["treatment"] == "measured" and entry["engine"] == "infinity"
    ]
    require(
        measured_orders.count("infinity/faiss") == 3,
        "Measured schedule lacks three Infinity/FAISS pairs",
    )
    require(
        measured_orders.count("faiss/infinity") == 3,
        "Measured schedule lacks three FAISS/Infinity pairs",
    )


def validate_records_against_schedule(records: list[dict[str, Any]]) -> None:
    expected = schedule()
    require(len(records) == len(expected), "Run records do not cover the full schedule")
    schedule_keys = (
        "sequence",
        "pair",
        "pair_order",
        "treatment",
        "engine",
        "participants",
    )
    for index, (record, entry) in enumerate(zip(records, expected)):
        actual_member = {key: record.get(key) for key in schedule_keys}
        expected_member = {key: entry[key] for key in schedule_keys}
        require(
            actual_member == expected_member,
            f"Run record {index} does not match its scheduled member",
        )


def validate_run_chronology(records: list[dict[str, Any]]) -> None:
    previous_end: int | None = None
    for index, record in enumerate(records):
        started_ns = record.get("started_at_unix_ns")
        ended_ns = record.get("ended_at_unix_ns")
        require(
            type(started_ns) is int and started_ns > 0,
            f"Run {index} has an invalid start timestamp",
        )
        require(
            type(ended_ns) is int and ended_ns >= started_ns,
            f"Run {index} has an invalid end timestamp",
        )
        if previous_end is not None:
            require(
                started_ns >= previous_end,
                f"Run {index} overlaps or predates the previous member",
            )
        previous_end = ended_ns


def parse_swap_usage(text: str) -> dict[str, Any]:
    values: dict[str, int] = {}
    for name in ("total", "used", "free"):
        matches = re.findall(
            rf"\b{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT]?)\b",
            text,
            re.IGNORECASE,
        )
        require(len(matches) == 1, f"Expected one swap {name} value")
        amount = float(matches[0][0])
        suffix = matches[0][1].upper()
        multiplier = {
            "": 1,
            "K": 1024,
            "M": 1024**2,
            "G": 1024**3,
            "T": 1024**4,
        }[suffix]
        values[f"{name}_bytes"] = round(amount * multiplier)
    values["raw"] = text.strip()
    return values


def parse_battery_percent(text: str) -> int:
    matches = re.findall(r"\b([0-9]{1,3})%;", text)
    require(len(matches) == 1, "Expected one battery percentage")
    value = int(matches[0])
    require(0 <= value <= 100, f"Invalid battery percentage: {value}%")
    return value


def parse_power_source(text: str) -> str:
    ac_count = text.count("Now drawing from 'AC Power'")
    battery_count = text.count("Now drawing from 'Battery Power'")
    require(
        (ac_count, battery_count) in ((1, 0), (0, 1)),
        "Expected exactly one active power source",
    )
    return "ac" if ac_count == 1 else "battery"


def power_profile(text: str, source: str) -> str:
    heading = {"ac": "AC Power:", "battery": "Battery Power:"}.get(source)
    require(heading is not None, f"Unknown power source: {source}")
    profiles: dict[str, list[str]] = {}
    active_heading: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"AC Power:", "Battery Power:"}:
            require(stripped not in profiles, f"Duplicate {stripped} profile")
            active_heading = stripped
            profiles[active_heading] = []
        elif active_heading is not None:
            profiles[active_heading].append(line)
    require(
        set(profiles) == {"AC Power:", "Battery Power:"},
        "Power settings do not contain exactly one profile per source",
    )
    return "\n".join(profiles[heading])


def parse_thermal_nominal(text: str) -> bool:
    normalized = [
        line.strip().removeprefix("Note: ").strip()
        for line in text.splitlines()
        if line.strip()
    ]
    thermal_lines = [line for line in normalized if "thermal warning" in line.lower()]
    performance_lines = [
        line for line in normalized if "performance warning" in line.lower()
    ]
    return thermal_lines == [
        "No thermal warning level has been recorded"
    ] and performance_lines == ["No performance warning level has been recorded"]


def check_server_ports() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for port in SERVER_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            result = probe.connect_ex(("127.0.0.1", port))
        closed = result == errno.ECONNREFUSED
        checks.append(
            {
                "host": "127.0.0.1",
                "port": port,
                "connect_errno": result,
                "closed": closed,
            }
        )
        require(
            closed,
            f"Port {port} is open or could not be proven closed (errno {result})",
        )
    return checks


def capture_member_preflight(
    *,
    cwd: Path,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    battery = run_text(["/usr/bin/pmset", "-g", "batt"], cwd=cwd).stdout
    power = run_text(["/usr/bin/pmset", "-g", "custom"], cwd=cwd).stdout
    thermal = run_text(["/usr/bin/pmset", "-g", "therm"], cwd=cwd).stdout
    swap_text = run_text(["/usr/sbin/sysctl", "vm.swapusage"], cwd=cwd).stdout
    processes = run_text(
        [
            "/bin/ps",
            "-axo",
            "pid=,ppid=,%cpu=,%mem=,state=,etime=,command=",
        ],
        cwd=cwd,
    ).stdout

    host_name = f"{prefix}.host-preflight.txt"
    process_name = f"{prefix}.processes.txt"
    (output_dir / host_name).write_text(
        "[battery]\n"
        + battery
        + "\n[custom]\n"
        + power
        + "\n[thermal]\n"
        + thermal
        + "\n[swap]\n"
        + swap_text,
        encoding="utf-8",
    )
    (output_dir / process_name).write_text(processes, encoding="utf-8")

    power_source = parse_power_source(battery)
    battery_percent = parse_battery_percent(battery)
    require(
        BATTERY_MINIMUM_PERCENT <= battery_percent <= 100,
        "Battery charge is outside the required "
        f"{BATTERY_MINIMUM_PERCENT}-100% range: {battery_percent}%",
    )
    active_power_profile = power_profile(power, power_source)
    low_power_values = re.findall(
        r"(?m)^\s*lowpowermode\s+([01])\s*$",
        active_power_profile,
    )
    require(
        low_power_values == ["0"],
        f"Low Power Mode is enabled or unknown in the {power_source} profile",
    )
    require(
        parse_thermal_nominal(thermal),
        "Thermal or performance warning state is not nominal",
    )
    swap = parse_swap_usage(swap_text)
    ports = check_server_ports()
    return {
        "captured_at": utc_now(),
        "host_raw_path": host_name,
        "process_listing_path": process_name,
        "power_source": power_source,
        "power_policy": POWER_POLICY,
        "ac_power": power_source == "ac",
        "battery_percent": battery_percent,
        "battery_minimum_percent": BATTERY_MINIMUM_PERCENT,
        "low_power_mode": False,
        "thermal_nominal": True,
        "swap": swap,
        "ports": ports,
    }


def capture_member_postflight(
    *,
    cwd: Path,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    battery = run_text(["/usr/bin/pmset", "-g", "batt"], cwd=cwd).stdout
    power = run_text(["/usr/bin/pmset", "-g", "custom"], cwd=cwd).stdout
    thermal = run_text(["/usr/bin/pmset", "-g", "therm"], cwd=cwd).stdout
    swap_text = run_text(["/usr/sbin/sysctl", "vm.swapusage"], cwd=cwd).stdout
    path_name = f"{prefix}.host-postflight.txt"
    (output_dir / path_name).write_text(
        "[battery]\n"
        + battery
        + "\n[custom]\n"
        + power
        + "\n[thermal]\n"
        + thermal
        + "\n[swap]\n"
        + swap_text,
        encoding="utf-8",
    )

    power_source = parse_power_source(battery)
    battery_percent = parse_battery_percent(battery)
    require(
        BATTERY_MINIMUM_PERCENT <= battery_percent <= 100,
        "Battery charge is outside the required "
        f"{BATTERY_MINIMUM_PERCENT}-100% range: {battery_percent}%",
    )
    active_power_profile = power_profile(power, power_source)
    low_power_values = re.findall(
        r"(?m)^\s*lowpowermode\s+([01])\s*$",
        active_power_profile,
    )
    require(
        low_power_values == ["0"],
        f"Low Power Mode changed in the {power_source} profile",
    )
    require(
        parse_thermal_nominal(thermal),
        "Post-run thermal or performance warning state is not nominal",
    )
    return {
        "captured_at": utc_now(),
        "raw_path": path_name,
        "power_source": power_source,
        "battery_percent": battery_percent,
        "battery_minimum_percent": BATTERY_MINIMUM_PERCENT,
        "low_power_mode": False,
        "thermal_nominal": True,
        "swap": parse_swap_usage(swap_text),
    }


def parse_max_rss(stderr: str) -> int | None:
    matches = re.findall(
        r"^\s*(\d+)\s+maximum resident set size\s*$",
        stderr,
        re.MULTILINE,
    )
    return int(matches[0]) if len(matches) == 1 else None


def parse_process_swaps(stderr: str) -> int | None:
    matches = re.findall(r"^\s*(\d+)\s+swaps\s*$", stderr, re.MULTILINE)
    return int(matches[0]) if len(matches) == 1 else None


def binary_for_engine(args: argparse.Namespace, engine: str) -> Path:
    if engine == "infinity":
        return args.infinity_binary
    require(engine == "faiss", f"Unknown engine: {engine}")
    return args.faiss_binary


def benchmark_environment(participants: int) -> dict[str, str]:
    require(participants > 0, "Benchmark participant count must be positive")
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


def tool_environment() -> dict[str, str]:
    """Return the fixed environment used by provenance and build tools."""
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "TMPDIR": "/private/tmp",
    }


def unsafe_parent_environment_names(
    environment: Mapping[str, str],
) -> list[str]:
    return sorted(
        name
        for name in environment
        if (
            name in UNSAFE_PARENT_ENVIRONMENT_NAMES
            or name.startswith(UNSAFE_PARENT_ENVIRONMENT_PREFIXES)
        )
    )


def unsafe_parent_environment_policy_record(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": PARENT_ENVIRONMENT_POLICY_SCHEMA_VERSION,
        "exact_names": sorted(UNSAFE_PARENT_ENVIRONMENT_NAMES),
        "prefixes": list(UNSAFE_PARENT_ENVIRONMENT_PREFIXES),
        "observed_unsafe_names": unsafe_parent_environment_names(environment),
    }


def reject_unsafe_parent_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    parent = os.environ if environment is None else environment
    policy = unsafe_parent_environment_policy_record(parent)
    require(
        policy["observed_unsafe_names"] == [],
        "Campaign parent environment contains unsafe compiler, loader, "
        f"toolchain, Git, or Python injection variables: "
        f"{policy['observed_unsafe_names']}",
    )
    return policy


def verify_runtime_artifacts(expected_hashes: dict[str, str]) -> dict[str, Any]:
    require(bool(expected_hashes), "Runtime artifact set is empty")
    artifacts: list[dict[str, Any]] = []
    for raw_path, expected_hash in sorted(expected_hashes.items()):
        path = Path(raw_path)
        require(path.is_file(), f"Runtime artifact is missing: {path}")
        actual_hash = sha256(path)
        require(
            actual_hash == expected_hash,
            f"Runtime artifact changed: {path}",
        )
        artifacts.append(
            {
                "path": str(path),
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
            }
        )
    return {"verified_at": utc_now(), "artifacts": artifacts}


def _libproc() -> Any:
    global _LIBPROC
    require(
        platform.system() == "Darwin",
        "Process evidence requires the macOS libproc API",
    )
    if _LIBPROC is None:
        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        except OSError as error:
            raise D0Failure(f"Could not load macOS libproc: {error}") from error
        library.proc_listpgrppids.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_listpgrppids.restype = ctypes.c_int
        library.proc_listchildpids.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_listchildpids.restype = ctypes.c_int
        library.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_pidinfo.restype = ctypes.c_int
        library.proc_pidpath.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        library.proc_pidpath.restype = ctypes.c_int
        _LIBPROC = library
    return _LIBPROC


def _libproc_error(operation: str, identifier: int) -> D0Failure:
    error_number = ctypes.get_errno()
    detail = os.strerror(error_number) if error_number else "unknown error"
    return D0Failure(f"{operation} failed for {identifier}: {detail}")


def _libproc_pid_list(function_name: str, identifier: int) -> list[int]:
    require(identifier > 0, f"{function_name} identifier must be positive")
    function = getattr(_libproc(), function_name)
    ctypes.set_errno(0)
    capacity_hint = function(identifier, None, 0)
    if capacity_hint < 0:
        raise _libproc_error(function_name, identifier)
    capacity = max(64, capacity_hint + 32)
    require(
        capacity <= PROCESS_MAXIMUM_COUNT,
        f"{function_name} requested an excessive PID capacity: {capacity}",
    )
    values = (ctypes.c_int * capacity)()
    ctypes.set_errno(0)
    count = function(identifier, values, ctypes.sizeof(values))
    if count < 0:
        raise _libproc_error(function_name, identifier)
    require(
        count <= capacity,
        f"{function_name} returned more PIDs than the allocated capacity",
    )
    pids = [int(values[index]) for index in range(count)]
    require(
        all(pid > 0 for pid in pids),
        f"{function_name} returned a nonpositive PID",
    )
    require(
        len(set(pids)) == len(pids),
        f"{function_name} returned duplicate PIDs",
    )
    return sorted(pids)


def process_executable_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    require(
        stat.S_ISREG(status.st_mode),
        f"Process executable is not a regular file: {resolved}",
    )
    require(
        status.st_dev > 0
        and status.st_ino > 0
        and status.st_size > 0,
        f"Process executable identity is invalid: {resolved}",
    )
    return {
        "path": str(resolved),
        "device": status.st_dev,
        "inode": status.st_ino,
        "size": status.st_size,
    }


def _mapped_executable_identity(
    pid: int,
    executable_path: Path,
) -> dict[str, Any]:
    require(
        ctypes.sizeof(_ProcRegionInfo) == 96
        and ctypes.sizeof(_VinfoStat) == 136
        and ctypes.sizeof(_VnodeInfo) == 152
        and ctypes.sizeof(_VnodeInfoPath) == 1176
        and ctypes.sizeof(_ProcRegionWithPathInfo) == 1272,
        "PROC_PIDREGIONPATHINFO ctypes layout differs from the macOS ABI",
    )
    region = _ProcRegionWithPathInfo()
    ctypes.set_errno(0)
    # Address zero asks libproc for the lowest mapped region. Requiring that
    # region to be executable and to name proc_pidpath binds the main Mach-O.
    copied = _libproc().proc_pidinfo(
        pid,
        PROC_PIDREGIONPATHINFO,
        0,
        ctypes.byref(region),
        ctypes.sizeof(region),
    )
    if copied < 0:
        raise _libproc_error("proc_pidinfo(PROC_PIDREGIONPATHINFO)", pid)
    require(
        copied == ctypes.sizeof(region),
        "proc_pidinfo(PROC_PIDREGIONPATHINFO) returned "
        f"{copied} bytes for PID {pid}, expected {ctypes.sizeof(region)}",
    )

    mapping = region.prp_prinfo
    required_protection = VM_PROT_READ | VM_PROT_EXECUTE
    require(
        int(mapping.pri_address) > 0 and int(mapping.pri_size) > 0,
        f"PID {pid} has an invalid first mapped region",
    )
    require(
        int(mapping.pri_protection) & required_protection == required_protection,
        f"PID {pid} first mapped region is not readable and executable",
    )

    raw_mapping_path = bytes(region.prp_vip.vip_path).split(b"\0", 1)[0]
    require(
        raw_mapping_path != b"" and b"\0" not in raw_mapping_path,
        f"PID {pid} executable mapped region has an invalid path",
    )
    try:
        mapping_path = Path(os.fsdecode(raw_mapping_path))
    except UnicodeError as error:
        raise D0Failure(
            f"Could not decode executable mapped-region path for PID {pid}: "
            f"{error}"
        ) from error
    require(
        mapping_path.is_absolute(),
        f"PID {pid} executable mapped-region path is not absolute",
    )
    require(
        os.path.normpath(mapping_path) == os.path.normpath(executable_path),
        f"PID {pid} first executable mapped region differs from proc_pidpath: "
        f"{mapping_path} != {executable_path}",
    )

    status = region.prp_vip.vip_vi.vi_stat
    require(
        stat.S_ISREG(int(status.vst_mode)),
        f"PID {pid} executable mapped region is not a regular file",
    )
    require(
        int(status.vst_dev) > 0
        and int(status.vst_ino) > 0
        and int(status.vst_size) > 0,
        f"PID {pid} executable mapped-region vnode identity is invalid",
    )
    return {
        "path": str(executable_path),
        "device": int(status.vst_dev),
        "inode": int(status.vst_ino),
        "size": int(status.vst_size),
    }


def capture_process_identity(pid: int) -> dict[str, Any]:
    require(pid > 0, "Process PID must be positive")
    information = _ProcBsdInfo()
    ctypes.set_errno(0)
    copied = _libproc().proc_pidinfo(
        pid,
        PROC_PIDTBSDINFO,
        0,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    if copied < 0:
        raise _libproc_error("proc_pidinfo", pid)
    require(
        copied == ctypes.sizeof(information),
        f"proc_pidinfo returned {copied} bytes for PID {pid}, "
        f"expected {ctypes.sizeof(information)}",
    )
    require(
        int(information.pbi_pid) == pid,
        f"proc_pidinfo returned a different PID for {pid}",
    )

    path_buffer = ctypes.create_string_buffer(PROC_PIDPATHINFO_MAXSIZE)
    ctypes.set_errno(0)
    path_bytes = _libproc().proc_pidpath(
        pid,
        path_buffer,
        ctypes.sizeof(path_buffer),
    )
    if path_bytes <= 0:
        raise _libproc_error("proc_pidpath", pid)
    raw_path = bytes(path_buffer.value)
    require(
        raw_path != b"" and b"\0" not in raw_path,
        f"proc_pidpath returned an invalid path for PID {pid}",
    )
    try:
        executable_path = Path(os.fsdecode(raw_path)).resolve(strict=True)
    except (OSError, UnicodeError) as error:
        raise D0Failure(
            f"Could not resolve executable path for PID {pid}: {error}"
        ) from error
    executable = _mapped_executable_identity(pid, executable_path)
    status_code = int(information.pbi_status)
    status_name = PROCESS_STATUS_NAMES.get(status_code)
    require(
        status_name is not None,
        f"PID {pid} has unknown process status {status_code}",
    )
    raw_name = bytes(information.pbi_name) or bytes(information.pbi_comm)
    process_name = os.fsdecode(raw_name)
    require(process_name != "", f"PID {pid} has an empty process name")
    return {
        "pid": pid,
        "parent_pid": int(information.pbi_ppid),
        "process_group": int(information.pbi_pgid),
        "status_code": status_code,
        "status": status_name,
        "start_time_seconds": int(information.pbi_start_tvsec),
        "start_time_microseconds": int(information.pbi_start_tvusec),
        "name": process_name,
        "executable_path": executable["path"],
        "executable_device": executable["device"],
        "executable_inode": executable["inode"],
        "executable_size": executable["size"],
        "child_pids": process_child_pids(pid),
    }


def process_child_pids(pid: int) -> list[int]:
    require(pid > 0, "Parent process PID must be positive")
    return _libproc_pid_list("proc_listchildpids", pid)


def process_thread_count(pid: int) -> int:
    require(pid > 0, "Thread-count PID must be positive")
    require(
        ctypes.sizeof(_ProcBsdInfo) == 136
        and ctypes.sizeof(_ProcTaskInfo) == 96
        and ctypes.sizeof(_ProcTaskAllInfo) == 232,
        "PROC_PIDTASKALLINFO ctypes layout differs from the macOS ABI",
    )
    information = _ProcTaskAllInfo()
    ctypes.set_errno(0)
    copied = _libproc().proc_pidinfo(
        pid,
        PROC_PIDTASKALLINFO,
        0,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    if copied <= 0:
        raise _libproc_error("proc_pidinfo(PROC_PIDTASKALLINFO)", pid)
    require(
        copied == ctypes.sizeof(information)
        and int(information.pbsd.pbi_pid) == pid,
        "PROC_PIDTASKALLINFO returned malformed process identity",
    )
    count = int(information.ptinfo.pti_threadnum)
    require(count > 0, "PROC_PIDTASKALLINFO returned no process threads")
    return count


def process_group_member_pids(process_group: int) -> list[int]:
    require(process_group > 0, "Attestation process group must be positive")
    return _libproc_pid_list("proc_listpgrppids", process_group)


def process_group_snapshot(
    process_group: int,
    *,
    cwd: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    del cwd
    require(timeout_seconds > 0, "Process snapshot timeout must be positive")
    return [
        capture_process_identity(pid)
        for pid in process_group_member_pids(process_group)
    ]


def _darwin_sigchld_action() -> tuple[int, int]:
    global _SIGACTION_LIBRARY
    require(sys.platform == "darwin", "SIGCHLD policy requires Darwin")
    require(
        ctypes.sizeof(_DarwinSigaction) == 16
        and _DarwinSigaction.mask.offset == 8
        and _DarwinSigaction.flags.offset == 12,
        "SIGCHLD policy requires the Darwin arm64 sigaction ABI",
    )
    if _SIGACTION_LIBRARY is None:
        library = ctypes.CDLL(None, use_errno=True)
        library.sigaction.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_DarwinSigaction),
            ctypes.POINTER(_DarwinSigaction),
        ]
        library.sigaction.restype = ctypes.c_int
        _SIGACTION_LIBRARY = library
    action = _DarwinSigaction()
    ctypes.set_errno(0)
    result = _SIGACTION_LIBRARY.sigaction(
        signal.SIGCHLD,
        None,
        ctypes.byref(action),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return int(action.handler or 0), int(action.flags)


def _require_waitable_sigchld_policy() -> None:
    handler, flags = _darwin_sigchld_action()
    require(
        signal.getsignal(signal.SIGCHLD) is signal.SIG_DFL
        and handler == int(signal.SIG_DFL)
        and flags & SA_NOCLDWAIT == 0,
        "Gated attested launch requires waitable default SIGCHLD policy",
    )


def _ensure_waitable_sigchld_policy() -> None:
    while True:
        try:
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            _require_waitable_sigchld_policy()
            return
        except BaseException:
            try:
                time.sleep(ATTESTATION_STOP_POLL_SECONDS)
            except BaseException:
                pass


def _require_controlled_parent_signal_handlers() -> None:
    for signal_number in signal.valid_signals():
        if signal_number in {
            signal.SIGKILL,
            signal.SIGSTOP,
            *ATTESTATION_DEFERRED_SIGNALS,
        }:
            continue
        require(
            signal.getsignal(signal_number) in {signal.SIG_DFL, signal.SIG_IGN},
            "Gated attested launch rejects uncontrolled signal handlers",
        )


def _configure_attestation_waitid_api() -> Any:
    global _WAITID_LIBRARY
    if _WAITID_LIBRARY is None:
        require(sys.platform == "darwin", "Attestation waitid requires Darwin")
        require(
            ctypes.sizeof(ctypes.c_int) == 4
            and ctypes.sizeof(ctypes.c_uint) == 4
            and ctypes.sizeof(ctypes.c_void_p) == 8
            and ctypes.sizeof(ctypes.c_long) == 8
            and ctypes.sizeof(_DarwinSiginfo) == 104
            and _DarwinSiginfo.si_pid.offset == 12
            and _DarwinSiginfo.si_status.offset == 20,
            "Attestation waitid requires the Darwin arm64 siginfo_t ABI",
        )
        library = ctypes.CDLL(None, use_errno=True)
        library.waitid.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(_DarwinSiginfo),
            ctypes.c_int,
        ]
        library.waitid.restype = ctypes.c_int
        _WAITID_LIBRARY = library
    return _WAITID_LIBRARY


def _observe_attested_terminal_nonconsuming(
    pid: int,
) -> _TerminalObservation | None:
    require(pid > 0, "Attested process PID must be positive")
    information = _DarwinSiginfo()
    ctypes.set_errno(0)
    result = _configure_attestation_waitid_api().waitid(
        P_PID,
        ctypes.c_uint(pid),
        ctypes.byref(information),
        WNOHANG | WNOWAIT | WEXITED,
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
        and information.si_code in WAITID_CHILD_CODES,
        "waitid returned malformed attested-child information: "
        f"signo={information.si_signo}, pid={information.si_pid}, "
        f"code={information.si_code}, status={information.si_status}",
    )
    if information.si_code not in TERMINAL_CHILD_CODES:
        return None
    return _TerminalObservation(
        pid=int(information.si_pid),
        code=int(information.si_code),
        status=int(information.si_status),
    )


def _publish_attested_terminal(
    child: _AttestedChildState,
    observation: _TerminalObservation,
) -> None:
    require(
        observation.pid == child.process.pid
        and child.lifecycle
        in {
            _AttestedLifecycle.OWNED,
            _AttestedLifecycle.TERMINAL_OBSERVED,
        },
        "Attested terminal observation is inconsistent with ownership",
    )
    if child.lifecycle == _AttestedLifecycle.TERMINAL_OBSERVED:
        require(
            child.terminal == observation
            and child.terminal_observed_at_monotonic_ns is not None,
            "Attested terminal observation changed after publication",
        )
        return
    child.terminal = observation
    child.terminal_observed_at_monotonic_ns = time.monotonic_ns()
    child.lifecycle = _AttestedLifecycle.TERMINAL_OBSERVED


def _wait_for_attested_terminal(
    child: _AttestedChildState,
    *,
    deadline: float,
    cancellation_event: threading.Event | None = None,
    cancellation_action: Any | None = None,
    progress_action: Any | None = None,
) -> bool:
    require(
        (cancellation_event is None) == (cancellation_action is None),
        "Attested cancellation event and action must be provided together",
    )
    require(
        child.lifecycle
        in {
            _AttestedLifecycle.OWNED,
            _AttestedLifecycle.TERMINAL_OBSERVED,
        },
        "Attested child cannot be observed after reap starts",
    )
    cancellation_handled = False
    while child.terminal is None:
        if progress_action is not None:
            progress_action()
        if (
            cancellation_event is not None
            and cancellation_event.is_set()
            and not cancellation_handled
        ):
            cancellation_action()
            cancellation_handled = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        observation = _observe_attested_terminal_nonconsuming(child.process.pid)
        if observation is not None:
            _publish_attested_terminal(child, observation)
            return True
        time.sleep(min(ATTESTATION_STOP_POLL_SECONDS, remaining))
    require(
        child.lifecycle == _AttestedLifecycle.TERMINAL_OBSERVED,
        "Attested terminal observation was published incompletely",
    )
    return True


def _wait_for_attested_group_quiescence(
    child: _AttestedChildState,
    *,
    deadline: float,
) -> bool:
    require(
        child.terminal is not None
        and child.lifecycle == _AttestedLifecycle.TERMINAL_OBSERVED
        and child.group_state
        in {
            _AttestedGroupState.ACTIVE,
            _AttestedGroupState.KILL_SENT,
            _AttestedGroupState.QUIESCENT,
        },
        "Attested process group cannot be inspected after reap starts",
    )
    if child.group_state == _AttestedGroupState.QUIESCENT:
        require(
            child.pre_reap_member_pids == [child.process.pid]
            and child.pre_reap_observed_at_monotonic_ns is not None,
            "Attested process group quiescence was published incompletely",
        )
        return True
    while True:
        members = process_group_member_pids(child.process.pid)
        require(
            members == sorted(set(members)) and all(pid > 0 for pid in members),
            "Attested process group PID list is malformed",
        )
        if members == [child.process.pid]:
            child.pre_reap_member_pids = members
            child.pre_reap_observed_at_monotonic_ns = time.monotonic_ns()
            child.group_state = _AttestedGroupState.QUIESCENT
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(ATTESTATION_STOP_POLL_SECONDS, remaining))


def _reap_attested_child(child: _AttestedChildState) -> None:
    require(
        child.terminal is not None
        and child.lifecycle
        in {
            _AttestedLifecycle.TERMINAL_OBSERVED,
            _AttestedLifecycle.REAP_IN_PROGRESS,
        }
        and child.group_state == _AttestedGroupState.QUIESCENT
        and child.pre_reap_member_pids == [child.process.pid]
        and child.pre_reap_observed_at_monotonic_ns is not None,
        "Attested child cannot be reaped before exact leader-only quiescence",
    )
    child.lifecycle = _AttestedLifecycle.REAP_IN_PROGRESS
    try:
        observed_pid, status_code = os.waitpid(child.process.pid, 0)
    except ChildProcessError as error:
        child.lifecycle = _AttestedLifecycle.REAPED
        child.group_state = _AttestedGroupState.EMPTY
        child.process.returncode = sys.maxsize
        raise D0Failure(
            "Attested child was already reaped; wait status is unavailable"
        ) from error

    child.lifecycle = _AttestedLifecycle.REAPED
    child.group_state = _AttestedGroupState.EMPTY
    child.process.returncode = sys.maxsize
    require(
        observed_pid == child.process.pid,
        f"Attested reap returned an unexpected PID: {observed_pid}",
    )
    terminal = child.terminal
    if terminal.code == CLD_EXITED:
        require(
            os.WIFEXITED(status_code)
            and os.WEXITSTATUS(status_code) == terminal.status,
            "Attested waitpid exit status differs from waitid",
        )
        returncode = terminal.status
    else:
        require(
            terminal.code in {CLD_KILLED, CLD_DUMPED}
            and os.WIFSIGNALED(status_code)
            and os.WTERMSIG(status_code) == terminal.status,
            "Attested waitpid signal status differs from waitid",
        )
        returncode = -terminal.status
    child.returncode = returncode
    child.process.returncode = returncode
    child.reap_status_validated = True
    child.reap_completed_at_monotonic_ns = time.monotonic_ns()


def _process_executable_from_snapshot(
    process: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": process["executable_path"],
        "device": process["executable_device"],
        "inode": process["executable_inode"],
        "size": process["executable_size"],
    }


def validate_live_benchmark_topology(
    processes: list[dict[str, Any]],
    *,
    process_group: int,
    supervisor_pid: int,
    expected_wrapper_executable: dict[str, Any],
    expected_benchmark_executable: dict[str, Any],
    phase: str,
) -> int:
    require(
        len(processes) == 2,
        f"The {phase} process group contains {len(processes)} processes "
        "instead of the time wrapper and benchmark child",
    )
    by_pid = {int(process["pid"]): process for process in processes}
    require(
        len(by_pid) == len(processes),
        f"The {phase} process group contains duplicate PIDs",
    )
    wrapper = by_pid.get(process_group)
    require(
        wrapper is not None,
        f"The {phase} process group lost its leader",
    )
    benchmark_candidates = [
        process for pid, process in by_pid.items() if pid != process_group
    ]
    require(
        len(benchmark_candidates) == 1,
        f"The {phase} process group lacks a unique benchmark child",
    )
    benchmark = benchmark_candidates[0]
    benchmark_pid = int(benchmark["pid"])
    require(
        wrapper["parent_pid"] == supervisor_pid,
        f"The {phase} time wrapper has an unexpected parent",
    )
    require(
        wrapper["process_group"] == process_group
        and benchmark["process_group"] == process_group,
        f"The {phase} benchmark topology escaped its process group",
    )
    require(
        wrapper["status"] not in {"creating", "stopped", "zombie"},
        f"The {phase} time wrapper has invalid status {wrapper['status']}",
    )
    require(
        benchmark["parent_pid"] == process_group,
        f"The {phase} benchmark is not a direct child of the time wrapper",
    )
    require(
        benchmark["status"] == "stopped",
        f"The {phase} benchmark child is not stopped",
    )
    require(
        wrapper["child_pids"] == [benchmark_pid],
        f"The {phase} time wrapper child set differs",
    )
    require(
        benchmark["child_pids"] == [],
        f"The {phase} benchmark spawned descendants",
    )
    require(
        _process_executable_from_snapshot(wrapper)
        == expected_wrapper_executable,
        f"The {phase} time wrapper executable identity differs: "
        f"observed={_process_executable_from_snapshot(wrapper)!r}, "
        f"expected={expected_wrapper_executable!r}",
    )
    require(
        _process_executable_from_snapshot(benchmark)
        == expected_benchmark_executable,
        f"The {phase} benchmark executable identity differs: "
        f"observed={_process_executable_from_snapshot(benchmark)!r}, "
        f"expected={expected_benchmark_executable!r}",
    )
    return benchmark_pid


def wait_for_attestation_stop(
    process_group: int,
    phase: str,
    *,
    cwd: Path,
    deadline: float,
    supervisor_pid: int,
    expected_wrapper_executable: dict[str, Any],
    expected_benchmark_executable: dict[str, Any],
) -> dict[str, Any]:
    require(
        phase in ATTESTATION_BARRIER_PHASES,
        f"Unexpected attestation barrier phase: {phase}",
    )
    while True:
        remaining = deadline - time.monotonic()
        require(
            remaining > 0,
            f"Timed out waiting for the {phase} attestation stop",
        )
        processes = process_group_snapshot(
            process_group,
            cwd=cwd,
            timeout_seconds=max(0.1, min(1.0, remaining)),
        )
        stopped = [
            process for process in processes if process["status"] == "stopped"
        ]
        if stopped:
            stopped_pid = validate_live_benchmark_topology(
                processes,
                process_group=process_group,
                supervisor_pid=supervisor_pid,
                expected_wrapper_executable=expected_wrapper_executable,
                expected_benchmark_executable=expected_benchmark_executable,
                phase=phase,
            )
            observed_at_ns = time.monotonic_ns()
            os.kill(stopped_pid, signal.SIGCONT)
            return {
                "phase": phase,
                "process_group": process_group,
                "stopped_pid": stopped_pid,
                "observed_at_monotonic_ns": observed_at_ns,
                "resumed_at_monotonic_ns": time.monotonic_ns(),
                "processes": processes,
            }
        time.sleep(min(ATTESTATION_STOP_POLL_SECONDS, remaining))


def _stable_process_identity(process: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in process.items()
        if key not in {"status_code", "status"}
    }


def benchmark_process_evidence(
    barriers: list[dict[str, Any]],
    *,
    process_group: int,
    supervisor_pid: int,
    expected_wrapper_executable: dict[str, Any],
    expected_benchmark_executable: dict[str, Any],
    terminal_observation: dict[str, Any],
    pre_reap_quiescence: dict[str, Any],
    reap: dict[str, Any],
) -> dict[str, Any]:
    require(
        [barrier.get("phase") for barrier in barriers]
        == list(ATTESTATION_BARRIER_PHASES),
        "Benchmark process evidence requires all four ordered barriers",
    )
    wrapper_identity: dict[str, Any] | None = None
    benchmark_identity: dict[str, Any] | None = None
    benchmark_pid: int | None = None
    for barrier in barriers:
        phase = str(barrier["phase"])
        processes = barrier.get("processes")
        require(
            isinstance(processes, list),
            f"The {phase} process snapshot is not a list",
        )
        current_benchmark_pid = validate_live_benchmark_topology(
            processes,
            process_group=process_group,
            supervisor_pid=supervisor_pid,
            expected_wrapper_executable=expected_wrapper_executable,
            expected_benchmark_executable=expected_benchmark_executable,
            phase=phase,
        )
        by_pid = {int(process["pid"]): process for process in processes}
        current_wrapper_identity = _stable_process_identity(
            by_pid[process_group]
        )
        current_benchmark_identity = _stable_process_identity(
            by_pid[current_benchmark_pid]
        )
        if wrapper_identity is None:
            wrapper_identity = current_wrapper_identity
            benchmark_identity = current_benchmark_identity
            benchmark_pid = current_benchmark_pid
        else:
            require(
                current_wrapper_identity == wrapper_identity,
                "Time-wrapper process identity changed across barriers",
            )
            require(
                current_benchmark_identity == benchmark_identity
                and current_benchmark_pid == benchmark_pid,
                "Benchmark-child process identity changed across barriers",
            )
    require(
        set(terminal_observation)
        == {"observed_at_monotonic_ns", "pid", "code", "status"},
        "Terminal process evidence keys differ",
    )
    terminal_observed_at_ns = terminal_observation["observed_at_monotonic_ns"]
    terminal_pid = terminal_observation["pid"]
    terminal_code = terminal_observation["code"]
    terminal_status = terminal_observation["status"]
    require(
        type(terminal_observed_at_ns) is int
        and terminal_observed_at_ns >= barriers[-1]["resumed_at_monotonic_ns"]
        and terminal_pid == process_group
        and type(terminal_code) is int
        and type(terminal_status) is int
        and terminal_code in TERMINAL_CHILD_CODES
        and terminal_status >= 0,
        "Benchmark terminal observation is invalid",
    )
    require(
        set(pre_reap_quiescence)
        == {
            "observed_at_monotonic_ns",
            "process_group",
            "leader_pid",
            "member_pids",
            "no_members_except_leader",
        },
        "Pre-reap process evidence keys differ",
    )
    observed_at_ns = pre_reap_quiescence["observed_at_monotonic_ns"]
    member_pids = pre_reap_quiescence["member_pids"]
    require(
        type(observed_at_ns) is int
        and observed_at_ns >= terminal_observed_at_ns,
        "Pre-reap observation time is invalid",
    )
    require(
        pre_reap_quiescence["process_group"] == process_group
        and pre_reap_quiescence["leader_pid"] == process_group
        and member_pids == [process_group]
        and pre_reap_quiescence["no_members_except_leader"] is True,
        "Benchmark process group was not quiescent before reap",
    )
    require(
        set(reap)
        == {
            "completed_at_monotonic_ns",
            "pid",
            "status_validated",
            "returncode",
        },
        "Reap process evidence keys differ",
    )
    reap_completed_at_ns = reap["completed_at_monotonic_ns"]
    returncode = reap["returncode"]
    expected_returncode = (
        terminal_status if terminal_code == CLD_EXITED else -terminal_status
    )
    require(
        type(reap_completed_at_ns) is int
        and reap_completed_at_ns >= observed_at_ns
        and reap["pid"] == process_group
        and reap["status_validated"] is True
        and type(returncode) is int
        and returncode == expected_returncode,
        "Benchmark reap status differs from its terminal observation",
    )
    return {
        "schema_version": PROCESS_EVIDENCE_SCHEMA_VERSION,
        "supervisor_pid": supervisor_pid,
        "process_group": process_group,
        "wrapper_executable": expected_wrapper_executable,
        "benchmark_executable": expected_benchmark_executable,
        "barrier_phases": list(ATTESTATION_BARRIER_PHASES),
        "benchmark_pid": benchmark_pid,
        "terminal_observation": terminal_observation,
        "pre_reap_quiescence": pre_reap_quiescence,
        "reap": reap,
    }


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = list(getattr(error, "_infinity_notes", ()))
    notes.append(note)
    setattr(error, "_infinity_notes", tuple(notes))


def _add_exception_note_noexcept(error: BaseException, note: str) -> None:
    try:
        _add_exception_note(error, note)
    except BaseException:
        pass


def _add_secondary_exception_note_noexcept(
    primary: BaseException,
    prefix: str,
    secondary: BaseException,
) -> None:
    try:
        secondary_type = type(secondary).__name__
        detail = str(secondary)
        if detail:
            note = f"{prefix} with {secondary_type}: {detail}"
        else:
            note = f"{prefix} with {secondary_type}"
    except BaseException:
        note = prefix
    _add_exception_note_noexcept(primary, note)


def _append_under_lock_noexcept(
    lock: Any,
    items: Any,
    value: Any,
) -> None:
    try:
        with _rlock_guard(lock):
            items.append(value)
    except BaseException:
        pass


@contextmanager
def _rlock_guard(lock: Any) -> Iterable[None]:
    acquired = False
    acquisition_ambiguous = False
    try:
        try:
            acquired = lock.acquire()
        except BaseException:
            acquisition_ambiguous = True
            raise
        require(acquired, "State RLock acquisition returned false")
        yield
    finally:
        if acquired or acquisition_ambiguous:
            try:
                lock.release()
            except BaseException:
                pass


def _supervise_attested_process_once(
    process: Any,
    *,
    cwd: Path,
    timeout_seconds: float,
    expected_wrapper_executable: dict[str, Any],
    expected_benchmark_executable: dict[str, Any],
    _deferred_sigint_state: _DeferredSigintState | None = None,
    _initial_error: BaseException | None = None,
) -> dict[str, Any]:
    require(process.pid > 0, "Attested process PID must be positive")
    require(process.stdout is not None, "Attested process stdout pipe is missing")
    require(process.stderr is not None, "Attested process stderr pipe is missing")
    require(timeout_seconds > 0, "Attested process timeout must be positive")

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    pending_barrier_lines: list[str] = []
    barriers: list[dict[str, Any]] = []
    errors: list[str] = []
    deferred_exceptions: list[BaseException] = []
    state_lock = threading.RLock()
    lifecycle_lock = threading.Lock()
    termination_requested = threading.Event()
    deadline = time.monotonic() + timeout_seconds
    pipe_close_attempts: set[str] = set()
    supervisor_pid = os.getpid()
    child = _AttestedChildState(process=process)

    def raise_deferred_owner_interrupt() -> None:
        if (
            _deferred_sigint_state is not None
            and _deferred_sigint_state.requested
        ):
            raise _deferred_signal_exception(_deferred_sigint_state)

    def append_error(message: str) -> None:
        _append_under_lock_noexcept(state_lock, errors, message)

    def describe_error(error: BaseException) -> str:
        try:
            error_type = type(error).__name__
        except BaseException:
            error_type = "BaseException"
        try:
            detail = str(error)
        except BaseException:
            detail = ""
        if detail:
            try:
                return f"{error_type}: {detail}"
            except BaseException:
                pass
        return error_type

    def record_cleanup_failure(
        operation: str,
        error: BaseException,
    ) -> None:
        try:
            description = describe_error(error)
        except BaseException:
            description = "BaseException"
        try:
            message = (
                f"Attested process cleanup failed during {operation}: "
                f"{description}"
            )
        except BaseException:
            message = "Attested process cleanup failed"
        append_error(message)

    def defer_non_exception(error: BaseException) -> None:
        try:
            is_exception = isinstance(error, Exception)
        except BaseException:
            is_exception = False
        if is_exception:
            return
        _append_under_lock_noexcept(state_lock, deferred_exceptions, error)

    def terminate_group() -> list[BaseException]:
        with lifecycle_lock:
            if (
                child.lifecycle
                in {
                    _AttestedLifecycle.REAP_IN_PROGRESS,
                    _AttestedLifecycle.REAPED,
                }
                or child.group_state
                in {
                    _AttestedGroupState.KILL_SENT,
                    _AttestedGroupState.QUIESCENT,
                    _AttestedGroupState.EMPTY,
                    _AttestedGroupState.UNRESOLVED,
                }
            ):
                return []
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                child.group_state = _AttestedGroupState.KILL_SENT
                return []
            except BaseException as error:
                record_cleanup_failure("kill process group", error)
                return [error]
            child.group_state = _AttestedGroupState.KILL_SENT
        return []

    def record_error(message: str) -> None:
        append_error(message)
        termination_requested.set()

    def perform_requested_termination() -> None:
        for error in terminate_group():
            defer_non_exception(error)

    def drain_stdout() -> None:
        try:
            for line in process.stdout:
                stdout_parts.append(line)
                with _rlock_guard(state_lock):
                    if line.startswith("attestation_barrier="):
                        pending_barrier_lines.append(line)
        except BaseException as error:
            defer_non_exception(error)
            record_error(f"attestation stdout supervision failed: {error}")

    def drain_stderr() -> None:
        try:
            stderr_parts.extend(process.stderr)
        except BaseException as error:
            defer_non_exception(error)
            record_error(f"attestation stderr capture failed: {error}")

    output_threads: list[tuple[str, threading.Thread]] = []
    settled_output_threads: set[str] = set()

    def process_pending_barriers() -> None:
        raise_deferred_owner_interrupt()
        while True:
            with _rlock_guard(state_lock):
                if not pending_barrier_lines:
                    return
                line = pending_barrier_lines.pop(0)
                expected_index = len(barriers)
            try:
                require(
                    line.endswith("\n") and line.count("=") == 1,
                    "Attestation barrier marker is malformed",
                )
                phase = line.removesuffix("\n").split("=", 1)[1]
                require(
                    expected_index < len(ATTESTATION_BARRIER_PHASES),
                    "Attested process emitted too many barrier markers",
                )
                require(
                    phase == ATTESTATION_BARRIER_PHASES[expected_index],
                    f"Attestation barrier {phase} is out of order",
                )
                observation = wait_for_attestation_stop(
                    process.pid,
                    phase,
                    cwd=cwd,
                    deadline=deadline,
                    supervisor_pid=supervisor_pid,
                    expected_wrapper_executable=expected_wrapper_executable,
                    expected_benchmark_executable=expected_benchmark_executable,
                )
                with _rlock_guard(state_lock):
                    require(
                        len(barriers) == expected_index,
                        "Attestation barrier state changed concurrently",
                    )
                    barriers.append(observation)
            except BaseException as error:
                defer_non_exception(error)
                record_error(f"attestation barrier supervision failed: {error}")
                return

    def close_pipes() -> list[BaseException]:
        failures: list[BaseException] = []
        for name, pipe in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        ):
            if name in pipe_close_attempts:
                continue
            pipe_close_attempts.add(name)
            try:
                pipe.close()
            except BaseException as error:
                record_cleanup_failure(f"close {name}", error)
                failures.append(error)
        return failures

    def join_output_threads(
        *,
        cleanup: bool,
    ) -> tuple[list[BaseException], list[str]]:
        failures: list[BaseException] = []
        alive: list[str] = []
        for name, thread in output_threads:
            if name in settled_output_threads:
                continue
            joined = False
            try:
                thread.join(timeout=ATTESTATION_THREAD_JOIN_SECONDS)
                joined = True
            except BaseException as error:
                record_cleanup_failure(f"join {name} thread", error)
                failures.append(error)
            try:
                if thread.is_alive():
                    alive.append(name)
                    if cleanup:
                        append_error(
                            f"Attested process cleanup failed during join {name} "
                            f"thread: did not terminate within "
                            f"{ATTESTATION_THREAD_JOIN_SECONDS} seconds"
                        )
                elif joined:
                    settled_output_threads.add(name)
            except BaseException as error:
                record_cleanup_failure(f"inspect {name} thread", error)
                failures.append(error)
        return failures, alive

    def cleanup_after_abnormal_exit_once() -> list[BaseException]:
        failures: list[BaseException] = []
        if child.lifecycle not in {
            _AttestedLifecycle.REAP_IN_PROGRESS,
            _AttestedLifecycle.REAPED,
        }:
            failures.extend(terminate_group())
        failures.extend(close_pipes())
        join_failures, _ = join_output_threads(cleanup=True)
        failures.extend(join_failures)
        if child.cleanup_deadline is None:
            child.cleanup_deadline = (
                time.monotonic() + ATTESTATION_POST_KILL_WAIT_SECONDS
            )
        settle_deadline = child.cleanup_deadline

        if child.lifecycle == _AttestedLifecycle.OWNED:
            try:
                if not _wait_for_attested_terminal(
                    child,
                    deadline=settle_deadline,
                ):
                    error = TimeoutError(
                        "did not terminate within "
                        f"{ATTESTATION_POST_KILL_WAIT_SECONDS} seconds"
                    )
                    record_cleanup_failure("observe terminal process", error)
                    failures.append(error)
            except BaseException as error:
                record_cleanup_failure("observe terminal process", error)
                failures.append(error)
                defer_non_exception(error)

        if (
            child.lifecycle == _AttestedLifecycle.TERMINAL_OBSERVED
            and child.group_state != _AttestedGroupState.QUIESCENT
        ):
            try:
                if not _wait_for_attested_group_quiescence(
                    child,
                    deadline=settle_deadline,
                ):
                    error = TimeoutError(
                        "process group retained members after bounded cleanup"
                    )
                    record_cleanup_failure(
                        "establish pre-reap group quiescence",
                        error,
                    )
                    failures.append(error)
            except BaseException as error:
                record_cleanup_failure(
                    "establish pre-reap group quiescence",
                    error,
                )
                failures.append(error)
                defer_non_exception(error)

        if (
            child.lifecycle
            in {
                _AttestedLifecycle.TERMINAL_OBSERVED,
                _AttestedLifecycle.REAP_IN_PROGRESS,
            }
            and child.group_state == _AttestedGroupState.QUIESCENT
            and child.pre_reap_member_pids == [process.pid]
            and child.pre_reap_observed_at_monotonic_ns is not None
        ):
            try:
                with lifecycle_lock:
                    if child.lifecycle != _AttestedLifecycle.REAPED:
                        _reap_attested_child(child)
            except BaseException as error:
                record_cleanup_failure("reap process", error)
                failures.append(error)
                defer_non_exception(error)
        return failures

    def cleanup_after_abnormal_exit() -> list[BaseException]:
        failures: list[BaseException] = []

        def remember_failure(error: BaseException) -> None:
            try:
                failures.append(error)
            except BaseException:
                pass
            defer_non_exception(error)

        for attempt in range(2):
            try:
                failures.extend(cleanup_after_abnormal_exit_once())
            except BaseException as error:
                record_cleanup_failure(
                    "complete cleanup attempt",
                    error,
                )
                remember_failure(error)
            if child.lifecycle == _AttestedLifecycle.REAPED:
                break
        while True:
            try:
                if (
                    child.lifecycle == _AttestedLifecycle.REAPED
                    or getattr(process, "production_owned", False) is not True
                ):
                    break
                child.cleanup_deadline = (
                    time.monotonic() + ATTESTATION_POST_KILL_WAIT_SECONDS
                )
                failures.extend(cleanup_after_abnormal_exit_once())
                if child.lifecycle != _AttestedLifecycle.REAPED:
                    time.sleep(ATTESTATION_STOP_POLL_SECONDS)
            except BaseException as error:
                record_cleanup_failure("quarantined cleanup retry", error)
                remember_failure(error)
        if child.lifecycle != _AttestedLifecycle.REAPED:
            if child.group_state != _AttestedGroupState.QUIESCENT:
                child.group_state = _AttestedGroupState.UNRESOLVED
            append_error(
                "Attested process cleanup exhausted without a validated reap"
            )
        return failures

    timed_out = False
    cleanup_required = False
    primary_error: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    try:
        try:
            if _initial_error is not None:
                raise _initial_error
            raise_deferred_owner_interrupt()
            output_threads = [
                (
                    "stdout",
                    threading.Thread(
                        target=drain_stdout,
                        name=f"attestation-stdout-{process.pid}",
                    ),
                ),
                (
                    "stderr",
                    threading.Thread(
                        target=drain_stderr,
                        name=f"attestation-stderr-{process.pid}",
                    ),
                ),
            ]
            for _, thread in output_threads:
                thread.start()
            try:
                terminated = _wait_for_attested_terminal(
                    child,
                    deadline=deadline,
                    cancellation_event=termination_requested,
                    cancellation_action=perform_requested_termination,
                    progress_action=process_pending_barriers,
                )
            except BaseException:
                raise
            if not terminated:
                timed_out = True
                cleanup_required = True
            else:
                join_failures, alive_threads = join_output_threads(cleanup=False)
                process_pending_barriers()
                if join_failures:
                    primary_error = join_failures[0]
                    cleanup_required = True
                if alive_threads:
                    append_error("Attested process output threads did not terminate")
                    cleanup_required = True
                with _rlock_guard(state_lock):
                    if deferred_exceptions and primary_error is None:
                        primary_error = deferred_exceptions[0]
                    cleanup_required = cleanup_required or bool(errors)
                    cleanup_required = cleanup_required or bool(deferred_exceptions)
                if not cleanup_required:
                    try:
                        quiescence_deadline = (
                            time.monotonic() + ATTESTATION_POST_KILL_WAIT_SECONDS
                        )
                        if not _wait_for_attested_group_quiescence(
                            child,
                            deadline=quiescence_deadline,
                        ):
                            append_error(
                                "Attested process group retained members before reap"
                            )
                            cleanup_required = True
                        else:
                            with lifecycle_lock:
                                _reap_attested_child(child)
                    except BaseException as error:
                        primary_error = error
                        cleanup_required = True
        except BaseException as error:
            primary_error = error
            cleanup_required = True
    finally:
        if cleanup_required:
            cleanup_failures = cleanup_after_abnormal_exit()
        else:
            close_failures = close_pipes()
            if close_failures:
                primary_error = close_failures[0]
                cleanup_failures = cleanup_after_abnormal_exit()

    with _rlock_guard(state_lock):
        deferred_snapshot = list(deferred_exceptions)
    secondary_exceptions = [
        error
        for error in (*deferred_snapshot, *cleanup_failures)
        if error is not primary_error
    ]
    if primary_error is not None:
        if (
            _deferred_sigint_state is not None
            and _deferred_sigint_state.requested
            and _deferred_signal_exception(_deferred_sigint_state)
            is not primary_error
        ):
            _add_exception_note_noexcept(
                primary_error,
                "A termination signal was deferred while the attested child "
                "was settled",
            )
        for error in secondary_exceptions:
            _add_secondary_exception_note_noexcept(
                primary_error,
                "Attested process secondary failure",
                error,
            )
        raise primary_error
    owner_interrupt = (
        _deferred_signal_exception(_deferred_sigint_state)
        if (
            _deferred_sigint_state is not None
            and _deferred_sigint_state.requested
        )
        else None
    )
    if owner_interrupt is not None:
        for error in secondary_exceptions:
            if error is not owner_interrupt:
                _add_secondary_exception_note_noexcept(
                    owner_interrupt,
                    "Attested process secondary failure",
                    error,
                )
        raise owner_interrupt
    fatal_error = next(
        (
            error
            for error in secondary_exceptions
            if not isinstance(error, Exception)
        ),
        None,
    )
    if fatal_error is not None:
        raise fatal_error

    phases = [str(barrier["phase"]) for barrier in barriers]
    if phases != list(ATTESTATION_BARRIER_PHASES):
        append_error(
            "Attested process did not complete the ordered four-barrier protocol: "
            f"{phases}"
        )
    process_evidence: dict[str, Any] = {}
    if (
        phases == list(ATTESTATION_BARRIER_PHASES)
        and child.terminal is not None
        and child.terminal_observed_at_monotonic_ns is not None
        and child.pre_reap_member_pids is not None
        and child.pre_reap_observed_at_monotonic_ns is not None
        and child.lifecycle == _AttestedLifecycle.REAPED
        and child.group_state == _AttestedGroupState.EMPTY
        and child.reap_status_validated
        and child.returncode is not None
        and child.reap_completed_at_monotonic_ns is not None
    ):
        terminal_observation = {
            "observed_at_monotonic_ns": (
                child.terminal_observed_at_monotonic_ns
            ),
            "pid": process.pid,
            "code": child.terminal.code,
            "status": child.terminal.status,
        }
        pre_reap_quiescence = {
            "observed_at_monotonic_ns": (
                child.pre_reap_observed_at_monotonic_ns
            ),
            "process_group": process.pid,
            "leader_pid": process.pid,
            "member_pids": child.pre_reap_member_pids,
            "no_members_except_leader": True,
        }
        reap = {
            "completed_at_monotonic_ns": (
                child.reap_completed_at_monotonic_ns
            ),
            "pid": process.pid,
            "status_validated": True,
            "returncode": child.returncode,
        }
        try:
            process_evidence = benchmark_process_evidence(
                barriers,
                process_group=process.pid,
                supervisor_pid=supervisor_pid,
                expected_wrapper_executable=expected_wrapper_executable,
                expected_benchmark_executable=expected_benchmark_executable,
                terminal_observation=terminal_observation,
                pre_reap_quiescence=pre_reap_quiescence,
                reap=reap,
            )
        except D0Failure as error:
            append_error(str(error))
    return {
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "timed_out": timed_out,
        "barriers": barriers,
        "benchmark_process_evidence": process_evidence,
        "errors": errors,
    }


def supervise_attested_process(
    process: Any,
    *,
    cwd: Path,
    timeout_seconds: float,
    expected_wrapper_executable: dict[str, Any],
    expected_benchmark_executable: dict[str, Any],
    _deferred_sigint_state: _DeferredSigintState | None = None,
    _initial_error: BaseException | None = None,
) -> dict[str, Any]:
    first_error = _initial_error
    while True:
        try:
            result = _supervise_attested_process_once(
                process,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                expected_wrapper_executable=expected_wrapper_executable,
                expected_benchmark_executable=expected_benchmark_executable,
                _deferred_sigint_state=_deferred_sigint_state,
                _initial_error=first_error,
            )
        except BaseException as error:
            if first_error is None:
                first_error = error
            elif error is not first_error:
                _add_secondary_exception_note_noexcept(
                    first_error,
                    "Attested process supervision retry failed",
                    error,
                )
        else:
            if first_error is None:
                return result

        try:
            settled = process.returncode is not None
        except BaseException as error:
            settled = False
            if first_error is None:
                first_error = error
            elif error is not first_error:
                _add_secondary_exception_note_noexcept(
                    first_error,
                    "Attested process settlement-state read failed",
                    error,
                )
        if settled:
            require(first_error is not None, "Settled supervision lost its error")
            raise first_error

        try:
            production_owned = (
                getattr(process, "production_owned", False) is True
            )
        except BaseException as error:
            production_owned = isinstance(process, _GatedTextProcess)
            if first_error is None:
                first_error = error
            elif error is not first_error:
                _add_secondary_exception_note_noexcept(
                    first_error,
                    "Attested process ownership-state read failed",
                    error,
                )
        if not production_owned:
            require(first_error is not None, "Failed supervision lost its error")
            raise first_error
        try:
            time.sleep(ATTESTATION_STOP_POLL_SECONDS)
        except BaseException as error:
            if first_error is None:
                first_error = error
            elif error is not first_error:
                _add_secondary_exception_note_noexcept(
                    first_error,
                    "Attested process supervision retry sleep failed",
                    error,
                )


def _close_descriptor_quietly(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except BaseException:
        pass


def _write_gated_child_failure(
    descriptor: int,
    *,
    stage: int,
    error_number: int,
) -> None:
    try:
        payload = (
            b"F"
            + bytes((stage & 0xFF,))
            + int(error_number).to_bytes(4, "little", signed=True)
        )
        os.write(descriptor, payload)
    except BaseException:
        pass


def _gated_exec_default_signal_numbers(
    blockable_signals: set[signal.Signals],
) -> tuple[int, ...]:
    default_signal_numbers = {
        int(signal_number)
        for signal_number in blockable_signals
        if signal.getsignal(signal_number) not in {signal.SIG_DFL, signal.SIG_IGN}
    }
    default_signal_numbers.update(
        int(signal_number)
        for signal_number in ATTESTATION_DEFERRED_SIGNALS
    )
    default_signal_numbers.add(int(signal.SIGPIPE))
    if hasattr(signal, "SIGXFSZ"):
        default_signal_numbers.add(int(signal.SIGXFSZ))
    return tuple(sorted(default_signal_numbers))


def _run_gated_attested_child(
    *,
    command: tuple[str, ...],
    environment: dict[str, str],
    previous_signal_mask: set[signal.Signals],
    default_signal_numbers: tuple[int, ...],
    command_descriptor: int,
    status_descriptor: int,
    stdout_descriptor: int,
    stderr_descriptor: int,
    devnull_descriptor: int,
    cwd_descriptor: int,
    parent_descriptors: tuple[int, ...],
) -> None:
    for descriptor in parent_descriptors:
        _close_descriptor_quietly(descriptor)
    stage = 1
    try:
        os.setsid()
        stage = 2
        require(
            os.write(status_descriptor, b"H") == 1,
            "Gated child could not publish ownership",
        )
        stage = 3
        action = os.read(command_descriptor, 1)
        if action != b"E":
            os._exit(125)
        stage = 4
        os.dup2(devnull_descriptor, 0)
        os.dup2(stdout_descriptor, 1)
        os.dup2(stderr_descriptor, 2)
        stage = 5
        os.fchdir(cwd_descriptor)
        stage = 6
        for signal_number in default_signal_numbers:
            signal.signal(signal_number, signal.SIG_DFL)
        require(
            os.write(status_descriptor, b"R") == 1,
            "Gated child could not publish exec readiness",
        )
        stage = 7
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
        stage = 8
        os.execve(command[0], command, environment)
    except OSError as error:
        _write_gated_child_failure(
            status_descriptor,
            stage=stage,
            error_number=error.errno or 0,
        )
    except BaseException:
        _write_gated_child_failure(
            status_descriptor,
            stage=stage,
            error_number=0,
        )
    os._exit(126)


def _settle_pre_exec_gated_child(pid: int, *, group_verified: bool) -> None:
    require(pid > 0, "Pre-exec gated child PID must be positive")
    while True:
        try:
            observation = _observe_attested_terminal_nonconsuming(pid)
        except InterruptedError:
            continue
        except ChildProcessError:
            return
        except BaseException:
            pass
        else:
            if observation is not None:
                break
            try:
                if group_verified:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException:
                pass
        try:
            time.sleep(ATTESTATION_STOP_POLL_SECONDS)
        except BaseException:
            pass

    # A nonconsuming terminal observation pins the child as our zombie. From
    # here onward, never inspect or signal its process group; only settle reap.
    while True:
        try:
            observed_pid, _ = os.waitpid(pid, 0)
        except ChildProcessError:
            return
        except BaseException:
            try:
                observation = _observe_attested_terminal_nonconsuming(pid)
            except ChildProcessError:
                return
            except BaseException:
                observation = None
        else:
            if observed_pid == pid:
                return
        try:
            time.sleep(ATTESTATION_STOP_POLL_SECONDS)
        except BaseException:
            pass


def _close_ambiguous_fork_gate(descriptor: int) -> None:
    while True:
        try:
            os.close(descriptor)
            return
        except OSError as error:
            if error.errno == errno.EBADF:
                return
        except BaseException:
            pass
        try:
            time.sleep(ATTESTATION_STOP_POLL_SECONDS)
        except BaseException:
            pass


def _settle_ambiguous_fork_child() -> None:
    # Signals are blocked and child exclusivity was rechecked immediately
    # before fork. If fork took effect before its Python result raised, the
    # closed exec gate makes that sole child exit without reaching exec.
    while True:
        try:
            observed_pid, _ = os.waitpid(-1, 0)
        except ChildProcessError:
            return
        except OSError as error:
            if error.errno == errno.ECHILD:
                return
        except BaseException:
            pass
        else:
            if observed_pid > 0:
                continue
        try:
            time.sleep(ATTESTATION_STOP_POLL_SECONDS)
        except BaseException:
            pass


def _fork_gated_attested_process(
    command: list[str],
    *,
    launch_cwd: Path,
    environment: Mapping[str, str],
    state: _GatedLaunchState,
    deadline: float,
    deferred: _DeferredSigintState,
) -> _GatedTextProcess:
    require(sys.platform == "darwin", "Gated attested launch requires Darwin")
    _require_waitable_sigchld_policy()
    _require_controlled_parent_signal_handlers()
    require(
        threading.active_count() == 1
        and process_thread_count(os.getpid()) == 1,
        "Gated attested launch requires a process-wide single-threaded supervisor",
    )
    require(
        state.phase == _AttestedLaunchPhase.REGISTERED
        and state.process is None,
        "Gated attested launch state is not fresh",
    )
    require(
        process_child_pids(os.getpid()) == [],
        "Gated attested launch requires no preexisting child processes",
    )
    cwd_descriptor = -1
    devnull_descriptor = -1
    command_read = -1
    command_write = -1
    status_read = -1
    status_write = -1
    stdout_read = -1
    stdout_write = -1
    stderr_read = -1
    stderr_write = -1
    stdout_stream: Any | None = None
    stderr_stream: Any | None = None
    previous_mask: set[signal.Signals] | None = None
    pid = -1
    group_verified = False
    child_settled = False
    process: _GatedTextProcess | None = None
    primary_error: BaseException | None = None
    try:
        cwd_descriptor = os.open(
            launch_cwd,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        devnull_descriptor = os.open(
            "/dev/null",
            os.O_RDONLY | os.O_CLOEXEC,
        )
        command_read, command_write = os.pipe()
        status_read, status_write = os.pipe()
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()
        descriptors = (
            cwd_descriptor,
            devnull_descriptor,
            command_read,
            command_write,
            status_read,
            status_write,
            stdout_read,
            stdout_write,
            stderr_read,
            stderr_write,
        )
        require(
            all(descriptor > 2 for descriptor in descriptors)
            and all(
                not os.get_inheritable(descriptor)
                for descriptor in descriptors
            ),
            "Gated attested launch descriptors must be close-on-exec",
        )
        command_tuple = tuple(command)
        environment_copy = dict(environment)
        blockable_signals = {
            signal_number
            for signal_number in signal.valid_signals()
            if signal_number not in {signal.SIGKILL, signal.SIGSTOP}
        }
        default_signal_numbers = _gated_exec_default_signal_numbers(
            blockable_signals
        )
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            set(),
        )
        signal.pthread_sigmask(
            signal.SIG_BLOCK,
            blockable_signals,
        )
        _require_waitable_sigchld_policy()
        _require_controlled_parent_signal_handlers()
        require(
            process_child_pids(os.getpid()) == [],
            "Gated attested launch gained a child before fork",
        )
        handshake_deadline = min(
            deadline,
            time.monotonic() + ATTESTATION_LAUNCH_GATE_SECONDS,
        )
        try:
            pid = os.fork()
        except BaseException:
            _close_ambiguous_fork_gate(command_write)
            command_write = -1
            _settle_ambiguous_fork_child()
            raise
        if pid == 0:
            _run_gated_attested_child(
                command=command_tuple,
                environment=environment_copy,
                previous_signal_mask=previous_mask,
                default_signal_numbers=default_signal_numbers,
                command_descriptor=command_read,
                status_descriptor=status_write,
                stdout_descriptor=stdout_write,
                stderr_descriptor=stderr_write,
                devnull_descriptor=devnull_descriptor,
                cwd_descriptor=cwd_descriptor,
                parent_descriptors=(
                    command_write,
                    status_read,
                    stdout_read,
                    stderr_read,
                ),
            )
            os._exit(127)

        require(pid > 0, "fork returned a nonpositive child PID")
        _ensure_waitable_sigchld_policy()
        _require_controlled_parent_signal_handlers()
        state.phase = _AttestedLaunchPhase.FORKED
        _close_descriptor_quietly(command_read)
        command_read = -1
        _close_descriptor_quietly(status_write)
        status_write = -1
        _close_descriptor_quietly(stdout_write)
        stdout_write = -1
        _close_descriptor_quietly(stderr_write)
        stderr_write = -1
        _close_descriptor_quietly(devnull_descriptor)
        devnull_descriptor = -1
        _close_descriptor_quietly(cwd_descriptor)
        cwd_descriptor = -1

        stdout_stream = os.fdopen(
            stdout_read,
            "r",
            encoding="utf-8",
            errors="strict",
        )
        stdout_read = -1
        stderr_stream = os.fdopen(
            stderr_read,
            "r",
            encoding="utf-8",
            errors="strict",
        )
        stderr_read = -1
        process = _GatedTextProcess(
            pid=pid,
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
        state.process = process
        state.command_descriptor = command_write
        command_write = -1
        state.status_descriptor = status_read
        status_read = -1
        marker = _read_gated_launch_byte(
            state.status_descriptor,
            deadline=handshake_deadline,
            deferred=deferred,
            context="gated child ownership handshake",
        )
        if marker == b"F":
            _raise_gated_child_failure(
                state.status_descriptor,
                deadline=handshake_deadline,
                deferred=deferred,
            )
        require(marker == b"H", "Gated child ownership handshake differs")
        require(
            os.getpgid(pid) == pid and os.getsid(pid) == pid,
            "Gated child session or process group differs from its PID",
        )
        group_verified = True
        state.phase = _AttestedLaunchPhase.GROUP_VERIFIED
    except BaseException as error:
        primary_error = error

    def settle_owned_child() -> None:
        nonlocal child_settled
        require(primary_error is not None, "Pre-exec settlement lacks an error")
        while not child_settled:
            try:
                _settle_pre_exec_gated_child(
                    pid,
                    group_verified=group_verified,
                )
                child_settled = True
                if process is not None:
                    process.returncode = sys.maxsize
            except BaseException as cleanup_error:
                _add_secondary_exception_note_noexcept(
                    primary_error,
                    "Pre-exec gated cleanup retry failed",
                    cleanup_error,
                )
                try:
                    time.sleep(ATTESTATION_STOP_POLL_SECONDS)
                except BaseException as sleep_error:
                    _add_secondary_exception_note_noexcept(
                        primary_error,
                        "Pre-exec gated cleanup retry sleep failed",
                        sleep_error,
                    )

    if primary_error is not None and pid > 0:
        settle_owned_child()

    if previous_mask is not None:
        restoration_failure_recorded = False
        while True:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                break
            except BaseException as restoration_error:
                if primary_error is None:
                    primary_error = restoration_error
                elif not restoration_failure_recorded:
                    _add_secondary_exception_note_noexcept(
                        primary_error,
                        "Gated launch signal-mask restoration failed",
                        restoration_error,
                    )
                restoration_failure_recorded = True
                if pid > 0 and not child_settled:
                    settle_owned_child()
                try:
                    time.sleep(ATTESTATION_STOP_POLL_SECONDS)
                except BaseException as sleep_error:
                    _add_secondary_exception_note_noexcept(
                        primary_error,
                        "Gated launch signal-mask restoration retry sleep failed",
                        sleep_error,
                    )

    if primary_error is not None and pid > 0 and not child_settled:
        settle_owned_child()

    if primary_error is not None:
        try:
            _close_gated_launch_control(state)
        except BaseException as cleanup_error:
            _add_secondary_exception_note_noexcept(
                primary_error,
                "Gated launch control cleanup failed",
                cleanup_error,
            )
        for name, stream in (
            ("stdout", stdout_stream),
            ("stderr", stderr_stream),
        ):
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as cleanup_error:
                _add_secondary_exception_note_noexcept(
                    primary_error,
                    "Gated launch stream cleanup failed",
                    cleanup_error,
                )

    for descriptor in (
        cwd_descriptor,
        devnull_descriptor,
        command_read,
        command_write,
        status_read,
        status_write,
        stdout_read,
        stdout_write,
        stderr_read,
        stderr_write,
    ):
        _close_descriptor_quietly(descriptor)

    if primary_error is not None:
        raise primary_error
    require(
        process is not None,
        "Gated attested launch returned without an owned process",
    )
    return process


def _read_gated_launch_byte(
    descriptor: int,
    *,
    deadline: float,
    deferred: _DeferredSigintState,
    context: str,
) -> bytes:
    while True:
        if deferred.requested:
            raise _deferred_signal_exception(deferred)
        remaining = deadline - time.monotonic()
        require(remaining > 0, f"Timed out during {context}")
        try:
            readable, _, _ = select.select(
                [descriptor],
                [],
                [],
                min(ATTESTATION_STOP_POLL_SECONDS, remaining),
            )
        except InterruptedError:
            continue
        if not readable:
            continue
        return os.read(descriptor, 1)


def _read_gated_launch_exact(
    descriptor: int,
    size: int,
    *,
    deadline: float,
    deferred: _DeferredSigintState,
    context: str,
) -> bytes:
    parts: list[bytes] = []
    remaining_bytes = size
    while remaining_bytes:
        first = _read_gated_launch_byte(
            descriptor,
            deadline=deadline,
            deferred=deferred,
            context=context,
        )
        require(first != b"", f"Gated child truncated {context}")
        parts.append(first)
        remaining_bytes -= 1
    return b"".join(parts)


def _raise_gated_child_failure(
    descriptor: int,
    *,
    deadline: float,
    deferred: _DeferredSigintState,
) -> None:
    detail = _read_gated_launch_exact(
        descriptor,
        5,
        deadline=deadline,
        deferred=deferred,
        context="gated child failure report",
    )
    stage = detail[0]
    error_number = int.from_bytes(detail[1:], "little", signed=True)
    message = os.strerror(error_number) if error_number > 0 else "unknown error"
    raise D0Failure(
        f"Gated child failed before exec at stage {stage}: {message}"
    )


def _complete_gated_attested_exec(
    state: _GatedLaunchState,
    *,
    deadline: float,
    deferred: _DeferredSigintState,
    expected_wrapper_executable: dict[str, Any],
) -> None:
    require(
        state.phase == _AttestedLaunchPhase.GROUP_VERIFIED
        and state.process is not None
        and state.command_descriptor >= 0
        and state.status_descriptor >= 0,
        "Gated attested launch is not ready for exec",
    )
    if deferred.requested:
        raise _deferred_signal_exception(deferred)
    require(
        os.write(state.command_descriptor, b"E") == 1,
        "Could not release the gated child for exec",
    )
    _close_descriptor_quietly(state.command_descriptor)
    state.command_descriptor = -1
    state.phase = _AttestedLaunchPhase.EXEC_RELEASED

    marker = _read_gated_launch_byte(
        state.status_descriptor,
        deadline=deadline,
        deferred=deferred,
        context="gated child exec-ready confirmation",
    )
    if marker == b"F":
        _raise_gated_child_failure(
            state.status_descriptor,
            deadline=deadline,
            deferred=deferred,
        )
    require(marker == b"R", "Gated child exec-ready confirmation differs")
    marker = _read_gated_launch_byte(
        state.status_descriptor,
        deadline=deadline,
        deferred=deferred,
        context="gated child exec confirmation",
    )
    if marker == b"F":
        _raise_gated_child_failure(
            state.status_descriptor,
            deadline=deadline,
            deferred=deferred,
        )
    require(marker == b"", "Gated child emitted an invalid exec confirmation")
    _close_descriptor_quietly(state.status_descriptor)
    state.status_descriptor = -1

    while True:
        if deferred.requested:
            raise _deferred_signal_exception(deferred)
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Timed out verifying the gated exec identity")
        try:
            process = capture_process_identity(state.process.pid)
        except (D0Failure, OSError):
            process = {}
        if (
            process.get("pid") == state.process.pid
            and process.get("parent_pid") == os.getpid()
            and process.get("process_group") == state.process.pid
            and _process_executable_from_snapshot(process)
            == expected_wrapper_executable
            and process.get("status") not in {"creating", "zombie"}
        ):
            state.phase = _AttestedLaunchPhase.EXEC_CONFIRMED
            return
        observation = _observe_attested_terminal_nonconsuming(
            state.process.pid
        )
        require(
            observation is None,
            "Gated child terminated before exec identity verification",
        )
        time.sleep(min(ATTESTATION_STOP_POLL_SECONDS, remaining))


def _close_gated_launch_control(state: _GatedLaunchState) -> None:
    _close_descriptor_quietly(state.command_descriptor)
    _close_descriptor_quietly(state.status_descriptor)
    state.command_descriptor = -1
    state.status_descriptor = -1


def launch_and_supervise_attested_process(
    command: list[str],
    *,
    launch_cwd: Path,
    environment: Mapping[str, str],
    cwd: Path,
    timeout_seconds: float,
    expected_wrapper_executable: dict[str, Any],
    expected_benchmark_executable: dict[str, Any],
) -> dict[str, Any]:
    require(
        bool(command) and all(type(argument) is str for argument in command),
        "Attested launch command is invalid",
    )
    require(
        Path(command[0]).is_absolute()
        and Path(command[0]).is_file()
        and launch_cwd.is_absolute()
        and launch_cwd.is_dir(),
        "Attested launch paths are invalid",
    )
    require(
        all(
            type(name) is str
            and name != ""
            and "=" not in name
            and type(value) is str
            and "\0" not in value
            for name, value in environment.items()
        ),
        "Attested launch environment is invalid",
    )
    require(timeout_seconds > 0, "Attested process timeout must be positive")
    require(
        threading.current_thread() is threading.main_thread(),
        "Attested process launch must run on the main thread",
    )
    original_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in ATTESTATION_DEFERRED_SIGNALS
    }
    require(
        original_handlers[signal.SIGINT] is signal.default_int_handler
        and all(
            original_handlers[signal_number]
            in {signal.SIG_DFL, signal.SIG_IGN}
            for signal_number in ATTESTATION_DEFERRED_SIGNALS
            if signal_number != signal.SIGINT
        ),
        "Attested process launch requires default termination-signal handlers",
    )
    deferred_errors: dict[int, BaseException] = {
        signal.SIGINT: KeyboardInterrupt(),
        **{
            signal_number: SystemExit(128 + int(signal_number))
            for signal_number in ATTESTATION_DEFERRED_SIGNALS
            if signal_number != signal.SIGINT
        },
    }
    deferred = _DeferredSigintState(
        interruption=deferred_errors[signal.SIGINT],
        interruptions=deferred_errors,
    )

    def defer_termination(
        signal_number: int,
        frame: Any,
    ) -> None:
        del frame
        if signal_number not in deferred_errors or deferred.requested:
            return
        deferred.request = (
            signal_number,
            deferred_errors[signal_number],
        )

    launch_state = _GatedLaunchState()
    process: _GatedTextProcess | None = None
    supervision: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    deadline = time.monotonic() + timeout_seconds
    installed_handlers: list[int] = []
    try:
        for signal_number in ATTESTATION_DEFERRED_SIGNALS:
            if original_handlers[signal_number] == signal.SIG_IGN:
                continue
            signal.signal(signal_number, defer_termination)
            installed_handlers.append(signal_number)
        process = _fork_gated_attested_process(
            command,
            launch_cwd=launch_cwd,
            environment=environment,
            state=launch_state,
            deadline=deadline,
            deferred=deferred,
        )
        try:
            _complete_gated_attested_exec(
                launch_state,
                deadline=deadline,
                deferred=deferred,
                expected_wrapper_executable=expected_wrapper_executable,
            )
            launch_error: BaseException | None = None
        except BaseException as error:
            launch_error = error
        finally:
            _close_gated_launch_control(launch_state)
        try:
            remaining_timeout = max(
                ATTESTATION_STOP_POLL_SECONDS,
                deadline - time.monotonic(),
            )
        except BaseException as error:
            if launch_error is None:
                launch_error = error
            else:
                _add_secondary_exception_note_noexcept(
                    launch_error,
                    "Attested launch timeout calculation failed",
                    error,
                )
            remaining_timeout = ATTESTATION_STOP_POLL_SECONDS
        supervision = supervise_attested_process(
            process,
            cwd=cwd,
            timeout_seconds=remaining_timeout,
            expected_wrapper_executable=expected_wrapper_executable,
            expected_benchmark_executable=expected_benchmark_executable,
            _deferred_sigint_state=deferred,
            _initial_error=launch_error,
        )
    except BaseException as error:
        primary_error = error
        if process is None:
            process = launch_state.process
        if process is not None and process.returncode is None:
            try:
                supervise_attested_process(
                    process,
                    cwd=cwd,
                    timeout_seconds=ATTESTATION_POST_KILL_WAIT_SECONDS,
                    expected_wrapper_executable=expected_wrapper_executable,
                    expected_benchmark_executable=expected_benchmark_executable,
                    _deferred_sigint_state=deferred,
                    _initial_error=primary_error,
                )
            except BaseException as settlement_error:
                if settlement_error is not primary_error:
                    _add_secondary_exception_note_noexcept(
                        primary_error,
                        "Attested launcher fallback settlement failed",
                        settlement_error,
                    )
    finally:
        _close_gated_launch_control(launch_state)
        for signal_number in reversed(installed_handlers):
            try:
                signal.signal(
                    signal_number,
                    original_handlers[signal_number],
                )
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                else:
                    _add_secondary_exception_note_noexcept(
                        primary_error,
                        "Attested process signal-handler restoration failed",
                        error,
                    )

    if primary_error is not None:
        if (
            deferred.requested
            and _deferred_signal_exception(deferred) is not primary_error
        ):
            _add_exception_note_noexcept(
                primary_error,
                f"Signal {deferred.signal_number} was deferred during "
                "attested process launch",
            )
        raise primary_error
    require(
        process is not None
        and supervision is not None
        and process.returncode is not None,
        "Attested process supervision returned without a settled child",
    )
    if deferred.requested:
        raise _deferred_signal_exception(deferred)
    return {
        **supervision,
        "pid": process.pid,
        "returncode": process.returncode,
    }


def index_timing_binding(
    barriers: list[dict[str, Any]],
    fields: dict[str, str],
    *,
    engine: str,
) -> dict[str, Any]:
    require(
        len(barriers) == len(ATTESTATION_BARRIER_PHASES),
        "Index timing binding requires all four attestation barriers",
    )
    before_index = barriers[1]
    after_index = barriers[2]
    require(
        before_index.get("phase") == "before-index"
        and after_index.get("phase") == "after-index",
        "Index timing barriers are missing or out of order",
    )
    outer_start = before_index.get("resumed_at_monotonic_ns")
    outer_end = after_index.get("observed_at_monotonic_ns")
    require(
        type(outer_start) is int and outer_start > 0,
        "Index timing outer start is invalid",
    )
    require(
        type(outer_end) is int and outer_end >= outer_start,
        "Index timing outer end is invalid",
    )
    outer_ns = outer_end - outer_start
    cold_build_ns = parse_unsigned_field(fields, f"{engine}_cold_build_ns")
    require(
        cold_build_ns <= outer_ns,
        "Reported cold-build duration exceeds the parent-observed index window",
    )
    boundary_overhead_ns = outer_ns - cold_build_ns
    require(
        boundary_overhead_ns <= INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS,
        "Parent-observed index-window overhead exceeds the protocol maximum",
    )
    return {
        "schema_version": INDEX_TIMING_BINDING_SCHEMA_VERSION,
        "clock": "python-time-monotonic-ns",
        "outer_start_phase": "before-index.resumed",
        "outer_end_phase": "after-index.observed",
        "outer_start_monotonic_ns": outer_start,
        "outer_end_monotonic_ns": outer_end,
        "outer_index_window_ns": outer_ns,
        "reported_cold_build_ns": cold_build_ns,
        "boundary_overhead_ns": boundary_overhead_ns,
        "maximum_boundary_overhead_ns": (
            INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
        ),
    }


def run_member(
    entry: dict[str, Any],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    binary_hashes: dict[str, str],
    campaign_binding: dict[str, Any],
    runtime_artifact_hashes: dict[str, str],
    dataset_path: Path,
    dataset: dict[str, Any],
    truth: dict[str, Any],
) -> dict[str, Any]:
    sequence = int(entry["sequence"])
    engine = str(entry["engine"])
    prefix = f"{sequence:02d}-{entry['pair']}-{engine}"
    binary = binary_for_engine(args, engine)
    member_binding = member_campaign_binding(entry, campaign_binding)
    require(
        member_binding["binary_sha256"] == binary_hashes[engine],
        f"{engine} campaign binary differs from preflight",
    )
    require(
        member_binding["dataset_sha256"] == dataset["sha256"],
        "Campaign dataset differs from the member dataset",
    )
    sidecar_name = audit_sidecar_path(prefix)
    sidecar_path = output_dir / sidecar_name
    require(
        not sidecar_path.exists() and not sidecar_path.is_symlink(),
        f"Audit sidecar path already exists before {prefix}",
    )

    require(not dataset_path.is_symlink(), "Benchmark dataset must not be a symlink")
    require(
        dataset_path.resolve()
        == (output_dir.resolve() / str(dataset["path"])).resolve(),
        "Benchmark dataset is not the frozen file in the evidence directory",
    )
    dataset_before = verify_dataset(dataset_path, dataset)
    require(
        sha256(binary) == binary_hashes[engine],
        f"{engine} benchmark binary changed before {prefix}",
    )
    runtime_artifacts_before = verify_runtime_artifacts(runtime_artifact_hashes)
    host_before = capture_member_preflight(
        cwd=args.repo,
        output_dir=output_dir,
        prefix=prefix,
    )

    command = [
        "/usr/bin/time",
        "-lp",
        str(binary),
        dataset["path"],
        str(VECTORS),
        str(DIMENSIONS),
        str(M),
        str(EF_CONSTRUCTION),
        str(EF_SEARCH),
        str(CHUNK_SIZE),
        str(QUERY_COUNT),
        str(entry["participants"]),
        str(BUILD_GRAIN),
        sidecar_name,
        str(member_binding["campaign_nonce"]),
        str(member_binding["schedule_sequence"]),
        str(member_binding["role_id"]),
        str(member_binding["binary_sha256"]),
        str(member_binding["dataset_sha256"]),
    ]
    require(len(command) == 19, f"Frozen command length differs before {prefix}")
    environment = benchmark_environment(int(entry["participants"]))
    wrapper_process_identity = process_executable_identity(Path(command[0]))
    benchmark_process_identity = process_executable_identity(binary)

    idle = wait_for_idle(
        cwd=args.repo,
        output_dir=output_dir,
        prefix=prefix,
        timeout_seconds=args.idle_timeout_seconds,
        minimum_idle_percent=args.idle_minimum_percent,
    )
    require(
        idle_record_passes(
            idle,
            expected_minimum_idle_percent=args.idle_minimum_percent,
        ),
        f"CPU-idle preflight failed before {prefix}",
    )

    started_at = utc_now()
    started_ns = time.time_ns()
    idle_to_launch_ns = time.monotonic_ns() - int(idle["accepted_at_monotonic_ns"])
    require(
        0 <= idle_to_launch_ns <= IDLE_TO_LAUNCH_MAX_NS,
        f"Idle-to-launch delay is {idle_to_launch_ns} ns before {prefix}",
    )
    supervision = launch_and_supervise_attested_process(
        command,
        launch_cwd=output_dir,
        environment=environment,
        cwd=args.repo,
        timeout_seconds=args.member_timeout_seconds,
        expected_wrapper_executable=wrapper_process_identity,
        expected_benchmark_executable=benchmark_process_identity,
    )
    process_pid = int(supervision["pid"])
    process_returncode = int(supervision["returncode"])
    stdout = str(supervision["stdout"])
    stderr = str(supervision["stderr"])
    timed_out = bool(supervision["timed_out"])
    ended_ns = time.time_ns()

    stdout_name = f"{prefix}.stdout"
    stderr_name = f"{prefix}.stderr"
    (output_dir / stdout_name).write_text(stdout, encoding="utf-8")
    (output_dir / stderr_name).write_text(stderr, encoding="utf-8")

    errors: list[str] = []
    fields: dict[str, str] = {}
    attestation: dict[str, Any] = {}
    timing_binding: dict[str, Any] = {}
    errors.extend(str(error) for error in supervision["errors"])
    try:
        driver_stdout, attestation = split_attested_driver_output(stdout)
        fields = parse_driver_result(
            driver_stdout,
            engine=engine,
            participants=int(entry["participants"]),
            dataset=dataset,
            heldout=truth,
        )
        validate_driver_campaign_binding(
            fields,
            member_binding,
            context=prefix,
        )
        timing_binding = index_timing_binding(
            supervision["barriers"],
            fields,
            engine=engine,
        )
    except D0Failure as error:
        errors.append(str(error))
    audit: dict[str, Any] = {}
    sidecar_data: bytes | None = None
    if fields:
        try:
            sidecar_data = read_audit_sidecar(sidecar_path)
            audit = parse_audit_sidecar(
                sidecar_path,
                engine=engine,
                fields=fields,
                dataset=dataset,
                truth=truth,
                expected_campaign_nonce=str(member_binding["campaign_nonce"]),
                expected_schedule_sequence=int(
                    member_binding["schedule_sequence"]
                ),
                expected_role_id=int(member_binding["role_id"]),
                expected_binary_sha256=str(member_binding["binary_sha256"]),
                expected_dataset_sha256=str(member_binding["dataset_sha256"]),
                data=sidecar_data,
            )
        except D0Failure as error:
            errors.append(str(error))
    if process_returncode != 0:
        errors.append(f"process exited with code {process_returncode}")
    if timed_out:
        errors.append(f"process exceeded {args.member_timeout_seconds} second timeout")
    max_rss = parse_max_rss(stderr)
    if max_rss is None:
        errors.append("missing maximum resident set size")
    process_swaps = parse_process_swaps(stderr)
    if process_swaps is None:
        errors.append("missing process swap count")
    elif process_swaps != 0:
        errors.append(f"process reported {process_swaps} swaps")

    try:
        dataset_after = verify_dataset(dataset_path, dataset)
    except D0Failure as error:
        dataset_after = {"verified_at": utc_now(), "error": str(error)}
        errors.append(str(error))
    try:
        runtime_artifacts_after = verify_runtime_artifacts(runtime_artifact_hashes)
    except D0Failure as error:
        runtime_artifacts_after = {"verified_at": utc_now(), "error": str(error)}
        errors.append(str(error))
    swap_growth: int | None = None
    try:
        host_after = capture_member_postflight(
            cwd=args.repo,
            output_dir=output_dir,
            prefix=prefix,
        )
        swap_growth = int(host_after["swap"]["used_bytes"]) - int(
            host_before["swap"]["used_bytes"]
        )
    except D0Failure as error:
        host_after = {"captured_at": utc_now(), "error": str(error)}
        errors.append(str(error))

    sidecar_record: dict[str, Any] = {"path": sidecar_name}
    if sidecar_data is not None:
        sidecar_record.update(
            {
                "sha256": hashlib.sha256(sidecar_data).hexdigest(),
                "bytes": len(sidecar_data),
            }
        )

    record = {
        **entry,
        "status": "pass" if not errors else "fail",
        "scope": D0_SCOPE,
        "errors": errors,
        "pid": process_pid,
        "started_at": started_at,
        "started_at_unix_ns": started_ns,
        "ended_at_unix_ns": ended_ns,
        "command": command,
        "working_directory": ".",
        "binary": {
            "path": str(binary),
            "sha256": binary_hashes[engine],
        },
        "environment": environment,
        "idle": idle,
        "idle_to_launch_ns": idle_to_launch_ns,
        "host_before": host_before,
        "host_after": host_after,
        "global_swap_growth_bytes": swap_growth,
        "dataset_before": dataset_before,
        "dataset_after": dataset_after,
        "runtime_artifacts_before": runtime_artifacts_before,
        "runtime_artifacts_after": runtime_artifacts_after,
        "stdout_path": stdout_name,
        "stderr_path": stderr_name,
        "fields": fields,
        "attestation": attestation,
        "attestation_barriers": supervision["barriers"],
        "benchmark_process_evidence": supervision[
            "benchmark_process_evidence"
        ],
        "index_timing_binding": timing_binding,
        "audit_sidecar": sidecar_record,
        "audit": audit,
        "process_lifetime_maximum_resident_set_size": max_rss,
        "process_swaps": process_swaps,
    }
    (output_dir / f"{prefix}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    require(not errors, f"{prefix} failed validation: {errors}")
    return record


def relative_mad(values: list[float | int]) -> float:
    require(bool(values), "Cannot calculate relative MAD of an empty sample")
    median = statistics.median(values)
    require(median > 0, "Cannot calculate relative MAD with a zero median")
    return statistics.median(abs(value - median) for value in values) / median


def geometric_mean(values: list[float]) -> float:
    require(bool(values), "Cannot calculate geometric mean of an empty sample")
    require(
        all(value > 0 and math.isfinite(value) for value in values),
        "Invalid geometric-mean input",
    )
    return math.exp(statistics.mean(math.log(value) for value in values))


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
        f"{pair} {engine} has no unique recall point k={k} ef={ef}",
    )
    point = matches[0]
    require(
        set(point)
        == {"k", "ef", "eligible_hits", "possible_hits", "recall"},
        f"{pair} {engine} recall point keys differ k={k} ef={ef}",
    )
    eligible_hits = point.get("eligible_hits")
    possible_hits = point.get("possible_hits")
    require(
        type(eligible_hits) is int
        and type(possible_hits) is int
        and 0 <= eligible_hits <= possible_hits
        and possible_hits == HELDOUT_QUERY_COUNT * k,
        f"{pair} {engine} recall hit counts differ k={k} ef={ef}",
    )
    recall = Fraction(eligible_hits, possible_hits)
    recorded_recall = point.get("recall")
    require(
        type(recorded_recall) is float
        and math.isfinite(recorded_recall)
        and recorded_recall == float(recall),
        f"{pair} {engine} recall float differs from hit counts k={k} ef={ef}",
    )
    return recall


def summarize(
    records: list[dict[str, Any]],
    *,
    dataset: dict[str, Any],
    schedule_sha256: str,
    idle_minimum_percent: float = IDLE_MINIMUM_PERCENT,
) -> dict[str, Any]:
    validate_records_against_schedule(records)
    validate_run_chronology(records)
    measured = [record for record in records if record["treatment"] == "measured"]
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    durations: dict[str, list[int]] = {"infinity": [], "faiss": []}
    for record in measured:
        engine = str(record["engine"])
        pair = str(record["pair"])
        require(
            engine not in by_pair.setdefault(pair, {}),
            f"Duplicate {engine} member in {pair}",
        )
        by_pair[pair][engine] = record
        durations[engine].append(
            parse_unsigned_field(
                record["fields"],
                f"{engine}_cold_build_ns",
            )
        )

    ingestion_pairs: list[dict[str, Any]] = []
    for pair in ("P1", "P2", "P3", "P4", "P5", "P6"):
        members = by_pair.get(pair, {})
        require(set(members) == {"infinity", "faiss"}, f"Incomplete pair {pair}")
        infinity_ns = parse_unsigned_field(
            members["infinity"]["fields"],
            "infinity_cold_build_ns",
        )
        faiss_ns = parse_unsigned_field(
            members["faiss"]["fields"],
            "faiss_cold_build_ns",
        )
        require(
            members["infinity"]["pair_order"] == members["faiss"]["pair_order"],
            f"Pair-order disagreement in {pair}",
        )
        ingestion_pairs.append(
            {
                "pair": pair,
                "order": members["infinity"]["pair_order"],
                "infinity_cold_build_ns": infinity_ns,
                "faiss_cold_build_ns": faiss_ns,
                "infinity_vectors_per_second": VECTORS * 1e9 / infinity_ns,
                "faiss_vectors_per_second": VECTORS * 1e9 / faiss_ns,
                "ratio": faiss_ns / infinity_ns,
            }
        )

    strata: dict[str, Any] = {}
    for order in ("infinity/faiss", "faiss/infinity"):
        ratios = [pair["ratio"] for pair in ingestion_pairs if pair["order"] == order]
        require(len(ratios) == 3, f"Order stratum {order} does not contain three pairs")
        strata[order] = {
            "ratios": ratios,
            "geometric_mean_ratio": geometric_mean(ratios),
        }
    estimate = math.sqrt(
        strata["infinity/faiss"]["geometric_mean_ratio"]
        * strata["faiss/infinity"]["geometric_mean_ratio"]
    )
    ingestion_engines = {
        engine: {
            "wall_time_ns": values,
            "vectors_per_second": [VECTORS * 1e9 / value for value in values],
            "median_wall_time_ns": statistics.median(values),
            "median_vectors_per_second": statistics.median(
                VECTORS * 1e9 / value for value in values
            ),
            "relative_mad": relative_mad(values),
        }
        for engine, values in durations.items()
    }
    variability_accepted = all(
        details["relative_mad"] <= 0.10 for details in ingestion_engines.values()
    )
    ratio_relative_mad = relative_mad(
        [pair["ratio"] for pair in ingestion_pairs]
    )
    ratio_variability_accepted = ratio_relative_mad <= 0.10
    all_runs_valid = all(record["status"] == "pass" for record in records)
    idle_protocol_met = all(
        idle_record_passes(
            record["idle"],
            expected_minimum_idle_percent=idle_minimum_percent,
        )
        for record in records
    )
    idle_to_launch_met = all(
        isinstance(record.get("idle_to_launch_ns"), int)
        and 0 <= record["idle_to_launch_ns"] <= IDLE_TO_LAUNCH_MAX_NS
        for record in records
    )
    host_resource_protocol_met = all(
        record.get("process_swaps") == 0
        and type(record["host_before"].get("battery_percent")) is int
        and BATTERY_MINIMUM_PERCENT <= record["host_before"]["battery_percent"] <= 100
        and type(record["host_after"].get("battery_percent")) is int
        and BATTERY_MINIMUM_PERCENT <= record["host_after"]["battery_percent"] <= 100
        and record["host_before"].get("low_power_mode") is False
        and record["host_after"].get("low_power_mode") is False
        and record["host_before"].get("thermal_nominal") is True
        and record["host_after"].get("thermal_nominal") is True
        and isinstance(record.get("global_swap_growth_bytes"), int)
        for record in records
    )
    dataset_integrity_met = all(
        record["dataset_before"].get("sha256") == dataset["sha256"]
        and record["dataset_after"].get("sha256") == dataset["sha256"]
        and record["dataset_before"].get("fnv1a64") == dataset["fnv1a64"]
        and record["dataset_after"].get("fnv1a64") == dataset["fnv1a64"]
        for record in records
    )
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
    graph_members = [
        {
            "sequence": record["sequence"],
            "pair": record["pair"],
            "engine": record["engine"],
            **record["audit"]["graph"],
        }
        for record in records
    ]
    raw_graph_audit_met = (
        len(graph_members) == len(records)
        and all(
            member["valid"]
            and member["vertex_count"] == VECTORS
            and member["reachable_count"] == VECTORS
            for member in graph_members
        )
    )
    recall_points: list[dict[str, Any]] = []
    for k, ef in RECALL_POINTS:
        paired_deficits: list[dict[str, Any]] = []
        infinity_values: list[float] = []
        faiss_values: list[float] = []
        deficit_fractions: list[Fraction] = []
        for pair in ("P1", "P2", "P3", "P4", "P5", "P6"):
            members = by_pair[pair]
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
            paired_deficits.append(
                {
                    "pair": pair,
                    "infinity": recalls["infinity"],
                    "faiss": recalls["faiss"],
                    "infinity_deficit": float(deficit_fraction),
                    "infinity_deficit_exact": fraction_record(deficit_fraction),
                }
            )
        maximum_deficit_fraction = max(Fraction(0), max(deficit_fractions))
        recall_points.append(
            {
                "k": k,
                "ef": ef,
                "infinity_median": statistics.median(infinity_values),
                "faiss_median": statistics.median(faiss_values),
                "paired": paired_deficits,
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
    heldout_recall_audit_met = all(
        record["audit"]["recall"]["query_count"] == HELDOUT_QUERY_COUNT
        and record["audit"]["recall"]["query_seed"] == HELDOUT_QUERY_SEED
        and len(record["audit"]["recall"]["points"]) == len(RECALL_POINTS)
        for record in records
    )
    recall_parity_all_points = all(
        point["deficit_at_most_0_005"] for point in recall_points
    )

    query_pairs: list[dict[str, Any]] = []
    query_values: dict[str, dict[str, list[float | int]]] = {
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
    for pair in ("P1", "P2", "P3", "P4", "P5", "P6"):
        members = by_pair[pair]
        pair_query: dict[str, dict[str, float | int]] = {}
        pair_recall_fractions: dict[str, Fraction] = {}
        for engine in ("infinity", "faiss"):
            measured_query = members[engine]["audit"]["query"]
            throughput_wall_ns = measured_query.get("throughput_wall_ns")
            qps = measured_query.get("qps")
            tps = measured_query.get("tps")
            p50_ns = measured_query.get("p50_ns")
            p95_ns = measured_query.get("p95_ns")
            p99_ns = measured_query.get("p99_ns")
            require(
                type(throughput_wall_ns) is int
                and type(qps) is float
                and type(tps) is float
                and type(p50_ns) is int
                and type(p95_ns) is int
                and type(p99_ns) is int,
                f"{pair} {engine} query metric types differ",
            )
            recall_fraction = recall_point_fraction(
                members[engine],
                QUERY_K,
                QUERY_EF_SEARCH,
            )
            pair_recall_fractions[engine] = recall_fraction
            details: dict[str, float | int] = {
                "throughput_wall_ns": throughput_wall_ns,
                "qps": qps,
                "tps": tps,
                "p50_ns": p50_ns,
                "p95_ns": p95_ns,
                "p99_ns": p99_ns,
                "recall_at_10": float(recall_fraction),
            }
            require(
                details["p50_ns"] > 0
                and details["p50_ns"] <= details["p95_ns"] <= details["p99_ns"],
                f"{pair} {engine} query latency percentiles are invalid",
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
                "pair": pair,
                "order": members["infinity"]["pair_order"],
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

    query_strata: dict[str, Any] = {}
    for order in ("infinity/faiss", "faiss/infinity"):
        ratios = [
            pair["ratios"]["qps"] for pair in query_pairs if pair["order"] == order
        ]
        require(
            len(ratios) == 3,
            f"Query order stratum {order} does not contain three pairs",
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
        query_engines[engine] = {
            **values,
            "median_throughput_wall_ns": statistics.median(
                values["throughput_wall_ns"]
            ),
            "median_qps": statistics.median(values["qps"]),
            "median_tps": statistics.median(values["tps"]),
            "median_p50_ns": statistics.median(values["p50_ns"]),
            "median_p95_ns": statistics.median(values["p95_ns"]),
            "median_p99_ns": statistics.median(values["p99_ns"]),
            "median_recall_at_10": statistics.median(values["recall_at_10"]),
            "qps_relative_mad": relative_mad(values["qps"]),
        }
    query_qps_ratios = [pair["ratios"]["qps"] for pair in query_pairs]
    query_ratio_relative_mad = relative_mad(query_qps_ratios)
    query_qps_variability_accepted = all(
        details["qps_relative_mad"] <= 0.10
        for details in query_engines.values()
    )
    query_ratio_variability_accepted = query_ratio_relative_mad <= 0.10
    query_recall_floor_met = all(
        pair["recall_floor_met"] for pair in query_pairs
    )
    query_recall_gap_met = all(pair["recall_gap_met"] for pair in query_pairs)
    query_audit_met = all(
        record["audit"]["query"]["schema_version"]
        == QUERY_BENCHMARK_SCHEMA_VERSION
        and record["audit"]["query"]["timed_query_corpus_sha256"]
        == record["audit"]["recall"]["queries_sha256"]
        and len(record["audit"]["query"]["latency_samples_ns"])
        == QUERY_LATENCY_SAMPLE_COUNT
        and record["audit"]["query"]["latency_validated_operations"]
        == QUERY_LATENCY_SAMPLE_COUNT
        and record["audit"]["query"]["throughput_operations"]
        == QUERY_THROUGHPUT_OPERATIONS
        and record["audit"]["query"]["throughput_validated_operations"]
        == QUERY_THROUGHPUT_OPERATIONS
        and record["audit"]["query"]["per_query_checksum_count"]
        == HELDOUT_QUERY_COUNT
        and len(record["audit"]["query"]["per_query_checksums"])
        == HELDOUT_QUERY_COUNT
        and record["audit"]["query"]["latency_validated_result_checksum"]
        == record["audit"]["query"]["throughput_validated_result_checksum"]
        == record["audit"]["query"]["result_checksum"]
        == query_benchmark_checksum(
            record["audit"]["query"]["per_query_checksums"]
        )
        for record in records
    )
    query_summary = {
        "protocol": query_protocol(),
        "pairs": query_pairs,
        "strata": query_strata,
        "stratified_geometric_mean_qps_ratio": query_estimate,
        "qps_ratio_range": [
            min(query_qps_ratios),
            max(query_qps_ratios),
        ],
        "engines": query_engines,
        "paired_qps_ratio_relative_mad": query_ratio_relative_mad,
        "recall_floor_met": query_recall_floor_met,
        "absolute_recall_gap_met": query_recall_gap_met,
    }
    accepted = (
        all_runs_valid
        and variability_accepted
        and ratio_variability_accepted
        and query_qps_variability_accepted
        and query_ratio_variability_accepted
        and query_audit_met
        and query_recall_floor_met
        and query_recall_gap_met
        and idle_protocol_met
        and idle_to_launch_met
        and host_resource_protocol_met
        and dataset_integrity_met
        and raw_graph_audit_met
        and heldout_recall_audit_met
        and recall_parity_all_points
    )
    limitations = [
        "The workload is deterministic synthetic float32 data, not a canonical dataset.",
        "Held-out Recall@10/100 is a D0 quality audit, not a canonical workload result.",
        QUERY_SCOPE_LIMITATION,
        "No peak-active-worker or live-index-byte audit is present.",
        INDEXING_WINDOW_RSS_LIMITATION,
        "This development-only result cannot be relabeled as confirmatory evidence.",
    ]
    if not recall_parity_all_points:
        limitations.append(
            "Infinity exceeds the 0.005 paired recall-deficit target at one or "
            "more diagnostic search breadths."
        )
    if idle_minimum_percent < IDLE_MINIMUM_PERCENT:
        limitations.append(
            f"A development-only {idle_minimum_percent:g}% CPU-idle threshold "
            f"replaced the default {IDLE_MINIMUM_PERCENT:g}% threshold; this "
            "campaign is suitable for iteration, not qualification."
        )
    limitations.append(
        "System-wide swap growth was recorded but not used as a "
        "development acceptance gate; each benchmark process still had "
        "to report zero swaps."
    )
    swap_growth_values = [int(record["global_swap_growth_bytes"]) for record in records]
    return {
        "status": "pass" if accepted else "fail",
        "scope": D0_SCOPE,
        "claim_eligible": False,
        "power": {
            "sources": power_sources,
            "policies": power_policies,
        },
        "idle": {
            "minimum_percent": idle_minimum_percent,
            "window_seconds": IDLE_WINDOW_SECONDS,
            "policy": idle_policy(idle_minimum_percent),
        },
        "resources": {
            "global_swap_policy": GLOBAL_SWAP_POLICY,
            "maximum_observed_global_swap_growth_bytes": max(
                0,
                max(swap_growth_values),
            ),
            "indexing_window_peak_rss_established": False,
        },
        "quality": {
            "graph": {
                "all_valid_and_reachable": raw_graph_audit_met,
                "members": graph_members,
            },
            "recall": {
                "query_count": HELDOUT_QUERY_COUNT,
                "query_seed": HELDOUT_QUERY_SEED,
                "maximum_paired_deficit": MAXIMUM_PAIRED_RECALL_DEFICIT,
                "maximum_paired_deficit_exact": fraction_record(
                    Fraction(*MAXIMUM_PAIRED_RECALL_DEFICIT_FRACTION)
                ),
                "all_points_within_deficit": recall_parity_all_points,
                "points": recall_points,
            },
        },
        "workload": {
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
        },
        "ingestion": {
            "pairs": ingestion_pairs,
            "strata": strata,
            "stratified_geometric_mean_ratio": estimate,
            "ratio_range": [
                min(pair["ratio"] for pair in ingestion_pairs),
                max(pair["ratio"] for pair in ingestion_pairs),
            ],
            "engines": ingestion_engines,
            "paired_ratio_relative_mad": ratio_relative_mad,
        },
        "query": query_summary,
        "acceptance": {
            "all_runs_valid": all_runs_valid,
            "ingestion_relative_mad_at_most_0_10": variability_accepted,
            "ingestion_paired_ratio_relative_mad_at_most_0_10": (
                ratio_variability_accepted
            ),
            "query_qps_relative_mad_at_most_0_10": (
                query_qps_variability_accepted
            ),
            "query_paired_qps_ratio_relative_mad_at_most_0_10": (
                query_ratio_variability_accepted
            ),
            "query_evidence_complete": query_audit_met,
            "query_recall_floor_met": query_recall_floor_met,
            "query_absolute_recall_gap_at_most_0_005": query_recall_gap_met,
            "idle_protocol_met": idle_protocol_met,
            "idle_to_launch_at_most_1_second": idle_to_launch_met,
            "host_resource_protocol_met": host_resource_protocol_met,
            "dataset_integrity_met": dataset_integrity_met,
            "raw_graph_audit_met": raw_graph_audit_met,
            "heldout_recall_audit_met": heldout_recall_audit_met,
            "paired_recall_deficit_at_most_0_005": recall_parity_all_points,
            "exact_balanced_schedule_met": True,
        },
        "limitations": limitations,
    }


def render_summary(summary: dict[str, Any]) -> str:
    status = (
        "accepted development diagnostic"
        if summary["status"] == "pass"
        else "failed development diagnostic"
    )
    lines = [
        "# Native HNSW D0 Diagnostic",
        "",
        f"Status: {status}",
        "",
        f"Scope: `{D0_SCOPE}`. This result is not eligible for the 1.50x claim.",
        "",
        "Observed power sources: "
        f"{', '.join(summary['power']['sources'])}; policies: "
        f"{', '.join(summary['power']['policies'])}.",
        "",
        "| Pair | Order | Infinity vectors/s | FAISS vectors/s | Ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in summary["ingestion"]["pairs"]:
        lines.append(
            f"| {pair['pair']} | {pair['order']} | "
            f"{pair['infinity_vectors_per_second']:.1f} | "
            f"{pair['faiss_vectors_per_second']:.1f} | "
            f"{pair['ratio']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "Stratified geometric mean: "
            f"{summary['ingestion']['stratified_geometric_mean_ratio']:.4f}x",
            "",
            "Ratio range: "
            f"{summary['ingestion']['ratio_range'][0]:.4f}x to "
            f"{summary['ingestion']['ratio_range'][1]:.4f}x",
            "",
            "Relative MAD: "
            f"Infinity {summary['ingestion']['engines']['infinity']['relative_mad']:.4%}; "
            f"FAISS {summary['ingestion']['engines']['faiss']['relative_mad']:.4%}; "
            f"paired ratio {summary['ingestion']['paired_ratio_relative_mad']:.4%}.",
            "",
            "## Query",
            "",
            "| Pair | Infinity QPS | FAISS QPS | QPS ratio | Infinity p50/p95/p99 ns | FAISS p50/p95/p99 ns |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in summary["query"]["pairs"]:
        infinity_query = pair["infinity"]
        faiss_query = pair["faiss"]
        lines.append(
            f"| {pair['pair']} | {infinity_query['qps']:.1f} | "
            f"{faiss_query['qps']:.1f} | {pair['ratios']['qps']:.4f}x | "
            f"{infinity_query['p50_ns']}/{infinity_query['p95_ns']}/{infinity_query['p99_ns']} | "
            f"{faiss_query['p50_ns']}/{faiss_query['p95_ns']}/{faiss_query['p99_ns']} |"
        )
    lines.extend(
        [
            "",
            "Query stratified geometric-mean QPS ratio: "
            f"{summary['query']['stratified_geometric_mean_qps_ratio']:.4f}x.",
            "",
            "Query recall floor met: "
            f"{summary['query']['recall_floor_met']}; absolute paired recall-gap gate met: "
            f"{summary['query']['absolute_recall_gap_met']}.",
            "",
            f"Recorded idle protocol "
            f"({summary['idle']['minimum_percent']:g}% minimum for "
            f"{summary['idle']['window_seconds']} seconds, "
            f"{summary['idle']['policy']}) met: "
            f"{summary['acceptance']['idle_protocol_met']}.",
            "",
            "All accepted idle gates launched within one second: "
            f"{summary['acceptance']['idle_to_launch_at_most_1_second']}.",
            "",
            "Power, thermal, and zero benchmark-process swap protocol met: "
            f"{summary['acceptance']['host_resource_protocol_met']}.",
            "",
            "System-wide swap policy: "
            f"{summary['resources']['global_swap_policy']}; maximum observed "
            "growth "
            f"{summary['resources']['maximum_observed_global_swap_growth_bytes']} "
            "bytes.",
            "",
            "Raw graph audit valid and fully reachable for every member: "
            f"{summary['quality']['graph']['all_valid_and_reachable']}.",
            "",
            "All held-out recall points within the 0.005 paired deficit target: "
            f"{summary['quality']['recall']['all_points_within_deficit']}.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    return "\n".join(lines) + "\n"


def validate_completed_campaign(summary: dict[str, Any]) -> None:
    require(
        summary.get("status") in ("pass", "fail"),
        "Completed campaign has an invalid summary status",
    )


def parse_cmake_cache(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        name_and_type, value = line.split("=", 1)
        if ":" not in name_and_type:
            continue
        name, _ = name_and_type.split(":", 1)
        values[name] = value
    return values


def configured_compiler_paths(
    cache: Mapping[str, str],
    *,
    label: str,
) -> tuple[str, ...]:
    compilers: list[str] = []
    for key in CMAKE_COMPILER_CACHE_KEYS:
        value = cache.get(key)
        if value is None or value == "" or value.endswith("-NOTFOUND"):
            continue
        require(
            Path(value).is_absolute(),
            f"{label} {key} is not an absolute path",
        )
        if value not in compilers:
            compilers.append(value)
    require(
        cache.get("CMAKE_CXX_COMPILER") in compilers,
        f"{label} build has no configured C++ compiler",
    )
    return tuple(compilers)


def configured_scan_deps_paths(
    cache: Mapping[str, str],
    *,
    label: str,
) -> tuple[str, ...]:
    scanners: list[str] = []
    for key in CMAKE_SCAN_DEPS_CACHE_KEYS:
        value = cache.get(key)
        if value is None or value == "" or value.endswith("-NOTFOUND"):
            continue
        require(
            Path(value).is_absolute(),
            f"{label} {key} is not an absolute path",
        )
        if value not in scanners:
            scanners.append(value)
    return tuple(scanners)


def build_setting_names(engine: str) -> tuple[str, ...]:
    require(engine in ("infinity", "faiss"), f"Unknown build engine: {engine}")
    if engine == "infinity":
        return (*COMMON_BUILD_SETTINGS, *INFINITY_PRODUCTION_BUILD_SETTINGS)
    return COMMON_BUILD_SETTINGS


def resolve_vcpkg_benchmark_headers(
    cache: dict[str, str],
) -> tuple[Path, Path]:
    installed_value = cache.get("VCPKG_INSTALLED_DIR")
    triplet = cache.get("VCPKG_TARGET_TRIPLET")
    require(
        isinstance(installed_value, str) and installed_value != "",
        "Infinity build has no VCPKG_INSTALLED_DIR",
    )
    require(
        isinstance(triplet, str)
        and triplet != ""
        and "/" not in triplet
        and "\\" not in triplet,
        "Infinity build has no valid VCPKG_TARGET_TRIPLET",
    )
    installed = Path(installed_value).resolve()
    internal_value = cache.get("_VCPKG_INSTALLED_DIR")
    if internal_value:
        require(
            Path(internal_value).resolve() == installed,
            "Infinity vcpkg installed-directory cache entries disagree",
        )
    include_root = (installed / triplet / "include").resolve()
    ctpl_header = include_root / "ctpl_stl.h"
    simde_root = include_root / "simde"
    require(ctpl_header.is_file(), f"CTPL header is missing: {ctpl_header}")
    require(simde_root.is_dir(), f"SIMDe header tree is missing: {simde_root}")
    return ctpl_header.resolve(), simde_root.resolve()


def validate_infinity_production_build(
    cache: dict[str, str],
    *,
    binary: Path,
    repo: Path,
) -> None:
    require(
        binary.name == "infinity_hnsw_d0_production",
        "Infinity binary is not the production D0 target",
    )
    home_directory = cache.get("CMAKE_HOME_DIRECTORY")
    require(
        isinstance(home_directory, str)
        and home_directory != ""
        and Path(home_directory).resolve() == repo.resolve(),
        "Infinity production build has a different source root",
    )
    expected = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "ENABLE_JEMALLOC": "OFF",
        "VCPKG_MANIFEST_INSTALL": "OFF",
        "VCPKG_TARGET_TRIPLET": "arm64-osx",
        "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION": "ON",
        "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL": "OFF",
        "INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING": "ON",
    }
    for name, value in expected.items():
        require(
            cache.get(name) == value,
            f"Infinity production build requires {name}={value}",
        )
    require(
        bool(cache.get("INFINITY_BUILD_TIME_OVERRIDE")),
        "Infinity production build has no reproducible build-time override",
    )
    resolve_vcpkg_benchmark_headers(cache)


def find_build_metadata(binary: Path) -> tuple[Path, Path, Path]:
    for directory in (binary.parent, *binary.parents):
        cache = directory / "CMakeCache.txt"
        compile_commands = directory / "compile_commands.json"
        if cache.is_file() and compile_commands.is_file():
            return directory, cache, compile_commands
    raise D0Failure(
        f"Could not find CMakeCache.txt and compile_commands.json for {binary}"
    )


def resolve_executable(configured_path: str) -> Path:
    require(
        not configured_path.startswith("~"),
        f"Configured executable must not use home expansion: {configured_path}",
    )
    candidate = Path(configured_path)
    if not candidate.is_absolute():
        resolved = shutil.which(
            configured_path,
            path=tool_environment()["PATH"],
        )
        require(
            resolved is not None, f"Could not resolve executable: {configured_path}"
        )
        candidate = Path(resolved)
    require(candidate.is_file(), f"Configured executable is missing: {candidate}")
    return candidate.resolve()


def executable_record(
    configured_path: str,
    *,
    version_arguments: list[str],
    cwd: Path,
    output_dir: Path,
) -> dict[str, Any]:
    resolved = resolve_executable(configured_path)
    version = run_text(
        [str(resolved), *version_arguments],
        cwd=cwd,
        check=False,
    )
    version_text = (version.stdout + version.stderr).strip()
    require(version_text != "", f"Executable emitted no version text: {resolved}")
    return {
        "configured_path": configured_path,
        "resolved_path": str(resolved),
        **capture_content_blob(resolved, output_dir=output_dir),
        "version_command": [str(resolved), *version_arguments],
        "version_returncode": version.returncode,
        "version": version_text,
    }


def source_path_from_compile_entry(entry: dict[str, Any]) -> Path:
    working_directory = Path(str(entry["directory"]))
    require(
        working_directory.is_absolute(),
        "Compile entry working directory is not absolute",
    )
    return Path(
        lexical_absolute_path(
            str(entry["file"]),
            working_directory=working_directory,
        )
    )


def path_label(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def capture_content_blob(path: Path, *, output_dir: Path) -> dict[str, Any]:
    require(path.is_file(), f"Content-addressed input is missing: {path}")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        and len(data) == after.st_size,
        f"Content-addressed input changed while being read: {path}",
    )
    digest = hashlib.sha256(data).hexdigest()
    relative_path = Path("provenance") / "blobs" / digest[:2] / digest
    captured_path = output_dir / relative_path
    captured_path.parent.mkdir(parents=True, exist_ok=True)
    if captured_path.exists():
        require(
            captured_path.is_file()
            and captured_path.stat().st_size == len(data)
            and sha256(captured_path) == digest,
            f"Content-addressed blob collision or corruption: {captured_path}",
        )
    else:
        # Evidence blobs preserve bytes, not source metadata such as protected
        # macOS file flags, which may not be reproducible at the destination.
        captured_path.write_bytes(data)
    return {
        "captured_path": relative_path.as_posix(),
        "sha256": digest,
        "bytes": len(data),
    }


def compile_source_records(
    entries: list[dict[str, Any]],
    *,
    repo: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    paths = sorted(
        {source_path_from_compile_entry(entry) for entry in entries},
        key=lambda path: str(path).encode("utf-8"),
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        require(path.is_file(), f"Compiled source is missing: {path}")
        records.append(
            {
                "path": path_label(path, repo),
                "absolute_path": str(path),
                **capture_content_blob(path, output_dir=output_dir),
            }
        )
    return records


def source_tree_records(
    root: Path,
    *,
    repo: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    require(resolved_root.is_dir(), f"Source tree is missing: {resolved_root}")
    paths = sorted(
        [
            path
            for path in resolved_root.rglob("*")
            if path.is_file()
            and ".git" not in path.relative_to(resolved_root).parts
            and "__pycache__" not in path.relative_to(resolved_root).parts
            and path.suffix != ".pyc"
        ],
        key=lambda path: str(path).encode("utf-8"),
    )
    require(bool(paths), f"Source tree is empty: {resolved_root}")
    return [
        {
            "path": path_label(path, repo),
            "absolute_path": str(path),
            **capture_content_blob(path, output_dir=output_dir),
        }
        for path in paths
    ]


def macho_rpaths(path: Path, *, cwd: Path) -> list[Path]:
    output = run_text(
        ["/usr/bin/otool", "-l", str(path)],
        cwd=cwd,
        check=False,
    )
    if output.returncode != 0:
        return []
    rpaths: list[Path] = []
    lines = output.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "cmd LC_RPATH":
            continue
        for candidate in lines[index + 1 : index + 5]:
            match = re.match(r"\s*path (.+?) \(offset \d+\)", candidate)
            if match:
                raw = match.group(1)
                raw = raw.replace("@loader_path", str(path.parent))
                raw = raw.replace("@executable_path", str(path.parent))
                rpaths.append(Path(raw).resolve())
                break
    return rpaths


def resolve_macho_dependency(
    name: str,
    *,
    image: Path,
    executable: Path,
    rpaths: Iterable[Path],
) -> Path | None:
    if name.startswith("@loader_path/"):
        return (image.parent / name.removeprefix("@loader_path/")).resolve()
    if name.startswith("@executable_path/"):
        return (executable.parent / name.removeprefix("@executable_path/")).resolve()
    if name.startswith("@rpath/"):
        suffix = name.removeprefix("@rpath/")
        for rpath in rpaths:
            candidate = (rpath / suffix).resolve()
            if candidate.is_file():
                return candidate
        return None
    if name.startswith("/"):
        return Path(name).resolve()
    return (image.parent / name).resolve()


def dependency_manifest(binary: Path, *, cwd: Path) -> dict[str, Any]:
    executable_rpaths = macho_rpaths(binary, cwd=cwd)
    queue = [binary]
    inspected: set[Path] = set()
    dependencies: dict[str, dict[str, Any]] = {}
    raw_outputs: dict[str, str] = {}
    while queue:
        image = queue.pop(0)
        if image in inspected:
            continue
        inspected.add(image)
        completed = run_text(
            ["/usr/bin/otool", "-L", str(image)],
            cwd=cwd,
            check=False,
        )
        require(completed.returncode == 0, f"otool -L failed for {image}")
        raw_outputs[str(image)] = completed.stdout
        image_rpaths = [*macho_rpaths(image, cwd=cwd), *executable_rpaths]
        for line in completed.stdout.splitlines()[1:]:
            match = re.match(r"\s*(\S+)\s+\(compatibility version", line)
            if not match:
                continue
            install_name = match.group(1)
            resolved = resolve_macho_dependency(
                install_name,
                image=image,
                executable=binary,
                rpaths=image_rpaths,
            )
            key = str(resolved) if resolved is not None else install_name
            record = dependencies.setdefault(
                key,
                {
                    "install_names": [],
                    "referenced_by": [],
                    "resolved_path": str(resolved) if resolved is not None else None,
                    "sha256": None,
                    "bytes": None,
                    "unavailable_reason": None,
                },
            )
            if install_name not in record["install_names"]:
                record["install_names"].append(install_name)
            if str(image) not in record["referenced_by"]:
                record["referenced_by"].append(str(image))
            if resolved is None:
                record["unavailable_reason"] = "unresolved install name"
                require(
                    install_name.startswith(("/usr/lib/", "/System/Library/")),
                    f"Could not resolve non-system dependency {install_name}",
                )
            elif resolved.is_file():
                record["sha256"] = sha256(resolved)
                record["bytes"] = resolved.stat().st_size
                if resolved not in inspected:
                    queue.append(resolved)
            else:
                record["unavailable_reason"] = "provided by the macOS dyld shared cache"
                require(
                    install_name.startswith(("/usr/lib/", "/System/Library/")),
                    f"Dependency path is missing: {resolved}",
                )
    return {
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "dependencies": sorted(
            dependencies.values(), key=lambda item: str(item["resolved_path"])
        ),
        "otool_outputs": raw_outputs,
    }


def git_pathspecs(root: Path, output_dir: Path) -> list[str]:
    try:
        relative_output = output_dir.relative_to(root)
    except ValueError:
        return ["."]
    return [".", f":(exclude){relative_output}"]


def capture_git_repository(
    root: Path,
    *,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    git_root = Path(
        run_text(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            cwd=root,
        ).stdout.strip()
    ).resolve()
    head = run_text(
        ["git", "-C", str(git_root), "rev-parse", "HEAD"],
        cwd=git_root,
    ).stdout.strip()
    tree = run_text(
        ["git", "-C", str(git_root), "rev-parse", "HEAD^{tree}"],
        cwd=git_root,
    ).stdout.strip()
    pathspecs = git_pathspecs(git_root, output_dir)
    status = run_text(
        [
            "git",
            "-C",
            str(git_root),
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            *pathspecs,
        ],
        cwd=git_root,
    ).stdout
    diff = run_text(
        [
            "git",
            "-C",
            str(git_root),
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            *pathspecs,
        ],
        cwd=git_root,
    ).stdout
    commit_object = run_text(
        ["git", "-C", str(git_root), "cat-file", "commit", head],
        cwd=git_root,
    ).stdout
    tree_listing = run_text(
        ["git", "-C", str(git_root), "ls-tree", "-r", "--full-tree", head],
        cwd=git_root,
    ).stdout
    status_name = f"{label}-git-status.txt"
    diff_name = f"{label}-source-diff.patch"
    commit_name = f"{label}-commit-object.txt"
    tree_name = f"{label}-tree-listing.txt"
    (output_dir / status_name).write_text(status, encoding="utf-8")
    (output_dir / diff_name).write_text(diff, encoding="utf-8")
    (output_dir / commit_name).write_text(commit_object, encoding="utf-8")
    (output_dir / tree_name).write_text(tree_listing, encoding="utf-8")
    return {
        "root": str(git_root),
        "head": head,
        "tree": tree,
        "dirty": bool(status),
        "status_path": status_name,
        "status_sha256": sha256(output_dir / status_name),
        "diff_path": diff_name,
        "diff_sha256": sha256(output_dir / diff_name),
        "commit_object_path": commit_name,
        "commit_object_sha256": sha256(output_dir / commit_name),
        "tree_listing_path": tree_name,
        "tree_listing_sha256": sha256(output_dir / tree_name),
    }


def git_object_id(kind: str, data: bytes) -> str:
    header = f"{kind} {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def parse_git_commit_identity(
    data: bytes,
    *,
    label: str,
) -> tuple[str, tuple[str, ...]]:
    header, separator, _message = data.partition(b"\n\n")
    require(separator == b"\n\n", f"{label} has no header terminator")
    trees: list[str] = []
    parents: list[str] = []
    for line in header.splitlines():
        if line.startswith(b" "):
            continue
        name, separator, value = line.partition(b" ")
        require(separator == b" " and value, f"{label} has a malformed header")
        if name not in (b"tree", b"parent"):
            continue
        try:
            object_id = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise D0Failure(f"{label} has a non-ASCII object ID") from error
        require(
            re.fullmatch(r"[0-9a-f]{40}", object_id) is not None,
            f"{label} has an invalid {name.decode('ascii')} object ID",
        )
        if name == b"tree":
            trees.append(object_id)
        else:
            parents.append(object_id)
    require(len(trees) == 1, f"{label} must contain exactly one tree")
    return trees[0], tuple(parents)


def write_immutable_capture(
    output_dir: Path,
    filename: str,
    data: bytes,
) -> Path:
    destination = output_dir / filename
    require(not destination.exists(), f"Evidence capture already exists: {destination}")
    destination.write_bytes(data)
    require(
        destination.stat().st_size == len(data)
        and hashlib.sha256(destination.read_bytes()).digest()
        == hashlib.sha256(data).digest(),
        f"Evidence capture differs after writing: {destination}",
    )
    return destination


def read_stable_regular_file(path: Path, *, label: str) -> bytes:
    require(path.is_absolute(), f"{label} path is not absolute")
    require(not path.is_symlink(), f"{label} must not be a symlink")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    data = path.read_bytes()
    after = path.stat()
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        and len(data) == after.st_size,
        f"{label} changed while being read",
    )
    return data


def canonical_changed_paths(data: bytes, *, label: str) -> tuple[str, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise D0Failure(f"{label} is not UTF-8") from error
    paths: list[str] = []
    for line_number, path in enumerate(text.splitlines(), start=1):
        pure = PurePosixPath(path)
        require(
            path == pure.as_posix()
            and not pure.is_absolute()
            and all(part not in ("", ".", "..") for part in pure.parts),
            f"{label} line {line_number} is not a canonical relative path",
        )
        require(path not in paths, f"{label} duplicates path {path!r}")
        paths.append(path)
    require(paths, f"{label} is empty")
    return tuple(paths)


def validate_compiled_sources_at_git_commit(
    records: list[dict[str, Any]],
    *,
    repository_root: Path,
    commit: str,
    output_dir: Path,
) -> None:
    require(records, "Linked FAISS build has no compiled source records")
    for record in records:
        source = Path(str(record["absolute_path"]))
        try:
            relative = source.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise D0Failure(
                f"Linked FAISS compiled source is outside its repository: {source}"
            ) from error
        listing = run_bytes(
            [
                "git",
                "-C",
                str(repository_root),
                "ls-tree",
                "-z",
                commit,
                "--",
                relative,
            ],
            cwd=repository_root,
        ).stdout
        entries = [entry for entry in listing.split(b"\0") if entry]
        require(
            len(entries) == 1,
            f"Linked FAISS compiled source is absent from derivative tree: {relative}",
        )
        metadata, separator, raw_path = entries[0].partition(b"\t")
        match = re.fullmatch(rb"([0-7]{6}) (blob) ([0-9a-f]{40})", metadata)
        require(
            separator == b"\t"
            and match is not None
            and raw_path == relative.encode("utf-8"),
            f"Linked FAISS derivative tree entry is invalid: {relative}",
        )
        captured = output_dir / str(record["captured_path"])
        captured_bytes = read_stable_regular_file(
            captured.resolve(),
            label=f"Captured linked FAISS source {relative}",
        )
        require(
            git_object_id("blob", captured_bytes)
            == match.group(3).decode("ascii"),
            f"Linked FAISS compiled source differs from derivative tree: {relative}",
        )


def capture_faiss_audited_reference(
    *,
    benchmark_repo: Path,
    faiss_source_root: Path,
    faiss_repository: dict[str, Any],
    faiss_source_records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    repository_root = Path(str(faiss_repository["root"])).resolve()
    require(
        repository_root == faiss_source_root.resolve(),
        "Linked FAISS source root differs from its captured repository root",
    )
    require(
        faiss_repository["dirty"] is False,
        "FAISS derivative source repository is not clean",
    )

    base_commit = run_bytes(
        ["git", "-C", str(repository_root), "cat-file", "commit", PINNED_FAISS_COMMIT],
        cwd=repository_root,
    ).stdout
    require(
        git_object_id("commit", base_commit) == PINNED_FAISS_COMMIT,
        "Pinned FAISS base commit bytes do not match the pinned commit",
    )
    base_tree, _base_parents = parse_git_commit_identity(
        base_commit,
        label="Pinned FAISS base commit",
    )
    require(
        base_tree == PINNED_FAISS_BASE_TREE,
        "Pinned FAISS base commit has the wrong tree",
    )
    base_listing = run_bytes(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-tree",
            "-r",
            "--full-tree",
            PINNED_FAISS_COMMIT,
        ],
        cwd=repository_root,
    ).stdout
    write_immutable_capture(
        output_dir,
        FAISS_BASE_COMMIT_OBJECT_FILENAME,
        base_commit,
    )
    write_immutable_capture(
        output_dir,
        FAISS_BASE_TREE_LISTING_FILENAME,
        base_listing,
    )

    derivative_head = str(faiss_repository["head"])
    require(
        derivative_head != PINNED_FAISS_COMMIT,
        "FAISS derivative cannot use the pinned base commit directly",
    )
    derivative_commit = run_bytes(
        ["git", "-C", str(repository_root), "cat-file", "commit", derivative_head],
        cwd=repository_root,
    ).stdout
    require(
        git_object_id("commit", derivative_commit) == derivative_head,
        "FAISS derivative commit bytes do not match HEAD",
    )
    derivative_tree, derivative_parents = parse_git_commit_identity(
        derivative_commit,
        label="FAISS derivative commit",
    )
    require(
        derivative_parents == (PINNED_FAISS_COMMIT,),
        "FAISS derivative must have exactly the pinned base commit as its parent",
    )
    require(
        derivative_tree == PINNED_FAISS_STATS_DISABLED_TREE
        and faiss_repository["tree"] == PINNED_FAISS_STATS_DISABLED_TREE,
        "FAISS derivative has the wrong tree",
    )
    require(
        (output_dir / str(faiss_repository["commit_object_path"])).read_bytes()
        == derivative_commit,
        "Captured FAISS derivative commit differs from the repository",
    )
    derivative_listing = run_bytes(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-tree",
            "-r",
            "--full-tree",
            derivative_head,
        ],
        cwd=repository_root,
    ).stdout
    require(
        (output_dir / str(faiss_repository["tree_listing_path"])).read_bytes()
        == derivative_listing,
        "Captured FAISS derivative tree listing differs from the repository",
    )

    changed_paths_bytes = run_bytes(
        [
            "git",
            "-C",
            str(repository_root),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            derivative_head,
        ],
        cwd=repository_root,
    ).stdout
    changed_paths = canonical_changed_paths(
        changed_paths_bytes,
        label="FAISS derivative changed paths",
    )
    require(
        changed_paths == PINNED_FAISS_STATS_PATCH_PATHS,
        "FAISS derivative changed paths differ from the pinned patch",
    )
    write_immutable_capture(
        output_dir,
        FAISS_CHANGED_PATHS_FILENAME,
        changed_paths_bytes,
    )

    derivative_diff = run_bytes(
        [
            "git",
            "-C",
            str(repository_root),
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            f"{PINNED_FAISS_COMMIT}..{derivative_head}",
        ],
        cwd=repository_root,
    ).stdout
    require(
        hashlib.sha256(derivative_diff).hexdigest()
        == PINNED_FAISS_STATS_PATCH_SHA256,
        "FAISS derivative diff does not match the pinned patch",
    )
    write_immutable_capture(
        output_dir,
        FAISS_DERIVATIVE_DIFF_FILENAME,
        derivative_diff,
    )

    patch_source = (benchmark_repo / FAISS_STATS_PATCH_SOURCE).resolve()
    patch_bytes = read_stable_regular_file(
        patch_source,
        label="Pinned FAISS stats-disabled patch",
    )
    require(
        hashlib.sha256(patch_bytes).hexdigest()
        == PINNED_FAISS_STATS_PATCH_SHA256,
        "Pinned FAISS stats-disabled patch hash differs",
    )
    require(
        patch_bytes == derivative_diff,
        "Pinned FAISS patch bytes differ from the derivative diff",
    )
    captured_patch = write_immutable_capture(
        output_dir,
        FAISS_STATS_PATCH_FILENAME,
        patch_bytes,
    )

    tagged_commit = run_text(
        [
            "git",
            "-C",
            str(repository_root),
            "rev-parse",
            "v1.15.0^{commit}",
        ],
        cwd=repository_root,
    ).stdout.strip()
    require(
        tagged_commit == PINNED_FAISS_COMMIT,
        "FAISS v1.15.0 tag does not resolve to the pinned base commit",
    )
    validate_compiled_sources_at_git_commit(
        faiss_source_records,
        repository_root=repository_root,
        commit=derivative_head,
        output_dir=output_dir,
    )
    return {
        "policy": "single-parent-pinned-tree-v1",
        "repository_root": str(repository_root),
        "base": {
            "commit": PINNED_FAISS_COMMIT,
            "tree": PINNED_FAISS_BASE_TREE,
            "tag": "v1.15.0",
        },
        "derivative": {
            "head": derivative_head,
            "tree": PINNED_FAISS_STATS_DISABLED_TREE,
            "parent": PINNED_FAISS_COMMIT,
            "dirty": False,
            "changed_paths": list(PINNED_FAISS_STATS_PATCH_PATHS),
        },
        "patch": {
            "source_path": str(patch_source),
            "captured_path": FAISS_STATS_PATCH_FILENAME,
            "sha256": sha256(captured_patch),
        },
        "feature": {
            "cmake_option": "FAISS_ENABLE_GLOBAL_HNSW_STATS",
            "required_value": "OFF",
            "compile_definition": "FAISS_DISABLE_GLOBAL_HNSW_STATS=1",
        },
    }


def compile_entry_arguments(
    entry: Any,
    *,
    label: str,
    index: int,
) -> list[str]:
    require(isinstance(entry, dict), f"{label} compile entry {index} is invalid")
    has_arguments = "arguments" in entry
    has_command = "command" in entry
    require(
        has_arguments is not has_command,
        f"{label} compile entry {index} must contain one command representation",
    )
    if has_arguments:
        arguments = entry["arguments"]
        require(
            isinstance(arguments, list)
            and arguments
            and all(isinstance(item, str) and item for item in arguments),
            f"{label} compile entry {index} arguments are invalid",
        )
        return list(arguments)
    command = entry["command"]
    require(
        isinstance(command, str) and command,
        f"{label} compile entry {index} command is invalid",
    )
    try:
        arguments = shlex.split(command)
    except ValueError as error:
        raise D0Failure(
            f"{label} compile entry {index} cannot be tokenized: {error}"
        ) from error
    require(arguments, f"{label} compile entry {index} command is empty")
    return arguments


def expand_response_arguments(
    arguments: list[str],
    *,
    working_directory: Path,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> list[str]:
    require(
        working_directory.is_absolute(),
        f"{label} response-file working directory is not absolute",
    )
    invocation_directory = Path(os.path.abspath(working_directory))
    expanded: list[str] = []

    def visit(tokens: list[str], active: tuple[Path, ...]) -> None:
        require(
            len(active) <= MAX_RESPONSE_DEPTH,
            f"{label} exceeds the response-file nesting limit",
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
                    f"{label} exceeds the expanded response-token limit",
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
            require(
                response_path not in active,
                f"{label} contains a response-file cycle at {response_path}",
            )
            require(
                response_path.is_file(),
                f"{label} response file is missing: {response_path}",
            )
            data = response_path.read_bytes()
            require(
                len(data) <= MAX_RESPONSE_FILE_BYTES,
                f"{label} response file exceeds the byte limit: {response_path}",
            )
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise D0Failure(
                    f"{label} response file is not UTF-8: {response_path}"
                ) from error
            try:
                nested = shlex.split(text)
            except ValueError as error:
                raise D0Failure(
                    f"{label} response file cannot be tokenized: "
                    f"{response_path}: {error}"
                ) from error
            response_references.add((invocation_directory, response_path))
            visit(nested, (*active, response_path))

    visit(arguments, ())
    return expanded


def response_file_records(
    references: set[tuple[Path, Path]],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "invocation_working_directory": str(working_directory),
            **capture_content_blob(path, output_dir=output_dir),
        }
        for working_directory, path in sorted(
            references,
            key=lambda item: (
                str(item[0]).encode("utf-8"),
                str(item[1]).encode("utf-8"),
            ),
        )
    ]


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
    label: str,
) -> dict[str, dict[str, Any]]:
    require("\r" not in text, f"{label} uses non-canonical line endings")
    lines = text.splitlines()
    result: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(lines):
        if lines[index] == "":
            index += 1
            continue
        match = NINJA_DEPS_HEADER.fullmatch(lines[index])
        require(match is not None, f"{label} has an invalid header on line {index + 1}")
        output = lexical_absolute_path(
            match.group("output"),
            working_directory=build_directory,
        )
        require(output not in result, f"{label} duplicates output {output}")
        count = int(match.group("count"))
        index += 1
        dependencies: list[str] = []
        for _ in range(count):
            require(index < len(lines), f"{label} truncates dependencies for {output}")
            line = lines[index]
            require(
                line.startswith("    ") and line[4:] != "",
                f"{label} has an invalid dependency on line {index + 1}",
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
            f"{label} has extra dependencies for {output}",
        )
        result[output] = {
            "output": output,
            "dependency_count": count,
            "deps_mtime": int(match.group("mtime")),
            "status": match.group("status"),
            "dependencies": dependencies,
        }
    require(result, f"{label} is empty")
    return result


def compile_output_lexical_path(
    arguments: list[str],
    *,
    working_directory: Path,
    label: str,
) -> str | None:
    if "-c" not in arguments:
        return None
    output_indices = [
        index for index, argument in enumerate(arguments) if argument == "-o"
    ]
    require(
        len(output_indices) == 1,
        f"{label} must contain exactly one separated -o output",
    )
    output_index = output_indices[0]
    require(output_index + 1 < len(arguments), f"{label} has a dangling -o")
    return lexical_absolute_path(
        arguments[output_index + 1],
        working_directory=working_directory,
    )


def module_output_paths(
    arguments: list[str],
    *,
    working_directory: Path,
    label: str,
) -> list[str]:
    outputs: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-fmodule-output":
            require(
                index + 1 < len(arguments),
                f"{label} has a dangling -fmodule-output",
            )
            value = arguments[index + 1]
            index += 2
        elif argument.startswith("-fmodule-output="):
            value = argument.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        require(value != "", f"{label} has an empty module output")
        outputs.append(
            lexical_absolute_path(value, working_directory=working_directory)
        )
    require(
        len(outputs) == len(set(outputs)),
        f"{label} duplicates a module output",
    )
    return outputs


def compile_direct_inputs(
    entry: dict[str, Any],
    arguments: list[str],
    *,
    working_directory: Path,
    label: str,
) -> list[dict[str, str]]:
    source = Path(str(entry.get("file", "")))
    require(str(source) != "", f"{label} has no source file")
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
            require(index + 1 < len(arguments), f"{label} has a dangling {argument}")
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
        require(value != "", f"{label} has an empty {role} path")
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
    label: str,
) -> list[Path]:
    directories: list[Path] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-F":
            require(index + 1 < len(arguments), f"{label} has a dangling -F")
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
    label: str,
) -> list[Path]:
    directories: list[Path] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-L":
            require(index + 1 < len(arguments), f"{label} has a dangling -L")
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
    label: str,
) -> list[Path]:
    require(
        name != ""
        and not name.startswith(":")
        and "/" not in name
        and "\0" not in name,
        f"{label} has an invalid library name {name!r}",
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


def resolve_library_input(
    name: str,
    *,
    search_directories: list[Path],
    search_paths_first: bool,
    label: str,
) -> str:
    for candidate in library_input_candidates(
        name,
        search_directories=search_directories,
        search_paths_first=search_paths_first,
        label=label,
    ):
        if candidate.is_file():
            return os.path.abspath(candidate)
    raise D0Failure(f"{label} cannot resolve library {name!r}")


def resolve_framework_input(
    name: str,
    *,
    search_directories: list[Path],
    label: str,
) -> str:
    require(
        name != "" and "/" not in name and "\0" not in name,
        f"{label} has an invalid framework name {name!r}",
    )
    matches: list[Path] = []
    for directory in search_directories:
        framework = directory / f"{name}.framework"
        for candidate in (framework / f"{name}.tbd", framework / name):
            if candidate.is_file():
                matches.append(candidate)
                break
    require(matches, f"{label} cannot resolve framework {name}")
    return os.path.abspath(matches[0])


def derive_link_inputs(
    arguments: list[str],
    *,
    working_directory: Path,
    sysroot: Path,
    label: str,
) -> list[dict[str, str]]:
    require(arguments, f"{label} is empty")
    framework_directories = framework_search_directories(
        arguments,
        working_directory=working_directory,
        sysroot=sysroot,
        label=label,
    )
    library_directories = library_search_directories(
        arguments,
        working_directory=working_directory,
        sysroot=sysroot,
        label=label,
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
            require(index + 1 < len(arguments), f"{label} has dangling {argument}")
            name = arguments[index + 1]
            inputs.append(
                {
                    "role": "framework-input",
                    "path": resolve_framework_input(
                        name,
                        search_directories=framework_directories,
                        label=label,
                    ),
                }
            )
            index += 2
            continue
        if argument in linker_file_options:
            require(index + 1 < len(arguments), f"{label} has dangling {argument}")
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
            require(index + 1 < len(arguments), f"{label} has dangling {argument}")
            index += 2
            continue
        if argument == "-Xlinker":
            raise D0Failure(f"{label} uses unsupported -Xlinker syntax")
        if argument.startswith("-Wl,"):
            pieces = argument.split(",")[1:]
            require(
                pieces and all(piece != "" for piece in pieces),
                f"{label} has an empty -Wl operand",
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
                    raise D0Failure(f"{label} uses unsupported linker filelist")
                if piece in no_value_flags:
                    offset += 1
                    continue
                if piece in value_flags:
                    require(
                        offset + 1 < len(pieces),
                        f"{label} has a dangling {piece}",
                    )
                    offset += 2
                    continue
                if piece in framework_options:
                    require(
                        offset + 1 < len(pieces),
                        f"{label} has a dangling {piece}",
                    )
                    inputs.append(
                        {
                            "role": "framework-input",
                            "path": resolve_framework_input(
                                pieces[offset + 1],
                                search_directories=framework_directories,
                                label=label,
                            ),
                        }
                    )
                    offset += 2
                    continue
                if piece in file_flags:
                    require(
                        offset + 1 < len(pieces) and pieces[offset + 1] != "",
                        f"{label} has a dangling {piece}",
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
                        f"{label} has a dangling {piece}",
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
                        f"{label} has a dangling -l",
                    )
                    library_name = pieces[offset + 1]
                    offset += 2
                elif piece.startswith("-l"):
                    library_name = piece[2:]
                    offset += 1
                else:
                    library_name = None
                if library_name is not None:
                    path = resolve_library_input(
                        library_name,
                        search_directories=library_directories,
                        search_paths_first=search_paths_first,
                        label=label,
                    )
                    kind = link_input_kind(path)
                    require(
                        kind is not None,
                        f"{label} resolved an unsupported library {path}",
                    )
                    inputs.append(
                        {
                            "role": f"link-library-{kind}",
                            "path": path,
                        }
                    )
                    continue
                if piece.startswith("-"):
                    raise D0Failure(
                        f"{label} uses unsupported -Wl option {piece!r}"
                    )
                kind = link_input_kind(piece)
                require(
                    kind is not None,
                    f"{label} has an unsupported -Wl operand {piece!r}",
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
            require(index + 1 < len(arguments), f"{label} has a dangling -l")
            library_name = arguments[index + 1]
            index += 2
        elif argument.startswith("-l"):
            library_name = argument[2:]
            index += 1
        else:
            library_name = None
        if library_name is not None:
            path = resolve_library_input(
                library_name,
                search_directories=library_directories,
                search_paths_first=search_paths_first,
                label=label,
            )
            kind = link_input_kind(path)
            require(kind is not None, f"{label} resolved an unsupported library {path}")
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
        require(kind is not None, f"{label} has an unclassified operand {argument!r}")
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
    require(inputs, f"{label} has no explicit linker inputs")
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
    label: str,
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
        raise D0Failure(
            f"{label} output is outside its Ninja build directory: {output_path}"
        ) from error
    require(relative.parts, f"{label} output is the build directory")
    require(
        all(part not in ("", ".", "..") for part in relative.parts),
        f"{label} output has an invalid Ninja target path",
    )
    return relative.as_posix()


def parse_ninja_targets(
    text: str,
    *,
    build_directory: Path,
    label: str,
) -> dict[str, str]:
    targets: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(line != "", f"{label} line {line_number} is empty")
        pieces = line.rsplit(": ", 1)
        require(
            len(pieces) == 2 and pieces[0] != "" and pieces[1] != "",
            f"{label} line {line_number} is malformed",
        )
        output = lexical_absolute_path(
            pieces[0],
            working_directory=build_directory,
        )
        previous = targets.get(output)
        require(
            previous is None or previous == pieces[1],
            f"{label} has conflicting rules for target {output}",
        )
        targets[output] = pieces[1]
    require(targets, f"{label} is empty")
    return targets


def parse_ninja_query(
    text: str,
    *,
    build_directory: Path,
    label: str,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line and not line.startswith(" "):
            require(
                line.endswith(":") and line != ":",
                f"{label} line {line_number} has a malformed target",
            )
            if current is not None:
                require(
                    current["rule"] is not None,
                    f"{label} target {current['output']} has no producer rule",
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
        require(current is not None, f"{label} line {line_number} precedes a target")
        if line.startswith("  input: "):
            require(section is None, f"{label} target repeats its input section")
            current["rule"] = line[len("  input: ") :]
            require(current["rule"] != "", f"{label} target has an empty rule")
            section = "inputs"
            continue
        if line == "  outputs:":
            require(
                current["rule"] is not None and section == "inputs",
                f"{label} target has an out-of-order outputs section",
            )
            section = "outputs"
            continue
        require(
            line.startswith("    ") and section in {"inputs", "outputs"},
            f"{label} line {line_number} has invalid indentation",
        )
        value = line[4:]
        require(value != "", f"{label} line {line_number} has an empty path")
        if section == "inputs":
            if value.startswith("|| "):
                edge = "order-only"
                value = value[3:]
            elif value.startswith("| "):
                edge = "implicit"
                value = value[2:]
            else:
                edge = "explicit"
            require(value != "", f"{label} line {line_number} has an empty input")
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
            require(value != "", f"{label} line {line_number} has an empty output")
            current["outputs"].append(
                lexical_absolute_path(
                    value,
                    working_directory=build_directory,
                )
            )
    if current is not None:
        require(
            current["rule"] is not None,
            f"{label} target {current['output']} has no producer rule",
        )
        nodes.append(current)
    require(nodes, f"{label} is empty")
    outputs = [str(node["output"]) for node in nodes]
    require(len(outputs) == len(set(outputs)), f"{label} repeats a queried target")
    return sorted(nodes, key=lambda node: str(node["output"]).encode("utf-8"))


def parse_ninja_clean_plan(
    text: str,
    *,
    build_directory: Path,
    label: str,
    requested_target: str | None = None,
    requested_targets: Iterable[str] | None = None,
) -> list[str]:
    if requested_targets is None:
        require(requested_target is not None, f"{label} has no requested target")
        normalized_targets = [requested_target]
    else:
        require(
            requested_target is None,
            f"{label} mixes single-product and product target arguments",
        )
        normalized_targets = list(requested_targets)
    require(
        normalized_targets
        and all(
            isinstance(target, str) and target != ""
            for target in normalized_targets
        )
        and len(normalized_targets) == len(set(normalized_targets)),
        f"{label} requested targets are invalid",
    )
    require(
        text.endswith("\n") and "\r" not in text,
        f"{label} is not canonical newline-terminated text",
    )
    lines = text.splitlines()
    require(
        len(lines) >= 3 and lines[0] == "Cleaning...",
        f"{label} has an unexpected header",
    )
    count_match = re.fullmatch(r"(0|[1-9][0-9]*) files\.", lines[-1])
    require(count_match is not None, f"{label} has an invalid file count")
    require(
        build_directory.is_absolute()
        and os.path.abspath(str(build_directory)) == str(build_directory),
        f"{label} build directory is not canonical absolute",
    )

    paths: list[str] = []
    seen: set[str] = set()
    targets: list[str] = []
    for line in lines[1:-1]:
        if line.startswith("Target "):
            targets.append(line.removeprefix("Target "))
            continue
        require(line.startswith("Remove "), f"{label} has an unexpected line")
        path_text = line.removeprefix("Remove ")
        require(
            path_text != ""
            and all(
                ord(character) >= 32 and ord(character) != 127
                for character in path_text
            ),
            f"{label} has an invalid removal path",
        )
        path = Path(path_text)
        require(
            not path.is_absolute()
            and path.as_posix() == path_text
            and all(part not in ("", ".", "..") for part in path.parts),
            f"{label} removal path is not canonical relative: {path_text}",
        )
        absolute_path = str(build_directory / path)
        require(
            absolute_path not in seen,
            f"{label} repeats removal path: {path_text}",
        )
        seen.add(absolute_path)
        paths.append(absolute_path)

    require(targets == normalized_targets, f"{label} target list differs")
    require(
        len(paths) == int(count_match.group(1)),
        f"{label} removal count differs",
    )
    return paths


def graph_proven_link_outputs(
    *,
    nodes: list[dict[str, Any]],
    selected_output: str,
    link_inputs: list[dict[str, str]],
    known_derived_outputs: set[str],
    build_directory: Path,
    label: str,
) -> list[str]:
    by_output = {str(node["output"]): node for node in nodes}
    selected = by_output.get(selected_output)
    require(selected is not None, f"{label} omits the selected output")
    require(selected["rule"] != "phony", f"{label} selected output is phony")
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


def normalize_build_products(
    *,
    build_directory: Path,
    label: str,
    products: Iterable[Mapping[str, Any]] | None = None,
    requested_target: str | None = None,
    expected_output: Path | None = None,
    compile_entries: list[dict[str, Any]] | None = None,
    link_arguments: list[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    legacy = products is None
    if legacy:
        require(
            requested_target is not None
            and expected_output is not None
            and compile_entries is not None
            and link_arguments is not None,
            f"{label} single-product closure arguments are incomplete",
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
            f"{label} mixes single-product and ordered-product arguments",
        )
        raw_products = list(products)
    require(raw_products, f"{label} has no build products")

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    targets: set[str] = set()
    outputs: set[str] = set()
    for index, value in enumerate(raw_products):
        product_label = f"{label} product {index}"
        require(isinstance(value, Mapping), f"{product_label} is not a mapping")
        require(
            set(value)
            == {
                "name",
                "requested_target",
                "expected_output",
                "compile_entries",
                "link_arguments",
            },
            f"{product_label} keys differ",
        )
        name = value["name"]
        target = value["requested_target"]
        product_entries = value["compile_entries"]
        product_link = value["link_arguments"]
        require(
            isinstance(name, str)
            and re.fullmatch(r"[a-z][a-z0-9-]*", name) is not None,
            f"{product_label} name is invalid",
        )
        require(name not in names, f"{label} duplicates product name {name}")
        require(
            isinstance(target, str) and target != "",
            f"{product_label} requested target is invalid",
        )
        require(
            target not in targets,
            f"{label} duplicates requested target {target}",
        )
        output = lexical_absolute_path(
            str(value["expected_output"]),
            working_directory=build_directory,
        )
        require(
            target
            == ninja_target_for_output(
                Path(output),
                build_directory=build_directory,
                label=product_label,
            ),
            f"{product_label} target does not name its physical output",
        )
        require(
            output not in outputs,
            f"{label} duplicates expected output {output}",
        )
        require(
            isinstance(product_entries, list) and product_entries,
            f"{product_label} compile entries are invalid",
        )
        require(
            isinstance(product_link, list)
            and product_link
            and all(
                isinstance(argument, str) and argument
                for argument in product_link
            ),
            f"{product_label} link arguments are invalid",
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
            f"{product_label} link output is invalid",
        )
        linked_output = lexical_absolute_path(
            product_link[output_indices[0] + 1],
            working_directory=build_directory,
        )
        require(
            linked_output == output,
            f"{product_label} link output differs from its expected output",
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
    return normalized, legacy


def capture_selected_target_graph(
    *,
    output_dir: Path,
    capture_prefix: str,
    build_directory: Path,
    build_tool: dict[str, Any],
    known_derived_outputs: set[str],
    label: str,
    products: list[dict[str, Any]] | None = None,
    requested_target: str | None = None,
    expected_output: Path | None = None,
    link_inputs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if products is None:
        require(
            requested_target is not None
            and expected_output is not None
            and link_inputs is not None,
            f"{label} single-product target graph arguments are incomplete",
        )
        graph_products = [
            {
                "name": "benchmark",
                "requested_target": requested_target,
                "expected_output": lexical_absolute_path(
                    str(expected_output),
                    working_directory=build_directory,
                ),
                "link_inputs": link_inputs,
            }
        ]
    else:
        require(
            requested_target is None
            and expected_output is None
            and link_inputs is None,
            f"{label} mixes single-product and ordered-product graph arguments",
        )
        graph_products = products
    require(graph_products, f"{label} target graph has no products")
    requested_targets = [
        str(product["requested_target"]) for product in graph_products
    ]
    selected_outputs = [
        lexical_absolute_path(
            str(product["expected_output"]),
            working_directory=build_directory,
        )
        for product in graph_products
    ]
    require(
        len(graph_products) == len(selected_outputs),
        f"{label} target graph product/output counts differ",
    )
    require(
        len(requested_targets) == len(set(requested_targets))
        and len(selected_outputs) == len(set(selected_outputs)),
        f"{label} target graph products are not unique",
    )
    targets_command = [
        build_tool["resolved_path"],
        "-C",
        str(build_directory),
        "-t",
        "targets",
        "all",
    ]
    targets_text = run_text(targets_command, cwd=build_directory).stdout
    targets_name = f"{capture_prefix}-ninja-targets.txt"
    (output_dir / targets_name).write_text(targets_text, encoding="utf-8")
    targets = parse_ninja_targets(
        targets_text,
        build_directory=build_directory,
        label=f"{label} Ninja targets",
    )
    for selected_output in selected_outputs:
        require(
            targets.get(selected_output) not in (None, "phony"),
            f"{label} selected physical output has no material Ninja producer: "
            f"{selected_output}",
        )
    build_prefix = f"{build_directory}/"
    candidate_outputs = {
        str(item["path"])
        for product in graph_products
        for item in product["link_inputs"]
        if (
            str(item["path"]).startswith(build_prefix)
            and targets.get(str(item["path"])) not in (None, "phony")
        )
    }
    query_targets = list(requested_targets)
    for path in sorted(candidate_outputs, key=lambda value: value.encode("utf-8")):
        target = ninja_target_for_output(
            Path(path),
            build_directory=build_directory,
            label=f"{label} linked output",
        )
        if target not in query_targets:
            query_targets.append(target)
    require(
        len(query_targets) == len(set(query_targets)),
        f"{label} target graph query contains duplicate targets",
    )
    query_command = [
        build_tool["resolved_path"],
        "-C",
        str(build_directory),
        "-t",
        "query",
        *query_targets,
    ]
    query_text = run_text(query_command, cwd=build_directory).stdout
    query_name = f"{capture_prefix}-ninja-query.txt"
    (output_dir / query_name).write_text(query_text, encoding="utf-8")
    nodes = parse_ninja_query(
        query_text,
        build_directory=build_directory,
        label=f"{label} Ninja query",
    )
    derived_link_outputs = sorted(
        {
            output
            for product, selected_output in zip(
                graph_products,
                selected_outputs,
            )
            for output in graph_proven_link_outputs(
                nodes=nodes,
                selected_output=selected_output,
                link_inputs=product["link_inputs"],
                known_derived_outputs=known_derived_outputs,
                build_directory=build_directory,
                label=f"{label} product {product['name']} target graph",
            )
        },
        key=lambda value: value.encode("utf-8"),
    )
    material_outputs = sorted(
        {
            *selected_outputs,
            *(
                path
                for path in known_derived_outputs
                if targets.get(path) not in (None, "phony")
            ),
            *candidate_outputs,
        },
        key=lambda value: value.encode("utf-8"),
    )
    clean_command = [
        build_tool["resolved_path"],
        "-v",
        "-n",
        "-C",
        str(build_directory),
        "-t",
        "clean",
        *requested_targets,
    ]
    clean_completed = run_text(clean_command, cwd=build_directory)
    require(
        clean_completed.stderr == "",
        f"{label} Ninja target clean plan emitted stderr",
    )
    clean_stdout_name = f"{capture_prefix}-ninja-clean-plan.stdout"
    clean_stderr_name = f"{capture_prefix}-ninja-clean-plan.stderr"
    (output_dir / clean_stdout_name).write_text(
        clean_completed.stdout,
        encoding="utf-8",
    )
    (output_dir / clean_stderr_name).write_text(
        clean_completed.stderr,
        encoding="utf-8",
    )
    clean_outputs = parse_ninja_clean_plan(
        clean_completed.stdout,
        build_directory=build_directory,
        requested_targets=requested_targets,
        label=f"{label} Ninja target clean plan",
    )
    clean_output_set = set(clean_outputs)
    require(
        {
            *selected_outputs,
            *material_outputs,
            *derived_link_outputs,
        }
        <= clean_output_set,
        f"{label} Ninja target clean plan omits graph-proven outputs",
    )
    return {
        "targets": {
            "command": targets_command,
            "captured_path": targets_name,
            "sha256": sha256(output_dir / targets_name),
            "records": len(targets),
        },
        "query": {
            "command": query_command,
            "captured_path": query_name,
            "sha256": sha256(output_dir / query_name),
            "nodes": nodes,
        },
        "clean_plan": {
            "command": clean_command,
            "returncode": clean_completed.returncode,
            "stdout": {
                "captured_path": clean_stdout_name,
                "sha256": sha256(output_dir / clean_stdout_name),
            },
            "stderr": {
                "captured_path": clean_stderr_name,
                "sha256": sha256(output_dir / clean_stderr_name),
            },
            "outputs": clean_outputs,
        },
        "selected_outputs": selected_outputs,
        "material_outputs": material_outputs,
        "derived_link_outputs": derived_link_outputs,
    }


def _settlement_stream_bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    require(isinstance(value, str), "Ninja settlement stream has an invalid type")
    return value.encode("utf-8")


def settlement_stream_record(value: str | bytes | None) -> dict[str, Any]:
    data = _settlement_stream_bytes(value)
    try:
        utf8_text: str | None = data.decode("utf-8")
    except UnicodeDecodeError:
        utf8_text = None
    return {
        "encoding": "base64",
        "content_base64": base64.b64encode(data).decode("ascii"),
        "utf8_text": utf8_text,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_exclusive_atomic(path: Path, data: bytes) -> None:
    require(path.parent.is_dir(), f"Evidence directory is missing: {path.parent}")
    require(not path.exists(), f"Evidence already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.parent.name}.{path.name}.",
        dir=path.parent.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        os.link(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def write_ninja_settlement_failure(
    *,
    output_dir: Path,
    capture_prefix: str,
    command: list[str],
    cwd: Path,
    returncode: int | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    reason: str,
    error: BaseException | None = None,
) -> None:
    stem = f"{capture_prefix}-closure-settlement"
    record_path = output_dir / f"{stem}.json"
    record: dict[str, Any] = {
        "settlement_failure_schema_version": (
            NINJA_SETTLEMENT_FAILURE_SCHEMA_VERSION
        ),
        "status": "fail",
        "reason": reason,
        "command": command,
        "working_directory": str(cwd),
        "returncode": returncode,
        "expected_stdout_marker": NINJA_NO_WORK_MARKER,
        "stdout": settlement_stream_record(stdout),
        "stderr": settlement_stream_record(stderr),
    }
    if error is not None:
        record["execution_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    write_exclusive_atomic(
        record_path,
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def settlement_publication_failure(
    primary_error: BaseException,
    evidence_error: BaseException,
) -> D0Failure:
    return D0Failure(
        f"{primary_error}; could not preserve Ninja settlement failure evidence "
        f"({type(evidence_error).__name__}): {evidence_error}"
    )


def require_build_capture_settled(
    *,
    output_dir: Path,
    capture_prefix: str,
    command: list[str],
    cwd: Path,
    label: str,
) -> None:
    try:
        completed = run_bytes(command, cwd=cwd, check=False)
    except BaseException as error:
        stdout = getattr(error, "stdout", getattr(error, "output", None))
        stderr = getattr(error, "stderr", None)
        try:
            write_ninja_settlement_failure(
                output_dir=output_dir,
                capture_prefix=capture_prefix,
                command=command,
                cwd=cwd,
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                reason="execution-exception",
                error=error,
            )
        except BaseException as evidence_error:
            raise settlement_publication_failure(
                error,
                evidence_error,
            ) from evidence_error
        raise
    if (
        completed.returncode == 0
        and NINJA_NO_WORK_MARKER_BYTES in completed.stdout
    ):
        return
    reason = (
        "nonzero-returncode"
        if completed.returncode != 0
        else "missing-no-work-marker"
    )
    primary_error = D0Failure(
        (
            f"{label} settlement command failed with exit status "
            f"{completed.returncode}"
        )
        if completed.returncode != 0
        else f"{label} became stale during capture"
    )
    try:
        write_ninja_settlement_failure(
            output_dir=output_dir,
            capture_prefix=capture_prefix,
            command=command,
            cwd=cwd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            reason=reason,
        )
    except BaseException as evidence_error:
        raise settlement_publication_failure(
            primary_error,
            evidence_error,
        ) from evidence_error
    raise primary_error


def capture_build_input_closure(
    *,
    output_dir: Path,
    capture_prefix: str,
    build_directory: Path,
    build_tool: dict[str, Any],
    cache: dict[str, str],
    response_references: set[tuple[Path, Path]],
    label: str,
    products: Iterable[Mapping[str, Any]] | None = None,
    requested_target: str | None = None,
    expected_output: Path | None = None,
    compile_entries: list[dict[str, Any]] | None = None,
    link_arguments: list[str] | None = None,
) -> dict[str, Any]:
    normalized_products, _ = normalize_build_products(
        build_directory=build_directory,
        label=label,
        products=products,
        requested_target=requested_target,
        expected_output=expected_output,
        compile_entries=compile_entries,
        link_arguments=link_arguments,
    )
    canonical_compile_entry_union(
        normalized_products,
        label=f"{label} closure",
        response_references=response_references,
    )
    deps_command = [
        build_tool["resolved_path"],
        "-C",
        str(build_directory),
        "-t",
        "deps",
    ]
    deps_text = run_text(deps_command, cwd=build_directory).stdout
    deps_name = f"{capture_prefix}-ninja-deps.txt"
    (output_dir / deps_name).write_text(deps_text, encoding="utf-8")
    all_deps = parse_ninja_deps(
        deps_text,
        build_directory=build_directory,
        label=f"{label} Ninja dependencies",
    )

    compile_units_by_output: dict[str, dict[str, Any]] = {}
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
        for index, entry in enumerate(product["compile_entries"]):
            entry_label = f"{label} product {product['name']} compile entry {index}"
            working_directory = Path(str(entry.get("directory", "")))
            require(
                working_directory.is_absolute(),
                f"{entry_label} directory is invalid",
            )
            raw_arguments = compile_entry_arguments(
                entry,
                label=f"{label} product {product['name']}",
                index=index,
            )
            arguments = expand_response_arguments(
                raw_arguments,
                working_directory=working_directory,
                label=entry_label,
                response_references=response_references,
            )
            output = compile_output_lexical_path(
                arguments,
                working_directory=working_directory,
                label=entry_label,
            )
            require(output is not None, f"{entry_label} is not a compilation")
            module_outputs = module_output_paths(
                arguments,
                working_directory=working_directory,
                label=entry_label,
            )
            direct_inputs = compile_direct_inputs(
                entry,
                arguments,
                working_directory=working_directory,
                label=entry_label,
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
                f"{label} product {product['name']} has conflicting "
                f"compile entries for {output}",
            )
            product_units_by_output.setdefault(output, (arguments, unit))
            previous_unit = compile_units_by_output.get(output)
            require(
                previous_unit is None
                or (
                    previous_unit == unit
                    and compile_arguments_by_output[output] == arguments
                ),
                f"{label} has conflicting compile entries for {output}",
            )
            compile_units_by_output.setdefault(output, unit)
            compile_arguments_by_output.setdefault(output, arguments)
            previous_inputs = direct_inputs_by_output.get(output)
            require(
                previous_inputs is None or previous_inputs == direct_inputs,
                f"{label} has conflicting direct inputs for {output}",
            )
            direct_inputs_by_output.setdefault(output, direct_inputs)
            seed_outputs.add(output)
            for module_output in module_outputs:
                previous_producer = module_producers.get(module_output)
                require(
                    previous_producer is None or previous_producer == output,
                    f"{label} has conflicting module output {module_output}",
                )
                module_producers.setdefault(module_output, output)
        product_compile_outputs[product["name"]] = sorted(
            product_units_by_output,
            key=lambda value: value.encode("utf-8"),
        )
    compile_units = list(compile_units_by_output.values())

    missing_outputs = sorted(seed_outputs - set(all_deps))
    require(
        not missing_outputs,
        f"{label} Ninja dependencies omit compile/module outputs: {missing_outputs}",
    )
    closure_outputs: set[str] = set()
    pending = sorted(seed_outputs, key=lambda value: value.encode("utf-8"))
    while pending:
        output = pending.pop()
        if output in closure_outputs:
            continue
        record = all_deps[output]
        require(record["status"] == "VALID", f"{label} has stale deps for {output}")
        closure_outputs.add(output)
        for dependency in record["dependencies"]:
            producer = module_producers.get(dependency, dependency)
            if producer in all_deps and producer not in closure_outputs:
                pending.append(producer)
            require(
                Path(dependency).suffix.lower() != ".pcm"
                or dependency in module_producers,
                f"{label} PCM dependency has no selected producer: {dependency}",
            )

    for output, direct_inputs in direct_inputs_by_output.items():
        recorded = set(all_deps[output]["dependencies"])
        for direct_input in direct_inputs:
            if direct_input["role"] == "module-input":
                require(
                    direct_input["path"] in module_producers,
                    f"{label} module input has no selected producer: "
                    f"{direct_input['path']}",
                )
                continue
            if direct_input["role"] == "module-map":
                continue
            require(
                direct_input["path"] in recorded,
                f"{label} direct input {direct_input['path']} is absent from "
                f"Ninja dependencies for {output}",
            )

    captured_products: list[dict[str, Any]] = []
    for product in normalized_products:
        product_link_inputs = derive_link_inputs(
            product["link_arguments"],
            working_directory=build_directory,
            sysroot=Path(cache["CMAKE_OSX_SYSROOT"]),
            label=f"{label} product {product['name']} link command",
        )
        captured_products.append(
            {
                "name": product["name"],
                "requested_target": product["requested_target"],
                "expected_output": product["expected_output"],
                "compile_outputs": product_compile_outputs[product["name"]],
                "link": {
                    "arguments": product["link_arguments"],
                    "inputs": product_link_inputs,
                },
            }
        )
    require(
        len(normalized_products) == len(captured_products),
        f"{label} normalized and captured product counts differ",
    )
    selected_target_graph = capture_selected_target_graph(
        output_dir=output_dir,
        capture_prefix=capture_prefix,
        build_directory=build_directory,
        build_tool=build_tool,
        products=[
            {
                "name": product["name"],
                "requested_target": product["requested_target"],
                "expected_output": product["expected_output"],
                "link_inputs": captured["link"]["inputs"],
            }
            for product, captured in zip(
                normalized_products,
                captured_products,
            )
        ],
        known_derived_outputs={
            *closure_outputs,
            *module_producers,
        },
        label=label,
    )
    roles: dict[str, set[str]] = {}

    def add_role(path: str, role: str) -> None:
        roles.setdefault(path, set()).add(role)

    dependency_graph: list[dict[str, Any]] = []
    for output in sorted(closure_outputs, key=lambda value: value.encode("utf-8")):
        record = all_deps[output]
        add_role(output, "ninja-output")
        for dependency in record["dependencies"]:
            add_role(dependency, "compiler-dependency")
        dependency_graph.append(record)
    for unit in compile_units:
        add_role(unit["output"], "compile-output")
        for module_output in unit["module_outputs"]:
            add_role(module_output, "module-output")
        for direct_input in unit["direct_inputs"]:
            add_role(direct_input["path"], direct_input["role"])
    for _, response_path in response_references:
        add_role(str(response_path), "response-file")
    for product in captured_products:
        for link_input in product["link"]["inputs"]:
            add_role(link_input["path"], link_input["role"])

    files: list[dict[str, Any]] = []
    for path_text in sorted(roles, key=lambda value: value.encode("utf-8")):
        path = Path(path_text)
        require(path.is_file(), f"{label} build input is missing: {path}")
        resolved_before = path.resolve()
        blob = capture_content_blob(path, output_dir=output_dir)
        require(
            path.resolve() == resolved_before and sha256(path) == blob["sha256"],
            f"{label} build input changed during capture: {path}",
        )
        files.append(
            {
                "path": path_text,
                "resolved_path": str(resolved_before),
                "type": closure_file_role(path_text),
                "roles": sorted(roles[path_text]),
                **blob,
            }
        )

    repeated_deps = run_text(deps_command, cwd=build_directory).stdout
    require(
        repeated_deps == deps_text,
        f"{label} Ninja dependencies changed during capture",
    )
    require_build_capture_settled(
        output_dir=output_dir,
        capture_prefix=capture_prefix,
        command=[
            build_tool["resolved_path"],
            "-C",
            str(build_directory),
            *[
                str(product["requested_target"])
                for product in normalized_products
            ],
        ],
        cwd=build_directory,
        label=label,
    )

    compile_units.sort(key=lambda value: value["output"].encode("utf-8"))
    closure = {
        "closure_schema_version": BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
        "products": captured_products,
        "build_directory": str(build_directory),
        "ninja_deps": {
            "command": deps_command,
            "captured_path": deps_name,
            "sha256": sha256(output_dir / deps_name),
            "records": len(all_deps),
        },
        "compile_units": compile_units,
        "dependency_graph": dependency_graph,
        "selected_target_graph": selected_target_graph,
        "files": files,
    }
    closure_name = f"{capture_prefix}-build-input-closure.json"
    write_json(output_dir / closure_name, closure)
    return {
        "closure_schema_version": BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
        "captured_path": closure_name,
        "sha256": sha256(output_dir / closure_name),
        "products": len(captured_products),
        "compile_units": len(compile_units),
        "dependency_outputs": len(dependency_graph),
        "files": len(files),
    }


def normalized_compile_arguments(arguments: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-MD", "-MMD"):
            index += 1
            continue
        if argument in ("-MF", "-MT", "-MQ"):
            require(
                index + 1 < len(arguments),
                f"Compile command has a dangling {argument}",
            )
            index += 2
            continue
        result.append(argument)
        index += 1
    return result


def validate_release_arguments(
    arguments: list[str],
    cache: dict[str, str],
    *,
    label: str,
) -> None:
    optimization = [
        argument
        for argument in arguments
        if re.fullmatch(r"-O(?:0|1|2|3|s|z|g|fast)?", argument)
    ]
    require(
        optimization and optimization[-1] == "-O3",
        f"{label} effective optimization is not -O3",
    )

    ndebug_defined: bool | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        macro: str | None = None
        defined: bool | None = None
        if argument in ("-D", "-U"):
            require(index + 1 < len(arguments), f"{label} has a dangling {argument}")
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
    require(ndebug_defined is True, f"{label} does not leave NDEBUG defined")

    architecture_values = [
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == "-arch"
    ]
    require(
        architecture_values == ["arm64"],
        f"{label} does not compile for exactly one arm64 architecture",
    )
    sysroot_values = [
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == "-isysroot"
    ]
    require(
        sysroot_values == [cache["CMAKE_OSX_SYSROOT"]],
        f"{label} sysroot differs from CMake",
    )
    deployment_flags = [
        argument
        for argument in arguments
        if argument.startswith("-mmacosx-version-min=")
    ]
    require(
        deployment_flags
        == [f"-mmacosx-version-min={cache['CMAKE_OSX_DEPLOYMENT_TARGET']}"],
        f"{label} deployment target differs from CMake",
    )

    lowered = [argument.lower() for argument in arguments]
    forbidden_prefixes = (
        "-flto",
        "-fprofile",
        "-ffast-math",
        "-funsafe-math",
        "-ffp-model=fast",
    )
    require("-ofast" not in lowered, f"{label} uses forbidden -Ofast")
    require(
        not any(argument.startswith(forbidden_prefixes) for argument in lowered),
        f"{label} uses a forbidden optimization policy",
    )
    for index, argument in enumerate(lowered):
        if argument in ("-march", "-mcpu"):
            require(index + 1 < len(lowered), f"{label} has a dangling {argument}")
            value = lowered[index + 1]
        elif argument.startswith(("-march=", "-mcpu=")):
            value = argument.split("=", 1)[1]
        else:
            continue
        require(value != "native", f"{label} uses a native CPU target")
        require(
            not re.fullmatch(r"(?:apple-|m[1-9]).*", value),
            f"{label} uses an Apple-specific CPU target",
        )


def validate_release_build(
    cache: dict[str, str],
    compile_entries: list[Any],
    *,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> None:
    require(cache.get("CMAKE_BUILD_TYPE") == "Release", f"{label} build is not Release")
    require(
        cache.get("CMAKE_OSX_ARCHITECTURES") == "arm64",
        f"{label} build is not fixed to arm64",
    )
    require(
        cache.get("CMAKE_OSX_DEPLOYMENT_TARGET") == "14.0",
        f"{label} deployment target is not 14.0",
    )
    require(cache.get("CMAKE_OSX_SYSROOT"), f"{label} build has no explicit sysroot")
    require(cache.get("CMAKE_LINKER"), f"{label} build has no linker")
    require(compile_entries, f"{label} compile database is empty")
    compilers = set(configured_compiler_paths(cache, label=label))
    for index, entry in enumerate(compile_entries):
        raw_arguments = compile_entry_arguments(entry, label=label, index=index)
        directory = Path(str(entry.get("directory", "")))
        require(
            directory.is_absolute(),
            f"{label} compile entry {index} directory is invalid",
        )
        arguments = expand_response_arguments(
            raw_arguments,
            working_directory=directory,
            label=f"{label} compile entry {index}",
            response_references=response_references,
        )
        require(
            arguments[0] in compilers,
            f"{label} compile entry {index} uses a different compiler",
        )
        validate_release_arguments(
            arguments,
            cache,
            label=f"{label} compile entry {index}",
        )


def validate_required_compile_definition(
    compile_entries: list[dict[str, Any]],
    *,
    name: str,
    value: str,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> None:
    expected = (True, value)
    for index, entry in enumerate(compile_entries):
        entry_label = f"{label} compile entry {index}"
        directory = Path(str(entry.get("directory", "")))
        require(directory.is_absolute(), f"{entry_label} directory is invalid")
        arguments = expand_response_arguments(
            compile_entry_arguments(entry, label=label, index=index),
            working_directory=directory,
            label=entry_label,
            response_references=response_references,
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
                    f"{entry_label} has a dangling {argument}",
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
            f"{entry_label} must define exactly {name}={value}",
        )


def validate_faiss_stats_disabled_build(
    cache: dict[str, str],
    compile_entries: list[dict[str, Any]],
    *,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> None:
    require(
        cache.get("FAISS_ENABLE_GLOBAL_HNSW_STATS") == "OFF",
        f"{label} requires FAISS_ENABLE_GLOBAL_HNSW_STATS=OFF",
    )
    validate_required_compile_definition(
        compile_entries,
        name="FAISS_DISABLE_GLOBAL_HNSW_STATS",
        value="1",
        label=label,
        response_references=response_references,
    )


def compile_output_path(
    arguments: list[str],
    *,
    working_directory: Path,
    label: str,
) -> Path | None:
    if "-c" not in arguments:
        return None
    output_indices = [
        index for index, argument in enumerate(arguments) if argument == "-o"
    ]
    require(
        len(output_indices) == 1,
        f"{label} must contain exactly one separated -o output",
    )
    output_index = output_indices[0]
    require(output_index + 1 < len(arguments), f"{label} has a dangling -o")
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
    label: str,
) -> list[list[str]]:
    scan_deps_tokens = scan_deps_tokens or set()
    shell_boundaries = {"&&", "||", ";", ">", ">>"}
    segments: list[list[str]] = []
    start = 0
    for index in range(len(tokens) + 1):
        if index != len(tokens) and tokens[index] not in shell_boundaries:
            continue
        segment = tokens[start:index]
        start = index + 1
        if not segment:
            continue
        compiler_indices = [
            token_index
            for token_index, token in enumerate(segment)
            if token in compiler_tokens
        ]
        scanner_indices = [
            token_index
            for token_index, token in enumerate(segment)
            if token in scan_deps_tokens
        ]
        require(
            len(scanner_indices) <= 1,
            f"{label} contains multiple configured dependency scanners",
        )
        if scanner_indices:
            require(
                len(compiler_indices) == 1,
                f"{label} dependency scan does not use exactly one configured compiler",
            )
            compiler_index = compiler_indices[0]
            compiler_arguments = segment[compiler_index:]
            compile_indices = [
                index
                for index, argument in enumerate(compiler_arguments)
                if argument == "-c"
            ]
            output_indices = [
                index
                for index, argument in enumerate(compiler_arguments)
                if argument == "-o"
            ]
            output_is_valid = (
                len(output_indices) == 1
                and output_indices[0] + 1 < len(compiler_arguments)
                and compiler_arguments[output_indices[0] + 1] != ""
                and not compiler_arguments[output_indices[0] + 1].startswith("-")
            )
            require(
                scanner_indices == [0]
                and segment[:compiler_index]
                == [segment[0], "-format=p1689", "--"]
                and len(compile_indices) == 1
                and output_is_valid,
                f"{label} has a malformed configured dependency scan",
            )
            continue
        compile_like = "-c" in segment and "-o" in segment
        require(
            not compile_like or len(compiler_indices) == 1,
            f"{label} compile action does not use exactly one configured compiler",
        )
        require(
            len(compiler_indices) <= 1,
            f"{label} contains multiple configured compiler tokens",
        )
        if compiler_indices:
            require(
                compiler_indices == [0],
                f"{label} configured compiler has an unsupported wrapper",
            )
            segments.append(segment[compiler_indices[0] :])
    return segments


def target_compile_entries(
    entries: list[dict[str, Any]],
    ninja_commands: str,
    *,
    compiler: str | None = None,
    cache: Mapping[str, str] | None = None,
    build_directory: Path,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> list[dict[str, Any]]:
    require(
        (compiler is None) != (cache is None),
        f"{label} must select either one legacy compiler or configured compilers",
    )
    compiler_tokens = (
        {compiler}
        if compiler is not None
        else set(configured_compiler_paths(cache, label=label))
    )
    scan_deps_tokens = (
        set()
        if cache is None
        else set(configured_scan_deps_paths(cache, label=label))
    )
    target_arguments_by_output: dict[Path, list[str]] = {}
    for line_number, line in enumerate(ninja_commands.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise D0Failure(
                f"{label} line {line_number} cannot be tokenized: {error}"
            ) from error
        for command_tokens in compiler_command_slices(
            tokens,
            compiler_tokens=compiler_tokens,
            scan_deps_tokens=scan_deps_tokens,
            label=f"{label} line {line_number}",
        ):
            arguments = expand_response_arguments(
                command_tokens,
                working_directory=build_directory,
                label=f"{label} compiler invocation on line {line_number}",
                response_references=response_references,
            )
            output = compile_output_path(
                arguments,
                working_directory=build_directory,
                label=f"{label} compiler invocation on line {line_number}",
            )
            if output is not None:
                previous = target_arguments_by_output.get(output)
                require(
                    previous is None or previous == arguments,
                    f"{label} has conflicting compiler invocations for {output}",
                )
                target_arguments_by_output.setdefault(output, arguments)

    require(
        target_arguments_by_output,
        f"{label} contains no compilation outputs",
    )
    selected_by_output: dict[Path, tuple[list[str], dict[str, Any]]] = {}
    for index, entry in enumerate(entries):
        raw_arguments = compile_entry_arguments(entry, label=label, index=index)
        directory = Path(str(entry.get("directory", "")))
        require(
            directory.is_absolute(),
            f"{label} compile entry {index} directory is invalid",
        )
        raw_output = compile_output_path(
            raw_arguments,
            working_directory=directory,
            label=f"{label} compile entry {index}",
        )
        if raw_output not in target_arguments_by_output:
            continue
        arguments = expand_response_arguments(
            raw_arguments,
            working_directory=directory,
            label=f"{label} compile entry {index}",
            response_references=response_references,
        )
        output = compile_output_path(
            arguments,
            working_directory=directory,
            label=f"{label} compile entry {index}",
        )
        require(
            output == raw_output,
            f"{label} response file changes compile entry {index} output",
        )
        require(
            arguments[0] in compiler_tokens,
            f"{label} compile entry {index} uses an unconfigured compiler",
        )
        previous = selected_by_output.get(output)
        require(
            previous is None or previous[0] == arguments,
            f"{label} compile database has conflicting records for {output}",
        )
        selected_by_output.setdefault(output, (arguments, entry))

    require(
        set(selected_by_output) == set(target_arguments_by_output),
        f"{label} compile database does not exactly cover target compilation outputs",
    )
    for output, (arguments, _) in selected_by_output.items():
        require(
            normalized_compile_arguments(arguments)
            == normalized_compile_arguments(target_arguments_by_output[output]),
            f"{label} has no exact compiler invocation for {output}",
        )
    return [
        selected_by_output[output][1]
        for output in sorted(
            selected_by_output,
            key=lambda value: str(value).encode("utf-8"),
        )
    ]


def canonical_compile_entry_union(
    products: Iterable[Mapping[str, Any]],
    *,
    combined_entries: list[dict[str, Any]] | None = None,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> list[dict[str, Any]]:
    def collect(
        groups: Iterable[tuple[str, list[dict[str, Any]]]],
        *,
        collection_label: str,
    ) -> dict[Path, tuple[list[str], dict[str, Any]]]:
        by_output: dict[Path, tuple[list[str], dict[str, Any]]] = {}
        for group_name, entries in groups:
            require(entries, f"{collection_label} {group_name} is empty")
            for index, entry in enumerate(entries):
                entry_label = f"{collection_label} {group_name} entry {index}"
                directory = Path(str(entry.get("directory", "")))
                require(
                    directory.is_absolute(),
                    f"{entry_label} directory is invalid",
                )
                arguments = expand_response_arguments(
                    compile_entry_arguments(
                        entry,
                        label=f"{collection_label} {group_name}",
                        index=index,
                    ),
                    working_directory=directory,
                    label=entry_label,
                    response_references=response_references,
                )
                output = compile_output_path(
                    arguments,
                    working_directory=directory,
                    label=entry_label,
                )
                require(output is not None, f"{entry_label} is not a compile")
                previous = by_output.get(output)
                require(
                    previous is None or previous[0] == arguments,
                    f"{collection_label} has conflicting compile entries "
                    f"for {output}",
                )
                by_output.setdefault(output, (arguments, entry))
        return by_output

    product_groups: list[tuple[str, list[dict[str, Any]]]] = []
    product_names: set[str] = set()
    for index, product in enumerate(products):
        product_label = f"{label} product {index}"
        require(isinstance(product, Mapping), f"{product_label} is not a mapping")
        name = product.get("name")
        entries = product.get("compile_entries")
        require(
            isinstance(name, str) and name and name not in product_names,
            f"{product_label} name is invalid or duplicated",
        )
        require(
            isinstance(entries, list),
            f"{product_label} compile entries are invalid",
        )
        product_names.add(name)
        product_groups.append((name, entries))
    require(product_groups, f"{label} has no product compile sets")
    product_union = collect(
        product_groups,
        collection_label=f"{label} product union",
    )
    if combined_entries is not None:
        combined_union = collect(
            [("combined capture", combined_entries)],
            collection_label=f"{label} combined union",
        )
        require(
            {
                output: arguments
                for output, (arguments, _) in product_union.items()
            }
            == {
                output: arguments
                for output, (arguments, _) in combined_union.items()
            },
            f"{label} canonical product union differs from combined capture",
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
    label: str,
    response_references: set[tuple[Path, Path]],
    products: Iterable[Mapping[str, Any]] | None = None,
    compile_entries: list[dict[str, Any]] | None = None,
    expected_output: Path | None = None,
) -> list[str] | list[list[str]]:
    product_mode = products is not None
    if not product_mode:
        require(
            compile_entries is not None and expected_output is not None,
            f"{label} single-product command arguments are incomplete",
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
            f"{label} mixes single-product and ordered-product command arguments",
        )
        command_products = list(products)
    require(command_products, f"{label} has no command products")

    expected_product_outputs: list[Path] = []
    product_names: set[str] = set()
    expected_by_output: dict[Path, list[str]] = {}
    expected_exact_by_output: dict[Path, list[str]] = {}
    for product_index, product in enumerate(command_products):
        product_label = f"{label} product {product_index}"
        require(isinstance(product, Mapping), f"{product_label} is not a mapping")
        require(
            set(product) == {"name", "expected_output", "compile_entries"},
            f"{product_label} keys differ",
        )
        name = product["name"]
        require(
            isinstance(name, str) and name and name not in product_names,
            f"{product_label} name is invalid or duplicated",
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
            f"{label} duplicates expected output {product_output}",
        )
        expected_product_outputs.append(product_output)
        product_entries = product["compile_entries"]
        require(
            isinstance(product_entries, list) and product_entries,
            f"{product_label} compile entries are invalid",
        )
        product_compile_arguments: dict[Path, list[str]] = {}
        for index, entry in enumerate(product_entries):
            entry_label = f"{product_label} compile entry {index}"
            raw_arguments = compile_entry_arguments(
                entry,
                label=product_label,
                index=index,
            )
            directory = Path(str(entry.get("directory", "")))
            require(directory.is_absolute(), f"{entry_label} directory is invalid")
            arguments = expand_response_arguments(
                raw_arguments,
                working_directory=directory,
                label=entry_label,
                response_references=response_references,
            )
            output = compile_output_path(
                arguments,
                working_directory=directory,
                label=entry_label,
            )
            require(output is not None, f"{entry_label} is not a compile")
            previous_product_arguments = product_compile_arguments.get(output)
            require(
                previous_product_arguments is None
                or previous_product_arguments == arguments,
                f"{product_label} has conflicting duplicate commands for {output}",
            )
            product_compile_arguments.setdefault(output, arguments)
            normalized_arguments = normalized_compile_arguments(arguments)
            previous_exact_arguments = expected_exact_by_output.get(output)
            require(
                previous_exact_arguments is None
                or previous_exact_arguments == arguments,
                f"{label} has conflicting compile commands for {output}",
            )
            expected_exact_by_output.setdefault(output, arguments)
            previous_arguments = expected_by_output.get(output)
            require(
                previous_arguments is None
                or previous_arguments == normalized_arguments,
                f"{label} has conflicting compile commands for {output}",
            )
            expected_by_output.setdefault(output, normalized_arguments)
    observed_outputs: set[Path] = set()
    observed_by_output: dict[Path, list[list[str]]] = {}
    observed_exact_by_output: dict[Path, list[str]] = {}
    compiler_tokens = set(configured_compiler_paths(cache, label=label))
    scan_deps_tokens = set(configured_scan_deps_paths(cache, label=label))
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
            raise D0Failure(
                f"{label} line {line_number} cannot be tokenized: {error}"
            ) from error
        for command_tokens in compiler_command_slices(
            tokens,
            compiler_tokens=compiler_tokens,
            scan_deps_tokens=scan_deps_tokens,
            label=f"{label} line {line_number}",
        ):
            arguments = expand_response_arguments(
                command_tokens,
                working_directory=build_directory,
                label=f"{label} compiler invocation {invocation_count + 1}",
                response_references=response_references,
            )
            invocation_count += 1
            validate_release_arguments(
                arguments,
                cache,
                label=f"{label} compiler invocation {invocation_count}",
            )
            output = compile_output_path(
                arguments,
                working_directory=build_directory,
                label=f"{label} compiler invocation {invocation_count}",
            )
            if output is not None:
                previous_arguments = observed_exact_by_output.get(output)
                require(
                    previous_arguments is None or previous_arguments == arguments,
                    f"{label} has conflicting compiler invocations for {output}",
                )
                observed_exact_by_output.setdefault(output, arguments)
                observed_outputs.add(output)
                observed_by_output.setdefault(output, []).append(
                    normalized_compile_arguments(arguments)
                )
            if "-c" not in arguments and "-o" in arguments:
                output_indices = [
                    index
                    for index, argument in enumerate(arguments)
                    if argument == "-o"
                ]
                require(
                    len(output_indices) == 1,
                    f"{label} link invocation must contain exactly one -o",
                )
                output_index = output_indices[0]
                require(
                    output_index + 1 < len(arguments),
                    f"{label} link invocation has a dangling -o",
                )
                observed_output = Path(
                    lexical_absolute_path(
                        arguments[output_index + 1],
                        working_directory=build_directory,
                    )
                )
                if observed_output in matching_link_arguments:
                    matching_link_arguments[observed_output].append(arguments)
    require(invocation_count > 0, f"{label} contains no compiler invocation")
    require(
        observed_outputs == set(expected_by_output),
        f"{label} does not exactly cover the compile database object outputs",
    )
    for output, expected_arguments in expected_by_output.items():
        require(
            expected_arguments in observed_by_output.get(output, []),
            f"{label} has no exact compiler invocation for {output}",
        )
    for output in expected_product_outputs:
        require(
            len(matching_link_arguments[output]) == 1,
            f"{label} must contain exactly one expected link command for {output}",
        )
    ordered_links = [
        matching_link_arguments[output][0] for output in expected_product_outputs
    ]
    return ordered_links if product_mode else ordered_links[0]


def preflight(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    dataset: dict[str, Any],
    dataset_record_path: Path,
    heldout: dict[str, Any],
    heldout_record_path: Path,
    schedule_path: Path,
) -> dict[str, Any]:
    parent_environment_policy = reject_unsafe_parent_environment()
    require(platform.machine() == "arm64", "D0 must run natively on arm64")
    require(
        args.infinity_binary != args.faiss_binary,
        "Infinity and FAISS binaries must be distinct paths",
    )
    disk_usage = shutil.disk_usage(args.repo)
    required_free_bytes = MINIMUM_DISK_RESERVE_BYTES + D0_WORST_CASE_NEW_BYTES
    require(
        disk_usage.free >= required_free_bytes,
        f"D0 requires {required_free_bytes} free bytes, found {disk_usage.free}",
    )
    performance_prefixes = (
        "ACCELERATE_",
        "BLAS_",
        "CPUPROFILE",
        "DYLD_",
        "KMP_",
        "Malloc",
        "MKL_",
        "OPENBLAS_",
        "OMP_",
        "VECLIB_",
    )
    parent_performance_environment = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(performance_prefixes)
    }

    binaries: dict[str, Any] = {}
    builds: dict[str, Any] = {}
    build_caches: dict[str, dict[str, str]] = {}
    all_source_records: dict[str, list[dict[str, Any]]] = {}
    dependency_manifests: dict[str, dict[str, Any]] = {}
    source_roots: set[Path] = {args.repo}
    for engine in ("infinity", "faiss"):
        response_references: set[tuple[Path, Path]] = set()
        binary = binary_for_engine(args, engine)
        description = run_text(
            ["/usr/bin/file", str(binary)],
            cwd=args.repo,
        ).stdout.strip()
        require(
            "Mach-O 64-bit executable arm64" in description,
            f"{engine} binary is not native arm64: {description}",
        )
        binary_hash = sha256(binary)
        binaries[engine] = {
            "path": str(binary),
            "description": description,
            "sha256": binary_hash,
            "bytes": binary.stat().st_size,
        }

        build_dir, cache_path, compile_commands_path = find_build_metadata(binary)
        ninja_target = ninja_target_for_output(
            binary,
            build_directory=build_dir,
            label=engine,
        )
        cache_text = cache_path.read_text(encoding="utf-8")
        cache = parse_cmake_cache(cache_text)

        all_compile_entries = json.loads(
            compile_commands_path.read_text(encoding="utf-8")
        )
        require(
            isinstance(all_compile_entries, list) and bool(all_compile_entries),
            f"{engine} compile database is empty",
        )

        cache_name = f"{engine}-CMakeCache.txt"
        compile_name = f"{engine}-compile_commands.json"
        ninja_commands_name = f"{engine}-ninja-commands.txt"
        ninja_noop_name = f"{engine}-ninja-noop.txt"
        shutil.copy2(cache_path, output_dir / cache_name)

        compiler = executable_record(
            cache["CMAKE_CXX_COMPILER"],
            version_arguments=["--version"],
            cwd=args.repo,
            output_dir=output_dir,
        )
        linker = executable_record(
            cache["CMAKE_LINKER"],
            version_arguments=["-v"],
            cwd=args.repo,
            output_dir=output_dir,
        )
        build_tool = executable_record(
            cache["CMAKE_MAKE_PROGRAM"],
            version_arguments=["--version"],
            cwd=args.repo,
            output_dir=output_dir,
        )
        ninja_commands = run_text(
            [
                build_tool["resolved_path"],
                "-C",
                str(build_dir),
                "-t",
                "commands",
                ninja_target,
            ],
            cwd=args.repo,
        ).stdout
        require(ninja_commands.strip() != "", f"{engine} Ninja command list is empty")
        compile_entries = target_compile_entries(
            all_compile_entries,
            ninja_commands,
            cache=cache,
            build_directory=build_dir,
            label=f"{engine} Ninja commands",
            response_references=response_references,
        )
        validate_release_build(
            cache,
            compile_entries,
            label=engine,
            response_references=response_references,
        )
        if engine == "infinity":
            validate_infinity_production_build(
                cache,
                binary=binary,
                repo=args.repo,
            )
        build_caches[engine] = cache
        (output_dir / compile_name).write_text(
            json.dumps(compile_entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        link_arguments = validate_ninja_commands(
            ninja_commands,
            cache=cache,
            compile_entries=compile_entries,
            build_directory=build_dir,
            expected_output=binary,
            label=f"{engine} Ninja commands",
            response_references=response_references,
        )
        (output_dir / ninja_commands_name).write_text(
            ninja_commands,
            encoding="utf-8",
        )
        ninja_noop = run_text(
            [
                build_tool["resolved_path"],
                "-C",
                str(build_dir),
                ninja_target,
            ],
            cwd=args.repo,
        ).stdout
        require(
            "ninja: no work to do." in ninja_noop,
            f"{engine} build tree is stale",
        )
        (output_dir / ninja_noop_name).write_text(
            ninja_noop,
            encoding="utf-8",
        )
        dependencies = dependency_manifest(binary, cwd=args.repo)
        dependency_manifests[engine] = dependencies
        dependencies_name = f"{engine}-dependencies.json"
        (output_dir / dependencies_name).write_text(
            json.dumps(dependencies, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        source_records = compile_source_records(
            compile_entries,
            repo=args.repo,
            output_dir=output_dir,
        )
        all_source_records[engine] = source_records
        home_directory = Path(cache["CMAKE_HOME_DIRECTORY"]).resolve()
        source_roots.add(home_directory)
        build_input_closure = capture_build_input_closure(
            output_dir=output_dir,
            capture_prefix=engine,
            requested_target=ninja_target,
            expected_output=binary,
            build_directory=build_dir,
            build_tool=build_tool,
            cache=cache,
            compile_entries=compile_entries,
            link_arguments=link_arguments,
            response_references=response_references,
            label=engine,
        )
        builds[engine] = {
            "build_directory": str(build_dir),
            "cmake_home_directory": str(home_directory),
            "cmake_cache": {
                "captured_path": cache_name,
                "sha256": sha256(output_dir / cache_name),
            },
            "compile_commands": {
                "captured_path": compile_name,
                "sha256": sha256(output_dir / compile_name),
                "entries": len(compile_entries),
            },
            "settings": {
                name: cache.get(name)
                for name in build_setting_names(engine)
            },
            "compiler": compiler,
            "linker": linker,
            "build_tool": build_tool,
            "ninja_commands": {
                "captured_path": ninja_commands_name,
                "sha256": sha256(output_dir / ninja_commands_name),
            },
            "ninja_noop": {
                "captured_path": ninja_noop_name,
                "sha256": sha256(output_dir / ninja_noop_name),
            },
            "dependencies": {
                "captured_path": dependencies_name,
                "sha256": sha256(output_dir / dependencies_name),
                "count": len(dependencies["dependencies"]),
            },
            "build_input_closure": build_input_closure,
            "response_files": response_file_records(
                response_references,
                output_dir=output_dir,
            ),
            "compiled_sources": len(source_records),
        }

    require(
        binaries["infinity"]["sha256"] != binaries["faiss"]["sha256"],
        "Infinity and FAISS binaries have identical content",
    )
    for tool_name in ("compiler", "linker", "build_tool"):
        require(
            builds["infinity"][tool_name]["resolved_path"]
            == builds["faiss"][tool_name]["resolved_path"],
            f"Infinity and FAISS use different {tool_name} paths",
        )
        require(
            builds["infinity"][tool_name]["sha256"]
            == builds["faiss"][tool_name]["sha256"],
            f"Infinity and FAISS use different {tool_name} binaries",
        )
    for setting in COMMON_BUILD_SETTINGS:
        require(
            builds["infinity"]["settings"][setting]
            == builds["faiss"]["settings"][setting],
            f"Infinity and FAISS differ in {setting}",
        )

    faiss_dependencies = [
        dependency
        for dependency in dependency_manifests["faiss"]["dependencies"]
        if dependency["resolved_path"] is not None
        and Path(dependency["resolved_path"]).name == "libfaiss.dylib"
    ]
    require(
        len(faiss_dependencies) == 1,
        "FAISS binary must resolve exactly one libfaiss.dylib, "
        f"found {len(faiss_dependencies)}",
    )
    faiss_library = Path(faiss_dependencies[0]["resolved_path"]).resolve()
    require(
        faiss_library.is_file(), f"Linked FAISS library is missing: {faiss_library}"
    )
    require(
        faiss_dependencies[0]["sha256"] == sha256(faiss_library),
        "Linked FAISS library hash changed during preflight",
    )

    faiss_library_build_dir, faiss_cache_path, faiss_compile_path = find_build_metadata(
        faiss_library
    )
    faiss_cache_text = faiss_cache_path.read_text(encoding="utf-8")
    faiss_cache = parse_cmake_cache(faiss_cache_text)
    faiss_compile_text = faiss_compile_path.read_text(encoding="utf-8")
    all_faiss_compile_entries = json.loads(faiss_compile_text)
    require(
        isinstance(all_faiss_compile_entries, list) and bool(all_faiss_compile_entries),
        "Linked FAISS compile database is empty",
    )
    require(
        faiss_library_build_dir in faiss_library.parents,
        "Linked FAISS library is not inside its recorded build directory",
    )

    faiss_library_compiler = executable_record(
        faiss_cache["CMAKE_CXX_COMPILER"],
        version_arguments=["--version"],
        cwd=args.repo,
        output_dir=output_dir,
    )
    faiss_library_linker = executable_record(
        faiss_cache["CMAKE_LINKER"],
        version_arguments=["-v"],
        cwd=args.repo,
        output_dir=output_dir,
    )
    faiss_library_build_tool = executable_record(
        faiss_cache["CMAKE_MAKE_PROGRAM"],
        version_arguments=["--version"],
        cwd=args.repo,
        output_dir=output_dir,
    )
    faiss_library_target = ninja_target_for_output(
        faiss_library,
        build_directory=faiss_library_build_dir,
        label="linked FAISS library",
    )
    faiss_ninja_commands_name = "faiss-library-ninja-commands.txt"
    faiss_ninja_commands = run_text(
        [
            faiss_library_build_tool["resolved_path"],
            "-C",
            str(faiss_library_build_dir),
            "-t",
            "commands",
            faiss_library_target,
        ],
        cwd=args.repo,
    ).stdout
    require(
        faiss_ninja_commands.strip() != "", "FAISS library Ninja command list is empty"
    )
    faiss_response_references: set[tuple[Path, Path]] = set()
    faiss_compile_entries = target_compile_entries(
        all_faiss_compile_entries,
        faiss_ninja_commands,
        cache=faiss_cache,
        build_directory=faiss_library_build_dir,
        label="FAISS library Ninja commands",
        response_references=faiss_response_references,
    )
    validate_release_build(
        faiss_cache,
        faiss_compile_entries,
        label="linked FAISS library",
        response_references=faiss_response_references,
    )
    validate_faiss_stats_disabled_build(
        faiss_cache,
        faiss_compile_entries,
        label="linked FAISS library",
        response_references=faiss_response_references,
    )
    faiss_link_arguments = validate_ninja_commands(
        faiss_ninja_commands,
        cache=faiss_cache,
        compile_entries=faiss_compile_entries,
        build_directory=faiss_library_build_dir,
        expected_output=faiss_library,
        label="FAISS library Ninja commands",
        response_references=faiss_response_references,
    )
    (output_dir / faiss_ninja_commands_name).write_text(
        faiss_ninja_commands,
        encoding="utf-8",
    )
    faiss_ninja_noop_name = "faiss-library-ninja-noop.txt"
    faiss_ninja_noop = run_text(
        [
            faiss_library_build_tool["resolved_path"],
            "-C",
            str(faiss_library_build_dir),
            faiss_library_target,
        ],
        cwd=args.repo,
    ).stdout
    require(
        "ninja: no work to do." in faiss_ninja_noop,
        "FAISS library build tree is stale",
    )
    (output_dir / faiss_ninja_noop_name).write_text(
        faiss_ninja_noop,
        encoding="utf-8",
    )
    for tool_name, tool in (
        ("compiler", faiss_library_compiler),
        ("linker", faiss_library_linker),
        ("build_tool", faiss_library_build_tool),
    ):
        require(
            tool["resolved_path"] == builds["faiss"][tool_name]["resolved_path"],
            f"Linked FAISS library uses a different {tool_name} path",
        )
        require(
            tool["sha256"] == builds["faiss"][tool_name]["sha256"],
            f"Linked FAISS library uses a different {tool_name} binary",
        )
    for setting in (
        "CMAKE_BUILD_TYPE",
        "CMAKE_OSX_ARCHITECTURES",
        "CMAKE_OSX_DEPLOYMENT_TARGET",
        "CMAKE_OSX_SYSROOT",
    ):
        require(
            faiss_cache.get(setting) == builds["faiss"]["settings"][setting],
            f"Linked FAISS library differs in {setting}",
        )

    libomp_dependencies = [
        dependency
        for dependency in dependency_manifests["faiss"]["dependencies"]
        if dependency["resolved_path"] is not None
        and Path(dependency["resolved_path"]).name == "libomp.dylib"
    ]
    require(
        len(libomp_dependencies) == 1,
        "FAISS dependency graph must resolve exactly one libomp.dylib, "
        f"found {len(libomp_dependencies)}",
    )
    linked_libomp = Path(libomp_dependencies[0]["resolved_path"]).resolve()
    require(
        sha256(linked_libomp) == PINNED_LIBOMP_SHA256,
        "Linked FAISS OpenMP runtime does not match the pinned D0 runtime",
    )
    configured_libomp = Path(faiss_cache.get("OpenMP_libomp_LIBRARY", "")).resolve()
    require(
        configured_libomp == linked_libomp,
        "Linked FAISS OpenMP runtime differs from its source-build configuration",
    )
    configured_omp_include = Path(
        faiss_cache.get("OpenMP_CXX_INCLUDE_DIR", "")
    ).resolve()
    configured_omp_header = configured_omp_include / "omp.h"
    require(
        configured_omp_header.is_file()
        and sha256(configured_omp_header) == PINNED_OMP_HEADER_SHA256,
        "FAISS omp.h does not match the pinned D0 header",
    )
    require(
        Path(build_caches["faiss"].get("LIBOMP_LIBRARY", "")).resolve()
        == linked_libomp,
        "FAISS wrapper links a different OpenMP runtime",
    )
    require(
        Path(build_caches["faiss"].get("LIBOMP_INCLUDE_DIR", "")).resolve()
        == configured_omp_include,
        "FAISS wrapper uses a different OpenMP header directory",
    )

    faiss_cache_name = "faiss-library-CMakeCache.txt"
    faiss_compile_name = "faiss-library-compile_commands.json"
    shutil.copy2(faiss_cache_path, output_dir / faiss_cache_name)
    (output_dir / faiss_compile_name).write_text(
        json.dumps(faiss_compile_entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    faiss_source_records = compile_source_records(
        faiss_compile_entries,
        repo=args.repo,
        output_dir=output_dir,
    )
    all_source_records["faiss_library"] = faiss_source_records
    faiss_source_root = Path(faiss_cache["CMAKE_HOME_DIRECTORY"]).resolve()
    source_roots.add(faiss_source_root)
    faiss_build_input_closure = capture_build_input_closure(
        output_dir=output_dir,
        capture_prefix="faiss-library",
        requested_target=faiss_library_target,
        expected_output=faiss_library,
        build_directory=faiss_library_build_dir,
        build_tool=faiss_library_build_tool,
        cache=faiss_cache,
        compile_entries=faiss_compile_entries,
        link_arguments=faiss_link_arguments,
        response_references=faiss_response_references,
        label="linked FAISS library",
    )
    linked_faiss_build = {
        "library": {
            "path": str(faiss_library),
            "sha256": sha256(faiss_library),
            "bytes": faiss_library.stat().st_size,
        },
        "build_directory": str(faiss_library_build_dir),
        "cmake_home_directory": str(faiss_source_root),
        "cmake_cache": {
            "captured_path": faiss_cache_name,
            "sha256": sha256(output_dir / faiss_cache_name),
        },
        "compile_commands": {
            "captured_path": faiss_compile_name,
            "sha256": sha256(output_dir / faiss_compile_name),
            "entries": len(faiss_compile_entries),
        },
        "settings": {
            name: faiss_cache.get(name)
            for name in (
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
            )
        },
        "compiler": faiss_library_compiler,
        "linker": faiss_library_linker,
        "build_tool": faiss_library_build_tool,
        "ninja_commands": {
            "captured_path": faiss_ninja_commands_name,
            "sha256": sha256(output_dir / faiss_ninja_commands_name),
        },
        "ninja_noop": {
            "captured_path": faiss_ninja_noop_name,
            "sha256": sha256(output_dir / faiss_ninja_noop_name),
        },
        "build_input_closure": faiss_build_input_closure,
        "response_files": response_file_records(
            faiss_response_references,
            output_dir=output_dir,
        ),
        "compiled_sources": len(faiss_source_records),
        "openmp_runtime": libomp_dependencies[0],
    }

    libomp_recipe = args.repo / "tools/apple_silicon/prepare_d0_libomp.sh"
    require(libomp_recipe.is_file(), f"D0 libomp recipe is missing: {libomp_recipe}")
    native_harness_root = (
        args.repo / "tools/apple_silicon/native_hnsw_smoke"
    ).resolve()
    ctpl_header, simde_root = resolve_vcpkg_benchmark_headers(
        build_caches["infinity"]
    )
    ctpl_root = ctpl_header.parent
    infinity_compile_commands = (
        output_dir / builds["infinity"]["compile_commands"]["captured_path"]
    ).read_text(encoding="utf-8")
    require(
        str(ctpl_root) in infinity_compile_commands,
        "Infinity compile database does not include the recorded vcpkg headers",
    )
    infinity_closure_path = (
        output_dir
        / builds["infinity"]["build_input_closure"]["captured_path"]
    )
    infinity_closure = json.loads(
        infinity_closure_path.read_text(encoding="utf-8")
    )
    closure_paths = {
        Path(record["resolved_path"]).resolve()
        for record in infinity_closure["files"]
    }
    require(
        ctpl_header in closure_paths,
        "Infinity build-input closure omits the CTPL header",
    )
    simde_closure_paths: list[Path] = []
    for path in closure_paths:
        try:
            path.relative_to(simde_root)
        except ValueError:
            continue
        simde_closure_paths.append(path)
    require(
        simde_closure_paths,
        "Infinity build-input closure omits the SIMDe headers",
    )
    ctpl_files = [
        {
            "path": path_label(ctpl_header, args.repo),
            "absolute_path": str(ctpl_header),
            **capture_content_blob(ctpl_header, output_dir=output_dir),
        }
    ]
    simde_files = source_tree_records(
        simde_root,
        repo=args.repo,
        output_dir=output_dir,
    )
    require(
        set(simde_closure_paths)
        <= {Path(record["absolute_path"]) for record in simde_files},
        "Recorded SIMDe tree does not cover every compiled SIMDe header",
    )
    benchmark_input_sources = {
        "native_harness": {
            "root": str(native_harness_root),
            "files": source_tree_records(
                native_harness_root,
                repo=args.repo,
                output_dir=output_dir,
            ),
        },
        "ctpl": {
            "root": str(ctpl_root),
            "files": ctpl_files,
        },
        "simde_headers": {
            "root": str(simde_root),
            "files": simde_files,
        },
        "libomp_recipe": {
            "path": path_label(libomp_recipe, args.repo),
            **capture_content_blob(libomp_recipe, output_dir=output_dir),
        },
        "omp_header": {
            "path": str(configured_omp_header),
            **capture_content_blob(configured_omp_header, output_dir=output_dir),
        },
        "libomp_runtime": {
            "path": str(linked_libomp),
            **capture_content_blob(linked_libomp, output_dir=output_dir),
        },
    }
    benchmark_sources_name = "benchmark-input-source-hashes.json"
    (output_dir / benchmark_sources_name).write_text(
        json.dumps(benchmark_input_sources, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sources_name = "compiled-source-hashes.json"
    (output_dir / sources_name).write_text(
        json.dumps(all_source_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    git_repositories: list[dict[str, Any]] = []
    seen_git_roots: set[Path] = set()
    for source_root in sorted(source_roots):
        git_root = Path(
            run_text(
                ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
                cwd=source_root,
            ).stdout.strip()
        ).resolve()
        if git_root in seen_git_roots:
            continue
        seen_git_roots.add(git_root)
        git_repositories.append(
            capture_git_repository(
                git_root,
                output_dir=output_dir,
                label=f"source-repository-{len(git_repositories):02d}",
            )
        )
    faiss_repository = next(
        (
            repository
            for repository in git_repositories
            if Path(repository["root"]).resolve() == faiss_source_root
        ),
        None,
    )
    require(faiss_repository is not None, "FAISS source repository was not captured")
    faiss_audited_reference = capture_faiss_audited_reference(
        benchmark_repo=args.repo,
        faiss_source_root=faiss_source_root,
        faiss_repository=faiss_repository,
        faiss_source_records=faiss_source_records,
        output_dir=output_dir,
    )

    runtime_artifact_hashes = {
        details["path"]: details["sha256"] for details in binaries.values()
    }
    time_wrapper_path = Path(
        process_executable_identity(Path("/usr/bin/time"))["path"]
    )
    runtime_artifact_hashes[str(time_wrapper_path)] = sha256(time_wrapper_path)
    for manifest in dependency_manifests.values():
        for dependency in manifest["dependencies"]:
            if dependency["resolved_path"] is not None and dependency["sha256"]:
                runtime_artifact_hashes[dependency["resolved_path"]] = dependency[
                    "sha256"
                ]
    for build in (*builds.values(), linked_faiss_build):
        for tool_name in ("compiler", "linker", "build_tool"):
            tool = build[tool_name]
            runtime_artifact_hashes[tool["resolved_path"]] = tool["sha256"]

    runner_path = Path(__file__).resolve()
    runner_capture_path = output_dir / RUNNER_CAPTURE_FILENAME
    shutil.copy2(runner_path, runner_capture_path)
    runner_sha256 = sha256(runner_path)
    require(
        sha256(runner_capture_path) == runner_sha256,
        "Captured D0 runner differs from the executing runner",
    )
    runtime_artifact_hashes[str(runner_path)] = runner_sha256
    verifier_path = runner_path.with_name("verify_native_hnsw_d0.py")
    require(verifier_path.is_file(), f"D0 verifier is missing: {verifier_path}")
    verifier_capture_path = output_dir / VERIFIER_CAPTURE_FILENAME
    shutil.copy2(verifier_path, verifier_capture_path)
    verifier_sha256 = sha256(verifier_path)
    require(
        sha256(verifier_capture_path) == verifier_sha256,
        "Captured D0 verifier differs from the source verifier",
    )
    runtime_artifact_hashes[str(verifier_path)] = verifier_sha256
    python_executable = executable_record(
        sys.executable,
        version_arguments=["--version"],
        cwd=args.repo,
        output_dir=output_dir,
    )
    runtime_artifact_hashes[python_executable["resolved_path"]] = python_executable[
        "sha256"
    ]
    runtime_artifact_snapshots: list[dict[str, Any]] = []
    for original_path, expected_digest in sorted(runtime_artifact_hashes.items()):
        snapshot = capture_content_blob(Path(original_path), output_dir=output_dir)
        require(
            snapshot["sha256"] == expected_digest,
            f"Runtime artifact changed before snapshot: {original_path}",
        )
        runtime_artifact_snapshots.append({"original_path": original_path, **snapshot})
    schedule_sha256 = sha256(schedule_path)
    campaign_binding = create_campaign_binding(
        binaries=binaries,
        dataset=dataset,
        schedule_sha256=schedule_sha256,
    )
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "execution_mode": (
            PREFLIGHT_ONLY_MODE if args.preflight_only else FULL_CAMPAIGN_MODE
        ),
        "captured_at": utc_now(),
        "scope": D0_SCOPE,
        "repo": str(args.repo),
        "runner": {
            "source_path": str(runner_path),
            "captured_path": RUNNER_CAPTURE_FILENAME,
            "sha256": runner_sha256,
            "bytes": runner_capture_path.stat().st_size,
            "argv": list(sys.argv),
            "invocation_working_directory": str(Path.cwd().resolve()),
            "python_executable": python_executable,
            "effective_arguments": {
                "repo": str(args.repo),
                "infinity_binary": str(args.infinity_binary),
                "faiss_binary": str(args.faiss_binary),
                "output_directory": str(args.output_dir),
                "idle_minimum_percent": args.idle_minimum_percent,
                "idle_timeout_seconds": args.idle_timeout_seconds,
                "member_timeout_seconds": args.member_timeout_seconds,
                "preflight_only": args.preflight_only,
            },
        },
        "verifier": {
            "source_path": str(verifier_path),
            "captured_path": VERIFIER_CAPTURE_FILENAME,
            "sha256": verifier_sha256,
            "bytes": verifier_capture_path.stat().st_size,
        },
        "host": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "kernel": platform.release(),
            "python": platform.python_version(),
            "disk": {
                "total_bytes": disk_usage.total,
                "used_bytes": disk_usage.used,
                "free_bytes": disk_usage.free,
                "minimum_reserve_bytes": MINIMUM_DISK_RESERVE_BYTES,
                "declared_worst_case_new_bytes": D0_WORST_CASE_NEW_BYTES,
                "required_free_bytes": required_free_bytes,
            },
        },
        "environment_control": {
            "parent_performance_environment": parent_performance_environment,
            "unsafe_parent_environment_policy": parent_environment_policy,
            "child_environment_template": benchmark_environment(MEASURED_PARTICIPANTS),
            "inherits_parent_environment": False,
        },
        "dataset": {
            **dataset,
            "record_path": dataset_record_path.name,
            "record_sha256": sha256(dataset_record_path),
        },
        "heldout": {
            **heldout,
            "record_path": heldout_record_path.name,
            "record_sha256": sha256(heldout_record_path),
        },
        "schedule": {
            "path": schedule_path.name,
            "sha256": schedule_sha256,
            "members": len(schedule()),
            "pair_plan": [
                {
                    "pair": pair,
                    "treatment": treatment,
                    "participants": participants,
                    "order": "/".join(engines),
                }
                for pair, treatment, participants, engines in PAIR_PLAN
            ],
        },
        "campaign_binding": campaign_binding,
        "binaries": binaries,
        "runtime_artifact_hashes": runtime_artifact_hashes,
        "runtime_artifact_snapshots": runtime_artifact_snapshots,
        "builds": builds,
        "linked_faiss_build": linked_faiss_build,
        "source_provenance": {
            "benchmark_input_sources_path": benchmark_sources_name,
            "benchmark_input_sources_sha256": sha256(
                output_dir / benchmark_sources_name
            ),
            "compiled_source_hashes_path": sources_name,
            "compiled_source_hashes_sha256": sha256(output_dir / sources_name),
            "git_repositories": git_repositories,
            "faiss_audited_reference": faiss_audited_reference,
        },
        "protocol": {
            "audit_sidecar_schema_version": AUDIT_SIDECAR_SCHEMA_VERSION,
            "attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
            "attestation_barrier_phases": list(ATTESTATION_BARRIER_PHASES),
            "index_timing_binding_schema_version": (
                INDEX_TIMING_BINDING_SCHEMA_VERSION
            ),
            "index_timing_maximum_boundary_overhead_ns": (
                INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
            ),
            "process_evidence_schema_version": PROCESS_EVIDENCE_SCHEMA_VERSION,
            "time_wrapper_path": str(time_wrapper_path),
            "query_performance": query_protocol(),
            "heldout_query_count": HELDOUT_QUERY_COUNT,
            "heldout_query_seed": HELDOUT_QUERY_SEED,
            "recall_points": [{"k": k, "ef": ef} for k, ef in RECALL_POINTS],
            "idle_minimum_percent": args.idle_minimum_percent,
            "idle_policy": idle_policy(args.idle_minimum_percent),
            "idle_window_seconds": IDLE_WINDOW_SECONDS,
            "idle_sample_count": IDLE_SAMPLE_COUNT,
            "idle_to_launch_max_ns": IDLE_TO_LAUNCH_MAX_NS,
            "server_ports": list(SERVER_PORTS),
            "power_policy": POWER_POLICY,
            "battery_minimum_percent": BATTERY_MINIMUM_PERCENT,
            "global_swap_policy": GLOBAL_SWAP_POLICY,
            "member_timeout_seconds": args.member_timeout_seconds,
            "idle_timeout_seconds": args.idle_timeout_seconds,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path) -> None:
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            entries.append(f"{sha256(path)}  {path.relative_to(output_dir)}")
    (output_dir / "MANIFEST.sha256").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--infinity-binary", type=Path, required=True)
    parser.add_argument("--faiss-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-idle-minimum-percent",
        dest="idle_minimum_percent",
        type=float,
        default=IDLE_MINIMUM_PERCENT,
        help=(
            "Development-only CPU-idle override; defaults to the frozen 95%% "
            "gate and accepts values from 25 through 95"
        ),
    )
    parser.add_argument("--idle-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--member-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and archive build provenance without running benchmark members",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    for option in (
        "--repo",
        "--infinity-binary",
        "--faiss-binary",
        "--output-dir",
        "--development-idle-minimum-percent",
        "--idle-timeout-seconds",
        "--member-timeout-seconds",
        "--preflight-only",
    ):
        occurrences = sum(
            argument == option or argument.startswith(f"{option}=")
            for argument in arguments
        )
        if occurrences > 1:
            parser.error(f"{option} must not be repeated")
    args = parser.parse_args(arguments)
    args.repo = args.repo.resolve()
    for name in ("infinity_binary", "faiss_binary"):
        binary = getattr(args, name)
        if not binary.is_absolute():
            binary = args.repo / binary
        binary = binary.resolve()
        require(binary.is_file(), f"Binary not found: {binary}")
        setattr(args, name, binary)
    args.output_dir = args.output_dir.resolve()
    require(
        args.infinity_binary != args.faiss_binary,
        "Infinity and FAISS binary paths must differ",
    )
    require(
        args.idle_timeout_seconds >= IDLE_WINDOW_SECONDS,
        f"Idle timeout must be at least {IDLE_WINDOW_SECONDS} seconds",
    )
    validate_idle_minimum_percent(args.idle_minimum_percent)
    require(args.member_timeout_seconds > 0, "Member timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    reject_unsafe_parent_environment()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        dataset_path = args.output_dir / DATASET_FILENAME
        dataset = create_dataset(dataset_path)
        dataset_record_path = args.output_dir / "dataset.json"
        write_json(dataset_record_path, dataset)
        truth = heldout_truth(dataset_path, dataset)
        heldout = heldout_manifest(truth)
        heldout_record_path = args.output_dir / HELDOUT_MANIFEST_FILENAME
        write_json(heldout_record_path, heldout)

        all_schedule = schedule()
        schedule_path = args.output_dir / "schedule.json"
        write_json(schedule_path, all_schedule)

        preflight_record = preflight(
            args,
            args.output_dir,
            dataset=dataset,
            dataset_record_path=dataset_record_path,
            heldout=heldout,
            heldout_record_path=heldout_record_path,
            schedule_path=schedule_path,
        )
        write_json(args.output_dir / "preflight.json", preflight_record)
        if args.preflight_only:
            result = {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "scope": D0_SCOPE,
                "execution_mode": PREFLIGHT_ONLY_MODE,
                "status": "pass",
                "completed_at": utc_now(),
                "benchmark_members_executed": 0,
                "campaign_complete": False,
                "artifact_sha256": {
                    "dataset.json": sha256(dataset_record_path),
                    HELDOUT_MANIFEST_FILENAME: sha256(heldout_record_path),
                    "preflight.json": sha256(args.output_dir / "preflight.json"),
                    "schedule.json": sha256(schedule_path),
                },
            }
            write_json(args.output_dir / "preflight-result.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        records: list[dict[str, Any]] = []
        binary_hashes = {
            engine: preflight_record["binaries"][engine]["sha256"]
            for engine in ("infinity", "faiss")
        }
        runtime_artifact_hashes = preflight_record["runtime_artifact_hashes"]
        campaign_binding = preflight_record["campaign_binding"]
        for entry in all_schedule:
            records.append(
                run_member(
                    entry,
                    args=args,
                    output_dir=args.output_dir,
                    binary_hashes=binary_hashes,
                    campaign_binding=campaign_binding,
                    runtime_artifact_hashes=runtime_artifact_hashes,
                    dataset_path=dataset_path,
                    dataset=dataset,
                    truth=truth,
                )
            )
            write_json(args.output_dir / "runs.json", records)

        summary = summarize(
            records,
            dataset=dataset,
            schedule_sha256=sha256(schedule_path),
            idle_minimum_percent=args.idle_minimum_percent,
        )
        write_json(args.output_dir / "summary.json", summary)
        (args.output_dir / "README.md").write_text(
            render_summary(summary),
            encoding="utf-8",
        )
        # A failed acceptance gate is a completed negative result. Only
        # operational failures belong in failure.json.
        validate_completed_campaign(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException as error:
        write_json(
            args.output_dir / "failure.json",
            {
                "status": "fail",
                "scope": D0_SCOPE,
                "captured_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        write_manifest(args.output_dir)


if __name__ == "__main__":
    main()
