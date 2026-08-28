#!/usr/bin/env python3
"""Run the development-only Apple HNSW Batch4 causal comparison."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.apple_silicon import native_hnsw_d0 as d0
from tools.apple_silicon import native_hnsw_exactness as exactness
from tools.apple_silicon import verify_native_hnsw_exactness as exactness_verifier

CAUSAL_SCHEMA_VERSION = 16
REQUIRED_EXACTNESS_SCHEMA_VERSION = 3
FULL_CAMPAIGN_MODE = "full-campaign"
PREFLIGHT_ONLY_MODE = "preflight-only"
PERFORMANCE_MEASUREMENT_SCOPE = "authenticated-evidence-instrumented-builds"
RUNNER_CAPTURE_FILENAME = "runner-native_hnsw_batch4_causal.py"
VERIFIER_CAPTURE_FILENAME = "verifier-native_hnsw_batch4_causal.py"
IDEMPOTENCE_SCHEMA_VERSION = 11
CAMPAIGN_BINDING_SCHEMA_VERSION = 1
NINJA_EXECUTION_METHOD = "directory-fd-fchdir-exec-v1"
OUTPUT_REMOVAL_METHOD = "descriptor-relative-quarantine-rename-v1"
NINJA_VERSION = "1.13.2"
NINJA_LOG_VERSION = 7
NINJA_LOG_HEADER = f"# ninja log v{NINJA_LOG_VERSION}\n".encode("ascii")
NINJA_IDEMPOTENCE_PROGRESS = "[0/2]"
BUILD_PRODUCT_TARGETS = (
    ("benchmark", "infinity_hnsw_d0_production"),
    ("exactness-emitter", "infinity_hnsw_exactness_emitter_production"),
)
NINJA_INVOCATION_ORDER = (
    "base-clean-plan",
    "combined-clean-plan",
    "rebuild",
    "dependency-report-before",
    "settlement",
    "dependency-report-after",
    "idempotence",
)
BUILD_REBUILD_TIMEOUT_SECONDS = 600.0
POST_KILL_WAIT_SECONDS = 5.0
BUILD_TREE_MAX_DEPTH = 128
BUILD_TREE_MAX_ENTRIES = 250_000
NINJA_DEPS_DATABASE_MAX_BYTES = 64 * 1024 * 1024
NINJA_DEPS_MAX_OUTPUTS = 100_000
NINJA_DEPS_MAX_DEPENDENCIES_PER_OUTPUT = 65_536
NINJA_DEPS_MAX_TOTAL_DEPENDENCIES = 4_000_000
NINJA_DEPS_MAX_NODES = 250_000
NINJA_DEPS_MAX_RECORDS = 500_000
NINJA_DEPS_SIGNATURE = b"# ninjadeps\n"
NINJA_DEPS_VERSION = 4
NINJA_DEPS_MAX_RECORD_BYTES = (1 << 19) - 1
NINJA_LOG_MAX_BYTES = 64 * 1024 * 1024
STAT_MANIFEST_KEYS = (
    "device",
    "inode",
    "mode",
    "links",
    "bytes",
    "mtime_ns",
    "ctime_ns",
)
BUILD_TREE_DESCRIPTOR_APIS_SUPPORTED = (
    os.scandir in os.supports_fd
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.readlink in os.supports_dir_fd
)
DESCRIPTOR_QUARANTINE_APIS_SUPPORTED = (
    os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)
RENAME_EXCL = 0x00000004
RENAME_NOFOLLOW_ANY = 0x00000010
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEATX_NP = getattr(_LIBC, "renameatx_np", None)
if _RENAMEATX_NP is not None:
    _RENAMEATX_NP.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEATX_NP.restype = ctypes.c_int
DESCRIPTOR_QUARANTINE_APIS_SUPPORTED = (
    DESCRIPTOR_QUARANTINE_APIS_SUPPORTED
    and _RENAMEATX_NP is not None
)
MINIMUM_SPEEDUP = 1.05
MAXIMUM_RECALL_DEFICIT = 0.005
MAXIMUM_RELATIVE_MAD = 0.10
MINIMUM_QUERY_TPS_RATIO = 0.95
MAXIMUM_QUERY_LATENCY_RATIO = 1.05
MINIMUM_QUERY_RECALL = 0.99
MAXIMUM_QUERY_RECALL_GAP = 0.005
QUERY_LATENCY_PERCENTILES = ("p50_ns", "p95_ns", "p99_ns")
MINIMUM_SPEEDUP_FRACTION = (105, 100)
MAXIMUM_RELATIVE_MAD_FRACTION = (1, 10)
MINIMUM_QUERY_TPS_FRACTION = (95, 100)
MAXIMUM_QUERY_LATENCY_FRACTION = (105, 100)
MINIMUM_QUERY_RECALL_FRACTION = (99, 100)
MAXIMUM_RECALL_DEFICIT_FRACTION = (5, 1000)
MAXIMUM_QUERY_RECALL_GAP_FRACTION = (5, 1000)
LOCAL_PYTHON_MODULES = (
    (
        "native_hnsw_d0",
        "module-native_hnsw_d0.py",
        Path(d0.__file__).resolve(),
    ),
    (
        "verify_native_hnsw_d0",
        "module-verify_native_hnsw_d0.py",
        Path(__file__).resolve().with_name("verify_native_hnsw_d0.py"),
    ),
    (
        "native_hnsw_exactness",
        "module-native_hnsw_exactness.py",
        Path(exactness.__file__).resolve(),
    ),
    (
        "verify_native_hnsw_exactness",
        "module-verify_native_hnsw_exactness.py",
        Path(exactness_verifier.__file__).resolve(),
    ),
)
SHARED_BUILD_SETTINGS = (
    "CMAKE_BUILD_TYPE",
    "ENABLE_JEMALLOC",
    "CMAKE_OSX_ARCHITECTURES",
    "CMAKE_OSX_DEPLOYMENT_TARGET",
    "CMAKE_OSX_SYSROOT",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_ASM_COMPILER",
    "CMAKE_C_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_ASM_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_COMMAND",
    "CMAKE_LINKER",
    "CMAKE_MAKE_PROGRAM",
    "CMAKE_HOME_DIRECTORY",
    "CMAKE_TOOLCHAIN_FILE",
    "CMAKE_EXPORT_COMPILE_COMMANDS",
    "VCPKG_TARGET_TRIPLET",
    "VCPKG_HOST_TRIPLET",
    "VCPKG_OVERLAY_PORTS",
    "VCPKG_OVERLAY_TRIPLETS",
    "VCPKG_USE_HOST_TOOLS",
    "VCPKG_MANIFEST_INSTALL",
    "VCPKG_INSTALLED_DIR",
    "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION",
    "INFINITY_BUILD_TIME_OVERRIDE",
)
REQUIRED_PRODUCTION_BUILD_VALUES = {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
    "CMAKE_OSX_ARCHITECTURES": "arm64",
    "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
    "ENABLE_JEMALLOC": "OFF",
    "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION": "ON",
    "INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE": "ON",
    "INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE": "ON",
    "VCPKG_HOST_TRIPLET": "arm64-osx",
    "VCPKG_MANIFEST_INSTALL": "OFF",
    "VCPKG_TARGET_TRIPLET": "arm64-osx",
    "VCPKG_USE_HOST_TOOLS": "ON",
}
REQUIRED_NONEMPTY_BUILD_SETTINGS = (
    "CMAKE_ASM_COMPILER",
    "CMAKE_ASM_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_COMMAND",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_C_COMPILER",
    "CMAKE_C_COMPILER_CLANG_SCAN_DEPS",
    "CMAKE_HOME_DIRECTORY",
    "CMAKE_LINKER",
    "CMAKE_MAKE_PROGRAM",
    "CMAKE_OSX_SYSROOT",
    "CMAKE_TOOLCHAIN_FILE",
    "INFINITY_BUILD_TIME_OVERRIDE",
    "VCPKG_INSTALLED_DIR",
    "VCPKG_OVERLAY_PORTS",
    "VCPKG_OVERLAY_TRIPLETS",
)
BUILD_TOOL_SPECS = (
    ("c_compiler", "CMAKE_C_COMPILER", ("--version",)),
    ("compiler", "CMAKE_CXX_COMPILER", ("--version",)),
    ("asm_compiler", "CMAKE_ASM_COMPILER", ("--version",)),
    (
        "scan_deps",
        "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS",
        ("--version",),
    ),
    ("linker", "CMAKE_LINKER", ("-v",)),
    ("build_tool", "CMAKE_MAKE_PROGRAM", ("--version",)),
    ("cmake", "CMAKE_COMMAND", ("--version",)),
)

PAIR_PLAN = (
    ("C1", "correctness", 1, ("control", "treatment")),
    ("W1", "warmup", d0.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("W2", "warmup", d0.MEASURED_PARTICIPANTS, ("treatment", "control")),
    ("P1", "measured", d0.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("P2", "measured", d0.MEASURED_PARTICIPANTS, ("treatment", "control")),
    ("P3", "measured", d0.MEASURED_PARTICIPANTS, ("treatment", "control")),
    ("P4", "measured", d0.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("P5", "measured", d0.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("P6", "measured", d0.MEASURED_PARTICIPANTS, ("treatment", "control")),
)

CAUSAL_ROLE_IDS = {
    "control": 1,
    "treatment": 2,
}


class CausalFailure(RuntimeError):
    """Raised when causal evidence cannot satisfy its frozen protocol."""


@dataclass
class BuildDirectoryAnchor:
    path: Path
    descriptor: int
    device: int
    inode: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> BuildDirectoryAnchor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class VaryingSwitchSpec:
    name: str
    control_value: str
    treatment_value: str
    compile_sources: tuple[str, ...]

    @property
    def macro(self) -> str:
        return f"-D{self.name}"

    def value_for_role(self, role: str) -> str:
        require(role in ("control", "treatment"), f"Unknown causal role: {role}")
        return self.control_value if role == "control" else self.treatment_value

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "macro": self.macro,
            "control_value": self.control_value,
            "treatment_value": self.treatment_value,
            "compile_sources": list(self.compile_sources),
        }


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    varying_switches: tuple[VaryingSwitchSpec, ...]
    held_switches: tuple[tuple[str, str], ...]
    scope: str

    def switch_states_for_role(self, role: str) -> dict[str, str]:
        require(role in ("control", "treatment"), f"Unknown causal role: {role}")
        states: dict[str, str] = {}
        for switch in self.varying_switches:
            require(
                switch.name not in states,
                f"{self.name} repeats varying switch {switch.name}",
            )
            require(
                switch.control_value in ("ON", "OFF")
                and switch.treatment_value in ("ON", "OFF")
                and switch.control_value != switch.treatment_value,
                f"{self.name} has an invalid contrast for {switch.name}",
            )
            states[switch.name] = switch.value_for_role(role)
        for name, value in self.held_switches:
            require(
                name not in states,
                f"{self.name} declares {name} as both varying and held",
            )
            require(
                value in ("ON", "OFF"),
                f"{self.name} has an invalid held state for {name}",
            )
            states[name] = value
        return states

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "varying_switches": [
                switch.record() for switch in self.varying_switches
            ],
            "held_switches": [
                {"name": name, "value": value}
                for name, value in self.held_switches
            ],
            "scope": self.scope,
        }


TRAVERSAL_SWITCH = "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL"
RECIPROCAL_PRUNING_SWITCH = (
    "INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING"
)
INCREMENTAL_RECIPROCAL_SWITCH = (
    "INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL"
)
INCREMENTAL_RECIPROCAL_SHADOW_SWITCH = (
    "INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW"
)
INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH = (
    "INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS"
)
INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH = (
    "INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE"
)
THRESHOLD_BATCH4_TRAVERSAL_SWITCH = (
    "INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL"
)
THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH = (
    "INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE"
)
LVQ_CAPTURE_SWITCH = "INFINITY_ENABLE_HNSW_LVQ_CAPTURE"
COMPACT_VERTEX_LOCKS_SWITCH = "INFINITY_ENABLE_APPLE_HNSW_COMPACT_VERTEX_LOCKS"
HNSW_ALG_SOURCE = "/storage/knn_index/knn_hnsw/hnsw_alg.cppm"
HNSW_COMMON_SOURCE = "/storage/knn_index/knn_hnsw/hnsw_common.cppm"
HNSW_LVQ_CAPTURE_SOURCE = "/storage/knn_index/knn_hnsw/hnsw_lvq_capture.cppm"
DIST_FUNC_L2_SOURCE = "/storage/knn_index/knn_hnsw/dist_func_l2.cppm"
DATA_STORE_SOURCE = "/storage/knn_index/knn_hnsw/data_store/data_store.cppm"
HNSW_EXACTNESS_EMITTER_SOURCE = (
    "/tools/apple_silicon/native_hnsw_smoke/hnsw_exactness_emitter.cpp"
)

SWITCH_COMPILE_SOURCES = {
    TRAVERSAL_SWITCH: (HNSW_ALG_SOURCE,),
    RECIPROCAL_PRUNING_SWITCH: (HNSW_ALG_SOURCE,),
    INCREMENTAL_RECIPROCAL_SWITCH: (HNSW_ALG_SOURCE, HNSW_COMMON_SOURCE),
    INCREMENTAL_RECIPROCAL_SHADOW_SWITCH: (
        HNSW_ALG_SOURCE,
        HNSW_COMMON_SOURCE,
    ),
    INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH: (
        HNSW_ALG_SOURCE,
        HNSW_COMMON_SOURCE,
    ),
    INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH: (
        HNSW_ALG_SOURCE,
        HNSW_COMMON_SOURCE,
        HNSW_EXACTNESS_EMITTER_SOURCE,
    ),
    THRESHOLD_BATCH4_TRAVERSAL_SWITCH: (HNSW_ALG_SOURCE,),
    THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH: (
        HNSW_ALG_SOURCE,
        HNSW_COMMON_SOURCE,
        HNSW_EXACTNESS_EMITTER_SOURCE,
    ),
    LVQ_CAPTURE_SWITCH: (
        HNSW_LVQ_CAPTURE_SOURCE,
        HNSW_ALG_SOURCE,
        DIST_FUNC_L2_SOURCE,
    ),
    COMPACT_VERTEX_LOCKS_SWITCH: (DATA_STORE_SOURCE,),
}

EXPERIMENTS = {
    "incremental-reciprocal": ExperimentSpec(
        name="incremental-reciprocal",
        varying_switches=(
            VaryingSwitchSpec(
                name=INCREMENTAL_RECIPROCAL_SWITCH,
                control_value="OFF",
                treatment_value="ON",
                compile_sources=(HNSW_ALG_SOURCE, HNSW_COMMON_SOURCE),
            ),
        ),
        held_switches=(
            (TRAVERSAL_SWITCH, "OFF"),
            (RECIPROCAL_PRUNING_SWITCH, "ON"),
            (THRESHOLD_BATCH4_TRAVERSAL_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_SHADOW_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH, "OFF"),
            (LVQ_CAPTURE_SWITCH, "OFF"),
            (COMPACT_VERTEX_LOCKS_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH, "ON"),
            (THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH, "ON"),
        ),
        scope="d0-apple-hnsw-incremental-reciprocal-causal-development-only",
    ),
    "threshold-only": ExperimentSpec(
        name="threshold-only",
        varying_switches=(
            VaryingSwitchSpec(
                name=THRESHOLD_BATCH4_TRAVERSAL_SWITCH,
                control_value="OFF",
                treatment_value="ON",
                compile_sources=(HNSW_ALG_SOURCE,),
            ),
        ),
        held_switches=(
            (INCREMENTAL_RECIPROCAL_SWITCH, "OFF"),
            (TRAVERSAL_SWITCH, "OFF"),
            (RECIPROCAL_PRUNING_SWITCH, "ON"),
            (INCREMENTAL_RECIPROCAL_SHADOW_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH, "OFF"),
            (LVQ_CAPTURE_SWITCH, "OFF"),
            (COMPACT_VERTEX_LOCKS_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH, "ON"),
            (THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH, "ON"),
        ),
        scope="d0-apple-hnsw-threshold-only-causal-development-only",
    ),
    "threshold-after-incremental": ExperimentSpec(
        name="threshold-after-incremental",
        varying_switches=(
            VaryingSwitchSpec(
                name=THRESHOLD_BATCH4_TRAVERSAL_SWITCH,
                control_value="OFF",
                treatment_value="ON",
                compile_sources=(HNSW_ALG_SOURCE,),
            ),
        ),
        held_switches=(
            (INCREMENTAL_RECIPROCAL_SWITCH, "ON"),
            (TRAVERSAL_SWITCH, "OFF"),
            (RECIPROCAL_PRUNING_SWITCH, "ON"),
            (INCREMENTAL_RECIPROCAL_SHADOW_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH, "OFF"),
            (LVQ_CAPTURE_SWITCH, "OFF"),
            (COMPACT_VERTEX_LOCKS_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH, "ON"),
            (THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH, "ON"),
        ),
        scope=(
            "d0-apple-hnsw-threshold-after-incremental-"
            "causal-development-only"
        ),
    ),
    "combined-optimized": ExperimentSpec(
        name="combined-optimized",
        varying_switches=(
            VaryingSwitchSpec(
                name=INCREMENTAL_RECIPROCAL_SWITCH,
                control_value="OFF",
                treatment_value="ON",
                compile_sources=(HNSW_ALG_SOURCE, HNSW_COMMON_SOURCE),
            ),
            VaryingSwitchSpec(
                name=THRESHOLD_BATCH4_TRAVERSAL_SWITCH,
                control_value="OFF",
                treatment_value="ON",
                compile_sources=(HNSW_ALG_SOURCE,),
            ),
        ),
        held_switches=(
            (TRAVERSAL_SWITCH, "OFF"),
            (RECIPROCAL_PRUNING_SWITCH, "ON"),
            (INCREMENTAL_RECIPROCAL_SHADOW_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH, "OFF"),
            (LVQ_CAPTURE_SWITCH, "OFF"),
            (COMPACT_VERTEX_LOCKS_SWITCH, "OFF"),
            (INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH, "ON"),
            (THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH, "ON"),
        ),
        scope="d0-apple-hnsw-combined-optimized-causal-development-only",
    ),
}

INCREMENTAL_WITNESS_FIELDS = (
    "incremental_treatment_compiled",
    "incremental_capture_armed",
    "incremental_eligible_branch_entered",
    "incremental_successful_unchanged_observed",
    "incremental_successful_updated_observed",
)
THRESHOLD_WITNESS_FIELDS = (
    "threshold_treatment_compiled",
    "threshold_capture_armed",
    "threshold_eligible_branch_entered",
    "threshold_rejected_lane_observed",
    "threshold_surviving_lane_observed",
)
WITNESS_FIELDS = INCREMENTAL_WITNESS_FIELDS + THRESHOLD_WITNESS_FIELDS
require_witness_field_alignment = tuple(d0.EXECUTION_WITNESS_FIELDS)
if WITNESS_FIELDS != require_witness_field_alignment:
    raise RuntimeError("Causal and D0 execution-witness fields differ")
del require_witness_field_alignment
EXPECTED_OPTIMIZATION_ROLE_MATRIX = {
    "incremental-reciprocal": {
        "control": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "OFF",
            INCREMENTAL_RECIPROCAL_SWITCH: "OFF",
        },
        "treatment": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "OFF",
            INCREMENTAL_RECIPROCAL_SWITCH: "ON",
        },
    },
    "threshold-only": {
        "control": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "OFF",
            INCREMENTAL_RECIPROCAL_SWITCH: "OFF",
        },
        "treatment": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "ON",
            INCREMENTAL_RECIPROCAL_SWITCH: "OFF",
        },
    },
    "threshold-after-incremental": {
        "control": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "OFF",
            INCREMENTAL_RECIPROCAL_SWITCH: "ON",
        },
        "treatment": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "ON",
            INCREMENTAL_RECIPROCAL_SWITCH: "ON",
        },
    },
    "combined-optimized": {
        "control": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "OFF",
            INCREMENTAL_RECIPROCAL_SWITCH: "OFF",
        },
        "treatment": {
            THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "ON",
            INCREMENTAL_RECIPROCAL_SWITCH: "ON",
        },
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalFailure(message)


def validate_experiment_spec(
    experiment: ExperimentSpec,
    *,
    context: str,
) -> None:
    require(
        bool(experiment.varying_switches),
        f"{context} has no varying switches",
    )
    require(
        experiment.name in EXPECTED_OPTIMIZATION_ROLE_MATRIX,
        f"{context} is not a schema-16 experiment",
    )
    varying_names = [switch.name for switch in experiment.varying_switches]
    held_names = [name for name, _ in experiment.held_switches]
    require(
        len(varying_names) == len(set(varying_names)),
        f"{context} repeats a varying switch",
    )
    require(
        len(held_names) == len(set(held_names)),
        f"{context} repeats a held switch",
    )
    require(
        set(varying_names).isdisjoint(held_names),
        f"{context} overlaps varying and held switches",
    )
    for switch in experiment.varying_switches:
        require(
            switch.name in SWITCH_COMPILE_SOURCES,
            f"{context} varying switch {switch.name} has no source map",
        )
        require(
            switch.compile_sources == SWITCH_COMPILE_SOURCES[switch.name]
            and bool(switch.compile_sources)
            and len(switch.compile_sources) == len(set(switch.compile_sources)),
            f"{context} varying switch {switch.name} source map differs",
        )
        require(
            switch.control_value in ("ON", "OFF")
            and switch.treatment_value in ("ON", "OFF")
            and switch.control_value != switch.treatment_value,
            f"{context} varying switch {switch.name} role values are invalid",
        )
    for name, value in experiment.held_switches:
        require(
            name in SWITCH_COMPILE_SOURCES,
            f"{context} held switch {name} has no source map",
        )
        require(
            value in ("ON", "OFF"),
            f"{context} held switch {name} value is invalid",
        )
    for role in ("control", "treatment"):
        states = experiment.switch_states_for_role(role)
        required_held_states = {
            TRAVERSAL_SWITCH: "OFF",
            RECIPROCAL_PRUNING_SWITCH: "ON",
            INCREMENTAL_RECIPROCAL_SHADOW_SWITCH: "OFF",
            INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH: "OFF",
            LVQ_CAPTURE_SWITCH: "OFF",
            COMPACT_VERTEX_LOCKS_SWITCH: "OFF",
            INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH: "ON",
            THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH: "ON",
        }
        require(
            all(states.get(name) == value for name, value in required_held_states.items()),
            f"{context} {role} common held-state contract differs",
        )
        require(
            INCREMENTAL_RECIPROCAL_SWITCH in states
            and THRESHOLD_BATCH4_TRAVERSAL_SWITCH in states,
            f"{context} {role} does not bind both witnessed optimizations",
        )
        require(
            all(
                states[name] == value
                for name, value in EXPECTED_OPTIMIZATION_ROLE_MATRIX[
                    experiment.name
                ][role].items()
            ),
            f"{context} {role} optimization role matrix differs",
        )


require(
    set(EXPERIMENTS) == set(EXPECTED_OPTIMIZATION_ROLE_MATRIX),
    "Schema-15 experiment allowlist and role matrix differ",
)
for _experiment_name, _experiment in EXPERIMENTS.items():
    validate_experiment_spec(
        _experiment,
        context=f"experiment {_experiment_name}",
    )
del _experiment_name
del _experiment


def expected_witnesses_for_experiment(
    experiment: ExperimentSpec,
) -> dict[str, dict[str, int]]:
    validate_experiment_spec(experiment, context=f"experiment {experiment.name}")
    expected: dict[str, dict[str, int]] = {}
    for role in ("control", "treatment"):
        states = experiment.switch_states_for_role(role)
        require(
            INCREMENTAL_RECIPROCAL_SWITCH in states
            and THRESHOLD_BATCH4_TRAVERSAL_SWITCH in states,
            f"{experiment.name} does not bind both witnessed optimizations",
        )
        incremental = int(states[INCREMENTAL_RECIPROCAL_SWITCH] == "ON")
        threshold = int(states[THRESHOLD_BATCH4_TRAVERSAL_SWITCH] == "ON")
        expected[role] = {
            **{name: incremental for name in INCREMENTAL_WITNESS_FIELDS},
            **{name: threshold for name in THRESHOLD_WITNESS_FIELDS},
        }
    return expected


def validate_expected_witnesses(
    value: Any,
    *,
    experiment: ExperimentSpec,
    context: str,
) -> dict[str, dict[str, int]]:
    require(type(value) is dict, f"{context} must be an object")
    require(
        set(value) == {"control", "treatment"},
        f"{context} must contain exactly control and treatment",
    )
    normalized: dict[str, dict[str, int]] = {}
    for role in ("control", "treatment"):
        witness = value[role]
        require(type(witness) is dict, f"{context} {role} must be an object")
        require(
            set(witness) == set(WITNESS_FIELDS),
            f"{context} {role} fields differ",
        )
        require(
            all(type(bit) is int and bit in (0, 1) for bit in witness.values()),
            f"{context} {role} contains a non-binary witness value",
        )
        normalized[role] = {
            name: witness[name] for name in WITNESS_FIELDS
        }
    require(
        normalized == expected_witnesses_for_experiment(experiment),
        f"{context} differs from the experiment role matrix",
    )
    return normalized


def validate_run_execution_witness(
    value: Any,
    *,
    experiment: ExperimentSpec,
    role: str,
    context: str,
) -> dict[str, int]:
    require(role in ("control", "treatment"), f"{context} role is invalid")
    require(type(value) is dict, f"{context} must be an object")
    require(set(value) == set(WITNESS_FIELDS), f"{context} fields differ")
    normalized: dict[str, int] = {}
    for name in WITNESS_FIELDS:
        bit = value[name]
        require(
            type(bit) is int and bit in (0, 1),
            f"{context} {name} is not binary",
        )
        normalized[name] = bit
    expected = expected_witnesses_for_experiment(experiment)[role]
    require(normalized == expected, f"{context} differs from the role matrix")
    for prefix, fields in (
        ("incremental", INCREMENTAL_WITNESS_FIELDS),
        ("threshold", THRESHOLD_WITNESS_FIELDS),
    ):
        compiled, armed, eligible, first_outcome, second_outcome = (
            normalized[name] for name in fields
        )
        require(
            not (armed or eligible or first_outcome or second_outcome)
            or compiled == 1,
            f"{context} {prefix} evidence exists without compiled treatment",
        )
        require(
            not (eligible or first_outcome or second_outcome) or armed == 1,
            f"{context} {prefix} evidence exists without an armed capture",
        )
        require(
            not (first_outcome or second_outcome) or eligible == 1,
            f"{context} {prefix} outcome exists without eligibility",
        )
    return normalized


def create_campaign_binding(
    *,
    roles: dict[str, dict[str, Any]],
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
    role_records: dict[str, Any] = {}
    for role, role_id in CAUSAL_ROLE_IDS.items():
        binary_sha256 = str(roles.get(role, {}).get("binary", {}).get("sha256", ""))
        require(
            re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is not None,
            f"Campaign {role} binary SHA-256 is invalid",
        )
        role_records[role] = {
            "role_id": role_id,
            "engine": "infinity",
            "binary_sha256": binary_sha256,
        }
    return {
        "schema_version": CAMPAIGN_BINDING_SCHEMA_VERSION,
        "campaign_nonce": secrets.token_hex(32),
        "dataset_sha256": dataset_sha256,
        "schedule_sha256": schedule_sha256,
        "roles": role_records,
    }


def member_campaign_binding(
    entry: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    role = str(entry["role"])
    require(role in CAUSAL_ROLE_IDS, f"Unknown campaign role: {role}")
    require(
        campaign.get("schema_version") == CAMPAIGN_BINDING_SCHEMA_VERSION,
        "Campaign binding schema differs",
    )
    role_record = campaign.get("roles", {}).get(role)
    require(isinstance(role_record, dict), f"Campaign binding omits {role}")
    require(
        role_record
        == {
            "role_id": CAUSAL_ROLE_IDS[role],
            "engine": "infinity",
            "binary_sha256": role_record.get("binary_sha256"),
        },
        f"Campaign binding for {role} differs",
    )
    result = {
        "campaign_nonce": campaign.get("campaign_nonce"),
        "schedule_sequence": int(entry["sequence"]),
        "role_id": CAUSAL_ROLE_IDS[role],
        "binary_sha256": role_record["binary_sha256"],
        "dataset_sha256": campaign.get("dataset_sha256"),
    }
    for name in ("campaign_nonce", "binary_sha256", "dataset_sha256"):
        require(
            isinstance(result[name], str)
            and re.fullmatch(r"[0-9a-f]{64}", result[name]) is not None,
            f"Campaign member {name} is invalid",
        )
    return result


def canonical_repo(repo: Path, *, label: str) -> Path:
    require(repo.is_absolute(), f"{label} repository path is not absolute")
    resolved = repo.resolve()
    require(repo == resolved, f"{label} repository path is not canonical")
    return resolved


def canonical_source(repo: Path, suffix: str) -> str:
    require(suffix.startswith("/"), f"Compile-source suffix is not absolute: {suffix}")
    if suffix.startswith("/tools/"):
        return str(repo / suffix.removeprefix("/"))
    return str(repo / "src" / suffix.removeprefix("/"))


def validate_cmake_home_directory(
    settings: dict[str, Any],
    *,
    repo: Path,
    label: str,
) -> None:
    repo = canonical_repo(repo, label=label)
    home_directory = settings.get("CMAKE_HOME_DIRECTORY")
    require(
        isinstance(home_directory, str)
        and Path(home_directory).is_absolute()
        and Path(home_directory).resolve() == repo,
        f"{label} CMAKE_HOME_DIRECTORY is not the canonical repository",
    )


def scan_compiler_macro_arguments(
    arguments: list[str],
    switches: set[str],
) -> tuple[
    dict[str, bool],
    dict[str, set[int]],
    dict[str, list[tuple[str, str, int]]],
]:
    require(
        all(isinstance(argument, str) for argument in arguments),
        "Compile arguments contain a non-string value",
    )
    states = {switch: False for switch in switches}
    argument_indexes = {switch: set() for switch in switches}
    occurrences: dict[str, list[tuple[str, str, int]]] = {
        switch: [] for switch in switches
    }
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        operation: str | None = None
        payload: str | None = None
        consumed = {index}
        if argument in ("-D", "-U"):
            require(
                index + 1 < len(arguments),
                f"Compiler option {argument} has no macro argument",
            )
            operation = argument[1]
            payload = arguments[index + 1]
            consumed.add(index + 1)
            index += 2
        elif argument.startswith("-D") and len(argument) > 2:
            operation = "D"
            payload = argument[2:]
            index += 1
        elif argument.startswith("-U") and len(argument) > 2:
            operation = "U"
            payload = argument[2:]
            index += 1
        else:
            index += 1
            continue

        name = payload.partition("=")[0] if operation == "D" else payload
        if name not in switches:
            continue
        states[name] = operation == "D"
        argument_indexes[name].update(consumed)
        occurrences[name].append((operation, payload, len(consumed)))
    return states, argument_indexes, occurrences


def without_argument_indexes(arguments: list[str], indexes: set[int]) -> list[str]:
    return [
        argument
        for index, argument in enumerate(arguments)
        if index not in indexes
    ]


def validate_compile_macro_states(
    arguments_by_source: dict[str, list[str]],
    *,
    experiment: ExperimentSpec,
    role: str,
    repo: Path,
) -> dict[str, dict[str, Any]]:
    require(role in ("control", "treatment"), f"Unknown causal role: {role}")
    repo = canonical_repo(repo, label=role)
    for switch in experiment.varying_switches:
        require(
            SWITCH_COMPILE_SOURCES.get(switch.name)
            == switch.compile_sources,
            f"{role} varying-source specification differs for {switch.name}",
        )
    switch_values = experiment.switch_states_for_role(role)
    require(
        all(name in SWITCH_COMPILE_SOURCES for name in switch_values),
        f"{role} experiment names a switch without a frozen source map",
    )
    switches = set(switch_values)
    expected_sources = {
        switch: {
            canonical_source(repo, suffix)
            for suffix in SWITCH_COMPILE_SOURCES[switch]
        }
        for switch in switches
    }

    for switch, suffixes in (
        (name, SWITCH_COMPILE_SOURCES[name]) for name in switches
    ):
        for suffix in suffixes:
            expected = canonical_source(repo, suffix)
            matching = sorted(
                source for source in arguments_by_source if source.endswith(suffix)
            )
            require(
                matching == [expected],
                f"{role} compile source for {switch} and {suffix} is not exactly "
                f"the canonical repository source",
            )

    scans: dict[str, dict[str, Any]] = {}
    for source, arguments in arguments_by_source.items():
        states, indexes, occurrences = scan_compiler_macro_arguments(
            arguments,
            switches,
        )
        scans[source] = {
            "states": states,
            "argument_indexes": indexes,
            "occurrences": occurrences,
        }
        for switch, value in switch_values.items():
            expected_defined = (
                value == "ON" and source in expected_sources[switch]
            )
            expected_occurrences = (
                [("D", switch, 1)] if expected_defined else []
            )
            require(
                states[switch] is expected_defined
                and occurrences[switch] == expected_occurrences,
                f"{role} effective macro state for {switch} on {source} has "
                f"encoding {occurrences[switch]!r}; expected "
                f"{'defined once canonically' if expected_defined else 'undefined with no compiler option'}",
            )
    return scans


def exact_fraction(value: Fraction | float | int) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def fraction_record(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def exact_median(values: list[Fraction]) -> Fraction:
    require(bool(values), "Cannot calculate an exact median from no values")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def exact_relative_mad(values: list[Fraction]) -> Fraction:
    center = exact_median(values)
    require(center > 0, "Exact relative MAD center is not positive")
    deviations = [abs(value - center) for value in values]
    return exact_median(deviations) / center


def exact_relative_mad_at_most(
    relative_mad: Fraction,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    return relative_mad <= Fraction(maximum_numerator, maximum_denominator)


def ratio_at_least(
    numerator: Fraction | float | int,
    denominator: Fraction | float | int,
    minimum_numerator: int,
    minimum_denominator: int,
) -> bool:
    return (
        exact_fraction(numerator) * minimum_denominator
        >= exact_fraction(denominator) * minimum_numerator
    )


def ratio_at_most(
    numerator: Fraction | float | int,
    denominator: Fraction | float | int,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    return (
        exact_fraction(numerator) * maximum_denominator
        <= exact_fraction(denominator) * maximum_numerator
    )


def difference_at_most(
    minuend: Fraction | float | int,
    subtrahend: Fraction | float | int,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    return (
        exact_fraction(minuend) - exact_fraction(subtrahend)
        <= Fraction(maximum_numerator, maximum_denominator)
    )


def absolute_difference_at_most(
    left: Fraction | float | int,
    right: Fraction | float | int,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    return (
        abs(exact_fraction(left) - exact_fraction(right))
        <= Fraction(maximum_numerator, maximum_denominator)
    )


def build_setting_names(experiment: ExperimentSpec) -> tuple[str, ...]:
    control_states = experiment.switch_states_for_role("control")
    treatment_states = experiment.switch_states_for_role("treatment")
    require(
        tuple(control_states) == tuple(treatment_states),
        f"{experiment.name} role switch sets differ",
    )
    return (
        *SHARED_BUILD_SETTINGS,
        *control_states,
    )


def validate_switch_states(
    cache: dict[str, str],
    *,
    role: str,
    experiment: ExperimentSpec,
) -> None:
    require(
        cache.get("ENABLE_JEMALLOC") == "OFF",
        f"{role} ENABLE_JEMALLOC is not OFF",
    )
    require(
        cache.get("VCPKG_MANIFEST_INSTALL") == "OFF",
        f"{role} VCPKG_MANIFEST_INSTALL is not OFF",
    )
    for name, value in experiment.switch_states_for_role(role).items():
        require(
            cache.get(name) == value,
            f"{role} switch {name} is not {value}",
        )


def validate_production_build_settings(
    cache: dict[str, str],
    *,
    role: str,
) -> None:
    for name, expected in REQUIRED_PRODUCTION_BUILD_VALUES.items():
        require(
            cache.get(name) == expected,
            f"{role} production build requires {name}={expected}",
        )
    for name in REQUIRED_NONEMPTY_BUILD_SETTINGS:
        require(
            isinstance(cache.get(name), str) and cache[name] != "",
            f"{role} production build setting {name} is missing",
        )
    scan_deps_paths = d0.configured_scan_deps_paths(cache, label=role)
    require(
        len(scan_deps_paths) == 1
        and all(
            cache[name] == scan_deps_paths[0]
            for name in d0.CMAKE_SCAN_DEPS_CACHE_KEYS
        ),
        f"{role} production build must use one configured dependency scanner",
    )


def role_binary(args: argparse.Namespace, role: str) -> Path:
    require(role in ("control", "treatment"), f"Unknown causal role: {role}")
    return args.control_binary if role == "control" else args.treatment_binary


def ordered_build_products(
    binary: Path,
    *,
    build_directory: Path,
    label: str,
) -> list[dict[str, Any]]:
    require(
        binary.name == BUILD_PRODUCT_TARGETS[0][1],
        f"{label} benchmark binary has an unexpected name: {binary.name}",
    )
    outputs = [
        binary,
        binary.with_name(BUILD_PRODUCT_TARGETS[1][1]),
    ]
    require(
        len(outputs) == len(BUILD_PRODUCT_TARGETS),
        f"{label} build product definition count differs",
    )
    products: list[dict[str, Any]] = []
    for (name, expected_target), output in zip(
        BUILD_PRODUCT_TARGETS,
        outputs,
    ):
        require(
            output.is_file()
            and output.resolve(strict=True) == output
            and d0.find_build_metadata(output)[0] == build_directory,
            f"{label} {name} output is not a canonical file in the role build: "
            f"{output}",
        )
        target = d0.ninja_target_for_output(
            output,
            build_directory=build_directory,
            label=f"{label} {name}",
        )
        require(
            Path(target).name == expected_target,
            f"{label} {name} target differs: {target}",
        )
        products.append(
            {
                "name": name,
                "requested_target": target,
                "expected_output": output,
            }
        )
    return products


def schedule() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pair, phase, participants, roles in PAIR_PLAN:
        pair_order = "/".join(roles)
        for role in roles:
            entries.append(
                {
                    "sequence": len(entries),
                    "pair": pair,
                    "phase": phase,
                    "pair_order": pair_order,
                    "role": role,
                    "engine": "infinity",
                    "participants": participants,
                }
            )
    validate_schedule(entries)
    return entries


def validate_schedule(entries: list[dict[str, Any]]) -> None:
    expected: list[dict[str, Any]] = []
    for pair, phase, participants, roles in PAIR_PLAN:
        for role in roles:
            expected.append(
                {
                    "sequence": len(expected),
                    "pair": pair,
                    "phase": phase,
                    "pair_order": "/".join(roles),
                    "role": role,
                    "engine": "infinity",
                    "participants": participants,
                }
            )
    require(entries == expected, "Schedule differs from the frozen causal order")
    measured = [
        entry
        for entry in entries
        if entry["phase"] == "measured" and entry["role"] == "control"
    ]
    require(len(measured) == 6, "Causal schedule must contain six measured pairs")
    require(
        sum(entry["pair_order"] == "control/treatment" for entry in measured) == 3,
        "Causal schedule lacks three control/treatment pairs",
    )
    require(
        sum(entry["pair_order"] == "treatment/control" for entry in measured) == 3,
        "Causal schedule lacks three treatment/control pairs",
    )


def validate_records_against_schedule(records: list[dict[str, Any]]) -> None:
    expected = schedule()
    require(len(records) == len(expected), "Records do not cover the full schedule")
    keys = (
        "sequence",
        "pair",
        "phase",
        "pair_order",
        "role",
        "engine",
        "participants",
    )
    for index, (record, entry) in enumerate(zip(records, expected)):
        require(
            {key: record.get(key) for key in keys}
            == {key: entry[key] for key in keys},
            f"Run record {index} differs from its scheduled member",
        )


def normalized_dependency_identity(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "install_names": dependency["install_names"],
            "resolved_path": dependency["resolved_path"],
            "sha256": dependency["sha256"],
            "bytes": dependency["bytes"],
            "unavailable_reason": dependency["unavailable_reason"],
        }
        for dependency in manifest["dependencies"]
    ]


def canonical_compile_arguments(
    entry: dict[str, Any],
    *,
    build_directory: Path,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> list[str]:
    raw = d0.compile_entry_arguments(entry, label=label, index=0)
    expanded = d0.expand_response_arguments(
        raw,
        working_directory=Path(str(entry["directory"])),
        label=label,
        response_references=response_references,
    )
    normalized = d0.normalized_compile_arguments(expanded)
    build_prefix = str(build_directory)
    return [argument.replace(build_prefix, "$BUILD") for argument in normalized]


def canonical_build_path(path: str, *, build_directory: Path) -> str:
    build_prefix = str(build_directory)
    if path == build_prefix:
        return "$BUILD"
    descendant_prefix = f"{build_prefix}{os.sep}"
    if path.startswith(descendant_prefix):
        return f"$BUILD/{path[len(descendant_prefix):]}"
    return path


def compile_argument_map(
    entries: list[dict[str, Any]],
    *,
    build_directory: Path,
    label: str,
    response_references: set[tuple[Path, Path]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for index, entry in enumerate(entries):
        source = canonical_build_path(
            str(d0.source_path_from_compile_entry(entry)),
            build_directory=build_directory,
        )
        require(source not in result, f"{label} duplicates compile source {source}")
        result[source] = canonical_compile_arguments(
            entry,
            build_directory=build_directory,
            label=f"{label} compile entry {index}",
            response_references=response_references,
        )
    return result


def canonical_link_arguments(
    arguments: list[str],
    *,
    build_directory: Path,
) -> list[str]:
    build_prefix = str(build_directory)
    return [argument.replace(build_prefix, "$BUILD") for argument in arguments]


def closure_input_identity(
    closure: dict[str, Any],
    *,
    build_directory: Path,
) -> list[dict[str, Any]]:
    derived_paths = {
        str(unit["output"])
        for unit in closure["compile_units"]
    }
    for unit in closure["compile_units"]:
        derived_paths.update(str(path) for path in unit["module_outputs"])
    derived_paths.update(
        str(record["output"]) for record in closure["dependency_graph"]
    )
    derived_paths.update(
        str(path)
        for path in closure["selected_target_graph"]["derived_link_outputs"]
    )
    products = closure.get("products")
    require(
        isinstance(products, list)
        and [product.get("name") for product in products]
        == [name for name, _ in BUILD_PRODUCT_TARGETS],
        "Build-input identity products differ from the ordered campaign products",
    )
    derived_paths.update(
        str(product["expected_output"])
        for product in products
    )
    build_prefix = str(build_directory)
    result: list[dict[str, Any]] = []
    for record in closure["files"]:
        path = str(record["path"])
        roles = [str(role) for role in record["roles"]]
        if (
            path in derived_paths
            or any(
                role in {"compile-output", "module-output", "ninja-output"}
                for role in roles
            )
        ):
            continue
        result.append(
            {
                "path": path.replace(build_prefix, "$BUILD"),
                "resolved_path": str(record["resolved_path"]).replace(
                    build_prefix,
                    "$BUILD",
                ),
                "type": record["type"],
                "roles": roles,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
        )
    return sorted(
        result,
        key=lambda record: (
            str(record["path"]).encode("utf-8"),
            str(record["resolved_path"]).encode("utf-8"),
        ),
    )


def artifact_reference(path: Path) -> dict[str, Any]:
    return {
        "captured_path": path.name,
        "sha256": d0.sha256(path),
        "bytes": path.stat().st_size,
    }


def stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def stat_manifest_fields(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "links": metadata.st_nlink,
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def descriptor_path(descriptor: int, *, context: str) -> Path:
    getpath = getattr(fcntl, "F_GETPATH", None)
    require(getpath is not None, f"{context} requires F_GETPATH")
    try:
        raw_path = fcntl.fcntl(descriptor, getpath, b"\0" * 1024)
    except OSError as error:
        raise CausalFailure(
            f"{context} descriptor path could not be inspected: {error}"
        ) from error
    require(
        b"\0" in raw_path,
        f"{context} descriptor path is not terminated",
    )
    path_text = os.fsdecode(raw_path.split(b"\0", 1)[0])
    try:
        path_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CausalFailure(
            f"{context} descriptor path is not UTF-8"
        ) from error
    path = Path(path_text)
    require(
        path_text != "" and path.is_absolute(),
        f"{context} descriptor path is invalid",
    )
    return path


def open_build_directory_anchor(build_directory: Path) -> BuildDirectoryAnchor:
    context = "Build directory anchor"
    require(
        build_directory.is_absolute()
        and os.path.abspath(build_directory) == str(build_directory),
        f"{context} path is not canonical absolute",
    )
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    require(
        directory_flag != 0 and nofollow != 0,
        f"{context} requires O_DIRECTORY and O_NOFOLLOW",
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | directory_flag
        | nofollow
    )
    try:
        lexical_before = build_directory.lstat()
        resolved_before = build_directory.resolve(strict=True)
        descriptor = os.open(build_directory, flags)
    except OSError as error:
        raise CausalFailure(f"{context} could not be opened safely: {error}") from error
    try:
        opened = os.fstat(descriptor)
        require(
            stat.S_ISDIR(lexical_before.st_mode)
            and not stat.S_ISLNK(lexical_before.st_mode)
            and resolved_before == build_directory
            and stat_identity(lexical_before) == stat_identity(opened)
            and descriptor_path(descriptor, context=context) == build_directory,
            f"{context} changed while being opened",
        )
        anchor = BuildDirectoryAnchor(
            path=build_directory,
            descriptor=descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        require_build_directory_binding(anchor, context=context)
        return anchor
    except BaseException:
        os.close(descriptor)
        raise


def require_build_directory_binding(
    anchor: BuildDirectoryAnchor,
    *,
    context: str,
) -> None:
    require(anchor.descriptor >= 0, f"{context} descriptor is closed")
    try:
        opened = os.fstat(anchor.descriptor)
        lexical = anchor.path.lstat()
        resolved = anchor.path.resolve(strict=True)
    except OSError as error:
        raise CausalFailure(f"{context} binding could not be inspected: {error}") from error
    require(
        stat.S_ISDIR(opened.st_mode)
        and stat.S_ISDIR(lexical.st_mode)
        and not stat.S_ISLNK(lexical.st_mode)
        and (opened.st_dev, opened.st_ino) == (anchor.device, anchor.inode)
        and stat_identity(opened) == stat_identity(lexical)
        and descriptor_path(anchor.descriptor, context=context) == anchor.path
        and resolved == anchor.path,
        f"{context} pathname no longer names the opened build directory",
    )


def require_ninja_lock_absent(
    anchor: BuildDirectoryAnchor,
    *,
    invocation: str,
    phase: str,
    observations: list[dict[str, Any]],
) -> None:
    require(
        invocation in NINJA_INVOCATION_ORDER and phase in {"before", "after"},
        "Ninja lock observation has an invalid invocation or phase",
    )
    try:
        os.stat(
            ".ninja_lock",
            dir_fd=anchor.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError as error:
        raise CausalFailure(
            f"{invocation} Ninja lock could not be inspected {phase}: {error}"
        ) from error
    else:
        raise CausalFailure(
            f"{invocation} Ninja lock is present {phase} anchored execution"
        )
    observations.append(
        {
            "sequence": len(observations),
            "invocation": invocation,
            "phase": phase,
            "relative_path": ".ninja_lock",
            "absent": True,
        }
    )


_ANCHORED_NINJA_SHIM = (
    "import os,stat,sys\n"
    "fd=int(sys.argv[1]);dev=int(sys.argv[2]);ino=int(sys.argv[3])\n"
    "value=os.fstat(fd)\n"
    "if not stat.S_ISDIR(value.st_mode) or "
    "(value.st_dev,value.st_ino)!=(dev,ino): os._exit(125)\n"
    "os.fchdir(fd)\n"
    "os.close(fd)\n"
    "os.execv(sys.argv[4],sys.argv[4:])\n"
)


def _execute_anchored_ninja_process(
    command: list[str],
    *,
    anchor: BuildDirectoryAnchor,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    wrapper_command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _ANCHORED_NINJA_SHIM,
        str(anchor.descriptor),
        str(anchor.device),
        str(anchor.inode),
        *command,
    ]
    try:
        process = subprocess.Popen(
            wrapper_command,
            pass_fds=(anchor.descriptor,),
            env=d0.tool_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        raise CausalFailure(f"Anchored Ninja could not be started: {error}") from error

    def kill_close_and_reap(*, failure: str) -> None:
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            cleanup_errors.append(("kill process group", error))
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            if pipe is not None:
                try:
                    pipe.close()
                except BaseException as error:
                    cleanup_errors.append((f"close {name}", error))
        try:
            process.wait(timeout=POST_KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired as wait_error:
            raise CausalFailure(
                f"{failure} and could not be reaped within "
                f"{POST_KILL_WAIT_SECONDS} seconds: {' '.join(command)}"
            ) from wait_error
        except BaseException as error:
            cleanup_errors.append(("reap process", error))
        if cleanup_errors:
            operations = ", ".join(operation for operation, _ in cleanup_errors)
            raise CausalFailure(
                f"{failure} cleanup failed during {operations}: "
                f"{' '.join(command)}"
            ) from cleanup_errors[0][1]

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        kill_close_and_reap(failure="Anchored Ninja timed out")
        raise CausalFailure(
            f"Anchored Ninja timed out after {timeout} seconds: "
            f"{' '.join(command)}"
        ) from error
    except BaseException as error:
        kill_close_and_reap(failure="Anchored Ninja communication failed")
        if not isinstance(error, Exception):
            raise
        raise CausalFailure(
            f"Anchored Ninja communication failed: {' '.join(command)}"
        ) from error
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run_anchored_ninja(
    command: list[str],
    *,
    anchor: BuildDirectoryAnchor,
    invocation: str,
    observations: list[dict[str, Any]],
    check: bool = True,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    require(
        command
        and Path(command[0]).is_absolute()
        and command.count("-C") == 1
        and command[command.index("-C") + 1 : command.index("-C") + 2] == ["."]
        and str(anchor.path) not in command,
        f"{invocation} Ninja command is not descriptor-relative",
    )
    require_build_directory_binding(
        anchor,
        context=f"{invocation} pre-execution build directory",
    )
    require_ninja_lock_absent(
        anchor,
        invocation=invocation,
        phase="before",
        observations=observations,
    )
    try:
        completed = _execute_anchored_ninja_process(
            command,
            anchor=anchor,
            timeout=timeout,
        )
    finally:
        require_build_directory_binding(
            anchor,
            context=f"{invocation} post-execution build directory",
        )
        require_ninja_lock_absent(
            anchor,
            invocation=invocation,
            phase="after",
            observations=observations,
        )
    if check and completed.returncode != 0:
        raise CausalFailure(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr}"
        )
    return completed


def stable_regular_file_descriptor(
    descriptor: int,
    *,
    context: str,
    maximum_bytes: int | None,
    capture_bytes: bool,
) -> tuple[dict[str, Any], bytes | None]:
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode),
            f"{context} is not a regular file",
        )
        if maximum_bytes is not None:
            require(
                before.st_size <= maximum_bytes,
                f"{context} exceeds the {maximum_bytes}-byte limit",
            )
        digest = hashlib.sha256()
        captured = bytearray() if capture_bytes else None
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if maximum_bytes is not None:
                require(
                    byte_count <= maximum_bytes,
                    f"{context} exceeds the {maximum_bytes}-byte limit",
                )
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise CausalFailure(f"{context} could not be read safely: {error}") from error
    require(
        stat_identity(before) == stat_identity(after)
        and byte_count == before.st_size,
        f"{context} changed while being read",
    )
    return (
        {
            **stat_manifest_fields(before),
            "sha256": digest.hexdigest(),
        },
        bytes(captured) if captured is not None else None,
    )


def stable_regular_file(
    path: Path,
    *,
    context: str,
    maximum_bytes: int | None = None,
    capture_bytes: bool,
) -> tuple[dict[str, Any], bytes | None]:
    require(
        path.is_absolute() and os.path.abspath(path) == str(path),
        f"{context} path is not canonical absolute",
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    require(
        nofollow != 0
        and directory_flag != 0
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd,
        f"{context} requires descriptor-relative filesystem APIs",
    )
    try:
        lexical_before = path.lstat()
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise CausalFailure(f"{context} could not be opened safely: {error}") from error
    require(
        stat.S_ISREG(lexical_before.st_mode)
        and resolved_path.is_absolute()
        and len(resolved_path.parts) >= 2
        and len(resolved_path.parts) <= BUILD_TREE_MAX_DEPTH + 2,
        f"{context} is not a physical regular file",
    )

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | close_on_exec | directory_flag | nofollow
    file_flags = os.O_RDONLY | close_on_exec | nofollow
    descriptors: list[int] = []
    directory_bindings: list[
        tuple[int, str, int, tuple[int, int, int]]
    ] = []
    try:
        root_descriptor = os.open(Path(resolved_path.anchor), directory_flags)
        descriptors.append(root_descriptor)
        current_descriptor = root_descriptor
        current_path = Path(resolved_path.anchor)
        for component in resolved_path.parent.parts[1:]:
            try:
                child_before = os.stat(
                    component,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_descriptor,
                )
                descriptors.append(child_descriptor)
                child_opened = os.fstat(child_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} ancestor could not be opened safely: "
                    f"{current_path / component}: {error}"
                ) from error
            binding_identity = (
                child_before.st_dev,
                child_before.st_ino,
                child_before.st_mode,
            )
            require(
                stat.S_ISDIR(child_before.st_mode)
                and binding_identity
                == (
                    child_opened.st_dev,
                    child_opened.st_ino,
                    child_opened.st_mode,
                ),
                f"{context} ancestor changed while being opened: "
                f"{current_path / component}",
            )
            directory_bindings.append(
                (
                    current_descriptor,
                    component,
                    child_descriptor,
                    binding_identity,
                )
            )
            current_descriptor = child_descriptor
            current_path /= component

        terminal_name = resolved_path.name
        try:
            terminal_before = os.stat(
                terminal_name,
                dir_fd=current_descriptor,
                follow_symlinks=False,
            )
            file_descriptor = os.open(
                terminal_name,
                file_flags,
                dir_fd=current_descriptor,
            )
            descriptors.append(file_descriptor)
            terminal_opened = os.fstat(file_descriptor)
        except OSError as error:
            raise CausalFailure(
                f"{context} could not be opened safely: {error}"
            ) from error
        require(
            stat.S_ISREG(terminal_before.st_mode)
            and stat_identity(terminal_before)
            == stat_identity(terminal_opened)
            == stat_identity(lexical_before),
            f"{context} changed while being opened",
        )
        record, content = stable_regular_file_descriptor(
            file_descriptor,
            context=context,
            maximum_bytes=maximum_bytes,
            capture_bytes=capture_bytes,
        )
        try:
            terminal_after = os.stat(
                terminal_name,
                dir_fd=current_descriptor,
                follow_symlinks=False,
            )
            lexical_after = path.lstat()
            resolved_after = path.resolve(strict=True)
        except OSError as error:
            raise CausalFailure(
                f"{context} changed while being read: {error}"
            ) from error
        record_identity = (
            record["device"],
            record["inode"],
            record["mode"],
            record["links"],
            record["bytes"],
            record["mtime_ns"],
            record["ctime_ns"],
        )
        require(
            record_identity
            == stat_identity(os.fstat(file_descriptor))
            == stat_identity(terminal_after)
            == stat_identity(lexical_after)
            and resolved_path == resolved_after,
            f"{context} changed while being read",
        )
        for (
            parent_descriptor,
            component,
            child_descriptor,
            binding_identity,
        ) in directory_bindings:
            try:
                child_at_path = os.stat(
                    component,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                child_opened = os.fstat(child_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} ancestor changed while being read: {error}"
                ) from error
            require(
                binding_identity
                == (
                    child_at_path.st_dev,
                    child_at_path.st_ino,
                    child_at_path.st_mode,
                )
                == (
                    child_opened.st_dev,
                    child_opened.st_ino,
                    child_opened.st_mode,
                ),
                f"{context} ancestor changed while being read",
            )
        record["resolved_path"] = str(resolved_path)
        return record, content
    except OSError as error:
        raise CausalFailure(f"{context} could not be opened safely: {error}") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def stable_regular_file_bytes(
    path: Path,
    *,
    context: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    identity, content = stable_regular_file(
        path,
        context=context,
        maximum_bytes=maximum_bytes,
        capture_bytes=True,
    )
    require(content is not None, f"{context} content capture failed")
    return identity, content


def write_stable_file_snapshot(
    source: Path,
    destination: Path,
    *,
    context: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    identity, content = stable_regular_file_bytes(
        source,
        context=context,
        maximum_bytes=maximum_bytes,
    )
    try:
        with destination.open("xb") as captured:
            captured.write(content)
            captured.flush()
            os.fsync(captured.fileno())
    except OSError as error:
        raise CausalFailure(f"{context} snapshot could not be written: {error}") from error
    require(
        d0.sha256(destination) == identity["sha256"]
        and destination.stat().st_size == identity["bytes"],
        f"{context} snapshot differs from its source",
    )
    return identity, content


def require_target_snapshot_binding(
    records: list[dict[str, Any]],
    path: Path,
    identity: dict[str, Any],
    *,
    context: str,
) -> None:
    matches = [record for record in records if record["path"] == str(path)]
    require(len(matches) == 1, f"{context} target identity is missing")
    record = matches[0]
    require(
        record["resolved_path"] == identity["resolved_path"]
        and record["sha256"] == identity["sha256"]
        and record["bytes"] == identity["bytes"]
        and record["mtime_ns"] == identity["mtime_ns"],
        f"{context} snapshot differs from the target identity",
    )


def require_tree_snapshot_binding(
    records: list[dict[str, Any]],
    relative_path: str,
    identity: dict[str, Any],
    *,
    context: str,
) -> None:
    matches = [record for record in records if record["path"] == relative_path]
    require(len(matches) == 1, f"{context} tree identity is missing")
    record = matches[0]
    require(
        all(record[key] == identity[key] for key in STAT_MANIFEST_KEYS)
        and record.get("sha256") == identity["sha256"],
        f"{context} snapshot differs from the tree identity",
    )


def tree_without_mutable_paths(
    records: list[dict[str, Any]],
    mutable_relative_paths: set[str],
) -> list[dict[str, Any]]:
    excluded = set(mutable_relative_paths)
    for value in mutable_relative_paths:
        parent = Path(value).parent
        while str(parent) not in ("", "."):
            excluded.add(parent.as_posix())
            parent = parent.parent
        excluded.add(".")
    return [record for record in records if record["path"] not in excluded]


def require_tree_root_binding(
    records: list[dict[str, Any]],
    anchor: BuildDirectoryAnchor,
    *,
    context: str,
) -> None:
    roots = [record for record in records if record["path"] == "."]
    require(
        len(roots) == 1
        and roots[0]["device"] == anchor.device
        and roots[0]["inode"] == anchor.inode,
        f"{context} is not bound to the opened build directory",
    )


def _build_tree_stat_manifest_once(
    build_directory: Path,
    *,
    excluded_relative_paths: set[str],
) -> list[dict[str, Any]]:
    require(
        build_directory.is_absolute()
        and build_directory.resolve(strict=True) == build_directory,
        f"Build tree root is not canonical: {build_directory}",
    )
    require(
        all(
            path not in ("", ".")
            and not Path(path).is_absolute()
            and all(part not in ("", ".", "..") for part in Path(path).parts)
            for path in excluded_relative_paths
        ),
        "Build tree exclusions are not canonical relative paths",
    )
    records: list[dict[str, Any]] = []
    excluded_seen: set[str] = set()

    def append_record(record: dict[str, Any]) -> None:
        require(
            len(records) < BUILD_TREE_MAX_ENTRIES,
            f"Build tree exceeds {BUILD_TREE_MAX_ENTRIES} entries",
        )
        records.append(record)

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    require(
        getattr(os, "O_DIRECTORY", 0) != 0
        and getattr(os, "O_NOFOLLOW", 0) != 0,
        "Build tree traversal requires O_DIRECTORY and O_NOFOLLOW",
    )
    require(
        BUILD_TREE_DESCRIPTOR_APIS_SUPPORTED,
        "Build tree traversal requires descriptor-relative filesystem APIs",
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    def stat_child(
        directory_descriptor: int,
        name: str,
        *,
        context: str,
    ) -> os.stat_result:
        try:
            return os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise CausalFailure(f"{context}: {error}") from error

    def visit(
        directory_descriptor: int,
        relative: str,
        depth: int,
    ) -> None:
        require(
            depth <= BUILD_TREE_MAX_DEPTH,
            f"Build tree exceeds depth {BUILD_TREE_MAX_DEPTH}: {relative}",
        )
        try:
            before = os.fstat(directory_descriptor)
        except OSError as error:
            raise CausalFailure(
                f"Build tree directory could not be inspected: {relative}: {error}"
            ) from error
        require(
            stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
            f"Build tree directory is not physical: {relative}",
        )
        append_record(
            {
                "path": relative,
                "type": "directory",
                **stat_manifest_fields(before),
            }
        )
        try:
            with os.scandir(directory_descriptor) as iterator:
                names = sorted(
                    (entry.name for entry in iterator),
                    key=lambda name: name.encode("utf-8"),
                )
        except OSError as error:
            raise CausalFailure(
                f"Build tree directory could not be traversed: {relative}: {error}"
            ) from error
        for name in names:
            require(
                name not in ("", ".", "..")
                and "/" not in name
                and "\0" not in name,
                f"Build tree contains an invalid entry name: {name!r}",
            )
            child_relative = (
                name if relative == "." else f"{relative}/{name}"
            )
            metadata = stat_child(
                directory_descriptor,
                name,
                context=(
                    f"Build tree entry could not be inspected: {child_relative}"
                ),
            )
            if child_relative in excluded_relative_paths:
                require(
                    stat.S_ISREG(metadata.st_mode),
                    f"Excluded build path is not a regular file: {child_relative}",
                )
                excluded_seen.add(child_relative)
                continue
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(name, dir_fd=directory_descriptor)
                except OSError as error:
                    raise CausalFailure(
                        "Build tree symlink changed during traversal: "
                        f"{child_relative}: {error}"
                    ) from error
                metadata_after = stat_child(
                    directory_descriptor,
                    name,
                    context=(
                        "Build tree symlink changed during traversal: "
                        f"{child_relative}"
                    ),
                )
                require(
                    stat_identity(metadata) == stat_identity(metadata_after),
                    f"Build tree symlink changed during traversal: {child_relative}",
                )
                append_record(
                    {
                        "path": child_relative,
                        "type": "symlink",
                        "target": target,
                        **stat_manifest_fields(metadata),
                    }
                )
            elif stat.S_ISDIR(metadata.st_mode):
                try:
                    child_descriptor = os.open(
                        name,
                        directory_flags,
                        dir_fd=directory_descriptor,
                    )
                except OSError as error:
                    raise CausalFailure(
                        "Build tree directory could not be opened safely: "
                        f"{child_relative}: {error}"
                    ) from error
                try:
                    opened = os.fstat(child_descriptor)
                    require(
                        stat_identity(metadata) == stat_identity(opened),
                        "Build tree directory changed before traversal: "
                        f"{child_relative}",
                    )
                    visit(child_descriptor, child_relative, depth + 1)
                    opened_after = os.fstat(child_descriptor)
                    parent_after = stat_child(
                        directory_descriptor,
                        name,
                        context=(
                            "Build tree directory changed during traversal: "
                            f"{child_relative}"
                        ),
                    )
                    require(
                        stat_identity(metadata)
                        == stat_identity(opened_after)
                        == stat_identity(parent_after),
                        "Build tree directory changed during traversal: "
                        f"{child_relative}",
                    )
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(metadata.st_mode):
                try:
                    file_descriptor = os.open(
                        name,
                        file_flags,
                        dir_fd=directory_descriptor,
                    )
                except OSError as error:
                    raise CausalFailure(
                        "Build tree file could not be opened safely: "
                        f"{child_relative}: {error}"
                    ) from error
                try:
                    identity, _ = stable_regular_file_descriptor(
                        file_descriptor,
                        context=f"Build tree file {child_relative}",
                        maximum_bytes=None,
                        capture_bytes=False,
                    )
                    parent_after = stat_child(
                        directory_descriptor,
                        name,
                        context=(
                            "Build tree file changed during hashing: "
                            f"{child_relative}"
                        ),
                    )
                    require(
                        stat_identity(metadata)
                        == (
                            identity["device"],
                            identity["inode"],
                            identity["mode"],
                            identity["links"],
                            identity["bytes"],
                            identity["mtime_ns"],
                            identity["ctime_ns"],
                        )
                        == stat_identity(parent_after),
                        f"Build tree file changed during hashing: {child_relative}",
                    )
                finally:
                    os.close(file_descriptor)
                append_record(
                    {
                        "path": child_relative,
                        "type": "file",
                        **{
                            key: identity[key]
                            for key in (
                                "device",
                                "inode",
                                "mode",
                                "links",
                                "bytes",
                                "mtime_ns",
                                "ctime_ns",
                                "sha256",
                            )
                        },
                    }
                )
            else:
                raise CausalFailure(
                    f"Build tree contains a non-regular entry: {child_relative}"
                )
        try:
            after = os.fstat(directory_descriptor)
        except OSError as error:
            raise CausalFailure(
                f"Build tree directory changed during traversal: {relative}: {error}"
            ) from error
        require(
            stat_identity(before) == stat_identity(after),
            f"Build tree directory changed during traversal: {relative}",
        )

    try:
        root_descriptor = os.open(build_directory, directory_flags)
    except OSError as error:
        raise CausalFailure(
            f"Build tree root could not be opened safely: {error}"
        ) from error
    try:
        root_before = os.fstat(root_descriptor)
        root_bound_path = descriptor_path(
            root_descriptor,
            context="Build tree root",
        )
        require(
            root_bound_path == build_directory
            and stat_identity(root_before)
            == stat_identity(build_directory.lstat()),
            "Build tree root changed before traversal",
        )
        visit(root_descriptor, ".", 0)
        root_after = os.fstat(root_descriptor)
        require(
            stat_identity(root_before)
            == stat_identity(root_after)
            == stat_identity(build_directory.lstat())
            and descriptor_path(
                root_descriptor,
                context="Build tree root",
            )
            == build_directory,
            "Build tree root changed during traversal",
        )
    finally:
        os.close(root_descriptor)
    require(
        excluded_seen == excluded_relative_paths,
        "Build tree exclusions were not all observed: "
        f"missing={sorted(excluded_relative_paths - excluded_seen)}",
    )
    return sorted(
        records,
        key=lambda record: (
            record["path"] != ".",
            str(record["path"]).encode("utf-8"),
        ),
    )


def validate_build_tree_inode_consistency(
    records: list[dict[str, Any]],
) -> None:
    by_inode: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        by_inode.setdefault(
            (record["device"], record["inode"]),
            [],
        ).append(record)
    for inode, inode_records in by_inode.items():
        first = inode_records[0]
        identity_keys = ("type", *STAT_MANIFEST_KEYS)
        if first["type"] == "file":
            identity_keys = (*identity_keys, "sha256")
        elif first["type"] == "symlink":
            identity_keys = (*identity_keys, "target")
        require(
            all(
                all(record[key] == first[key] for key in identity_keys)
                for record in inode_records[1:]
            ),
            f"Build tree inode {inode} has conflicting records",
        )
        require(
            first["links"] >= len(inode_records),
            f"Build tree inode {inode} has an impossible hard-link count",
        )
        require(
            first["type"] != "directory" or len(inode_records) == 1,
            f"Build tree repeats directory inode {inode}",
        )


def build_tree_stat_manifest(
    build_directory: Path,
    *,
    excluded_relative_paths: set[str],
) -> list[dict[str, Any]]:
    first = _build_tree_stat_manifest_once(
        build_directory,
        excluded_relative_paths=excluded_relative_paths,
    )
    second = _build_tree_stat_manifest_once(
        build_directory,
        excluded_relative_paths=excluded_relative_paths,
    )
    validate_build_tree_inode_consistency(first)
    validate_build_tree_inode_consistency(second)
    require(
        first == second,
        "Build tree changed between complete snapshots",
    )
    return second


def _target_file_identities_once(paths: set[Path]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda value: str(value).encode("utf-8")):
        require(
            path.is_absolute() and os.path.abspath(path) == str(path),
            f"Idempotence input path is not canonical absolute: {path}",
        )
        identity, _ = stable_regular_file(
            path,
            context=f"Idempotence input {path}",
            capture_bytes=False,
        )
        identities.append(
            {
                "path": str(path),
                "resolved_path": identity["resolved_path"],
                "sha256": identity["sha256"],
                **{
                    key: identity[key]
                    for key in STAT_MANIFEST_KEYS
                },
            }
        )
    return identities


def target_file_identities(paths: set[Path]) -> list[dict[str, Any]]:
    first = _target_file_identities_once(paths)
    second = _target_file_identities_once(paths)
    require(
        first == second,
        "Idempotence inputs changed between complete snapshots",
    )
    return second


def rebuild_metadata_paths(
    *,
    build_directory: Path,
    cache_path: Path,
    compile_path: Path,
) -> set[Path]:
    return {
        cache_path,
        compile_path,
        build_directory / "build.ninja",
        build_directory / "CMakeFiles/rules.ninja",
        build_directory / "CMakeFiles/VerifyGlobs.cmake",
        build_directory / "CMakeFiles/cmake.verify_globs",
        build_directory / ".ninja_deps",
        build_directory / ".ninja_log",
    }


def capture_preclosure_metadata(
    paths: set[Path],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda value: str(value).encode("utf-8")):
        metadata_before = path.lstat()
        require(
            stat.S_ISREG(metadata_before.st_mode),
            f"Pre-closure build metadata is not a regular file: {path}",
        )
        resolved_before = path.resolve(strict=True)
        blob = d0.capture_content_blob(path, output_dir=output_dir)
        metadata_after = path.lstat()
        require(
            (
                metadata_before.st_dev,
                metadata_before.st_ino,
                metadata_before.st_mode,
                metadata_before.st_nlink,
                metadata_before.st_size,
                metadata_before.st_mtime_ns,
                metadata_before.st_ctime_ns,
            )
            == (
                metadata_after.st_dev,
                metadata_after.st_ino,
                metadata_after.st_mode,
                metadata_after.st_nlink,
                metadata_after.st_size,
                metadata_after.st_mtime_ns,
                metadata_after.st_ctime_ns,
            )
            and resolved_before == path
            and path.resolve(strict=True) == resolved_before
            and d0.sha256(path) == blob["sha256"],
            f"Pre-closure build metadata changed during capture: {path}",
        )
        records.append(
            {
                "path": str(path),
                "resolved_path": str(resolved_before),
                "sha256": blob["sha256"],
                **stat_manifest_fields(metadata_before),
                "captured_path": blob["captured_path"],
            }
        )
    return records


def identity_content(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("path", "resolved_path", "sha256", "bytes")
    }


def require_closure_identity_binding(
    closure: dict[str, Any],
    *,
    derived_before: list[dict[str, Any]],
    immutable_before: list[dict[str, Any]],
    context: str,
) -> None:
    closure_files = {
        str(record["path"]): record
        for record in closure["files"]
    }
    require(
        len(closure_files) == len(closure["files"]),
        f"{context} closure contains duplicate file paths",
    )
    compile_outputs = [
        str(unit["output"])
        for unit in closure["compile_units"]
    ]
    require(
        len(compile_outputs) == len(set(compile_outputs)),
        f"{context} closure contains duplicate compile outputs",
    )
    for path in compile_outputs:
        record = closure_files.get(path)
        require(
            record is not None and "compile-output" in record["roles"],
            f"{context} compile output lacks an authenticated closure file: "
            f"{path}",
        )

    observed = {
        str(record["path"]): record
        for record in (*derived_before, *immutable_before)
    }
    require(
        len(observed) == len(derived_before) + len(immutable_before),
        f"{context} rebuild identity sets overlap",
    )
    for record in closure_files.values():
        path = str(record["path"])
        current = observed.get(path)
        require(
            current is not None,
            f"{context} closure file is absent before clean: {path}",
        )
        require(
            identity_content(current)
            == {
                key: record[key]
                for key in ("path", "resolved_path", "sha256", "bytes")
            },
            f"{context} closure file identity changed before clean: {path}",
        )


def rebuild_output_contract(
    closure: dict[str, Any],
    *,
    build_directory: Path,
    expected_output: Path,
) -> tuple[set[Path], set[Path], dict[Path, Path]]:
    selected_target_graph = closure["selected_target_graph"]
    products = closure.get("products")
    require(
        isinstance(products, list)
        and [product.get("name") for product in products]
        == [name for name, _ in BUILD_PRODUCT_TARGETS],
        "Rebuild products differ from the ordered campaign products",
    )
    selected_outputs = [
        Path(str(product["expected_output"]))
        for product in products
    ]
    require(
        selected_outputs
        and selected_outputs[0] == expected_output
        and selected_target_graph["selected_outputs"]
        == [str(path) for path in selected_outputs],
        "Selected target graph outputs differ from the ordered build products",
    )
    compile_outputs = {
        Path(str(unit["output"]))
        for unit in closure["compile_units"]
    }
    module_producers = {
        Path(str(path)): Path(str(unit["output"]))
        for unit in closure["compile_units"]
        for path in unit["module_outputs"]
    }
    require(
        len(module_producers)
        == sum(len(unit["module_outputs"]) for unit in closure["compile_units"]),
        "Module outputs have duplicate producers",
    )
    module_outputs = set(module_producers)
    response_outputs: set[Path] = set()
    for record in closure["files"]:
        if "response-file" not in record["roles"]:
            continue
        path = Path(str(record["path"]))
        try:
            relative_path = path.relative_to(build_directory)
        except ValueError:
            continue
        require(
            relative_path.parts,
            f"Build-local response file names the build directory: {path}",
        )
        response_outputs.add(path)

    freshness_outputs = {
        Path(str(path))
        for path in selected_target_graph["material_outputs"]
    }
    freshness_outputs.update(
        Path(str(path))
        for path in selected_target_graph["clean_plan"]["outputs"]
    )
    freshness_outputs.update(
        Path(str(path))
        for path in selected_target_graph["derived_link_outputs"]
    )
    freshness_outputs.update(compile_outputs)
    freshness_outputs.update(module_outputs)
    freshness_outputs.update(response_outputs)
    freshness_outputs.update(selected_outputs)

    exact_outputs = set(compile_outputs)
    exact_outputs.update(selected_outputs)
    require(
        exact_outputs <= freshness_outputs,
        "Exact rebuild outputs are not covered by the freshness set",
    )
    return freshness_outputs, exact_outputs, module_producers


def rebuild_content_deltas(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    before_by_path = {str(record["path"]): record for record in before}
    after_by_path = {str(record["path"]): record for record in after}
    require(
        len(before_by_path) == len(before)
        and len(after_by_path) == len(after)
        and set(before_by_path) == set(after_by_path),
        "Rebuild output identity path sets differ",
    )
    deltas: list[dict[str, Any]] = []
    for path in sorted(before_by_path, key=lambda value: value.encode("utf-8")):
        before_record = before_by_path[path]
        after_record = after_by_path[path]
        require(
            before_record["resolved_path"] == after_record["resolved_path"],
            f"Rebuild output resolved path changed: {path}",
        )
        if (
            before_record["sha256"] == after_record["sha256"]
            and before_record["bytes"] == after_record["bytes"]
        ):
            continue
        deltas.append(
            {
                "path": path,
                "before_sha256": before_record["sha256"],
                "before_bytes": before_record["bytes"],
                "after_sha256": after_record["sha256"],
                "after_bytes": after_record["bytes"],
            }
        )
    return deltas


def exact_link_action_count(
    text: str,
    *,
    expected_arguments: list[str],
    build_directory: Path,
    response_references: set[tuple[Path, Path]],
    context: str,
) -> int:
    require(expected_arguments, f"{context} expected link arguments are empty")
    compiler = expected_arguments[0]
    shell_boundaries = {"&&", "||", ";", ">", ">>"}
    matches = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        progress = re.fullmatch(r"\[\d+/\d+\] (.+)", line)
        if progress is None:
            continue
        try:
            tokens = shlex.split(progress.group(1))
        except ValueError as error:
            raise CausalFailure(
                f"{context} line {line_number} cannot be tokenized: {error}"
            ) from error
        for compiler_index, token in enumerate(tokens):
            if token != compiler:
                continue
            end = len(tokens)
            for index in range(compiler_index + 1, len(tokens)):
                if tokens[index] in shell_boundaries:
                    end = index
                    break
            arguments = d0.expand_response_arguments(
                tokens[compiler_index:end],
                working_directory=build_directory,
                label=f"{context} invocation on line {line_number}",
                response_references=response_references,
            )
            if arguments == expected_arguments:
                matches += 1
    return matches


def validate_idempotence_transcript(
    completed: subprocess.CompletedProcess[str],
    *,
    build_directory: Path,
    cmake_path: str,
) -> None:
    stdout_lines = completed.stdout.splitlines()
    stderr_lines = completed.stderr.splitlines()
    require(
        stdout_lines
        and stdout_lines[0]
        == "ninja: Entering directory `.'",
        "Ninja idempotence transcript has an unexpected directory header",
    )
    require(
        stdout_lines[-1:] == ["ninja: no work to do."],
        "Ninja idempotence transcript does not end with no work",
    )
    progress_prefix = f"{NINJA_IDEMPOTENCE_PROGRESS} "
    progress = [
        line.removeprefix(progress_prefix)
        for line in stdout_lines[1:-1]
        if line.startswith(progress_prefix)
    ]
    require(
        len(stdout_lines) == 3 and len(progress) == 1,
        "Ninja idempotence invocation executed an unexpected command",
    )
    verify_globs = build_directory / "CMakeFiles/VerifyGlobs.cmake"
    require(
        shlex.split(progress[0]) == [cmake_path, "-P", str(verify_globs)],
        "Ninja idempotence invocation did not execute only VerifyGlobs.cmake",
    )
    require(
        stderr_lines
        == [
            "ninja explain: "
            f"{build_directory}/CMakeFiles/VerifyGlobs.cmake_force is dirty"
        ],
        "Ninja idempotence explanation differs from the forced glob-check edge",
    )


def normalized_ninja_dependency_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        **stored_ninja_dependency_record(record),
        "status": record["status"],
    }


def stored_ninja_dependency_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    dependencies = list(record["dependencies"])
    require(
        record["dependency_count"] == len(dependencies),
        f"Ninja dependency count differs for {record['output']}",
    )
    return {
        "output": record["output"],
        "deps_mtime": record["deps_mtime"],
        "dependencies": sorted(
            dependencies,
            key=lambda value: value.encode("utf-8"),
        ),
    }


def ninja_dependency_input_identity(
    record: dict[str, Any],
) -> dict[str, Any]:
    stored = stored_ninja_dependency_record(record)
    return {
        "output": stored["output"],
        "dependencies": stored["dependencies"],
    }


def ninja_dependency_output_spellings(
    text: str,
    *,
    build_directory: Path,
    records: dict[str, dict[str, Any]],
    label: str,
) -> dict[str, str]:
    lines = text.splitlines()
    result: dict[str, str] = {}
    index = 0
    while index < len(lines):
        if lines[index] == "":
            index += 1
            continue
        match = d0.NINJA_DEPS_HEADER.fullmatch(lines[index])
        require(
            match is not None,
            f"{label} has an invalid header on line {index + 1}",
        )
        lexical = match.group("output")
        canonical = d0.lexical_absolute_path(
            lexical,
            working_directory=build_directory,
        )
        require(
            canonical not in result,
            f"{label} duplicates output {canonical}",
        )
        result[canonical] = lexical
        index += 1 + int(match.group("count"))
        require(
            index <= len(lines),
            f"{label} truncates dependencies for {canonical}",
        )
    require(
        set(result) == set(records),
        f"{label} output spellings differ from parsed records",
    )
    return result


def parse_ninja_dependency_database(
    content: bytes,
    *,
    build_directory: Path,
    expected_output_spellings: dict[str, str],
    label: str,
) -> dict[str, dict[str, Any]]:
    require(
        len(content) >= 16
        and content[: len(NINJA_DEPS_SIGNATURE)] == NINJA_DEPS_SIGNATURE
        and struct.unpack_from("<I", content, len(NINJA_DEPS_SIGNATURE))[0]
        == NINJA_DEPS_VERSION,
        f"{label} database has an invalid Ninja signature or version",
    )
    require(
        0 < len(expected_output_spellings) <= NINJA_DEPS_MAX_OUTPUTS,
        f"{label} expected output count is outside the parser limit",
    )
    require(
        Path(build_directory).is_absolute()
        and os.path.abspath(build_directory) == str(build_directory),
        f"{label} build directory is not canonical absolute",
    )
    lexical_values: list[str] = []
    for output, lexical in expected_output_spellings.items():
        require(
            isinstance(output, str)
            and isinstance(lexical, str)
            and Path(output).is_absolute()
            and os.path.abspath(output) == output
            and d0.lexical_absolute_path(
                lexical,
                working_directory=build_directory,
            )
            == output,
            f"{label} expected output spelling differs: {output}",
        )
        lexical_values.append(lexical)
    require(
        len(set(lexical_values)) == len(lexical_values),
        f"{label} expected output spellings are not unique",
    )
    nodes: list[str] = []
    node_ids: dict[str, int] = {}
    dependencies: dict[int, tuple[int, tuple[int, ...]]] = {}
    offset = 16
    record_count = 0
    total_dependency_count = 0
    while offset < len(content):
        require(
            len(content) - offset >= 4,
            f"{label} database truncates a record size",
        )
        encoded_size = struct.unpack_from("<I", content, offset)[0]
        offset += 4
        is_dependency = bool(encoded_size & 0x80000000)
        record_size = encoded_size & 0x7FFFFFFF
        record_count += 1
        require(
            record_count <= NINJA_DEPS_MAX_RECORDS
            and 0 < record_size <= NINJA_DEPS_MAX_RECORD_BYTES,
            f"{label} database record {record_count} exceeds the parser limit",
        )
        require(
            record_size <= len(content) - offset,
            f"{label} database truncates record {record_count}",
        )
        payload = content[offset : offset + record_size]
        offset += record_size
        if is_dependency:
            require(
                record_size >= 12 and record_size % 4 == 0,
                f"{label} database dependency record {record_count} is invalid",
            )
            dependency_count = record_size // 4 - 3
            require(
                dependency_count <= NINJA_DEPS_MAX_DEPENDENCIES_PER_OUTPUT,
                f"{label} database dependency record {record_count} "
                "exceeds the per-record dependency limit",
            )
            total_dependency_count += dependency_count
            require(
                total_dependency_count <= NINJA_DEPS_MAX_TOTAL_DEPENDENCIES,
                f"{label} database exceeds the total dependency limit",
            )
            values = struct.unpack(f"<{record_size // 4}I", payload)
            output_id, mtime_low, mtime_high, *dependency_ids = values
            require(
                output_id < len(nodes)
                and all(node_id < len(nodes) for node_id in dependency_ids),
                f"{label} database dependency record {record_count} "
                "references an invalid node",
            )
            mtime = mtime_low | (mtime_high << 32)
            require(
                mtime <= 0x7FFFFFFFFFFFFFFF,
                f"{label} database dependency record {record_count} "
                "has a negative mtime",
            )
            dependencies[output_id] = (mtime, tuple(dependency_ids))
            continue

        require(
            record_size >= 8 and record_size % 4 == 0,
            f"{label} database path record {record_count} is invalid",
        )
        path_and_padding = payload[:-4]
        path_bytes = path_and_padding.rstrip(b"\0")
        padding = len(path_and_padding) - len(path_bytes)
        require(
            path_bytes
            and padding <= 3
            and padding == (4 - len(path_bytes) % 4) % 4
            and b"\0" not in path_bytes,
            f"{label} database path record {record_count} has invalid padding",
        )
        checksum = struct.unpack_from("<I", payload, record_size - 4)[0]
        require(
            checksum == ((~len(nodes)) & 0xFFFFFFFF),
            f"{label} database path record {record_count} has an invalid ID",
        )
        try:
            path = path_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CausalFailure(
                f"{label} database path record {record_count} is not UTF-8"
            ) from error
        require(
            "\n" not in path
            and "\r" not in path
            and path not in node_ids
            and len(nodes) < NINJA_DEPS_MAX_NODES,
            f"{label} database path record {record_count} is invalid",
        )
        node_ids[path] = len(nodes)
        nodes.append(path)

    require(offset == len(content), f"{label} database has trailing bytes")
    result: dict[str, dict[str, Any]] = {}
    for output in sorted(
        expected_output_spellings,
        key=lambda value: value.encode("utf-8"),
    ):
        lexical = expected_output_spellings[output]
        output_id = node_ids.get(lexical)
        require(
            output_id is not None and output_id in dependencies,
            f"{label} database omits expected output {output}",
        )
        mtime, dependency_ids = dependencies[output_id]
        dependency_paths = [
            d0.lexical_absolute_path(
                nodes[node_id],
                working_directory=build_directory,
            )
            for node_id in dependency_ids
        ]
        result[output] = {
            "output": output,
            "dependency_count": len(dependency_paths),
            "deps_mtime": mtime,
            "dependencies": dependency_paths,
        }
    return result


def parse_archived_ninja_dependency_database(
    database_path: Path,
    *,
    build_directory: Path,
    expected_output_spellings: dict[str, str],
    label: str,
) -> dict[str, dict[str, Any]]:
    database_identity, database_content = stable_regular_file_bytes(
        database_path,
        context=f"{label} database",
        maximum_bytes=NINJA_DEPS_DATABASE_MAX_BYTES,
    )
    result = parse_ninja_dependency_database(
        database_content,
        build_directory=build_directory,
        expected_output_spellings=expected_output_spellings,
        label=label,
    )
    database_identity_after, database_content_after = stable_regular_file_bytes(
        database_path,
        context=f"{label} database after parse",
        maximum_bytes=NINJA_DEPS_DATABASE_MAX_BYTES,
    )
    require(
        database_identity_after == database_identity
        and database_content_after == database_content,
        f"{label} archived Ninja database changed during parse",
    )
    return result


def validate_ninja_dependency_archive(
    report: dict[str, dict[str, Any]],
    archive: dict[str, dict[str, Any]],
    *,
    label: str,
) -> None:
    require(
        set(report) == set(archive),
        f"{label} output membership differs from the archived Ninja database",
    )
    for output in sorted(report):
        require(
            stored_ninja_dependency_record(report[output])
            == stored_ninja_dependency_record(archive[output]),
            f"{label} record differs from the archived Ninja database: {output}",
        )


def validate_ninja_dependency_settlement(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    closure: dict[str, Any],
    role: str,
) -> tuple[list[str], list[str]]:
    closure_by_output = {
        str(record["output"]): record
        for record in closure["dependency_graph"]
    }
    closure_outputs = set(closure_by_output)
    require(
        len(closure_outputs) == len(closure["dependency_graph"]),
        f"{role} dependency graph contains duplicate outputs",
    )
    missing_before = sorted(closure_outputs - set(before))
    missing_after = sorted(closure_outputs - set(after))
    require(
        not missing_before and not missing_after,
        f"{role} Ninja dependency settlement omits selected outputs: "
        f"before={missing_before}, after={missing_after}",
    )

    for output in sorted(set(before) & set(after)):
        require(
            normalized_ninja_dependency_record(before[output])
            == normalized_ninja_dependency_record(after[output]),
            f"{role} retained Ninja dependency record changed: {output}",
        )
    for output in sorted(closure_outputs):
        require(
            before[output]["status"] == "VALID"
            and after[output]["status"] == "VALID",
            f"{role} selected Ninja dependency record is not VALID: {output}",
        )
        expected_identity = ninja_dependency_input_identity(
            closure_by_output[output]
        )
        require(
            ninja_dependency_input_identity(before[output])
            == expected_identity
            and ninja_dependency_input_identity(after[output])
            == expected_identity,
            f"{role} selected Ninja dependency inputs differ from the "
            f"captured closure: {output}",
        )

    added = sorted(
        set(after) - set(before),
        key=lambda value: value.encode("utf-8"),
    )
    removed = sorted(
        set(before) - set(after),
        key=lambda value: value.encode("utf-8"),
    )
    require(
        not added and not removed,
        f"{role} Ninja dependency settlement changed output membership: "
        f"added={added}, removed={removed}",
    )
    return added, removed


def validate_ninja_log_append(
    before: bytes,
    after: bytes,
    *,
    build_directory: Path,
    context: str,
) -> None:
    require(
        before.endswith(b"\n") and after.endswith(b"\n"),
        f"{context} Ninja log is not newline-terminated",
    )
    require(
        bool(before.splitlines(keepends=True))
        and bool(after.splitlines(keepends=True))
        and before.splitlines(keepends=True)[0] == NINJA_LOG_HEADER
        and after.splitlines(keepends=True)[0] == NINJA_LOG_HEADER,
        f"{context} Ninja log version is not exactly "
        f"v{NINJA_LOG_VERSION}",
    )
    require(
        after.startswith(before),
        f"{context} Ninja log was rewritten or recompacted; "
        "append-only proof unavailable",
    )
    require(
        len(after) > len(before),
        f"{context} Ninja v{NINJA_LOG_VERSION} log did not append a record",
    )
    try:
        appended_lines = after[len(before) :].decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise CausalFailure(f"{context} Ninja log append is not UTF-8") from error
    fields = appended_lines[0].split("\t") if len(appended_lines) == 1 else []
    require(
        len(fields) == 5
        and all(re.fullmatch(r"(0|[1-9][0-9]*)", value) for value in fields[:3])
        and int(fields[0]) <= int(fields[1])
        and 0 < int(fields[2]) <= 0x7FFFFFFFFFFFFFFF
        and fields[3] == str(build_directory / "CMakeFiles/cmake.verify_globs")
        and re.fullmatch(r"(?:0|[1-9a-f][0-9a-f]{0,15})", fields[4])
        is not None,
        f"{context} Ninja log append differs from the glob-check edge: "
        f"{appended_lines!r}",
    )


def parse_ninja_clean_transcript(
    completed: subprocess.CompletedProcess[str],
    *,
    build_directory: Path,
    requested_target: str,
    additional_requested_targets: list[str] | None = None,
    supplemental_targets: list[str] | None = None,
    context: str,
) -> list[Path]:
    require(completed.returncode == 0, f"{context} command failed")
    require(completed.stderr == "", f"{context} emitted stderr")
    require(
        completed.stdout.endswith("\n") and "\r" not in completed.stdout,
        f"{context} is not canonical newline-terminated text",
    )
    lines = completed.stdout.splitlines()
    require(
        len(lines) >= 3
        and lines[0] == "Cleaning...",
        f"{context} has an unexpected header",
    )
    count_match = re.fullmatch(r"(0|[1-9][0-9]*) files\.", lines[-1])
    require(count_match is not None, f"{context} has an invalid file count")

    require(
        build_directory.is_absolute()
        and os.path.abspath(str(build_directory)) == str(build_directory),
        f"{context} build directory is not canonical absolute",
    )
    paths: list[Path] = []
    seen: set[Path] = set()
    targets: list[str] = []
    for line in lines[1:-1]:
        if line.startswith("Target "):
            targets.append(line.removeprefix("Target "))
            continue
        require(line.startswith("Remove "), f"{context} has an unexpected line")
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
        absolute_path = build_directory / path
        try:
            relative_path = absolute_path.relative_to(build_directory)
        except ValueError as error:
            raise CausalFailure(
                f"{context} removal path escapes the build directory: {path_text}"
            ) from error
        require(
            relative_path.parts,
            f"{context} attempts to remove the build directory",
        )
        require(
            absolute_path not in seen,
            f"{context} repeats removal path: {path_text}",
        )
        seen.add(absolute_path)
        paths.append(absolute_path)

    require(
        targets
        == [
            requested_target,
            *(additional_requested_targets or []),
            *(supplemental_targets or []),
        ],
        f"{context} target list differs",
    )
    expected_count = int(count_match.group(1))
    require(
        len(paths) == expected_count,
        f"{context} removal count differs: transcript reports "
        f"{expected_count}, parsed {len(paths)}",
    )
    return paths


def paths_overlap(left: Path, right: Path) -> bool:
    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


def classify_ninja_clean_paths(
    paths: set[Path],
    *,
    build_directory: Path,
    context: str,
) -> tuple[set[Path], set[Path]]:
    require(
        build_directory.is_dir()
        and not build_directory.is_symlink()
        and build_directory.resolve(strict=True) == build_directory,
        f"{context} build directory is not a physical canonical directory",
    )
    existing: set[Path] = set()
    absent: set[Path] = set()
    for path in paths:
        relative_path = path.relative_to(build_directory)
        ancestor = build_directory
        for part in relative_path.parts[:-1]:
            ancestor /= part
            if not os.path.lexists(ancestor):
                break
            metadata = ancestor.lstat()
            require(
                stat.S_ISDIR(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode),
                f"{context} removal path has a non-directory or symlink "
                f"ancestor: {ancestor}",
            )
        if not os.path.lexists(path):
            absent.add(path)
            continue
        metadata = path.lstat()
        require(
            stat.S_ISREG(metadata.st_mode),
            f"{context} removal path is not a regular file: {path}",
        )
        existing.add(path)
    require(
        existing.isdisjoint(absent) and existing | absent == paths,
        f"{context} filesystem partition is incomplete",
    )
    return existing, absent


def renameatx_np_exclusive(
    source_name: str,
    destination_name: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
) -> None:
    require(
        _RENAMEATX_NP is not None
        and source_name not in ("", ".", "..")
        and destination_name not in ("", ".", "..")
        and "/" not in source_name
        and "/" not in destination_name,
        "Exclusive descriptor-relative rename is unavailable or invalid",
    )
    ctypes.set_errno(0)
    result = _RENAMEATX_NP(
        source_dir_fd,
        os.fsencode(source_name),
        destination_dir_fd,
        os.fsencode(destination_name),
        RENAME_EXCL | RENAME_NOFOLLOW_ANY,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            source_name,
        )


def descriptor_relative_quarantine_outputs(
    anchor: BuildDirectoryAnchor,
    quarantine_anchor: BuildDirectoryAnchor,
    derived_before: list[dict[str, Any]],
    protected_identities: list[dict[str, Any]],
    *,
    quarantine_path_prefix: str,
    context: str,
) -> list[dict[str, Any]]:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    require(
        directory_flag != 0
        and nofollow != 0
        and nonblock != 0
        and DESCRIPTOR_QUARANTINE_APIS_SUPPORTED,
        f"{context} requires descriptor-relative no-follow filesystem APIs",
    )
    expected_paths = [str(record["path"]) for record in derived_before]
    require(
        len(expected_paths) == len(set(expected_paths))
        and expected_paths
        == sorted(expected_paths, key=lambda value: value.encode("utf-8")),
        f"{context} expected outputs are not unique and ordered",
    )
    require_build_directory_binding(anchor, context=f"{context} build directory")
    require_build_directory_binding(
        quarantine_anchor,
        context=f"{context} quarantine directory",
    )
    require(
        quarantine_anchor.device == anchor.device
        and Path(quarantine_path_prefix).name == quarantine_path_prefix
        and quarantine_path_prefix not in ("", ".", ".."),
        f"{context} quarantine is not a same-filesystem evidence directory",
    )

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | close_on_exec | directory_flag | nofollow
    file_flags = os.O_RDONLY | close_on_exec | nofollow | nonblock
    protected_inodes = {
        (record["device"], record["inode"])
        for record in protected_identities
    }
    require(
        all(
            record["resolved_path"] == record["path"]
            and record["device"] == anchor.device
            for record in protected_identities
        ),
        f"{context} protected identity set is not bound to the build root",
    )
    expected_inodes = [
        (record["device"], record["inode"])
        for record in derived_before
    ]
    require(
        len(expected_inodes) == len(set(expected_inodes))
        and not (set(expected_inodes) & protected_inodes)
        and all(
            record["resolved_path"] == record["path"]
            and record["device"] == anchor.device
            and record["links"] == 1
            for record in derived_before
        ),
        f"{context} expected outputs alias each other, protected metadata, "
        "are hard-linked, or use another filesystem",
    )
    removed: list[dict[str, Any]] = []

    def require_ancestor_bindings(
        bindings: list[tuple[int, str, int, tuple[int, int, int]]],
        *,
        relative_path: Path,
        phase: str,
    ) -> None:
        require_build_directory_binding(
            anchor,
            context=f"{context} build directory {phase} {relative_path}",
        )
        for (
            parent_descriptor,
            component,
            child_descriptor,
            binding_identity,
        ) in bindings:
            try:
                child_at_path = os.stat(
                    component,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                child_opened = os.fstat(child_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} output ancestor changed {phase}: "
                    f"{relative_path}: {error}"
                ) from error
            require(
                binding_identity
                == (
                    child_at_path.st_dev,
                    child_at_path.st_ino,
                    child_at_path.st_mode,
                )
                == (
                    child_opened.st_dev,
                    child_opened.st_ino,
                    child_opened.st_mode,
                )
                and child_opened.st_dev == anchor.device,
                f"{context} output ancestor changed {phase}: {relative_path}",
            )

    for output_index, expected in enumerate(derived_before):
        path = Path(str(expected["path"]))
        resolved_path = str(expected["resolved_path"])
        quarantine_name = f"{output_index:08d}.output"
        try:
            os.stat(
                quarantine_name,
                dir_fd=quarantine_anchor.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        except OSError as error:
            raise CausalFailure(
                f"{context} quarantine destination could not be inspected: "
                f"{quarantine_name}: {error}"
            ) from error
        else:
            raise CausalFailure(
                f"{context} quarantine destination already exists: "
                f"{quarantine_name}"
            )
        try:
            relative_path = path.relative_to(anchor.path)
        except ValueError as error:
            raise CausalFailure(
                f"{context} output escapes the opened build directory: {path}"
            ) from error
        require(
            path.is_absolute()
            and os.path.abspath(path) == str(path)
            and resolved_path == str(path)
            and relative_path.parts
            and len(relative_path.parts) <= BUILD_TREE_MAX_DEPTH
            and all(part not in ("", ".", "..") for part in relative_path.parts),
            f"{context} output is not a physical canonical build path: {path}",
        )

        descriptors: list[int] = []
        directory_bindings: list[
            tuple[int, str, int, tuple[int, int, int]]
        ] = []
        try:
            current_descriptor = anchor.descriptor
            current_relative = Path()
            for component in relative_path.parts[:-1]:
                try:
                    child_before = os.stat(
                        component,
                        dir_fd=current_descriptor,
                        follow_symlinks=False,
                    )
                    child_descriptor = os.open(
                        component,
                        directory_flags,
                        dir_fd=current_descriptor,
                    )
                    descriptors.append(child_descriptor)
                    child_opened = os.fstat(child_descriptor)
                except OSError as error:
                    raise CausalFailure(
                        f"{context} output ancestor could not be opened safely: "
                        f"{relative_path}: {error}"
                    ) from error
                binding_identity = (
                    child_before.st_dev,
                    child_before.st_ino,
                    child_before.st_mode,
                )
                require(
                    stat.S_ISDIR(child_before.st_mode)
                    and child_before.st_dev == anchor.device
                    and binding_identity
                    == (
                        child_opened.st_dev,
                        child_opened.st_ino,
                        child_opened.st_mode,
                    ),
                    f"{context} output ancestor changed while being opened: "
                    f"{current_relative / component}",
                )
                directory_bindings.append(
                    (
                        current_descriptor,
                        component,
                        child_descriptor,
                        binding_identity,
                    )
                )
                current_descriptor = child_descriptor
                current_relative /= component

            terminal_name = relative_path.name
            try:
                terminal_before = os.stat(
                    terminal_name,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
                file_descriptor = os.open(
                    terminal_name,
                    file_flags,
                    dir_fd=current_descriptor,
                )
                descriptors.append(file_descriptor)
                terminal_opened = os.fstat(file_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} output could not be opened safely: "
                    f"{relative_path}: {error}"
                ) from error
            require(
                stat.S_ISREG(terminal_before.st_mode)
                and terminal_before.st_dev == anchor.device
                and stat_identity(terminal_before)
                == stat_identity(terminal_opened),
                f"{context} output changed while being opened: {relative_path}",
            )
            identity, _ = stable_regular_file_descriptor(
                file_descriptor,
                context=f"{context} output {relative_path}",
                maximum_bytes=None,
                capture_bytes=False,
            )
            require(
                identity["sha256"] == expected["sha256"]
                and all(
                    identity[key] == expected[key]
                    for key in STAT_MANIFEST_KEYS
                ),
                f"{context} output identity differs from derived-before: "
                f"{relative_path}",
            )
            try:
                terminal_current = os.stat(
                    terminal_name,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
                opened_current = os.fstat(file_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} output changed before quarantine: "
                    f"{relative_path}: {error}"
                ) from error
            require(
                stat_identity(terminal_current)
                == stat_identity(opened_current)
                == stat_identity(terminal_opened),
                f"{context} output changed before quarantine: {relative_path}",
            )
            require(
                (identity["device"], identity["inode"]) not in protected_inodes,
                f"{context} output aliases protected metadata: {relative_path}",
            )
            require_ancestor_bindings(
                directory_bindings,
                relative_path=relative_path,
                phase="before quarantine rename",
            )
            try:
                terminal_current = os.stat(
                    terminal_name,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
                opened_current = os.fstat(file_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} output changed at quarantine boundary: "
                    f"{relative_path}: {error}"
                ) from error
            require(
                stat_identity(terminal_current)
                == stat_identity(opened_current)
                == stat_identity(terminal_opened),
                f"{context} output changed at quarantine boundary: "
                f"{relative_path}",
            )
            try:
                renameatx_np_exclusive(
                    terminal_name,
                    quarantine_name,
                    source_dir_fd=current_descriptor,
                    destination_dir_fd=quarantine_anchor.descriptor,
                )
                quarantine_descriptor = os.open(
                    quarantine_name,
                    file_flags,
                    dir_fd=quarantine_anchor.descriptor,
                )
                descriptors.append(quarantine_descriptor)
                terminal_after = os.fstat(file_descriptor)
                quarantine_opened = os.fstat(quarantine_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} output could not be quarantined safely: "
                    f"{relative_path}: {error}"
                ) from error
            require(
                stat_identity(terminal_after)
                == stat_identity(quarantine_opened)
                and (
                    terminal_after.st_dev,
                    terminal_after.st_ino,
                    terminal_after.st_mode,
                    terminal_after.st_nlink,
                    terminal_after.st_size,
                    terminal_after.st_mtime_ns,
                )
                == (
                    identity["device"],
                    identity["inode"],
                    identity["mode"],
                    identity["links"],
                    identity["bytes"],
                    identity["mtime_ns"],
                ),
                f"{context} quarantined output identity differs: "
                f"{relative_path}",
            )
            quarantined_identity, _ = stable_regular_file_descriptor(
                quarantine_descriptor,
                context=f"{context} quarantined output {relative_path}",
                maximum_bytes=None,
                capture_bytes=False,
            )
            require(
                quarantined_identity["sha256"] == identity["sha256"]
                and all(
                    quarantined_identity[key] == terminal_after_value
                    for key, terminal_after_value in stat_manifest_fields(
                        terminal_after
                    ).items()
                ),
                f"{context} quarantined output content differs: {relative_path}",
            )
            try:
                os.stat(
                    terminal_name,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as error:
                raise CausalFailure(
                    f"{context} output absence could not be checked: "
                    f"{relative_path}: {error}"
                ) from error
            else:
                raise CausalFailure(
                    f"{context} output name remains after quarantine: "
                    f"{relative_path}"
                )
            try:
                quarantined_at_path = os.stat(
                    quarantine_name,
                    dir_fd=quarantine_anchor.descriptor,
                    follow_symlinks=False,
                )
                quarantine_current = os.fstat(quarantine_descriptor)
            except OSError as error:
                raise CausalFailure(
                    f"{context} quarantined output changed: "
                    f"{relative_path}: {error}"
                ) from error
            require(
                stat_identity(quarantined_at_path)
                == stat_identity(quarantine_current)
                == stat_identity(terminal_after),
                f"{context} quarantined output changed: {relative_path}",
            )

            require_ancestor_bindings(
                directory_bindings,
                relative_path=relative_path,
                phase="after quarantine rename",
            )
            require_build_directory_binding(
                quarantine_anchor,
                context=f"{context} quarantine after {relative_path}",
            )
            removed.append(
                {
                    "path": str(path),
                    "relative_path": relative_path.as_posix(),
                    "resolved_path": str(path),
                    "quarantined_path": (
                        f"{quarantine_path_prefix}/{quarantine_name}"
                    ),
                    "before": {
                        "sha256": identity["sha256"],
                        **{
                            key: identity[key]
                            for key in STAT_MANIFEST_KEYS
                        },
                    },
                    "after": {
                        "sha256": quarantined_identity["sha256"],
                        **{
                            key: quarantined_identity[key]
                            for key in STAT_MANIFEST_KEYS
                        },
                    },
                    "name_absent_after": True,
                }
            )
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
    require_build_directory_binding(
        anchor,
        context=f"{context} final build directory",
    )
    require_build_directory_binding(
        quarantine_anchor,
        context=f"{context} final quarantine directory",
    )
    return removed


def _capture_build_idempotence_anchored(
    *,
    role: str,
    output_dir: Path,
    build_directory: Path,
    anchor: BuildDirectoryAnchor,
    requested_target: str,
    build_tool: dict[str, Any],
    cache: dict[str, str],
    closure: dict[str, Any],
    cache_path: Path,
    compile_path: Path,
    expected_output: Path,
    link_arguments: list[str],
    response_references: set[tuple[Path, Path]],
    preclosure_metadata: list[dict[str, Any]],
    preclosure_metadata_path: Path,
) -> dict[str, Any]:
    mutable_relative_paths = {".ninja_log"}
    lock_observations: list[dict[str, Any]] = []
    ninja_log = build_directory / ".ninja_log"
    require(ninja_log.is_file(), f"{role} Ninja log is missing")
    require_build_directory_binding(
        anchor,
        context=f"{role} idempotence build directory",
    )
    closure_products = closure.get("products")
    require(
        isinstance(closure_products, list)
        and [product.get("name") for product in closure_products]
        == [name for name, _ in BUILD_PRODUCT_TARGETS],
        f"{role} idempotence products differ from the ordered campaign products",
    )
    requested_targets = [
        str(product["requested_target"])
        for product in closure_products
    ]
    require(
        requested_targets
        and requested_targets[0] == requested_target
        and len(requested_targets) == len(set(requested_targets)),
        f"{role} ordered rebuild targets differ from the benchmark target",
    )
    closure_link_arguments = [
        [str(argument) for argument in product["link"]["arguments"]]
        for product in closure_products
    ]
    require(
        closure_link_arguments
        and closure_link_arguments[0] == link_arguments,
        f"{role} benchmark link arguments differ from the closure",
    )

    (
        freshness_output_paths,
        exact_output_paths,
        module_producers,
    ) = rebuild_output_contract(
        closure,
        build_directory=build_directory,
        expected_output=expected_output,
    )
    closure_paths = {
        Path(str(record["path"]))
        for record in closure["files"]
    }
    all_metadata_paths = rebuild_metadata_paths(
        build_directory=build_directory,
        cache_path=cache_path,
        compile_path=compile_path,
    )
    protected_paths = all_metadata_paths - {
        build_directory / ".ninja_deps",
        build_directory / ".ninja_log",
    }
    clean_protected_paths = all_metadata_paths | {
        build_directory / ".ninja_log",
        build_directory / ".ninja_lock",
    }

    base_clean_command = [
        build_tool["resolved_path"],
        "-v",
        "-n",
        "-C",
        ".",
        "-t",
        "clean",
        *requested_targets,
    ]
    base_clean = run_anchored_ninja(
        base_clean_command,
        anchor=anchor,
        invocation="base-clean-plan",
        observations=lock_observations,
    )
    base_clean_stdout_path = (
        output_dir / f"{role}-rebuild-base-clean-plan.stdout"
    )
    base_clean_stderr_path = (
        output_dir / f"{role}-rebuild-base-clean-plan.stderr"
    )
    base_clean_stdout_path.write_text(base_clean.stdout, encoding="utf-8")
    base_clean_stderr_path.write_text(base_clean.stderr, encoding="utf-8")
    base_clean_path_list = parse_ninja_clean_transcript(
        base_clean,
        build_directory=build_directory,
        requested_target=requested_target,
        additional_requested_targets=requested_targets[1:],
        context=f"{role} base Ninja clean plan",
    )
    base_clean_paths = set(base_clean_path_list)
    frozen_base_clean_path_list = [
        Path(str(path))
        for path in closure["selected_target_graph"]["clean_plan"]["outputs"]
    ]
    frozen_base_clean_paths = set(frozen_base_clean_path_list)
    unexpected_base_clean_paths = sorted(
        str(path)
        for path in base_clean_paths - frozen_base_clean_paths
    )
    require(
        not unexpected_base_clean_paths,
        f"{role} base clean plan contains non-closure outputs: "
        f"{unexpected_base_clean_paths}",
    )
    missing_base_clean_paths = sorted(
        str(path)
        for path in frozen_base_clean_paths - base_clean_paths
    )
    require(
        not missing_base_clean_paths,
        f"{role} base clean plan omits frozen closure outputs: "
        f"{missing_base_clean_paths}",
    )
    require(
        base_clean_path_list == frozen_base_clean_path_list,
        f"{role} base clean plan order differs from the frozen closure",
    )
    supplemental_removal_paths = freshness_output_paths - base_clean_paths
    unsupported_supplemental_paths = sorted(
        str(path)
        for path in supplemental_removal_paths - set(module_producers)
    )
    require(
        not unsupported_supplemental_paths,
        f"{role} clean plan omits non-module graph-derived freshness outputs: "
        f"{unsupported_supplemental_paths}",
    )
    invalid_supplemental_paths: list[str] = []
    for path in supplemental_removal_paths:
        try:
            relative_path = path.relative_to(build_directory)
        except ValueError:
            invalid_supplemental_paths.append(str(path))
            continue
        if (
            not path.is_absolute()
            or os.path.abspath(str(path)) != str(path)
            or not relative_path.parts
        ):
            invalid_supplemental_paths.append(str(path))
    require(
        not invalid_supplemental_paths,
        f"{role} supplemental module outputs escape the build directory: "
        f"{sorted(invalid_supplemental_paths)}",
    )
    uncleaned_module_producers = sorted(
        f"{path} <- {module_producers[path]}"
        for path in supplemental_removal_paths
        if module_producers[path] not in base_clean_paths
    )
    require(
        not uncleaned_module_producers,
        f"{role} supplemental module outputs lack cleaned producers: "
        f"{uncleaned_module_producers}",
    )
    supplemental_targets = sorted(
        path.relative_to(build_directory).as_posix()
        for path in supplemental_removal_paths
    )
    removal_paths = freshness_output_paths
    protected_overlaps = sorted(
        {
            str(removal)
            for removal in removal_paths
            for protected in clean_protected_paths
            if paths_overlap(removal, protected)
        }
    )
    require(
        not protected_overlaps,
        f"{role} clean removal set overlaps protected build metadata: "
        f"{protected_overlaps}",
    )
    require(
        preclosure_metadata_path.is_file()
        and json.loads(preclosure_metadata_path.read_text(encoding="utf-8"))
        == preclosure_metadata,
        f"{role} pre-closure metadata artifact changed",
    )
    require(
        {Path(str(record["path"])) for record in preclosure_metadata}
        == all_metadata_paths,
        f"{role} pre-closure metadata path set differs",
    )
    preclosure_identities = [
        {
            key: record[key]
            for key in (
                "path",
                "resolved_path",
                "sha256",
                *STAT_MANIFEST_KEYS,
            )
        }
        for record in preclosure_metadata
    ]
    current_metadata = target_file_identities(all_metadata_paths)
    require(
        preclosure_identities == current_metadata,
        f"{role} build metadata changed after closure capture",
    )

    derived_paths, absent_before_paths = classify_ninja_clean_paths(
        removal_paths,
        build_directory=build_directory,
        context=f"{role} clean removal set",
    )
    missing_existing_freshness_outputs = sorted(
        str(path)
        for path in freshness_output_paths - derived_paths
    )
    require(
        not missing_existing_freshness_outputs,
        f"{role} graph-derived freshness outputs are missing before clean: "
        f"{missing_existing_freshness_outputs}",
    )
    selected_output_paths = {
        Path(str(path))
        for path in closure["selected_target_graph"]["selected_outputs"]
    }
    require(
        selected_output_paths <= derived_paths,
        f"{role} rebuild graph omits an ordered product output",
    )
    immutable_paths = closure_paths - removal_paths
    derived_before = target_file_identities(derived_paths)
    immutable_before = target_file_identities(immutable_paths)
    protected_before = target_file_identities(protected_paths)
    require(
        all(
            record["resolved_path"] == record["path"]
            for record in (*derived_before, *protected_before)
        ),
        f"{role} derived or protected build path resolves elsewhere",
    )
    derived_before_path = output_dir / f"{role}-rebuild-derived-before.json"
    immutable_before_path = output_dir / f"{role}-rebuild-immutable-before.json"
    protected_before_path = output_dir / f"{role}-rebuild-protected-before.json"
    d0.write_json(derived_before_path, derived_before)
    d0.write_json(immutable_before_path, immutable_before)
    d0.write_json(protected_before_path, protected_before)
    preclosure_by_path = {
        str(record["path"]): record
        for record in preclosure_identities
    }
    require(
        protected_before
        == [
            preclosure_by_path[str(path)]
            for path in sorted(
                protected_paths,
                key=lambda value: str(value).encode("utf-8"),
            )
        ],
        f"{role} protected build metadata changed before clean",
    )
    require_closure_identity_binding(
        closure,
        derived_before=derived_before,
        immutable_before=immutable_before,
        context=role,
    )

    dry_clean_command = [
        build_tool["resolved_path"],
        "-v",
        "-n",
        "-C",
        ".",
        "-t",
        "clean",
        *requested_targets,
        *supplemental_targets,
    ]
    dry_clean = run_anchored_ninja(
        dry_clean_command,
        anchor=anchor,
        invocation="combined-clean-plan",
        observations=lock_observations,
    )
    dry_clean_stdout_path = output_dir / f"{role}-rebuild-clean-plan.stdout"
    dry_clean_stderr_path = output_dir / f"{role}-rebuild-clean-plan.stderr"
    dry_clean_stdout_path.write_text(dry_clean.stdout, encoding="utf-8")
    dry_clean_stderr_path.write_text(dry_clean.stderr, encoding="utf-8")
    planned_clean_path_list = parse_ninja_clean_transcript(
        dry_clean,
        build_directory=build_directory,
        requested_target=requested_target,
        additional_requested_targets=requested_targets[1:],
        supplemental_targets=supplemental_targets,
        context=f"{role} combined Ninja clean plan",
    )
    planned_clean_paths = set(planned_clean_path_list)
    require(
        len(planned_clean_path_list) == len(removal_paths)
        and planned_clean_paths == removal_paths,
        f"{role} combined Ninja clean plan differs from the required "
        "removal set",
    )

    require(
        target_file_identities(all_metadata_paths) == preclosure_identities,
        f"{role} build metadata changed before clean execution",
    )
    require(
        target_file_identities(derived_paths) == derived_before,
        f"{role} rebuild outputs changed before clean execution",
    )
    require(
        target_file_identities(immutable_paths) == immutable_before,
        f"{role} immutable inputs changed before clean execution",
    )
    require(
        target_file_identities(protected_paths) == protected_before,
        f"{role} protected metadata changed before clean execution",
    )

    current_derived_paths, current_absent_paths = classify_ninja_clean_paths(
        removal_paths,
        build_directory=build_directory,
        context=f"{role} clean removal set pre-execution check",
    )
    require(
        current_derived_paths == derived_paths
        and current_absent_paths == absent_before_paths,
        f"{role} clean-removal filesystem state changed before execution",
    )
    quarantine_directory = output_dir / f"{role}-rebuild-quarantine"
    try:
        quarantine_directory.mkdir(mode=0o700)
    except OSError as error:
        raise CausalFailure(
            f"{role} output quarantine could not be created: {error}"
        ) from error
    with open_build_directory_anchor(quarantine_directory) as quarantine_anchor:
        removed_outputs = descriptor_relative_quarantine_outputs(
            anchor=anchor,
            quarantine_anchor=quarantine_anchor,
            derived_before=derived_before,
            protected_identities=preclosure_identities,
            quarantine_path_prefix=quarantine_directory.name,
            context=f"{role} descriptor-relative output quarantine",
        )
    removed_outputs_path = output_dir / f"{role}-rebuild-removed-outputs.json"
    d0.write_json(removed_outputs_path, removed_outputs)
    require(
        [
            {
                key: record[key]
                for key in (
                    "path",
                    "resolved_path",
                    "before",
                )
            }
            for record in removed_outputs
        ]
        == [
            {
                "path": record["path"],
                "resolved_path": record["resolved_path"],
                "before": {
                    "sha256": record["sha256"],
                    **{
                        key: record[key]
                        for key in STAT_MANIFEST_KEYS
                    },
                },
            }
            for record in derived_before
        ],
        f"{role} descriptor-relative removal evidence differs from "
        "derived-before",
    )
    remaining_removals, absent_after_removal = classify_ninja_clean_paths(
        removal_paths,
        build_directory=build_directory,
        context=f"{role} clean removal set post-execution check",
    )
    require(
        not remaining_removals and absent_after_removal == removal_paths,
        f"{role} clean removal set remains present: "
        f"{sorted(str(path) for path in remaining_removals)}",
    )
    immutable_after_clean = target_file_identities(immutable_paths)
    protected_after_clean = target_file_identities(protected_paths)
    require(
        immutable_after_clean == immutable_before,
        f"{role} clean changed immutable build inputs",
    )
    require(
        protected_after_clean == protected_before,
        f"{role} clean changed protected build metadata",
    )
    immutable_after_clean_path = (
        output_dir / f"{role}-rebuild-immutable-after-clean.json"
    )
    protected_after_clean_path = (
        output_dir / f"{role}-rebuild-protected-after-clean.json"
    )
    d0.write_json(immutable_after_clean_path, immutable_after_clean)
    d0.write_json(protected_after_clean_path, protected_after_clean)

    rebuild_command = [
        build_tool["resolved_path"],
        "-C",
        ".",
        "-v",
        "-d",
        "explain",
        *requested_targets,
    ]
    rebuild = run_anchored_ninja(
        rebuild_command,
        anchor=anchor,
        invocation="rebuild",
        observations=lock_observations,
        timeout=BUILD_REBUILD_TIMEOUT_SECONDS,
    )
    require(
        "ninja: no work to do." not in rebuild.stdout,
        f"{role} clean rebuild performed no material work",
    )
    rebuild_stdout_path = output_dir / f"{role}-rebuild.stdout"
    rebuild_stderr_path = output_dir / f"{role}-rebuild.stderr"
    rebuild_stdout_path.write_text(rebuild.stdout, encoding="utf-8")
    rebuild_stderr_path.write_text(rebuild.stderr, encoding="utf-8")
    derived_after = target_file_identities(derived_paths)
    derived_after_path = output_dir / f"{role}-rebuild-derived-after.json"
    d0.write_json(derived_after_path, derived_after)
    resurrected_absent = sorted(
        str(path)
        for path in absent_before_paths
        if os.path.lexists(path)
    )
    require(
        not resurrected_absent,
        f"{role} clean rebuild created outputs absent before clean: "
        f"{resurrected_absent}",
    )
    immutable_after = target_file_identities(immutable_paths)
    protected_after = target_file_identities(protected_paths)
    immutable_after_path = output_dir / f"{role}-rebuild-immutable-after.json"
    protected_after_path = output_dir / f"{role}-rebuild-protected-after.json"
    d0.write_json(immutable_after_path, immutable_after)
    d0.write_json(protected_after_path, protected_after)
    content_deltas = rebuild_content_deltas(derived_before, derived_after)
    content_deltas_path = output_dir / f"{role}-rebuild-content-deltas.json"
    d0.write_json(content_deltas_path, content_deltas)
    changed_exact_outputs = sorted(
        delta["path"]
        for delta in content_deltas
        if Path(str(delta["path"])) in exact_output_paths
    )
    require(
        not changed_exact_outputs,
        f"{role} clean rebuild changed exact compile/link outputs: "
        f"{changed_exact_outputs}",
    )
    require(
        immutable_after == immutable_before,
        f"{role} clean rebuild changed immutable build inputs",
    )
    require(
        protected_after == protected_before,
        f"{role} clean rebuild changed protected build metadata",
    )
    rebuilt_response_references: set[tuple[Path, Path]] = set()
    for product_index, product_link_arguments in enumerate(
        closure_link_arguments
    ):
        require(
            exact_link_action_count(
                rebuild.stdout,
                expected_arguments=product_link_arguments,
                build_directory=build_directory,
                response_references=rebuilt_response_references,
                context=f"{role} clean rebuild product {product_index}",
            )
            == 1,
            f"{role} clean rebuild does not contain exactly one selected link "
            f"action for product {product_index}",
        )
    require(
        rebuilt_response_references <= response_references,
        f"{role} clean rebuild references unexpected response files",
    )

    target_paths = {
        Path(str(record["path"]))
        for record in closure["files"]
    }
    target_paths.update(freshness_output_paths)
    target_paths.update(
        {
            cache_path,
            compile_path,
            build_directory / "build.ninja",
            build_directory / "CMakeFiles/rules.ninja",
            build_directory / "CMakeFiles/VerifyGlobs.cmake",
            build_directory / "CMakeFiles/cmake.verify_globs",
            build_directory / ".ninja_deps",
            *selected_output_paths,
        }
    )
    ninja_deps = build_directory / ".ninja_deps"
    dependency_report_command = [
        build_tool["resolved_path"],
        "-C",
        ".",
        "-t",
        "deps",
    ]
    settlement_target_before = target_file_identities(target_paths)
    post_rebuild_identities: dict[str, dict[str, Any]] = {}
    for record in (
        *derived_after,
        *immutable_after,
        *protected_after,
    ):
        path = str(record["path"])
        require(
            path not in post_rebuild_identities
            or post_rebuild_identities[path] == record,
            f"{role} post-rebuild identity sets disagree for {path}",
        )
        post_rebuild_identities[path] = record
    settlement_without_deps = {
        str(record["path"]): record
        for record in settlement_target_before
        if record["path"] != str(build_directory / ".ninja_deps")
    }
    require(
        settlement_without_deps == post_rebuild_identities,
        f"{role} settlement baseline differs from post-rebuild identities",
    )
    settlement_tree_before = build_tree_stat_manifest(
        build_directory,
        excluded_relative_paths=mutable_relative_paths,
    )
    require_tree_root_binding(
        settlement_tree_before,
        anchor,
        context=f"{role} pre-settlement build tree",
    )
    require(
        all(record["path"] != ".ninja_lock" for record in settlement_tree_before),
        f"{role} pre-settlement build tree contains .ninja_lock",
    )
    settlement_target_before_path = (
        output_dir / f"{role}-settlement-target-before.json"
    )
    settlement_tree_before_path = (
        output_dir / f"{role}-settlement-tree-before.json"
    )
    settlement_deps_before_path = (
        output_dir / f"{role}-settlement-ninja-deps-before.bin"
    )
    settlement_log_before_path = (
        output_dir / f"{role}-settlement-ninja-log-before.txt"
    )
    d0.write_json(settlement_target_before_path, settlement_target_before)
    d0.write_json(settlement_tree_before_path, settlement_tree_before)
    settlement_deps_before_identity, _ = write_stable_file_snapshot(
        ninja_deps,
        settlement_deps_before_path,
        context=f"{role} pre-settlement Ninja dependencies",
        maximum_bytes=NINJA_DEPS_DATABASE_MAX_BYTES,
    )
    settlement_log_before_identity, settlement_log_before_bytes = (
        write_stable_file_snapshot(
            ninja_log,
            settlement_log_before_path,
            context=f"{role} pre-settlement Ninja log",
            maximum_bytes=NINJA_LOG_MAX_BYTES,
        )
    )
    require_target_snapshot_binding(
        settlement_target_before,
        ninja_deps,
        settlement_deps_before_identity,
        context=f"{role} pre-settlement Ninja dependencies",
    )
    require_tree_snapshot_binding(
        settlement_tree_before,
        ".ninja_deps",
        settlement_deps_before_identity,
        context=f"{role} pre-settlement Ninja dependencies",
    )

    dependency_report_before = run_anchored_ninja(
        dependency_report_command,
        anchor=anchor,
        invocation="dependency-report-before",
        observations=lock_observations,
    )
    require(
        dependency_report_before.returncode == 0
        and dependency_report_before.stderr == "",
        f"{role} pre-settlement Ninja dependency report failed",
    )
    require(
        target_file_identities(target_paths) == settlement_target_before
        and build_tree_stat_manifest(
            build_directory,
            excluded_relative_paths=mutable_relative_paths,
        )
        == settlement_tree_before
        and stable_regular_file_bytes(
            ninja_log,
            context=f"{role} Ninja log after pre-settlement dependency report",
            maximum_bytes=NINJA_LOG_MAX_BYTES,
        )[1]
        == settlement_log_before_bytes,
        f"{role} pre-settlement Ninja dependency report changed the build",
    )
    dependency_report_before_stdout_path = (
        output_dir / f"{role}-settlement-ninja-deps-report-before.txt"
    )
    dependency_report_before_stderr_path = (
        output_dir / f"{role}-settlement-ninja-deps-report-before.stderr"
    )
    dependency_report_before_stdout_path.write_text(
        dependency_report_before.stdout,
        encoding="utf-8",
    )
    dependency_report_before_stderr_path.write_text(
        dependency_report_before.stderr,
        encoding="utf-8",
    )
    dependency_records_before = d0.parse_ninja_deps(
        dependency_report_before.stdout,
        build_directory=build_directory,
        label=f"{role} pre-settlement Ninja dependencies",
    )
    dependency_spellings_before = ninja_dependency_output_spellings(
        dependency_report_before.stdout,
        build_directory=build_directory,
        records=dependency_records_before,
        label=f"{role} pre-settlement Ninja dependencies",
    )
    dependency_archive_before = parse_archived_ninja_dependency_database(
        settlement_deps_before_path,
        build_directory=build_directory,
        expected_output_spellings=dependency_spellings_before,
        label=f"{role} pre-settlement archived Ninja dependencies",
    )
    validate_ninja_dependency_archive(
        dependency_records_before,
        dependency_archive_before,
        label=f"{role} pre-settlement Ninja dependencies",
    )

    settlement_command = [
        build_tool["resolved_path"],
        "-C",
        ".",
        "-v",
        "-d",
        "explain",
        *requested_targets,
    ]
    settlement_completed = run_anchored_ninja(
        settlement_command,
        anchor=anchor,
        invocation="settlement",
        observations=lock_observations,
    )
    validate_idempotence_transcript(
        settlement_completed,
        build_directory=build_directory,
        cmake_path=cache["CMAKE_COMMAND"],
    )
    settlement_stdout_path = output_dir / f"{role}-settlement.stdout"
    settlement_stderr_path = output_dir / f"{role}-settlement.stderr"
    settlement_stdout_path.write_text(
        settlement_completed.stdout,
        encoding="utf-8",
    )
    settlement_stderr_path.write_text(
        settlement_completed.stderr,
        encoding="utf-8",
    )

    settlement_target_after = target_file_identities(target_paths)
    settlement_tree_after = build_tree_stat_manifest(
        build_directory,
        excluded_relative_paths=mutable_relative_paths,
    )
    require_tree_root_binding(
        settlement_tree_after,
        anchor,
        context=f"{role} post-settlement build tree",
    )
    require(
        all(record["path"] != ".ninja_lock" for record in settlement_tree_after),
        f"{role} post-settlement build tree contains .ninja_lock",
    )
    settlement_target_after_path = (
        output_dir / f"{role}-settlement-target-after.json"
    )
    settlement_tree_after_path = (
        output_dir / f"{role}-settlement-tree-after.json"
    )
    settlement_deps_after_path = (
        output_dir / f"{role}-settlement-ninja-deps-after.bin"
    )
    settlement_log_after_path = (
        output_dir / f"{role}-settlement-ninja-log-after.txt"
    )
    d0.write_json(settlement_target_after_path, settlement_target_after)
    d0.write_json(settlement_tree_after_path, settlement_tree_after)
    settlement_deps_after_identity, _ = write_stable_file_snapshot(
        ninja_deps,
        settlement_deps_after_path,
        context=f"{role} post-settlement Ninja dependencies",
        maximum_bytes=NINJA_DEPS_DATABASE_MAX_BYTES,
    )
    settlement_log_after_identity, settlement_log_after_bytes = (
        write_stable_file_snapshot(
            ninja_log,
            settlement_log_after_path,
            context=f"{role} post-settlement Ninja log",
            maximum_bytes=NINJA_LOG_MAX_BYTES,
        )
    )
    require_target_snapshot_binding(
        settlement_target_after,
        ninja_deps,
        settlement_deps_after_identity,
        context=f"{role} post-settlement Ninja dependencies",
    )
    require_tree_snapshot_binding(
        settlement_tree_after,
        ".ninja_deps",
        settlement_deps_after_identity,
        context=f"{role} post-settlement Ninja dependencies",
    )

    dependency_report_after = run_anchored_ninja(
        dependency_report_command,
        anchor=anchor,
        invocation="dependency-report-after",
        observations=lock_observations,
    )
    require(
        dependency_report_after.returncode == 0
        and dependency_report_after.stderr == "",
        f"{role} post-settlement Ninja dependency report failed",
    )
    require(
        target_file_identities(target_paths) == settlement_target_after
        and build_tree_stat_manifest(
            build_directory,
            excluded_relative_paths=mutable_relative_paths,
        )
        == settlement_tree_after
        and stable_regular_file_bytes(
            ninja_log,
            context=f"{role} Ninja log after post-settlement dependency report",
            maximum_bytes=NINJA_LOG_MAX_BYTES,
        )[1]
        == settlement_log_after_bytes,
        f"{role} post-settlement Ninja dependency report changed the build",
    )
    dependency_report_after_stdout_path = (
        output_dir / f"{role}-settlement-ninja-deps-report-after.txt"
    )
    dependency_report_after_stderr_path = (
        output_dir / f"{role}-settlement-ninja-deps-report-after.stderr"
    )
    dependency_report_after_stdout_path.write_text(
        dependency_report_after.stdout,
        encoding="utf-8",
    )
    dependency_report_after_stderr_path.write_text(
        dependency_report_after.stderr,
        encoding="utf-8",
    )
    dependency_records_after = d0.parse_ninja_deps(
        dependency_report_after.stdout,
        build_directory=build_directory,
        label=f"{role} post-settlement Ninja dependencies",
    )
    dependency_spellings_after = ninja_dependency_output_spellings(
        dependency_report_after.stdout,
        build_directory=build_directory,
        records=dependency_records_after,
        label=f"{role} post-settlement Ninja dependencies",
    )
    dependency_archive_after = parse_archived_ninja_dependency_database(
        settlement_deps_after_path,
        build_directory=build_directory,
        expected_output_spellings=dependency_spellings_after,
        label=f"{role} post-settlement archived Ninja dependencies",
    )
    validate_ninja_dependency_archive(
        dependency_records_after,
        dependency_archive_after,
        label=f"{role} post-settlement Ninja dependencies",
    )
    added_dependency_outputs, removed_dependency_outputs = (
        validate_ninja_dependency_settlement(
            dependency_records_before,
            dependency_records_after,
            closure=closure,
            role=role,
        )
    )
    for phase, dependency_records, target_records in (
        (
            "before",
            dependency_records_before,
            settlement_target_before,
        ),
        (
            "after",
            dependency_records_after,
            settlement_target_after,
        ),
    ):
        target_by_path = {
            str(record["path"]): record
            for record in target_records
        }
        for dependency_record in closure["dependency_graph"]:
            output = str(dependency_record["output"])
            require(
                output in target_by_path
                and target_by_path[output]["mtime_ns"]
                == dependency_records[output]["deps_mtime"],
                f"{role} {phase} selected dependency status is not bound "
                f"to output mtime: {output}",
            )

    deps_path_text = str(ninja_deps)
    require(
        [
            record
            for record in settlement_target_before
            if record["path"] != deps_path_text
        ]
        == [
            record
            for record in settlement_target_after
            if record["path"] != deps_path_text
        ],
        f"{role} target closure changed outside .ninja_deps during settlement",
    )
    require(
        tree_without_mutable_paths(
            settlement_tree_before,
            {".ninja_deps", ".ninja_log"},
        )
        == tree_without_mutable_paths(
            settlement_tree_after,
            {".ninja_deps", ".ninja_log"},
        ),
        f"{role} build tree changed outside Ninja databases during settlement",
    )
    validate_ninja_log_append(
        settlement_log_before_bytes,
        settlement_log_after_bytes,
        build_directory=build_directory,
        context=f"{role} settlement",
    )

    target_before = target_file_identities(target_paths)
    tree_before = build_tree_stat_manifest(
        build_directory,
        excluded_relative_paths=mutable_relative_paths,
    )
    require_tree_root_binding(
        tree_before,
        anchor,
        context=f"{role} pre-idempotence build tree",
    )
    require(
        all(record["path"] != ".ninja_lock" for record in tree_before),
        f"{role} pre-idempotence build tree contains .ninja_lock",
    )
    require(
        target_before == settlement_target_after
        and tree_before == settlement_tree_after,
        f"{role} authoritative build state changed after settlement",
    )
    target_before_path = output_dir / f"{role}-idempotence-target-before.json"
    tree_before_path = output_dir / f"{role}-idempotence-tree-before.json"
    log_before_path = output_dir / f"{role}-idempotence-ninja-log-before.txt"
    d0.write_json(target_before_path, target_before)
    d0.write_json(tree_before_path, tree_before)
    _log_before_identity, log_before_bytes = write_stable_file_snapshot(
        ninja_log,
        log_before_path,
        context=f"{role} authoritative pre-idempotence Ninja log",
        maximum_bytes=NINJA_LOG_MAX_BYTES,
    )
    require(
        log_before_bytes == settlement_log_after_bytes,
        f"{role} authoritative Ninja log changed after settlement",
    )

    command = settlement_command
    completed = run_anchored_ninja(
        command,
        anchor=anchor,
        invocation="idempotence",
        observations=lock_observations,
    )
    validate_idempotence_transcript(
        completed,
        build_directory=build_directory,
        cmake_path=cache["CMAKE_COMMAND"],
    )
    stdout_path = output_dir / f"{role}-idempotence.stdout"
    stderr_path = output_dir / f"{role}-idempotence.stderr"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    target_after = target_file_identities(target_paths)
    tree_after = build_tree_stat_manifest(
        build_directory,
        excluded_relative_paths=mutable_relative_paths,
    )
    require_tree_root_binding(
        tree_after,
        anchor,
        context=f"{role} post-idempotence build tree",
    )
    require(
        all(record["path"] != ".ninja_lock" for record in tree_after),
        f"{role} post-idempotence build tree contains .ninja_lock",
    )
    require(
        target_after == target_before,
        f"{role} target closure changed during idempotence proof",
    )
    require(
        tree_without_mutable_paths(
            tree_after,
            {".ninja_log"},
        )
        == tree_without_mutable_paths(
            tree_before,
            {".ninja_log"},
        ),
        f"{role} build tree changed outside .ninja_log during idempotence proof",
    )
    target_after_path = output_dir / f"{role}-idempotence-target-after.json"
    tree_after_path = output_dir / f"{role}-idempotence-tree-after.json"
    log_after_path = output_dir / f"{role}-idempotence-ninja-log-after.txt"
    d0.write_json(target_after_path, target_after)
    d0.write_json(tree_after_path, tree_after)
    _log_after_identity, log_after_bytes = write_stable_file_snapshot(
        ninja_log,
        log_after_path,
        context=f"{role} authoritative post-idempotence Ninja log",
        maximum_bytes=NINJA_LOG_MAX_BYTES,
    )
    validate_ninja_log_append(
        log_before_bytes,
        log_after_bytes,
        build_directory=build_directory,
        context=f"{role} idempotence",
    )
    expected_lock_order = [
        (invocation, phase)
        for invocation in NINJA_INVOCATION_ORDER
        for phase in ("before", "after")
    ]
    require(
        [
            (record["invocation"], record["phase"])
            for record in lock_observations
        ]
        == expected_lock_order
        and all(
            record["sequence"] == index
            and record["relative_path"] == ".ninja_lock"
            and record["absent"] is True
            for index, record in enumerate(lock_observations)
        ),
        f"{role} Ninja lock observations are incomplete",
    )

    return {
        "idempotence_schema_version": IDEMPOTENCE_SCHEMA_VERSION,
        "execution": {
            "method": NINJA_EXECUTION_METHOD,
            "build_directory": str(build_directory),
            "root_device": anchor.device,
            "root_inode": anchor.inode,
            "ninja_log_version": NINJA_LOG_VERSION,
        },
        "ninja_lock_observations": lock_observations,
        "clean_rebuild": {
            "base_clean_command": base_clean_command,
            "base_clean_returncode": base_clean.returncode,
            "base_clean_stdout": artifact_reference(base_clean_stdout_path),
            "base_clean_stderr": artifact_reference(base_clean_stderr_path),
            "dry_clean_command": dry_clean_command,
            "dry_clean_returncode": dry_clean.returncode,
            "dry_clean_stdout": artifact_reference(dry_clean_stdout_path),
            "dry_clean_stderr": artifact_reference(dry_clean_stderr_path),
            "removal_method": OUTPUT_REMOVAL_METHOD,
            "removed_outputs": {
                **artifact_reference(removed_outputs_path),
                "files": len(removed_outputs),
            },
            "rebuild_command": rebuild_command,
            "rebuild_returncode": rebuild.returncode,
            "rebuild_stdout": artifact_reference(rebuild_stdout_path),
            "rebuild_stderr": artifact_reference(rebuild_stderr_path),
            "preclosure_metadata": {
                **artifact_reference(preclosure_metadata_path),
                "files": len(preclosure_metadata),
            },
            "supplemental_targets": supplemental_targets,
            "derived_before": {
                **artifact_reference(derived_before_path),
                "files": len(derived_before),
            },
            "absent_before": sorted(
                str(path) for path in absent_before_paths
            ),
            "derived_absent_after_clean": sorted(
                str(path) for path in derived_paths
            ),
            "derived_after": {
                **artifact_reference(derived_after_path),
                "files": len(derived_after),
            },
            "content_deltas": {
                **artifact_reference(content_deltas_path),
                "files": len(content_deltas),
            },
            "immutable_before": {
                **artifact_reference(immutable_before_path),
                "files": len(immutable_before),
            },
            "immutable_after_clean": {
                **artifact_reference(immutable_after_clean_path),
                "files": len(immutable_after_clean),
            },
            "immutable_after": {
                **artifact_reference(immutable_after_path),
                "files": len(immutable_after),
            },
            "protected_before": {
                **artifact_reference(protected_before_path),
                "files": len(protected_before),
            },
            "protected_after_clean": {
                **artifact_reference(protected_after_clean_path),
                "files": len(protected_after_clean),
            },
            "protected_after": {
                **artifact_reference(protected_after_path),
                "files": len(protected_after),
            },
        },
        "settlement": {
            "command": settlement_command,
            "returncode": settlement_completed.returncode,
            "stdout": artifact_reference(settlement_stdout_path),
            "stderr": artifact_reference(settlement_stderr_path),
            "dependency_report_command": dependency_report_command,
            "dependency_report_before": {
                "returncode": dependency_report_before.returncode,
                "stdout": artifact_reference(
                    dependency_report_before_stdout_path
                ),
                "stderr": artifact_reference(
                    dependency_report_before_stderr_path
                ),
                "records": len(dependency_records_before),
            },
            "dependency_report_after": {
                "returncode": dependency_report_after.returncode,
                "stdout": artifact_reference(
                    dependency_report_after_stdout_path
                ),
                "stderr": artifact_reference(
                    dependency_report_after_stderr_path
                ),
                "records": len(dependency_records_after),
            },
            "target_before": {
                **artifact_reference(settlement_target_before_path),
                "files": len(settlement_target_before),
            },
            "target_after": {
                **artifact_reference(settlement_target_after_path),
                "files": len(settlement_target_after),
            },
            "build_tree_before": {
                **artifact_reference(settlement_tree_before_path),
                "entries": len(settlement_tree_before),
            },
            "build_tree_after": {
                **artifact_reference(settlement_tree_after_path),
                "entries": len(settlement_tree_after),
            },
            "ninja_deps_before": artifact_reference(
                settlement_deps_before_path
            ),
            "ninja_deps_after": artifact_reference(
                settlement_deps_after_path
            ),
            "ninja_log_before": artifact_reference(
                settlement_log_before_path
            ),
            "ninja_log_after": artifact_reference(
                settlement_log_after_path
            ),
            "allowed_mutable_paths": [
                str(ninja_deps),
                str(ninja_log),
            ],
            "closure_dependency_outputs": len(
                closure["dependency_graph"]
            ),
            "added_dependency_outputs": added_dependency_outputs,
            "removed_dependency_outputs": removed_dependency_outputs,
        },
        "command": command,
        "returncode": completed.returncode,
        "stdout": artifact_reference(stdout_path),
        "stderr": artifact_reference(stderr_path),
        "target_before": {
            **artifact_reference(target_before_path),
            "files": len(target_before),
        },
        "target_after": {
            **artifact_reference(target_after_path),
            "files": len(target_after),
        },
        "build_tree_before": {
            **artifact_reference(tree_before_path),
            "entries": len(tree_before),
        },
        "build_tree_after": {
            **artifact_reference(tree_after_path),
            "entries": len(tree_after),
        },
        "allowed_mutable_paths": [str(ninja_log)],
        "ninja_log_before": artifact_reference(log_before_path),
        "ninja_log_after": artifact_reference(log_after_path),
    }


def capture_build_idempotence(
    *,
    role: str,
    output_dir: Path,
    build_directory: Path,
    requested_target: str,
    build_tool: dict[str, Any],
    cache: dict[str, str],
    closure: dict[str, Any],
    cache_path: Path,
    compile_path: Path,
    expected_output: Path,
    link_arguments: list[str],
    response_references: set[tuple[Path, Path]],
    preclosure_metadata: list[dict[str, Any]],
    preclosure_metadata_path: Path,
) -> dict[str, Any]:
    with open_build_directory_anchor(build_directory) as anchor:
        return _capture_build_idempotence_anchored(
            role=role,
            output_dir=output_dir,
            build_directory=build_directory,
            anchor=anchor,
            requested_target=requested_target,
            build_tool=build_tool,
            cache=cache,
            closure=closure,
            cache_path=cache_path,
            compile_path=compile_path,
            expected_output=expected_output,
            link_arguments=link_arguments,
            response_references=response_references,
            preclosure_metadata=preclosure_metadata,
            preclosure_metadata_path=preclosure_metadata_path,
        )


def capture_build(
    role: str,
    binary: Path,
    *,
    experiment: ExperimentSpec,
    repo: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    repo = canonical_repo(repo, label=role)
    description = d0.run_text(["/usr/bin/file", str(binary)], cwd=repo).stdout.strip()
    require(
        "Mach-O 64-bit executable arm64" in description,
        f"{role} binary is not native arm64: {description}",
    )
    build_directory, cache_path, compile_path = d0.find_build_metadata(binary)
    build_products = ordered_build_products(
        binary,
        build_directory=build_directory,
        label=role,
    )
    ninja_target = str(build_products[0]["requested_target"])
    product_descriptions = {
        "benchmark": description,
    }
    for product in build_products[1:]:
        product_description = d0.run_text(
            ["/usr/bin/file", str(product["expected_output"])],
            cwd=repo,
        ).stdout.strip()
        require(
            "Mach-O 64-bit executable arm64" in product_description,
            f"{role} {product['name']} is not native arm64: "
            f"{product_description}",
        )
        product_descriptions[str(product["name"])] = product_description
    cache_text = cache_path.read_text(encoding="utf-8")
    cache = d0.parse_cmake_cache(cache_text)
    validate_cmake_home_directory(cache, repo=repo, label=role)
    validate_production_build_settings(cache, role=role)
    validate_switch_states(cache, role=role, experiment=experiment)

    cache_name = f"{role}-CMakeCache.txt"
    full_compile_name = f"{role}-full-compile_commands.json"
    compile_name = f"{role}-compile_commands.json"
    commands_name = f"{role}-ninja-commands.txt"
    dependencies_name = f"{role}-dependencies.json"
    expanded_arguments_name = f"{role}-normalized-compile-arguments.json"
    expanded_link_arguments_name = f"{role}-normalized-link-arguments.json"
    shutil.copy2(cache_path, output_dir / cache_name)
    shutil.copy2(compile_path, output_dir / full_compile_name)
    preclosure_metadata_path = (
        output_dir / f"{role}-rebuild-preclosure-metadata.json"
    )
    preclosure_metadata = capture_preclosure_metadata(
        rebuild_metadata_paths(
            build_directory=build_directory,
            cache_path=cache_path,
            compile_path=compile_path,
        ),
        output_dir=output_dir,
    )
    d0.write_json(preclosure_metadata_path, preclosure_metadata)

    all_entries = json.loads(compile_path.read_text(encoding="utf-8"))
    require(isinstance(all_entries, list) and all_entries, f"{role} compile database is empty")
    tools = {
        record_name: d0.executable_record(
            cache[cache_name],
            version_arguments=list(version_arguments),
            cwd=repo,
            output_dir=output_dir,
        )
        for record_name, cache_name, version_arguments in BUILD_TOOL_SPECS
    }
    build_tool = tools["build_tool"]
    require(
        build_tool["version"] == NINJA_VERSION,
        f"{role} Ninja version is unsupported: {build_tool['version']!r}",
    )
    response_references: set[tuple[Path, Path]] = set()
    product_compile_sets: list[dict[str, Any]] = []
    product_compile_provenance: list[dict[str, Any]] = []
    independent_link_arguments: list[list[str]] = []
    for product in build_products:
        product_name = str(product["name"])
        product_command = [
            build_tool["resolved_path"],
            "-C",
            str(build_directory),
            "-t",
            "commands",
            str(product["requested_target"]),
        ]
        product_commands = d0.run_text(product_command, cwd=repo).stdout
        require(
            product_commands.strip() != "",
            f"{role} {product_name} Ninja command list is empty",
        )
        product_entries = d0.target_compile_entries(
            all_entries,
            product_commands,
            cache=cache,
            build_directory=build_directory,
            label=f"{role} {product_name} Ninja commands",
            response_references=response_references,
        )
        d0.validate_release_build(
            cache,
            product_entries,
            label=f"{role} {product_name}",
            response_references=response_references,
        )
        product_link_sets = d0.validate_ninja_commands(
            product_commands,
            cache=cache,
            build_directory=build_directory,
            label=f"{role} {product_name} Ninja commands",
            response_references=response_references,
            products=[
                {
                    "name": product_name,
                    "expected_output": product["expected_output"],
                    "compile_entries": product_entries,
                }
            ],
        )
        require(
            isinstance(product_link_sets, list)
            and len(product_link_sets) == 1
            and isinstance(product_link_sets[0], list),
            f"{role} {product_name} link command is invalid",
        )
        product_command_name = (
            f"{role}-{product_name}-ninja-commands.txt"
        )
        (output_dir / product_command_name).write_text(
            product_commands,
            encoding="utf-8",
        )
        product_compile_sets.append(
            {
                "name": product_name,
                "compile_entries": product_entries,
            }
        )
        product_compile_provenance.append(
            {
                "name": product_name,
                "requested_target": str(product["requested_target"]),
                "expected_output": str(product["expected_output"]),
                "compile_entries": len(product_entries),
                "ninja_commands": {
                    "command": product_command,
                    "captured_path": product_command_name,
                    "sha256": d0.sha256(output_dir / product_command_name),
                },
            }
        )
        independent_link_arguments.append(product_link_sets[0])

    combined_command = [
        build_tool["resolved_path"],
        "-C",
        str(build_directory),
        "-t",
        "commands",
        *[
            str(product["requested_target"])
            for product in build_products
        ],
    ]
    commands = d0.run_text(combined_command, cwd=repo).stdout
    require(commands.strip() != "", f"{role} combined Ninja command list is empty")
    combined_entries = d0.target_compile_entries(
        all_entries,
        commands,
        cache=cache,
        build_directory=build_directory,
        label=f"{role} combined Ninja commands",
        response_references=response_references,
    )
    entries = d0.canonical_compile_entry_union(
        product_compile_sets,
        combined_entries=combined_entries,
        label=f"{role} compile provenance",
        response_references=response_references,
    )
    d0.validate_release_build(
        cache,
        entries,
        label=role,
        response_references=response_references,
    )
    require(
        len(build_products) == len(product_compile_sets),
        f"{role} product compile-set count differs",
    )
    command_products = [
        {
            "name": product["name"],
            "expected_output": product["expected_output"],
            "compile_entries": product_set["compile_entries"],
        }
        for product, product_set in zip(
            build_products,
            product_compile_sets,
        )
    ]
    link_argument_sets = d0.validate_ninja_commands(
        commands,
        cache=cache,
        build_directory=build_directory,
        label=f"{role} Ninja commands",
        response_references=response_references,
        products=command_products,
    )
    require(
        isinstance(link_argument_sets, list)
        and len(link_argument_sets) == len(build_products)
        and all(
            isinstance(arguments, list)
            and all(isinstance(argument, str) for argument in arguments)
            for arguments in link_argument_sets
        ),
        f"{role} ordered link arguments are invalid",
    )
    require(
        link_argument_sets == independent_link_arguments,
        f"{role} combined link commands differ from independent product captures",
    )
    (output_dir / compile_name).write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / commands_name).write_text(commands, encoding="utf-8")

    dependencies = d0.dependency_manifest(binary, cwd=repo)
    d0.write_json(output_dir / dependencies_name, dependencies)
    exactness_product = build_products[1]
    exactness_path = Path(str(exactness_product["expected_output"]))
    exactness_dependencies_name = (
        f"{role}-exactness-emitter-dependencies.json"
    )
    exactness_dependencies = d0.dependency_manifest(
        exactness_path,
        cwd=repo,
    )
    d0.write_json(
        output_dir / exactness_dependencies_name,
        exactness_dependencies,
    )
    sources = d0.compile_source_records(entries, repo=repo, output_dir=output_dir)
    arguments = compile_argument_map(
        entries,
        build_directory=build_directory,
        label=role,
        response_references=response_references,
    )
    d0.write_json(output_dir / expanded_arguments_name, arguments)
    normalized_link_arguments = [
        canonical_link_arguments(
            arguments,
            build_directory=build_directory,
        )
        for arguments in link_argument_sets
    ]
    d0.write_json(
        output_dir / expanded_link_arguments_name,
        normalized_link_arguments,
    )
    require(
        len(build_products)
        == len(product_compile_sets)
        == len(link_argument_sets),
        f"{role} closure product evidence count differs",
    )
    closure = d0.capture_build_input_closure(
        output_dir=output_dir,
        capture_prefix=role,
        build_directory=build_directory,
        build_tool=build_tool,
        cache=cache,
        products=[
            {
                **product,
                "compile_entries": product_set["compile_entries"],
                "link_arguments": arguments,
            }
            for product, product_set, arguments in zip(
                build_products,
                product_compile_sets,
                link_argument_sets,
            )
        ],
        response_references=response_references,
        label=role,
    )
    closure_record = json.loads(
        (output_dir / closure["captured_path"]).read_text(encoding="utf-8")
    )
    idempotence = capture_build_idempotence(
        role=role,
        output_dir=output_dir,
        build_directory=build_directory,
        requested_target=ninja_target,
        build_tool=build_tool,
        cache=cache,
        closure=closure_record,
        cache_path=cache_path,
        compile_path=compile_path,
        expected_output=binary,
        link_arguments=link_argument_sets[0],
        response_references=response_references,
        preclosure_metadata=preclosure_metadata,
        preclosure_metadata_path=preclosure_metadata_path,
    )
    binary_blob = d0.capture_content_blob(binary, output_dir=output_dir)
    exactness_blob = d0.capture_content_blob(
        exactness_path,
        output_dir=output_dir,
    )
    settings = {
        name: cache.get(name)
        for name in build_setting_names(experiment)
    }
    switch_states = experiment.switch_states_for_role(role)
    record = {
        "role": role,
        "engine": "infinity",
        "switch_states": switch_states,
        "binary": {
            "path": str(binary),
            "description": description,
            "sha256": d0.sha256(binary),
            "bytes": binary.stat().st_size,
            **binary_blob,
        },
        "build_directory": str(build_directory),
        "settings": settings,
        "cmake_cache": {
            "captured_path": cache_name,
            "sha256": d0.sha256(output_dir / cache_name),
        },
        "full_compile_commands": {
            "captured_path": full_compile_name,
            "sha256": d0.sha256(output_dir / full_compile_name),
            "entries": len(all_entries),
        },
        "compile_commands": {
            "captured_path": compile_name,
            "sha256": d0.sha256(output_dir / compile_name),
            "entries": len(entries),
        },
        "normalized_compile_arguments": {
            "captured_path": expanded_arguments_name,
            "sha256": d0.sha256(output_dir / expanded_arguments_name),
        },
        "normalized_link_arguments": {
            "captured_path": expanded_link_arguments_name,
            "sha256": d0.sha256(output_dir / expanded_link_arguments_name),
        },
        "ninja_commands": {
            "captured_path": commands_name,
            "sha256": d0.sha256(output_dir / commands_name),
        },
        "product_compile_provenance": product_compile_provenance,
        "ninja_idempotence": idempotence,
        "dependencies": {
            "captured_path": dependencies_name,
            "sha256": d0.sha256(output_dir / dependencies_name),
            "identity": normalized_dependency_identity(dependencies),
        },
        "exactness_emitter": {
            "product_index": 1,
            "product_name": "exactness-emitter",
            "path": str(exactness_path),
            "description": product_descriptions["exactness-emitter"],
            "sha256": d0.sha256(exactness_path),
            "bytes": exactness_path.stat().st_size,
            **exactness_blob,
        },
        "exactness_dependencies": {
            "captured_path": exactness_dependencies_name,
            "sha256": d0.sha256(
                output_dir / exactness_dependencies_name
            ),
            "identity": normalized_dependency_identity(
                exactness_dependencies
            ),
        },
        **tools,
        "compiled_sources": sources,
        "response_files": d0.response_file_records(
            response_references,
            output_dir=output_dir,
        ),
        "build_input_closure": closure,
    }
    record["_causal_validation"] = {
        "normalized_link_arguments": normalized_link_arguments,
        "closure_input_identity": closure_input_identity(
            closure_record,
            build_directory=build_directory,
        ),
    }
    runtime_hashes = {
        record["binary"]["path"]: record["binary"]["sha256"],
        record["exactness_emitter"]["path"]: record[
            "exactness_emitter"
        ]["sha256"],
    }
    runtime_hashes.update(
        {
            tool["resolved_path"]: tool["sha256"]
            for tool in tools.values()
        }
    )
    for dependency in dependencies["dependencies"]:
        if dependency["resolved_path"] is not None and dependency["sha256"] is not None:
            previous = runtime_hashes.setdefault(
                dependency["resolved_path"],
                dependency["sha256"],
            )
            require(
                previous == dependency["sha256"],
                f"{role} benchmark runtime dependency identity conflicts",
            )
    for dependency in exactness_dependencies["dependencies"]:
        if dependency["resolved_path"] is not None and dependency["sha256"] is not None:
            previous = runtime_hashes.setdefault(
                dependency["resolved_path"],
                dependency["sha256"],
            )
            require(
                previous == dependency["sha256"],
                f"{role} exactness runtime dependency identity conflicts",
            )
    return record, runtime_hashes


def source_identity(build: dict[str, Any]) -> list[tuple[str, str, int]]:
    build_directory = Path(str(build["build_directory"]))
    return sorted(
        (
            canonical_build_path(
                str(source["absolute_path"]),
                build_directory=build_directory,
            ),
            str(source["sha256"]),
            int(source["bytes"]),
        )
        for source in build["compiled_sources"]
    )


def compare_builds(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    experiment: ExperimentSpec,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo = canonical_repo(repo, label="causal")
    control_switch_states = experiment.switch_states_for_role("control")
    treatment_switch_states = experiment.switch_states_for_role("treatment")
    require(
        control["binary"]["path"] != treatment["binary"]["path"],
        "Control and treatment binary paths are identical",
    )
    require(
        control["binary"]["sha256"] != treatment["binary"]["sha256"],
        "Control and treatment binary contents are identical",
    )
    require(
        control["exactness_emitter"]["path"]
        != treatment["exactness_emitter"]["path"],
        "Control and treatment exactness emitter paths are identical",
    )
    require(
        control["exactness_emitter"]["sha256"]
        != treatment["exactness_emitter"]["sha256"],
        "Control and treatment exactness emitter contents are identical",
    )
    for setting in (
        *SHARED_BUILD_SETTINGS,
        *(name for name, _ in experiment.held_switches),
    ):
        require(
            control["settings"][setting] == treatment["settings"][setting],
            f"Builds differ in {setting}",
        )
    for role, build, expected_states in (
        ("control", control, control_switch_states),
        ("treatment", treatment, treatment_switch_states),
    ):
        require(
            build.get("switch_states") == expected_states,
            f"{role} build switch_states differ from the experiment",
        )
        for name, value in expected_states.items():
            require(
                build["settings"].get(name) == value,
                f"{role} build setting for {name} is not {value}",
            )
    for tool, _, _ in BUILD_TOOL_SPECS:
        require(
            control[tool]["resolved_path"] == treatment[tool]["resolved_path"]
            and control[tool]["sha256"] == treatment[tool]["sha256"],
            f"Builds use different {tool} binaries",
        )
    require(
        source_identity(control) == source_identity(treatment),
        "Control and treatment compiled-source identities differ",
    )
    require(
        control["dependencies"]["identity"] == treatment["dependencies"]["identity"],
        "Control and treatment runtime dependency identities differ",
    )
    require(
        control["exactness_dependencies"]["identity"]
        == treatment["exactness_dependencies"]["identity"],
        "Control and treatment exactness runtime dependency identities differ",
    )
    for role, build in (("control", control), ("treatment", treatment)):
        validate_cmake_home_directory(
            build["settings"],
            repo=repo,
            label=role,
        )

    control_arguments = json.loads(
        (output_dir / control["normalized_compile_arguments"]["captured_path"]).read_text(
            encoding="utf-8"
        )
    )
    treatment_arguments = json.loads(
        (output_dir / treatment["normalized_compile_arguments"]["captured_path"]).read_text(
            encoding="utf-8"
        )
    )
    require(
        set(control_arguments) == set(treatment_arguments),
        "Control and treatment compile-source sets differ",
    )
    control_scans = validate_compile_macro_states(
        control_arguments,
        experiment=experiment,
        role="control",
        repo=repo,
    )
    treatment_scans = validate_compile_macro_states(
        treatment_arguments,
        experiment=experiment,
        role="treatment",
        repo=repo,
    )
    varying_sources = {
        switch.name: [
            canonical_source(repo, suffix)
            for suffix in switch.compile_sources
        ]
        for switch in experiment.varying_switches
    }
    for source in sorted(control_arguments):
        control_values = list(control_arguments[source])
        treatment_values = list(treatment_arguments[source])
        control_varying_indexes: set[int] = set()
        treatment_varying_indexes: set[int] = set()
        for switch in experiment.varying_switches:
            control_varying_indexes.update(
                control_scans[source]["argument_indexes"][switch.name]
            )
            treatment_varying_indexes.update(
                treatment_scans[source]["argument_indexes"][switch.name]
            )
        control_values = without_argument_indexes(
            control_values,
            control_varying_indexes,
        )
        treatment_values = without_argument_indexes(
            treatment_values,
            treatment_varying_indexes,
        )
        require(
            control_values == treatment_values,
            f"Compile arguments differ beyond declared varying switches for {source}",
        )
    require(
        control["_causal_validation"]["normalized_link_arguments"]
        == treatment["_causal_validation"]["normalized_link_arguments"],
        "Link arguments differ beyond build-directory paths",
    )
    require(
        control["_causal_validation"]["closure_input_identity"]
        == treatment["_causal_validation"]["closure_input_identity"],
        "Build-input closures differ outside derived outputs",
    )
    return {
        "experiment": experiment.name,
        "varying_switches": [
            switch.record() for switch in experiment.varying_switches
        ],
        "held_switches": [
            {"name": name, "value": value}
            for name, value in experiment.held_switches
        ],
        "control_switch_states": control_switch_states,
        "treatment_switch_states": treatment_switch_states,
        "engine": "infinity",
        "source_identity_equal": True,
        "runtime_dependency_identity_equal": True,
        "exactness_runtime_dependency_identity_equal": True,
        "toolchain_identity_equal": True,
        "normalized_compile_arguments_equal_after_declared_switches": True,
        "normalized_link_arguments_equal": True,
        "build_input_identity_equal": True,
        "varying_compile_sources": varying_sources,
        "permitted_compile_differences": [
            switch.macro for switch in experiment.varying_switches
        ],
    }


def runtime_snapshots(
    hashes: dict[str, str],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for path_text, expected in sorted(hashes.items()):
        snapshot = d0.capture_content_blob(Path(path_text), output_dir=output_dir)
        require(snapshot["sha256"] == expected, f"Runtime artifact changed: {path_text}")
        snapshots.append({"original_path": path_text, **snapshot})
    return snapshots


def capture_program(
    source_path: Path,
    capture_name: str,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    capture_path = output_dir / capture_name
    shutil.copy2(source_path, capture_path)
    digest = d0.sha256(source_path)
    require(d0.sha256(capture_path) == digest, f"Captured program differs: {capture_name}")
    return {
        "source_path": str(source_path),
        "captured_path": capture_name,
        "sha256": digest,
        "bytes": capture_path.stat().st_size,
    }


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
    require(
        exactness.EXACTNESS_SCHEMA_VERSION
        == REQUIRED_EXACTNESS_SCHEMA_VERSION,
        "Causal exactness producer schema version mismatch: "
        f"expected {REQUIRED_EXACTNESS_SCHEMA_VERSION}, "
        f"found {exactness.EXACTNESS_SCHEMA_VERSION}",
    )
    require(
        exactness_verifier.SCHEMA_VERSION
        == REQUIRED_EXACTNESS_SCHEMA_VERSION,
        "Causal exactness verifier schema version mismatch: "
        f"expected {REQUIRED_EXACTNESS_SCHEMA_VERSION}, "
        f"found {exactness_verifier.SCHEMA_VERSION}",
    )
    experiment = args.experiment_spec
    expected_witnesses = validate_expected_witnesses(
        expected_witnesses_for_experiment(experiment),
        experiment=experiment,
        context="causal expected witnesses",
    )
    parent_environment_policy = d0.reject_unsafe_parent_environment()
    require(platform.machine() == "arm64", "Causal campaign must run on native arm64")
    require(args.control_binary != args.treatment_binary, "Role binaries must use distinct paths")
    disk = shutil.disk_usage(args.repo)
    required_free = d0.MINIMUM_DISK_RESERVE_BYTES + d0.D0_WORST_CASE_NEW_BYTES
    require(
        disk.free >= required_free,
        f"Causal campaign requires {required_free} free bytes, found {disk.free}",
    )

    roles: dict[str, Any] = {}
    runtime_hashes: dict[str, str] = {}
    for role in ("control", "treatment"):
        build, build_runtime_hashes = capture_build(
            role,
            role_binary(args, role),
            experiment=experiment,
            repo=args.repo,
            output_dir=output_dir,
        )
        roles[role] = build
        for path, digest in build_runtime_hashes.items():
            previous = runtime_hashes.setdefault(path, digest)
            require(previous == digest, f"Runtime artifact differs between builds: {path}")
    contrast = compare_builds(
        roles["control"],
        roles["treatment"],
        experiment=experiment,
        repo=args.repo,
        output_dir=output_dir,
    )
    for role in ("control", "treatment"):
        roles[role].pop("_causal_validation")

    runner_path = Path(__file__).resolve()
    verifier_path = runner_path.with_name("verify_native_hnsw_batch4_causal.py")
    require(verifier_path.is_file(), f"Causal verifier is missing: {verifier_path}")
    runner = capture_program(runner_path, RUNNER_CAPTURE_FILENAME, output_dir=output_dir)
    verifier = capture_program(
        verifier_path,
        VERIFIER_CAPTURE_FILENAME,
        output_dir=output_dir,
    )
    local_python_modules = [
        {
            "module": module,
            **capture_program(
                source_path,
                capture_name,
                output_dir=output_dir,
            ),
        }
        for module, capture_name, source_path in LOCAL_PYTHON_MODULES
    ]
    python = d0.executable_record(
        sys.executable,
        version_arguments=["--version"],
        cwd=args.repo,
        output_dir=output_dir,
    )
    for program in (runner, verifier, *local_python_modules):
        runtime_hashes[program["source_path"]] = program["sha256"]
    runtime_hashes[python["resolved_path"]] = python["sha256"]
    time_wrapper_path = Path(
        d0.process_executable_identity(Path("/usr/bin/time"))["path"]
    )
    runtime_hashes[str(time_wrapper_path)] = d0.sha256(time_wrapper_path)
    exactness_gate = exactness.run_exactness_gate(
        dataset_path=output_dir / str(dataset["path"]),
        emitters={
            role: Path(str(roles[role]["exactness_emitter"]["path"]))
            for role in ("control", "treatment")
        },
        output_dir=output_dir,
        timeout_seconds=args.member_timeout_seconds,
        expected_emitters={
            role: roles[role]["exactness_emitter"]
            for role in ("control", "treatment")
        },
        expected_witnesses=expected_witnesses,
    )
    exactness_gate["independent_verification"] = (
        exactness_verifier.validate_exactness_gate(
            output_dir,
            exactness_gate["independent_verifier_record"],
            expected_witnesses=expected_witnesses,
        )
    )
    snapshots = runtime_snapshots(runtime_hashes, output_dir=output_dir)

    git_repository = d0.capture_git_repository(
        args.repo,
        output_dir=output_dir,
        label="source-repository",
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
    schedule_sha256 = d0.sha256(schedule_path)
    campaign_binding = create_campaign_binding(
        roles=roles,
        dataset=dataset,
        schedule_sha256=schedule_sha256,
    )
    return {
        "causal_schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment": experiment.record(),
        "expected_witnesses": expected_witnesses,
        "scope": experiment.scope,
        "execution_mode": (
            PREFLIGHT_ONLY_MODE if args.preflight_only else FULL_CAMPAIGN_MODE
        ),
        "captured_at": d0.utc_now(),
        "repo": str(args.repo),
        "runner": {
            **runner,
            "argv": list(sys.argv),
            "invocation_working_directory": str(Path.cwd().resolve()),
            "python_executable": python,
            "effective_arguments": {
                "repo": str(args.repo),
                "control_binary": str(args.control_binary),
                "treatment_binary": str(args.treatment_binary),
                "experiment": experiment.name,
                "output_directory": str(args.output_dir),
                "idle_minimum_percent": args.idle_minimum_percent,
                "idle_timeout_seconds": args.idle_timeout_seconds,
                "member_timeout_seconds": args.member_timeout_seconds,
                "preflight_only": args.preflight_only,
            },
        },
        "verifier": verifier,
        "local_python_modules": local_python_modules,
        "host": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "kernel": platform.release(),
            "python": platform.python_version(),
            "disk": {
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
                "minimum_reserve_bytes": d0.MINIMUM_DISK_RESERVE_BYTES,
                "declared_worst_case_new_bytes": d0.D0_WORST_CASE_NEW_BYTES,
                "required_free_bytes": required_free,
            },
        },
        "environment_control": {
            "parent_performance_environment": parent_performance_environment,
            "unsafe_parent_environment_policy": parent_environment_policy,
            "tool_environment_template": d0.tool_environment(),
            "tools_inherit_parent_environment": False,
            "child_environment_template": d0.benchmark_environment(
                d0.MEASURED_PARTICIPANTS
            ),
            "inherits_parent_environment": False,
        },
        "dataset": {
            **dataset,
            "record_path": dataset_record_path.name,
            "record_sha256": d0.sha256(dataset_record_path),
        },
        "heldout": {
            **heldout,
            "record_path": heldout_record_path.name,
            "record_sha256": d0.sha256(heldout_record_path),
        },
        "schedule": {
            "path": schedule_path.name,
            "sha256": schedule_sha256,
            "members": len(schedule()),
            "pair_plan": [
                {
                    "pair": pair,
                    "phase": phase,
                    "participants": participants,
                    "order": "/".join(pair_roles),
                }
                for pair, phase, participants, pair_roles in PAIR_PLAN
            ],
        },
        "campaign_binding": campaign_binding,
        "roles": roles,
        "causal_contrast": contrast,
        "exactness_gate": exactness_gate,
        "runtime_artifact_hashes": runtime_hashes,
        "runtime_artifact_snapshots": snapshots,
        "source_provenance": {"git_repository": git_repository},
        "protocol": {
            "audit_sidecar_schema_version": d0.AUDIT_SIDECAR_SCHEMA_VERSION,
            "query_benchmark": d0.query_protocol(),
            "attestation_schema_version": d0.ATTESTATION_SCHEMA_VERSION,
            "attestation_barrier_phases": list(d0.ATTESTATION_BARRIER_PHASES),
            "index_timing_binding_schema_version": (
                d0.INDEX_TIMING_BINDING_SCHEMA_VERSION
            ),
            "index_timing_maximum_boundary_overhead_ns": (
                d0.INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
            ),
            "process_evidence_schema_version": d0.PROCESS_EVIDENCE_SCHEMA_VERSION,
            "exactness_schema_version": REQUIRED_EXACTNESS_SCHEMA_VERSION,
            "exactness_timeout_seconds_per_run": args.member_timeout_seconds,
            "performance_measurement_scope": PERFORMANCE_MEASUREMENT_SCOPE,
            "time_wrapper_path": str(time_wrapper_path),
            "heldout_query_count": d0.HELDOUT_QUERY_COUNT,
            "heldout_query_seed": d0.HELDOUT_QUERY_SEED,
            "recall_points": [{"k": k, "ef": ef} for k, ef in d0.RECALL_POINTS],
            "minimum_speedup": MINIMUM_SPEEDUP,
            "maximum_recall_deficit": MAXIMUM_RECALL_DEFICIT,
            "maximum_tps_relative_mad": MAXIMUM_RELATIVE_MAD,
            "maximum_query_latency_relative_mad": MAXIMUM_RELATIVE_MAD,
            "minimum_query_tps_ratio": MINIMUM_QUERY_TPS_RATIO,
            "maximum_query_latency_ratio": MAXIMUM_QUERY_LATENCY_RATIO,
            "minimum_query_recall": MINIMUM_QUERY_RECALL,
            "maximum_query_recall_gap": MAXIMUM_QUERY_RECALL_GAP,
            "require_single_thread_semantic_payload_exact_match": True,
            "require_every_measured_pair_faster": True,
            "outlier_policy": "retain-all",
            "idle_minimum_percent": args.idle_minimum_percent,
            "idle_policy": d0.idle_policy(args.idle_minimum_percent),
            "idle_window_seconds": d0.IDLE_WINDOW_SECONDS,
            "idle_sample_count": d0.IDLE_SAMPLE_COUNT,
            "idle_to_launch_max_ns": d0.IDLE_TO_LAUNCH_MAX_NS,
            "server_ports": list(d0.SERVER_PORTS),
            "power_policy": d0.POWER_POLICY,
            "battery_minimum_percent": d0.BATTERY_MINIMUM_PERCENT,
            "global_swap_policy": d0.GLOBAL_SWAP_POLICY,
            "member_timeout_seconds": args.member_timeout_seconds,
            "idle_timeout_seconds": args.idle_timeout_seconds,
        },
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
    role = str(entry["role"])
    prefix = f"{entry['sequence']:02d}-{entry['pair']}-{role}"
    binary = role_binary(args, role)
    member_binding = member_campaign_binding(entry, campaign_binding)
    require(
        member_binding["binary_sha256"] == binary_hashes[role],
        f"{role} campaign binary differs from preflight",
    )
    require(
        member_binding["dataset_sha256"] == dataset["sha256"],
        "Campaign dataset differs from the member dataset",
    )
    sidecar_name = d0.audit_sidecar_path(prefix)
    sidecar_path = output_dir / sidecar_name
    require(not sidecar_path.exists(), f"Audit sidecar already exists: {sidecar_path}")
    require(not dataset_path.is_symlink(), "Benchmark dataset must not be a symlink")
    dataset_before = d0.verify_dataset(dataset_path, dataset)
    require(d0.sha256(binary) == binary_hashes[role], f"{role} binary changed")
    runtime_before = d0.verify_runtime_artifacts(runtime_artifact_hashes)
    host_before = d0.capture_member_preflight(
        cwd=args.repo,
        output_dir=output_dir,
        prefix=prefix,
    )
    command = [
        "/usr/bin/time",
        "-lp",
        str(binary),
        dataset["path"],
        str(d0.VECTORS),
        str(d0.DIMENSIONS),
        str(d0.M),
        str(d0.EF_CONSTRUCTION),
        str(d0.EF_SEARCH),
        str(d0.CHUNK_SIZE),
        str(d0.QUERY_COUNT),
        str(entry["participants"]),
        str(d0.BUILD_GRAIN),
        sidecar_name,
        str(member_binding["campaign_nonce"]),
        str(member_binding["schedule_sequence"]),
        str(member_binding["role_id"]),
        str(member_binding["binary_sha256"]),
        str(member_binding["dataset_sha256"]),
    ]
    environment = d0.benchmark_environment(int(entry["participants"]))
    wrapper_process_identity = d0.process_executable_identity(Path(command[0]))
    benchmark_process_identity = d0.process_executable_identity(binary)
    idle = d0.wait_for_idle(
        cwd=args.repo,
        output_dir=output_dir,
        prefix=prefix,
        timeout_seconds=args.idle_timeout_seconds,
        minimum_idle_percent=args.idle_minimum_percent,
    )
    require(
        d0.idle_record_passes(
            idle,
            expected_minimum_idle_percent=args.idle_minimum_percent,
        ),
        f"CPU-idle preflight failed before {prefix}",
    )

    started_at = d0.utc_now()
    started_ns = time.time_ns()
    idle_to_launch_ns = time.monotonic_ns() - int(idle["accepted_at_monotonic_ns"])
    require(
        0 <= idle_to_launch_ns <= d0.IDLE_TO_LAUNCH_MAX_NS,
        f"Idle-to-launch delay exceeded one second before {prefix}",
    )
    process = subprocess.Popen(
        command,
        cwd=output_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    supervision = d0.supervise_attested_process(
        process,
        cwd=args.repo,
        timeout_seconds=args.member_timeout_seconds,
        expected_wrapper_executable=wrapper_process_identity,
        expected_benchmark_executable=benchmark_process_identity,
    )
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
        driver_stdout, attestation = d0.split_attested_driver_output(stdout)
        fields = d0.parse_driver_result(
            driver_stdout,
            engine="infinity",
            participants=int(entry["participants"]),
            dataset=dataset,
            heldout=truth,
        )
        timing_binding = d0.index_timing_binding(
            supervision["barriers"],
            fields,
            engine="infinity",
        )
    except d0.D0Failure as error:
        errors.append(str(error))
    audit: dict[str, Any] = {}
    sidecar_data: bytes | None = None
    if fields:
        try:
            sidecar_data = d0.read_audit_sidecar(sidecar_path)
            audit = d0.parse_audit_sidecar(
                sidecar_path,
                engine="infinity",
                fields=fields,
                dataset=dataset,
                truth=truth,
                expected_campaign_nonce=str(member_binding["campaign_nonce"]),
                expected_schedule_sequence=int(
                    member_binding["schedule_sequence"]
                ),
                expected_role_id=int(member_binding["role_id"]),
                expected_binary_sha256=str(
                    member_binding["binary_sha256"]
                ),
                expected_dataset_sha256=str(
                    member_binding["dataset_sha256"]
                ),
                data=sidecar_data,
            )
            validate_run_execution_witness(
                audit.get("execution_witness"),
                experiment=args.experiment_spec,
                role=role,
                context=f"{prefix} execution witness",
            )
        except d0.D0Failure as error:
            errors.append(str(error))
        except CausalFailure as error:
            errors.append(str(error))
    if process.returncode != 0:
        errors.append(f"process exited with code {process.returncode}")
    if timed_out:
        errors.append(f"process exceeded {args.member_timeout_seconds} second timeout")
    max_rss = d0.parse_max_rss(stderr)
    if max_rss is None:
        errors.append("missing maximum resident set size")
    process_swaps = d0.parse_process_swaps(stderr)
    if process_swaps is None:
        errors.append("missing process swap count")
    elif process_swaps != 0:
        errors.append(f"process reported {process_swaps} swaps")

    try:
        dataset_after = d0.verify_dataset(dataset_path, dataset)
    except d0.D0Failure as error:
        dataset_after = {"verified_at": d0.utc_now(), "error": str(error)}
        errors.append(str(error))
    try:
        runtime_after = d0.verify_runtime_artifacts(runtime_artifact_hashes)
    except d0.D0Failure as error:
        runtime_after = {"verified_at": d0.utc_now(), "error": str(error)}
        errors.append(str(error))
    swap_growth: int | None = None
    try:
        host_after = d0.capture_member_postflight(
            cwd=args.repo,
            output_dir=output_dir,
            prefix=prefix,
        )
        swap_growth = int(host_after["swap"]["used_bytes"]) - int(
            host_before["swap"]["used_bytes"]
        )
    except d0.D0Failure as error:
        host_after = {"captured_at": d0.utc_now(), "error": str(error)}
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
        "experiment": args.experiment_spec.name,
        "scope": args.experiment_spec.scope,
        "errors": errors,
        "pid": process.pid,
        "started_at": started_at,
        "started_at_unix_ns": started_ns,
        "ended_at_unix_ns": ended_ns,
        "command": command,
        "working_directory": ".",
        "binary": {"path": str(binary), "sha256": binary_hashes[role]},
        "environment": environment,
        "idle": idle,
        "idle_to_launch_ns": idle_to_launch_ns,
        "host_before": host_before,
        "host_after": host_after,
        "global_swap_growth_bytes": swap_growth,
        "dataset_before": dataset_before,
        "dataset_after": dataset_after,
        "runtime_artifacts_before": runtime_before,
        "runtime_artifacts_after": runtime_after,
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
    d0.write_json(output_dir / f"{prefix}.json", record)
    require(not errors, f"{prefix} failed validation: {errors}")
    return record


def recall_point_fraction(record: dict[str, Any], k: int, ef: int) -> Fraction:
    matches = [
        point
        for point in record["audit"]["recall"]["points"]
        if point["k"] == k and point["ef"] == ef
    ]
    require(len(matches) == 1, f"Run lacks a unique recall point k={k}, ef={ef}")
    point = matches[0]
    require(
        set(point)
        == {"k", "ef", "eligible_hits", "possible_hits", "recall"},
        f"Run recall point keys differ k={k}, ef={ef}",
    )
    eligible_hits = point.get("eligible_hits")
    possible_hits = point.get("possible_hits")
    require(
        type(eligible_hits) is int
        and type(possible_hits) is int
        and 0 <= eligible_hits <= possible_hits
        and possible_hits == d0.HELDOUT_QUERY_COUNT * k,
        f"Run has invalid recall hit counts k={k}, ef={ef}",
    )
    recall = Fraction(eligible_hits, possible_hits)
    recorded_recall = point.get("recall")
    require(
        type(recorded_recall) is float
        and math.isfinite(recorded_recall)
        and recorded_recall == float(recall),
        f"Run recall float differs from hit counts k={k}, ef={ef}",
    )
    return recall


def recall_point(record: dict[str, Any], k: int, ef: int) -> float:
    return float(recall_point_fraction(record, k, ef))


def correctness_semantic_payload(record: dict[str, Any]) -> dict[str, Any]:
    audit = record["audit"]
    fields = record["fields"]
    graph = audit["graph"]
    require(
        isinstance(graph.get("levels_sha256"), str),
        "Correctness audit lacks the graph level hash",
    )
    returned_results: list[dict[str, Any]] = []
    for k, ef in d0.RECALL_POINTS:
        ids_name = f"infinity_returned_ids_ef_{ef}_k_{k}"
        distances_name = (
            f"infinity_returned_distances_sha256_ef_{ef}_k_{k}"
        )
        require(
            isinstance(fields.get(ids_name), str),
            f"Correctness fields lack {ids_name}",
        )
        require(
            isinstance(fields.get(distances_name), str),
            f"Correctness fields lack {distances_name}",
        )
        returned_results.append(
            {
                "k": k,
                "ef": ef,
                "returned_ids": fields[ids_name],
                "returned_distances_sha256": fields[distances_name],
            }
        )
    query_checksum = audit["query"].get("result_checksum")
    require(
        type(query_checksum) is int,
        "Correctness audit lacks an integer query result checksum",
    )
    return {
        "graph": graph,
        "levels_sha256": graph["levels_sha256"],
        "recall": audit["recall"],
        "returned_results": returned_results,
        "query_result_checksum": query_checksum,
    }


def measured_query_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    query = record["audit"]["query"]
    expected_keys = {
        *d0.query_protocol(),
        "timed_query_corpus_sha256",
        "latency_samples_ns",
        "latency_validated_operations",
        "latency_validated_result_checksum",
        "throughput_operations",
        "throughput_validated_operations",
        "throughput_wall_ns",
        "throughput_validated_result_checksum",
        "result_checksum",
        "per_query_checksum_count",
        "per_query_checksums",
        "qps",
        "tps",
        "p50_ns",
        "p95_ns",
        "p99_ns",
    }
    require(
        type(query) is dict and set(query) == expected_keys,
        f"{record['pair']} {record['role']} query evidence shape differs",
    )
    reconstructed = d0.query_performance_record(
        timed_query_corpus_sha256=query["timed_query_corpus_sha256"],
        latency_samples_ns=query["latency_samples_ns"],
        latency_validated_operations=query["latency_validated_operations"],
        latency_validated_result_checksum=(
            query["latency_validated_result_checksum"]
        ),
        throughput_operations=query["throughput_operations"],
        throughput_validated_operations=query[
            "throughput_validated_operations"
        ],
        throughput_wall_ns=query["throughput_wall_ns"],
        throughput_validated_result_checksum=(
            query["throughput_validated_result_checksum"]
        ),
        result_checksum=query["result_checksum"],
        per_query_checksums=query["per_query_checksums"],
    )
    require(
        query == reconstructed,
        f"{record['pair']} {record['role']} query evidence differs from raw measurements",
    )
    require(
        query["timed_query_corpus_sha256"]
        == record["audit"]["recall"].get("queries_sha256"),
        f"{record['pair']} {record['role']} timed query corpus differs from replay",
    )
    operations = query.get("throughput_operations")
    wall_ns = query.get("throughput_wall_ns")
    require(
        type(operations) is int and operations > 0,
        f"{record['pair']} {record['role']} query operation count is invalid",
    )
    require(
        type(wall_ns) is int and wall_ns > 0,
        f"{record['pair']} {record['role']} query wall time is invalid",
    )
    tps_fraction = Fraction(operations * 1_000_000_000, wall_ns)
    tps = float(tps_fraction)
    recorded_tps = float(query["tps"])
    p50_ns = query.get("p50_ns")
    p95_ns = query.get("p95_ns")
    p99_ns = query.get("p99_ns")
    require(
        math.isfinite(recorded_tps)
        and math.isclose(recorded_tps, tps, rel_tol=1e-12, abs_tol=1e-12),
        f"{record['pair']} {record['role']} recorded query TPS differs from raw evidence",
    )
    require(
        math.isfinite(tps) and tps > 0.0,
        f"{record['pair']} {record['role']} query TPS is invalid",
    )
    require(
        type(p50_ns) is int
        and type(p95_ns) is int
        and type(p99_ns) is int
        and 0 < p50_ns <= p95_ns <= p99_ns,
        f"{record['pair']} {record['role']} query latency percentiles are invalid",
    )
    return {
        "tps": tps,
        "p50_ns": p50_ns,
        "p95_ns": p95_ns,
        "p99_ns": p99_ns,
        "recall_at_10": recall_point(
            record,
            d0.QUERY_K,
            d0.QUERY_EF_SEARCH,
        ),
    }


def measured_query_tps_fraction(record: dict[str, Any]) -> Fraction:
    query = record["audit"]["query"]
    operations = query.get("throughput_operations")
    wall_ns = query.get("throughput_wall_ns")
    require(
        type(operations) is int and operations > 0,
        f"{record['pair']} {record['role']} query operation count is invalid",
    )
    require(
        type(wall_ns) is int and wall_ns > 0,
        f"{record['pair']} {record['role']} query wall time is invalid",
    )
    return Fraction(operations * 1_000_000_000, wall_ns)


def summarize(
    records: list[dict[str, Any]],
    *,
    experiment: ExperimentSpec,
    dataset: dict[str, Any],
    schedule_sha256: str,
    idle_minimum_percent: float,
) -> dict[str, Any]:
    validate_records_against_schedule(records)
    require(
        all(record.get("experiment") == experiment.name for record in records),
        "Run records differ from the selected experiment",
    )
    for record in records:
        validate_run_execution_witness(
            record.get("audit", {}).get("execution_witness"),
            experiment=experiment,
            role=str(record.get("role")),
            context=f"{record.get('pair')} {record.get('role')} execution witness",
        )
    measured_run_execution_witnesses_valid = True
    measured = [record for record in records if record["phase"] == "measured"]
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    durations: dict[str, list[int]] = {"control": [], "treatment": []}
    for record in measured:
        pair = str(record["pair"])
        role = str(record["role"])
        require(role not in by_pair.setdefault(pair, {}), f"Duplicate {role} in {pair}")
        by_pair[pair][role] = record
        durations[role].append(
            d0.parse_unsigned_field(record["fields"], "infinity_cold_build_ns")
        )

    pairs: list[dict[str, Any]] = []
    for pair in ("P1", "P2", "P3", "P4", "P5", "P6"):
        members = by_pair.get(pair, {})
        require(set(members) == {"control", "treatment"}, f"Incomplete pair {pair}")
        control_ns = d0.parse_unsigned_field(
            members["control"]["fields"], "infinity_cold_build_ns"
        )
        treatment_ns = d0.parse_unsigned_field(
            members["treatment"]["fields"], "infinity_cold_build_ns"
        )
        order = str(members["control"]["pair_order"])
        require(order == members["treatment"]["pair_order"], f"Order differs in {pair}")
        pairs.append(
            {
                "pair": pair,
                "order": order,
                "control_cold_build_ns": control_ns,
                "treatment_cold_build_ns": treatment_ns,
                "control_vectors_per_second": d0.VECTORS * 1e9 / control_ns,
                "treatment_vectors_per_second": d0.VECTORS * 1e9 / treatment_ns,
                "treatment_speedup": control_ns / treatment_ns,
            }
        )
    strata: dict[str, Any] = {}
    for order in ("control/treatment", "treatment/control"):
        order_pairs = [pair for pair in pairs if pair["order"] == order]
        values = [pair["treatment_speedup"] for pair in order_pairs]
        require(len(values) == 3, f"Order stratum {order} must contain three pairs")
        speedup_product = Fraction(
            math.prod(pair["control_cold_build_ns"] for pair in order_pairs),
            math.prod(pair["treatment_cold_build_ns"] for pair in order_pairs),
        )
        strata[order] = {
            "treatment_speedups": values,
            "geometric_mean_speedup": d0.geometric_mean(values),
            "speedup_product_exact": fraction_record(speedup_product),
        }
    overall_speedup_product = Fraction(
        math.prod(pair["control_cold_build_ns"] for pair in pairs),
        math.prod(pair["treatment_cold_build_ns"] for pair in pairs),
    )
    required_speedup_product = Fraction(
        *MINIMUM_SPEEDUP_FRACTION
    ) ** len(pairs)
    minimum_effect = overall_speedup_product >= required_speedup_product
    estimate = (
        MINIMUM_SPEEDUP
        if overall_speedup_product == required_speedup_product
        else math.pow(float(overall_speedup_product), 1.0 / len(pairs))
    )
    minimum_speedup_comparison_exact = {
        "observed_speedup_product": fraction_record(overall_speedup_product),
        "required_speedup_product": fraction_record(required_speedup_product),
        "root_degree": len(pairs),
        "met": minimum_effect,
    }
    exact_duration_relative_mads = {
        role: exact_relative_mad([Fraction(value) for value in values])
        for role, values in durations.items()
    }
    variability = {
        role: {
            "cold_build_ns": values,
            "median_cold_build_ns": statistics.median(values),
            "relative_mad": float(exact_duration_relative_mads[role]),
        }
        for role, values in durations.items()
    }
    exact_paired_ratio_relative_mad = exact_relative_mad(
        [
            Fraction(
                pair["control_cold_build_ns"],
                pair["treatment_cold_build_ns"],
            )
            for pair in pairs
        ]
    )
    paired_ratio_relative_mad = float(exact_paired_ratio_relative_mad)

    query_values: dict[str, dict[str, list[float | int]]] = {
        role: {
            "tps": [],
            "p50_ns": [],
            "p95_ns": [],
            "p99_ns": [],
            "recall_at_10": [],
        }
        for role in ("control", "treatment")
    }
    query_tps_fractions: dict[str, list[Fraction]] = {
        "control": [],
        "treatment": [],
    }
    query_tps_ratio_fractions: list[Fraction] = []
    query_latency_ratio_fractions: dict[str, list[Fraction]] = {
        percentile: [] for percentile in QUERY_LATENCY_PERCENTILES
    }
    query_pairs: list[dict[str, Any]] = []
    for pair in ("P1", "P2", "P3", "P4", "P5", "P6"):
        members = by_pair[pair]
        control_query = measured_query_metrics(members["control"])
        treatment_query = measured_query_metrics(members["treatment"])
        control_tps_fraction = measured_query_tps_fraction(members["control"])
        treatment_tps_fraction = measured_query_tps_fraction(members["treatment"])
        for role, values in (
            ("control", control_query),
            ("treatment", treatment_query),
        ):
            for name, value in values.items():
                query_values[role][name].append(value)
        query_tps_fractions["control"].append(control_tps_fraction)
        query_tps_fractions["treatment"].append(treatment_tps_fraction)
        tps_ratio_fraction = treatment_tps_fraction / control_tps_fraction
        query_tps_ratio_fractions.append(tps_ratio_fraction)
        tps_ratio = float(tps_ratio_fraction)
        latency_ratio_fractions = {
            percentile: Fraction(
                int(treatment_query[percentile]),
                int(control_query[percentile]),
            )
            for percentile in QUERY_LATENCY_PERCENTILES
        }
        for percentile, ratio in latency_ratio_fractions.items():
            query_latency_ratio_fractions[percentile].append(ratio)
        control_recall = float(control_query["recall_at_10"])
        treatment_recall = float(treatment_query["recall_at_10"])
        control_recall_fraction = recall_point_fraction(
            members["control"],
            d0.QUERY_K,
            d0.QUERY_EF_SEARCH,
        )
        treatment_recall_fraction = recall_point_fraction(
            members["treatment"],
            d0.QUERY_K,
            d0.QUERY_EF_SEARCH,
        )
        recall_gap_fraction = abs(
            control_recall_fraction - treatment_recall_fraction
        )
        recall_gap = float(recall_gap_fraction)
        query_pairs.append(
            {
                "pair": pair,
                "order": members["control"]["pair_order"],
                "control": control_query,
                "treatment": treatment_query,
                "treatment_over_control": {
                    "tps": tps_ratio,
                    "p50_latency": float(latency_ratio_fractions["p50_ns"]),
                    "p95_latency": float(latency_ratio_fractions["p95_ns"]),
                    "p99_latency": float(latency_ratio_fractions["p99_ns"]),
                },
                "absolute_recall_gap": recall_gap,
                "absolute_recall_gap_exact": fraction_record(recall_gap_fraction),
                "tps_nonregression": ratio_at_least(
                    treatment_tps_fraction,
                    control_tps_fraction,
                    *MINIMUM_QUERY_TPS_FRACTION,
                ),
                "p50_nonregression": ratio_at_most(
                    treatment_query["p50_ns"],
                    control_query["p50_ns"],
                    *MAXIMUM_QUERY_LATENCY_FRACTION,
                ),
                "p95_nonregression": ratio_at_most(
                    treatment_query["p95_ns"],
                    control_query["p95_ns"],
                    *MAXIMUM_QUERY_LATENCY_FRACTION,
                ),
                "p99_nonregression": ratio_at_most(
                    treatment_query["p99_ns"],
                    control_query["p99_ns"],
                    *MAXIMUM_QUERY_LATENCY_FRACTION,
                ),
                "recall_floor_met": (
                    ratio_at_least(
                        control_recall_fraction,
                        1,
                        *MINIMUM_QUERY_RECALL_FRACTION,
                    )
                    and ratio_at_least(
                        treatment_recall_fraction,
                        1,
                        *MINIMUM_QUERY_RECALL_FRACTION,
                    )
                ),
                "recall_gap_met": absolute_difference_at_most(
                    control_recall_fraction,
                    treatment_recall_fraction,
                    *MAXIMUM_QUERY_RECALL_GAP_FRACTION,
                ),
            }
        )
    exact_query_role_tps_relative_mads = {
        role: exact_relative_mad(values)
        for role, values in query_tps_fractions.items()
    }
    exact_query_role_latency_relative_mads = {
        percentile: {
            role: exact_relative_mad(
                [Fraction(int(value)) for value in query_values[role][percentile]]
            )
            for role in ("control", "treatment")
        }
        for percentile in QUERY_LATENCY_PERCENTILES
    }
    query_roles: dict[str, dict[str, Any]] = {}
    for role, values in query_values.items():
        query_roles[role] = {
            **values,
            "median_tps": statistics.median(values["tps"]),
            "median_p50_ns": statistics.median(values["p50_ns"]),
            "median_p95_ns": statistics.median(values["p95_ns"]),
            "median_p99_ns": statistics.median(values["p99_ns"]),
            "median_recall_at_10": statistics.median(values["recall_at_10"]),
            "tps_relative_mad": float(
                exact_query_role_tps_relative_mads[role]
            ),
        }
    exact_query_paired_tps_ratio_relative_mad = exact_relative_mad(
        query_tps_ratio_fractions
    )
    query_paired_tps_ratio_relative_mad = float(
        exact_query_paired_tps_ratio_relative_mad
    )
    exact_query_paired_latency_ratio_relative_mads = {
        percentile: exact_relative_mad(ratios)
        for percentile, ratios in query_latency_ratio_fractions.items()
    }
    query_latency_variability = {
        percentile: {
            "role_relative_mad": {
                role: float(relative_mad)
                for role, relative_mad in (
                    exact_query_role_latency_relative_mads[percentile].items()
                )
            },
            "paired_treatment_over_control_ratio_relative_mad": float(
                exact_query_paired_latency_ratio_relative_mads[percentile]
            ),
        }
        for percentile in QUERY_LATENCY_PERCENTILES
    }
    query_tps_variability_accepted = all(
        exact_relative_mad_at_most(
            relative_mad,
            *MAXIMUM_RELATIVE_MAD_FRACTION,
        )
        for relative_mad in exact_query_role_tps_relative_mads.values()
    )
    query_ratio_variability_accepted = exact_relative_mad_at_most(
        exact_query_paired_tps_ratio_relative_mad,
        *MAXIMUM_RELATIVE_MAD_FRACTION,
    )
    query_role_latency_variability_accepted = {
        percentile: all(
            exact_relative_mad_at_most(
                relative_mad,
                *MAXIMUM_RELATIVE_MAD_FRACTION,
            )
            for relative_mad in role_relative_mads.values()
        )
        for percentile, role_relative_mads in (
            exact_query_role_latency_relative_mads.items()
        )
    }
    query_paired_latency_ratio_variability_accepted = {
        percentile: exact_relative_mad_at_most(
            relative_mad,
            *MAXIMUM_RELATIVE_MAD_FRACTION,
        )
        for percentile, relative_mad in (
            exact_query_paired_latency_ratio_relative_mads.items()
        )
    }
    query_tps_nonregression = all(
        pair["tps_nonregression"] for pair in query_pairs
    )
    query_p50_nonregression = all(
        pair["p50_nonregression"] for pair in query_pairs
    )
    query_p95_nonregression = all(
        pair["p95_nonregression"] for pair in query_pairs
    )
    query_p99_nonregression = all(
        pair["p99_nonregression"] for pair in query_pairs
    )
    query_recall_floor_met = all(
        pair["recall_floor_met"] for pair in query_pairs
    )
    query_recall_gap_met = all(pair["recall_gap_met"] for pair in query_pairs)

    recall_points: list[dict[str, Any]] = []
    for k, ef in d0.RECALL_POINTS:
        paired: list[dict[str, Any]] = []
        controls: list[float] = []
        treatments: list[float] = []
        deficit_fractions: list[Fraction] = []
        for pair in ("P1", "P2", "P3", "P4", "P5", "P6"):
            members = by_pair[pair]
            control_fraction = recall_point_fraction(
                members["control"], k, ef
            )
            treatment_fraction = recall_point_fraction(
                members["treatment"], k, ef
            )
            control_recall = float(control_fraction)
            treatment_recall = float(treatment_fraction)
            controls.append(control_recall)
            treatments.append(treatment_recall)
            deficit_fraction = control_fraction - treatment_fraction
            deficit_fractions.append(deficit_fraction)
            paired.append(
                {
                    "pair": pair,
                    "control": control_recall,
                    "treatment": treatment_recall,
                    "treatment_deficit": float(deficit_fraction),
                    "treatment_deficit_exact": fraction_record(deficit_fraction),
                    "within_0_005": difference_at_most(
                        control_fraction,
                        treatment_fraction,
                        *MAXIMUM_RECALL_DEFICIT_FRACTION,
                    ),
                }
            )
        recall_points.append(
            {
                "k": k,
                "ef": ef,
                "control_median": statistics.median(controls),
                "treatment_median": statistics.median(treatments),
                "maximum_treatment_deficit": max(
                    0.0, max(point["treatment_deficit"] for point in paired)
                ),
                "maximum_treatment_deficit_exact": fraction_record(
                    max(Fraction(0), max(deficit_fractions))
                ),
                "all_pairs_within_0_005": all(
                    point["within_0_005"] for point in paired
                ),
                "paired": paired,
            }
        )

    all_runs_valid = all(record["status"] == "pass" for record in records)
    graph_valid = all(
        record["audit"]["graph"]["valid"]
        and record["audit"]["graph"]["vertex_count"] == d0.VECTORS
        and record["audit"]["graph"]["reachable_count"] == d0.VECTORS
        for record in records
    )
    correctness = {
        str(record["role"]): record
        for record in records
        if record["phase"] == "correctness"
    }
    require(
        set(correctness) == {"control", "treatment"},
        "Correctness pair is incomplete",
    )
    single_thread_semantic_payload_exact_match = (
        correctness_semantic_payload(correctness["control"])
        == correctness_semantic_payload(correctness["treatment"])
    )
    recall_nonregression = all(
        point["all_pairs_within_0_005"] for point in recall_points
    )
    every_pair_faster = all(
        pair["control_cold_build_ns"] > pair["treatment_cold_build_ns"]
        for pair in pairs
    )
    strata_faster = all(
        math.prod(
            pair["control_cold_build_ns"]
            for pair in pairs
            if pair["order"] == order
        )
        > math.prod(
            pair["treatment_cold_build_ns"]
            for pair in pairs
            if pair["order"] == order
        )
        for order in strata
    )
    variability_accepted = all(
        exact_relative_mad_at_most(
            relative_mad,
            *MAXIMUM_RELATIVE_MAD_FRACTION,
        )
        for relative_mad in exact_duration_relative_mads.values()
    )
    ratio_variability_accepted = exact_relative_mad_at_most(
        exact_paired_ratio_relative_mad,
        *MAXIMUM_RELATIVE_MAD_FRACTION,
    )
    idle_protocol_met = all(
        d0.idle_record_passes(
            record["idle"],
            expected_minimum_idle_percent=idle_minimum_percent,
        )
        for record in records
    )
    idle_to_launch_met = all(
        isinstance(record.get("idle_to_launch_ns"), int)
        and 0 <= record["idle_to_launch_ns"] <= d0.IDLE_TO_LAUNCH_MAX_NS
        for record in records
    )
    host_resource_protocol_met = all(
        record.get("process_swaps") == 0
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
    accepted = all(
        (
            all_runs_valid,
            graph_valid,
            single_thread_semantic_payload_exact_match,
            recall_nonregression,
            every_pair_faster,
            strata_faster,
            minimum_effect,
            variability_accepted,
            ratio_variability_accepted,
            query_tps_variability_accepted,
            query_ratio_variability_accepted,
            all(query_role_latency_variability_accepted.values()),
            all(query_paired_latency_ratio_variability_accepted.values()),
            query_tps_nonregression,
            query_p50_nonregression,
            query_p95_nonregression,
            query_p99_nonregression,
            query_recall_floor_met,
            query_recall_gap_met,
            idle_protocol_met,
            idle_to_launch_met,
            host_resource_protocol_met,
            dataset_integrity_met,
            measured_run_execution_witnesses_valid,
        )
    )
    swap_growth = [int(record["global_swap_growth_bytes"]) for record in records]
    return {
        "status": "pass" if accepted else "fail",
        "experiment": experiment.record(),
        "expected_witnesses": expected_witnesses_for_experiment(experiment),
        "scope": experiment.scope,
        "performance_measurement_scope": PERFORMANCE_MEASUREMENT_SCOPE,
        "claim_eligible": False,
        "decision": (
            f"accept-{experiment.name}" if accepted else f"reject-{experiment.name}"
        ),
        "workload": {
            "vectors": d0.VECTORS,
            "dimensions": d0.DIMENSIONS,
            "M": d0.M,
            "ef_construction": d0.EF_CONSTRUCTION,
            "ef_search": d0.EF_SEARCH,
            "participants": d0.MEASURED_PARTICIPANTS,
            "data_fnv1a64": dataset["fnv1a64"],
            "data_sha256": dataset["sha256"],
            "data_bytes": dataset["bytes"],
            "schedule_sha256": schedule_sha256,
        },
        "performance": {
            "minimum_speedup": MINIMUM_SPEEDUP,
            "pairs": pairs,
            "strata": strata,
            "stratified_geometric_mean_speedup": estimate,
            "stratified_geometric_mean_speedup_is_approximate": True,
            "minimum_speedup_comparison_exact": minimum_speedup_comparison_exact,
            "speedup_range": [
                min(pair["treatment_speedup"] for pair in pairs),
                max(pair["treatment_speedup"] for pair in pairs),
            ],
            "variability": variability,
            "paired_ratio_relative_mad": paired_ratio_relative_mad,
        },
        "query_performance": {
            "maximum_relative_mad": MAXIMUM_RELATIVE_MAD,
            "minimum_tps_ratio": MINIMUM_QUERY_TPS_RATIO,
            "maximum_p50_p95_p99_latency_ratio": MAXIMUM_QUERY_LATENCY_RATIO,
            "minimum_recall": MINIMUM_QUERY_RECALL,
            "maximum_absolute_recall_gap": MAXIMUM_QUERY_RECALL_GAP,
            "pairs": query_pairs,
            "roles": query_roles,
            "paired_tps_ratio_relative_mad": (
                query_paired_tps_ratio_relative_mad
            ),
            "latency_variability": query_latency_variability,
            "role_tps_relative_mad_at_most_0_10": (
                query_tps_variability_accepted
            ),
            "paired_tps_ratio_relative_mad_at_most_0_10": (
                query_ratio_variability_accepted
            ),
            "all_pairs_tps_at_least_0_95x": query_tps_nonregression,
            "all_pairs_p50_at_most_1_05x": query_p50_nonregression,
            "all_pairs_p95_at_most_1_05x": query_p95_nonregression,
            "all_pairs_p99_at_most_1_05x": query_p99_nonregression,
            "all_pairs_recall_at_least_0_99": query_recall_floor_met,
            "all_pairs_recall_gap_at_most_0_005": query_recall_gap_met,
        },
        "quality": {
            "graph_all_valid_and_reachable": graph_valid,
            "single_thread_semantic_payload_exact_match": (
                single_thread_semantic_payload_exact_match
            ),
            "maximum_paired_recall_deficit": MAXIMUM_RECALL_DEFICIT,
            "recall_all_pairs_within_deficit": recall_nonregression,
            "recall_points": recall_points,
        },
        "resources": {
            "global_swap_policy": d0.GLOBAL_SWAP_POLICY,
            "maximum_observed_global_swap_growth_bytes": max(0, max(swap_growth)),
            "indexing_window_peak_rss_established": False,
        },
        "acceptance": {
            "all_runs_valid": all_runs_valid,
            "exact_balanced_schedule_met": True,
            "measured_run_execution_witnesses_valid": (
                measured_run_execution_witnesses_valid
            ),
            "all_graphs_valid_and_reachable": graph_valid,
            "single_thread_semantic_payload_exact_match": (
                single_thread_semantic_payload_exact_match
            ),
            "paired_recall_deficit_at_most_0_005": recall_nonregression,
            "every_measured_pair_faster": every_pair_faster,
            "both_order_strata_faster": strata_faster,
            "stratified_geometric_mean_at_least_1_05": minimum_effect,
            "relative_mad_at_most_0_10": variability_accepted,
            "paired_ratio_relative_mad_at_most_0_10": ratio_variability_accepted,
            "query_tps_relative_mad_at_most_0_10": (
                query_tps_variability_accepted
            ),
            "query_paired_tps_ratio_relative_mad_at_most_0_10": (
                query_ratio_variability_accepted
            ),
            "query_p50_role_latency_relative_mad_at_most_0_10": (
                query_role_latency_variability_accepted["p50_ns"]
            ),
            "query_p95_role_latency_relative_mad_at_most_0_10": (
                query_role_latency_variability_accepted["p95_ns"]
            ),
            "query_p99_role_latency_relative_mad_at_most_0_10": (
                query_role_latency_variability_accepted["p99_ns"]
            ),
            "query_p50_paired_latency_ratio_relative_mad_at_most_0_10": (
                query_paired_latency_ratio_variability_accepted["p50_ns"]
            ),
            "query_p95_paired_latency_ratio_relative_mad_at_most_0_10": (
                query_paired_latency_ratio_variability_accepted["p95_ns"]
            ),
            "query_p99_paired_latency_ratio_relative_mad_at_most_0_10": (
                query_paired_latency_ratio_variability_accepted["p99_ns"]
            ),
            "query_tps_at_least_0_95x": query_tps_nonregression,
            "query_p50_latency_at_most_1_05x": query_p50_nonregression,
            "query_p95_latency_at_most_1_05x": query_p95_nonregression,
            "query_p99_latency_at_most_1_05x": query_p99_nonregression,
            "query_recall_both_at_least_0_99": query_recall_floor_met,
            "query_recall_gap_at_most_0_005": query_recall_gap_met,
            "idle_protocol_met": idle_protocol_met,
            "idle_to_launch_at_most_1_second": idle_to_launch_met,
            "host_resource_protocol_met": host_resource_protocol_met,
            "dataset_integrity_met": dataset_integrity_met,
            "outliers_removed": False,
        },
        "limitations": [
            "This is a development-only Infinity control/treatment causal result.",
            "It is not an Infinity-versus-FAISS result and is not claim-eligible.",
            "Performance applies only to authenticated evidence-instrumented builds; evidence-disabled release binaries require a separate final benchmark.",
            "The workload is deterministic synthetic float32 data, not a canonical dataset.",
            "Whole-process RSS includes post-timing audits; indexing-window peak RSS is unestablished.",
            "No measured member is removed as an outlier.",
        ],
    }


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# Apple HNSW Batch4 {summary['experiment']['name']} Causal Checkpoint",
        "",
        f"Status: `{summary['status']}`; decision: `{summary['decision']}`.",
        "",
        "This is a development-only Infinity control/treatment result, not a FAISS claim.",
        "Performance is scoped to authenticated evidence-instrumented builds.",
        "",
        "| Pair | Order | Control ns | Treatment ns | Speedup |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in summary["performance"]["pairs"]:
        lines.append(
            f"| {pair['pair']} | {pair['order']} | "
            f"{pair['control_cold_build_ns']} | "
            f"{pair['treatment_cold_build_ns']} | "
            f"{pair['treatment_speedup']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "Stratified geometric-mean speedup: "
            f"`{summary['performance']['stratified_geometric_mean_speedup']:.4f}x` "
            "(approximate report; acceptance uses the serialized exact product).",
            "",
            "Exact minimum-speedup gate: "
            f"`{summary['performance']['minimum_speedup_comparison_exact']['met']}`.",
            "",
            "Every measured pair faster: "
            f"`{summary['acceptance']['every_measured_pair_faster']}`.",
            "",
            "All paired held-out recall deficits at most 0.005: "
            f"`{summary['quality']['recall_all_pairs_within_deficit']}`.",
            "",
            "## Query Performance",
            "",
            "| Pair | Control TPS | Treatment TPS | TPS ratio | "
            "p50 ratio | p95 ratio | p99 ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in summary["query_performance"]["pairs"]:
        ratios = pair["treatment_over_control"]
        lines.append(
            f"| {pair['pair']} | {pair['control']['tps']:.2f} | "
            f"{pair['treatment']['tps']:.2f} | {ratios['tps']:.4f}x | "
            f"{ratios['p50_latency']:.4f}x | {ratios['p95_latency']:.4f}x | "
            f"{ratios['p99_latency']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "Query TPS nonregression: "
            f"`{summary['query_performance']['all_pairs_tps_at_least_0_95x']}`; "
            "p50/p95/p99 latency nonregression: "
            f"`{summary['query_performance']['all_pairs_p50_at_most_1_05x']}`/"
            f"`{summary['query_performance']['all_pairs_p95_at_most_1_05x']}`/"
            f"`{summary['query_performance']['all_pairs_p99_at_most_1_05x']}`.",
            "",
            "Control TPS relative MAD: "
            f"`{summary['query_performance']['roles']['control']['tps_relative_mad']:.4%}`.",
            "",
            "Treatment TPS relative MAD: "
            f"`{summary['query_performance']['roles']['treatment']['tps_relative_mad']:.4%}`.",
            "",
            "Paired TPS-ratio relative MAD: "
            f"`{summary['query_performance']['paired_tps_ratio_relative_mad']:.4%}`.",
            "",
            "Query variability gates: per-role "
            f"`{summary['query_performance']['role_tps_relative_mad_at_most_0_10']}`; "
            "paired-ratio "
            f"`{summary['query_performance']['paired_tps_ratio_relative_mad_at_most_0_10']}`.",
            "",
        ]
    )
    for percentile in QUERY_LATENCY_PERCENTILES:
        label = percentile.removesuffix("_ns")
        variability = summary["query_performance"]["latency_variability"][
            percentile
        ]
        role_relative_mad = variability["role_relative_mad"]
        lines.extend(
            [
                f"{label} latency relative MAD: control "
                f"`{role_relative_mad['control']:.4%}`; treatment "
                f"`{role_relative_mad['treatment']:.4%}`; paired "
                "treatment/control ratio "
                f"`{variability['paired_treatment_over_control_ratio_relative_mad']:.4%}`.",
                "",
                f"{label} latency variability gates: per-role "
                f"`{summary['acceptance'][f'query_{label}_role_latency_relative_mad_at_most_0_10']}`; "
                "paired-ratio "
                f"`{summary['acceptance'][f'query_{label}_paired_latency_ratio_relative_mad_at_most_0_10']}`.",
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    return "\n".join(lines) + "\n"


def validate_completed_campaign(summary: dict[str, Any]) -> None:
    require(
        summary.get("status") in ("pass", "fail"),
        "Completed campaign has an invalid summary status",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--control-binary", type=Path, required=True)
    parser.add_argument("--treatment-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-idle-minimum-percent",
        dest="idle_minimum_percent",
        type=float,
        default=d0.IDLE_MINIMUM_PERCENT,
    )
    parser.add_argument("--idle-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--member-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    for option in (
        "--repo",
        "--experiment",
        "--control-binary",
        "--treatment-binary",
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
    args.experiment_spec = EXPERIMENTS[args.experiment]
    args.repo = args.repo.resolve()
    for name in ("control_binary", "treatment_binary"):
        binary = getattr(args, name)
        if not binary.is_absolute():
            binary = args.repo / binary
        binary = binary.resolve()
        require(binary.is_file(), f"Binary not found: {binary}")
        setattr(args, name, binary)
    args.output_dir = args.output_dir.resolve()
    require(args.control_binary != args.treatment_binary, "Role binary paths must differ")
    d0.validate_idle_minimum_percent(args.idle_minimum_percent)
    require(
        args.idle_timeout_seconds >= d0.IDLE_WINDOW_SECONDS,
        f"Idle timeout must be at least {d0.IDLE_WINDOW_SECONDS} seconds",
    )
    require(args.member_timeout_seconds > 0, "Member timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    d0.reject_unsafe_parent_environment()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        dataset_path = args.output_dir / d0.DATASET_FILENAME
        dataset = d0.create_dataset(dataset_path)
        dataset_record_path = args.output_dir / "dataset.json"
        d0.write_json(dataset_record_path, dataset)
        truth = d0.heldout_truth(dataset_path, dataset)
        heldout = d0.heldout_manifest(truth)
        heldout_record_path = args.output_dir / d0.HELDOUT_MANIFEST_FILENAME
        d0.write_json(heldout_record_path, heldout)
        all_schedule = schedule()
        schedule_path = args.output_dir / "schedule.json"
        d0.write_json(schedule_path, all_schedule)
        preflight_record = preflight(
            args,
            args.output_dir,
            dataset=dataset,
            dataset_record_path=dataset_record_path,
            heldout=heldout,
            heldout_record_path=heldout_record_path,
            schedule_path=schedule_path,
        )
        d0.write_json(args.output_dir / "preflight.json", preflight_record)
        if args.preflight_only:
            result = {
                "causal_schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment": args.experiment_spec.record(),
                "scope": args.experiment_spec.scope,
                "execution_mode": PREFLIGHT_ONLY_MODE,
                "status": "pass",
                "completed_at": d0.utc_now(),
                "benchmark_members_executed": 0,
                "campaign_complete": False,
                "artifact_sha256": {
                    "dataset.json": d0.sha256(dataset_record_path),
                    d0.HELDOUT_MANIFEST_FILENAME: d0.sha256(
                        heldout_record_path
                    ),
                    "preflight.json": d0.sha256(
                        args.output_dir / "preflight.json"
                    ),
                    "schedule.json": d0.sha256(schedule_path),
                },
            }
            d0.write_json(args.output_dir / "preflight-result.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        binary_hashes = {
            role: preflight_record["roles"][role]["binary"]["sha256"]
            for role in ("control", "treatment")
        }
        campaign_binding = preflight_record["campaign_binding"]
        runtime_hashes = preflight_record["runtime_artifact_hashes"]
        records: list[dict[str, Any]] = []
        for entry in all_schedule:
            records.append(
                run_member(
                    entry,
                    args=args,
                    output_dir=args.output_dir,
                    binary_hashes=binary_hashes,
                    campaign_binding=campaign_binding,
                    runtime_artifact_hashes=runtime_hashes,
                    dataset_path=dataset_path,
                    dataset=dataset,
                    truth=truth,
                )
            )
            d0.write_json(args.output_dir / "runs.json", records)
        summary = summarize(
            records,
            experiment=args.experiment_spec,
            dataset=dataset,
            schedule_sha256=d0.sha256(schedule_path),
            idle_minimum_percent=args.idle_minimum_percent,
        )
        d0.write_json(args.output_dir / "summary.json", summary)
        (args.output_dir / "README.md").write_text(
            render_summary(summary),
            encoding="utf-8",
        )
        # A rejected hypothesis is still a complete campaign. Only operational
        # failures belong in failure.json.
        validate_completed_campaign(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException as error:
        d0.write_json(
            args.output_dir / "failure.json",
            {
                "status": "fail",
                "experiment": args.experiment_spec.record(),
                "scope": args.experiment_spec.scope,
                "captured_at": d0.utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        d0.write_manifest(args.output_dir)


if __name__ == "__main__":
    main()
