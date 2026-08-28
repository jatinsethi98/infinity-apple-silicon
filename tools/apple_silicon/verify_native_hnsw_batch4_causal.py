#!/usr/bin/env python3
"""Independently verify Apple HNSW Batch4 causal evidence."""

from __future__ import annotations

if __name__ == "__main__":
    try:
        _injected_helper = _AUTHENTICATED_NATIVE_HNSW_D0_HELPER
        _injected_exactness_helper = (
            _AUTHENTICATED_NATIVE_HNSW_EXACTNESS_HELPER
        )
        _injected_context = _AUTHENTICATED_NATIVE_HNSW_BATCH4_CONTEXT
    except NameError as error:
        raise RuntimeError(
            "Direct execution is disabled; use the authenticated Batch4 bootstrap"
        ) from error
else:
    _injected_helper = None
    _injected_exactness_helper = None
    _injected_context = None

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

if _injected_helper is not None:
    v = _injected_helper
    if not isinstance(v, types.ModuleType):
        raise RuntimeError("Authenticated Batch4 helper injection is invalid")
    exactness_v = _injected_exactness_helper
    if not isinstance(exactness_v, types.ModuleType):
        raise RuntimeError(
            "Authenticated Batch4 exactness-helper injection is invalid"
        )
    del _AUTHENTICATED_NATIVE_HNSW_D0_HELPER
    del _AUTHENTICATED_NATIVE_HNSW_EXACTNESS_HELPER
    del _AUTHENTICATED_NATIVE_HNSW_BATCH4_CONTEXT
else:
    from tools.apple_silicon import verify_native_hnsw_d0 as v
    from tools.apple_silicon import verify_native_hnsw_exactness as exactness_v
del _injected_helper
del _injected_exactness_helper

CAUSAL_SCHEMA_VERSION = 16
REQUIRED_EXACTNESS_SCHEMA_VERSION = 3
ARCHIVED_CAUSAL_SCHEMA_VERSION = 15
# Archived schema 15 is self-authenticating only with its captured verifier.
ARCHIVED_SCHEMA_15_VERIFIER_POLICY = "use-manifest-captured-verifier"
FULL_CAMPAIGN_MODE = "full-campaign"
PREFLIGHT_ONLY_MODE = "preflight-only"
PERFORMANCE_MEASUREMENT_SCOPE = "authenticated-evidence-instrumented-builds"
RUNNER_CAPTURE_FILENAME = "runner-native_hnsw_batch4_causal.py"
VERIFIER_CAPTURE_FILENAME = "verifier-native_hnsw_batch4_causal.py"
MINIMUM_SPEEDUP = 1.05
MAXIMUM_RECALL_DEFICIT = 0.005
MAXIMUM_RELATIVE_MAD = 0.10
QUERY_LATENCY_PERCENTILES = ("p50_ns", "p95_ns", "p99_ns")
MINIMUM_QUERY_TPS_RATIO = 0.95
MAXIMUM_QUERY_LATENCY_RATIO = 1.05
MINIMUM_QUERY_RECALL = 0.99
MAXIMUM_QUERY_RECALL_GAP = 0.005
MINIMUM_SPEEDUP_FRACTION = (105, 100)
MAXIMUM_RELATIVE_MAD_FRACTION = (1, 10)
MINIMUM_QUERY_TPS_FRACTION = (95, 100)
MAXIMUM_QUERY_LATENCY_FRACTION = (105, 100)
MINIMUM_QUERY_RECALL_FRACTION = (99, 100)
MAXIMUM_RECALL_DEFICIT_FRACTION = (5, 1000)
MAXIMUM_QUERY_RECALL_GAP_FRACTION = (5, 1000)
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
STAT_MANIFEST_KEYS = (
    "device",
    "inode",
    "mode",
    "links",
    "bytes",
    "mtime_ns",
    "ctime_ns",
)
LOCAL_PYTHON_MODULES = (
    (
        "native_hnsw_d0",
        "module-native_hnsw_d0.py",
        "tools/apple_silicon/native_hnsw_d0.py",
    ),
    (
        "verify_native_hnsw_d0",
        "module-verify_native_hnsw_d0.py",
        "tools/apple_silicon/verify_native_hnsw_d0.py",
    ),
    (
        "native_hnsw_exactness",
        "module-native_hnsw_exactness.py",
        "tools/apple_silicon/native_hnsw_exactness.py",
    ),
    (
        "verify_native_hnsw_exactness",
        "module-verify_native_hnsw_exactness.py",
        "tools/apple_silicon/verify_native_hnsw_exactness.py",
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
BUILD_TOOL_CACHE_KEYS = {
    record_name: cache_name
    for record_name, cache_name, _ in BUILD_TOOL_SPECS
}

PAIR_PLAN = (
    ("C1", "correctness", 1, ("control", "treatment")),
    ("W1", "warmup", v.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("W2", "warmup", v.MEASURED_PARTICIPANTS, ("treatment", "control")),
    ("P1", "measured", v.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("P2", "measured", v.MEASURED_PARTICIPANTS, ("treatment", "control")),
    ("P3", "measured", v.MEASURED_PARTICIPANTS, ("treatment", "control")),
    ("P4", "measured", v.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("P5", "measured", v.MEASURED_PARTICIPANTS, ("control", "treatment")),
    ("P6", "measured", v.MEASURED_PARTICIPANTS, ("treatment", "control")),
)
CAUSAL_ROLE_IDS = {
    "control": 1,
    "treatment": 2,
}

RUN_RECORD_FIELDS = frozenset(
    {
        "sequence",
        "pair",
        "phase",
        "pair_order",
        "role",
        "engine",
        "participants",
        "status",
        "experiment",
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


class CausalVerificationError(RuntimeError):
    """Raised when causal evidence fails independent reconstruction."""


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
COMPACT_VERTEX_LOCKS_SWITCH = (
    "INFINITY_ENABLE_APPLE_HNSW_COMPACT_VERTEX_LOCKS"
)
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
if WITNESS_FIELDS != tuple(v.EXECUTION_WITNESS_FIELDS):
    raise RuntimeError("Causal verifier and D0 execution-witness fields differ")
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
        raise CausalVerificationError(message)


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
    witnesses = v.expect_mapping(value, context)
    v.expect_exact_keys(
        witnesses,
        {"control", "treatment"},
        context,
    )
    normalized: dict[str, dict[str, int]] = {}
    for role in ("control", "treatment"):
        witness = v.expect_mapping(witnesses[role], f"{context} {role}")
        require(
            set(witness) == set(WITNESS_FIELDS),
            f"{context} {role} fields differ",
        )
        normalized_witness: dict[str, int] = {}
        for name in WITNESS_FIELDS:
            bit = v.expect_int(
                witness[name],
                f"{context} {role} {name}",
                minimum=0,
            )
            require(bit <= 1, f"{context} {role} {name} is not binary")
            normalized_witness[name] = bit
        normalized[role] = normalized_witness
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
    witness = v.expect_mapping(value, context)
    v.expect_exact_keys(witness, set(WITNESS_FIELDS), context)
    normalized: dict[str, int] = {}
    for name in WITNESS_FIELDS:
        bit = v.expect_int(witness[name], f"{context} {name}", minimum=0)
        require(bit <= 1, f"{context} {name} is not binary")
        normalized[name] = bit
    require(
        normalized == expected_witnesses_for_experiment(experiment)[role],
        f"{context} differs from the role matrix",
    )
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


def json_value_equal_exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            json_value_equal_exact(actual[key], expected[key])
            for key in actual
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            json_value_equal_exact(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def authenticated_context() -> dict[str, Any] | None:
    if _injected_context is None:
        return None
    require(
        isinstance(_injected_context, Mapping),
        "Authenticated Batch4 context is invalid",
    )
    context = dict(_injected_context)
    v.expect_exact_keys(
        context,
        {
            "evidence_directory",
            "directory_descriptor",
            "directory_identity",
            "preflight_only",
            "verifier",
            "helper",
            "exactness_helper",
        },
        "authenticated Batch4 context",
    )
    evidence_directory = v.expect_string(
        context["evidence_directory"],
        "authenticated evidence directory",
    )
    require(
        Path(evidence_directory).is_absolute(),
        "Authenticated evidence directory is not absolute",
    )
    require(
        type(context["preflight_only"]) is bool,
        "Authenticated preflight-only mode is not boolean",
    )
    descriptor = v.expect_int(
        context["directory_descriptor"],
        "authenticated evidence directory descriptor",
        minimum=0,
    )
    identity = context["directory_identity"]
    require(
        isinstance(identity, tuple) and len(identity) == 9,
        "Authenticated evidence directory identity is invalid",
    )
    for index, value in enumerate(identity):
        v.expect_int(
            value,
            f"authenticated evidence directory identity field {index}",
            minimum=0,
        )
    try:
        descriptor_stat = os.fstat(descriptor)
        actual_identity = (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
            descriptor_stat.st_mode,
            descriptor_stat.st_nlink,
            descriptor_stat.st_uid,
            descriptor_stat.st_gid,
            descriptor_stat.st_size,
            descriptor_stat.st_mtime_ns,
            descriptor_stat.st_ctime_ns,
        )
    except OSError as error:
        raise CausalVerificationError(
            f"Authenticated evidence directory descriptor is invalid: {error}"
        ) from error
    require(
        actual_identity == identity,
        "Authenticated evidence directory descriptor identity differs",
    )
    for name, expected_filename in (
        ("verifier", VERIFIER_CAPTURE_FILENAME),
        ("helper", "module-verify_native_hnsw_d0.py"),
        (
            "exactness_helper",
            "module-verify_native_hnsw_exactness.py",
        ),
    ):
        source = v.expect_mapping(
            context[name],
            f"authenticated {name}",
        )
        v.expect_exact_keys(
            source,
            {"filename", "sha256", "bytes"},
            f"authenticated {name}",
        )
        require(
            source["filename"] == expected_filename,
            f"Authenticated {name} filename differs",
        )
        v.expect_sha256(source["sha256"], f"authenticated {name} SHA-256")
        v.expect_int(
            source["bytes"],
            f"authenticated {name} byte count",
            minimum=1,
        )
    return context


def validate_campaign_binding(
    value: Any,
    *,
    dataset_sha256: str,
    schedule_sha256: str,
    roles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    campaign = v.expect_mapping(value, "campaign binding")
    v.expect_exact_keys(
        campaign,
        {
            "schema_version",
            "campaign_nonce",
            "dataset_sha256",
            "schedule_sha256",
            "roles",
        },
        "campaign binding",
    )
    require(
        v.expect_int(
            campaign["schema_version"],
            "campaign binding schema version",
            minimum=1,
        )
        == CAMPAIGN_BINDING_SCHEMA_VERSION,
        "Campaign binding schema differs",
    )
    nonce = v.expect_sha256(
        campaign["campaign_nonce"],
        "campaign binding nonce",
    )
    bound_dataset = v.expect_sha256(
        campaign["dataset_sha256"],
        "campaign binding dataset SHA-256",
    )
    bound_schedule = v.expect_sha256(
        campaign["schedule_sha256"],
        "campaign binding schedule SHA-256",
    )
    require(
        bound_dataset == dataset_sha256,
        "Campaign binding dataset differs from dataset.json",
    )
    require(
        bound_schedule == schedule_sha256,
        "Campaign binding schedule differs from schedule.json",
    )

    role_values = v.expect_mapping(
        campaign["roles"],
        "campaign binding roles",
    )
    v.expect_exact_keys(
        role_values,
        set(CAUSAL_ROLE_IDS),
        "campaign binding roles",
    )
    normalized_roles: dict[str, dict[str, Any]] = {}
    for role, expected_role_id in CAUSAL_ROLE_IDS.items():
        context = f"campaign binding {role}"
        record = v.expect_mapping(role_values[role], context)
        v.expect_exact_keys(
            record,
            {"role_id", "engine", "binary_sha256"},
            context,
        )
        require(
            v.expect_int(record["role_id"], f"{context} role ID", minimum=1)
            == expected_role_id,
            f"Campaign binding {role} role ID differs",
        )
        require(
            v.expect_string(record["engine"], f"{context} engine")
            == "infinity",
            f"Campaign binding {role} engine differs",
        )
        binary_sha256 = v.expect_sha256(
            record["binary_sha256"],
            f"{context} binary SHA-256",
        )
        require(
            binary_sha256 == roles[role]["binary"]["sha256"],
            f"Campaign binding {role} binary differs from preflight",
        )
        normalized_roles[role] = {
            "role_id": expected_role_id,
            "engine": "infinity",
            "binary_sha256": binary_sha256,
        }
    return {
        "schema_version": CAMPAIGN_BINDING_SCHEMA_VERSION,
        "campaign_nonce": nonce,
        "dataset_sha256": bound_dataset,
        "schedule_sha256": bound_schedule,
        "roles": normalized_roles,
    }


def member_campaign_binding(
    member: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, str | int]:
    role = str(member["role"])
    require(role in CAUSAL_ROLE_IDS, f"Unknown campaign role: {role}")
    role_record = campaign["roles"][role]
    return {
        "campaign_nonce": campaign["campaign_nonce"],
        "schedule_sequence": int(member["sequence"]),
        "role_id": CAUSAL_ROLE_IDS[role],
        "binary_sha256": role_record["binary_sha256"],
        "dataset_sha256": campaign["dataset_sha256"],
    }


def canonical_repo(repo: Path, *, label: str) -> Path:
    require(repo.is_absolute(), f"{label} repository path is not absolute")
    require(
        os.path.abspath(repo) == str(repo),
        f"{label} repository path is not canonical",
    )
    return repo


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
        and os.path.abspath(home_directory) == home_directory
        and home_directory == str(repo),
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


def exact_median(values: Sequence[Fraction]) -> Fraction:
    require(bool(values), "Cannot calculate an exact median from no values")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def exact_relative_mad(values: Sequence[Fraction]) -> Fraction:
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


def exact_ratio_at_least(
    numerator: Fraction,
    denominator: Fraction,
    minimum_numerator: int,
    minimum_denominator: int,
) -> bool:
    require(
        numerator > 0 and denominator > 0,
        "Exact ratio operands must be positive",
    )
    return (
        numerator * minimum_denominator
        >= denominator * minimum_numerator
    )


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


def validate_toolchain_identity(
    control_tools: dict[str, dict[str, Any]],
    treatment_tools: dict[str, dict[str, Any]],
) -> None:
    for tool, _, _ in BUILD_TOOL_SPECS:
        require(
            control_tools[tool]["resolved_path"]
            == treatment_tools[tool]["resolved_path"]
            and control_tools[tool]["sha256"]
            == treatment_tools[tool]["sha256"],
            f"Role builds use different {tool} binaries",
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


def validate_production_build_settings(
    settings: dict[str, Any],
    *,
    context: str,
) -> None:
    for name, expected in REQUIRED_PRODUCTION_BUILD_VALUES.items():
        require(
            settings.get(name) == expected,
            f"{context} production build requires {name}={expected}",
        )
    for name in REQUIRED_NONEMPTY_BUILD_SETTINGS:
        require(
            isinstance(settings.get(name), str) and settings[name] != "",
            f"{context} production build setting {name} is missing",
        )
    scan_deps_paths = v.configured_scan_deps_paths(
        settings,
        context=context,
    )
    require(
        len(scan_deps_paths) == 1
        and all(
            settings[name] == scan_deps_paths[0]
            for name in v.CMAKE_SCAN_DEPS_CACHE_KEYS
        ),
        f"{context} production build must use one configured dependency scanner",
    )


def validate_experiment(value: Any, context: str) -> ExperimentSpec:
    record = v.expect_mapping(value, context)
    v.expect_exact_keys(
        record,
        {
            "name",
            "varying_switches",
            "held_switches",
            "scope",
        },
        context,
    )
    name = v.expect_string(record["name"], f"{context} name")
    require(name in EXPERIMENTS, f"{context} names an unsupported experiment")
    experiment = EXPERIMENTS[name]
    require(record == experiment.record(), f"{context} differs from the frozen specification")
    return experiment


def frozen_schedule() -> list[dict[str, Any]]:
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
    return entries


def validate_schedule(value: Any) -> list[dict[str, Any]]:
    entries = v.expect_list(value, "schedule.json")
    expected = frozen_schedule()
    keys = {
        "sequence",
        "pair",
        "phase",
        "pair_order",
        "role",
        "engine",
        "participants",
    }
    for index, entry_value in enumerate(entries):
        entry = v.expect_mapping(entry_value, f"schedule member {index}")
        v.expect_exact_keys(entry, keys, f"schedule member {index}")
        v.expect_int(entry["sequence"], f"schedule member {index} sequence", minimum=0)
        v.expect_int(
            entry["participants"],
            f"schedule member {index} participants",
            minimum=1,
        )
        for key in ("pair", "phase", "pair_order", "role", "engine"):
            v.expect_string(entry[key], f"schedule member {index} {key}")
    require(entries == expected, "schedule.json differs from the frozen causal order")
    return entries


def validate_capture(
    evidence_dir: Path,
    value: Any,
    *,
    context: str,
    expected_path: str | None = None,
) -> dict[str, Any]:
    record = v.expect_mapping(value, context)
    path = v.validate_capture_reference(
        evidence_dir,
        record,
        context=context,
        expected_path=expected_path,
    )
    require(
        v.expect_int(record["bytes"], f"{context} bytes", minimum=1)
        == path.stat().st_size,
        f"{context} byte count differs",
    )
    return record


def validate_source_records(
    evidence_dir: Path,
    value: Any,
    *,
    context: str,
) -> list[dict[str, Any]]:
    records = v.expect_list(value, context)
    require(records, f"{context} is empty")
    labels: set[str] = set()
    absolute_paths: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, record_value in enumerate(records):
        item_context = f"{context} item {index}"
        record = v.expect_mapping(record_value, item_context)
        v.expect_exact_keys(
            record,
            {"path", "absolute_path", "captured_path", "sha256", "bytes"},
            item_context,
        )
        absolute = v.expect_string(record["absolute_path"], f"{item_context} absolute path")
        require(
            Path(absolute).is_absolute() and os.path.abspath(absolute) == absolute,
            f"{item_context} path is not canonical absolute",
        )
        label = v.expect_string(record["path"], f"{item_context} path")
        require(label != "", f"{item_context} path label is empty")
        require(label not in labels, f"{context} duplicates source label {label!r}")
        require(
            absolute not in absolute_paths,
            f"{context} duplicates source path {absolute!r}",
        )
        labels.add(label)
        absolute_paths.add(absolute)
        captured = v.validate_capture_reference(
            evidence_dir,
            record,
            context=item_context,
        )
        size = v.expect_int(record["bytes"], f"{item_context} bytes", minimum=0)
        require(captured.stat().st_size == size, f"{item_context} size differs")
        validated.append(record)
    require(
        [str(record["absolute_path"]) for record in validated]
        == sorted(absolute_paths, key=lambda path: path.encode("utf-8")),
        f"{context} is not in canonical absolute-path order",
    )
    return validated


def source_identity(build: dict[str, Any]) -> list[tuple[str, str, int]]:
    build_directory = Path(
        v.expect_string(build["build_directory"], "build directory")
    )
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


def normalized_dependency_identity(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
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


def source_path_from_compile_entry(entry: Any, context: str) -> str:
    mapping = v.expect_mapping(entry, context)
    directory = Path(
        v.expect_string(mapping.get("directory"), f"{context} directory")
    )
    require(directory.is_absolute(), f"{context} directory is not absolute")
    source = Path(v.expect_string(mapping.get("file"), f"{context} file"))
    if not source.is_absolute():
        source = directory / source
    return v.lexical_absolute_path(
        str(source),
        working_directory=directory,
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
        v.lexical_absolute_path(
            argv[0],
            working_directory=Path(working_directory),
        )
        == runner_source_path,
        "Preflight runner argv[0] does not identify its source path",
    )
    option_names = {
        "--repo": "repo",
        "--experiment": "experiment",
        "--control-binary": "control_binary",
        "--treatment-binary": "treatment_binary",
        "--output-dir": "output_directory",
        "--development-idle-minimum-percent": "idle_minimum_percent",
        "--idle-timeout-seconds": "idle_timeout_seconds",
        "--member-timeout-seconds": "member_timeout_seconds",
    }
    parsed: dict[str, str] = {}
    preflight_only = False
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--preflight-only":
            require(
                not preflight_only,
                "Preflight runner argv duplicates --preflight-only",
            )
            preflight_only = True
            index += 1
            continue

        option, separator, joined_value = token.partition("=")
        require(
            option in option_names,
            f"Preflight runner argv contains unknown option {option!r}",
        )
        name = option_names[option]
        require(name not in parsed, f"Preflight runner argv duplicates {option}")
        if separator:
            value = joined_value
            index += 1
        else:
            require(
                index + 1 < len(argv),
                f"Preflight runner argv omits {option} value",
            )
            value = argv[index + 1]
            index += 2
        require(
            isinstance(value, str) and value and not value.startswith("--"),
            f"Preflight runner argv has an invalid {option} value",
        )
        parsed[name] = value

    for required in (
        "experiment",
        "control_binary",
        "treatment_binary",
        "output_directory",
    ):
        require(required in parsed, f"Preflight runner argv omits {required}")
    require(
        parsed["experiment"] in EXPERIMENTS,
        "Preflight runner argv experiment is invalid",
    )

    resolved_repo = v.lexical_absolute_path(
        parsed.get("repo", working_directory),
        working_directory=Path(working_directory),
    )
    path_values = {
        "repo": resolved_repo,
        "control_binary": v.lexical_absolute_path(
            parsed["control_binary"],
            working_directory=Path(resolved_repo),
        ),
        "treatment_binary": v.lexical_absolute_path(
            parsed["treatment_binary"],
            working_directory=Path(resolved_repo),
        ),
        "output_directory": v.lexical_absolute_path(
            parsed["output_directory"],
            working_directory=Path(working_directory),
        ),
    }
    numeric_defaults = {
        "idle_minimum_percent": v.IDLE_MINIMUM_PERCENT,
        "idle_timeout_seconds": 300.0,
        "member_timeout_seconds": 300.0,
    }
    numeric_values: dict[str, float] = {}
    for name, default in numeric_defaults.items():
        try:
            value = float(parsed[name]) if name in parsed else default
        except ValueError as error:
            raise CausalVerificationError(
                f"Preflight runner argv {name} is not numeric"
            ) from error
        require(
            math.isfinite(value),
            f"Preflight runner argv {name} is not finite",
        )
        numeric_values[name] = value
    return {
        **path_values,
        "experiment": parsed["experiment"],
        **numeric_values,
        "preflight_only": preflight_only,
    }


def canonical_compile_argument_map(
    entries: list[Any],
    *,
    build_directory: Path,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
    context: str,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    build_prefix = str(build_directory)
    for index, entry in enumerate(entries):
        entry_context = f"{context} compile entry {index}"
        mapping = v.expect_mapping(entry, entry_context)
        working_directory = Path(
            v.expect_string(
                mapping.get("directory"),
                f"{entry_context} directory",
            )
        )
        raw = v.compile_entry_arguments(mapping, context=entry_context)
        expanded = v.expand_response_arguments(
            raw,
            working_directory=working_directory,
            response_files=response_files,
            used_response_files=used_response_files,
            context=entry_context,
        )
        source = canonical_build_path(
            source_path_from_compile_entry(mapping, entry_context),
            build_directory=build_directory,
        )
        require(source not in result, f"{context} duplicates compile source {source}")
        result[source] = [
            argument.replace(build_prefix, "$BUILD")
            for argument in v.normalized_compile_arguments(
                expanded,
                entry_context,
            )
        ]
    return result


def canonical_build_path(path: str, *, build_directory: Path) -> str:
    build_prefix = str(build_directory)
    if path == build_prefix:
        return "$BUILD"
    descendant_prefix = f"{build_prefix}{os.sep}"
    if path.startswith(descendant_prefix):
        return f"$BUILD/{path[len(descendant_prefix):]}"
    return path


def canonical_link_arguments(
    arguments: list[str],
    *,
    build_directory: Path,
) -> list[str]:
    build_prefix = str(build_directory)
    return [argument.replace(build_prefix, "$BUILD") for argument in arguments]


def ordered_build_products(
    binary_path: str,
    *,
    build_directory: Path,
    context: str,
) -> list[dict[str, str]]:
    benchmark = Path(binary_path)
    require(
        benchmark.name == BUILD_PRODUCT_TARGETS[0][1],
        f"{context} benchmark binary has an unexpected name",
    )
    outputs = [
        benchmark,
        benchmark.with_name(BUILD_PRODUCT_TARGETS[1][1]),
    ]
    require(
        len(outputs) == len(BUILD_PRODUCT_TARGETS),
        f"{context} build product definition count differs",
    )
    products: list[dict[str, str]] = []
    for (name, expected_target), output in zip(
        BUILD_PRODUCT_TARGETS,
        outputs,
    ):
        require(
            output.is_absolute()
            and os.path.abspath(str(output)) == str(output),
            f"{context} {name} output is not canonical absolute",
        )
        target = v.ninja_target_for_output(
            output,
            build_directory=build_directory,
            context=f"{context} {name}",
        )
        require(
            Path(target).name == expected_target,
            f"{context} {name} target differs",
        )
        products.append(
            {
                "name": name,
                "requested_target": target,
                "expected_output": str(output),
            }
        )
    return products


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
    selected_target_graph = v.expect_mapping(
        closure["selected_target_graph"],
        "build-input identity selected target graph",
    )
    derived_paths.update(
        v.expect_string(path, "build-input identity graph-proven output")
        for path in v.expect_list(
            selected_target_graph["derived_link_outputs"],
            "build-input identity graph-proven outputs",
        )
    )
    products = v.expect_list(
        closure["products"],
        "build-input identity products",
    )
    derived_paths.update(
        v.expect_string(
            v.expect_mapping(
                product,
                f"build-input identity product {index}",
            )["expected_output"],
            f"build-input identity product {index} expected output",
        )
        for index, product in enumerate(products)
    )
    build_prefix = str(build_directory)
    result: list[dict[str, Any]] = []
    for index, value in enumerate(closure["files"]):
        context = f"build-input identity file {index}"
        record = v.expect_mapping(value, context)
        path = v.expect_string(record["path"], f"{context} path")
        roles = [
            v.expect_string(role, f"{context} role")
            for role in v.expect_list(record["roles"], f"{context} roles")
        ]
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
                "resolved_path": v.expect_string(
                    record["resolved_path"],
                    f"{context} resolved path",
                ).replace(build_prefix, "$BUILD"),
                "type": v.expect_string(record["type"], f"{context} type"),
                "roles": roles,
                "sha256": v.expect_sha256(record["sha256"], f"{context} sha256"),
                "bytes": v.expect_int(
                    record["bytes"],
                    f"{context} bytes",
                    minimum=0,
                ),
            }
        )
    return sorted(
        result,
        key=lambda record: (
            str(record["path"]).encode("utf-8"),
            str(record["resolved_path"]).encode("utf-8"),
        ),
    )


def validate_git_repository(
    evidence_dir: Path,
    value: Any,
    *,
    expected_root: str,
    build_directory: Path,
    source_records: list[dict[str, Any]],
) -> None:
    context = "preflight source Git repository"
    repository = v.expect_mapping(value, context)
    v.expect_exact_keys(
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
    require(repository["root"] == expected_root, f"{context} root differs")
    for name in ("head", "tree"):
        require(
            v.GIT_OBJECT_ID.fullmatch(
                v.expect_string(repository[name], f"{context} {name}")
            )
            is not None,
            f"{context} {name} is not a Git object ID",
        )
    status_path = v.validate_capture_reference(
        evidence_dir,
        {
            "status_path": repository["status_path"],
            "sha256": repository["status_sha256"],
        },
        context=f"{context} status",
        path_key="status_path",
        expected_path="source-repository-git-status.txt",
    )
    diff_path = v.validate_capture_reference(
        evidence_dir,
        {
            "diff_path": repository["diff_path"],
            "sha256": repository["diff_sha256"],
        },
        context=f"{context} diff",
        path_key="diff_path",
        expected_path="source-repository-source-diff.patch",
    )
    dirty = v._read_text(status_path, f"{context} status") != ""
    require(type(repository["dirty"]) is bool, f"{context} dirty flag is invalid")
    require(repository["dirty"] is dirty, f"{context} dirty flag differs")
    v._read_text(diff_path, f"{context} diff")

    commit_path = v.validate_capture_reference(
        evidence_dir,
        {
            "commit_object_path": repository["commit_object_path"],
            "sha256": repository["commit_object_sha256"],
        },
        context=f"{context} commit",
        path_key="commit_object_path",
        expected_path="source-repository-commit-object.txt",
    )
    commit_bytes = v.read_verified_bytes(
        commit_path,
        f"{context} commit object",
    )
    require(
        v.git_object_id("commit", commit_bytes) == repository["head"],
        f"{context} commit bytes differ from HEAD",
    )
    commit_lines = commit_bytes.splitlines()
    require(
        commit_lines
        and commit_lines[0] == f"tree {repository['tree']}".encode("ascii"),
        f"{context} commit tree differs",
    )
    tree_path = v.validate_capture_reference(
        evidence_dir,
        {
            "tree_listing_path": repository["tree_listing_path"],
            "sha256": repository["tree_listing_sha256"],
        },
        context=f"{context} tree listing",
        path_key="tree_listing_path",
        expected_path="source-repository-tree-listing.txt",
    )
    tree_entries = v.parse_git_tree_listing(
        v._read_text(tree_path, f"{context} tree listing"),
        f"{context} tree listing",
    )
    require(
        v.git_tree_id(tree_entries, f"{context} tree listing")
        == repository["tree"],
        f"{context} tree listing differs from the root tree",
    )
    if repository["dirty"] is False:
        root_path = Path(expected_root)
        for source in source_records:
            absolute = v.expect_string(
                source["absolute_path"],
                f"{context} pristine source absolute path",
            )
            try:
                Path(absolute).relative_to(build_directory)
            except ValueError:
                pass
            else:
                continue
            try:
                relative = Path(absolute).relative_to(root_path).as_posix()
            except ValueError:
                continue
            tree_entry = tree_entries.get(relative)
            require(
                tree_entry is not None and tree_entry[1] == "blob",
                f"{context} pristine source {relative!r} is absent from HEAD",
            )
            _captured_text, captured_path = v.relative_evidence_path(
                evidence_dir,
                source["captured_path"],
                f"{context} pristine source {relative}",
            )
            require(
                v.git_object_id(
                    "blob",
                    v.read_verified_bytes(
                        captured_path,
                        f"{context} pristine source {relative}",
                    ),
                )
                == tree_entry[2],
                f"{context} pristine source {relative!r} differs from HEAD",
            )


def validate_idempotence_artifact(
    evidence_dir: Path,
    value: Any,
    *,
    context: str,
    expected_path: str,
    count_key: str | None = None,
) -> tuple[Path, int | None]:
    record = v.expect_mapping(value, context)
    expected_keys = {"captured_path", "sha256", "bytes"}
    if count_key is not None:
        expected_keys.add(count_key)
    v.expect_exact_keys(record, expected_keys, context)
    path = v.validate_capture_reference(
        evidence_dir,
        record,
        context=context,
        expected_path=expected_path,
    )
    require(
        path.stat().st_size
        == v.expect_int(record["bytes"], f"{context} bytes", minimum=0),
        f"{context} size differs",
    )
    count = (
        v.expect_int(record[count_key], f"{context} {count_key}", minimum=0)
        if count_key is not None
        else None
    )
    return path, count


def validate_idempotence_target_identities(
    value: Any,
    *,
    context: str,
    require_resolved_path_identity: bool = False,
) -> list[dict[str, Any]]:
    records = v.expect_list(value, context)
    paths: list[str] = []
    result: list[dict[str, Any]] = []
    for index, value_record in enumerate(records):
        item_context = f"{context} item {index}"
        record = v.expect_mapping(value_record, item_context)
        v.expect_exact_keys(
            record,
            {
                "path",
                "resolved_path",
                "sha256",
                *STAT_MANIFEST_KEYS,
            },
            item_context,
        )
        path = v.expect_string(record["path"], f"{item_context} path")
        resolved_path = v.expect_string(
            record["resolved_path"],
            f"{item_context} resolved path",
        )
        require(
            Path(path).is_absolute() and os.path.abspath(path) == path,
            f"{item_context} path is not canonical absolute",
        )
        require(
            Path(resolved_path).is_absolute(),
            f"{item_context} resolved path is not absolute",
        )
        if require_resolved_path_identity:
            require(
                resolved_path == path,
                f"{item_context} resolved path differs from path",
            )
        v.expect_sha256(record["sha256"], f"{item_context} sha256")
        v.expect_int(record["device"], f"{item_context} device", minimum=0)
        v.expect_int(record["inode"], f"{item_context} inode", minimum=1)
        mode = v.expect_int(record["mode"], f"{item_context} mode", minimum=0)
        require(stat.S_ISREG(mode), f"{item_context} mode is not regular")
        v.expect_int(record["links"], f"{item_context} links", minimum=1)
        v.expect_int(record["bytes"], f"{item_context} bytes", minimum=0)
        v.expect_int(record["mtime_ns"], f"{item_context} mtime", minimum=0)
        v.expect_int(record["ctime_ns"], f"{item_context} ctime", minimum=0)
        require(path not in paths, f"{context} duplicates {path}")
        paths.append(path)
        result.append(record)
    require(
        paths == sorted(paths, key=lambda path: path.encode("utf-8")),
        f"{context} paths are not canonically ordered",
    )
    return result


def validate_descriptor_relative_quarantines(
    value: Any,
    *,
    evidence_dir: Path,
    evidence_manifest: dict[str, str] | None,
    role: str,
    context: str,
    build_directory: Path,
    root_device: int,
    derived_before: list[dict[str, Any]],
    protected_identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = v.expect_list(value, context)
    expected_by_path = {
        v.expect_string(record["path"], f"{context} expected path"): record
        for record in derived_before
    }
    result: list[dict[str, Any]] = []
    paths: list[str] = []
    quarantine_paths: list[str] = []
    protected_inodes = {
        (
            v.expect_int(record["device"], f"{context} protected device"),
            v.expect_int(record["inode"], f"{context} protected inode"),
        )
        for record in protected_identities
    }
    require(
        all(
            v.expect_int(
                record["device"],
                f"{context} protected device",
                minimum=0,
            )
            == root_device
            for record in protected_identities
        ),
        f"{context} protected identity uses another filesystem",
    )
    expected_inodes = [
        (
            v.expect_int(record["device"], f"{context} expected device"),
            v.expect_int(record["inode"], f"{context} expected inode"),
        )
        for record in derived_before
    ]
    require(
        len(expected_inodes) == len(set(expected_inodes))
        and not (set(expected_inodes) & protected_inodes)
        and all(device == root_device for device, _inode in expected_inodes)
        and all(
            v.expect_int(
                record["links"],
                f"{context} expected links",
                minimum=1,
            )
            == 1
            for record in derived_before
        ),
        f"{context} expected outputs alias each other, protected metadata, "
        "are hard-linked, or use another filesystem",
    )

    def validate_stat_identity(
        value: Any,
        *,
        identity_context: str,
        minimum_links: int,
    ) -> dict[str, Any]:
        identity = v.expect_mapping(value, identity_context)
        v.expect_exact_keys(
            identity,
            {"sha256", *STAT_MANIFEST_KEYS},
            identity_context,
        )
        v.expect_sha256(identity["sha256"], f"{identity_context} sha256")
        v.expect_int(identity["device"], f"{identity_context} device", minimum=0)
        v.expect_int(identity["inode"], f"{identity_context} inode", minimum=1)
        mode = v.expect_int(
            identity["mode"],
            f"{identity_context} mode",
            minimum=0,
        )
        require(stat.S_ISREG(mode), f"{identity_context} mode is not regular")
        v.expect_int(
            identity["links"],
            f"{identity_context} links",
            minimum=minimum_links,
        )
        v.expect_int(identity["bytes"], f"{identity_context} bytes", minimum=0)
        v.expect_int(
            identity["mtime_ns"],
            f"{identity_context} mtime",
            minimum=0,
        )
        v.expect_int(
            identity["ctime_ns"],
            f"{identity_context} ctime",
            minimum=0,
        )
        return identity

    for index, value_record in enumerate(records):
        item_context = f"{context} item {index}"
        record = v.expect_mapping(value_record, item_context)
        v.expect_exact_keys(
            record,
            {
                "path",
                "relative_path",
                "resolved_path",
                "quarantined_path",
                "before",
                "after",
                "name_absent_after",
            },
            item_context,
        )
        path = v.expect_string(record["path"], f"{item_context} path")
        relative_text = v.expect_string(
            record["relative_path"],
            f"{item_context} relative path",
        )
        resolved_path = v.expect_string(
            record["resolved_path"],
            f"{item_context} resolved path",
        )
        quarantined_text, quarantined_path = v.relative_evidence_path(
            evidence_dir,
            record["quarantined_path"],
            f"{item_context} quarantined path",
        )
        expected_quarantined_text = (
            f"{role}-rebuild-quarantine/{index:08d}.output"
        )
        try:
            expected_relative = Path(path).relative_to(build_directory)
        except ValueError as error:
            raise CausalVerificationError(
                f"{item_context} path escapes the build directory"
            ) from error
        relative_path = Path(relative_text)
        require(
            Path(path).is_absolute()
            and os.path.abspath(path) == path
            and resolved_path == path
            and relative_path.as_posix() == relative_text
            and not relative_path.is_absolute()
            and relative_path.parts
            and all(part not in ("", ".", "..") for part in relative_path.parts)
            and expected_relative == relative_path,
            f"{item_context} path is not canonical descriptor-relative",
        )
        require(
            quarantined_text == expected_quarantined_text
            and quarantined_text not in quarantine_paths,
            f"{item_context} quarantined path differs",
        )
        before = validate_stat_identity(
            record["before"],
            identity_context=f"{item_context} before",
            minimum_links=1,
        )
        after = validate_stat_identity(
            record["after"],
            identity_context=f"{item_context} after",
            minimum_links=1,
        )
        require(
            before["device"] == root_device
            and after["device"] == root_device
            and all(
                after[key] == before[key]
                for key in (
                    "device",
                    "inode",
                    "mode",
                    "bytes",
                    "mtime_ns",
                    "sha256",
                )
            )
            and after["links"] == before["links"]
            and after["ctime_ns"] >= before["ctime_ns"]
            and record["name_absent_after"] is True,
            f"{item_context} quarantine transition is invalid",
        )
        expected = expected_by_path.get(path)
        require(
            expected is not None
            and expected["resolved_path"] == path
            and before
            == {
                "sha256": expected["sha256"],
                **{
                    key: expected[key]
                    for key in STAT_MANIFEST_KEYS
                },
            },
            f"{item_context} identity differs from derived-before",
        )
        inode_key = (before["device"], before["inode"])
        quarantined_metadata = v.verified_file_stat(
            quarantined_path,
            f"{item_context} quarantined output",
        )
        quarantined_live_identity = {
            "device": quarantined_metadata.st_dev,
            "inode": quarantined_metadata.st_ino,
            "mode": quarantined_metadata.st_mode,
            "links": quarantined_metadata.st_nlink,
            "bytes": quarantined_metadata.st_size,
            "mtime_ns": quarantined_metadata.st_mtime_ns,
            "ctime_ns": quarantined_metadata.st_ctime_ns,
        }
        require(
            inode_key not in protected_inodes
            and stat.S_ISREG(quarantined_metadata.st_mode)
            and quarantined_live_identity == {
                key: after[key]
                for key in STAT_MANIFEST_KEYS
            }
            and v.sha256_file(quarantined_path) == after["sha256"],
            f"{item_context} aliases protected metadata or quarantined "
            "identity/content differs",
        )
        require(path not in paths, f"{context} duplicates {path}")
        paths.append(path)
        quarantine_paths.append(quarantined_text)
        result.append(record)
    require(
        paths == sorted(paths, key=lambda path: path.encode("utf-8")),
        f"{context} paths are not canonically ordered",
    )
    require(
        set(paths) == set(expected_by_path),
        f"{context} path set differs from derived-before",
    )
    if evidence_manifest is not None:
        quarantine_prefix = f"{role}-rebuild-quarantine/"
        expected_quarantine_paths = set(quarantine_paths)
        manifest_quarantine_paths = {
            path
            for path in evidence_manifest
            if path.startswith(quarantine_prefix)
        }
        require(
            manifest_quarantine_paths == expected_quarantine_paths,
            f"{context} manifest quarantine path set differs: "
            f"missing={sorted(expected_quarantine_paths - manifest_quarantine_paths)}, "
            f"unexpected={sorted(manifest_quarantine_paths - expected_quarantine_paths)}",
        )
        quarantine_directory = evidence_dir / quarantine_prefix.rstrip("/")
        quarantine_metadata, quarantine_entries = v.verified_directory_snapshot(
            quarantine_directory,
            f"{context} quarantine directory",
        )
        require(
            stat.S_ISDIR(quarantine_metadata.st_mode)
            and not stat.S_ISLNK(quarantine_metadata.st_mode)
            and quarantine_entries
            == sorted(Path(path).name for path in expected_quarantine_paths),
            f"{context} quarantine directory contains unexpected entries",
        )
    return result


def rebuild_metadata_paths(
    *,
    build_directory: Path,
) -> set[str]:
    return {
        str(build_directory / "CMakeCache.txt"),
        str(build_directory / "compile_commands.json"),
        str(build_directory / "build.ninja"),
        str(build_directory / "CMakeFiles/rules.ninja"),
        str(build_directory / "CMakeFiles/VerifyGlobs.cmake"),
        str(build_directory / "CMakeFiles/cmake.verify_globs"),
        str(build_directory / ".ninja_deps"),
        str(build_directory / ".ninja_log"),
    }


def validate_preclosure_metadata(
    evidence_dir: Path,
    value: Any,
    *,
    context: str,
    expected_paths: set[str],
    build_directory: Path,
    cache_capture: Path,
    compile_capture: Path,
) -> list[dict[str, Any]]:
    records = v.expect_list(value, context)
    paths: list[str] = []
    result: list[dict[str, Any]] = []
    for index, value_record in enumerate(records):
        item_context = f"{context} item {index}"
        record = v.expect_mapping(value_record, item_context)
        v.expect_exact_keys(
            record,
            {
                "path",
                "resolved_path",
                "sha256",
                *STAT_MANIFEST_KEYS,
                "captured_path",
            },
            item_context,
        )
        path = v.expect_string(record["path"], f"{item_context} path")
        resolved_path = v.expect_string(
            record["resolved_path"],
            f"{item_context} resolved path",
        )
        require(
            Path(path).is_absolute()
            and os.path.abspath(path) == path
            and Path(resolved_path).is_absolute(),
            f"{item_context} path is not canonical absolute",
        )
        require(
            resolved_path == path,
            f"{item_context} resolved path differs from path",
        )
        digest = v.expect_sha256(record["sha256"], f"{item_context} sha256")
        v.expect_int(record["device"], f"{item_context} device", minimum=0)
        v.expect_int(record["inode"], f"{item_context} inode", minimum=1)
        mode = v.expect_int(record["mode"], f"{item_context} mode", minimum=0)
        require(stat.S_ISREG(mode), f"{item_context} mode is not regular")
        v.expect_int(record["links"], f"{item_context} links", minimum=1)
        size = v.expect_int(record["bytes"], f"{item_context} bytes", minimum=0)
        v.expect_int(record["mtime_ns"], f"{item_context} mtime", minimum=0)
        v.expect_int(record["ctime_ns"], f"{item_context} ctime", minimum=0)
        captured_text, captured_path = v.relative_evidence_path(
            evidence_dir,
            record["captured_path"],
            f"{item_context} captured path",
        )
        require(
            captured_text == f"provenance/blobs/{digest[:2]}/{digest}",
            f"{item_context} is not stored at its content-addressed path",
        )
        data = v.read_verified_bytes(captured_path, f"{item_context} content")
        require(
            len(data) == size and hashlib.sha256(data).hexdigest() == digest,
            f"{item_context} captured content differs",
        )
        require(path not in paths, f"{context} duplicates {path}")
        paths.append(path)
        result.append(record)
    require(
        paths == sorted(paths, key=lambda path: path.encode("utf-8")),
        f"{context} paths are not canonically ordered",
    )
    require(set(paths) == expected_paths, f"{context} path set differs")
    by_path = {record["path"]: record for record in result}
    known_captures = {
        str(build_directory / "CMakeCache.txt"): cache_capture,
        str(build_directory / "compile_commands.json"): compile_capture,
    }
    for path, capture in known_captures.items():
        require(
            path in by_path
            and by_path[path]["sha256"] == v.sha256_file(capture)
            and by_path[path]["bytes"] == capture.stat().st_size,
            f"{context} identity differs from archived capture: {path}",
        )
    return result


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
    closure_values = v.expect_list(
        closure["files"],
        f"{context} closure files",
    )
    closure_files: dict[str, dict[str, Any]] = {}
    for index, value_record in enumerate(closure_values):
        record = v.expect_mapping(value_record, f"{context} closure file {index}")
        path = v.expect_string(
            record["path"],
            f"{context} closure file {index} path",
        )
        require(
            path not in closure_files,
            f"{context} closure contains duplicate file paths",
        )
        closure_files[path] = record
    compile_outputs = [
        v.expect_string(
            v.expect_mapping(unit, f"{context} compile unit {index}")["output"],
            f"{context} compile unit {index} output",
        )
        for index, unit in enumerate(
            v.expect_list(closure["compile_units"], f"{context} compile units")
        )
    ]
    require(
        len(compile_outputs) == len(set(compile_outputs)),
        f"{context} closure contains duplicate compile outputs",
    )
    for path in compile_outputs:
        record = closure_files.get(path)
        require(
            record is not None
            and "compile-output"
            in v.expect_list(record["roles"], f"{context} closure file roles"),
            f"{context} compile output lacks an authenticated closure file: "
            f"{path}",
        )

    observed = {
        v.expect_string(record["path"], f"{context} observed path"): record
        for record in (*derived_before, *immutable_before)
    }
    require(
        len(observed) == len(derived_before) + len(immutable_before),
        f"{context} rebuild identity sets overlap",
    )
    for index, record in enumerate(closure_files.values()):
        path = v.expect_string(
            record["path"],
            f"{context} closure file {index} path",
        )
        current = observed.get(path)
        require(
            current is not None,
            f"{context} closure file is absent before clean: {path}",
        )
        require(
            identity_content(current)
            == {
                "path": path,
                "resolved_path": v.expect_string(
                    record["resolved_path"],
                    f"{context} closure file {index} resolved path",
                ),
                "sha256": v.expect_sha256(
                    record["sha256"],
                    f"{context} closure file {index} sha256",
                ),
                "bytes": v.expect_int(
                    record["bytes"],
                    f"{context} closure file {index} bytes",
                    minimum=0,
                ),
            },
            f"{context} closure file identity changed before clean: {path}",
        )


def rebuild_output_contract(
    closure: dict[str, Any],
    *,
    build_directory: Path,
    binary_path: str,
) -> tuple[set[str], set[str], dict[str, str]]:
    selected_target_graph = v.expect_mapping(
        closure["selected_target_graph"],
        "rebuild selected target graph",
    )
    product_values = v.expect_list(
        closure["products"],
        "rebuild products",
    )
    product_names: list[str] = []
    selected_outputs: list[str] = []
    for index, product_value in enumerate(product_values):
        product = v.expect_mapping(
            product_value,
            f"rebuild product {index}",
        )
        product_names.append(
            v.expect_string(
                product["name"],
                f"rebuild product {index} name",
            )
        )
        selected_outputs.append(
            v.expect_string(
                product["expected_output"],
                f"rebuild product {index} expected output",
            )
        )
    require(
        product_names == [name for name, _ in BUILD_PRODUCT_TARGETS]
        and selected_outputs
        and selected_outputs[0] == binary_path
        and v.expect_list(
            selected_target_graph["selected_outputs"],
            "rebuild selected outputs",
        )
        == selected_outputs,
        "Rebuild selected outputs differ from the ordered build products",
    )
    compile_outputs: set[str] = set()
    module_producers: dict[str, str] = {}
    for index, unit_value in enumerate(
        v.expect_list(closure["compile_units"], "rebuild compile units")
    ):
        unit = v.expect_mapping(unit_value, f"rebuild compile unit {index}")
        output = v.expect_string(
            unit["output"],
            f"rebuild compile unit {index} output",
        )
        compile_outputs.add(output)
        for module_index, path in enumerate(
            v.expect_list(
                unit["module_outputs"],
                f"rebuild compile unit {index} module outputs",
            )
        ):
            module_output = v.expect_string(
                path,
                f"rebuild compile unit {index} module output {module_index}",
            )
            require(
                module_output not in module_producers,
                f"Rebuild module output has duplicate producers: "
                f"{module_output}",
            )
            module_producers[module_output] = output
    module_outputs = set(module_producers)

    response_outputs: set[str] = set()
    for index, record_value in enumerate(
        v.expect_list(closure["files"], "rebuild closure files")
    ):
        record = v.expect_mapping(record_value, f"rebuild closure file {index}")
        roles = {
            v.expect_string(role, f"rebuild closure file {index} role")
            for role in v.expect_list(
                record["roles"],
                f"rebuild closure file {index} roles",
            )
        }
        if "response-file" not in roles:
            continue
        path = v.expect_string(
            record["path"],
            f"rebuild closure file {index} path",
        )
        try:
            relative_path = Path(path).relative_to(build_directory)
        except ValueError:
            continue
        require(
            relative_path.parts,
            f"Build-local response file names the build directory: {path}",
        )
        response_outputs.add(path)

    freshness_outputs = {
        v.expect_string(path, "rebuild material output")
        for path in v.expect_list(
            selected_target_graph["material_outputs"],
            "rebuild material outputs",
        )
    }
    clean_plan = v.expect_mapping(
        selected_target_graph["clean_plan"],
        "rebuild selected target clean plan",
    )
    freshness_outputs.update(
        v.expect_string(path, "rebuild target clean output")
        for path in v.expect_list(
            clean_plan["outputs"],
            "rebuild target clean outputs",
        )
    )
    freshness_outputs.update(
        v.expect_string(path, "rebuild derived link output")
        for path in v.expect_list(
            selected_target_graph["derived_link_outputs"],
            "rebuild derived link outputs",
        )
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
    before_by_path = {
        v.expect_string(record["path"], "rebuild before path"): record
        for record in before
    }
    after_by_path = {
        v.expect_string(record["path"], "rebuild after path"): record
        for record in after
    }
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


def validate_rebuild_content_deltas(
    value: Any,
    *,
    context: str,
) -> list[dict[str, Any]]:
    records = v.expect_list(value, context)
    paths: list[str] = []
    result: list[dict[str, Any]] = []
    for index, value_record in enumerate(records):
        item_context = f"{context} item {index}"
        record = v.expect_mapping(value_record, item_context)
        v.expect_exact_keys(
            record,
            {
                "path",
                "before_sha256",
                "before_bytes",
                "after_sha256",
                "after_bytes",
            },
            item_context,
        )
        path = v.expect_string(record["path"], f"{item_context} path")
        require(
            Path(path).is_absolute() and os.path.abspath(path) == path,
            f"{item_context} path is not canonical absolute",
        )
        v.expect_sha256(record["before_sha256"], f"{item_context} before sha256")
        v.expect_sha256(record["after_sha256"], f"{item_context} after sha256")
        v.expect_int(record["before_bytes"], f"{item_context} before bytes", minimum=0)
        v.expect_int(record["after_bytes"], f"{item_context} after bytes", minimum=0)
        require(
            record["before_sha256"] != record["after_sha256"]
            or record["before_bytes"] != record["after_bytes"],
            f"{item_context} does not describe a content change",
        )
        require(path not in paths, f"{context} duplicates {path}")
        paths.append(path)
        result.append(record)
    require(
        paths == sorted(paths, key=lambda path: path.encode("utf-8")),
        f"{context} paths are not canonically ordered",
    )
    return result


def exact_link_action_count(
    text: str,
    *,
    expected_arguments: list[str],
    build_directory: Path,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
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
            raise CausalVerificationError(
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
            arguments = v.expand_response_arguments(
                tokens[compiler_index:end],
                working_directory=build_directory,
                response_files=response_files,
                used_response_files=used_response_files,
                context=f"{context} invocation on line {line_number}",
            )
            if arguments == expected_arguments:
                matches += 1
    return matches


def validate_idempotence_tree_manifest(
    value: Any,
    *,
    context: str,
) -> list[dict[str, Any]]:
    records = v.expect_list(value, context)
    require(
        0 < len(records) <= BUILD_TREE_MAX_ENTRIES,
        f"{context} entry count is outside the runner limit",
    )
    paths: list[str] = []
    result: list[dict[str, Any]] = []
    for index, value_record in enumerate(records):
        item_context = f"{context} item {index}"
        record = v.expect_mapping(value_record, item_context)
        entry_type = v.expect_string(record.get("type"), f"{item_context} type")
        expected_keys = {"path", "type", *STAT_MANIFEST_KEYS}
        if entry_type == "symlink":
            expected_keys.add("target")
        elif entry_type == "file":
            expected_keys.add("sha256")
        else:
            require(
                entry_type == "directory",
                f"{item_context} has unsupported type",
            )
        v.expect_exact_keys(record, expected_keys, item_context)
        path = v.expect_string(record["path"], f"{item_context} path")
        path_object = PurePosixPath(path)
        require(
            path != ""
            and "\0" not in path
            and not path_object.is_absolute()
            and path_object.as_posix() == path
            and (
                path == "."
                or all(
                    part not in ("", ".", "..")
                    for part in path_object.parts
                )
            ),
            f"{item_context} path is not a canonical relative path",
        )
        require(
            path != "." or entry_type == "directory",
            f"{item_context} root entry is not a directory",
        )
        require(path != ".ninja_log", f"{context} includes the mutable Ninja log")
        for key in STAT_MANIFEST_KEYS:
            minimum = 1 if key in ("inode", "mode", "links") else 0
            v.expect_int(record[key], f"{item_context} {key}", minimum=minimum)
        mode = record["mode"]
        require(
            (
                entry_type == "directory"
                and stat.S_ISDIR(mode)
                and not stat.S_ISLNK(mode)
            )
            or (entry_type == "file" and stat.S_ISREG(mode))
            or (entry_type == "symlink" and stat.S_ISLNK(mode)),
            f"{item_context} mode does not match its type",
        )
        depth = 0 if path == "." else len(path_object.parts)
        maximum_depth = (
            BUILD_TREE_MAX_DEPTH
            if entry_type == "directory"
            else BUILD_TREE_MAX_DEPTH + 1
        )
        require(
            depth <= maximum_depth,
            f"{item_context} exceeds the runner depth limit",
        )
        if entry_type == "symlink":
            target = v.expect_string(
                record["target"],
                f"{item_context} target",
            )
            try:
                target_bytes = target.encode("utf-8")
            except UnicodeEncodeError as error:
                raise CausalVerificationError(
                    f"{item_context} target is not UTF-8"
                ) from error
            require(
                "\0" not in target and record["bytes"] == len(target_bytes),
                f"{item_context} target size is invalid",
            )
        elif entry_type == "file":
            v.expect_sha256(record["sha256"], f"{item_context} sha256")
        require(path not in paths, f"{context} duplicates {path}")
        paths.append(path)
        result.append(record)
    require(
        paths
        == sorted(
            paths,
            key=lambda path: (
                path != ".",
                path.encode("utf-8"),
            ),
        ),
        f"{context} paths are not canonically ordered",
    )
    require(
        paths and paths[0] == "." and paths.count(".") == 1,
        f"{context} does not contain exactly one root directory record",
    )
    by_path = {record["path"]: record for record in result}
    for path in paths:
        if path == ".":
            continue
        parent = PurePosixPath(path).parent.as_posix()
        require(
            parent in by_path and by_path[parent]["type"] == "directory",
            f"{context} path {path} lacks a directory parent",
        )
    by_inode: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in result:
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
            f"{context} inode {inode} has conflicting records",
        )
        require(
            first["links"] >= len(inode_records),
            f"{context} inode {inode} has an impossible hard-link count",
        )
        require(
            first["type"] != "directory" or len(inode_records) == 1,
            f"{context} repeats a directory inode {inode}",
        )
    return result


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


def validate_idempotence_transcript(
    stdout: str,
    stderr: str,
    *,
    build_directory: Path,
    cmake_path: str,
    context: str,
) -> None:
    stdout_lines = stdout.splitlines()
    stderr_lines = stderr.splitlines()
    require(
        stdout_lines
        and stdout_lines[0]
        == "ninja: Entering directory `.'",
        f"{context} has an unexpected directory header",
    )
    require(
        stdout_lines[-1:] == ["ninja: no work to do."],
        f"{context} does not end with no work",
    )
    progress_prefix = f"{NINJA_IDEMPOTENCE_PROGRESS} "
    progress = [
        line.removeprefix(progress_prefix)
        for line in stdout_lines[1:-1]
        if line.startswith(progress_prefix)
    ]
    require(
        len(stdout_lines) == 3 and len(progress) == 1,
        f"{context} executed an unexpected command",
    )
    require(
        shlex.split(progress[0])
        == [
            cmake_path,
            "-P",
            str(build_directory / "CMakeFiles/VerifyGlobs.cmake"),
        ],
        f"{context} did not execute only VerifyGlobs.cmake",
    )
    require(
        stderr_lines
        == [
            "ninja explain: "
            f"{build_directory}/CMakeFiles/VerifyGlobs.cmake_force is dirty"
        ],
        f"{context} explanation differs from the forced glob-check edge",
    )


def normalized_ninja_dependency_record(
    record: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    return {
        **stored_ninja_dependency_record(record, context=context),
        "status": v.expect_string(record["status"], f"{context} status"),
    }


def stored_ninja_dependency_record(
    record: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    output = v.expect_string(record["output"], f"{context} output")
    dependencies = [
        v.expect_string(dependency, f"{context} dependency")
        for dependency in v.expect_list(
            record["dependencies"],
            f"{context} dependencies",
        )
    ]
    require(
        v.expect_int(
            record["dependency_count"],
            f"{context} dependency count",
            minimum=0,
        )
        == len(dependencies),
        f"{context} dependency count differs",
    )
    return {
        "output": output,
        "deps_mtime": v.expect_int(
            record["deps_mtime"],
            f"{context} deps mtime",
            minimum=0,
        ),
        "dependencies": sorted(
            dependencies,
            key=lambda value: value.encode("utf-8"),
        ),
    }


def ninja_dependency_input_identity(
    record: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    stored = stored_ninja_dependency_record(record, context=context)
    return {
        "output": stored["output"],
        "dependencies": stored["dependencies"],
    }


def ninja_dependency_output_spellings(
    text: str,
    *,
    build_directory: Path,
    records: dict[str, dict[str, Any]],
    context: str,
) -> dict[str, str]:
    lines = text.splitlines()
    result: dict[str, str] = {}
    index = 0
    while index < len(lines):
        if lines[index] == "":
            index += 1
            continue
        match = v.NINJA_DEPS_HEADER.fullmatch(lines[index])
        require(
            match is not None,
            f"{context} has an invalid header on line {index + 1}",
        )
        lexical = match.group("output")
        canonical = v.lexical_absolute_path(
            lexical,
            working_directory=build_directory,
        )
        require(
            canonical not in result,
            f"{context} duplicates output {canonical}",
        )
        result[canonical] = lexical
        index += 1 + int(match.group("count"))
        require(
            index <= len(lines),
            f"{context} truncates dependencies for {canonical}",
        )
    require(
        set(result) == set(records),
        f"{context} output spellings differ from parsed records",
    )
    return result


def parse_ninja_dependency_database(
    content: bytes,
    *,
    build_directory: Path,
    expected_output_spellings: dict[str, str],
    context: str,
) -> dict[str, dict[str, Any]]:
    require(
        len(content) >= 16
        and content[: len(NINJA_DEPS_SIGNATURE)] == NINJA_DEPS_SIGNATURE
        and struct.unpack_from("<I", content, len(NINJA_DEPS_SIGNATURE))[0]
        == NINJA_DEPS_VERSION,
        f"{context} database has an invalid Ninja signature or version",
    )
    require(
        0 < len(expected_output_spellings) <= NINJA_DEPS_MAX_OUTPUTS,
        f"{context} expected output count is outside the parser limit",
    )
    require(
        Path(build_directory).is_absolute()
        and os.path.abspath(build_directory) == str(build_directory),
        f"{context} build directory is not canonical absolute",
    )
    lexical_values: list[str] = []
    for output, lexical in expected_output_spellings.items():
        require(
            isinstance(output, str)
            and isinstance(lexical, str)
            and Path(output).is_absolute()
            and os.path.abspath(output) == output
            and v.lexical_absolute_path(
                lexical,
                working_directory=build_directory,
            )
            == output,
            f"{context} expected output spelling differs: {output}",
        )
        lexical_values.append(lexical)
    require(
        len(set(lexical_values)) == len(lexical_values),
        f"{context} expected output spellings are not unique",
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
            f"{context} database truncates a record size",
        )
        encoded_size = struct.unpack_from("<I", content, offset)[0]
        offset += 4
        is_dependency = bool(encoded_size & 0x80000000)
        record_size = encoded_size & 0x7FFFFFFF
        record_count += 1
        require(
            record_count <= NINJA_DEPS_MAX_RECORDS
            and 0 < record_size <= NINJA_DEPS_MAX_RECORD_BYTES,
            f"{context} database record {record_count} exceeds the parser limit",
        )
        require(
            record_size <= len(content) - offset,
            f"{context} database truncates record {record_count}",
        )
        payload = content[offset : offset + record_size]
        offset += record_size
        if is_dependency:
            require(
                record_size >= 12 and record_size % 4 == 0,
                f"{context} database dependency record {record_count} is invalid",
            )
            dependency_count = record_size // 4 - 3
            require(
                dependency_count <= NINJA_DEPS_MAX_DEPENDENCIES_PER_OUTPUT,
                f"{context} database dependency record {record_count} "
                "exceeds the per-record dependency limit",
            )
            total_dependency_count += dependency_count
            require(
                total_dependency_count <= NINJA_DEPS_MAX_TOTAL_DEPENDENCIES,
                f"{context} database exceeds the total dependency limit",
            )
            values = struct.unpack(f"<{record_size // 4}I", payload)
            output_id, mtime_low, mtime_high, *dependency_ids = values
            require(
                output_id < len(nodes)
                and all(node_id < len(nodes) for node_id in dependency_ids),
                f"{context} database dependency record {record_count} "
                "references an invalid node",
            )
            mtime = mtime_low | (mtime_high << 32)
            require(
                mtime <= 0x7FFFFFFFFFFFFFFF,
                f"{context} database dependency record {record_count} "
                "has a negative mtime",
            )
            dependencies[output_id] = (mtime, tuple(dependency_ids))
            continue

        require(
            record_size >= 8 and record_size % 4 == 0,
            f"{context} database path record {record_count} is invalid",
        )
        path_and_padding = payload[:-4]
        path_bytes = path_and_padding.rstrip(b"\0")
        padding = len(path_and_padding) - len(path_bytes)
        require(
            path_bytes
            and padding <= 3
            and padding == (4 - len(path_bytes) % 4) % 4
            and b"\0" not in path_bytes,
            f"{context} database path record {record_count} has invalid padding",
        )
        checksum = struct.unpack_from("<I", payload, record_size - 4)[0]
        require(
            checksum == ((~len(nodes)) & 0xFFFFFFFF),
            f"{context} database path record {record_count} has an invalid ID",
        )
        try:
            path = path_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CausalVerificationError(
                f"{context} database path record {record_count} is not UTF-8"
            ) from error
        require(
            "\n" not in path
            and "\r" not in path
            and path not in node_ids
            and len(nodes) < NINJA_DEPS_MAX_NODES,
            f"{context} database path record {record_count} is invalid",
        )
        node_ids[path] = len(nodes)
        nodes.append(path)

    require(offset == len(content), f"{context} database has trailing bytes")
    result: dict[str, dict[str, Any]] = {}
    for output in sorted(
        expected_output_spellings,
        key=lambda value: value.encode("utf-8"),
    ):
        lexical = expected_output_spellings[output]
        output_id = node_ids.get(lexical)
        require(
            output_id is not None and output_id in dependencies,
            f"{context} database omits expected output {output}",
        )
        mtime, dependency_ids = dependencies[output_id]
        dependency_paths = [
            v.lexical_absolute_path(
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
    context: str,
) -> dict[str, dict[str, Any]]:
    try:
        database_content = v.read_verified_bytes(
            database_path,
            f"{context} database",
            maximum_bytes=NINJA_DEPS_DATABASE_MAX_BYTES,
        )
    except (OSError, v.VerificationError) as error:
        raise CausalVerificationError(f"{context} failed: {error}") from error
    result = parse_ninja_dependency_database(
        database_content,
        build_directory=build_directory,
        expected_output_spellings=expected_output_spellings,
        context=context,
    )
    try:
        database_content_after = v.read_verified_bytes(
            database_path,
            f"{context} database after parse",
            maximum_bytes=NINJA_DEPS_DATABASE_MAX_BYTES,
        )
    except (OSError, v.VerificationError) as error:
        raise CausalVerificationError(f"{context} failed: {error}") from error
    require(
        database_content_after == database_content,
        f"{context} archived Ninja database changed during parse",
    )
    return result


def validate_ninja_dependency_archive(
    report: dict[str, dict[str, Any]],
    archive: dict[str, dict[str, Any]],
    *,
    context: str,
) -> None:
    require(
        set(report) == set(archive),
        f"{context} output membership differs from the archived Ninja database",
    )
    for output in sorted(report):
        require(
            stored_ninja_dependency_record(
                report[output],
                context=f"{context} report {output}",
            )
            == stored_ninja_dependency_record(
                archive[output],
                context=f"{context} archive {output}",
            ),
            f"{context} record differs from the archived Ninja database: "
            f"{output}",
        )


def validate_ninja_dependency_settlement(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    closure: dict[str, Any],
    context: str,
) -> tuple[list[str], list[str], int]:
    closure_outputs: set[str] = set()
    closure_by_output: dict[str, dict[str, Any]] = {}
    dependency_graph = v.expect_list(
        closure["dependency_graph"],
        f"{context} closure dependency graph",
    )
    for index, value in enumerate(dependency_graph):
        record = v.expect_mapping(
            value,
            f"{context} closure dependency record {index}",
        )
        output = v.expect_string(
            record["output"],
            f"{context} closure dependency output {index}",
        )
        require(
            output not in closure_outputs,
            f"{context} closure dependency graph duplicates {output}",
        )
        closure_outputs.add(output)
        closure_by_output[output] = record

    missing_before = sorted(closure_outputs - set(before))
    missing_after = sorted(closure_outputs - set(after))
    require(
        not missing_before and not missing_after,
        f"{context} omits selected outputs: "
        f"before={missing_before}, after={missing_after}",
    )
    for output in sorted(set(before) & set(after)):
        require(
            normalized_ninja_dependency_record(
                before[output],
                context=f"{context} before {output}",
            )
            == normalized_ninja_dependency_record(
                after[output],
                context=f"{context} after {output}",
            ),
            f"{context} retained record changed: {output}",
        )
    for output in sorted(closure_outputs):
        require(
            before[output]["status"] == "VALID"
            and after[output]["status"] == "VALID",
            f"{context} selected record is not VALID: {output}",
        )
        expected_identity = ninja_dependency_input_identity(
            closure_by_output[output],
            context=f"{context} closure {output}",
        )
        require(
            ninja_dependency_input_identity(
                before[output],
                context=f"{context} before {output}",
            )
            == expected_identity
            and ninja_dependency_input_identity(
                after[output],
                context=f"{context} after {output}",
            )
            == expected_identity,
            f"{context} selected dependency inputs differ from the captured "
            f"closure: {output}",
        )

    added = sorted(set(after) - set(before), key=lambda value: value.encode("utf-8"))
    removed = sorted(
        set(before) - set(after),
        key=lambda value: value.encode("utf-8"),
    )
    require(
        not added and not removed,
        f"{context} changed output membership: "
        f"added={added}, removed={removed}",
    )
    return added, removed, len(closure_outputs)


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
        raise CausalVerificationError(
            f"{context} Ninja log append is not UTF-8"
        ) from error
    fields = appended_lines[0].split("\t") if len(appended_lines) == 1 else []
    require(
        len(fields) == 5
        and all(re.fullmatch(r"(0|[1-9][0-9]*)", value) for value in fields[:3])
        and int(fields[0]) <= int(fields[1])
        and 0 < int(fields[2]) <= 0x7FFFFFFFFFFFFFFF
        and fields[3] == str(build_directory / "CMakeFiles/cmake.verify_globs")
        and re.fullmatch(r"(?:0|[1-9a-f][0-9a-f]{0,15})", fields[4])
        is not None,
        f"{context} Ninja log append differs from the glob-check edge",
    )


def parse_ninja_clean_transcript(
    stdout: str,
    stderr: str,
    *,
    build_directory: Path,
    requested_target: str,
    additional_requested_targets: list[str] | None = None,
    supplemental_targets: list[str] | None = None,
    context: str,
) -> list[str]:
    require(stderr == "", f"{context} emitted stderr")
    require(
        stdout.endswith("\n") and "\r" not in stdout,
        f"{context} is not canonical newline-terminated text",
    )
    lines = stdout.splitlines()
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

    paths: list[str] = []
    seen: set[str] = set()
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
            raise CausalVerificationError(
                f"{context} removal path escapes the build directory: {path_text}"
            ) from error
        require(
            relative_path.parts,
            f"{context} attempts to remove the build directory",
        )
        canonical_path = str(absolute_path)
        require(
            canonical_path not in seen,
            f"{context} repeats removal path: {path_text}",
        )
        seen.add(canonical_path)
        paths.append(canonical_path)

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


def paths_overlap(left: str, right: str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def validate_build_idempotence(
    evidence_dir: Path,
    value: Any,
    *,
    role: str,
    build_directory: Path,
    requested_target: str,
    build_tool_path: str,
    cmake_path: str,
    closure: dict[str, Any],
    binary_path: str,
    binary_sha256: str,
    binary_bytes: int,
    cache_capture: Path,
    compile_capture: Path,
    link_arguments: list[str],
    response_files: dict[tuple[str, str], Path],
    evidence_manifest: dict[str, str],
) -> None:
    context = f"preflight {role} build idempotence"
    closure_product_values = v.expect_list(
        closure["products"],
        f"{context} products",
    )
    closure_products = [
        v.expect_mapping(product, f"{context} product {index}")
        for index, product in enumerate(closure_product_values)
    ]
    requested_targets = [
        v.expect_string(
            product["requested_target"],
            f"{context} product {index} requested target",
        )
        for index, product in enumerate(closure_products)
    ]
    require(
        requested_targets
        and requested_targets[0] == requested_target
        and len(requested_targets) == len(set(requested_targets)),
        f"{context} ordered rebuild targets differ from the benchmark target",
    )
    closure_link_arguments = [
        [
            v.expect_string(
                argument,
                f"{context} product {product_index} link argument",
            )
            for argument in v.expect_list(
                v.expect_mapping(
                    product["link"],
                    f"{context} product {product_index} link",
                )["arguments"],
                f"{context} product {product_index} link arguments",
            )
        ]
        for product_index, product in enumerate(closure_products)
    ]
    require(
        closure_link_arguments
        and closure_link_arguments[0] == link_arguments,
        f"{context} benchmark link arguments differ from the closure",
    )
    record = v.expect_mapping(value, context)
    v.expect_exact_keys(
        record,
        {
            "idempotence_schema_version",
            "execution",
            "ninja_lock_observations",
            "clean_rebuild",
            "settlement",
            "command",
            "returncode",
            "stdout",
            "stderr",
            "target_before",
            "target_after",
            "build_tree_before",
            "build_tree_after",
            "allowed_mutable_paths",
            "ninja_log_before",
            "ninja_log_after",
        },
        context,
    )
    require(
        v.expect_int(
            record["idempotence_schema_version"],
            f"{context} schema",
            minimum=1,
        )
        == IDEMPOTENCE_SCHEMA_VERSION,
        f"{context} schema is unsupported",
    )
    execution = v.expect_mapping(record["execution"], f"{context} execution")
    v.expect_exact_keys(
        execution,
        {
            "method",
            "build_directory",
            "root_device",
            "root_inode",
            "ninja_log_version",
        },
        f"{context} execution",
    )
    require(
        execution["method"] == NINJA_EXECUTION_METHOD
        and execution["build_directory"] == str(build_directory)
        and v.expect_int(
            execution["root_device"],
            f"{context} execution root device",
            minimum=0,
        )
        >= 0
        and v.expect_int(
            execution["root_inode"],
            f"{context} execution root inode",
            minimum=1,
        )
        >= 1
        and v.expect_int(
            execution["ninja_log_version"],
            f"{context} execution Ninja log version",
            minimum=1,
        )
        == NINJA_LOG_VERSION,
        f"{context} execution anchor differs",
    )
    root_device = v.expect_int(
        execution["root_device"],
        f"{context} execution root device",
        minimum=0,
    )
    lock_observations = v.expect_list(
        record["ninja_lock_observations"],
        f"{context} Ninja lock observations",
    )
    expected_lock_order = [
        (invocation, phase)
        for invocation in NINJA_INVOCATION_ORDER
        for phase in ("before", "after")
    ]
    require(
        len(lock_observations) == len(expected_lock_order),
        f"{context} Ninja lock observation count differs",
    )
    for index, (value, expected_order) in enumerate(
        zip(lock_observations, expected_lock_order)
    ):
        observation = v.expect_mapping(
            value,
            f"{context} Ninja lock observation {index}",
        )
        v.expect_exact_keys(
            observation,
            {
                "sequence",
                "invocation",
                "phase",
                "relative_path",
                "absent",
            },
            f"{context} Ninja lock observation {index}",
        )
        require(
            v.expect_int(
                observation["sequence"],
                f"{context} Ninja lock observation sequence",
                minimum=0,
            )
            == index
            and (
                v.expect_string(
                    observation["invocation"],
                    f"{context} Ninja lock observation invocation",
                ),
                v.expect_string(
                    observation["phase"],
                    f"{context} Ninja lock observation phase",
                ),
            )
            == expected_order
            and observation["relative_path"] == ".ninja_lock"
            and observation["absent"] is True,
            f"{context} Ninja lock observation {index} differs",
        )
    require(
        record["command"]
        == [
            build_tool_path,
            "-C",
            ".",
            "-v",
            "-d",
            "explain",
            *requested_targets,
        ],
        f"{context} command differs",
    )
    require(
        record["command"].count(build_tool_path) == 1,
        f"{context} command does not contain exactly one Ninja invocation",
    )
    require(
        v.expect_int(record["returncode"], f"{context} return code") == 0,
        f"{context} command failed",
    )
    clean_rebuild = v.expect_mapping(
        record["clean_rebuild"],
        f"{context} clean rebuild",
    )
    v.expect_exact_keys(
        clean_rebuild,
        {
            "base_clean_command",
            "base_clean_returncode",
            "base_clean_stdout",
            "base_clean_stderr",
            "dry_clean_command",
            "dry_clean_returncode",
            "dry_clean_stdout",
            "dry_clean_stderr",
            "removal_method",
            "removed_outputs",
            "rebuild_command",
            "rebuild_returncode",
            "rebuild_stdout",
            "rebuild_stderr",
            "preclosure_metadata",
            "supplemental_targets",
            "derived_before",
            "absent_before",
            "derived_absent_after_clean",
            "derived_after",
            "content_deltas",
            "immutable_before",
            "immutable_after_clean",
            "immutable_after",
            "protected_before",
            "protected_after_clean",
            "protected_after",
        },
        f"{context} clean rebuild",
    )
    expected_base_clean_command = [
        build_tool_path,
        "-v",
        "-n",
        "-C",
        ".",
        "-t",
        "clean",
        *requested_targets,
    ]
    expected_rebuild_command = [
        build_tool_path,
        "-C",
        ".",
        "-v",
        "-d",
        "explain",
        *requested_targets,
    ]
    require(
        clean_rebuild["base_clean_command"] == expected_base_clean_command,
        f"{context} base clean command differs",
    )
    require(
        clean_rebuild["rebuild_command"] == expected_rebuild_command,
        f"{context} rebuild command differs",
    )
    require(
        v.expect_int(
            clean_rebuild["base_clean_returncode"],
            f"{context} base clean return code",
        )
        == 0,
        f"{context} base clean command failed",
    )
    require(
        v.expect_int(
            clean_rebuild["dry_clean_returncode"],
            f"{context} dry clean return code",
        )
        == 0,
        f"{context} dry clean command failed",
    )
    require(
        v.expect_int(
            clean_rebuild["rebuild_returncode"],
            f"{context} rebuild return code",
        )
        == 0,
        f"{context} rebuild command failed",
    )
    transcript_paths: dict[str, Path] = {}
    for name, expected_name in (
        (
            "base_clean_stdout",
            f"{role}-rebuild-base-clean-plan.stdout",
        ),
        (
            "base_clean_stderr",
            f"{role}-rebuild-base-clean-plan.stderr",
        ),
        ("dry_clean_stdout", f"{role}-rebuild-clean-plan.stdout"),
        ("dry_clean_stderr", f"{role}-rebuild-clean-plan.stderr"),
        ("rebuild_stdout", f"{role}-rebuild.stdout"),
        ("rebuild_stderr", f"{role}-rebuild.stderr"),
    ):
        artifact_path, _ = validate_idempotence_artifact(
            evidence_dir,
            clean_rebuild[name],
            context=f"{context} {name.replace('_', ' ')}",
            expected_path=expected_name,
        )
        transcript_paths[name] = artifact_path
        if name == "rebuild_stdout":
            require(
                "ninja: no work to do."
                not in v._read_text(artifact_path, f"{context} rebuild stdout"),
                f"{context} rebuild performed no material work",
            )

    base_clean_stdout = v._read_text(
        transcript_paths["base_clean_stdout"],
        f"{context} base clean stdout",
    )
    base_clean_stderr = v._read_text(
        transcript_paths["base_clean_stderr"],
        f"{context} base clean stderr",
    )
    base_clean_path_list = parse_ninja_clean_transcript(
        base_clean_stdout,
        base_clean_stderr,
        build_directory=build_directory,
        requested_target=requested_target,
        additional_requested_targets=requested_targets[1:],
        context=f"{context} base clean transcript",
    )
    base_clean_paths = set(base_clean_path_list)
    (
        freshness_output_paths,
        exact_output_paths,
        module_producers,
    ) = rebuild_output_contract(
        closure,
        build_directory=build_directory,
        binary_path=binary_path,
    )
    clean_plan = v.expect_mapping(
        v.expect_mapping(
            closure["selected_target_graph"],
            f"{context} selected target graph",
        )["clean_plan"],
        f"{context} frozen target clean plan",
    )
    frozen_base_clean_path_list = [
        v.expect_string(path, f"{context} frozen target clean output")
        for path in v.expect_list(
            clean_plan["outputs"],
            f"{context} frozen target clean outputs",
        )
    ]
    frozen_base_clean_paths = set(frozen_base_clean_path_list)
    unexpected_base_clean_paths = sorted(
        base_clean_paths - frozen_base_clean_paths
    )
    require(
        not unexpected_base_clean_paths,
        f"{context} base clean plan contains non-closure outputs: "
        f"{unexpected_base_clean_paths}",
    )
    missing_base_clean_paths = sorted(
        frozen_base_clean_paths - base_clean_paths
    )
    require(
        not missing_base_clean_paths,
        f"{context} base clean plan omits frozen closure outputs: "
        f"{missing_base_clean_paths}",
    )
    require(
        base_clean_path_list == frozen_base_clean_path_list,
        f"{context} base clean plan order differs from the frozen closure",
    )
    supplemental_targets = [
        v.expect_string(path, f"{context} supplemental target")
        for path in v.expect_list(
            clean_rebuild["supplemental_targets"],
            f"{context} supplemental targets",
        )
    ]
    require(
        supplemental_targets
        == sorted(
            set(supplemental_targets),
            key=lambda path: path.encode("utf-8"),
        ),
        f"{context} supplemental targets are not unique and ordered",
    )
    supplemental_removal_paths: set[str] = set()
    for target in supplemental_targets:
        path = Path(target)
        require(
            not path.is_absolute()
            and path.as_posix() == target
            and path.parts
            and all(part not in ("", ".", "..") for part in path.parts),
            f"{context} supplemental target is not canonical relative: "
            f"{target}",
        )
        supplemental_removal_paths.add(str(build_directory / path))
    expected_supplemental_removals = freshness_output_paths - base_clean_paths
    require(
        supplemental_removal_paths == expected_supplemental_removals,
        f"{context} supplemental targets differ from graph-derived outputs",
    )
    require(
        supplemental_removal_paths <= set(module_producers),
        f"{context} supplemental targets contain non-module outputs",
    )
    require(
        all(
            module_producers[path] in base_clean_paths
            for path in supplemental_removal_paths
        ),
        f"{context} supplemental module output producer is not cleaned",
    )
    expected_dry_clean_command = [
        *expected_base_clean_command,
        *supplemental_targets,
    ]
    require(
        clean_rebuild["dry_clean_command"] == expected_dry_clean_command,
        f"{context} dry clean command differs",
    )
    require(
        clean_rebuild["removal_method"] == OUTPUT_REMOVAL_METHOD,
        f"{context} output removal method differs",
    )

    dry_clean_stdout = v._read_text(
        transcript_paths["dry_clean_stdout"],
        f"{context} dry clean stdout",
    )
    dry_clean_stderr = v._read_text(
        transcript_paths["dry_clean_stderr"],
        f"{context} dry clean stderr",
    )
    planned_clean_path_list = parse_ninja_clean_transcript(
        dry_clean_stdout,
        dry_clean_stderr,
        build_directory=build_directory,
        requested_target=requested_target,
        additional_requested_targets=requested_targets[1:],
        supplemental_targets=supplemental_targets,
        context=f"{context} combined dry clean transcript",
    )
    planned_clean_paths = set(planned_clean_path_list)
    removal_paths = freshness_output_paths
    require(
        len(planned_clean_path_list) == len(removal_paths)
        and planned_clean_paths == removal_paths,
        f"{context} combined clean plan differs from required removals",
    )
    closure_paths = {
        v.expect_string(item["path"], f"{context} closure path")
        for item in v.expect_list(closure["files"], f"{context} closure files")
    }
    all_metadata_paths = rebuild_metadata_paths(build_directory=build_directory)
    protected_paths = all_metadata_paths - {
        str(build_directory / ".ninja_deps"),
        str(build_directory / ".ninja_log"),
    }
    clean_protected_paths = all_metadata_paths | {
        str(build_directory / ".ninja_log"),
        str(build_directory / ".ninja_lock"),
    }
    protected_overlaps = sorted(
        {
            removal
            for removal in removal_paths
            for protected in clean_protected_paths
            if paths_overlap(removal, protected)
        }
    )
    require(
        not protected_overlaps,
        f"{context} clean removal set overlaps protected build metadata",
    )
    removed_outputs_path, removed_output_count = validate_idempotence_artifact(
        evidence_dir,
        clean_rebuild["removed_outputs"],
        context=f"{context} removed outputs",
        expected_path=f"{role}-rebuild-removed-outputs.json",
        count_key="files",
    )
    preclosure_path, preclosure_count = validate_idempotence_artifact(
        evidence_dir,
        clean_rebuild["preclosure_metadata"],
        context=f"{context} pre-closure metadata",
        expected_path=f"{role}-rebuild-preclosure-metadata.json",
        count_key="files",
    )
    preclosure_metadata = validate_preclosure_metadata(
        evidence_dir,
        v.read_json(preclosure_path, f"{context} pre-closure metadata"),
        context=f"{context} pre-closure metadata",
        expected_paths=all_metadata_paths,
        build_directory=build_directory,
        cache_capture=cache_capture,
        compile_capture=compile_capture,
    )
    require(
        preclosure_count == len(preclosure_metadata),
        f"{context} pre-closure metadata count differs",
    )
    immutable_paths = closure_paths - removal_paths

    rebuild_identities: dict[str, list[dict[str, Any]]] = {}
    identity_specs = {
        "derived_before": (
            f"{role}-rebuild-derived-before.json",
            None,
            True,
        ),
        "derived_after": (
            f"{role}-rebuild-derived-after.json",
            None,
            True,
        ),
        "immutable_before": (
            f"{role}-rebuild-immutable-before.json",
            immutable_paths,
            False,
        ),
        "immutable_after_clean": (
            f"{role}-rebuild-immutable-after-clean.json",
            immutable_paths,
            False,
        ),
        "immutable_after": (
            f"{role}-rebuild-immutable-after.json",
            immutable_paths,
            False,
        ),
        "protected_before": (
            f"{role}-rebuild-protected-before.json",
            protected_paths,
            True,
        ),
        "protected_after_clean": (
            f"{role}-rebuild-protected-after-clean.json",
            protected_paths,
            True,
        ),
        "protected_after": (
            f"{role}-rebuild-protected-after.json",
            protected_paths,
            True,
        ),
    }
    for name, (
        expected_name,
        expected_paths,
        require_resolved_path_identity,
    ) in identity_specs.items():
        path, count = validate_idempotence_artifact(
            evidence_dir,
            clean_rebuild[name],
            context=f"{context} {name.replace('_', ' ')}",
            expected_path=expected_name,
            count_key="files",
        )
        identities = validate_idempotence_target_identities(
            v.read_json(path, f"{context} {name.replace('_', ' ')}"),
            context=f"{context} {name.replace('_', ' ')}",
            require_resolved_path_identity=require_resolved_path_identity,
        )
        require(
            count == len(identities),
            f"{context} {name.replace('_', ' ')} count differs",
        )
        require(
            expected_paths is None
            or {
                v.expect_string(item["path"], f"{context} identity path")
                for item in identities
            }
            == expected_paths,
            f"{context} {name.replace('_', ' ')} path set differs",
        )
        rebuild_identities[name] = identities

    derived_paths = {
        v.expect_string(item["path"], f"{context} derived path")
        for item in rebuild_identities["derived_before"]
    }
    absent_before = [
        v.expect_string(path, f"{context} absent-before path")
        for path in v.expect_list(
            clean_rebuild["absent_before"],
            f"{context} absent-before paths",
        )
    ]
    require(
        absent_before
        == sorted(set(absent_before), key=lambda path: path.encode("utf-8")),
        f"{context} absent-before paths are not unique and ordered",
    )
    absent_before_paths = set(absent_before)
    require(
        derived_paths <= removal_paths
        and freshness_output_paths <= derived_paths,
        f"{context} existing clean-removal output set differs",
    )
    require(
        derived_paths.isdisjoint(absent_before_paths)
        and derived_paths | absent_before_paths == removal_paths,
        f"{context} clean-removal filesystem partition differs",
    )
    removed_outputs = validate_descriptor_relative_quarantines(
        v.read_json(removed_outputs_path, f"{context} removed outputs"),
        evidence_dir=evidence_dir,
        evidence_manifest=evidence_manifest,
        role=role,
        context=f"{context} removed outputs",
        build_directory=build_directory,
        root_device=root_device,
        derived_before=rebuild_identities["derived_before"],
        protected_identities=preclosure_metadata,
    )
    require(
        removed_output_count == len(removed_outputs),
        f"{context} removed output count differs",
    )
    expected_quarantine_namespace = {
        v.expect_string(
            item["quarantined_path"],
            f"{context} quarantined path",
        )
        for item in removed_outputs
    }
    expected_quarantine_namespace.add(
        removed_outputs_path.relative_to(evidence_dir).as_posix()
    )
    actual_quarantine_namespace = {
        path
        for path in evidence_manifest
        if path.startswith(f"{role}-rebuild-quarantine")
        or path.startswith(f"{role}-rebuild-removed-outputs")
    }
    require(
        actual_quarantine_namespace == expected_quarantine_namespace,
        f"{context} quarantine evidence namespace differs: "
        f"missing={sorted(expected_quarantine_namespace - actual_quarantine_namespace)}, "
        f"unexpected={sorted(actual_quarantine_namespace - expected_quarantine_namespace)}",
    )
    require(
        {
            v.expect_string(item["path"], f"{context} derived path")
            for item in rebuild_identities["derived_after"]
        }
        == derived_paths,
        f"{context} derived after path set differs",
    )
    require(
        clean_rebuild["derived_absent_after_clean"] == sorted(derived_paths),
        f"{context} clean absence set differs",
    )
    require_closure_identity_binding(
        closure,
        derived_before=rebuild_identities["derived_before"],
        immutable_before=rebuild_identities["immutable_before"],
        context=context,
    )
    preclosure_by_path = {
        v.expect_string(record["path"], f"{context} pre-closure path"): record
        for record in preclosure_metadata
    }
    for protected_record in rebuild_identities["protected_before"]:
        path = v.expect_string(
            protected_record["path"],
            f"{context} protected path",
        )
        require(
            {
                key: protected_record[key]
                for key in (
                    "path",
                    "resolved_path",
                    "sha256",
                    *STAT_MANIFEST_KEYS,
                )
            }
            == {
                key: preclosure_by_path[path][key]
                for key in (
                    "path",
                    "resolved_path",
                    "sha256",
                    *STAT_MANIFEST_KEYS,
                )
            },
            f"{context} protected metadata changed after closure capture: {path}",
        )

    content_deltas_path, content_delta_count = validate_idempotence_artifact(
        evidence_dir,
        clean_rebuild["content_deltas"],
        context=f"{context} content deltas",
        expected_path=f"{role}-rebuild-content-deltas.json",
        count_key="files",
    )
    content_deltas = validate_rebuild_content_deltas(
        v.read_json(content_deltas_path, f"{context} content deltas"),
        context=f"{context} content deltas",
    )
    require(
        content_delta_count == len(content_deltas),
        f"{context} content delta count differs",
    )
    expected_content_deltas = rebuild_content_deltas(
        rebuild_identities["derived_before"],
        rebuild_identities["derived_after"],
    )
    require(
        content_deltas == expected_content_deltas,
        f"{context} content deltas differ from rebuilt output identities",
    )
    changed_exact_outputs = sorted(
        delta["path"]
        for delta in content_deltas
        if v.expect_string(delta["path"], f"{context} content delta path")
        in exact_output_paths
    )
    require(
        not changed_exact_outputs,
        f"{context} exact compile/link outputs changed",
    )
    require(
        rebuild_identities["immutable_before"]
        == rebuild_identities["immutable_after_clean"]
        == rebuild_identities["immutable_after"],
        f"{context} immutable inputs changed during clean rebuild",
    )
    require(
        rebuild_identities["protected_before"]
        == rebuild_identities["protected_after_clean"]
        == rebuild_identities["protected_after"],
        f"{context} protected metadata changed during clean rebuild",
    )
    for phase in ("derived_before", "derived_after"):
        derived_by_path = {
            v.expect_string(item["path"], f"{context} derived path"): item
            for item in rebuild_identities[phase]
        }
        require(
            binary_path in derived_by_path
            and derived_by_path[binary_path]["sha256"] == binary_sha256
            and derived_by_path[binary_path]["bytes"] == binary_bytes,
            f"{context} rebuilt binary identity differs",
        )
    rebuilt_response_files: set[tuple[str, str]] = set()
    rebuild_stdout = v._read_text(
        transcript_paths["rebuild_stdout"],
        f"{context} rebuild stdout",
    )
    for product_index, product_link_arguments in enumerate(
        closure_link_arguments
    ):
        require(
            exact_link_action_count(
                rebuild_stdout,
                expected_arguments=product_link_arguments,
                build_directory=build_directory,
                response_files=response_files,
                used_response_files=rebuilt_response_files,
                context=f"{context} rebuild transcript product {product_index}",
            )
            == 1,
            f"{context} rebuild does not contain exactly one selected link "
            f"action for product {product_index}",
        )
    require(
        rebuilt_response_files <= set(response_files),
        f"{context} rebuild references unexpected response files",
    )

    settlement_context = f"{context} settlement"
    settlement = v.expect_mapping(record["settlement"], settlement_context)
    v.expect_exact_keys(
        settlement,
        {
            "command",
            "returncode",
            "stdout",
            "stderr",
            "dependency_report_command",
            "dependency_report_before",
            "dependency_report_after",
            "target_before",
            "target_after",
            "build_tree_before",
            "build_tree_after",
            "ninja_deps_before",
            "ninja_deps_after",
            "ninja_log_before",
            "ninja_log_after",
            "allowed_mutable_paths",
            "closure_dependency_outputs",
            "added_dependency_outputs",
            "removed_dependency_outputs",
        },
        settlement_context,
    )
    expected_noop_command = [
        build_tool_path,
        "-C",
        ".",
        "-v",
        "-d",
        "explain",
        *requested_targets,
    ]
    require(
        settlement["command"] == expected_noop_command,
        f"{settlement_context} command differs",
    )
    require(
        settlement["command"].count(build_tool_path) == 1,
        f"{settlement_context} command does not contain exactly one Ninja "
        "invocation",
    )
    require(
        v.expect_int(
            settlement["returncode"],
            f"{settlement_context} return code",
        )
        == 0,
        f"{settlement_context} command failed",
    )
    settlement_stdout_path, _ = validate_idempotence_artifact(
        evidence_dir,
        settlement["stdout"],
        context=f"{settlement_context} stdout",
        expected_path=f"{role}-settlement.stdout",
    )
    settlement_stderr_path, _ = validate_idempotence_artifact(
        evidence_dir,
        settlement["stderr"],
        context=f"{settlement_context} stderr",
        expected_path=f"{role}-settlement.stderr",
    )
    validate_idempotence_transcript(
        v._read_text(
            settlement_stdout_path,
            f"{settlement_context} stdout",
        ),
        v._read_text(
            settlement_stderr_path,
            f"{settlement_context} stderr",
        ),
        build_directory=build_directory,
        cmake_path=cmake_path,
        context=f"{settlement_context} transcript",
    )

    expected_dependency_report_command = [
        build_tool_path,
        "-C",
        ".",
        "-t",
        "deps",
    ]
    require(
        settlement["dependency_report_command"]
        == expected_dependency_report_command,
        f"{settlement_context} dependency report command differs",
    )
    dependency_records: dict[str, dict[str, dict[str, Any]]] = {}
    dependency_output_spellings: dict[str, dict[str, str]] = {}
    for phase in ("before", "after"):
        report_context = f"{settlement_context} dependency report {phase}"
        report = v.expect_mapping(
            settlement[f"dependency_report_{phase}"],
            report_context,
        )
        v.expect_exact_keys(
            report,
            {"returncode", "stdout", "stderr", "records"},
            report_context,
        )
        require(
            v.expect_int(report["returncode"], f"{report_context} return code")
            == 0,
            f"{report_context} command failed",
        )
        report_stdout_path, _ = validate_idempotence_artifact(
            evidence_dir,
            report["stdout"],
            context=f"{report_context} stdout",
            expected_path=(
                f"{role}-settlement-ninja-deps-report-{phase}.txt"
            ),
        )
        report_stderr_path, _ = validate_idempotence_artifact(
            evidence_dir,
            report["stderr"],
            context=f"{report_context} stderr",
            expected_path=(
                f"{role}-settlement-ninja-deps-report-{phase}.stderr"
            ),
        )
        require(
            v._read_text(report_stderr_path, f"{report_context} stderr") == "",
            f"{report_context} emitted stderr",
        )
        report_stdout = v._read_text(
            report_stdout_path,
            f"{report_context} stdout",
        )
        dependency_records[phase] = v.parse_ninja_deps(
            report_stdout,
            build_directory=build_directory,
            context=report_context,
        )
        dependency_output_spellings[phase] = (
            ninja_dependency_output_spellings(
                report_stdout,
                build_directory=build_directory,
                records=dependency_records[phase],
                context=report_context,
            )
        )
        require(
            v.expect_int(
                report["records"],
                f"{report_context} record count",
                minimum=1,
            )
            == len(dependency_records[phase]),
            f"{report_context} record count differs",
        )
    (
        expected_added_dependency_outputs,
        expected_removed_dependency_outputs,
        closure_dependency_output_count,
    ) = validate_ninja_dependency_settlement(
        dependency_records["before"],
        dependency_records["after"],
        closure=closure,
        context=f"{settlement_context} Ninja dependencies",
    )
    require(
        v.expect_int(
            settlement["closure_dependency_outputs"],
            f"{settlement_context} closure dependency output count",
            minimum=1,
        )
        == closure_dependency_output_count,
        f"{settlement_context} closure dependency output count differs",
    )
    for key, expected_values in (
        ("added_dependency_outputs", expected_added_dependency_outputs),
        ("removed_dependency_outputs", expected_removed_dependency_outputs),
    ):
        values = [
            v.expect_string(value, f"{settlement_context} {key} item")
            for value in v.expect_list(
                settlement[key],
                f"{settlement_context} {key}",
            )
        ]
        require(
            values == expected_values,
            f"{settlement_context} {key.replace('_', ' ')} differ",
        )

    expected_target_paths = {
        v.expect_string(item["path"], f"{context} closure path")
        for item in v.expect_list(closure["files"], f"{context} closure files")
    }
    expected_target_paths.update(freshness_output_paths)
    expected_target_paths.update(
        {
            str(build_directory / "CMakeCache.txt"),
            str(build_directory / "compile_commands.json"),
            str(build_directory / "build.ninja"),
            str(build_directory / "CMakeFiles/rules.ninja"),
            str(build_directory / "CMakeFiles/VerifyGlobs.cmake"),
            str(build_directory / "CMakeFiles/cmake.verify_globs"),
            str(build_directory / ".ninja_deps"),
            binary_path,
        }
    )
    settlement_target_records: dict[str, list[dict[str, Any]]] = {}
    for phase in ("before", "after"):
        path, count = validate_idempotence_artifact(
            evidence_dir,
            settlement[f"target_{phase}"],
            context=f"{settlement_context} target {phase}",
            expected_path=f"{role}-settlement-target-{phase}.json",
            count_key="files",
        )
        identities = validate_idempotence_target_identities(
            v.read_json(path, f"{settlement_context} target {phase}"),
            context=f"{settlement_context} target {phase}",
        )
        require(
            count == len(identities),
            f"{settlement_context} target {phase} count differs",
        )
        require(
            {
                v.expect_string(
                    item["path"],
                    f"{settlement_context} target path",
                )
                for item in identities
            }
            == expected_target_paths,
            f"{settlement_context} target {phase} path set differs",
        )
        settlement_target_records[phase] = identities
    ninja_deps_path_text = str(build_directory / ".ninja_deps")
    require(
        [
            item
            for item in settlement_target_records["before"]
            if item["path"] != ninja_deps_path_text
        ]
        == [
            item
            for item in settlement_target_records["after"]
            if item["path"] != ninja_deps_path_text
        ],
        f"{settlement_context} target closure changed outside .ninja_deps",
    )
    post_rebuild_by_path: dict[str, dict[str, Any]] = {}
    for item in (
        *rebuild_identities["derived_after"],
        *rebuild_identities["immutable_after"],
        *rebuild_identities["protected_after"],
    ):
        path = v.expect_string(
            item["path"],
            f"{settlement_context} post-rebuild path",
        )
        require(
            path not in post_rebuild_by_path
            or post_rebuild_by_path[path] == item,
            f"{settlement_context} post-rebuild identities disagree for "
            f"{path}",
        )
        post_rebuild_by_path[path] = item
    require(
        {
            v.expect_string(
                item["path"],
                f"{settlement_context} target path",
            ): item
            for item in settlement_target_records["before"]
            if item["path"] != ninja_deps_path_text
        }
        == post_rebuild_by_path,
        f"{settlement_context} baseline differs from post-rebuild identities",
    )

    settlement_database_paths: dict[str, Path] = {}
    for phase in ("before", "after"):
        database_path, _ = validate_idempotence_artifact(
            evidence_dir,
            settlement[f"ninja_deps_{phase}"],
            context=f"{settlement_context} Ninja deps database {phase}",
            expected_path=f"{role}-settlement-ninja-deps-{phase}.bin",
        )
        target_by_path = {
            v.expect_string(
                item["path"],
                f"{settlement_context} target path",
            ): item
            for item in settlement_target_records[phase]
        }
        require(
            target_by_path[ninja_deps_path_text]["sha256"]
            == v.sha256_file(database_path)
            and target_by_path[ninja_deps_path_text]["bytes"]
            == database_path.stat().st_size,
            f"{settlement_context} Ninja deps database {phase} identity "
            "differs",
        )
        settlement_database_paths[phase] = database_path

    for phase in ("before", "after"):
        archive = parse_archived_ninja_dependency_database(
            settlement_database_paths[phase],
            build_directory=build_directory,
            expected_output_spellings=dependency_output_spellings[phase],
            context=(
                f"{settlement_context} archived Ninja dependencies {phase}"
            ),
        )
        validate_ninja_dependency_archive(
            dependency_records[phase],
            archive,
            context=f"{settlement_context} Ninja dependencies {phase}",
        )

    closure_outputs = {
        v.expect_string(
            v.expect_mapping(
                value,
                f"{settlement_context} closure dependency record",
            )["output"],
            f"{settlement_context} closure dependency output",
        )
        for value in v.expect_list(
            closure["dependency_graph"],
            f"{settlement_context} closure dependency graph",
        )
    }
    for phase in ("before", "after"):
        target_by_path = {
            v.expect_string(
                item["path"],
                f"{settlement_context} target path",
            ): item
            for item in settlement_target_records[phase]
        }
        for output in closure_outputs:
            require(
                output in target_by_path
                and target_by_path[output]["mtime_ns"]
                == dependency_records[phase][output]["deps_mtime"],
                f"{settlement_context} {phase} selected dependency status "
                f"is not bound to output mtime: {output}",
            )

    settlement_tree_records: dict[str, list[dict[str, Any]]] = {}
    for phase in ("before", "after"):
        path, count = validate_idempotence_artifact(
            evidence_dir,
            settlement[f"build_tree_{phase}"],
            context=f"{settlement_context} build tree {phase}",
            expected_path=f"{role}-settlement-tree-{phase}.json",
            count_key="entries",
        )
        tree = validate_idempotence_tree_manifest(
            v.read_json(path, f"{settlement_context} build tree {phase}"),
            context=f"{settlement_context} build tree {phase}",
        )
        require(
            count == len(tree),
            f"{settlement_context} build tree {phase} count differs",
        )
        require(
            ".ninja_deps"
            in {
                v.expect_string(
                    item["path"],
                    f"{settlement_context} build tree path",
                )
                for item in tree
            },
            f"{settlement_context} build tree {phase} omits .ninja_deps",
        )
        root_records = [
            item
            for item in tree
            if v.expect_string(
                item["path"],
                f"{settlement_context} build tree path",
            )
            == "."
        ]
        require(
            len(root_records) == 1
            and root_records[0]["device"] == execution["root_device"]
            and root_records[0]["inode"] == execution["root_inode"]
            and ".ninja_lock"
            not in {
                v.expect_string(
                    item["path"],
                    f"{settlement_context} build tree path",
                )
                for item in tree
            },
            f"{settlement_context} build tree {phase} is not bound to the "
            "execution root or contains .ninja_lock",
        )
        settlement_tree_records[phase] = tree
    require(
        tree_without_mutable_paths(
            settlement_tree_records["before"],
            {".ninja_deps", ".ninja_log"},
        )
        == tree_without_mutable_paths(
            settlement_tree_records["after"],
            {".ninja_deps", ".ninja_log"},
        ),
        f"{settlement_context} build tree changed outside Ninja databases",
    )
    require(
        settlement["allowed_mutable_paths"]
        == [
            str(build_directory / ".ninja_deps"),
            str(build_directory / ".ninja_log"),
        ],
        f"{settlement_context} mutable path set differs",
    )
    settlement_log_paths: dict[str, Path] = {}
    for phase in ("before", "after"):
        settlement_log_paths[phase], _ = validate_idempotence_artifact(
            evidence_dir,
            settlement[f"ninja_log_{phase}"],
            context=f"{settlement_context} Ninja log {phase}",
            expected_path=f"{role}-settlement-ninja-log-{phase}.txt",
        )
    settlement_log_before = v.read_verified_bytes(
        settlement_log_paths["before"],
        f"{settlement_context} Ninja log before",
    )
    settlement_log_after = v.read_verified_bytes(
        settlement_log_paths["after"],
        f"{settlement_context} Ninja log after",
    )
    validate_ninja_log_append(
        settlement_log_before,
        settlement_log_after,
        build_directory=build_directory,
        context=settlement_context,
    )

    stdout_path, _ = validate_idempotence_artifact(
        evidence_dir,
        record["stdout"],
        context=f"{context} stdout",
        expected_path=f"{role}-idempotence.stdout",
    )
    stderr_path, _ = validate_idempotence_artifact(
        evidence_dir,
        record["stderr"],
        context=f"{context} stderr",
        expected_path=f"{role}-idempotence.stderr",
    )
    validate_idempotence_transcript(
        v._read_text(stdout_path, f"{context} stdout"),
        v._read_text(stderr_path, f"{context} stderr"),
        build_directory=build_directory,
        cmake_path=cmake_path,
        context=f"{context} transcript",
    )

    target_records: dict[str, list[dict[str, Any]]] = {}
    for phase in ("before", "after"):
        path, count = validate_idempotence_artifact(
            evidence_dir,
            record[f"target_{phase}"],
            context=f"{context} target {phase}",
            expected_path=f"{role}-idempotence-target-{phase}.json",
            count_key="files",
        )
        target_records[phase] = validate_idempotence_target_identities(
            v.read_json(path, f"{context} target {phase}"),
            context=f"{context} target {phase}",
        )
        require(
            count == len(target_records[phase]),
            f"{context} target {phase} count differs",
        )
    require(
        target_records["before"] == target_records["after"],
        f"{context} target closure changed",
    )
    require(
        settlement_target_records["after"] == target_records["before"],
        f"{context} authoritative target state changed after settlement",
    )
    target_by_path = {
        v.expect_string(item["path"], f"{context} target path"): item
        for item in target_records["before"]
    }
    require(
        set(target_by_path) == expected_target_paths,
        f"{context} target file set differs from the build closure",
    )
    known_files = {
        binary_path: (binary_sha256, binary_bytes),
        str(build_directory / "CMakeCache.txt"): (
            v.sha256_file(cache_capture),
            cache_capture.stat().st_size,
        ),
        str(build_directory / "compile_commands.json"): (
            v.sha256_file(compile_capture),
            compile_capture.stat().st_size,
        ),
    }
    for path, (digest, size) in known_files.items():
        require(
            target_by_path[path]["sha256"] == digest
            and target_by_path[path]["bytes"] == size,
            f"{context} identity differs for {path}",
        )

    tree_records: dict[str, list[dict[str, Any]]] = {}
    for phase in ("before", "after"):
        path, count = validate_idempotence_artifact(
            evidence_dir,
            record[f"build_tree_{phase}"],
            context=f"{context} build tree {phase}",
            expected_path=f"{role}-idempotence-tree-{phase}.json",
            count_key="entries",
        )
        tree_records[phase] = validate_idempotence_tree_manifest(
            v.read_json(path, f"{context} build tree {phase}"),
            context=f"{context} build tree {phase}",
        )
        require(
            count == len(tree_records[phase]),
            f"{context} build tree {phase} count differs",
        )
        root_records = [
            item
            for item in tree_records[phase]
            if v.expect_string(item["path"], f"{context} build tree path")
            == "."
        ]
        require(
            len(root_records) == 1
            and root_records[0]["device"] == execution["root_device"]
            and root_records[0]["inode"] == execution["root_inode"]
            and ".ninja_lock"
            not in {
                v.expect_string(item["path"], f"{context} build tree path")
                for item in tree_records[phase]
            },
            f"{context} build tree {phase} is not bound to the execution root "
            "or contains .ninja_lock",
        )
    require(
        tree_without_mutable_paths(
            tree_records["before"],
            {".ninja_log"},
        )
        == tree_without_mutable_paths(
            tree_records["after"],
            {".ninja_log"},
        ),
        f"{context} build tree changed outside the permitted Ninja log",
    )
    require(
        settlement_tree_records["after"] == tree_records["before"],
        f"{context} authoritative build tree changed after settlement",
    )
    tree_paths = {
        v.expect_string(item["path"], f"{context} build tree path")
        for item in tree_records["before"]
    }
    absent_relative_paths = {
        Path(path).relative_to(build_directory).as_posix()
        for path in absent_before_paths
    }
    require(
        tree_paths.isdisjoint(absent_relative_paths),
        f"{context} clean rebuild created outputs absent before clean",
    )
    require(
        record["allowed_mutable_paths"] == [str(build_directory / ".ninja_log")],
        f"{context} mutable path set differs",
    )

    log_paths: dict[str, Path] = {}
    for phase in ("before", "after"):
        log_paths[phase], _ = validate_idempotence_artifact(
            evidence_dir,
            record[f"ninja_log_{phase}"],
            context=f"{context} Ninja log {phase}",
            expected_path=f"{role}-idempotence-ninja-log-{phase}.txt",
        )
    log_before = v.read_verified_bytes(
        log_paths["before"],
        f"{context} Ninja log before",
    )
    log_after = v.read_verified_bytes(
        log_paths["after"],
        f"{context} Ninja log after",
    )
    require(
        settlement_log_after == log_before,
        f"{context} authoritative Ninja log changed after settlement",
    )
    validate_ninja_log_append(
        log_before,
        log_after,
        build_directory=build_directory,
        context=context,
    )


def target_compile_entries(
    entries: list[Any],
    ninja_commands: str,
    *,
    cache: dict[str, str],
    build_directory: Path,
    response_files: dict[tuple[str, str], Path],
    used_response_files: set[tuple[str, str]],
    context: str,
) -> list[Any]:
    compiler_tokens = set(
        v.configured_compiler_paths(cache, context=context)
    )
    target_arguments: dict[Path, list[str]] = {}
    for line_number, line in enumerate(ninja_commands.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise CausalVerificationError(
                f"{context} line {line_number} cannot be tokenized: {error}"
            ) from error
        for command_tokens in v.compiler_command_slices(
            tokens,
            compiler_tokens=compiler_tokens,
            scan_deps_tokens=set(
                v.configured_scan_deps_paths(cache, context=context)
            ),
            context=f"{context} line {line_number}",
        ):
            arguments = v.expand_response_arguments(
                command_tokens,
                working_directory=build_directory,
                response_files=response_files,
                used_response_files=used_response_files,
                context=f"{context} compiler invocation on line {line_number}",
            )
            output = v.compile_output_path(
                arguments,
                working_directory=build_directory,
                context=f"{context} compiler invocation on line {line_number}",
            )
            if output is None:
                continue
            previous = target_arguments.get(output)
            require(
                previous is None or previous == arguments,
                f"{context} has conflicting compiler invocations for {output}",
            )
            target_arguments.setdefault(output, arguments)
    require(target_arguments, f"{context} contains no compilation outputs")

    selected: dict[Path, tuple[list[str], Any]] = {}
    for index, entry_value in enumerate(entries):
        entry_context = f"{context} compile entry {index}"
        entry = v.expect_mapping(entry_value, entry_context)
        directory = Path(
            v.expect_string(
                entry.get("directory"),
                f"{entry_context} directory",
            )
        )
        require(
            directory.is_absolute(),
            f"{entry_context} directory is not absolute",
        )
        raw_arguments = v.compile_entry_arguments(
            entry,
            context=entry_context,
        )
        raw_output = v.compile_output_path(
            raw_arguments,
            working_directory=directory,
            context=entry_context,
        )
        if raw_output not in target_arguments:
            continue
        arguments = v.expand_response_arguments(
            raw_arguments,
            working_directory=directory,
            response_files=response_files,
            used_response_files=used_response_files,
            context=entry_context,
        )
        output = v.compile_output_path(
            arguments,
            working_directory=directory,
            context=entry_context,
        )
        require(
            output == raw_output,
            f"{entry_context} response file changes its output",
        )
        require(
            arguments[0] in compiler_tokens,
            f"{entry_context} uses an unconfigured compiler",
        )
        previous = selected.get(output)
        require(
            previous is None or previous[0] == arguments,
            f"{context} compile database has conflicting records for {output}",
        )
        selected.setdefault(output, (arguments, entry_value))
    require(
        set(selected) == set(target_arguments),
        f"{context} compile database does not exactly cover target outputs",
    )
    for output, (arguments, _) in selected.items():
        require(
            v.normalized_compile_arguments(arguments, context)
            == v.normalized_compile_arguments(
                target_arguments[output],
                context,
            ),
            f"{context} has no exact compiler invocation for {output}",
        )
    return [
        selected[output][1]
        for output in sorted(
            selected,
            key=lambda value: str(value).encode("utf-8"),
        )
    ]


def validate_build(
    evidence_dir: Path,
    role: str,
    value: Any,
    *,
    experiment: ExperimentSpec,
    repo: Path,
    evidence_manifest: dict[str, str],
    runtime_hashes: dict[str, str],
    runtime_paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = f"preflight {role} build"
    build = v.expect_mapping(value, context)
    v.expect_exact_keys(
        build,
        {
            "role",
            "engine",
            "switch_states",
            "binary",
            "build_directory",
            "settings",
            "cmake_cache",
            "full_compile_commands",
            "compile_commands",
            "normalized_compile_arguments",
            "normalized_link_arguments",
            "ninja_commands",
            "product_compile_provenance",
            "ninja_idempotence",
            "dependencies",
            "exactness_emitter",
            "exactness_dependencies",
            "c_compiler",
            "compiler",
            "asm_compiler",
            "scan_deps",
            "linker",
            "build_tool",
            "cmake",
            "compiled_sources",
            "response_files",
            "build_input_closure",
        },
        context,
    )
    require(build["role"] == role, f"{context} role differs")
    require(build["engine"] == "infinity", f"{context} engine is not infinity")
    expected_switch_states = experiment.switch_states_for_role(role)
    switch_states = v.expect_mapping(
        build["switch_states"],
        f"{context} switch states",
    )
    v.expect_exact_keys(
        switch_states,
        set(expected_switch_states),
        f"{context} switch states",
    )
    for name, expected in expected_switch_states.items():
        require(
            v.expect_string(
                switch_states[name],
                f"{context} switch state {name}",
            )
            == expected,
            f"{context} switch state {name} differs",
        )
    require(
        Path(v.expect_string(build["build_directory"], f"{context} directory")).is_absolute(),
        f"{context} directory is not absolute",
    )
    build_directory = Path(str(build["build_directory"]))

    binary = v.expect_mapping(build["binary"], f"{context} binary")
    v.expect_exact_keys(
        binary,
        {
            "path",
            "description",
            "sha256",
            "bytes",
            "captured_path",
        },
        f"{context} binary",
    )
    binary_path = v.expect_string(binary["path"], f"{context} binary path")
    require(Path(binary_path).is_absolute(), f"{context} binary path is not absolute")
    expected_build_products = ordered_build_products(
        binary_path,
        build_directory=build_directory,
        context=context,
    )
    require(
        "Mach-O 64-bit executable arm64"
        in v.expect_string(binary["description"], f"{context} description"),
        f"{context} binary is not native arm64",
    )
    binary_digest = v.expect_sha256(binary["sha256"], f"{context} binary sha256")
    require(
        runtime_hashes.get(binary_path) == binary_digest,
        f"{context} binary is absent from runtime artifacts",
    )
    binary_capture = v.validate_capture_reference(
        evidence_dir,
        binary,
        context=f"{context} binary",
    )
    require(
        binary_capture.stat().st_size
        == v.expect_int(binary["bytes"], f"{context} binary bytes", minimum=1),
        f"{context} binary size differs",
    )
    exactness_emitter = v.expect_mapping(
        build["exactness_emitter"],
        f"{context} exactness emitter",
    )
    v.expect_exact_keys(
        exactness_emitter,
        {
            "product_index",
            "product_name",
            "path",
            "description",
            "sha256",
            "bytes",
            "captured_path",
        },
        f"{context} exactness emitter",
    )
    require(
        exactness_emitter["product_index"] == 1
        and exactness_emitter["product_name"] == "exactness-emitter",
        f"{context} exactness emitter product binding differs",
    )
    exactness_path = v.expect_string(
        exactness_emitter["path"],
        f"{context} exactness emitter path",
    )
    require(
        exactness_path
        == str(expected_build_products[1]["expected_output"]),
        f"{context} exactness emitter path differs from product 1",
    )
    require(
        "Mach-O 64-bit executable arm64"
        in v.expect_string(
            exactness_emitter["description"],
            f"{context} exactness emitter description",
        ),
        f"{context} exactness emitter is not native arm64",
    )
    exactness_digest = v.expect_sha256(
        exactness_emitter["sha256"],
        f"{context} exactness emitter sha256",
    )
    require(
        runtime_hashes.get(exactness_path) == exactness_digest,
        f"{context} exactness emitter is absent from runtime artifacts",
    )
    exactness_capture = v.validate_capture_reference(
        evidence_dir,
        exactness_emitter,
        context=f"{context} exactness emitter",
    )
    require(
        exactness_capture.stat().st_size
        == v.expect_int(
            exactness_emitter["bytes"],
            f"{context} exactness emitter bytes",
            minimum=1,
        ),
        f"{context} exactness emitter size differs",
    )

    cache_record = v.expect_mapping(build["cmake_cache"], f"{context} cache")
    v.expect_exact_keys(
        cache_record,
        {"captured_path", "sha256"},
        f"{context} cache",
    )
    cache_path = v.validate_capture_reference(
        evidence_dir,
        cache_record,
        context=f"{context} cache",
        expected_path=f"{role}-CMakeCache.txt",
    )
    cache = v.parse_cmake_cache(
        v._read_text(cache_path, f"{context} cache"),
        f"{context} cache",
    )
    settings = v.expect_mapping(build["settings"], f"{context} settings")
    expected_setting_keys = set(build_setting_names(experiment))
    v.expect_exact_keys(settings, expected_setting_keys, f"{context} settings")
    require(
        settings
        == {
            name: cache.get(name)
            for name in build_setting_names(experiment)
        },
        f"{context} settings differ from the captured cache",
    )
    validate_production_build_settings(settings, context=context)
    require(settings["CMAKE_BUILD_TYPE"] == "Release", f"{context} is not Release")
    require(settings["CMAKE_OSX_ARCHITECTURES"] == "arm64", f"{context} is not arm64")
    require(
        settings["CMAKE_OSX_DEPLOYMENT_TARGET"] == "14.0",
        f"{context} deployment target differs",
    )
    require(
        settings["CMAKE_EXPORT_COMPILE_COMMANDS"] == "ON",
        f"{context} does not export compile commands",
    )
    require(
        settings["INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION"] == "ON",
        f"{context} does not build the production D0 target",
    )
    require(
        settings["ENABLE_JEMALLOC"] == "OFF",
        f"{context} does not use system malloc",
    )
    require(
        settings["VCPKG_MANIFEST_INSTALL"] == "OFF",
        f"{context} enables vcpkg manifest installation",
    )
    require(
        isinstance(settings["INFINITY_BUILD_TIME_OVERRIDE"], str)
        and settings["INFINITY_BUILD_TIME_OVERRIDE"] != "",
        f"{context} has no reproducible build-time override",
    )
    validate_cmake_home_directory(settings, repo=repo, label=context)
    for name in (
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_ASM_COMPILER",
        "CMAKE_LINKER",
        "CMAKE_MAKE_PROGRAM",
        "CMAKE_HOME_DIRECTORY",
        "CMAKE_TOOLCHAIN_FILE",
        "VCPKG_TARGET_TRIPLET",
        "VCPKG_HOST_TRIPLET",
        "VCPKG_OVERLAY_PORTS",
        "VCPKG_OVERLAY_TRIPLETS",
        "VCPKG_INSTALLED_DIR",
    ):
        require(
            isinstance(settings[name], str) and settings[name] != "",
            f"{context} setting {name} is missing",
    )
    for name, value in expected_switch_states.items():
        require(
            cache.get(name) == value
            and settings[name] == value
            and switch_states[name] == value,
            f"{context} switch {name} is not {value}",
        )

    full_compile_reference = v.expect_mapping(
        build["full_compile_commands"],
        f"{context} full compile database",
    )
    v.expect_exact_keys(
        full_compile_reference,
        {"captured_path", "sha256", "entries"},
        f"{context} full compile database",
    )
    full_compile_path = v.validate_capture_reference(
        evidence_dir,
        full_compile_reference,
        context=f"{context} full compile database",
        expected_path=f"{role}-full-compile_commands.json",
    )
    full_compile_entries = v.expect_list(
        v.read_json(full_compile_path, f"{context} full compile database"),
        f"{context} full compile database",
    )
    require(full_compile_entries, f"{context} full compile database is empty")
    require(
        v.expect_int(
            full_compile_reference["entries"],
            f"{context} full compile entry count",
            minimum=1,
        )
        == len(full_compile_entries),
        f"{context} full compile entry count differs",
    )

    compile_reference = v.expect_mapping(
        build["compile_commands"],
        f"{context} compile database",
    )
    v.expect_exact_keys(
        compile_reference,
        {"captured_path", "sha256", "entries"},
        f"{context} compile database",
    )
    compile_path = v.validate_capture_reference(
        evidence_dir,
        compile_reference,
        context=f"{context} compile database",
        expected_path=f"{role}-compile_commands.json",
    )
    compile_entries = v.expect_list(
        v.read_json(compile_path, f"{context} compile database"),
        f"{context} compile database",
    )
    require(
        v.expect_int(
            compile_reference["entries"],
            f"{context} compile entry count",
            minimum=1,
        )
        == len(compile_entries),
        f"{context} compile entry count differs",
    )
    full_compile_identities = {
        json.dumps(entry, sort_keys=True, separators=(",", ":"))
        for entry in full_compile_entries
    }
    require(
        len(full_compile_identities) == len(full_compile_entries),
        f"{context} full compile database contains duplicate entries",
    )
    response_files = v.validate_response_file_records(
        evidence_dir,
        build["response_files"],
        f"{context} response files",
    )
    used_response_files: set[tuple[str, str]] = set()
    product_provenance = v.expect_list(
        build["product_compile_provenance"],
        f"{context} product compile provenance",
    )
    require(
        len(product_provenance) == len(expected_build_products),
        f"{context} product compile provenance count differs",
    )
    product_compile_sets: list[dict[str, Any]] = []
    product_command_details: list[dict[str, Any]] = []
    independent_link_arguments: list[list[str]] = []
    for index, (value, product) in enumerate(
        zip(
            product_provenance,
            expected_build_products,
        )
    ):
        product_context = f"{context} product compile provenance {index}"
        record = v.expect_mapping(value, product_context)
        v.expect_exact_keys(
            record,
            {
                "name",
                "requested_target",
                "expected_output",
                "compile_entries",
                "ninja_commands",
            },
            product_context,
        )
        product_name = str(product["name"])
        require(
            record["name"] == product_name
            and record["requested_target"]
            == str(product["requested_target"])
            and record["expected_output"]
            == str(product["expected_output"]),
            f"{product_context} product identity differs",
        )
        command_record = v.expect_mapping(
            record["ninja_commands"],
            f"{product_context} Ninja commands",
        )
        v.expect_exact_keys(
            command_record,
            {"command", "captured_path", "sha256"},
            f"{product_context} Ninja commands",
        )
        command_path = v.validate_capture_reference(
            evidence_dir,
            command_record,
            context=f"{product_context} Ninja commands",
            expected_path=f"{role}-{product_name}-ninja-commands.txt",
        )
        command_text = v._read_text(
            command_path,
            f"{product_context} Ninja commands",
        )
        product_entries = target_compile_entries(
            full_compile_entries,
            command_text,
            cache=cache,
            build_directory=build_directory,
            response_files=response_files,
            used_response_files=used_response_files,
            context=f"{product_context} Ninja commands",
        )
        require(
            v.expect_int(
                record["compile_entries"],
                f"{product_context} compile entry count",
                minimum=1,
            )
            == len(product_entries),
            f"{product_context} compile entry count differs",
        )
        product_compile_sets.append(
            {
                "name": product_name,
                "compile_entries": product_entries,
            }
        )
        product_links = v.validate_ninja_commands(
            command_text,
            cache=cache,
            build_directory=build_directory,
            context=f"{product_context} Ninja commands",
            response_files=response_files,
            used_response_files=used_response_files,
            products=[
                {
                    "name": product_name,
                    "expected_output": product["expected_output"],
                    "compile_entries": product_entries,
                }
            ],
        )
        require(
            isinstance(product_links, list)
            and len(product_links) == 1
            and isinstance(product_links[0], list),
            f"{product_context} link command is invalid",
        )
        independent_link_arguments.append(product_links[0])
        product_command_details.append(
            {
                "record": command_record,
                "product": product,
            }
        )
    canonical_union = v.canonical_compile_entry_union(
        product_compile_sets,
        combined_entries=compile_entries,
        context=f"{context} compile provenance",
        response_files=response_files,
        used_response_files=used_response_files,
    )
    require(
        canonical_union == compile_entries,
        f"{context} compile database is not the canonical product union",
    )
    require(
        all(
            json.dumps(entry, sort_keys=True, separators=(",", ":"))
            in full_compile_identities
            for entry in canonical_union
        ),
        f"{context} product compile union is not in the full database",
    )
    v.validate_release_compile_inputs(
        cache,
        compile_entries,
        context=context,
        response_files=response_files,
        used_response_files=used_response_files,
    )

    compiled_sources = validate_source_records(
        evidence_dir,
        build["compiled_sources"],
        context=f"{context} compiled sources",
    )
    require(
        v.compile_source_paths(
            compile_entries,
            f"{context} compile database",
        )
        == {str(source["absolute_path"]) for source in compiled_sources},
        f"{context} compiled-source records differ from the compile database",
    )
    build["compiled_sources"] = compiled_sources

    tools: dict[str, dict[str, Any]] = {}
    for tool, cache_key, _ in BUILD_TOOL_SPECS:
        tools[tool] = v.validate_executable_record(
            evidence_dir,
            build[tool],
            f"{context} {tool}",
            runtime_artifacts=runtime_hashes,
            require_configured_path_resolution=True,
        )
        require(
            tools[tool]["configured_path"] == cache.get(cache_key),
            f"{context} {tool} differs from its cache",
        )
    require(
        tools["build_tool"]["version"] == NINJA_VERSION,
        f"{context} Ninja version is unsupported",
    )
    for detail in product_command_details:
        command_record = detail["record"]
        product = detail["product"]
        require(
            command_record["command"]
            == [
                tools["build_tool"]["resolved_path"],
                "-C",
                str(build_directory),
                "-t",
                "commands",
                str(product["requested_target"]),
            ],
            f"{context} {product['name']} Ninja command invocation differs",
        )

    commands_reference = v.expect_mapping(
        build["ninja_commands"],
        f"{context} Ninja commands",
    )
    v.expect_exact_keys(
        commands_reference,
        {"captured_path", "sha256"},
        f"{context} Ninja commands",
    )
    commands_path = v.validate_capture_reference(
        evidence_dir,
        commands_reference,
        context=f"{context} Ninja commands",
        expected_path=f"{role}-ninja-commands.txt",
    )
    require(
        len(expected_build_products) == len(product_compile_sets),
        f"{context} product compile-set count differs",
    )
    command_products = [
        {
            "name": product["name"],
            "expected_output": product["expected_output"],
            "compile_entries": product_set["compile_entries"],
        }
        for product, product_set in zip(
            expected_build_products,
            product_compile_sets,
        )
    ]
    link_argument_sets = v.validate_ninja_commands(
        v._read_text(commands_path, f"{context} Ninja commands"),
        cache=cache,
        build_directory=build_directory,
        context=f"{context} Ninja commands",
        response_files=response_files,
        used_response_files=used_response_files,
        products=command_products,
    )
    require(
        isinstance(link_argument_sets, list)
        and len(link_argument_sets) == len(expected_build_products)
        and all(
            isinstance(arguments, list)
            and all(isinstance(argument, str) for argument in arguments)
            for arguments in link_argument_sets
        ),
        f"{context} ordered link arguments are invalid",
    )
    require(
        link_argument_sets == independent_link_arguments,
        f"{context} combined links differ from independent product captures",
    )

    canonical_arguments = canonical_compile_argument_map(
        compile_entries,
        build_directory=build_directory,
        response_files=response_files,
        used_response_files=used_response_files,
        context=context,
    )
    normalized_reference = v.expect_mapping(
        build["normalized_compile_arguments"],
        f"{context} normalized compile arguments",
    )
    v.expect_exact_keys(
        normalized_reference,
        {"captured_path", "sha256"},
        f"{context} normalized compile arguments",
    )
    normalized_path = v.validate_capture_reference(
        evidence_dir,
        normalized_reference,
        context=f"{context} normalized compile arguments",
        expected_path=f"{role}-normalized-compile-arguments.json",
    )
    require(
        v.read_json(
            normalized_path,
            f"{context} normalized compile arguments",
        )
        == canonical_arguments,
        f"{context} normalized compile arguments were not independently reproduced",
    )
    normalized_link = [
        canonical_link_arguments(
            arguments,
            build_directory=build_directory,
        )
        for arguments in link_argument_sets
    ]
    normalized_link_reference = v.expect_mapping(
        build["normalized_link_arguments"],
        f"{context} normalized link arguments",
    )
    v.expect_exact_keys(
        normalized_link_reference,
        {"captured_path", "sha256"},
        f"{context} normalized link arguments",
    )
    normalized_link_path = v.validate_capture_reference(
        evidence_dir,
        normalized_link_reference,
        context=f"{context} normalized link arguments",
        expected_path=f"{role}-normalized-link-arguments.json",
    )
    require(
        v.read_json(
            normalized_link_path,
            f"{context} normalized link arguments",
        )
        == normalized_link,
        f"{context} normalized link arguments were not independently reproduced",
    )

    require(
        len(expected_build_products)
        == len(product_compile_sets)
        == len(link_argument_sets),
        f"{context} closure product evidence count differs",
    )
    closure = v.validate_build_input_closure(
        evidence_dir,
        build["build_input_closure"],
        capture_prefix=role,
        build_directory=build_directory,
        build_tool_path=tools["build_tool"]["resolved_path"],
        cache=cache,
        products=[
            {
                **product,
                "compile_entries": product_set["compile_entries"],
                "link_arguments": arguments,
            }
            for product, product_set, arguments in zip(
                expected_build_products,
                product_compile_sets,
                link_argument_sets,
            )
        ],
        response_files=response_files,
        used_response_files=used_response_files,
        context=f"{context} build-input closure",
    )
    require(
        closure["products"][1]["name"] == "exactness-emitter"
        and closure["products"][1]["expected_output"] == exactness_path,
        f"{context} closure product 1 does not bind the exactness emitter",
    )
    v.validate_compiled_sources_against_closure(
        {
            str(source["absolute_path"]): source
            for source in compiled_sources
        },
        closure,
        context=context,
    )
    require(
        used_response_files == set(response_files),
        f"{context} contains unreferenced response-file records",
    )

    dependencies = v.expect_mapping(build["dependencies"], f"{context} dependencies")
    v.expect_exact_keys(
        dependencies,
        {"captured_path", "sha256", "identity"},
        f"{context} dependencies",
    )
    dependency_identity = v.expect_list(
        dependencies["identity"],
        f"{context} dependency identity",
    )
    manifest = v.validate_dependency_manifest(
        evidence_dir,
        {
            "captured_path": dependencies["captured_path"],
            "sha256": dependencies["sha256"],
            "count": len(dependency_identity),
        },
        expected_path=f"{role}-dependencies.json",
        expected_binary_path=binary_path,
        expected_binary_sha256=binary_digest,
        runtime_artifacts=runtime_hashes,
        runtime_artifact_paths=runtime_paths,
        context=f"{context} dependencies",
    )
    require(
        dependency_identity == normalized_dependency_identity(manifest),
        f"{context} dependency identity differs from archived Mach-O evidence",
    )
    exactness_dependencies = v.expect_mapping(
        build["exactness_dependencies"],
        f"{context} exactness dependencies",
    )
    v.expect_exact_keys(
        exactness_dependencies,
        {"captured_path", "sha256", "identity"},
        f"{context} exactness dependencies",
    )
    exactness_dependency_identity = v.expect_list(
        exactness_dependencies["identity"],
        f"{context} exactness dependency identity",
    )
    exactness_manifest = v.validate_dependency_manifest(
        evidence_dir,
        {
            "captured_path": exactness_dependencies["captured_path"],
            "sha256": exactness_dependencies["sha256"],
            "count": len(exactness_dependency_identity),
        },
        expected_path=f"{role}-exactness-emitter-dependencies.json",
        expected_binary_path=exactness_path,
        expected_binary_sha256=exactness_digest,
        runtime_artifacts=runtime_hashes,
        runtime_artifact_paths=runtime_paths,
        context=f"{context} exactness dependencies",
    )
    require(
        exactness_dependency_identity
        == normalized_dependency_identity(exactness_manifest),
        f"{context} exactness dependency identity differs from archived evidence",
    )

    recorded_build_tool_path = tools["build_tool"]["resolved_path"]
    validate_build_idempotence(
        evidence_dir,
        build["ninja_idempotence"],
        role=role,
        build_directory=build_directory,
        requested_target=v.ninja_target_for_output(
            Path(binary_path),
            build_directory=build_directory,
            context=context,
        ),
        build_tool_path=recorded_build_tool_path,
        cmake_path=tools["cmake"]["configured_path"],
        closure=closure,
        binary_path=binary_path,
        binary_sha256=binary_digest,
        binary_bytes=v.expect_int(
            binary["bytes"],
            f"{context} binary bytes",
            minimum=1,
        ),
        cache_capture=cache_path,
        compile_capture=full_compile_path,
        link_arguments=link_argument_sets[0],
        response_files=response_files,
        evidence_manifest=evidence_manifest,
    )
    return build, {
        "cache": cache,
        "canonical_compile_arguments": canonical_arguments,
        "canonical_link_arguments": normalized_link,
        "closure_input_identity": closure_input_identity(
            closure,
            build_directory=build_directory,
        ),
        "dependency_identity": normalized_dependency_identity(manifest),
        "exactness_dependency_identity": normalized_dependency_identity(
            exactness_manifest
        ),
        "tools": tools,
    }


def validate_exactness_process_gate(
    evidence_dir: Path,
    value: Any,
    *,
    roles: dict[str, dict[str, Any]],
    timeout_seconds: float,
    expected_witnesses: dict[str, dict[str, int]],
    directory_descriptor: int | None = None,
    directory_identity: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    context = "preflight exactness gate"
    require(
        exactness_v.SCHEMA_VERSION == REQUIRED_EXACTNESS_SCHEMA_VERSION,
        f"{context} authenticated verifier schema differs",
    )
    gate = v.expect_mapping(value, context)
    v.expect_exact_keys(
        gate,
        {
            "schema_version",
            "status",
            "expected_witnesses",
            "execution_order",
            "timeout_seconds_per_run",
            "dataset",
            "runs",
            "comparisons",
            "independent_verifier_record",
            "independent_verification",
        },
        context,
    )
    require(
        v.expect_int(
            gate["schema_version"],
            f"{context} schema version",
            minimum=1,
        )
        == REQUIRED_EXACTNESS_SCHEMA_VERSION
        and gate["status"] == "pass",
        f"{context} status or schema differs",
    )
    require(
        json_value_equal_exact(
            gate["expected_witnesses"],
            expected_witnesses,
        ),
        f"{context} expected_witnesses differ",
    )
    expected_order = [
        "control-0",
        "control-1",
        "treatment-0",
        "treatment-1",
    ]
    require(
        gate["execution_order"] == expected_order,
        f"{context} execution order differs",
    )
    require(
        gate["timeout_seconds_per_run"] == timeout_seconds,
        f"{context} timeout differs",
    )
    require(
        json_value_equal_exact(
            gate["dataset"],
            {
            "path": v.DATASET_FILENAME,
            "sha256": v.DATASET_SHA256,
            "bytes": v.VECTORS * v.DIMENSIONS * v.FLOAT32_BYTES,
            },
        ),
        f"{context} dataset differs",
    )
    runs = v.expect_list(gate["runs"], f"{context} runs")
    require(len(runs) == 4, f"{context} must contain four runs")
    expected_pairs = (
        ("control", 0),
        ("control", 1),
        ("treatment", 0),
        ("treatment", 1),
    )
    artifact_names = (
        "graph",
        "save_to_ptr",
        "witness",
        "stdout",
        "stderr",
    )
    rich_artifacts: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    rich_lifecycles: dict[tuple[str, int], dict[str, Any]] = {}
    emitter_records: dict[str, dict[str, Any]] = {}
    producer_records: dict[tuple[str, int], dict[str, Any]] = {}
    previous_ended_monotonic_ns = 0
    previous_ended_unix_ns = 0
    for sequence, (run_value, expected_pair) in enumerate(
        zip(runs, expected_pairs)
    ):
        role, repetition = expected_pair
        run_context = f"{context} {role}-{repetition}"
        run = v.expect_mapping(run_value, run_context)
        v.expect_exact_keys(
            run,
            {
                "sequence",
                "role",
                "repetition",
                "status",
                "command",
                "environment",
                "started_at_unix_ns",
                "ended_at_unix_ns",
                "started_at_monotonic_ns",
                "resumed_at_monotonic_ns",
                "ended_at_monotonic_ns",
                "emitter",
                "process_evidence",
                "artifacts",
                "producer_verification",
            },
            run_context,
        )
        require(
            v.expect_int(
                run["sequence"],
                f"{run_context} sequence",
                minimum=0,
            )
            == sequence
            and v.expect_string(run["role"], f"{run_context} role") == role
            and v.expect_int(
                run["repetition"],
                f"{run_context} repetition",
                minimum=0,
            )
            == repetition
            and run["status"] == "pass",
            f"{run_context} identity or status differs",
        )
        emitter = v.expect_mapping(run["emitter"], f"{run_context} emitter")
        v.expect_exact_keys(
            emitter,
            {"path", "sha256", "bytes", "identity"},
            f"{run_context} emitter",
        )
        build_emitter = roles[role]["exactness_emitter"]
        emitter_bytes = v.expect_int(
            emitter["bytes"],
            f"{run_context} emitter bytes",
            minimum=1,
        )
        require(
            emitter["path"] == build_emitter["path"]
            and emitter["sha256"] == build_emitter["sha256"]
            and emitter_bytes == build_emitter["bytes"],
            f"{run_context} emitter differs from its role build",
        )
        previous_emitter = emitter_records.setdefault(role, dict(emitter))
        require(
            json_value_equal_exact(emitter, previous_emitter),
            f"{run_context} emitter identity changed across repetitions",
        )
        emitter_identity = v.expect_mapping(
            emitter["identity"],
            f"{run_context} emitter identity",
        )
        v.expect_exact_keys(
            emitter_identity,
            set(STAT_MANIFEST_KEYS),
            f"{run_context} emitter identity",
        )
        for name in STAT_MANIFEST_KEYS:
            v.expect_int(
                emitter_identity[name],
                f"{run_context} emitter {name}",
                minimum=0,
            )
        require(
            stat.S_ISREG(emitter_identity["mode"])
            and emitter_identity["links"] == 1
            and emitter_identity["bytes"] == emitter_bytes,
            f"{run_context} emitter filesystem identity is invalid",
        )
        expected_command = [
            emitter["path"],
            "--dataset-fd",
            "10",
            "--graph-fd",
            "11",
            "--save-to-ptr-fd",
            "12",
            "--execution-evidence-fd",
            "13",
        ]
        require(
            run["command"] == expected_command,
            f"{run_context} command differs",
        )
        require(
            run["environment"] == v.expected_environment(1),
            f"{run_context} environment differs",
        )
        started_unix = v.expect_int(
            run["started_at_unix_ns"],
            f"{run_context} start time",
            minimum=1,
        )
        ended_unix = v.expect_int(
            run["ended_at_unix_ns"],
            f"{run_context} end time",
            minimum=started_unix,
        )
        started_mono = v.expect_int(
            run["started_at_monotonic_ns"],
            f"{run_context} monotonic start",
            minimum=1,
        )
        resumed_mono = v.expect_int(
            run["resumed_at_monotonic_ns"],
            f"{run_context} monotonic resume",
            minimum=started_mono,
        )
        ended_mono = v.expect_int(
            run["ended_at_monotonic_ns"],
            f"{run_context} monotonic end",
            minimum=resumed_mono,
        )
        require(
            ended_unix >= started_unix and ended_mono >= resumed_mono,
            f"{run_context} timestamps are not ordered",
        )
        require(
            started_unix >= previous_ended_unix_ns
            and started_mono >= previous_ended_monotonic_ns,
            f"{run_context} overlaps the previous exactness run",
        )
        require(
            ended_mono - resumed_mono
            <= int(timeout_seconds * 1_000_000_000),
            f"{run_context} elapsed time exceeds the configured timeout",
        )
        previous_ended_unix_ns = ended_unix
        previous_ended_monotonic_ns = ended_mono

        process_evidence = v.expect_mapping(
            run["process_evidence"],
            f"{run_context} process evidence",
        )
        v.expect_exact_keys(
            process_evidence,
            {
                "schema_version",
                "spawn_method",
                "spawn_flags",
                "fixed_descriptors",
                "suspended",
                "lifecycle",
            },
            f"{run_context} process evidence",
        )
        require(
            v.expect_int(
                process_evidence["schema_version"],
                f"{run_context} process evidence schema version",
                minimum=1,
            )
            == exactness_v.PROCESS_EVIDENCE_SCHEMA_VERSION
            and process_evidence["spawn_method"]
            == "darwin-posix-spawn-start-suspended-v1"
            and process_evidence["spawn_flags"]
            == {
                "value": 0x4480,
                "start_suspended": 0x0080,
                "setsid": 0x0400,
                "cloexec_default": 0x4000,
            }
            and process_evidence["fixed_descriptors"]
            == {
                "dataset": 10,
                "graph": 11,
                "save_to_ptr": 12,
                "witness": 13,
            },
            f"{run_context} spawn contract differs",
        )
        suspended = v.expect_mapping(
            process_evidence["suspended"],
            f"{run_context} suspended process",
        )
        v.expect_exact_keys(
            suspended,
            {
                "observed_at_monotonic_ns",
                "supervisor_pid",
                "pid",
                "process_group",
                "session",
                "processes",
                "mapped_macho_matches_held_descriptor",
                "direct_child",
                "no_descendants",
            },
            f"{run_context} suspended process",
        )
        pid = v.expect_int(
            suspended["pid"],
            f"{run_context} PID",
            minimum=1,
        )
        supervisor_pid = v.expect_int(
            suspended["supervisor_pid"],
            f"{run_context} supervisor PID",
            minimum=1,
        )
        observed_at = v.expect_int(
            suspended["observed_at_monotonic_ns"],
            f"{run_context} suspended observation time",
            minimum=started_mono,
        )
        require(
            v.expect_int(
                suspended["process_group"],
                f"{run_context} process group",
                minimum=1,
            )
            == pid
            and v.expect_int(
                suspended["session"],
                f"{run_context} session",
                minimum=1,
            )
            == pid
            and suspended["mapped_macho_matches_held_descriptor"] is True
            and suspended["direct_child"] is True
            and suspended["no_descendants"] is True,
            f"{run_context} suspended topology differs",
        )
        processes = v.expect_list(
            suspended["processes"],
            f"{run_context} suspended processes",
        )
        require(
            len(processes) == 1,
            f"{run_context} suspended process group is not singular",
        )
        process = v.expect_mapping(
            processes[0],
            f"{run_context} suspended child",
        )
        v.expect_exact_keys(
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
            f"{run_context} suspended child",
        )
        for name in (
            "pid",
            "parent_pid",
            "process_group",
            "status_code",
            "start_time_seconds",
            "start_time_microseconds",
            "executable_device",
            "executable_inode",
            "executable_size",
        ):
            v.expect_int(
                process[name],
                f"{run_context} suspended child {name}",
                minimum=0,
            )
        child_pids = v.expect_list(
            process["child_pids"],
            f"{run_context} suspended child PIDs",
        )
        for child_index, child_pid in enumerate(child_pids):
            v.expect_int(
                child_pid,
                f"{run_context} suspended child PID {child_index}",
                minimum=1,
            )
        require(
            process["pid"] == pid
            and process["parent_pid"] == supervisor_pid
            and process["process_group"] == pid
            and process["status"] == "stopped"
            and child_pids == []
            and process["executable_path"] == emitter["path"]
            and process["executable_device"] == emitter_identity["device"]
            and process["executable_inode"] == emitter_identity["inode"]
            and process["executable_size"] == emitter_identity["bytes"],
            f"{run_context} mapped child identity differs",
        )
        require(
            started_mono
            <= observed_at
            <= resumed_mono,
            f"{run_context} suspension timestamp is outside the run",
        )
        lifecycle = v.expect_mapping(
            process_evidence["lifecycle"],
            f"{run_context} lifecycle",
        )
        v.expect_exact_keys(
            lifecycle,
            {
                "schema_version",
                "leader_pid",
                "terminal_observation",
                "pre_reap_quiescence",
                "reap",
            },
            f"{run_context} lifecycle",
        )
        lifecycle_schema = v.expect_int(
            lifecycle["schema_version"],
            f"{run_context} lifecycle schema version",
            minimum=1,
        )
        lifecycle_leader = v.expect_int(
            lifecycle["leader_pid"],
            f"{run_context} lifecycle leader PID",
            minimum=1,
        )
        terminal_lifecycle = v.expect_mapping(
            lifecycle["terminal_observation"],
            f"{run_context} lifecycle terminal observation",
        )
        terminal_observed_at = v.expect_int(
            terminal_lifecycle.get("observed_at_monotonic_ns"),
            f"{run_context} lifecycle terminal observation time",
            minimum=resumed_mono,
        )
        reap_lifecycle = v.expect_mapping(
            lifecycle["reap"],
            f"{run_context} lifecycle reap",
        )
        reap_completed_at = v.expect_int(
            reap_lifecycle.get("completed_at_monotonic_ns"),
            f"{run_context} lifecycle reap completion time",
            minimum=terminal_observed_at,
        )
        require(
            lifecycle_schema == exactness_v.PROCESS_EVIDENCE_SCHEMA_VERSION
            and lifecycle_leader == pid
            and terminal_observed_at <= ended_mono
            and reap_completed_at <= ended_mono,
            f"{run_context} lifecycle is not bound to the suspended run",
        )
        rich_lifecycles[role, repetition] = dict(lifecycle)

        artifact_values = v.expect_mapping(
            run["artifacts"],
            f"{run_context} artifacts",
        )
        v.expect_exact_keys(
            artifact_values,
            set(artifact_names),
            f"{run_context} artifacts",
        )
        validated_artifacts: dict[str, dict[str, Any]] = {}
        for artifact_name in artifact_names:
            artifact_context = f"{run_context} {artifact_name}"
            artifact = v.expect_mapping(
                artifact_values[artifact_name],
                artifact_context,
            )
            v.expect_exact_keys(
                artifact,
                {"captured_path", "sha256", "bytes", "identity"},
                artifact_context,
            )
            artifact_path = v.validate_capture_reference(
                evidence_dir,
                artifact,
                context=artifact_context,
            )
            identity = v.expect_mapping(
                artifact["identity"],
                f"{artifact_context} identity",
            )
            v.expect_exact_keys(
                identity,
                set(STAT_MANIFEST_KEYS),
                f"{artifact_context} identity",
            )
            for name in STAT_MANIFEST_KEYS:
                v.expect_int(
                    identity[name],
                    f"{artifact_context} identity {name}",
                    minimum=0,
                )
            current = artifact_path.stat()
            expected_identity = {
                "device": current.st_dev,
                "inode": current.st_ino,
                "mode": current.st_mode,
                "links": current.st_nlink,
                "bytes": current.st_size,
                "mtime_ns": current.st_mtime_ns,
                "ctime_ns": current.st_ctime_ns,
            }
            require(
                json_value_equal_exact(identity, expected_identity)
                and current.st_size == artifact["bytes"],
                f"{artifact_context} terminal filesystem identity differs",
            )
            validated_artifacts[artifact_name] = dict(artifact)
        rich_artifacts[role, repetition] = validated_artifacts

        producer = v.expect_mapping(
            run["producer_verification"],
            f"{run_context} producer verification",
        )
        v.expect_exact_keys(
            producer,
            {
                "producer_parser",
                "stdout",
                "stderr",
                "graph",
                "save_to_ptr",
                "witness",
            },
            f"{run_context} producer verification",
        )
        require(
            producer["producer_parser"]
            == "tools.apple_silicon.native_hnsw_exactness",
            f"{run_context} producer parser differs",
        )
        producer_records[role, repetition] = dict(producer)
        for artifact_name in ("stdout", "stderr", "graph", "save_to_ptr", "witness"):
            producer_artifact = v.expect_mapping(
                producer[artifact_name],
                f"{run_context} producer {artifact_name}",
            )
            require(
                v.expect_sha256(
                    producer_artifact.get("sha256"),
                    f"{run_context} producer {artifact_name} SHA-256",
                )
                == validated_artifacts[artifact_name]["sha256"]
                and v.expect_int(
                    producer_artifact.get("bytes"),
                    f"{run_context} producer {artifact_name} bytes",
                    minimum=0,
                )
                == validated_artifacts[artifact_name]["bytes"],
                f"{run_context} producer {artifact_name} identity differs",
            )

    independent_record = v.expect_mapping(
        gate["independent_verifier_record"],
        f"{context} independent verifier record",
    )
    v.expect_exact_keys(
        independent_record,
        {"schema_version", "expected_witnesses", "dataset", "roles"},
        f"{context} independent verifier record",
    )
    require(
        v.expect_int(
            independent_record["schema_version"],
            f"{context} independent verifier schema version",
            minimum=1,
        )
        == REQUIRED_EXACTNESS_SCHEMA_VERSION,
        f"{context} independent verifier schema differs",
    )
    require(
        json_value_equal_exact(
            independent_record["expected_witnesses"],
            expected_witnesses,
        ),
        f"{context} independent expected_witnesses differ",
    )
    independent_roles = v.expect_mapping(
        independent_record["roles"],
        f"{context} independent roles",
    )
    v.expect_exact_keys(
        independent_roles,
        {"control", "treatment"},
        f"{context} independent roles",
    )
    for role in ("control", "treatment"):
        independent_role = v.expect_mapping(
            independent_roles[role],
            f"{context} independent {role}",
        )
        v.expect_exact_keys(
            independent_role,
            {"runs"},
            f"{context} independent {role}",
        )
        minimal_runs = v.expect_list(
            independent_role["runs"],
            f"{context} independent {role} runs",
        )
        require(
            len(minimal_runs) == 2,
            f"{context} independent {role} run count differs",
        )
        for ordinal, minimal_value in enumerate(minimal_runs, 1):
            minimal = v.expect_mapping(
                minimal_value,
                f"{context} independent {role} run {ordinal}",
            )
            v.expect_exact_keys(
                minimal,
                {
                    "ordinal",
                    "lifecycle",
                    "stdout",
                    "stderr",
                    "graph",
                    "save_to_ptr",
                    "witness",
                },
                f"{context} independent {role} run {ordinal}",
            )
            require(
                v.expect_int(
                    minimal["ordinal"],
                    f"{context} independent {role} ordinal",
                    minimum=1,
                )
                == ordinal,
                f"{context} independent {role} run identity differs",
            )
            require(
                json_value_equal_exact(
                    minimal["lifecycle"],
                    rich_lifecycles[role, ordinal - 1],
                ),
                f"{context} independent {role} lifecycle differs",
            )
            for artifact_name in artifact_names:
                rich = rich_artifacts[role, ordinal - 1][artifact_name]
                require(
                    json_value_equal_exact(
                        minimal[artifact_name],
                        {
                            "path": rich["captured_path"],
                            "sha256": rich["sha256"],
                            "bytes": rich["bytes"],
                        },
                    ),
                    f"{context} independent {role} {artifact_name} differs",
                )
    (
        independent_summary,
        independent_producer_semantics,
    ) = exactness_v.validate_exactness_gate_with_producer_semantics(
        evidence_dir,
        independent_record,
        expected_witnesses=expected_witnesses,
        directory_descriptor=directory_descriptor,
        directory_identity=directory_identity,
    )
    for role, repetition in expected_pairs:
        require(
            json_value_equal_exact(
                producer_records[role, repetition],
                independent_producer_semantics[role][repetition],
            ),
            f"{context} {role}-{repetition} producer semantics differ "
            "from independent reconstruction",
        )
    require(
        json_value_equal_exact(
            gate["independent_verification"], independent_summary
        ),
        f"{context} archived independent verification differs",
    )
    comparisons = v.expect_mapping(
        gate["comparisons"],
        f"{context} comparisons",
    )
    v.expect_exact_keys(
        comparisons,
        {
            "within_role_byte_determinism",
            "cross_role_stdout_byte_equal",
            "cross_role_graph_byte_equal",
            "cross_role_save_to_ptr_byte_equal",
            "cross_role_witness_byte_equal",
            "cross_role_witness_relation_matches_expected",
            "graph_sha256",
            "save_to_ptr_sha256",
            "stdout_sha256",
            "control_witness_sha256",
            "treatment_witness_sha256",
        },
        f"{context} comparisons",
    )
    expected_witnesses_equal = (
        expected_witnesses["control"] == expected_witnesses["treatment"]
    )
    require(
        all(
            comparisons[name] is True
            for name in (
                "within_role_byte_determinism",
                "cross_role_stdout_byte_equal",
                "cross_role_graph_byte_equal",
                "cross_role_save_to_ptr_byte_equal",
                "cross_role_witness_relation_matches_expected",
            )
        )
        and comparisons["cross_role_witness_byte_equal"]
        is expected_witnesses_equal
        and independent_summary["expected_witnesses"]
        == expected_witnesses
        and independent_summary["cross_role_witness_byte_equal"]
        is expected_witnesses_equal
        and comparisons["graph_sha256"]
        == independent_summary["graph"]["sha256"]
        and comparisons["save_to_ptr_sha256"]
        == independent_summary["save_to_ptr"]["sha256"]
        and comparisons["stdout_sha256"]
        == rich_artifacts["control", 0]["stdout"]["sha256"]
        and comparisons["control_witness_sha256"]
        == rich_artifacts["control", 0]["witness"]["sha256"]
        and comparisons["treatment_witness_sha256"]
        == rich_artifacts["treatment", 0]["witness"]["sha256"],
        f"{context} comparison summary differs",
    )
    return independent_summary


def validate_preflight(
    evidence_dir: Path,
    value: Any,
    *,
    dataset: dict[str, Any],
    heldout: dict[str, Any],
    evidence_manifest: dict[str, str],
    schedule_sha256: str,
    expected_mode: str,
    authenticated_context: dict[str, Any] | None = None,
) -> tuple[
    ExperimentSpec,
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, str],
    float,
]:
    preflight = v.expect_mapping(value, "preflight.json")
    causal_schema_version = v.expect_int(
        preflight.get("causal_schema_version"),
        "preflight causal schema version",
        minimum=1,
    )
    require(
        causal_schema_version != ARCHIVED_CAUSAL_SCHEMA_VERSION,
        "Archived causal schema 15 must use its manifest-captured verifier",
    )
    require(
        causal_schema_version == CAUSAL_SCHEMA_VERSION,
        "Unsupported causal schema version",
    )
    v.expect_exact_keys(
        preflight,
        {
            "causal_schema_version",
            "experiment",
            "expected_witnesses",
            "scope",
            "execution_mode",
            "captured_at",
            "repo",
            "runner",
            "verifier",
            "local_python_modules",
            "host",
            "environment_control",
            "dataset",
            "heldout",
            "schedule",
            "campaign_binding",
            "roles",
            "causal_contrast",
            "exactness_gate",
            "runtime_artifact_hashes",
            "runtime_artifact_snapshots",
            "source_provenance",
            "protocol",
        },
        "preflight.json",
    )
    experiment = validate_experiment(preflight["experiment"], "preflight experiment")
    expected_witnesses = validate_expected_witnesses(
        preflight["expected_witnesses"],
        experiment=experiment,
        context="preflight expected_witnesses",
    )
    require(preflight["scope"] == experiment.scope, "Preflight scope differs")
    require(preflight["execution_mode"] == expected_mode, "Preflight mode differs")
    require(
        v.expect_string(preflight["captured_at"], "preflight captured_at").endswith("Z"),
        "Preflight timestamp is not UTC",
    )
    repo = v.expect_string(preflight["repo"], "preflight repo")
    require(
        Path(repo).is_absolute(),
        "Preflight repository path is not absolute",
    )
    repo_path = canonical_repo(Path(repo), label="preflight")

    runner = v.expect_mapping(preflight["runner"], "preflight runner")
    v.expect_exact_keys(
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
    runner_capture = validate_capture(
        evidence_dir,
        runner,
        context="preflight runner",
        expected_path=RUNNER_CAPTURE_FILENAME,
    )
    verifier_value = v.expect_mapping(preflight["verifier"], "preflight verifier")
    v.expect_exact_keys(
        verifier_value,
        {"source_path", "captured_path", "sha256", "bytes"},
        "preflight verifier",
    )
    verifier_capture = validate_capture(
        evidence_dir,
        verifier_value,
        context="preflight verifier",
        expected_path=VERIFIER_CAPTURE_FILENAME,
    )
    if authenticated_context is not None:
        require(
            verifier_capture["sha256"]
            == authenticated_context["verifier"]["sha256"]
            and verifier_capture["bytes"]
            == authenticated_context["verifier"]["bytes"],
            "Authenticated verifier differs from the preflight capture",
        )
    require(
        Path(v.expect_string(runner["source_path"], "runner source path")).is_absolute(),
        "Runner source path is not absolute",
    )
    require(
        Path(
            v.expect_string(
                verifier_capture["source_path"],
                "verifier source path",
            )
        ).is_absolute(),
        "Verifier source path is not absolute",
    )
    require(
        runner["source_path"]
        == str(Path(repo) / "tools/apple_silicon/native_hnsw_batch4_causal.py"),
        "Runner source path differs from the recorded repository",
    )
    require(
        verifier_capture["source_path"]
        == str(
            Path(repo)
            / "tools/apple_silicon/verify_native_hnsw_batch4_causal.py"
        ),
        "Verifier source path differs from the recorded repository",
    )
    module_values = v.expect_list(
        preflight["local_python_modules"],
        "preflight local Python modules",
    )
    require(
        len(module_values) == len(LOCAL_PYTHON_MODULES),
        "Preflight local Python module count differs",
    )
    local_modules: list[dict[str, Any]] = []
    for index, (value, expected) in enumerate(
        zip(module_values, LOCAL_PYTHON_MODULES)
    ):
        module, capture_name, relative_source = expected
        context = f"preflight local Python module {index}"
        record = v.expect_mapping(value, context)
        v.expect_exact_keys(
            record,
            {
                "module",
                "source_path",
                "captured_path",
                "sha256",
                "bytes",
            },
            context,
        )
        require(record["module"] == module, f"{context} name differs")
        capture = validate_capture(
            evidence_dir,
            record,
            context=context,
            expected_path=capture_name,
        )
        require(
            capture["source_path"] == str(Path(repo) / relative_source),
            f"{context} source path differs",
        )
        local_modules.append(capture)
    if authenticated_context is not None:
        helper_captures = [
            module
            for module in local_modules
            if module["captured_path"]
            == authenticated_context["helper"]["filename"]
        ]
        require(
            len(helper_captures) == 1
            and helper_captures[0]["sha256"]
            == authenticated_context["helper"]["sha256"]
            and helper_captures[0]["bytes"]
            == authenticated_context["helper"]["bytes"],
            "Authenticated helper differs from the preflight capture",
        )
        exactness_helper_captures = [
            module
            for module in local_modules
            if module["captured_path"]
            == authenticated_context["exactness_helper"]["filename"]
        ]
        require(
            len(exactness_helper_captures) == 1
            and exactness_helper_captures[0]["sha256"]
            == authenticated_context["exactness_helper"]["sha256"]
            and exactness_helper_captures[0]["bytes"]
            == authenticated_context["exactness_helper"]["bytes"],
            "Authenticated exactness helper differs from the preflight capture",
        )

    runtime_value = v.expect_mapping(
        preflight["runtime_artifact_hashes"],
        "runtime artifact hashes",
    )
    runtime_hashes: dict[str, str] = {}
    for path_value, digest_value in runtime_value.items():
        path = v.expect_string(path_value, "runtime artifact path")
        require(Path(path).is_absolute(), f"Runtime artifact is not absolute: {path}")
        runtime_hashes[path] = v.expect_sha256(
            digest_value,
            f"runtime artifact {path}",
        )
    require(
        runtime_hashes.get(runner["source_path"]) == runner["sha256"],
        "Runtime artifacts omit the runner",
    )
    require(
        runtime_hashes.get(verifier_capture["source_path"])
        == verifier_capture["sha256"],
        "Runtime artifacts omit the verifier",
    )
    for module in local_modules:
        require(
            runtime_hashes.get(module["source_path"]) == module["sha256"],
            f"Runtime artifacts omit local module {module['module']}",
        )
    v.validate_executable_record(
        evidence_dir,
        runner["python_executable"],
        "runner Python",
        runtime_artifacts=runtime_hashes,
    )
    runtime_paths = v.validate_runtime_artifact_snapshots(
        evidence_dir,
        preflight["runtime_artifact_snapshots"],
        expected_hashes=runtime_hashes,
    )

    expected_dataset = {
        **dataset,
        "record_path": "dataset.json",
        "record_sha256": v.sha256_file(evidence_dir / "dataset.json"),
    }
    require(preflight["dataset"] == expected_dataset, "Preflight dataset differs")
    expected_heldout = {
        **heldout,
        "record_path": v.HELDOUT_MANIFEST_FILENAME,
        "record_sha256": v.sha256_file(
            evidence_dir / v.HELDOUT_MANIFEST_FILENAME
        ),
    }
    require(preflight["heldout"] == expected_heldout, "Preflight held-out data differs")
    schedule_record = v.expect_mapping(preflight["schedule"], "preflight schedule")
    require(
        schedule_record
        == {
            "path": "schedule.json",
            "sha256": schedule_sha256,
            "members": 18,
            "pair_plan": [
                {
                    "pair": pair,
                    "phase": phase,
                    "participants": participants,
                    "order": "/".join(roles),
                }
                for pair, phase, participants, roles in PAIR_PLAN
            ],
        },
        "Preflight schedule record differs",
    )

    protocol = v.expect_mapping(preflight["protocol"], "preflight protocol")
    required_protocol = {
        "audit_sidecar_schema_version": v.AUDIT_SIDECAR_SCHEMA_VERSION,
        "query_benchmark": v.expected_query_protocol(),
        "attestation_schema_version": v.ATTESTATION_SCHEMA_VERSION,
        "attestation_barrier_phases": list(v.ATTESTATION_BARRIER_PHASES),
        "index_timing_binding_schema_version": (
            v.INDEX_TIMING_BINDING_SCHEMA_VERSION
        ),
        "index_timing_maximum_boundary_overhead_ns": (
            v.INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
        ),
        "process_evidence_schema_version": v.PROCESS_EVIDENCE_SCHEMA_VERSION,
        "exactness_schema_version": REQUIRED_EXACTNESS_SCHEMA_VERSION,
        "performance_measurement_scope": PERFORMANCE_MEASUREMENT_SCOPE,
        "time_wrapper_path": "/usr/bin/time",
        "heldout_query_count": v.HELDOUT_QUERY_COUNT,
        "heldout_query_seed": v.HELDOUT_QUERY_SEED,
        "recall_points": [{"k": k, "ef": ef} for k, ef in v.RECALL_POINTS],
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
        "idle_window_seconds": v.IDLE_WINDOW_SECONDS,
        "idle_sample_count": v.IDLE_SAMPLE_COUNT,
        "idle_to_launch_max_ns": v.IDLE_TO_LAUNCH_MAX_NS,
        "server_ports": v.SERVER_PORTS,
        "power_policy": v.POWER_POLICY,
        "battery_minimum_percent": v.BATTERY_MINIMUM_PERCENT,
        "global_swap_policy": v.GLOBAL_SWAP_POLICY,
    }
    v.expect_exact_keys(
        protocol,
        {
            *required_protocol,
            "idle_minimum_percent",
            "idle_policy",
            "member_timeout_seconds",
            "idle_timeout_seconds",
            "exactness_timeout_seconds_per_run",
        },
        "preflight protocol",
    )
    for key, expected in required_protocol.items():
        require(protocol.get(key) == expected, f"Preflight protocol {key} differs")
    idle_minimum = v.expect_number(
        protocol["idle_minimum_percent"],
        "preflight idle minimum",
    )
    require(
        v.DEVELOPMENT_IDLE_MINIMUM_FLOOR_PERCENT
        <= idle_minimum
        <= v.IDLE_MINIMUM_PERCENT,
        "Preflight idle minimum is outside the permitted range",
    )
    require(
        protocol["idle_policy"] == v.idle_policy(idle_minimum),
        "Preflight idle policy differs",
    )
    require(
        v.expect_number(protocol["member_timeout_seconds"], "member timeout") > 0,
        "Member timeout is not positive",
    )
    require(
        protocol["exactness_timeout_seconds_per_run"]
        == protocol["member_timeout_seconds"],
        "Exactness timeout differs from the member timeout",
    )
    require(
        v.expect_number(protocol["idle_timeout_seconds"], "idle timeout")
        >= v.IDLE_WINDOW_SECONDS,
        "Idle timeout is too short",
    )

    roles_value = v.expect_mapping(preflight["roles"], "preflight roles")
    v.expect_exact_keys(roles_value, {"control", "treatment"}, "preflight roles")
    roles: dict[str, dict[str, Any]] = {}
    build_details: dict[str, dict[str, Any]] = {}
    for role in ("control", "treatment"):
        roles[role], build_details[role] = validate_build(
            evidence_dir,
            role,
            roles_value[role],
            experiment=experiment,
            repo=repo_path,
            evidence_manifest=evidence_manifest,
            runtime_hashes=runtime_hashes,
            runtime_paths=runtime_paths,
        )
    require(
        roles["control"]["binary"]["path"] != roles["treatment"]["binary"]["path"],
        "Role binary paths are identical",
    )
    require(
        roles["control"]["binary"]["sha256"]
        != roles["treatment"]["binary"]["sha256"],
        "Role binary hashes are identical",
    )
    require(
        roles["control"]["exactness_emitter"]["path"]
        != roles["treatment"]["exactness_emitter"]["path"],
        "Role exactness emitter paths are identical",
    )
    require(
        roles["control"]["exactness_emitter"]["sha256"]
        != roles["treatment"]["exactness_emitter"]["sha256"],
        "Role exactness emitter hashes are identical",
    )
    for setting in (
        *SHARED_BUILD_SETTINGS,
        *(name for name, _ in experiment.held_switches),
    ):
        require(
            roles["control"]["settings"][setting]
            == roles["treatment"]["settings"][setting],
            f"Role builds differ in {setting}",
        )
    require(
        source_identity(roles["control"]) == source_identity(roles["treatment"]),
        "Role compiled-source identities differ",
    )
    require(
        build_details["control"]["dependency_identity"]
        == build_details["treatment"]["dependency_identity"],
        "Role dependency identities differ",
    )
    require(
        build_details["control"]["exactness_dependency_identity"]
        == build_details["treatment"]["exactness_dependency_identity"],
        "Role exactness dependency identities differ",
    )
    validate_exactness_process_gate(
        evidence_dir,
        preflight["exactness_gate"],
        roles=roles,
        timeout_seconds=float(
            protocol["exactness_timeout_seconds_per_run"]
        ),
        expected_witnesses=expected_witnesses,
        directory_descriptor=(
            authenticated_context["directory_descriptor"]
            if authenticated_context is not None
            else None
        ),
        directory_identity=(
            authenticated_context["directory_identity"]
            if authenticated_context is not None
            else None
        ),
    )
    validate_toolchain_identity(
        build_details["control"]["tools"],
        build_details["treatment"]["tools"],
    )
    campaign_binding = validate_campaign_binding(
        preflight["campaign_binding"],
        dataset_sha256=dataset["sha256"],
        schedule_sha256=schedule_sha256,
        roles=roles,
    )

    control_arguments = build_details["control"]["canonical_compile_arguments"]
    treatment_arguments = build_details["treatment"]["canonical_compile_arguments"]
    require(
        set(control_arguments) == set(treatment_arguments),
        "Role compile-source sets differ",
    )
    control_scans = validate_compile_macro_states(
        control_arguments,
        experiment=experiment,
        role="control",
        repo=repo_path,
    )
    treatment_scans = validate_compile_macro_states(
        treatment_arguments,
        experiment=experiment,
        role="treatment",
        repo=repo_path,
    )
    varying_sources = {
        switch.name: [
            canonical_source(repo_path, suffix)
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
            f"Role compile arguments differ beyond declared varying switches for {source}",
        )
    require(
        build_details["control"]["canonical_link_arguments"]
        == build_details["treatment"]["canonical_link_arguments"],
        "Role link arguments differ beyond build-directory paths",
    )
    require(
        build_details["control"]["closure_input_identity"]
        == build_details["treatment"]["closure_input_identity"],
        "Role build-input closures differ outside derived outputs",
    )

    contrast = v.expect_mapping(preflight["causal_contrast"], "causal contrast")
    expected_contrast = {
        "experiment": experiment.name,
        "varying_switches": [
            switch.record() for switch in experiment.varying_switches
        ],
        "held_switches": [
            {"name": name, "value": value}
            for name, value in experiment.held_switches
        ],
        "control_switch_states": experiment.switch_states_for_role("control"),
        "treatment_switch_states": experiment.switch_states_for_role(
            "treatment"
        ),
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
    require(
        contrast == expected_contrast,
        "Causal contrast differs from independent reconstruction",
    )
    effective_arguments = v.expect_mapping(
        runner["effective_arguments"],
        "preflight runner effective arguments",
    )
    v.expect_exact_keys(
        effective_arguments,
        {
            "repo",
            "control_binary",
            "treatment_binary",
            "experiment",
            "output_directory",
            "idle_minimum_percent",
            "idle_timeout_seconds",
            "member_timeout_seconds",
            "preflight_only",
        },
        "preflight runner effective arguments",
    )
    require(
        type(effective_arguments["preflight_only"]) is bool,
        "Preflight runner preflight_only argument is not boolean",
    )
    for name in (
        "idle_minimum_percent",
        "idle_timeout_seconds",
        "member_timeout_seconds",
    ):
        v.expect_number(
            effective_arguments[name],
            f"preflight runner effective {name}",
        )
    require(
        effective_arguments
        == {
            "repo": repo,
            "control_binary": roles["control"]["binary"]["path"],
            "treatment_binary": roles["treatment"]["binary"]["path"],
            "experiment": experiment.name,
            "output_directory": str(evidence_dir),
            "idle_minimum_percent": idle_minimum,
            "idle_timeout_seconds": protocol["idle_timeout_seconds"],
            "member_timeout_seconds": protocol["member_timeout_seconds"],
            "preflight_only": expected_mode == PREFLIGHT_ONLY_MODE,
        },
        "Preflight runner effective arguments differ",
    )
    argv = v.expect_list(runner["argv"], "preflight runner argv")
    invocation_working_directory = v.expect_string(
        runner["invocation_working_directory"],
        "preflight runner working directory",
    )
    require(
        Path(invocation_working_directory).is_absolute()
        and os.path.abspath(invocation_working_directory)
        == invocation_working_directory,
        "Preflight runner working directory is not canonical absolute",
    )
    require(
        parse_runner_argv(
            argv,
            working_directory=invocation_working_directory,
            runner_source_path=v.expect_string(
                runner["source_path"],
                "preflight runner source path",
            ),
        )
        == effective_arguments,
        "Preflight runner effective arguments differ from argv",
    )

    host = v.expect_mapping(preflight["host"], "preflight host")
    v.expect_exact_keys(
        host,
        {"machine", "macos", "kernel", "python", "disk"},
        "preflight host",
    )
    require(host["machine"] == "arm64", "Preflight host is not arm64")
    for name in ("macos", "kernel", "python"):
        require(
            v.expect_string(host[name], f"preflight host {name}") != "",
            f"Preflight host {name} is empty",
        )
    disk = v.expect_mapping(host["disk"], "preflight host disk")
    v.expect_exact_keys(
        disk,
        {
            "total_bytes",
            "used_bytes",
            "free_bytes",
            "minimum_reserve_bytes",
            "declared_worst_case_new_bytes",
            "required_free_bytes",
        },
        "preflight host disk",
    )
    total = v.expect_int(disk["total_bytes"], "preflight disk total", minimum=1)
    used = v.expect_int(disk["used_bytes"], "preflight disk used", minimum=0)
    free = v.expect_int(disk["free_bytes"], "preflight disk free", minimum=0)
    reserve = v.expect_int(
        disk["minimum_reserve_bytes"],
        "preflight disk reserve",
        minimum=1,
    )
    worst_case = v.expect_int(
        disk["declared_worst_case_new_bytes"],
        "preflight disk worst case",
        minimum=1,
    )
    required = v.expect_int(
        disk["required_free_bytes"],
        "preflight disk required",
        minimum=1,
    )
    require(total == used + free, "Preflight disk accounting differs")
    require(required == reserve + worst_case, "Preflight disk requirement differs")
    require(free >= required, "Preflight disk reserve was not met")

    environment = v.expect_mapping(
        preflight["environment_control"],
        "preflight environment",
    )
    v.expect_exact_keys(
        environment,
        {
            "parent_performance_environment",
            "unsafe_parent_environment_policy",
            "tool_environment_template",
            "tools_inherit_parent_environment",
            "child_environment_template",
            "inherits_parent_environment",
        },
        "preflight environment",
    )
    parent_environment = v.expect_mapping(
        environment["parent_performance_environment"],
        "preflight parent performance environment",
    )
    require(
        all(
            isinstance(name, str)
            and name
            and isinstance(value, str)
            for name, value in parent_environment.items()
        ),
        "Preflight parent performance environment is invalid",
    )
    v.validate_parent_environment_policy(
        environment["unsafe_parent_environment_policy"],
        context="preflight unsafe parent environment policy",
    )
    require(
        environment.get("tool_environment_template")
        == v.expected_tool_environment()
        and environment.get("tools_inherit_parent_environment") is False
        and
        environment.get("child_environment_template")
        == v.expected_environment(v.MEASURED_PARTICIPANTS)
        and environment.get("inherits_parent_environment") is False,
        "Preflight child environment differs",
    )
    source_provenance = v.expect_mapping(
        preflight["source_provenance"],
        "source provenance",
    )
    v.expect_exact_keys(
        source_provenance,
        {"git_repository"},
        "source provenance",
    )
    validate_git_repository(
        evidence_dir,
        source_provenance["git_repository"],
        expected_root=repo,
        build_directory=Path(
            v.expect_string(
                roles["control"]["build_directory"],
                "control build directory",
            )
        ),
        source_records=roles["control"]["compiled_sources"],
    )
    return experiment, roles, campaign_binding, runtime_hashes, idle_minimum


def validate_run_schedule_identity(
    record: dict[str, Any],
    member: dict[str, Any],
    *,
    schedule_keys: Sequence[str],
    context: str,
) -> None:
    for name, minimum in (("sequence", 0), ("participants", 1)):
        require(
            type(record.get(name)) is int,
            f"{context} {name} must be a JSON integer",
        )
        v.expect_int(record.get(name), f"{context} {name}", minimum=minimum)
    for name in ("pair", "phase", "pair_order", "role", "engine"):
        v.expect_string(record.get(name), f"{context} {name}")
    require(
        {key: record.get(key) for key in schedule_keys}
        == {key: member[key] for key in schedule_keys},
        f"{context} differs from the frozen schedule",
    )


def validate_runs(
    evidence_dir: Path,
    value: Any,
    *,
    experiment: ExperimentSpec,
    schedule: list[dict[str, Any]],
    dataset: dict[str, Any],
    roles: dict[str, dict[str, Any]],
    campaign_binding: dict[str, Any],
    runtime_hashes: dict[str, str],
    idle_minimum: float,
    truth: dict[str, Any],
) -> list[dict[str, Any]]:
    raw = v.expect_list(value, "runs.json")
    require(len(raw) == len(schedule) == 18, "runs.json must contain 18 members")
    records: list[dict[str, Any]] = []
    schedule_keys = (
        "sequence",
        "pair",
        "phase",
        "pair_order",
        "role",
        "engine",
        "participants",
    )
    for index, (record_value, member) in enumerate(zip(raw, schedule)):
        prefix = f"{index:02d}-{member['pair']}-{member['role']}"
        context = f"run {prefix}"
        record = v.expect_mapping(record_value, context)
        v.expect_exact_keys(record, RUN_RECORD_FIELDS, context)
        validate_run_schedule_identity(
            record,
            member,
            schedule_keys=schedule_keys,
            context=context,
        )
        member_path = evidence_dir / f"{prefix}.json"
        require(member_path.is_file(), f"{context} member JSON is missing")
        require(
            v.read_json(member_path, f"{context} member JSON") == record,
            f"{context} differs from its member JSON",
        )
        require(record["status"] == "pass", f"{context} did not pass")
        require(
            record["experiment"] == experiment.name,
            f"{context} experiment differs",
        )
        require(record["scope"] == experiment.scope, f"{context} scope differs")
        require(record["errors"] == [], f"{context} records errors")
        require(record["engine"] == "infinity", f"{context} engine is not infinity")
        role = str(record["role"])
        participants = v.expect_int(
            record["participants"],
            f"{context} participants",
            minimum=1,
        )
        binary = roles[role]["binary"]
        member_binding = member_campaign_binding(member, campaign_binding)
        require(
            record["binary"]
            == {"path": binary["path"], "sha256": binary["sha256"]},
            f"{context} binary identity differs",
        )
        expected_sidecar = f"{prefix}{v.AUDIT_SIDECAR_SUFFIX}"
        require(
            record["command"]
            == v.expected_command(
                binary["path"],
                dataset["path"],
                participants,
                expected_sidecar,
                campaign_nonce=str(member_binding["campaign_nonce"]),
                schedule_sequence=int(member_binding["schedule_sequence"]),
                role_id=int(member_binding["role_id"]),
                binary_sha256=str(member_binding["binary_sha256"]),
                dataset_sha256=str(member_binding["dataset_sha256"]),
            ),
            f"{context} command differs",
        )
        require(record["working_directory"] == ".", f"{context} working directory differs")
        require(
            record["environment"] == v.expected_environment(participants),
            f"{context} environment differs",
        )
        v.validate_dataset_fingerprint(
            record["dataset_before"],
            dataset=dataset,
            context=f"{context} dataset_before",
        )
        v.validate_dataset_fingerprint(
            record["dataset_after"],
            dataset=dataset,
            context=f"{context} dataset_after",
        )
        v.validate_runtime_artifacts(
            record["runtime_artifacts_before"],
            expected_hashes=runtime_hashes,
            context=f"{context} runtime artifacts before",
        )
        v.validate_runtime_artifacts(
            record["runtime_artifacts_after"],
            expected_hashes=runtime_hashes,
            context=f"{context} runtime artifacts after",
        )
        v.validate_host_resources(
            evidence_dir,
            record,
            power_policy=v.POWER_POLICY,
            battery_minimum_percent=v.BATTERY_MINIMUM_PERCENT,
            global_swap_policy_name=v.GLOBAL_SWAP_POLICY,
            context=context,
            prefix=prefix,
        )
        idle = v.validate_idle(
            evidence_dir,
            record["idle"],
            context=context,
            minimum_idle_percent=idle_minimum,
        )
        delay = v.expect_int(
            record["idle_to_launch_ns"],
            f"{context} idle-to-launch delay",
            minimum=0,
        )
        require(delay <= v.IDLE_TO_LAUNCH_MAX_NS, f"{context} launch delay is too long")
        started_ns = v.expect_int(
            record["started_at_unix_ns"],
            f"{context} start",
            minimum=1,
        )
        ended_ns = v.expect_int(
            record["ended_at_unix_ns"],
            f"{context} end",
            minimum=started_ns,
        )
        require(started_ns >= idle["accepted_at_unix_ns"], f"{context} predates idle gate")
        require(ended_ns >= started_ns, f"{context} ends before it starts")

        fields = v.validate_driver_fields(
            record["fields"],
            engine="infinity",
            participants=int(record["participants"]),
            dataset=dataset,
            context=context,
        )
        expected_binding_fields = {
            "campaign_nonce": str(member_binding["campaign_nonce"]),
            "schedule_sequence": str(member_binding["schedule_sequence"]),
            "role_id": str(member_binding["role_id"]),
            "binary_sha256": str(member_binding["binary_sha256"]),
            "dataset_sha256": str(member_binding["dataset_sha256"]),
        }
        require(
            {
                name: fields[name]
                for name in expected_binding_fields
            }
            == expected_binding_fields,
            f"{context} stdout campaign binding differs from preflight",
        )
        stdout_name = f"{prefix}.stdout"
        stderr_name = f"{prefix}.stderr"
        require(record["stdout_path"] == stdout_name, f"{context} stdout path differs")
        require(record["stderr_path"] == stderr_name, f"{context} stderr path differs")
        stdout_path = evidence_dir / stdout_name
        stderr_path = evidence_dir / stderr_name
        captured_fields, captured_attestation = v.parse_attested_driver_stdout(
            v._read_text(stdout_path, f"{context} stdout"),
            "infinity",
            f"{context} stdout",
        )
        require(captured_fields == fields, f"{context} stdout differs from fields")
        v.expect_normalized_equal(
            record["attestation"],
            captured_attestation,
            f"{context} attestation",
        )
        v.validate_attestation_barriers(
            record["attestation_barriers"],
            process_group=v.expect_int(
                record["pid"],
                f"{context} PID",
                minimum=1,
            ),
            context=context,
        )
        v.validate_benchmark_process_evidence(
            record["benchmark_process_evidence"],
            barriers_value=record["attestation_barriers"],
            process_group=v.expect_int(
                record["pid"],
                f"{context} PID",
                minimum=1,
            ),
            expected_benchmark_path=binary["path"],
            expected_benchmark_bytes=v.expect_int(
                binary["bytes"],
                f"{context} preflight benchmark bytes",
                minimum=1,
            ),
            context=context,
        )
        v.validate_index_timing_binding(
            record["index_timing_binding"],
            barriers_value=record["attestation_barriers"],
            fields=fields,
            engine="infinity",
            context=context,
        )

        sidecar_record = v.expect_mapping(record["audit_sidecar"], f"{context} sidecar")
        v.expect_exact_keys(
            sidecar_record,
            {"path", "sha256", "bytes"},
            f"{context} sidecar",
        )
        sidecar_name, sidecar_path = v.relative_evidence_path(
            evidence_dir,
            sidecar_record["path"],
            f"{context} sidecar path",
        )
        require(sidecar_name == expected_sidecar, f"{context} sidecar name differs")
        data = v.read_audit_sidecar(sidecar_path, f"{context} sidecar")
        digest = hashlib.sha256(data).hexdigest()
        require(
            digest == sidecar_record["sha256"] == fields["audit_sidecar_sha256"],
            f"{context} sidecar hashes disagree",
        )
        require(
            len(data)
            == sidecar_record["bytes"]
            == v.parse_unsigned_field(fields, "audit_sidecar_bytes", context),
            f"{context} sidecar sizes disagree",
        )
        reconstructed, details = v.parse_audit_sidecar(
            data,
            engine="infinity",
            fields=fields,
            truth=truth,
            context=f"{context} sidecar",
            expected_campaign_nonce=str(member_binding["campaign_nonce"]),
            expected_schedule_sequence=int(
                member_binding["schedule_sequence"]
            ),
            expected_role_id=int(member_binding["role_id"]),
            expected_binary_sha256=str(member_binding["binary_sha256"]),
            expected_dataset_sha256=str(member_binding["dataset_sha256"]),
        )
        v.validate_audit_stdout_fields(
            fields,
            reconstructed,
            details,
            engine="infinity",
            context=context,
        )
        v.expect_normalized_equal(record["audit"], reconstructed, f"{context} audit")
        validate_run_execution_witness(
            reconstructed.get("execution_witness"),
            experiment=experiment,
            role=role,
            context=f"{context} execution witness",
        )

        stderr = v._read_text(stderr_path, f"{context} stderr")
        rss = v.expect_int(
            record["process_lifetime_maximum_resident_set_size"],
            f"{context} maximum RSS",
            minimum=1,
        )
        rss_matches = v.MAXIMUM_RSS.findall(stderr)
        require(len(rss_matches) == 1 and int(rss_matches[0]) == rss, f"{context} RSS differs")
        swaps = v.expect_int(record["process_swaps"], f"{context} swaps", minimum=0)
        swap_matches = v.PROCESS_SWAPS.findall(stderr)
        require(
            len(swap_matches) == 1 and int(swap_matches[0]) == swaps == 0,
            f"{context} swap record differs",
        )
        records.append(record)
    validate_run_chronology(records)
    return records


def validate_run_chronology(records: Sequence[dict[str, Any]]) -> None:
    previous_end: int | None = None
    for index, record in enumerate(records):
        started_ns = v.expect_int(
            record.get("started_at_unix_ns"),
            f"run {index} chronology start",
            minimum=1,
        )
        ended_ns = v.expect_int(
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
    return statistics.median(float(value) for value in values)


def relative_mad(values: Sequence[int | float]) -> float:
    return float(exact_relative_mad([exact_fraction(value) for value in values]))


def geometric_mean(values: Sequence[float]) -> float:
    require(values and all(value > 0 and math.isfinite(value) for value in values), "Invalid geometric mean input")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def recall_point_fraction(
    record: dict[str, Any],
    k: int,
    ef: int,
) -> Fraction:
    points = [
        point
        for point in record["audit"]["recall"]["points"]
        if point["k"] == k and point["ef"] == ef
    ]
    require(len(points) == 1, f"Run lacks unique recall k={k}, ef={ef}")
    point = v.expect_mapping(points[0], f"recall k={k}, ef={ef}")
    v.expect_exact_keys(
        point,
        {"k", "ef", "eligible_hits", "possible_hits", "recall"},
        f"recall k={k}, ef={ef}",
    )
    eligible_hits = v.expect_int(
        point.get("eligible_hits"),
        f"recall k={k}, ef={ef} eligible hits",
        minimum=0,
    )
    possible_hits = v.expect_int(
        point.get("possible_hits"),
        f"recall k={k}, ef={ef} possible hits",
        minimum=1,
    )
    require(
        possible_hits == v.HELDOUT_QUERY_COUNT * k
        and eligible_hits <= possible_hits,
        f"Recall hit counts differ k={k}, ef={ef}",
    )
    recall = Fraction(eligible_hits, possible_hits)
    recorded_recall = point.get("recall")
    require(
        type(recorded_recall) is float
        and math.isfinite(recorded_recall)
        and recorded_recall == float(recall),
        f"Recall float differs from hit counts k={k}, ef={ef}",
    )
    return recall


def recall_point(record: dict[str, Any], k: int, ef: int) -> float:
    return float(recall_point_fraction(record, k, ef))


def correctness_semantic_payload(record: dict[str, Any]) -> dict[str, Any]:
    audit = v.expect_mapping(record["audit"], "correctness audit")
    fields = v.expect_mapping(record["fields"], "correctness fields")
    graph = v.expect_mapping(audit["graph"], "correctness graph")
    levels_sha256 = v.expect_sha256(
        graph.get("levels_sha256"),
        "correctness graph level hash",
    )
    returned_results: list[dict[str, Any]] = []
    for k, ef in v.RECALL_POINTS:
        ids_name = f"infinity_returned_ids_ef_{ef}_k_{k}"
        distances_name = f"infinity_returned_distances_sha256_ef_{ef}_k_{k}"
        returned_results.append(
            {
                "k": k,
                "ef": ef,
                "returned_ids": v.expect_string(
                    fields.get(ids_name),
                    f"correctness {ids_name}",
                ),
                "returned_distances_sha256": v.expect_sha256(
                    fields.get(distances_name),
                    f"correctness {distances_name}",
                ),
            }
        )
    query = v.expect_mapping(audit["query"], "correctness query")
    query_checksum = v.expect_int(
        query.get("result_checksum"),
        "correctness query result checksum",
        minimum=0,
    )
    return {
        "graph": graph,
        "levels_sha256": levels_sha256,
        "recall": audit["recall"],
        "returned_results": returned_results,
        "query_result_checksum": query_checksum,
    }


def nearest_rank(samples: Sequence[int], percentile: float) -> int:
    require(samples, "Cannot calculate a percentile from no samples")
    require(
        0.0 < percentile <= 1.0,
        "Nearest-rank percentile must be in (0, 1]",
    )
    require(
        all(type(sample) is int and sample >= 0 for sample in samples),
        "Query latency samples must be nonnegative integers",
    )
    ordered = sorted(samples)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def measured_query_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    context = f"{record['pair']} {record['role']} query"
    query = v.expect_mapping(record["audit"]["query"], context)
    protocol = v.expected_query_protocol()
    metric_fields = {
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
    v.expect_exact_keys(query, {*protocol, *metric_fields}, context)
    require(
        {name: query[name] for name in protocol} == protocol,
        f"{context} protocol differs",
    )
    timed_query_corpus_sha256 = v.expect_sha256(
        query["timed_query_corpus_sha256"],
        f"{context} timed query corpus SHA-256",
    )
    samples = [
        v.expect_int(sample, f"{context} latency sample", minimum=0)
        for sample in v.expect_list(
            query["latency_samples_ns"],
            f"{context} latency samples",
        )
    ]
    latency_validated_operations = v.expect_int(
        query["latency_validated_operations"],
        f"{context} latency validated operations",
        minimum=1,
    )
    latency_validated_result_checksum = v.expect_int(
        query["latency_validated_result_checksum"],
        f"{context} latency validated result checksum",
        minimum=0,
    )
    operations = v.expect_int(
        query["throughput_operations"],
        f"{context} throughput operations",
        minimum=1,
    )
    throughput_validated_operations = v.expect_int(
        query["throughput_validated_operations"],
        f"{context} throughput validated operations",
        minimum=1,
    )
    wall_ns = v.expect_int(
        query["throughput_wall_ns"],
        f"{context} throughput wall time",
        minimum=1,
    )
    throughput_validated_result_checksum = v.expect_int(
        query["throughput_validated_result_checksum"],
        f"{context} throughput validated result checksum",
        minimum=0,
    )
    result_checksum = v.expect_int(
        query["result_checksum"],
        f"{context} result checksum",
        minimum=0,
    )
    per_query_checksum_count = v.expect_int(
        query["per_query_checksum_count"],
        f"{context} per-query checksum count",
        minimum=1,
    )
    per_query_checksums = [
        v.expect_int(
            checksum,
            f"{context} per-query checksum",
            minimum=0,
        )
        for checksum in v.expect_list(
            query["per_query_checksums"],
            f"{context} per-query checksums",
        )
    ]
    require(
        all(
            value <= v.FNV1A64_MASK
            for value in (
                latency_validated_result_checksum,
                throughput_validated_result_checksum,
                result_checksum,
                *per_query_checksums,
            )
        ),
        f"{context} checksum is outside uint64",
    )
    reconstructed = v.query_performance_record(
        timed_query_corpus_sha256,
        samples,
        latency_validated_operations,
        latency_validated_result_checksum,
        operations,
        throughput_validated_operations,
        wall_ns,
        throughput_validated_result_checksum,
        result_checksum,
        per_query_checksums,
    )
    v.expect_normalized_equal(
        query,
        reconstructed,
        f"{context} reconstructed evidence",
    )
    recall = v.expect_mapping(record["audit"]["recall"], f"{context} recall")
    require(
        query["timed_query_corpus_sha256"]
        == v.expect_sha256(
            recall.get("queries_sha256"),
            f"{context} replay query corpus SHA-256",
        ),
        f"{context} timed query corpus differs from replay",
    )
    require(
        len(samples) == v.QUERY_LATENCY_SAMPLE_COUNT,
        f"{context} latency sample count differs",
    )
    require(
        operations == v.QUERY_THROUGHPUT_OPERATIONS,
        f"{context} throughput operation count differs",
    )
    require(
        per_query_checksum_count == v.HELDOUT_QUERY_COUNT,
        f"{context} per-query checksum count differs",
    )
    tps = operations * 1e9 / wall_ns
    p50_ns = nearest_rank(samples, 0.50)
    p95_ns = nearest_rank(samples, 0.95)
    p99_ns = nearest_rank(samples, 0.99)
    v.expect_close(query["tps"], tps, f"{context} recorded TPS")
    v.expect_close(query["qps"], tps, f"{context} recorded QPS")
    require(query["p50_ns"] == p50_ns, f"{context} recorded p50 differs")
    require(query["p95_ns"] == p95_ns, f"{context} recorded p95 differs")
    require(query["p99_ns"] == p99_ns, f"{context} recorded p99 differs")
    require(
        math.isfinite(tps)
        and tps > 0.0
        and 0 < p50_ns <= p95_ns <= p99_ns,
        f"{context} reconstructed metrics are invalid",
    )
    return {
        "tps": tps,
        "p50_ns": p50_ns,
        "p95_ns": p95_ns,
        "p99_ns": p99_ns,
        "recall_at_10": recall_point(
            record,
            v.QUERY_K,
            v.QUERY_EF_SEARCH,
        ),
    }


def measured_query_tps_fraction(record: dict[str, Any]) -> Fraction:
    """Return query TPS exactly from its integer measurement fields."""
    context = f"{record['pair']} {record['role']} query"
    query = v.expect_mapping(record["audit"]["query"], context)
    operations = v.expect_int(
        query["throughput_operations"],
        f"{context} throughput operations",
        minimum=1,
    )
    wall_ns = v.expect_int(
        query["throughput_wall_ns"],
        f"{context} throughput wall time",
        minimum=1,
    )
    require(
        operations == v.QUERY_THROUGHPUT_OPERATIONS,
        f"{context} throughput operation count differs",
    )
    return Fraction(operations * 1_000_000_000, wall_ns)


def recompute(records: list[dict[str, Any]]) -> dict[str, Any]:
    experiment_names = {
        str(record.get("experiment")) for record in records
    }
    require(
        len(experiment_names) == 1
        and next(iter(experiment_names)) in EXPERIMENTS,
        "Run records do not bind one supported experiment",
    )
    experiment = EXPERIMENTS[next(iter(experiment_names))]
    for record in records:
        validate_run_execution_witness(
            record.get("audit", {}).get("execution_witness"),
            experiment=experiment,
            role=str(record.get("role")),
            context=f"{record.get('pair')} {record.get('role')} execution witness",
        )
    measured = [record for record in records if record["phase"] == "measured"]
    require(len(measured) == 12, "Expected 12 measured members")
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    durations: dict[str, list[int]] = {"control": [], "treatment": []}
    for record in measured:
        members = by_pair.setdefault(str(record["pair"]), {})
        role = str(record["role"])
        require(role not in members, f"Duplicate role in {record['pair']}")
        members[role] = record
        durations[role].append(
            v.parse_unsigned_field(record["fields"], "infinity_cold_build_ns", role)
        )
    pairs: list[dict[str, Any]] = []
    for pair_name in ("P1", "P2", "P3", "P4", "P5", "P6"):
        members = by_pair.get(pair_name, {})
        require(set(members) == {"control", "treatment"}, f"Incomplete {pair_name}")
        control_ns = v.parse_unsigned_field(
            members["control"]["fields"],
            "infinity_cold_build_ns",
            pair_name,
        )
        treatment_ns = v.parse_unsigned_field(
            members["treatment"]["fields"],
            "infinity_cold_build_ns",
            pair_name,
        )
        order = str(members["control"]["pair_order"])
        require(order == members["treatment"]["pair_order"], f"{pair_name} order differs")
        pairs.append(
            {
                "pair": pair_name,
                "order": order,
                "control_cold_build_ns": control_ns,
                "treatment_cold_build_ns": treatment_ns,
                "control_vectors_per_second": v.VECTORS * 1e9 / control_ns,
                "treatment_vectors_per_second": v.VECTORS * 1e9 / treatment_ns,
                "treatment_speedup": control_ns / treatment_ns,
            }
        )
    strata: dict[str, Any] = {}
    for order in ("control/treatment", "treatment/control"):
        order_pairs = [pair for pair in pairs if pair["order"] == order]
        speedups = [pair["treatment_speedup"] for pair in order_pairs]
        require(len(speedups) == 3, f"Order stratum {order} is incomplete")
        speedup_product = Fraction(
            math.prod(pair["control_cold_build_ns"] for pair in order_pairs),
            math.prod(pair["treatment_cold_build_ns"] for pair in order_pairs),
        )
        strata[order] = {
            "treatment_speedups": speedups,
            "geometric_mean_speedup": geometric_mean(speedups),
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
            "median_cold_build_ns": median(values),
            "relative_mad": float(exact_duration_relative_mads[role]),
        }
        for role, values in durations.items()
    }
    exact_ratio_mad = exact_relative_mad(
        [
            Fraction(
                pair["control_cold_build_ns"],
                pair["treatment_cold_build_ns"],
            )
            for pair in pairs
        ]
    )
    ratio_mad = float(exact_ratio_mad)

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
    for pair_name in ("P1", "P2", "P3", "P4", "P5", "P6"):
        members = by_pair[pair_name]
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
        # Float TPS is reporting-only; all acceptance gates use these exact
        # operation-count/wall-time ratios.
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
            v.QUERY_K,
            v.QUERY_EF_SEARCH,
        )
        treatment_recall_fraction = recall_point_fraction(
            members["treatment"],
            v.QUERY_K,
            v.QUERY_EF_SEARCH,
        )
        recall_gap_fraction = abs(
            control_recall_fraction - treatment_recall_fraction
        )
        recall_gap = float(recall_gap_fraction)
        query_pairs.append(
            {
                "pair": pair_name,
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
                "tps_nonregression": exact_ratio_at_least(
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
    query_roles: dict[str, dict[str, Any]] = {}
    for role, values in query_values.items():
        query_roles[role] = {
            **values,
            "median_tps": median(values["tps"]),
            "median_p50_ns": median(values["p50_ns"]),
            "median_p95_ns": median(values["p95_ns"]),
            "median_p99_ns": median(values["p99_ns"]),
            "median_recall_at_10": median(values["recall_at_10"]),
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
    exact_query_role_latency_relative_mads = {
        percentile: {
            role: exact_relative_mad(
                [Fraction(int(value)) for value in values[percentile]]
            )
            for role, values in query_values.items()
        }
        for percentile in QUERY_LATENCY_PERCENTILES
    }
    exact_query_paired_latency_ratio_relative_mads = {
        percentile: exact_relative_mad(ratios)
        for percentile, ratios in query_latency_ratio_fractions.items()
    }
    query_latency_variability = {
        percentile: {
            "role_relative_mad": {
                role: float(relative_mad)
                for role, relative_mad in role_mads.items()
            },
            "paired_treatment_over_control_ratio_relative_mad": float(
                exact_query_paired_latency_ratio_relative_mads[percentile]
            ),
        }
        for percentile, role_mads in exact_query_role_latency_relative_mads.items()
    }
    query_role_latency_variability_accepted = {
        percentile: all(
            exact_relative_mad_at_most(
                relative_mad,
                *MAXIMUM_RELATIVE_MAD_FRACTION,
            )
            for relative_mad in role_mads.values()
        )
        for percentile, role_mads in exact_query_role_latency_relative_mads.items()
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
    for k, ef in v.RECALL_POINTS:
        controls: list[float] = []
        treatments: list[float] = []
        paired: list[dict[str, Any]] = []
        deficit_fractions: list[Fraction] = []
        for pair_name in ("P1", "P2", "P3", "P4", "P5", "P6"):
            members = by_pair[pair_name]
            control_fraction = recall_point_fraction(
                members["control"], k, ef
            )
            treatment_fraction = recall_point_fraction(
                members["treatment"], k, ef
            )
            control = float(control_fraction)
            treatment = float(treatment_fraction)
            controls.append(control)
            treatments.append(treatment)
            deficit_fraction = control_fraction - treatment_fraction
            deficit_fractions.append(deficit_fraction)
            deficit = float(deficit_fraction)
            paired.append(
                {
                    "pair": pair_name,
                    "control": control,
                    "treatment": treatment,
                    "treatment_deficit": deficit,
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
                "control_median": median(controls),
                "treatment_median": median(treatments),
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
    graph_valid = all(
        record["audit"]["graph"]["valid"] is True
        and record["audit"]["graph"]["vertex_count"] == v.VECTORS
        and record["audit"]["graph"]["reachable_count"] == v.VECTORS
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
    return {
        "pairs": pairs,
        "strata": strata,
        "estimate": estimate,
        "minimum_effect": minimum_effect,
        "minimum_speedup_comparison_exact": minimum_speedup_comparison_exact,
        "variability": variability,
        "paired_ratio_relative_mad": ratio_mad,
        "query_pairs": query_pairs,
        "query_roles": query_roles,
        "query_paired_tps_ratio_relative_mad": (
            query_paired_tps_ratio_relative_mad
        ),
        "query_latency_variability": query_latency_variability,
        "query_role_latency_variability_accepted": (
            query_role_latency_variability_accepted
        ),
        "query_paired_latency_ratio_variability_accepted": (
            query_paired_latency_ratio_variability_accepted
        ),
        "variability_accepted": all(
            exact_relative_mad_at_most(
                relative_mad,
                *MAXIMUM_RELATIVE_MAD_FRACTION,
            )
            for relative_mad in exact_duration_relative_mads.values()
        ),
        "ratio_variability_accepted": exact_relative_mad_at_most(
            exact_ratio_mad,
            *MAXIMUM_RELATIVE_MAD_FRACTION,
        ),
        "query_tps_variability_accepted": all(
            exact_relative_mad_at_most(
                relative_mad,
                *MAXIMUM_RELATIVE_MAD_FRACTION,
            )
            for relative_mad in exact_query_role_tps_relative_mads.values()
        ),
        "query_ratio_variability_accepted": exact_relative_mad_at_most(
            exact_query_paired_tps_ratio_relative_mad,
            *MAXIMUM_RELATIVE_MAD_FRACTION,
        ),
        "query_tps_nonregression": query_tps_nonregression,
        "query_p50_nonregression": query_p50_nonregression,
        "query_p95_nonregression": query_p95_nonregression,
        "query_p99_nonregression": query_p99_nonregression,
        "query_recall_floor_met": query_recall_floor_met,
        "query_recall_gap_met": query_recall_gap_met,
        "measured_run_execution_witnesses_valid": True,
        "recall_points": recall_points,
        "graph_valid": graph_valid,
        "single_thread_semantic_payload_exact_match": (
            single_thread_semantic_payload_exact_match
        ),
    }


def hypothesis_outcome(
    experiment: ExperimentSpec,
    accepted: bool,
) -> dict[str, Any]:
    return {
        "hypothesis_accepted": accepted,
        "decision": (
            f"accept-{experiment.name}"
            if accepted
            else f"reject-{experiment.name}"
        ),
    }


def validate_summary_outcome(
    summary: dict[str, Any],
    *,
    experiment: ExperimentSpec,
    accepted: bool,
) -> None:
    outcome = hypothesis_outcome(experiment, accepted)
    require(
        summary["status"] == ("pass" if accepted else "fail"),
        "Summary status differs",
    )
    require(
        summary["decision"] == outcome["decision"],
        "Summary decision differs",
    )


def compare_summary(
    value: Any,
    *,
    experiment: ExperimentSpec,
    evidence_dir: Path,
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    dataset: dict[str, Any],
    schedule_sha256: str,
    idle_minimum: float,
) -> bool:
    summary = v.expect_mapping(value, "summary.json")
    expected_keys = {
        "status",
        "experiment",
        "expected_witnesses",
        "scope",
        "performance_measurement_scope",
        "claim_eligible",
        "decision",
        "workload",
        "performance",
        "query_performance",
        "quality",
        "resources",
        "acceptance",
        "limitations",
    }
    v.expect_exact_keys(summary, expected_keys, "summary.json")
    recall_nonregression = all(
        point["all_pairs_within_0_005"] for point in metrics["recall_points"]
    )
    every_pair_faster = all(
        pair["control_cold_build_ns"] > pair["treatment_cold_build_ns"]
        for pair in metrics["pairs"]
    )
    strata_faster = all(
        math.prod(
            pair["control_cold_build_ns"]
            for pair in metrics["pairs"]
            if pair["order"] == order
        )
        > math.prod(
            pair["treatment_cold_build_ns"]
            for pair in metrics["pairs"]
            if pair["order"] == order
        )
        for order in metrics["strata"]
    )
    minimum_effect = metrics["minimum_effect"]
    variability_accepted = metrics["variability_accepted"]
    ratio_variability = metrics["ratio_variability_accepted"]
    query_tps_variability = metrics["query_tps_variability_accepted"]
    query_ratio_variability = metrics["query_ratio_variability_accepted"]
    query_role_latency_variability = metrics[
        "query_role_latency_variability_accepted"
    ]
    query_paired_latency_ratio_variability = metrics[
        "query_paired_latency_ratio_variability_accepted"
    ]
    query_tps_nonregression = metrics["query_tps_nonregression"]
    query_p50_nonregression = metrics["query_p50_nonregression"]
    query_p95_nonregression = metrics["query_p95_nonregression"]
    query_p99_nonregression = metrics["query_p99_nonregression"]
    query_recall_floor_met = metrics["query_recall_floor_met"]
    query_recall_gap_met = metrics["query_recall_gap_met"]
    idle_protocol = all(
        v.validate_idle(
            evidence_dir,
            record["idle"],
            context=f"summary idle {index}",
            minimum_idle_percent=idle_minimum,
        )
        is not None
        for index, record in enumerate(records)
    )
    idle_to_launch = all(
        type(record.get("idle_to_launch_ns")) is int
        and 0 <= record["idle_to_launch_ns"] <= v.IDLE_TO_LAUNCH_MAX_NS
        for record in records
    )
    host_resources = all(
        record["process_swaps"] == 0
        and record["host_before"]["low_power_mode"] is False
        and record["host_after"]["low_power_mode"] is False
        and record["host_before"]["thermal_nominal"] is True
        and record["host_after"]["thermal_nominal"] is True
        and type(record["global_swap_growth_bytes"]) is int
        for record in records
    )
    dataset_integrity = all(
        record["dataset_before"]["sha256"] == dataset["sha256"]
        and record["dataset_after"]["sha256"] == dataset["sha256"]
        and record["dataset_before"]["fnv1a64"] == dataset["fnv1a64"]
        and record["dataset_after"]["fnv1a64"] == dataset["fnv1a64"]
        for record in records
    )
    accepted = all(
        (
            metrics["graph_valid"],
            metrics["single_thread_semantic_payload_exact_match"],
            recall_nonregression,
            every_pair_faster,
            strata_faster,
            minimum_effect,
            variability_accepted,
            ratio_variability,
            query_tps_variability,
            query_ratio_variability,
            all(query_role_latency_variability.values()),
            all(query_paired_latency_ratio_variability.values()),
            query_tps_nonregression,
            query_p50_nonregression,
            query_p95_nonregression,
            query_p99_nonregression,
            query_recall_floor_met,
            query_recall_gap_met,
            idle_protocol,
            idle_to_launch,
            host_resources,
            dataset_integrity,
            metrics["measured_run_execution_witnesses_valid"],
        )
    )
    validate_experiment(summary["experiment"], "summary experiment")
    require(summary["experiment"] == experiment.record(), "Summary experiment differs")
    validate_expected_witnesses(
        summary["expected_witnesses"],
        experiment=experiment,
        context="summary expected_witnesses",
    )
    validate_summary_outcome(
        summary,
        experiment=experiment,
        accepted=accepted,
    )
    require(summary["scope"] == experiment.scope, "Summary scope differs")
    require(
        summary["performance_measurement_scope"]
        == PERFORMANCE_MEASUREMENT_SCOPE,
        "Summary performance measurement scope differs",
    )
    require(summary["claim_eligible"] is False, "Summary is incorrectly claim-eligible")
    expected_workload = {
        "vectors": v.VECTORS,
        "dimensions": v.DIMENSIONS,
        "M": v.M,
        "ef_construction": v.EF_CONSTRUCTION,
        "ef_search": v.EF_SEARCH,
        "participants": v.MEASURED_PARTICIPANTS,
        "data_fnv1a64": dataset["fnv1a64"],
        "data_sha256": dataset["sha256"],
        "data_bytes": dataset["bytes"],
        "schedule_sha256": schedule_sha256,
    }
    require(summary["workload"] == expected_workload, "Summary workload differs")
    performance = v.expect_mapping(summary["performance"], "summary performance")
    require(performance["minimum_speedup"] == MINIMUM_SPEEDUP, "Minimum speedup differs")
    v.expect_normalized_equal(performance["pairs"], metrics["pairs"], "summary pairs")
    v.expect_normalized_equal(performance["strata"], metrics["strata"], "summary strata")
    v.expect_close(
        performance["stratified_geometric_mean_speedup"],
        metrics["estimate"],
        "summary estimate",
    )
    require(
        performance["stratified_geometric_mean_speedup_is_approximate"] is True,
        "Summary estimate is not marked approximate",
    )
    require(
        performance["minimum_speedup_comparison_exact"]
        == metrics["minimum_speedup_comparison_exact"],
        "Summary exact minimum-speedup comparison differs",
    )
    v.expect_normalized_equal(
        performance["variability"],
        metrics["variability"],
        "summary variability",
    )
    v.expect_close(
        performance["paired_ratio_relative_mad"],
        metrics["paired_ratio_relative_mad"],
        "summary ratio MAD",
    )
    require(
        performance["speedup_range"]
        == [
            min(pair["treatment_speedup"] for pair in metrics["pairs"]),
            max(pair["treatment_speedup"] for pair in metrics["pairs"]),
        ],
        "Summary speedup range differs",
    )
    expected_query_performance = {
        "maximum_relative_mad": MAXIMUM_RELATIVE_MAD,
        "minimum_tps_ratio": MINIMUM_QUERY_TPS_RATIO,
        "maximum_p50_p95_p99_latency_ratio": MAXIMUM_QUERY_LATENCY_RATIO,
        "minimum_recall": MINIMUM_QUERY_RECALL,
        "maximum_absolute_recall_gap": MAXIMUM_QUERY_RECALL_GAP,
        "pairs": metrics["query_pairs"],
        "roles": metrics["query_roles"],
        "paired_tps_ratio_relative_mad": metrics[
            "query_paired_tps_ratio_relative_mad"
        ],
        "latency_variability": metrics["query_latency_variability"],
        "role_tps_relative_mad_at_most_0_10": query_tps_variability,
        "paired_tps_ratio_relative_mad_at_most_0_10": (
            query_ratio_variability
        ),
        "all_pairs_tps_at_least_0_95x": query_tps_nonregression,
        "all_pairs_p50_at_most_1_05x": query_p50_nonregression,
        "all_pairs_p95_at_most_1_05x": query_p95_nonregression,
        "all_pairs_p99_at_most_1_05x": query_p99_nonregression,
        "all_pairs_recall_at_least_0_99": query_recall_floor_met,
        "all_pairs_recall_gap_at_most_0_005": query_recall_gap_met,
    }
    v.expect_normalized_equal(
        summary["query_performance"],
        expected_query_performance,
        "summary query performance",
    )
    quality = v.expect_mapping(summary["quality"], "summary quality")
    v.expect_exact_keys(
        quality,
        {
            "graph_all_valid_and_reachable",
            "single_thread_semantic_payload_exact_match",
            "maximum_paired_recall_deficit",
            "recall_all_pairs_within_deficit",
            "recall_points",
        },
        "summary quality",
    )
    require(
        quality["graph_all_valid_and_reachable"] == metrics["graph_valid"]
        and quality["single_thread_semantic_payload_exact_match"]
        == metrics["single_thread_semantic_payload_exact_match"]
        and quality["maximum_paired_recall_deficit"] == MAXIMUM_RECALL_DEFICIT
        and quality["recall_all_pairs_within_deficit"] == recall_nonregression,
        "Summary quality scalars differ",
    )
    v.expect_normalized_equal(
        quality["recall_points"],
        metrics["recall_points"],
        "summary recall points",
    )
    acceptance = v.expect_mapping(summary["acceptance"], "summary acceptance")
    expected_acceptance = {
        "all_runs_valid": True,
        "exact_balanced_schedule_met": True,
        "measured_run_execution_witnesses_valid": metrics[
            "measured_run_execution_witnesses_valid"
        ],
        "all_graphs_valid_and_reachable": metrics["graph_valid"],
        "single_thread_semantic_payload_exact_match": metrics[
            "single_thread_semantic_payload_exact_match"
        ],
        "paired_recall_deficit_at_most_0_005": recall_nonregression,
        "every_measured_pair_faster": every_pair_faster,
        "both_order_strata_faster": strata_faster,
        "stratified_geometric_mean_at_least_1_05": minimum_effect,
        "relative_mad_at_most_0_10": variability_accepted,
        "paired_ratio_relative_mad_at_most_0_10": ratio_variability,
        "query_tps_relative_mad_at_most_0_10": query_tps_variability,
        "query_paired_tps_ratio_relative_mad_at_most_0_10": (
            query_ratio_variability
        ),
        "query_p50_role_latency_relative_mad_at_most_0_10": (
            query_role_latency_variability["p50_ns"]
        ),
        "query_p95_role_latency_relative_mad_at_most_0_10": (
            query_role_latency_variability["p95_ns"]
        ),
        "query_p99_role_latency_relative_mad_at_most_0_10": (
            query_role_latency_variability["p99_ns"]
        ),
        "query_p50_paired_latency_ratio_relative_mad_at_most_0_10": (
            query_paired_latency_ratio_variability["p50_ns"]
        ),
        "query_p95_paired_latency_ratio_relative_mad_at_most_0_10": (
            query_paired_latency_ratio_variability["p95_ns"]
        ),
        "query_p99_paired_latency_ratio_relative_mad_at_most_0_10": (
            query_paired_latency_ratio_variability["p99_ns"]
        ),
        "query_tps_at_least_0_95x": query_tps_nonregression,
        "query_p50_latency_at_most_1_05x": query_p50_nonregression,
        "query_p95_latency_at_most_1_05x": query_p95_nonregression,
        "query_p99_latency_at_most_1_05x": query_p99_nonregression,
        "query_recall_both_at_least_0_99": query_recall_floor_met,
        "query_recall_gap_at_most_0_005": query_recall_gap_met,
        "idle_protocol_met": idle_protocol,
        "idle_to_launch_at_most_1_second": idle_to_launch,
        "host_resource_protocol_met": host_resources,
        "dataset_integrity_met": dataset_integrity,
        "outliers_removed": False,
    }
    v.expect_exact_keys(
        acceptance,
        set(expected_acceptance),
        "summary acceptance",
    )
    require(acceptance == expected_acceptance, "Summary acceptance differs")
    maximum_swap_growth = max(
        0, max(int(record["global_swap_growth_bytes"]) for record in records)
    )
    require(
        summary["resources"]
        == {
            "global_swap_policy": v.GLOBAL_SWAP_POLICY,
            "maximum_observed_global_swap_growth_bytes": maximum_swap_growth,
            "indexing_window_peak_rss_established": False,
        },
        "Summary resources differ",
    )
    limitations = v.expect_list(summary["limitations"], "summary limitations")
    require(
        any("not an Infinity-versus-FAISS result" in item for item in limitations),
        "Summary omits the non-FAISS limitation",
    )
    require(
        any("No measured member is removed" in item for item in limitations),
        "Summary omits the no-outlier-deletion limitation",
    )
    require(
        any(
            "evidence-disabled release binaries require a separate final benchmark"
            in item
            for item in limitations
        ),
        "Summary omits the evidence-instrumented-build limitation",
    )
    return accepted


def validate_preflight_result(
    value: Any,
    *,
    experiment: ExperimentSpec,
    dataset_sha256: str,
    heldout_sha256: str,
    preflight_sha256: str,
    schedule_sha256: str,
) -> None:
    result = v.expect_mapping(value, "preflight-result.json")
    v.expect_exact_keys(
        result,
        {
            "causal_schema_version",
            "experiment",
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
    result_schema_version = v.expect_int(
        result["causal_schema_version"],
        "preflight result causal schema version",
        minimum=1,
    )
    require(
        result_schema_version != ARCHIVED_CAUSAL_SCHEMA_VERSION,
        "Archived causal schema 15 must use its manifest-captured verifier",
    )
    require(
        result_schema_version == CAUSAL_SCHEMA_VERSION,
        "Preflight result schema version differs",
    )
    validate_experiment(result["experiment"], "preflight result experiment")
    require(
        result["experiment"] == experiment.record(),
        "Preflight result experiment differs",
    )
    require(result["scope"] == experiment.scope, "Preflight result scope differs")
    require(
        result["execution_mode"] == PREFLIGHT_ONLY_MODE,
        "Preflight result execution mode differs",
    )
    require(result["status"] == "pass", "Preflight result did not pass")
    require(
        v.expect_string(
            result["completed_at"],
            "preflight result completion time",
        ).endswith("Z"),
        "Preflight result completion time is not UTC",
    )
    require(
        v.expect_int(
            result["benchmark_members_executed"],
            "preflight result benchmark member count",
            minimum=0,
        )
        == 0,
        "Preflight-only evidence claims benchmark members",
    )
    require(
        type(result["campaign_complete"]) is bool
        and result["campaign_complete"] is False,
        "Preflight-only evidence claims a complete campaign",
    )
    artifacts = v.expect_mapping(
        result["artifact_sha256"],
        "preflight result artifact hashes",
    )
    require(
        artifacts
        == {
            "dataset.json": dataset_sha256,
            v.HELDOUT_MANIFEST_FILENAME: heldout_sha256,
            "preflight.json": preflight_sha256,
            "schedule.json": schedule_sha256,
        },
        "Preflight result artifact hashes differ",
    )


def closure_records_for_orphan_check(
    evidence_dir: Path,
    preflight_value: Any,
) -> list[Any]:
    preflight = v.expect_mapping(preflight_value, "preflight.json")
    roles = v.expect_mapping(preflight.get("roles"), "preflight roles")
    values: list[Any] = []
    for role in ("control", "treatment"):
        build = v.expect_mapping(roles.get(role), f"preflight {role} build")
        reference = v.expect_mapping(
            build.get("build_input_closure"),
            f"preflight {role} build-input closure",
        )
        _, path = v.relative_evidence_path(
            evidence_dir,
            reference.get("captured_path"),
            f"preflight {role} build-input closure path",
        )
        values.append(
            v.read_json(path, f"preflight {role} build-input closure")
        )
    return values


def verify_evidence(
    evidence_directory: Path,
    *,
    preflight_only: bool = False,
    authenticated_context_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_dir = Path(evidence_directory)
    require(evidence_dir.exists() and evidence_dir.is_dir(), "Evidence directory is missing")
    require(not evidence_dir.is_symlink(), "Evidence directory must not be a symlink")
    evidence_dir = evidence_dir.resolve()
    reader: v.VerifiedEvidence | None = None
    try:
        directory_descriptor = (
            authenticated_context_value["directory_descriptor"]
            if authenticated_context_value is not None
            else None
        )
        directory_identity = (
            authenticated_context_value["directory_identity"]
            if authenticated_context_value is not None
            else None
        )
        reader = v.VerifiedEvidence(
            evidence_dir,
            directory_descriptor=directory_descriptor,
            directory_identity=directory_identity,
        )
        manifest = reader.open()
        require("failure.json" not in manifest, "Evidence contains failure.json")
        required = {
            "dataset.json",
            v.DATASET_FILENAME,
            v.HELDOUT_MANIFEST_FILENAME,
            "schedule.json",
            "preflight.json",
            RUNNER_CAPTURE_FILENAME,
            VERIFIER_CAPTURE_FILENAME,
        }
        require(not (required - set(manifest)), "Evidence is missing common artifacts")
        schedule_path = evidence_dir / "schedule.json"
        schedule = validate_schedule(v.read_json(schedule_path, "schedule.json"))
        dataset = v.validate_dataset(
            evidence_dir,
            v.read_json(evidence_dir / "dataset.json", "dataset.json"),
        )
        truth = v.heldout_truth_for_validated_dataset(
            evidence_dir / v.DATASET_FILENAME,
            dataset["sha256"],
        )
        heldout = v.validate_heldout_manifest(
            evidence_dir,
            v.read_json(
                evidence_dir / v.HELDOUT_MANIFEST_FILENAME,
                v.HELDOUT_MANIFEST_FILENAME,
            ),
            dataset=dataset,
            truth=truth,
        )
        expected_mode = PREFLIGHT_ONLY_MODE if preflight_only else FULL_CAMPAIGN_MODE
        preflight_value = v.read_json(evidence_dir / "preflight.json", "preflight.json")
        (
            experiment,
            roles,
            campaign_binding,
            runtime_hashes,
            idle_minimum,
        ) = validate_preflight(
            evidence_dir,
            preflight_value,
            dataset=dataset,
            heldout=heldout,
            evidence_manifest=manifest,
            schedule_sha256=v.sha256_file(schedule_path),
            expected_mode=expected_mode,
            authenticated_context=authenticated_context_value,
        )
        closure_values = closure_records_for_orphan_check(
            evidence_dir,
            preflight_value,
        )
        if preflight_only:
            require("preflight-result.json" in manifest, "Preflight result is missing")
            require(
                "runs.json" not in manifest
                and "summary.json" not in manifest
                and "README.md" not in manifest,
                "Preflight-only evidence contains run results",
            )
            member_prefixes = tuple(
                f"{member['sequence']:02d}-{member['pair']}-{member['role']}."
                for member in schedule
            )
            member_files = sorted(
                path for path in manifest if path.startswith(member_prefixes)
            )
            require(
                not member_files,
                f"Preflight-only evidence contains member files: {member_files}",
            )
            validate_preflight_result(
                v.read_json(
                    evidence_dir / "preflight-result.json",
                    "preflight-result.json",
                ),
                experiment=experiment,
                dataset_sha256=v.sha256_file(evidence_dir / "dataset.json"),
                heldout_sha256=v.sha256_file(
                    evidence_dir / v.HELDOUT_MANIFEST_FILENAME
                ),
                preflight_sha256=v.sha256_file(evidence_dir / "preflight.json"),
                schedule_sha256=v.sha256_file(schedule_path),
            )
            v.verify_no_orphan_content_blobs(
                manifest,
                preflight_value,
                *closure_values,
            )
            return {
                "status": "PREFLIGHT_PASS",
                "experiment": experiment.name,
                "schedule_members": len(schedule),
                "benchmark_members_executed": 0,
            }

        require(
            "preflight-result.json" not in manifest,
            "Full campaign evidence contains preflight-result.json",
        )
        require(
            "runs.json" in manifest
            and "summary.json" in manifest
            and "README.md" in manifest,
            "Campaign results are missing",
        )
        records = validate_runs(
            evidence_dir,
            v.read_json(evidence_dir / "runs.json", "runs.json"),
            experiment=experiment,
            schedule=schedule,
            dataset=dataset,
            roles=roles,
            campaign_binding=campaign_binding,
            runtime_hashes=runtime_hashes,
            idle_minimum=idle_minimum,
            truth=truth,
        )
        metrics = recompute(records)
        hypothesis_accepted = compare_summary(
            v.read_json(evidence_dir / "summary.json", "summary.json"),
            experiment=experiment,
            evidence_dir=evidence_dir,
            metrics=metrics,
            records=records,
            dataset=dataset,
            schedule_sha256=v.sha256_file(schedule_path),
            idle_minimum=idle_minimum,
        )
        v.verify_no_orphan_content_blobs(
            manifest,
            preflight_value,
            *records,
            *closure_values,
        )
        return {
            "status": "PASS",
            "experiment": experiment.name,
            **hypothesis_outcome(experiment, hypothesis_accepted),
            "schedule_members": len(schedule),
            "measured_pairs": len(metrics["pairs"]),
            "stratified_geometric_mean_speedup": metrics["estimate"],
            "minimum_speedup_comparison_exact": metrics[
                "minimum_speedup_comparison_exact"
            ],
            "query_performance": {
                "pairs": metrics["query_pairs"],
                "roles": metrics["query_roles"],
                "role_tps_relative_mad_at_most_0_10": metrics[
                    "query_tps_variability_accepted"
                ],
                "paired_tps_ratio_relative_mad": metrics[
                    "query_paired_tps_ratio_relative_mad"
                ],
                "latency_variability": metrics["query_latency_variability"],
                "paired_tps_ratio_relative_mad_at_most_0_10": metrics[
                    "query_ratio_variability_accepted"
                ],
                "all_pairs_tps_at_least_0_95x": metrics[
                    "query_tps_nonregression"
                ],
                "all_pairs_p50_at_most_1_05x": metrics[
                    "query_p50_nonregression"
                ],
                "all_pairs_p95_at_most_1_05x": metrics[
                    "query_p95_nonregression"
                ],
                "all_pairs_p99_at_most_1_05x": metrics[
                    "query_p99_nonregression"
                ],
                "all_pairs_recall_at_least_0_99": metrics[
                    "query_recall_floor_met"
                ],
                "all_pairs_recall_gap_at_most_0_005": metrics[
                    "query_recall_gap_met"
                ],
            },
            "quality_nonregression": all(
                point["all_pairs_within_0_005"]
                for point in metrics["recall_points"]
            ),
        }
    except v.VerificationError as error:
        raise CausalVerificationError(str(error)) from error
    finally:
        if reader is not None:
            try:
                reader.close()
            except v.VerificationError as error:
                raise CausalVerificationError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments.count("--preflight-only") > 1:
        parser.error("--preflight-only must not be repeated")
    return parser.parse_args(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        context = authenticated_context()
        require(
            context is not None,
            "Direct execution is disabled; use the authenticated Batch4 bootstrap",
        )
        authenticated_root = Path(context["evidence_directory"])
        require(
            args.evidence_directory.resolve(strict=True) == authenticated_root,
            "Verifier evidence directory differs from the authenticated root",
        )
        require(
            args.preflight_only is context["preflight_only"],
            "Verifier mode differs from the authenticated mode",
        )
        result = verify_evidence(
            args.evidence_directory,
            preflight_only=args.preflight_only,
            authenticated_context_value=context,
        )
    except (CausalVerificationError, OSError) as error:
        print(
            json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
