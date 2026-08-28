from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path, PurePosixPath
from unittest import mock

from tools.apple_silicon import native_hnsw_batch4_causal as causal
from tools.apple_silicon import native_hnsw_d0 as d0
from tools.apple_silicon import native_hnsw_exactness as exactness
from tools.apple_silicon import verify_native_hnsw_batch4_causal as verify
from tools.apple_silicon import verify_native_hnsw_exactness as exactness_verify


DATASET = {
    "sha256": "a" * 64,
    "fnv1a64": 123,
    "bytes": 456,
}

NINJA_DEPS_HEADER = b"# ninjadeps\n" + struct.pack("<I", 4)


def ninja_deps_record(
    payload: bytes,
    *,
    dependency: bool = False,
    declared_size: int | None = None,
) -> bytes:
    size = len(payload) if declared_size is None else declared_size
    if dependency:
        size |= 0x80000000
    return struct.pack("<I", size) + payload


def ninja_deps_path_record(
    path: object,
    node_id: int,
    *,
    padding: int | None = None,
    checksum: int | None = None,
    declared_size: int | None = None,
) -> bytes:
    path_bytes = path.encode("utf-8") if isinstance(path, str) else bytes(path)
    if padding is None:
        padding = (4 - len(path_bytes) % 4) % 4
    if checksum is None:
        checksum = (~node_id) & 0xFFFFFFFF
    payload = path_bytes + b"\0" * padding + struct.pack("<I", checksum)
    return ninja_deps_record(payload, declared_size=declared_size)


def ninja_deps_dependency_record(
    output_id: int,
    mtime: int,
    dependency_ids: list[int],
    *,
    declared_size: int | None = None,
) -> bytes:
    payload = struct.pack(
        f"<{3 + len(dependency_ids)}I",
        output_id,
        mtime & 0xFFFFFFFF,
        (mtime >> 32) & 0xFFFFFFFF,
        *dependency_ids,
    )
    return ninja_deps_record(
        payload,
        dependency=True,
        declared_size=declared_size,
    )


def ninja_deps_database(
    paths: list[object],
    dependencies: list[tuple[int, int, list[int]]],
) -> bytes:
    return b"".join(
        (
            NINJA_DEPS_HEADER,
            *(
                ninja_deps_path_record(path, node_id)
                for node_id, path in enumerate(paths)
            ),
            *(
                ninja_deps_dependency_record(output_id, mtime, dependency_ids)
                for output_id, mtime, dependency_ids in dependencies
            ),
        )
    )


def parse_ninja_deps_fixture(
    module: object,
    content: bytes,
    *,
    build_directory: Path,
    expected_output_spellings: dict[str, str],
) -> dict[str, dict]:
    context = {"label": "test parser"} if module is causal else {
        "context": "test parser"
    }
    return module.parse_ninja_dependency_database(
        content,
        build_directory=build_directory,
        expected_output_spellings=expected_output_spellings,
        **context,
    )


def parse_archived_ninja_deps_fixture(
    module: object,
    database_path: Path,
    *,
    build_directory: Path,
    expected_output_spellings: dict[str, str],
) -> dict[str, dict]:
    context = {"label": "test archive"} if module is causal else {
        "context": "test archive"
    }
    return module.parse_archived_ninja_dependency_database(
        database_path,
        build_directory=build_directory,
        expected_output_spellings=expected_output_spellings,
        **context,
    )


def ninja_clean_transcript(
    target: str,
    paths: list[str],
    *,
    additional_targets: list[str] | None = None,
    supplemental_targets: list[str] | None = None,
) -> str:
    return "\n".join(
        [
            "Cleaning...",
            f"Target {target}",
            *(f"Target {item}" for item in (additional_targets or [])),
            *(f"Target {item}" for item in (supplemental_targets or [])),
            *(f"Remove {path}" for path in paths),
            f"{len(paths)} files.",
            "",
        ]
    )


def fixture_selected_target_graph(
    target: Path,
    *,
    material_outputs: list[Path] | None = None,
    clean_outputs: list[Path] | None = None,
    derived_link_outputs: list[Path] | None = None,
) -> dict:
    return {
        "selected_outputs": [str(target)],
        "material_outputs": [
            str(path) for path in (material_outputs or [target])
        ],
        "derived_link_outputs": [
            str(path) for path in (derived_link_outputs or [])
        ],
        "clean_plan": {
            "outputs": [str(path) for path in (clean_outputs or [target])],
        },
    }


def fixture_exactness_output(benchmark_output: Path) -> Path:
    return (
        benchmark_output.parent
        / "infinity_hnsw_exactness_emitter_production"
    )


def schema_16_closure_fixture(
    closure: dict,
    *,
    benchmark_output: Path,
    benchmark_target: str,
    benchmark_link_arguments: list[str],
    exactness_output: Path | None = None,
    exactness_link_arguments: list[str] | None = None,
) -> dict:
    exactness_output = exactness_output or fixture_exactness_output(
        benchmark_output
    )
    if not exactness_output.exists():
        exactness_output.write_bytes(b"exactness fixture\n")
    exactness_target = "infinity_hnsw_exactness_emitter_production"
    closure["products"] = [
        {
            "name": "benchmark",
            "requested_target": benchmark_target,
            "expected_output": str(benchmark_output),
            "link": {"arguments": benchmark_link_arguments},
        },
        {
            "name": "exactness-emitter",
            "requested_target": exactness_target,
            "expected_output": str(exactness_output),
            "link": {
                "arguments": exactness_link_arguments
                or [
                    "/tools/clang++",
                    "-o",
                    exactness_output.name,
                ]
            },
        },
    ]
    graph = closure["selected_target_graph"]
    graph["selected_outputs"] = [
        str(benchmark_output),
        str(exactness_output),
    ]
    for key in ("material_outputs",):
        if str(exactness_output) not in graph[key]:
            graph[key].append(str(exactness_output))
    clean_outputs = graph["clean_plan"]["outputs"]
    if str(exactness_output) not in clean_outputs:
        clean_outputs.append(str(exactness_output))
    return closure


def live_ninja_clean_outputs(
    ninja: str,
    build_directory: Path,
    target: str,
    *,
    additional_targets: list[str] | None = None,
) -> list[Path]:
    completed = subprocess.run(
        [
            ninja,
            "-v",
            "-n",
            "-C",
            str(build_directory),
            "-t",
            "clean",
            target,
            *(additional_targets or []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return causal.parse_ninja_clean_transcript(
        completed,
        build_directory=build_directory,
        requested_target=target,
        additional_requested_targets=additional_targets,
        context=f"{target} fixture clean plan",
    )


def fixture_evidence_manifest(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): causal.d0.sha256(path)
        for path in directory.rglob("*")
        if path.is_file()
    }


def fixture_link_arguments(
    ninja: str,
    build_directory: Path,
    target: str,
    compile_entries: list[dict],
    binary: Path,
) -> list[str]:
    compiler = causal.d0.compile_entry_arguments(
        compile_entries[0],
        label="fixture",
        index=0,
    )[0]
    commands = subprocess.run(
        [ninja, "-C", str(build_directory), "-t", "commands", target],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    matches: list[list[str]] = []
    shell_boundaries = {"&&", "||", ";", ">", ">>"}
    for line in commands.splitlines():
        tokens = shlex.split(line)
        for compiler_index, token in enumerate(tokens):
            if token != compiler:
                continue
            end = len(tokens)
            for index in range(compiler_index + 1, len(tokens)):
                if tokens[index] in shell_boundaries:
                    end = index
                    break
            arguments = tokens[compiler_index:end]
            if "-c" in arguments or arguments.count("-o") != 1:
                continue
            output_index = arguments.index("-o")
            if output_index + 1 >= len(arguments):
                continue
            output = Path(arguments[output_index + 1])
            if not output.is_absolute():
                output = build_directory / output
            if Path(os.path.abspath(output)) == binary:
                matches.append(arguments)
    if len(matches) != 1:
        raise AssertionError(f"expected one fixture link action, found {len(matches)}")
    return matches[0]


def fixture_compile_output(
    compile_entries: list[dict],
    *,
    target: str,
) -> Path:
    marker = f"CMakeFiles/{target}.dir/"
    matches: list[Path] = []
    for index, entry in enumerate(compile_entries):
        arguments = causal.d0.compile_entry_arguments(
            entry,
            label=f"{target} fixture",
            index=index,
        )
        output = causal.d0.compile_output_path(
            arguments,
            working_directory=Path(entry["directory"]),
            label=f"{target} fixture",
        )
        if output is not None and marker in output.as_posix():
            matches.append(output)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {target} fixture compile output, found {matches}"
        )
    return matches[0]


def capture_preclosure_fixture(
    build_directory: Path,
    output_dir: Path,
    *,
    create_missing: bool = False,
) -> tuple[list[dict], Path]:
    cache_path = build_directory / "CMakeCache.txt"
    compile_path = build_directory / "compile_commands.json"
    paths = causal.rebuild_metadata_paths(
        build_directory=build_directory,
        cache_path=cache_path,
        compile_path=compile_path,
    )
    if create_missing:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(f"fixture metadata: {path.name}\n".encode())
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise AssertionError(f"fixture build metadata is missing: {missing}")
    records = causal.capture_preclosure_metadata(paths, output_dir=output_dir)
    captured_path = output_dir / "control-rebuild-preclosure-metadata.json"
    causal.d0.write_json(captured_path, records)
    return records, captured_path


def closure_file_record(
    path: Path,
    roles: list[str],
    *,
    output_dir: Path,
) -> dict:
    resolved = path.resolve(strict=True)
    blob = causal.d0.capture_content_blob(path, output_dir=output_dir)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "type": causal.d0.closure_file_role(str(path)),
        "roles": roles,
        **blob,
    }


def exactness_lifecycle_fixture(pid: int, base_time: int) -> dict:
    return {
        "schema_version": exactness.EXACTNESS_PROCESS_EVIDENCE_SCHEMA_VERSION,
        "leader_pid": pid,
        "terminal_observation": {
            "observed_at_monotonic_ns": base_time,
            "pid": pid,
            "idtype": exactness.P_PID,
            "options": (
                exactness.WNOHANG | exactness.WEXITED | exactness.WNOWAIT
            ),
            "code": exactness.CLD_EXITED,
            "status": 0,
        },
        "pre_reap_quiescence": {
            "observed_at_monotonic_ns": base_time + 1,
            "process_group": pid,
            "leader_pid": pid,
            "member_pids": [pid],
            "no_members_except_leader": True,
        },
        "reap": {
            "completed_at_monotonic_ns": base_time + 2,
            "requested_pid": pid,
            "options": 0,
            "returned_pid": pid,
            "raw_status": 0,
            "status_validated": True,
            "returncode": 0,
        },
    }


def exactness_gate_fixture(
    output_dir: Path,
    *,
    expected_witnesses: dict[str, dict[str, int]] | None = None,
) -> tuple[dict, dict[str, dict], dict]:
    expected_witnesses = expected_witnesses or (
        causal.expected_witnesses_for_experiment(
            causal.EXPERIMENTS["combined-optimized"]
        )
    )
    artifact_names = ("graph", "save_to_ptr", "witness", "stdout", "stderr")
    roles = {
        role: {
            "exactness_emitter": {
                "path": f"/build/{role}/infinity_hnsw_exactness_emitter_production",
                "sha256": digest * 64,
                "bytes": 4096,
            }
        }
        for role, digest in (("control", "a"), ("treatment", "b"))
    }
    runs: list[dict] = []
    independent_roles = {
        "control": {"runs": []},
        "treatment": {"runs": []},
    }
    artifacts_by_run: dict[tuple[str, int], dict[str, dict]] = {}
    for sequence, (role, repetition) in enumerate(exactness.EXACTNESS_RUN_ORDER):
        artifacts: dict[str, dict] = {}
        for artifact_name in artifact_names:
            path = output_dir / f"{role}-{repetition}-{artifact_name}.bin"
            path.write_bytes(
                f"{role}:{repetition}:{artifact_name}\n".encode("ascii")
            )
            status = path.stat()
            artifacts[artifact_name] = {
                "captured_path": path.name,
                "sha256": d0.sha256(path),
                "bytes": status.st_size,
                "identity": {
                    "device": status.st_dev,
                    "inode": status.st_ino,
                    "mode": status.st_mode,
                    "links": status.st_nlink,
                    "bytes": status.st_size,
                    "mtime_ns": status.st_mtime_ns,
                    "ctime_ns": status.st_ctime_ns,
                },
            }
        artifacts_by_run[role, repetition] = artifacts
        emitter = roles[role]["exactness_emitter"]
        emitter_identity = {
            "device": 1,
            "inode": 100 if role == "control" else 200,
            "mode": stat.S_IFREG | 0o755,
            "links": 1,
            "bytes": emitter["bytes"],
            "mtime_ns": 1,
            "ctime_ns": 1,
        }
        pid = 1000 + sequence
        started_mono = 10_000 + sequence * 100
        resumed_mono = started_mono + 20
        ended_mono = resumed_mono + 20
        lifecycle = exactness_lifecycle_fixture(pid, resumed_mono + 10)
        runs.append(
            {
                "sequence": sequence,
                "role": role,
                "repetition": repetition,
                "status": "pass",
                "command": [
                    emitter["path"],
                    "--dataset-fd",
                    "10",
                    "--graph-fd",
                    "11",
                    "--save-to-ptr-fd",
                    "12",
                    "--execution-evidence-fd",
                    "13",
                ],
                "environment": d0.benchmark_environment(1),
                "started_at_unix_ns": 1_000_000 + sequence * 100,
                "ended_at_unix_ns": 1_000_050 + sequence * 100,
                "started_at_monotonic_ns": started_mono,
                "resumed_at_monotonic_ns": resumed_mono,
                "ended_at_monotonic_ns": ended_mono,
                "emitter": {**emitter, "identity": emitter_identity},
                "process_evidence": {
                    "schema_version": (
                        exactness.EXACTNESS_PROCESS_EVIDENCE_SCHEMA_VERSION
                    ),
                    "spawn_method": "darwin-posix-spawn-start-suspended-v1",
                    "spawn_flags": {
                        "value": exactness.SPAWN_FLAGS,
                        "start_suspended": exactness.POSIX_SPAWN_START_SUSPENDED,
                        "setsid": exactness.POSIX_SPAWN_SETSID,
                        "cloexec_default": exactness.POSIX_SPAWN_CLOEXEC_DEFAULT,
                    },
                    "fixed_descriptors": {
                        "dataset": exactness.CHILD_DATASET_FD,
                        "graph": exactness.CHILD_GRAPH_FD,
                        "save_to_ptr": exactness.CHILD_SAVE_TO_PTR_FD,
                        "witness": exactness.CHILD_WITNESS_FD,
                    },
                    "suspended": {
                        "observed_at_monotonic_ns": started_mono + 10,
                        "supervisor_pid": 900,
                        "pid": pid,
                        "process_group": pid,
                        "session": pid,
                        "processes": [
                            {
                                "pid": pid,
                                "parent_pid": 900,
                                "process_group": pid,
                                "status_code": 17,
                                "status": "stopped",
                                "start_time_seconds": 1,
                                "start_time_microseconds": 2,
                                "name": "exactness-emitter",
                                "executable_path": emitter["path"],
                                "executable_device": emitter_identity["device"],
                                "executable_inode": emitter_identity["inode"],
                                "executable_size": emitter_identity["bytes"],
                                "child_pids": [],
                            }
                        ],
                        "mapped_macho_matches_held_descriptor": True,
                        "direct_child": True,
                        "no_descendants": True,
                    },
                    "lifecycle": copy.deepcopy(lifecycle),
                },
                "artifacts": artifacts,
                "producer_verification": {
                    "producer_parser": (
                        "tools.apple_silicon.native_hnsw_exactness"
                    ),
                    **{
                        name: {
                            "sha256": record["sha256"],
                            "bytes": record["bytes"],
                        }
                        for name, record in artifacts.items()
                    },
                },
            }
        )
        independent_roles[role]["runs"].append(
            {
                "ordinal": repetition + 1,
                "lifecycle": copy.deepcopy(lifecycle),
                **{
                    name: {
                        "path": record["captured_path"],
                        "sha256": record["sha256"],
                        "bytes": record["bytes"],
                    }
                    for name, record in artifacts.items()
                },
            }
        )

    reference = artifacts_by_run["control", 0]
    independent_summary = {
        "schema_version": exactness_verify.SCHEMA_VERSION,
        "status": "PASS",
        "expected_witnesses": copy.deepcopy(expected_witnesses),
        "cross_role_witness_byte_equal": (
            expected_witnesses["control"]
            == expected_witnesses["treatment"]
        ),
        "graph": {
            "sha256": reference["graph"]["sha256"],
            "bytes": reference["graph"]["bytes"],
        },
        "save_to_ptr": {
            "sha256": reference["save_to_ptr"]["sha256"],
            "bytes": reference["save_to_ptr"]["bytes"],
        },
    }
    return (
        {
            "schema_version": exactness.EXACTNESS_SCHEMA_VERSION,
            "status": "pass",
            "expected_witnesses": copy.deepcopy(expected_witnesses),
            "execution_order": [
                f"{role}-{repetition}"
                for role, repetition in exactness.EXACTNESS_RUN_ORDER
            ],
            "timeout_seconds_per_run": 300.0,
            "dataset": {
                "path": d0.DATASET_FILENAME,
                "sha256": d0.DATASET_SHA256,
                "bytes": d0.VECTORS * d0.DIMENSIONS * d0.FLOAT32_BYTES,
            },
            "runs": runs,
            "comparisons": {
                "within_role_byte_determinism": True,
                "cross_role_stdout_byte_equal": True,
                "cross_role_graph_byte_equal": True,
                "cross_role_save_to_ptr_byte_equal": True,
                "cross_role_witness_byte_equal": (
                    expected_witnesses["control"]
                    == expected_witnesses["treatment"]
                ),
                "cross_role_witness_relation_matches_expected": True,
                "graph_sha256": independent_summary["graph"]["sha256"],
                "save_to_ptr_sha256": (
                    independent_summary["save_to_ptr"]["sha256"]
                ),
                "stdout_sha256": reference["stdout"]["sha256"],
                "control_witness_sha256": reference["witness"]["sha256"],
                "treatment_witness_sha256": artifacts_by_run[
                    "treatment", 0
                ]["witness"]["sha256"],
            },
            "independent_verifier_record": {
                "schema_version": exactness_verify.SCHEMA_VERSION,
                "expected_witnesses": copy.deepcopy(expected_witnesses),
                "dataset": {
                    "path": d0.DATASET_FILENAME,
                    "sha256": d0.DATASET_SHA256,
                    "bytes": d0.VECTORS * d0.DIMENSIONS * d0.FLOAT32_BYTES,
                },
                "roles": independent_roles,
            },
            "independent_verification": independent_summary,
        },
        roles,
        independent_summary,
    )


def set_query_wall_ns(record: dict, wall_ns: int) -> None:
    operations = verify.v.QUERY_THROUGHPUT_OPERATIONS
    tps = operations * 1e9 / wall_ns
    record["audit"]["query"].update(
        {
            "throughput_operations": operations,
            "throughput_wall_ns": wall_ns,
            "qps": tps,
            "tps": tps,
        }
    )


def set_query_latency_percentiles(
    record: dict,
    *,
    p50_ns: int,
    p95_ns: int,
    p99_ns: int,
) -> None:
    sample_count = verify.v.QUERY_LATENCY_SAMPLE_COUNT
    record["audit"]["query"].update(
        {
            "latency_samples_ns": (
                [p50_ns] * (sample_count // 2)
                + [p95_ns] * (sample_count * 45 // 100)
                + [p99_ns] * (sample_count * 5 // 100)
            ),
            "p50_ns": p50_ns,
            "p95_ns": p95_ns,
            "p99_ns": p99_ns,
        }
    )


def set_recall_hits(point: dict, eligible_hits: int) -> None:
    possible_hits = point["possible_hits"]
    point["eligible_hits"] = eligible_hits
    point["recall"] = eligible_hits / float(possible_hits)


def fixture_query_checksums() -> list[int]:
    return list(range(1, d0.HELDOUT_QUERY_COUNT + 1))


def fixture_query_evidence(
    *,
    target_tps: float,
    p50_ns: int,
    p95_ns: int,
    p99_ns: int,
    per_query_checksums: list[int] | None = None,
) -> dict:
    operations = d0.QUERY_THROUGHPUT_OPERATIONS
    wall_ns = round(operations * 1e9 / target_tps)
    latency_samples = (
        [p50_ns] * (d0.QUERY_LATENCY_SAMPLE_COUNT // 2)
        + [p95_ns] * (d0.QUERY_LATENCY_SAMPLE_COUNT * 45 // 100)
        + [p99_ns] * (d0.QUERY_LATENCY_SAMPLE_COUNT * 5 // 100)
    )
    checksums = list(per_query_checksums or fixture_query_checksums())
    result_checksum = d0.query_benchmark_checksum(checksums)
    return d0.query_performance_record(
        timed_query_corpus_sha256="d" * 64,
        latency_samples_ns=latency_samples,
        latency_validated_operations=d0.QUERY_LATENCY_SAMPLE_COUNT,
        latency_validated_result_checksum=result_checksum,
        throughput_operations=operations,
        throughput_validated_operations=operations,
        throughput_wall_ns=wall_ns,
        throughput_validated_result_checksum=result_checksum,
        result_checksum=result_checksum,
        per_query_checksums=checksums,
    )


def with_raw_query_evidence(records: list[dict]) -> list[dict]:
    for record in records:
        query = record["audit"]["query"]
        record["audit"]["query"] = fixture_query_evidence(
            target_tps=float(query["tps"]),
            p50_ns=int(query["p50_ns"]),
            p95_ns=int(query["p95_ns"]),
            p99_ns=int(query["p99_ns"]),
            per_query_checksums=query.get("per_query_checksums"),
        )
    return records


def passing_idle() -> dict:
    return {
        "status": "pass",
        "accepted_at_monotonic_ns": time.monotonic_ns(),
        "accepted_at_unix_ns": time.time_ns(),
        "required_minimum_idle_percent": 95.0,
        "required_window_seconds": d0.IDLE_WINDOW_SECONDS,
        "accepted_attempt": 1,
        "attempts": [
            {
                "attempt": 1,
                "status": "pass",
                "minimum_idle_percent": 96.0,
                "covered_seconds": d0.IDLE_WINDOW_SECONDS,
            }
        ],
    }


def summary_records(
    *,
    experiment: causal.ExperimentSpec = causal.EXPERIMENTS[
        "incremental-reciprocal"
    ],
    control_ns: int = 110,
    treatment_ns: int = 100,
) -> list[dict]:
    records: list[dict] = []
    for entry in causal.schedule():
        role = str(entry["role"])
        duration = control_ns if role == "control" else treatment_ns
        fields = {"infinity_cold_build_ns": str(duration)}
        for k, ef in d0.RECALL_POINTS:
            fields[f"infinity_returned_ids_ef_{ef}_k_{k}"] = (
                f"{k}:{ef}:identical"
            )
            fields[
                f"infinity_returned_distances_sha256_ef_{ef}_k_{k}"
            ] = f"{k:02x}{ef:04x}".ljust(64, "0")
        fingerprint = {
            "sha256": DATASET["sha256"],
            "fnv1a64": DATASET["fnv1a64"],
        }
        records.append(
            {
                **entry,
                "status": "pass",
                "experiment": experiment.name,
                "fields": fields,
                "idle": passing_idle(),
                "idle_to_launch_ns": 1,
                "process_swaps": 0,
                "host_before": {
                    "low_power_mode": False,
                    "thermal_nominal": True,
                },
                "host_after": {
                    "low_power_mode": False,
                    "thermal_nominal": True,
                },
                "global_swap_growth_bytes": 0,
                "dataset_before": fingerprint,
                "dataset_after": fingerprint,
                "audit": {
                    "execution_witness": copy.deepcopy(
                        causal.expected_witnesses_for_experiment(experiment)[role]
                    ),
                    "graph": {
                        "valid": True,
                        "vertex_count": d0.VECTORS,
                        "reachable_count": d0.VECTORS,
                        "levels_sha256": "c" * 64,
                    },
                    "recall": {
                        "query_count": d0.HELDOUT_QUERY_COUNT,
                        "query_seed": d0.HELDOUT_QUERY_SEED,
                        "queries_sha256": "d" * 64,
                        "truth_sha256": "e" * 64,
                        "points": [
                            {
                                "k": k,
                                "ef": ef,
                                "eligible_hits": d0.HELDOUT_QUERY_COUNT * k,
                                "possible_hits": d0.HELDOUT_QUERY_COUNT * k,
                                "recall": 1.0,
                            }
                            for k, ef in d0.RECALL_POINTS
                        ]
                    },
                    "query": fixture_query_evidence(
                        target_tps=1000.0,
                        p50_ns=100,
                        p95_ns=150,
                        p99_ns=200,
                    ),
                },
            }
        )
    return records


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_exact_role_aware_and_balanced(self) -> None:
        entries = causal.schedule()
        self.assertEqual(len(entries), 18)
        self.assertEqual([entry["sequence"] for entry in entries], list(range(18)))
        self.assertTrue(all(entry["engine"] == "infinity" for entry in entries))
        measured = [
            entry
            for entry in entries
            if entry["phase"] == "measured" and entry["role"] == "control"
        ]
        self.assertEqual(
            sum(entry["pair_order"] == "control/treatment" for entry in measured),
            3,
        )
        self.assertEqual(
            sum(entry["pair_order"] == "treatment/control" for entry in measured),
            3,
        )
        self.assertEqual(entries, verify.frozen_schedule())

    def test_schedule_rejects_reordering_or_wrong_engine(self) -> None:
        entries = causal.schedule()
        reordered = copy.deepcopy(entries)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(causal.CausalFailure, "frozen causal order"):
            causal.validate_schedule(reordered)

        wrong_engine = copy.deepcopy(entries)
        wrong_engine[0]["engine"] = "faiss"
        with self.assertRaisesRegex(causal.CausalFailure, "frozen causal order"):
            causal.validate_schedule(wrong_engine)
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "frozen causal order",
        ):
            verify.validate_schedule(wrong_engine)

    def test_run_chronology_rejects_overlap(self) -> None:
        records = [
            {"started_at_unix_ns": 10, "ended_at_unix_ns": 20},
            {"started_at_unix_ns": 20, "ended_at_unix_ns": 30},
        ]
        verify.validate_run_chronology(records)
        records[1]["started_at_unix_ns"] = 19
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "overlaps or predates",
        ):
            verify.validate_run_chronology(records)


class SummaryTests(unittest.TestCase):
    def summarize(self, records: list[dict]) -> dict:
        return causal.summarize(
            records,
            experiment=causal.EXPERIMENTS["incremental-reciprocal"],
            dataset=DATASET,
            schedule_sha256="b" * 64,
            idle_minimum_percent=95.0,
        )

    def compare_verified_summary(
        self,
        summary: dict,
        records: list[dict],
    ) -> bool:
        metrics = verify.recompute(records)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(verify.v, "validate_idle", return_value={}),
        ):
            return verify.compare_summary(
                summary,
                experiment=verify.EXPERIMENTS["incremental-reciprocal"],
                evidence_dir=Path(directory),
                metrics=metrics,
                records=records,
                dataset=DATASET,
                schedule_sha256="b" * 64,
                idle_minimum=95.0,
            )

    def test_known_ten_percent_speedup_passes_and_recomputes(self) -> None:
        records = with_raw_query_evidence(summary_records())
        summary = self.summarize(records)
        metrics = verify.recompute(records)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["decision"], "accept-incremental-reciprocal")
        self.assertEqual(
            summary["performance_measurement_scope"],
            causal.PERFORMANCE_MEASUREMENT_SCOPE,
        )
        self.assertTrue(
            any(
                "evidence-disabled release binaries require a separate final benchmark"
                in limitation
                for limitation in summary["limitations"]
            )
        )
        self.assertAlmostEqual(
            summary["performance"]["stratified_geometric_mean_speedup"],
            1.1,
        )
        self.assertAlmostEqual(metrics["estimate"], 1.1)
        self.assertTrue(summary["acceptance"]["every_measured_pair_faster"])
        self.assertTrue(
            summary["acceptance"][
                "single_thread_semantic_payload_exact_match"
            ]
        )
        self.assertNotIn("single_thread_audit_exact_match", summary["quality"])
        self.assertNotIn("single_thread_audit_exact_match", summary["acceptance"])
        self.assertTrue(summary["acceptance"]["query_tps_at_least_0_95x"])
        self.assertTrue(
            summary["acceptance"]["query_tps_relative_mad_at_most_0_10"]
        )
        self.assertTrue(
            summary["acceptance"][
                "query_paired_tps_ratio_relative_mad_at_most_0_10"
            ]
        )
        self.assertTrue(
            summary["acceptance"]["query_p50_latency_at_most_1_05x"]
        )
        self.assertTrue(
            summary["acceptance"]["query_p95_latency_at_most_1_05x"]
        )
        self.assertTrue(
            summary["acceptance"]["query_p99_latency_at_most_1_05x"]
        )
        self.assertFalse(summary["acceptance"]["outliers_removed"])

    def test_one_non_faster_pair_fails(self) -> None:
        records = summary_records()
        for record in records:
            if record["pair"] == "P1" and record["role"] == "treatment":
                record["fields"]["infinity_cold_build_ns"] = "110"
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["acceptance"]["every_measured_pair_faster"])

    def test_summary_binds_expected_witnesses_and_measurement_scope(self) -> None:
        records = with_raw_query_evidence(summary_records())
        summary = self.summarize(records)
        changed_witness = copy.deepcopy(summary)
        changed_witness["expected_witnesses"]["control"][
            causal.INCREMENTAL_WITNESS_FIELDS[0]
        ] = 1
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "expected_witnesses differs from the experiment role matrix",
        ):
            self.compare_verified_summary(changed_witness, records)

        changed_scope = copy.deepcopy(summary)
        changed_scope["performance_measurement_scope"] = "release-binaries"
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "performance measurement scope differs",
        ):
            self.compare_verified_summary(changed_scope, records)

    def test_effect_below_five_percent_fails(self) -> None:
        summary = self.summarize(summary_records(control_ns=104, treatment_ns=100))
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["stratified_geometric_mean_at_least_1_05"]
        )

    def test_exact_five_percent_speedup_boundary_passes(self) -> None:
        summary = self.summarize(
            summary_records(control_ns=105, treatment_ns=100)
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(
            summary["performance"]["stratified_geometric_mean_speedup"],
            1.05,
        )
        exact = summary["performance"]["minimum_speedup_comparison_exact"]
        self.assertEqual(
            exact["observed_speedup_product"],
            exact["required_speedup_product"],
        )
        self.assertEqual(exact["root_degree"], 6)
        self.assertTrue(exact["met"])
        self.assertTrue(
            summary["acceptance"]["stratified_geometric_mean_at_least_1_05"]
        )

    def test_rounded_geometric_mean_cannot_override_exact_failure(self) -> None:
        summary = self.summarize(
            summary_records(control_ns=1_049_999, treatment_ns=1_000_000)
        )
        rendered = causal.render_summary(summary)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["performance"]["minimum_speedup_comparison_exact"]["met"]
        )
        self.assertIn("`1.0500x` (approximate report", rendered)
        self.assertIn("Exact minimum-speedup gate: `False`.", rendered)

    def test_ratio_gates_do_not_accept_nearby_out_of_bounds_values(self) -> None:
        for module in (causal, verify):
            with self.subTest(module=module.__name__, gate="minimum"):
                self.assertTrue(module.ratio_at_least(0.95, 1, 95, 100))
                self.assertFalse(
                    module.ratio_at_least(0.94999999999, 1, 95, 100)
                )
            with self.subTest(module=module.__name__, gate="maximum"):
                self.assertTrue(module.ratio_at_most(1.05, 1, 105, 100))
                self.assertFalse(
                    module.ratio_at_most(1.05000000001, 1, 105, 100)
                )

    def test_rejected_hypothesis_is_a_completed_campaign(self) -> None:
        summary = self.summarize(summary_records(control_ns=104, treatment_ns=100))
        causal.validate_completed_campaign(summary)
        verify.validate_summary_outcome(
            summary,
            experiment=verify.EXPERIMENTS["incremental-reciprocal"],
            accepted=False,
        )
        with self.assertRaisesRegex(causal.CausalFailure, "summary status"):
            causal.validate_completed_campaign({"status": "interrupted"})
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "Summary status differs",
        ):
            verify.validate_summary_outcome(
                summary,
                experiment=verify.EXPERIMENTS["incremental-reciprocal"],
                accepted=True,
            )
        self.assertEqual(
            verify.hypothesis_outcome(
                verify.EXPERIMENTS["incremental-reciprocal"],
                False,
            ),
            {
                "hypothesis_accepted": False,
                "decision": "reject-incremental-reciprocal",
            },
        )

    def test_exact_recall_deficit_boundary_passes(self) -> None:
        records = summary_records()
        for record in records:
            if (
                record["pair"] == "P1"
                and record["role"] == "treatment"
                and record["phase"] == "measured"
            ):
                point = next(
                    point
                    for point in record["audit"]["recall"]["points"]
                    if point["k"] == 100
                )
                set_recall_hits(point, point["possible_hits"] - 32)
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(
            summary["acceptance"]["paired_recall_deficit_at_most_0_005"]
        )
        point_summary = next(
            point for point in summary["quality"]["recall_points"] if point["k"] == 100
        )
        self.assertEqual(
            point_summary["maximum_treatment_deficit_exact"],
            {"numerator": 1, "denominator": 200},
        )

    def test_recall_deficit_beyond_boundary_fails(self) -> None:
        records = summary_records()
        for record in records:
            if (
                record["pair"] == "P1"
                and record["role"] == "treatment"
                and record["phase"] == "measured"
            ):
                point = next(
                    point
                    for point in record["audit"]["recall"]["points"]
                    if point["k"] == 100
                )
                set_recall_hits(point, point["possible_hits"] - 33)
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["paired_recall_deficit_at_most_0_005"]
        )

    def test_recall_hit_count_records_are_strict(self) -> None:
        mutations = {
            "boolean hits": lambda point: point.update({"eligible_hits": True}),
            "negative hits": lambda point: point.update({"eligible_hits": -1}),
            "too many hits": lambda point: point.update(
                {"eligible_hits": point["possible_hits"] + 1}
            ),
            "wrong denominator": lambda point: point.update(
                {"possible_hits": point["possible_hits"] + 1}
            ),
            "float mismatch": lambda point: point.update({"recall": 0.5}),
            "extra key": lambda point: point.update({"unexpected": 1}),
        }
        for label, mutate in mutations.items():
            for module in (causal, verify):
                with self.subTest(label=label, module=module.__name__):
                    record = summary_records()[0]
                    point = record["audit"]["recall"]["points"][0]
                    mutate(point)
                    with self.assertRaises(
                        (
                            causal.CausalFailure,
                            verify.CausalVerificationError,
                            verify.v.VerificationError,
                        )
                    ):
                        module.recall_point_fraction(
                            record,
                            point["k"],
                            point["ef"],
                        )

    def test_c1_timing_samples_are_excluded_from_semantic_match(self) -> None:
        summary = self.summarize(summary_records())
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(
            summary["quality"]["single_thread_semantic_payload_exact_match"]
        )

    def test_c1_semantic_payload_binds_required_results(self) -> None:
        mutations = {
            "graph levels": lambda record: record["audit"]["graph"].update(
                {"levels_sha256": "d" * 64}
            ),
            "recall": lambda record: record["audit"]["recall"]["points"][0].update(
                {
                    "eligible_hits": (
                        record["audit"]["recall"]["points"][0]["possible_hits"] - 1
                    ),
                    "recall": (
                        record["audit"]["recall"]["points"][0]["possible_hits"] - 1
                    )
                    / record["audit"]["recall"]["points"][0]["possible_hits"],
                }
            ),
            "returned IDs": lambda record: record["fields"].update(
                {
                    "infinity_returned_ids_ef_64_k_10": "different",
                }
            ),
            "distance hash": lambda record: record["fields"].update(
                {
                    "infinity_returned_distances_sha256_ef_64_k_10": "e" * 64,
                }
            ),
            "query checksum": lambda record: record["audit"]["query"].update(
                {"result_checksum": 654321}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                records = summary_records()
                treatment = next(
                    record
                    for record in records
                    if record["phase"] == "correctness"
                    and record["role"] == "treatment"
                )
                mutate(treatment)
                summary = self.summarize(records)
                self.assertEqual(summary["status"], "fail")
                self.assertFalse(
                    summary["quality"][
                        "single_thread_semantic_payload_exact_match"
                    ]
                )

    def test_single_thread_graph_difference_fails(self) -> None:
        records = summary_records()
        for record in records:
            if record["phase"] == "correctness" and record["role"] == "treatment":
                record["audit"]["graph"]["reachable_count"] -= 1
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["single_thread_semantic_payload_exact_match"]
        )

    def test_legacy_duplicate_exactness_alias_is_rejected(self) -> None:
        records = with_raw_query_evidence(summary_records())
        for section in ("quality", "acceptance"):
            with self.subTest(section=section):
                summary = self.summarize(records)
                summary[section]["single_thread_audit_exact_match"] = True
                with self.assertRaisesRegex(
                    verify.v.VerificationError,
                    f"summary {section} keys differ",
                ):
                    self.compare_verified_summary(summary, records)

    def test_query_metrics_are_reported_separately_by_pair(self) -> None:
        summary = self.summarize(summary_records())
        query = summary["query_performance"]
        self.assertEqual(len(query["pairs"]), 6)
        self.assertEqual(query["roles"]["control"]["median_tps"], 1000.0)
        self.assertEqual(query["roles"]["treatment"]["median_p50_ns"], 100)
        first = query["pairs"][0]
        self.assertEqual(first["control"]["p95_ns"], 150)
        self.assertEqual(first["treatment"]["p99_ns"], 200)
        self.assertEqual(first["treatment_over_control"]["tps"], 1.0)
        self.assertEqual(query["roles"]["control"]["tps_relative_mad"], 0.0)
        self.assertEqual(query["paired_tps_ratio_relative_mad"], 0.0)

    def test_schema2_query_record_matches_independent_verifier(self) -> None:
        producer = fixture_query_evidence(
            target_tps=1000.0,
            p50_ns=100,
            p95_ns=150,
            p99_ns=200,
        )
        independent = verify.v.query_performance_record(
            producer["timed_query_corpus_sha256"],
            producer["latency_samples_ns"],
            producer["latency_validated_operations"],
            producer["latency_validated_result_checksum"],
            producer["throughput_operations"],
            producer["throughput_validated_operations"],
            producer["throughput_wall_ns"],
            producer["throughput_validated_result_checksum"],
            producer["result_checksum"],
            producer["per_query_checksums"],
        )

        self.assertEqual(producer, independent)
        self.assertEqual(
            independent["per_query_checksum_count"],
            d0.HELDOUT_QUERY_COUNT,
        )

    def test_high_query_role_tps_relative_mad_fails(self) -> None:
        records = summary_records()
        for record in records:
            if record["phase"] == "measured":
                record["audit"]["query"]["tps"] = (
                    500.0 if record["pair"] in {"P1", "P2", "P3"} else 1500.0
                )
        with_raw_query_evidence(records)
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["query_tps_relative_mad_at_most_0_10"]
        )
        self.assertTrue(
            summary["acceptance"][
                "query_paired_tps_ratio_relative_mad_at_most_0_10"
            ]
        )
        self.assertGreater(
            metrics["query_roles"]["control"]["tps_relative_mad"],
            causal.MAXIMUM_RELATIVE_MAD,
        )
        self.assertEqual(metrics["query_paired_tps_ratio_relative_mad"], 0.0)

    def test_high_paired_query_tps_ratio_relative_mad_fails(self) -> None:
        records = summary_records()
        for record in records:
            if record["phase"] != "measured":
                continue
            first_half = record["pair"] in {"P1", "P2", "P3"}
            if record["role"] == "control":
                record["audit"]["query"]["tps"] = (
                    1050.0 if first_half else 950.0
                )
            else:
                record["audit"]["query"]["tps"] = (
                    998.0 if first_half else 1140.0
                )
        with_raw_query_evidence(records)
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(
            summary["acceptance"]["query_tps_relative_mad_at_most_0_10"]
        )
        self.assertFalse(
            summary["acceptance"][
                "query_paired_tps_ratio_relative_mad_at_most_0_10"
            ]
        )
        self.assertTrue(summary["acceptance"]["query_tps_at_least_0_95x"])
        self.assertGreater(
            metrics["query_paired_tps_ratio_relative_mad"],
            causal.MAXIMUM_RELATIVE_MAD,
        )

    def test_query_tps_relative_mad_exact_boundary_passes(self) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] == "measured":
                set_query_wall_ns(
                    record,
                    (
                        11_000_022
                        if record["pair"] in {"P1", "P2", "P3"}
                        else 9_000_018
                    ),
                )
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(
            summary["acceptance"]["query_tps_relative_mad_at_most_0_10"]
        )
        self.assertEqual(
            summary["query_performance"]["roles"]["control"][
                "tps_relative_mad"
            ],
            0.10,
        )
        self.assertTrue(metrics["query_tps_variability_accepted"])
        self.assertTrue(self.compare_verified_summary(summary, records))

    def test_query_tps_relative_mad_one_ns_inside_passes(self) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] == "measured":
                set_query_wall_ns(
                    record,
                    (
                        11_000_021
                        if record["pair"] in {"P1", "P2", "P3"}
                        else 9_000_018
                    ),
                )
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(
            summary["acceptance"]["query_tps_relative_mad_at_most_0_10"]
        )
        self.assertLess(
            metrics["query_roles"]["control"]["tps_relative_mad"],
            causal.MAXIMUM_RELATIVE_MAD,
        )
        self.assertTrue(metrics["query_tps_variability_accepted"])
        self.assertTrue(self.compare_verified_summary(summary, records))

    def test_query_tps_relative_mad_beyond_boundary_fails(self) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] == "measured":
                set_query_wall_ns(
                    record,
                    (
                        11_000_023
                        if record["pair"] in {"P1", "P2", "P3"}
                        else 9_000_018
                    ),
                )
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["query_tps_relative_mad_at_most_0_10"]
        )
        self.assertFalse(metrics["query_tps_variability_accepted"])
        self.assertFalse(self.compare_verified_summary(summary, records))

    def test_paired_query_tps_ratio_relative_mad_exact_boundary_passes(
        self,
    ) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] != "measured":
                continue
            if record["role"] == "control":
                set_query_wall_ns(
                    record,
                    (
                        9_900_000
                        if record["pair"] in {"P1", "P2", "P3"}
                        else 12_100_000
                    ),
                )
            else:
                set_query_wall_ns(record, 10_000_000)
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(
            summary["acceptance"]["query_tps_relative_mad_at_most_0_10"]
        )
        self.assertTrue(
            summary["acceptance"][
                "query_paired_tps_ratio_relative_mad_at_most_0_10"
            ]
        )
        self.assertEqual(
            summary["query_performance"]["roles"]["control"][
                "tps_relative_mad"
            ],
            0.10,
        )
        self.assertEqual(
            summary["query_performance"]["paired_tps_ratio_relative_mad"],
            0.10,
        )
        self.assertTrue(metrics["query_ratio_variability_accepted"])
        self.assertTrue(self.compare_verified_summary(summary, records))

    def test_paired_query_tps_ratio_relative_mad_one_ns_inside_passes(
        self,
    ) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] != "measured":
                continue
            if record["role"] == "control":
                set_query_wall_ns(
                    record,
                    (
                        9_900_000
                        if record["pair"] in {"P1", "P2", "P3"}
                        else 12_099_999
                    ),
                )
            else:
                set_query_wall_ns(record, 10_000_000)
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(
            summary["acceptance"][
                "query_paired_tps_ratio_relative_mad_at_most_0_10"
            ]
        )
        self.assertLess(
            metrics["query_paired_tps_ratio_relative_mad"],
            causal.MAXIMUM_RELATIVE_MAD,
        )
        self.assertTrue(metrics["query_ratio_variability_accepted"])
        self.assertTrue(self.compare_verified_summary(summary, records))

    def test_paired_query_tps_ratio_relative_mad_one_ns_outside_fails(
        self,
    ) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] != "measured":
                continue
            if record["role"] == "control":
                set_query_wall_ns(
                    record,
                    (
                        9_900_000
                        if record["pair"] in {"P1", "P2", "P3"}
                        else 12_100_001
                    ),
                )
            else:
                set_query_wall_ns(record, 10_000_000)
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"][
                "query_paired_tps_ratio_relative_mad_at_most_0_10"
            ]
        )
        self.assertFalse(metrics["query_ratio_variability_accepted"])
        self.assertFalse(self.compare_verified_summary(summary, records))

    def test_ingestion_mad_exact_boundary_passes_independent_verifier(
        self,
    ) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] != "measured":
                continue
            if record["role"] == "control":
                duration = (
                    9_000_018
                    if record["pair"] in {"P1", "P2", "P3"}
                    else 11_000_022
                )
            else:
                duration = 8_000_016
            record["fields"]["infinity_cold_build_ns"] = str(duration)
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(
            summary["performance"]["variability"]["control"]["relative_mad"],
            0.10,
        )
        self.assertEqual(
            summary["performance"]["paired_ratio_relative_mad"],
            0.10,
        )
        self.assertTrue(metrics["variability_accepted"])
        self.assertTrue(metrics["ratio_variability_accepted"])
        self.assertTrue(self.compare_verified_summary(summary, records))

    def test_ingestion_mad_one_ns_outside_fails_independent_verifier(
        self,
    ) -> None:
        records = with_raw_query_evidence(summary_records())
        for record in records:
            if record["phase"] != "measured":
                continue
            if record["role"] == "control":
                duration = (
                    9_000_018
                    if record["pair"] in {"P1", "P2", "P3"}
                    else 11_000_023
                )
            else:
                duration = 8_000_016
            record["fields"]["infinity_cold_build_ns"] = str(duration)
        summary = self.summarize(records)
        metrics = verify.recompute(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["relative_mad_at_most_0_10"]
        )
        self.assertFalse(
            summary["acceptance"]["paired_ratio_relative_mad_at_most_0_10"]
        )
        self.assertFalse(metrics["variability_accepted"])
        self.assertFalse(metrics["ratio_variability_accepted"])
        self.assertFalse(self.compare_verified_summary(summary, records))

    def test_query_latency_role_relative_mad_boundaries(self) -> None:
        cases = {
            "p50_ns": lambda value: (value, 150, 200),
            "p95_ns": lambda value: (50, value, 200),
            "p99_ns": lambda value: (50, 75, value),
        }
        for percentile, percentiles_for_value in cases.items():
            label = percentile.removesuffix("_ns")
            acceptance_key = (
                f"query_{label}_role_latency_relative_mad_at_most_0_10"
            )
            for boundary, high, expected in (
                ("exact", 110, True),
                ("one_ns_outside", 111, False),
            ):
                with self.subTest(
                    percentile=percentile,
                    boundary=boundary,
                ):
                    records = with_raw_query_evidence(summary_records())
                    for record in records:
                        if record["phase"] != "measured":
                            continue
                        value = (
                            90
                            if record["pair"] in {"P1", "P2", "P3"}
                            else high
                        )
                        p50_ns, p95_ns, p99_ns = percentiles_for_value(value)
                        set_query_latency_percentiles(
                            record,
                            p50_ns=p50_ns,
                            p95_ns=p95_ns,
                            p99_ns=p99_ns,
                        )
                    summary = self.summarize(records)
                    metrics = verify.recompute(records)
                    variability = summary["query_performance"][
                        "latency_variability"
                    ][percentile]

                    self.assertIs(summary["acceptance"][acceptance_key], expected)
                    self.assertIs(
                        metrics["query_role_latency_variability_accepted"][
                            percentile
                        ],
                        expected,
                    )
                    self.assertEqual(
                        variability["role_relative_mad"]["control"],
                        (0.10 if expected else 21 / 201),
                    )
                    self.assertEqual(
                        variability["role_relative_mad"]["treatment"],
                        (0.10 if expected else 21 / 201),
                    )
                    self.assertEqual(
                        variability[
                            "paired_treatment_over_control_ratio_relative_mad"
                        ],
                        0.0,
                    )
                    self.assertEqual(summary["status"], "pass" if expected else "fail")
                    self.assertEqual(
                        self.compare_verified_summary(summary, records),
                        expected,
                    )

    def test_query_paired_latency_ratio_relative_mad_boundaries(self) -> None:
        cases = {
            "p50_ns": lambda value: (value, 250, 300),
            "p95_ns": lambda value: (100, value, 300),
            "p99_ns": lambda value: (100, 125, value),
        }
        for percentile, percentiles_for_value in cases.items():
            label = percentile.removesuffix("_ns")
            role_acceptance_key = (
                f"query_{label}_role_latency_relative_mad_at_most_0_10"
            )
            paired_acceptance_key = (
                f"query_{label}_paired_latency_ratio_relative_mad_at_most_0_10"
            )
            for boundary, high_control, expected in (
                ("exact", 180, True),
                ("one_ns_outside", 179, False),
            ):
                with self.subTest(
                    percentile=percentile,
                    boundary=boundary,
                ):
                    records = with_raw_query_evidence(summary_records())
                    for record in records:
                        if record["phase"] != "measured":
                            continue
                        first_half = record["pair"] in {"P1", "P2", "P3"}
                        value = (
                            180
                            if first_half and record["role"] == "control"
                            else 153
                            if first_half
                            else high_control
                            if record["role"] == "control"
                            else 187
                        )
                        p50_ns, p95_ns, p99_ns = percentiles_for_value(value)
                        set_query_latency_percentiles(
                            record,
                            p50_ns=p50_ns,
                            p95_ns=p95_ns,
                            p99_ns=p99_ns,
                        )
                    summary = self.summarize(records)
                    metrics = verify.recompute(records)
                    variability = summary["query_performance"][
                        "latency_variability"
                    ][percentile]

                    self.assertTrue(summary["acceptance"][role_acceptance_key])
                    self.assertTrue(
                        metrics["query_role_latency_variability_accepted"][
                            percentile
                        ]
                    )
                    self.assertIs(
                        summary["acceptance"][paired_acceptance_key],
                        expected,
                    )
                    self.assertIs(
                        metrics[
                            "query_paired_latency_ratio_variability_accepted"
                        ][percentile],
                        expected,
                    )
                    paired_mad = variability[
                        "paired_treatment_over_control_ratio_relative_mad"
                    ]
                    if expected:
                        self.assertEqual(paired_mad, 0.10)
                    else:
                        self.assertGreater(
                            paired_mad,
                            causal.MAXIMUM_RELATIVE_MAD,
                        )
                    self.assertEqual(summary["status"], "pass" if expected else "fail")
                    self.assertEqual(
                        self.compare_verified_summary(summary, records),
                        expected,
                    )

    def test_query_report_contains_separate_variability_lines(self) -> None:
        rendered = causal.render_summary(self.summarize(summary_records()))
        self.assertIn("Control TPS relative MAD: `0.0000%`.", rendered)
        self.assertIn("Treatment TPS relative MAD: `0.0000%`.", rendered)
        self.assertIn("Paired TPS-ratio relative MAD: `0.0000%`.", rendered)
        self.assertIn(
            "Query variability gates: per-role `True`; paired-ratio `True`.",
            rendered,
        )
        for percentile in ("p50", "p95", "p99"):
            self.assertIn(
                f"{percentile} latency relative MAD: control `0.0000%`; "
                "treatment `0.0000%`; paired treatment/control ratio "
                "`0.0000%`.",
                rendered,
            )
            self.assertIn(
                f"{percentile} latency variability gates: per-role `True`; "
                "paired-ratio `True`.",
                rendered,
            )

    def test_verifier_recomputes_tps_from_raw_operations_and_wall_time(
        self,
    ) -> None:
        records = with_raw_query_evidence(summary_records())
        record = next(
            item
            for item in records
            if item["phase"] == "measured"
            and item["pair"] == "P1"
            and item["role"] == "control"
        )
        query = record["audit"]["query"]
        query["throughput_wall_ns"] = 5_120_000_000
        query["qps"] = 2000.0
        query["tps"] = 2000.0

        metrics = verify.recompute(records)

        self.assertEqual(metrics["query_pairs"][0]["control"]["tps"], 2000.0)
        self.assertEqual(
            metrics["query_roles"]["control"]["tps"][0],
            2000.0,
        )

    def test_verifier_rejects_recorded_query_throughput_tampering(self) -> None:
        for field in ("qps", "tps"):
            with self.subTest(field=field):
                records = with_raw_query_evidence(summary_records())
                record = next(
                    item
                    for item in records
                    if item["phase"] == "measured"
                )
                record["audit"]["query"][field] += 1.0
                with self.assertRaisesRegex(
                    verify.v.VerificationError,
                    f"reconstructed evidence {field}",
                ):
                    verify.recompute(records)

    def test_verifier_requires_exact_query_keys_and_protocol(self) -> None:
        mutations = {
            "extra key": lambda query: query.update({"unexpected": True}),
            "missing raw key": lambda query: query.pop("throughput_wall_ns"),
            "protocol tamper": lambda query: query.update(
                {"throughput_concurrency": query["throughput_concurrency"] + 1}
            ),
            "corpus tamper": lambda query: query.update(
                {"timed_query_corpus_sha256": "f" * 64}
            ),
            "latency count tamper": lambda query: query.update(
                {
                    "latency_validated_operations": (
                        query["latency_validated_operations"] - 1
                    )
                }
            ),
            "latency checksum tamper": lambda query: query.update(
                {
                    "latency_validated_result_checksum": (
                        query["latency_validated_result_checksum"] ^ 1
                    )
                }
            ),
            "throughput validated count tamper": lambda query: query.update(
                {
                    "throughput_validated_operations": (
                        query["throughput_validated_operations"] - 1
                    )
                }
            ),
            "throughput checksum tamper": lambda query: query.update(
                {
                    "throughput_validated_result_checksum": (
                        query["throughput_validated_result_checksum"] ^ 1
                    )
                }
            ),
            "per-query count tamper": lambda query: query.update(
                {
                    "per_query_checksum_count": (
                        query["per_query_checksum_count"] - 1
                    )
                }
            ),
            "per-query ordering tamper": lambda query: query.update(
                {"per_query_checksums": list(reversed(query["per_query_checksums"]))}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                records = with_raw_query_evidence(summary_records())
                record = next(
                    item
                    for item in records
                    if item["phase"] == "measured"
                )
                mutate(record["audit"]["query"])
                with self.assertRaises(
                    (
                        verify.CausalVerificationError,
                        verify.v.VerificationError,
                    )
                ):
                    verify.recompute(records)

    def test_verifier_rejects_query_variability_summary_tampering(
        self,
    ) -> None:
        records = with_raw_query_evidence(summary_records())
        summary = self.summarize(records)
        self.assertTrue(self.compare_verified_summary(summary, records))
        mutations = {
            "extra query key": lambda value: value[
                "query_performance"
            ].update({"unexpected": True}),
            "missing query key": lambda value: value[
                "query_performance"
            ].pop("paired_tps_ratio_relative_mad"),
            "role MAD": lambda value: value["query_performance"]["roles"][
                "control"
            ].update({"tps_relative_mad": 0.01}),
            "paired-ratio MAD": lambda value: value[
                "query_performance"
            ].update({"paired_tps_ratio_relative_mad": 0.01}),
            "role gate": lambda value: value["acceptance"].update(
                {"query_tps_relative_mad_at_most_0_10": False}
            ),
            "paired-ratio gate": lambda value: value["acceptance"].update(
                {
                    "query_paired_tps_ratio_relative_mad_at_most_0_10": False,
                }
            ),
            "latency role MAD": lambda value: value["query_performance"][
                "latency_variability"
            ]["p50_ns"]["role_relative_mad"].update({"control": 0.01}),
            "latency paired-ratio MAD": lambda value: value[
                "query_performance"
            ]["latency_variability"]["p50_ns"].update(
                {
                    "paired_treatment_over_control_ratio_relative_mad": 0.01,
                }
            ),
            "latency role gate": lambda value: value["acceptance"].update(
                {
                    "query_p50_role_latency_relative_mad_at_most_0_10": False,
                }
            ),
            "latency paired-ratio gate": lambda value: value[
                "acceptance"
            ].update(
                {
                    "query_p50_paired_latency_ratio_relative_mad_at_most_0_10": False,
                }
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                tampered = copy.deepcopy(summary)
                mutate(tampered)
                with self.assertRaises(
                    (
                        verify.CausalVerificationError,
                        verify.v.VerificationError,
                    )
                ):
                    self.compare_verified_summary(tampered, records)

    def test_verifier_rejects_exact_decision_field_tampering(self) -> None:
        records = with_raw_query_evidence(summary_records())
        summary = self.summarize(records)
        mutations = {
            "minimum speedup product": lambda value: value["performance"][
                "minimum_speedup_comparison_exact"
            ]["observed_speedup_product"].update({"numerator": 1}),
            "approximate marker": lambda value: value["performance"].update(
                {"stratified_geometric_mean_speedup_is_approximate": False}
            ),
            "query recall gap": lambda value: value["query_performance"]["pairs"][
                0
            ]["absolute_recall_gap_exact"].update({"numerator": 1}),
            "maximum recall deficit": lambda value: value["quality"][
                "recall_points"
            ][0]["maximum_treatment_deficit_exact"].update({"numerator": 1}),
            "p50 gate": lambda value: value["acceptance"].update(
                {"query_p50_latency_at_most_1_05x": False}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                tampered = copy.deepcopy(summary)
                mutate(tampered)
                with self.assertRaises(
                    (
                        verify.CausalVerificationError,
                        verify.v.VerificationError,
                    )
                ):
                    self.compare_verified_summary(tampered, records)

    def test_query_tps_below_0_95x_fails(self) -> None:
        records = summary_records()
        for record in records:
            if record["phase"] == "measured" and record["role"] == "treatment":
                record["audit"]["query"]["tps"] = 949.0
        with_raw_query_evidence(records)
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["acceptance"]["query_tps_at_least_0_95x"])

    def test_paired_query_tps_ratio_raw_timing_boundaries(self) -> None:
        cases = (
            ("one_ns_below", 9_499_999, False),
            ("exact_boundary", 9_500_000, True),
            ("one_ns_above", 9_500_001, True),
        )
        for label, control_wall_ns, expected in cases:
            with self.subTest(label=label):
                records = with_raw_query_evidence(summary_records())
                for record in records:
                    if record["phase"] != "measured":
                        continue
                    set_query_wall_ns(
                        record,
                        (
                            control_wall_ns
                            if record["role"] == "control"
                            else 10_000_000
                        ),
                    )
                summary = self.summarize(records)
                metrics = verify.recompute(records)
                self.assertIs(
                    summary["acceptance"]["query_tps_at_least_0_95x"],
                    expected,
                )
                self.assertIs(metrics["query_tps_nonregression"], expected)
                self.assertEqual(
                    all(
                        pair["tps_nonregression"]
                        for pair in metrics["query_pairs"]
                    ),
                    expected,
                )
                self.assertEqual(
                    self.compare_verified_summary(summary, records),
                    summary["status"] == "pass",
                )

    def test_query_latency_raw_percentile_boundaries(self) -> None:
        cases = (
            (
                "p50",
                (100, 200, 300),
                (105, 200, 300),
                (106, 200, 300),
                "query_p50_latency_at_most_1_05x",
                "query_p50_nonregression",
            ),
            (
                "p95",
                (50, 100, 200),
                (50, 105, 200),
                (50, 106, 200),
                "query_p95_latency_at_most_1_05x",
                "query_p95_nonregression",
            ),
            (
                "p99",
                (50, 75, 100),
                (50, 75, 105),
                (50, 75, 106),
                "query_p99_latency_at_most_1_05x",
                "query_p99_nonregression",
            ),
        )
        for (
            percentile,
            control_values,
            boundary_values,
            outside_values,
            acceptance_key,
            metrics_key,
        ) in cases:
            for label, treatment_values, expected in (
                ("exact_boundary", boundary_values, True),
                ("one_ns_outside", outside_values, False),
            ):
                with self.subTest(percentile=percentile, label=label):
                    records = with_raw_query_evidence(summary_records())
                    for record in records:
                        if record["phase"] != "measured":
                            continue
                        values = (
                            control_values
                            if record["role"] == "control"
                            else treatment_values
                        )
                        set_query_latency_percentiles(
                            record,
                            p50_ns=values[0],
                            p95_ns=values[1],
                            p99_ns=values[2],
                        )
                    summary = self.summarize(records)
                    metrics = verify.recompute(records)
                    self.assertIs(summary["acceptance"][acceptance_key], expected)
                    self.assertIs(metrics[metrics_key], expected)
                    self.assertEqual(
                        self.compare_verified_summary(summary, records),
                        summary["status"] == "pass",
                    )

    def test_query_p95_above_1_05x_fails(self) -> None:
        records = summary_records()
        for record in records:
            if record["phase"] == "measured" and record["role"] == "treatment":
                set_query_latency_percentiles(
                    record,
                    p50_ns=100,
                    p95_ns=158,
                    p99_ns=200,
                )
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["query_p95_latency_at_most_1_05x"]
        )
        self.assertTrue(
            summary["acceptance"]["query_p99_latency_at_most_1_05x"]
        )

    def test_query_p99_above_1_05x_fails(self) -> None:
        records = summary_records()
        for record in records:
            if record["phase"] == "measured" and record["role"] == "treatment":
                set_query_latency_percentiles(
                    record,
                    p50_ns=100,
                    p95_ns=150,
                    p99_ns=211,
                )
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["query_p99_latency_at_most_1_05x"]
        )

    def test_query_recall_floor_and_gap_are_gated(self) -> None:
        records = summary_records()
        for record in records:
            if record["phase"] == "measured" and record["role"] == "treatment":
                for point in record["audit"]["recall"]["points"]:
                    if (point["k"], point["ef"]) == (
                        d0.QUERY_K,
                        d0.QUERY_EF_SEARCH,
                    ):
                        set_recall_hits(point, point["possible_hits"] - 7)
        summary = self.summarize(records)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["query_recall_both_at_least_0_99"]
        )
        self.assertFalse(
            summary["acceptance"]["query_recall_gap_at_most_0_005"]
        )

    def test_query_recall_hit_count_gap_boundaries(self) -> None:
        for missing_hits, expected in ((3, True), (4, False)):
            with self.subTest(missing_hits=missing_hits):
                records = with_raw_query_evidence(summary_records())
                for record in records:
                    if (
                        record["phase"] != "measured"
                        or record["role"] != "treatment"
                    ):
                        continue
                    point = next(
                        point
                        for point in record["audit"]["recall"]["points"]
                        if (point["k"], point["ef"])
                        == (d0.QUERY_K, d0.QUERY_EF_SEARCH)
                    )
                    set_recall_hits(
                        point,
                        point["possible_hits"] - missing_hits,
                    )
                summary = self.summarize(records)
                metrics = verify.recompute(records)
                self.assertIs(
                    summary["acceptance"]["query_recall_gap_at_most_0_005"],
                    expected,
                )
                self.assertIs(metrics["query_recall_gap_met"], expected)
                self.assertEqual(
                    self.compare_verified_summary(summary, records),
                    summary["status"] == "pass",
                )
                first_gap = summary["query_performance"]["pairs"][0][
                    "absolute_recall_gap_exact"
                ]
                expected_gap = Fraction(
                    missing_hits,
                    d0.HELDOUT_QUERY_COUNT * d0.QUERY_K,
                )
                self.assertEqual(
                    first_gap,
                    {
                        "numerator": expected_gap.numerator,
                        "denominator": expected_gap.denominator,
                    },
                )

    def test_missing_or_duplicate_member_is_rejected(self) -> None:
        records = summary_records()
        with self.assertRaisesRegex(causal.CausalFailure, "full schedule"):
            self.summarize(records[:-1])
        records[1] = records[0]
        with self.assertRaisesRegex(causal.CausalFailure, "scheduled member"):
            self.summarize(records)

    def test_mismatched_experiment_is_rejected(self) -> None:
        records = summary_records()
        records[0]["experiment"] = "threshold-only"
        with self.assertRaisesRegex(causal.CausalFailure, "selected experiment"):
            self.summarize(records)


class BuildComparisonTests(unittest.TestCase):
    REPO = Path("/repo")

    def create_ninja_deps_fixture(
        self,
        root: Path,
        *,
        absolute_output: bool = False,
    ) -> tuple[Path, dict[str, dict], dict[str, str], bytes]:
        ninja_value = shutil.which("ninja")
        compiler = shutil.which("cc")
        self.assertIsNotNone(ninja_value)
        self.assertIsNotNone(compiler)
        ninja = Path(ninja_value).resolve(strict=True)
        build_directory = root / "deps-build"
        build_directory.mkdir()
        (build_directory / "source.c").write_text(
            "int value(void) { return 7; }\n",
            encoding="utf-8",
        )
        output = (
            str(build_directory / "object.o")
            if absolute_output
            else "object.o"
        )
        (build_directory / "build.ninja").write_text(
            "\n".join(
                (
                    "rule cc",
                    f"  command = {compiler} -MMD -MF $out.d -c $in -o $out",
                    "  depfile = $out.d",
                    "  deps = gcc",
                    f"build {output}: cc source.c",
                    "",
                )
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [str(ninja), "-C", str(build_directory)],
            check=True,
            capture_output=True,
            text=True,
        )
        report = subprocess.run(
            [str(ninja), "-C", str(build_directory), "-t", "deps"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        records = causal.d0.parse_ninja_deps(
            report,
            build_directory=build_directory,
            label="test Ninja dependencies",
        )
        spellings = causal.ninja_dependency_output_spellings(
            report,
            build_directory=build_directory,
            records=records,
            label="test Ninja dependencies",
        )
        return (
            build_directory,
            records,
            spellings,
            (build_directory / ".ninja_deps").read_bytes(),
        )

    def build_pair(
        self,
        experiment: causal.ExperimentSpec,
        output_dir: Path,
    ) -> tuple[dict, dict]:
        source_suffixes = tuple(
            dict.fromkeys(
                suffix
                for suffixes in causal.SWITCH_COMPILE_SOURCES.values()
                for suffix in suffixes
            )
        )
        sources = [
            causal.canonical_source(self.REPO, suffix)
            for suffix in source_suffixes
        ]
        sources.append(str(self.REPO / "src/other.cppm"))
        arguments: dict[str, dict[str, list[str]]] = {
            "control": {},
            "treatment": {},
        }
        for role in ("control", "treatment"):
            for source in sources:
                values = ["clang++", "-c", source]
                switch_values = experiment.switch_states_for_role(role)
                for switch, value in switch_values.items():
                    if (
                        value == "ON"
                        and source
                        in {
                            causal.canonical_source(self.REPO, suffix)
                            for suffix in causal.SWITCH_COMPILE_SOURCES[switch]
                        }
                    ):
                        values.insert(1, f"-D{switch}")
                arguments[role][source] = values
            (output_dir / f"{role}-arguments.json").write_text(
                json.dumps(arguments[role]),
                encoding="utf-8",
            )

        shared_settings = {name: "same" for name in causal.SHARED_BUILD_SETTINGS}
        shared_settings["CMAKE_HOME_DIRECTORY"] = str(self.REPO)
        compiled_sources = [
            {
                "absolute_path": source,
                "sha256": "f" * 64,
                "bytes": 1,
            }
            for source in sources
        ]

        def build(role: str) -> dict:
            settings = {
                **shared_settings,
                **experiment.switch_states_for_role(role),
            }
            binary_path = f"/tmp/{role}"
            exactness_path = f"/tmp/{role}-exactness-emitter"
            record = {
                "switch_states": experiment.switch_states_for_role(role),
                "build_directory": f"/build/{role}",
                "binary": {
                    "path": binary_path,
                    "sha256": ("a" if role == "control" else "b") * 64,
                },
                "settings": settings,
                "compiled_sources": compiled_sources,
                "dependencies": {"identity": []},
                "exactness_emitter": {
                    "product_index": 1,
                    "product_name": "exactness-emitter",
                    "path": exactness_path,
                    "sha256": ("c" if role == "control" else "d") * 64,
                    "bytes": 4096,
                },
                "exactness_dependencies": {"identity": []},
                "product_compile_provenance": [
                    {
                        "name": name,
                        "requested_target": target,
                        "expected_output": (
                            binary_path if index == 0 else exactness_path
                        ),
                        "compile_entries": len(sources),
                        "ninja_commands": {
                            "command": [
                                "/tools/build_tool",
                                "-C",
                                f"/build/{role}",
                                "-t",
                                "commands",
                                target,
                            ],
                            "captured_path": (
                                f"{role}-{name}-ninja-commands.txt"
                            ),
                            "sha256": "e" * 64,
                        },
                    }
                    for index, (name, target) in enumerate(
                        causal.BUILD_PRODUCT_TARGETS
                    )
                ],
                "normalized_compile_arguments": {
                    "captured_path": f"{role}-arguments.json",
                },
                "_causal_validation": {
                    "normalized_link_arguments": ["clang++", "-o", "$BUILD/bin"],
                    "closure_input_identity": [],
                },
            }
            for tool, _, _ in causal.BUILD_TOOL_SPECS:
                record[tool] = {
                    "resolved_path": f"/tools/{tool}",
                    "sha256": "c" * 64,
                }
            return record

        return build("control"), build("treatment")

    def test_build_local_source_paths_are_role_normalized(self) -> None:
        common_source = {
            "absolute_path": "/repo/src/common.cppm",
            "sha256": "a" * 64,
            "bytes": 11,
        }
        control = {
            "build_directory": "/repo/build/control",
            "compiled_sources": [
                common_source,
                {
                    "absolute_path": (
                        "/repo/build/control/src/generated/"
                        "compilation_config.cppm"
                    ),
                    "sha256": "b" * 64,
                    "bytes": 17,
                },
            ],
        }
        treatment = {
            "build_directory": "/repo/build/treatment",
            "compiled_sources": [
                common_source,
                {
                    "absolute_path": (
                        "/repo/build/treatment/src/generated/"
                        "compilation_config.cppm"
                    ),
                    "sha256": "b" * 64,
                    "bytes": 17,
                },
            ],
        }
        expected = [
            ("$BUILD/src/generated/compilation_config.cppm", "b" * 64, 17),
            ("/repo/src/common.cppm", "a" * 64, 11),
        ]
        self.assertEqual(causal.source_identity(control), expected)
        self.assertEqual(causal.source_identity(treatment), expected)
        self.assertEqual(verify.source_identity(control), expected)
        self.assertEqual(verify.source_identity(treatment), expected)

    def test_build_path_normalization_requires_path_boundary(self) -> None:
        build_directory = Path("/repo/build/control")
        sibling = "/repo/build/control-other/generated.cppm"
        for normalize in (
            causal.canonical_build_path,
            verify.canonical_build_path,
        ):
            with self.subTest(normalize=normalize.__module__):
                self.assertEqual(
                    normalize(
                        "/repo/build/control/generated.cppm",
                        build_directory=build_directory,
                    ),
                    "$BUILD/generated.cppm",
                )
                self.assertEqual(
                    normalize(sibling, build_directory=build_directory),
                    sibling,
                )

    def test_compile_argument_maps_normalize_build_local_source_keys(
        self,
    ) -> None:
        for role in ("control", "treatment"):
            with self.subTest(role=role):
                build_directory = Path(f"/repo/build/{role}")
                source = (
                    build_directory
                    / "src/generated/compilation_config.cppm"
                )
                entry = {
                    "directory": str(build_directory),
                    "file": str(source),
                    "arguments": [
                        "/tools/clang++",
                        "-c",
                        str(source),
                        "-o",
                        str(build_directory / "config.o"),
                    ],
                }
                expected = {
                    "$BUILD/src/generated/compilation_config.cppm": [
                        "/tools/clang++",
                        "-c",
                        "$BUILD/src/generated/compilation_config.cppm",
                        "-o",
                        "$BUILD/config.o",
                    ]
                }
                self.assertEqual(
                    causal.compile_argument_map(
                        [entry],
                        build_directory=build_directory,
                        label=role,
                        response_references=set(),
                    ),
                    expected,
                )
                self.assertEqual(
                    verify.canonical_compile_argument_map(
                        [entry],
                        build_directory=build_directory,
                        response_files={},
                        used_response_files=set(),
                        context=role,
                    ),
                    expected,
                )

    def test_schema_16_build_fixture_covers_both_products(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        with tempfile.TemporaryDirectory() as directory:
            control, treatment = self.build_pair(
                experiment,
                Path(directory),
            )
            self.assertEqual(
                [
                    record["name"]
                    for record in control["product_compile_provenance"]
                ],
                ["benchmark", "exactness-emitter"],
            )
            self.assertEqual(
                [
                    record["requested_target"]
                    for record in control["product_compile_provenance"]
                ],
                [
                    "infinity_hnsw_d0_production",
                    "infinity_hnsw_exactness_emitter_production",
                ],
            )
            self.assertEqual(
                control["product_compile_provenance"][1]["expected_output"],
                control["exactness_emitter"]["path"],
            )
            self.assertNotEqual(
                control["exactness_emitter"]["path"],
                treatment["exactness_emitter"]["path"],
            )
            self.assertEqual(
                control["exactness_dependencies"]["identity"],
                treatment["exactness_dependencies"]["identity"],
            )

            for label, mutate, message in (
                (
                    "emitter path",
                    lambda value: value["exactness_emitter"].update(
                        {"path": control["exactness_emitter"]["path"]}
                    ),
                    "exactness emitter paths are identical",
                ),
                (
                    "emitter hash",
                    lambda value: value["exactness_emitter"].update(
                        {"sha256": control["exactness_emitter"]["sha256"]}
                    ),
                    "exactness emitter contents are identical",
                ),
                (
                    "dependency identity",
                    lambda value: value["exactness_dependencies"].update(
                        {"identity": [{"path": "/different"}]}
                    ),
                    "exactness runtime dependency identities differ",
                ),
            ):
                with self.subTest(label=label):
                    changed = copy.deepcopy(treatment)
                    mutate(changed)
                    with self.assertRaisesRegex(
                        causal.CausalFailure,
                        message,
                    ):
                        causal.compare_builds(
                            control,
                            changed,
                            experiment=experiment,
                            repo=self.REPO,
                            output_dir=Path(directory),
                        )

    def test_incremental_macro_is_treatment_only_on_two_sources(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            contrast = causal.compare_builds(
                control,
                treatment,
                experiment=experiment,
                repo=self.REPO,
                output_dir=output_dir,
            )
        self.assertEqual(
            contrast["control_switch_states"][
                causal.INCREMENTAL_RECIPROCAL_SWITCH
            ],
            "OFF",
        )
        self.assertEqual(
            contrast["treatment_switch_states"][
                causal.INCREMENTAL_RECIPROCAL_SWITCH
            ],
            "ON",
        )
        self.assertEqual(
            len(
                contrast["varying_compile_sources"][
                    causal.INCREMENTAL_RECIPROCAL_SWITCH
                ]
            ),
            2,
        )

    def test_opaque_build_local_link_input_remains_in_identity(self) -> None:
        build_directory = Path("/build")
        closure = {
            "products": [
                {
                    "name": "benchmark",
                    "expected_output": "/build/runner",
                },
                {
                    "name": "exactness-emitter",
                    "expected_output": "/build/exactness-emitter",
                },
            ],
            "compile_units": [],
            "dependency_graph": [],
            "selected_target_graph": {
                "selected_outputs": [
                    "/build/runner",
                    "/build/exactness-emitter",
                ],
                "derived_link_outputs": ["/build/libderived.a"],
            },
            "files": [
                {
                    "path": "/build/libderived.a",
                    "resolved_path": "/build/libderived.a",
                    "type": "archive",
                    "roles": ["link-archive"],
                    "sha256": "a" * 64,
                    "bytes": 1,
                },
                {
                    "path": "/build/libopaque.a",
                    "resolved_path": "/build/libopaque.a",
                    "type": "archive",
                    "roles": ["link-archive"],
                    "sha256": "b" * 64,
                    "bytes": 2,
                },
            ],
        }
        expected = [
            {
                "path": "$BUILD/libopaque.a",
                "resolved_path": "$BUILD/libopaque.a",
                "type": "archive",
                "roles": ["link-archive"],
                "sha256": "b" * 64,
                "bytes": 2,
            }
        ]
        self.assertEqual(
            causal.closure_input_identity(
                closure,
                build_directory=build_directory,
            ),
            expected,
        )
        self.assertEqual(
            verify.closure_input_identity(
                closure,
                build_directory=build_directory,
            ),
            expected,
        )
        changed = copy.deepcopy(closure)
        changed["files"][1]["sha256"] = "c" * 64
        self.assertNotEqual(
            causal.closure_input_identity(
                closure,
                build_directory=build_directory,
            ),
            causal.closure_input_identity(
                changed,
                build_directory=build_directory,
            ),
        )

    def test_incremental_macro_must_appear_exactly_once_on_each_source(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        varying = experiment.varying_switches[0]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            path = output_dir / "treatment-arguments.json"
            arguments = json.loads(path.read_text(encoding="utf-8"))
            common = f"/repo/src{causal.HNSW_COMMON_SOURCE}"
            arguments[common].remove(varying.macro)
            path.write_text(json.dumps(arguments), encoding="utf-8")
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "treatment effective macro state .* expected defined",
            ):
                causal.compare_builds(
                    control,
                    treatment,
                    experiment=experiment,
                    repo=self.REPO,
                    output_dir=output_dir,
                )

    def test_execution_evidence_macro_is_required_on_all_three_sources(
        self,
    ) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        evidence_switch = (
            causal.INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH
        )
        for suffix in causal.SWITCH_COMPILE_SOURCES[evidence_switch]:
            with self.subTest(source=suffix), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                control, treatment = self.build_pair(experiment, output_dir)
                path = output_dir / "treatment-arguments.json"
                arguments = json.loads(path.read_text(encoding="utf-8"))
                source = causal.canonical_source(self.REPO, suffix)
                arguments[source].remove(f"-D{evidence_switch}")
                path.write_text(json.dumps(arguments), encoding="utf-8")
                with self.assertRaisesRegex(
                    causal.CausalFailure,
                    "effective macro state .* expected defined",
                ):
                    causal.compare_builds(
                        control,
                        treatment,
                        experiment=experiment,
                        repo=self.REPO,
                        output_dir=output_dir,
                    )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "effective macro state .* expected defined",
                ):
                    verify.validate_compile_macro_states(
                        arguments,
                        experiment=verify.EXPERIMENTS[experiment.name],
                        role="treatment",
                        repo=self.REPO,
                    )

    def test_threshold_macro_is_treatment_only(self) -> None:
        experiment = causal.EXPERIMENTS["threshold-only"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            contrast = causal.compare_builds(
                control,
                treatment,
                experiment=experiment,
                repo=self.REPO,
                output_dir=output_dir,
            )
        self.assertEqual(
            contrast["control_switch_states"][
                causal.THRESHOLD_BATCH4_TRAVERSAL_SWITCH
            ],
            "OFF",
        )
        self.assertEqual(
            contrast["treatment_switch_states"][
                causal.THRESHOLD_BATCH4_TRAVERSAL_SWITCH
            ],
            "ON",
        )
        self.assertEqual(
            contrast["varying_compile_sources"],
            {
                causal.THRESHOLD_BATCH4_TRAVERSAL_SWITCH: [
                    causal.canonical_source(
                        self.REPO,
                        causal.HNSW_ALG_SOURCE,
                    )
                ]
            },
        )

    def test_combined_experiment_normalizes_both_varying_macros(self) -> None:
        experiment = causal.EXPERIMENTS["combined-optimized"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            contrast = causal.compare_builds(
                control,
                treatment,
                experiment=experiment,
                repo=self.REPO,
                output_dir=output_dir,
            )
        self.assertEqual(
            contrast["permitted_compile_differences"],
            [switch.macro for switch in experiment.varying_switches],
        )
        self.assertEqual(
            contrast["control_switch_states"],
            experiment.switch_states_for_role("control"),
        )
        self.assertEqual(
            contrast["treatment_switch_states"],
            experiment.switch_states_for_role("treatment"),
        )

    def test_combined_experiment_rejects_each_missing_varying_macro(self) -> None:
        experiment = causal.EXPERIMENTS["combined-optimized"]
        for varying in experiment.varying_switches:
            with self.subTest(switch=varying.name), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                control, treatment = self.build_pair(experiment, output_dir)
                path = output_dir / "treatment-arguments.json"
                arguments = json.loads(path.read_text(encoding="utf-8"))
                source = causal.canonical_source(
                    self.REPO,
                    varying.compile_sources[0],
                )
                arguments[source].remove(varying.macro)
                path.write_text(json.dumps(arguments), encoding="utf-8")
                with self.assertRaisesRegex(
                    causal.CausalFailure,
                    "expected defined once canonically",
                ):
                    causal.compare_builds(
                        control,
                        treatment,
                        experiment=experiment,
                        repo=self.REPO,
                        output_dir=output_dir,
                    )

    def test_off_varying_switch_cannot_be_explicitly_undefined(self) -> None:
        experiment = causal.EXPERIMENTS["combined-optimized"]
        for varying in experiment.varying_switches:
            with self.subTest(switch=varying.name), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                control, treatment = self.build_pair(experiment, output_dir)
                path = output_dir / "control-arguments.json"
                arguments = json.loads(path.read_text(encoding="utf-8"))
                source = causal.canonical_source(
                    self.REPO,
                    varying.compile_sources[0],
                )
                arguments[source].append(f"-U{varying.name}")
                path.write_text(json.dumps(arguments), encoding="utf-8")
                with self.assertRaisesRegex(
                    causal.CausalFailure,
                    "expected undefined with no compiler option",
                ):
                    causal.compare_builds(
                        control,
                        treatment,
                        experiment=experiment,
                        repo=self.REPO,
                        output_dir=output_dir,
                    )

    def test_undeclared_compile_difference_is_not_normalized(self) -> None:
        experiment = causal.EXPERIMENTS["combined-optimized"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            path = output_dir / "treatment-arguments.json"
            arguments = json.loads(path.read_text(encoding="utf-8"))
            arguments[str(self.REPO / "src/other.cppm")].append(
                "-DINFINITY_UNDECLARED_CAUSAL_DIFFERENCE"
            )
            path.write_text(json.dumps(arguments), encoding="utf-8")
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "differ beyond declared varying switches",
            ):
                causal.compare_builds(
                    control,
                    treatment,
                    experiment=experiment,
                    repo=self.REPO,
                    output_dir=output_dir,
                )

    def test_build_switch_states_reject_missing_extra_or_wrong_values(self) -> None:
        experiment = causal.EXPERIMENTS["combined-optimized"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            mutations = {
                "missing": lambda states: states.pop(
                    causal.THRESHOLD_BATCH4_TRAVERSAL_SWITCH
                ),
                "extra": lambda states: states.update({"UNDECLARED": "ON"}),
                "wrong": lambda states: states.update(
                    {causal.INCREMENTAL_RECIPROCAL_SWITCH: "OFF"}
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    changed = copy.deepcopy(treatment)
                    mutate(changed["switch_states"])
                    with self.assertRaisesRegex(
                        causal.CausalFailure,
                        "switch_states differ from the experiment",
                    ):
                        causal.compare_builds(
                            control,
                            changed,
                            experiment=experiment,
                            repo=self.REPO,
                            output_dir=output_dir,
                        )

    def test_noncanonical_macro_forms_are_rejected(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        varying = experiment.varying_switches[0]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            path = output_dir / "treatment-arguments.json"
            arguments = json.loads(path.read_text(encoding="utf-8"))
            replacements = {
                causal.HNSW_ALG_SOURCE: [
                    "-D",
                    f"{varying.name}=1",
                ],
                causal.HNSW_COMMON_SOURCE: [
                    f"-D{varying.name}=enabled",
                ],
            }
            for suffix, replacement in replacements.items():
                source = causal.canonical_source(self.REPO, suffix)
                index = arguments[source].index(varying.macro)
                arguments[source][index : index + 1] = replacement
            path.write_text(json.dumps(arguments), encoding="utf-8")

            with self.assertRaisesRegex(
                causal.CausalFailure,
                "expected defined once canonically",
            ):
                causal.compare_builds(
                    control,
                    treatment,
                    experiment=experiment,
                    repo=self.REPO,
                    output_dir=output_dir,
                )
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "expected defined once canonically",
            ):
                verify.validate_compile_macro_states(
                    arguments,
                    experiment=verify.EXPERIMENTS[experiment.name],
                    role="treatment",
                    repo=self.REPO,
                )

    def test_trailing_undef_overrides_a_feature_define(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        varying = experiment.varying_switches[0]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            path = output_dir / "treatment-arguments.json"
            arguments = json.loads(path.read_text(encoding="utf-8"))
            source = causal.canonical_source(self.REPO, causal.HNSW_ALG_SOURCE)
            arguments[source].extend(["-U", varying.name])
            path.write_text(json.dumps(arguments), encoding="utf-8")

            with self.assertRaisesRegex(
                causal.CausalFailure,
                "effective macro state .* expected defined",
            ):
                causal.compare_builds(
                    control,
                    treatment,
                    experiment=experiment,
                    repo=self.REPO,
                    output_dir=output_dir,
                )
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "effective macro state .* expected defined",
            ):
                verify.validate_compile_macro_states(
                    arguments,
                    experiment=verify.EXPERIMENTS[experiment.name],
                    role="treatment",
                    repo=self.REPO,
                )

    def test_feature_macro_cannot_leak_to_an_unlisted_source(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        varying = experiment.varying_switches[0]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            path = output_dir / "treatment-arguments.json"
            arguments = json.loads(path.read_text(encoding="utf-8"))
            arguments[str(self.REPO / "src/other.cppm")].append(
                varying.macro
            )
            path.write_text(json.dumps(arguments), encoding="utf-8")
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "effective macro state .* expected undefined",
            ):
                causal.compare_builds(
                    control,
                    treatment,
                    experiment=experiment,
                    repo=self.REPO,
                    output_dir=output_dir,
                )

    def test_held_switch_effective_state_and_leakage_are_rejected(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        cases = {
            "ordered held undef": lambda arguments: arguments[
                causal.canonical_source(self.REPO, causal.HNSW_ALG_SOURCE)
            ].extend(["-U", causal.RECIPROCAL_PRUNING_SWITCH]),
            "disabled held define leakage": lambda arguments: arguments[
                str(self.REPO / "src/other.cppm")
            ].append(f"-D{causal.INCREMENTAL_RECIPROCAL_SHADOW_SWITCH}=1"),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                control, treatment = self.build_pair(experiment, output_dir)
                path = output_dir / "treatment-arguments.json"
                arguments = json.loads(path.read_text(encoding="utf-8"))
                mutate(arguments)
                path.write_text(json.dumps(arguments), encoding="utf-8")
                with self.assertRaisesRegex(
                    causal.CausalFailure,
                    "effective macro state",
                ):
                    causal.compare_builds(
                        control,
                        treatment,
                        experiment=experiment,
                        repo=self.REPO,
                        output_dir=output_dir,
                    )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "effective macro state",
                ):
                    verify.validate_compile_macro_states(
                        arguments,
                        experiment=verify.EXPERIMENTS[experiment.name],
                        role="treatment",
                        repo=self.REPO,
                    )

    def test_noncanonical_duplicate_feature_source_is_rejected(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            control, treatment = self.build_pair(experiment, output_dir)
            duplicate = f"/alternate{causal.HNSW_COMMON_SOURCE}"
            for role in ("control", "treatment"):
                path = output_dir / f"{role}-arguments.json"
                arguments = json.loads(path.read_text(encoding="utf-8"))
                canonical = causal.canonical_source(
                    self.REPO,
                    causal.HNSW_COMMON_SOURCE,
                )
                arguments[duplicate] = list(arguments[canonical])
                path.write_text(json.dumps(arguments), encoding="utf-8")
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "not exactly the canonical repository source",
            ):
                causal.compare_builds(
                    control,
                    treatment,
                    experiment=experiment,
                    repo=self.REPO,
                    output_dir=output_dir,
                )

    def test_cmake_home_directory_must_bind_to_canonical_repo(self) -> None:
        for module, error in (
            (causal, causal.CausalFailure),
            (verify, verify.CausalVerificationError),
        ):
            with self.subTest(module=module.__name__):
                module.validate_cmake_home_directory(
                    {"CMAKE_HOME_DIRECTORY": str(self.REPO)},
                    repo=self.REPO,
                    label="test",
                )
                with self.assertRaisesRegex(
                    error,
                    "CMAKE_HOME_DIRECTORY",
                ):
                    module.validate_cmake_home_directory(
                        {"CMAKE_HOME_DIRECTORY": "/different/repo"},
                        repo=self.REPO,
                        label="test",
                    )

    def test_verifier_compares_every_build_tool_identity(self) -> None:
        tools = {
            name: {
                "resolved_path": f"/tools/{name}",
                "sha256": "a" * 64,
            }
            for name, _, _ in verify.BUILD_TOOL_SPECS
        }
        verify.validate_toolchain_identity(tools, copy.deepcopy(tools))
        for name, _, _ in verify.BUILD_TOOL_SPECS:
            with self.subTest(tool=name):
                treatment = copy.deepcopy(tools)
                treatment[name]["sha256"] = "b" * 64
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    f"different {name}",
                ):
                    verify.validate_toolchain_identity(tools, treatment)

    def test_ninja_dependency_settlement_accepts_identical_stored_map(
        self,
    ) -> None:
        live_output = "/build/live.o"
        outside_output = "/build/outside.o"

        def record(output: str, dependency: str) -> dict:
            return {
                "output": output,
                "dependency_count": 1,
                "deps_mtime": 123,
                "status": "VALID",
                "dependencies": [dependency],
            }

        before = {
            live_output: record(live_output, "/source/live.cpp"),
            outside_output: record(outside_output, "/source/outside.cpp"),
        }
        after = copy.deepcopy(before)
        closure = {
            "dependency_graph": [copy.deepcopy(before[live_output])]
        }
        self.assertEqual(
            causal.validate_ninja_dependency_settlement(
                before,
                after,
                closure=closure,
                role="control",
            ),
            ([], []),
        )
        self.assertEqual(
            verify.validate_ninja_dependency_settlement(
                before,
                after,
                closure=closure,
                context="test settlement",
            ),
            ([], [], 1),
        )

    def test_ninja_dependency_archive_binds_stored_fields(self) -> None:
        output = "/build/live.o"
        report = {
            output: {
                "output": output,
                "dependency_count": 2,
                "deps_mtime": 123,
                "status": "VALID",
                "dependencies": [
                    "/source/live.cpp",
                    "/source/live.h",
                ],
            }
        }
        parsed_archive = copy.deepcopy(report)
        parsed_archive[output]["dependencies"].reverse()
        parsed_archive[output]["status"] = "STALE"
        causal.validate_ninja_dependency_archive(
            report,
            parsed_archive,
            label="test archive",
        )
        verify.validate_ninja_dependency_archive(
            report,
            parsed_archive,
            context="test archive",
        )
        mutations = {
            "membership": lambda record: record.update(
                {
                    "dependencies": [
                        "/source/live.cpp",
                        "/source/substituted.h",
                    ]
                }
            ),
            "multiplicity": lambda record: record.update(
                {
                    "dependencies": [
                        "/source/live.cpp",
                        "/source/live.cpp",
                    ]
                }
            ),
            "mtime": lambda record: record.update({"deps_mtime": 124}),
        }
        for label, mutate in mutations.items():
            with self.subTest(case=label):
                mutated = copy.deepcopy(parsed_archive)
                mutate(mutated[output])
                with self.assertRaises(causal.CausalFailure):
                    causal.validate_ninja_dependency_archive(
                        report,
                        mutated,
                        label="test archive",
                    )
                with self.assertRaises(verify.CausalVerificationError):
                    verify.validate_ninja_dependency_archive(
                        report,
                        mutated,
                        context="test archive",
                    )

    def test_ninja_dependency_settlement_rejects_semantic_changes(
        self,
    ) -> None:
        live_output = "/build/live.o"
        outside_output = "/build/outside.o"

        def record(
            output: str,
            dependency: str,
            *,
            deps_mtime: int = 123,
            status: str = "VALID",
        ) -> dict:
            return {
                "output": output,
                "dependency_count": 1,
                "deps_mtime": deps_mtime,
                "status": status,
                "dependencies": [dependency],
            }

        baseline = {
            live_output: record(live_output, "/source/live.cpp"),
            outside_output: record(outside_output, "/source/outside.cpp"),
        }
        closure = {
            "dependency_graph": [copy.deepcopy(baseline[live_output])]
        }
        matching_substitution = copy.deepcopy(baseline)
        matching_substitution[live_output] = record(
            live_output,
            "/source/substituted.cpp",
        )
        cases = {
            "matching before and after differ from closure": matching_substitution,
            "selected dependency": {
                live_output: record(live_output, "/source/substituted.cpp"),
                outside_output: baseline[outside_output],
            },
            "selected mtime": {
                live_output: record(
                    live_output,
                    "/source/live.cpp",
                    deps_mtime=124,
                ),
                outside_output: baseline[outside_output],
            },
            "selected stale": {
                live_output: record(
                    live_output,
                    "/source/live.cpp",
                    status="STALE",
                ),
                outside_output: baseline[outside_output],
            },
            "selected removal": {
                outside_output: baseline[outside_output],
            },
            "outside removal": {
                live_output: baseline[live_output],
            },
            "outside addition": {
                **baseline,
                "/build/added.o": record(
                    "/build/added.o",
                    "/source/added.cpp",
                ),
            },
            "retained outside record": {
                live_output: baseline[live_output],
                outside_output: record(
                    outside_output,
                    "/source/substituted.cpp",
                ),
            },
        }
        for label, after in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(causal.CausalFailure):
                    causal.validate_ninja_dependency_settlement(
                        baseline,
                        after,
                        closure=closure,
                        role="control",
                    )
                with self.assertRaises(verify.CausalVerificationError):
                    verify.validate_ninja_dependency_settlement(
                        baseline,
                        after,
                        closure=closure,
                        context="test settlement",
                    )

    def test_clean_transcript_parsers_accept_relative_paths_and_spaces(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            stdout = ninja_clean_transcript(
                "target with spaces",
                ["relative output.o", "second output.o"],
            )
            completed = subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=stdout,
                stderr="",
            )
            self.assertEqual(
                causal.parse_ninja_clean_transcript(
                    completed,
                    build_directory=build_directory,
                    requested_target="target with spaces",
                    context="test clean",
                ),
                [
                    build_directory / "relative output.o",
                    build_directory / "second output.o",
                ],
            )
            self.assertEqual(
                verify.parse_ninja_clean_transcript(
                    stdout,
                    "",
                    build_directory=build_directory,
                    requested_target="target with spaces",
                    context="test clean",
                ),
                [
                    str(build_directory / "relative output.o"),
                    str(build_directory / "second output.o"),
                ],
            )
            combined_stdout = ninja_clean_transcript(
                "target with spaces",
                ["relative output.o", "module output.pcm"],
                supplemental_targets=["module output.pcm"],
            )
            combined = subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=combined_stdout,
                stderr="",
            )
            self.assertEqual(
                causal.parse_ninja_clean_transcript(
                    combined,
                    build_directory=build_directory,
                    requested_target="target with spaces",
                    supplemental_targets=["module output.pcm"],
                    context="test combined clean",
                ),
                [
                    build_directory / "relative output.o",
                    build_directory / "module output.pcm",
                ],
            )
            self.assertEqual(
                verify.parse_ninja_clean_transcript(
                    combined_stdout,
                    "",
                    build_directory=build_directory,
                    requested_target="target with spaces",
                    supplemental_targets=["module output.pcm"],
                    context="test combined clean",
                ),
                [
                    str(build_directory / "relative output.o"),
                    str(build_directory / "module output.pcm"),
                ],
            )

    def test_clean_transcript_parsers_reject_unsafe_or_ambiguous_input(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            cases = (
                (
                    ninja_clean_transcript("target", ["../outside"]),
                    "",
                    "canonical relative",
                ),
                (
                    ninja_clean_transcript("target", ["/outside"]),
                    "",
                    "canonical relative",
                ),
                (
                    ninja_clean_transcript("target", ["output", "output"]),
                    "",
                    "repeats removal path",
                ),
                (
                    ninja_clean_transcript(
                        "target",
                        ["output", str(build_directory / "output")],
                    ),
                    "",
                    "canonical relative",
                ),
                (
                    "Cleaning...\nTarget target\nRemove output\n2 files.\n",
                    "",
                    "removal count differs",
                ),
                (
                    "Cleaning...\nTarget target\nRemove output\n01 files.\n",
                    "",
                    "invalid file count",
                ),
                (
                    "Cleaning...\nTarget target\nUnexpected output\n1 files.\n",
                    "",
                    "unexpected line",
                ),
                (
                    ninja_clean_transcript("target", ["output"]).replace(
                        "\n",
                        "\r\n",
                    ),
                    "",
                    "canonical newline-terminated text",
                ),
                (
                    ninja_clean_transcript("target", ["output"]),
                    "warning\n",
                    "emitted stderr",
                ),
            )
            for stdout, stderr, message in cases:
                with self.subTest(message=message):
                    completed = subprocess.CompletedProcess(
                        ["/tools/ninja"],
                        0,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    with self.assertRaisesRegex(causal.CausalFailure, message):
                        causal.parse_ninja_clean_transcript(
                            completed,
                            build_directory=build_directory,
                            requested_target="target",
                            context="test clean",
                        )
                    with self.assertRaisesRegex(
                        verify.CausalVerificationError,
                        message,
                    ):
                        verify.parse_ninja_clean_transcript(
                            stdout,
                            stderr,
                            build_directory=build_directory,
                            requested_target="target",
                            context="test clean",
                        )

    def test_clean_path_classifier_rejects_symlinks_and_tracks_absence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            outside_directory = root / "outside"
            build_directory.mkdir()
            outside_directory.mkdir()
            regular = build_directory / "regular.o"
            absent = build_directory / "absent.modmap"
            regular.write_bytes(b"object")
            existing, missing = causal.classify_ninja_clean_paths(
                {regular, absent},
                build_directory=build_directory,
                context="test clean",
            )
            self.assertEqual(existing, {regular})
            self.assertEqual(missing, {absent})

            outside = outside_directory / "sentinel"
            outside.write_bytes(b"outside")
            terminal_symlink = build_directory / "terminal-link"
            terminal_symlink.symlink_to(outside)
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "not a regular file",
            ):
                causal.classify_ninja_clean_paths(
                    {terminal_symlink},
                    build_directory=build_directory,
                    context="test clean",
                )

            ancestor_symlink = build_directory / "linked-directory"
            ancestor_symlink.symlink_to(outside_directory, target_is_directory=True)
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "symlink ancestor",
            ):
                causal.classify_ninja_clean_paths(
                    {ancestor_symlink / "sentinel"},
                    build_directory=build_directory,
                    context="test clean",
                )
            self.assertEqual(outside.read_bytes(), b"outside")

    def test_build_tree_manifest_hashes_content_and_tracks_directories(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            nested = build_directory / "nested"
            nested.mkdir()
            artifact = nested / "artifact.o"
            artifact.write_bytes(b"before")
            original_mtime = artifact.stat().st_mtime_ns
            before = causal.build_tree_stat_manifest(
                build_directory,
                excluded_relative_paths=set(),
            )
            artifact.write_bytes(b"after!")
            os.utime(
                artifact,
                ns=(original_mtime, original_mtime),
            )
            after = causal.build_tree_stat_manifest(
                build_directory,
                excluded_relative_paths=set(),
            )
            before_by_path = {item["path"]: item for item in before}
            after_by_path = {item["path"]: item for item in after}
            self.assertEqual(before_by_path["."]["type"], "directory")
            self.assertEqual(before_by_path["nested"]["type"], "directory")
            self.assertEqual(
                set(before_by_path["nested"]),
                {"path", "type", *causal.STAT_MANIFEST_KEYS},
            )
            self.assertEqual(
                before_by_path["nested/artifact.o"]["bytes"],
                after_by_path["nested/artifact.o"]["bytes"],
            )
            self.assertEqual(
                before_by_path["nested/artifact.o"]["mtime_ns"],
                after_by_path["nested/artifact.o"]["mtime_ns"],
            )
            self.assertNotEqual(
                before_by_path["nested/artifact.o"]["sha256"],
                after_by_path["nested/artifact.o"]["sha256"],
            )

    def test_build_tree_manifest_fails_closed_on_traversal_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            with mock.patch.object(
                causal.os,
                "scandir",
                side_effect=PermissionError("denied"),
            ), self.assertRaisesRegex(
                causal.CausalFailure,
                "could not be traversed",
            ):
                causal.build_tree_stat_manifest(
                    build_directory,
                    excluded_relative_paths=set(),
                )

    def test_build_tree_manifest_rejects_directory_replacement_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            original_scandir = os.scandir

            def mutating_scandir(path: os.PathLike[str]) -> os.ScandirIterator:
                iterator = original_scandir(path)
                os.chmod(path, 0o755)
                return iterator

            with mock.patch.object(
                causal.os,
                "scandir",
                side_effect=mutating_scandir,
            ), self.assertRaisesRegex(
                causal.CausalFailure,
                "directory changed during traversal",
            ):
                causal.build_tree_stat_manifest(
                    build_directory,
                    excluded_relative_paths=set(),
                )

    def test_build_tree_manifest_rejects_excluded_log_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            quarantine_directory = root / "quarantine"
            build_directory.mkdir()
            quarantine_directory.mkdir()
            outside = root / "outside.log"
            outside.write_bytes(b"# ninja log v7\n")
            (build_directory / ".ninja_log").symlink_to(outside)
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "Excluded build path is not a regular file",
            ):
                causal.build_tree_stat_manifest(
                    build_directory,
                    excluded_relative_paths={".ninja_log"},
                )

    def test_stable_snapshot_rejects_mutation_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "mutable.log"
            path.write_bytes(b"# ninja log v7\n")
            original_read = os.read
            mutated = False

            def mutating_read(descriptor: int, count: int) -> bytes:
                nonlocal mutated
                content = original_read(descriptor, count)
                if content and not mutated:
                    with path.open("ab") as destination:
                        destination.write(b"x")
                        destination.flush()
                        os.fsync(destination.fileno())
                    mutated = True
                return content

            with mock.patch.object(
                causal.os,
                "read",
                side_effect=mutating_read,
            ), self.assertRaisesRegex(
                causal.CausalFailure,
                "changed while being read",
            ):
                causal.stable_regular_file_bytes(
                    path,
                    context="test mutable snapshot",
                    maximum_bytes=1024,
                )

    def test_stable_snapshot_rejects_mutation_after_descriptor_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "mutable.log"
            path.write_bytes(b"# ninja log v7\n")
            original = causal.stable_regular_file_descriptor
            mutated = False

            def mutate_after_hash(*args: object, **kwargs: object) -> object:
                nonlocal mutated
                result = original(*args, **kwargs)
                if not mutated:
                    path.write_bytes(b"changed after hash\n")
                    mutated = True
                return result

            with mock.patch.object(
                causal,
                "stable_regular_file_descriptor",
                side_effect=mutate_after_hash,
            ), self.assertRaisesRegex(
                causal.CausalFailure,
                "changed while being read",
            ):
                causal.stable_regular_file_bytes(
                    path,
                    context="test post-hash mutation",
                    maximum_bytes=1024,
                )

    def test_build_tree_manifest_rejects_change_between_complete_passes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            artifact = build_directory / "artifact.o"
            artifact.write_bytes(b"first")
            metadata = artifact.stat()
            original = causal._build_tree_stat_manifest_once
            calls = 0

            def mutate_between_passes(*args: object, **kwargs: object) -> object:
                nonlocal calls
                result = original(*args, **kwargs)
                calls += 1
                if calls == 1:
                    artifact.write_bytes(b"other")
                    os.utime(
                        artifact,
                        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
                    )
                return result

            with mock.patch.object(
                causal,
                "_build_tree_stat_manifest_once",
                side_effect=mutate_between_passes,
            ), self.assertRaisesRegex(
                causal.CausalFailure,
                "changed between complete snapshots",
            ):
                causal.build_tree_stat_manifest(
                    build_directory,
                    excluded_relative_paths=set(),
                )
            self.assertEqual(calls, 2)

    def test_target_identities_reject_restored_ancestor_substitution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            live = root / "live"
            original = root / "original"
            substitute = root / "substitute"
            live.mkdir()
            substitute.mkdir()
            (live / "artifact.o").write_bytes(b"trusted")
            (substitute / "artifact.o").write_bytes(b"hostile")
            target = live / "artifact.o"
            live.rename(original)
            substitute.rename(live)
            descriptor_reader = causal.stable_regular_file_descriptor
            restored = False

            def restore_after_hash(*args: object, **kwargs: object) -> object:
                nonlocal restored
                result = descriptor_reader(*args, **kwargs)
                if not restored:
                    live.rename(substitute)
                    original.rename(live)
                    restored = True
                return result

            with mock.patch.object(
                causal,
                "stable_regular_file_descriptor",
                side_effect=restore_after_hash,
            ), self.assertRaisesRegex(
                causal.CausalFailure,
                "changed while being read",
            ):
                causal.target_file_identities({target})
            self.assertTrue(restored)
            self.assertEqual(target.read_bytes(), b"trusted")

    def test_target_identities_accept_distinct_hardlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            first = root / "first.o"
            second = root / "second.o"
            first.write_bytes(b"same inode")
            os.link(first, second)
            records = causal.target_file_identities({first, second})
            self.assertEqual(
                [record["path"] for record in records],
                [str(first), str(second)],
            )
            self.assertEqual(
                {record["sha256"] for record in records},
                {hashlib.sha256(b"same inode").hexdigest()},
            )

    def test_build_tree_manifest_uses_descriptor_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            nested = build_directory / "nested"
            nested.mkdir()
            (nested / "artifact.o").write_bytes(b"object")
            original_scandir = os.scandir
            scandir_arguments: list[object] = []

            def tracking_scandir(path: object) -> os.ScandirIterator:
                scandir_arguments.append(path)
                return original_scandir(path)

            with mock.patch.object(
                causal.os,
                "scandir",
                side_effect=tracking_scandir,
            ):
                causal.build_tree_stat_manifest(
                    build_directory,
                    excluded_relative_paths=set(),
                )
            self.assertTrue(scandir_arguments)
            self.assertTrue(
                all(isinstance(argument, int) for argument in scandir_arguments)
            )
            self.assertEqual(len(scandir_arguments), 4)

    def test_verifier_rejects_impossible_tree_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            nested = build_directory / "nested"
            nested.mkdir()
            artifact = nested / "artifact.o"
            artifact.write_bytes(b"object")
            os.link(artifact, nested / "alias.o")
            (build_directory / "!artifact").write_bytes(b"early")
            (build_directory / "link").symlink_to("nested/artifact.o")
            records = causal.build_tree_stat_manifest(
                build_directory,
                excluded_relative_paths=set(),
            )
        self.assertEqual(records[0]["path"], ".")
        verify.validate_idempotence_tree_manifest(
            records,
            context="test tree",
        )
        by_path = {record["path"]: record for record in records}

        wrong_mode = copy.deepcopy(records)
        wrong_mode[-1]["mode"] = stat.S_IFDIR | 0o755
        missing_parent = [
            copy.deepcopy(by_path["."]),
            copy.deepcopy(by_path["nested/artifact.o"]),
        ]
        noncanonical = copy.deepcopy(records)
        noncanonical[-1]["path"] = "nested//artifact.o"
        conflicting_hardlink = copy.deepcopy(records)
        hardlink = next(
            record
            for record in conflicting_hardlink
            if record["path"] == "nested/alias.o"
        )
        hardlink["sha256"] = "f" * 64
        with self.assertRaisesRegex(
            causal.CausalFailure,
            "conflicting records",
        ):
            causal.validate_build_tree_inode_consistency(conflicting_hardlink)
        nul_symlink = copy.deepcopy(records)
        symlink = next(
            record
            for record in nul_symlink
            if record["path"] == "link"
        )
        symlink["target"] = "bad\0target"
        symlink["bytes"] = len(symlink["target"].encode("utf-8"))
        for label, value, message in (
            ("wrong mode", wrong_mode, "mode does not match"),
            ("missing parent", missing_parent, "lacks a directory parent"),
            ("noncanonical", noncanonical, "canonical relative path"),
            ("hardlink conflict", conflicting_hardlink, "conflicting records"),
            ("NUL symlink", nul_symlink, "target size is invalid"),
        ):
            with self.subTest(case=label), self.assertRaisesRegex(
                verify.CausalVerificationError,
                message,
            ):
                verify.validate_idempotence_tree_manifest(
                    value,
                    context="test tree",
                )

        with mock.patch.object(
            verify,
            "BUILD_TREE_MAX_ENTRIES",
            len(records) - 1,
        ), self.assertRaisesRegex(
            verify.CausalVerificationError,
            "entry count",
        ):
            verify.validate_idempotence_tree_manifest(
                records,
                context="test tree",
            )

        directory_mode = stat.S_IFDIR | 0o755
        file_mode = stat.S_IFREG | 0o644
        boundary: list[dict[str, object]] = []
        parts: list[str] = []
        for depth in range(verify.BUILD_TREE_MAX_DEPTH + 1):
            path = "." if depth == 0 else "/".join(parts)
            boundary.append(
                {
                    "path": path,
                    "type": "directory",
                    "device": 1,
                    "inode": depth + 1,
                    "mode": directory_mode,
                    "links": 1,
                    "bytes": 0,
                    "mtime_ns": 0,
                    "ctime_ns": 0,
                }
            )
            parts.append(f"d{depth}")
        boundary.append(
            {
                "path": f"{boundary[-1]['path']}/artifact.o",
                "type": "file",
                "device": 1,
                "inode": len(boundary) + 1,
                "mode": file_mode,
                "links": 1,
                "bytes": 1,
                "mtime_ns": 0,
                "ctime_ns": 0,
                "sha256": "a" * 64,
            }
        )
        boundary.sort(key=lambda record: str(record["path"]).encode("utf-8"))
        verify.validate_idempotence_tree_manifest(
            boundary,
            context="test boundary tree",
        )
        too_deep = copy.deepcopy(boundary)
        deepest = max(
            (
                record
                for record in too_deep
                if record["type"] == "directory"
            ),
            key=lambda record: len(PurePosixPath(str(record["path"])).parts),
        )
        child_path = f"{deepest['path']}/too-deep"
        too_deep.append(
            {
                **copy.deepcopy(deepest),
                "path": child_path,
                "inode": 10_000,
            }
        )
        too_deep.sort(key=lambda record: str(record["path"]).encode("utf-8"))
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "depth limit",
        ):
            verify.validate_idempotence_tree_manifest(
                too_deep,
                context="test deep tree",
            )

    def test_ninja_v4_parser_accepts_all_padding_and_maximum_mtime(
        self,
    ) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        content = ninja_deps_database(
            ["out0", "abc", "de", "f"],
            [(0, 0x7FFFFFFFFFFFFFFF, [1, 2, 3, 3])],
        )
        spellings = {str(build_directory / "out0"): "out0"}
        expected = {
            str(build_directory / "out0"): {
                "output": str(build_directory / "out0"),
                "dependency_count": 4,
                "deps_mtime": 0x7FFFFFFFFFFFFFFF,
                "dependencies": [
                    str(build_directory / "abc"),
                    str(build_directory / "de"),
                    str(build_directory / "f"),
                    str(build_directory / "f"),
                ],
            }
        }
        for module in (causal, verify):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    parse_ninja_deps_fixture(
                        module,
                        content,
                        build_directory=build_directory,
                        expected_output_spellings=spellings,
                    ),
                    expected,
                )

    def test_ninja_v4_parser_last_record_wins_and_preserves_duplicate_deps(
        self,
    ) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        content = ninja_deps_database(
            ["out.o", "first.h", "second.h"],
            [
                (0, 100, [1]),
                (0, 200, [2, 2, 1]),
            ],
        )
        spellings = {str(build_directory / "out.o"): "out.o"}
        for module in (causal, verify):
            with self.subTest(module=module.__name__):
                parsed = parse_ninja_deps_fixture(
                    module,
                    content,
                    build_directory=build_directory,
                    expected_output_spellings=spellings,
                )
                self.assertEqual(
                    parsed[str(build_directory / "out.o")],
                    {
                        "output": str(build_directory / "out.o"),
                        "dependency_count": 3,
                        "deps_mtime": 200,
                        "dependencies": [
                            str(build_directory / "second.h"),
                            str(build_directory / "second.h"),
                            str(build_directory / "first.h"),
                        ],
                    },
                )
                zero_final = parse_ninja_deps_fixture(
                    module,
                    ninja_deps_database(
                        ["out.o", "first.h"],
                        [(0, 100, [1]), (0, 300, [])],
                    ),
                    build_directory=build_directory,
                    expected_output_spellings=spellings,
                )
                self.assertEqual(
                    zero_final[str(build_directory / "out.o")],
                    {
                        "output": str(build_directory / "out.o"),
                        "dependency_count": 0,
                        "deps_mtime": 300,
                        "dependencies": [],
                    },
                )

    def test_ninja_v4_parser_rejects_malformed_envelope_and_sizes(
        self,
    ) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        spellings = {str(build_directory / "out.o"): "out.o"}
        valid = ninja_deps_database(["out.o"], [(0, 1, [])])
        path_prefix = (
            NINJA_DEPS_HEADER + ninja_deps_path_record("out.o", 0)
        )
        cases = {
            "signature": b"! ninjadeps\n" + valid[12:],
            "version": b"# ninjadeps\n" + struct.pack("<I", 3) + valid[16:],
            "zero size": NINJA_DEPS_HEADER + struct.pack("<I", 0),
            "oversized record": (
                NINJA_DEPS_HEADER
                + struct.pack("<I", causal.NINJA_DEPS_MAX_RECORD_BYTES + 1)
            ),
            "truncated size": valid + b"\0\1",
            "truncated payload": (
                NINJA_DEPS_HEADER + struct.pack("<I", 8) + b"abc"
            ),
            "short path": (
                NINJA_DEPS_HEADER + ninja_deps_record(b"\0" * 4)
            ),
            "unaligned path": (
                NINJA_DEPS_HEADER
                + ninja_deps_record(b"abc" + struct.pack("<I", 0xFFFFFFFF))
            ),
            "short dependency": (
                path_prefix
                + ninja_deps_record(b"\0" * 8, dependency=True)
            ),
            "unaligned dependency": (
                path_prefix
                + ninja_deps_record(b"\0" * 13, dependency=True)
            ),
        }
        for name, content in cases.items():
            for module, error in (
                (causal, causal.CausalFailure),
                (verify, verify.CausalVerificationError),
            ):
                with self.subTest(case=name, module=module.__name__):
                    with self.assertRaises(error):
                        parse_ninja_deps_fixture(
                            module,
                            content,
                            build_directory=build_directory,
                            expected_output_spellings=spellings,
                        )

    def test_ninja_v4_parser_rejects_malformed_paths(self) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        spellings = {str(build_directory / "out.o"): "out.o"}
        cases = {
            "empty": ninja_deps_path_record(b"", 0),
            "padding": ninja_deps_path_record(b"abc", 0, padding=5),
            "embedded NUL": ninja_deps_path_record(b"a\0b", 0),
            "UTF-8": ninja_deps_path_record(b"\xff", 0),
            "newline": ninja_deps_path_record(b"a\n", 0),
            "carriage return": ninja_deps_path_record(b"a\r", 0),
            "checksum": ninja_deps_path_record("out.o", 0, checksum=0),
            "skipped ID": ninja_deps_path_record("out.o", 1),
            "duplicate": (
                ninja_deps_path_record("out.o", 0)
                + ninja_deps_path_record("out.o", 1)
            ),
        }
        for name, records in cases.items():
            content = NINJA_DEPS_HEADER + records
            for module, error in (
                (causal, causal.CausalFailure),
                (verify, verify.CausalVerificationError),
            ):
                with self.subTest(case=name, module=module.__name__):
                    with self.assertRaises(error):
                        parse_ninja_deps_fixture(
                            module,
                            content,
                            build_directory=build_directory,
                            expected_output_spellings=spellings,
                        )

    def test_ninja_v4_parser_rejects_invalid_ids_and_negative_mtime(
        self,
    ) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        spellings = {str(build_directory / "out.o"): "out.o"}
        paths = b"".join(
            (
                ninja_deps_path_record("out.o", 0),
                ninja_deps_path_record("input.h", 1),
            )
        )
        cases = {
            "future output ID": ninja_deps_dependency_record(2, 1, []),
            "future dependency ID": ninja_deps_dependency_record(0, 1, [2]),
            "maximum dependency ID": ninja_deps_dependency_record(
                0,
                1,
                [0xFFFFFFFF],
            ),
            "negative mtime": ninja_deps_dependency_record(0, 1 << 63, [1]),
        }
        for name, dependency in cases.items():
            content = NINJA_DEPS_HEADER + paths + dependency
            for module, error in (
                (causal, causal.CausalFailure),
                (verify, verify.CausalVerificationError),
            ):
                with self.subTest(case=name, module=module.__name__):
                    with self.assertRaises(error):
                        parse_ninja_deps_fixture(
                            module,
                            content,
                            build_directory=build_directory,
                            expected_output_spellings=spellings,
                        )

    def test_ninja_v4_parser_filters_dead_records_and_binds_outputs(
        self,
    ) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        output = str(build_directory / "out.o")
        content = ninja_deps_database(
            ["out.o", "input.h", "dead.o"],
            [(0, 1, [1]), (2, 2, [])],
        )
        for module, error in (
            (causal, causal.CausalFailure),
            (verify, verify.CausalVerificationError),
        ):
            with self.subTest(module=module.__name__):
                parsed = parse_ninja_deps_fixture(
                    module,
                    content,
                    build_directory=build_directory,
                    expected_output_spellings={output: "out.o"},
                )
                self.assertEqual(set(parsed), {output})

            invalid_spellings = (
                {},
                {
                    output: "out.o",
                    str(build_directory / "alias.o"): "out.o",
                },
                {str(build_directory / "wrong.o"): "out.o"},
                {str(build_directory / "missing.o"): "missing.o"},
            )
            for spellings in invalid_spellings:
                with self.subTest(
                    module=module.__name__,
                    spellings=spellings,
                ), self.assertRaises(error):
                    parse_ninja_deps_fixture(
                        module,
                        content,
                        build_directory=build_directory,
                        expected_output_spellings=spellings,
                    )

            without_dependency = ninja_deps_database(["out.o"], [])
            with self.subTest(
                module=module.__name__,
                case="missing dependency record",
            ), self.assertRaises(error):
                parse_ninja_deps_fixture(
                    module,
                    without_dependency,
                    build_directory=build_directory,
                    expected_output_spellings={output: "out.o"},
                )

    def test_ninja_v4_parser_enforces_all_resource_limits(self) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        output = str(build_directory / "out.o")
        spellings = {output: "out.o"}
        content = ninja_deps_database(
            ["out.o", "one.h", "two.h"],
            [(0, 1, [1, 2])],
        )
        limits = {
            "NINJA_DEPS_MAX_OUTPUTS": 0,
            "NINJA_DEPS_MAX_RECORD_BYTES": 7,
            "NINJA_DEPS_MAX_NODES": 1,
            "NINJA_DEPS_MAX_RECORDS": 2,
            "NINJA_DEPS_MAX_DEPENDENCIES_PER_OUTPUT": 1,
            "NINJA_DEPS_MAX_TOTAL_DEPENDENCIES": 1,
        }
        for module, error in (
            (causal, causal.CausalFailure),
            (verify, verify.CausalVerificationError),
        ):
            for constant, value in limits.items():
                with self.subTest(
                    module=module.__name__,
                    constant=constant,
                ), mock.patch.object(
                    module,
                    constant,
                    value,
                ), self.assertRaises(error):
                    parse_ninja_deps_fixture(
                        module,
                        content,
                        build_directory=build_directory,
                        expected_output_spellings=spellings,
                    )

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / ".ninja_deps"
            database_path.write_bytes(content)
            for module, error in (
                (causal, causal.CausalFailure),
                (verify, verify.CausalVerificationError),
            ):
                with self.subTest(
                    module=module.__name__,
                    constant="NINJA_DEPS_DATABASE_MAX_BYTES",
                ), mock.patch.object(
                    module,
                    "NINJA_DEPS_DATABASE_MAX_BYTES",
                    1,
                ), self.assertRaises(error):
                    parse_archived_ninja_deps_fixture(
                        module,
                        database_path,
                        build_directory=build_directory,
                        expected_output_spellings=spellings,
                    )

    def test_ninja_v4_archive_rejects_mutation_during_parse(self) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        output = str(build_directory / "out.o")
        spellings = {output: "out.o"}
        content = ninja_deps_database(["out.o"], [(0, 1, [])])
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / ".ninja_deps"
            for module, error in (
                (causal, causal.CausalFailure),
                (verify, verify.CausalVerificationError),
            ):
                database_path.write_bytes(content)
                parser = module.parse_ninja_dependency_database

                def mutate_after_parse(*args, **kwargs):
                    result = parser(*args, **kwargs)
                    database_path.write_bytes(content + b"\0")
                    return result

                with self.subTest(module=module.__name__), mock.patch.object(
                    module,
                    "parse_ninja_dependency_database",
                    side_effect=mutate_after_parse,
                ), self.assertRaisesRegex(error, "changed during parse"):
                    parse_archived_ninja_deps_fixture(
                        module,
                        database_path,
                        build_directory=build_directory,
                        expected_output_spellings=spellings,
                    )

    def test_ninja_v4_archive_never_executes_subprocesses(self) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        output = str(build_directory / "out.o")
        spellings = {output: "out.o"}
        content = ninja_deps_database(["out.o"], [(0, 1, [])])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / ".ninja_deps"
            database_path.write_bytes(content)
            (root / "build.ninja").write_text(
                "include /untrusted/project/input.ninja\n",
                encoding="utf-8",
            )
            for module in (causal, verify):
                with self.subTest(module=module.__name__), mock.patch.object(
                    subprocess,
                    "Popen",
                    side_effect=AssertionError("subprocess execution"),
                ), mock.patch.object(
                    subprocess,
                    "run",
                    side_effect=AssertionError("subprocess execution"),
                ):
                    self.assertEqual(
                        set(
                            parse_archived_ninja_deps_fixture(
                                module,
                                database_path,
                                build_directory=build_directory,
                                expected_output_spellings=spellings,
                            )
                        ),
                        {output},
                    )

    def test_ninja_v4_runner_and_verifier_match_for_every_truncation(
        self,
    ) -> None:
        build_directory = Path("/tmp/ninja-v4-build")
        spellings = {str(build_directory / "out.o"): "out.o"}
        content = ninja_deps_database(
            ["out.o", "input.h"],
            [(0, 123, [1, 1])],
        )
        for size in range(len(content) + 1):
            outcomes = []
            for module, error in (
                (causal, causal.CausalFailure),
                (verify, verify.CausalVerificationError),
            ):
                try:
                    parsed = parse_ninja_deps_fixture(
                        module,
                        content[:size],
                        build_directory=build_directory,
                        expected_output_spellings=spellings,
                    )
                except error:
                    outcomes.append(("error", None))
                else:
                    outcomes.append(("success", parsed))
            with self.subTest(size=size):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_ninja_v4_parser_matches_real_relative_and_absolute_databases(
        self,
    ) -> None:
        for absolute_output in (False, True):
            with self.subTest(absolute_output=absolute_output):
                with tempfile.TemporaryDirectory() as directory:
                    (
                        build_directory,
                        records,
                        spellings,
                        content,
                    ) = self.create_ninja_deps_fixture(
                        Path(directory).resolve(),
                        absolute_output=absolute_output,
                    )
                    if absolute_output:
                        self.assertEqual(
                            spellings,
                            {
                                str(build_directory / "object.o"): str(
                                    build_directory / "object.o"
                                )
                            },
                        )
                    else:
                        self.assertEqual(
                            spellings,
                            {str(build_directory / "object.o"): "object.o"},
                        )
                    for module in (causal, verify):
                        parsed = parse_ninja_deps_fixture(
                            module,
                            content,
                            build_directory=build_directory,
                            expected_output_spellings=spellings,
                        )
                        if module is causal:
                            causal.validate_ninja_dependency_archive(
                                records,
                                parsed,
                                label="real Ninja database",
                            )
                        else:
                            verify.validate_ninja_dependency_archive(
                                records,
                                parsed,
                                context="real Ninja database",
                            )

    def test_ninja_log_append_rejects_malformed_numeric_fields(self) -> None:
        build_directory = Path("/tmp/build")
        before = b"# ninja log v7\n"
        output = str(
            build_directory / "CMakeFiles/cmake.verify_globs"
        ).encode("utf-8")
        malformed = before + b"x\t2\t3\t" + output + b"\t4\n"
        for module, error in (
            (causal, causal.CausalFailure),
            (verify, verify.CausalVerificationError),
        ):
            with self.subTest(module=module.__name__), self.assertRaisesRegex(
                error,
                "glob-check edge",
            ):
                module.validate_ninja_log_append(
                    before,
                    malformed,
                    build_directory=build_directory,
                    context="test malformed log",
                )

    def test_ninja_log_append_requires_exact_v7(self) -> None:
        build_directory = Path("/tmp/build")
        output = str(
            build_directory / "CMakeFiles/cmake.verify_globs"
        ).encode("utf-8")
        row = b"1\t2\t3\t" + output + b"\t4\n"
        for module, error in (
            (causal, causal.CausalFailure),
            (verify, verify.CausalVerificationError),
        ):
            with self.subTest(module=module.__name__, version=7):
                module.validate_ninja_log_append(
                    b"# ninja log v7\n",
                    b"# ninja log v7\n" + row,
                    build_directory=build_directory,
                    context="test v7 log",
                )
            for version in (5, 6, 8):
                with (
                    self.subTest(module=module.__name__, version=version),
                    self.assertRaisesRegex(error, "v7|version"),
                ):
                    header = f"# ninja log v{version}\n".encode("ascii")
                    module.validate_ninja_log_append(
                        header,
                        header + row,
                        build_directory=build_directory,
                        context="test wrong log version",
                    )
            for command_hash in (b"04", b"1234567890abcdef0", b"ABC"):
                with (
                    self.subTest(
                        module=module.__name__,
                        command_hash=command_hash,
                    ),
                    self.assertRaisesRegex(error, "glob-check edge"),
                ):
                    module.validate_ninja_log_append(
                        b"# ninja log v7\n",
                        (
                            b"# ninja log v7\n"
                            + b"1\t2\t3\t"
                            + output
                            + b"\t"
                            + command_hash
                            + b"\n"
                        ),
                        build_directory=build_directory,
                        context="test noncanonical v7 hash",
                    )

    def test_ninja_log_recompaction_has_explicit_diagnostic(self) -> None:
        build_directory = Path("/tmp/build")
        output = str(
            build_directory / "CMakeFiles/cmake.verify_globs"
        ).encode("utf-8")
        before = (
            b"# ninja log v7\n"
            + b"1\t2\t3\t"
            + output
            + b"\t4\n"
        )
        after = (
            b"# ninja log v7\n"
            + b"5\t6\t7\t"
            + output
            + b"\t8\n"
        )
        for module, error in (
            (causal, causal.CausalFailure),
            (verify, verify.CausalVerificationError),
        ):
            with self.subTest(module=module.__name__), self.assertRaisesRegex(
                error,
                "rewritten or recompacted; append-only proof unavailable",
            ):
                module.validate_ninja_log_append(
                    before,
                    after,
                    build_directory=build_directory,
                    context="test recompacted log",
                )

    def test_idempotence_transcript_requires_exact_live_progress_total(
        self,
    ) -> None:
        build_directory = Path("/tmp/build")
        cmake = "/tools/cmake"
        verify_globs = build_directory / "CMakeFiles/VerifyGlobs.cmake"
        stderr = (
            "ninja explain: "
            f"{build_directory}/CMakeFiles/VerifyGlobs.cmake_force is dirty\n"
        )

        def stdout(total: int) -> str:
            return (
                "ninja: Entering directory `.'\n"
                f"[0/{total}] {cmake} -P {verify_globs}\n"
                "ninja: no work to do.\n"
            )

        causal.validate_idempotence_transcript(
            subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=stdout(2),
                stderr=stderr,
            ),
            build_directory=build_directory,
            cmake_path=cmake,
        )
        verify.validate_idempotence_transcript(
            stdout(2),
            stderr,
            build_directory=build_directory,
            cmake_path=cmake,
            context="test live progress",
        )
        for total in (1, 3):
            with (
                self.subTest(total=total, module="producer"),
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "unexpected command",
                ),
            ):
                causal.validate_idempotence_transcript(
                    subprocess.CompletedProcess(
                        ["/tools/ninja"],
                        0,
                        stdout=stdout(total),
                        stderr=stderr,
                    ),
                    build_directory=build_directory,
                    cmake_path=cmake,
                )
            with (
                self.subTest(total=total, module="verifier"),
                self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "unexpected command",
                ),
            ):
                verify.validate_idempotence_transcript(
                    stdout(total),
                    stderr,
                    build_directory=build_directory,
                    cmake_path=cmake,
                    context="test wrong progress",
                )

    def test_anchored_ninja_rejects_root_and_ancestor_replacement(self) -> None:
        for swap_ancestor in (False, True):
            with self.subTest(swap_ancestor=swap_ancestor):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    parent = root / "parent"
                    build_directory = parent / "build"
                    parent.mkdir()
                    build_directory.mkdir()
                    command = [
                        "/bin/sh",
                        "-c",
                        "/bin/pwd > anchor-marker",
                        "fixture",
                        "-C",
                        ".",
                    ]
                    original_popen = causal.subprocess.Popen

                    with causal.open_build_directory_anchor(
                        build_directory
                    ) as anchor:
                        observations: list[dict] = []

                        def replace_then_launch(
                            *args: object,
                            **kwargs: object,
                        ) -> subprocess.Popen:
                            if swap_ancestor:
                                moved_parent = root / "moved-parent"
                                parent.rename(moved_parent)
                                parent.mkdir()
                                replacement = parent / "build"
                                replacement.mkdir()
                            else:
                                moved_build = root / "moved-build"
                                build_directory.rename(moved_build)
                                replacement = build_directory
                                replacement.mkdir()
                            return original_popen(*args, **kwargs)

                        with (
                            mock.patch.object(
                                causal.subprocess,
                                "Popen",
                                side_effect=replace_then_launch,
                            ),
                            self.assertRaisesRegex(
                                causal.CausalFailure,
                                "pathname no longer names",
                            ),
                        ):
                            causal.run_anchored_ninja(
                                command,
                                anchor=anchor,
                                invocation="base-clean-plan",
                                observations=observations,
                            )

                    original_build = (
                        root / "moved-parent" / "build"
                        if swap_ancestor
                        else root / "moved-build"
                    )
                    self.assertTrue(
                        (original_build / "anchor-marker").is_file()
                    )
                    self.assertFalse(
                        (parent / "build" / "anchor-marker").exists()
                    )

    def test_anchored_ninja_rejects_ninja_lock_before_and_after(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            command = ["/usr/bin/true", "-C", "."]
            with causal.open_build_directory_anchor(
                build_directory
            ) as anchor:
                lock = build_directory / ".ninja_lock"
                lock.write_text("busy", encoding="utf-8")
                with (
                    mock.patch.object(
                        causal,
                        "_execute_anchored_ninja_process",
                    ) as execute,
                    self.assertRaisesRegex(
                        causal.CausalFailure,
                        "lock is present before",
                    ),
                ):
                    causal.run_anchored_ninja(
                        command,
                        anchor=anchor,
                        invocation="base-clean-plan",
                        observations=[],
                    )
                execute.assert_not_called()
                lock.unlink()

                def leave_lock(
                    command: list[str],
                    *,
                    anchor: causal.BuildDirectoryAnchor,
                    timeout: float,
                ) -> subprocess.CompletedProcess:
                    lock.write_text("left behind", encoding="utf-8")
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout="",
                        stderr="",
                    )

                with (
                    mock.patch.object(
                        causal,
                        "_execute_anchored_ninja_process",
                        side_effect=leave_lock,
                    ),
                    self.assertRaisesRegex(
                        causal.CausalFailure,
                        "lock is present after",
                    ),
                ):
                    causal.run_anchored_ninja(
                        command,
                        anchor=anchor,
                        invocation="base-clean-plan",
                        observations=[],
                    )

    def test_anchored_ninja_timeout_cleanup_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            command = ["/usr/bin/false", "-C", "."]
            for reap_succeeds in (True, False):
                with self.subTest(reap_succeeds=reap_succeeds):
                    process = mock.Mock()
                    process.pid = 12345
                    process.stdout = mock.Mock()
                    process.stderr = mock.Mock()
                    process.communicate.side_effect = subprocess.TimeoutExpired(
                        command,
                        0.01,
                    )
                    if reap_succeeds:
                        process.wait.return_value = -signal.SIGKILL
                    else:
                        process.wait.side_effect = subprocess.TimeoutExpired(
                            command,
                            0.01,
                        )
                    expected_message = (
                        "timed out after"
                        if reap_succeeds
                        else "could not be reaped within"
                    )
                    with (
                        causal.open_build_directory_anchor(
                            build_directory
                        ) as anchor,
                        mock.patch.object(
                            causal.subprocess,
                            "Popen",
                            return_value=process,
                        ),
                        mock.patch.object(causal.os, "killpg") as killpg,
                        self.assertRaisesRegex(
                            causal.CausalFailure,
                            expected_message,
                        ),
                    ):
                        causal._execute_anchored_ninja_process(
                            command,
                            anchor=anchor,
                            timeout=0.01,
                        )
                    process.communicate.assert_called_once_with(timeout=0.01)
                    process.wait.assert_called_once_with(
                        timeout=causal.POST_KILL_WAIT_SECONDS
                    )
                    process.stdout.close.assert_called_once_with()
                    process.stderr.close.assert_called_once_with()
                    killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_anchored_ninja_communication_failure_cleanup_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            command = ["/usr/bin/false", "-C", "."]
            for error in (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"), KeyboardInterrupt()):
                with self.subTest(error=type(error).__name__):
                    process = mock.Mock()
                    process.pid = 12345
                    process.stdout = mock.Mock()
                    process.stderr = mock.Mock()
                    process.communicate.side_effect = error
                    process.wait.return_value = -signal.SIGKILL
                    expected_exception = (
                        causal.CausalFailure
                        if isinstance(error, Exception)
                        else KeyboardInterrupt
                    )
                    with (
                        causal.open_build_directory_anchor(
                            build_directory
                        ) as anchor,
                        mock.patch.object(
                            causal.subprocess,
                            "Popen",
                            return_value=process,
                        ),
                        mock.patch.object(causal.os, "killpg") as killpg,
                        self.assertRaises(expected_exception),
                    ):
                        causal._execute_anchored_ninja_process(
                            command,
                            anchor=anchor,
                            timeout=0.01,
                        )
                    process.wait.assert_called_once_with(
                        timeout=causal.POST_KILL_WAIT_SECONDS
                    )
                    process.stdout.close.assert_called_once_with()
                    process.stderr.close.assert_called_once_with()
                    killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_anchored_ninja_cleanup_attempts_every_step_after_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            command = ["/usr/bin/false", "-C", "."]
            for stage in ("kill", "stdout", "wait"):
                with self.subTest(stage=stage):
                    process = mock.Mock()
                    process.pid = 12345
                    process.stdout = mock.Mock()
                    process.stderr = mock.Mock()
                    process.communicate.side_effect = RuntimeError("primary")
                    process.wait.return_value = -signal.SIGKILL
                    if stage == "stdout":
                        process.stdout.close.side_effect = OSError("close")
                    if stage == "wait":
                        process.wait.side_effect = OSError("wait")
                    kill_error = (
                        PermissionError("kill")
                        if stage == "kill"
                        else None
                    )
                    with (
                        causal.open_build_directory_anchor(
                            build_directory
                        ) as anchor,
                        mock.patch.object(
                            causal.subprocess,
                            "Popen",
                            return_value=process,
                        ),
                        mock.patch.object(
                            causal.os,
                            "killpg",
                            side_effect=kill_error,
                        ) as killpg,
                        self.assertRaisesRegex(
                            causal.CausalFailure,
                            "cleanup failed during",
                        ),
                    ):
                        causal._execute_anchored_ninja_process(
                            command,
                            anchor=anchor,
                            timeout=0.01,
                        )
                    killpg.assert_called_once_with(12345, signal.SIGKILL)
                    process.stdout.close.assert_called_once_with()
                    process.stderr.close.assert_called_once_with()
                    process.wait.assert_called_once_with(
                        timeout=causal.POST_KILL_WAIT_SECONDS
                    )

    def test_anchored_ninja_shim_supports_python_39_and_313(self) -> None:
        interpreters = (
            Path("/usr/bin/python3"),
            Path("/opt/homebrew/bin/python3.13"),
        )
        self.assertTrue(all(path.is_file() for path in interpreters))
        with tempfile.TemporaryDirectory() as directory:
            build_directory = Path(directory).resolve()
            marker = build_directory / "shim-marker"
            command = [
                "/bin/sh",
                "-c",
                "/bin/pwd > shim-marker",
                "fixture",
                "-C",
                ".",
            ]
            for interpreter in interpreters:
                with self.subTest(interpreter=str(interpreter)):
                    marker.unlink(missing_ok=True)
                    observations: list[dict] = []
                    with (
                        causal.open_build_directory_anchor(
                            build_directory
                        ) as anchor,
                        mock.patch.object(
                            causal.sys,
                            "executable",
                            str(interpreter),
                        ),
                    ):
                        completed = causal.run_anchored_ninja(
                            command,
                            anchor=anchor,
                            invocation="base-clean-plan",
                            observations=observations,
                        )
                    self.assertEqual(completed.returncode, 0)
                    self.assertEqual(
                        marker.read_text(encoding="utf-8").strip(),
                        str(build_directory),
                    )
                    self.assertEqual(len(observations), 2)

    def test_clean_plan_rejects_protected_path_before_real_clean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target"
            target.write_bytes(b"target")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            transcript = ninja_clean_transcript(
                "target",
                [
                    "target",
                    fixture_exactness_output(target).name,
                    "build.ninja",
                ],
                additional_targets=[
                    "infinity_hnsw_exactness_emitter_production"
                ],
            )
            completed = subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=transcript,
                stderr="",
            )
            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    return_value=completed,
                ) as run_text,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "base clean plan contains non-closure outputs",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=schema_16_closure_fixture(
                        {
                            "compile_units": [],
                            "files": [],
                            "selected_target_graph": (
                                fixture_selected_target_graph(target)
                            ),
                        },
                        benchmark_output=target,
                        benchmark_target="target",
                        benchmark_link_arguments=[
                            "/tools/clang++",
                            "-o",
                            "target",
                        ],
                    ),
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["/tools/clang++", "-o", "target"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            run_text.assert_called_once()

    def test_clean_plan_must_cover_non_module_graph_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target"
            omitted = build_directory / "opaque-generated-artifact"
            target.write_bytes(b"target")
            omitted.write_bytes(b"module")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            completed = subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=ninja_clean_transcript(
                    "target",
                    ["target", fixture_exactness_output(target).name],
                    additional_targets=[
                        "infinity_hnsw_exactness_emitter_production"
                    ],
                ),
                stderr="",
            )
            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    return_value=completed,
                ) as run_text,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "omits non-module graph-derived freshness outputs",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=schema_16_closure_fixture(
                        {
                            "compile_units": [
                                {
                                    "output": str(target),
                                    "module_outputs": [],
                                }
                            ],
                            "files": [
                                {
                                    "path": str(target),
                                    "roles": ["compile-output"],
                                },
                                {
                                    "path": str(omitted),
                                    "roles": ["opaque-output"],
                                },
                            ],
                            "selected_target_graph": (
                                fixture_selected_target_graph(
                                    target,
                                    material_outputs=[target, omitted],
                                )
                            ),
                        },
                        benchmark_output=target,
                        benchmark_target="target",
                        benchmark_link_arguments=[
                            "clang++",
                            "-o",
                            "target",
                        ],
                    ),
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["clang++", "-o", "target"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            run_text.assert_called_once()

    def test_supplemental_module_output_must_stay_in_build_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target.o"
            escaped_module = root / "escaped.pcm"
            target.write_bytes(b"object")
            escaped_module.write_bytes(b"module")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            completed = subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=ninja_clean_transcript(
                    "target.o",
                    ["target.o", fixture_exactness_output(target).name],
                    additional_targets=[
                        "infinity_hnsw_exactness_emitter_production"
                    ],
                ),
                stderr="",
            )
            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    return_value=completed,
                ) as run_text,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "supplemental module outputs escape the build directory",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target.o",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=schema_16_closure_fixture(
                        {
                            "compile_units": [
                                {
                                    "output": str(target),
                                    "module_outputs": [
                                        str(escaped_module)
                                    ],
                                }
                            ],
                            "files": [
                                {
                                    "path": str(target),
                                    "roles": ["compile-output"],
                                },
                                {
                                    "path": str(escaped_module),
                                    "roles": ["module-output"],
                                },
                            ],
                            "selected_target_graph": (
                                fixture_selected_target_graph(target)
                            ),
                        },
                        benchmark_output=target,
                        benchmark_target="target.o",
                        benchmark_link_arguments=[
                            "clang++",
                            "-o",
                            "target.o",
                        ],
                    ),
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["clang++", "-o", "target.o"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            run_text.assert_called_once()
            self.assertTrue(escaped_module.is_file())

    def test_descriptor_relative_quarantine_records_identity_and_uses_dir_fds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            quarantine_directory = root / "quarantine"
            build_directory.mkdir()
            quarantine_directory.mkdir()
            nested = build_directory / "nested"
            nested.mkdir()
            output = nested / "output.o"
            output.write_bytes(b"object")
            derived_before = causal.target_file_identities({output})
            original_open = os.open
            original_rename = causal.renameatx_np_exclusive
            open_calls: list[tuple[object, int, int | None]] = []
            rename_calls: list[tuple[str, str, int, int]] = []

            def tracking_open(
                path: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                open_calls.append((path, flags, dir_fd))
                return original_open(path, flags, mode, dir_fd=dir_fd)

            def tracking_rename(
                source_name: str,
                destination_name: str,
                *,
                source_dir_fd: int,
                destination_dir_fd: int,
            ) -> None:
                rename_calls.append(
                    (
                        source_name,
                        destination_name,
                        source_dir_fd,
                        destination_dir_fd,
                    )
                )
                original_rename(
                    source_name,
                    destination_name,
                    source_dir_fd=source_dir_fd,
                    destination_dir_fd=destination_dir_fd,
                )

            with causal.open_build_directory_anchor(
                build_directory
            ) as anchor, causal.open_build_directory_anchor(
                quarantine_directory
            ) as quarantine_anchor, mock.patch.object(
                causal.os,
                "open",
                side_effect=tracking_open,
            ), mock.patch.object(
                causal,
                "renameatx_np_exclusive",
                side_effect=tracking_rename,
            ):
                removed = causal.descriptor_relative_quarantine_outputs(
                    anchor,
                    quarantine_anchor,
                    derived_before,
                    [],
                    quarantine_path_prefix="quarantine",
                    context="test removal",
                )
            self.assertFalse(output.exists())
            self.assertEqual(len(rename_calls), 1)
            self.assertEqual(rename_calls[0][:2], ("output.o", "00000000.output"))
            self.assertEqual(
                (quarantine_directory / "00000000.output").read_bytes(),
                b"object",
            )
            directory_open = next(call for call in open_calls if call[0] == "nested")
            self.assertTrue(directory_open[1] & os.O_DIRECTORY)
            self.assertTrue(directory_open[1] & os.O_NOFOLLOW)
            terminal_open = next(call for call in open_calls if call[0] == "output.o")
            self.assertTrue(terminal_open[1] & os.O_NOFOLLOW)
            self.assertFalse(terminal_open[1] & os.O_DIRECTORY)
            self.assertEqual(len(removed), 1)
            self.assertEqual(removed[0]["path"], str(output))
            self.assertEqual(removed[0]["relative_path"], "nested/output.o")
            self.assertEqual(
                removed[0]["before"]["sha256"],
                hashlib.sha256(b"object").hexdigest(),
            )
            self.assertEqual(removed[0]["before"]["links"], 1)
            self.assertEqual(removed[0]["after"]["links"], 1)
            self.assertEqual(
                removed[0]["quarantined_path"],
                "quarantine/00000000.output",
            )
            self.assertIs(removed[0]["name_absent_after"], True)
            self.assertEqual(
                causal.OUTPUT_REMOVAL_METHOD,
                "descriptor-relative-quarantine-rename-v1",
            )

    def test_descriptor_relative_quarantine_rejects_unprotected_hardlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            quarantine_directory = root / "quarantine"
            build_directory.mkdir()
            quarantine_directory.mkdir()
            output = build_directory / "output.o"
            survivor = build_directory / "survivor.o"
            output.write_bytes(b"object")
            os.link(output, survivor)
            derived_before = causal.target_file_identities({output})
            with (
                causal.open_build_directory_anchor(build_directory) as anchor,
                causal.open_build_directory_anchor(
                    quarantine_directory
                ) as quarantine_anchor,
                mock.patch.object(
                    causal,
                    "renameatx_np_exclusive",
                ) as rename,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "hard-linked",
                ),
            ):
                causal.descriptor_relative_quarantine_outputs(
                    anchor,
                    quarantine_anchor,
                    derived_before,
                    [],
                    quarantine_path_prefix="quarantine",
                    context="test removal",
                )
            rename.assert_not_called()
            self.assertEqual(output.read_bytes(), b"object")
            self.assertEqual(survivor.read_bytes(), b"object")

    def test_descriptor_relative_quarantine_rejects_descendant_swap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            nested = build_directory / "nested"
            outside = root / "outside"
            quarantine_directory = root / "quarantine"
            build_directory.mkdir()
            nested.mkdir()
            outside.mkdir()
            quarantine_directory.mkdir()
            output = nested / "output.o"
            outside_output = outside / "output.o"
            output.write_bytes(b"trusted")
            outside_output.write_bytes(b"outside")
            derived_before = causal.target_file_identities({output})
            moved = build_directory / "moved-nested"
            original_reader = causal.stable_regular_file_descriptor
            swapped = False

            def swap_after_hash(*args: object, **kwargs: object) -> object:
                nonlocal swapped
                result = original_reader(*args, **kwargs)
                if not swapped:
                    nested.rename(moved)
                    nested.symlink_to(outside, target_is_directory=True)
                    swapped = True
                return result

            with (
                causal.open_build_directory_anchor(build_directory) as anchor,
                causal.open_build_directory_anchor(
                    quarantine_directory
                ) as quarantine_anchor,
                mock.patch.object(
                    causal,
                    "stable_regular_file_descriptor",
                    side_effect=swap_after_hash,
                ),
                mock.patch.object(
                    causal,
                    "renameatx_np_exclusive",
                ) as rename,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "ancestor changed before quarantine rename",
                ),
            ):
                causal.descriptor_relative_quarantine_outputs(
                    anchor,
                    quarantine_anchor,
                    derived_before,
                    [],
                    quarantine_path_prefix="quarantine",
                    context="test removal",
                )
            self.assertTrue(swapped)
            rename.assert_not_called()
            self.assertEqual((moved / "output.o").read_bytes(), b"trusted")
            self.assertEqual(outside_output.read_bytes(), b"outside")

    def test_descriptor_relative_quarantine_preserves_boundary_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            quarantine_directory = root / "quarantine"
            build_directory.mkdir()
            quarantine_directory.mkdir()
            output = build_directory / "output.o"
            moved = build_directory / "trusted-moved.o"
            output.write_bytes(b"trusted")
            derived_before = causal.target_file_identities({output})
            original_rename = causal.renameatx_np_exclusive

            def replace_at_boundary(
                source_name: str,
                destination_name: str,
                *,
                source_dir_fd: int,
                destination_dir_fd: int,
            ) -> None:
                output.rename(moved)
                output.write_bytes(b"replacement")
                original_rename(
                    source_name,
                    destination_name,
                    source_dir_fd=source_dir_fd,
                    destination_dir_fd=destination_dir_fd,
                )

            with (
                causal.open_build_directory_anchor(build_directory) as anchor,
                causal.open_build_directory_anchor(
                    quarantine_directory
                ) as quarantine_anchor,
                mock.patch.object(
                    causal,
                    "renameatx_np_exclusive",
                    side_effect=replace_at_boundary,
                ),
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "quarantined output identity differs",
                ),
            ):
                causal.descriptor_relative_quarantine_outputs(
                    anchor,
                    quarantine_anchor,
                    derived_before,
                    [],
                    quarantine_path_prefix="quarantine",
                    context="test removal",
                )
            self.assertEqual(moved.read_bytes(), b"trusted")
            self.assertFalse(output.exists())
            self.assertEqual(
                (quarantine_directory / "00000000.output").read_bytes(),
                b"replacement",
            )

    def test_exclusive_quarantine_rename_never_replaces_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_directory = root / "source"
            destination_directory = root / "destination"
            source_directory.mkdir()
            destination_directory.mkdir()
            source = source_directory / "output.o"
            destination = destination_directory / "00000000.output"
            source.write_bytes(b"source")
            destination.write_bytes(b"destination")
            with causal.open_build_directory_anchor(
                source_directory
            ) as source_anchor, causal.open_build_directory_anchor(
                destination_directory
            ) as destination_anchor, self.assertRaises(OSError):
                causal.renameatx_np_exclusive(
                    source.name,
                    destination.name,
                    source_dir_fd=source_anchor.descriptor,
                    destination_dir_fd=destination_anchor.descriptor,
                )
            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(destination.read_bytes(), b"destination")

    def test_descriptor_relative_quarantine_rejects_protected_hardlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            quarantine_directory = root / "quarantine"
            build_directory.mkdir()
            quarantine_directory.mkdir()
            protected = build_directory / "build.ninja"
            output = build_directory / "output.o"
            protected.write_bytes(b"shared")
            os.link(protected, output)
            derived_before = causal.target_file_identities({output})
            protected_identities = causal.target_file_identities({protected})
            with (
                causal.open_build_directory_anchor(build_directory) as anchor,
                causal.open_build_directory_anchor(
                    quarantine_directory
                ) as quarantine_anchor,
                mock.patch.object(
                    causal,
                    "renameatx_np_exclusive",
                ) as rename,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "alias.*protected metadata",
                ),
            ):
                causal.descriptor_relative_quarantine_outputs(
                    anchor,
                    quarantine_anchor,
                    derived_before,
                    protected_identities,
                    quarantine_path_prefix="quarantine",
                    context="test removal",
                )
            rename.assert_not_called()
            self.assertEqual(protected.read_bytes(), b"shared")
            self.assertEqual(output.read_bytes(), b"shared")

    def test_descriptor_relative_quarantine_opens_fifo_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            quarantine_directory = root / "quarantine"
            build_directory.mkdir()
            quarantine_directory.mkdir()
            output = build_directory / "output.o"
            output.write_bytes(b"object")
            derived_before = causal.target_file_identities({output})
            output.unlink()
            os.mkfifo(output)
            started = time.monotonic()
            with (
                causal.open_build_directory_anchor(build_directory) as anchor,
                causal.open_build_directory_anchor(
                    quarantine_directory
                ) as quarantine_anchor,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "changed while being opened",
                ),
            ):
                causal.descriptor_relative_quarantine_outputs(
                    anchor,
                    quarantine_anchor,
                    derived_before,
                    [],
                    quarantine_path_prefix="quarantine",
                    context="test removal",
                )
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertTrue(stat.S_ISFIFO(output.lstat().st_mode))

    def test_clean_rebuild_must_restore_every_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target"
            byproduct = build_directory / "opaque-byproduct"
            target.write_bytes(b"target")
            byproduct.write_bytes(b"byproduct")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            transcript = ninja_clean_transcript(
                "target",
                [
                    "target",
                    "opaque-byproduct",
                    fixture_exactness_output(target).name,
                ],
                additional_targets=[
                    "infinity_hnsw_exactness_emitter_production"
                ],
            )
            calls = 0

            def run_text(
                command: list[str],
                *,
                anchor: causal.BuildDirectoryAnchor,
                timeout: float = 60.0,
            ) -> subprocess.CompletedProcess:
                nonlocal calls
                calls += 1
                if calls == 3:
                    target.write_bytes(b"target")
                    fixture_exactness_output(target).write_bytes(
                        b"exactness fixture\n"
                    )
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=(
                            "ninja: Entering directory `.'\n"
                            "[1/2] clang++ input.o -o target\n"
                            "[2/2] /tools/clang++ -o "
                            "infinity_hnsw_exactness_emitter_production\n"
                        ),
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=transcript,
                    stderr="",
                )

            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    side_effect=run_text,
                ),
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "Idempotence input .*opaque-byproduct "
                    "could not be opened safely",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=schema_16_closure_fixture(
                        {
                            "compile_units": [
                                {
                                    "output": str(target),
                                    "module_outputs": [str(byproduct)],
                                }
                            ],
                            "files": [
                                closure_file_record(
                                    target,
                                    ["compile-output"],
                                    output_dir=output_dir,
                                ),
                                closure_file_record(
                                    byproduct,
                                    ["module-output"],
                                    output_dir=output_dir,
                                ),
                            ],
                            "selected_target_graph": (
                                fixture_selected_target_graph(
                                    target,
                                    clean_outputs=[target, byproduct],
                                )
                            ),
                        },
                        benchmark_output=target,
                        benchmark_target="target",
                        benchmark_link_arguments=[
                            "clang++",
                            "input.o",
                            "-o",
                            "target",
                        ],
                    ),
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["clang++", "input.o", "-o", "target"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )

    def test_build_manifest_replacement_after_clean_plan_prevents_unlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target"
            target.write_bytes(b"target")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            dry = subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=ninja_clean_transcript(
                    "target",
                    ["target", fixture_exactness_output(target).name],
                    additional_targets=[
                        "infinity_hnsw_exactness_emitter_production"
                    ],
                ),
                stderr="",
            )
            build_manifest = build_directory / "build.ninja"
            original = build_manifest.read_bytes()
            original_mtime_ns = build_manifest.stat().st_mtime_ns
            calls = 0

            def replace_after_plan(
                command: list[str],
                *,
                anchor: causal.BuildDirectoryAnchor,
                timeout: float = 60.0,
            ) -> subprocess.CompletedProcess:
                nonlocal calls
                calls += 1
                if calls == 2:
                    build_manifest.write_bytes(
                        bytes([original[0] ^ 1]) + original[1:]
                    )
                    os.utime(
                        build_manifest,
                        ns=(original_mtime_ns, original_mtime_ns),
                    )
                return dry

            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    side_effect=replace_after_plan,
                ) as run_text,
                mock.patch.object(
                    causal,
                    "descriptor_relative_quarantine_outputs",
                ) as quarantine_outputs,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "build metadata changed before clean execution",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=schema_16_closure_fixture(
                        {
                            "compile_units": [],
                            "files": [],
                            "selected_target_graph": (
                                fixture_selected_target_graph(target)
                            ),
                        },
                        benchmark_output=target,
                        benchmark_target="target",
                        benchmark_link_arguments=[
                            "/tools/clang++",
                            "-o",
                            "target",
                        ],
                    ),
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["/tools/clang++", "-o", "target"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            self.assertEqual(run_text.call_count, 2)
            quarantine_outputs.assert_not_called()
            self.assertEqual(target.read_bytes(), b"target")

    def test_rebuild_and_noop_commands_each_contain_ninja_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target"
            module_output = build_directory / "target.impl.pcm"
            target.write_bytes(b"target")
            module_output.write_bytes(b"module-before")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            captured_commands: list[list[str]] = []

            def run_text(
                command: list[str],
                *,
                anchor: causal.BuildDirectoryAnchor,
                timeout: float = 60.0,
            ) -> subprocess.CompletedProcess:
                captured_commands.append(command)
                if "-t" in command:
                    tool = command[command.index("-t") + 1]
                    if tool == "deps":
                        return subprocess.CompletedProcess(
                            command,
                            0,
                            stdout=(
                                f"{target}: #deps 1, deps mtime "
                                f"{target.stat().st_mtime_ns} (VALID)\n"
                                "    /tmp/input.cpp\n\n"
                            ),
                            stderr="",
                        )
                    self.assertEqual(tool, "clean")
                    combined = "target.impl.pcm" in command
                    self.assertIn("-n", command)
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=ninja_clean_transcript(
                            "target",
                            (
                                [
                                    "target",
                                    fixture_exactness_output(target).name,
                                    "target.impl.pcm",
                                ]
                                if combined
                                else [
                                    "target",
                                    fixture_exactness_output(target).name,
                                ]
                            ),
                            additional_targets=[
                                "infinity_hnsw_exactness_emitter_production"
                            ],
                            supplemental_targets=(
                                ["target.impl.pcm"] if combined else []
                            ),
                        ),
                        stderr="",
                    )
                if len(captured_commands) == 3:
                    target.write_bytes(b"target")
                    module_output.write_bytes(b"module-after")
                    fixture_exactness_output(target).write_bytes(
                        b"exactness fixture\n"
                    )
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=(
                            "ninja: Entering directory `.'\n"
                            "[1/2] clang++ input.o -o target\n"
                            "[2/2] /tools/clang++ -o "
                            "infinity_hnsw_exactness_emitter_production\n"
                        ),
                        stderr="ninja explain: output target is dirty\n",
                    )
                verify_globs = build_directory / "CMakeFiles/VerifyGlobs.cmake"
                with (build_directory / ".ninja_log").open(
                    "a",
                    encoding="utf-8",
                ) as ninja_log:
                    ninja_log.write(
                        "1\t2\t3\t"
                        f"{build_directory}/CMakeFiles/cmake.verify_globs"
                        "\t4\n"
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "ninja: Entering directory `.'\n"
                        f"[0/2] /tmp/cmake -P {verify_globs}\n"
                        "ninja: no work to do.\n"
                    ),
                    stderr=(
                        "ninja explain: "
                        f"{build_directory}/CMakeFiles/VerifyGlobs.cmake_force "
                        "is dirty\n"
                    ),
                )

            def current_synthetic_archive() -> dict[str, dict]:
                return {
                    str(target): {
                        "output": str(target),
                        "dependency_count": 1,
                        "deps_mtime": target.stat().st_mtime_ns,
                        "status": "STALE",
                        "dependencies": ["/tmp/input.cpp"],
                    }
                }

            with mock.patch.object(
                causal,
                "_execute_anchored_ninja_process",
                side_effect=run_text,
            ), mock.patch.object(
                causal,
                "parse_archived_ninja_dependency_database",
                side_effect=lambda *_args, **_kwargs: current_synthetic_archive(),
            ):
                closure = schema_16_closure_fixture(
                    {
                        "compile_units": [
                            {
                                "output": str(target),
                                "module_outputs": [str(module_output)],
                            }
                        ],
                        "dependency_graph": [
                            {
                                "output": str(target),
                                "dependency_count": 1,
                                "deps_mtime": target.stat().st_mtime_ns,
                                "status": "VALID",
                                "dependencies": ["/tmp/input.cpp"],
                            }
                        ],
                        "files": [
                            closure_file_record(
                                target,
                                ["compile-output"],
                                output_dir=output_dir,
                            ),
                            closure_file_record(
                                module_output,
                                ["module-output"],
                                output_dir=output_dir,
                            ),
                        ],
                        "selected_target_graph": (
                            fixture_selected_target_graph(target)
                        ),
                    },
                    benchmark_output=target,
                    benchmark_target="target",
                    benchmark_link_arguments=[
                        "clang++",
                        "input.o",
                        "-o",
                        "target",
                    ],
                )
                record = causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target",
                    build_tool={
                        "resolved_path": "/tmp/ninja",
                        "sha256": "0" * 64,
                    },
                    cache={"CMAKE_COMMAND": "/tmp/cmake"},
                    closure=closure,
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["clang++", "input.o", "-o", "target"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            self.assertEqual(
                record["clean_rebuild"]["supplemental_targets"],
                ["target.impl.pcm"],
            )
            self.assertEqual(len(captured_commands), 7)
            clean_commands = [
                command
                for command in captured_commands
                if "-t" in command
                and command[command.index("-t") + 1] == "clean"
            ]
            self.assertEqual(len(clean_commands), 2)
            self.assertTrue(all("-n" in command for command in clean_commands))
            self.assertTrue(
                all(
                    command.count("/tmp/ninja") == 1
                    for command in captured_commands
                )
            )
            self.assertTrue(
                all(command[0] == "/tmp/ninja" for command in captured_commands)
            )
            synthetic_archive = current_synthetic_archive()
            verify_arguments = {
                "role": "control",
                "build_directory": build_directory,
                "requested_target": "target",
                "build_tool_path": "/tmp/ninja",
                "cmake_path": "/tmp/cmake",
                "closure": closure,
                "binary_path": str(target),
                "binary_sha256": causal.d0.sha256(target),
                "binary_bytes": target.stat().st_size,
                "cache_capture": build_directory / "CMakeCache.txt",
                "compile_capture": build_directory / "compile_commands.json",
                "link_arguments": ["clang++", "input.o", "-o", "target"],
                "response_files": {},
            }

            def current_evidence_manifest() -> dict[str, str]:
                return {
                    path.relative_to(output_dir).as_posix(): causal.d0.sha256(path)
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }

            def verify_record(value: dict) -> None:
                with mock.patch.object(
                    verify,
                    "parse_archived_ninja_dependency_database",
                    return_value=synthetic_archive,
                ):
                    verify.validate_build_idempotence(
                        output_dir,
                        value,
                        evidence_manifest=current_evidence_manifest(),
                        **verify_arguments,
                    )

            verify_record(record)
            removed_reference = record["clean_rebuild"]["removed_outputs"]
            removed_path = output_dir / removed_reference["captured_path"]
            removed_records = json.loads(
                removed_path.read_text(encoding="utf-8")
            )
            quarantine_directory = output_dir / "control-rebuild-quarantine"
            orphan = quarantine_directory / "99999999.output"
            orphan.write_bytes(b"orphan")
            try:
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "manifest quarantine path set differs",
                ):
                    verify_record(record)
            finally:
                orphan.unlink()
            nested_quarantine = quarantine_directory / "nested"
            nested_quarantine.mkdir()
            try:
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "quarantine directory contains unexpected entries",
                ):
                    verify_record(record)
            finally:
                nested_quarantine.rmdir()
            quarantine_metadata_alias = (
                output_dir / "control-rebuild-removed-outputs.backup.json"
            )
            quarantine_metadata_alias.write_bytes(removed_path.read_bytes())
            try:
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "quarantine evidence namespace differs",
                ):
                    verify_record(record)
            finally:
                quarantine_metadata_alias.unlink()
            tampered_removed = copy.deepcopy(removed_records)
            tampered_removed[0]["after"]["links"] += 1
            try:
                causal.d0.write_json(removed_path, tampered_removed)
                tampered_record = copy.deepcopy(record)
                tampered_record["clean_rebuild"]["removed_outputs"].update(
                    {
                        "sha256": causal.d0.sha256(removed_path),
                        "bytes": removed_path.stat().st_size,
                    }
                )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "quarantine transition is invalid",
                ):
                    verify_record(tampered_record)
            finally:
                causal.d0.write_json(removed_path, removed_records)
            for label, mutate, message in (
                (
                    "execution method",
                    lambda value: value["execution"].update(
                        {"method": "path-cwd"}
                    ),
                    "execution anchor differs",
                ),
                (
                    "pathname command",
                    lambda value: value["command"].__setitem__(
                        2,
                        str(build_directory),
                    ),
                    "command differs",
                ),
                (
                    "lock observation",
                    lambda value: value["ninja_lock_observations"][0].update(
                        {"absent": False}
                    ),
                    "lock observation 0 differs",
                ),
            ):
                with self.subTest(tamper=label):
                    tampered_record = copy.deepcopy(record)
                    mutate(tampered_record)
                    with self.assertRaisesRegex(
                        verify.CausalVerificationError,
                        message,
                    ):
                        verify_record(tampered_record)
            for tampered_targets in (
                [],
                sorted(
                    [
                        "target.impl.pcm",
                        "fabricated.pcm",
                    ]
                ),
            ):
                with self.subTest(
                    supplemental_targets=tampered_targets,
                ):
                    tampered_record = copy.deepcopy(record)
                    tampered_record["clean_rebuild"][
                        "supplemental_targets"
                    ] = tampered_targets
                    with self.assertRaisesRegex(
                        verify.CausalVerificationError,
                        "supplemental targets differ",
                    ), mock.patch.object(
                        verify,
                        "parse_archived_ninja_dependency_database",
                        return_value=synthetic_archive,
                    ):
                        verify.validate_build_idempotence(
                            output_dir,
                            tampered_record,
                            role="control",
                            build_directory=build_directory,
                            requested_target="target",
                            build_tool_path="/tmp/ninja",
                            cmake_path="/tmp/cmake",
                            closure=closure,
                            binary_path=str(target),
                            binary_sha256=causal.d0.sha256(target),
                            binary_bytes=target.stat().st_size,
                            cache_capture=build_directory / "CMakeCache.txt",
                            compile_capture=(
                                build_directory / "compile_commands.json"
                            ),
                            link_arguments=[
                                "clang++",
                                "input.o",
                                "-o",
                                "target",
                            ],
                            response_files={},
                            evidence_manifest=current_evidence_manifest(),
                        )

            quarantined_path = output_dir / removed_records[0]["quarantined_path"]
            quarantined_bytes = quarantined_path.read_bytes()
            quarantined_path.unlink()
            quarantined_path.write_bytes(quarantined_bytes)
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "quarantined identity/content differs",
            ):
                verify_record(record)

    def test_source_substitution_after_closure_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            source = root / "source.cpp"
            target = build_directory / "target.o"
            source.write_bytes(b"original source\n")
            target.write_bytes(b"object")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            closure = schema_16_closure_fixture(
                {
                    "compile_units": [
                        {"output": str(target), "module_outputs": []},
                    ],
                    "files": [
                        closure_file_record(
                            source,
                            ["compile-source"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            target,
                            ["compile-output"],
                            output_dir=output_dir,
                        ),
                    ],
                    "selected_target_graph": fixture_selected_target_graph(
                        target
                    ),
                },
                benchmark_output=target,
                benchmark_target="target.o",
                benchmark_link_arguments=[
                    "clang++",
                    "-o",
                    "target.o",
                ],
            )
            source_mtime_ns = source.stat().st_mtime_ns

            def substitute_source(
                command: list[str],
                *,
                anchor: causal.BuildDirectoryAnchor,
                timeout: float = 60.0,
            ) -> subprocess.CompletedProcess:
                source.write_bytes(b"replaced source\n")
                os.utime(source, ns=(source_mtime_ns, source_mtime_ns))
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=ninja_clean_transcript(
                        "target.o",
                        [
                            "target.o",
                            fixture_exactness_output(target).name,
                        ],
                        additional_targets=[
                            "infinity_hnsw_exactness_emitter_production"
                        ],
                    ),
                    stderr="",
                )

            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    side_effect=substitute_source,
                ) as run_text,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "closure file identity changed before clean.*source.cpp",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target.o",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=closure,
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["clang++", "-o", "target.o"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            run_text.assert_called_once()

    def test_build_manifest_substitution_after_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target"
            target.write_bytes(b"target")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            build_manifest = build_directory / "build.ninja"
            manifest_before = build_manifest.read_bytes()
            manifest_mtime_ns = build_manifest.stat().st_mtime_ns

            def substitute_manifest(
                command: list[str],
                *,
                anchor: causal.BuildDirectoryAnchor,
                timeout: float = 60.0,
            ) -> subprocess.CompletedProcess:
                build_manifest.write_bytes(
                    bytes([manifest_before[0] ^ 1]) + manifest_before[1:]
                )
                os.utime(
                    build_manifest,
                    ns=(manifest_mtime_ns, manifest_mtime_ns),
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=ninja_clean_transcript(
                        "target",
                        [
                            "target",
                            fixture_exactness_output(target).name,
                        ],
                        additional_targets=[
                            "infinity_hnsw_exactness_emitter_production"
                        ],
                    ),
                    stderr="",
                )

            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    side_effect=substitute_manifest,
                ) as run_text,
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "build metadata changed after closure capture",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=schema_16_closure_fixture(
                        {
                            "compile_units": [],
                            "files": [],
                            "selected_target_graph": (
                                fixture_selected_target_graph(target)
                            ),
                        },
                        benchmark_output=target,
                        benchmark_target="target",
                        benchmark_link_arguments=[
                            "clang++",
                            "-o",
                            "target",
                        ],
                    ),
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["clang++", "-o", "target"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            run_text.assert_called_once()

    def test_build_manifest_snapshot_to_baseline_race_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_directory = root / "build"
            output_dir = root / "evidence"
            build_directory.mkdir()
            output_dir.mkdir()
            (build_directory / ".ninja_log").write_text(
                "# ninja log v7\n",
                encoding="utf-8",
            )
            target = build_directory / "target.o"
            target.write_bytes(b"object")
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(
                    build_directory,
                    output_dir,
                    create_missing=True,
                )
            )
            build_manifest = build_directory / "build.ninja"
            manifest_before = build_manifest.read_bytes()
            manifest_mtime_ns = build_manifest.stat().st_mtime_ns
            all_metadata_paths = causal.rebuild_metadata_paths(
                build_directory=build_directory,
                cache_path=build_directory / "CMakeCache.txt",
                compile_path=build_directory / "compile_commands.json",
            )
            original_identities = causal.target_file_identities
            substituted = False

            def identities_with_substitution(paths: set[Path]) -> list[dict]:
                nonlocal substituted
                result = original_identities(paths)
                if paths == all_metadata_paths and not substituted:
                    build_manifest.write_bytes(
                        bytes([manifest_before[0] ^ 1]) + manifest_before[1:]
                    )
                    os.utime(
                        build_manifest,
                        ns=(manifest_mtime_ns, manifest_mtime_ns),
                    )
                    substituted = True
                return result

            completed = subprocess.CompletedProcess(
                ["/tools/ninja"],
                0,
                stdout=ninja_clean_transcript(
                    "target.o",
                    ["target.o", fixture_exactness_output(target).name],
                    additional_targets=[
                        "infinity_hnsw_exactness_emitter_production"
                    ],
                ),
                stderr="",
            )
            with (
                mock.patch.object(
                    causal,
                    "_execute_anchored_ninja_process",
                    return_value=completed,
                ) as run_text,
                mock.patch.object(
                    causal,
                    "target_file_identities",
                    side_effect=identities_with_substitution,
                ),
                self.assertRaisesRegex(
                    causal.CausalFailure,
                    "protected build metadata changed before clean",
                ),
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="target.o",
                    build_tool={"resolved_path": "/tools/ninja"},
                    cache={"CMAKE_COMMAND": "/tools/cmake"},
                    closure=schema_16_closure_fixture(
                        {
                            "compile_units": [
                                {
                                    "output": str(target),
                                    "module_outputs": [],
                                },
                            ],
                            "files": [
                                closure_file_record(
                                    target,
                                    ["compile-output"],
                                    output_dir=output_dir,
                                ),
                            ],
                            "selected_target_graph": (
                                fixture_selected_target_graph(target)
                            ),
                        },
                        benchmark_output=target,
                        benchmark_target="target.o",
                        benchmark_link_arguments=[
                            "clang++",
                            "-o",
                            "target.o",
                        ],
                    ),
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=target,
                    link_arguments=["clang++", "-o", "target.o"],
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            run_text.assert_called_once()

    def test_compile_output_requires_authenticated_closure_file(self) -> None:
        closure = {
            "compile_units": [
                {"output": "/build/missing.o", "module_outputs": []},
            ],
            "files": [],
        }
        with self.assertRaisesRegex(
            causal.CausalFailure,
            "compile output lacks an authenticated closure file",
        ):
            causal.require_closure_identity_binding(
                closure,
                derived_before=[],
                immutable_before=[],
                context="test",
            )
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "compile output lacks an authenticated closure file",
        ):
            verify.require_closure_identity_binding(
                closure,
                derived_before=[],
                immutable_before=[],
                context="test",
            )

    def test_selected_link_action_and_final_noop_tampering_are_rejected(self) -> None:
        build_directory = Path("/build")
        link_arguments = ["clang++", "input.o", "-o", "target"]
        valid_rebuild = "[1/1] clang++ input.o -o target\n"
        self.assertEqual(
            causal.exact_link_action_count(
                valid_rebuild,
                expected_arguments=link_arguments,
                build_directory=build_directory,
                response_references=set(),
                context="test",
            ),
            1,
        )
        self.assertEqual(
            verify.exact_link_action_count(
                valid_rebuild,
                expected_arguments=link_arguments,
                build_directory=build_directory,
                response_files={},
                used_response_files=set(),
                context="test",
            ),
            1,
        )
        for text in (
            "[1/1] clang++ different.o -o target\n",
            valid_rebuild + "[2/2] clang++ input.o -o target\n",
        ):
            with self.subTest(text=text):
                self.assertNotEqual(
                    causal.exact_link_action_count(
                        text,
                        expected_arguments=link_arguments,
                        build_directory=build_directory,
                        response_references=set(),
                        context="test",
                    ),
                    1,
                )
                self.assertNotEqual(
                    verify.exact_link_action_count(
                        text,
                        expected_arguments=link_arguments,
                        build_directory=build_directory,
                        response_files={},
                        used_response_files=set(),
                        context="test",
                    ),
                    1,
                )

        cmake = "/tools/cmake"
        verify_globs = build_directory / "CMakeFiles/VerifyGlobs.cmake"
        valid_stdout = (
            f"ninja: Entering directory `{build_directory}'\n"
            f"[0/1] {cmake} -P {verify_globs}\n"
            "ninja: no work to do.\n"
        )
        valid_stderr = (
            "ninja explain: "
            f"{build_directory}/CMakeFiles/VerifyGlobs.cmake_force is dirty\n"
        )
        for stdout in (
            valid_stdout.replace(
                "ninja: no work to do.\n",
                "[1/1] clang++ input.o -o target\n",
            ),
            valid_stdout.replace(
                "ninja: no work to do.\n",
                "[1/2] clang++ input.o -o target\n"
                "ninja: no work to do.\n",
            ),
        ):
            completed = subprocess.CompletedProcess(
                ["ninja"],
                0,
                stdout=stdout,
                stderr=valid_stderr,
            )
            with self.subTest(stdout=stdout), self.assertRaises(
                causal.CausalFailure
            ):
                causal.validate_idempotence_transcript(
                    completed,
                    build_directory=build_directory,
                    cmake_path=cmake,
                )
            with self.subTest(stdout=stdout), self.assertRaises(
                verify.CausalVerificationError
            ):
                verify.validate_idempotence_transcript(
                    stdout,
                    valid_stderr,
                    build_directory=build_directory,
                    cmake_path=cmake,
                    context="test",
                )

    def test_content_delta_validator_rejects_reordering_and_false_deltas(
        self,
    ) -> None:
        records = [
            {
                "path": "/build/a",
                "before_sha256": "a" * 64,
                "before_bytes": 1,
                "after_sha256": "b" * 64,
                "after_bytes": 1,
            },
            {
                "path": "/build/b",
                "before_sha256": "c" * 64,
                "before_bytes": 2,
                "after_sha256": "d" * 64,
                "after_bytes": 2,
            },
        ]
        self.assertEqual(
            verify.validate_rebuild_content_deltas(
                records,
                context="test deltas",
            ),
            records,
        )
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "not canonically ordered",
        ):
            verify.validate_rebuild_content_deltas(
                list(reversed(records)),
                context="test deltas",
            )
        false_delta = copy.deepcopy(records[:1])
        false_delta[0]["after_sha256"] = false_delta[0]["before_sha256"]
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "does not describe a content change",
        ):
            verify.validate_rebuild_content_deltas(
                false_delta,
                context="test deltas",
            )

    def test_clean_rebuild_exposes_backdated_source_change(self) -> None:
        cmake = shutil.which("cmake")
        ninja = shutil.which("ninja")
        self.assertIsNotNone(cmake)
        self.assertIsNotNone(ninja)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_dir = root / "source"
            build_directory = root / "build"
            output_dir = root / "evidence"
            source_dir.mkdir()
            output_dir.mkdir()
            source = source_dir / "main.c"
            source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            exactness_target = (
                "infinity_hnsw_exactness_emitter_production"
            )
            (source_dir / "CMakeLists.txt").write_text(
                "\n".join(
                    (
                        "cmake_minimum_required(VERSION 3.20)",
                        "project(rebuild_fixture LANGUAGES C)",
                        'file(GLOB fixture_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")',
                        "add_executable(app ${fixture_sources})",
                        f"add_executable({exactness_target} ${{fixture_sources}})",
                        'set(dynamic_output "${CMAKE_CURRENT_BINARY_DIR}/implicit-response.modmap")',
                        'add_custom_command(OUTPUT "${dynamic_output}" COMMAND "${CMAKE_COMMAND}" -E touch "${dynamic_output}" VERBATIM)',
                        'add_custom_target(dynamic_output_target DEPENDS "${dynamic_output}")',
                        "add_dependencies(app dynamic_output_target)",
                        f"add_dependencies({exactness_target} dynamic_output_target)",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    str(cmake),
                    "-S",
                    str(source_dir),
                    "-B",
                    str(build_directory),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    str(cmake),
                    "--build",
                    str(build_directory),
                    "--target",
                    "app",
                    exactness_target,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            compile_entries = json.loads(
                (build_directory / "compile_commands.json").read_text(
                    encoding="utf-8"
                )
            )
            object_path = fixture_compile_output(
                compile_entries,
                target="app",
            )
            exactness_object_path = fixture_compile_output(
                compile_entries,
                target=exactness_target,
            )
            binary = build_directory / "app"
            exactness_binary = build_directory / exactness_target
            dynamic_output = build_directory / "implicit-response.modmap"
            self.assertTrue(dynamic_output.is_file())
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(build_directory, output_dir)
            )
            link_arguments = fixture_link_arguments(
                str(ninja),
                build_directory,
                "app",
                compile_entries,
                binary,
            )
            exactness_link_arguments = fixture_link_arguments(
                str(ninja),
                build_directory,
                exactness_target,
                compile_entries,
                exactness_binary,
            )
            old_timestamp = min(
                object_path.stat().st_mtime_ns,
                binary.stat().st_mtime_ns,
            ) - 1_000_000_000
            source.write_text("int main(void) { return 1; }\n", encoding="utf-8")
            os.utime(source, ns=(old_timestamp, old_timestamp))
            closure = schema_16_closure_fixture(
                {
                    "files": [
                        closure_file_record(
                            source,
                            ["compile-source"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            object_path,
                            ["compile-output"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            exactness_object_path,
                            ["compile-output"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            dynamic_output,
                            ["module-output"],
                            output_dir=output_dir,
                        ),
                    ],
                    "compile_units": [
                        {
                            "output": str(object_path),
                            "module_outputs": [str(dynamic_output)],
                        },
                        {
                            "output": str(exactness_object_path),
                            "module_outputs": [],
                        },
                    ],
                    "dependency_graph": [
                        {
                            "output": str(object_path),
                            "dependency_count": 1,
                            "deps_mtime": object_path.stat().st_mtime_ns,
                            "status": "VALID",
                            "dependencies": [str(source)],
                        },
                        {
                            "output": str(exactness_object_path),
                            "dependency_count": 1,
                            "deps_mtime": (
                                exactness_object_path.stat().st_mtime_ns
                            ),
                            "status": "VALID",
                            "dependencies": [str(source)],
                        },
                    ],
                    "selected_target_graph": fixture_selected_target_graph(
                        binary,
                        material_outputs=[
                            object_path,
                            exactness_object_path,
                            binary,
                            exactness_binary,
                        ],
                        clean_outputs=live_ninja_clean_outputs(
                            str(ninja),
                            build_directory,
                            "app",
                            additional_targets=[exactness_target],
                        ),
                    ),
                },
                benchmark_output=binary,
                benchmark_target="app",
                benchmark_link_arguments=link_arguments,
                exactness_output=exactness_binary,
                exactness_link_arguments=exactness_link_arguments,
            )
            with self.assertRaisesRegex(
                causal.CausalFailure,
                "changed exact compile/link outputs",
            ):
                causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="app",
                    build_tool={"resolved_path": str(ninja)},
                    cache={"CMAKE_COMMAND": str(cmake)},
                    closure=closure,
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=binary,
                    link_arguments=link_arguments,
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )

    def test_clean_rebuild_and_final_noop_verify_independently(self) -> None:
        cmake = shutil.which("cmake")
        ninja = shutil.which("ninja")
        self.assertIsNotNone(cmake)
        self.assertIsNotNone(ninja)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_dir = root / "source"
            build_directory = root / "build"
            output_dir = root / "evidence"
            source_dir.mkdir()
            output_dir.mkdir()
            live_ninja = root / "live-ninja"
            shutil.copy2(Path(ninja).resolve(strict=True), live_ninja)
            live_ninja.chmod(0o700)
            source = source_dir / "main.c"
            source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            custom_input = source_dir / "intermediate-input.txt"
            custom_input.write_text("before\n", encoding="utf-8")
            exactness_target = (
                "infinity_hnsw_exactness_emitter_production"
            )
            (source_dir / "CMakeLists.txt").write_text(
                "\n".join(
                    (
                        "cmake_minimum_required(VERSION 3.20)",
                        "project(rebuild_fixture LANGUAGES C)",
                        'file(GLOB fixture_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")',
                        "add_executable(app ${fixture_sources})",
                        f"add_executable({exactness_target} ${{fixture_sources}})",
                        'set(dynamic_output "${CMAKE_CURRENT_BINARY_DIR}/implicit-response.modmap")',
                        'add_custom_command(OUTPUT "${dynamic_output}" COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/intermediate-input.txt" "${dynamic_output}" DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/intermediate-input.txt" VERBATIM)',
                        'add_custom_target(dynamic_output_target DEPENDS "${dynamic_output}")',
                        "add_dependencies(app dynamic_output_target)",
                        f"add_dependencies({exactness_target} dynamic_output_target)",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    str(cmake),
                    "-S",
                    str(source_dir),
                    "-B",
                    str(build_directory),
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    str(cmake),
                    "--build",
                    str(build_directory),
                    "--target",
                    "app",
                    exactness_target,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            compile_entries = json.loads(
                (build_directory / "compile_commands.json").read_text(
                    encoding="utf-8"
                )
            )
            object_path = fixture_compile_output(
                compile_entries,
                target="app",
            )
            exactness_object_path = fixture_compile_output(
                compile_entries,
                target=exactness_target,
            )
            binary = build_directory / "app"
            exactness_binary = build_directory / exactness_target
            dynamic_output = build_directory / "implicit-response.modmap"
            self.assertTrue(dynamic_output.is_file())
            preclosure_metadata, preclosure_metadata_path = (
                capture_preclosure_fixture(build_directory, output_dir)
            )
            old_timestamp = dynamic_output.stat().st_mtime_ns - 1_000_000_000
            custom_input.write_text("after\n", encoding="utf-8")
            os.utime(custom_input, ns=(old_timestamp, old_timestamp))
            link_arguments = fixture_link_arguments(
                str(ninja),
                build_directory,
                "app",
                compile_entries,
                binary,
            )
            exactness_link_arguments = fixture_link_arguments(
                str(ninja),
                build_directory,
                exactness_target,
                compile_entries,
                exactness_binary,
            )
            closure = schema_16_closure_fixture(
                {
                    "files": [
                        closure_file_record(
                            source,
                            ["compile-source"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            custom_input,
                            ["custom-input"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            object_path,
                            ["compile-output"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            exactness_object_path,
                            ["compile-output"],
                            output_dir=output_dir,
                        ),
                        closure_file_record(
                            dynamic_output,
                            ["module-output"],
                            output_dir=output_dir,
                        ),
                    ],
                    "compile_units": [
                        {
                            "output": str(object_path),
                            "module_outputs": [str(dynamic_output)],
                        },
                        {
                            "output": str(exactness_object_path),
                            "module_outputs": [],
                        },
                    ],
                    "dependency_graph": [
                        {
                            "output": str(object_path),
                            "dependency_count": 1,
                            "deps_mtime": object_path.stat().st_mtime_ns,
                            "status": "VALID",
                            "dependencies": [str(source)],
                        },
                        {
                            "output": str(exactness_object_path),
                            "dependency_count": 1,
                            "deps_mtime": (
                                exactness_object_path.stat().st_mtime_ns
                            ),
                            "status": "VALID",
                            "dependencies": [str(source)],
                        },
                    ],
                    "selected_target_graph": fixture_selected_target_graph(
                        binary,
                        material_outputs=[
                            object_path,
                            exactness_object_path,
                            binary,
                            exactness_binary,
                        ],
                        clean_outputs=live_ninja_clean_outputs(
                            str(ninja),
                            build_directory,
                            "app",
                            additional_targets=[exactness_target],
                        ),
                    ),
                },
                benchmark_output=binary,
                benchmark_target="app",
                benchmark_link_arguments=link_arguments,
                exactness_output=exactness_binary,
                exactness_link_arguments=exactness_link_arguments,
            )
            execute_anchored = causal._execute_anchored_ninja_process

            def execute_with_system_ninja(
                command: list[str],
                *,
                anchor: causal.BuildDirectoryAnchor,
                timeout: float,
            ) -> subprocess.CompletedProcess:
                self.assertEqual(command[0], str(live_ninja))
                return execute_anchored(
                    [str(ninja), *command[1:]],
                    anchor=anchor,
                    timeout=timeout,
                )

            with mock.patch.object(
                causal,
                "_execute_anchored_ninja_process",
                side_effect=execute_with_system_ninja,
            ):
                record = causal.capture_build_idempotence(
                    role="control",
                    output_dir=output_dir,
                    build_directory=build_directory,
                    requested_target="app",
                    build_tool={"resolved_path": str(live_ninja)},
                    cache={"CMAKE_COMMAND": str(cmake)},
                    closure=closure,
                    cache_path=build_directory / "CMakeCache.txt",
                    compile_path=build_directory / "compile_commands.json",
                    expected_output=binary,
                    link_arguments=link_arguments,
                    response_references=set(),
                    preclosure_metadata=preclosure_metadata,
                    preclosure_metadata_path=preclosure_metadata_path,
                )
            live_ninja.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            live_ninja.chmod(0o700)
            verify.validate_build_idempotence(
                output_dir,
                record,
                role="control",
                build_directory=build_directory,
                requested_target="app",
                build_tool_path=str(live_ninja),
                cmake_path=str(cmake),
                closure=closure,
                binary_path=str(binary),
                binary_sha256=causal.d0.sha256(binary),
                binary_bytes=binary.stat().st_size,
                cache_capture=build_directory / "CMakeCache.txt",
                compile_capture=build_directory / "compile_commands.json",
                link_arguments=link_arguments,
                response_files={},
                evidence_manifest=fixture_evidence_manifest(output_dir),
            )
            content_deltas = json.loads(
                (
                    output_dir / "control-rebuild-content-deltas.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                [delta["path"] for delta in content_deltas],
                [str(dynamic_output)],
            )
            derived_before_path = (
                output_dir / "control-rebuild-derived-before.json"
            )
            derived_before = json.loads(
                derived_before_path.read_text(encoding="utf-8")
            )
            self.assertIn(
                str(dynamic_output),
                {item["path"] for item in derived_before},
            )
            derived_after_path = output_dir / "control-rebuild-derived-after.json"
            content_deltas_path = output_dir / "control-rebuild-content-deltas.json"
            derived_after = json.loads(
                derived_after_path.read_text(encoding="utf-8")
            )
            removed_outputs_path = (
                output_dir / "control-rebuild-removed-outputs.json"
            )
            removed_outputs = json.loads(
                removed_outputs_path.read_text(encoding="utf-8")
            )
            verify_arguments = {
                "role": "control",
                "build_directory": build_directory,
                "requested_target": "app",
                "build_tool_path": str(live_ninja),
                "cmake_path": str(cmake),
                "closure": closure,
                "binary_path": str(binary),
                "binary_sha256": causal.d0.sha256(binary),
                "binary_bytes": binary.stat().st_size,
                "cache_capture": build_directory / "CMakeCache.txt",
                "compile_capture": build_directory / "compile_commands.json",
                "link_arguments": link_arguments,
                "response_files": {},
                "evidence_manifest": fixture_evidence_manifest(output_dir),
            }

            def update_reference(reference: dict, path: Path) -> None:
                reference.update(
                    {
                        "sha256": causal.d0.sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )

            report_after_path = (
                output_dir
                / record["settlement"]["dependency_report_after"]["stdout"][
                    "captured_path"
                ]
            )
            report_before_path = (
                output_dir
                / record["settlement"]["dependency_report_before"]["stdout"][
                    "captured_path"
                ]
            )
            report_before_bytes = report_before_path.read_bytes()
            report_after_bytes = report_after_path.read_bytes()
            source_bytes = str(source).encode("utf-8")
            self.assertIn(source_bytes, report_before_bytes)
            self.assertIn(source_bytes, report_after_bytes)
            jointly_tampered_before = report_before_bytes.replace(
                source_bytes,
                source_bytes + b".substituted",
                1,
            )
            tampered_report = report_after_bytes.replace(
                source_bytes,
                source_bytes + b".substituted",
                1,
            )
            try:
                report_before_path.write_bytes(jointly_tampered_before)
                report_after_path.write_bytes(tampered_report)
                test_record = copy.deepcopy(record)
                update_reference(
                    test_record["settlement"]["dependency_report_before"][
                        "stdout"
                    ],
                    report_before_path,
                )
                update_reference(
                    test_record["settlement"]["dependency_report_after"][
                        "stdout"
                    ],
                    report_after_path,
                )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "selected dependency inputs differ from the captured closure",
                ):
                    verify.validate_build_idempotence(
                        output_dir,
                        test_record,
                        **verify_arguments,
                    )
            finally:
                report_before_path.write_bytes(report_before_bytes)
                report_after_path.write_bytes(report_after_bytes)

            try:
                report_after_path.write_bytes(tampered_report)
                test_record = copy.deepcopy(record)
                update_reference(
                    test_record["settlement"]["dependency_report_after"][
                        "stdout"
                    ],
                    report_after_path,
                )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "retained record changed",
                ):
                    verify.validate_build_idempotence(
                        output_dir,
                        test_record,
                        **verify_arguments,
                    )
            finally:
                report_after_path.write_bytes(report_after_bytes)

            deps_after_path = (
                output_dir
                / record["settlement"]["ninja_deps_after"]["captured_path"]
            )
            deps_after_bytes = deps_after_path.read_bytes()
            try:
                deps_after_path.write_bytes(deps_after_bytes + b"tampered")
                test_record = copy.deepcopy(record)
                update_reference(
                    test_record["settlement"]["ninja_deps_after"],
                    deps_after_path,
                )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "database after identity differs",
                ):
                    verify.validate_build_idempotence(
                        output_dir,
                        test_record,
                        **verify_arguments,
                    )
            finally:
                deps_after_path.write_bytes(deps_after_bytes)

            settlement_target_after_path = (
                output_dir
                / record["settlement"]["target_after"]["captured_path"]
            )
            settlement_target_after = json.loads(
                settlement_target_after_path.read_text(encoding="utf-8")
            )
            tampered_settlement_target = copy.deepcopy(
                settlement_target_after
            )
            settlement_binary = [
                item
                for item in tampered_settlement_target
                if item["path"] == str(binary)
            ]
            self.assertEqual(len(settlement_binary), 1)
            settlement_binary[0]["mtime_ns"] += 1
            try:
                causal.d0.write_json(
                    settlement_target_after_path,
                    tampered_settlement_target,
                )
                test_record = copy.deepcopy(record)
                update_reference(
                    test_record["settlement"]["target_after"],
                    settlement_target_after_path,
                )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "target closure changed outside",
                ):
                    verify.validate_build_idempotence(
                        output_dir,
                        test_record,
                        **verify_arguments,
                    )
            finally:
                causal.d0.write_json(
                    settlement_target_after_path,
                    settlement_target_after,
                )

            strict_target_after_path = (
                output_dir / record["target_after"]["captured_path"]
            )
            strict_target_after = json.loads(
                strict_target_after_path.read_text(encoding="utf-8")
            )
            tampered_strict_target = copy.deepcopy(strict_target_after)
            strict_deps = [
                item
                for item in tampered_strict_target
                if item["path"] == str(build_directory / ".ninja_deps")
            ]
            self.assertEqual(len(strict_deps), 1)
            strict_deps[0]["mtime_ns"] += 1
            try:
                causal.d0.write_json(
                    strict_target_after_path,
                    tampered_strict_target,
                )
                test_record = copy.deepcopy(record)
                update_reference(
                    test_record["target_after"],
                    strict_target_after_path,
                )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "target closure changed",
                ):
                    verify.validate_build_idempotence(
                        output_dir,
                        test_record,
                        **verify_arguments,
                    )
            finally:
                causal.d0.write_json(
                    strict_target_after_path,
                    strict_target_after,
                )

            strict_log_before_path = (
                output_dir / record["ninja_log_before"]["captured_path"]
            )
            strict_log_before = strict_log_before_path.read_bytes()
            try:
                strict_log_before_path.write_bytes(
                    strict_log_before
                    + (
                        "1\t2\t3\t"
                        f"{build_directory}/CMakeFiles/cmake.verify_globs"
                        "\t4\n"
                    ).encode("utf-8")
                )
                test_record = copy.deepcopy(record)
                update_reference(
                    test_record["ninja_log_before"],
                    strict_log_before_path,
                )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "authoritative Ninja log changed",
                ):
                    verify.validate_build_idempotence(
                        output_dir,
                        test_record,
                        **verify_arguments,
                    )
            finally:
                strict_log_before_path.write_bytes(strict_log_before)

            def write_artifact(
                test_record: dict,
                name: str,
                path: Path,
                value: list[dict],
            ) -> None:
                causal.d0.write_json(path, value)
                test_record["clean_rebuild"][name].update(
                    {
                        "sha256": causal.d0.sha256(path),
                        "bytes": path.stat().st_size,
                        "files": len(value),
                    }
                )

            for identity_name in (
                "derived_before",
                "protected_after_clean",
            ):
                with self.subTest(resolved_path_tamper=identity_name):
                    reference = record["clean_rebuild"][identity_name]
                    identity_path = output_dir / reference["captured_path"]
                    identities = json.loads(
                        identity_path.read_text(encoding="utf-8")
                    )
                    tampered_identities = copy.deepcopy(identities)
                    tampered_identities[0]["resolved_path"] += ".substituted"
                    try:
                        test_record = copy.deepcopy(record)
                        write_artifact(
                            test_record,
                            identity_name,
                            identity_path,
                            tampered_identities,
                        )
                        with self.assertRaisesRegex(
                            verify.CausalVerificationError,
                            "resolved path differs from path",
                        ):
                            verify.validate_build_idempotence(
                                output_dir,
                                test_record,
                                **verify_arguments,
                            )
                    finally:
                        causal.d0.write_json(identity_path, identities)

            delta_tamper_cases = {
                "missing": [],
                "additional": content_deltas
                + [
                    {
                        "path": "/zz-added-output",
                        "before_sha256": "1" * 64,
                        "before_bytes": 1,
                        "after_sha256": "2" * 64,
                        "after_bytes": 1,
                    }
                ],
                "falsified": [
                    {
                        **content_deltas[0],
                        "after_sha256": "e" * 64,
                    }
                ],
            }
            for label, tampered_deltas in delta_tamper_cases.items():
                with self.subTest(delta_tamper=label):
                    test_record = copy.deepcopy(record)
                    write_artifact(
                        test_record,
                        "derived_after",
                        derived_after_path,
                        derived_after,
                    )
                    write_artifact(
                        test_record,
                        "content_deltas",
                        content_deltas_path,
                        tampered_deltas,
                    )
                    with self.assertRaisesRegex(
                        verify.CausalVerificationError,
                        "content deltas differ",
                    ):
                        verify.validate_build_idempotence(
                            output_dir,
                            test_record,
                            **verify_arguments,
                        )

            for exact_path in (
                object_path,
                exactness_object_path,
                binary,
                exactness_binary,
            ):
                with self.subTest(exact_output=str(exact_path)):
                    test_record = copy.deepcopy(record)
                    tampered_after = copy.deepcopy(derived_after)
                    matching = [
                        item
                        for item in tampered_after
                        if item["path"] == str(exact_path)
                    ]
                    self.assertEqual(len(matching), 1)
                    matching[0]["sha256"] = "f" * 64
                    coherent_deltas = causal.rebuild_content_deltas(
                        derived_before,
                        tampered_after,
                    )
                    write_artifact(
                        test_record,
                        "derived_after",
                        derived_after_path,
                        tampered_after,
                    )
                    write_artifact(
                        test_record,
                        "content_deltas",
                        content_deltas_path,
                        coherent_deltas,
                    )
                    with self.assertRaisesRegex(
                        verify.CausalVerificationError,
                        "exact compile/link outputs changed",
                    ):
                        verify.validate_build_idempotence(
                            output_dir,
                            test_record,
                            **verify_arguments,
                        )

            test_record = copy.deepcopy(record)
            tampered_before = copy.deepcopy(derived_before)
            tampered_after = copy.deepcopy(derived_after)
            for identities in (tampered_before, tampered_after):
                matching = [
                    item
                    for item in identities
                    if item["path"] == str(object_path)
                ]
                self.assertEqual(len(matching), 1)
                matching[0]["sha256"] = "9" * 64
            coherent_deltas = causal.rebuild_content_deltas(
                tampered_before,
                tampered_after,
            )
            tampered_removed_outputs = copy.deepcopy(removed_outputs)
            matching_removed = [
                item
                for item in tampered_removed_outputs
                if item["path"] == str(object_path)
            ]
            self.assertEqual(len(matching_removed), 1)
            matching_removed[0]["before"]["sha256"] = "9" * 64
            matching_removed[0]["after"]["sha256"] = "9" * 64
            write_artifact(
                test_record,
                "derived_before",
                derived_before_path,
                tampered_before,
            )
            write_artifact(
                test_record,
                "derived_after",
                derived_after_path,
                tampered_after,
            )
            write_artifact(
                test_record,
                "content_deltas",
                content_deltas_path,
                coherent_deltas,
            )
            write_artifact(
                test_record,
                "removed_outputs",
                removed_outputs_path,
                tampered_removed_outputs,
            )
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "quarantined identity/content differs",
            ):
                verify.validate_build_idempotence(
                    output_dir,
                    test_record,
                    **verify_arguments,
                )

            write_artifact(
                record,
                "derived_before",
                derived_before_path,
                derived_before,
            )
            write_artifact(
                record,
                "derived_after",
                derived_after_path,
                derived_after,
            )
            write_artifact(
                record,
                "content_deltas",
                content_deltas_path,
                content_deltas,
            )
            write_artifact(
                record,
                "removed_outputs",
                removed_outputs_path,
                removed_outputs,
            )
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "rebuild does not contain exactly one selected link action",
            ):
                rebuild_stdout_path = output_dir / "control-rebuild.stdout"
                rebuild_stdout = rebuild_stdout_path.read_text(encoding="utf-8")
                rebuild_stdout_path.write_text(
                    rebuild_stdout.replace(
                        " -o app ",
                        " -o substituted-app ",
                    ),
                    encoding="utf-8",
                )
                record["clean_rebuild"]["rebuild_stdout"].update(
                    {
                        "sha256": causal.d0.sha256(rebuild_stdout_path),
                        "bytes": rebuild_stdout_path.stat().st_size,
                    }
                )
                verify.validate_build_idempotence(
                    output_dir,
                    record,
                    **verify_arguments,
                )


class Schema16ExactnessTests(unittest.TestCase):
    @staticmethod
    def producer_semantics(gate: dict) -> dict[str, list[dict]]:
        return {
            role: [
                run["producer_verification"]
                for run in gate["runs"]
                if run["role"] == role
            ]
            for role in ("control", "treatment")
        }

    @staticmethod
    def lifecycle_verifier(
        independent_summary: dict,
        producer_semantics: dict[str, list[dict]],
    ):
        def validate(
            _evidence_dir: Path,
            record: dict,
            **_kwargs: object,
        ) -> tuple[dict, dict[str, list[dict]]]:
            for role in ("control", "treatment"):
                for ordinal, run in enumerate(
                    record["roles"][role]["runs"],
                    1,
                ):
                    exactness_verify._validate_lifecycle(
                        run["lifecycle"],
                        context=f"{role} exactness run {ordinal} lifecycle",
                    )
            return independent_summary, producer_semantics

        return validate

    @staticmethod
    def first_lifecycles(gate: dict) -> tuple[dict, dict]:
        return (
            gate["runs"][0]["process_evidence"]["lifecycle"],
            gate["independent_verifier_record"]["roles"]["control"]["runs"][
                0
            ]["lifecycle"],
        )

    def mutate_both_lifecycles(self, gate: dict, mutate) -> None:
        for lifecycle in self.first_lifecycles(gate):
            mutate(lifecycle)

    def assert_gate_rejected(
        self,
        output_dir: Path,
        gate: dict,
        *,
        roles: dict[str, dict],
        independent_summary: dict,
        producer_semantics: dict[str, list[dict]],
        message: str,
    ) -> None:
        with (
            mock.patch.object(
                verify.exactness_v,
                "validate_exactness_gate_with_producer_semantics",
                side_effect=self.lifecycle_verifier(
                    independent_summary,
                    producer_semantics,
                ),
            ),
            self.assertRaisesRegex(
                (
                    verify.CausalVerificationError,
                    verify.v.VerificationError,
                    exactness_verify.ExactnessVerificationError,
                ),
                message,
            ),
        ):
            verify.validate_exactness_process_gate(
                output_dir,
                gate,
                roles=roles,
                timeout_seconds=300.0,
                expected_witnesses=independent_summary[
                    "expected_witnesses"
                ],
            )

    def test_exactness_process_gate_binds_four_descriptor_pinned_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            gate, roles, independent_summary = exactness_gate_fixture(
                output_dir
            )
            producer_semantics = self.producer_semantics(gate)
            self.assertEqual(gate["schema_version"], 3)
            self.assertEqual(
                gate["independent_verifier_record"]["schema_version"],
                3,
            )
            for run in gate["runs"]:
                rich = run["process_evidence"]["lifecycle"]
                minimal = gate["independent_verifier_record"]["roles"][
                    run["role"]
                ]["runs"][run["repetition"]]["lifecycle"]
                self.assertEqual(rich, minimal)
                self.assertIsNot(rich, minimal)
                self.assertIsNot(
                    rich["terminal_observation"],
                    minimal["terminal_observation"],
                )
                self.assertIsNot(
                    rich["pre_reap_quiescence"],
                    minimal["pre_reap_quiescence"],
                )
                self.assertIsNot(rich["reap"], minimal["reap"])
            with (
                mock.patch.object(
                    verify.exactness_v,
                    "validate_exactness_gate_with_producer_semantics",
                    side_effect=self.lifecycle_verifier(
                        independent_summary,
                        producer_semantics,
                    ),
                ) as independent_verify,
            ):
                self.assertEqual(
                    verify.validate_exactness_process_gate(
                        output_dir,
                        gate,
                        roles=roles,
                        timeout_seconds=300.0,
                        expected_witnesses=independent_summary[
                            "expected_witnesses"
                        ],
                    ),
                    independent_summary,
                )
            independent_verify.assert_called_once()
            self.assertEqual(
                independent_verify.call_args.args,
                (output_dir, gate["independent_verifier_record"]),
            )
            self.assertEqual(
                set(independent_verify.call_args.kwargs),
                {
                    "expected_witnesses",
                    "directory_descriptor",
                    "directory_identity",
                },
            )

            mutations = (
                (
                    "schema",
                    lambda value: value.update({"schema_version": 0}),
                    "must be at least 1",
                ),
                (
                    "boolean schema",
                    lambda value: value.update({"schema_version": True}),
                    "must be an integer",
                ),
                (
                    "run order",
                    lambda value: value["execution_order"].reverse(),
                    "execution order differs",
                ),
                (
                    "gate expected witnesses",
                    lambda value: value["expected_witnesses"]["control"].update(
                        {
                            causal.INCREMENTAL_WITNESS_FIELDS[0]: 1,
                        }
                    ),
                    "expected_witnesses differ",
                ),
                (
                    "independent expected witnesses",
                    lambda value: value["independent_verifier_record"][
                        "expected_witnesses"
                    ]["control"].update(
                        {
                            causal.INCREMENTAL_WITNESS_FIELDS[0]: 1,
                        }
                    ),
                    "independent expected_witnesses differ",
                ),
                (
                    "fixed descriptor",
                    lambda value: value["runs"][0]["process_evidence"][
                        "fixed_descriptors"
                    ].update({"dataset": 9}),
                    "spawn contract differs",
                ),
                (
                    "spawn flags",
                    lambda value: value["runs"][0]["process_evidence"][
                        "spawn_flags"
                    ].update({"value": 0}),
                    "spawn contract differs",
                ),
                (
                    "mapped descriptor",
                    lambda value: value["runs"][0]["process_evidence"][
                        "suspended"
                    ].update(
                        {"mapped_macho_matches_held_descriptor": False}
                    ),
                    "suspended topology differs",
                ),
                (
                    "descendant",
                    lambda value: value["runs"][0]["process_evidence"][
                        "suspended"
                    ].update({"no_descendants": False}),
                    "suspended topology differs",
                ),
                (
                    "mapped executable",
                    lambda value: value["runs"][0]["process_evidence"][
                        "suspended"
                    ]["processes"][0].update(
                        {"executable_path": "/substituted"}
                    ),
                    "mapped child identity differs",
                ),
                (
                    "boolean emitter links",
                    lambda value: value["runs"][0]["emitter"][
                        "identity"
                    ].update({"links": True}),
                    "must be an integer",
                ),
                (
                    "boolean suspension timestamp",
                    lambda value: value["runs"][0]["process_evidence"][
                        "suspended"
                    ].update({"observed_at_monotonic_ns": True}),
                    "must be an integer",
                ),
                (
                    "boolean artifact links",
                    lambda value: value["runs"][0]["artifacts"]["stderr"][
                        "identity"
                    ].update({"links": True}),
                    "must be an integer",
                ),
                (
                    "boolean producer stderr bytes",
                    lambda value: value["runs"][0]["producer_verification"][
                        "stderr"
                    ].update({"bytes": False}),
                    "must be an integer",
                ),
                (
                    "boolean independent ordinal",
                    lambda value: value["independent_verifier_record"][
                        "roles"
                    ]["control"]["runs"][0].update({"ordinal": True}),
                    "must be an integer",
                ),
                (
                    "boolean archived independent schema",
                    lambda value: value["independent_verification"].update(
                        {"schema_version": True}
                    ),
                    "archived independent verification differs",
                ),
                (
                    "boolean repetition",
                    lambda value: value["runs"][0].update(
                        {"repetition": False}
                    ),
                    "must be an integer",
                ),
                (
                    "overlapping runs",
                    lambda value: value["runs"][1].update(
                        {
                            "started_at_unix_ns": (
                                value["runs"][0]["ended_at_unix_ns"] - 1
                            ),
                            "started_at_monotonic_ns": (
                                value["runs"][0][
                                    "ended_at_monotonic_ns"
                                ]
                                - 1
                            ),
                        }
                    ),
                    "overlaps the previous exactness run",
                ),
                (
                    "elapsed timeout",
                    lambda value: value["runs"][0].update(
                        {
                            "ended_at_monotonic_ns": (
                                value["runs"][0][
                                    "resumed_at_monotonic_ns"
                                ]
                                + 301_000_000_000
                            )
                        }
                    ),
                    "elapsed time exceeds the configured timeout",
                ),
                (
                    "stdout comparison hash",
                    lambda value: value["comparisons"].update(
                        {"stdout_sha256": "0" * 64}
                    ),
                    "comparison summary differs",
                ),
                (
                    "cross-role witness relation",
                    lambda value: value["comparisons"].update(
                        {"cross_role_witness_byte_equal": True}
                    ),
                    "comparison summary differs",
                ),
                (
                    "control witness comparison hash",
                    lambda value: value["comparisons"].update(
                        {"control_witness_sha256": "0" * 64}
                    ),
                    "comparison summary differs",
                ),
                (
                    "treatment witness comparison hash",
                    lambda value: value["comparisons"].update(
                        {"treatment_witness_sha256": "0" * 64}
                    ),
                    "comparison summary differs",
                ),
                (
                    "producer semantics",
                    lambda value: value["runs"][0][
                        "producer_verification"
                    ]["graph"].update({"forged_semantics": True}),
                    "producer semantics differ from independent reconstruction",
                ),
                (
                    "terminal inode",
                    lambda value: value["runs"][0]["artifacts"]["graph"][
                        "identity"
                    ].update(
                        {
                            "inode": (
                                value["runs"][0]["artifacts"]["graph"][
                                    "identity"
                                ]["inode"]
                                + 1
                            )
                        }
                    ),
                    "terminal filesystem identity differs",
                ),
            )
            for label, mutate, message in mutations:
                with self.subTest(label=label):
                    changed = copy.deepcopy(gate)
                    mutate(changed)
                    self.assert_gate_rejected(
                        output_dir,
                        changed,
                        roles=roles,
                        independent_summary=independent_summary,
                        producer_semantics=producer_semantics,
                        message=message,
                    )

    def test_legacy_exactness_schemas_and_shapes_are_rejected(self) -> None:
        def legacy_rich_process_evidence(value: dict) -> None:
            process_evidence = value["runs"][0]["process_evidence"]
            lifecycle = process_evidence.pop("lifecycle")
            process_evidence["pre_reap_quiescence"] = lifecycle[
                "pre_reap_quiescence"
            ]
            process_evidence["exit_code"] = lifecycle["reap"]["returncode"]

        def legacy_independent_run(value: dict) -> None:
            run = value["independent_verifier_record"]["roles"]["control"][
                "runs"
            ][0]
            lifecycle = run.pop("lifecycle")
            run["returncode"] = lifecycle["reap"]["returncode"]

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            gate, roles, independent_summary = exactness_gate_fixture(
                output_dir
            )
            producer_semantics = self.producer_semantics(gate)
            mutations = (
                (
                    "legacy gate schema",
                    lambda value: value.update({"schema_version": 1}),
                    "status or schema differs",
                ),
                (
                    "legacy process evidence schema",
                    lambda value: value["runs"][0]["process_evidence"].update(
                        {"schema_version": 1}
                    ),
                    "spawn contract differs",
                ),
                (
                    "legacy lifecycle schema",
                    lambda value: self.mutate_both_lifecycles(
                        value,
                        lambda lifecycle: lifecycle.update(
                            {"schema_version": 1}
                        ),
                    ),
                    "lifecycle is not bound to the suspended run",
                ),
                (
                    "legacy independent schema",
                    lambda value: value[
                        "independent_verifier_record"
                    ].update({"schema_version": 1}),
                    "independent verifier schema differs",
                ),
                (
                    "legacy rich process shape",
                    legacy_rich_process_evidence,
                    "process evidence keys differ",
                ),
                (
                    "legacy independent run shape",
                    legacy_independent_run,
                    "independent control run 1 keys differ",
                ),
            )
            for label, mutate, message in mutations:
                with self.subTest(label=label):
                    changed = copy.deepcopy(gate)
                    mutate(changed)
                    self.assert_gate_rejected(
                        output_dir,
                        changed,
                        roles=roles,
                        independent_summary=independent_summary,
                        producer_semantics=producer_semantics,
                        message=message,
                    )

    def test_rich_and_independent_lifecycle_divergence_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            gate, roles, independent_summary = exactness_gate_fixture(
                output_dir
            )
            producer_semantics = self.producer_semantics(gate)
            mutations = (
                (
                    "rich only",
                    lambda value: self.first_lifecycles(value)[0]["reap"].update(
                        {"raw_status": 7 << 8}
                    ),
                ),
                (
                    "independent only",
                    lambda value: self.first_lifecycles(value)[1][
                        "terminal_observation"
                    ].update(
                        {
                            "options": (
                                exactness.WNOHANG | exactness.WEXITED
                            )
                        }
                    ),
                ),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    changed = copy.deepcopy(gate)
                    mutate(changed)
                    self.assert_gate_rejected(
                        output_dir,
                        changed,
                        roles=roles,
                        independent_summary=independent_summary,
                        producer_semantics=producer_semantics,
                        message="independent control lifecycle differs",
                    )

    def test_lifecycle_semantic_regressions_are_rejected(self) -> None:
        def mutate_both(value: dict, mutate) -> None:
            self.mutate_both_lifecycles(value, mutate)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            gate, roles, independent_summary = exactness_gate_fixture(
                output_dir
            )
            producer_semantics = self.producer_semantics(gate)
            mutations = (
                (
                    "wrong leader",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle.update(
                            {"leader_pid": 9_999}
                        ),
                    ),
                    "lifecycle is not bound to the suspended run",
                ),
                (
                    "missing WNOWAIT",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle[
                            "terminal_observation"
                        ].update(
                            {
                                "options": (
                                    exactness.WNOHANG | exactness.WEXITED
                                )
                            }
                        ),
                    ),
                    "terminal observation options differ",
                ),
                (
                    "empty pre-reap members",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle[
                            "pre_reap_quiescence"
                        ].update({"member_pids": []}),
                    ),
                    "does not prove exact leader-only membership",
                ),
                (
                    "extra pre-reap member",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle[
                            "pre_reap_quiescence"
                        ].update(
                            {
                                "member_pids": [
                                    lifecycle["leader_pid"],
                                    lifecycle["leader_pid"] + 1,
                                ]
                            }
                        ),
                    ),
                    "does not prove exact leader-only membership",
                ),
                (
                    "raw wait status mismatch",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle["reap"].update(
                            {"raw_status": 7 << 8}
                        ),
                    ),
                    "raw status does not prove successful exit",
                ),
                (
                    "pre-reap before terminal",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle[
                            "pre_reap_quiescence"
                        ].update(
                            {
                                "observed_at_monotonic_ns": (
                                    lifecycle["terminal_observation"][
                                        "observed_at_monotonic_ns"
                                    ]
                                    - 1
                                )
                            }
                        ),
                    ),
                    "pre-reap quiescence time is below",
                ),
                (
                    "reap before pre-reap",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle["reap"].update(
                            {
                                "completed_at_monotonic_ns": (
                                    lifecycle["pre_reap_quiescence"][
                                        "observed_at_monotonic_ns"
                                    ]
                                    - 1
                                )
                            }
                        ),
                    ),
                    "reap completion time is below",
                ),
                (
                    "boolean leader PID",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle.update(
                            {"leader_pid": True}
                        ),
                    ),
                    "leader PID must be an integer",
                ),
                (
                    "boolean terminal timestamp",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle[
                            "terminal_observation"
                        ].update({"observed_at_monotonic_ns": True}),
                    ),
                    "terminal observation time must be an integer",
                ),
                (
                    "boolean terminal status",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle[
                            "terminal_observation"
                        ].update({"status": False}),
                    ),
                    "terminal observation status must be an integer",
                ),
                (
                    "boolean raw status",
                    lambda value: mutate_both(
                        value,
                        lambda lifecycle: lifecycle["reap"].update(
                            {"raw_status": False}
                        ),
                    ),
                    "reap raw status must be an integer",
                ),
            )
            for label, mutate, message in mutations:
                with self.subTest(label=label):
                    changed = copy.deepcopy(gate)
                    mutate(changed)
                    self.assert_gate_rejected(
                        output_dir,
                        changed,
                        roles=roles,
                        independent_summary=independent_summary,
                        producer_semantics=producer_semantics,
                        message=message,
                    )

    def test_authenticated_exactness_helper_schema_mismatch_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            gate, roles, _ = exactness_gate_fixture(output_dir)
            with (
                mock.patch.object(verify.exactness_v, "SCHEMA_VERSION", 1),
                self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "authenticated verifier schema differs",
                ),
            ):
                verify.validate_exactness_process_gate(
                    output_dir,
                    gate,
                    roles=roles,
                    timeout_seconds=300.0,
                    expected_witnesses=gate["expected_witnesses"],
                )


class CommandLineTests(unittest.TestCase):
    def test_main_rejects_parent_environment_before_creating_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            with (
                mock.patch.object(
                    causal,
                    "parse_args",
                    return_value=type("Args", (), {"output_dir": output})(),
                ),
                mock.patch.object(
                    causal.d0,
                    "reject_unsafe_parent_environment",
                    side_effect=causal.d0.D0Failure("unsafe parent"),
                ),
            ):
                with self.assertRaisesRegex(causal.d0.D0Failure, "unsafe parent"):
                    causal.main()
            self.assertFalse(output.exists())

    def test_schema_16_required_experiments_bind_exact_witness_matrices(
        self,
    ) -> None:
        expected_edges = {
            "incremental-reciprocal": {
                "control": (0, 0),
                "treatment": (0, 1),
            },
            "threshold-only": {
                "control": (0, 0),
                "treatment": (1, 0),
            },
            "threshold-after-incremental": {
                "control": (0, 1),
                "treatment": (1, 1),
            },
            "combined-optimized": {
                "control": (0, 0),
                "treatment": (1, 1),
            },
        }
        for experiment_name, expected_roles in expected_edges.items():
            with self.subTest(experiment=experiment_name):
                witnesses = causal.expected_witnesses_for_experiment(
                    causal.EXPERIMENTS[experiment_name]
                )
                for role, (threshold, incremental) in expected_roles.items():
                    self.assertEqual(
                        [witnesses[role][name] for name in causal.WITNESS_FIELDS],
                        [incremental] * 5 + [threshold] * 5,
                    )

    def test_malformed_experiment_specs_fail_closed(self) -> None:
        for module, error in (
            (causal, causal.CausalFailure),
            (verify, verify.CausalVerificationError),
        ):
            base = module.EXPERIMENTS["combined-optimized"]
            threshold_only = module.EXPERIMENTS["threshold-only"]
            first = base.varying_switches[0]
            cases = {
                "empty varying": replace(base, varying_switches=()),
                "duplicate varying": replace(
                    base,
                    varying_switches=(first, first),
                ),
                "duplicate held": replace(
                    base,
                    held_switches=(
                        *base.held_switches,
                        base.held_switches[0],
                    ),
                ),
                "varying held overlap": replace(
                    base,
                    held_switches=(
                        *base.held_switches,
                        (first.name, "OFF"),
                    ),
                ),
                "unknown switch": replace(
                    base,
                    varying_switches=(
                        module.VaryingSwitchSpec(
                            name="INFINITY_UNKNOWN_SWITCH",
                            control_value="OFF",
                            treatment_value="ON",
                            compile_sources=(module.HNSW_ALG_SOURCE,),
                        ),
                    ),
                ),
                "invalid role values": replace(
                    base,
                    varying_switches=(
                        replace(
                            first,
                            control_value="OFF",
                            treatment_value="OFF",
                        ),
                    ),
                ),
                "wrong source map": replace(
                    base,
                    varying_switches=(
                        replace(
                            first,
                            compile_sources=(module.HNSW_ALG_SOURCE,),
                        ),
                    ),
                ),
                "wrong role matrix": replace(
                    threshold_only,
                    held_switches=tuple(
                        (
                            name,
                            "ON"
                            if name == module.INCREMENTAL_RECIPROCAL_SWITCH
                            else value,
                        )
                        for name, value in threshold_only.held_switches
                    ),
                ),
            }
            for label, malformed in cases.items():
                with self.subTest(module=module.__name__, case=label):
                    with self.assertRaises(error):
                        module.validate_experiment_spec(
                            malformed,
                            context="malformed experiment",
                        )

    def test_malformed_expected_witnesses_fail_closed(self) -> None:
        for module, errors in (
            (causal, (causal.CausalFailure,)),
            (
                verify,
                (
                    verify.CausalVerificationError,
                    verify.v.VerificationError,
                ),
            ),
        ):
            experiment = module.EXPERIMENTS["threshold-after-incremental"]
            valid = module.expected_witnesses_for_experiment(experiment)

            def wrong_value(value: object) -> dict:
                changed = copy.deepcopy(valid)
                changed["control"][
                    module.INCREMENTAL_WITNESS_FIELDS[0]
                ] = value
                return changed

            cases = {
                "missing role": {
                    "control": copy.deepcopy(valid["control"]),
                },
                "extra role": {
                    **copy.deepcopy(valid),
                    "other": copy.deepcopy(valid["control"]),
                },
                "missing field": copy.deepcopy(valid),
                "extra field": copy.deepcopy(valid),
                "boolean": wrong_value(True),
                "string": wrong_value("1"),
                "float": wrong_value(1.0),
                "negative": wrong_value(-1),
                "two": wrong_value(2),
                "swapped roles": {
                    "control": copy.deepcopy(valid["treatment"]),
                    "treatment": copy.deepcopy(valid["control"]),
                },
            }
            cases["missing field"]["control"].pop(module.WITNESS_FIELDS[0])
            cases["extra field"]["control"]["unexpected"] = 0
            for label, malformed in cases.items():
                with self.subTest(module=module.__name__, case=label):
                    with self.assertRaises(errors):
                        module.validate_expected_witnesses(
                            malformed,
                            experiment=experiment,
                            context="malformed witnesses",
                        )

    def test_schema_16_preflight_captures_authenticated_exactness_contract(
        self,
    ) -> None:
        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory).resolve()
            dataset_record_path = output_dir / "dataset.json"
            heldout_record_path = output_dir / d0.HELDOUT_MANIFEST_FILENAME
            schedule_path = output_dir / "schedule.json"
            for path, content in (
                (dataset_record_path, b"{}\n"),
                (heldout_record_path, b"{}\n"),
                (schedule_path, b"[]\n"),
            ):
                path.write_bytes(content)
            dataset = {
                "path": d0.DATASET_FILENAME,
                "sha256": d0.DATASET_SHA256,
                "bytes": d0.VECTORS * d0.DIMENSIONS * d0.FLOAT32_BYTES,
            }
            heldout = {"path": "heldout.bin"}
            experiment = causal.EXPERIMENTS["threshold-only"]
            expected_witnesses = causal.expected_witnesses_for_experiment(
                experiment
            )
            args = type(
                "Args",
                (),
                {
                    "repo": repo,
                    "control_binary": Path("/build/control/benchmark"),
                    "treatment_binary": Path("/build/treatment/benchmark"),
                    "output_dir": output_dir,
                    "experiment_spec": experiment,
                    "idle_minimum_percent": d0.IDLE_MINIMUM_PERCENT,
                    "idle_timeout_seconds": 300.0,
                    "member_timeout_seconds": 300.0,
                    "preflight_only": True,
                },
            )()

            def role_build(role: str) -> dict:
                digest = "a" if role == "control" else "b"
                emitter_digest = "c" if role == "control" else "d"
                return {
                    "role": role,
                    "binary": {
                        "path": str(getattr(args, f"{role}_binary")),
                        "sha256": digest * 64,
                    },
                    "exactness_emitter": {
                        "path": f"/build/{role}/exactness-emitter",
                        "sha256": emitter_digest * 64,
                        "bytes": 4096,
                    },
                    "_causal_validation": {},
                }

            builds = {
                role: role_build(role)
                for role in ("control", "treatment")
            }

            def capture_build(
                role: str,
                _binary: Path,
                **_kwargs: object,
            ) -> tuple[dict, dict[str, str]]:
                build = copy.deepcopy(builds[role])
                return (
                    build,
                    {
                        build["binary"]["path"]: build["binary"]["sha256"],
                        build["exactness_emitter"]["path"]: (
                            build["exactness_emitter"]["sha256"]
                        ),
                    },
                )

            independent_record = {
                "schema_version": exactness_verify.SCHEMA_VERSION,
                "expected_witnesses": expected_witnesses,
                "dataset": dataset,
                "roles": {
                    "control": {"runs": []},
                    "treatment": {"runs": []},
                },
            }
            gate = {
                "schema_version": exactness.EXACTNESS_SCHEMA_VERSION,
                "expected_witnesses": expected_witnesses,
                "independent_verifier_record": independent_record,
            }
            independent_summary = {
                "schema_version": exactness_verify.SCHEMA_VERSION,
                "status": "PASS",
                "expected_witnesses": expected_witnesses,
            }
            python_record = {
                "resolved_path": str(Path(sys.executable).resolve()),
                "sha256": "e" * 64,
            }
            disk = type(
                "Disk",
                (),
                {
                    "total": 1 << 50,
                    "used": 1,
                    "free": 1 << 49,
                },
            )()
            with (
                mock.patch.object(
                    causal.d0,
                    "reject_unsafe_parent_environment",
                    return_value={"status": "pass"},
                ),
                mock.patch.object(causal.platform, "machine", return_value="arm64"),
                mock.patch.object(causal.shutil, "disk_usage", return_value=disk),
                mock.patch.object(
                    causal,
                    "capture_build",
                    side_effect=capture_build,
                ),
                mock.patch.object(
                    causal,
                    "compare_builds",
                    return_value={"experiment": experiment.name},
                ),
                mock.patch.object(
                    causal.d0,
                    "executable_record",
                    return_value=python_record,
                ),
                mock.patch.object(
                    causal.d0,
                    "process_executable_identity",
                    return_value={"path": "/usr/bin/time"},
                ),
                mock.patch.object(
                    causal.exactness,
                    "run_exactness_gate",
                    return_value=copy.deepcopy(gate),
                ) as run_exactness,
                mock.patch.object(
                    causal.exactness_verifier,
                    "validate_exactness_gate",
                    return_value=independent_summary,
                ) as verify_exactness,
                mock.patch.object(
                    causal,
                    "runtime_snapshots",
                    return_value=[],
                ),
                mock.patch.object(
                    causal.d0,
                    "capture_git_repository",
                    return_value={"root": str(repo)},
                ),
            ):
                preflight = causal.preflight(
                    args,
                    output_dir,
                    dataset=dataset,
                    dataset_record_path=dataset_record_path,
                    heldout=heldout,
                    heldout_record_path=heldout_record_path,
                    schedule_path=schedule_path,
                )

            self.assertEqual(preflight["causal_schema_version"], 16)
            self.assertEqual(
                preflight["expected_witnesses"],
                expected_witnesses,
            )
            self.assertEqual(
                preflight["exactness_gate"]["independent_verification"],
                independent_summary,
            )
            expected_modules = [
                (
                    "native_hnsw_d0",
                    "module-native_hnsw_d0.py",
                ),
                (
                    "verify_native_hnsw_d0",
                    "module-verify_native_hnsw_d0.py",
                ),
                (
                    "native_hnsw_exactness",
                    "module-native_hnsw_exactness.py",
                ),
                (
                    "verify_native_hnsw_exactness",
                    "module-verify_native_hnsw_exactness.py",
                ),
            ]
            self.assertEqual(
                [
                    (record["module"], record["captured_path"])
                    for record in preflight["local_python_modules"]
                ],
                expected_modules,
            )
            for record in preflight["local_python_modules"]:
                self.assertEqual(
                    preflight["runtime_artifact_hashes"][
                        record["source_path"]
                    ],
                    record["sha256"],
                )
            self.assertEqual(
                preflight["protocol"]["exactness_schema_version"],
                exactness.EXACTNESS_SCHEMA_VERSION,
            )
            self.assertEqual(
                preflight["protocol"]["exactness_schema_version"],
                3,
            )
            self.assertEqual(
                preflight["protocol"]["exactness_timeout_seconds_per_run"],
                args.member_timeout_seconds,
            )
            self.assertEqual(
                preflight["protocol"]["performance_measurement_scope"],
                causal.PERFORMANCE_MEASUREMENT_SCOPE,
            )
            self.assertEqual(
                preflight["protocol"]["query_benchmark"],
                d0.query_protocol(),
            )
            run_exactness.assert_called_once_with(
                dataset_path=output_dir / d0.DATASET_FILENAME,
                emitters={
                    role: Path(
                        builds[role]["exactness_emitter"]["path"]
                    )
                    for role in ("control", "treatment")
                },
                output_dir=output_dir,
                timeout_seconds=args.member_timeout_seconds,
                expected_emitters={
                    role: builds[role]["exactness_emitter"]
                    for role in ("control", "treatment")
                },
                expected_witnesses=expected_witnesses,
            )
            verify_exactness.assert_called_once_with(
                output_dir,
                independent_record,
                expected_witnesses=expected_witnesses,
            )

            changed = copy.deepcopy(preflight)
            changed["expected_witnesses"]["control"][
                causal.THRESHOLD_WITNESS_FIELDS[0]
            ] = 1
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "expected_witnesses differs from the experiment role matrix",
            ):
                verify.validate_preflight(
                    output_dir,
                    changed,
                    dataset=dataset,
                    heldout=heldout,
                    evidence_manifest=fixture_evidence_manifest(output_dir),
                    schedule_sha256=causal.d0.sha256(schedule_path),
                    expected_mode=verify.PREFLIGHT_ONLY_MODE,
                )

            changed = copy.deepcopy(preflight)
            changed["protocol"]["query_benchmark"]["measured_passes"] += 1
            with (
                mock.patch.object(
                    verify.v,
                    "validate_executable_record",
                ),
                mock.patch.object(
                    verify.v,
                    "validate_runtime_artifact_snapshots",
                    return_value={},
                ),
                self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "Preflight protocol query_benchmark differs",
                ),
            ):
                verify.validate_preflight(
                    output_dir,
                    changed,
                    dataset=dataset,
                    heldout=heldout,
                    evidence_manifest=fixture_evidence_manifest(output_dir),
                    schedule_sha256=causal.d0.sha256(schedule_path),
                    expected_mode=verify.PREFLIGHT_ONLY_MODE,
                )

            changed = copy.deepcopy(preflight)
            changed["protocol"]["exactness_schema_version"] = 1
            with (
                mock.patch.object(
                    verify.v,
                    "validate_executable_record",
                ),
                mock.patch.object(
                    verify.v,
                    "validate_runtime_artifact_snapshots",
                    return_value={},
                ),
                self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "Preflight protocol exactness_schema_version differs",
                ),
            ):
                verify.validate_preflight(
                    output_dir,
                    changed,
                    dataset=dataset,
                    heldout=heldout,
                    evidence_manifest=fixture_evidence_manifest(output_dir),
                    schedule_sha256=causal.d0.sha256(schedule_path),
                    expected_mode=verify.PREFLIGHT_ONLY_MODE,
                )

    def test_schema_16_verifier_build_keys_require_product_evidence(
        self,
    ) -> None:
        build_keys = {
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
        }
        arguments = {
            "experiment": verify.EXPERIMENTS["incremental-reciprocal"],
            "repo": Path("/repo"),
            "evidence_manifest": {},
            "runtime_hashes": {},
            "runtime_paths": {},
        }
        for missing in (
            "product_compile_provenance",
            "exactness_emitter",
            "exactness_dependencies",
        ):
            with self.subTest(missing=missing):
                build = dict.fromkeys(build_keys)
                build.pop(missing)
                with self.assertRaisesRegex(
                    (
                        verify.CausalVerificationError,
                        verify.v.VerificationError,
                    ),
                    rf"keys differ.*{missing}",
                ):
                    verify.validate_build(
                        Path("/evidence"),
                        "control",
                        build,
                        **arguments,
                    )

        complete = dict.fromkeys(build_keys)
        complete.update(
            {
                "role": "control",
                "engine": "infinity",
                "switch_states": arguments[
                    "experiment"
                ].switch_states_for_role("control"),
            }
        )
        switch_mutations = {
            "missing": lambda states: states.pop(
                verify.INCREMENTAL_RECIPROCAL_SWITCH
            ),
            "extra": lambda states: states.update({"UNDECLARED": "ON"}),
            "wrong": lambda states: states.update(
                {verify.INCREMENTAL_RECIPROCAL_SWITCH: "ON"}
            ),
            "boolean": lambda states: states.update(
                {verify.INCREMENTAL_RECIPROCAL_SWITCH: False}
            ),
        }
        for label, mutate in switch_mutations.items():
            with self.subTest(switch_states=label):
                changed = copy.deepcopy(complete)
                mutate(changed["switch_states"])
                with self.assertRaises(
                    (
                        verify.CausalVerificationError,
                        verify.v.VerificationError,
                    )
                ):
                    verify.validate_build(
                        Path("/evidence"),
                        "control",
                        changed,
                        **arguments,
                    )
        with self.assertRaisesRegex(
            (verify.CausalVerificationError, verify.v.VerificationError),
            "directory must be a string",
        ):
            verify.validate_build(
                Path("/evidence"),
                "control",
                complete,
                **arguments,
            )

    def test_runner_file_path_entry_point_works_without_pythonpath(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(
                    repo
                    / "tools"
                    / "apple_silicon"
                    / "native_hnsw_batch4_causal.py"
                ),
                "--help",
            ],
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_captured_verifier_rejects_unauthenticated_direct_execution(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        source_dir = repo / "tools" / "apple_silicon"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verifier = root / "verifier-native_hnsw_batch4_causal.py"
            shutil.copy2(source_dir / "verify_native_hnsw_batch4_causal.py", verifier)
            shutil.copy2(
                source_dir / "verify_native_hnsw_d0.py",
                root / "module-verify_native_hnsw_d0.py",
            )
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-B", str(verifier), "--help"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("authenticated Batch4 bootstrap", result.stderr)

    def test_module_mode_verifier_rejects_unauthenticated_execution(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.apple_silicon.verify_native_hnsw_batch4_causal",
                "--help",
            ],
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("authenticated Batch4 bootstrap", result.stderr)

    def test_verifier_mode_flag_rejects_abbreviation_and_repetition(self) -> None:
        for arguments in (
            ["/tmp/evidence", "--preflight"],
            ["/tmp/evidence", "--preflight-only", "--preflight-only"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    verify.parse_args(arguments)

    def test_runner_rejects_abbreviated_and_repeated_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            treatment = root / "treatment"
            control.write_bytes(b"control")
            treatment.write_bytes(b"treatment")
            base = [
                "--repo",
                str(root),
                "--experiment",
                "threshold-only",
                "--control-binary",
                str(control),
                "--treatment-binary",
                str(treatment),
                "--output-dir",
                str(root / "evidence"),
            ]
            for arguments in (
                ["--exper", *base[2:]],
                [*base, "--experiment", "threshold-only"],
            ):
                with self.subTest(arguments=arguments):
                    with self.assertRaises(SystemExit):
                        causal.parse_args(arguments)

    def test_causal_runner_argv_is_independently_reconstructed(self) -> None:
        runner = "/repo/tools/apple_silicon/native_hnsw_batch4_causal.py"
        argv = [
            runner,
            "--repo",
            "/repo",
            "--experiment",
            "incremental-reciprocal",
            "--control-binary",
            "build/control/bin",
            "--treatment-binary=/repo/build/treatment/bin",
            "--output-dir",
            "/evidence",
            "--preflight-only",
        ]
        self.assertEqual(
            verify.parse_runner_argv(
                argv,
                working_directory="/work",
                runner_source_path=runner,
            ),
            {
                "repo": "/repo",
                "experiment": "incremental-reciprocal",
                "control_binary": "/repo/build/control/bin",
                "treatment_binary": "/repo/build/treatment/bin",
                "output_directory": "/evidence",
                "idle_minimum_percent": d0.IDLE_MINIMUM_PERCENT,
                "idle_timeout_seconds": 300.0,
                "member_timeout_seconds": 300.0,
                "preflight_only": True,
            },
        )
        experiment_index = argv.index("--experiment") + 1
        for experiment_name in causal.EXPERIMENTS:
            with self.subTest(experiment=experiment_name):
                changed = list(argv)
                changed[experiment_index] = experiment_name
                parsed = verify.parse_runner_argv(
                    changed,
                    working_directory="/work",
                    runner_source_path=runner,
                )
                self.assertEqual(parsed["experiment"], experiment_name)
        for mutation in (
            [
                *argv,
                "--idle-timeout-seconds",
                "1",
                "--idle-timeout-seconds",
                "2",
            ],
            [*argv, "--preflight-only"],
            [*argv, "--unknown", "value"],
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(verify.CausalVerificationError):
                    verify.parse_runner_argv(
                        mutation,
                        working_directory="/work",
                        runner_source_path=runner,
                    )

    def test_compile_source_identity_is_lexical_not_live_symlink_resolution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.cpp"
            second = root / "second.cpp"
            link = root / "source.cpp"
            first.write_text("first", encoding="ascii")
            second.write_text("second", encoding="ascii")
            link.symlink_to(first.name)
            entry = {"directory": str(root), "file": link.name}
            self.assertEqual(
                verify.source_path_from_compile_entry(entry, "fixture"),
                str(link),
            )
            link.unlink()
            link.symlink_to(second.name)
            self.assertEqual(
                verify.source_path_from_compile_entry(entry, "fixture"),
                str(link),
            )

    def test_duplicate_absolute_source_path_with_different_hash_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "source.cpp"
            capture.write_bytes(b"source")
            record = {
                "path": "source.cpp",
                "absolute_path": "/repo/source.cpp",
                "captured_path": capture.name,
                "sha256": hashlib.sha256(b"source").hexdigest(),
                "bytes": len(b"source"),
            }
            duplicate = {
                **record,
                "path": "other-label.cpp",
                "sha256": "f" * 64,
            }
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "duplicates source path",
            ):
                verify.validate_source_records(
                    root,
                    [record, duplicate],
                    context="fixture sources",
                )

    def test_compiled_source_bytes_must_match_build_closure(self) -> None:
        source = {
            "path": "src/source.cpp",
            "absolute_path": "/repo/src/source.cpp",
            "captured_path": "provenance/blobs/aa/" + "a" * 64,
            "sha256": "a" * 64,
            "bytes": 10,
        }
        closure = {
            "files": [
                {
                    "path": source["absolute_path"],
                    "roles": ["compile-source"],
                    "captured_path": source["captured_path"],
                    "sha256": source["sha256"],
                    "bytes": source["bytes"],
                }
            ]
        }
        verify.v.validate_compiled_sources_against_closure(
            {source["absolute_path"]: source},
            closure,
            context="fixture",
        )
        source["sha256"] = "b" * 64
        with self.assertRaisesRegex(
            verify.v.VerificationError,
            "content differs from the build closure",
        ):
            verify.v.validate_compiled_sources_against_closure(
                {source["absolute_path"]: source},
                closure,
                context="fixture",
            )

    def test_clean_git_source_bytes_are_bound_to_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_bytes = b"int fixture;\n"
            source_capture = root / "captured-source.cpp"
            source_capture.write_bytes(source_bytes)
            blob = verify.v.git_object_id("blob", source_bytes)
            listing_text = f"100644 blob {blob}\tfile.cpp\n"
            listing_entries = verify.v.parse_git_tree_listing(
                listing_text,
                "fixture tree",
            )
            tree = verify.v.git_tree_id(listing_entries, "fixture tree")
            commit_bytes = (
                f"tree {tree}\n"
                "author Fixture <fixture@example.com> 0 +0000\n"
                "committer Fixture <fixture@example.com> 0 +0000\n"
                "\nfixture\n"
            ).encode("ascii")
            head = verify.v.git_object_id("commit", commit_bytes)
            captures = {
                "source-repository-git-status.txt": b"",
                "source-repository-source-diff.patch": b"",
                "source-repository-commit-object.txt": commit_bytes,
                "source-repository-tree-listing.txt": listing_text.encode("ascii"),
            }
            for name, data in captures.items():
                (root / name).write_bytes(data)

            repository = {
                "root": "/repo",
                "head": head,
                "tree": tree,
                "dirty": False,
                "status_path": "source-repository-git-status.txt",
                "status_sha256": hashlib.sha256(
                    captures["source-repository-git-status.txt"]
                ).hexdigest(),
                "diff_path": "source-repository-source-diff.patch",
                "diff_sha256": hashlib.sha256(
                    captures["source-repository-source-diff.patch"]
                ).hexdigest(),
                "commit_object_path": "source-repository-commit-object.txt",
                "commit_object_sha256": hashlib.sha256(commit_bytes).hexdigest(),
                "tree_listing_path": "source-repository-tree-listing.txt",
                "tree_listing_sha256": hashlib.sha256(
                    listing_text.encode("ascii")
                ).hexdigest(),
            }
            source = {
                "absolute_path": "/repo/file.cpp",
                "captured_path": source_capture.name,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
            }
            verify.validate_git_repository(
                root,
                repository,
                expected_root="/repo",
                build_directory=Path("/repo/build/control"),
                source_records=[source],
            )

            generated_bytes = b"export module compilation_config;\n"
            generated_capture = root / "generated-compilation-config.cppm"
            generated_capture.write_bytes(generated_bytes)
            generated = {
                "absolute_path": (
                    "/repo/build/control/src/generated/"
                    "compilation_config.cppm"
                ),
                "captured_path": generated_capture.name,
                "sha256": hashlib.sha256(generated_bytes).hexdigest(),
                "bytes": len(generated_bytes),
            }
            verify.validate_git_repository(
                root,
                repository,
                expected_root="/repo",
                build_directory=Path("/repo/build/control"),
                source_records=[source, generated],
            )

            substituted = b"int substituted;\n"
            source_capture.write_bytes(substituted)
            source.update(
                {
                    "sha256": hashlib.sha256(substituted).hexdigest(),
                    "bytes": len(substituted),
                }
            )
            with self.assertRaisesRegex(
                verify.CausalVerificationError,
                "pristine source .* differs from HEAD",
            ):
                verify.validate_git_repository(
                    root,
                    repository,
                    expected_root="/repo",
                    build_directory=Path("/repo/build/control"),
                    source_records=[source],
                )

    def test_integer_schema_and_returncode_fields_reject_booleans(self) -> None:
        record = {
            "idempotence_schema_version": True,
            "execution": {
                "method": verify.NINJA_EXECUTION_METHOD,
                "build_directory": "/build",
                "root_device": 1,
                "root_inode": 2,
                "ninja_log_version": verify.NINJA_LOG_VERSION,
            },
            "ninja_lock_observations": [
                {
                    "sequence": index,
                    "invocation": invocation,
                    "phase": phase,
                    "relative_path": ".ninja_lock",
                    "absent": True,
                }
                for index, (invocation, phase) in enumerate(
                    (
                        (invocation, phase)
                        for invocation in verify.NINJA_INVOCATION_ORDER
                        for phase in ("before", "after")
                    )
                )
            ],
            "clean_rebuild": None,
            "settlement": None,
            "command": [],
            "returncode": 0,
            "stdout": None,
            "stderr": None,
            "target_before": None,
            "target_after": None,
            "build_tree_before": None,
            "build_tree_after": None,
            "allowed_mutable_paths": None,
            "ninja_log_before": None,
            "ninja_log_after": None,
        }
        arguments = {
            "role": "control",
            "build_directory": Path("/build"),
            "requested_target": "target",
            "build_tool_path": "/tools/ninja",
            "cmake_path": "/tools/cmake",
            "closure": {
                "files": [],
                "products": [
                    {
                        "name": "benchmark",
                        "requested_target": "target",
                        "expected_output": "/build/target",
                        "link": {"arguments": []},
                    },
                    {
                        "name": "exactness-emitter",
                        "requested_target": (
                            "infinity_hnsw_exactness_emitter_production"
                        ),
                        "expected_output": "/build/exactness-emitter",
                        "link": {"arguments": []},
                    },
                ],
            },
            "binary_path": "/build/target",
            "binary_sha256": "a" * 64,
            "binary_bytes": 1,
            "cache_capture": Path("/capture/cache"),
            "compile_capture": Path("/capture/compile"),
            "link_arguments": [],
            "response_files": {},
            "evidence_manifest": {},
        }
        with self.assertRaisesRegex(
            (verify.CausalVerificationError, verify.v.VerificationError),
            "schema must be an integer",
        ):
            verify.validate_build_idempotence(Path("/evidence"), record, **arguments)

        record["idempotence_schema_version"] = 10
        with self.assertRaisesRegex(
            (verify.CausalVerificationError, verify.v.VerificationError),
            "schema is unsupported",
        ):
            verify.validate_build_idempotence(Path("/evidence"), record, **arguments)

        record["idempotence_schema_version"] = verify.IDEMPOTENCE_SCHEMA_VERSION
        record["command"] = [
            arguments["build_tool_path"],
            "-C",
            ".",
            "-v",
            "-d",
            "explain",
            arguments["requested_target"],
            "infinity_hnsw_exactness_emitter_production",
        ]
        record["returncode"] = False
        with self.assertRaisesRegex(
            (verify.CausalVerificationError, verify.v.VerificationError),
            "return code must be an integer",
        ):
            verify.validate_build_idempotence(Path("/evidence"), record, **arguments)

    def test_cli_keeps_roles_independent_from_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            treatment = root / "treatment"
            control.write_bytes(b"control")
            treatment.write_bytes(b"treatment")
            args = causal.parse_args(
                [
                    "--repo",
                    str(root),
                    "--experiment",
                    "threshold-only",
                    "--control-binary",
                    str(control),
                    "--treatment-binary",
                    str(treatment),
                    "--output-dir",
                    str(root / "evidence"),
                ]
            )
        self.assertEqual(args.control_binary, control.resolve())
        self.assertEqual(args.treatment_binary, treatment.resolve())
        self.assertEqual(args.experiment_spec, causal.EXPERIMENTS["threshold-only"])
        self.assertEqual(
            args.experiment_spec.switch_states_for_role("control")[
                causal.INCREMENTAL_RECIPROCAL_SWITCH
            ],
            "OFF",
        )
        self.assertEqual(args.idle_minimum_percent, 95.0)

    def test_cli_accepts_exact_schema_16_experiment_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            treatment = root / "treatment"
            control.write_bytes(b"control")
            treatment.write_bytes(b"treatment")
            for experiment_name in (
                "incremental-reciprocal",
                "threshold-only",
                "threshold-after-incremental",
                "combined-optimized",
            ):
                with self.subTest(experiment=experiment_name):
                    args = causal.parse_args(
                        [
                            "--repo",
                            str(root),
                            "--experiment",
                            experiment_name,
                            "--control-binary",
                            str(control),
                            "--treatment-binary",
                            str(treatment),
                            "--output-dir",
                            str(root / f"evidence-{experiment_name}"),
                        ]
                    )
                    self.assertEqual(
                        args.experiment_spec,
                        causal.EXPERIMENTS[experiment_name],
                    )

    def test_cli_rejects_same_binary_for_both_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"same")
            with self.assertRaisesRegex(causal.CausalFailure, "paths must differ"):
                causal.parse_args(
                    [
                        "--repo",
                        str(root),
                        "--experiment",
                        "combined-optimized",
                        "--control-binary",
                        str(binary),
                        "--treatment-binary",
                        str(binary),
                        "--output-dir",
                        str(root / "evidence"),
                    ]
                )

    def test_experiment_allowlists_are_independently_identical(self) -> None:
        self.assertEqual(causal.CAUSAL_SCHEMA_VERSION, verify.CAUSAL_SCHEMA_VERSION)
        self.assertEqual(causal.CAUSAL_SCHEMA_VERSION, 16)
        self.assertEqual(verify.ARCHIVED_CAUSAL_SCHEMA_VERSION, 15)
        self.assertEqual(
            verify.ARCHIVED_SCHEMA_15_VERIFIER_POLICY,
            "use-manifest-captured-verifier",
        )
        self.assertEqual(causal.REQUIRED_EXACTNESS_SCHEMA_VERSION, 3)
        self.assertEqual(verify.REQUIRED_EXACTNESS_SCHEMA_VERSION, 3)
        self.assertEqual(
            causal.PERFORMANCE_MEASUREMENT_SCOPE,
            verify.PERFORMANCE_MEASUREMENT_SCOPE,
        )
        self.assertEqual(
            causal.IDEMPOTENCE_SCHEMA_VERSION,
            verify.IDEMPOTENCE_SCHEMA_VERSION,
        )
        self.assertEqual(causal.IDEMPOTENCE_SCHEMA_VERSION, 11)
        self.assertEqual(
            d0.BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
            verify.v.BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
        )
        self.assertEqual(d0.BUILD_INPUT_CLOSURE_SCHEMA_VERSION, 4)
        self.assertEqual(
            causal.CAMPAIGN_BINDING_SCHEMA_VERSION,
            verify.CAMPAIGN_BINDING_SCHEMA_VERSION,
        )
        self.assertEqual(
            causal.BUILD_PRODUCT_TARGETS,
            verify.BUILD_PRODUCT_TARGETS,
        )
        self.assertEqual(
            causal.BUILD_PRODUCT_TARGETS,
            (
                ("benchmark", "infinity_hnsw_d0_production"),
                (
                    "exactness-emitter",
                    "infinity_hnsw_exactness_emitter_production",
                ),
            ),
        )
        self.assertEqual(
            exactness.EXACTNESS_SCHEMA_VERSION,
            exactness_verify.SCHEMA_VERSION,
        )
        self.assertEqual(exactness.EXACTNESS_SCHEMA_VERSION, 3)
        expected_local_modules = (
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
        self.assertEqual(verify.LOCAL_PYTHON_MODULES, expected_local_modules)
        self.assertEqual(
            tuple(
                (
                    module,
                    capture_name,
                    str(source.relative_to(Path(__file__).resolve().parents[3])),
                )
                for module, capture_name, source in causal.LOCAL_PYTHON_MODULES
            ),
            expected_local_modules,
        )
        self.assertEqual(causal.CAUSAL_ROLE_IDS, verify.CAUSAL_ROLE_IDS)
        self.assertEqual(
            set(causal.EXPERIMENTS),
            {
                "incremental-reciprocal",
                "threshold-only",
                "threshold-after-incremental",
                "combined-optimized",
            },
        )
        self.assertEqual(
            {
                name: experiment.record()
                for name, experiment in causal.EXPERIMENTS.items()
            },
            {
                name: experiment.record()
                for name, experiment in verify.EXPERIMENTS.items()
            },
        )
        self.assertEqual(
            causal.EXPECTED_OPTIMIZATION_ROLE_MATRIX,
            verify.EXPECTED_OPTIMIZATION_ROLE_MATRIX,
        )
        self.assertEqual(causal.SHARED_BUILD_SETTINGS, verify.SHARED_BUILD_SETTINGS)
        self.assertEqual(
            causal.REQUIRED_PRODUCTION_BUILD_VALUES,
            verify.REQUIRED_PRODUCTION_BUILD_VALUES,
        )
        self.assertEqual(
            causal.REQUIRED_NONEMPTY_BUILD_SETTINGS,
            verify.REQUIRED_NONEMPTY_BUILD_SETTINGS,
        )
        self.assertEqual(causal.BUILD_TOOL_SPECS, verify.BUILD_TOOL_SPECS)
        self.assertEqual(
            causal.SWITCH_COMPILE_SOURCES,
            verify.SWITCH_COMPILE_SOURCES,
        )
        self.assertEqual(causal.PAIR_PLAN, verify.PAIR_PLAN)
        for name in (
            "MINIMUM_SPEEDUP",
            "MAXIMUM_RECALL_DEFICIT",
            "MAXIMUM_RELATIVE_MAD",
            "MAXIMUM_RELATIVE_MAD_FRACTION",
            "MINIMUM_QUERY_TPS_RATIO",
            "MAXIMUM_QUERY_LATENCY_RATIO",
            "MINIMUM_QUERY_RECALL",
            "MAXIMUM_QUERY_RECALL_GAP",
            "MINIMUM_SPEEDUP_FRACTION",
            "MINIMUM_QUERY_TPS_FRACTION",
            "MAXIMUM_QUERY_LATENCY_FRACTION",
            "MINIMUM_QUERY_RECALL_FRACTION",
            "MAXIMUM_RECALL_DEFICIT_FRACTION",
            "MAXIMUM_QUERY_RECALL_GAP_FRACTION",
        ):
            self.assertEqual(
                getattr(causal, name),
                getattr(verify, name),
                name,
            )
        self.assertEqual(causal.MINIMUM_SPEEDUP_FRACTION, (105, 100))
        self.assertEqual(causal.MAXIMUM_RELATIVE_MAD_FRACTION, (1, 10))
        self.assertEqual(causal.MINIMUM_QUERY_TPS_FRACTION, (95, 100))
        self.assertEqual(causal.MAXIMUM_QUERY_LATENCY_FRACTION, (105, 100))
        self.assertEqual(causal.MINIMUM_QUERY_RECALL_FRACTION, (99, 100))
        self.assertEqual(causal.MAXIMUM_RECALL_DEFICIT_FRACTION, (5, 1000))
        self.assertEqual(
            causal.MAXIMUM_QUERY_RECALL_GAP_FRACTION,
            (5, 1000),
        )
        self.assertIn("ENABLE_JEMALLOC", causal.SHARED_BUILD_SETTINGS)
        self.assertIn("VCPKG_MANIFEST_INSTALL", causal.SHARED_BUILD_SETTINGS)
        self.assertGreaterEqual(d0.D0_WORST_CASE_NEW_BYTES, 8 * 1024**3)

    def test_pre_schema_16_evidence_is_rejected_explicitly(self) -> None:
        preflight = {
            key: None
            for key in (
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
            )
        }
        for schema_version in (11, 12, 13, 14):
            with self.subTest(preflight_schema=schema_version):
                preflight["causal_schema_version"] = schema_version
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "Unsupported causal schema version",
                ):
                    verify.validate_preflight(
                        Path("/evidence"),
                        preflight,
                        dataset={},
                        heldout={},
                        evidence_manifest={},
                        schedule_sha256="a" * 64,
                        expected_mode=verify.PREFLIGHT_ONLY_MODE,
                    )
        preflight["causal_schema_version"] = 15
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "manifest-captured verifier",
        ):
            verify.validate_preflight(
                Path("/evidence"),
                preflight,
                dataset={},
                heldout={},
                evidence_manifest={},
                schedule_sha256="a" * 64,
                expected_mode=verify.PREFLIGHT_ONLY_MODE,
            )
        result = {
            key: None
            for key in (
                "causal_schema_version",
                "experiment",
                "scope",
                "execution_mode",
                "status",
                "completed_at",
                "benchmark_members_executed",
                "campaign_complete",
                "artifact_sha256",
            )
        }
        for schema_version in (11, 12, 13, 14):
            with self.subTest(result_schema=schema_version):
                result["causal_schema_version"] = schema_version
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "Preflight result schema version differs",
                ):
                    verify.validate_preflight_result(
                        result,
                        experiment=verify.EXPERIMENTS[
                            "incremental-reciprocal"
                        ],
                        dataset_sha256="a" * 64,
                        heldout_sha256="b" * 64,
                        preflight_sha256="c" * 64,
                        schedule_sha256="d" * 64,
                    )
        result["causal_schema_version"] = 15
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "manifest-captured verifier",
        ):
            verify.validate_preflight_result(
                result,
                experiment=verify.EXPERIMENTS["incremental-reciprocal"],
                dataset_sha256="a" * 64,
                heldout_sha256="b" * 64,
                preflight_sha256="c" * 64,
                schedule_sha256="d" * 64,
            )

    def test_campaign_binding_freezes_role_binary_dataset_and_schedule(self) -> None:
        roles = {
            "control": {"binary": {"sha256": "b" * 64}},
            "treatment": {"binary": {"sha256": "c" * 64}},
        }
        with mock.patch.object(causal.secrets, "token_hex", return_value="d" * 64):
            campaign = causal.create_campaign_binding(
                roles=roles,
                dataset={"sha256": "a" * 64},
                schedule_sha256="e" * 64,
            )
        self.assertEqual(
            campaign,
            {
                "schema_version": causal.CAMPAIGN_BINDING_SCHEMA_VERSION,
                "campaign_nonce": "d" * 64,
                "dataset_sha256": "a" * 64,
                "schedule_sha256": "e" * 64,
                "roles": {
                    "control": {
                        "role_id": 1,
                        "engine": "infinity",
                        "binary_sha256": "b" * 64,
                    },
                    "treatment": {
                        "role_id": 2,
                        "engine": "infinity",
                        "binary_sha256": "c" * 64,
                    },
                },
            },
        )
        binding = causal.member_campaign_binding(
            {
                "sequence": 17,
                "role": "treatment",
            },
            campaign,
        )
        self.assertEqual(
            binding,
            {
                "campaign_nonce": "d" * 64,
                "schedule_sequence": 17,
                "role_id": 2,
                "binary_sha256": "c" * 64,
                "dataset_sha256": "a" * 64,
            },
        )

    def test_campaign_binding_rejects_role_replay(self) -> None:
        campaign = {
            "schema_version": causal.CAMPAIGN_BINDING_SCHEMA_VERSION,
            "campaign_nonce": "d" * 64,
            "dataset_sha256": "a" * 64,
            "schedule_sha256": "e" * 64,
            "roles": {
                "control": {
                    "role_id": 2,
                    "engine": "infinity",
                    "binary_sha256": "b" * 64,
                }
            },
        }
        with self.assertRaisesRegex(causal.CausalFailure, "control differs"):
            causal.member_campaign_binding(
                {"sequence": 0, "role": "control"},
                campaign,
            )

    def test_independent_verifier_binds_campaign_to_preflight(self) -> None:
        roles = {
            "control": {"binary": {"sha256": "b" * 64}},
            "treatment": {"binary": {"sha256": "c" * 64}},
        }
        campaign = {
            "schema_version": verify.CAMPAIGN_BINDING_SCHEMA_VERSION,
            "campaign_nonce": "d" * 64,
            "dataset_sha256": "a" * 64,
            "schedule_sha256": "e" * 64,
            "roles": {
                "control": {
                    "role_id": 1,
                    "engine": "infinity",
                    "binary_sha256": "b" * 64,
                },
                "treatment": {
                    "role_id": 2,
                    "engine": "infinity",
                    "binary_sha256": "c" * 64,
                },
            },
        }
        self.assertEqual(
            verify.validate_campaign_binding(
                campaign,
                dataset_sha256="a" * 64,
                schedule_sha256="e" * 64,
                roles=roles,
            ),
            campaign,
        )
        mutations = {
            "schedule": lambda value: value.update(
                {"schedule_sha256": "f" * 64}
            ),
            "role": lambda value: value["roles"]["control"].update(
                {"role_id": 2}
            ),
            "binary": lambda value: value["roles"]["treatment"].update(
                {"binary_sha256": "f" * 64}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                tampered = copy.deepcopy(campaign)
                mutate(tampered)
                with self.assertRaises(verify.CausalVerificationError):
                    verify.validate_campaign_binding(
                        tampered,
                        dataset_sha256="a" * 64,
                        schedule_sha256="e" * 64,
                        roles=roles,
                    )

    def test_run_schedule_identity_rejects_json_booleans_for_integers(self) -> None:
        schedule_keys = (
            "sequence",
            "pair",
            "phase",
            "pair_order",
            "role",
            "engine",
            "participants",
        )
        member = copy.deepcopy(causal.schedule()[1])
        for name in ("sequence", "participants"):
            with self.subTest(name=name):
                record = copy.deepcopy(member)
                record[name] = True
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    "must be a JSON integer",
                ):
                    verify.validate_run_schedule_identity(
                        record,
                        member,
                        schedule_keys=schedule_keys,
                        context="fixture",
                    )

    def test_incremental_experiment_has_frozen_role_and_held_switches(self) -> None:
        experiment = causal.EXPERIMENTS["incremental-reciprocal"]
        varying = experiment.varying_switches[0]
        self.assertEqual(varying.control_value, "OFF")
        self.assertEqual(varying.treatment_value, "ON")
        self.assertEqual(
            varying.compile_sources,
            (causal.HNSW_ALG_SOURCE, causal.HNSW_COMMON_SOURCE),
        )
        self.assertEqual(
            dict(experiment.held_switches),
            {
                causal.TRAVERSAL_SWITCH: "OFF",
                causal.RECIPROCAL_PRUNING_SWITCH: "ON",
                causal.THRESHOLD_BATCH4_TRAVERSAL_SWITCH: "OFF",
                causal.INCREMENTAL_RECIPROCAL_SHADOW_SWITCH: "OFF",
                causal.INCREMENTAL_RECIPROCAL_DIAGNOSTICS_SWITCH: "OFF",
                causal.LVQ_CAPTURE_SWITCH: "OFF",
                causal.COMPACT_VERTEX_LOCKS_SWITCH: "OFF",
                causal.INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE_SWITCH: "ON",
                causal.THRESHOLD_BATCH4_EXECUTION_EVIDENCE_SWITCH: "ON",
            },
        )
        for role, expected in (("control", "OFF"), ("treatment", "ON")):
            cache = {
                "ENABLE_JEMALLOC": "OFF",
                "VCPKG_MANIFEST_INSTALL": "OFF",
                varying.name: expected,
                **dict(experiment.held_switches),
            }
            causal.validate_switch_states(
                cache,
                role=role,
                experiment=experiment,
            )

    def test_threshold_experiment_requires_frozen_held_states(self) -> None:
        experiment = causal.EXPERIMENTS["threshold-only"]
        cache = {
            **experiment.switch_states_for_role("control"),
            "ENABLE_JEMALLOC": "OFF",
            "VCPKG_MANIFEST_INSTALL": "OFF",
        }
        causal.validate_switch_states(
            cache,
            role="control",
            experiment=experiment,
        )
        cache[causal.TRAVERSAL_SWITCH] = "ON"
        with self.assertRaisesRegex(causal.CausalFailure, "switch"):
            causal.validate_switch_states(
                cache,
                role="control",
                experiment=experiment,
            )

    def test_causal_experiment_requires_system_malloc(self) -> None:
        experiment = causal.EXPERIMENTS["threshold-only"]
        cache = {
            **experiment.switch_states_for_role("control"),
            "ENABLE_JEMALLOC": "ON",
            "VCPKG_MANIFEST_INSTALL": "OFF",
        }
        with self.assertRaisesRegex(causal.CausalFailure, "ENABLE_JEMALLOC"):
            causal.validate_switch_states(
                cache,
                role="control",
                experiment=experiment,
            )

    def test_causal_experiment_disables_manifest_install(self) -> None:
        experiment = causal.EXPERIMENTS["threshold-only"]
        cache = {
            **experiment.switch_states_for_role("control"),
            "ENABLE_JEMALLOC": "OFF",
            "VCPKG_MANIFEST_INSTALL": "ON",
        }
        with self.assertRaisesRegex(causal.CausalFailure, "VCPKG_MANIFEST_INSTALL"):
            causal.validate_switch_states(
                cache,
                role="control",
                experiment=experiment,
            )

    def test_production_build_settings_fail_closed(self) -> None:
        cache = {
            **causal.REQUIRED_PRODUCTION_BUILD_VALUES,
            **{
                name: f"/fixture/{name.lower()}"
                for name in causal.REQUIRED_NONEMPTY_BUILD_SETTINGS
            },
        }
        scanner = "/fixture/clang-scan-deps"
        cache.update(
            {
                name: scanner
                for name in d0.CMAKE_SCAN_DEPS_CACHE_KEYS
            }
        )
        causal.validate_production_build_settings(cache, role="control")
        verify.validate_production_build_settings(cache, context="control")

        for name in causal.REQUIRED_NONEMPTY_BUILD_SETTINGS:
            with self.subTest(missing=name):
                invalid = copy.deepcopy(cache)
                invalid.pop(name)
                with self.assertRaisesRegex(causal.CausalFailure, name):
                    causal.validate_production_build_settings(
                        invalid,
                        role="control",
                    )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    name,
                ):
                    verify.validate_production_build_settings(
                        invalid,
                        context="control",
                    )

        for name, expected in causal.REQUIRED_PRODUCTION_BUILD_VALUES.items():
            with self.subTest(wrong=name):
                invalid = copy.deepcopy(cache)
                invalid[name] = f"not-{expected}"
                with self.assertRaisesRegex(causal.CausalFailure, name):
                    causal.validate_production_build_settings(
                        invalid,
                        role="treatment",
                    )
                with self.assertRaisesRegex(
                    verify.CausalVerificationError,
                    name,
                ):
                    verify.validate_production_build_settings(
                        invalid,
                        context="treatment",
                    )

        mismatched = copy.deepcopy(cache)
        mismatched["CMAKE_ASM_COMPILER_CLANG_SCAN_DEPS"] = (
            "/fixture/other-clang-scan-deps"
        )
        with self.assertRaisesRegex(
            causal.CausalFailure,
            "one configured dependency scanner",
        ):
            causal.validate_production_build_settings(
                mismatched,
                role="control",
            )
        with self.assertRaisesRegex(
            verify.CausalVerificationError,
            "one configured dependency scanner",
        ):
            verify.validate_production_build_settings(
                mismatched,
                context="control",
            )


if __name__ == "__main__":
    unittest.main()
