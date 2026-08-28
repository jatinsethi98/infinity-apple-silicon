#!/usr/bin/env python3
"""Run and prove a complete native Infinity HNSW restart cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import tomllib
import traceback
import uuid
from pathlib import Path
from typing import Any


READY_MARKER = "Infinity is started in standalone mode."
SHUTDOWN_MARKER = "Shutdown infinity server successfully"
TRACE_PROOF = "KnnScan: brute force task: 0, index task: 1"
INDEX_NAME = "idx_embedding_hnsw"
INDEX_PATH_RE = re.compile(
    r"^db_\d+/tbl_\d+/idx_\d+/seg_\d+/chunk_\d+\.idx$"
)
PERSIST_INDEX_RE = re.compile(
    r"Persist local path (?P<path>\S+\.idx) to "
    r"(?:composed|dedicated) ObjAddr "
    r"\((?P<key>[^,\s]+), (?P<offset>\d+), (?P<size>\d+)\)"
)
CHECKPOINT_INDEX_RE = re.compile(
    r"PM key: pm\|object\|(?P<path>\S+\.idx), "
    r"value: (?P<value>\{[^\n]+\})"
)
QUERY_OBJECT_RE = re.compile(
    r"GetObjCache (?:(?P<current>current) )?object "
    r"(?P<key>[^,\s]+)"
    r"(?:, file_path: (?P<path>\S+), ref count \d+| ref count \d+)"
)
VERIFY_INDEX_MUTATION_PATTERNS = {
    "create_index_transaction": re.compile(
        r"(?:NewTxn::CreateIndex|Command: CREATE INDEX)"
    ),
    "index_buffer_save": re.compile(
        r"BufferObj::Save (?:begin, )?.*?/idx_\d+/.+\.idx"
    ),
    "index_object_persist": re.compile(
        r"Persist local path \S+/idx_\d+/\S+\.idx"
    ),
    "flush_rpc": re.compile(r"THRIFT: Flush Type:"),
    "write_transaction": re.compile(r"Committing WRITE txn"),
    "nonempty_hnsw_recovery": re.compile(
        r"RecoverMemIndex .* append_ranges (?!0: \[\])"
    ),
}


class CycleFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CycleFailure(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def hash_files(root: Path) -> dict[str, dict[str, Any]]:
    require(root.is_dir(), f"Expected persistence directory {root}")
    return {
        str(path.relative_to(root)): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def hash_source_tree(root: Path, suffix: str) -> dict[str, Any]:
    files = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob(f"*{suffix}"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    digest = hashlib.sha256()
    for relative_path, file_hash in files.items():
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return {
        "root": str(root),
        "file_count": len(files),
        "aggregate_sha256": digest.hexdigest(),
        "files": files,
    }


def configured_runtime_paths(config: dict[str, Any]) -> dict[str, Path]:
    locations = {
        "log_dir": config["log"]["log_dir"],
        "persistence_dir": config["storage"]["persistence_dir"],
        "data_dir": config["storage"]["data_dir"],
        "catalog_dir": config["storage"]["catalog_dir"],
        "snapshot_dir": config["storage"]["snapshot_dir"],
        "temp_dir": config["buffer"]["temp_dir"],
        "wal_dir": config["wal"]["wal_dir"],
    }
    return {
        name: Path(location).expanduser().resolve()
        for name, location in locations.items()
    }


def configured_endpoints(config: dict[str, Any]) -> list[dict[str, Any]]:
    network = config["network"]
    server_host = str(network["server_address"])
    endpoints = [
        {
            "name": name,
            "host": server_host,
            "port": int(network[name]),
        }
        for name in ("client_port", "http_port", "postgres_port")
    ]
    endpoints.append(
        {
            "name": "peer_port",
            "host": str(network["peer_ip"]),
            "port": int(network["peer_port"]),
        }
    )
    return sorted(endpoints, key=lambda item: (item["host"], item["port"]))


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def endpoint_states(
    endpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {**endpoint, "open": port_is_open(endpoint["host"], endpoint["port"])}
        for endpoint in endpoints
    ]


def wait_for_server(
    process: subprocess.Popen[bytes],
    host: str,
    port: int,
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    marker_seen_at_ns: int | None = None
    socket_seen_at_ns: int | None = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise CycleFailure(
                f"Server PID {process.pid} exited during startup with "
                f"code {return_code}"
            )
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        if marker_seen_at_ns is None and READY_MARKER in log_text:
            marker_seen_at_ns = time.time_ns()
        if socket_seen_at_ns is None and port_is_open(host, port):
            socket_seen_at_ns = time.time_ns()
        if marker_seen_at_ns is not None and socket_seen_at_ns is not None:
            return {
                "marker": READY_MARKER,
                "marker_seen_at_unix_ns": marker_seen_at_ns,
                "socket": f"{host}:{port}",
                "socket_seen_at_unix_ns": socket_seen_at_ns,
            }
        time.sleep(0.05)
    missing = []
    if marker_seen_at_ns is None:
        missing.append("ready marker")
    if socket_seen_at_ns is None:
        missing.append(f"socket {host}:{port}")
    raise CycleFailure(
        f"Server PID {process.pid} startup timed out after "
        f"{timeout_seconds} seconds; missing {', '.join(missing)}"
    )


def wait_for_endpoints_close(
    endpoints: list[dict[str, Any]],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        open_endpoints = [
            state for state in endpoint_states(endpoints) if state["open"]
        ]
        if not open_endpoints:
            return
        time.sleep(0.05)
    rendered = ", ".join(
        f"{state['name']}={state['host']}:{state['port']}"
        for state in endpoint_states(endpoints)
        if state["open"]
    )
    raise CycleFailure(f"Configured server endpoints remained open: {rendered}")


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_process_group_exit(
    process_group_id: int,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_group_exists(process_group_id):
            return True
        time.sleep(0.05)
    return not process_group_exists(process_group_id)


def signal_process_group(process_group_id: int, sig: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return


def stop_server(process: subprocess.Popen[bytes], timeout_seconds: float) -> int:
    process_group_id = process.pid
    signal_process_group(process_group_id, signal.SIGTERM)
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        signal_process_group(process_group_id, signal.SIGKILL)
        process.wait()
        wait_for_process_group_exit(process_group_id, timeout_seconds)
        raise CycleFailure(
            f"Server process group {process_group_id} did not stop after "
            "SIGTERM"
        )

    if not wait_for_process_group_exit(process_group_id, timeout_seconds):
        signal_process_group(process_group_id, signal.SIGKILL)
        wait_for_process_group_exit(process_group_id, timeout_seconds)
        raise CycleFailure(
            f"Server process group {process_group_id} retained child processes "
            "after SIGTERM"
        )
    return return_code


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def is_index_path(path: str) -> bool:
    return INDEX_PATH_RE.fullmatch(path) is not None


def parse_persisted_index(log_text: str) -> dict[str, Any]:
    matches = [
        match
        for match in PERSIST_INDEX_RE.finditer(log_text)
        if is_index_path(match.group("path"))
    ]
    require(
        len(matches) == 1,
        f"Expected exactly one persisted HNSW index object, found "
        f"{len(matches)}",
    )
    match = matches[0]
    return {
        "local_path": match.group("path"),
        "object_key": match.group("key"),
        "part_offset": int(match.group("offset")),
        "part_size": int(match.group("size")),
        "log_line": line_number(log_text, match.start()),
    }


def parse_checkpoint_index(log_text: str, local_path: str) -> dict[str, Any]:
    matches = [
        match
        for match in CHECKPOINT_INDEX_RE.finditer(log_text)
        if match.group("path") == local_path
    ]
    require(
        len(matches) == 1,
        f"Expected one checkpoint mapping for {local_path}, found "
        f"{len(matches)}",
    )
    match = matches[0]
    value = json.loads(match.group("value"))
    require(
        {"obj_key", "part_offset", "part_size"} <= value.keys(),
        f"Incomplete checkpoint object mapping for {local_path}: {value!r}",
    )
    return {
        "local_path": local_path,
        "object_key": str(value["obj_key"]),
        "part_offset": int(value["part_offset"]),
        "part_size": int(value["part_size"]),
        "log_line": line_number(log_text, match.start()),
    }


def parse_forced_query_object(log_text: str) -> dict[str, Any]:
    use_marker = f"Use index: {INDEX_NAME}"
    use_offsets = [
        match.start()
        for match in re.finditer(
            rf"{re.escape(use_marker)}\r?$",
            log_text,
            flags=re.MULTILINE,
        )
    ]
    require(
        len(use_offsets) == 1,
        f"Expected one forced use of {INDEX_NAME}, found {len(use_offsets)}",
    )
    use_offset = use_offsets[0]
    trace_offset = log_text.find(TRACE_PROOF, use_offset)
    require(
        trace_offset >= 0,
        f"Forced query lacks indexed-only trace proof: {TRACE_PROOF}",
    )
    finish_offset = log_text.find("KnnScan: 0 task finished", trace_offset)
    require(finish_offset >= 0, "Forced indexed task did not report completion")
    block = log_text[trace_offset:finish_offset]
    object_matches = list(QUERY_OBJECT_RE.finditer(block))
    require(
        len(object_matches) == 1,
        "Forced HNSW task did not access exactly one index object",
    )
    match = object_matches[0]
    object_key = match.group("key")
    object_path = match.group("path")
    put_pattern = re.compile(
        rf"PutObjCache (?:current )?object {re.escape(object_key)} ref count 0"
    )
    require(
        put_pattern.search(block) is not None,
        f"Forced query did not release index object {object_key}",
    )
    return {
        "index_name": INDEX_NAME,
        "object_key": object_key,
        "local_path": object_path,
        "use_index_log_line": line_number(log_text, use_offset),
        "trace_log_line": line_number(log_text, trace_offset),
        "object_access_log_line": line_number(
            log_text,
            trace_offset + match.start(),
        ),
    }


def reject_verify_index_mutation(log_text: str) -> None:
    violations = [
        name
        for name, pattern in VERIFY_INDEX_MUTATION_PATTERNS.items()
        if pattern.search(log_text) is not None
    ]
    require(
        not violations,
        "Verify phase rebuilt or persisted index state: "
        + ", ".join(violations),
    )


def correlate_index_proof(
    initialize_log: str,
    verify_log: str,
) -> dict[str, Any]:
    persisted = parse_persisted_index(initialize_log)
    checkpoint = parse_checkpoint_index(
        verify_log,
        persisted["local_path"],
    )
    initialize_query = parse_forced_query_object(initialize_log)
    verify_query = parse_forced_query_object(verify_log)
    expected = {
        "object_key": persisted["object_key"],
        "part_offset": persisted["part_offset"],
        "part_size": persisted["part_size"],
    }
    actual = {
        "object_key": checkpoint["object_key"],
        "part_offset": checkpoint["part_offset"],
        "part_size": checkpoint["part_size"],
    }
    require(actual == expected, "Checkpoint mapping differs from persisted index")
    deserialize_marker = f"Deserialize added object {persisted['object_key']}"
    deserialize_offset = verify_log.find(deserialize_marker)
    require(
        deserialize_offset >= 0,
        f"Verify startup did not deserialize index object "
        f"{persisted['object_key']}",
    )
    require(
        initialize_query["object_key"] == persisted["object_key"],
        "Initialize query did not read the newly persisted index object",
    )
    require(
        verify_query["object_key"] == persisted["object_key"],
        "Verify query did not read the checkpoint-restored index object",
    )
    require(
        verify_query["local_path"] is not None
        and verify_query["local_path"].endswith(persisted["local_path"]),
        "Verify query object path differs from checkpoint index path",
    )
    require(
        "wal entry size: 0" in verify_log,
        "Verify restart did not prove an empty WAL replay tail",
    )
    require(
        "append_ranges 0: []" in verify_log,
        "Verify restart did not prove an empty HNSW recovery append range",
    )
    reject_verify_index_mutation(verify_log)
    return {
        "persisted": persisted,
        "checkpoint_mapping": checkpoint,
        "checkpoint_deserialize": {
            "object_key": persisted["object_key"],
            "log_line": line_number(verify_log, deserialize_offset),
        },
        "initialize_query": initialize_query,
        "verify_query": verify_query,
        "verify_index_mutation_patterns_absent": sorted(
            VERIFY_INDEX_MUTATION_PATTERNS
        ),
    }


def run_client(
    args: argparse.Namespace,
    phase: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    evidence_path = output_dir / f"{phase}.json"
    command = [
        str(args.python),
        str(args.client),
        "--phase",
        phase,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--rows",
        str(args.rows),
        "--dimensions",
        str(args.dimensions),
        "--batch-size",
        str(args.batch_size),
        "--run-id",
        run_id,
        "--output",
        str(evidence_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.sdk_path)
    completed = subprocess.run(
        command,
        cwd=args.repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=args.client_timeout,
        check=False,
    )
    (output_dir / f"{phase}.stdout").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output_dir / f"{phase}.stderr").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    require(
        completed.returncode == 0,
        f"{phase} client failed with code {completed.returncode}",
    )
    require(evidence_path.is_file(), f"{phase} client emitted no JSON evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    require(evidence.get("status") == "pass", f"{phase} did not report pass")
    require(evidence.get("run_id") == run_id, f"{phase} run ID mismatch")
    return evidence


def run_server_phase(
    args: argparse.Namespace,
    phase: str,
    run_id: str,
    output_dir: Path,
    endpoints: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    log_path = output_dir / f"server-{phase}.log"
    phase_path = output_dir / f"server-{phase}-lifecycle.json"
    lifecycle: dict[str, Any] = {
        "status": "fail",
        "phase": phase,
        "started_at_unix_ns": time.time_ns(),
        "log": str(log_path),
        "process_group_signaling": True,
    }
    process: subprocess.Popen[bytes] | None = None
    log_file: Any = None
    try:
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            [str(args.server), f"--config={args.config}"],
            cwd=args.repo,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        lifecycle["pid"] = process.pid
        lifecycle["process_group_id"] = process.pid
        lifecycle["readiness"] = wait_for_server(
            process,
            args.host,
            args.port,
            log_path,
            args.startup_timeout,
        )
        evidence = run_client(args, phase, run_id, output_dir)
        return_code = stop_server(process, args.shutdown_timeout)
        lifecycle["exit_code"] = return_code
        require(
            return_code == 0,
            f"Server {phase} exited with code {return_code}",
        )
        wait_for_endpoints_close(endpoints, args.shutdown_timeout)
        log_file.flush()
        log_file.close()
        log_file = None
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        require(READY_MARKER in log_text, f"Server {phase} was never ready")
        require(
            SHUTDOWN_MARKER in log_text,
            f"Server {phase} did not report clean shutdown",
        )
        lifecycle["forced_query"] = parse_forced_query_object(log_text)
        if phase == "verify":
            reject_verify_index_mutation(log_text)
        lifecycle["configured_endpoints_after_shutdown"] = endpoint_states(
            endpoints
        )
        lifecycle["status"] = "pass"
        return lifecycle, evidence, log_text
    except BaseException as error:
        lifecycle["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        cleanup_errors: list[str] = []
        if process is not None and (
            process.poll() is None or process_group_exists(process.pid)
        ):
            try:
                lifecycle["cleanup_exit_code"] = stop_server(
                    process,
                    args.shutdown_timeout,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                signal_process_group(process.pid, signal.SIGKILL)
                if process.poll() is None:
                    try:
                        process.wait(timeout=args.shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        cleanup_errors.append("parent remained after SIGKILL")
                wait_for_process_group_exit(
                    process.pid,
                    args.shutdown_timeout,
                )
        if process is not None:
            lifecycle["exit_code"] = process.poll()
            lifecycle["process_group_gone"] = not process_group_exists(
                process.pid
            )
        if log_file is not None:
            log_file.flush()
            log_file.close()
        lifecycle["configured_endpoints_at_end"] = endpoint_states(endpoints)
        lifecycle["ended_at_unix_ns"] = time.time_ns()
        if cleanup_errors:
            lifecycle["cleanup_errors"] = cleanup_errors
            lifecycle["status"] = "fail"
        write_json(phase_path, lifecycle)


def command_record(
    command: list[str],
    cwd: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def cmake_cache_value(cache_text: str, key: str) -> str | None:
    match = re.search(
        rf"^{re.escape(key)}:[^=]+=(.+)$",
        cache_text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match is not None else None


def capture_toolchain(args: argparse.Namespace) -> dict[str, Any]:
    build_dir = args.server.parent.parent
    cache_path = build_dir / "CMakeCache.txt"
    build_path = build_dir / "build.ninja"
    result: dict[str, Any] = {
        "host": {
            name: command_record(command, args.repo)
            for name, command in {
                "sw_vers": ["sw_vers"],
                "uname": ["uname", "-a"],
                "xcodebuild": ["xcodebuild", "-version"],
                "sdk_path": ["xcrun", "--show-sdk-path"],
                "sdk_version": ["xcrun", "--show-sdk-version"],
            }.items()
        },
        "cmake_cache": (
            {"path": str(cache_path), "sha256": sha256(cache_path)}
            if cache_path.is_file()
            else None
        ),
        "build_ninja": (
            {"path": str(build_path), "sha256": sha256(build_path)}
            if build_path.is_file()
            else None
        ),
    }
    if cache_path.is_file():
        cache_text = cache_path.read_text(encoding="utf-8", errors="replace")
        compiler_value = cmake_cache_value(cache_text, "CMAKE_CXX_COMPILER")
        if compiler_value is not None:
            compiler = Path(compiler_value).resolve()
            result["cxx_compiler"] = {
                "path": str(compiler),
                "sha256": sha256(compiler) if compiler.is_file() else None,
                "version": command_record([str(compiler), "--version"], args.repo),
            }
    return result


def capture_artifact_identity(
    args: argparse.Namespace,
    config_bytes: bytes,
) -> dict[str, Any]:
    python_resolved = args.python.resolve()
    return {
        "server": {
            "path": str(args.server),
            "sha256": sha256(args.server),
        },
        "config": {
            "path": str(args.config),
            "sha256": sha256_bytes(config_bytes),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "client": {
            "path": str(args.client),
            "sha256": sha256(args.client),
        },
        "python": {
            "path": str(args.python),
            "resolved_path": str(python_resolved),
            "sha256": sha256(python_resolved),
        },
        "sdk": hash_source_tree(args.sdk_path, ".py"),
    }


def capture_git_provenance(
    repo: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    payloads = {
        "source-git-status.txt": status,
        "source-diff.patch": diff,
    }
    return (
        {
            "head": head.decode("utf-8"),
            "status": {
                "path": "source-git-status.txt",
                "sha256": sha256_bytes(status),
                "size_bytes": len(status),
            },
            "binary_diff": {
                "path": "source-diff.patch",
                "sha256": sha256_bytes(diff),
                "size_bytes": len(diff),
            },
        },
        payloads,
    )


def binary_description(server: Path, repo: Path) -> dict[str, Any]:
    file_record = command_record(["file", str(server)], repo)
    require(file_record["return_code"] == 0, "`file` failed for server binary")
    require(
        "Mach-O 64-bit executable arm64" in file_record["stdout"],
        f"Server is not a native arm64 Mach-O: {file_record['stdout'].strip()}",
    )
    return {
        "file": file_record,
        "dependencies": command_record(["otool", "-L", str(server)], repo),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--sdk-path", type=Path, required=True)
    parser.add_argument(
        "--client",
        type=Path,
        default=Path("tools/apple_silicon/native_server_hnsw_smoke.py"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23871)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--client-timeout", type=float, default=60.0)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    for name in ("server", "config", "python", "sdk_path", "client"):
        path = getattr(args, name)
        if not path.is_absolute():
            path = args.repo / path
        if name == "python":
            path = Path(os.path.abspath(path))
        else:
            path = path.resolve()
        require(path.exists(), f"--{name.replace('_', '-')} not found: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    require(args.rows > 731, "--rows must be greater than smoke target row 731")
    require(args.dimensions > 0, "--dimensions must be positive")
    require(args.batch_size > 0, "--batch-size must be positive")
    return args


def load_phase_lifecycle(output_dir: Path, phase: str) -> dict[str, Any] | None:
    path = output_dir / f"server-{phase}-lifecycle.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_cycle(
    args: argparse.Namespace,
    lifecycle: dict[str, Any],
    config_bytes: bytes,
) -> None:
    config = tomllib.loads(config_bytes.decode("utf-8"))
    require(
        config["network"]["server_address"] == args.host,
        "Config server address does not match --host",
    )
    require(
        int(config["network"]["client_port"]) == args.port,
        "Config client port does not match --port",
    )
    require(
        config["log"]["log_level"].lower() == "trace",
        "Lifecycle proof requires log_level = trace",
    )
    runtime_paths = configured_runtime_paths(config)
    endpoints = configured_endpoints(config)
    lifecycle["config"]["runtime_paths"] = {
        name: str(path) for name, path in runtime_paths.items()
    }
    lifecycle["config"]["endpoints"] = endpoints
    for state in endpoint_states(endpoints):
        require(
            not state["open"],
            f"Refusing to run while {state['name']} "
            f"{state['host']}:{state['port']} is open",
        )
    for name, path in runtime_paths.items():
        require(
            not path.exists(),
            f"Fresh lifecycle requires absent {name}: {path}",
        )

    lifecycle["server_binary"] = binary_description(args.server, args.repo)
    initialize_lifecycle, initialize_evidence, initialize_log = (
        run_server_phase(
            args,
            "initialize",
            lifecycle["run_id"],
            args.output_dir,
            endpoints,
        )
    )
    lifecycle["initialize"] = initialize_lifecycle
    expected_sdk_root = args.sdk_path.resolve()
    initialize_sdk_module = Path(
        initialize_evidence["platform"]["sdk_module"]
    ).resolve()
    require(
        initialize_sdk_module.is_relative_to(expected_sdk_root),
        f"Initialize imported SDK outside {expected_sdk_root}: "
        f"{initialize_sdk_module}",
    )

    persistence_dir = runtime_paths["persistence_dir"]
    persisted_after_initialize = hash_files(persistence_dir)
    require(
        any(
            name != "KEY_EMPTY" and metadata["size_bytes"] > 0
            for name, metadata in persisted_after_initialize.items()
        ),
        "Initialize phase produced no nonempty persistence object",
    )

    verify_lifecycle, verify_evidence, verify_log = run_server_phase(
        args,
        "verify",
        lifecycle["run_id"],
        args.output_dir,
        endpoints,
    )
    lifecycle["verify"] = verify_lifecycle
    verify_sdk_module = Path(verify_evidence["platform"]["sdk_module"]).resolve()
    require(
        verify_sdk_module == initialize_sdk_module,
        "Initialize and verify imported different SDK modules",
    )
    persisted_after_verify = hash_files(persistence_dir)

    require(
        initialize_lifecycle["pid"] != verify_lifecycle["pid"],
        "Initialize and verify used the same server PID",
    )
    require(
        persisted_after_verify == persisted_after_initialize,
        "Persistence object set or content changed across read-only restart",
    )
    require(
        initialize_evidence["validation"]["query"]["result_ids"]
        == verify_evidence["validation"]["query"]["result_ids"],
        "Result IDs changed after restart",
    )
    require(
        initialize_evidence["validation"]["query"]["distances"]
        == verify_evidence["validation"]["query"]["distances"],
        "Distances changed after restart",
    )
    initialize_digest = initialize_evidence["validation"][
        "complete_dataset"
    ]["canonical_float32_sha256"]
    verify_digest = verify_evidence["validation"]["complete_dataset"][
        "canonical_float32_sha256"
    ]
    require(
        initialize_digest == verify_digest,
        "Complete ID/vector digest changed after restart",
    )

    index_proof = correlate_index_proof(initialize_log, verify_log)
    index_object = index_proof["persisted"]
    object_metadata = persisted_after_initialize.get(
        index_object["object_key"]
    )
    require(
        object_metadata is not None,
        "Persisted HNSW object key is absent from persistence directory",
    )
    require(
        object_metadata["size_bytes"]
        >= index_object["part_offset"] + index_object["part_size"],
        "Persisted HNSW object is shorter than its checkpoint mapping",
    )

    final_identity = capture_artifact_identity(args, config_bytes)
    require(
        final_identity == lifecycle["artifact_identity_before"],
        "Binary, config, runner, client, Python, or SDK changed during cycle",
    )
    lifecycle.update(
        {
            "status": "pass",
            "sdk": {
                "root": str(expected_sdk_root),
                "module": str(initialize_sdk_module),
                "module_sha256": sha256(initialize_sdk_module),
            },
            "complete_dataset_float32_sha256": initialize_digest,
            "index_object_proof": index_proof,
            "persisted_objects_after_initialize": persisted_after_initialize,
            "persisted_objects_after_verify": persisted_after_verify,
            "artifact_identity_after": final_identity,
        }
    )


def main() -> None:
    args = parse_args()
    require(
        not args.output_dir.exists(),
        f"Output directory already exists: {args.output_dir}",
    )
    config_bytes = args.config.read_bytes()
    git_provenance, git_payloads = capture_git_provenance(args.repo)
    run_id = str(uuid.uuid4())
    lifecycle: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": "native-server-hnsw-v2",
        "evidence_class": "functional_persistence",
        "benchmark_eligible": False,
        "claim": (
            "A native arm64 Infinity server persists and reloads the same "
            "physical HNSW index object across a clean restart."
        ),
        "limitations": [
            "Synthetic smoke workload; all timings are diagnostic.",
            "Does not exercise a nonempty WAL replay tail or crash recovery.",
            "Does not support an Infinity-versus-FAISS performance claim.",
        ],
        "status": "fail",
        "run_id": run_id,
        "started_at_unix_ns": time.time_ns(),
        "repo": str(args.repo),
        "invocation": {
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
            "startup_timeout_seconds": args.startup_timeout,
            "shutdown_timeout_seconds": args.shutdown_timeout,
            "client_timeout_seconds": args.client_timeout,
        },
        "git": git_provenance,
        "config": {
            "path": str(args.config),
            "sha256": sha256_bytes(config_bytes),
        },
        "artifact_identity_before": capture_artifact_identity(
            args,
            config_bytes,
        ),
        "toolchain": capture_toolchain(args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in git_payloads.items():
        (args.output_dir / relative_path).write_bytes(payload)

    try:
        run_cycle(args, lifecycle, config_bytes)
    except BaseException as error:
        for phase in ("initialize", "verify"):
            phase_lifecycle = load_phase_lifecycle(args.output_dir, phase)
            if phase_lifecycle is not None:
                lifecycle[phase] = phase_lifecycle
        lifecycle["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        lifecycle["ended_at_unix_ns"] = time.time_ns()
        write_json(args.output_dir / "lifecycle.json", lifecycle)

    print(json.dumps(lifecycle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
