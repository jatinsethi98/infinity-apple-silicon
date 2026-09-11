"""Run the committed ``python/parallel_test`` concurrency suite on macOS/arm64.

This suite has never been run on macOS. It cannot be run as-committed because it
hardcodes the default thrift port (23817) in ``python/parallel_test/common/
common_values.py`` and would collide with whatever instance a human is running on
the default ports. This driver runs each committed module against a private
``eval-spare-a`` instance without editing any committed file: the actual port
override happens in a child process (``_parallel_runner.py``) that rewrites
``common_values`` in memory before pytest collects the module.

Each committed module is one parametrized test here. For every module the driver:

* starts ``eval-spare-a`` **fresh** (so one module's DROP DATABASE / crash cannot
  poison the next),
* runs the module in a child ``pytest`` process with a wall-clock timeout, so a
  hang is reported as a failure instead of stalling the whole evaluation,
* captures the child's full output under ``build/eval/logs/`` and its per-test
  results as JUnit XML,
* after the run, checks the **server's own log** (``inst.log_errors``) and whether
  the server process is still alive — a clean client result over a server-side
  error or a crashed server is a defect the child pytest cannot see,
* records a machine-readable summary under ``build/eval/results/``.

Run it (long; launch in the background and poll)::

    uv run pytest test/eval_macos/run_parallel_suite.py -v

The classification of each failure as ``portability`` (path/port/environment) vs
``engine-defect`` (a real race or wrong answer) is done by a human reading the
recorded evidence; this driver only gathers that evidence deterministically.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
import _race_secondary_index as race  # noqa: E402

REPO_ROOT = harness.REPO_ROOT
RUNNER = Path(__file__).resolve().parent / "_parallel_runner.py"
PT_DIR = REPO_ROOT / "python" / "parallel_test"
LOG_DIR = REPO_ROOT / "build" / "eval" / "logs"
INSTANCE_NAME = "eval-spare-a"

# One entry per committed module. ``timeout`` is a wall-clock ceiling for the whole
# module (the module's own tests run for a fixed wall time each; these ceilings sit
# well above the observed runtime so a trip means a genuine hang, not a slow box).
MODULES = [
    {"file": "test_sdkbase.py", "timeout": 60},
    {"file": "test_insert_parallel.py", "timeout": 360},
    {"file": "test_insert_delete_parallel.py", "timeout": 600},
    {"file": "test_insert_delete_update.py", "timeout": 180},
    {"file": "test_ddl_and_insert_delete.py", "timeout": 180},
    {"file": "test_ddl_parallel.py", "timeout": 180},
    {"file": "test_index_parallel.py", "timeout": 420},
    {"file": "test_chaos.py", "timeout": 300},
]
IDS = [m["file"] for m in MODULES]


def _parse_junit(xml_path: Path) -> dict:
    """Pull per-testcase outcomes out of the child's JUnit XML."""
    result = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "cases": []}
    if not xml_path.exists():
        return result
    tree = ET.parse(xml_path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    for suite in suites:
        result["tests"] += int(suite.get("tests", 0))
        result["failures"] += int(suite.get("failures", 0))
        result["errors"] += int(suite.get("errors", 0))
        result["skipped"] += int(suite.get("skipped", 0))
        for case in suite.findall("testcase"):
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            status = "passed"
            detail = ""
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            if failure is not None:
                status = "failed"
                detail = (failure.get("message") or "")[:600]
            elif error is not None:
                status = "error"
                detail = (error.get("message") or "")[:600]
            elif skipped is not None:
                status = "skipped"
                detail = (skipped.get("message") or "")[:300]
            result["cases"].append({"name": name, "status": status,
                                    "detail": detail, "time": case.get("time")})
    return result


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


@pytest.fixture(scope="session")
def server():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    inst = harness.instance(INSTANCE_NAME)
    yield inst
    inst.stop()


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_parallel_module(server, module):
    if harness.free_disk_bytes() < 5 * 1024**3:
        pytest.skip("less than 5 GiB free disk")

    name = module["file"]
    stem = name[:-3]
    log_path = LOG_DIR / f"{stem}.log"
    xml_path = LOG_DIR / f"{stem}.xml"
    module_path = str(PT_DIR / name)

    # Fresh server per module: isolation, and a crash in one module does not
    # silently corrupt the next.
    server.start(fresh=True)
    start_pid = server.pid()
    log_off = server.log_offset()

    cmd = [sys.executable, str(RUNNER), str(server.thrift_port), module_path,
           str(xml_path)]
    outcome = "completed"
    rc = None
    t0 = time.monotonic()
    with open(log_path, "w") as logfh:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=logfh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=module["timeout"])
        except subprocess.TimeoutExpired:
            outcome = "hang"
            # Kill the whole child process group; leave the SERVER alone.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = None
    elapsed = round(time.monotonic() - t0, 1)

    junit = _parse_junit(xml_path)
    server_errs = server.log_errors(log_off)
    alive_after = server.pid()
    server_crashed = alive_after is None or alive_after != start_pid

    summary = {
        "module": name,
        "outcome": outcome,
        "child_returncode": rc,
        "elapsed_s": elapsed,
        "timeout_s": module["timeout"],
        "tests": junit["tests"],
        "failures": junit["failures"],
        "errors": junit["errors"],
        "skipped": junit["skipped"],
        "cases": junit["cases"],
        "server_error_lines": server_errs[:50],
        "server_error_count": len(server_errs),
        "server_crashed": server_crashed,
        "pid_before": start_pid,
        "pid_after": alive_after,
        "log": str(log_path),
        "junit": str(xml_path),
    }
    harness.record(f"parallel__{stem}", summary)

    # Build a single actionable failure message that carries the evidence.
    problems = []
    if outcome == "hang":
        problems.append(
            f"HANG: child did not finish within {module['timeout']}s "
            f"(killed at {elapsed}s)")
    if server_crashed:
        problems.append(
            f"SERVER CRASH: pid {start_pid} -> {alive_after} during module")
    if rc not in (0, None) and outcome != "hang":
        problems.append(f"child pytest returncode={rc}")
    if junit["failures"] or junit["errors"]:
        bad = [c for c in junit["cases"] if c["status"] in ("failed", "error")]
        problems.append(
            f"{junit['failures']} failed / {junit['errors']} errored: "
            + "; ".join(f"{c['name']} [{c['status']}] {c['detail']}" for c in bad))
    if server_errs:
        problems.append(
            f"{len(server_errs)} server-side error line(s); first: {server_errs[0]}")

    if problems:
        pytest.fail(
            f"[{name}] " + " | ".join(problems)
            + f"\n--- child log tail ---\n{_tail(log_path)}",
            pytrace=False,
        )


def test_secondary_index_range_race_stress(server):
    """Regression for the ``RcuMultiMap::range`` data race fixed in ccc58a3da.

    The committed ``test_multiple_index_types_parallel`` touches this path but only
    lightly (~285 range reads among ten index types, swallowing every exception).
    This isolates it and pushes ~16k range queries per 8s round against concurrent
    inserts into the same in-memory secondary index, asserting the soundness
    invariants a torn red-black tree would break (see ``_race_secondary_index``).
    Deterministic: RNG is seeded and each round runs a fixed wall-clock window; the
    executed-query count is asserted so a future reader sees how hard it was pushed.
    """
    if harness.free_disk_bytes() < 5 * 1024**3:
        pytest.skip("less than 5 GiB free disk")
    from infinity.common import NetworkAddress
    from infinity.connection_pool import ConnectionPool

    server.start(fresh=True)
    start_pid = server.pid()
    log_off = server.log_offset()

    rounds = 2
    total_q = 0
    total_ins = 0
    total_rows = 0
    violations: list = []
    query_errors: list = []
    pool = ConnectionPool(NetworkAddress("127.0.0.1", server.thrift_port))
    try:
        for i in range(rounds):
            st = race.run(pool, seed=1000 + i, seconds=8.0)
            total_q += st.range_queries
            total_ins += st.inserts
            total_rows += st.rows_returned
            violations.extend(st.violations)
            query_errors.extend(st.query_errors)
    finally:
        pool.destroy()

    server_errs = server.log_errors(log_off)
    alive_after = server.pid()
    server_crashed = alive_after is None or alive_after != start_pid

    harness.record("parallel__secondary_index_range_race", {
        "rounds": rounds,
        "range_queries": total_q,
        "inserts": total_ins,
        "rows_returned": total_rows,
        "violations": violations[:50],
        "violation_count": len(violations),
        "query_errors": query_errors[:50],
        "query_error_count": len(query_errors),
        "server_error_lines": server_errs[:50],
        "server_error_count": len(server_errs),
        "server_crashed": server_crashed,
        "pid_before": start_pid,
        "pid_after": alive_after,
    })

    problems = []
    if server_crashed:
        problems.append(f"SERVER CRASH: pid {start_pid} -> {alive_after}")
    if violations:
        problems.append(
            f"{len(violations)} soundness violation(s); first: {violations[0]}")
    if query_errors:
        problems.append(
            f"{len(query_errors)} range-query error(s); first: {query_errors[0]}")
    if server_errs:
        problems.append(
            f"{len(server_errs)} server-side error line(s); first: {server_errs[0]}")
    if total_q < 2000:
        problems.append(
            f"only {total_q} range queries executed; stress did not actually run")
    if problems:
        pytest.fail("[secondary_index_range_race] " + " | ".join(problems),
                    pytrace=False)
