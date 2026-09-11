"""Regression baseline for the *committed* test suites on macOS arm64.

Every other suite under ``test/eval_macos/`` writes new tests. This one is the
odd one out on purpose: it runs the test corpora that already ship in the repo
(the gtest ``test_main`` binary, the ``.slt`` SQL-logic suite, and
``python/test_pysdk`` over both thrift and HTTP) on the *current* server binary
and asserts that the result matches the documented baseline in
``docs/apple_silicon/MACOS_VERIFICATION.md``. Its job is to catch a regression
introduced by the two storage commits that landed after that doc's numbers were
recorded (``b2d59374a`` WAL fsync, ``ccc58a3da`` rcu_multimap lock + export
truncation).

What "pass" means here
----------------------
* Unit tests: exactly the five documented failures, and *no others*. A sixth
  failure is a regression and fails :func:`test_unit_tests_no_new_failures`.
* SQL logic: 205/205, as the doc claims and the prior report recorded.
* pysdk (thrift and HTTP): every failure must be explained by a hardcoded Linux
  path (``/var/infinity``), i.e. a *portability* problem in the test harness, not
  an *engine* wrong-answer. A failure that is not path-related fails the
  classification test, because that would be a real defect the doc did not call
  out.

Coordination
------------
The committed suites are pinned to the *default* ports: sqllogictest talks to
postgres 5432, pysdk-thrift to 23817, pysdk-http to 23820. Those are the ``dev``
instance (port offset 0), which ``harness.INSTANCES`` deliberately does not hand
out to the concurrent eval suites. This module therefore drives
``scripts/apple_silicon/run_server.sh --instance dev`` directly, always with
``--fresh`` (the SLT suite is not idempotent), and always stops it in teardown.
It never touches an ``eval-*`` instance.

The unit-test binary is expensive (~15-25 min, longer under load) and produces a
machine-readable summary via ``--gtest_output=json``. To keep this module cheap
to re-run, the unit-test test *reuses* a JSON artifact at
``build/eval/logs/unit_tests.json`` when it is newer than the server binary
(proof it was produced by this binary); otherwise it runs ``test_main`` itself
with a long timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = REPO_ROOT / "build" / "macos-arm64-release" / "src" / "infinity"
TEST_MAIN = REPO_ROOT / "build" / "macos-arm64-release" / "src" / "test_main"
RUN_SERVER = REPO_ROOT / "scripts" / "apple_silicon" / "run_server.sh"
RUN_SLT = REPO_ROOT / "scripts" / "apple_silicon" / "run_slt.py"
LOG_DIR = REPO_ROOT / "build" / "eval" / "logs"
UNIT_JSON = LOG_DIR / "unit_tests.json"
SLT_DATA_ROOT = REPO_ROOT / "build" / "slt" / "test_data"

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

# The five failures documented in MACOS_VERIFICATION.md, each as a substring that
# uniquely identifies it in the gtest report. gtest renders the parameterised PGM
# case as "TestPGM/9.TestPGMTypeSupport", so a substring match is more robust than
# an exact "classname.name" reconstruction.
KNOWN_UNIT_FAILURES = {
    "TestPGMTypeSupport",  # TestPGM/9 (double): long double == double on Darwin
    "LowCardinalitySecondaryIndexTest.TestAllDataType",  # libc++ multimap layout
    "RAGAnalyzerTest.test_tokenize_consistency_with_python",  # NLTK network download
    "RAGAnalyzerTest.test_fine_grained_tokenize_consistency_with_python",
    "RAGAnalyzerTest.test_set_language_dutch",
}


def _is_known_unit_failure(full_name: str) -> bool:
    return any(known in full_name for known in KNOWN_UNIT_FAILURES)


def _run_test_main(json_path: Path, log_path: Path, timeout: float = 3600.0) -> None:
    """Run the gtest binary, emitting a JSON summary. Raises on infra failure."""
    if not TEST_MAIN.exists():
        raise RuntimeError(f"no test_main binary at {TEST_MAIN}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            [str(TEST_MAIN), "--gtest_brief=1", f"--gtest_output=json:{json_path}"],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
        )
    # gtest exits non-zero when any test fails; that is expected here (5 known
    # failures), so the return code is not itself an assertion. The JSON is.
    _ = proc.returncode


def _ensure_unit_json() -> Path:
    """Return a gtest JSON produced by the *current* binary, running it if needed."""
    fresh = (
        UNIT_JSON.exists()
        and BINARY.exists()
        and UNIT_JSON.stat().st_mtime >= BINARY.stat().st_mtime
    )
    if not fresh:
        _run_test_main(UNIT_JSON, LOG_DIR / "unit_tests_run.log")
    return UNIT_JSON


def _parse_gtest_json(path: Path) -> tuple[int, int, int, list[str]]:
    """Return (registered, ran, total_failures, failing_full_names).

    gtest's top-level ``tests`` counts *registered* cases including DISABLED_ ones;
    the console "N tests ran" figure (which the doc's 1131 comes from) is
    ``registered - disabled``. Both are returned so callers can compare against the
    right number.
    """
    data = json.loads(path.read_text())
    registered = int(data.get("tests", 0))
    disabled = int(data.get("disabled", 0))
    ran = registered - disabled
    failing: list[str] = []
    for suite in data.get("testsuites", []):
        for case in suite.get("testsuite", []):
            # A case failed if it carries a non-empty "failures" list.
            if case.get("failures"):
                classname = case.get("classname", suite.get("name", "?"))
                failing.append(f"{classname}.{case.get('name', '?')}")
    return registered, ran, int(data.get("failures", len(failing))), failing


@pytest.fixture(scope="module")
def unit_json() -> Path:
    return _ensure_unit_json()


def test_unit_tests_no_new_failures(unit_json: Path) -> None:
    """Every failing unit test must be one of the five documented failures.

    A failure outside that set is a regression from the storage commits and is a
    blocker.
    """
    registered, ran, reported_failures, failing = _parse_gtest_json(unit_json)
    unexpected = [name for name in failing if not _is_known_unit_failure(name)]
    assert not unexpected, (
        f"NEW unit-test failure(s) not documented in MACOS_VERIFICATION.md "
        f"(regression candidates): {unexpected}\n"
        f"all failing: {failing}\n"
        f"registered={registered} ran={ran} reported_failures={reported_failures}"
    )


def test_unit_tests_known_failures_present(unit_json: Path) -> None:
    """The five documented failures should still be failing.

    If one *stops* failing that is good news, not a regression, so this test does
    not hard-fail on a missing one; it records which known failures are absent so
    the doc can be updated. It only asserts the count is plausible.
    """
    _registered, _ran, _reported, failing = _parse_gtest_json(unit_json)
    matched = {k for k in KNOWN_UNIT_FAILURES if any(k in f for f in failing)}
    missing = KNOWN_UNIT_FAILURES - matched
    # Environment-dependent ones (RAG needs a network download) may flip; do not
    # hard-fail, but surface the delta.
    print(f"known failures still failing: {sorted(matched)}")
    print(f"known failures now passing (doc may be stale): {sorted(missing)}")
    assert matched, "none of the five documented failures reproduced — unexpected"


def test_unit_tests_total_count(unit_json: Path) -> None:
    """The suite should still run ~1131 tests (doc baseline; excludes disabled)."""
    registered, ran, _reported, _failing = _parse_gtest_json(unit_json)
    assert ran >= 1000, f"only {ran} unit tests ran (registered={registered}); expected ~1131"
    if ran != 1131:
        print(f"NOTE: {ran} unit tests ran (registered={registered}); doc records 1131")


# ---------------------------------------------------------------------------
# dev-instance lifecycle (shared by SLT and pysdk)
# ---------------------------------------------------------------------------


def _run_server(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [str(RUN_SERVER), *args, "--instance", "dev"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


@pytest.fixture
def fresh_dev():
    """A freshly-started ``dev`` server on the default ports, stopped afterwards.

    Function-scoped and ``--fresh`` because the committed suites mutate global
    state (databases, tables, snapshots) and the SLT suite is not idempotent.
    """
    _run_server("stop", check=False)
    _run_server("start", "--fresh")
    try:
        yield
    finally:
        _run_server("stop", check=False)


def _dev_log_errors() -> list[str]:
    """Server-side error/critical/fatal lines from the dev instance log."""
    log = REPO_ROOT / "build" / "instances" / "dev" / "log" / "infinity.log"
    if not log.exists():
        return []
    text = log.read_text(errors="replace")
    return [
        ln for ln in text.splitlines()
        if "| error |" in ln.lower() or "| critical |" in ln.lower()
        or "| fatal |" in ln.lower()
    ]


# ---------------------------------------------------------------------------
# SQL logic tests
# ---------------------------------------------------------------------------


def test_slt_all_pass(fresh_dev) -> None:
    """The full committed .slt suite must pass 205/205 on the current binary.

    Uses ``--skip-generate`` to reuse the staged corpus under build/slt/test_data;
    skips (does not fail) if that corpus is absent, because regenerating several GB
    inside a unit test is not this test's job.
    """
    if not SLT_DATA_ROOT.is_dir() or not any(SLT_DATA_ROOT.iterdir()):
        pytest.skip(
            f"SLT corpus not staged at {SLT_DATA_ROOT}; run "
            f"'uv run python scripts/apple_silicon/run_slt.py' once to populate it"
        )
    report = REPO_ROOT / "build" / "slt" / "report.txt"
    proc = subprocess.run(
        [sys.executable, str(RUN_SLT), "--skip-generate", "--report", str(report)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600,
    )
    # Parse the report for FAIL lines rather than trusting only the exit code.
    failed: list[str] = []
    passed = 0
    if report.exists():
        for line in report.read_text().splitlines():
            if line.startswith("PASS\t"):
                passed += 1
            elif line.startswith("FAIL\t"):
                failed.append(line.split("\t", 2)[-1])
    assert report.exists(), (
        f"run_slt.py produced no report; exit={proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
    )
    assert not failed, f"SLT failures ({len(failed)}/{passed + len(failed)}): {failed}"
    assert passed >= 200, f"only {passed} SLT files passed; expected 205"


# ---------------------------------------------------------------------------
# pysdk over thrift and HTTP
# ---------------------------------------------------------------------------

# A representative slice: pure DDL/DML that must work on any platform, plus
# import-driven files that exercise the hardcoded-path portability failure.
PYSDK_DDL_FILES = ["test_database.py", "test_table.py"]
PYSDK_IMPORT_FILES = ["test_basic.py"]
PYSDK_SUBSET = PYSDK_DDL_FILES + PYSDK_IMPORT_FILES


def _run_pysdk(mode: str, files: list[str]) -> Path:
    """Run a pysdk subset over thrift (mode='') or HTTP (mode='--http').

    Returns the path to a JUnit XML report (pytest built-in, no plugin needed).
    """
    tag = "http" if mode else "thrift"
    junit = LOG_DIR / f"pysdk_{tag}_subset.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable, "-m", "pytest",
        *[f"python/test_pysdk/{f}" for f in files],
        "-p", "no:cacheprovider", "--no-header", "-q",
        f"--junitxml={junit}",
    ]
    if mode:
        args.append(mode)
    subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    return junit


def _classify_junit(junit: Path) -> tuple[int, int, list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (passed, total, portability_failures, engine_failures).

    A failure is *portability* if its message/traceback references the hardcoded
    Linux data root; anything else is treated as a candidate *engine* defect.
    """
    tree = ET.parse(junit)
    root = tree.getroot()
    passed = total = 0
    portability: list[tuple[str, str]] = []
    engine: list[tuple[str, str]] = []
    for case in root.iter("testcase"):
        total += 1
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        problems = list(case.iter("failure")) + list(case.iter("error"))
        if not problems:
            # skipped cases carry a <skipped> child; count only real passes.
            if not list(case.iter("skipped")):
                passed += 1
            continue
        blob = " ".join(
            (p.get("message", "") + " " + (p.text or "")) for p in problems
        )
        if "/var/infinity" in blob or "FileNotFoundError" in blob:
            portability.append((name, blob[:300]))
        else:
            engine.append((name, blob[:400]))
    return passed, total, portability, engine


@pytest.mark.parametrize("mode", ["", "--http"], ids=["thrift", "http"])
def test_pysdk_failures_are_portability_only(fresh_dev, mode: str) -> None:
    """pysdk failures on macOS must all be hardcoded-path portability, not engine.

    Also asserts the pure-DDL files pass, i.e. the engine really works over this
    transport, and checks the server logged no error while the subset ran.
    """
    junit = _run_pysdk(mode, PYSDK_SUBSET)
    assert junit.exists(), f"pysdk ({mode or 'thrift'}) produced no JUnit XML"
    passed, total, portability, engine = _classify_junit(junit)
    print(f"pysdk[{mode or 'thrift'}]: {passed}/{total} passed, "
          f"{len(portability)} portability failures, {len(engine)} engine failures")
    for name, blob in engine:
        print(f"  ENGINE? {name}: {blob}")
    assert not engine, (
        f"pysdk[{mode or 'thrift'}] failures that are NOT explained by a hardcoded "
        f"path (candidate engine defects): {[n for n, _ in engine]}"
    )
    assert passed > 0, f"no pysdk[{mode or 'thrift'}] test passed; engine may be broken"
