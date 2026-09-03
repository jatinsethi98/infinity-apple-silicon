#!/usr/bin/env python3
"""Run the sqllogictest suites against a natively-built Infinity on macOS.

Why this exists instead of `tools/sqllogictest.py`
-------------------------------------------------

Two reasons, and neither is a macOS special case in the engine.

1. **The suite bakes an absolute Linux path into the SQL.** 94 of the committed
   `.slt` files contain statements like

       COPY t FROM '/var/infinity/test_data/embedding_float_dim4.csv' ...

   The path is resolved by the *server*, not by the test runner, so no runner
   flag can redirect it. On Linux CI the suite runs as root in a container and
   `/var/infinity` is simply writable. On macOS `/var` is root-owned and
   `/var/infinity` does not exist, so those files cannot run as written.

   Rather than edit 94 upstream files (a large diff that would have to be carried
   forever and would conflict with every upstream change to the suite), this
   driver materialises a *rewritten copy* of the tree under `build/slt/sql`,
   substituting the data root. Upstream files are never modified, and dropping
   this driver is enough to go back to the stock layout.

2. **A bring-up run needs the whole failure list, not the first failure.**
   `tools/sqllogictest.py` raises on the first non-zero exit, which is the right
   behaviour for a green suite guarding a merge and the wrong behaviour when you
   are finding out for the first time which subsystems do not work on a new
   platform. This driver runs everything and prints a grouped summary.

The generated `.slt` files (the ones produced by `tools/generate_*.py`) do not
need rewriting: they embed whatever data root they are told to use, and they are
gitignored, so we simply tell the generators to use the local root.

Usage
-----
Run it with the project's uv environment, not the system interpreter: several of
the parquet generators import pyarrow, which is a project dependency and is not
present in a stock system python.

    # server must already be running (see run_server.sh)
    uv run python scripts/apple_silicon/run_slt.py                    # everything
    uv run python scripts/apple_silicon/run_slt.py --suite dml         # one subtree
    uv run python scripts/apple_silicon/run_slt.py --suite dml/delete  # narrower
    uv run python scripts/apple_silicon/run_slt.py --file test_delete.slt
    uv run python scripts/apple_silicon/run_slt.py --skip-generate     # reuse data

The first run generates several GB of test data and is slow; use --skip-generate
on subsequent runs. The generators self-skip work whose output already exists, so
a rerun without the flag is cheap but not free.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The path the committed .slt files were written against. Substituted for the
# local data root when the tree is materialised.
UPSTREAM_DATA_ROOT = "/var/infinity/test_data"


@dataclass
class Result:
    path: str
    ok: bool
    seconds: float
    detail: str


BUILD_ROOT = REPO_ROOT / "build"


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def ensure_tools_importable() -> None:
    """Put tools/ on sys.path so its generators and helpers import by bare name.

    They import each other that way, and tools/sqllogictest.py is written to run with
    the repo root as cwd, so both have to be arranged before touching any of them.
    """
    tools = str(REPO_ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def require_deletable(path: Path, flag: str) -> Path:
    """Refuse to recursively delete anything that is not ours to delete.

    Two of this script's directories are wiped and rebuilt on every run. Both are
    settable from the command line, so without this check `--sql-root .` would delete
    the working tree and `--data-root /` would be worse. Constraining them to live
    under build/ makes the destructive operations unable to escape, which is a
    property worth having even though nobody would pass those flags on purpose.
    """
    resolved = path.resolve()
    build_root = BUILD_ROOT.resolve()
    if resolved == build_root or build_root not in resolved.parents:
        sys.exit(
            f"error: {flag} must be a directory strictly beneath {build_root}, "
            f"because it is deleted and recreated on every run.\n"
            f"  got: {resolved}"
        )
    return resolved


def find_sqllogictest(explicit: str | None) -> str:
    """Locate the sqllogictest binary, preferring the repo-local install."""
    if explicit:
        if not shutil.which(explicit) and not Path(explicit).is_file():
            sys.exit(f"error: --sqllogictest {explicit} is not executable")
        return explicit

    local = REPO_ROOT / "build" / "tools" / "bin" / "sqllogictest"
    if local.is_file():
        return str(local)

    found = shutil.which("sqllogictest")
    if found:
        return found

    sys.exit(
        "error: sqllogictest not found.\n"
        "  run: scripts/apple_silicon/install_test_tools.sh"
    )


# Kept in the order tools/sqllogictest.py invokes them, minus the two it has
# commented out. Listed explicitly rather than globbed so that a generator added
# upstream is a visible change here rather than a silent omission. Every one of
# these exposes `generate(generate_if_exists: bool, copy_dir: str)`.
GENERATORS = [
    "generate_big", "generate_fvecs", "generate_sort", "generate_limit",
    "generate_aggregate", "generate_top", "generate_top_varchar",
    "generate_compact", "generate_hnsw_with_delete", "generate_index_scan",
    "generate_many_import", "generate_big_point_query_test_fastroughfilter",
    "generate_many_import_drop", "generate_mem_hnsw", "generate_big_sparse",
    "generate_csr", "generate_bvecs", "generate_emvb_test_data",
    "generate_test_parquet", "generate_sparse_parquet",
    "generate_embedding_parquet", "generate_varchar_parquet",
    "generate_tensor_parquet", "generate_tensor_array_parquet",
    "generate_multivector_parquet", "generate_multivector_knn_scan",
    "generate_groupby1", "generate_unnest", "generate_large_import",
]


def generate_test_data(data_root: Path, regenerate: bool) -> None:
    """Run the upstream generators, pointing them at the local data root.

    The generators write their data files into `data_root` and their .slt files
    into test/sql (where they are gitignored). Because we pass the local root,
    the SQL they emit already refers to the right place and needs no rewriting.

    They are imported and called in-process rather than shelled out to: they are
    plain modules with a uniform entry point, and importing them directly means
    a failure surfaces as a traceback pointing at the generator instead of as an
    opaque non-zero exit.
    """
    # Fail here with a useful message rather than 20 generators later with an
    # ImportError: pyarrow is a project dependency, so a missing one means the
    # script is running under the wrong interpreter.
    if importlib.util.find_spec("pyarrow") is None:
        sys.exit(
            "error: pyarrow is not importable, so the parquet generators cannot run.\n"
            f"  this interpreter is {sys.executable}\n"
            "  re-run with the project environment:  uv run python scripts/apple_silicon/run_slt.py"
        )

    log(f"generating test data into {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)

    ensure_tools_importable()
    # The generators also resolve some paths relative to the process cwd.
    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        for name in GENERATORS:
            module = importlib.import_module(name)
            entry = getattr(module, "generate", None)
            if entry is None:
                sys.exit(f"error: tools/{name}.py has no generate() entry point")
            try:
                entry(regenerate, str(data_root))
            except Exception as exc:  # noqa: BLE001 - report which one, then stop
                raise SystemExit(
                    f"error: tools/{name}.py generate() failed: {exc!r}"
                ) from exc
    finally:
        os.chdir(previous_cwd)
    log(f"generation finished ({len(GENERATORS)} generators)")


def reset_export_tmp(data_root: Path) -> Path:
    """Recreate data_root/tmp empty before every run.

    The export suites write to fixed filenames under this directory and then read
    them back. Two things make a stale file actively dangerous rather than merely
    untidy: the engine opens output files with O_RDWR | O_CREAT and no O_TRUNC
    (src/storage/io/virtual_store_impl.cpp:127) and writes from offset zero, so a
    shorter second export leaves the tail of the previous one in place; and the
    tests then import what they exported. The result is a pass or a failure that
    depends on what the previous run happened to write.

    Upstream's runner clears this directory for the same reason
    (tools/sqllogictest.py). Not doing it is not a portability difference — it is
    just a missing step.
    """
    tmp_dir = require_deletable(data_root / "tmp", "--data-root")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    return tmp_dir


def copy_static_data(data_root: Path) -> None:
    """Flatten test/data into the data root, as the suites expect.

    Delegates to upstream's copy_all rather than reimplementing it: the .slt files
    refer to bare filenames under the data root instead of the nested layout in
    test/data, and the symlink handling (skip links rather than dereference them) is
    a subtlety worth having in exactly one place.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    ensure_tools_importable()
    sqllogictest = importlib.import_module("sqllogictest")
    sqllogictest.copy_all(str(REPO_ROOT / "test" / "data"), str(data_root))


def materialise_suite(data_root: Path, sql_out: Path, suite: str | None = None) -> int:
    """Copy test/sql to sql_out, rewriting the baked-in data root.

    Only `suite`'s subtree is walked when one is given, so a targeted run does not
    read and rewrite all 205 files to execute one of them.

    Returns the number of files whose contents were actually changed. That count is a
    sanity signal worth watching: if it drops to zero after an upstream merge, either
    the suite stopped hardcoding the path (good, delete this driver) or the constant
    moved (bad, and this number is how you notice).
    """
    src = REPO_ROOT / "test" / "sql"
    require_deletable(sql_out, "--sql-root")
    if sql_out.exists():
        shutil.rmtree(sql_out)
    sql_out.mkdir(parents=True)
    if suite:
        src = src / suite
        sql_out = sql_out / suite
        sql_out.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            sys.exit(f"error: no such suite: {suite}")

    # Rewrite the root only at a real path boundary, so a longer sibling directory
    # (/var/infinity/test_data_2, /var/infinity/test_data.old, test_data+backup) is
    # left alone. Requiring the next character to be a separator, a quote, whitespace
    # or end-of-string is stricter than excluding word characters, which still matched
    # "." and "+".
    pattern = re.compile(re.escape(UPSTREAM_DATA_ROOT) + r"(?=[/'\"\s]|$)")
    rewritten = 0

    for dirpath, _dirnames, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        (sql_out / rel).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            s = Path(dirpath) / name
            d = sql_out / rel / name
            if not name.endswith(".slt"):
                shutil.copyfile(s, d)
                continue
            text = s.read_text(encoding="utf-8", errors="surrogateescape")
            new, n = pattern.subn(str(data_root), text)
            if n:
                rewritten += 1
            d.write_text(new, encoding="utf-8", errors="surrogateescape")

    log(f"materialised suite at {sql_out} ({rewritten} files had the data root rewritten)")
    return rewritten


def collect_files(sql_root: Path, suite: str | None, only_file: str | None) -> list[Path]:
    base = sql_root / suite if suite else sql_root
    if not base.exists():
        sys.exit(f"error: no such suite: {base.relative_to(sql_root.parent)}")
    files = sorted(p for p in base.rglob("*.slt"))
    if only_file:
        files = [p for p in files if p.name == only_file]
        if not files:
            sys.exit(f"error: no .slt file named {only_file} under {base}")
    return files


def run(binary: str, files: list[Path], sql_root: Path, timeout: int) -> list[Result]:
    results: list[Result] = []
    width = max((len(str(p.relative_to(sql_root))) for p in files), default=10)
    for i, path in enumerate(files, 1):
        rel = str(path.relative_to(sql_root))
        print(f"[{i}/{len(files)}] {rel:<{width}} ", end="", flush=True)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [binary, str(path)], capture_output=True, text=True, timeout=timeout,
            )
            elapsed = time.monotonic() - start
            ok = proc.returncode == 0
            # sqllogictest prints the actual expected-vs-actual diff on stdout and
            # only a "some test case failed" summary on stderr, so keep both and
            # lead with stdout. Preferring stderr yields a report you cannot act on.
            detail = "" if ok else "\n".join(
                part.rstrip() for part in (proc.stdout, proc.stderr) if part and part.strip()
            )[-6000:]
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            ok, detail = False, f"timed out after {timeout}s"
        print(f"{'OK  ' if ok else 'FAIL'} {elapsed:7.2f}s", flush=True)
        results.append(Result(rel, ok, elapsed, detail))
    return results


def summarise(results: list[Result], report: Path | None) -> int:
    failed = [r for r in results if not r.ok]
    passed = len(results) - len(failed)

    print()
    print("=" * 78)
    print(f"{passed}/{len(results)} passed, {len(failed)} failed")
    slowest = sorted(results, key=lambda r: -r.seconds)[:5]
    if slowest:
        print("slowest: " + ", ".join(f"{r.path} ({r.seconds:.1f}s)" for r in slowest))

    if failed:
        # Group by top-level suite so a systemic failure in one subsystem reads
        # as one problem rather than as thirty.
        print()
        print("failures by suite:")
        groups: dict[str, list[Result]] = {}
        for r in failed:
            groups.setdefault(r.path.split("/")[0], []).append(r)
        for suite in sorted(groups):
            print(f"  {suite}: {len(groups[suite])}")
            for r in groups[suite]:
                print(f"    {r.path}")

    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w") as fh:
            for r in results:
                fh.write(f"{'PASS' if r.ok else 'FAIL'}\t{r.seconds:.2f}\t{r.path}\n")
                if not r.ok:
                    fh.write("--- detail ---\n")
                    fh.write(r.detail.rstrip() + "\n")
                    fh.write("--- end ---\n")
        print(f"\nfull report: {report}")

    return 1 if failed else 0


def main() -> int:
    default_root = REPO_ROOT / "build" / "slt"

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", help="subtree of test/sql to run, e.g. dml or dml/delete")
    ap.add_argument("--file", dest="only_file", help="run only the .slt file with this basename")
    ap.add_argument("--data-root", type=Path, default=default_root / "test_data",
                    help="where COPY-able test data is staged (default: build/slt/test_data)")
    ap.add_argument("--sql-root", type=Path, default=default_root / "sql",
                    help="where the rewritten suite is materialised (default: build/slt/sql)")
    ap.add_argument("--sqllogictest", help="path to the sqllogictest binary")
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse previously generated data (much faster on reruns)")
    ap.add_argument("--regenerate", action="store_true",
                    help="force generators to rewrite data that already exists")
    ap.add_argument("--timeout", type=int, default=1800, help="per-file timeout in seconds")
    ap.add_argument("--report", type=Path, default=default_root / "report.txt",
                    help="write a full pass/fail report here")
    args = ap.parse_args()

    binary = find_sqllogictest(args.sqllogictest)
    data_root = args.data_root.resolve()
    sql_root = args.sql_root.resolve()

    log(f"sqllogictest: {binary}")

    if not args.skip_generate:
        generate_test_data(data_root, args.regenerate)
        copy_static_data(data_root)
    else:
        if not data_root.is_dir():
            sys.exit(f"error: --skip-generate given but {data_root} does not exist")
        log(f"reusing staged data in {data_root}")

    # Independent of generation: the export suites need this empty even on a
    # --skip-generate rerun, because the previous run is exactly what dirties it.
    reset_export_tmp(data_root)

    # Always re-materialise, scoped to what will run: cheap, and a source edit to a
    # .slt file is then picked up without having to remember a flag.
    rewritten = materialise_suite(data_root, sql_root, args.suite)
    if rewritten == 0:
        log("warning: no file needed the data root rewritten — if upstream stopped "
            f"hardcoding {UPSTREAM_DATA_ROOT}, this driver is obsolete; if the constant "
            "moved, the rewrite is now silently doing nothing")

    files = collect_files(sql_root, args.suite, args.only_file)
    log(f"running {len(files)} test file(s)")
    results = run(binary, files, sql_root, args.timeout)
    return summarise(results, args.report)


if __name__ == "__main__":
    sys.exit(main())
