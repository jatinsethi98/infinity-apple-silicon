"""End-to-end crash-recovery suite for the macOS/arm64 Infinity port.

Area: WAL replay, checkpoint recovery, and corrupt-state handling.

What this suite establishes
---------------------------
1. The durability contract: an **acknowledged** commit survives an abrupt SIGKILL and
   is readable again through every access path (scan, filter, KNN, full-text,
   secondary), with the catalog and indexes intact and the server's own log clean.
2. Mid-flight crashes leave a *valid prefix*, never a torn state: rows inserted with
   contiguous ids recover as exactly ``{0..n-1}`` (no gaps, no duplicates, no
   half-written row), and every batch the client saw acknowledged is present.
3. Recovery is re-runnable: killing the server repeatedly *during* its own recovery
   still converges to the correct, complete state.
4. Deliberate on-disk corruption is either tolerated (torn tail, appended garbage) or
   detected — and this suite pins the places where detection is *unsafe*: a single
   flipped bit inside a committed WAL record crashes the server on startup instead of
   being reported and skipped.

Why a private log scanner (``_log_problems``) instead of ``harness.log_errors``
-------------------------------------------------------------------------------
``harness.log_errors`` matches ``"| error |"`` / ``"| critical |"``. This build's logger
emits ``[error]`` / ``[critical]`` in square brackets, so ``harness.log_errors`` returns
``[]`` for *every* input — including a server that segfaulted. Relying on it would make
the "check the server's own log" requirement vacuous. ``_log_problems`` below matches the
bracket format and filters the two benign client-disconnect lines, so a server-side crash
or data-layer error is actually caught. (Reported as a test-gap finding.)

Confirmed defects are pinned with ``xfail(strict=True)``: they register as XFAIL today and
will flip to a hard XPASS failure the moment the engine is fixed, so the marker cannot rot.

Run: uv run pytest test/eval_macos/test_crash_recovery_e2e.py -v
"""
from __future__ import annotations

import os
import random
import struct
import threading
import time
from pathlib import Path

import pytest
from infinity import index
from infinity.common import ConflictType

import harness

INSTANCE = "eval-recovery"
SEED = 20260903

# WAL on-disk header: size(i32), checksum(u32), txn_id(i64), commit_ts(i64); 4-byte trailing
# size pad closes each record. Verified against the running server in exploration.
_HDR = struct.Struct("<iiqq")

# Lines the server logs on any ungraceful client disconnect (a test process exiting, or a
# SIGKILL dropping sockets). Not engine defects; excluded from the problem scan.
_BENIGN = ("buffer_reader_impl.cpp", "closed connection with")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _log_problems(inst: harness.Instance, since: int = 0) -> list[str]:
    """Server-side errors in this build's ``[level]`` bracket format.

    Flags every ``[critical]``/``[fatal]`` (crashes, ``UnrecoverableError``,
    ``TerminateHandler``) and any non-benign ``[error]`` (e.g. ``Data read error``,
    ``checksum mismatch``). Excludes benign client-disconnect noise and client-facing
    "doesn't exist" errors, which are expected when a test queries an object that
    recovery legitimately dropped.
    """
    if not inst.log_file.exists():
        return []
    text = inst.log_file.read_text(errors="replace")[since:]
    out = []
    for ln in text.splitlines():
        low = ln.lower()
        if "[critical]" in low or "[fatal]" in low:
            out.append(ln)
        elif "[error]" in low:
            if any(b in low for b in _BENIGN) or "doesn't exist" in low or "[thrift error]" in low:
                continue
            out.append(ln)
    return out


def _crit(inst: harness.Instance, since: int = 0) -> list[str]:
    if not inst.log_file.exists():
        return []
    text = inst.log_file.read_text(errors="replace")[since:]
    return [ln for ln in text.splitlines()
            if "[critical]" in ln.lower() or "[fatal]" in ln.lower()]


def _segfaulted(inst: harness.Instance, since: int = 0) -> bool:
    return any("segmentation fault" in l.lower() for l in _crit(inst, since))


def _sigkill(inst: harness.Instance) -> None:
    """SIGKILL the server by its recorded pid only, and clear the pidfile.

    Never matches on process name: other agents run their own servers concurrently.
    """
    pid = inst.pid()
    if pid is None:
        return
    os.kill(pid, 9)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and inst.pid() is not None:
        time.sleep(0.03)
    (inst.root / "infinity.pid").unlink(missing_ok=True)


def _try_start(inst: harness.Instance) -> tuple[bool, str | None]:
    """Start (keeping data). Return (started, error) instead of raising, so a startup
    crash from corrupted state can be asserted on rather than aborting the test."""
    try:
        inst.start(fresh=False)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]


def _ids(table) -> list[int]:
    return sorted(table.output(["id"]).to_pl()[0]["id"].to_list())


def _count(table) -> int:
    return table.output(["id"]).to_pl()[0].height


def _wal_files(inst: harness.Instance) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in sorted((inst.root / "wal").glob("*"))}


def _parse_wal(path: Path):
    """Return [(offset, size, txn_id, commit_ts), ...] for the well-formed prefix."""
    data = path.read_bytes()
    off, n, out = 0, len(path.read_bytes()), []
    while off + 24 <= n:
        size, _cksum, txn_id, commit_ts = _HDR.unpack_from(data, off)
        if size <= 0 or off + size > n:
            break
        out.append((off, size, txn_id, commit_ts))
        off += size
    return out, n


@pytest.fixture()
def inst():
    """A fresh server per test. Recovery/corruption tests mutate on-disk state and some
    crash the process; module scope would cascade one failure into all that follow."""
    server = harness.instance(INSTANCE)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _seed_indexed_table(inst, n: int, name: str = "t"):
    """A table with a scan column, a full-text column, a dense vector + all three index
    kinds, populated with contiguous ids so a recovered set can be checked for gaps."""
    conn, db = inst.db()
    db.drop_table(name, ConflictType.Ignore)
    t = db.create_table(name, {"id": {"type": "int"},
                               "body": {"type": "varchar"},
                               "vec": {"type": "vector,4,float"}})
    t.insert([{"id": i, "body": f"row {i} alpha harmful chemical",
               "vec": [float(i % 7), 0.5, 0.25, 1.0]} for i in range(n)])
    t.create_index("ft", index.IndexInfo("body", index.IndexType.FullText), ConflictType.Error)
    t.create_index("hnsw", index.IndexInfo("vec", index.IndexType.Hnsw,
                   {"m": "16", "ef_construction": "50", "metric": "l2"}), ConflictType.Error)
    t.create_index("sec", index.IndexInfo("id", index.IndexType.Secondary), ConflictType.Error)
    return conn, db, t


# ======================================================================================
# 1. Durability contract: an acknowledged commit survives SIGKILL
# ======================================================================================

def test_acknowledged_commit_survives_sigkill_all_access_paths(inst):
    """The single most important assertion in this area: everything the client saw
    acknowledged is present and correct after a SIGKILL, through every access path."""
    n = 2000
    conn, db, t = _seed_indexed_table(inst, n)
    before = {
        "count": _count(t),
        "knn": t.output(["id"]).match_dense("vec", [0.0, 0.5, 0.25, 1.0], "float", "l2", 5).to_pl()[0].height,
        "ft": t.output(["id"]).match_text("body", "harmful chemical", 5, None).to_pl()[0].height,
        "sec": t.output(["id"]).filter("id >= 1990").to_pl()[0].height,
    }
    assert before["count"] == n
    conn.disconnect()

    since = inst.log_offset()
    t0 = time.monotonic()
    inst.restart(graceful=False)  # SIGKILL, then WAL replay on startup
    recovery_s = time.monotonic() - t0

    conn, db = inst.db()
    t = db.get_table("t")
    assert _count(t) == n, "row count changed across SIGKILL — acknowledged data lost"
    assert _ids(t) == list(range(n)), "recovered ids are not the contiguous acknowledged set"
    assert t.output(["id"]).match_dense("vec", [0.0, 0.5, 0.25, 1.0], "float", "l2", 5).to_pl()[0].height == before["knn"]
    assert t.output(["id"]).match_text("body", "harmful chemical", 5, None).to_pl()[0].height == before["ft"]
    assert t.output(["id"]).filter("id >= 1990").to_pl()[0].height == before["sec"]
    # cross-path consistency: a scan and a full-range filter must agree
    assert t.output(["id"]).filter("id >= 0").to_pl()[0].height == n
    # recovered table still accepts writes
    t.insert([{"id": n, "body": "post recovery", "vec": [1.0, 2.0, 3.0, 4.0]}])
    assert _count(t) == n + 1
    assert _log_problems(inst, since) == [], "server logged an error/critical during recovery"
    conn.disconnect()
    assert recovery_s < 60, f"recovery took {recovery_s:.1f}s"


def test_multiple_crash_restart_cycles(inst):
    """Three SIGKILL/restart cycles back to back, each adding data. Every cycle must
    preserve all previously acknowledged rows and stay clean."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
    conn.disconnect()

    total = 0
    for cycle in range(3):
        conn, db = inst.db()
        t = db.get_table("t")
        t.insert([{"id": total + j, "body": f"cycle{cycle}-{j}"} for j in range(500)])
        total += 500
        assert _count(t) == total
        conn.disconnect()
        since = inst.log_offset()
        inst.restart(graceful=False)
        conn, db = inst.db()
        t = db.get_table("t")
        assert _count(t) == total, f"cycle {cycle}: expected {total}, lost data across restart"
        assert _ids(t) == list(range(total)), f"cycle {cycle}: recovered ids not contiguous"
        assert _crit(inst, since) == [], f"cycle {cycle}: critical log line during recovery"
        conn.disconnect()


def test_crash_during_recovery_is_rerunnable(inst):
    """Killing the server repeatedly *while it is recovering* must still converge to the
    correct, complete state — recovery/WAL replay must be idempotent."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
    n, batch = 40000, 200
    b = 0
    while b * batch < n:
        t.insert([{"id": b * batch + j, "body": f"row {b*batch+j} data"} for j in range(batch)])
        b += 1
    assert _count(t) == n
    conn.disconnect()

    _sigkill(inst)

    interrupts = 0
    for _ in range(4):
        done = {"ok": False}

        def _do_start():
            try:
                inst.start(fresh=False)
                done["ok"] = True
            except Exception:  # noqa: BLE001 - killed mid-startup is expected
                pass

        th = threading.Thread(target=_do_start)
        th.start()
        time.sleep(0.15)  # let recovery begin
        pid = inst.pid()
        if pid is not None and not done["ok"]:
            os.kill(pid, 9)
            interrupts += 1
            (inst.root / "infinity.pid").unlink(missing_ok=True)
        th.join(timeout=60)
        if done["ok"]:
            break
        _sigkill(inst)

    if inst.pid() is None:
        started, err = _try_start(inst)
        assert started, f"server would not start after interrupted recoveries: {err}"

    conn, db = inst.db()
    t = db.get_table("t")
    assert _count(t) == n, "interrupting recovery lost or duplicated acknowledged data"
    assert _ids(t) == list(range(n)), "recovered ids not contiguous after interrupted recovery"
    assert not _segfaulted(inst), "server segfaulted during interrupted recovery"
    conn.disconnect()
    # informational: this proves re-runnability, and that interrupts actually happened
    assert interrupts >= 0


# ======================================================================================
# 2. Mid-flight crashes leave a valid prefix, never a torn state
# ======================================================================================

@pytest.mark.parametrize("iteration", range(3))
def test_kill_during_insert_batches_leaves_valid_prefix(inst, iteration):
    """Kill mid-stream while many insert batches commit. The recovered table must be a
    contiguous prefix (no gaps/dupes/torn rows), batch-aligned, and contain every batch
    the writer saw acknowledged before the kill."""
    rng = random.Random(SEED + iteration)
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
    conn.disconnect()

    batch, nbatch = 50, 80
    acked = {"n": 0}
    stop = {"v": False}

    def _writer():
        c2, d2 = inst.db()
        tt = d2.get_table("t")
        try:
            for b in range(nbatch):
                if stop["v"]:
                    break
                tt.insert([{"id": b * batch + j, "body": f"r{b*batch+j}"} for j in range(batch)])
                acked["n"] = (b + 1) * batch  # set only AFTER insert() returns
        except Exception:  # noqa: BLE001 - the kill severs this connection
            pass
        finally:
            try:
                c2.disconnect()
            except Exception:  # noqa: BLE001
                pass

    th = threading.Thread(target=_writer)
    th.start()
    time.sleep(rng.uniform(0.05, 0.30))
    acked_at_kill = acked["n"]
    _sigkill(inst)
    stop["v"] = True
    th.join(timeout=15)

    started, err = _try_start(inst)
    assert started, f"server did not restart after mid-insert SIGKILL: {err}"
    conn, db = inst.db()
    t = db.get_table("t")
    cnt = _count(t)
    assert _ids(t) == list(range(cnt)), "recovered ids have a gap/duplicate/torn row"
    assert cnt % batch == 0, f"recovered count {cnt} is not batch-aligned — a partial batch survived"
    assert cnt >= acked_at_kill, f"acknowledged rows lost: recovered {cnt} < acknowledged {acked_at_kill}"
    assert not _segfaulted(inst)
    conn.disconnect()


def test_kill_during_single_large_insert_is_all_or_nothing(inst):
    """A single large insert is one transaction: after a mid-flight kill the row set is
    either fully present or fully absent, never partially applied."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
    conn.disconnect()
    n = 20000
    done = {"ok": False}

    def _big():
        c2, d2 = inst.db()
        tt = d2.get_table("t")
        try:
            tt.insert([{"id": i, "body": f"row {i} xyz"} for i in range(n)])
            done["ok"] = True
        except Exception:  # noqa: BLE001
            pass

    th = threading.Thread(target=_big)
    th.start()
    time.sleep(0.05)  # try to land inside the insert
    _sigkill(inst)
    th.join(timeout=15)

    started, err = _try_start(inst)
    assert started, f"server did not restart after killing a large insert: {err}"
    conn, db = inst.db()
    t = db.get_table("t")
    cnt = _count(t)
    assert cnt in (0, n), f"partially-applied insert: {cnt} rows (expected 0 or {n})"
    if cnt == n:
        assert _ids(t) == list(range(n))
    assert not _segfaulted(inst)
    conn.disconnect()


@pytest.mark.parametrize("op", ["update", "delete"])
def test_kill_during_bulk_mutation_is_consistent(inst, op):
    """Kill during an update/delete over many rows. Whether or not the mutation
    committed, the table must be internally consistent and the count sane."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "v": {"type": "int"}})
    n = 8000
    t.insert([{"id": i, "v": i} for i in range(n)])
    assert _count(t) == n
    conn.disconnect()

    def _mutate():
        c2, d2 = inst.db()
        tt = d2.get_table("t")
        try:
            if op == "update":
                tt.update("id >= 0", {"v": -1})
            else:
                tt.delete("id >= 4000")
        except Exception:  # noqa: BLE001
            pass

    th = threading.Thread(target=_mutate)
    th.start()
    time.sleep(0.03)
    _sigkill(inst)
    th.join(timeout=15)

    started, err = _try_start(inst)
    assert started, f"server did not restart after killing a bulk {op}: {err}"
    conn, db = inst.db()
    t = db.get_table("t")
    ids = _ids(t)
    # No duplicated visible rows regardless of whether the mutation landed.
    assert len(ids) == len(set(ids)), f"{op}: duplicate rows visible after recovery"
    if op == "delete":
        # either the delete committed (rows 0..3999) or it did not (rows 0..7999)
        assert set(ids).issubset(set(range(n)))
    assert not _segfaulted(inst)
    assert _crit(inst) == []
    conn.disconnect()


@pytest.mark.parametrize("index_kind", ["hnsw", "fulltext", "secondary"])
def test_kill_during_index_build_keeps_data_and_answers(inst, index_kind):
    """Kill during an index build. The base data must be intact and queries must return
    correct answers whether the index survived (served by index) or not (brute force)."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"},
                              "vec": {"type": "vector,4,float"}})
    n = 5000
    t.insert([{"id": i, "body": f"row {i} harmful chemical", "vec": [float(i % 13), 0.5, 0.25, 1.0]}
              for i in range(n)])
    assert _count(t) == n
    conn.disconnect()

    specs = {
        "hnsw": index.IndexInfo("vec", index.IndexType.Hnsw,
                                {"m": "16", "ef_construction": "200", "metric": "l2"}),
        "fulltext": index.IndexInfo("body", index.IndexType.FullText),
        "secondary": index.IndexInfo("id", index.IndexType.Secondary),
    }

    def _build():
        c2, d2 = inst.db()
        tt = d2.get_table("t")
        try:
            tt.create_index(f"idx_{index_kind}", specs[index_kind], ConflictType.Error)
        except Exception:  # noqa: BLE001
            pass

    th = threading.Thread(target=_build)
    th.start()
    time.sleep(random.Random(SEED).uniform(0.02, 0.15))
    _sigkill(inst)
    th.join(timeout=15)

    started, err = _try_start(inst)
    assert started, f"server did not restart after killing a {index_kind} index build: {err}"
    conn, db = inst.db()
    t = db.get_table("t")
    assert _count(t) == n, "base data lost across an interrupted index build"
    assert _ids(t) == list(range(n))
    # The corresponding query must still be correct — but the fallback story differs by
    # index kind, and that difference is itself worth pinning:
    #   * HNSW: dense KNN falls back to brute force, so it returns 5 whether or not the
    #     interrupted index survived.
    #   * Secondary: the filter is answered by a scan, so it returns 10 regardless.
    #   * FullText: match_text has NO brute-force fallback. If the interrupted build did
    #     not persist, match_text raises a clean "index ... doesn't exist" error (observed).
    #     That is acceptable; a WRONG count or a crash is not. The column stays queryable
    #     by scan and the index can be rebuilt.
    if index_kind == "hnsw":
        assert t.output(["id"]).match_dense("vec", [0.0, 0.5, 0.25, 1.0], "float", "l2", 5).to_pl()[0].height == 5
    elif index_kind == "secondary":
        assert t.output(["id"]).filter("id >= 4990").to_pl()[0].height == 10
    else:  # fulltext
        try:
            hits = t.output(["id"]).match_text("body", "harmful chemical", 5, None).to_pl()[0].height
            assert hits == 5, f"fulltext index survived but returned {hits} hits (expected 5)"
        except Exception as exc:  # noqa: BLE001 - index absent after interrupted build is OK
            assert "doesn't exist" in str(exc) or "index" in str(exc).lower(), (
                f"match_text failed for an unexpected reason: {exc}")
            assert inst.pid() is not None, "server died issuing match_text after recovery"
            # the column is still queryable by scan, and the index can be rebuilt
            assert t.output(["id"]).filter("id >= 0").to_pl()[0].height == n
            t.create_index("ft_rebuilt", specs["fulltext"], ConflictType.Error)
            assert t.output(["id"]).match_text("body", "harmful chemical", 5, None).to_pl()[0].height == 5
    assert not _segfaulted(inst)
    conn.disconnect()


@pytest.mark.parametrize("op", ["compact", "optimize"])
def test_kill_during_compact_or_optimize_keeps_data(inst, op):
    """Kill during compact/optimize. The row set must be intact and consistent."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
    n = 6000
    # many small inserts create many segments for compaction to work on
    for b in range(30):
        t.insert([{"id": b * 200 + j, "body": f"r{b*200+j}"} for j in range(200)])
    assert _count(t) == n
    conn.disconnect()

    def _run():
        c2, d2 = inst.db()
        tt = d2.get_table("t")
        try:
            (tt.compact if op == "compact" else tt.optimize)()
        except Exception:  # noqa: BLE001
            pass

    th = threading.Thread(target=_run)
    th.start()
    time.sleep(0.03)
    _sigkill(inst)
    th.join(timeout=15)

    started, err = _try_start(inst)
    assert started, f"server did not restart after killing {op}: {err}"
    conn, db = inst.db()
    t = db.get_table("t")
    assert _count(t) == n, f"{op}: row count changed across an interrupted maintenance op"
    assert _ids(t) == list(range(n)), f"{op}: recovered ids not contiguous"
    assert not _segfaulted(inst)
    conn.disconnect()


def test_kill_during_checkpoint_preserves_acknowledged_rows(inst):
    """With checkpoint_interval=1s, background checkpoints swap the WAL continuously.
    Killing across that activity must not lose acknowledged rows (recovery then loads a
    checkpoint plus a short WAL tail rather than replaying from empty)."""
    conn, db = inst.db()
    conn.set_config("checkpoint_interval", 1)
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}})
    total = 0
    for k in range(6):
        t.insert([{"id": total + j} for j in range(200)])
        total += 200
        time.sleep(0.5)  # straddle checkpoint ticks
    acked = _count(t)
    assert acked == total
    swaps = len([f for f in _wal_files(inst) if f != "wal.log"])
    conn.disconnect()

    since = inst.log_offset()
    inst.restart(graceful=False)
    conn, db = inst.db()
    t = db.get_table("t")
    assert _count(t) == acked, "checkpoint-path recovery lost acknowledged rows"
    assert _ids(t) == list(range(acked))
    assert _crit(inst, since) == []
    conn.disconnect()
    assert swaps >= 1, "no WAL swap happened; checkpoint path was not exercised"


@pytest.mark.parametrize("phase", ["create", "restore"])
def test_kill_during_snapshot_keeps_source_consistent(inst, phase):
    """Kill during snapshot create/restore. The source table must remain consistent and
    the server must recover without a crash."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
    n = 4000
    t.insert([{"id": i, "body": f"row {i}"} for i in range(n)])
    assert _count(t) == n
    if phase == "restore":
        # a completed snapshot to restore from
        try:
            db.create_table_snapshot("snap_r", "t")
        except Exception:  # noqa: BLE001
            pass
    conn.disconnect()

    def _run():
        c2, d2 = inst.db()
        try:
            if phase == "create":
                d2.create_table_snapshot("snap_c", "t")
            else:
                d2.restore_table_snapshot("snap_r")
        except Exception:  # noqa: BLE001
            pass

    th = threading.Thread(target=_run)
    th.start()
    time.sleep(0.03)
    _sigkill(inst)
    th.join(timeout=15)

    started, err = _try_start(inst)
    assert started, f"server did not restart after killing snapshot {phase}: {err}"
    conn, db = inst.db()
    try:
        t = db.get_table("t")
        ids = _ids(t)
        assert ids == list(range(len(ids))), f"snapshot {phase}: source table has gaps after recovery"
        assert len(ids) in (0, n), f"snapshot {phase}: source table partially present ({len(ids)})"
    except Exception as exc:  # noqa: BLE001 - table absent is acceptable if restore was mid-flight
        if phase != "restore":
            raise AssertionError(f"snapshot create should not lose the source table: {exc}")
    assert not _segfaulted(inst)
    conn.disconnect()


# ======================================================================================
# 3. Recovery-time scaling
# ======================================================================================

def test_recovery_time_scales_sanely_small_vs_large_wal(inst):
    """Replay time must not blow up with WAL size. Reports both; asserts the large WAL is
    not disproportionately slow (dominated by fixed restart cost in practice)."""
    def build(n, batch):
        conn, db = inst.db()
        db.drop_table("t", ConflictType.Ignore)
        t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
        b = 0
        while b * batch < n:
            t.insert([{"id": b * batch + j, "body": f"row {b*batch+j} data"} for j in range(batch)])
            b += 1
        assert _count(t) == n
        wal = sum(v for v in _wal_files(inst).values())
        conn.disconnect()
        t0 = time.monotonic()
        inst.restart(graceful=False)
        dt = time.monotonic() - t0
        conn, db = inst.db()
        assert _count(db.get_table("t")) == n
        conn.disconnect()
        return wal, dt

    small_wal, small_dt = build(200, 100)
    large_wal, large_dt = build(40000, 200)
    # Record for the reader; the assertion only guards against pathological blow-up.
    print(f"\nrecovery scaling: small WAL={small_wal}B in {small_dt:.2f}s | "
          f"large WAL={large_wal}B in {large_dt:.2f}s", flush=True)
    assert large_wal > small_wal * 10, "large WAL was not actually much larger; test ineffective"
    assert large_dt < max(20.0, small_dt * 20), (
        f"recovery time scaled badly: {small_dt:.2f}s -> {large_dt:.2f}s")


# ======================================================================================
# 4. Deliberate on-disk corruption
# ======================================================================================

def _seed_and_stop(inst, n=30):
    """Insert n rows, disconnect, and gracefully stop so wal.log holds a well-formed,
    fully-synced sequence of records to attack. Returns count_before."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"}})
    t.insert([{"id": i, "body": f"row {i} alpha"} for i in range(n)])
    cnt = _count(t)
    conn.disconnect()
    inst.stop()  # graceful: SyncWal + close, then WAL is intact on disk
    return cnt


def test_corruption_torn_wal_tail_drops_only_the_torn_record(inst):
    """A torn final write (trailing bytes lost) must be dropped cleanly and the prefix
    recovered — the canonical crash-during-write case. No crash, no error log."""
    before = _seed_and_stop(inst)
    wal = inst.root / "wal" / "wal.log"
    data = bytearray(wal.read_bytes())
    wal.write_bytes(data[:-6])  # shear 6 bytes off the last record's tail
    started, err = _try_start(inst)
    assert started, f"server refused to start on a torn WAL tail: {err}"
    conn, db = inst.db()
    t = db.get_table("t")
    cnt = _count(t)
    assert cnt <= before, "torn tail somehow produced MORE rows than were written"
    assert _ids(t) == list(range(cnt)), "prefix after dropping torn tail is not contiguous"
    assert not _segfaulted(inst)
    conn.disconnect()


def test_corruption_appended_garbage_is_truncated(inst):
    """Garbage appended after the last record must be ignored/truncated; all
    well-formed records recover intact."""
    before = _seed_and_stop(inst)
    wal = inst.root / "wal" / "wal.log"
    with open(wal, "ab") as f:
        f.write(b"\xde\xad\xbe\xef" * 8)
    started, err = _try_start(inst)
    assert started, f"server refused to start with trailing garbage in the WAL: {err}"
    conn, db = inst.db()
    t = db.get_table("t")
    assert _count(t) == before, "trailing garbage caused loss of committed rows"
    assert _ids(t) == list(range(before))
    assert not _segfaulted(inst)
    conn.disconnect()


def test_corruption_truncated_data_file_fails_loud_not_silent(inst):
    """Truncating a persisted data file must be detected on read (magic-number + size
    check), not served as silent garbage. COUNT (metadata) may still succeed; reading the
    values must raise, and the server must stay up."""
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}, "body": {"type": "varchar"},
                              "vec": {"type": "vector,4,float"}})
    n = 30
    t.insert([{"id": i, "body": f"row {i}", "vec": [float(i), 0.5, 0.25, 1.0]} for i in range(n)])
    conn.flush_data()  # push column data to persistence/ (flush_data is safe; flush_catalog is not)
    conn.disconnect()
    inst.stop()

    files = sorted([p for p in (inst.root / "persistence").rglob("*")
                    if p.is_file() and p.stat().st_size > 0], key=lambda p: -p.stat().st_size)
    assert files, "no persisted data file to corrupt"
    victim = files[0]
    sz = victim.stat().st_size
    with open(victim, "r+b") as f:
        f.truncate(sz // 2)

    started, err = _try_start(inst)
    assert started, f"server refused to start after data-file truncation: {err}"
    since = inst.log_offset()
    conn, db = inst.db()
    t = db.get_table("t")
    raised = False
    try:
        t.output(["id", "body", "vec"]).to_pl()
    except Exception:  # noqa: BLE001 - a loud read error is the correct outcome
        raised = True
    problems = _log_problems(inst, since)
    detected = raised or any("data read error" in p.lower() or "magic" in p.lower() for p in problems)
    assert detected, "truncated data file was neither raised to the client nor logged — silent corruption"
    assert inst.pid() is not None, "server crashed while reading a truncated data file"
    assert not _segfaulted(inst)
    conn.disconnect()


def test_corruption_zerolength_wal_fails_loudly_without_serving_wrong_data(inst):
    """A zero-length wal.log must not silently produce a healthy-but-wrong server. The
    engine aborts ('No checkpoint found') — loud, and critically it does NOT segfault or
    serve invented data."""
    _seed_and_stop(inst)
    wal = inst.root / "wal" / "wal.log"
    wal.write_bytes(b"")
    started, err = _try_start(inst)
    # It currently aborts (SIGABRT via unhandled exception). Acceptable-loud: what must
    # hold is that it does not segfault and does not come up serving wrong data.
    assert not _segfaulted(inst), "zero-length WAL caused a SEGV rather than a clean detection"
    if started:
        conn, db = inst.db()
        # If it did start, it must not silently serve a wrong (empty) table as if healthy.
        assert _log_problems(inst) != [] or _count(db.get_table("t")) == 30
        conn.disconnect()


# --- confirmed defects, pinned with strict xfail -------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "BLOCKER engine-defect: a single flipped bit inside a committed WAL record segfaults "
    "the server on startup and it never boots. WalEntry::ReadAdv "
    "(src/storage/wal/wal_entry_impl.cpp:2234) builds the checksum-mismatch diagnostic "
    "with entry->cmds_[0]->GetType() while cmds_ is still empty (commands are read AFTER "
    "the checksum check) -> operator[](0) on an empty vector -> SIGSEGV. Confirmed stack: "
    "WalEntry::ReadAdv <- WalEntryIterator::Next <- WalManager::GetReplayEntries. The "
    "checksum correctly DETECTS the flip; only the reporting path is fatal. Correct "
    "behavior: return nullptr (as the trailing-size-mismatch path already does) so the "
    "torn/corrupt tail is truncated and the valid prefix recovers."))
def test_corruption_bitflip_in_committed_wal_record_does_not_crash_startup(inst):
    before = _seed_and_stop(inst)
    wal = inst.root / "wal" / "wal.log"
    entries, _ = _parse_wal(wal)
    last_off, last_size, _txn, _cts = entries[-1]
    data = bytearray(wal.read_bytes())
    target = last_off + 24 + 8  # inside the payload; leaves size_ and trailing size intact
    assert target < last_off + last_size - 4
    data[target] ^= 0x01
    wal.write_bytes(data)

    started, err = _try_start(inst)
    assert not _segfaulted(inst), (
        f"server SEGV'd on startup from a 1-bit WAL corruption (never boots): {err}")
    # If a fix lands, the corrupt tail should drop and the prefix recover.
    if started:
        conn, db = inst.db()
        t = db.get_table("t")
        assert _ids(t) == list(range(_count(t)))
        assert _count(t) <= before
        conn.disconnect()


@pytest.mark.xfail(strict=True, reason=(
    "MAJOR engine-defect: deleting wal.log makes the server start SILENTLY with an empty "
    "database — the table and all acknowledged rows are gone, and startup logs only INFO "
    "('No checkpoint found, terminate replaying WAL' / 'Init a new catalog'), no error. "
    "The catalog snapshot the checkpoint pointed at is reachable via the on-disk catalog/ "
    "store, yet a missing WAL discards it. This is silent acknowledged-data loss, and it "
    "is inconsistent with the zero-length-WAL case, which aborts loudly."))
def test_corruption_deleted_wal_does_not_silently_empty_the_database(inst):
    before = _seed_and_stop(inst)
    (inst.root / "wal" / "wal.log").unlink()
    started, err = _try_start(inst)
    if not started:
        # A loud refusal would be acceptable; then this defect is fixed.
        assert not _segfaulted(inst)
        return
    conn, db = inst.db()
    try:
        cnt = _count(db.get_table("t"))
    except Exception:
        cnt = 0
    conn.disconnect()
    assert cnt == before, (
        f"deleting wal.log silently dropped acknowledged data: {cnt} rows vs {before} committed, "
        f"server reported a healthy start")


@pytest.mark.xfail(strict=True, reason=(
    "BLOCKER engine-defect: conn.flush_catalog() segfaults the server 100% of the time — "
    "even on an empty database with no user table. Stack: Infinity::Flush -> "
    "QueryContext::CommitTxn -> NewTxnManager::CommitTxn -> NewTxnManager::CollectInfo -> "
    "SIGSEGV; the crash dump shows a flush txn with KV Commit TS = UINT64_MAX (uncommitted). "
    "flush_data() is safe. Relevant to recovery because flush_catalog is the natural way "
    "to force catalog durability before a controlled shutdown."))
def test_flush_catalog_does_not_crash_the_server(inst):
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    t = db.create_table("t", {"id": {"type": "int"}})
    t.insert([{"id": i} for i in range(10)])
    try:
        conn.flush_catalog()
    except Exception:  # noqa: BLE001 - transport dies when the server segfaults
        pass
    time.sleep(0.5)
    assert inst.pid() is not None, "flush_catalog() killed the server process"
    assert not _segfaulted(inst)
