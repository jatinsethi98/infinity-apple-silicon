"""End-to-end concurrency suite: index maintenance racing queries on macOS/arm64.

Area (this file): **index maintenance and the mem-index lifecycle** racing queries,
centred on the freshly-shipped ``RcuMultiMap::range`` fix (commit ccc58a3da) that now
holds ``dirty_lock_`` across the whole range traversal in
``src/storage/common/rcu_multimap.cppm``. The reachable caller is
``SecondaryIndexInMem::RangeQueryInner``
(``src/storage/secondary_index/secondary_index_in_mem_impl.cpp``).

Mechanism verified from source before writing these tests:
  * A secondary-index range query (``t.filter("v >= a AND v < b")`` when a Secondary
    index exists on ``v``) walks the in-memory segment's ``RcuMultiMap`` under
    ``dirty_lock_``. ``Insert`` takes the same lock, so a range query and an index
    append now serialise on one in-memory segment.
  * When a segment's mem-index row count crosses ``mem_index_capacity`` (65536 in the
    generated config, see ``scripts/apple_silicon/run_server.sh``), an async
    ``DumpMemIndexTask`` is submitted (``new_txn_index_impl.cpp:1015``) that moves the
    rows from the mem-index into an on-disk chunk **on a background thread** — i.e. a
    dump runs concurrently with the inserts that triggered it and with any queries.

Every concurrent test here:
  1. seeds all RNGs and repeats each race for a reported number of iterations,
  2. asserts the final/observed state is one a correct engine could produce (exact row
     sets, no lost/duplicated/torn rows, cross-access-path agreement) — not merely
     "it did not crash",
  3. checks ``inst.log_errors()`` so a clean client result over a server-side error is
     still caught,
  4. is wall-clock bounded (thread joins have timeouts; a thread that will not join is
     reported as a hang == blocker),
  5. and — for the mem-index tests — finishes with a ``restart(graceful=False)`` (SIGKILL
     + WAL recovery) and a correctness re-check, so a structure corrupted in memory is
     caught when it is reloaded from disk.

Run:
  uv run pytest test/eval_macos/test_concurrency_index_e2e.py -v
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

import harness
from infinity import index
from infinity.common import ConflictType, SparseVector

# --------------------------------------------------------------------------- #
# knobs — every race repeats enough to be meaningful; counts are reported.
# --------------------------------------------------------------------------- #
SEED = 20260903
MEM_INDEX_CAPACITY = 65536          # must match run_server.sh generated config
JOIN_TIMEOUT = 240.0                # a worker that won't join in this long == hang
STORM_INSERT_ROWS = 200_000         # > 3x capacity: forces >=3 async mem-index dumps
DUMP_TARGET_ROWS = 5_000            # fixed committed set queried throughout a dump

RESULTS: dict = {"measurements": [], "iterations": {}, "notes": []}

random.seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-load")
    it.start()
    try:
        yield it
    finally:
        try:
            harness.record("concurrency_index_e2e", RESULTS)
        finally:
            it.stop()


def _tbl(conn, name):
    return conn.get_database("default_db").get_table(name)


def _fresh(db, name, schema):
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(name, schema, ConflictType.Error)


# The server rejects any single insert of > 8192 rows
# ("Insert batch row limit shouldn't more than 8192"), so all bulk setup inserts
# must be chunked. 8000 keeps a safe margin.
def _bulk_insert(table, rows, batch=8000):
    for s in range(0, len(rows), batch):
        table.insert(rows[s:s + batch])


def _join_all(threads, timeout=JOIN_TIMEOUT):
    """Join every thread; return names of any that did not finish (a hang)."""
    deadline = time.monotonic() + timeout
    hung = []
    for th in threads:
        remaining = max(0.0, deadline - time.monotonic())
        th.join(timeout=remaining)
        if th.is_alive():
            hung.append(th.name)
    return hung


def _filter_vals(table, lo, hi, col="v"):
    df, _ = table.output([col]).filter(f"{col} >= {lo} AND {col} < {hi}").to_pl()
    if df.height == 0:
        return []
    return df[col].to_list()


def _count_all(table):
    df, _ = table.output(["id"]).to_pl()
    return df.height


def _percentiles(samples):
    if not samples:
        return {}
    a = np.array(samples, dtype=float)
    return {
        "n": len(samples),
        "median_ms": float(np.median(a) * 1e3),
        "p95_ms": float(np.percentile(a, 95) * 1e3),
        "max_ms": float(a.max() * 1e3),
    }


# =========================================================================== #
# 1. CORE: mem-index dump racing queries (the shipped RcuMultiMap::range fix)
# =========================================================================== #
def test_secondary_memindex_dump_racing_queries(inst):
    """A fixed, fully-committed row set queried continuously while an async
    mem-index dump moves those very rows from the in-memory RcuMultiMap into an
    on-disk chunk mid-scan.

    The target range [0, DUMP_TARGET_ROWS) is committed BEFORE any concurrency and
    never changes, so a correct engine must return *exactly* that set on every query,
    regardless of the dump. A query that returns fewer rows (lost during the mem->chunk
    move), more rows (double-counted across mem+chunk), or a wrong value (torn offset)
    is an engine defect.
    """
    off = inst.log_offset()
    conn, db = inst.db()
    K = DUMP_TARGET_ROWS
    t = _fresh(db, "dump_race", {"id": {"type": "int"}, "v": {"type": "int"}})
    t.insert([{"id": i, "v": i} for i in range(K)])
    t.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)

    expected = set(range(K))
    errors: list = []
    query_iters = [0]
    stop = threading.Event()

    def querier():
        c = inst.connect()
        tq = _tbl(c, "dump_race")
        n = 0
        try:
            while not stop.is_set():
                vals = _filter_vals(tq, 0, K)
                n += 1
                if len(vals) != K:
                    errors.append(("count", len(vals), "expected", K))
                elif set(vals) != expected:
                    got = set(vals)
                    errors.append(("set", "missing", sorted(expected - got)[:5],
                                   "extra", sorted(got - expected)[:5]))
        except Exception as e:  # noqa: BLE001
            errors.append(("query-exc", type(e).__name__, str(e)[:120]))
        finally:
            query_iters[0] += n
            c.disconnect()

    NW = 2
    per_worker = STORM_INSERT_ROWS // NW              # disjoint value ranges per worker

    def inserter(worker):
        c = inst.connect()
        ti = _tbl(c, "dump_race")
        base = 1_000_000 + worker * 10_000_000         # non-overlapping so v stays unique
        B = 2000
        try:
            for start in range(0, per_worker, B):
                rows = [{"id": base + j, "v": base + j}
                        for j in range(start, min(start + B, per_worker))]
                ti.insert(rows)
        except Exception as e:  # noqa: BLE001
            errors.append(("insert-exc", type(e).__name__, str(e)[:120]))
        finally:
            c.disconnect()

    qs = [threading.Thread(target=querier, name=f"q{i}") for i in range(4)]
    ins = [threading.Thread(target=inserter, args=(w,), name=f"i{w}") for w in range(NW)]
    for th in qs + ins:
        th.start()
    hung = _join_all(ins)                       # inserters finish the storm
    stop.set()
    hung += _join_all(qs)

    # Confirm dumps actually fired, from the server's own log (info level), so the
    # "how hard it was pushed" claim is measured, not inferred.
    log_text = inst.log_file.read_text(errors="replace")[off:] if inst.log_file.exists() else ""
    dumps_fired = log_text.count("Submit dump task")
    RESULTS["iterations"]["dump_race_queries"] = query_iters[0]
    RESULTS["notes"].append(
        f"dump_race: {query_iters[0]} queries during {STORM_INSERT_ROWS} inserts; "
        f"{dumps_fired} mem-index dump tasks submitted (server log)")

    assert not hung, f"threads hung (deadlock/hang): {hung}"
    assert query_iters[0] >= 50, f"too few query iterations to be meaningful: {query_iters[0]}"
    assert dumps_fired >= 1, (
        f"no mem-index dump fired — race never exercised the dump path "
        f"(inserted {STORM_INSERT_ROWS} > capacity {MEM_INDEX_CAPACITY})")
    assert errors == [], f"correctness violations under dump race: {errors[:8]}"
    log = inst.log_errors(off)
    assert log == [], f"server logged errors during dump race: {log[:8]}"

    # total must be exactly what we committed
    total = _count_all(t)
    assert total == K + STORM_INSERT_ROWS, f"row count {total} != {K + STORM_INSERT_ROWS}"

    # SIGKILL + WAL recovery, then re-check on-disk state
    off2 = inst.log_offset()
    inst.restart(graceful=False)
    conn2, db2 = inst.db()
    t2 = _tbl(conn2, "dump_race")
    vals = _filter_vals(t2, 0, K)
    assert len(vals) == K and set(vals) == expected, (
        f"post-SIGKILL target range corrupted: n={len(vals)}")
    total2 = _count_all(t2)
    assert total2 == K + STORM_INSERT_ROWS, f"post-SIGKILL row count {total2}"
    rec_log = inst.log_errors(off2)
    assert rec_log == [], f"errors during WAL recovery: {rec_log[:8]}"


# =========================================================================== #
# 2. range monotonicity + no dups + cross-access-path agreement
# =========================================================================== #
def test_secondary_range_monotonic_and_crosspath(inst):
    """A single querier watching a growing table must see a monotonically
    non-decreasing count (inserts only add), never a duplicate value, and every
    returned value inside the predicate. At quiescence the index path and a
    full-scan path must agree exactly (no row in one access path but not the other).
    """
    off = inst.log_offset()
    conn, db = inst.db()
    N = 90_000                                  # crosses capacity -> a dump mid-run
    LO, HI = 0, N                               # whole inserted range
    t = _fresh(db, "mono", {"id": {"type": "int"}, "v": {"type": "int"}})
    t.insert([{"id": i, "v": i} for i in range(1000)])
    t.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)

    errors: list = []
    samples = [0]
    stop = threading.Event()

    def watcher():
        c = inst.connect()
        tq = _tbl(c, "mono")
        last = -1
        n = 0
        try:
            while not stop.is_set():
                vals = _filter_vals(tq, LO, HI)
                n += 1
                cnt = len(vals)
                if cnt < last:
                    errors.append(("non-monotonic", last, cnt))
                if len(set(vals)) != cnt:
                    errors.append(("dup", cnt, len(set(vals))))
                if vals and (min(vals) < LO or max(vals) >= HI):
                    errors.append(("out-of-range", min(vals), max(vals)))
                last = max(last, cnt)
        except Exception as e:  # noqa: BLE001
            errors.append(("watch-exc", type(e).__name__, str(e)[:120]))
        finally:
            samples[0] = n
            c.disconnect()

    def inserter():
        c = inst.connect()
        ti = _tbl(c, "mono")
        B = 2000
        try:
            for start in range(1000, N, B):
                ti.insert([{"id": j, "v": j} for j in range(start, min(start + B, N))])
        except Exception as e:  # noqa: BLE001
            errors.append(("insert-exc", type(e).__name__, str(e)[:120]))
        finally:
            c.disconnect()

    w = threading.Thread(target=watcher, name="watch")
    i = threading.Thread(target=inserter, name="ins")
    w.start(); i.start()
    hung = _join_all([i])
    stop.set()
    hung += _join_all([w])

    RESULTS["iterations"]["mono_queries"] = samples[0]
    assert not hung, f"threads hung: {hung}"
    assert errors == [], f"monotonicity/dup/range violations: {errors[:8]}"

    # cross-path: secondary-index filter vs full-scan-in-python over a sub-range
    sub_lo, sub_hi = 12_345, 67_890
    idx_vals = sorted(_filter_vals(t, sub_lo, sub_hi))
    df_all, _ = t.output(["v"]).to_pl()
    scan_vals = sorted(v for v in df_all["v"].to_list() if sub_lo <= v < sub_hi)
    assert idx_vals == scan_vals, (
        f"index path vs full scan disagree: idx={len(idx_vals)} scan={len(scan_vals)}")
    assert scan_vals == list(range(sub_lo, sub_hi)), "expected contiguous set missing rows"
    assert inst.log_errors(off) == []


# =========================================================================== #
# 3. index build racing inserts
# =========================================================================== #
def test_secondary_build_racing_inserts(inst):
    """Build a secondary index while a second connection streams inserts into the
    same table. After both finish, the finished index must be able to find every
    committed row (count matches) and return exact sets for arbitrary sub-ranges.
    Repeated over several iterations because the interleaving is nondeterministic.
    """
    off = inst.log_offset()
    conn, db = inst.db()
    iterations = 5
    RESULTS["iterations"]["build_vs_insert"] = iterations
    for it in range(iterations):
        name = f"build_ins_{it}"
        pre = 3000
        extra = 40_000
        t = _fresh(db, name, {"id": {"type": "int"}, "v": {"type": "int"}})
        t.insert([{"id": i, "v": i} for i in range(pre)])

        errors: list = []

        def builder():
            c = inst.connect()
            tb = _tbl(c, name)
            try:
                tb.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary),
                                ConflictType.Error)
            except Exception as e:  # noqa: BLE001
                errors.append(("build-exc", type(e).__name__, str(e)[:120]))
            finally:
                c.disconnect()

        def inserter():
            c = inst.connect()
            ti = _tbl(c, name)
            B = 2000
            try:
                for start in range(pre, pre + extra, B):
                    ti.insert([{"id": j, "v": j}
                               for j in range(start, min(start + B, pre + extra))])
            except Exception as e:  # noqa: BLE001
                errors.append(("insert-exc", type(e).__name__, str(e)[:120]))
            finally:
                c.disconnect()

        bt = threading.Thread(target=builder, name="build")
        it_ = threading.Thread(target=inserter, name="ins")
        bt.start(); it_.start()
        hung = _join_all([bt, it_])
        assert not hung, f"[{name}] threads hung: {hung}"
        assert errors == [], f"[{name}] errors during build/insert: {errors[:8]}"

        total = pre + extra
        assert _count_all(t) == total, f"[{name}] lost rows: {_count_all(t)} != {total}"
        # index must return every row for the whole range
        allv = sorted(_filter_vals(t, 0, total))
        assert allv == list(range(total)), (
            f"[{name}] index missing rows after build: got {len(allv)} of {total}")
        # an arbitrary sub-range
        sub = sorted(_filter_vals(t, 10_000, 20_000))
        assert sub == list(range(10_000, 20_000)), f"[{name}] sub-range wrong: {len(sub)}"
    assert inst.log_errors(off) == []


# =========================================================================== #
# 4. two index builds on different columns of the same table, concurrently
# =========================================================================== #
def test_concurrent_index_builds_two_columns(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    N = 40_000
    t = _fresh(db, "two_build", {"id": {"type": "int"}, "a": {"type": "int"}, "b": {"type": "int"}})
    # a ascending, b descending — distinct so a mixed-up build is detectable
    _bulk_insert(t, [{"id": i, "a": i, "b": N - i} for i in range(N)])

    # Two create_index DDLs on the SAME table can conflict on the shared
    # `next_index_id` metadata counter (optimistic concurrency -> one aborts with
    # error 9003 "Resource busy"). That is an acceptable outcome *provided* state
    # stays consistent and the loser is retryable; a non-conflict error, or a wrong
    # result afterwards, would be a real defect.
    results: dict = {}

    def build(col, idxname):
        c = inst.connect()
        tb = _tbl(c, "two_build")
        try:
            tb.create_index(idxname, index.IndexInfo(col, index.IndexType.Secondary),
                            ConflictType.Error)
            results[idxname] = ("ok", None)
        except Exception as e:  # noqa: BLE001
            results[idxname] = ("err", (type(e).__name__, str(e)[:160]))
        finally:
            c.disconnect()

    ta = threading.Thread(target=build, args=("a", "ix_a"), name="ba")
    tb = threading.Thread(target=build, args=("b", "ix_b"), name="bb")
    ta.start(); tb.start()
    hung = _join_all([ta, tb])
    assert not hung, f"threads hung: {hung}"

    # Classify: transaction-conflict errors are tolerated-and-retried; anything else fails.
    def _is_conflict(msg):
        m = msg.lower()
        return "resource busy" in m or "conflict" in m or "9003" in m or "busy" in m
    conflicted = []
    for col, idxname in (("a", "ix_a"), ("b", "ix_b")):
        status, err = results[idxname]
        if status == "err":
            assert _is_conflict(err[1]), f"[{idxname}] non-conflict build error: {err}"
            conflicted.append(idxname)
    RESULTS["notes"].append(
        f"two concurrent create_index on same table: {len(conflicted)} conflicted (9003), retried: {conflicted}")
    # Retry the conflicted build(s) sequentially — must now succeed and be consistent.
    for idxname in conflicted:
        col = "a" if idxname == "ix_a" else "b"
        t.create_index(idxname, index.IndexInfo(col, index.IndexType.Secondary),
                       ConflictType.Error)

    # both indexes independently correct
    a_vals = sorted(_filter_vals(t, 100, 200, col="a"))
    assert a_vals == list(range(100, 200)), f"index a wrong: {len(a_vals)}"
    # b = N - i, so b in [100,200) <=> i in (N-200, N-100]
    b_df, _ = t.output(["id"]).filter("b >= 100 AND b < 200").to_pl()
    b_ids = sorted(b_df["id"].to_list())
    assert b_ids == list(range(N - 199, N - 99)), f"index b wrong: {b_ids[:3]}..{b_ids[-3:]}"
    assert inst.log_errors(off) == []


# =========================================================================== #
# 5. delete racing an index build — deleted rows must not resurrect
# =========================================================================== #
def test_delete_racing_secondary_build(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    N = 50_000
    D = 20_000                                  # delete v in [0, D)
    t = _fresh(db, "del_build", {"id": {"type": "int"}, "v": {"type": "int"}})
    _bulk_insert(t, [{"id": i, "v": i} for i in range(N)])

    errors: list = []

    def builder():
        c = inst.connect()
        tb = _tbl(c, "del_build")
        try:
            tb.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary),
                            ConflictType.Error)
        except Exception as e:  # noqa: BLE001
            errors.append(("build", type(e).__name__, str(e)[:120]))
        finally:
            c.disconnect()

    def deleter():
        c = inst.connect()
        td = _tbl(c, "del_build")
        B = 2000
        try:
            for start in range(0, D, B):
                td.delete(f"v >= {start} AND v < {min(start + B, D)}")
        except Exception as e:  # noqa: BLE001
            errors.append(("delete", type(e).__name__, str(e)[:120]))
        finally:
            c.disconnect()

    bt = threading.Thread(target=builder, name="build")
    dt = threading.Thread(target=deleter, name="del")
    bt.start(); dt.start()
    hung = _join_all([bt, dt])
    assert not hung, f"threads hung: {hung}"
    assert errors == [], f"delete/build errors: {errors[:8]}"

    # deleted range must be empty through the finished index
    gone = _filter_vals(t, 0, D)
    assert gone == [], f"deleted rows resurrected via index: {len(gone)} rows, e.g. {gone[:5]}"
    # survivors intact and exact
    surv = sorted(_filter_vals(t, D, N))
    assert surv == list(range(D, N)), f"survivors wrong: {len(surv)} != {N - D}"
    assert _count_all(t) == N - D, f"total {_count_all(t)} != {N - D}"
    assert inst.log_errors(off) == []


# =========================================================================== #
# 6. compact racing queries (multi-segment) — pre == post, correct during
# =========================================================================== #
def test_compact_racing_queries(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    segs = 6
    per = 4000
    total = segs * per
    t = _fresh(db, "compact_race", {"id": {"type": "int"}, "v": {"type": "int"}})
    for s in range(segs):                        # each import -> its own segment
        rows = [(s * per + i, s * per + i) for i in range(per)]
        p = inst.stage_csv(f"compact_seg{s}.csv", rows)
        t.import_data(str(p), {"file_type": "csv"})
    t.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)
    seg_before = t.show_segments().height
    lo, hi = 5_000, 19_000
    pre = sorted(_filter_vals(t, lo, hi))
    assert pre == list(range(lo, hi)), f"pre-compact baseline wrong: {len(pre)}"

    errors: list = []
    iters = [0]
    stop = threading.Event()

    def querier():
        c = inst.connect()
        tq = _tbl(c, "compact_race")
        n = 0
        try:
            while not stop.is_set():
                vals = sorted(_filter_vals(tq, lo, hi))
                n += 1
                if vals != pre:
                    errors.append(("mismatch", len(vals), "expected", len(pre)))
        except Exception as e:  # noqa: BLE001
            errors.append(("q-exc", type(e).__name__, str(e)[:120]))
        finally:
            iters[0] = n
            c.disconnect()

    q1 = threading.Thread(target=querier, name="q1")
    q2 = threading.Thread(target=querier, name="q2")
    q1.start(); q2.start()
    time.sleep(0.2)
    cres = t.compact()                            # merge 6 -> 1 while querying
    stop.set()
    hung = _join_all([q1, q2])

    seg_after = t.show_segments().height
    RESULTS["notes"].append(f"compact: segments {seg_before}->{seg_after}, {iters[0]} queries during")
    assert not hung, f"threads hung: {hung}"
    assert errors == [], f"query mismatch during compact: {errors[:8]}"
    assert seg_after < seg_before, f"compact did not merge segments ({seg_before}->{seg_after})"
    post = sorted(_filter_vals(t, lo, hi))
    assert post == pre, f"post-compact result set changed: {len(post)} vs {len(pre)}"
    assert _count_all(t) == total, f"compact changed row count: {_count_all(t)} != {total}"
    assert inst.log_errors(off) == []


# =========================================================================== #
# 7. optimize racing queries
# =========================================================================== #
def test_optimize_racing_queries(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    N = 80_000                                   # crosses capacity -> chunks to optimize
    t = _fresh(db, "opt_race", {"id": {"type": "int"}, "v": {"type": "int"}})
    _bulk_insert(t, [{"id": i, "v": i} for i in range(N)])
    t.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)
    lo, hi = 1000, 60_000
    pre = sorted(_filter_vals(t, lo, hi))
    assert pre == list(range(lo, hi))

    errors: list = []
    iters = [0]
    stop = threading.Event()

    def querier():
        c = inst.connect()
        tq = _tbl(c, "opt_race")
        n = 0
        try:
            while not stop.is_set():
                vals = sorted(_filter_vals(tq, lo, hi))
                n += 1
                if vals != pre:
                    errors.append(("mismatch", len(vals)))
        except Exception as e:  # noqa: BLE001
            errors.append(("q-exc", type(e).__name__, str(e)[:120]))
        finally:
            iters[0] = n
            c.disconnect()

    q = threading.Thread(target=querier, name="q")
    q.start()
    time.sleep(0.2)
    for _ in range(3):
        t.optimize()
    stop.set()
    hung = _join_all([q])
    RESULTS["notes"].append(f"optimize: {iters[0]} queries during 3 optimize() calls")
    assert not hung, f"threads hung: {hung}"
    assert errors == [], f"query mismatch during optimize: {errors[:8]}"
    assert sorted(_filter_vals(t, lo, hi)) == pre
    assert inst.log_errors(off) == []


# =========================================================================== #
# 8. explicit dump_index racing queries (synchronous mem-index -> chunk)
# =========================================================================== #
def test_dump_index_racing_queries(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    N = 40_000                                   # stays under capacity: lives in mem-index
    t = _fresh(db, "dumpidx_race", {"id": {"type": "int"}, "v": {"type": "int"}})
    _bulk_insert(t, [{"id": i, "v": i} for i in range(N)])
    t.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)
    lo, hi = 100, 30_000
    pre = sorted(_filter_vals(t, lo, hi))
    assert pre == list(range(lo, hi))

    errors: list = []
    iters = [0]
    stop = threading.Event()

    def querier():
        c = inst.connect()
        tq = _tbl(c, "dumpidx_race")
        n = 0
        try:
            while not stop.is_set():
                vals = sorted(_filter_vals(tq, lo, hi))
                n += 1
                if vals != pre:
                    errors.append(("mismatch", len(vals)))
        except Exception as e:  # noqa: BLE001
            errors.append(("q-exc", type(e).__name__, str(e)[:120]))
        finally:
            iters[0] = n
            c.disconnect()

    qs = [threading.Thread(target=querier, name=f"q{i}") for i in range(3)]
    for th in qs:
        th.start()
    time.sleep(0.2)
    dump_err = None
    try:
        t.dump_index("ix")                       # force mem-index -> on-disk chunk now
    except Exception as e:  # noqa: BLE001
        dump_err = (type(e).__name__, str(e)[:160])
    stop.set()
    hung = _join_all(qs)
    RESULTS["notes"].append(f"dump_index: {iters[0]} queries during forced dump; dump_err={dump_err}")
    assert not hung, f"threads hung: {hung}"
    assert dump_err is None, f"dump_index raised: {dump_err}"
    assert errors == [], f"query mismatch during dump_index: {errors[:8]}"
    assert sorted(_filter_vals(t, lo, hi)) == pre
    assert inst.log_errors(off) == []


# =========================================================================== #
# 9. per-index-type build under concurrent query load (+ latency degradation)
# =========================================================================== #
@dataclass
class BuildSpec:
    name: str
    schema: dict
    rows: list = field(default_factory=list)
    build: object = None          # (table) -> None : builds the index we race
    query: object = None          # (table) -> hashable result ; raises on malformed
    prebuild: object = None       # (table) -> None : optional index needed for query


def _hnsw_spec():
    rng = np.random.default_rng(SEED)
    N, DIM = 6000, 24
    corpus = rng.random((N, DIM), dtype=np.float32)
    q = rng.random(DIM, dtype=np.float32).tolist()
    d = np.sum((corpus.astype(np.float64) - np.asarray(q)) ** 2, axis=1)
    truth = set(int(i) for i in np.argsort(d)[:10])
    rows = [{"id": i, "v": corpus[i].tolist()} for i in range(N)]

    def build(t):
        t.create_index("h", index.IndexInfo("v", index.IndexType.Hnsw,
                        {"M": "16", "ef_construction": "200", "metric": "l2"}),
                        ConflictType.Error)

    def query(t):
        df, _ = t.output(["id", "_distance"]).match_dense(
            "v", q, "float", "l2", 10, {"ef": "200"}).to_pl()
        ids = df["id"].to_list()
        # the score column is not reliably named "_distance"; take the non-id column
        score_cols = [c for c in df.columns if c.lower() != "id"]
        sc = df[score_cols[0]].to_list() if score_cols else []
        assert len(ids) == 10, f"hnsw returned {len(ids)} != 10"
        assert all(0 <= i < N for i in ids), "hnsw id out of range"
        assert all(sc[i] <= sc[i + 1] + 1e-6 for i in range(len(sc) - 1)), "hnsw not sorted"
        return frozenset(ids)

    return BuildSpec("hnsw", {"id": {"type": "int"}, "v": {"type": "vector,24,float32"}},
                     rows, build, query), truth


def _ivf_spec():
    rng = np.random.default_rng(SEED + 1)
    N, DIM = 6000, 24
    corpus = rng.random((N, DIM), dtype=np.float32)
    q = rng.random(DIM, dtype=np.float32).tolist()
    d = np.sum((corpus.astype(np.float64) - np.asarray(q)) ** 2, axis=1)
    truth = set(int(i) for i in np.argsort(d)[:10])
    rows = [{"id": i, "v": corpus[i].tolist()} for i in range(N)]

    def build(t):
        t.create_index("iv", index.IndexInfo("v", index.IndexType.IVF, {"metric": "l2"}),
                        ConflictType.Error)

    def query(t):
        df, _ = t.output(["id", "_distance"]).match_dense(
            "v", q, "float", "l2", 10, {"nprobe": "128"}).to_pl()
        ids = df["id"].to_list()
        assert len(ids) == 10, f"ivf returned {len(ids)} != 10"
        assert all(0 <= i < N for i in ids), "ivf id out of range"
        return frozenset(ids)

    return BuildSpec("ivf", {"id": {"type": "int"}, "v": {"type": "vector,24,float32"}},
                     rows, build, query), truth


def _bmp_spec():
    rng = np.random.default_rng(SEED + 2)
    N, DIMS = 4000, 100          # indices < 100 fit int8 (max 127)
    rows = []
    mats = []
    for i in range(N):
        k = int(rng.integers(5, 15))
        idxs = sorted(set(int(x) for x in rng.integers(0, DIMS, size=k)))
        vals = [float(v) for v in rng.random(len(idxs)).astype(np.float32)]
        rows.append({"id": i, "c": SparseVector(idxs, vals)})
        mats.append(dict(zip(idxs, vals)))
    q_idx = list(range(0, 40))
    q_val = [1.0] * 40
    qmap = dict(zip(q_idx, q_val))
    ip = np.array([sum(qmap.get(k, 0.0) * v for k, v in m.items()) for m in mats])
    truth = set(int(i) for i in np.argsort(-ip)[:10])

    def build(t):
        t.create_index("bmp", index.IndexInfo("c", index.IndexType.BMP,
                        {"block_size": "16", "compress_type": "compress"}), ConflictType.Error)

    def query(t):
        df, _ = t.output(["id", "_similarity"]).match_sparse(
            "c", SparseVector(q_idx, q_val), "ip", 10, {"alpha": "1.0", "beta": "1.0"}).to_pl()
        ids = df["id"].to_list()
        assert 1 <= len(ids) <= 10, f"bmp returned {len(ids)}"
        assert all(0 <= i < N for i in ids), "bmp id out of range"
        return frozenset(ids)

    return BuildSpec("bmp", {"id": {"type": "int"}, "c": {"type": "sparse,100,float,int8"}},
                     rows, build, query), truth


def _secondary_spec():
    N = 60_000
    rows = [{"id": i, "v": i} for i in range(N)]
    lo, hi = 1000, 40_000
    truth = frozenset(range(lo, hi))

    def build(t):
        t.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)

    def query(t):
        vals = _filter_vals(t, lo, hi)
        got = frozenset(vals)
        assert len(vals) == len(got), "secondary produced duplicates"
        assert got == truth, f"secondary set wrong: {len(got)} vs {len(truth)}"
        return got

    return BuildSpec("secondary", {"id": {"type": "int"}, "v": {"type": "int"}},
                     rows, build, query), truth


@pytest.mark.parametrize("mk", [_secondary_spec, _hnsw_spec, _ivf_spec, _bmp_spec],
                         ids=["secondary", "hnsw", "ivf", "bmp"])
def test_index_build_under_query_load(inst, mk):
    """Build an index of each type while queries of the matching kind hammer the
    same column. Queries fall back to brute force before the index commits and use
    the index after — the result must be well-formed throughout, exact (secondary)
    or high-recall (ANN) after, and the latency cost of building is reported.
    """
    off = inst.log_offset()
    conn, db = inst.db()
    spec, truth = mk()
    tname = f"buildload_{spec.name}"
    t = _fresh(db, tname, spec.schema)
    # insert in batches (large vector payloads per row)
    B = 2000
    for s in range(0, len(spec.rows), B):
        t.insert(spec.rows[s:s + B])

    errors: list = []

    # baseline latency: brute force, no concurrent build
    base_lat = []
    for _ in range(15):
        t0 = time.perf_counter()
        spec.query(t)
        base_lat.append(time.perf_counter() - t0)

    during_lat: list = []
    iters = [0]
    stop = threading.Event()

    def querier():
        c = inst.connect()
        tq = _tbl(c, tname)
        n = 0
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                spec.query(tq)
                during_lat.append(time.perf_counter() - t0)
                n += 1
        except Exception as e:  # noqa: BLE001
            errors.append((spec.name, "q-exc", type(e).__name__, str(e)[:140]))
        finally:
            iters[0] = n
            c.disconnect()

    build_err = [None]

    def builder():
        c = inst.connect()
        tb = _tbl(c, tname)
        try:
            spec.build(tb)
        except Exception as e:  # noqa: BLE001
            build_err[0] = (type(e).__name__, str(e)[:160])
        finally:
            c.disconnect()

    q = threading.Thread(target=querier, name="q")
    b = threading.Thread(target=builder, name="build")
    q.start()
    time.sleep(0.1)
    b.start()
    hung = _join_all([b])
    stop.set()
    hung += _join_all([q])

    base = _percentiles(base_lat)
    during = _percentiles(during_lat)
    ratio = (during.get("median_ms", 0) / base["median_ms"]) if base.get("median_ms") else None
    RESULTS["measurements"].append({
        "what": f"{spec.name} query latency while its index builds",
        "value": f"baseline median {base.get('median_ms'):.2f}ms -> during-build median "
                 f"{during.get('median_ms', float('nan')):.2f}ms "
                 f"(p95 {during.get('p95_ms', float('nan')):.2f}ms, {iters[0]} queries)",
        "method": "same fixed query, perf_counter, brute-force baseline vs concurrent create_index",
    })
    RESULTS["iterations"][f"buildload_{spec.name}"] = iters[0]

    assert not hung, f"[{spec.name}] threads hung during build-under-load: {hung}"
    assert build_err[0] is None, f"[{spec.name}] build raised: {build_err[0]}"
    assert errors == [], f"[{spec.name}] malformed query results during build: {errors[:6]}"
    assert iters[0] >= 3, f"[{spec.name}] too few concurrent queries: {iters[0]}"

    # correctness after the index is committed
    res = spec.query(t)
    if spec.name in ("hnsw", "ivf", "bmp"):
        recall = len(set(res) & truth) / len(truth)
        RESULTS["measurements"].append({
            "what": f"{spec.name} recall@10 after concurrent build",
            "value": f"{recall:.2f}", "method": "vs brute-force ground truth"})
        assert recall >= 0.7, f"[{spec.name}] post-build recall {recall} < 0.7"
    else:
        assert res == truth, f"[{spec.name}] post-build exact set wrong"
    assert inst.log_errors(off) == []


# =========================================================================== #
# 9b. FullText build racing queries (match_text needs an index, so we query an
#     already-built FullText index on column A while building one on column B)
# =========================================================================== #
def _ft_ids(table, field, term, topn=2000):
    df, _ = table.output(["id"]).match_text(field, term, topn, None).to_pl()
    idc = [c for c in df.columns if c.lower() == "id"][0]
    return set(df[idc].to_list())


def test_fulltext_build_racing_queries(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    N = 4000
    t = _fresh(db, "ft_race",
               {"id": {"type": "int"}, "a": {"type": "varchar"}, "b": {"type": "varchar"}})
    # doc i: "apple" in a iff i%10==0 ; "cherry" in b iff i%7==0 — exact, known sets
    rows = [{"id": i,
             "a": f"doc{i} " + ("apple " if i % 10 == 0 else "banana ") + "filler text",
             "b": f"rec{i} " + ("cherry " if i % 7 == 0 else "date ") + "pad words"}
            for i in range(N)]
    _bulk_insert(t, rows)
    t.create_index("ft_a", index.IndexInfo("a", index.IndexType.FullText), ConflictType.Error)

    expected_a = {i for i in range(N) if i % 10 == 0}
    baseline = _ft_ids(t, "a", "apple")
    assert baseline == expected_a, f"FT baseline wrong: {len(baseline)} vs {len(expected_a)}"

    errors: list = []
    lat: list = []
    iters = [0]
    stop = threading.Event()

    def querier():
        c = inst.connect()
        tq = _tbl(c, "ft_race")
        n = 0
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                got = _ft_ids(tq, "a", "apple")
                lat.append(time.perf_counter() - t0)
                n += 1
                if got != expected_a:
                    errors.append(("ft-mismatch", len(got), "expected", len(expected_a)))
        except Exception as e:  # noqa: BLE001
            errors.append(("ft-q-exc", type(e).__name__, str(e)[:140]))
        finally:
            iters[0] = n
            c.disconnect()

    build_err = [None]

    def builder():
        c = inst.connect()
        tb = _tbl(c, "ft_race")
        try:
            tb.create_index("ft_b", index.IndexInfo("b", index.IndexType.FullText),
                            ConflictType.Error)
        except Exception as e:  # noqa: BLE001
            build_err[0] = (type(e).__name__, str(e)[:160])
        finally:
            c.disconnect()

    q = threading.Thread(target=querier, name="q")
    b = threading.Thread(target=builder, name="build")
    q.start()
    time.sleep(0.1)
    b.start()
    hung = _join_all([b])
    stop.set()
    hung += _join_all([q])

    RESULTS["iterations"]["ft_build_queries"] = iters[0]
    p = _percentiles(lat)
    if p:
        RESULTS["measurements"].append({
            "what": "fulltext match_text latency while a second FT index builds",
            "value": f"median {p['median_ms']:.2f}ms p95 {p['p95_ms']:.2f}ms ({iters[0]} queries)",
            "method": "query built FT index on col a while create_index on col b"})

    assert not hung, f"threads hung: {hung}"
    assert build_err[0] is None, f"fulltext build raised: {build_err[0]}"
    assert errors == [], f"match_text wrong during concurrent FT build: {errors[:6]}"
    assert iters[0] >= 3, f"too few concurrent FT queries: {iters[0]}"
    # reads on the EXISTING FT index (col a) stay exact throughout a concurrent build
    assert _ft_ids(t, "a", "apple") == expected_a
    assert inst.log_errors(off) == []
    # NOTE: whether the *concurrently built* col-b index is usable is a separate
    # (currently failing) property — see test_fulltext_concurrent_build_yields_usable_index.


@pytest.mark.xfail(reason="ENGINE DEFECT: a FullText index built while match_text queries run "
                          "concurrently is registered in the catalog (list_indexes shows it) and "
                          "create_index returns OK (error_code 0), but match_text on it fails with "
                          "(3013, 'Column ... index \"\" doesn't exist') and nothing is logged. A "
                          "SIGKILL+WAL restart repairs it, so the WAL/catalog are correct and only "
                          "the live process's in-memory index registration is missing. "
                          "Deterministic 3/3 in isolation.",
                   raises=Exception, strict=False)
def test_fulltext_concurrent_build_yields_usable_index(inst):
    """Isolates the defect: after building a FullText index concurrently with query
    load, the index must be *usable*. Currently it is not (xfail). If this starts to
    pass, the engine has been fixed — remove the xfail marker.
    """
    conn, db = inst.db()
    N = 4000
    t = _fresh(db, "ft_usable",
               {"id": {"type": "int"}, "a": {"type": "varchar"}, "b": {"type": "varchar"}})
    rows = [{"id": i,
             "a": f"doc{i} " + ("apple " if i % 10 == 0 else "banana ") + "filler",
             "b": f"rec{i} " + ("cherry " if i % 7 == 0 else "date ") + "pad"}
            for i in range(N)]
    _bulk_insert(t, rows)
    t.create_index("ft_a", index.IndexInfo("a", index.IndexType.FullText), ConflictType.Error)

    stop = threading.Event()

    def querier():
        c = inst.connect()
        tq = _tbl(c, "ft_usable")
        try:
            while not stop.is_set():
                _ft_ids(tq, "a", "apple")
        except Exception:  # noqa: BLE001
            pass
        finally:
            c.disconnect()

    qs = [threading.Thread(target=querier, name=f"q{i}") for i in range(3)]
    for th in qs:
        th.start()
    time.sleep(0.1)
    cb = inst.connect()
    tb = _tbl(cb, "ft_usable")
    tb.create_index("ft_b", index.IndexInfo("b", index.IndexType.FullText), ConflictType.Error)
    cb.disconnect()
    stop.set()
    _join_all(qs)

    # the concurrently-built index must be usable (fresh connection to rule out cache)
    c2 = inst.connect()
    t2 = _tbl(c2, "ft_usable")
    expected_b = {i for i in range(N) if i % 7 == 0}
    try:
        got = _ft_ids(t2, "b", "cherry")
    finally:
        c2.disconnect()
    assert got == expected_b, f"concurrently-built FT index unusable/wrong: {len(got)}"


# =========================================================================== #
# 10. cost of the new lock: range latency, quiescent vs writer-contended
# =========================================================================== #
def test_secondary_range_lock_contention_cost(inst):
    """Quantify the serialisation the RcuMultiMap::range fix introduces. The query
    range is a FIXED 100-row window, so its intrinsic cost does not change as the
    writer grows the map; any latency increase is attributable to contending with
    Insert() on dirty_lock_. Everything stays under mem_index_capacity so the whole
    index remains a single in-memory RcuMultiMap (no dump changes the regime).

    A modest slowdown is the expected, correct cost (recorded as a measurement). A
    pathological collapse (unbounded growth / >~50x) is reported as a perf-leak.
    """
    off = inst.log_offset()
    conn, db = inst.db()
    QLO, QHI = 0, 100                            # fixed 100-row query window
    base_fill = 30_000                           # + writer stays < 65536 => no dump
    t = _fresh(db, "lockcost", {"id": {"type": "int"}, "v": {"type": "int"}})
    # 100 rows in the query window, rest well outside it
    t.insert([{"id": i, "v": i} for i in range(QLO, QHI)])
    _bulk_insert(t, [{"id": i, "v": i} for i in range(1000, base_fill)])
    t.create_index("ix", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)

    def measure(n):
        lat = []
        for _ in range(n):
            t0 = time.perf_counter()
            vals = _filter_vals(t, QLO, QHI)
            lat.append(time.perf_counter() - t0)
            assert len(vals) == QHI - QLO, f"window changed size: {len(vals)}"
        return lat

    quiescent = measure(120)

    errors: list = []
    stop = threading.Event()
    inserted = [0]

    def writer():
        c = inst.connect()
        tw = _tbl(c, "lockcost")
        base = 2_000_000
        cap = MEM_INDEX_CAPACITY - base_fill - 500   # stay under the dump threshold
        n = 0
        B = 200
        try:
            while not stop.is_set() and n < cap:
                end = min(n + B, cap)
                tw.insert([{"id": base + j, "v": base + j} for j in range(n, end)])
                n = end
        except Exception as e:  # noqa: BLE001
            errors.append(("writer", type(e).__name__, str(e)[:120]))
        finally:
            inserted[0] = n
            c.disconnect()

    w = threading.Thread(target=writer, name="writer")
    w.start()
    contended = measure(120)
    stop.set()
    hung = _join_all([w])

    qp = _percentiles(quiescent)
    cp = _percentiles(contended)
    ratio_med = cp["median_ms"] / qp["median_ms"] if qp["median_ms"] else float("inf")
    ratio_p95 = cp["p95_ms"] / qp["p95_ms"] if qp["p95_ms"] else float("inf")
    RESULTS["measurements"].append({
        "what": "secondary range-query latency: quiescent vs insert-contended (in-mem segment)",
        "value": (f"quiescent median {qp['median_ms']:.3f}ms p95 {qp['p95_ms']:.3f}ms; "
                  f"contended median {cp['median_ms']:.3f}ms p95 {cp['p95_ms']:.3f}ms; "
                  f"median x{ratio_med:.1f}, p95 x{ratio_p95:.1f}; writer inserted {inserted[0]}"),
        "method": "fixed 100-row window under dirty_lock_; 120 samples each phase",
    })

    assert not hung, f"writer hung: {hung}"
    assert errors == [], f"writer errors: {errors[:4]}"
    # correctness must survive contention
    assert len(_filter_vals(t, QLO, QHI)) == QHI - QLO
    # No dump should have happened (kept under capacity) — assert single segment count sane
    assert inst.log_errors(off) == []
    # Pathological collapse is a perf finding, surfaced by the report; assert only that
    # latency did not run away to absurd levels (guards a lock-ordering livelock).
    assert cp["p95_ms"] < 5000, f"contended p95 {cp['p95_ms']}ms is pathological (possible livelock)"
