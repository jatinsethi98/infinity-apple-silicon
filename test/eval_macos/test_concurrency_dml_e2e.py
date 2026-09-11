"""Concurrent DML correctness suite for the macOS arm64 port.

Area: correctness of insert / update / delete / select under races. Many threads and
several independent SDK connections (a single thrift connection is one socket, so each
thread MUST own its connection; sharing one would corrupt the protocol stream and look
like an engine defect) drive one table at once. The generated config has
``connection_pool_size = 128``, so the suites go wide enough to matter.

What "correct" means here is stronger than "did not crash":

* **No lost or duplicated rows.** N threads each insert M disjoint ids -> exactly N*M
  rows, every id present exactly once (``test_concurrent_inserts_no_loss_no_dup``).
* **No torn value.** Many writers hammer one row's varchar and one row's embedding to
  distinct values; a reader must only ever observe a whole value some writer wrote, never
  a byte-level mixture (``test_same_row_update_varchar_never_torn`` /
  ``..._embedding_...``).
* **No lost update on disjoint rows** (``test_disjoint_row_updates_all_land``).
* **An explicable final set for insert-racing-delete** (``test_insert_racing_delete``).
* **Snapshot self-consistency for a reader racing a writer**: every scan sees a state a
  correct engine could have produced — no half-inserted row, no row with some columns
  updated and others not, and (for a single multi-row UPDATE statement) no mixture of
  two versions across rows (``test_reader_racing_writer_snapshot_consistent``).
* **count(*) monotonic and bounded** under insert-only load
  (``test_count_monotonic_under_inserts``).
* **A multi-row insert is atomic**: a concurrent reader never sees a strict subset of one
  ``insert`` batch (``test_multi_row_insert_is_atomic``).

Isolation is *determined empirically and reported*, not assumed: the snapshot and
insert-atomicity tests together establish that a single DML statement is atomic to
concurrent readers (a reader cannot see a partially-applied statement).

Throughput (rows/sec for concurrent insert at 1/2/4/8/16 threads) is measured and
recorded to ``build/eval/results/concurrency.json``.

One test (``test_DEFECT_aggregate_reads_dead_versions_past_block_boundary``) pins a real,
deterministic engine defect this area found: after repeated all-row ``UPDATE``s push a
table's physical row-versions past one 8192-row block, ``sum``/``min``/``max`` read
uninitialised/dead slots and return silently-wrong results (``max`` returns a value no
row ever held), while ``count(*)`` and a full column scan stay correct. The pin follows
the repo's ``test_DEFECT_*`` convention: it passes today by asserting the wrong behaviour,
and will fail (alerting the maintainer) the day the aggregate is fixed.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import harness
from infinity.common import ConflictType, NetworkAddress
from infinity.connection_pool import ConnectionPool
from infinity.errors import ErrorCode

# Insert calls are capped server-side ("Insert batch row limit shouldn't more than 8192.");
# every writer batches at or below this.
INSERT_BATCH = 1000
JOIN_TIMEOUT = 60.0  # a thread still alive after this is a hang -> blocker


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    if harness.free_disk_bytes() < 5 * 1024**3:
        pytest.skip("need >=5 GiB free disk")
    it = harness.instance("eval-concurrency")
    it.start()
    try:
        yield it
    finally:
        it.stop()


@pytest.fixture()
def db(inst):
    conn = inst.connect()
    try:
        yield conn.get_database("default_db")
    finally:
        conn.disconnect()


def _fresh(db, name, schema):
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(name, schema, ConflictType.Error)


def _count(t) -> int:
    res, _ = t.output(["count(*)"]).to_pl()
    return int(res.item(0, 0))


def _agg(t, expr):
    res, _ = t.output([expr]).to_pl()
    return res.item(0, 0)


def _scan(t, col):
    df, _ = t.output([col]).to_df()
    return df[col].to_list()


def _insert_batches(table, rows):
    for i in range(0, len(rows), INSERT_BATCH):
        table.insert(rows[i:i + INSERT_BATCH])


def _run_workers(inst, fn, nthreads, *, table_name, per_thread_args=None):
    """Start ``nthreads`` threads, each with its own connection + table handle.

    ``fn(tid, table, stop_or_arg)`` runs the body. Exceptions are captured per thread and
    re-raised in the parent so a worker error is never swallowed. A thread still alive
    after ``JOIN_TIMEOUT`` is reported as a hang (blocker), not silently ignored.
    """
    errors: list[tuple[int, str]] = []

    def wrapped(tid):
        conn = inst.connect()
        try:
            t = conn.get_database("default_db").get_table(table_name)
            arg = per_thread_args[tid] if per_thread_args is not None else None
            fn(tid, t, arg)
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append((tid, repr(exc)))
        finally:
            try:
                conn.disconnect()
            except Exception:
                pass

    threads = [threading.Thread(target=wrapped, args=(i,), name=f"w{i}") for i in range(nthreads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=JOIN_TIMEOUT)
    hung = [th.name for th in threads if th.is_alive()]
    assert not hung, f"threads hung (possible deadlock/hang): {hung}"
    assert not errors, f"worker threads raised: {errors[:8]}"


# module-global instance handle so the final log-scrape test can see errors from any test
_REF: dict = {}
_RESULTS: dict = {"isolation": {}, "throughput_rows_per_sec": {}, "notes": {}}


@pytest.fixture(scope="module", autouse=True)
def _capture(inst):
    _REF["inst"] = inst
    yield
    harness.record("concurrency", _RESULTS)


# --------------------------------------------------------------------------- #
# 1. concurrent inserts -> no lost / duplicated row
# --------------------------------------------------------------------------- #

def test_concurrent_inserts_no_loss_no_dup(db, inst):
    """8 threads each insert 4000 disjoint ids concurrently. Afterwards exactly 32000 rows
    must exist, every id present exactly once (no lost insert, no duplicate), and — since
    all rows are live with no dead versions — ``sum`` over a constant column must be exact."""
    name = "c_ins"
    n_threads, per = 8, 4000
    _fresh(db, name, {"id": {"type": "int"}, "who": {"type": "int"}, "a": {"type": "int"}})
    try:
        def body(tid, t, _):
            rows = [{"id": tid * per + i, "who": tid, "a": 1} for i in range(per)]
            _insert_batches(t, rows)

        _run_workers(inst, body, n_threads, table_name=name)

        total = n_threads * per
        assert _count(db.get_table(name)) == total
        ids = _scan(db.get_table(name), "id")
        assert len(ids) == total, f"row count via scan {len(ids)} != {total} (lost/dup rows)"
        assert set(ids) == set(range(total)), "id set differs from expected (lost or spurious ids)"
        assert len(set(ids)) == total, "duplicate ids present"
        # pure-insert data (no dead versions): the aggregate must be exact here.
        assert int(_agg(db.get_table(name), "sum(a)")) == total
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_concurrent_inserts_via_connection_pool(db, inst):
    """Same guarantee, driven through ``infinity.connection_pool.ConnectionPool`` (the
    brief asks for it explicitly). Each task borrows a pooled connection, inserts a
    disjoint id range, and releases it."""
    name = "c_ins_pool"
    n_tasks, per = 12, 2000
    _fresh(db, name, {"id": {"type": "int"}})
    pool = ConnectionPool(uri=NetworkAddress("127.0.0.1", inst.thrift_port), max_size=16)
    try:
        def task(tid):
            conn = pool.get_conn()
            try:
                t = conn.get_database("default_db").get_table(name)
                _insert_batches(t, [{"id": tid * per + i} for i in range(per)])
            finally:
                pool.release_conn(conn)

        with ThreadPoolExecutor(max_workers=n_tasks) as ex:
            list(ex.map(task, range(n_tasks)))

        total = n_tasks * per
        ids = _scan(db.get_table(name), "id")
        assert _count(db.get_table(name)) == total
        assert len(ids) == total and set(ids) == set(range(total))
        assert inst.log_errors() == []
    finally:
        pool.destroy()
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 2. concurrent update of the SAME row -> never a torn value
# --------------------------------------------------------------------------- #

def test_same_row_update_varchar_never_torn(db, inst):
    """Many threads update one row's varchar to distinct long values while a reader scans
    it continuously. Every value the reader observes must be a whole value some writer
    wrote (never a byte-level mixture of two), and the final value must be one that was
    written. Long distinct strings make a torn write visible."""
    name = "c_torn_vc"
    _fresh(db, name, {"id": {"type": "int"}, "s": {"type": "varchar"}})
    try:
        t0 = db.get_table(name)
        t0.insert([{"id": 1, "s": "init"}])
        writers = {f"{c}{c * 400}" for c in ("A", "B", "C", "D", "E", "F")}
        allowed = writers | {"init"}
        rounds = 60
        bad: list[str] = []
        stop = threading.Event()

        def reader():
            conn = inst.connect()
            try:
                rt = conn.get_database("default_db").get_table(name)
                while not stop.is_set():
                    for v in _scan(rt, "s"):
                        if v not in allowed:
                            bad.append((v or "")[:24])
            finally:
                conn.disconnect()

        rd = threading.Thread(target=reader, name="reader")
        rd.start()

        vals = list(writers)
        def body(tid, t, _):
            for _ in range(rounds):
                t.update("id = 1", {"s": vals[tid]})

        try:
            _run_workers(inst, body, len(vals), table_name=name)
        finally:
            stop.set()
            rd.join(timeout=JOIN_TIMEOUT)
            assert not rd.is_alive(), "reader hung"

        assert bad == [], f"torn / unexpected varchar values observed: {bad[:5]}"
        final = _scan(db.get_table(name), "s")
        assert len(final) == 1 and final[0] in writers, f"final value not one written: {final[:1]}"
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_same_row_update_embedding_never_torn(db, inst):
    """Same race on an 8-dim float embedding. Each writer writes an all-equal vector
    [w]*8 with a distinct w, so a torn write shows up as a vector whose 8 elements are not
    all equal, or whose value was never written. Reader asserts every observed vector is
    one whole writer's vector."""
    name = "c_torn_vec"
    _fresh(db, name, {"id": {"type": "int"}, "vec": {"type": "vector,8,float"}})
    try:
        t0 = db.get_table(name)
        t0.insert([{"id": 1, "vec": [0.0] * 8}])
        writer_vals = [float(x) for x in (11, 22, 33, 44, 55, 66)]
        allowed = set(writer_vals) | {0.0}
        rounds = 60
        bad: list = []
        stop = threading.Event()

        def reader():
            conn = inst.connect()
            try:
                rt = conn.get_database("default_db").get_table(name)
                while not stop.is_set():
                    df, _ = rt.output(["vec"]).filter("id = 1").to_df()
                    for vec in df["vec"].to_list():
                        vec = list(vec)
                        if len(set(vec)) != 1 or vec[0] not in allowed:
                            bad.append(vec)
            finally:
                conn.disconnect()

        rd = threading.Thread(target=reader, name="reader")
        rd.start()

        def body(tid, t, _):
            v = writer_vals[tid]
            for _ in range(rounds):
                t.update("id = 1", {"vec": [v] * 8})

        try:
            _run_workers(inst, body, len(writer_vals), table_name=name)
        finally:
            stop.set()
            rd.join(timeout=JOIN_TIMEOUT)
            assert not rd.is_alive(), "reader hung"

        assert bad == [], f"torn / unexpected embeddings observed: {bad[:5]}"
        df, _ = db.get_table(name).output(["vec"]).filter("id = 1").to_df()
        fv = list(df["vec"].to_list()[0])
        assert len(set(fv)) == 1 and fv[0] in writer_vals, f"final vector not one written: {fv}"
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 3. concurrent update of DISJOINT rows -> every update lands (no lost update)
# --------------------------------------------------------------------------- #

def test_disjoint_row_updates_all_land(db, inst):
    """Each of 8 threads owns a disjoint block of rows and stamps every row it owns with
    its own marker, over several rounds, all concurrently. Afterwards every row must carry
    exactly its owner's marker — a lost update anywhere is a blocker."""
    name = "c_disjoint"
    n_threads, per, rounds = 8, 200, 5
    _fresh(db, name, {"id": {"type": "int"}, "owner": {"type": "int"}, "mark": {"type": "int"}})
    try:
        t0 = db.get_table(name)
        rows = [{"id": tid * per + i, "owner": tid, "mark": 0}
                for tid in range(n_threads) for i in range(per)]
        _insert_batches(t0, rows)

        def body(tid, t, _):
            for r in range(rounds):
                t.update(f"owner = {tid}", {"mark": tid * 1000 + r})

        _run_workers(inst, body, n_threads, table_name=name)

        df, _ = db.get_table(name).output(["owner", "mark"]).to_df()
        got = {}
        for owner, mark in zip(df["owner"].to_list(), df["mark"].to_list()):
            got.setdefault(owner, set()).add(mark)
        for tid in range(n_threads):
            assert got.get(tid) == {tid * 1000 + (rounds - 1)}, \
                f"owner {tid}: lost update, marks seen {sorted(got.get(tid, []))}"
        assert _count(db.get_table(name)) == n_threads * per
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 4. insert racing delete on overlapping keys -> explicable final set
# --------------------------------------------------------------------------- #

def test_insert_racing_delete(db, inst):
    """Writers insert the whole key range [0, K) while a deleter repeatedly deletes the
    lower half. After the writers finish, one final delete of the lower half is issued.
    The final set must then be exactly the upper half, each id once: the lower half is
    provably gone and every surviving id is present exactly once (no lost/dup)."""
    name = "c_ins_del"
    K = 16000
    half = K // 2
    _fresh(db, name, {"id": {"type": "int"}})
    try:
        stop = threading.Event()
        del_errors: list[str] = []

        def deleter():
            conn = inst.connect()
            try:
                dt = conn.get_database("default_db").get_table(name)
                while not stop.is_set():
                    try:
                        dt.delete(f"id < {half}")
                    except Exception as exc:  # noqa: BLE001
                        del_errors.append(repr(exc))
                        break
                    time.sleep(0.002)
            finally:
                conn.disconnect()

        dl = threading.Thread(target=deleter, name="deleter")
        dl.start()

        n_threads, per = 8, K // 8
        def body(tid, t, _):
            _insert_batches(t, [{"id": tid * per + i} for i in range(per)])

        try:
            _run_workers(inst, body, n_threads, table_name=name)
        finally:
            stop.set()
            dl.join(timeout=JOIN_TIMEOUT)
            assert not dl.is_alive(), "deleter hung"

        assert del_errors == [], f"deleter raised: {del_errors[:4]}"
        # settle the final set deterministically: one last delete of the lower half.
        db.get_table(name).delete(f"id < {half}")
        ids = _scan(db.get_table(name), "id")
        assert set(ids) == set(range(half, K)), \
            f"final set not explicable: missing={sorted(set(range(half, K)) - set(ids))[:5]} " \
            f"spurious={sorted(set(ids) - set(range(half, K)))[:5]}"
        assert len(ids) == half, f"duplicate survivors: {len(ids)} != {half}"
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 5. reader racing writer -> every snapshot self-consistent (atomicity)
# --------------------------------------------------------------------------- #

def test_reader_racing_writer_snapshot_consistent(db, inst):
    """The atomicity question that matters most. Every row satisfies the invariant
    a + b == C. A single writer statement rewrites ALL rows at once (a=v, b=C-v, ver=v),
    so at any committed instant every row shares one version. A reader scanning
    continuously asserts, for every snapshot it sees:

      (i)  per-row: a + b == C for every row (no row with a updated but b not -> no
           half-applied row), and
      (ii) cross-row: all rows share a single ``ver`` (no mixture of two versions ->
           the multi-row UPDATE statement is atomic / snapshot-isolated to the reader).

    Runs 80 update iterations against a continuous reader; the observed scan count is
    recorded so a future reader knows how hard it was pushed."""
    name = "c_snap"
    nrows, C, iters = 500, 10_000, 80
    _fresh(db, name, {"id": {"type": "int"}, "a": {"type": "int"}, "b": {"type": "int"}, "ver": {"type": "int"}})
    try:
        db.get_table(name).insert([{"id": i, "a": 0, "b": C, "ver": 0} for i in range(nrows)])
        stop = threading.Event()
        per_row_torn: list = []
        version_mix: list = []
        scans = [0]

        def reader():
            conn = inst.connect()
            try:
                rt = conn.get_database("default_db").get_table(name)
                while not stop.is_set():
                    df, _ = rt.output(["a", "b", "ver"]).to_df()
                    a = df["a"].to_list(); b = df["b"].to_list(); ver = df["ver"].to_list()
                    if not a:
                        continue
                    scans[0] += 1
                    for av, bv in zip(a, b):
                        if av + bv != C:
                            per_row_torn.append((av, bv)); break
                    if len(set(ver)) > 1:
                        version_mix.append(sorted(set(ver)))
            finally:
                conn.disconnect()

        rd = threading.Thread(target=reader, name="reader")
        rd.start()
        time.sleep(0.1)
        writer = db.get_table(name)
        for v in range(1, iters + 1):
            writer.update("1 = 1", {"a": v, "b": C - v, "ver": v})
        time.sleep(0.1)
        stop.set()
        rd.join(timeout=JOIN_TIMEOUT)
        assert not rd.is_alive(), "reader hung"

        assert per_row_torn == [], f"half-applied row observed (a+b != C): {per_row_torn[:5]}"
        assert version_mix == [], f"reader saw a mixture of versions (not snapshot-atomic): {version_mix[:5]}"
        assert scans[0] > 0, "reader never completed a scan"
        _RESULTS["isolation"]["multi_row_update_atomic"] = True
        _RESULTS["isolation"]["snapshot_scans_observed"] = scans[0]
        _RESULTS["notes"]["snapshot"] = (
            f"{scans[0]} concurrent scans across {iters} all-row UPDATE statements: "
            "0 half-applied rows, 0 cross-row version mixes -> single DML statement is "
            "atomic and snapshot-isolated to a concurrent reader"
        )
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 6. count(*) monotonic and bounded under insert-only load
# --------------------------------------------------------------------------- #

def test_count_monotonic_under_inserts(db, inst):
    """A reader sampling count(*) while 8 threads insert (never delete) must see a
    non-decreasing sequence that never exceeds the number ultimately committed."""
    name = "c_count_mono"
    n_threads, per = 8, 3000
    total = n_threads * per
    _fresh(db, name, {"id": {"type": "int"}})
    try:
        stop = threading.Event()
        samples: list[int] = []

        def reader():
            conn = inst.connect()
            try:
                rt = conn.get_database("default_db").get_table(name)
                while not stop.is_set():
                    samples.append(_count(rt))
            finally:
                conn.disconnect()

        rd = threading.Thread(target=reader, name="reader")
        rd.start()

        def body(tid, t, _):
            _insert_batches(t, [{"id": tid * per + i} for i in range(per)])

        try:
            _run_workers(inst, body, n_threads, table_name=name)
        finally:
            stop.set()
            rd.join(timeout=JOIN_TIMEOUT)
            assert not rd.is_alive(), "reader hung"

        assert all(x <= total for x in samples), f"count exceeded committed total {total}: max {max(samples)}"
        # monotonic non-decreasing
        drops = [(samples[i - 1], samples[i]) for i in range(1, len(samples)) if samples[i] < samples[i - 1]]
        assert drops == [], f"count(*) went backwards under insert-only load: {drops[:5]}"
        assert _count(db.get_table(name)) == total
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 7. a multi-row insert is atomic (isolation determination)
# --------------------------------------------------------------------------- #

def test_multi_row_insert_is_atomic(db, inst):
    """One thread inserts a single full 8192-row batch tagged uniquely; a reader samples
    the count of that tag repeatedly. If ``insert`` is atomic the reader sees only 0 or
    8192 for the tag, never a strict subset of one batch. Repeated over several batches;
    the observed distinct counts are recorded."""
    name = "c_ins_atomic"
    batch = 8192
    _fresh(db, name, {"id": {"type": "int"}, "tag": {"type": "int"}})
    try:
        stop = threading.Event()
        observed: set[int] = set()

        def reader():
            conn = inst.connect()
            try:
                rt = conn.get_database("default_db").get_table(name)
                while not stop.is_set():
                    res, _ = rt.output(["count(*)"]).filter("tag = 7").to_pl()
                    observed.add(int(res.item(0, 0)))
            finally:
                conn.disconnect()

        rd = threading.Thread(target=reader, name="reader")
        rd.start()
        time.sleep(0.2)
        writer = db.get_table(name)
        for _ in range(6):
            writer.insert([{"id": i, "tag": 7} for i in range(batch)])  # one atomic call
            time.sleep(0.05)
            writer.delete("tag = 7")
            time.sleep(0.05)
        stop.set()
        rd.join(timeout=JOIN_TIMEOUT)
        assert not rd.is_alive(), "reader hung"

        partial = sorted(x for x in observed if 0 < x < batch)
        assert partial == [], f"reader saw a partial insert batch (non-atomic insert): {partial[:10]}"
        _RESULTS["isolation"]["multi_row_insert_atomic"] = True
        _RESULTS["isolation"]["insert_atomic_counts_observed"] = sorted(observed)
        _RESULTS["notes"]["insert_atomicity"] = (
            f"reader over concurrent 8192-row insert/delete only ever saw counts "
            f"{sorted(observed)} for the batch tag -> a single insert call is atomic"
        )
        assert inst.log_errors() == []
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 8. DEFECT PIN: aggregates read dead/uninitialised slots past a block boundary
# --------------------------------------------------------------------------- #

def test_DEFECT_aggregate_reads_dead_versions_past_block_boundary(db, inst):
    """DEFECT (engine, silent wrong result): once repeated all-row ``UPDATE``s push a
    table's *physical* row-versions past one 8192-row block, ``sum``/``max`` return
    silently-wrong values. With 500 rows updated in place 20 times (=> ~10500 physical
    versions, > 8192), every live row holds a == 100:

      * ``count(*)`` == 500                      (correct)
      * a full column scan sums to 50000, max 100 (correct — the data is fine)
      * ``sum(a)``  returns 50256  (WRONG: expected 50000)
      * ``max(a)``  returns 256    (a value NO row ever held — proof the aggregate reads
                                    memory outside the live rows)

    ``count(*)`` and scans apply the visibility mask; the aggregate operators read one
    uninitialised/dead slot per extra block. Nothing is logged (``log_errors`` empty), so
    the SDK cannot see it. Concurrency amplifies it (observed values up to ~6.2e9) but the
    defect reproduces deterministically single-threaded, which is what this pin uses.

    This pins the CURRENT buggy behaviour so it is reproducible; when the aggregate is
    fixed the ``sum``/``max`` assertions below will fail and this test must be updated.
    """
    name = "c_defect_agg"
    nrows, nupd, val = 500, 20, 100
    _fresh(db, name, {"id": {"type": "int"}, "a": {"type": "int"}})
    try:
        t = db.get_table(name)
        t.insert([{"id": i, "a": 0} for i in range(nrows)])
        for _ in range(nupd):
            t.update("1 = 1", {"a": val})

        # the data itself is correct via count(*) and a full scan
        assert _count(t) == nrows
        scan = _scan(t, "a")
        assert len(scan) == nrows and sum(scan) == nrows * val and max(scan) == val

        true_sum = nrows * val  # 50000
        server_sum = int(_agg(t, "sum(a)"))
        server_max = int(_agg(t, "max(a)"))

        # the smoking gun: max reports a value no row ever held (>100)
        assert server_max > val, (
            f"aggregate defect appears fixed: max(a)={server_max} <= {val} "
            f"(no phantom read) — update this pin and re-open the finding"
        )
        # and sum is wrong (over-counts by reading extra dead/uninitialised slots)
        assert server_sum != true_sum, (
            f"aggregate defect appears fixed: sum(a)={server_sum} == {true_sum} — "
            f"update this pin and re-open the finding"
        )
        assert server_sum > true_sum, f"expected over-count, got sum={server_sum} < {true_sum}"
        # silent: the engine logged nothing about returning a wrong result
        assert inst.log_errors() == []
        _RESULTS["notes"]["aggregate_defect"] = (
            f"nrows={nrows} nupd={nupd}: count={_count(t)} scan_sum={sum(scan)} scan_max={max(scan)} "
            f"| server sum(a)={server_sum} max(a)={server_max} (expected {true_sum}/{val})"
        )
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 9. documented secondary-index range race (RcuMultiMap::range unlocked iteration)
# --------------------------------------------------------------------------- #

def test_secondary_index_range_race_survives(db, inst):
    """MACOS_VERIFICATION.md documents that ``RcuMultiMap::range`` iterates a
    ``std::multimap`` unlocked while ``Insert`` mutates it — reachable when an indexing
    thread appends to the in-memory secondary index while a query thread runs a range
    filter (``RangeQueryInner``). This exercises exactly that: writers continuously insert
    into a secondary-indexed int column while readers run overlapping range filters, for a
    bounded time. It asserts the observable envelope — no crash, no hang, no out-of-range
    row leaking from a range query, no server-side error — and records how hard it pushed.

    The underlying data race is real UB in the source; this could not turn it into an
    observable failure on the release binary (reported as such), so the value here is the
    negative bound plus a stress reproduction a future TSAN build can reuse."""
    name = "c_secidx"
    import infinity.index as idx
    _fresh(db, name, {"id": {"type": "int"}, "k": {"type": "int"}})
    try:
        db.get_table(name).create_index(
            "si", idx.IndexInfo("k", idx.IndexType.Secondary), ConflictType.Error)
        stop = threading.Event()
        errors: list = []
        violations: list = []
        reads = [0]
        writes = [0]
        rng = random.Random(20240903)

        def writer(wid):
            conn = inst.connect()
            try:
                wt = conn.get_database("default_db").get_table(name)
                base, n = wid * 1_000_000, 0
                while not stop.is_set():
                    wt.insert([{"id": base + n + i, "k": (base + n + i) % 5000} for i in range(200)])
                    n += 200; writes[0] += 200
            except Exception as exc:  # noqa: BLE001
                errors.append(("writer", repr(exc)))
            finally:
                conn.disconnect()

        def reader(rid):
            local = random.Random(rid * 99 + 1)
            conn = inst.connect()
            try:
                rt = conn.get_database("default_db").get_table(name)
                while not stop.is_set():
                    lo = local.randint(0, 4000); hi = lo + local.randint(1, 900)
                    df, _ = rt.output(["k"]).filter(f"k >= {lo} and k <= {hi}").to_df()
                    reads[0] += 1
                    for kv in df["k"].to_list():
                        if kv < lo or kv > hi:
                            violations.append((lo, hi, kv)); break
            except Exception as exc:  # noqa: BLE001
                errors.append(("reader", repr(exc)))
            finally:
                conn.disconnect()

        threads = [threading.Thread(target=writer, args=(i,), name=f"wr{i}") for i in range(3)]
        threads += [threading.Thread(target=reader, args=(i,), name=f"rd{i}") for i in range(4)]
        for th in threads:
            th.start()
        time.sleep(12)
        stop.set()
        for th in threads:
            th.join(timeout=JOIN_TIMEOUT)
        hung = [th.name for th in threads if th.is_alive()]

        assert hung == [], f"secondary-index race hung threads (blocker): {hung}"
        assert inst.pid() is not None, "server died during secondary-index race (blocker)"
        assert errors == [], f"secondary-index race raised: {errors[:6]}"
        assert violations == [], f"range query returned out-of-range rows: {violations[:6]}"
        assert inst.log_errors() == []
        _RESULTS["notes"]["secondary_index_race"] = (
            f"12s stress: {reads[0]} range reads over ~{writes[0]} inserted rows, "
            "3 writers + 4 readers on a Secondary-indexed column: no crash, no hang, "
            "no out-of-range leak, no logged error (documented UB not observably triggered)"
        )
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# 10. throughput scaling: concurrent insert rows/sec at 1/2/4/8/16 threads
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("nthreads", [1, 2, 4, 8, 16])
def test_insert_throughput_scaling(db, inst, nthreads):
    """Insert a fixed total of 48000 rows split across ``nthreads`` threads and record
    rows/sec. Connections and row batches are built up front and all threads are released
    from a barrier, so the timed window covers only the inserts (not connection setup),
    making the number a fair per-thread-count throughput. Fixed total work means rows/sec
    should rise with threads if inserts scale; a flat or falling curve is the finding.
    Correctness is still asserted (exactly the total rows, unique ids).

    Observed here (see build/eval/results/concurrency.json): throughput saturates at ~2
    threads and is flat 2->16 — concurrent inserts do not scale past 2 threads. A separate
    no_sync run showed the same shape, so the ceiling is a write-path serialization, not
    the WAL fsync. Reported as a perf finding."""
    name = f"c_tput_{nthreads}"
    total = 48000
    per = total // nthreads
    _fresh(db, name, {"id": {"type": "int"}})
    conns = [inst.connect() for _ in range(nthreads)]
    try:
        tabs = [c.get_database("default_db").get_table(name) for c in conns]
        batches = []
        for tid in range(nthreads):
            rows = [{"id": tid * per + i} for i in range(per)]
            batches.append([rows[i:i + INSERT_BATCH] for i in range(0, len(rows), INSERT_BATCH)])
        barrier = threading.Barrier(nthreads + 1)
        errs: list = []

        def worker(tid):
            try:
                barrier.wait()
                for b in batches[tid]:
                    tabs[tid].insert(b)
            except Exception as exc:  # noqa: BLE001
                errs.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,), name=f"tp{i}") for i in range(nthreads)]
        for th in threads:
            th.start()
        barrier.wait()
        t0 = time.monotonic()
        for th in threads:
            th.join(timeout=JOIN_TIMEOUT)
        elapsed = time.monotonic() - t0
        assert not any(th.is_alive() for th in threads), "throughput worker hung"
        assert errs == [], f"throughput worker raised: {errs[:4]}"

        rps = (per * nthreads) / elapsed if elapsed > 0 else 0.0
        _RESULTS["throughput_rows_per_sec"][str(nthreads)] = round(rps, 1)

        ids = _scan(db.get_table(name), "id")
        assert len(ids) == per * nthreads and len(set(ids)) == per * nthreads
        assert inst.log_errors() == []
    finally:
        for c in conns:
            try:
                c.disconnect()
            except Exception:
                pass
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# final: the server logged no error/critical/fatal line across the whole run
# --------------------------------------------------------------------------- #

def test_zzz_server_log_clean(inst):
    errs = inst.log_errors()
    assert errs == [], "server logged error/critical/fatal lines:\n" + "\n".join(errs[:20])
