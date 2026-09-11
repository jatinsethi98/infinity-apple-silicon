"""End-to-end verification of the secondary index, filtering and range-query surface
on macOS arm64.

Focus (per the evaluation brief):

* A regression test for commit ccc58a3da, which made ``RcuMultiMap::range`` hold
  ``dirty_lock_`` across the whole traversal so that a query thread walking the in-memory
  secondary index cannot observe a red-black tree mid-rebalance while an indexing thread
  inserts into it. ``test_concurrent_insert_vs_range_query`` drives exactly that path
  (concurrent INSERTs + ``filter('x < v < y')`` INDEX SCANs against one in-mem segment,
  staying under mem_index_capacity=65536) and asserts every range query returns exactly
  the rows that satisfy it, with no crash, no hang, and no server-side error. It also
  reports throughput, because the fix serialises range queries against inserts.

* The whole filter expression surface: comparison operators, IN/NOT IN, IS NULL,
  LIKE, boolean logic, arithmetic, strings/unicode, dates; index_scan-vs-table_scan
  selection proven with EXPLAIN; Secondary and SecondaryFunctional indexes; and range
  queries crossing the in-mem-index dump.

* Negative cases as first-class tests, looking for both silent-accept and crash/corruption.

Every server-mutating group also checks ``inst.log_errors`` — a clean client result over a
server-side error is a defect the SDK cannot see.

Run:  uv run pytest test/eval_macos/test_secondary_filter_e2e.py -v
"""

from __future__ import annotations

import threading
import time

import pytest

import harness
from infinity import index
from infinity.common import ConflictType, InfinityException
from infinity.errors import ErrorCode

SEC = index.IndexType.Secondary
SECF = index.IndexType.SecondaryFunctional


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-secondary")
    it.start()
    try:
        yield it
    finally:
        it.stop()


@pytest.fixture(autouse=True)
def _ensure_alive(inst):
    """Bring the server back if a previous test crashed it.

    One test in this suite (int32 overflow) SIGSEGVs the server on purpose to document
    an engine defect; without this, every subsequent test would fail with a transport
    error that has nothing to do with what it is testing.
    """
    if inst.pid() is None:
        inst.start(fresh=True)
    yield


# ------------------------------------------------------------------------------ helpers

def _table(inst, name, schema, rows=None, indexes=None):
    conn, db = inst.db()
    db.drop_table(name, ConflictType.Ignore)
    t = db.create_table(name, schema, ConflictType.Error)
    if rows:
        t.insert(rows)
    for idx_name, target, itype in (indexes or []):
        t.create_index(idx_name, index.IndexInfo(target, itype), ConflictType.Error)
    return conn, db, t


def _vals(t, cols, filt, col):
    r, _ = t.output(cols).filter(filt).to_pl()
    return sorted(r[col].to_list())


def _plan_kind(t, filt, cols=("id",)):
    e = t.output(list(cols)).filter(filt).explain()
    text = "\n".join(e.to_series().to_list())
    if "INDEX SCAN" in text:
        return "INDEX SCAN"
    if "TABLE SCAN" in text:
        return "TABLE SCAN"
    return "OTHER:\n" + text


# ------------------------------------------------------------------ filter surface (+)

def test_comparison_operators(inst):
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "cmp",
        {"id": {"type": "int"}, "v": {"type": "int"}},
        rows=[{"id": i, "v": i} for i in range(20)],
        indexes=[("idx_v", "v", SEC)],
    )
    try:
        assert _vals(t, ["v"], "v = 7", "v") == [7]
        assert _vals(t, ["v"], "v < 3", "v") == [0, 1, 2]
        assert _vals(t, ["v"], "v <= 3", "v") == [0, 1, 2, 3]
        assert _vals(t, ["v"], "v > 16", "v") == [17, 18, 19]
        assert _vals(t, ["v"], "v >= 16", "v") == [16, 17, 18, 19]
        assert _vals(t, ["v"], "v > 5 AND v < 10", "v") == [6, 7, 8, 9]
        # != must return the complement of the single matching row.
        assert _vals(t, ["v"], "v != 7", "v") == [i for i in range(20) if i != 7]
    finally:
        db.drop_table("cmp", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_in_notin_empty(inst):
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "inq",
        {"id": {"type": "int"}, "v": {"type": "int"}},
        rows=[{"id": i, "v": i} for i in range(20)],
        indexes=[("idx_v", "v", SEC)],
    )
    try:
        assert _vals(t, ["v"], "v IN (1, 3, 5)", "v") == [1, 3, 5]
        assert _vals(t, ["v"], "v NOT IN (1, 3, 5)", "v") == [i for i in range(20) if i not in (1, 3, 5)]
        # empty IN list must match nothing (not everything, not an error).
        assert _vals(t, ["v"], "v IN ()", "v") == []
    finally:
        db.drop_table("inq", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_is_null_and_not_null(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    db.drop_table("nul", ConflictType.Ignore)
    # Insert some rows with an explicit NULL in the indexed column.
    t = db.create_table("nul", {"id": {"type": "int"}, "v": {"type": "int"}}, ConflictType.Error)
    t.insert([{"id": 0, "v": 10}, {"id": 1, "v": 20}])
    t.insert([{"id": 2, "v": None}])
    t.create_index("idx_v", index.IndexInfo("v", SEC), ConflictType.Error)
    try:
        assert _vals(t, ["id"], "v IS NULL", "id") == [2]
        assert _vals(t, ["id"], "v IS NOT NULL", "id") == [0, 1]
    finally:
        db.drop_table("nul", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_like(inst):
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "lk",
        {"id": {"type": "int"}, "s": {"type": "varchar"}},
        rows=[{"id": i, "s": f"row{i:02d}"} for i in range(20)],
        indexes=[("idx_s", "s", SEC)],
    )
    try:
        assert _vals(t, ["id"], "s LIKE 'row0%'", "id") == list(range(10))
        assert _vals(t, ["id"], "s LIKE '%'", "id") == list(range(20))
        assert _vals(t, ["id"], "s LIKE 'row_5'", "id") == [5, 15]  # single-char wildcard
        assert _vals(t, ["id"], "s = 'row07'", "id") == [7]
    finally:
        db.drop_table("lk", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_boolean_logic_and_parens(inst):
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "bl",
        {"id": {"type": "int"}, "v": {"type": "int"}},
        rows=[{"id": i, "v": i} for i in range(20)],
        indexes=[("idx_v", "v", SEC)],
    )
    try:
        assert _vals(t, ["v"], "(v > 3 AND v < 8) OR v = 15", "v") == [4, 5, 6, 7, 15]
        assert _vals(t, ["v"], "NOT (v < 18)", "v") == [18, 19]
        assert _vals(t, ["v"], "v < 3 OR v > 17", "v") == [0, 1, 2, 18, 19]
        assert _vals(t, ["v"], "((v >= 5) AND (v <= 9)) AND NOT (v = 7)", "v") == [5, 6, 8, 9]
    finally:
        db.drop_table("bl", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_arithmetic_in_predicate(inst):
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "ar",
        {"id": {"type": "int"}, "v": {"type": "int"}},
        rows=[{"id": i, "v": i} for i in range(20)],
        indexes=[("idx_v", "v", SEC)],
    )
    try:
        assert _vals(t, ["v"], "v + 1 > 10", "v") == list(range(10, 20))
        assert _vals(t, ["v"], "v * 2 = 10", "v") == [5]
        assert _vals(t, ["v"], "v - 3 = 0", "v") == [3]
    finally:
        db.drop_table("ar", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_string_unicode_and_sql_injection_is_data(inst):
    """A string predicate must be treated as data, never parsed as SQL."""
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "u",
        {"id": {"type": "int"}, "s": {"type": "varchar"}},
        rows=[
            {"id": 1, "s": "naïve café"},
            {"id": 2, "s": "o'brien"},
            {"id": 3, "s": "x' OR '1'='1"},
            {"id": 4, "s": "日本語"},
            {"id": 5, "s": "plain"},
        ],
        indexes=[("idx_s", "s", SEC)],
    )
    try:
        assert _vals(t, ["id"], "s = 'naïve café'", "id") == [1]
        assert _vals(t, ["id"], "s = '日本語'", "id") == [4]
        # The injection-shaped string matches only the row whose *data* is that string.
        r, _ = t.output(["id", "s"]).filter("s = 'x'' OR ''1''=''1'").to_pl()
        assert r["id"].to_list() == [3]
        assert r["s"].to_list() == ["x' OR '1'='1"]
        # An embedded single quote, escaped by doubling, is one row of data.
        assert _vals(t, ["id"], "s = 'o''brien'", "id") == [2]
    finally:
        db.drop_table("u", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_date_and_timestamp_comparison(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    db.drop_table("dt", ConflictType.Ignore)
    t = db.create_table(
        "dt",
        {"id": {"type": "int"}, "d": {"type": "date"}, "ts": {"type": "timestamp"}},
        ConflictType.Error,
    )
    t.insert([
        {"id": 1, "d": "2024-01-01", "ts": "2024-01-01 00:00:00"},
        {"id": 2, "d": "2024-06-15", "ts": "2024-06-15 12:30:00"},
        {"id": 3, "d": "2025-01-01", "ts": "2025-01-01 08:00:00"},
    ])
    t.create_index("idx_d", index.IndexInfo("d", SEC), ConflictType.Error)
    t.create_index("idx_ts", index.IndexInfo("ts", SEC), ConflictType.Error)
    try:
        assert _vals(t, ["id"], "d > '2024-03-01'", "id") == [2, 3]
        assert _vals(t, ["id"], "d >= '2024-06-15' AND d < '2025-01-01'", "id") == [2]
        assert _vals(t, ["id"], "ts > '2024-06-15 12:00:00'", "id") == [2, 3]
    finally:
        db.drop_table("dt", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


# --------------------------------------------------------------- plan selection (EXPLAIN)

def test_index_scan_vs_table_scan_selection(inst):
    """Prove the secondary index is actually chosen for equality and range, and that
    without an index the same query falls back to a table scan. Also pins the CURRENT
    (observed) behaviour that IN and LIKE do NOT use the secondary index -- a built index
    that is never chosen for those predicates."""
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "pl",
        {"id": {"type": "int"}, "v": {"type": "int"}, "s": {"type": "varchar"}},
        rows=[{"id": i, "v": i, "s": f"row{i:02d}"} for i in range(20)],
        indexes=[("idx_v", "v", SEC), ("idx_s", "s", SEC)],
    )
    try:
        # Index IS used for equality and range on an indexed column.
        assert _plan_kind(t, "v = 7") == "INDEX SCAN"
        assert _plan_kind(t, "v > 5 AND v < 10") == "INDEX SCAN"
        assert _plan_kind(t, "s = 'row07'") == "INDEX SCAN"

        # Observed limitation: IN and LIKE do NOT use the secondary index.
        assert _plan_kind(t, "v IN (1, 3, 5)") == "TABLE SCAN"
        assert _plan_kind(t, "s LIKE 'row0%'") == "TABLE SCAN"
    finally:
        db.drop_table("pl", ConflictType.Ignore)

    # A table with no index must table-scan.
    _, _, tn = _table(
        inst, "pl2",
        {"id": {"type": "int"}, "v": {"type": "int"}},
        rows=[{"id": i, "v": i} for i in range(20)],
    )
    try:
        assert _plan_kind(tn, "v > 5") == "TABLE SCAN"
    finally:
        db.drop_table("pl2", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_secondary_functional_index_query_and_plan(inst):
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "fn",
        {"c1": {"type": "int"}, "c2": {"type": "varchar"}},
        rows=[
            {"c1": 1, "c2": "hello world"},
            {"c1": 2, "c2": "thank you"},
            {"c1": 3, "c2": "hello world"},
            {"c1": 4, "c2": "thank you"},
        ],
        indexes=[("fidx", "sqrt(c1)", SECF), ("sidx", "substring(c2, 0, 5)", SECF)],
    )
    try:
        assert _vals(t, ["c1"], "sqrt(c1) > 1", "c1") == [2, 3, 4]
        assert _vals(t, ["c1"], "sqrt(c1) > 2", "c1") == []  # max sqrt(4)=2, strict > excludes
        assert _vals(t, ["c1"], "substring(c2, 0, 5) = 'hello'", "c1") == [1, 3]
        # The functional index must actually be chosen.
        assert _plan_kind(t, "sqrt(c1) > 1", cols=["c1"]) == "INDEX SCAN"
    finally:
        db.drop_table("fn", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_range_across_mem_index_dump(inst):
    """A range query must return the same rows before and after the in-mem index is
    dumped to an on-disk chunk."""
    off = inst.log_offset()
    conn, db, t = _table(
        inst, "dmp",
        {"id": {"type": "int"}, "v": {"type": "int"}},
        indexes=[("idx_v", "v", SEC)],
    )
    t.insert([{"id": i, "v": i % 100} for i in range(500)])
    try:
        before = sorted(t.output(["id"]).filter("v > 40 AND v < 60").to_pl()[0]["id"].to_list())
        t.dump_index("idx_v")  # force the in-mem segment out to disk
        after = sorted(t.output(["id"]).filter("v > 40 AND v < 60").to_pl()[0]["id"].to_list())
        expect = sorted(i for i in range(500) if 40 < (i % 100) < 60)
        assert before == expect, f"before dump: {len(before)} != expected {len(expect)}"
        assert after == expect, f"after dump: {len(after)} != expected {len(expect)}"
    finally:
        db.drop_table("dmp", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


# ----------------------------------------------- CONCURRENCY: the ccc58a3da regression

def test_concurrent_insert_vs_range_query(inst):
    """Regression for ccc58a3da: drive concurrent INSERTs (indexing thread appends to the
    in-mem secondary index) against concurrent range queries (query thread walks the same
    RcuMultiMap via SecondaryIndexInMem::RangeQueryInner). Before the fix, range() copied
    the dirty_map_ pointer under the lock and then iterated it unlocked while Insert()
    rebalanced the same std::multimap -- a red-black-tree walk racing a rebalance.

    Invariants asserted for every range query, under load:
      * every returned value satisfies the predicate (LO < v < HI);
      * no duplicate rows;
      * every returned value is a value that was actually inserted;
    and after quiescence, the range returns EXACTLY the satisfying set.
    No crash, no hang, no server-side error. Throughput is reported.
    """
    off = inst.log_offset()
    conn, db = inst.db()
    db.drop_table("conc", ConflictType.Ignore)
    t = db.create_table("conc", {"id": {"type": "int"}, "v": {"type": "int"}}, ConflictType.Error)
    # Build the index on the EMPTY table so ALL rows flow through the in-mem append path
    # (RcuMultiMap::Insert), maximising overlap with range().
    t.create_index("idx_v", index.IndexInfo("v", SEC), ConflictType.Error)

    TOTAL = 50000          # < mem_index_capacity (65536): stays in the in-mem index
    BATCH = 10             # small batches -> many Insert calls -> more interleavings
    N_INS = 5
    N_QRY = 6
    LO, HI = 100, 40000    # a wide window so each range() walks a large slice of the map

    errors: list = []
    query_count = [0]
    insert_count = [0]
    stop = threading.Event()
    universe = set(range(TOTAL))

    def inserter(worker: int):
        c = inst.connect()
        tt = c.get_database("default_db").get_table("conc")
        try:
            i = worker * BATCH
            stride = N_INS * BATCH
            while i < TOTAL:
                block = [{"id": j, "v": j} for j in range(i, min(i + BATCH, TOTAL))]
                tt.insert(block)
                insert_count[0] += len(block)
                i += stride
        except Exception as e:  # noqa: BLE001
            errors.append(("insert", worker, type(e).__name__, str(e)[:120]))
        finally:
            c.disconnect()

    def querier(worker: int):
        c = inst.connect()
        tt = c.get_database("default_db").get_table("conc")
        try:
            local = 0
            while not stop.is_set():
                r, _ = tt.output(["v"]).filter(f"v > {LO} AND v < {HI}").to_pl()
                vals = r["v"].to_list()
                local += 1
                if vals:
                    lo_v, hi_v = min(vals), max(vals)
                    if lo_v <= LO or hi_v >= HI:
                        errors.append(("range-bounds", worker, lo_v, hi_v))
                    if len(set(vals)) != len(vals):
                        errors.append(("duplicates", worker, len(vals), len(set(vals))))
                    if not set(vals).issubset(universe):
                        errors.append(("foreign-values", worker, len(set(vals) - universe)))
                    # value count can never exceed the number of satisfying keys
                    if len(vals) > (HI - LO - 1):
                        errors.append(("overcount", worker, len(vals)))
            query_count[0] += local
        except Exception as e:  # noqa: BLE001
            errors.append(("query", worker, type(e).__name__, str(e)[:120]))
        finally:
            c.disconnect()

    t0 = time.monotonic()
    ins = [threading.Thread(target=inserter, args=(w,)) for w in range(N_INS)]
    qrs = [threading.Thread(target=querier, args=(w,)) for w in range(N_QRY)]
    for th in qrs:
        th.start()
    for th in ins:
        th.start()
    for th in ins:
        th.join(timeout=180)
    # if any inserter is still alive here, we hung
    hung_inserter = any(th.is_alive() for th in ins)
    stop.set()
    for th in qrs:
        th.join(timeout=30)
    hung_querier = any(th.is_alive() for th in qrs)
    dt = time.monotonic() - t0

    try:
        assert not hung_inserter, "inserter thread did not finish within 180s (possible hang/deadlock)"
        assert not hung_querier, "querier thread did not finish within 30s (possible hang/deadlock)"
        assert inst.pid() is not None, "server process died during concurrent insert/range load"
        assert errors == [], f"concurrency invariant violations: {errors[:10]} (total {len(errors)})"

        # Quiescent exact check: the range must now equal the full satisfying set.
        got = sorted(t.output(["v"]).filter(f"v > {LO} AND v < {HI}").to_pl()[0]["v"].to_list())
        expect = [v for v in range(TOTAL) if LO < v < HI]
        assert got == expect, f"final range wrong: got {len(got)} rows, expected {len(expect)}"

        total_rows = t.output(["v"]).to_pl()[0].height
        assert total_rows == TOTAL, f"row count after inserts: {total_rows} != {TOTAL}"

        server_errs = inst.log_errors(off)
        assert server_errs == [], f"server logged errors during concurrency: {server_errs[:5]}"

        qps = query_count[0] / dt if dt else 0.0
        ips = insert_count[0] / dt if dt else 0.0
        print(
            f"\n[concurrency] {query_count[0]} range queries + {insert_count[0]} rows inserted "
            f"in {dt:.1f}s -> {qps:.0f} queries/s, {ips:.0f} rows/s "
            f"({N_INS} inserters, {N_QRY} queriers, window ({LO},{HI}))"
        )
        harness.record("secondary_concurrency", {
            "queries": query_count[0], "rows": insert_count[0], "seconds": round(dt, 2),
            "queries_per_s": round(qps, 1), "rows_per_s": round(ips, 1),
            "n_inserters": N_INS, "n_queriers": N_QRY, "errors": len(errors),
        })
    finally:
        db.drop_table("conc", ConflictType.Ignore)
        conn.disconnect()


# ------------------------------------------------------------------- NEGATIVE cases

def test_neg_missing_column(inst):
    conn, db, t = _table(inst, "n1", {"id": {"type": "int"}, "v": {"type": "int"}},
                         rows=[{"id": 0, "v": 0}], indexes=[("idx_v", "v", SEC)])
    off = inst.log_offset()
    try:
        with pytest.raises(InfinityException) as ei:
            t.output(["id"]).filter("nope > 5").to_pl()
        assert int(ei.value.error_code) == int(ErrorCode.COLUMN_NOT_EXIST), ei.value.error_msg
        assert inst.pid() is not None
    finally:
        db.drop_table("n1", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_type_mismatch_int_column_vs_string(inst):
    """Comparing an int column against a non-numeric string literal. This is SILENTLY
    ACCEPTED and returns zero rows rather than a DATA_TYPE_MISMATCH error -- documented
    here so the behaviour is pinned. A numeric string IS coerced. The point of the test
    is that neither shape crashes the server or logs an error."""
    conn, db, t = _table(
        inst, "n2", {"id": {"type": "int"}, "v": {"type": "int"}},
        rows=[{"id": i, "v": i} for i in range(20)], indexes=[("idx_v", "v", SEC)],
    )
    off = inst.log_offset()
    try:
        # non-numeric string -> silently 0 rows (no type error raised)
        r, _ = t.output(["id"]).filter("v = 'abc'").to_pl()
        assert len(r) == 0
        r, _ = t.output(["id"]).filter("v > 'abc'").to_pl()
        assert len(r) == 0
        # numeric string -> coerced to the column type
        assert _vals(t, ["v"], "v > '5'", "v") == list(range(6, 20))
        assert _vals(t, ["v"], "v = '7'", "v") == [7]
        assert inst.pid() is not None
    finally:
        db.drop_table("n2", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_unterminated_string_literal(inst):
    conn, db, t = _table(inst, "n3", {"id": {"type": "int"}, "s": {"type": "varchar"}},
                         rows=[{"id": 0, "s": "a"}], indexes=[("idx_s", "s", SEC)])
    off = inst.log_offset()
    try:
        # sqlglot rejects the malformed literal client-side; nothing reaches the server.
        with pytest.raises(Exception) as ei:
            t.output(["id"]).filter("s = 'abc").to_pl()
        assert not isinstance(ei.value, InfinityException) or ei.value.error_code != ErrorCode.OK
        assert inst.pid() is not None
    finally:
        db.drop_table("n3", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_between_raises_sdk_recursion(inst):
    """BETWEEN -- a core range predicate named in the brief -- is not handled by the
    thrift SDK's expression walker. Unhandled sqlglot nodes fall through to
    ``traverse_conditions(cons[1])`` (utils.py:638) and recurse forever, so BETWEEN
    raises RecursionError CLIENT-SIDE and never reaches the server. SDK defect."""
    conn, db, t = _table(inst, "n4", {"id": {"type": "int"}, "v": {"type": "int"}},
                         rows=[{"id": i, "v": i} for i in range(20)], indexes=[("idx_v", "v", SEC)])
    off = inst.log_offset()
    try:
        with pytest.raises(RecursionError):
            t.output(["v"]).filter("v BETWEEN 5 AND 9").to_pl()
        assert inst.pid() is not None, "server must survive a client-side SDK failure"
    finally:
        db.drop_table("n4", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_eq_null_raises_sdk_recursion(inst):
    """``v = NULL`` (three-valued-logic question in the brief) is not expressible through
    the SDK: the NULL literal is an unhandled node and recurses forever the same way.
    So the answer is neither 'false' nor a server error -- it is a client-side
    RecursionError. IS NULL / IS NOT NULL are the supported spellings (tested above)."""
    conn, db, t = _table(inst, "n5", {"id": {"type": "int"}, "v": {"type": "int"}},
                         rows=[{"id": i, "v": i} for i in range(20)], indexes=[("idx_v", "v", SEC)])
    off = inst.log_offset()
    try:
        with pytest.raises(RecursionError):
            t.output(["v"]).filter("v = NULL").to_pl()
        assert inst.pid() is not None
    finally:
        db.drop_table("n5", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_range_lower_exceeds_upper(inst):
    conn, db, t = _table(inst, "n6", {"id": {"type": "int"}, "v": {"type": "int"}},
                         rows=[{"id": i, "v": i} for i in range(20)], indexes=[("idx_v", "v", SEC)])
    off = inst.log_offset()
    try:
        # empty range -> zero rows, no error, no crash.
        assert _vals(t, ["v"], "v > 10 AND v < 5", "v") == []
        assert inst.pid() is not None
    finally:
        db.drop_table("n6", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_in_empty_and_huge_list(inst):
    conn, db, t = _table(inst, "n7", {"id": {"type": "int"}, "v": {"type": "int"}},
                         rows=[{"id": i, "v": i} for i in range(20)], indexes=[("idx_v", "v", SEC)])
    off = inst.log_offset()
    try:
        assert _vals(t, ["v"], "v IN ()", "v") == []
        # a 5000-element IN list, only a few of which exist
        huge = ", ".join(str(x) for x in range(5000))
        r, _ = t.output(["v"]).filter(f"v IN ({huge})").to_pl()
        assert sorted(r["v"].to_list()) == list(range(20))
        assert inst.pid() is not None
    finally:
        db.drop_table("n7", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_like_pathological_wildcards(inst):
    conn, db, t = _table(inst, "n8", {"id": {"type": "int"}, "s": {"type": "varchar"}},
                         rows=[{"id": i, "s": f"aaa{i:02d}bbb"} for i in range(20)],
                         indexes=[("idx_s", "s", SEC)])
    off = inst.log_offset()
    try:
        # many consecutive wildcards must not hang or blow up.
        r, _ = t.output(["id"]).filter("s LIKE '%%%%%a%%%%%'").to_pl()
        assert sorted(r["id"].to_list()) == list(range(20))
        r, _ = t.output(["id"]).filter("s LIKE '%zzz%'").to_pl()
        assert len(r) == 0
        assert inst.pid() is not None
    finally:
        db.drop_table("n8", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_secondary_index_on_embedding_rejected(inst):
    conn, db = inst.db()
    db.drop_table("n9", ConflictType.Ignore)
    t = db.create_table("n9", {"id": {"type": "int"}, "e": {"type": "vector,4,float"}}, ConflictType.Error)
    off = inst.log_offset()
    try:
        with pytest.raises(InfinityException) as ei:
            t.create_index("bad", index.IndexInfo("e", SEC), ConflictType.Error)
        assert int(ei.value.error_code) == int(ErrorCode.INVALID_INDEX_DEFINITION), ei.value.error_msg
        assert inst.pid() is not None
    finally:
        db.drop_table("n9", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_duplicate_index_name(inst):
    conn, db, t = _table(inst, "n10", {"id": {"type": "int"}, "v": {"type": "int"}},
                         rows=[{"id": 0, "v": 0}], indexes=[("idx_v", "v", SEC)])
    off = inst.log_offset()
    try:
        with pytest.raises(InfinityException) as ei:
            t.create_index("idx_v", index.IndexInfo("v", SEC), ConflictType.Error)
        assert int(ei.value.error_code) == int(ErrorCode.DUPLICATE_INDEX_NAME), ei.value.error_msg
        assert inst.pid() is not None
    finally:
        db.drop_table("n10", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_neg_int32_overflow_bound_crashes_server(inst):
    """ENGINE DEFECT (crash). A filter whose bound overflows the indexed INTEGER column
    -- ``v > 2147483647`` (INT32_MAX) or beyond -- SIGSEGVs the server, but ONLY when a
    Secondary index exists: the identical query on a non-indexed column returns 0 rows
    and the server survives. The crash is in the index-scan bound construction reached
    from InfinityThriftService::Select. ``v > 2147483646`` is fine, so the boundary is
    exactly the type's max.

    This test asserts the CORRECT behaviour (survive, return 0 rows); it therefore FAILS
    while the defect is present. It restores the server afterwards so teardown is clean.
    Kept LAST so its crash cannot disturb the other tests.
    """
    conn, db, t = _table(
        inst, "ovf", {"id": {"type": "int"}, "v": {"type": "int"}},
        rows=[{"id": i, "v": i} for i in range(20)], indexes=[("idx_v", "v", SEC)],
    )
    crashed = False
    rows = None
    try:
        try:
            r, _ = t.output(["id"]).filter("v > 2147483647").to_pl()
            rows = len(r)
        except Exception:  # noqa: BLE001 - transport error is the symptom of the crash
            pass
        # The process may still be dying when the client error surfaces; give it a moment
        # so crash detection is deterministic (avoids a flaky pid()-not-yet-None race).
        for _ in range(50):
            if inst.pid() is None:
                break
            time.sleep(0.1)
        crashed = inst.pid() is None
    finally:
        # Restore for a clean module teardown regardless of outcome.
        if inst.pid() is None:
            inst.start(fresh=True)
        else:
            try:
                conn.disconnect()
            except Exception:  # noqa: BLE001 - connection may be dead post-crash
                pass

    assert not crashed, (
        "SERVER CRASHED (SIGSEGV) on filter('v > 2147483647') against an INTEGER column "
        "with a Secondary index. Expected: survive and return 0 rows (the value is out of "
        "the column range). The same filter on a non-indexed column is safe -- defect is in "
        "the secondary index-scan bound computation."
    )
    assert rows == 0
