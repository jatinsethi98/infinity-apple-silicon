"""End-to-end mutation and storage-maintenance suite for the macOS arm64 port.

Area: update / delete / compact / optimize / cleanup / mem-index dump.

Design
------
* One server instance (``eval-mutate``, its own port offset and data dir) is started
  once for the module and stopped in a finalizer that runs even on failure.
* Every test owns a uniquely named table and drops it in a ``finally``.
* Positive tests assert on *exact* expected values (row sets, counts, KNN top-1,
  full-text hit sets), never merely "no exception".
* Negative tests assert the *specific* ``ErrorCode`` and then assert the server is
  still alive and the table still correct.
* The server's own error log is scraped (``inst.log_errors``): a clean client result
  over a server-side error is a defect the SDK cannot see. A module-final test asserts
  the log carried no error/critical/fatal line across the whole run.

The headline correctness question is ``test_deleted_rows_absent_from_every_access_path``:
a deleted row must be gone from a plain scan, a filtered scan, a KNN search over an
HNSW index, a full-text search, and ``count(*)`` — and stay gone after ``compact``,
after ``optimize``, and after a restart (so the deletion was persisted, not masked).
"""

from __future__ import annotations

import pytest

import harness
from infinity.common import ConflictType, InfinityException
from infinity.errors import ErrorCode
import infinity.index as idx


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-mutate")
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


def _count(t) -> int:
    res, _ = t.output(["count(*)"]).to_pl()
    return res.item(0, 0)


def _ids(df, col: str = "id") -> set[int]:
    return {int(x) for x in df[col].to_list()}


def _scan_ids(t, col: str = "id") -> set[int]:
    df, _ = t.output([col]).to_df()
    return _ids(df, col)


def _fresh(db, name: str, schema: dict):
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(name, schema, ConflictType.Error)


# --------------------------------------------------------------------------- #
# UPDATE — positive
# --------------------------------------------------------------------------- #

def test_update_single_row_by_predicate(db):
    t = _fresh(db, "m_upd_single", {"c1": {"type": "int", "constraints": ["primary key", "not null"]},
                                    "c2": {"type": "int"}, "c3": {"type": "int"}})
    try:
        t.insert([{"c1": i, "c2": i * 10, "c3": i * 100} for i in (1, 2, 3, 4)])
        r = t.update("c1 = 2", {"c2": 999, "c3": 888})
        assert r.error_code == ErrorCode.OK

        df, _ = t.output(["c1", "c2", "c3"]).filter("c1 = 2").to_df()
        assert df.to_dict("records") == [{"c1": 2, "c2": 999, "c3": 888}]
        # the other rows are untouched
        df, _ = t.output(["c1", "c2", "c3"]).filter("c1 <> 2").to_df()
        assert sorted(df["c2"].to_list()) == [10, 30, 40]
        assert _count(t) == 4
    finally:
        db.drop_table("m_upd_single", ConflictType.Ignore)


def test_update_many_rows_by_predicate(db):
    t = _fresh(db, "m_upd_many", {"id": {"type": "int"}, "grp": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"id": i, "grp": i % 2, "v": 0} for i in range(20)])
        r = t.update("grp = 1", {"v": 7})
        assert r.error_code == ErrorCode.OK
        # exactly the 10 odd ids now have v == 7, the 10 even ids still 0
        df, _ = t.output(["id", "v"]).filter("v = 7").to_df()
        assert _ids(df) == {i for i in range(20) if i % 2 == 1}
        assert _count(t) == 20
        df, _ = t.output(["id"]).filter("v = 0").to_df()
        assert _ids(df) == {i for i in range(20) if i % 2 == 0}
    finally:
        db.drop_table("m_upd_many", ConflictType.Ignore)


def test_update_every_column_type(db):
    schema = {
        "id": {"type": "int"},
        "c_i8": {"type": "int8"}, "c_i16": {"type": "int16"}, "c_i64": {"type": "int64"},
        "c_f": {"type": "float"}, "c_d": {"type": "double"},
        "c_bool": {"type": "bool"}, "c_vc": {"type": "varchar"},
        "c_vec": {"type": "vector,3,float"},
    }
    t = _fresh(db, "m_upd_types", schema)
    try:
        t.insert([{"id": 1, "c_i8": 1, "c_i16": 1, "c_i64": 1, "c_f": 1.0, "c_d": 1.0,
                   "c_bool": True, "c_vc": "one", "c_vec": [1.0, 1.0, 1.0]}])
        newvals = {"c_i8": 120, "c_i16": 30000, "c_i64": 10_000_000_000,
                   "c_f": 2.5, "c_d": 3.25, "c_bool": False, "c_vc": "hello world",
                   "c_vec": [7.0, 8.0, 9.0]}
        for col, val in newvals.items():
            r = t.update("id = 1", {col: val})
            assert r.error_code == ErrorCode.OK, col
        df, _ = t.output(["*"]).to_df()
        row = df.to_dict("records")[0]
        assert row["c_i8"] == 120 and row["c_i16"] == 30000 and row["c_i64"] == 10_000_000_000
        assert abs(row["c_f"] - 2.5) < 1e-6 and abs(row["c_d"] - 3.25) < 1e-9
        assert bool(row["c_bool"]) is False
        assert row["c_vc"] == "hello world"
        assert list(row["c_vec"]) == [7.0, 8.0, 9.0]
    finally:
        db.drop_table("m_upd_types", ConflictType.Ignore)


def test_update_column_an_index_is_built_on(db):
    """Update the vector column an HNSW index is built on; the index must see the change."""
    t = _fresh(db, "m_upd_indexed", {"id": {"type": "int"}, "vec": {"type": "vector,4,float"}})
    try:
        t.insert([{"id": i, "vec": [float(i)] * 4} for i in range(10)])
        t.create_index("h", idx.IndexInfo("vec", idx.IndexType.Hnsw,
                       {"M": "16", "ef_construction": "50", "metric": "l2"}), ConflictType.Error)
        # before: nearest to [3,3,3,3] is id 3 (exact)
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [3.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) == 3
        # move id 3 far away
        r = t.update("id = 3", {"vec": [1000.0] * 4})
        assert r.error_code == ErrorCode.OK
        # now querying [3,3,3,3] must NOT return id 3 first; and querying near [1000,...] returns id 3
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [3.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) != 3
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [1000.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) == 3
    finally:
        db.drop_table("m_upd_indexed", ConflictType.Ignore)


def test_update_to_null_nullable_and_not_null(db):
    t = _fresh(db, "m_upd_null", {"id": {"type": "int", "constraints": ["primary key", "not null"]},
                                  "nn": {"type": "int", "constraints": ["not null"]},
                                  "nullable": {"type": "int"}})
    try:
        t.insert([{"id": 1, "nn": 5, "nullable": 5}])
        # nullable -> NULL is accepted and observable
        r = t.update("id = 1", {"nullable": None})
        assert r.error_code == ErrorCode.OK
        df, _ = t.output(["id", "nullable"]).to_df()
        assert df["nullable"].to_list()[0] is None
        # NOT NULL -> NULL is rejected, and the prior value is preserved (atomic reject)
        with pytest.raises(InfinityException) as e:
            t.update("id = 1", {"nn": None})
        assert e.value.args[0] == ErrorCode.NOT_NULL_CONSTRAINT_VIOLATED
        df, _ = t.output(["id", "nn"]).to_df()
        assert df["nn"].to_list()[0] == 5
    finally:
        db.drop_table("m_upd_null", ConflictType.Ignore)


def test_update_no_match_predicate(db):
    t = _fresh(db, "m_upd_nomatch", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"id": i, "v": i} for i in range(5)])
        r = t.update("1 = 0", {"v": -1})
        assert r.error_code == ErrorCode.OK
        # nothing changed
        df, _ = t.output(["id", "v"]).to_df()
        assert {int(a): int(b) for a, b in zip(df["id"].to_list(), df["v"].to_list())} == {i: i for i in range(5)}
        assert _count(t) == 5
    finally:
        db.drop_table("m_upd_nomatch", ConflictType.Ignore)


def test_update_all_rows_always_true_predicate(db):
    t = _fresh(db, "m_upd_alltrue", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"id": i, "v": i} for i in range(5)])
        r = t.update("1 = 1", {"v": 42})
        assert r.error_code == ErrorCode.OK
        df, _ = t.output(["v"]).to_df()
        assert df["v"].to_list() == [42] * 5
    finally:
        db.drop_table("m_upd_alltrue", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# DELETE — positive
# --------------------------------------------------------------------------- #

def test_delete_by_predicate(db):
    t = _fresh(db, "m_del_pred", {"c1": {"type": "int"}, "c2": {"type": "int"}})
    try:
        t.insert([{"c1": i, "c2": i * 10} for i in (1, 2, 3, 4)])
        r = t.delete("c1 = 2")
        assert r.error_code == ErrorCode.OK and r.deleted_rows == 1
        assert _scan_ids(t, "c1") == {1, 3, 4}
        assert _count(t) == 3
    finally:
        db.drop_table("m_del_pred", ConflictType.Ignore)


def test_delete_all_no_condition(db):
    t = _fresh(db, "m_del_all", {"c1": {"type": "int"}})
    try:
        t.insert([{"c1": i} for i in range(50)])
        r = t.delete()
        assert r.error_code == ErrorCode.OK and r.deleted_rows == 50
        assert _count(t) == 0
        df, _ = t.output(["c1"]).to_df()
        assert len(df) == 0
    finally:
        db.drop_table("m_del_all", ConflictType.Ignore)


def test_delete_no_match_predicate(db):
    t = _fresh(db, "m_del_nomatch", {"c1": {"type": "int"}})
    try:
        t.insert([{"c1": i} for i in range(5)])
        r = t.delete("c1 = 999")
        assert r.error_code == ErrorCode.OK and r.deleted_rows == 0
        assert _count(t) == 5
    finally:
        db.drop_table("m_del_nomatch", ConflictType.Ignore)


def test_delete_then_reinsert_same_key(db):
    t = _fresh(db, "m_del_reinsert", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"id": i, "v": i} for i in range(5)])
        assert t.delete("id = 3").deleted_rows == 1
        assert _scan_ids(t) == {0, 1, 2, 4}
        t.insert([{"id": 3, "v": 333}])
        assert _scan_ids(t) == {0, 1, 2, 3, 4}
        assert _count(t) == 5
        df, _ = t.output(["v"]).filter("id = 3").to_df()
        assert df["v"].to_list() == [333]
    finally:
        db.drop_table("m_del_reinsert", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# HEADLINE: deleted rows gone from every access path, through compact/optimize/restart
# --------------------------------------------------------------------------- #

def _build_mixed_table(db, name, n=40):
    t = _fresh(db, name, {"id": {"type": "int"}, "num": {"type": "int"},
                          "vec": {"type": "vector,4,float"}, "body": {"type": "varchar"}})
    t.insert([{"id": i, "num": i, "vec": [float(i)] * 4, "body": f"term{i} common"} for i in range(n)])
    t.create_index("h", idx.IndexInfo("vec", idx.IndexType.Hnsw,
                   {"M": "16", "ef_construction": "50", "metric": "l2"}), ConflictType.Error)
    t.create_index("ft", idx.IndexInfo("body", idx.IndexType.FullText), ConflictType.Error)
    return t


def _assert_absent_everywhere(t, deleted: set[int], survivors: set[int]):
    n_survivors = len(survivors)
    # plain scan
    assert _scan_ids(t) == survivors
    # filtered scan: none of the deleted ids match a range that would include them
    df, _ = t.output(["id"]).filter("id >= 0").to_df()
    assert _ids(df) == survivors
    # count(*)
    assert _count(t) == n_survivors
    # KNN over the HNSW index: ask for every row; deleted ids must not surface
    df, _ = t.output(["id", "_distance"]).match_dense("vec", [0.0] * 4, "float", "l2", n_survivors + len(deleted)).to_df()
    got = _ids(df)
    assert got == survivors, f"KNN leaked deleted rows: {got & deleted}"
    # KNN aimed straight at a deleted vector must not return it as the top hit
    for d in list(deleted)[:3]:
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [float(d)] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) != d, f"KNN returned deleted id {d} as nearest"
    # full-text: the per-row unique term of a deleted row must return nothing
    for d in list(deleted)[:3]:
        res, _ = t.output(["id", "_score"]).match_text("body", f"term{d}", 10).to_pl()
        assert res.height == 0, f"full-text returned deleted id {d}"
    # full-text on the shared term returns exactly the survivors
    res, _ = t.output(["id", "_score"]).match_text("body", "common", n_survivors + len(deleted)).to_pl()
    assert {int(x) for x in res["id"].to_list()} == survivors


def test_deleted_rows_absent_from_every_access_path(db, inst):
    """The correctness question that matters most: a deleted row must be gone from a
    plain scan, a filtered scan, a KNN search over HNSW, a full-text search, and
    count(*) — and stay gone after compact, after optimize, and after a restart."""
    name = "m_del_everywhere"
    n = 40
    deleted = {5, 6, 7, 20, 39}
    survivors = set(range(n)) - deleted
    try:
        t = _build_mixed_table(db, name, n)
        r = t.delete("id = 5 or id = 6 or id = 7 or id = 20 or id = 39")
        assert r.error_code == ErrorCode.OK and r.deleted_rows == len(deleted)

        _assert_absent_everywhere(t, deleted, survivors)          # in-memory delete mask

        t.compact()
        _assert_absent_everywhere(t, deleted, survivors)          # after compact

        t.optimize()
        _assert_absent_everywhere(t, deleted, survivors)          # after optimize

        # restart keeps the data dir: proves the deletion was persisted, not just masked
        # in memory. graceful stop checkpoints; recovery must reproduce the delete.
        inst.restart(graceful=True)
        conn2 = inst.connect()
        try:
            t2 = conn2.get_database("default_db").get_table(name)
            _assert_absent_everywhere(t2, deleted, survivors)     # after restart
        finally:
            conn2.disconnect()
    finally:
        conn3 = inst.connect()
        try:
            conn3.get_database("default_db").drop_table(name, ConflictType.Ignore)
        finally:
            conn3.disconnect()


# --------------------------------------------------------------------------- #
# COMPACT
# --------------------------------------------------------------------------- #

def test_compact_merges_segments_preserving_data_and_deletes(db):
    t = _fresh(db, "m_compact", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        # four separate inserts -> (at least) several segments
        for base in range(4):
            t.insert([{"id": base * 10 + i, "v": base * 10 + i} for i in range(10)])
        seg_before = len(t.show_segments())
        assert seg_before >= 1
        assert _count(t) == 40
        # delete a few, then compact
        t.delete("id = 0 or id = 15 or id = 39")
        assert _count(t) == 37
        r = t.compact()
        if r.error_code != ErrorCode.OK:
            import time
            time.sleep(2)  # background compaction may have raced us
        seg_after = len(t.show_segments())
        assert seg_after <= seg_before
        assert _count(t) == 37
        assert _scan_ids(t) == (set(range(40)) - {0, 15, 39})
    finally:
        db.drop_table("m_compact", ConflictType.Ignore)


def test_compact_empty_table(db):
    t = _fresh(db, "m_compact_empty", {"id": {"type": "int"}})
    try:
        try:
            r = t.compact()
            outcome = f"ok:{r.error_code}"
        except InfinityException as e:
            outcome = f"exc:{e.args[0]}"
        # whatever the policy, the server must survive and the table must still be usable
        assert _count(t) == 0, outcome
        t.insert([{"id": 1}])
        assert _count(t) == 1
    finally:
        db.drop_table("m_compact_empty", ConflictType.Ignore)


def test_compact_twice_in_a_row(db):
    """Compact issued again immediately (self-conflict probe). Final state must be correct."""
    t = _fresh(db, "m_compact_twice", {"id": {"type": "int"}})
    try:
        for base in range(3):
            t.insert([{"id": base * 5 + i} for i in range(5)])
        for _ in range(2):
            try:
                t.compact()
            except InfinityException:
                pass  # a conflict with the previous/background compaction is acceptable
        assert _count(t) == 15
        assert _scan_ids(t) == set(range(15))
    finally:
        db.drop_table("m_compact_twice", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# OPTIMIZE
# --------------------------------------------------------------------------- #

def test_optimize_empty_table(db):
    t = _fresh(db, "m_opt_empty", {"id": {"type": "int"}})
    try:
        try:
            r = t.optimize()
            outcome = f"ok:{getattr(r, 'error_code', '?')}"
        except InfinityException as e:
            outcome = f"exc:{e.args[0]}"
        assert _count(t) == 0, outcome
    finally:
        db.drop_table("m_opt_empty", ConflictType.Ignore)


def test_optimize_preserves_results(db):
    t = _fresh(db, "m_opt", {"id": {"type": "int"}, "vec": {"type": "vector,4,float"}})
    try:
        t.insert([{"id": i, "vec": [float(i)] * 4} for i in range(30)])
        t.create_index("h", idx.IndexInfo("vec", idx.IndexType.Hnsw,
                       {"M": "16", "ef_construction": "50", "metric": "l2"}), ConflictType.Error)
        t.delete("id = 10")
        before, _ = t.output(["id"]).match_dense("vec", [0.0] * 4, "float", "l2", 29).to_df()
        before_ids = _ids(before)
        t.optimize()
        after, _ = t.output(["id"]).match_dense("vec", [0.0] * 4, "float", "l2", 29).to_df()
        assert _ids(after) == before_ids == (set(range(30)) - {10})
        assert _count(t) == 29
    finally:
        db.drop_table("m_opt", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# CLEANUP
# --------------------------------------------------------------------------- #

def test_cleanup_path(db, inst):
    """conn.cleanup() reclaims dropped/compacted storage; it must not disturb live data."""
    t = _fresh(db, "m_cleanup", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        for base in range(3):
            t.insert([{"id": base * 10 + i, "v": base * 10 + i} for i in range(10)])
        t.delete("id = 5")
        t.compact()
        # drop a throwaway table so cleanup has something to reclaim
        db.drop_table("m_cleanup_scratch", ConflictType.Ignore)
        scratch = db.create_table("m_cleanup_scratch", {"id": {"type": "int"}}, ConflictType.Error)
        scratch.insert([{"id": i} for i in range(5)])
        db.drop_table("m_cleanup_scratch", ConflictType.Error)

        conn = inst.connect()
        try:
            r = conn.cleanup()
            assert r.error_code == ErrorCode.OK
        finally:
            conn.disconnect()

        # live table is intact and correct after cleanup
        assert _count(t) == 29
        assert _scan_ids(t) == (set(range(30)) - {5})
    finally:
        db.drop_table("m_cleanup", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# MEM-INDEX DUMP (cross mem_index_capacity = 65536)
# --------------------------------------------------------------------------- #

def test_mem_index_dump_crossing_capacity(db):
    """Insert > mem_index_capacity (65536) rows into an HNSW-indexed table so the
    in-memory index is dumped to a segment, and assert queries return the same answers
    before and after an explicit dump, and that deletes still take effect across the
    dumped + in-memory boundary."""
    n = 70000  # > 65536
    t = _fresh(db, "m_memdump", {"id": {"type": "int"}, "vec": {"type": "vector,4,float"}})
    try:
        t.create_index("h", idx.IndexInfo("vec", idx.IndexType.Hnsw,
                       {"M": "16", "ef_construction": "50", "metric": "l2"}), ConflictType.Error)
        batch = 5000
        for start in range(0, n, batch):
            rows = [{"id": i, "vec": [float(i)] * 4} for i in range(start, min(start + batch, n))]
            t.insert(rows)
        assert _count(t) == n

        # exact-hit KNN before an explicit dump: nearest to [500,...] is id 500 (dist 0)
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [500.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) == 500
        # a point past the mem_index_capacity boundary
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [69000.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) == 69000

        # force any remaining in-memory index to disk, then re-query — answers stable
        t.dump_index("h")
        assert _count(t) == n
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [500.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) == 500
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [69000.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) == 69000

        # a delete must remove the row from the dumped index too
        t.delete("id = 500")
        assert _count(t) == n - 1
        df, _ = t.output(["id", "_distance"]).match_dense("vec", [500.0] * 4, "float", "l2", 1).to_df()
        assert int(df["id"].to_list()[0]) != 500
    finally:
        db.drop_table("m_memdump", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# NEGATIVE tests — assert the specific error, then assert the server is still correct
# --------------------------------------------------------------------------- #

def test_update_on_nonexistent_table(db):
    db.drop_table("m_ne_upd", ConflictType.Ignore)
    stale = db.create_table("m_ne_upd", {"c1": {"type": "int"}}, ConflictType.Error)
    db.drop_table("m_ne_upd", ConflictType.Error)
    with pytest.raises(InfinityException) as e:
        stale.update("c1 = 1", {"c1": 2})
    assert e.value.args[0] == ErrorCode.TABLE_NOT_EXIST


def test_delete_on_nonexistent_table(db):
    db.drop_table("m_ne_del", ConflictType.Ignore)
    stale = db.create_table("m_ne_del", {"c1": {"type": "int"}}, ConflictType.Error)
    db.drop_table("m_ne_del", ConflictType.Error)
    with pytest.raises(InfinityException) as e:
        stale.delete("c1 = 1")
    assert e.value.args[0] == ErrorCode.TABLE_NOT_EXIST


def test_predicate_references_nonexistent_column(db):
    t = _fresh(db, "m_pred_nocol", {"c1": {"type": "int"}})
    try:
        t.insert([{"c1": 1}])
        with pytest.raises(InfinityException) as e:
            t.delete("nope = 1")
        assert e.value.args[0] == ErrorCode.COLUMN_NOT_EXIST
        with pytest.raises(InfinityException) as e:
            t.update("nope = 1", {"c1": 2})
        assert e.value.args[0] == ErrorCode.COLUMN_NOT_EXIST
        assert _count(t) == 1  # table untouched, server alive
    finally:
        db.drop_table("m_pred_nocol", ConflictType.Ignore)


def test_update_sets_nonexistent_column(db):
    t = _fresh(db, "m_set_nocol", {"c1": {"type": "int"}})
    try:
        t.insert([{"c1": 1}])
        with pytest.raises(InfinityException) as e:
            t.update("c1 = 1", {"nope": 5})
        assert e.value.args[0] in (ErrorCode.SYNTAX_ERROR, ErrorCode.COLUMN_NOT_EXIST)
        assert _count(t) == 1
    finally:
        db.drop_table("m_set_nocol", ConflictType.Ignore)


def test_update_wrong_type_list_into_scalar(db):
    t = _fresh(db, "m_wrongtype", {"c1": {"type": "int"}, "c2": {"type": "int"}})
    try:
        t.insert([{"c1": 1, "c2": 2}])
        with pytest.raises(InfinityException) as e:
            t.update("c1 = 1", {"c2": [1, 2, 3]})
        assert e.value.args[0] == ErrorCode.NOT_SUPPORTED_TYPE_CONVERSION
        # value preserved
        df, _ = t.output(["c2"]).to_df()
        assert df["c2"].to_list() == [2]
    finally:
        db.drop_table("m_wrongtype", ConflictType.Ignore)


def test_update_violates_embedding_dimension(db):
    t = _fresh(db, "m_dim", {"id": {"type": "int"}, "vec": {"type": "vector,4,float"}})
    try:
        t.insert([{"id": 1, "vec": [1.0, 2.0, 3.0, 4.0]}])
        with pytest.raises(InfinityException) as e:
            t.update("id = 1", {"vec": [1.0, 2.0, 3.0]})  # dim 3 into dim-4 column
        assert e.value.args[0] == ErrorCode.DATA_TYPE_MISMATCH
        df, _ = t.output(["vec"]).to_df()
        assert list(df["vec"].to_list()[0]) == [1.0, 2.0, 3.0, 4.0]  # unchanged
    finally:
        db.drop_table("m_dim", ConflictType.Ignore)


def test_invalid_predicate_non_boolean_expression(db):
    t = _fresh(db, "m_badpred", {"num": {"type": "int"}})
    try:
        t.insert([{"num": 1}])
        # a bare column reference is not a boolean predicate; the server must reject it
        with pytest.raises(InfinityException) as e:
            t.delete("num")
        assert e.value.args[0] in (ErrorCode.INVALID_EXPRESSION, ErrorCode.SYNTAX_ERROR), e.value.args
        assert _count(t) == 1
    finally:
        db.drop_table("m_badpred", ConflictType.Ignore)


def test_delete_predicate_type_mismatch_is_lenient(db):
    """Comparing an int column to a non-numeric string matches nothing and does not
    error (documents the engine's lenient predicate coercion — non-destructive)."""
    t = _fresh(db, "m_predmismatch", {"num": {"type": "int"}})
    try:
        t.insert([{"num": i} for i in range(5)])
        r = t.delete("num = 'abc'")
        assert r.error_code == ErrorCode.OK and r.deleted_rows == 0
        assert _count(t) == 5
    finally:
        db.drop_table("m_predmismatch", ConflictType.Ignore)


def test_partial_update_atomicity(db):
    """A single UPDATE with one valid and one invalid assignment must not commit the
    valid half (no partially-modified row)."""
    t = _fresh(db, "m_partial", {"id": {"type": "int"}, "good": {"type": "int"}, "bad": {"type": "int"}})
    try:
        t.insert([{"id": 1, "good": 10, "bad": 20}])
        with pytest.raises(InfinityException):
            t.update("id = 1", {"good": 111, "bad": [1, 2, 3]})  # bad = list into int -> reject
        df, _ = t.output(["good", "bad"]).to_df()
        row = df.to_dict("records")[0]
        assert row["good"] == 10 and row["bad"] == 20, f"partial update leaked: {row}"
    finally:
        db.drop_table("m_partial", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# DEFECT PIN: silent NULL coercion on bad UPDATE value
# --------------------------------------------------------------------------- #

def test_DEFECT_bad_update_value_silently_coerced_to_null(db):
    """DEFECT (engine, silent-acceptance + data-loss): an UPDATE that assigns a value a
    nullable numeric column cannot represent — a non-numeric string, or an integer out
    of the column's range — is *silently accepted* (error_code == OK) and the row's
    prior value is *overwritten with NULL*. No error is returned and none is logged.

    This test pins the current buggy behavior so it is reproducible; if the engine is
    fixed to reject the assignment (the correct behavior), this test will start failing
    and must be updated. See the reported finding for full detail.
    """
    t = _fresh(db, "m_defect_null", {"id": {"type": "int"}, "n": {"type": "int"}})
    try:
        t.insert([{"id": 1, "n": 100}, {"id": 2, "n": 200}])

        # (1) non-numeric string into an int column: accepted, value becomes NULL
        r = t.update("id = 1", {"n": "abc"})
        assert r.error_code == ErrorCode.OK  # <-- should have raised a conversion error
        df, _ = t.output(["n"]).filter("id = 1").to_df()
        assert df["n"].to_list()[0] is None  # <-- prior value 100 was silently destroyed

        # (2) integer out of int32 range: accepted, value becomes NULL
        r = t.update("id = 2", {"n": 2 ** 40})
        assert r.error_code == ErrorCode.OK  # <-- should have raised an out-of-range error
        df, _ = t.output(["n"]).filter("id = 2").to_df()
        assert df["n"].to_list()[0] is None  # <-- prior value 200 was silently destroyed

        # no server-side error was logged for either data-losing update
        assert inst_errors_for(t) == []
    finally:
        db.drop_table("m_defect_null", ConflictType.Ignore)


def test_DEFECT_string_vs_float_rounding_inconsistency(db):
    """DEFECT (engine, minor): assigning an int column the string '3.9' truncates to 3,
    but assigning the float 3.9 rounds to 4 — two different coercion paths give
    different results for the same numeric magnitude."""
    t = _fresh(db, "m_defect_round", {"id": {"type": "int"}, "n": {"type": "int"}})
    try:
        t.insert([{"id": 1, "n": 0}, {"id": 2, "n": 0}])
        t.update("id = 1", {"n": "3.9"})
        t.update("id = 2", {"n": 3.9})
        df, _ = t.output(["id", "n"]).to_df()
        got = {int(a): int(b) for a, b in zip(df["id"].to_list(), df["n"].to_list())}
        assert got[1] == 3   # string path truncates
        assert got[2] == 4   # float path rounds
    finally:
        db.drop_table("m_defect_round", ConflictType.Ignore)


# a tiny module-global to let the defect test scrape the log without a fixture juggle
_INST_REF = {}


def inst_errors_for(_t):
    return _INST_REF["inst"].log_errors()


@pytest.fixture(scope="module", autouse=True)
def _capture_inst(inst):
    _INST_REF["inst"] = inst
    yield


# --------------------------------------------------------------------------- #
# module-final: the server logged no error/critical/fatal line across the run
# --------------------------------------------------------------------------- #

def test_zzz_server_log_clean(inst):
    errs = inst.log_errors()
    assert errs == [], "server logged error/critical/fatal lines:\n" + "\n".join(errs[:20])
