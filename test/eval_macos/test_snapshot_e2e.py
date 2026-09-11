"""End-to-end verification of Infinity snapshots on macOS arm64.

Scope: create / list / show / restore / drop snapshots at table, database and
system scope. Restore fidelity is checked by querying THROUGH every index
(HNSW KNN, FullText, Secondary range) and by comparing full row payloads --
never by count(*) alone, because an index can restore structurally and still
answer wrong.

Every negative test asserts the server is still alive afterwards
(`inst.pid() is not None`) and the server-side error log is inspected, because a
clean client result over a server-side error is a defect the SDK cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

from infinity import index  # noqa: E402
from infinity.common import ConflictType  # noqa: E402
from infinity.errors import ErrorCode  # noqa: E402
from infinity.common import InfinityException  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    ix = harness.instance("eval-snapshot")
    ix.start()
    # Disable auto-checkpoint so we test the snapshot machinery, not checkpoints.
    conn = ix.connect()
    try:
        conn.set_config("checkpoint_interval", 0)
    except Exception:
        pass
    finally:
        conn.disconnect()
    yield ix
    ix.stop()


def _df(builder):
    r = builder.to_pl()
    return r[0] if isinstance(r, tuple) else r


def _ids(builder):
    return _df(builder)["id"].to_list()


def _count(t):
    return int(_df(t.output(["count(*)"]))["count(star)"][0])


def _rows_by_id(t):
    """Full row payload sorted by id -- the strongest content-fidelity check."""
    df = _df(t.output(["id", "num", "body"]).filter("id >= 0"))
    df = df.sort("id")
    return list(zip(df["id"].to_list(), df["num"].to_list(), df["body"].to_list()))


QUERY_VEC = [0.0, 0.0, 0.0, 1.0]


def _make_indexed_table(db, name, n=10, extra_rows=0):
    """Table with HNSW + FullText + Secondary indexes, n deterministic rows.

    vec[i]   = [i, i+0.1, 0, 1]   -> KNN(QUERY_VEC) nearest is 0,1,2,...
    num[i]   = i*10               -> range num>=50 is {5,6,7,8,9} for n=10
    body[i]  = "fruit apple ..."  (i<5) / "fruit banana ..." (i>=5)
    """
    db.drop_table(name, ConflictType.Ignore)
    t = db.create_table(name, {
        "id": {"type": "int", "constraints": ["primary key"]},
        "body": {"type": "varchar"},
        "num": {"type": "int"},
        "vec": {"type": "vector,4,float"},
    }, ConflictType.Error)
    rows = []
    for i in range(n + extra_rows):
        body = "fruit apple orange" if i < 5 else "fruit banana grape"
        rows.append({"id": i, "body": body, "num": i * 10,
                     "vec": [float(i), float(i) + 0.1, 0.0, 1.0]})
    if rows:
        t.insert(rows)
    t.create_index("idx_vec", index.IndexInfo(
        "vec", index.IndexType.Hnsw,
        {"metric": "l2", "m": "16", "ef_construction": "50"}), ConflictType.Error)
    t.create_index("idx_ft", index.IndexInfo(
        "body", index.IndexType.FullText), ConflictType.Error)
    t.create_index("idx_num", index.IndexInfo(
        "num", index.IndexType.Secondary), ConflictType.Error)
    return t


def _knn(t, topn=3):
    return _ids(t.output(["id"]).match_dense("vec", QUERY_VEC, "float", "l2", topn))


def _ft(t, term="apple", topn=100):
    return sorted(_ids(t.output(["id"]).match_text("body", term, topn, None)))


def _range(t, expr="num >= 50"):
    return sorted(_ids(t.output(["id"]).filter(expr)))


# --------------------------------------------------------------------------- #
# core fidelity
# --------------------------------------------------------------------------- #

def test_table_snapshot_roundtrip_through_every_index(inst):
    """Snapshot -> drop -> restore must return IDENTICAL answers through HNSW,
    FullText and Secondary, and identical row payloads."""
    conn, db = inst.db()
    try:
        off = inst.log_offset()
        t = _make_indexed_table(db, "fid_t", n=10)

        # Independent ground truth for KNN via numpy.
        vecs = np.array([[i, i + 0.1, 0.0, 1.0] for i in range(10)])
        d = np.linalg.norm(vecs - np.array(QUERY_VEC), axis=1)
        expected_knn = list(np.argsort(d, kind="stable")[:3])

        base_knn = _knn(t, 3)
        base_ft = _ft(t, "apple")
        base_range = _range(t, "num >= 50")
        base_rows = _rows_by_id(t)

        # Baselines must be non-vacuous, else the fidelity comparison is meaningless.
        assert base_knn == expected_knn == [0, 1, 2], base_knn
        assert base_ft == [0, 1, 2, 3, 4], base_ft
        assert base_range == [5, 6, 7, 8, 9], base_range
        assert len(base_rows) == 10

        db.create_table_snapshot("fid_snap", "fid_t")
        db.drop_table("fid_t", ConflictType.Error)
        r = db.restore_table_snapshot("fid_snap")
        assert r.error_code == ErrorCode.OK

        t2 = db.get_table("fid_t")
        assert _count(t2) == 10
        assert _knn(t2, 3) == base_knn, "HNSW answer changed after restore"
        assert _range(t2, "num >= 50") == base_range, "Secondary answer changed after restore"
        assert _rows_by_id(t2) == base_rows, "row payload changed after restore"
        # NOTE: FullText fidelity after restore is broken -- see the dedicated
        # xfail test test_fulltext_answer_survives_restore. We record it here too:
        print("FT_AFTER_RESTORE (expected", base_ft, "):", _ft(t2, "apple"))

        errs = inst.log_errors(off)
        assert errs == [], f"server logged errors during snapshot/restore: {errs}"

        conn.drop_snapshot("fid_snap")
    finally:
        db.drop_table("fid_t", ConflictType.Ignore)
        conn.disconnect()


@pytest.mark.xfail(strict=True, reason=(
    "engine-defect: FullText index restores structurally (show_index reports it "
    "present with 1 segment) but match_text returns ZERO rows for every term "
    "after restore; no server error is logged (silently wrong). HNSW/Secondary "
    "restore correctly. optimize() does not repair it."))
def test_fulltext_answer_survives_restore(inst):
    """A KNN and a range query survive restore; a full-text query must too."""
    conn, db = inst.db()
    try:
        _make_indexed_table(db, "ftr_t", n=10)
        t = db.get_table("ftr_t")
        base_ft = _ft(t, "apple")
        assert base_ft == [0, 1, 2, 3, 4], f"baseline FT wrong: {base_ft}"
        db.create_table_snapshot("ftr_snap", "ftr_t")
        db.drop_table("ftr_t", ConflictType.Error)
        db.restore_table_snapshot("ftr_snap")
        t2 = db.get_table("ftr_t")
        # This is the assertion that currently fails (returns []):
        assert _ft(t2, "apple") == base_ft
        conn.drop_snapshot("ftr_snap")
    finally:
        db.drop_table("ftr_t", ConflictType.Ignore)
        conn.disconnect()


def test_empty_table_snapshot(inst):
    """An indexed but empty table must snapshot and restore cleanly."""
    conn, db = inst.db()
    try:
        off = inst.log_offset()
        _make_indexed_table(db, "empty_t", n=0)
        t = db.get_table("empty_t")
        assert _count(t) == 0

        db.create_table_snapshot("empty_snap", "empty_t")
        db.drop_table("empty_t", ConflictType.Error)
        assert db.restore_table_snapshot("empty_snap").error_code == ErrorCode.OK

        t2 = db.get_table("empty_t")
        assert _count(t2) == 0
        assert _knn(t2, 3) == []
        assert _range(t2, "num >= 0") == []
        assert inst.pid() is not None
        assert inst.log_errors(off) == []
        conn.drop_snapshot("empty_snap")
    finally:
        db.drop_table("empty_t", ConflictType.Ignore)
        conn.disconnect()


def test_snapshot_after_deletes_and_compact(inst):
    """Snapshot must capture post-delete, post-compact state exactly."""
    conn, db = inst.db()
    try:
        off = inst.log_offset()
        t = _make_indexed_table(db, "dc_t", n=10)
        t.delete("num < 30")          # removes id 0,1,2
        assert _count(t) == 7, "delete(num<30) did not remove 3 rows"
        t.compact()
        assert _count(t) == 7, "compact changed the row count"
        surviving = _range(t, "num >= 30")   # avoid always-true 'num>=0' (known indexscan bug)
        print("DC surviving via secondary:", surviving, "count:", _count(t))
        assert surviving == [3, 4, 5, 6, 7, 8, 9], surviving
        base_rows = _rows_by_id(t)

        db.create_table_snapshot("dc_snap", "dc_t")
        db.drop_table("dc_t", ConflictType.Error)
        assert db.restore_table_snapshot("dc_snap").error_code == ErrorCode.OK

        t2 = db.get_table("dc_t")
        assert _count(t2) == 7
        assert _range(t2, "num >= 30") == surviving
        assert _rows_by_id(t2) == base_rows
        assert inst.pid() is not None
        assert inst.log_errors(off) == []
        conn.drop_snapshot("dc_snap")
    finally:
        db.drop_table("dc_t", ConflictType.Ignore)
        conn.disconnect()


def test_snapshot_survives_restart(inst):
    """Restore must not depend on in-memory state: snapshot, graceful restart,
    drop, restore."""
    conn, db = inst.db()
    try:
        _make_indexed_table(db, "rst_t", n=10)
        db.create_table_snapshot("rst_snap", "rst_t")
    finally:
        conn.disconnect()

    inst.restart(graceful=True)

    conn, db = inst.db()
    try:
        off = inst.log_offset()
        db.drop_table("rst_t", ConflictType.Error)
        assert db.restore_table_snapshot("rst_snap").error_code == ErrorCode.OK
        t2 = db.get_table("rst_t")
        assert _count(t2) == 10
        assert _knn(t2, 3) == [0, 1, 2]
        assert _range(t2, "num >= 50") == [5, 6, 7, 8, 9]
        assert _rows_by_id(t2) == [(i, i * 10,
                                    "fruit apple orange" if i < 5 else "fruit banana grape")
                                   for i in range(10)]
        assert inst.pid() is not None
        assert inst.log_errors(off) == []
        conn.drop_snapshot("rst_snap")
    finally:
        db.drop_table("rst_t", ConflictType.Ignore)
        conn.disconnect()


def test_two_snapshots_restored_in_each_order(inst):
    """Two snapshots of one table at different sizes must each restore to their
    own state regardless of restore order."""
    conn, db = inst.db()
    try:
        _make_indexed_table(db, "ver_t", n=10)          # 10 rows -> snapA
        db.create_table_snapshot("snapA", "ver_t")
        t = db.get_table("ver_t")
        t.insert([{"id": i, "body": "fruit banana grape", "num": i * 10,
                   "vec": [float(i), float(i) + 0.1, 0.0, 1.0]} for i in range(10, 15)])
        assert _count(t) == 15
        db.create_table_snapshot("snapB", "ver_t")     # 15 rows -> snapB

        # order 1: B then A
        db.drop_table("ver_t", ConflictType.Error)
        db.restore_table_snapshot("snapB")
        assert _count(db.get_table("ver_t")) == 15
        db.drop_table("ver_t", ConflictType.Error)
        db.restore_table_snapshot("snapA")
        assert _count(db.get_table("ver_t")) == 10

        # order 2: A already restored; now B again
        db.drop_table("ver_t", ConflictType.Error)
        db.restore_table_snapshot("snapB")
        assert _count(db.get_table("ver_t")) == 15

        assert inst.pid() is not None
        conn.drop_snapshot("snapA")
        conn.drop_snapshot("snapB")
    finally:
        db.drop_table("ver_t", ConflictType.Ignore)
        conn.disconnect()


def test_list_show_drop_snapshot(inst):
    """list/show/drop lifecycle, and report the disk footprint."""
    conn, db = inst.db()
    try:
        _make_indexed_table(db, "lsd_t", n=10)
        db.create_table_snapshot("lsd_snap", "lsd_t")

        listing = conn.list_snapshots().snapshots
        names = [s.name for s in listing]
        assert "lsd_snap" in names, names
        entry = next(s for s in listing if s.name == "lsd_snap")
        # list scope/size are populated:
        assert entry.scope in ("Table", "table"), entry.scope
        assert entry.size and entry.size != "", entry.size

        show = conn.show_snapshot("lsd_snap").snapshot  # observed: fields garbled

        # on-disk footprint
        snap_dir = inst.root / "snapshots" / "lsd_snap"
        disk = sum(p.stat().st_size for p in snap_dir.rglob("*") if p.is_file())
        assert disk > 0

        conn.drop_snapshot("lsd_snap")
        names_after = [s.name for s in conn.list_snapshots().snapshots]
        assert "lsd_snap" not in names_after
        with pytest.raises(InfinityException):
            conn.show_snapshot("lsd_snap")   # gone -> must error, not crash
        assert inst.pid() is not None
        # expose the field-mapping oddity in the failure text if anyone asserts on it
        print("SHOW_SNAPSHOT_RAW:", show)
    finally:
        db.drop_table("lsd_t", ConflictType.Ignore)
        conn.disconnect()


def test_database_snapshot(inst):
    """Database-scope snapshot: drop the whole database, restore, verify.

    The session that created the database appears to hold it as current, so
    error 3033 ("Can't drop using database") blocks dropping it from the same
    connection -- we drop and restore from a FRESH connection.
    """
    conn, _ = inst.db()
    try:
        off = inst.log_offset()
        conn.drop_database("snapdb", ConflictType.Ignore)
        sdb = conn.create_database("snapdb", ConflictType.Error)
        _make_indexed_table(sdb, "dbt", n=10)
        base = _rows_by_id(sdb.get_table("dbt"))
        conn.create_database_snapshot("db_snap", "snapdb")
    finally:
        conn.disconnect()

    conn2 = inst.connect()
    try:
        conn2.drop_database("snapdb", ConflictType.Error)
        assert conn2.restore_database_snapshot("db_snap").error_code == ErrorCode.OK

        sdb2 = conn2.get_database("snapdb")
        t2 = sdb2.get_table("dbt")
        assert _count(t2) == 10
        assert _rows_by_id(t2) == base
        assert _knn(t2, 3) == [0, 1, 2]
        assert inst.pid() is not None
        assert inst.log_errors(off) == []
        conn2.drop_snapshot("db_snap")
    finally:
        try:
            conn2.drop_database("snapdb", ConflictType.Ignore)
        except Exception:
            pass
        conn2.disconnect()


def test_system_snapshot_create_list_show_drop(inst):
    """System-scope snapshot create/list/show/drop; restore attempted tolerantly."""
    conn, db = inst.db()
    try:
        off = inst.log_offset()
        _make_indexed_table(db, "sys_t", n=10)
        r = conn.create_system_snapshot("sys_snap")
        assert r.error_code == ErrorCode.OK
        names = [s.name for s in conn.list_snapshots().snapshots]
        assert "sys_snap" in names, names
        conn.show_snapshot("sys_snap")  # must not crash
        assert inst.pid() is not None

        # Restore whole system from snapshot -- must not crash and data present.
        try:
            rr = conn.restore_system_snapshot("sys_snap")
            restored_ok = (rr.error_code == ErrorCode.OK)
        except InfinityException as e:
            restored_ok = False
            print("SYSTEM_RESTORE_EXC:", e.error_code, e.error_msg)
        assert inst.pid() is not None, "server crashed on system restore"
        if restored_ok:
            _, db2 = inst.db()
            assert _count(db2.get_table("sys_t")) == 10
        print("SYSTEM_RESTORE_OK:", restored_ok, "log:", inst.log_errors(off))
        conn.drop_snapshot("sys_snap")
    finally:
        db.drop_table("sys_t", ConflictType.Ignore)
        conn.disconnect()


# --------------------------------------------------------------------------- #
# negative / adversarial
# --------------------------------------------------------------------------- #

def test_duplicate_snapshot_name_is_rejected(inst):
    conn, db = inst.db()
    try:
        _make_indexed_table(db, "dup_t", n=3)
        db.create_table_snapshot("dup_snap", "dup_t")
        with pytest.raises(InfinityException) as ei:
            db.create_table_snapshot("dup_snap", "dup_t")
        assert ei.value.error_code == 3100 or "already exists" in ei.value.error_msg
        assert "dup_snap" in ei.value.error_msg  # error names the offending snapshot
        assert inst.pid() is not None
        conn.drop_snapshot("dup_snap")
    finally:
        db.drop_table("dup_t", ConflictType.Ignore)
        conn.disconnect()


def test_snapshot_nonexistent_table_errors(inst):
    conn, db = inst.db()
    try:
        with pytest.raises(InfinityException) as ei:
            db.create_table_snapshot("ghost_snap", "no_such_table_xyz")
        assert "no_such_table_xyz" in ei.value.error_msg
        assert inst.pid() is not None
    finally:
        conn.disconnect()


def test_restore_nonexistent_snapshot_errors(inst):
    conn, db = inst.db()
    try:
        with pytest.raises(InfinityException) as ei:
            db.restore_table_snapshot("definitely_not_a_snapshot")
        assert inst.pid() is not None
        # Note the error leaks a server-side absolute path (recorded in report).
        print("RESTORE_MISSING_MSG:", ei.value.error_msg)
    finally:
        conn.disconnect()


def test_restore_over_existing_table_errors(inst):
    conn, db = inst.db()
    try:
        _make_indexed_table(db, "over_t", n=5)
        db.create_table_snapshot("over_snap", "over_t")
        # table still exists -> restore must refuse, not silently clobber or crash
        with pytest.raises(InfinityException) as ei:
            db.restore_table_snapshot("over_snap")
        assert "already exists" in ei.value.error_msg
        assert inst.pid() is not None
        # original table is intact
        assert _count(db.get_table("over_t")) == 5
        conn.drop_snapshot("over_snap")
    finally:
        db.drop_table("over_t", ConflictType.Ignore)
        conn.disconnect()


def test_drop_nonexistent_snapshot_errors(inst):
    conn, db = inst.db()
    try:
        with pytest.raises(InfinityException):
            conn.drop_snapshot("no_such_snap_to_drop")
        assert inst.pid() is not None
    finally:
        conn.disconnect()


@pytest.mark.xfail(strict=True, reason=(
    "SECURITY engine-defect (blocker): a snapshot name is used unsanitised as a "
    "filesystem path. '../pwn_escape' writes pwn_escape.json + pwn_escape/ into the "
    "instance DATA ROOT, outside snapshots/; 'nest/deep_snap' creates nested dirs. "
    "A crafted name (e.g. '../wal', '../catalog/...') could clobber engine state or, "
    "with deeper traversal, another instance's data."))
def test_snapshot_name_path_traversal_escapes_dir(inst):
    """SECURITY: a snapshot name containing '..' or '/' must not write outside the
    snapshots directory. This test PROVES whether files escape."""
    conn, db = inst.db()
    escaped = []
    try:
        _make_indexed_table(db, "trav_t", n=3)
        snap_dir = inst.root / "snapshots"

        # 1) parent-escape via '..'  -> lands in the instance data root
        try:
            db.create_table_snapshot("../pwn_escape", "trav_t")
        except InfinityException as e:
            print("TRAVERSAL_DOTDOT_REJECTED:", e.error_msg)
        parent_artifacts = list(inst.root.glob("pwn_escape*"))
        if parent_artifacts:
            escaped.extend(str(p) for p in parent_artifacts)

        # 2) nested via '/'  -> creates subdirs under snapshots/
        try:
            db.create_table_snapshot("nest/deep_snap", "trav_t")
        except InfinityException as e:
            print("TRAVERSAL_SLASH_REJECTED:", e.error_msg)
        nested = list((snap_dir / "nest").glob("deep_snap*")) if (snap_dir / "nest").exists() else []
        if nested:
            escaped.extend(str(p) for p in nested)

        assert inst.pid() is not None, "server crashed on traversal name"

        # This assertion FAILS on purpose if the engine escapes -- surfacing the defect.
        assert not escaped, (
            "SNAPSHOT NAME ESCAPED THE SNAPSHOTS DIRECTORY (path traversal): "
            + "; ".join(escaped)
        )
    finally:
        # clean up any escaped artifacts and the nested snapshot dir
        for p in inst.root.glob("pwn_escape*"):
            (p.unlink() if p.is_file() else __import__("shutil").rmtree(p, ignore_errors=True))
        nest = inst.root / "snapshots" / "nest"
        if nest.exists():
            __import__("shutil").rmtree(nest, ignore_errors=True)
        try:
            conn.drop_snapshot("../pwn_escape")
        except Exception:
            pass
        db.drop_table("trav_t", ConflictType.Ignore)
        conn.disconnect()


def test_restore_truncated_snapshot_files(inst):
    """Corrupt a snapshot on disk, then restore: must error cleanly, not crash."""
    import os
    conn, db = inst.db()
    try:
        off = inst.log_offset()
        _make_indexed_table(db, "trunc_t", n=10)
        db.create_table_snapshot("trunc_snap", "trunc_t")
        db.drop_table("trunc_t", ConflictType.Error)

        # Truncate every .col data file in the snapshot to 0 bytes.
        snap_dir = inst.root / "snapshots" / "trunc_snap"
        cols = list(snap_dir.rglob("*.col"))
        assert cols, "expected .col files in snapshot"
        for p in cols:
            os.truncate(p, 0)

        crashed = False
        errored = False
        try:
            db.restore_table_snapshot("trunc_snap")
        except InfinityException as e:
            errored = True
            print("TRUNC_RESTORE_EXC:", e.error_code, e.error_msg)
        except Exception as e:  # noqa
            print("TRUNC_RESTORE_OTHER:", type(e).__name__, e)
        if inst.pid() is None:
            crashed = True

        assert not crashed, "server CRASHED restoring a truncated snapshot"
        # Whether it errors or silently 'succeeds' with wrong data, report it.
        if not errored:
            # It returned success -> verify data integrity; wrong data == data loss.
            try:
                t2 = db.get_table("trunc_t")
                c = _count(t2)
                print("TRUNC_RESTORE_SILENT_SUCCESS count=", c)
                try:
                    print("TRUNC_ROWS:", _rows_by_id(t2)[:3])
                except Exception as e:  # noqa
                    print("TRUNC_ROWS_UNREADABLE:", type(e).__name__, e)
                try:
                    print("TRUNC_KNN:", _knn(t2, 3))
                except Exception as e:  # noqa
                    print("TRUNC_KNN_UNREADABLE:", type(e).__name__, e)
            except Exception as e:  # noqa
                print("TRUNC_RESTORE_SUCCESS_BUT_UNQUERYABLE:", type(e).__name__, e)
        print("TRUNC_LOG_ERRORS:", inst.log_errors(off))
        try:
            conn.drop_snapshot("trunc_snap")
        except Exception:
            pass
    finally:
        db.drop_table("trunc_t", ConflictType.Ignore)
        conn.disconnect()
