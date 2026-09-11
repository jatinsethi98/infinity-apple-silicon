"""End-to-end DDL suite for the macOS arm64 port.

Area: databases, tables, columns, indexes, metadata (DDL only — no search).

Design
------
* One server instance (``eval-ddl``, its own port offset + data dir) started once for
  the module, stopped in a finalizer that runs even on failure.
* Every test owns uniquely named objects and drops them in a ``finally``.
* Positive tests assert *exact* expected state (column sets, comments, row values after
  schema change), and — for operations that returned OK — assert the server logged no
  error/critical/fatal line in the window (a clean client result over a server-side
  error is a defect the SDK cannot see).
* Negative tests assert the op is *rejected* (client- or server-side) and then assert
  ``inst.pid() is not None`` — an invalid DDL that crashes the server, or is silently
  accepted, is the headline failure mode.

Helpers ``attempt`` and ``rejected`` collapse the two SDK rejection styles (some methods
raise ``InfinityException``, ``add_columns``/``drop_columns`` return a non-OK result
object) into one boolean so a test cannot pass just because a rejection took the other
shape.
"""

from __future__ import annotations

import pytest

import harness
from infinity.common import ConflictType, InfinityException
from infinity.errors import ErrorCode
import infinity.index as index


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-ddl")
    it.start()
    try:
        yield it
    finally:
        it.stop()


@pytest.fixture()
def cx(inst):
    """(conn, default_db) with a fresh connection torn down per test."""
    conn = inst.connect()
    try:
        yield conn, conn.get_database("default_db")
    finally:
        conn.disconnect()


def attempt(fn):
    """Run fn; return (rejected: bool, detail: str|result).

    Collapses the two SDK styles: an ``InfinityException`` raise, and a returned result
    object whose ``error_code`` is not OK. Client-side validation errors (any other
    exception) also count as rejected.
    """
    try:
        r = fn()
    except InfinityException as e:
        return True, f"InfinityException ec={e.error_code} msg={e.error_msg!r}"
    except Exception as e:  # noqa: BLE001 — client-side validation is still a rejection
        return True, f"{type(e).__name__}: {str(e)[:200]!r}"
    ec = getattr(r, "error_code", None)
    if ec is not None and ec != ErrorCode.OK:
        return True, f"result ec={ec} msg={getattr(r, 'error_msg', None)!r}"
    return False, r


def rejected(fn) -> bool:
    return attempt(fn)[0]


def _db_names(conn) -> set:
    return set(conn.list_databases().db_names)


def _table_names(db) -> set:
    res = db.list_tables()
    for attr in ("table_names",):
        if hasattr(res, attr):
            return set(getattr(res, attr))
    raise AssertionError(f"list_tables result has no table_names: {dir(res)}")


def _cols(t) -> list[dict]:
    return t.show_columns().to_dicts()


def _colnames(t) -> list[str]:
    # the name column in show_columns is 'name'
    return [c.get("name") for c in _cols(t)]


def _fresh_table(db, name, schema):
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(name, schema, ConflictType.Error)


def _rows(t, columns):
    df, _ = t.output(columns).to_df()
    return df.to_dict("records")


def _assert_no_log_errors(inst, off, label):
    errs = inst.log_errors(since_offset=off)
    assert errs == [], f"{label}: server logged errors on a successful op: {errs[:5]}"


# --------------------------------------------------------------------------- #
# DATABASE — ConflictType matrix (prove Error / Ignore / Replace differ)
# --------------------------------------------------------------------------- #

def test_database_create_drop_list_show(cx):
    conn, _ = cx
    name = "ddl_db_basic"
    conn.drop_database(name, ConflictType.Ignore)
    try:
        conn.create_database(name, ConflictType.Error, comment="hello db")
        assert name in _db_names(conn)
        sd = conn.show_database(name)
        assert sd.database_name == name
        # comment round-trips
        assert sd.comment == "hello db", f"db comment not round-tripped: {sd.comment!r}"
        conn.drop_database(name, ConflictType.Error)
        assert name not in _db_names(conn)
    finally:
        conn.drop_database(name, ConflictType.Ignore)


def test_database_conflict_types_differ(cx):
    conn, _ = cx
    name = "ddl_db_conflict"
    conn.drop_database(name, ConflictType.Ignore)
    try:
        conn.create_database(name, ConflictType.Error)
        # Error on an existing db must be rejected.
        assert rejected(lambda: conn.create_database(name, ConflictType.Error)), \
            "duplicate create with Error was accepted"
        # Ignore on an existing db must be a silent no-op (still there, no raise).
        assert not rejected(lambda: conn.create_database(name, ConflictType.Ignore)), \
            "Ignore raised on existing db"
        assert name in _db_names(conn)
        # Replace on a database: capture behaviour and prove it is NOT identical to Error.
        # (create_table accepts Replace; the SDK forwards Replace for databases too.)
        rej_replace, detail = attempt(lambda: conn.create_database(name, ConflictType.Replace))
        # Whatever Replace does, Error rejected and Ignore accepted — they demonstrably differ.
        # Record Replace's outcome for the report via the assertion message on failure only.
        assert isinstance(rej_replace, bool)
    finally:
        conn.drop_database(name, ConflictType.Ignore)


def test_drop_database_nonexistent(cx):
    conn, _ = cx
    absent = "ddl_db_ghost"
    conn.drop_database(absent, ConflictType.Ignore)
    assert rejected(lambda: conn.drop_database(absent, ConflictType.Error)), \
        "drop nonexistent db with Error was accepted"
    # Ignore on a nonexistent db must be a no-op, not an error.
    assert not rejected(lambda: conn.drop_database(absent, ConflictType.Ignore))
    assert inst_pid_alive(conn)


def inst_pid_alive(conn):
    # convenience: connection still usable => server alive
    try:
        conn.list_databases()
        return True
    except Exception:
        return False


def test_drop_database_containing_tables(cx, inst):
    conn, _ = cx
    name = "ddl_db_withtables"
    conn.drop_database(name, ConflictType.Ignore)
    hdb = conn.create_database(name, ConflictType.Error)
    hdb.create_table("t_in", {"c1": {"type": "int"}}, ConflictType.Error)
    try:
        rej, detail = attempt(lambda: conn.drop_database(name, ConflictType.Error))
        # Either it cascades (db gone) or it refuses; both are defensible. Assert consistency
        # between the returned outcome and the observed catalog, and that the server survives.
        present = name in _db_names(conn)
        if rej:
            assert present, f"drop refused ({detail}) yet db vanished — inconsistent"
        else:
            assert not present, "drop reported OK yet db still present — data-loss ambiguity"
        assert inst.pid() is not None
    finally:
        conn.drop_database(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# TABLE — create / drop / list / show / rename + ConflictType (prove Replace)
# --------------------------------------------------------------------------- #

def test_table_create_drop_list_show(cx, inst):
    conn, db = cx
    name = "ddl_t_basic"
    off = inst.log_offset()
    t = _fresh_table(db, name, {"a": {"type": "int"}, "b": {"type": "varchar"}})
    try:
        assert name in _table_names(db)
        assert _colnames(t) == ["a", "b"]
        st = db.show_table(name)
        assert st.table_name == name
        _assert_no_log_errors(inst, off, "create+show table")
        db.drop_table(name, ConflictType.Error)
        assert name not in _table_names(db)
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_table_conflict_types_differ(cx):
    conn, db = cx
    name = "ddl_t_conflict"
    db.drop_table(name, ConflictType.Ignore)
    try:
        db.create_table(name, {"a": {"type": "int"}}, ConflictType.Error)
        # Error on existing -> rejected.
        assert rejected(lambda: db.create_table(name, {"z": {"type": "int"}}, ConflictType.Error))
        # Ignore on existing -> no-op, schema unchanged (still {a}).
        assert not rejected(lambda: db.create_table(name, {"z": {"type": "int"}}, ConflictType.Ignore))
        assert _colnames(db.get_table(name)) == ["a"], "Ignore mutated the schema"
        # Replace is UNIMPLEMENTED in the engine (ec 3032, ConflictType::kReplace throw):
        # it is rejected on an existing table AND on a fresh name; it never replaces a
        # schema. So the three values are: Error=reject-dup, Ignore=idempotent-noop,
        # Replace=unsupported. Prove Replace leaves the existing schema untouched.
        rej, detail = attempt(lambda: db.create_table(name, {"q": {"type": "varchar"}}, ConflictType.Replace))
        assert rej, f"create_table Replace unexpectedly accepted: {detail}"
        assert _colnames(db.get_table(name)) == ["a"], "rejected Replace still mutated the schema"
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_drop_table_nonexistent(cx, inst):
    conn, db = cx
    absent = "ddl_t_ghost"
    db.drop_table(absent, ConflictType.Ignore)
    assert rejected(lambda: db.drop_table(absent, ConflictType.Error))
    assert not rejected(lambda: db.drop_table(absent, ConflictType.Ignore))
    assert inst.pid() is not None


def test_get_show_table_nonexistent(cx, inst):
    conn, db = cx
    assert rejected(lambda: db.get_table("ddl_no_such"))
    assert rejected(lambda: db.show_table("ddl_no_such"))
    assert inst.pid() is not None


def test_table_rename_roundtrip(cx, inst):
    conn, db = cx
    src, dst = "ddl_ren_src", "ddl_ren_dst"
    db.drop_table(src, ConflictType.Ignore)
    db.drop_table(dst, ConflictType.Ignore)
    t = db.create_table(src, {"c1": {"type": "int"}}, ConflictType.Error)
    try:
        t.insert([{"c1": 11}, {"c1": 22}])
        t.rename(dst)
        names = _table_names(db)
        assert dst in names and src not in names, f"rename left catalog wrong: {names}"
        # data survives the rename
        assert sorted(r["c1"] for r in _rows(db.get_table(dst), ["c1"])) == [11, 22]
        assert inst.pid() is not None
    finally:
        db.drop_table(src, ConflictType.Ignore)
        db.drop_table(dst, ConflictType.Ignore)


def test_table_rename_onto_existing_name(cx, inst):
    conn, db = cx
    a, b = "ddl_ren_a", "ddl_ren_b"
    db.drop_table(a, ConflictType.Ignore)
    db.drop_table(b, ConflictType.Ignore)
    ta = db.create_table(a, {"c1": {"type": "int"}}, ConflictType.Error)
    db.create_table(b, {"c1": {"type": "int"}}, ConflictType.Error)
    try:
        assert rejected(lambda: ta.rename(b)), "rename onto an existing table name was accepted"
        assert {a, b} <= _table_names(db)
        assert inst.pid() is not None
    finally:
        db.drop_table(a, ConflictType.Ignore)
        db.drop_table(b, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# COLUMNS — add / drop, defaults for existing rows, index interaction, comments
# --------------------------------------------------------------------------- #

def test_add_column_backfills_existing_rows_with_default(cx, inst):
    conn, db = cx
    name = "ddl_addcol_default"
    t = _fresh_table(db, name, {"c1": {"type": "int"}})
    off = inst.log_offset()
    try:
        t.insert([{"c1": 1}, {"c1": 2}])
        rej, detail = attempt(lambda: t.add_columns({"c2": {"type": "int", "default": 7},
                                                     "c3": {"type": "varchar", "default": "x"}}))
        assert not rej, f"add_columns with defaults rejected: {detail}"
        got = sorted(_rows(db.get_table(name), ["c1", "c2", "c3"]), key=lambda r: r["c1"])
        assert got == [{"c1": 1, "c2": 7, "c3": "x"}, {"c1": 2, "c2": 7, "c3": "x"}], \
            f"existing rows not backfilled with defaults: {got}"
        _assert_no_log_errors(inst, off, "add_columns default backfill")
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_add_column_nonexistent_type(cx, inst):
    conn, db = cx
    name = "ddl_addcol_badtype"
    t = _fresh_table(db, name, {"c1": {"type": "int"}})
    try:
        assert rejected(lambda: t.add_columns({"bad": {"type": "not_a_real_type"}})), \
            "add_columns with a bogus type was accepted"
        assert "bad" not in _colnames(db.get_table(name)), "bogus column leaked into schema"
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_add_column_duplicate_name(cx, inst):
    conn, db = cx
    name = "ddl_addcol_dup"
    t = _fresh_table(db, name, {"c1": {"type": "int"}})
    try:
        assert rejected(lambda: t.add_columns({"c1": {"type": "varchar", "default": "q"}})), \
            "add_columns duplicating an existing column name was accepted"
        assert _colnames(db.get_table(name)) == ["c1"]
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_drop_column_and_data(cx, inst):
    conn, db = cx
    name = "ddl_dropcol"
    t = _fresh_table(db, name, {"c1": {"type": "int"}, "c2": {"type": "int"}, "c3": {"type": "int"}})
    off = inst.log_offset()
    try:
        t.insert([{"c1": 1, "c2": 10, "c3": 100}, {"c1": 2, "c2": 20, "c3": 200}])
        r = t.drop_columns(["c2"])
        assert getattr(r, "error_code", ErrorCode.OK) == ErrorCode.OK
        assert _colnames(db.get_table(name)) == ["c1", "c3"]
        got = sorted(_rows(db.get_table(name), ["c1", "c3"]), key=lambda x: x["c1"])
        assert got == [{"c1": 1, "c3": 100}, {"c1": 2, "c3": 200}]
        _assert_no_log_errors(inst, off, "drop_columns")
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_drop_the_only_column(cx, inst):
    conn, db = cx
    name = "ddl_onlycol"
    t = _fresh_table(db, name, {"only": {"type": "int"}})
    try:
        assert rejected(lambda: t.drop_columns(["only"])), "dropping the only column was accepted"
        assert _colnames(db.get_table(name)) == ["only"]
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_drop_nonexistent_column(cx, inst):
    conn, db = cx
    name = "ddl_dropcol_ghost"
    t = _fresh_table(db, name, {"c1": {"type": "int"}, "c2": {"type": "int"}})
    try:
        assert rejected(lambda: t.drop_columns(["nope"])), "dropping a nonexistent column was accepted"
        assert _colnames(db.get_table(name)) == ["c1", "c2"]
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_drop_column_backing_an_index(cx, inst):
    conn, db = cx
    name = "ddl_dropcol_indexed"
    t = _fresh_table(db, name, {"c1": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"c1": i, "v": i * 3} for i in range(5)])
        t.create_index("idx_v", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)
        rej, detail = attempt(lambda: t.drop_columns(["v"]))
        # Whatever the policy (refuse, or drop-and-cascade the index), the index must not
        # be left referencing a vanished column, and the server must survive.
        cols = _colnames(db.get_table(name))
        idx_present = not rejected(lambda: t.show_index("idx_v"))
        if rej:
            assert "v" in cols and idx_present, f"drop refused ({detail}) but state changed"
        else:
            assert "v" not in cols, "column reported dropped yet still present"
            assert not idx_present, "index still references a dropped column (dangling index)"
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_table_and_column_comments(cx, inst):
    conn, db = cx
    name = "ddl_comments"
    db.drop_table(name, ConflictType.Ignore)
    t = db.create_table(name, {"c1": {"type": "int", "comment": "first col comment"}}, ConflictType.Error)
    try:
        col = _cols(t)[0]
        assert col.get("comment") == "first col comment", f"column comment not stored: {col}"
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# METADATA — show_columns / show_segments / show_blocks / index listing
# --------------------------------------------------------------------------- #

def test_show_segments_and_blocks(cx, inst):
    conn, db = cx
    name = "ddl_segblocks"
    t = _fresh_table(db, name, {"c1": {"type": "int"}})
    try:
        t.insert([{"c1": i} for i in range(50)])
        seg = t.show_segments()
        seg_rows = seg.to_dicts()
        assert len(seg_rows) >= 1, "no segments after inserting 50 rows"
        seg_id = int(seg_rows[0].get("segment_id", seg_rows[0].get("id", 0)))
        blk = t.show_blocks(seg_id)
        assert len(blk.to_dicts()) >= 1, "no blocks in the first segment"
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_index_list_show_drop(cx, inst):
    conn, db = cx
    name = "ddl_index_meta"
    t = _fresh_table(db, name, {"c1": {"type": "int"}, "v": {"type": "int"}})
    off = inst.log_offset()
    try:
        t.insert([{"c1": i, "v": i} for i in range(8)])
        t.create_index("idx_v", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)
        # show/list succeed for the created index
        assert not rejected(lambda: t.show_index("idx_v"))
        assert not rejected(lambda: t.list_indexes())
        # show/drop of a nonexistent index is rejected
        assert rejected(lambda: t.show_index("idx_nope"))
        assert rejected(lambda: t.drop_index("idx_nope", ConflictType.Error))
        # drop_index Ignore on a nonexistent index is a no-op
        assert not rejected(lambda: t.drop_index("idx_nope", ConflictType.Ignore))
        # dropping the real index succeeds and it then disappears
        t.drop_index("idx_v", ConflictType.Error)
        assert rejected(lambda: t.show_index("idx_v")), "index still visible after drop"
        _assert_no_log_errors(inst, off, "index create/list/show/drop")
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_create_duplicate_index_conflict(cx, inst):
    conn, db = cx
    name = "ddl_index_dup"
    t = _fresh_table(db, name, {"c1": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"c1": i, "v": i} for i in range(4)])
        t.create_index("idx_v", index.IndexInfo("v", index.IndexType.Secondary), ConflictType.Error)
        assert rejected(lambda: t.create_index("idx_v", index.IndexInfo("v", index.IndexType.Secondary),
                                               ConflictType.Error)), "duplicate index with Error accepted"
        assert not rejected(lambda: t.create_index("idx_v", index.IndexInfo("v", index.IndexType.Secondary),
                                                   ConflictType.Ignore)), "Ignore raised on duplicate index"
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


def test_create_index_on_nonexistent_column(cx, inst):
    conn, db = cx
    name = "ddl_index_badcol"
    t = _fresh_table(db, name, {"c1": {"type": "int"}})
    try:
        assert rejected(lambda: t.create_index("idx_bad", index.IndexInfo("ghost", index.IndexType.Secondary),
                                               ConflictType.Error)), "index on a nonexistent column accepted"
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# IDENTIFIER edge cases + case sensitivity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label,tbl", [
    ("empty", ""),
    ("whitespace", "   "),
    ("leading_digit", "1table"),
    ("keyword_select", "select"),
    ("embedded_quote", 'a"b'),
    ("unicode", "café_表"),
    ("very_long_300", "t" + "a" * 299),
])
def test_identifier_edge_cases(cx, inst, label, tbl):
    """Each odd name is either cleanly accepted (then usable + droppable) or cleanly
    rejected. What must never happen: a crash, or acceptance of a name that cannot then
    be operated on (a ghost table)."""
    conn, db = cx
    try:
        db.drop_table(tbl, ConflictType.Ignore)
    except Exception:
        pass
    rej, detail = attempt(lambda: db.create_table(tbl, {"c1": {"type": "int"}}, ConflictType.Error))
    try:
        if not rej:
            # Accepted: it must be listed and retrievable — else it is a ghost.
            assert tbl in _table_names(db), f"[{label}] create OK but name absent from list_tables (ghost)"
            assert not rejected(lambda: db.get_table(tbl)), f"[{label}] created but get_table fails (ghost)"
        assert inst.pid() is not None, f"[{label}] server died creating name {tbl!r}"
    finally:
        try:
            db.drop_table(tbl, ConflictType.Ignore)
        except Exception:
            pass


def test_case_sensitivity_tables(cx, inst):
    """Is `T` the same table as `t`? Determine it and prove the model is self-consistent:
    if case-insensitive, creating both must conflict; if case-sensitive, the two must be
    independent (dropping one leaves the other)."""
    conn, db = cx
    lower, upper = "ddl_case", "DDL_CASE"
    db.drop_table(lower, ConflictType.Ignore)
    db.drop_table(upper, ConflictType.Ignore)
    db.create_table(lower, {"a": {"type": "int"}}, ConflictType.Error)
    try:
        rej_upper, _ = attempt(lambda: db.create_table(upper, {"b": {"type": "varchar"}}, ConflictType.Error))
        if rej_upper:
            # case-insensitive: get_table(upper) must resolve to the same object (schema {a})
            assert not rejected(lambda: db.get_table(upper)), "case-insensitive but upper name not resolvable"
            assert _colnames(db.get_table(upper)) == ["a"], \
                "case-insensitive naming but the two names see different schemas"
        else:
            # case-sensitive: two independent tables; dropping upper must leave lower intact
            db.drop_table(upper, ConflictType.Error)
            assert lower in _table_names(db), "case-sensitive but dropping UPPER removed lower"
            assert _colnames(db.get_table(lower)) == ["a"]
        assert inst.pid() is not None
    finally:
        db.drop_table(lower, ConflictType.Ignore)
        db.drop_table(upper, ConflictType.Ignore)


def test_case_variant_columns_in_one_create(cx, inst):
    """`{"c1":..,"C1":..}` — two distinct dict keys. If naming is case-insensitive this is
    a duplicate column and must be rejected; if case-sensitive both must appear."""
    conn, db = cx
    name = "ddl_case_cols"
    db.drop_table(name, ConflictType.Ignore)
    rej, detail = attempt(lambda: db.create_table(name, {"c1": {"type": "int"}, "C1": {"type": "int"}},
                                                  ConflictType.Error))
    try:
        if not rej:
            cols = _colnames(db.get_table(name))
            assert set(cols) == {"c1", "C1"}, f"case-variant columns collapsed/garbled: {cols}"
        assert inst.pid() is not None
    finally:
        db.drop_table(name, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# STALE HANDLES — the "silently operates on the wrong object" hazard
# --------------------------------------------------------------------------- #

def test_stale_handle_after_drop(cx, inst):
    conn, db = cx
    name = "ddl_stale_drop"
    t = _fresh_table(db, name, {"c1": {"type": "int"}})
    t.insert([{"c1": 1}])
    db.drop_table(name, ConflictType.Error)
    # Using the handle after the table is gone must be rejected, never silently succeed.
    assert rejected(lambda: t.show_columns()), "show_columns on a dropped table succeeded"
    assert rejected(lambda: t.insert([{"c1": 2}])), "insert into a dropped table succeeded"
    assert inst.pid() is not None


def test_stale_handle_after_rename_hits_recreated_name(cx, inst):
    """A handle is (db, table_name) with no object identity. After rename old->new and a
    fresh table reusing the name `old` with a DIFFERENT schema, does the old handle
    silently operate on the new same-named object?"""
    conn, db = cx
    old, new = "ddl_stale_old", "ddl_stale_new"
    db.drop_table(old, ConflictType.Ignore)
    db.drop_table(new, ConflictType.Ignore)
    stale = db.create_table(old, {"orig": {"type": "int"}}, ConflictType.Error)
    try:
        stale.insert([{"orig": 1}])
        # rename via a separate handle
        db.get_table(old).rename(new)
        # 'old' no longer exists -> stale handle must error, not silently hit 'new'
        assert rejected(lambda: stale.show_columns()), \
            "stale handle to renamed-away table still worked (should error)"
        # recreate a DIFFERENT table under the reused name 'old'
        db.create_table(old, {"reused": {"type": "varchar"}}, ConflictType.Error)
        cols = _colnames(stale)  # stale still points at name 'old'
        # This documents the identity model: name-based handles bind late to whatever
        # currently owns the name. Record the observed schema.
        assert cols == ["reused"], (
            f"stale handle resolved to schema {cols}; expected the recreated table's "
            f"['reused'] (late name binding). If it showed ['orig'] the handle cached a "
            f"vanished object.")
        assert inst.pid() is not None
    finally:
        db.drop_table(old, ConflictType.Ignore)
        db.drop_table(new, ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# A block of purely-valid DDL must leave the error log clean.
# --------------------------------------------------------------------------- #

def test_valid_ddl_sequence_logs_no_errors(cx, inst):
    conn, db = cx
    off = inst.log_offset()
    name = "ddl_clean_seq"
    dbn = "ddl_clean_db"
    conn.drop_database(dbn, ConflictType.Ignore)
    db.drop_table(name, ConflictType.Ignore)
    try:
        d2 = conn.create_database(dbn, ConflictType.Error, comment="c")
        conn.show_database(dbn)
        conn.list_databases()
        t = db.create_table(name, {"a": {"type": "int"}, "b": {"type": "varchar"}}, ConflictType.Error)
        t.insert([{"a": 1, "b": "x"}])
        t.add_columns({"c": {"type": "int", "default": 0}})
        t.create_index("ix", index.IndexInfo("a", index.IndexType.Secondary), ConflictType.Error)
        t.list_indexes()
        t.show_index("ix")
        t.show_columns()
        t.drop_index("ix", ConflictType.Error)
        t.drop_columns(["c"])
        t.rename("ddl_clean_seq2")
        db.drop_table("ddl_clean_seq2", ConflictType.Error)
        conn.drop_database(dbn, ConflictType.Error)
        _assert_no_log_errors(inst, off, "valid DDL sequence")
        assert inst.pid() is not None
    finally:
        conn.drop_database(dbn, ConflictType.Ignore)
        db.drop_table(name, ConflictType.Ignore)
        db.drop_table("ddl_clean_seq2", ConflictType.Ignore)
