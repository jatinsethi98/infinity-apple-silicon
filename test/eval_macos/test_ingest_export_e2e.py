"""End-to-end ingest/export verification for the macOS arm64 port.

Area: INSERT, COPY FROM (import_data), COPY TO (export_data).

Every test uses its own fresh table on the shared ``eval-insert`` instance and asserts
concrete expected values (row counts, exact bytes, numpy/text round-trips), never merely
"no exception". Defect-documenting tests assert the behaviour that *should* hold, so a
failure is a real engine defect and a pass means it was fixed. Each negative test also
asserts the server is still alive (``inst.pid() is not None``) and, at teardown, that the
server logged no error-level lines during the run.

Findings are reported out-of-band via the evaluation's StructuredOutput; this file is the
reproducible artifact behind them.
"""

from __future__ import annotations

import math
import struct

import pytest

import harness
from infinity.common import ConflictType, InfinityException


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def inst():
    server = harness.instance("eval-insert")
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture()
def db(inst):
    conn, database = inst.db()
    try:
        yield database
    finally:
        conn.disconnect()


_counter = {"n": 0}


def fresh(db, cols):
    """Create a uniquely-named table; drop any prior namesake first."""
    _counter["n"] += 1
    name = f"ie_{_counter['n']}"
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(name, cols), name


def count(t) -> int:
    d = t.output(["count(*)"]).to_pl()[0].to_dicts()[0]
    return int(next(iter(d.values())))


def scalar(t, expr):
    d = t.output([expr]).to_pl()[0].to_dicts()[0]
    return next(iter(d.values()))


# =========================================================================== INSERT


def test_insert_single_row(db):
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "float"}, "c3": {"type": "varchar"}})
    t.insert([{"c1": 7, "c2": 2.5, "c3": "hello"}])
    rows = t.output(["c1", "c2", "c3"]).to_pl()[0].to_dicts()
    assert rows == [{"c1": 7, "c2": 2.5, "c3": "hello"}]


def test_insert_many_rows_one_call(db):
    t, _ = fresh(db, {"a": {"type": "int"}})
    n = 5000  # under the 8192-per-call server limit
    t.insert([{"a": i} for i in range(n)])
    assert count(t) == n
    assert int(scalar(t, "sum(a)")) == sum(range(n))


def test_insert_batch_spanning_blocks(db):
    """70000 rows > mem_index_capacity (65536): must span several blocks correctly."""
    t, _ = fresh(db, {"a": {"type": "int"}})
    n = 70000
    chunk = 8000  # server rejects inserts > 8192 rows
    for start in range(0, n, chunk):
        t.insert([{"a": i} for i in range(start, min(start + chunk, n))])
    assert count(t) == n
    assert int(scalar(t, "sum(a)")) == sum(range(n))
    assert int(scalar(t, "max(a)")) == n - 1


def test_insert_null_into_nullable_column(db):
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "varchar"}})
    t.insert([{"c1": None, "c2": "a"}])
    assert t.output(["c1", "c2"]).to_pl()[0].to_dicts() == [{"c1": None, "c2": "a"}]


def test_insert_missing_column_fills_null(db):
    """Columns absent from the insert dict are nullable-defaulted to NULL (by design)."""
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "float"}, "c3": {"type": "varchar"}})
    t.insert([{"c1": 5}])
    assert t.output(["c1", "c2", "c3"]).to_pl()[0].to_dicts() == [{"c1": 5, "c2": None, "c3": None}]


# ------------------------------------------------------------------- INSERT negatives


def test_insert_extra_unknown_column_rejected(db, inst):
    t, _ = fresh(db, {"c1": {"type": "int"}})
    with pytest.raises(InfinityException) as ei:
        t.insert([{"c1": 1, "nope": 9}])
    assert "not found" in ei.value.error_msg.lower()
    assert count(t) == 0
    assert inst.pid() is not None


def test_insert_empty_list_rejected(db, inst):
    t, _ = fresh(db, {"c1": {"type": "int"}})
    with pytest.raises(InfinityException):
        t.insert([])
    assert inst.pid() is not None


def test_insert_vector_dim_mismatch_rejected(db, inst):
    t, _ = fresh(db, {"a": {"type": "int"}, "v": {"type": "vector,4,float"}})
    t.insert([{"a": 1, "v": [1.0, 2.0, 3.0, 4.0]}])
    for bad in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]):
        with pytest.raises(InfinityException) as ei:
            t.insert([{"a": 2, "v": bad}])
        assert "embedding" in ei.value.error_msg.lower()
    assert count(t) == 1  # only the valid row survives
    assert inst.pid() is not None


def test_insert_wrong_type_should_be_rejected(db, inst):
    """DEFECT: a string into an int column is silently accepted and stored as NULL.

    Observed: insert([{"c1":"notint","c2":1.0}]) -> OK, row stored as {"c1": null,...}.
    A type mismatch must be rejected, not silently coerced to NULL.
    """
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "float"}})
    with pytest.raises(InfinityException):
        t.insert([{"c1": "notint", "c2": 1.0}])
    assert inst.pid() is not None


def test_insert_int8_overflow_should_be_rejected(db, inst):
    """DEFECT: 300 into an int8 column is silently accepted, stored as NULL."""
    t, _ = fresh(db, {"a": {"type": "int8"}})
    with pytest.raises(InfinityException):
        t.insert([{"a": 300}])
    assert inst.pid() is not None


# =========================================================================== COPY FROM


def test_import_csv_no_header(db):
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "float"}, "c3": {"type": "varchar"}})
    path = _stage_csv(t, db, [[10, 1.5, "a"], [11, 2.5, "b"]])
    t.import_data(str(path), {"header": False})
    assert t.output(["c1", "c2", "c3"]).to_pl()[0].sort("c1").to_dicts() == [
        {"c1": 10, "c2": 1.5, "c3": "a"},
        {"c1": 11, "c2": 2.5, "c3": "b"},
    ]


def test_import_csv_custom_delimiter(db, _inst):
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "varchar"}})
    path = _inst.stage_csv("delim_semi.csv", [[1, "x"], [2, "y"]], delimiter=";")
    t.import_data(str(path), {"header": False, "delimiter": ";"})
    assert count(t) == 2
    assert t.output(["c1", "c2"]).to_pl()[0].sort("c1").to_dicts() == [
        {"c1": 1, "c2": "x"}, {"c1": 2, "c2": "y"}]


def test_import_csv_header_false_reads_header_as_data(db, _inst):
    """With header=False every line is data (documents the parser's baseline)."""
    t, _ = fresh(db, {"c1": {"type": "varchar"}, "c2": {"type": "varchar"}})
    path = _inst.stage_csv("hf.csv", [["v1", "v2"]], header=["c1", "c2"])
    t.import_data(str(path), {"header": False})
    assert t.output(["c1", "c2"]).to_pl()[0].sort("c1").to_dicts() == [
        {"c1": "c1", "c2": "c2"}, {"c1": "v1", "c2": "v2"}]


def test_import_jsonl(db, _inst):
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "float"}, "c3": {"type": "varchar"}})
    path = _inst.stage_jsonl("in.jsonl", [{"c1": 1, "c2": 1.5, "c3": "aa"},
                                          {"c1": 2, "c2": 2.5, "c3": "bb"}])
    t.import_data(str(path), {"file_type": "jsonl"})
    assert t.output(["c1", "c2", "c3"]).to_pl()[0].sort("c1").to_dicts() == [
        {"c1": 1, "c2": 1.5, "c3": "aa"}, {"c1": 2, "c2": 2.5, "c3": "bb"}]


def test_import_parquet_unsupported_rejected_cleanly(db, _inst, inst):
    """PARQUET import is not supported on this build. The SDK has no parquet
    CopyFileType, and the HTTP layer rejects it with error 3032 "Not supported file
    type parquet". We assert the rejection is graceful: an error is returned, the table
    stays empty, and the server does not crash. (Coverage gap: parquet ingest is a
    documented capability the brief asked for but the engine does not implement.)
    """
    requests = pytest.importorskip("requests")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    t, name = fresh(db, {"col1": {"type": "bool"}, "col2": {"type": "int8"},
                         "col3": {"type": "int64"}, "col4": {"type": "float"}})
    tbl = pa.table({
        "col1": [True, False, True],
        "col2": pa.array([1, 2, 3], pa.int8()),
        "col3": pa.array([10, 11, 12], pa.int64()),
        "col4": pa.array([0.5, 1.5, 2.5], pa.float32()),
    })
    pqpath = _inst.data_root() / "gen.parquet"
    pq.write_table(tbl, str(pqpath))
    # SDK path: parquet is not a recognised file_type client-side.
    with pytest.raises(InfinityException):
        t.import_data(str(pqpath), {"file_type": "parquet"})
    # HTTP path: graceful rejection, not a crash.
    url = f"{_inst.http_base}/databases/default_db/tables/{name}"
    r = requests.put(url, json={"file_path": str(pqpath), "file_type": "parquet"},
                     headers={"accept": "application/json", "content-type": "application/json"})
    assert r.status_code != 200
    body = r.json()
    assert body["error_code"] == 3032 and "not supported file type" in body["error_msg"].lower()
    assert count(t) == 0
    assert inst.pid() is not None


# ------------------------------------------------------------------- COPY FROM negatives


def test_import_nonexistent_file(db, _inst, inst):
    t, _ = fresh(db, {"a": {"type": "int"}})
    with pytest.raises(InfinityException) as ei:
        t.import_data(str(_inst.data_root() / "does_not_exist.csv"), {"header": False})
    assert "isn't found" in ei.value.error_msg or "not found" in ei.value.error_msg.lower()
    assert count(t) == 0
    assert inst.pid() is not None


def test_import_too_many_columns_rejected_table_empty(db, _inst, inst):
    t, _ = fresh(db, {"a": {"type": "int"}, "b": {"type": "int"}, "c": {"type": "int"}})
    path = _inst.stage_file("too_many.csv", "1,2,3,4\n5,6,7,8\n")
    with pytest.raises(InfinityException) as ei:
        t.import_data(str(path), {"header": False})
    assert "column count mismatch" in ei.value.error_msg.lower()
    assert count(t) == 0  # rejected import must not partially apply
    assert inst.pid() is not None


def test_import_too_few_columns_should_be_rejected(db, _inst, inst):
    """DEFECT: too-FEW-column CSV is silently accepted (missing cols NULL-filled),
    while too-MANY columns is correctly rejected. The invariant a rejected/invalid
    import leaves the table empty is violated: 2 rows land with c=NULL.
    """
    t, _ = fresh(db, {"a": {"type": "int"}, "b": {"type": "int"}, "c": {"type": "int"}})
    path = _inst.stage_file("too_few.csv", "1,2\n3,4\n")
    before = count(t)
    try:
        t.import_data(str(path), {"header": False})
    except InfinityException:
        pass  # rejection is the correct behaviour
    assert count(t) == before, (
        f"too-few-column CSV silently imported {count(t)} rows: "
        f"{t.output(['a', 'b', 'c']).to_pl()[0].to_dicts()}")
    assert inst.pid() is not None


def test_import_bad_value_rejected_table_empty(db, _inst, inst):
    t, _ = fresh(db, {"a": {"type": "int"}, "b": {"type": "varchar"}})
    path = _inst.stage_file("badval.csv", "notanint,x\n")
    with pytest.raises(InfinityException) as ei:
        t.import_data(str(path), {"header": False})
    assert "parse integer" in ei.value.error_msg.lower()
    assert count(t) == 0
    assert inst.pid() is not None


def test_import_unsupported_filetype_rejected(db, _inst, inst):
    t, _ = fresh(db, {"a": {"type": "int"}})
    path = _inst.stage_file("x.csv", "1\n")
    with pytest.raises(InfinityException) as ei:
        t.import_data(str(path), {"file_type": "xml"})
    assert "unrecognized import file type" in ei.value.error_msg.lower()
    assert inst.pid() is not None


def test_import_multichar_delimiter_rejected(db, _inst, inst):
    t, _ = fresh(db, {"a": {"type": "int"}, "b": {"type": "int"}})
    path = _inst.stage_file("x.csv", "1,2\n")
    with pytest.raises(InfinityException):
        t.import_data(str(path), {"delimiter": "||"})
    assert inst.pid() is not None


def test_import_unknown_option_rejected_not_silently_ignored(db, _inst, inst):
    """A misspelt option must NOT be silently ignored (brief's worst case a)."""
    t, _ = fresh(db, {"a": {"type": "int"}})
    path = _inst.stage_file("x.csv", "1\n")
    with pytest.raises(InfinityException) as ei:
        t.import_data(str(path), {"headr": True})
    assert "unknown import parameter" in ei.value.error_msg.lower()
    assert inst.pid() is not None


def test_import_directory_should_be_rejected(db, _inst, inst):
    """DEFECT (minor): importing a directory path returns OK (0 rows) instead of erroring."""
    t, _ = fresh(db, {"a": {"type": "int"}})
    with pytest.raises(InfinityException):
        t.import_data(str(_inst.data_root()), {"header": False})
    assert inst.pid() is not None


def test_import_int8_overflow_should_be_rejected(db, _inst, inst):
    """DEFECT: CSV value 300 into an int8 column is silently accepted and stored as 0
    (and, inconsistently, as NULL when the same value is inserted via the SDK).
    """
    t, _ = fresh(db, {"a": {"type": "int8"}})
    path = _inst.stage_file("ov.csv", "300\n")
    try:
        t.import_data(str(path), {"header": False})
    except InfinityException:
        assert inst.pid() is not None
        return  # rejection would be correct
    stored = t.output(["a"]).to_pl()[0]["a"].to_list()
    assert stored == [300], f"int8 overflow silently stored {stored} instead of rejecting"
    assert inst.pid() is not None


def test_import_csv_header_true_skips_header(db, _inst, inst):
    """DEFECT: header=True does NOT skip the header row on import; the engine tries to
    parse the header cell 'c1' as an integer and fails with a parser error.
    """
    t, _ = fresh(db, {"c1": {"type": "int"}, "c2": {"type": "float"}, "c3": {"type": "varchar"}})
    path = _inst.stage_csv("headed.csv", [[100, 2.5, "x"], [101, 3.5, "y"]],
                           header=["c1", "c2", "c3"])
    try:
        t.import_data(str(path), {"header": True})
    except InfinityException as e:
        pytest.fail(f"header=True import raised instead of skipping header row: "
                    f"code={int(e.error_code)} msg={e.error_msg!r}")
    assert t.output(["c1", "c2", "c3"]).to_pl()[0].sort("c1").to_dicts() == [
        {"c1": 100, "c2": 2.5, "c3": "x"}, {"c1": 101, "c2": 3.5, "c3": "y"}]
    assert inst.pid() is not None


# =========================================================================== COPY TO


def test_export_csv_column_subset(db, _inst):
    t, _ = fresh(db, {"a": {"type": "int"}, "b": {"type": "int"}, "c": {"type": "varchar"}})
    t.insert([{"a": 1, "b": 2, "c": "hi"}, {"a": 3, "b": 4, "c": "bye"}])
    path = _unique(_inst, "subset.csv")
    t.export_data(str(path), {"header": False, "file_type": "csv"}, columns=["a", "c"])
    assert path.read_text() == "1,hi\n3,bye\n"


def test_export_jsonl(db, _inst):
    t, _ = fresh(db, {"a": {"type": "int"}, "b": {"type": "int"}, "c": {"type": "varchar"}})
    t.insert([{"a": 1, "b": 2, "c": "hi"}, {"a": 3, "b": 4, "c": "bye"}])
    path = _unique(_inst, "out.jsonl")
    t.export_data(str(path), {"file_type": "jsonl"})
    lines = [l for l in path.read_text().splitlines() if l]
    import json as _json
    got = sorted((_json.loads(l) for l in lines), key=lambda d: d["a"])
    assert got == [{"a": 1, "b": 2, "c": "hi"}, {"a": 3, "b": 4, "c": "bye"}]


def test_export_float_roundtrip_lossless(db, _inst):
    """Text formatting of doubles must round-trip exactly (classic loss point)."""
    t, _ = fresh(db, {"id": {"type": "int"}, "f": {"type": "double"}})
    vals = [0.1, 1.0 / 3.0, math.pi, 1e-7, 1234567.891234, -2.718281828459045]
    t.insert([{"id": i, "f": v} for i, v in enumerate(vals)])
    path = _unique(_inst, "flo.csv")
    t.export_data(str(path), {"header": False, "file_type": "csv"})
    t2, _ = fresh(db, {"id": {"type": "int"}, "f": {"type": "double"}})
    t2.import_data(str(path), {"header": False})
    got = t2.output(["id", "f"]).to_pl()[0].sort("id")["f"].to_list()
    assert got == vals


def test_export_large_roundtrip(db, _inst):
    t, _ = fresh(db, {"a": {"type": "int"}, "s": {"type": "varchar"}})
    t.insert([{"a": i, "s": "X" * 40} for i in range(300)])
    path = _unique(_inst, "large.csv")
    t.export_data(str(path), {"header": False, "file_type": "csv"})
    t2, _ = fresh(db, {"a": {"type": "int"}, "s": {"type": "varchar"}})
    t2.import_data(str(path), {"header": False})
    assert count(t2) == 300
    assert int(scalar(t2, "sum(a)")) == sum(range(300))
    assert set(t2.output(["s"]).to_pl()[0]["s"].to_list()) == {"X" * 40}


def test_export_autocreates_missing_directory(db, _inst):
    """Behaviour: exporting into a nonexistent directory creates it (not a defect)."""
    t, _ = fresh(db, {"a": {"type": "int"}})
    t.insert([{"a": 42}])
    path = _inst.data_root() / f"made_dir_{_counter['n']}" / "x.csv"
    assert not path.parent.exists()
    t.export_data(str(path), {"file_type": "csv"})
    assert path.exists() and path.read_text() == "42\n"


# --------------------------------------------------- COPY TO: overwrite / truncation


def test_export_overwrite_refused_leaves_original_intact(db, _inst, inst):
    """Regression around commit ccc58a3da (un-truncated writes).

    COPY TO refuses to overwrite an existing file (error 7002). We assert that a
    refused overwrite does NOT truncate or partially rewrite the original file: after
    exporting a large result then attempting to export a tiny result over the same
    path, the file is byte-identical to the large export. If a shorter write had
    truncated in place (the pre-fix bug) or left a stale tail, this fails.
    """
    big_t, _ = fresh(db, {"a": {"type": "int"}, "s": {"type": "varchar"}})
    big_t.insert([{"a": i, "s": "X" * 40} for i in range(300)])
    path = _unique(_inst, "trunc_target.csv")
    big_t.export_data(str(path), {"header": False, "file_type": "csv"})
    original = path.read_bytes()
    assert len(original) > 10000

    small_t, _ = fresh(db, {"a": {"type": "int"}, "s": {"type": "varchar"}})
    small_t.insert([{"a": 7, "s": "z"}])
    with pytest.raises(InfinityException) as ei:
        small_t.export_data(str(path), {"header": False, "file_type": "csv"})
    assert "already existed" in ei.value.error_msg.lower()
    assert path.read_bytes() == original, "refused overwrite mutated/truncated the original file"
    assert inst.pid() is not None


# --------------------------------------------------- COPY TO: DEFECTS


def test_export_csv_escapes_special_chars_roundtrip(db, _inst, inst):
    """DEFECT (blocker): CSV export does not quote/escape varchar values containing the
    delimiter, double-quotes, or newlines. The exported file is not valid RFC-4180 CSV:
    a value 'a,b' is written as a bare 'a,b' (extra column), and 'new\\nline' is written
    with a literal newline (extra row). Re-importing the export fails with a column-count
    mismatch, so an export/re-import round-trip loses data.
    """
    t, _ = fresh(db, {"id": {"type": "int"}, "s": {"type": "varchar"}})
    wild = [{"id": 1, "s": "a,b"}, {"id": 2, "s": 'q"uote'},
            {"id": 3, "s": "new\nline"}, {"id": 4, "s": "ünïçödé 🚀"}]
    t.insert(wild)
    path = _unique(_inst, "wild.csv")
    t.export_data(str(path), {"header": False, "file_type": "csv"})
    exported = path.read_text()

    t2, _ = fresh(db, {"id": {"type": "int"}, "s": {"type": "varchar"}})
    try:
        t2.import_data(str(path), {"header": False})
    except InfinityException as e:
        pytest.fail(f"export/re-import round-trip failed; export was not valid CSV: "
                    f"exported={exported!r} import_error={e.error_msg!r}")
    got = {r["id"]: r["s"] for r in t2.output(["id", "s"]).to_pl()[0].to_dicts()}
    assert got == {r["id"]: r["s"] for r in wild}
    assert inst.pid() is not None


def test_export_csv_header_true_emits_header(db, _inst, inst):
    """DEFECT: export with header=True writes no header row."""
    t, _ = fresh(db, {"a": {"type": "int"}, "c": {"type": "varchar"}})
    t.insert([{"a": 1, "c": "hi"}, {"a": 3, "c": "bye"}])
    path = _unique(_inst, "hdr_out.csv")
    t.export_data(str(path), {"header": True, "file_type": "csv"}, columns=["a", "c"])
    lines = path.read_text().splitlines()
    assert lines and lines[0] == "a,c", (
        f"export header=True emitted no header line; first line was {lines[:1]!r}")
    assert inst.pid() is not None


# --------------------------------------------------------------------------- teardown check


def test_zz_no_server_side_errors_logged(inst):
    """A clean client result over a server-logged error is a defect the SDK cannot see."""
    errs = inst.log_errors()
    # Parser/type errors from negative tests are surfaced to the client as exceptions,
    # not logged at error level; any error-level log line here is unexpected.
    assert errs == [], "server logged error-level lines during ingest suite:\n" + "\n".join(errs[:20])


# --------------------------------------------------------------------------- helpers


@pytest.fixture()
def _inst(inst):
    return inst


def _unique(inst, name: str):
    _counter["n"] += 1
    p = inst.data_root() / f"{_counter['n']}_{name}"
    if p.exists():
        p.unlink()
    return p


def _stage_csv(t, db, rows):
    inst = harness.instance("eval-insert")  # same registered instance; only for path building
    _counter["n"] += 1
    return inst.stage_csv(f"imp_{_counter['n']}.csv", rows)
