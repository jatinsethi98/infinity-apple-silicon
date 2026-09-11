"""Regression tests for inputs that kill the server process.

Every test here corresponds to a confirmed way for an ordinary client to take the whole
server down with a single query. They are grouped in one file because they share a
requirement that shapes the fixture: **each test needs a fresh server**, since a test that
kills the server would otherwise cascade into every test after it. That cascade is how two
of these defects were originally found — a suite with a module-scoped server reported one
real failure followed by twenty-nine "Could not connect", which reads like a broken
harness rather than a crashed engine.

Each crash was confirmed on a fresh server per case, with a non-crashing control that
isolates the responsible code path.

Defect 1 — folded predicate on a Secondary-indexed integer column
-----------------------------------------------------------------
Any comparison the optimizer folds to always-false or always-true crashes the server, but
only when the column carries a Secondary index. On an int8 column (range [-128, 127])::

    WHERE v > 127     -- always false      WHERE v >= -128   -- always true
    WHERE v < -128    -- always false      WHERE v <= 127    -- always true

Confirmed for int8, int16, int32 and int64, for ``>``, ``>=``, ``<``, ``<=`` and ``=``, and
through an ``AND`` compound. NOT reproducible for ``!=``, for ``OR``, for a float column,
or without the index. These are ordinary boundary conditions, not malformed input, which
is what makes it serious: any client with query access can kill the server, and a user can
trip it by accident.

Mechanism: ``FilterExpressionPushDownHelper::UnwindCast``
(``src/planner/optimizer/index_scan/filter_expression_push_down_helper_impl.cpp``, the
early return at :357 and the three ``IntegralContinueUnwind`` false branches at :395,
:403, :411) signals the folded case with a sentinel tuple
``{0, Value::MakeNull(), nullptr, kAlwaysFalse|kAlwaysTrue}`` — column id ``0``, and a
**null expression pointer**. The contract is that callers check ``compare_type`` before
touching the rest. Two consumers in
``src/planner/optimizer/index_scan/filter_expression_push_down_indexscanfilter_impl.cpp``
do not: ``SolveForColVal`` (:567-576) calls
``new_candidate_column_index_map_.at(column_id)`` with the sentinel ``0``, and
``SolveForFuncVal`` (:578-590) dereferences the null pointer outright via
``static_cast<FunctionExpression *>(base_expression.get())->ExtractFunctionInfo()``.

Observed under lldb: ``EXC_BAD_ACCESS`` at ``ldr x8, [x8, #0x8]`` / ``blr x8`` — a call
through a garbage vtable slot — inside the heavily-inlined
``InfinityThriftService::Select``. The process dies **without writing an error or critical
line to infinity.log**, so it never reaches the engine's own error paths.

The switch at ``filter_expression_push_down_indexscanfilter_impl.cpp:824-829`` *does*
handle these compare types correctly (returning ``IndexFilterEvaluatorAllTrue`` /
``AllFalse``); the two lambdas run earlier and never reach it. Any fix must keep
always-false and always-true distinct — collapsing them would turn a crash into silently
wrong results, which is worse.

Defect 2 — match_sparse topn is not validated
---------------------------------------------
``match_sparse`` with ``topn = -1`` or ``topn = 2**63-1`` crashes the server.
``match_dense`` rejects the same values cleanly with an ``InfinityException``, so the
validation exists in the engine and is simply missing on the sparse path. ``topn = 0`` is
rejected on both paths.

Defect 3 — string into a numeric column is silently stored as NULL
------------------------------------------------------------------
Not a crash, but the same class of missing validation, and kept here so it is not lost.
Inserting ``"abc"`` into an INTEGER column succeeds (thrift: no exception; HTTP: 200 with
``error_code: 0``) and stores NULL. ``3.7`` into an INTEGER column silently stores ``4``.
Meanwhile ``True`` into an INTEGER column IS rejected with
``Can't cast from Boolean to Integer@src/function/cast/bool_cast.cppm:67`` — so the cast
layer validates some conversions and silently substitutes NULL for others. The client is
told the write succeeded either way.
"""

from __future__ import annotations

import pytest
from infinity import index
from infinity.common import ConflictType, SparseVector

import harness

INSTANCE = "eval-spare-b"


@pytest.fixture()
def inst():
    """A fresh server per test — these tests kill the server.

    Module or session scope would make every test after the first crash fail for the wrong
    reason, which is precisely how these defects were originally mis-reported.
    """
    server = harness.instance(INSTANCE)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _assert_alive(inst, what: str):
    assert inst.pid() is not None, f"SERVER PROCESS DIED (SIGSEGV) on {what}"


# --------------------------------------------------------------------------------------
# Defect 1: folded predicate on a Secondary-indexed integer column
# --------------------------------------------------------------------------------------

# int8 is used deliberately: its bounds are small, literal, and unambiguously in range for
# the type, which rules out "the literal overflowed" as an explanation. The sample value is
# 126 rather than 127 because inserting INT8_MAX and then indexing it fails separately with
# "The value 127 is reserved as a sentinel" (see test_int8_max_is_indexable below), and that
# must not mask this defect.
_INT8_ROWS = [{"id": 0, "v": 1}, {"id": 1, "v": 2}, {"id": 2, "v": 126}]


def _indexed_int_table(inst, coltype: str = "int8", with_index: bool = True, rows=None):
    conn, db = inst.db()
    db.drop_table("t", ConflictType.Ignore)
    table = db.create_table("t", {"id": {"type": "int"}, "v": {"type": coltype}})
    table.insert(rows if rows is not None else _INT8_ROWS)
    if with_index:
        table.create_index("idx_v", index.IndexInfo("v", index.IndexType.Secondary))
    return conn, table


@pytest.mark.parametrize("filt,folds_to,expected_rows", [
    ("v > 127", "always-false", 0),
    ("v < -128", "always-false", 0),
    ("v >= -128", "always-true", 3),
    ("v <= 127", "always-true", 3),
])
def test_folded_predicate_does_not_crash(inst, filt, folds_to, expected_rows):
    """Survival AND the right answer.

    The row count is asserted as well as survival, so that a 'fix' which mapped
    always-true onto always-false — turning a crash into silently wrong results — could
    not pass this test.
    """
    conn, table = _indexed_int_table(inst)
    try:
        result = table.output(["id"]).filter(filt).to_pl()[0]
    except Exception as exc:  # noqa: BLE001 - report what actually happened
        _assert_alive(inst, f"filter({filt!r}) [{folds_to}]")
        pytest.fail(f"filter({filt!r}) [{folds_to}] raised {type(exc).__name__}: {exc}")
    _assert_alive(inst, f"filter({filt!r}) [{folds_to}]")
    assert result.height == expected_rows, (
        f"filter({filt!r}) folds to {folds_to}; expected {expected_rows} rows, "
        f"got {result.height}"
    )
    assert inst.log_errors() == []
    conn.disconnect()


@pytest.mark.parametrize("coltype,filt", [
    ("int8", "v > 127"),
    ("int16", "v > 32767"),
    ("int", "v > 2147483647"),
    ("int64", "v > 9223372036854775807"),
])
def test_folded_predicate_across_integer_widths(inst, coltype, filt):
    """Not specific to one integer width."""
    conn, table = _indexed_int_table(inst, coltype=coltype)
    try:
        result = table.output(["id"]).filter(filt).to_pl()[0]
    except Exception as exc:  # noqa: BLE001
        _assert_alive(inst, f"{coltype} filter({filt!r})")
        pytest.fail(f"{coltype} filter({filt!r}) raised {type(exc).__name__}: {exc}")
    _assert_alive(inst, f"{coltype} filter({filt!r})")
    assert result.height == 0
    conn.disconnect()


def test_folded_predicate_in_and_compound(inst):
    """An AND compound containing a folded predicate reaches the same path."""
    conn, table = _indexed_int_table(inst)
    try:
        result = table.output(["id"]).filter("v > 127 AND id >= 0").to_pl()[0]
    except Exception as exc:  # noqa: BLE001
        _assert_alive(inst, "AND compound with a folded predicate")
        pytest.fail(f"AND compound raised {type(exc).__name__}: {exc}")
    _assert_alive(inst, "AND compound with a folded predicate")
    assert result.height == 0
    conn.disconnect()


@pytest.mark.parametrize("filt,expected_rows", [
    ("v > 126", 0),
    ("v >= -127", 3),
    ("v < 127", 3),
])
def test_ordinary_predicate_control(inst, filt, expected_rows):
    """Control: predicates one step inside the bound must keep working.

    Without this, a 'fix' that disabled the secondary index-scan path altogether would
    look like a pass.
    """
    conn, table = _indexed_int_table(inst)
    result = table.output(["id"]).filter(filt).to_pl()[0]
    _assert_alive(inst, f"ordinary filter({filt!r})")
    assert result.height == expected_rows
    conn.disconnect()


def test_no_index_control(inst):
    """Control: without the Secondary index the same predicate is handled correctly.

    This is what localises the defect to the index-scan filter path rather than to
    expression evaluation or the type system.
    """
    conn, table = _indexed_int_table(inst, with_index=False)
    result = table.output(["id"]).filter("v > 127").to_pl()[0]
    _assert_alive(inst, "no-index control")
    assert result.height == 0
    conn.disconnect()


def test_int8_max_is_indexable(inst):
    """Separate defect: INT8_MAX cannot be carried by a Secondary index.

    Inserting 127 into an int8 column and then building a Secondary index over it fails
    with a thrift-level ``TApplicationException: The value 127 is reserved as a sentinel``.
    127 is a legal int8 value, so either it must be indexable or the engine must reject it
    at insert time with a real error — failing at index-build time with a
    serialization-layer message is neither.
    """
    conn, table = _indexed_int_table(
        inst, rows=[{"id": 0, "v": 1}, {"id": 1, "v": 127}]
    )
    assert table.output(["id"]).filter("v = 127").to_pl()[0].height == 1
    conn.disconnect()


# --------------------------------------------------------------------------------------
# Defect 2: match_sparse does not validate topn
# --------------------------------------------------------------------------------------

def _sparse_table(inst):
    conn, db = inst.db()
    db.drop_table("sp", ConflictType.Ignore)
    table = db.create_table("sp", {"id": {"type": "int"},
                                   "v": {"type": "sparse,100,float,int"}})
    table.insert([{"id": i, "v": SparseVector([i, i + 1], [1.0, 2.0])} for i in range(5)])
    table.create_index("idx_sp", index.IndexInfo("v", index.IndexType.BMP))
    return conn, table


@pytest.mark.parametrize("topn", [-1, 2**63 - 1])
def test_match_sparse_invalid_topn_is_rejected_not_fatal(inst, topn):
    """An invalid topn must produce an error, not kill the server.

    ``match_dense`` already rejects these values cleanly (see the control below), so the
    engine has the validation — the sparse path just does not reach it.
    """
    conn, table = _sparse_table(inst)
    query = table.output(["id"]).match_sparse(
        "v", SparseVector([0, 1], [1.0, 1.0]), "ip", topn
    )
    with pytest.raises(Exception):
        query.to_pl()
    _assert_alive(inst, f"match_sparse(topn={topn})")
    conn.disconnect()


@pytest.mark.parametrize("topn,expected_rows", [(1, 1), (3, 3), (1000, 5)])
def test_match_sparse_valid_topn_control(inst, topn, expected_rows):
    """Control: valid topn values work, including one larger than the row count."""
    conn, table = _sparse_table(inst)
    result = table.output(["id"]).match_sparse(
        "v", SparseVector([0, 1], [1.0, 1.0]), "ip", topn
    ).to_pl()[0]
    _assert_alive(inst, f"match_sparse(topn={topn})")
    assert result.height == expected_rows
    conn.disconnect()


@pytest.mark.parametrize("topn", [0, -1])
def test_match_dense_invalid_topn_control(inst, topn):
    """Control: the dense path rejects the same invalid topn cleanly and survives.

    This is what makes defect 2 a missing-validation bug on one code path rather than an
    undefined-input question.
    """
    conn, db = inst.db()
    db.drop_table("dn", ConflictType.Ignore)
    table = db.create_table("dn", {"id": {"type": "int"}, "v": {"type": "vector,4,float"}})
    table.insert([{"id": i, "v": [float(i)] * 4} for i in range(5)])
    table.create_index("idx_dn", index.IndexInfo(
        "v", index.IndexType.Hnsw,
        {"m": "16", "ef_construction": "50", "metric": "l2"},
    ))
    with pytest.raises(Exception):
        table.output(["id"]).match_dense("v", [1.0] * 4, "float", "l2", topn).to_pl()
    _assert_alive(inst, f"match_dense(topn={topn})")
    conn.disconnect()


# --------------------------------------------------------------------------------------
# Defect 3: silent type coercion on insert
# --------------------------------------------------------------------------------------

def test_string_into_integer_column_is_rejected(inst):
    """A non-numeric string must not be silently stored as NULL.

    Currently the insert reports success and the row holds NULL, so the caller has no way
    to know the value it supplied was discarded.
    """
    conn, db = inst.db()
    db.drop_table("tc", ConflictType.Ignore)
    table = db.create_table("tc", {"id": {"type": "int"}, "n": {"type": "int"}})
    with pytest.raises(Exception):
        table.insert([{"id": 1, "n": "abc"}])
    _assert_alive(inst, 'insert "abc" into an INTEGER column')
    assert table.output(["id"]).to_pl()[0].height == 0, (
        'insert of "abc" into an INTEGER column was accepted; the row must not exist'
    )
    conn.disconnect()


def test_float_into_integer_column_is_rejected_or_documented(inst):
    """3.7 into an INTEGER column is silently truncated to 4.

    Rounding a value the caller supplied without telling them is a data-integrity problem
    whichever direction it rounds. Pinned here so the behaviour is a decision rather than
    an accident.
    """
    conn, db = inst.db()
    db.drop_table("tc2", ConflictType.Ignore)
    table = db.create_table("tc2", {"id": {"type": "int"}, "n": {"type": "int"}})
    table.insert([{"id": 1, "n": 3.7}])
    stored = table.output(["n"]).to_pl()[0]
    _assert_alive(inst, "insert 3.7 into an INTEGER column")
    assert stored.height == 1
    pytest.fail(
        f"3.7 inserted into an INTEGER column was silently stored as {stored['n'][0]} "
        "with no error or warning"
    )


def test_bool_into_integer_column_is_rejected_control(inst):
    """Control: the cast layer DOES reject bool -> integer.

    Its error is ``Can't cast from Boolean to Integer@src/function/cast/bool_cast.cppm:67``.
    That the same layer silently substitutes NULL for a failed string parse is the
    inconsistency defect 3 is about.
    """
    conn, db = inst.db()
    db.drop_table("tc3", ConflictType.Ignore)
    table = db.create_table("tc3", {"id": {"type": "int"}, "n": {"type": "int"}})
    with pytest.raises(Exception):
        table.insert([{"id": 1, "n": True}])
    _assert_alive(inst, "insert True into an INTEGER column")
    conn.disconnect()


# --------------------------------------------------------------------------------------
# Defect 4: flush_catalog() segfaults the server
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("populate", [False, True], ids=["empty-db", "with-data"])
def test_flush_catalog_does_not_crash(inst, populate):
    """``conn.flush_catalog()`` kills the server with SIGSEGV, 100% of the time.

    Reproduced on an empty database and on one holding 100 rows. The server logs a
    checkpoint transaction and then ``[critical] Error: Segmentation fault: 11``::

        [critical] Txn ID: 2, Text: checkpoint, Begin TS: 1, Commit TS: 4, ... State: Committed
        [critical] Error: Segmentation fault: 11

    This compounds the WAL-corruption defect below. That one loses every commit since the
    last checkpoint, and the shipped ``checkpoint_interval`` is 86400s — so the obvious
    mitigation is to take a checkpoint yourself before risky work. This is the API for
    doing that, and calling it takes the server down.
    """
    conn, db = inst.db()
    if populate:
        db.drop_table("fc", ConflictType.Ignore)
        table = db.create_table("fc", {"id": {"type": "int"}})
        table.insert([{"id": i} for i in range(100)])
    flush = getattr(conn, "flush_catalog", None)
    if flush is None:
        pytest.skip("this SDK build exposes no flush_catalog()")
    try:
        flush()
    except Exception as exc:  # noqa: BLE001
        _assert_alive(inst, "flush_catalog()")
        pytest.fail(f"flush_catalog() raised {type(exc).__name__}: {exc}")
    _assert_alive(inst, "flush_catalog()")
    conn.disconnect()


# --------------------------------------------------------------------------------------
# Defect 5: one corrupt WAL entry deletes the whole WAL and aborts startup
# --------------------------------------------------------------------------------------

def test_single_corrupt_wal_entry_does_not_destroy_the_database():
    """A single flipped bit in the WAL must not discard committed data or abort startup.

    Observed, with ``wal_flush = full_fsync`` so every commit was durably synced:

    1. ``[warning] Found bad wal entry .../wal.log@0`` — the corruption IS detected.
    2. ``[warning] Remove wal log .../wal.log`` — the engine then deletes the **entire WAL
       file**, not just the bad entry or the tail after it. Every acknowledged commit since
       the last checkpoint is gone, and ``wal/`` is left empty.
    3. With no WAL, no checkpoint can be found:
       ``[critical] WAL replay: No checkpoint found in wal@src/storage/wal/wal_manager_impl.cpp:803``
    4. That calls ``UnrecoverableError``, which throws ``UnrecoverableException``. Nothing
       catches it on the startup path, so ``TerminateHandler`` runs and the process dies with
       ``Abort trap: 6``.

    Two things make this worse than it first looks. The shipped ``checkpoint_interval`` is
    86400s, so in a fresh or lightly-used database there is no checkpoint at all and the
    window of loss is every commit ever made. And the abort happens **after** the server has
    bound its PostgreSQL, HTTP, thrift and peer listeners, so a health check that only probes
    a port sees a healthy server moments before it dies.

    This also bounds what commit b2d59374a bought: the WAL is now genuinely fsynced, but a
    single corrupt byte in it means the durably-written data is discarded anyway.

    Correct behaviour is to truncate at the first bad entry and replay everything before it,
    or to refuse to start with a diagnosable error — not to delete the log and abort.

    Uses its own server rather than the shared fixture because it must corrupt files while
    the server is stopped and then restart against them.
    """
    server = harness.instance(INSTANCE)
    server.start()
    try:
        conn, db = server.db()
        db.drop_table("walt", ConflictType.Ignore)
        table = db.create_table("walt", {"id": {"type": "int"}, "v": {"type": "varchar"}})
        table.insert([{"id": i, "v": f"row-{i}"} for i in range(200)])
        assert table.output(["count(*)"]).to_pl()[0].row(0)[0] == 200
        conn.disconnect()
    finally:
        server.stop()

    wal_dir = server.root / "wal"
    wals = sorted(wal_dir.glob("wal*"), key=lambda p: p.stat().st_mtime)
    assert wals, f"expected a WAL file under {wal_dir}"
    target = wals[-1]

    # Offset 40 is deep inside the first record's payload: past the 24-byte WalEntryHeader
    # (i32 size_ + u32 checksum_ + i64 txn_id_ + TxnTimeStamp commit_ts_) and well clear of
    # the trailing size field, so the size guards still pass and only the CRC differs. That
    # is what routes execution into the checksum-mismatch branch specifically.
    with open(target, "r+b") as fh:
        fh.seek(40)
        original = fh.read(1)
        fh.seek(40)
        fh.write(bytes([original[0] ^ 0x40]))

    started = True
    try:
        server.start(fresh=False)
    except harness.HarnessError:
        started = False

    try:
        assert started, (
            f"server aborted on startup after one bit was flipped in {target.name}; "
            "the WAL was deleted and no checkpoint remained"
        )
        assert wal_dir.exists() and any(wal_dir.iterdir()), (
            "the WAL directory was emptied: one corrupt entry deleted the whole log"
        )
        conn, db = server.db()
        recovered = db.get_table("walt").output(["count(*)"]).to_pl()[0].row(0)[0]
        conn.disconnect()
        assert recovered == 200, (
            f"expected the 200 durably-committed rows to survive one corrupt WAL entry, "
            f"recovered {recovered}"
        )
    finally:
        server.stop()
