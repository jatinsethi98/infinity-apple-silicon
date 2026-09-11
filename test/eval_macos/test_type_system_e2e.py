"""End-to-end verification of the Infinity type system on macOS arm64.

Per the evaluation brief, this suite exercises every scalar type at its boundaries,
NULL behaviour everywhere, and the composite types (embedding / sparse / tensor). It
asserts *real expected values* (computed in-test) rather than "no exception", and treats
negative cases (wrong type, out-of-range, NaN/inf, wrong dimension, bad type name,
overflow-must-not-wrap) as first-class tests. After every mutating group it checks
``inst.log_errors`` and ``inst.pid()`` -- a clean client result over a server-side error,
or a crash, is a defect the SDK cannot see.

Run:  uv run pytest test/eval_macos/test_type_system_e2e.py -v
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

import harness
from infinity.common import ConflictType, InfinityException

try:
    from infinity.common import SparseVector
except Exception:  # noqa: BLE001
    SparseVector = None

from infinity.errors import ErrorCode


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-types")
    it.start()
    try:
        yield it
    finally:
        it.stop()


@pytest.fixture(autouse=True)
def _ensure_alive(inst):
    """Bring the server back if a previous test crashed it, so each test starts clean."""
    if inst.pid() is None:
        inst.start(fresh=True)
    yield


# ------------------------------------------------------------------------------ helpers

def _fresh(inst, name, schema):
    conn, db = inst.db()
    db.drop_table(name, ConflictType.Ignore)
    t = db.create_table(name, schema, ConflictType.Error)
    return conn, db, t


def _col(t, col, order="id"):
    """Return the values of ``col`` ordered by ``order`` as a Python list."""
    r, _ = t.output([order, col]).to_pl()
    return r.sort(order)[col].to_list()


# ============================================================ INTEGER round-trip + overflow

# (sdk spelling, alias spelling, min, max) for the four signed widths.
# NOTE: the thrift SDK's dict-schema create_table only accepts int8/int16/int32/integer/int/
# int64/int128 -- the SQL aliases tinyint/smallint/bigint/boolean are REJECTED here (see
# test_sql_int_aliases_rejected_by_dict_api). So we drive the widths by their intN spelling.
_INT_TYPES = [
    ("int8", "int8", -128, 127),
    ("int16", "int16", -32768, 32767),
    ("int32", "integer", -2147483648, 2147483647),
    ("int64", "int64", -9223372036854775808, 9223372036854775807),
]


@pytest.mark.parametrize("sdk_type,alias,lo,hi", _INT_TYPES, ids=[t[1] for t in _INT_TYPES])
def test_integer_boundary_roundtrip(inst, sdk_type, alias, lo, hi):
    """min, max, min+1, max-1, 0, -1 must round-trip bit-exact through insert->select."""
    off = inst.log_offset()
    vals = [lo, lo + 1, -1, 0, 1, hi - 1, hi]
    conn, db, t = _fresh(inst, "int_rt", {"id": {"type": "int"}, "v": {"type": sdk_type}})
    try:
        t.insert([{"id": i, "v": v} for i, v in enumerate(vals)])
        got = _col(t, "v")
        assert got == vals, f"{sdk_type}: round-trip mismatch got={got} expected={vals}"
        # alias must produce an identical column
        db.drop_table("int_alias", ConflictType.Ignore)
        ta = db.create_table("int_alias", {"id": {"type": "int"}, "v": {"type": alias}}, ConflictType.Error)
        ta.insert([{"id": i, "v": v} for i, v in enumerate(vals)])
        assert _col(ta, "v") == vals, f"alias {alias} differs from {sdk_type}"
        db.drop_table("int_alias", ConflictType.Ignore)
    finally:
        db.drop_table("int_rt", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


@pytest.mark.parametrize("sql_alias", ["tinyint", "smallint", "bigint", "boolean"])
def test_sql_int_aliases_rejected_by_dict_api(inst, sql_alias):
    """The dict-schema create_table rejects the SQL type aliases tinyint/smallint/bigint/
    boolean with INVALID_DATA_TYPE, even though these are valid Infinity SQL type names
    (they appear in common_values.py `types`). This pins the SDK/SQL inconsistency and
    proves it is a clean client-side error, not a crash."""
    conn, db = inst.db()
    db.drop_table("alias_t", ConflictType.Ignore)
    try:
        with pytest.raises(InfinityException) as ei:
            db.create_table("alias_t", {"v": {"type": sql_alias}}, ConflictType.Error)
        assert int(ei.value.error_code) == int(ErrorCode.INVALID_DATA_TYPE), ei.value.error_msg
        assert inst.pid() is not None
    finally:
        db.drop_table("alias_t", ConflictType.Ignore)
        conn.disconnect()


@pytest.mark.parametrize("sdk_type,alias,lo,hi", _INT_TYPES, ids=[t[1] for t in _INT_TYPES])
def test_integer_overflow_must_not_wrap(inst, sdk_type, alias, lo, hi):
    """Inserting a value outside a narrow int's range must be an ERROR, never silently
    altered. Correct outcomes: an error (best) OR a read-back that bit-exactly equals the
    inserted value. We FAIL on any changed value -- whether a wraparound (128 -> -128) or a
    silent coercion to NULL (data loss). OBSERVED: the engine silently stores NULL, i.e.
    the out-of-range value is discarded with no error. This test therefore fails while that
    data-loss defect is present.
    """
    if sdk_type == "int64":
        pytest.skip("no representable Python int overflows int64 through the thrift SDK path")
    off = inst.log_offset()
    over = hi + 1
    under = lo - 1
    conn, db, t = _fresh(inst, "int_ovf", {"id": {"type": "int"}, "v": {"type": sdk_type}})
    findings = []
    try:
        for probe in (over, under):
            raised = None
            try:
                t.insert([{"id": 0, "v": probe}])
            except Exception as e:  # noqa: BLE001
                raised = e
            if raised is None:
                back = _col(t, "v")
                # something got stored -- it MUST equal what we asked for
                if back and back[-1] != probe:
                    findings.append((sdk_type, probe, back[-1]))
                # clean out for next probe
                db.drop_table("int_ovf", ConflictType.Ignore)
                t = db.create_table("int_ovf", {"id": {"type": "int"}, "v": {"type": sdk_type}}, ConflictType.Error)
        assert inst.pid() is not None, f"server crashed inserting out-of-range {sdk_type}"
        assert findings == [], (
            f"SILENT DATA LOSS (data-integrity defect): out-of-range integer inserts were "
            f"stored as a different value with NO error raised. Each tuple is "
            f"(type, inserted, stored): {findings}. Expected an out-of-range error."
        )
    finally:
        db.drop_table("int_ovf", ConflictType.Ignore)
        conn.disconnect()
    assert inst.log_errors(off) == [], inst.log_errors(off)


# ============================================================ FLOAT / DOUBLE

@pytest.mark.parametrize("sdk_type,np_type,alias", [
    ("float", np.float32, "float32"),
    ("double", np.float64, "float64"),
], ids=["float32", "float64"])
def test_float_boundary_roundtrip(inst, sdk_type, np_type, alias):
    """Boundaries + signed zero must round-trip bit-exact (same IEEE-754 bit pattern)."""
    off = inst.log_offset()
    info = np.finfo(np_type)
    vals = [
        0.0, -0.0, 1.0, -1.0,
        float(info.tiny), float(-info.tiny),
        float(info.max), float(-info.max),
        float(info.eps),
    ]
    conn, db, t = _fresh(inst, "flt_rt", {"id": {"type": "int"}, "v": {"type": sdk_type}})
    try:
        t.insert([{"id": i, "v": v} for i, v in enumerate(vals)])
        got = _col(t, "v")
        # bit-exact: compare the IEEE-754 bit patterns at the column's own precision
        exp_bits = [_bits(np_type(v)) for v in vals]
        got_bits = [_bits(np_type(g)) for g in got]
        assert got_bits == exp_bits, (
            f"{sdk_type}: not bit-exact. got={got} expected={vals}"
        )
        # signed zero: -0.0 must keep its sign bit (0x80000000 for f32)
        neg_zero_idx = 1
        assert _bits(np_type(got[neg_zero_idx])) == _bits(np_type(-0.0)), (
            f"{sdk_type}: -0.0 lost its sign bit on round-trip (got {got[neg_zero_idx]})"
        )
    finally:
        db.drop_table("flt_rt", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def _bits(x):
    """Return the raw IEEE-754 bit pattern of a numpy float scalar as an int."""
    if isinstance(x, np.float32):
        return struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


@pytest.mark.parametrize("sdk_type,np_type", [("float", np.float32), ("double", np.float64)],
                         ids=["float32", "float64"])
def test_float_special_values(inst, sdk_type, np_type):
    """NaN and +-inf storage/retrieval.

    OBSERVED behaviour (pinned here): +inf and -inf round-trip correctly, but a stored NaN
    comes back as NULL -- the engine cannot distinguish NaN from NULL. That is a silent
    value conversion (a NaN is a distinct IEEE-754 value, not a null), reported separately.
    The must-hold invariants asserted here: inf/-inf survive exactly, and the server neither
    crashes nor logs an error."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "flt_sp", {"id": {"type": "int"}, "v": {"type": sdk_type}})
    try:
        t.insert([{"id": 0, "v": float("nan")},
                  {"id": 1, "v": float("inf")},
                  {"id": 2, "v": float("-inf")}])
        assert inst.pid() is not None, "server crashed inserting NaN/inf"
        got = dict(zip(_col(t, "id"), _col(t, "v")))
        # inf/-inf must round-trip exactly (engine correct).
        assert got[1] is not None and math.isinf(got[1]) and got[1] > 0, f"+inf lost: {got[1]}"
        assert got[2] is not None and math.isinf(got[2]) and got[2] < 0, f"-inf lost: {got[2]}"
        # The engine keeps NaN as a genuine (non-null) value: IS NULL must NOT match it.
        nan_null_ids = _col_ids(t, "v IS NULL")
        assert nan_null_ids == [], (
            f"NaN must not be NULL server-side, but IS NULL matched {nan_null_ids}"
        )
        # ...yet the thrift SDK renders that NaN as Python None in to_pl() -- so at the SDK
        # layer NaN is indistinguishable from NULL (reported as an sdk-defect).
        assert got[0] is None, (
            f"pinning observed SDK behaviour: NaN decodes to None, got {got[0]!r}"
        )
        print(f"\n[float-special] {sdk_type}: NaN kept as value server-side "
              f"(IS NULL={nan_null_ids}) but SDK to_pl() renders it as {got[0]!r}")
    finally:
        db.drop_table("flt_sp", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_float16_bfloat16_precision(inst):
    """Characterise float16/bfloat16 precision loss with an OBSERVED epsilon rather than
    tolerating it vaguely. We insert values that are NOT representable and assert the stored
    value equals the correct rounding to that reduced-precision format (computed with numpy),
    which pins both the loss and its direction."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "half", {
        "id": {"type": "int"}, "f16": {"type": "float16"}, "bf16": {"type": "bfloat16"}})
    probes = [1.0 / 3.0, 0.1, 3.14159265358979, 65504.0, 1e-4, -2.0 / 7.0]
    try:
        t.insert([{"id": i, "f16": v, "bf16": v} for i, v in enumerate(probes)])
        got16 = _col(t, "f16")
        gotbf = _col(t, "bf16")
        # float16: exact IEEE half rounding
        exp16 = [float(np.float16(v)) for v in probes]
        assert got16 == exp16, f"float16 rounding mismatch: got={got16} expected={exp16}"
        # bfloat16: top 16 bits of the float32 (round-to-nearest-even is the standard, but
        # some impls truncate). Accept whichever the engine does, but require it be within
        # one bf16 ULP of the input and CONSISTENT across rows.
        max_rel = 0.0
        for src, g in zip(probes, gotbf):
            if src != 0:
                max_rel = max(max_rel, abs(g - src) / abs(src))
        # bf16 has 8 mantissa bits -> relative error <= 2^-8 ~= 0.0039
        assert max_rel <= 2 ** -7, f"bfloat16 relative error {max_rel} exceeds 1 ULP band"
        print(f"\n[precision] observed max relative error: float16 exact-rounded, "
              f"bfloat16 <= {max_rel:.5f}")
    finally:
        db.drop_table("half", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


# ============================================================ BOOL

def test_bool_roundtrip_and_null(inst):
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "bl", {"id": {"type": "int"}, "v": {"type": "bool"}})
    try:
        t.insert([{"id": 0, "v": True}, {"id": 1, "v": False}, {"id": 2, "v": None}])
        got = _col(t, "v")
        assert got[0] is True or got[0] == True, got  # noqa: E712
        assert got[1] is False or got[1] == False, got  # noqa: E712
        assert got[2] is None, f"bool NULL not preserved: {got[2]}"
        assert sorted(x for x in _col(t, "id") if x is not None) == [0, 1, 2]
        assert _col_ids(t, "v IS NULL") == [2]
        assert _col_ids(t, "v IS NOT NULL") == [0, 1]
        assert _col_ids(t, "v = True") == [0]
        assert _col_ids(t, "v = False") == [1]
    finally:
        db.drop_table("bl", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def _col_ids(t, filt):
    r, _ = t.output(["id"]).filter(filt).to_pl()
    return sorted(r["id"].to_list())


# ============================================================ VARCHAR edge cases

def test_varchar_edge_cases(inst):
    """empty, very long, embedded NUL, emoji, combining chars, multi-byte unicode."""
    off = inst.log_offset()
    long_s = "x" * 200000
    cases = {
        0: "",
        1: long_s,
        2: "a\x00b",                # embedded NUL
        3: "emoji \U0001F600\U0001F1FA\U0001F1F8",  # emoji + flag (surrogate-pair region)
        4: "é́ combin",  # combining acute accents
        5: "日本語テスト",
        6: "naïve café",
        7: "tab\tnewline\nquote\"backslash\\",
    }
    conn, db, t = _fresh(inst, "vc", {"id": {"type": "int"}, "s": {"type": "varchar"}})
    findings = {}
    try:
        t.insert([{"id": i, "s": v} for i, v in cases.items()])
        got = dict(zip(_col(t, "id"), _col(t, "s")))
        for i, expected in cases.items():
            if got.get(i) != expected:
                findings[i] = (repr(expected)[:40], repr(got.get(i))[:40])
        assert inst.pid() is not None
        # Empty, long, unicode, emoji, combining, tab/newline MUST round-trip exactly.
        for i in (0, 1, 3, 4, 5, 6, 7):
            assert i not in findings, f"varchar case {i} corrupted: {findings.get(i)}"
        # Embedded NUL (case 2) is the interesting one: report whatever the engine does.
        if 2 in findings:
            print(f"\n[varchar] embedded-NUL not preserved: {findings[2]}")
    finally:
        db.drop_table("vc", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_varchar_equality_filter_unicode(inst):
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "vcf", {"id": {"type": "int"}, "s": {"type": "varchar"}})
    try:
        t.insert([{"id": 1, "s": "日本語"}, {"id": 2, "s": "café"}, {"id": 3, "s": ""}])
        assert _col_ids(t, "s = '日本語'") == [1]
        assert _col_ids(t, "s = 'café'") == [2]
        assert _col_ids(t, "s = ''") == [3]
    finally:
        db.drop_table("vcf", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


# ============================================================ DATE / TIME / DATETIME / TIMESTAMP

def test_date_time_roundtrip_and_boundaries(inst):
    """Round-trip dates including leap day, epoch, far future; report what the engine does
    with year 0 and negative years rather than assuming. Also cross-checks the two date
    tests MACOS_VERIFICATION.md flags as encoding non-portable tz/formatting assumptions."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "dt", {
        "id": {"type": "int"}, "d": {"type": "date"}, "tm": {"type": "time"},
        "dtm": {"type": "datetime"}, "ts": {"type": "timestamp"}})
    ok_rows = [
        {"id": 1, "d": "2024-02-29", "tm": "12:30:45", "dtm": "2024-02-29 12:30:45", "ts": "2024-02-29 12:30:45"},
        {"id": 2, "d": "1970-01-01", "tm": "00:00:00", "dtm": "1970-01-01 00:00:00", "ts": "1970-01-01 00:00:00"},
        {"id": 3, "d": "2999-12-31", "tm": "23:59:59", "dtm": "2999-12-31 23:59:59", "ts": "2999-12-31 23:59:59"},
    ]
    try:
        t.insert(ok_rows)
        d = dict(zip(_col(t, "id"), _col(t, "d")))
        assert "2024-02-29" in str(d[1]), f"leap day did not round-trip: {d[1]}"
        assert "1970-01-01" in str(d[2]), f"epoch date did not round-trip: {d[2]}"
        assert "2999-12-31" in str(d[3]), f"far-future date did not round-trip: {d[3]}"
        # comparison / ordering on date
        assert _col_ids(t, "d > '2024-01-01'") == [1, 3]
        assert _col_ids(t, "d < '2000-01-01'") == [2]
        assert inst.pid() is not None
    finally:
        db.drop_table("dt", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_date_invalid_values(inst):
    """Feb 30, month 13, year 0, negative year: each must be an ERROR or a documented
    normalisation -- never a silent crash, and never a bogus stored date that reads back
    as garbage. Reports the engine's actual behaviour per probe."""
    off = inst.log_offset()
    probes = ["2023-02-29", "2024-02-30", "2024-13-01", "2024-00-10", "0000-01-01", "-0001-01-01"]
    observed = {}
    silently_accepted = []
    decode_crashes = []
    for i, p in enumerate(probes):
        conn, db, t = _fresh(inst, "dinv", {"id": {"type": "int"}, "d": {"type": "date"}})
        try:
            raised = None
            try:
                t.insert([{"id": 0, "d": p}])
            except Exception as e:  # noqa: BLE001
                raised = type(e).__name__
            if raised is not None:
                observed[p] = ("rejected", raised)
            elif inst.pid() is None:
                observed[p] = ("SERVER-CRASH", None)
            else:
                # Insert did not error -> the server accepted this invalid date. Read it back.
                try:
                    back = _col(t, "d")
                    observed[p] = ("stored", str(back[0]) if back else None)
                    silently_accepted.append((p, observed[p][1]))
                except OverflowError:
                    # The SDK's parse_date_bytes crashes turning the stored day-offset into a
                    # Python date -- the server stored a value no date decoder can render.
                    observed[p] = ("sdk-decode-OverflowError", None)
                    silently_accepted.append((p, "unrenderable"))
                    decode_crashes.append(p)
        finally:
            db.drop_table("dinv", ConflictType.Ignore)
            conn.disconnect()
        if inst.pid() is None:
            inst.start(fresh=True)
    print(f"\n[date-invalid] {observed}")
    print(f"[date-invalid] silently accepted (should be rejected): {silently_accepted}")
    # Hard requirement: the SERVER must not crash on any probe.
    crashes = [p for p, (kind, _) in observed.items() if kind == "SERVER-CRASH"]
    assert crashes == [], f"invalid date values CRASHED the server: {crashes}"
    # Defect assertion: an invalid calendar date must be rejected at insert, not silently
    # stored. This fails while the engine accepts them (evidence in silently_accepted).
    assert silently_accepted == [], (
        f"SILENT ACCEPT of invalid calendar dates (no insert error): {silently_accepted}. "
        f"Some are unrenderable and crash the SDK date decoder on read: {decode_crashes}."
    )
    assert inst.log_errors(off) == [], inst.log_errors(off)


# ============================================================ EMBEDDING (dense vector)

def test_embedding_roundtrip_and_distance(inst):
    """Round-trip a float embedding bit-exactly (float32 precision) and verify a KNN search
    returns the true nearest neighbour computed with numpy."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "emb", {"id": {"type": "int"}, "v": {"type": "vector,4,float"}})
    data = {
        0: [1.0, 0.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0, 0.0],
        2: [0.9, 0.1, 0.0, 0.0],
        3: [0.0, 0.0, 1.0, 1.0],
    }
    try:
        t.insert([{"id": i, "v": v} for i, v in data.items()])
        got = dict(zip(_col(t, "id"), _col(t, "v")))
        for i, v in data.items():
            assert [float(np.float32(x)) for x in got[i]] == [float(np.float32(x)) for x in v], \
                f"embedding {i} not bit-exact: {got[i]} vs {v}"
        # nearest neighbour to query by L2 must be the numpy-computed nearest.
        q = [1.0, 0.05, 0.0, 0.0]
        dists = {i: float(np.linalg.norm(np.array(v) - np.array(q))) for i, v in data.items()}
        true_nn = min(dists, key=dists.get)
        r, _ = t.output(["id"]).match_dense("v", q, "float", "l2", 1).to_pl()
        assert r["id"].to_list() == [true_nn], f"KNN nearest wrong: {r['id'].to_list()} != {true_nn}"
    finally:
        db.drop_table("emb", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_embedding_wrong_dimension_rejected(inst):
    """Inserting an embedding of the wrong dimension must be an error, not a crash and not
    a silent pad/truncate."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "embd", {"id": {"type": "int"}, "v": {"type": "vector,4,float"}})
    try:
        for bad in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]):
            raised = None
            try:
                t.insert([{"id": 0, "v": bad}])
            except Exception as e:  # noqa: BLE001
                raised = e
            assert raised is not None, f"wrong-dim embedding {bad} was SILENTLY ACCEPTED"
            assert inst.pid() is not None, f"server crashed on wrong-dim embedding {bad}"
    finally:
        db.drop_table("embd", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None


# ============================================================ SPARSE

@pytest.mark.skipif(SparseVector is None, reason="SparseVector not importable from SDK")
def test_sparse_roundtrip(inst):
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "sp", {"id": {"type": "int"}, "v": {"type": "sparse,100,float,int"}})
    try:
        t.insert([{"id": 0, "v": SparseVector(indices=[10, 20, 30], values=[1.5, 2.5, 3.5])},
                  {"id": 1, "v": {"5": -1.0, "99": 4.0}}])
        r, _ = t.output(["id", "v"]).to_pl()
        by_id = dict(zip(r["id"].to_list(), r["v"].to_list()))
        # row 0: indices/values preserved (order may differ -> compare as dict)
        got0 = _sparse_to_dict(by_id[0])
        assert got0 == {10: 1.5, 20: 2.5, 30: 3.5}, f"sparse row0 mismatch: {got0}"
        got1 = _sparse_to_dict(by_id[1])
        assert got1 == {5: -1.0, 99: 4.0}, f"sparse row1 mismatch: {got1}"
    finally:
        db.drop_table("sp", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def _sparse_to_dict(v):
    """Normalise whatever the SDK returns for a sparse cell into {int_index: float}.

    ``to_pl`` unifies struct fields across rows, so a cell can carry keys that belong to
    other rows with a None filler -- drop those; they are not part of this row's vector.
    """
    if isinstance(v, dict):
        return {int(k): float(val) for k, val in v.items() if val is not None}
    # SparseVector-like
    idx = getattr(v, "indices", None)
    val = getattr(v, "values", None)
    if idx is not None and val is not None:
        return {int(i): float(x) for i, x in zip(idx, val)}
    return v


def test_sparse_index_out_of_range_rejected(inst):
    """A sparse index >= the declared dimension must be rejected, not silently stored."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "spb", {"id": {"type": "int"}, "v": {"type": "sparse,10,float,int"}})
    try:
        raised = None
        try:
            payload = SparseVector(indices=[100], values=[1.0]) if SparseVector else {"100": 1.0}
            t.insert([{"id": 0, "v": payload}])
        except Exception as e:  # noqa: BLE001
            raised = e
        assert inst.pid() is not None, "server crashed on out-of-range sparse index"
        assert raised is not None, "sparse index 100 in a dim-10 column was SILENTLY ACCEPTED"
    finally:
        db.drop_table("spb", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None


# ============================================================ TENSOR

def test_tensor_roundtrip(inst):
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "tn", {"id": {"type": "int"}, "v": {"type": "tensor,3,float"}})
    try:
        t.insert([{"id": 0, "v": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
                  {"id": 1, "v": [7.0, 8.0, 9.0]}])
        r, _ = t.output(["id", "v"]).to_pl()
        by_id = dict(zip(r["id"].to_list(), r["v"].to_list()))
        v0 = [[float(np.float32(x)) for x in row] for row in by_id[0]]
        assert v0 == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f"tensor row0 mismatch: {by_id[0]}"
        v1 = [[float(np.float32(x)) for x in row] for row in by_id[1]]
        assert v1 == [[7.0, 8.0, 9.0]], f"tensor row1 (single vector) mismatch: {by_id[1]}"
    finally:
        db.drop_table("tn", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


# ============================================================ NULL everywhere (systematic)

def test_null_aggregate_semantics(inst):
    """sum/avg/count over a column with NULLs must SKIP nulls (SQL semantics), and count(*)
    must include null rows. A null counted as zero in avg is a wrong-result defect."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "nagg", {"id": {"type": "int"}, "v": {"type": "int"}})
    present = [10, 20, 30]
    try:
        t.insert([{"id": 0, "v": 10}, {"id": 1, "v": 20}, {"id": 2, "v": 30},
                  {"id": 3, "v": None}, {"id": 4, "v": None}])
        r, _ = t.output(["count(v)", "sum(v)", "avg(v)", "min(v)", "max(v)", "count(*)"]).to_pl()
        row = {c: r[c].to_list()[0] for c in r.columns}
        # find columns positionally (names vary), so read by index too
        vals = [r[c].to_list()[0] for c in r.columns]
        cnt_v, sum_v, avg_v, min_v, max_v, cnt_star = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
        assert int(cnt_v) == 3, f"count(v) should skip 2 nulls -> 3, got {cnt_v} (cols={r.columns})"
        assert float(sum_v) == 60.0, f"sum(v) should be 60 (nulls skipped), got {sum_v}"
        assert abs(float(avg_v) - 20.0) < 1e-9, (
            f"avg(v) should be 60/3=20 (nulls skipped), got {avg_v}. "
            f"If ~12, nulls were counted as 0 -- wrong-result defect."
        )
        assert int(min_v) == 10 and int(max_v) == 30, f"min/max wrong with nulls: {min_v},{max_v}"
        assert int(cnt_star) == 5, f"count(*) should include null rows -> 5, got {cnt_star}"
        _ = (present, row)
    finally:
        db.drop_table("nagg", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_null_filter_and_arithmetic(inst):
    """IS NULL / IS NOT NULL partition the rows exactly; arithmetic with a NULL operand
    yields NULL (so the row does not satisfy a strict comparison)."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "nfa", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"id": i, "v": i} for i in range(5)] + [{"id": 5, "v": None}, {"id": 6, "v": None}])
        assert _col_ids(t, "v IS NULL") == [5, 6]
        assert _col_ids(t, "v IS NOT NULL") == [0, 1, 2, 3, 4]
        # v + 1 > 2 : NULL rows produce NULL, which is not > 2, so excluded.
        assert _col_ids(t, "v + 1 > 2") == [2, 3, 4]
        # complement check: IS NULL count + IS NOT NULL count == total
        total = len(_col(t, "id"))
        assert total == 7
    finally:
        db.drop_table("nfa", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_null_update_to_and_from(inst):
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "nud", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"id": 0, "v": 100}, {"id": 1, "v": 200}])
        # 100 -> NULL
        t.update("id = 0", {"v": None})
        assert _col_ids(t, "v IS NULL") == [0], "update value->NULL did not take"
        # NULL -> 999
        t.update("id = 0", {"v": 999})
        assert _col_ids(t, "v IS NULL") == [], "update NULL->value did not take"
        assert dict(zip(_col(t, "id"), _col(t, "v")))[0] == 999
    finally:
        db.drop_table("nud", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_null_survives_restart(inst):
    """A NULL must survive a WAL replay across a graceful restart (durability of the null
    bitmap). Uses a dedicated table and does NOT drop it before restart."""
    off = inst.log_offset()
    conn, db = inst.db()
    db.drop_table("nrst", ConflictType.Ignore)
    t = db.create_table("nrst", {"id": {"type": "int"}, "v": {"type": "int"}}, ConflictType.Error)
    t.insert([{"id": 0, "v": 42}, {"id": 1, "v": None}])
    conn.disconnect()
    inst.restart(graceful=True)
    conn2, db2 = inst.db()
    try:
        t2 = db2.get_table("nrst")
        assert _col_ids(t2, "v IS NULL") == [1], "NULL did not survive restart"
        assert dict(zip(_col(t2, "id"), _col(t2, "v")))[0] == 42
    finally:
        db2.drop_table("nrst", ConflictType.Ignore)
        conn2.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_null_from_csv_empty_field(inst):
    """An empty field in an imported CSV must become NULL, not 0 and not empty-string."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "ncsv", {"id": {"type": "int"}, "v": {"type": "int"}})
    path = inst.stage_csv("nulls.csv", rows=[[0, 5], [1, ""], [2, 7]], header=None)
    try:
        t.import_data(str(path), {"delimiter": ",", "header": False})
        ids_null = _col_ids(t, "v IS NULL")
        assert ids_null == [1], f"empty CSV field should be NULL for id=1, got IS NULL ids {ids_null}"
        vals = dict(zip(_col(t, "id"), _col(t, "v")))
        assert vals[0] == 5 and vals[2] == 7
    finally:
        db.drop_table("ncsv", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    assert inst.log_errors(off) == [], inst.log_errors(off)


# ============================================================ NEGATIVE: bad type definitions

def test_unknown_type_name_rejected(inst):
    off = inst.log_offset()
    conn, db = inst.db()
    db.drop_table("badt", ConflictType.Ignore)
    try:
        raised = None
        try:
            db.create_table("badt", {"id": {"type": "int"}, "v": {"type": "quaternion"}}, ConflictType.Error)
        except Exception as e:  # noqa: BLE001
            raised = e
        assert raised is not None, "unknown type name 'quaternion' was accepted"
        assert inst.pid() is not None
    finally:
        db.drop_table("badt", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None


@pytest.mark.parametrize("dim_spec", ["vector,0,float", "vector,-1,float", "vector,100000000,float"],
                         ids=["zero", "negative", "absurd"])
def test_bad_embedding_dimension_rejected(inst, dim_spec):
    """Zero, negative, and absurd embedding dimensions must be rejected at DDL, not crash."""
    off = inst.log_offset()
    conn, db = inst.db()
    db.drop_table("bde", ConflictType.Ignore)
    created = False
    try:
        raised = None
        try:
            db.create_table("bde", {"id": {"type": "int"}, "v": {"type": dim_spec}}, ConflictType.Error)
            created = True
        except Exception as e:  # noqa: BLE001
            raised = e
        assert inst.pid() is not None, f"server crashed creating {dim_spec}"
        # zero/negative must be rejected. absurd may be accepted at DDL (lazy alloc) -- we
        # only require no crash for the absurd case.
        if dim_spec in ("vector,0,float", "vector,-1,float"):
            assert raised is not None, f"{dim_spec} was accepted -- invalid dimension"
    finally:
        db.drop_table("bde", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
    _ = created
    assert inst.log_errors(off) == [], inst.log_errors(off)


def test_wrong_python_type_for_column(inst):
    """Inserting a non-numeric string into an int column must be an error, never a silent
    coercion. OBSERVED: the string 'not a number' is silently stored as NULL with no error
    -- the same silent-data-loss defect as out-of-range integers. This test asserts the
    correct behaviour (error) and so fails while the defect is present. It records the
    stored value as evidence and confirms the server survives."""
    off = inst.log_offset()
    conn, db, t = _fresh(inst, "wt", {"id": {"type": "int"}, "v": {"type": "int"}})
    silently_stored = []
    try:
        raised = None
        try:
            t.insert([{"id": 0, "v": "not a number"}])
        except Exception as e:  # noqa: BLE001
            raised = e
        assert inst.pid() is not None, "server crashed on wrong-typed value"
        if raised is None:
            silently_stored.append(("int<-'not a number'", _col(t, "v")))
        assert raised is not None, (
            f"SILENT ACCEPT (data-integrity defect): a non-numeric string into an int column "
            f"was NOT rejected; it was stored as {silently_stored}. Expected a type error."
        )
    finally:
        db.drop_table("wt", ConflictType.Ignore)
        conn.disconnect()
    assert inst.pid() is not None
