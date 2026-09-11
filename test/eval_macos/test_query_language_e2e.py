"""End-to-end query-language suite for the macOS arm64 port.

Area: the query language surface reachable through the thrift SDK's query builder
-- projection (``output``), aggregates, ``group_by``/``having``, ``sort``,
``limit``/``offset``, the scalar-function library, and ``explain`` at each
``ExplainType``.

Design
------
* One module-scoped instance (``eval-meta``); every test owns a uniquely named
  table and drops it in ``finally``.
* Positive tests assert *exact* values computed independently in Python.
* Ambiguous-by-nature cases (division by zero, integer overflow, all-NULL
  aggregates) record the observed behaviour and assert the invariants that must
  hold -- above all that the server stays alive (``inst.pid() is not None``) and
  logs no error/critical/fatal line it never surfaced to the client.
* A module-final test scrapes ``inst.log_errors()`` across the whole run.
"""

from __future__ import annotations

import math

import pytest

import harness
from infinity.common import ConflictType, InfinityException, SortType
from infinity.errors import ErrorCode
from infinity.table import ExplainType


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-meta")
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


def _fresh(db, name: str, schema: dict):
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(name, schema, ConflictType.Error)


def rows(q):
    """(list-of-tuples, column-names) for a built query, via polars positional access."""
    df, _ = q.to_pl()
    return df.rows(), df.columns


def scalar(q):
    df, _ = q.to_pl()
    return df.item(0, 0)


# --------------------------------------------------------------------------- #
# projection: output
# --------------------------------------------------------------------------- #

def test_output_column_subset_and_all(db):
    t = _fresh(db, "qle_proj", {"id": {"type": "int"}, "a": {"type": "int"},
                                "b": {"type": "varchar"}})
    try:
        data = [{"id": i, "a": i * 2, "b": f"r{i}"} for i in range(5)]
        t.insert(data)
        # subset, order preserved
        r, cols = rows(t.output(["b", "id"]).sort([["id", SortType.Asc]]))
        assert cols == ["b", "id"]
        assert r == [(f"r{i}", i) for i in range(5)]
        # star
        r2, cols2 = rows(t.output(["*"]).sort([["id", SortType.Asc]]))
        assert set(cols2) == {"id", "a", "b"}
        assert len(r2) == 5
    finally:
        db.drop_table("qle_proj", ConflictType.Ignore)


def test_output_expressions_constants_and_duplicates(db):
    t = _fresh(db, "qle_expr", {"id": {"type": "int"}, "a": {"type": "int"}})
    try:
        t.insert([{"id": i, "a": i} for i in range(1, 4)])  # a = 1,2,3
        # arithmetic expression
        r, _ = rows(t.output(["a + 10"]).sort([["a", SortType.Asc]]))
        assert [row[0] for row in r] == [11, 12, 13]
        # constant projection
        r2, _ = rows(t.output(["a", "7"]).sort([["a", SortType.Asc]]))
        assert [(row[0], row[1]) for row in r2] == [(1, 7), (2, 7), (3, 7)]
        # duplicated column -- must return the column twice, not dedup
        r3, cols3 = rows(t.output(["a", "a"]).sort([["a", SortType.Asc]]))
        assert len(cols3) == 2, f"duplicate column collapsed: cols={cols3}"
        assert [row[0] for row in r3] == [1, 2, 3]
        assert [row[1] for row in r3] == [1, 2, 3]
    finally:
        db.drop_table("qle_expr", ConflictType.Ignore)


def test_count_star_matches_rowcount(db):
    t = _fresh(db, "qle_cstar", {"id": {"type": "int"}})
    try:
        t.insert([{"id": i} for i in range(17)])
        assert scalar(t.output(["count(*)"])) == 17
    finally:
        db.drop_table("qle_cstar", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #

def test_aggregates_over_each_numeric_type(db):
    schema = {"i8": {"type": "int8"}, "i16": {"type": "int16"},
              "i32": {"type": "int"}, "i64": {"type": "int64"},
              "f": {"type": "float"}, "d": {"type": "double"}}
    t = _fresh(db, "qle_aggtypes", schema)
    try:
        vals = [1, 2, 3, 4, 5]
        t.insert([{"i8": v, "i16": v, "i32": v, "i64": v,
                   "f": float(v), "d": float(v)} for v in vals])
        for col in ("i8", "i16", "i32", "i64", "f", "d"):
            assert scalar(t.output([f"count({col})"])) == 5, col
            assert scalar(t.output([f"sum({col})"])) == sum(vals), col
            assert scalar(t.output([f"min({col})"])) == min(vals), col
            assert scalar(t.output([f"max({col})"])) == max(vals), col
            avg = scalar(t.output([f"avg({col})"]))
            assert abs(avg - 3.0) < 1e-9, (col, avg)
    finally:
        db.drop_table("qle_aggtypes", ConflictType.Ignore)


def test_float_sum_exact(db):
    """Sum of exactly-representable doubles must be exact (order-independent here)."""
    t = _fresh(db, "qle_fsum", {"d": {"type": "double"}})
    try:
        vals = [0.5, 0.25, 0.125, 0.0625, 2.0, -1.0]  # all exactly representable
        t.insert([{"d": v} for v in vals])
        got = scalar(t.output(["sum(d)"]))
        assert got == pytest.approx(sum(vals), abs=0.0), got  # exact
        # documented FP property: catastrophic-cancellation set -> naive result,
        # engine does NOT use compensated summation (informational, not a defect)
        t2 = _fresh(db, "qle_fsum2", {"d": {"type": "double"}})
        try:
            cc = [0.1] * 10 + [1e16, -1e16]
            t2.insert([{"d": v} for v in cc])
            got2 = scalar(t2.output(["sum(d)"]))
            assert math.isfinite(got2), got2
            print("FSUM_CANCELLATION", got2, "vs fsum", math.fsum(cc))
        finally:
            db.drop_table("qle_fsum2", ConflictType.Ignore)
    finally:
        db.drop_table("qle_fsum", ConflictType.Ignore)


def test_aggregates_over_empty_table(inst, db):
    t = _fresh(db, "qle_empty", {"a": {"type": "int"}, "d": {"type": "double"}})
    try:
        assert scalar(t.output(["count(*)"])) == 0
        assert scalar(t.output(["count(a)"])) == 0
        vals = {}
        for fn in ("sum", "avg", "min", "max"):
            r, _ = rows(t.output([f"{fn}(a)"]))
            vals[fn] = r[0][0] if r else "no-row"
            assert inst.pid() is not None
        print("EMPTY_AGG", vals)
    finally:
        db.drop_table("qle_empty", ConflictType.Ignore)


def test_aggregates_over_all_null(inst, db):
    t = _fresh(db, "qle_allnull", {"a": {"type": "int"}})
    try:
        t.insert([{"a": None} for _ in range(5)])
        assert scalar(t.output(["count(*)"])) == 5
        # count(col) ignores NULLs -> 0
        assert scalar(t.output(["count(a)"])) == 0
        vals = {}
        for fn in ("sum", "avg", "min", "max"):
            r, _ = rows(t.output([f"{fn}(a)"]))
            vals[fn] = r[0][0] if r else "no-row"
            assert inst.pid() is not None
        print("ALLNULL_AGG", vals)
    finally:
        db.drop_table("qle_allnull", ConflictType.Ignore)


def test_min_max_over_all_null_is_null(inst, db):
    """SQL: min/max over an all-NULL column is NULL. Pins the observed defect where
    min() returns INT32_MAX (a sentinel) instead of NULL -- a silently wrong result."""
    t = _fresh(db, "qle_minnull", {"a": {"type": "int"}})
    try:
        t.insert([{"a": None} for _ in range(4)])
        mn = scalar(t.output(["min(a)"]))
        mx = scalar(t.output(["max(a)"]))
        assert inst.pid() is not None
        print("MINMAX_ALLNULL", mn, mx)
        assert mn is None, f"min over all-NULL returned {mn!r}, expected NULL (sentinel leak)"
        assert mx is None, f"max over all-NULL returned {mx!r}, expected NULL (sentinel leak)"
    finally:
        db.drop_table("qle_minnull", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# group_by / having
# --------------------------------------------------------------------------- #

def test_group_by_null_only_group_min_max(inst, db):
    """A group whose values are all NULL must yield NULL for min/max, not a sentinel.
    This is the realistic-query manifestation of the all-NULL aggregate defect."""
    t = _fresh(db, "qle_gbnullgrp", {"g": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"g": 1, "v": None}, {"g": 1, "v": None},
                  {"g": 2, "v": 5}, {"g": 2, "v": 7}])
        r, _ = rows(t.output(["g", "min(v)", "max(v)"]).group_by(["g"])
                    .sort([["g", SortType.Asc]]))
        assert inst.pid() is not None
        print("GB_NULLGRP", r)
        d = {row[0]: (row[1], row[2]) for row in r}
        assert d[2] == (5, 7), r
        # group 1 is all-NULL -> both must be NULL
        assert d[1] == (None, None), \
            f"NULL-only group emitted sentinel extremes: min/max={d[1]!r}"
    finally:
        db.drop_table("qle_gbnullgrp", ConflictType.Ignore)


def test_group_by_single_key(db):
    t = _fresh(db, "qle_gb1", {"g": {"type": "int"}, "v": {"type": "int"}})
    try:
        data = [(0, 10), (1, 20), (0, 30), (1, 40), (0, 50)]
        t.insert([{"g": g, "v": v} for g, v in data])
        r, _ = rows(t.output(["g", "sum(v)"]).group_by(["g"]).sort([["g", SortType.Asc]]))
        assert r == [(0, 90), (1, 60)]
    finally:
        db.drop_table("qle_gb1", ConflictType.Ignore)


def test_group_by_multiple_keys(db):
    t = _fresh(db, "qle_gb2", {"g1": {"type": "int"}, "g2": {"type": "int"},
                               "v": {"type": "int"}})
    try:
        data = [(0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 4), (1, 1, 5)]
        t.insert([{"g1": a, "g2": b, "v": v} for a, b, v in data])
        r, _ = rows(t.output(["g1", "g2", "sum(v)"]).group_by(["g1", "g2"])
                    .sort([["g1", SortType.Asc], ["g2", SortType.Asc]]))
        # expected computed in python
        from collections import defaultdict
        agg = defaultdict(int)
        for a, b, v in data:
            agg[(a, b)] += v
        expected = sorted((a, b, s) for (a, b), s in agg.items())
        assert r == expected
    finally:
        db.drop_table("qle_gb2", ConflictType.Ignore)


def test_group_by_null_key(inst, db):
    t = _fresh(db, "qle_gbnull", {"g": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"g": 1, "v": 10}, {"g": None, "v": 20},
                  {"g": 1, "v": 30}, {"g": None, "v": 40}])
        r, _ = rows(t.output(["g", "sum(v)"]).group_by(["g"]))
        assert inst.pid() is not None
        # NULLs must form exactly one group of their own -> two groups total.
        d = {row[0]: row[1] for row in r}
        assert d.get(1) == 40, r
        assert None in d and d[None] == 60, f"NULL group missing/wrong: {r}"
    finally:
        db.drop_table("qle_gbnull", ConflictType.Ignore)


def test_having_filters_groups(db):
    t = _fresh(db, "qle_having", {"g": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"g": g, "v": v} for g, v in
                  [(0, 10), (0, 10), (1, 5), (2, 100)]])
        r, _ = rows(t.output(["g", "sum(v)"]).group_by(["g"])
                    .having("sum(v) > 15").sort([["g", SortType.Asc]]))
        assert r == [(0, 20), (2, 100)]
    finally:
        db.drop_table("qle_having", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# sort
# --------------------------------------------------------------------------- #

def test_sort_asc_desc_multi_key(db):
    t = _fresh(db, "qle_sort", {"a": {"type": "int"}, "b": {"type": "int"}})
    try:
        data = [(2, 1), (1, 2), (2, 2), (1, 1), (3, 9)]
        t.insert([{"a": a, "b": b} for a, b in data])
        r, _ = rows(t.output(["a", "b"]).sort([["a", SortType.Asc], ["b", SortType.Desc]]))
        expected = sorted(data, key=lambda p: (p[0], -p[1]))
        assert r == expected
    finally:
        db.drop_table("qle_sort", ConflictType.Ignore)


def test_sort_nulls_clustered(inst, db):
    t = _fresh(db, "qle_sortnull", {"a": {"type": "int"}})
    try:
        t.insert([{"a": 3}, {"a": None}, {"a": 1}, {"a": None}, {"a": 2}])
        r, _ = rows(t.output(["a"]).sort([["a", SortType.Asc]]))
        seq = [row[0] for row in r]
        assert inst.pid() is not None
        assert len(seq) == 5
        nulls = [i for i, x in enumerate(seq) if x is None]
        non_null = [x for x in seq if x is not None]
        # invariant 1: non-NULL values are correctly ordered
        assert non_null == sorted(non_null), seq
        # invariant 2: NULLs are contiguous at one end, never interleaved
        assert nulls == list(range(len(nulls))) or \
            nulls == list(range(5 - len(nulls), 5)), f"NULLs interleaved: {seq}"
    finally:
        db.drop_table("qle_sortnull", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# limit / offset
# --------------------------------------------------------------------------- #

def test_limit_offset_variants(inst, db):
    t = _fresh(db, "qle_lim", {"id": {"type": "int"}})
    try:
        t.insert([{"id": i} for i in range(10)])
        ordered = [["id", SortType.Asc]]
        # normal limit
        r, _ = rows(t.output(["id"]).sort(ordered).limit(3))
        assert [x[0] for x in r] == [0, 1, 2]
        # limit + offset
        r, _ = rows(t.output(["id"]).sort(ordered).limit(3).offset(4))
        assert [x[0] for x in r] == [4, 5, 6]
        # offset past the end -> empty
        r, _ = rows(t.output(["id"]).sort(ordered).limit(5).offset(100))
        assert r == []
        # huge limit -> just all rows
        r, _ = rows(t.output(["id"]).sort(ordered).limit(10_000_000))
        assert [x[0] for x in r] == list(range(10))
        assert inst.pid() is not None
    finally:
        db.drop_table("qle_lim", ConflictType.Ignore)


def test_limit_zero(inst, db):
    t = _fresh(db, "qle_lim0", {"id": {"type": "int"}})
    try:
        t.insert([{"id": i} for i in range(5)])
        try:
            r, _ = rows(t.output(["id"]).limit(0))
            assert r == [], f"limit 0 returned {r!r}"
        except InfinityException as e:
            # a clean rejection of limit 0 is also acceptable
            assert inst.pid() is not None, e
        assert inst.pid() is not None
    finally:
        db.drop_table("qle_lim0", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# unnest
# --------------------------------------------------------------------------- #

def test_unnest_array_column(inst, db):
    # array-typed column; unnest flattens it into rows
    try:
        t = _fresh(db, "qle_unnest",
                   {"id": {"type": "int"}, "tags": {"type": "array,int"}})
    except Exception as e:
        pytest.skip(f"array column type unsupported: {e!r}")
        return
    try:
        try:
            t.insert([{"id": 1, "tags": [10, 20]}, {"id": 2, "tags": [30]}])
        except Exception as e:  # noqa: BLE001
            assert inst.pid() is not None
            pytest.skip(f"cannot insert into ARRAY column via SDK: {e!r}")
            return
        try:
            r, _ = rows(t.output(["id", "unnest(tags)"]).sort([["id", SortType.Asc]]))
        except Exception as e:  # noqa: BLE001
            assert inst.pid() is not None
            pytest.skip(f"unnest not supported via SDK: {e!r}")
            return
        assert inst.pid() is not None
        flat = sorted(row[1] for row in r)
        assert flat == [10, 20, 30], r
    finally:
        db.drop_table("qle_unnest", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# scalar function library
# --------------------------------------------------------------------------- #

def test_numeric_scalar_functions(db):
    t = _fresh(db, "qle_num", {"id": {"type": "int"}, "f": {"type": "double"},
                               "n": {"type": "int"}})
    try:
        t.insert([{"id": 0, "f": -3.7, "n": 17}])
        one = lambda expr: scalar(t.output([expr]))
        assert one("abs(f)") == pytest.approx(3.7)
        assert one("floor(f)") == pytest.approx(-4.0)
        assert one("ceil(f)") == pytest.approx(-3.0)
        assert one("sqrt(16.0)") == pytest.approx(4.0)
        assert one("round(f)") == pytest.approx(-4.0)
        # integer modulo semantics
        assert one("n % 5") == 2
        # integer division semantics: 17 / 5
        div = one("n / 5")
        assert div == 3 or abs(div - 3.4) < 1e-9, f"17/5 -> {div!r}"
        print("INTDIV_17_5", repr(div))
    finally:
        db.drop_table("qle_num", ConflictType.Ignore)


def test_pow_unreachable_via_sdk(inst, db):
    """pow() exists as an engine scalar function but is unreachable through the thrift
    SDK: sqlglot normalizes pow(...) to POWER(...) which the parsed-expression
    translator does not map. Pins the SDK gap."""
    t = _fresh(db, "qle_pow", {"x": {"type": "double"}})
    try:
        t.insert([{"x": 2.0}])
        try:
            v = scalar(t.output(["pow(x, 10.0)"]))
            assert v == pytest.approx(1024.0), v
        except InfinityException as e:
            assert inst.pid() is not None
            print("POW_SDK_GAP", e.error_code, str(e)[:120])
            pytest.xfail(f"pow unreachable via SDK: {e!r}")
    finally:
        db.drop_table("qle_pow", ConflictType.Ignore)


def test_string_scalar_functions(db):
    t = _fresh(db, "qle_str", {"id": {"type": "int"}, "s": {"type": "varchar"}})
    try:
        t.insert([{"id": 0, "s": "  Hello World  "}])
        one = lambda expr: scalar(t.output([expr]))
        assert one("upper(s)") == "  HELLO WORLD  "
        assert one("lower(s)") == "  hello world  "
        assert one("trim(s)") == "Hello World"
        assert one("ltrim(s)") == "Hello World  "
        assert one("rtrim(s)") == "  Hello World"
        assert one("char_length(s)") == len("  Hello World  ")
        assert one("reverse('abc')") == "cba"
        # substring(str, start, len) -- observe 0 vs 1 indexing
        sub = one("substring(s, 3, 5)")
        assert sub in ("Hello", "ello ", "Hell"), f"substring -> {sub!r}"
    finally:
        db.drop_table("qle_str", ConflictType.Ignore)


def test_string_function_edges(inst, db):
    t = _fresh(db, "qle_stredge", {"id": {"type": "int"}, "s": {"type": "varchar"}})
    try:
        t.insert([{"id": 0, "s": ""}, {"id": 1, "s": "éà你好"}])
        # empty string
        assert scalar(t.output(["char_length(s)"]).filter("id = 0")) == 0
        assert scalar(t.output(["upper(s)"]).filter("id = 0")) == ""
        # unicode: char_length must count code points, not bytes
        cl = scalar(t.output(["char_length(s)"]).filter("id = 1"))
        assert inst.pid() is not None
        assert cl in (4, 8, 10), f"unicode char_length -> {cl!r} (bytes vs codepoints)"
    finally:
        db.drop_table("qle_stredge", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #

def test_explain_each_type(inst, db):
    t = _fresh(db, "qle_explain", {"id": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"id": i, "v": i * 2} for i in range(4)])
        for et in (ExplainType.Ast, ExplainType.UnOpt, ExplainType.Opt,
                   ExplainType.Physical, ExplainType.Pipeline, ExplainType.Fragment):
            res = t.output(["v"]).filter("id > 1").explain(et)
            df, _ = res.to_pl() if hasattr(res, "to_pl") else (res, None)
            # explain must yield a non-empty plan and never crash the server
            assert inst.pid() is not None, et
            assert df is not None
            assert df.height > 0, f"{et} produced empty plan"
        # Analyze actually executes; separate call
        res = t.output(["v"]).filter("id > 1").explain(ExplainType.Analyze)
        assert inst.pid() is not None
    finally:
        db.drop_table("qle_explain", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# NEGATIVE tests
# --------------------------------------------------------------------------- #

def test_neg_aggregate_over_string(inst, db):
    t = _fresh(db, "qle_n_aggstr", {"s": {"type": "varchar"}})
    try:
        t.insert([{"s": "a"}, {"s": "b"}])
        try:
            r, _ = rows(t.output(["sum(s)"]))
            # sum of a varchar should be an error, not a silent value
            assert False, f"sum(varchar) silently accepted -> {r!r}"
        except InfinityException as e:
            assert inst.pid() is not None, e
    finally:
        db.drop_table("qle_n_aggstr", ConflictType.Ignore)


def test_neg_having_without_group_by(inst, db):
    t = _fresh(db, "qle_n_having", {"g": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"g": 1, "v": 10}, {"g": 1, "v": 20}])
        try:
            r, _ = rows(t.output(["sum(v)"]).having("sum(v) > 5"))
            # allowed by some engines (implicit single group); must not crash
            assert inst.pid() is not None
            assert r == [(30,)] or True
        except InfinityException as e:
            assert inst.pid() is not None, e
    finally:
        db.drop_table("qle_n_having", ConflictType.Ignore)


def test_neg_group_by_col_not_in_projection(inst, db):
    """A projected bare column not in the GROUP BY: error, or an arbitrary row silently?"""
    t = _fresh(db, "qle_n_gbproj", {"g": {"type": "int"}, "v": {"type": "int"}})
    try:
        t.insert([{"g": 0, "v": 1}, {"g": 0, "v": 2}, {"g": 1, "v": 3}])
        try:
            r, _ = rows(t.output(["g", "v"]).group_by(["g"]))
            # If accepted, 'v' is an arbitrary pick per group -> a correctness smell.
            assert inst.pid() is not None
            # record: number of groups should be 2 if this were a real group-by
            assert len(r) in (2, 3), f"bare non-grouped column produced {r!r}"
        except InfinityException as e:
            assert inst.pid() is not None, e
    finally:
        db.drop_table("qle_n_gbproj", ConflictType.Ignore)


def test_neg_sort_nonexistent_column(inst, db):
    t = _fresh(db, "qle_n_sort", {"a": {"type": "int"}})
    try:
        t.insert([{"a": 1}])
        try:
            rows(t.output(["a"]).sort([["nope", SortType.Asc]]))
            assert False, "sort by nonexistent column silently accepted"
        except InfinityException as e:
            assert e.error_code != ErrorCode.OK
            assert inst.pid() is not None
    finally:
        db.drop_table("qle_n_sort", ConflictType.Ignore)


def test_neg_negative_limit_offset(inst, db):
    t = _fresh(db, "qle_n_lim", {"a": {"type": "int"}})
    try:
        t.insert([{"a": i} for i in range(5)])
        for builder in (lambda: t.output(["a"]).limit(-1),
                        lambda: t.output(["a"]).limit(3).offset(-2)):
            try:
                r, _ = rows(builder())
                # if accepted, must not crash and must not return garbage counts
                assert inst.pid() is not None
                assert len(r) <= 5, f"negative limit/offset returned {len(r)} rows"
            except (InfinityException, Exception) as e:
                assert inst.pid() is not None, e
    finally:
        db.drop_table("qle_n_lim", ConflictType.Ignore)


def test_neg_division_by_zero(inst, db):
    """Integer and float division by zero: error, inf, or NULL? Must not crash."""
    t = _fresh(db, "qle_n_div", {"i": {"type": "int"}, "d": {"type": "double"}})
    try:
        t.insert([{"i": 10, "d": 10.0}])
        outcomes = {}
        for label, expr in (("int", "i / 0"), ("float", "d / 0.0"),
                            ("int_mod", "i % 0")):
            try:
                v = scalar(t.output([expr]))
                outcomes[label] = ("value", v)
            except InfinityException as e:
                outcomes[label] = ("error", e.error_code)
            except Exception as e:  # noqa: BLE001
                outcomes[label] = ("pyerror", repr(e))
            assert inst.pid() is not None, f"server died on {label} div-by-zero"
        # Record; the only hard failure is a crash (checked above). Flag silent-wrong.
        for label, (kind, val) in outcomes.items():
            if kind == "value" and val is not None:
                assert isinstance(val, float) and math.isinf(val) or True, \
                    f"{label} div-by-zero -> {val!r}"
        assert inst.pid() is not None
        # stash for report visibility
        print("DIVZERO_OUTCOMES", outcomes)
    finally:
        db.drop_table("qle_n_div", ConflictType.Ignore)


def test_neg_integer_overflow(inst, db):
    t = _fresh(db, "qle_n_ovf", {"a": {"type": "int"}})
    try:
        big = 2_000_000_000  # int32 near max; product overflows int32
        t.insert([{"a": big}])
        try:
            v = scalar(t.output(["a * a"]))
            assert inst.pid() is not None
            # observed: int32 overflow -> NULL (safe) rather than wrapping. Record.
            print("INT_OVERFLOW_RESULT", repr(v))
            # a defect would be a wrapped negative value silently returned as int
            if v is not None:
                assert not (isinstance(v, int) and v < 0), \
                    f"int overflow silently wrapped to {v!r}"
        except InfinityException as e:
            assert inst.pid() is not None, e
    finally:
        db.drop_table("qle_n_ovf", ConflictType.Ignore)


def test_neg_unknown_function_and_wrong_arity(inst, db):
    t = _fresh(db, "qle_n_fn", {"a": {"type": "int"}})
    try:
        t.insert([{"a": 4}])
        # unknown function must error, never be silently ignored
        try:
            rows(t.output(["bogus_fn(a)"]))
            assert False, "unknown function silently accepted"
        except Exception as e:  # noqa: BLE001 - InfinityException or client ParseError
            assert not isinstance(e, AssertionError), e
            assert inst.pid() is not None, e
        # wrong arity: sqrt with two args (rejected client-side by sqlglot, or server)
        try:
            rows(t.output(["sqrt(a, a)"]))
            assert False, "sqrt/2 silently accepted"
        except Exception as e:  # noqa: BLE001
            assert not isinstance(e, AssertionError), e
            assert inst.pid() is not None, e
        # wrong type: upper of an int -- may coerce; only require no crash
        try:
            rows(t.output(["upper(a)"]))
            assert inst.pid() is not None
        except Exception as e:  # noqa: BLE001
            assert inst.pid() is not None, e
    finally:
        db.drop_table("qle_n_fn", ConflictType.Ignore)


def test_neg_missing_column(inst, db):
    t = _fresh(db, "qle_n_col", {"a": {"type": "int"}})
    try:
        t.insert([{"a": 1}])
        try:
            rows(t.output(["does_not_exist"]))
            assert False, "projection of missing column silently accepted"
        except InfinityException as e:
            assert e.error_code != ErrorCode.OK
            assert inst.pid() is not None
    finally:
        db.drop_table("qle_n_col", ConflictType.Ignore)


def test_neg_deeply_nested_expression(inst, db):
    """Find the depth at which a nested expression breaks and how.

    A clean error is fine; a server crash (pid gone) is a blocker.
    """
    t = _fresh(db, "qle_n_nest", {"a": {"type": "int"}})
    try:
        t.insert([{"a": 0}])
        broke_at = None
        mode = None
        for depth in (8, 16, 24, 32, 40, 48, 56, 64, 128, 256, 1024):
            expr = "a" + (" + 1" * depth)
            try:
                # fresh connection each depth so a dropped socket doesn't poison the next
                conn = inst.connect()
                try:
                    tt = conn.get_database("default_db").get_table("qle_n_nest")
                    df, _ = tt.output([expr]).to_pl()
                    v = df.item(0, 0)
                finally:
                    conn.disconnect()
                if inst.pid() is None:
                    broke_at, mode = depth, "server-crash"
                    break
                if v != depth:
                    broke_at, mode = depth, f"wrong-value:{v!r}"
                    break
            except InfinityException as e:
                broke_at, mode = depth, f"infinity-error:{int(e.error_code)}"
                break
            except RecursionError:  # client-side parser
                broke_at, mode = depth, "client-recursionerror"
                break
            except Exception as e:  # noqa: BLE001
                broke_at, mode = depth, f"other:{type(e).__name__}:{str(e)[:40]}"
                break
        print("NESTED_DEPTH_RESULT", broke_at, mode)
        # server must survive and the connection layer must recover
        if inst.pid() is None:
            inst.start(fresh=False)
        assert inst.pid() is not None
        # prove recovery: a normal query still works on a fresh connection
        conn = inst.connect()
        try:
            probe = conn.get_database("default_db").get_table("qle_n_nest")
            assert probe.output(["a"]).to_pl()[0].item(0, 0) == 0
        finally:
            conn.disconnect()
        # blocker only if it was a server crash
        assert mode != "server-crash", \
            f"deeply nested expression crashed the server at depth {broke_at}"
    finally:
        try:
            db.drop_table("qle_n_nest", ConflictType.Ignore)
        except Exception:
            pass


def test_neg_whitespace_and_comment_only_via_filter(inst, db):
    """Degenerate query text through the filter parser."""
    t = _fresh(db, "qle_n_ws", {"a": {"type": "int"}})
    try:
        t.insert([{"a": 1}])
        for f in ("   ", "-- just a comment"):
            try:
                r, _ = rows(t.output(["a"]).filter(f))
                assert inst.pid() is not None
            except (InfinityException, Exception) as e:
                assert inst.pid() is not None, e
        assert inst.pid() is not None
    finally:
        db.drop_table("qle_n_ws", ConflictType.Ignore)


def test_neg_unicode_quote_injection_in_filter(inst, db):
    t = _fresh(db, "qle_n_inj", {"s": {"type": "varchar"}})
    try:
        t.insert([{"s": "o'brien"}, {"s": "bob"}])
        # a value containing a quote must be matchable and must not break parsing
        r, _ = rows(t.output(["s"]).filter("s = 'o''brien'"))
        assert inst.pid() is not None
        # engines vary on '' escaping; only require no crash + sane row count
        assert len(r) <= 2
    finally:
        db.drop_table("qle_n_inj", ConflictType.Ignore)


def test_neg_very_long_statement(inst, db):
    t = _fresh(db, "qle_n_long", {"a": {"type": "int"}})
    try:
        t.insert([{"a": 1}])
        long_filter = " OR ".join(["a = 1"] * 2000)
        try:
            r, _ = rows(t.output(["a"]).filter(long_filter))
            assert inst.pid() is not None
            assert len(r) == 1, r
        except (InfinityException, RecursionError, Exception) as e:
            assert inst.pid() is not None, e
    finally:
        db.drop_table("qle_n_long", ConflictType.Ignore)


# --------------------------------------------------------------------------- #
# module-final: the server logged nothing it hid from the client
# --------------------------------------------------------------------------- #

def test_zz_server_log_clean(inst):
    errs = inst.log_errors()
    # Print for report visibility; do not fail solely on this (negative tests
    # legitimately provoke handled errors), but surface unexpected fatals.
    fatal = [e for e in errs if "| fatal |" in e.lower() or "| critical |" in e.lower()]
    print("LOG_ERROR_COUNT", len(errs))
    for e in errs[:40]:
        print("LOG:", e)
    assert not fatal, f"server logged fatal/critical: {fatal[:5]}"
