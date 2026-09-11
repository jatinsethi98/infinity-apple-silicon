"""End-to-end full-text search evaluation for the macOS arm64 port.

Covers: FullText index build, ``match_text`` over one and several fields with field
boosts, the query operators the engine documents (test/sql/dql/fulltext/*.slt), the
options ``bm25_param_k1`` / ``bm25_param_b`` / ``threshold`` / ``topn`` / ``operator`` /
``similarity``, highlighting, and the dictionary-backed ``chinese`` (jieba) analyzer.

The BM25 kernel is pinned from src/storage/invertedindex/search/bm25_ranker_impl.cpp:

    smooth_idf = log(1 + (N - df + 0.5) / (df + 0.5))
    smooth_tf  = (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / avgdl))
    score     += smooth_idf * smooth_tf * weight

The option-parsing worst case (docs/apple_silicon/MACOS_VERIFICATION.md: unrecognised
search options are SILENTLY IGNORED) is attacked directly: every option name is taken
verbatim from src/planner/bound_select_statement_impl.cpp, each is proven to move the
score in the BM25-required direction, and misspelled / wrong-named options are checked
to see whether they raise or vanish.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

from infinity.common import ConflictType  # noqa: E402
from infinity.index import IndexInfo, IndexType  # noqa: E402


# --------------------------------------------------------------------------- #
# BM25 reference (exact kernel from bm25_ranker_impl.cpp)
# --------------------------------------------------------------------------- #

def _idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25(n_docs: int, df: int, tf: int, dl: int, avgdl: float,
         k1: float = 1.2, b: float = 0.75, weight: float = 1.0) -> float:
    smooth_tf = (k1 + 1.0) * tf / (tf + k1 * (1.0 - b + b * dl / avgdl))
    return _idf(n_docs, df) * smooth_tf * weight


# The pinned corpus: 4 docs, column `body`, standard analyzer (whitespace tokens).
#   id0 "kiwi"                  kiwi tf=1  len 1
#   id1 "kiwi kiwi"             kiwi tf=2  len 2
#   id2 "kiwi alpha beta gamma" kiwi tf=1  len 4
#   id3 "mango"                 mango tf=1 len 1
CORPUS = [
    (0, "kiwi"),
    (1, "kiwi kiwi"),
    (2, "kiwi alpha beta gamma"),
    (3, "mango"),
]
N = 4
DF_KIWI = 3
DF_MANGO = 1
AVGDL = (1 + 2 + 4 + 1) / 4.0  # = 2.0


# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    ins = harness.instance("eval-fulltext")
    ins.start()
    try:
        yield ins
    finally:
        ins.stop()


def _fresh_table(db, name, rows=CORPUS, analyzer=None, index=True,
                 extra_varchar=False):
    db.drop_table(name, ConflictType.Ignore)
    schema = {"id": {"type": "int"}, "body": {"type": "varchar"}}
    if extra_varchar:
        schema["body2"] = {"type": "varchar"}
    tbl = db.create_table(name, schema, ConflictType.Error)
    payload = []
    for r in rows:
        rec = {"id": r[0], "body": r[1]}
        if extra_varchar:
            rec["body2"] = r[1]
        payload.append(rec)
    tbl.insert(payload)
    if index:
        params = {"analyzer": analyzer} if analyzer else None
        res = tbl.create_index("ft_body",
                               IndexInfo("body", IndexType.FullText, params),
                               ConflictType.Error)
        assert res.error_code == 0, res
    return tbl


def _score_col(df):
    for c in df.columns:
        cl = str(c).lower()
        if "score" in cl:
            return c
    raise AssertionError(f"no score column in {list(df.columns)}")


def _id_col(df):
    for c in df.columns:
        if str(c).lower() == "id":
            return c
    raise AssertionError(f"no id column in {list(df.columns)}")


def search(tbl, fields, text, topn, extra=None, out=("id", "body", "_score"),
           highlight=None):
    """Run a match_text query and return {id: score} plus the raw df."""
    qb = tbl.output(list(out))
    if highlight:
        qb = qb.highlight(list(highlight))
    qb = qb.match_text(fields, text, topn, extra)
    df, _extra = qb.to_df()
    sc = _score_col(df)
    idc = _id_col(df)
    scores = {int(i): float(s) for i, s in zip(df[idc].tolist(), df[sc].tolist())}
    return scores, df


# --------------------------------------------------------------------------- #
# 1. BM25 exact-value proofs (the only proof a tuning option was applied)
# --------------------------------------------------------------------------- #

def test_bm25_default_absolute_scores(inst):
    """Default (k1=1.2, b=0.75) scores match the pinned kernel exactly."""
    _conn, db = inst.db()
    off = inst.log_offset()
    tbl = _fresh_table(db, "ft_bm25_default")
    scores, df = search(tbl, "body", "kiwi", 10)
    exp0 = bm25(N, DF_KIWI, tf=1, dl=1, avgdl=AVGDL)
    exp1 = bm25(N, DF_KIWI, tf=2, dl=2, avgdl=AVGDL)
    exp2 = bm25(N, DF_KIWI, tf=1, dl=4, avgdl=AVGDL)
    assert scores.keys() == {0, 1, 2}, f"mango must not match; got {scores}"
    assert abs(scores[0] - exp0) < 5e-3, (scores[0], exp0)
    assert abs(scores[1] - exp1) < 5e-3, (scores[1], exp1)
    assert abs(scores[2] - exp2) < 5e-3, (scores[2], exp2)
    # tf-monotonic and length-penalised: id1(tf2) > id0(tf1,len1) > id2(tf1,len4)
    assert scores[1] > scores[0] > scores[2]
    assert inst.log_errors(off) == []
    _conn.disconnect()


def test_bm25_param_b_applied(inst):
    """b=0 must remove length normalisation: two tf=1 docs of different length
    become EQUAL, and each equals IDF exactly (smooth_tf==1). If bm25_param_b is
    silently ignored this fails because id2(len4) keeps its default penalty."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_bm25_b")
    scores, _ = search(tbl, "body", "kiwi", 10, extra={"bm25_param_b": 0.0})
    idf_kiwi = _idf(N, DF_KIWI)
    assert abs(scores[0] - idf_kiwi) < 5e-3, (scores[0], idf_kiwi)
    assert abs(scores[2] - idf_kiwi) < 5e-3, (scores[2], idf_kiwi)
    assert abs(scores[0] - scores[2]) < 2e-3, (
        f"b=0 did not disable length norm: id0={scores[0]} id2={scores[2]} "
        f"-> bm25_param_b may be silently ignored")
    _conn.disconnect()


def test_bm25_param_k1_applied(inst):
    """With b=0 the tf-saturation ratio score(tf=2)/score(tf=1) == 2(k1+1)/(2+k1),
    a pure function of k1 (IDF and length cancel). Assert the exact ratio for
    three k1 values; a stuck constexpr k1=1.2 would give 1.375 for all three."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_bm25_k1")
    for k1, want in [(0.5, 2 * 1.5 / 2.5), (1.2, 2 * 2.2 / 3.2), (3.0, 2 * 4.0 / 5.0)]:
        scores, _ = search(tbl, "body", "kiwi", 10,
                           extra={"bm25_param_b": 0.0, "bm25_param_k1": k1})
        ratio = scores[1] / scores[0]
        assert abs(ratio - want) < 1e-2, (
            f"k1={k1}: ratio {ratio:.4f} != expected {want:.4f} "
            f"-> bm25_param_k1 not applied as given")
    _conn.disconnect()


def test_rarer_term_scores_higher(inst):
    """A rarer term (mango df=1) contributes strictly more than a common one
    (kiwi df=3). With b=0, tf=1 the score equals IDF exactly."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_idf")
    kiwi, _ = search(tbl, "body", "kiwi", 10, extra={"bm25_param_b": 0.0})
    mango, _ = search(tbl, "body", "mango", 10, extra={"bm25_param_b": 0.0})
    assert abs(mango[3] - _idf(N, DF_MANGO)) < 5e-3
    assert abs(kiwi[0] - _idf(N, DF_KIWI)) < 5e-3
    assert mango[3] > kiwi[0], (mango[3], kiwi[0])
    _conn.disconnect()


def test_field_boost_multiplies_score(inst):
    """`body^5` must scale every BM25 contribution by the boost weight (5x)."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_boost")
    base, _ = search(tbl, "body", "kiwi", 10, extra={"bm25_param_b": 0.0})
    boosted, _ = search(tbl, "body^5", "kiwi", 10, extra={"bm25_param_b": 0.0})
    for i in base:
        assert abs(boosted[i] - 5.0 * base[i]) < 1e-2, (i, boosted[i], base[i])
    _conn.disconnect()


# --------------------------------------------------------------------------- #
# 2. Silent-ignore attack (the documented worst case)
# --------------------------------------------------------------------------- #

def test_misspelled_option_is_silently_ignored(inst):
    """A plausible-but-wrong option name changes nothing and raises nothing.
    This is the documented failure mode; quantify it."""
    _conn, db = inst.db()
    off = inst.log_offset()
    tbl = _fresh_table(db, "ft_misspell")
    baseline, _ = search(tbl, "body", "kiwi", 10)
    # bm25_param_kk1 (double-k typo) and bm25_paramb: neither key is ever looked up.
    typo, df = search(tbl, "body", "kiwi", 10,
                      extra={"bm25_param_kk1": 99.0, "bm25_paramb": 99.0})
    assert typo == baseline, (
        f"misspelled options altered the result?? baseline={baseline} typo={typo}")
    assert inst.log_errors(off) == [], "server logged nothing about the bad option"
    assert inst.pid() is not None
    _conn.disconnect()


def test_wrong_named_threshold_is_silently_ignored(inst):
    """`threshold` is the real score-cutoff option; `score_threshold` (the tensor
    option name) is NOT recognised for full text. Passing it must NOT filter —
    a naive test author would believe they had a cutoff and be exercising nothing."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_wrongthresh")
    # Real option filters (below): with default scoring id2 ~0.253 is dropped by 0.35.
    real, _ = search(tbl, "body", "kiwi", 10, extra={"threshold": 0.35})
    assert set(real.keys()) == {0, 1}, f"threshold=0.35 should drop id2: {real}"
    # Wrong name: no filtering at all -> all three kiwi docs returned.
    wrong, _ = search(tbl, "body", "kiwi", 10, extra={"score_threshold": 0.35})
    assert set(wrong.keys()) == {0, 1, 2}, (
        f"score_threshold appears to have filtered -> unexpected; got {wrong}")
    assert inst.pid() is not None
    _conn.disconnect()


def test_threshold_filters_by_score(inst):
    """`threshold` keeps only rows with score >= threshold, matching computed values."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_thresh")
    allrows, _ = search(tbl, "body", "kiwi", 10)
    cut = 0.35
    expected = {i for i, s in allrows.items() if s >= cut}
    got, _ = search(tbl, "body", "kiwi", 10, extra={"threshold": cut})
    assert set(got.keys()) == expected, (got, allrows, expected)
    _conn.disconnect()


def test_topn_limits_rows(inst):
    """`topn` bounds the returned row count to the top-scoring N."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_topn")
    got, _ = search(tbl, "body", "kiwi", 1)
    assert len(got) == 1, got
    # highest scorer is id1 (tf=2)
    assert set(got.keys()) == {1}, got
    _conn.disconnect()


# --------------------------------------------------------------------------- #
# 3. Operators / query syntax (from test/sql/dql/fulltext/*.slt)
# --------------------------------------------------------------------------- #

def test_operator_and_vs_or(inst):
    """operator=and requires all terms; operator=or (default-ish) requires any."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_op")
    or_rows, _ = search(tbl, "body", "kiwi mango", 10, extra={"operator": "or"})
    and_rows, _ = search(tbl, "body", "kiwi mango", 10, extra={"operator": "and"})
    # OR matches any doc containing kiwi OR mango => all 4 docs.
    assert set(or_rows.keys()) == {0, 1, 2, 3}, or_rows
    # AND requires both kiwi AND mango in one doc => none of our docs have both.
    assert and_rows == {}, and_rows
    _conn.disconnect()


def test_phrase_query(inst):
    """A quoted phrase matches only adjacent tokens in order."""
    _conn, db = inst.db()
    rows = [(0, "alpha beta gamma"), (1, "beta alpha gamma"), (2, "alpha gamma beta")]
    tbl = _fresh_table(db, "ft_phrase", rows=rows)
    got, _ = search(tbl, "body", '"alpha beta"', 10)
    assert set(got.keys()) == {0}, f'phrase "alpha beta" should match only id0: {got}'
    _conn.disconnect()


def test_term_boost_in_query(inst):
    """Per-term boost `term^w` scales that term's contribution."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_termboost")
    base, _ = search(tbl, "body", "kiwi", 10, extra={"bm25_param_b": 0.0})
    boosted, _ = search(tbl, "body", "kiwi^3", 10, extra={"bm25_param_b": 0.0})
    for i in base:
        assert abs(boosted[i] - 3.0 * base[i]) < 1e-2, (i, boosted[i], base[i])
    _conn.disconnect()


def test_multi_field_search(inst):
    """match_text over two fields with independent boosts sums per-field BM25."""
    _conn, db = inst.db()
    db.drop_table("ft_multi", ConflictType.Ignore)
    tbl = db.create_table("ft_multi",
                          {"id": {"type": "int"}, "title": {"type": "varchar"},
                           "body": {"type": "varchar"}}, ConflictType.Error)
    tbl.insert([{"id": 0, "title": "kiwi", "body": "kiwi"},
                {"id": 1, "title": "grape", "body": "kiwi"}])
    tbl.create_index("ft_t", IndexInfo("title", IndexType.FullText), ConflictType.Error)
    tbl.create_index("ft_b", IndexInfo("body", IndexType.FullText), ConflictType.Error)
    both, _ = search(tbl, "title,body", "kiwi", 10, out=("id", "_score"))
    # id0 has kiwi in both fields, id1 only in body -> id0 scores higher.
    assert both[0] > both[1], both
    _conn.disconnect()


# --------------------------------------------------------------------------- #
# 4. Highlighting
# --------------------------------------------------------------------------- #

def test_highlight_wraps_matches(inst):
    """HIGHLIGHT wraps matched terms in <em> and leaves other text intact."""
    _conn, db = inst.db()
    rows = [(0, "the kiwi is a small brown bird")]
    tbl = _fresh_table(db, "ft_hl", rows=rows)
    qb = tbl.output(["id", "body"]).highlight(["body"])
    qb = qb.match_text("body", "kiwi", 10, None)
    df, _ = qb.to_df()
    text = str(df["body"].tolist()[0])
    assert "<em>kiwi</em>" in text, text
    assert "small brown bird" in text, text
    _conn.disconnect()


# --------------------------------------------------------------------------- #
# 5. Analyzer (dictionary-backed CJK)
# --------------------------------------------------------------------------- #

def test_chinese_analyzer_segmentation(inst):
    """The jieba-backed 'chinese' analyzer segments a space-free CJK run so that
    a dictionary term inside it (清华大学) matches. Reaching a hit proves the
    resource_dir dictionaries loaded on macOS."""
    _conn, db = inst.db()
    off = inst.log_offset()
    rows = [(0, "我来到北京清华大学"), (1, "小明硕士毕业于中国科学院计算所")]
    tbl = _fresh_table(db, "ft_cjk", rows=rows, analyzer="chinese")
    got, _ = search(tbl, "body", "清华大学", 10)
    assert 0 in got and 1 not in got, f"expected only id0 to match 清华大学: {got}"
    assert inst.log_errors(off) == []
    _conn.disconnect()


def test_mixed_cjk_latin_and_case_folding(inst):
    """Standard analyzer folds case for Latin; CJK adjacency still tokenizes."""
    _conn, db = inst.db()
    rows = [(0, "Kiwi FRUIT 水果")]
    tbl = _fresh_table(db, "ft_case", rows=rows)
    lower, _ = search(tbl, "body", "kiwi", 10)
    upper, _ = search(tbl, "body", "KIWI", 10)
    assert set(lower.keys()) == {0} and set(upper.keys()) == {0}, (lower, upper)
    _conn.disconnect()


# --------------------------------------------------------------------------- #
# 6. Delete / restart durability
# --------------------------------------------------------------------------- #

def test_search_after_delete(inst):
    """A deleted row must disappear from full-text results."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_delete")
    before, _ = search(tbl, "body", "kiwi", 10)
    assert 1 in before
    tbl.delete("id = 1")
    after, _ = search(tbl, "body", "kiwi", 10)
    assert 1 not in after, after
    assert set(after.keys()) == {0, 2}, after
    _conn.disconnect()


def test_search_after_restart(inst):
    """Full-text index survives a graceful restart (WAL/checkpoint recovery)."""
    _conn, db = inst.db()
    _fresh_table(db, "ft_restart")
    _conn.disconnect()
    inst.restart(graceful=True)
    _conn2, db2 = inst.db()
    tbl = db2.get_table("ft_restart")
    got, _ = search(tbl, "body", "kiwi", 10)
    assert set(got.keys()) == {0, 1, 2}, got
    assert inst.pid() is not None
    _conn2.disconnect()


# --------------------------------------------------------------------------- #
# 7. Empty / degenerate queries
# --------------------------------------------------------------------------- #

def test_empty_query_returns_no_rows(inst):
    _conn, db = inst.db()
    off = inst.log_offset()
    tbl = _fresh_table(db, "ft_empty")
    got, _ = search(tbl, "body", "", 10)
    assert got == {}, got
    assert inst.pid() is not None
    assert inst.log_errors(off) == []
    _conn.disconnect()


def test_whitespace_only_query_clean(inst):
    """Whitespace-only query: empty("") returns 0 rows, but "     " raises
    error 3052 instead. Accept either, require no crash; the inconsistency is
    recorded as a finding."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_ws")
    try:
        got, _ = search(tbl, "body", "     ", 10)
        assert got == {}, got
    except Exception as e:
        assert "3052" in str(e) or "match" in str(e).lower(), e
    assert inst.pid() is not None
    _conn.disconnect()


def test_punctuation_only_query_clean(inst):
    """Punctuation-only query must not crash; 0 rows or a clean error both ok."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_punct")
    try:
        got, _ = search(tbl, "body", ".", 10)
        assert got == {}, got
    except Exception:
        pass
    assert inst.pid() is not None
    _conn.disconnect()


# --------------------------------------------------------------------------- #
# 8. Negative tests — invalid input must ERROR, not crash, not silently pass
# --------------------------------------------------------------------------- #

def _expect_error(fn):
    with pytest.raises(Exception) as ei:  # noqa: PT011 - broad on purpose
        fn()
    return ei.value


def test_negative_topn_rejected(inst):
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_neg_topn")
    _expect_error(lambda: search(tbl, "body", "kiwi", 0))
    _expect_error(lambda: search(tbl, "body", "kiwi", -5))
    assert inst.pid() is not None
    _conn.disconnect()


def test_bm25_param_out_of_range_rejected(inst):
    """bm25_param_b in [0,1], k1 >= 0 per the parser; violations must error."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_neg_bm25")
    _expect_error(lambda: search(tbl, "body", "kiwi", 10, extra={"bm25_param_b": 5.0}))
    _expect_error(lambda: search(tbl, "body", "kiwi", 10, extra={"bm25_param_k1": -1.0}))
    assert inst.pid() is not None
    _conn.disconnect()


def test_unknown_operator_rejected(inst):
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_neg_op")
    _expect_error(lambda: search(tbl, "body", "kiwi", 10, extra={"operator": "xor"}))
    assert inst.pid() is not None
    _conn.disconnect()


def test_unknown_similarity_rejected(inst):
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_neg_sim")
    _expect_error(lambda: search(tbl, "body", "kiwi", 10, extra={"similarity": "cosine"}))
    assert inst.pid() is not None
    _conn.disconnect()


def test_nonexistent_field_rejected(inst):
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_neg_field")
    _expect_error(lambda: search(tbl, "nope", "kiwi", 10))
    assert inst.pid() is not None
    _conn.disconnect()


def test_non_indexed_field_rejected(inst):
    """A varchar column without a FullText index cannot be matched."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_neg_noidx", extra_varchar=True)  # body2 has no index
    _expect_error(lambda: search(tbl, "body2", "kiwi", 10))
    assert inst.pid() is not None
    _conn.disconnect()


def test_match_text_on_non_varchar_rejected(inst):
    """match_text against an int column must error, not crash."""
    _conn, db = inst.db()
    tbl = _fresh_table(db, "ft_neg_int")
    _expect_error(lambda: search(tbl, "id", "1", 10))
    assert inst.pid() is not None
    _conn.disconnect()


def test_unbalanced_quote_and_paren(inst):
    """Malformed query syntax must be rejected or handled without crashing."""
    _conn, db = inst.db()
    off = inst.log_offset()
    tbl = _fresh_table(db, "ft_neg_quote")
    for q in ['"kiwi', '(kiwi', 'kiwi)']:
        try:
            search(tbl, "body", q, 10)
        except Exception:
            pass  # raising is acceptable
    assert inst.pid() is not None, "server died on malformed query syntax"
    _conn.disconnect()


def test_unknown_analyzer_rejected_at_build(inst):
    """Building a FullText index with a bogus analyzer must be rejected, not
    silently defaulted (fulltext.slt rejects analyzer=jieba/ngram this way)."""
    _conn, db = inst.db()
    db.drop_table("ft_neg_analyzer", ConflictType.Ignore)
    tbl = db.create_table("ft_neg_analyzer",
                          {"id": {"type": "int"}, "body": {"type": "varchar"}},
                          ConflictType.Error)
    tbl.insert([{"id": 0, "body": "kiwi"}])
    raised = False
    err_code = None
    try:
        res = tbl.create_index("bad",
                               IndexInfo("body", IndexType.FullText,
                                         {"analyzer": "no_such_analyzer"}),
                               ConflictType.Error)
        err_code = res.error_code
    except Exception:
        raised = True
    assert raised or err_code not in (0, None), (
        f"bogus analyzer accepted silently (error_code={err_code})")
    assert inst.pid() is not None
    _conn.disconnect()


def test_query_metacharacters_escaped(inst):
    """A term that is only the query language's own metacharacters must not crash
    the server; it should match nothing or error cleanly."""
    _conn, db = inst.db()
    off = inst.log_offset()
    tbl = _fresh_table(db, "ft_meta")
    for q in ["^", ":", "*", "\\", "kiwi^^"]:
        try:
            search(tbl, "body", q, 10)
        except Exception:
            pass
    assert inst.pid() is not None, "server crashed on metacharacter-only query"
    _conn.disconnect()
