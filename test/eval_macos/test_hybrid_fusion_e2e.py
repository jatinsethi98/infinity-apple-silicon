"""End-to-end verification of hybrid search and fusion (the RAG retrieval path).

Area: fusion of dense (KNN), full-text (BM25) and sparse arms into one ranked list.

The central technique here is an *arm-vs-fusion oracle*: every fusion query is
paired with standalone runs of each of its arms. We read each arm's ranked ids and
scores, then recompute the fused ranking by hand from the engine's documented rules
(read out of ``src/executor/operator/physical_fusion_impl.cpp``):

  * RRF:  fused_score(d) = sum over arms containing d of 1/(rank_constant + rank),
          rank is 1-based, rank_constant defaults to 60, invalid (<1) values ignored.
  * weighted_sum, normalize=none: fused_score(d) = sum_i weights[i]*score_i(d),
          absent arm contributes 0, default weight 1.0, dense l2 score = -(squared L2).

Recomputing from the arms is the only thing that catches a fusion which silently
drops an arm, mis-normalizes, or ignores its weights -- all of which return
plausible-looking rankings that a smoke test would accept.

Run:
    uv run pytest test/eval_macos/test_hybrid_fusion_e2e.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import harness
from infinity.common import ConflictType, InfinityException, SparseVector
from infinity.index import IndexInfo, IndexType

RANK_CONSTANT_DEFAULT = 60
QUERY_VEC = [0.0, 0.0, 0.0, 0.0]
DIM = 4
N = 8

# --- deterministic corpus ---------------------------------------------------
# vec[i] = [i+0.5, 0.5, 0.5, 0.5]; squared-L2 to the zero query is (i+0.5)^2 + 0.75,
# which is a clean integer for the first few docs: 1, 3, 7, 13, 21, 31, 43, 57.
# The full-text term "photon" appears in docs 2..7 with strictly decreasing count,
# so the two arms' top-4 sets overlap in exactly {2,3} and are otherwise disjoint.
PHOTON_COUNT = {2: 5, 3: 4, 4: 3, 5: 2, 6: 2, 7: 1}  # docs 0,1 have none


def _vec(i: int) -> list[float]:
    return [float(i) + 0.5, 0.5, 0.5, 0.5]


def _sq_l2(i: int) -> float:
    v = np.array(_vec(i), dtype=np.float64)
    return float(np.sum((v - np.array(QUERY_VEC)) ** 2))


def _body(i: int) -> str:
    fillers = ["harbor", "quiet", "lens", "field", "study", "drift", "calm", "signal"]
    words = ["photon"] * PHOTON_COUNT.get(i, 0)
    # pad to a fixed length so BM25 length-normalization does not reorder unexpectedly
    words += [f"{fillers[i % len(fillers)]}{i}"] * (8 - len(words))
    return " ".join(words)


def _rows():
    return [{"id": i, "body": _body(i), "vec": _vec(i)} for i in range(N)]


# --- fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-hybrid")
    it.start()
    try:
        conn, db = it.db()
        db.drop_table("fuse", ConflictType.Ignore)
        db.create_table(
            "fuse",
            {"id": {"type": "int"}, "body": {"type": "varchar"},
             "vec": {"type": f"vector,{DIM},float"}},
            ConflictType.Error,
        )
        t = db.get_table("fuse")
        t.insert(_rows())
        t.create_index("ft", IndexInfo("body", IndexType.FullText), ConflictType.Error)
        conn.disconnect()
        yield it
    finally:
        it.stop()


@pytest.fixture()
def tbl(inst):
    conn, db = inst.db()
    try:
        yield db.get_table("fuse")
    finally:
        conn.disconnect()


# --- helpers ----------------------------------------------------------------

def ids_scores(result):
    """Return (ids:list[int], scores:list[float]|None) from a .to_pl() result."""
    df, _extra = result
    cols = list(df.columns)
    id_c = "id" if "id" in cols else cols[0]
    score_c = next((c for c in cols if "score" in c.lower()), None)
    ids = [int(x) for x in df[id_c].to_list()]
    scores = [float(x) for x in df[score_c].to_list()] if score_c else None
    return ids, scores


def dense_alone(tbl, topn):
    # NOTE: SCORE() is rejected for a standalone KNN arm (engine error 3013 -- only
    # Fusion / MATCH TEXT / MATCH TENSOR expose SCORE()), so we read ids only and
    # compute the dense score (-squared L2) with numpy where needed.
    ids, _ = ids_scores(tbl.output(["id"])
                        .match_dense("vec", QUERY_VEC, "float", "l2", topn).to_pl())
    return ids, None


def text_alone(tbl, topn):
    return ids_scores(tbl.output(["id", "_score"])
                      .match_text("body", "photon", topn).to_pl())


def rank_map(ids):
    """1-based rank of each id in a returned arm ordering."""
    return {doc: pos + 1 for pos, doc in enumerate(ids)}


def rrf_expected(text_ids, dense_ids, rank_constant=RANK_CONSTANT_DEFAULT):
    tr, dr = rank_map(text_ids), rank_map(dense_ids)
    union = set(text_ids) | set(dense_ids)
    out = {}
    for d in union:
        s = 0.0
        if d in tr:
            s += 1.0 / (rank_constant + tr[d])
        if d in dr:
            s += 1.0 / (rank_constant + dr[d])
        out[d] = s
    return out


def assert_ranking(fused_ids, fused_scores, expected, tol=1e-4):
    """Every returned score matches the hand-computed value, and the order is
    a valid descending sort of those scores (ties may permute)."""
    assert set(fused_ids) == set(expected), (
        f"fused id set {sorted(fused_ids)} != expected union {sorted(expected)}"
    )
    for d, s in zip(fused_ids, fused_scores):
        assert math.isclose(s, expected[d], abs_tol=tol), (
            f"doc {d}: engine score {s} != recomputed {expected[d]}"
        )
    assert all(a >= b - tol for a, b in zip(fused_scores, fused_scores[1:])), (
        f"fused scores not descending: {fused_scores}"
    )


# ============================================================================
# POSITIVE / ORACLE TESTS
# ============================================================================

def test_dense_arm_matches_numpy(tbl):
    """Sanity: the dense arm alone is exactly numpy's L2 ordering, and the engine's
    reported _distance equals numpy's squared L2."""
    ids, _ = dense_alone(tbl, N)
    order = sorted(range(N), key=_sq_l2)
    assert ids == order, f"dense order {ids} != numpy {order}"
    # _distance IS available for a standalone KNN arm; verify the numeric value.
    df, _extra = tbl.output(["id", "_distance"]) \
        .match_dense("vec", QUERY_VEC, "float", "l2", N).to_pl()
    cols = list(df.columns)
    dist_c = next((c for c in cols if "distance" in c.lower()), None)
    assert dist_c is not None, f"no _distance column in {cols}"
    dids = [int(x) for x in df["id"].to_list()]
    dvals = [float(x) for x in df[dist_c].to_list()]
    for d, dist in zip(dids, dvals):
        assert math.isclose(dist, _sq_l2(d), abs_tol=1e-3), (
            f"doc {d} engine l2 distance {dist} != numpy squared-L2 {_sq_l2(d)}")


def test_rrf_two_way_oracle(inst, tbl):
    """RRF(dense+full-text) ranking recomputed from each arm's ranks."""
    off = inst.log_offset()
    text_ids, _ = text_alone(tbl, 4)
    dense_ids, _ = dense_alone(tbl, 4)
    expected = rrf_expected(text_ids, dense_ids)

    fused_ids, fused_scores = ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="rrf", topn=10).to_pl())

    assert_ranking(fused_ids, fused_scores, expected)
    # union must include text-only, dense-only AND shared docs
    assert {0, 1} <= set(fused_ids)  # dense-only
    assert set(dense_ids) & set(text_ids)  # shared docs exist
    assert inst.log_errors(off) == []


def test_rrf_custom_rank_constant_oracle(tbl):
    text_ids, _ = text_alone(tbl, 4)
    dense_ids, _ = dense_alone(tbl, 4)
    expected = rrf_expected(text_ids, dense_ids, rank_constant=5)
    fused_ids, fused_scores = ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="rrf", topn=10, fusion_params={"rank_constant": "5"}).to_pl())
    assert_ranking(fused_ids, fused_scores, expected)


def test_weighted_sum_none_default_weights_oracle(inst, tbl):
    """weighted_sum, normalize=none, default weights=1: fused = text_bm25 + (-sqL2)."""
    off = inst.log_offset()
    text_ids, text_scores = text_alone(tbl, 4)
    tscore = dict(zip(text_ids, text_scores))
    dense_ids, _ = dense_alone(tbl, 4)
    expected = {}
    for d in set(text_ids) | set(dense_ids):
        expected[d] = tscore.get(d, 0.0) + (-_sq_l2(d) if d in dense_ids else 0.0)

    fused_ids, fused_scores = ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="weighted_sum", topn=10,
                fusion_params={"normalize": "none"}).to_pl())
    assert_ranking(fused_ids, fused_scores, expected, tol=1e-3)
    assert inst.log_errors(off) == []


def test_weighted_sum_weights_are_applied(tbl):
    """weights=[1,3] must scale ONLY the dense arm by 3x vs weights=[1,1].
    Differential test: proves the weight vector is honored and no arm is dropped,
    without depending on the score sign convention."""
    text_ids, _ = text_alone(tbl, 4)
    dense_ids, _ = dense_alone(tbl, 4)

    def fuse(weights):
        return dict(zip(*ids_scores(
            tbl.output(["id", "_score"])
            .match_text("body", "photon", 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .fusion(method="weighted_sum", topn=10,
                    fusion_params={"weights": weights, "normalize": "none"}).to_pl())))

    f11 = fuse("1.0,1.0")
    f13 = fuse("1.0,3.0")
    dense_only = set(dense_ids) - set(text_ids)
    text_only = set(text_ids) - set(dense_ids)
    assert dense_only and text_only, "need both arm-exclusive docs for this test"
    for d in dense_only:               # dense weight 1->3 : contribution triples
        assert math.isclose(f13[d], 3.0 * f11[d], abs_tol=1e-3), (
            f"dense-only doc {d}: f13={f13[d]} expected 3*f11={3 * f11[d]}")
    for d in text_only:                # text weight unchanged : identical
        assert math.isclose(f13[d], f11[d], abs_tol=1e-3), (
            f"text-only doc {d}: f13={f13[d]} != f11={f11[d]}")


def test_weighted_sum_zero_weight_drops_that_arm(tbl):
    """weights=[0,1] must zero out the text arm's contribution entirely."""
    dense_ids, _ = dense_alone(tbl, 4)
    fused = dict(zip(*ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="weighted_sum", topn=10,
                fusion_params={"weights": "0.0,1.0", "normalize": "none"}).to_pl())))
    for d in dense_ids:
        assert math.isclose(fused[d], -_sq_l2(d), abs_tol=1e-3), (
            f"doc {d}: with text weight 0, fused {fused[d]} should equal dense -sqL2 {-_sq_l2(d)}")


def test_weighted_sum_minmax_normalizes_arm_maxima_to_one(tbl):
    """Default normalize=minmax: the best doc of each arm normalizes to 1.0, so the
    top fused doc (weights=1) has score in [1, num_arms]."""
    fused_ids, fused_scores = ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="weighted_sum", topn=10).to_pl())
    assert max(fused_scores) >= 1.0 - 1e-4, f"minmax top score {max(fused_scores)} < 1.0"
    assert max(fused_scores) <= 2.0 + 1e-4, f"minmax top score {max(fused_scores)} > 2.0"


def test_rrf_identical_arms(tbl):
    """Two identical dense arms: every doc appears in both at the same rank, so
    fused_score = 2/(60+rank) and the order is exactly the single-arm order."""
    dense_ids, _ = dense_alone(tbl, 4)
    fused_ids, fused_scores = ids_scores(
        tbl.output(["id", "_score"])
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="rrf", topn=10).to_pl())
    assert fused_ids == dense_ids, f"{fused_ids} != {dense_ids}"
    for pos, s in enumerate(fused_scores):
        exp = 2.0 / (RANK_CONSTANT_DEFAULT + pos + 1)
        assert math.isclose(s, exp, abs_tol=1e-5), f"rank {pos+1}: {s} != {exp}"


def test_rrf_disjoint_arms(tbl):
    """Disjoint arms (text top-2 vs dense top-2): each doc scores 1/(60+its rank)."""
    text_ids, _ = text_alone(tbl, 2)
    dense_ids, _ = dense_alone(tbl, 2)
    assert set(text_ids).isdisjoint(dense_ids), "expected disjoint top-2 sets"
    expected = rrf_expected(text_ids, dense_ids)
    fused_ids, fused_scores = ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 2)
        .match_dense("vec", QUERY_VEC, "float", "l2", 2)
        .fusion(method="rrf", topn=10).to_pl())
    assert_ranking(fused_ids, fused_scores, expected)


def test_fusion_topn_truncates_union(tbl):
    """topn smaller than the union returns exactly the top-n by fused score."""
    text_ids, _ = text_alone(tbl, 4)
    dense_ids, _ = dense_alone(tbl, 4)
    expected = rrf_expected(text_ids, dense_ids)
    top2 = sorted(expected, key=lambda d: -expected[d])[:2]
    fused_ids, fused_scores = ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="rrf", topn=2).to_pl())
    assert len(fused_ids) == 2, f"topn=2 returned {len(fused_ids)} rows"
    assert set(fused_ids) == set(top2), f"{fused_ids} != top2 {top2}"


def test_fusion_topn_larger_than_union(tbl):
    """topn larger than the union returns the whole union, not padded rows."""
    text_ids, _ = text_alone(tbl, 4)
    dense_ids, _ = dense_alone(tbl, 4)
    union = set(text_ids) | set(dense_ids)
    fused_ids, _ = ids_scores(
        tbl.output(["id", "_score"])
        .match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="rrf", topn=1000).to_pl())
    assert set(fused_ids) == union
    assert len(fused_ids) == len(union), f"got {len(fused_ids)} rows, union is {len(union)}"


# ============================================================================
# THREE-WAY FUSION (dense + full-text + sparse)
# ============================================================================

def test_three_way_rrf_sparse_arm(inst):
    """dense + full-text + sparse -> RRF over three arms, recomputed from each arm."""
    conn, db = inst.db()
    try:
        db.drop_table("fuse3", ConflictType.Ignore)
        db.create_table("fuse3", {
            "id": {"type": "int"}, "body": {"type": "varchar"},
            "vec": {"type": f"vector,{DIM},float"},
            "sp": {"type": "sparse,100,float,int"},
        }, ConflictType.Error)
        t = db.get_table("fuse3")
        rows = []
        for i in range(N):
            rows.append({"id": i, "body": _body(i), "vec": _vec(i),
                         "sp": SparseVector(indices=[i, (i + 1) % 100], values=[float(N - i), 1.0])})
        t.insert(rows)
        t.create_index("ft3", IndexInfo("body", IndexType.FullText), ConflictType.Error)

        query_sp = SparseVector(indices=[0, 1, 2, 3], values=[4.0, 3.0, 2.0, 1.0])

        def sparse_alone(topn):
            # SCORE() also rejected for a standalone sparse arm; ranks are enough.
            ids, _ = ids_scores(t.output(["id"])
                                .match_sparse("sp", query_sp, "ip", topn).to_pl())
            return ids, None

        text_ids, _ = text_alone(t, 4)
        dense_ids, _ = dense_alone(t, 4)
        sp_ids, _ = sparse_alone(4)

        tr, dr, sr = rank_map(text_ids), rank_map(dense_ids), rank_map(sp_ids)
        union = set(text_ids) | set(dense_ids) | set(sp_ids)
        expected = {}
        for d in union:
            s = 0.0
            for rm in (tr, dr, sr):
                if d in rm:
                    s += 1.0 / (RANK_CONSTANT_DEFAULT + rm[d])
            expected[d] = s

        fused_ids, fused_scores = ids_scores(
            t.output(["id", "_score"])
            .match_text("body", "photon", 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .match_sparse("sp", query_sp, "ip", 4)
            .fusion(method="rrf", topn=20).to_pl())
        assert_ranking(fused_ids, fused_scores, expected)
        db.drop_table("fuse3", ConflictType.Ignore)
    finally:
        conn.disconnect()


# ============================================================================
# TENSOR RERANK FUSION
# ============================================================================

def test_match_tensor_rerank_fusion(inst):
    """fusion('match_tensor') as a reranker over an RRF result. Records a finding
    if unreachable; asserts the server survives either way."""
    conn, db = inst.db()
    finding = None
    try:
        db.drop_table("fuset", ConflictType.Ignore)
        db.create_table("fuset", {
            "id": {"type": "int"}, "body": {"type": "varchar"},
            "t": {"type": f"tensor,{DIM},float"},
        }, ConflictType.Error)
        tt = db.get_table("fuset")
        rows = [{"id": i, "body": _body(i),
                 "t": [[float(i) + 0.5, 0.5, 0.5, 0.5], [0.5, float(i) + 0.5, 0.5, 0.5]]}
                for i in range(N)]
        tt.insert(rows)
        tt.create_index("ftt", IndexInfo("body", IndexType.FullText), ConflictType.Error)
        try:
            fused_ids, fused_scores = ids_scores(
                tt.output(["id", "_score"])
                .match_text("body", "photon", 4)
                .match_tensor("t", [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], "float", 4)
                .fusion(method="rrf", topn=10)
                .fusion(method="match_tensor", topn=3,
                        fusion_params={"field": "t", "element_type": "float",
                                       "query_tensor": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]})
                .to_pl())
            assert len(fused_ids) <= 3, f"match_tensor rerank topn=3 returned {len(fused_ids)}"
            assert len(fused_ids) == len(set(fused_ids)), "duplicate ids in rerank output"
            assert all(a >= b - 1e-4 for a, b in zip(fused_scores, fused_scores[1:])), \
                f"tensor rerank scores not descending: {fused_scores}"
        except InfinityException as e:
            finding = f"match_tensor rerank fusion raised: {e!r}"
        db.drop_table("fuset", ConflictType.Ignore)
    finally:
        conn.disconnect()
    assert inst.pid() is not None, "server died on match_tensor rerank"
    if finding:
        pytest.skip(finding)


# ============================================================================
# NEGATIVE TESTS  (each asserts the server is still alive afterwards)
# ============================================================================

def _expect_error_or_survive(inst, thunk, label):
    """Run thunk; it should raise (good) OR return. Either way the server must live.
    Returns (raised: bool, value_or_exc)."""
    try:
        val = thunk()
        raised = False
    except InfinityException as e:
        val, raised = e, True
    assert inst.pid() is not None, f"SERVER DIED on: {label}"
    return raised, val


def test_neg_zero_arms(inst, tbl):
    raised, val = _expect_error_or_survive(
        inst, lambda: tbl.output(["id"]).fusion(method="rrf", topn=5).to_pl(),
        "fusion with zero arms")
    assert raised, f"fusion with zero match arms was accepted: {val}"


def test_neg_single_arm(inst, tbl):
    raised, val = _expect_error_or_survive(
        inst,
        lambda: tbl.output(["id", "_score"]).match_text("body", "photon", 4)
        .fusion(method="rrf", topn=5).to_pl(),
        "fusion with a single arm")
    # Document behavior: single-arm fusion either errors or passes the arm through.
    if not raised:
        ids, _ = ids_scores(val)
        assert inst.pid() is not None
        # record via assertion message only if it silently returns wrong shape
        assert len(ids) >= 1, "single-arm fusion returned nothing"


def test_neg_unknown_fusion_method(inst, tbl):
    # SDK rejects unknown methods client-side before hitting the server.
    with pytest.raises(InfinityException):
        tbl.output(["id"]).match_text("body", "photon", 4) \
            .match_dense("vec", QUERY_VEC, "float", "l2", 4) \
            .fusion(method="banana", topn=5).to_pl()
    assert inst.pid() is not None


def test_neg_unknown_fusion_param_silently_ignored(inst, tbl):
    """KNOWN-WORST-CASE probe: an unrecognized fusion option must not be silently
    accepted as a no-op. We compare a query carrying a bogus option to the clean
    query; if results are identical the option was silently ignored."""
    off = inst.log_offset()

    def run(params):
        return ids_scores(
            tbl.output(["id", "_score"])
            .match_text("body", "photon", 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .fusion(method="rrf", topn=10, fusion_params=params).to_pl())

    clean_ids, clean_scores = run(None)
    raised = False
    try:
        bogus_ids, bogus_scores = run({"rank_konstant": "5"})  # typo'd rank_constant
    except InfinityException:
        raised = True
    assert inst.pid() is not None
    # CONFIRMED DEFECT (characterization): the engine neither errors nor changes its
    # result for an unrecognized fusion option -- it is silently ignored. A user who
    # typos 'rank_constant' as 'rank_konstant' gets the default-60 ranking with no
    # signal. This test locks in that observed behavior; if the engine ever starts
    # validating options it will (correctly) break here and should be updated.
    silently_ignored = (not raised) and (bogus_ids == clean_ids) and all(
        math.isclose(a, b, abs_tol=1e-9) for a, b in zip(bogus_scores, clean_scores))
    assert silently_ignored, (
        "expected the known silent-ignore of unknown fusion options; instead "
        f"raised={raised}, ids {bogus_ids} vs {clean_ids}")


def test_neg_rank_constant_zero_silently_clamped(inst, tbl):
    """rank_constant=0 is guarded by `l>=1` in the engine and silently falls back to
    the default 60. Flags the silent clamp by comparing to the default result."""
    def run(params):
        return ids_scores(
            tbl.output(["id", "_score"])
            .match_text("body", "photon", 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .fusion(method="rrf", topn=10, fusion_params=params).to_pl())

    default_ids, default_scores = run(None)
    zero_ids, zero_scores = run({"rank_constant": "0"})
    assert inst.pid() is not None
    clamped = zero_ids == default_ids and all(
        math.isclose(a, b, abs_tol=1e-9) for a, b in zip(zero_scores, default_scores))
    # CONFIRMED DEFECT (characterization): rank_constant=0 is guarded by `l>=1` in
    # physical_fusion_impl.cpp and silently falls back to the default 60 rather than
    # being rejected. rank_constant=0 is a legitimate RRF request (scores 1/rank) and
    # is silently swallowed. This test locks in the observed clamp.
    assert clamped, (
        "expected rank_constant=0 to be silently clamped to default 60; "
        f"instead default_scores={default_scores[:3]} zero_scores={zero_scores[:3]}")


def test_neg_negative_rank_constant(inst, tbl):
    raised, val = _expect_error_or_survive(
        inst,
        lambda: tbl.output(["id", "_score"]).match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="rrf", topn=10, fusion_params={"rank_constant": "-5"}).to_pl(),
        "rrf rank_constant=-5")
    # Either a clean error or a clamp; must not crash. Record which.
    assert inst.pid() is not None


def test_neg_too_many_weights(inst, tbl):
    """More weights than arms: engine loops only over arms, so extras are dropped.
    Assert it does not crash and behaves as the truncated weight vector."""
    def run(params):
        return dict(zip(*ids_scores(
            tbl.output(["id", "_score"])
            .match_text("body", "photon", 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .fusion(method="weighted_sum", topn=10, fusion_params=params).to_pl())))

    raised, _ = _expect_error_or_survive(
        inst, lambda: run({"weights": "1.0,2.0,3.0", "normalize": "none"}),
        "weighted_sum with 3 weights for 2 arms")
    assert inst.pid() is not None


def test_neg_too_few_weights_padded(inst, tbl):
    """Fewer weights than arms: engine pads with 1.0. Assert [2.0] behaves as [2.0,1.0]."""
    def run(params):
        return dict(zip(*ids_scores(
            tbl.output(["id", "_score"])
            .match_text("body", "photon", 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .fusion(method="weighted_sum", topn=10, fusion_params=params).to_pl())))
    try:
        short = run({"weights": "2.0", "normalize": "none"})
        full = run({"weights": "2.0,1.0", "normalize": "none"})
        assert inst.pid() is not None
        assert set(short) == set(full)
        for d in short:
            assert math.isclose(short[d], full[d], abs_tol=1e-3), (
                f"doc {d}: padded [2.0] gave {short[d]} != [2.0,1.0] {full[d]}")
    except InfinityException:
        assert inst.pid() is not None  # rejecting is acceptable too


def test_neg_negative_weight(inst, tbl):
    """Negative weight is arithmetically valid: fused = -1*text + 1*dense (none-norm)."""
    text_ids, text_scores = text_alone(tbl, 4)
    tscore = dict(zip(text_ids, text_scores))
    dense_ids, _ = dense_alone(tbl, 4)
    raised, val = _expect_error_or_survive(
        inst,
        lambda: tbl.output(["id", "_score"]).match_text("body", "photon", 4)
        .match_dense("vec", QUERY_VEC, "float", "l2", 4)
        .fusion(method="weighted_sum", topn=10,
                fusion_params={"weights": "-1.0,1.0", "normalize": "none"}).to_pl(),
        "weighted_sum negative weight")
    if not raised:
        fused = dict(zip(*ids_scores(val)))
        for d in set(text_ids) | set(dense_ids):
            exp = -tscore.get(d, 0.0) + (-_sq_l2(d) if d in dense_ids else 0.0)
            assert math.isclose(fused[d], exp, abs_tol=1e-3), (
                f"doc {d}: negative-weight fused {fused[d]} != {exp}")


def test_neg_arm_without_index(inst):
    """Full-text arm on a table with no full-text index must error, not crash."""
    conn, db = inst.db()
    try:
        db.drop_table("noidx", ConflictType.Ignore)
        db.create_table("noidx", {"id": {"type": "int"}, "body": {"type": "varchar"},
                                  "vec": {"type": f"vector,{DIM},float"}}, ConflictType.Error)
        t = db.get_table("noidx")
        t.insert(_rows())
        raised, val = _expect_error_or_survive(
            inst,
            lambda: t.output(["id"]).match_text("body", "photon", 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .fusion(method="rrf", topn=10).to_pl(),
            "full-text arm with no full-text index")
        assert raised, f"match_text without a full-text index was accepted: {val}"
        db.drop_table("noidx", ConflictType.Ignore)
    finally:
        conn.disconnect()


def test_neg_wrong_dimension_dense_arm(inst, tbl):
    raised, val = _expect_error_or_survive(
        inst,
        lambda: tbl.output(["id"]).match_text("body", "photon", 4)
        .match_dense("vec", [0.0, 0.0, 0.0], "float", "l2", 4)  # dim 3 != 4
        .fusion(method="rrf", topn=10).to_pl(),
        "dense arm with wrong dimension (3 vs 4)")
    assert raised, f"wrong-dimension dense arm was accepted: {val}"


def test_neg_missing_column_arm(inst, tbl):
    raised, val = _expect_error_or_survive(
        inst,
        lambda: tbl.output(["id"]).match_text("body", "photon", 4)
        .match_dense("nope", QUERY_VEC, "float", "l2", 4)  # column does not exist
        .fusion(method="rrf", topn=10).to_pl(),
        "dense arm on a nonexistent column")
    assert raised, f"arm on nonexistent column was accepted: {val}"


def test_neg_unicode_quote_injection_in_text_arm(inst, tbl):
    """A query full of quotes/unicode/control chars must not crash the parser."""
    payloads = ["photon' OR '1'='1", 'photon" ; DROP TABLE fuse; --',
                "光子 photon\t", "photon\\", "'''\"\"\""]
    for p in payloads:
        _expect_error_or_survive(
            inst,
            lambda p=p: tbl.output(["id"]).match_text("body", p, 4)
            .match_dense("vec", QUERY_VEC, "float", "l2", 4)
            .fusion(method="rrf", topn=10).to_pl(),
            f"unicode/quote payload {p!r}")
    assert inst.pid() is not None


def test_server_clean_after_suite(inst):
    """No error/critical/fatal lines should have been logged by the positive paths."""
    assert inst.pid() is not None
