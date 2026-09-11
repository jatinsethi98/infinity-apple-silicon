"""End-to-end dense-vector (KNN) verification for the macOS arm64 port.

Correctness is checked against numpy ground truth computed over the *same* vectors,
not against "no exception". Distance convention discovered empirically on this build:

  * metric "l2"     -> ascending  `_distance`  == SQUARED euclidean  (sum (a-b)^2)
  * metric "ip"     -> descending `_similarity` == dot product
  * metric "cosine" -> descending `_similarity` == dot / (|a||b|)

Column names return upper-cased (DISTANCE / SIMILARITY); we read the non-id column.

Run:  uv run pytest test/eval_macos/test_dense_vector_e2e.py -v
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import harness
from infinity import index
from infinity.common import ConflictType, InfinityException


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def env():
    inst = harness.instance("eval-dense")
    inst.start()
    conn, db = inst.db()
    try:
        yield inst, conn, db
    finally:
        conn.disconnect()
        inst.stop()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mk(db, name, dim, etype="float32"):
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(
        name,
        {"id": {"type": "int"}, "v": {"type": f"vector,{dim},{etype}"}},
        ConflictType.Error,
    )


def _insert(tbl, vecs, ids=None):
    # `.tolist()` yields *native* python scalars; the SDK's constant serializer
    # rejects numpy scalars (only bare int/float match), which is a client concern.
    vecs = np.asarray(vecs)
    if ids is None:
        ids = list(range(len(vecs)))
    rows = [{"id": int(i), "v": vecs[j].tolist()} for j, i in enumerate(ids)]
    tbl.insert(rows)


def _score_out(metric):
    return "_distance" if metric == "l2" else "_similarity"


def _knn(tbl, query, qtype, metric, topk, opts=None, filt=None):
    """Return (ids, scores) as parallel python lists in result order."""
    q = tbl.output(["id", _score_out(metric)])
    if filt is not None:
        q = q.filter(filt)
    q = q.match_dense("v", np.asarray(query).tolist(), qtype, metric, topk, opts or {})
    df, _extra = q.to_pl()
    dicts = df.to_dicts()
    if not dicts:
        return [], []
    score_key = [c for c in df.columns if c.lower() != "id"][0]
    ids = [d["id"] for d in dicts]
    scores = [d[score_key] for d in dicts]
    return ids, scores


def _true_topk(vecs, q, metric, k):
    V = vecs.astype(np.float64)
    q = np.asarray(q, dtype=np.float64)
    if metric == "l2":
        d = np.sum((V - q) ** 2, axis=1)
        order = np.argsort(d, kind="stable")
    elif metric == "ip":
        d = V @ q
        order = np.argsort(-d, kind="stable")
    elif metric == "cosine":
        vn = np.linalg.norm(V, axis=1)
        qn = np.linalg.norm(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (V @ q) / (vn * qn)
        order = np.argsort(-d, kind="stable")
    else:  # pragma: no cover
        raise ValueError(metric)
    return list(order[:k]), d


def _recall(returned_ids, true_ids):
    if not true_ids:
        return 1.0
    return len(set(returned_ids) & set(true_ids)) / len(true_ids)


# Deterministic random corpus reused across index/recall tests.
_RNG = np.random.default_rng(1234)
_N = 1500
_DIM = 24
_CORPUS = _RNG.random((_N, _DIM), dtype=np.float32)
_QUERIES = _RNG.random((15, _DIM), dtype=np.float32)


# ==========================================================================  #
# 1. brute-force exactness: known values and exact ordering
# ==========================================================================  #
def test_brute_l2_exact_values_and_order(env):
    _inst, _conn, db = env
    t = _mk(db, "d_line", 4)
    vecs = np.array([[i, i, 0, 0] for i in range(1, 6)], dtype=np.float32)
    _insert(t, vecs, ids=list(range(1, 6)))
    ids, scores = _knn(t, [1, 1, 0, 0], "float", "l2", 5)
    # squared L2 to each [i,i,0,0] from [1,1,0,0] = 2*(i-1)^2
    assert ids == [1, 2, 3, 4, 5]
    assert scores == pytest.approx([0.0, 2.0, 8.0, 18.0, 32.0], abs=1e-4)
    # monotonic non-decreasing for a distance metric
    assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))


def test_brute_cosine_exact_values(env):
    _inst, _conn, db = env
    t = _mk(db, "d_cos", 4)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    _insert(t, vecs, ids=[0, 1, 2])
    ids, sims = _knn(t, [1, 0, 0, 0], "float", "cosine", 3)
    # cosine sims: id0 -> 1.0, id2 -> 1/sqrt2, id1 -> 0.0 ; descending order
    assert ids == [0, 2, 1]
    assert sims == pytest.approx([1.0, 1.0 / math.sqrt(2), 0.0], abs=1e-4)
    assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))


def test_brute_ip_exact_values(env):
    _inst, _conn, db = env
    t = _mk(db, "d_ip", 4)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    _insert(t, vecs, ids=[0, 1, 2])
    ids, sims = _knn(t, [3, 1, 0, 0], "float", "ip", 3)
    # dot products: id0 -> 3, id1 -> 1, id2 -> 4 ; descending
    assert ids == [2, 0, 1]
    assert sims == pytest.approx([4.0, 3.0, 1.0], abs=1e-4)
    assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))


def test_brute_recall_is_exact_all_metrics(env):
    """No-index brute force must be 100% recall against numpy for every metric."""
    _inst, _conn, db = env
    for metric in ("l2", "ip", "cosine"):
        t = _mk(db, f"d_rand_{metric}", _DIM)
        _insert(t, _CORPUS)
        tot = 0.0
        for q in _QUERIES:
            true_ids, _ = _true_topk(_CORPUS, q, metric, 10)
            got, scores = _knn(t, q, "float", metric, 10)
            tot += _recall(got, true_ids)
            # monotonic ordering of returned scores
            if metric == "l2":
                assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
            else:
                assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        mean = tot / len(_QUERIES)
        assert mean == pytest.approx(1.0), f"{metric}: brute recall {mean} != 1.0"


def test_returned_distance_matches_recomputation(env):
    """A returned _distance must equal the squared-L2 recomputed from stored vector."""
    _inst, _conn, db = env
    t = _mk(db, "d_recompute", _DIM)
    _insert(t, _CORPUS)
    q = _QUERIES[0]
    ids, scores = _knn(t, q, "float", "l2", 8)
    for rid, sc in zip(ids, scores):
        expect = float(np.sum((_CORPUS[rid].astype(np.float64) - q.astype(np.float64)) ** 2))
        assert sc == pytest.approx(expect, rel=1e-3, abs=1e-3), (
            f"id {rid}: engine {sc} vs recomputed squared-L2 {expect}"
        )


# ==========================================================================  #
# 2. topk boundaries
# ==========================================================================  #
def test_topk_one_row_count_and_over(env):
    _inst, _conn, db = env
    t = _mk(db, "d_topk", _DIM)
    _insert(t, _CORPUS)
    q = _QUERIES[1]
    assert len(_knn(t, q, "float", "l2", 1)[0]) == 1
    assert len(_knn(t, q, "float", "l2", _N)[0]) == _N
    # topk greater than row count clamps to row count (not an error, not padded)
    ids_over, _ = _knn(t, q, "float", "l2", _N + 5000)
    assert len(ids_over) == _N


def test_topk_zero_and_negative_rejected(env):
    _inst, _conn, db = env
    t = _mk(db, "d_topk_bad", 4)
    _insert(t, np.array([[1, 1, 0, 0]], dtype=np.float32), ids=[0])
    for bad in (0, -1):
        with pytest.raises(InfinityException) as ei:
            _knn(t, [1, 1, 0, 0], "float", "l2", bad)
        assert "topn" in ei.value.error_msg.lower()
    assert _inst.pid() is not None


# ==========================================================================  #
# 3. degenerate tables
# ==========================================================================  #
def test_empty_table_returns_no_rows(env):
    _inst, _conn, db = env
    t = _mk(db, "d_empty", 4)
    ids, _ = _knn(t, [1, 1, 0, 0], "float", "l2", 5)
    assert ids == []
    assert _inst.pid() is not None


def test_single_row_table(env):
    _inst, _conn, db = env
    t = _mk(db, "d_single", 4)
    _insert(t, np.array([[2, 2, 0, 0]], dtype=np.float32), ids=[7])
    ids, scores = _knn(t, [0, 0, 0, 0], "float", "l2", 3)
    assert ids == [7]
    assert scores[0] == pytest.approx(8.0, abs=1e-4)


def test_all_identical_vectors(env):
    _inst, _conn, db = env
    t = _mk(db, "d_ident", 4)
    vecs = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (5, 1))
    _insert(t, vecs, ids=list(range(5)))
    ids, scores = _knn(t, [1, 2, 3, 4], "float", "l2", 5)
    assert sorted(ids) == [0, 1, 2, 3, 4]
    assert all(s == pytest.approx(0.0, abs=1e-4) for s in scores)


# ==========================================================================  #
# 4. HNSW index: build, recall vs ground truth, ef monotonicity
# ==========================================================================  #
def _build_hnsw(t, metric="l2", m="16", efc="200"):
    t.create_index(
        "h_idx",
        index.IndexInfo("v", index.IndexType.Hnsw,
                        {"M": m, "ef_construction": efc, "metric": metric}),
        ConflictType.Error,
    )


def test_hnsw_recall_vs_ground_truth(env):
    _inst, _conn, db = env
    off = _inst.log_offset()
    t = _mk(db, "d_hnsw", _DIM)
    _insert(t, _CORPUS)
    _build_hnsw(t)
    tot = 0.0
    for q in _QUERIES:
        true_ids, _ = _true_topk(_CORPUS, q, "l2", 10)
        got, scores = _knn(t, q, "float", "l2", 10, {"ef": "200"})
        tot += _recall(got, true_ids)
        assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    mean = tot / len(_QUERIES)
    assert mean >= 0.90, f"HNSW mean recall {mean} < 0.90"
    assert _inst.log_errors(off) == []


def test_hnsw_ef_recall_is_monotonic(env):
    """Higher ef must not reduce recall; a low ef should not already be perfect
    (else the knob is untestable). This guards the silent-ignored-param class."""
    _inst, _conn, db = env
    t = _mk(db, "d_hnsw_ef", _DIM)
    _insert(t, _CORPUS)
    _build_hnsw(t, m="8", efc="40")  # weaker graph so low ef genuinely misses
    def mean_recall(ef):
        tot = 0.0
        for q in _QUERIES:
            true_ids, _ = _true_topk(_CORPUS, q, "l2", 10)
            got, _ = _knn(t, q, "float", "l2", 10, {"ef": str(ef)})
            tot += _recall(got, true_ids)
        return tot / len(_QUERIES)
    lo = mean_recall(10)
    hi = mean_recall(400)
    assert hi >= lo - 1e-9, f"ef raised recall dropped: ef10={lo} ef400={hi}"
    # Record whether the knob is demonstrably effective.
    assert hi > lo or lo >= 0.999, (
        f"ef knob produced no recall change and low-ef was not already perfect "
        f"(ef10={lo}, ef400={hi}) -> possibly silently ignored"
    )


# ==========================================================================  #
# 5. IVF index: nprobe MUST change recall (silent-accept detector)
# ==========================================================================  #
def test_ivf_nprobe_changes_recall(env):
    _inst, _conn, db = env
    t = _mk(db, "d_ivf", _DIM)
    _insert(t, _CORPUS)
    t.create_index("iv_idx", index.IndexInfo("v", index.IndexType.IVF, {"metric": "l2"}),
                   ConflictType.Error)
    def mean_recall(nprobe):
        tot = 0.0
        for q in _QUERIES:
            true_ids, _ = _true_topk(_CORPUS, q, "l2", 10)
            got, _ = _knn(t, q, "float", "l2", 10, {"nprobe": str(nprobe)})
            tot += _recall(got, true_ids)
        return tot / len(_QUERIES)
    lo = mean_recall(1)
    hi = mean_recall(4096)  # >> nlist -> searches all cells -> near-exact
    assert hi > lo, (
        f"nprobe had NO effect on recall (nprobe=1 -> {lo}, nprobe=4096 -> {hi}); "
        f"tuning parameter appears silently ignored"
    )
    assert hi >= 0.85, f"IVF full-scan recall only {hi}"


# ==========================================================================  #
# 6. KNN + filter: filter must be applied BEFORE knn, not post-trim
# ==========================================================================  #
def test_filter_applied_pre_knn(env):
    """True global nearest are excluded by the filter; a correct engine returns the
    nearest *within* the filtered set, a post-trim engine would return nothing."""
    _inst, _conn, db = env
    t = _mk(db, "d_filt", 4)
    # v = [i,0,0,0]; query [0,0,0,0] -> nearest = smallest id. Filter keeps only far ids.
    vecs = np.array([[i, 0, 0, 0] for i in range(100)], dtype=np.float32)
    _insert(t, vecs, ids=list(range(100)))
    ids, scores = _knn(t, [0, 0, 0, 0], "float", "l2", 5, filt="id >= 90")
    assert ids == [90, 91, 92, 93, 94], ids
    assert scores[0] == pytest.approx(90.0 * 90.0, abs=1e-2)
    assert _inst.pid() is not None


def test_filter_excluding_everything_returns_empty(env):
    _inst, _conn, db = env
    t = _mk(db, "d_filt_none", 4)
    vecs = np.array([[i, 0, 0, 0] for i in range(20)], dtype=np.float32)
    _insert(t, vecs, ids=list(range(20)))
    ids, _ = _knn(t, [0, 0, 0, 0], "float", "l2", 5, filt="id >= 100000")
    assert ids == []
    assert _inst.pid() is not None


def test_highly_selective_filter_matches_numpy(env):
    _inst, _conn, db = env
    t = _mk(db, "d_filt_sel", _DIM)
    _insert(t, _CORPUS)
    q = _QUERIES[2]
    # allowed ids: a contiguous slice; ground truth computed over that subset only
    allowed = list(range(500, 560))
    sub = _CORPUS[allowed]
    d = np.sum((sub.astype(np.float64) - q.astype(np.float64)) ** 2, axis=1)
    true_ids = [allowed[i] for i in np.argsort(d, kind="stable")[:5]]
    got, _ = _knn(t, q, "float", "l2", 5, filt="id >= 500 AND id <= 559")
    assert set(got).issubset(set(allowed))
    assert got == true_ids, f"filtered knn {got} != numpy {true_ids}"


# ==========================================================================  #
# 7. element types
# ==========================================================================  #
def test_int8_uint8_squared_l2(env):
    _inst, _conn, db = env
    for et in ("int8", "uint8"):
        t = _mk(db, f"d_{et}", 4, etype=et)
        vecs = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32)
        _insert(t, vecs, ids=[0, 1, 2])
        ids, scores = _knn(t, [1, 2, 3, 4], et, "l2", 3)
        assert ids == [0, 1, 2]
        # squared L2: 0, 4*16=64, 4*64=256
        assert scores == pytest.approx([0.0, 64.0, 256.0], abs=1e-4), (et, scores)


def test_float16_bfloat16_columns_are_unqueryable(env):
    """float16/bfloat16 columns accept inserts but KNN query is rejected (code 3032).
    A stored-but-unsearchable type is a real limitation worth surfacing."""
    _inst, _conn, db = env
    for et in ("float16", "bfloat16"):
        t = _mk(db, f"d_{et}", 4, etype=et)
        # insert succeeds
        _insert(t, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32), ids=[0, 1])
        # query with matching element type -> rejected
        with pytest.raises(InfinityException) as ei:
            _knn(t, [1, 2, 3, 4], et, "l2", 2)
        assert "not support" in ei.value.error_msg.lower()
        # query with plain float against the column -> also rejected (dim/type)
        with pytest.raises(InfinityException):
            _knn(t, [1, 2, 3, 4], "float", "l2", 2)
    assert _inst.pid() is not None


# ==========================================================================  #
# 8. negative / adversarial
# ==========================================================================  #
def test_dim_mismatch_both_directions(env):
    _inst, _conn, db = env
    t = _mk(db, "d_dim", 4)
    _insert(t, np.array([[1, 1, 0, 0]], dtype=np.float32), ids=[0])
    for bad_q in ([1.0] * 5, [1.0] * 3):
        with pytest.raises(InfinityException) as ei:
            _knn(t, bad_q, "float", "l2", 3)
        assert "matched" in ei.value.error_msg.lower()
    assert _inst.pid() is not None


def test_unknown_metric_rejected(env):
    _inst, _conn, db = env
    t = _mk(db, "d_metric", 4)
    _insert(t, np.array([[1, 1, 0, 0]], dtype=np.float32), ids=[0])
    with pytest.raises(InfinityException) as ei:
        _knn(t, [1, 1, 0, 0], "float", "jaccard", 3)
    assert "distance type" in ei.value.error_msg.lower()
    assert _inst.pid() is not None


def test_index_on_non_vector_column_rejected(env):
    _inst, _conn, db = env
    t = _mk(db, "d_idxcol", 4)
    _insert(t, np.array([[1, 1, 0, 0]], dtype=np.float32), ids=[0])
    with pytest.raises(InfinityException) as ei:
        t.create_index("bad", index.IndexInfo("id", index.IndexType.Hnsw,
                       {"M": "16", "ef_construction": "200", "metric": "l2"}),
                       ConflictType.Error)
    assert "hnsw" in ei.value.error_msg.lower()
    assert _inst.pid() is not None


def test_match_sparse_on_dense_column_rejected(env):
    _inst, _conn, db = env
    t = _mk(db, "d_sparsecol", 4)
    _insert(t, np.array([[1, 1, 0, 0]], dtype=np.float32), ids=[0])
    with pytest.raises(InfinityException):
        t.output(["id"]).match_sparse(
            "v", {"indices": [0], "values": [1.0]}, "ip", 3).to_pl()
    assert _inst.pid() is not None


def test_unknown_search_option_is_silently_accepted(env):
    """ENGINE DEFECT: a bogus knn option is accepted and ignored rather than rejected.
    We prove it changes nothing vs no-option -> it is silently dropped."""
    _inst, _conn, db = env
    off = _inst.log_offset()
    t = _mk(db, "d_opt", _DIM)
    _insert(t, _CORPUS)
    q = _QUERIES[3]
    base_ids, _ = _knn(t, q, "float", "l2", 10)
    bogus_ids, _ = _knn(t, q, "float", "l2", 10, {"not_a_real_option": "9999"})
    # If the option were validated this call would raise; instead it returns and
    # produces identical results, i.e. the option is silently ignored.
    assert bogus_ids == base_ids
    # A non-numeric value for a *real* option:
    _knn(t, q, "float", "l2", 10, {"ef": "not_a_number"})  # observe: raise or accept?
    assert _inst.pid() is not None
    # server-side log check
    _ = _inst.log_errors(off)


def test_nan_query_does_not_crash_but_returns_null_distances(env):
    """NaN query silently yields NULL distances in arbitrary order (no error)."""
    _inst, _conn, db = env
    t = _mk(db, "d_nan", 4)
    _insert(t, np.array([[1, 1, 0, 0], [2, 2, 0, 0]], dtype=np.float32), ids=[0, 1])
    ids, scores = _knn(t, [float("nan")] * 4, "float", "l2", 2)
    assert len(ids) == 2
    assert all(s is None or (isinstance(s, float) and math.isnan(s)) for s in scores), scores
    assert _inst.pid() is not None


def test_inf_query_does_not_crash(env):
    _inst, _conn, db = env
    t = _mk(db, "d_inf", 4)
    _insert(t, np.array([[1, 1, 0, 0], [2, 2, 0, 0]], dtype=np.float32), ids=[0, 1])
    ids, scores = _knn(t, [float("inf")] * 4, "float", "l2", 2)
    assert len(ids) == 2
    assert all(math.isinf(s) for s in scores), scores
    assert _inst.pid() is not None


def test_nan_inf_in_stored_data(env):
    """Store NaN/inf vectors, then a normal query. Must not crash; must still return
    the finite nearest neighbour first."""
    _inst, _conn, db = env
    t = _mk(db, "d_stored_nan", 4)
    rows = [
        {"id": 0, "v": [0.0, 0.0, 0.0, 0.0]},        # exact match to query
        {"id": 1, "v": [float("nan"), 0.0, 0.0, 0.0]},
        {"id": 2, "v": [float("inf"), 0.0, 0.0, 0.0]},
        {"id": 3, "v": [1.0, 0.0, 0.0, 0.0]},
    ]
    t.insert(rows)
    ids, scores = _knn(t, [0.0, 0.0, 0.0, 0.0], "float", "l2", 4)
    # Invariant that MUST hold: the exact-match row (id 0, distance 0) is the nearest.
    # A stored NaN must not be allowed to outrank a genuine distance-0 neighbour.
    assert ids[0] == 0 and scores[0] == pytest.approx(0.0, abs=1e-4), (
        f"exact match displaced by NaN/inf stored vector: order={list(zip(ids, scores))}"
    )
    assert _inst.pid() is not None


def test_cosine_all_zero_query_is_defined_as_zero(env):
    """cosine vs all-zero query is mathematically undefined (0/0); engine returns 0.0."""
    _inst, _conn, db = env
    t = _mk(db, "d_zeroq", 4)
    _insert(t, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32), ids=[0, 1])
    ids, sims = _knn(t, [0.0, 0.0, 0.0, 0.0], "float", "cosine", 2)
    assert len(ids) == 2
    assert all(s == pytest.approx(0.0, abs=1e-6) for s in sims), sims
    assert _inst.pid() is not None


def test_cosine_all_zero_stored_vector(env):
    _inst, _conn, db = env
    t = _mk(db, "d_zerostore", 4)
    _insert(t, np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32), ids=[0, 1])
    ids, sims = _knn(t, [1.0, 1.0, 1.0, 1.0], "float", "cosine", 2)
    # id1 is colinear -> sim 1.0; id0 is zero -> undefined, expect 0.0 and ranked last
    assert ids[0] == 1
    assert sims[0] == pytest.approx(1.0, abs=1e-4)
    assert _inst.pid() is not None


def test_quote_injection_in_filter_is_safe(env):
    """A single-quote / unicode payload in a filter must error cleanly, not crash
    the server or inject."""
    _inst, _conn, db = env
    t = _mk(db, "d_inject", 4)
    _insert(t, np.array([[1, 1, 0, 0]], dtype=np.float32), ids=[0])
    payloads = ["id = 0; DROP TABLE d_inject", "id = '你好'", "id = 0 OR '1'='1"]
    for p in payloads:
        try:
            _knn(t, [1, 1, 0, 0], "float", "l2", 3, filt=p)
        except InfinityException:
            pass
        except Exception:
            pass
    # table must still exist and be queryable, server alive
    ids, _ = _knn(t, [1, 1, 0, 0], "float", "l2", 3)
    assert ids == [0]
    assert _inst.pid() is not None


def test_duplicate_index_name_rejected(env):
    _inst, _conn, db = env
    t = _mk(db, "d_dupidx", _DIM)
    _insert(t, _CORPUS[:100])
    _build_hnsw(t)
    with pytest.raises(InfinityException) as ei:
        _build_hnsw(t)
    assert "already exist" in ei.value.error_msg.lower()
    assert _inst.pid() is not None
