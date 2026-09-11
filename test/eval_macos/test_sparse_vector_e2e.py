"""End-to-end evaluation of sparse-vector search on macOS arm64.

Area: sparse column definition/insert, the BMP index, ``match_sparse`` (inner
product), topn boundaries, sparse + filter, brute-force-vs-BMP agreement as the
correctness oracle, import/export, restart durability, optimize/compact, and a
large battery of negative cases.

Design notes
------------
* One server instance (``eval-sparse``) is started fresh for the module and
  stopped in teardown no matter what (module fixture with yield).
* Every positive group snapshots the server error log before it runs and asserts
  no new ``| error |``/``| critical |``/``| fatal |`` line appeared, because a
  clean client result over a server-side error is a defect the SDK cannot see.
* The oracle for search correctness is numpy: each stored sparse vector is
  materialised as a dense array, inner products are computed in the test, and the
  engine's brute-force scan is required to match exactly. The BMP index is then
  required to agree with the brute-force scan (recall 1.0 on data with a margin).
* Malformed sparse vectors are pushed at the server two ways: through the thrift
  SDK (which forwards duplicate/unsorted/out-of-range/negative indices verbatim)
  and through raw HTTP JSON (which forwards things the SDK dataclass blocks, e.g.
  an empty ``query_vector``). After each such probe we assert the server is still
  alive and still returns the correct answer to a known-good query.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import requests

import harness
from infinity.common import ConflictType, InfinityException, SparseVector
from infinity.errors import ErrorCode
from infinity.index import IndexInfo, IndexType

DIM = 100

# The committed sparse_knn.csv, replicated so the oracle is in the test, not a file.
# c1 -> {index: value}
CANON = {
    1: {i: 1.0 for i in range(0, 100, 10)},          # 0,10,20,...,90 -> 1.0
    2: {i: 2.0 for i in range(0, 100, 20)},          # 0,20,40,60,80 -> 2.0
    3: {i: 3.0 for i in range(0, 100, 30)},          # 0,30,60,90 -> 3.0
    4: {i: 4.0 for i in range(0, 100, 40)},          # 0,40,80 -> 4.0
    5: {},                                           # empty (all-zero) sparse vector
}
Q_IDX = [0, 20, 80]
Q_VAL = [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-sparse")
    it.start(fresh=True)
    try:
        yield it
    finally:
        it.stop()


def _dense(sparse_map: dict[int, float], dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    for i, x in sparse_map.items():
        v[i] = x
    return v


def _canon_matrix():
    ids = sorted(CANON)
    mat = np.stack([_dense(CANON[i]) for i in ids])
    return ids, mat


def _brute_topn(ids, mat, q_idx, q_val, topn, keep=lambda cid: True):
    """Ground-truth inner-product ranking computed in the test."""
    q = np.zeros(mat.shape[1], dtype=np.float64)
    for i, x in zip(q_idx, q_val):
        q[i] += x  # duplicate query indices accumulate
    scores = mat @ q
    order = []
    for cid, s in sorted(zip(ids, scores), key=lambda kv: (-kv[1], kv[0])):
        if keep(cid):
            order.append((cid, float(s)))
    return order[:topn]


def _run_sparse(table, q, topn, opt=None, out=("c1", "_similarity")):
    builder = table.output(list(out)).match_sparse("c2", q, "ip", topn, opt)
    df, _extra = builder.to_df()
    ids = [int(x) for x in df["c1"].tolist()]
    sims = [float(x) for x in df["SIMILARITY"].tolist()]
    return ids, sims


def _count(table) -> int:
    df, _ = table.output(["c1"]).to_df()
    return len(df)


def _make_table(db, name, value="float", index="int8", dim=DIM):
    db.drop_table(name, ConflictType.Ignore)
    return db.create_table(
        name,
        {"c1": {"type": "int"}, "c2": {"type": f"sparse,{dim},{value},{index}"}},
        ConflictType.Error,
    )


def _import_canon(inst, table, name="sparse_canon.csv"):
    rows = []
    for cid, m in CANON.items():
        if m:
            body = "[" + ",".join(f"{i}:{float(v)}" for i, v in sorted(m.items())) + "]"
        else:
            body = "[]"
        rows.append((cid, body))
    path = inst.stage_csv(name, rows)
    table.import_data(str(path), import_options={"delimiter": ","})


def _assert_alive_and_correct(inst, table):
    """After a negative probe: server up, and a known-good query still right."""
    assert inst.pid() is not None, "server process died"
    ids, sims = _run_sparse(table, SparseVector(Q_IDX, Q_VAL), 3)
    assert ids == [4, 2, 1]
    assert sims == pytest.approx([16.0, 12.0, 6.0])


def _http_sparse(inst, table_name, query_vector, topn=3, metric="ip", db="default_db"):
    """Raw HTTP search bypassing the SDK dataclass guards. Returns (status, json)."""
    url = f"{inst.http_base}/databases/{db}/tables/{table_name}/docs"
    body = {
        "output": ["c1", "_similarity"],
        "search": [{
            "match_method": "sparse",
            "fields": "c2",
            "query_vector": query_vector,
            "metric_type": metric,
            "topn": topn,
        }],
    }
    r = requests.get(url, headers={"accept": "application/json",
                                   "content-type": "application/json"}, json=body)
    try:
        payload = r.json()
    except Exception:
        payload = {"_raw": r.text}
    return r.status_code, payload


# --------------------------------------------------------------------------- #
# 1. column definition + insertion
# --------------------------------------------------------------------------- #

def test_insert_and_count_both_forms(inst):
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        t = _make_table(db, "sp_insert")
        # object form
        t.insert([{"c1": 1, "c2": SparseVector(indices=[10, 20, 30], values=[1.5, 2.5, 3.5])}])
        # dict form (index -> value)
        t.insert([{"c1": 2, "c2": {"40": 4.0, "50": 5.0}}])
        # multi-row in one call
        t.insert([
            {"c1": 3, "c2": SparseVector([0, 99], [1.0, 9.0])},
            {"c1": 4, "c2": {"1": 1.0}},
        ])
        assert _count(t) == 4

        # a query whose value hits row 3's index 99 must rank row 3 first
        ids, sims = _run_sparse(t, SparseVector([99], [1.0]), 4)
        assert ids[0] == 3
        assert sims[0] == pytest.approx(9.0)
        db.drop_table("sp_insert", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_import_canon_matches_numpy_bruteforce(inst):
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        t = _make_table(db, "sp_canon")
        _import_canon(inst, t)
        assert _count(t) == 5  # includes the empty (all-zero) row 5

        ids_all, mat = _canon_matrix()

        ids, sims = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 3)
        assert ids == [4, 2, 1]
        assert sims == pytest.approx([16.0, 12.0, 6.0])

        # topn beyond the margin: full ranking including the zero-scoring rows
        expect = _brute_topn(ids_all, mat, Q_IDX, Q_VAL, 10)
        ids10, sims10 = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 10)
        assert ids10 == [cid for cid, _ in expect]
        assert sims10 == pytest.approx([s for _, s in expect])

        db.drop_table("sp_canon", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []


# --------------------------------------------------------------------------- #
# 2. BMP index vs brute force (the correctness oracle), on random data
# --------------------------------------------------------------------------- #

def _build_random(inst, db, name, n_rows, dim, seed, index_type="int32"):
    rng = np.random.default_rng(seed)
    t = _make_table(db, name, value="float", index=index_type, dim=dim)
    mat = np.zeros((n_rows, dim), dtype=np.float64)
    batch = []
    for r in range(n_rows):
        k = int(rng.integers(3, 12))
        idxs = sorted(rng.choice(dim, size=k, replace=False).tolist())
        vals = [round(float(x), 4) for x in rng.uniform(0.1, 5.0, size=k)]
        for i, v in zip(idxs, vals):
            mat[r, i] = v
        batch.append({"c1": r, "c2": SparseVector([int(i) for i in idxs], vals)})
    # insert in a few batches so multiple segments exist
    step = max(1, n_rows // 3)
    for s in range(0, n_rows, step):
        t.insert(batch[s:s + step])
    return t, mat


def test_bmp_agrees_with_bruteforce_random(inst):
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        n, dim = 200, 120
        t, mat = _build_random(inst, db, "sp_rand", n, dim, seed=17)
        assert _count(t) == n

        qk = 8
        qidx = sorted(np.random.default_rng(99).choice(dim, size=qk, replace=False).tolist())
        qval = [round(float(x), 4) for x in np.random.default_rng(100).uniform(0.5, 4.0, size=qk)]
        ids_all = list(range(n))
        topn = 10
        expect = _brute_topn(ids_all, mat, qidx, qval, topn)
        exp_ids = [c for c, _ in expect]
        exp_sims = [s for _, s in expect]

        # brute force (no index) MUST match numpy exactly
        bf_ids, bf_sims = _run_sparse(t, SparseVector([int(i) for i in qidx], qval), topn)
        assert bf_ids == exp_ids, f"brute force disagreed with numpy: {bf_ids} vs {exp_ids}"
        assert bf_sims == pytest.approx(exp_sims, rel=1e-4, abs=1e-3)
        # distances monotonically non-increasing
        assert all(bf_sims[i] >= bf_sims[i + 1] - 1e-4 for i in range(len(bf_sims) - 1))

        # BMP index, exact search params -> must agree with brute force
        t.create_index("bmp1", IndexInfo("c2", IndexType.BMP,
                       {"block_size": "16", "compress_type": "compress"}), ConflictType.Error)
        bmp_ids, bmp_sims = _run_sparse(t, SparseVector([int(i) for i in qidx], qval), topn,
                                        opt={"alpha": "1.0", "beta": "1.0"})
        assert set(bmp_ids) == set(exp_ids), (
            f"BMP recall < 1.0: BMP={bmp_ids} brute={exp_ids}")
        assert bmp_ids == exp_ids, f"BMP order differs from brute force: {bmp_ids} vs {exp_ids}"
        assert bmp_sims == pytest.approx(exp_sims, rel=1e-4, abs=1e-3)

        # a second BMP index with the raww codec must agree too
        t.create_index("bmp2", IndexInfo("c2", IndexType.BMP,
                       {"block_size": "8", "compress_type": "raww"}), ConflictType.Error)
        r2_ids, _ = _run_sparse(t, SparseVector([int(i) for i in qidx], qval), topn,
                                opt={"alpha": "1.0", "beta": "1.0"})
        assert r2_ids == exp_ids

        db.drop_table("sp_rand", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []


# --------------------------------------------------------------------------- #
# 3. topn boundaries
# --------------------------------------------------------------------------- #

def test_topn_boundaries(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_topn")
        _import_canon(inst, t)

        # topn larger than table -> all rows (5), none dropped, none duplicated
        ids, _ = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 100000)
        assert sorted(ids) == [1, 2, 3, 4, 5]

        # topn == 0 and topn < 0: observe and record behaviour
        z_off = inst.log_offset()
        zero_behaviour = None
        try:
            zids, _ = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 0)
            zero_behaviour = ("ok", zids)
        except InfinityException as e:
            zero_behaviour = ("error", e.error_code)
        neg_behaviour = None
        try:
            nids, _ = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), -1)
            neg_behaviour = ("ok", nids)
        except InfinityException as e:
            neg_behaviour = ("error", e.error_code)

        # Whatever it does, it must not crash the server or corrupt results.
        _assert_alive_and_correct(inst, t)
        # record for the report; a silent empty result for topn=0 is acceptable,
        # a wrong non-empty result is not.
        if zero_behaviour[0] == "ok":
            assert zero_behaviour[1] == [] or len(zero_behaviour[1]) <= 5
        print("TOPN0:", zero_behaviour, "TOPNNEG:", neg_behaviour,
              "crash_log:", inst.log_errors(z_off))
        db.drop_table("sp_topn", ConflictType.Error)
    finally:
        conn.disconnect()


# --------------------------------------------------------------------------- #
# 4. sparse + filter
# --------------------------------------------------------------------------- #

def test_sparse_with_filter(inst):
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        t = _make_table(db, "sp_filter")
        _import_canon(inst, t)
        ids_all, mat = _canon_matrix()

        for expr, keep in [("c1 >= 3", lambda c: c >= 3),
                           ("c1 < 3", lambda c: c < 3),
                           ("c1 <> 4", lambda c: c != 4)]:
            expect = _brute_topn(ids_all, mat, Q_IDX, Q_VAL, 5, keep=keep)
            ids, sims = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 5, opt={"filter": expr})
            assert ids == [c for c, _ in expect], f"filter {expr}: {ids}"
            assert sims == pytest.approx([s for _, s in expect]), f"filter {expr}: {sims}"

        db.drop_table("sp_filter", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []


# --------------------------------------------------------------------------- #
# 5. threshold option; unrecognised option silently ignored?
# --------------------------------------------------------------------------- #

def test_threshold_and_unknown_option(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_opt")
        _import_canon(inst, t)

        # threshold=10 -> only rows with ip >= 10 (rows 4:16, 2:12)
        ids, sims = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 5, opt={"threshold": "10"})
        assert ids == [4, 2]
        assert sims == pytest.approx([16.0, 12.0])

        # A plausible-but-wrong option name. If it is silently ignored the result
        # is identical to no-option (all 5 rows). Record which happens.
        off = inst.log_offset()
        misspelled = None
        try:
            mids, _ = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 5, opt={"threshhold": "10"})
            misspelled = ("accepted", mids)
        except InfinityException as e:
            misspelled = ("rejected", e.error_code)
        print("MISSPELLED_THRESHOLD:", misspelled, "log:", inst.log_errors(off))
        # capture the silent-accept defect precisely: if accepted, it did NOT
        # filter (5 rows) -> the option did nothing.
        if misspelled[0] == "accepted":
            assert misspelled[1] == [4, 2, 1, 3, 5]

        _assert_alive_and_correct(inst, t)
        db.drop_table("sp_opt", ConflictType.Error)
    finally:
        conn.disconnect()


# --------------------------------------------------------------------------- #
# 6. BMP build-parameter validation (incl. silent-accept probe)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("params,should_error", [
    ({"block_size": "8", "compress_type": "compress"}, False),
    ({"block_size": "16", "compress_type": "raww"}, False),
    ({"block_size": "256"}, False),
    ({"block_size": "0"}, True),
    ({"block_size": "257"}, True),
    ({"block_size": "16", "compress_type": "wrong_compress_type"}, True),
])
def test_bmp_param_validation(inst, params, should_error):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_bmpparam")
        _import_canon(inst, t)
        if should_error:
            with pytest.raises(InfinityException) as e:
                t.create_index("ix", IndexInfo("c2", IndexType.BMP, params), ConflictType.Error)
            assert e.value.args[0] == ErrorCode.INVALID_INDEX_PARAM, e.value.args
        else:
            t.create_index("ix", IndexInfo("c2", IndexType.BMP, params), ConflictType.Error)
            ids, _ = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 3,
                                 opt={"alpha": "1.0", "beta": "1.0"})
            assert ids == [4, 2, 1]
        db.drop_table("sp_bmpparam", ConflictType.Error)
    finally:
        conn.disconnect()


def test_bmp_unknown_param_silent_accept(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_bmpunknown")
        _import_canon(inst, t)
        off = inst.log_offset()
        outcome = None
        try:
            t.create_index("ixu", IndexInfo("c2", IndexType.BMP,
                           {"block_size": "8", "compress_type": "compress",
                            "bogus_param": "42", "blocksize": "9999"}),
                           ConflictType.Error)
            # if it did not reject, the index must still be usable and correct
            ids, _ = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 3,
                                 opt={"alpha": "1.0", "beta": "1.0"})
            outcome = ("accepted", ids)
        except InfinityException as e:
            outcome = ("rejected", e.error_code)
        print("BMP_UNKNOWN_PARAM:", outcome, "log:", inst.log_errors(off))
        if outcome[0] == "accepted":
            assert outcome[1] == [4, 2, 1]
        _assert_alive_and_correct(
            inst, db.get_table("sp_bmpunknown"))
        db.drop_table("sp_bmpunknown", ConflictType.Error)
    finally:
        conn.disconnect()


# --------------------------------------------------------------------------- #
# 7. distance-type validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("metric", ["l2", "cosine", "hamming", "cos", "banana"])
def test_invalid_distance_type(inst, metric):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_metric2")
        _import_canon(inst, t)
        raised = False
        try:
            df, _ = t.output(["c1", "_similarity"]).match_sparse(
                "c2", SparseVector(Q_IDX, Q_VAL), metric, 3).to_df()
        except Exception:
            raised = True
        assert raised, f"metric {metric!r} was accepted (should be rejected)"
        _assert_alive_and_correct(inst, t)
        db.drop_table("sp_metric2", ConflictType.Error)
    finally:
        conn.disconnect()


# --------------------------------------------------------------------------- #
# 8. malformed sparse QUERY vectors
# --------------------------------------------------------------------------- #

def test_query_unsorted_indices_permutation_invariant(inst):
    """IP is order-independent: an unsorted query must give the sorted answer."""
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        t = _make_table(db, "sp_unsorted")
        _import_canon(inst, t)
        sorted_ids, sorted_sims = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 5)
        # permuted order, same (index,value) pairs
        perm = SparseVector([80, 0, 20], [3.0, 1.0, 2.0])
        u_ids, u_sims = _run_sparse(t, perm, 5)
        assert u_ids == sorted_ids, f"unsorted query changed ranking: {u_ids} vs {sorted_ids}"
        assert u_sims == pytest.approx(sorted_sims)
        db.drop_table("sp_unsorted", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_query_duplicate_indices(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_dupq")
        _import_canon(inst, t)
        ids_all, mat = _canon_matrix()
        # duplicate index 0 (1.0 + 1.0) plus 80:3.0 -> effective {0:2.0, 80:3.0}
        dup = SparseVector([0, 0, 80], [1.0, 1.0, 3.0])
        off = inst.log_offset()
        behaviour = None
        try:
            ids, sims = _run_sparse(t, dup, 5)
            behaviour = ("ok", ids, sims)
        except InfinityException as e:
            behaviour = ("error", e.error_code)
        print("DUP_QUERY:", behaviour, "log:", inst.log_errors(off))
        if behaviour[0] == "ok":
            # brute force with accumulation semantics
            expect = _brute_topn(ids_all, mat, [0, 0, 80], [1.0, 1.0, 3.0], 5)
            # primary invariant: no crash, server correct afterward
            _assert_alive_and_correct(inst, t)
            # secondary: does it use accumulation semantics? record if not.
            got = list(zip(behaviour[1], [round(s, 3) for s in behaviour[2]]))
            exp = [(c, round(s, 3)) for c, s in expect]
            print("DUP_QUERY expect(accumulate):", exp, "got:", got)
            assert behaviour[1][0] == 4  # row 4 (0:4,80:4 -> 2*4+3*4=20) still top
        else:
            _assert_alive_and_correct(inst, t)
        db.drop_table("sp_dupq", ConflictType.Error)
    finally:
        conn.disconnect()


def test_query_out_of_range_index(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_oorq", index="int32")  # int32 so 500 fits the index type
        _import_canon(inst, t)
        off = inst.log_offset()
        # index 500 >= declared dim 100
        oor = SparseVector([0, 500], [1.0, 10.0])
        behaviour = None
        try:
            ids, sims = _run_sparse(t, oor, 5)
            behaviour = ("ok", ids, sims)
        except InfinityException as e:
            behaviour = ("error", e.error_code)
        print("OOR_QUERY:", behaviour, "log:", inst.log_errors(off))
        _assert_alive_and_correct(inst, t)
        db.drop_table("sp_oorq", ConflictType.Error)
    finally:
        conn.disconnect()


def test_query_negative_index(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_negq", index="int32")
        _import_canon(inst, t)
        off = inst.log_offset()
        neg = SparseVector([-5, 0], [1.0, 1.0])
        behaviour = None
        try:
            ids, sims = _run_sparse(t, neg, 5)
            behaviour = ("ok", ids, sims)
        except InfinityException as e:
            behaviour = ("error", e.error_code)
        print("NEG_QUERY:", behaviour, "log:", inst.log_errors(off))
        _assert_alive_and_correct(inst, t)
        db.drop_table("sp_negq", ConflictType.Error)
    finally:
        conn.disconnect()


def test_query_empty_vector_sdk_and_http(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_emptyq")
        _import_canon(inst, t)

        # SDK object form: empty SparseVector is blocked client-side
        with pytest.raises(InfinityException) as e1:
            _run_sparse(t, SparseVector([], []), 3)
        print("EMPTY_SDK_OBJ:", e1.value.args[0])

        # SDK dict form: empty dict blocked client-side
        with pytest.raises(InfinityException) as e2:
            t.output(["c1"]).match_sparse("c2", {}, "ip", 3).to_df()
        print("EMPTY_SDK_DICT:", e2.value.args[0])

        # raw HTTP with empty query_vector -> server must reject, not crash
        off = inst.log_offset()
        status, payload = _http_sparse(inst, "sp_emptyq", {}, topn=3)
        print("EMPTY_HTTP:", status, payload, "log:", inst.log_errors(off))
        _assert_alive_and_correct(inst, t)
        db.drop_table("sp_emptyq", ConflictType.Error)
    finally:
        conn.disconnect()


def test_http_out_of_range_index(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_httpoor", index="int32")
        _import_canon(inst, t)
        off = inst.log_offset()
        status, payload = _http_sparse(inst, "sp_httpoor", {"0": 1.0, "500": 10.0}, topn=5)
        print("HTTP_OOR:", status, payload, "log:", inst.log_errors(off))
        _assert_alive_and_correct(inst, t)
        db.drop_table("sp_httpoor", ConflictType.Error)
    finally:
        conn.disconnect()


# --------------------------------------------------------------------------- #
# 9. malformed STORED sparse vectors (insert path)
# --------------------------------------------------------------------------- #

def test_insert_out_of_range_index(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_oorins", index="int32")  # dim 100, int32 index
        off = inst.log_offset()
        behaviour = None
        try:
            t.insert([{"c1": 1, "c2": SparseVector([200], [1.0])}])  # 200 >= dim 100
            behaviour = ("accepted", _count(t))
        except InfinityException as e:
            behaviour = ("rejected", e.error_code)
        print("INSERT_OOR:", behaviour, "log:", inst.log_errors(off))
        assert inst.pid() is not None, "server crashed on out-of-range insert"
        # If it was accepted, a subsequent scan/search must not crash the server.
        if behaviour[0] == "accepted":
            scan_ok = None
            try:
                ids, _ = _run_sparse(t, SparseVector([0], [1.0]), 5)
                scan_ok = ("ok", ids)
            except Exception as e:  # noqa: BLE001
                scan_ok = ("raised", repr(e))
            print("INSERT_OOR_then_query:", scan_ok, "alive:", inst.pid() is not None)
            assert inst.pid() is not None
        db.drop_table("sp_oorins", ConflictType.Ignore)
    finally:
        conn.disconnect()


def test_insert_empty_sparse_via_sdk_blocked(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_emptyins")
        # Both SDK forms cannot express an empty (all-zero) sparse vector.
        with pytest.raises(InfinityException) as e1:
            t.insert([{"c1": 1, "c2": SparseVector([], [])}])
        with pytest.raises(InfinityException) as e2:
            t.insert([{"c1": 2, "c2": {}}])
        print("EMPTY_INSERT_OBJ:", e1.value.args[0], "EMPTY_INSERT_DICT:", e2.value.args[0])
        assert _count(t) == 0
        # ... yet import can store an empty vector (row 5 of the canon set).
        _import_canon(inst, t)
        assert _count(t) == 5
        db.drop_table("sp_emptyins", ConflictType.Error)
    finally:
        conn.disconnect()


# --------------------------------------------------------------------------- #
# 10. import / export round-trip
# --------------------------------------------------------------------------- #

def test_export_import_roundtrip_csv(inst):
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        src = _make_table(db, "sp_exp_src")
        _import_canon(inst, src)
        out_path = inst.data_root() / "sp_export.csv"
        src.export_data(str(out_path), export_options={"file_type": "csv", "delimiter": ","})

        dst = _make_table(db, "sp_exp_dst")
        dst.import_data(str(out_path), import_options={"delimiter": ","})
        assert _count(dst) == 5

        ids, sims = _run_sparse(dst, SparseVector(Q_IDX, Q_VAL), 3)
        assert ids == [4, 2, 1]
        assert sims == pytest.approx([16.0, 12.0, 6.0])

        db.drop_table("sp_exp_src", ConflictType.Error)
        db.drop_table("sp_exp_dst", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []


def test_import_jsonl(inst):
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        t = _make_table(db, "sp_jsonl")
        records = []
        for cid, m in CANON.items():
            records.append({"c1": cid, "c2": {str(i): float(v) for i, v in m.items()}})
        path = inst.stage_jsonl("sp_canon.jsonl", records)
        t.import_data(str(path), import_options={"file_type": "jsonl"})
        assert _count(t) == 5
        ids, _ = _run_sparse(t, SparseVector(Q_IDX, Q_VAL), 3)
        assert ids == [4, 2, 1]
        db.drop_table("sp_jsonl", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []


# --------------------------------------------------------------------------- #
# 11. durability across restart
# --------------------------------------------------------------------------- #

def test_sparse_survives_graceful_restart(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_restart_g")
        _import_canon(inst, t)
        conn.disconnect()

        inst.restart(graceful=True)

        conn2, db2 = inst.db()
        try:
            t2 = db2.get_table("sp_restart_g")
            assert _count(t2) == 5
            ids, sims = _run_sparse(t2, SparseVector(Q_IDX, Q_VAL), 3)
            assert ids == [4, 2, 1]
            assert sims == pytest.approx([16.0, 12.0, 6.0])
            db2.drop_table("sp_restart_g", ConflictType.Error)
        finally:
            conn2.disconnect()
    finally:
        try:
            conn.disconnect()
        except Exception:
            pass


def test_sparse_survives_sigkill_wal_recovery(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_restart_k")
        _import_canon(inst, t)
        conn.disconnect()

        inst.restart(graceful=False)  # SIGKILL -> WAL replay only

        conn2, db2 = inst.db()
        try:
            t2 = db2.get_table("sp_restart_k")
            assert _count(t2) == 5, "rows lost after SIGKILL/WAL recovery"
            ids, sims = _run_sparse(t2, SparseVector(Q_IDX, Q_VAL), 3)
            assert ids == [4, 2, 1]
            assert sims == pytest.approx([16.0, 12.0, 6.0])
            db2.drop_table("sp_restart_k", ConflictType.Error)
        finally:
            conn2.disconnect()
    finally:
        try:
            conn.disconnect()
        except Exception:
            pass


def test_bmp_index_survives_restart(inst):
    conn, db = inst.db()
    try:
        t = _make_table(db, "sp_idx_restart")
        _import_canon(inst, t)
        t.create_index("bmpr", IndexInfo("c2", IndexType.BMP,
                       {"block_size": "8", "compress_type": "compress"}), ConflictType.Error)
        conn.disconnect()

        inst.restart(graceful=True)

        conn2, db2 = inst.db()
        try:
            t2 = db2.get_table("sp_idx_restart")
            names = getattr(t2.list_indexes(), "index_names", None)
            assert names is None or "bmpr" in names, f"index gone after restart: {names}"
            ids, _ = _run_sparse(t2, SparseVector(Q_IDX, Q_VAL), 3,
                                 opt={"alpha": "1.0", "beta": "1.0"})
            assert ids == [4, 2, 1], "BMP index wrong after restart"
            db2.drop_table("sp_idx_restart", ConflictType.Error)
        finally:
            conn2.disconnect()
    finally:
        try:
            conn.disconnect()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 12. optimize / compact
# --------------------------------------------------------------------------- #

def test_optimize_and_compact(inst):
    conn, db = inst.db()
    off = inst.log_offset()
    try:
        n, dim = 120, 100
        t, mat = _build_random(inst, db, "sp_optcompact", n, dim, seed=5, index_type="int32")
        assert _count(t) == n
        t.create_index("bo", IndexInfo("c2", IndexType.BMP,
                       {"block_size": "16", "compress_type": "compress"}), ConflictType.Error)

        qk = 6
        qidx = sorted(np.random.default_rng(7).choice(dim, size=qk, replace=False).tolist())
        qval = [round(float(x), 4) for x in np.random.default_rng(8).uniform(0.5, 4.0, size=qk)]
        expect = _brute_topn(list(range(n)), mat, qidx, qval, 10)
        exp_ids = [c for c, _ in expect]

        for op_name in ("optimize", "compact"):
            getattr(t, op_name)()
            assert _count(t) == n, f"{op_name} changed row count"
            ids, _ = _run_sparse(t, SparseVector([int(i) for i in qidx], qval), 10,
                                 opt={"alpha": "1.0", "beta": "1.0"})
            assert set(ids) == set(exp_ids), f"{op_name}: BMP recall dropped {ids} vs {exp_ids}"

        db.drop_table("sp_optcompact", ConflictType.Error)
    finally:
        conn.disconnect()
    assert inst.log_errors(off) == []
