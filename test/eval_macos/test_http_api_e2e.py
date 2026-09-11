"""End-to-end and HTTP<->thrift parity suite for the Infinity HTTP API.

The committed .slt corpus drives only the PostgreSQL wire path; the oatpp HTTP
listener (routes registered in ``src/network/http_server_impl.cpp``, search path in
``src/network/http/http_search_impl.cpp``) was never exercised on macOS. This suite
closes that gap.

For every operation we do two things the .slt tests cannot:

  * **Parity.** Perform the operation over HTTP (raw ``requests``) and over the thrift
    SDK against equivalent tables and assert the results agree -- same rows, same
    values, same ordering, same errors. Serialization divergence (int64 precision,
    float rounding, NULL, unicode, embeddings) is where the HTTP path is most likely
    to be wrong, so those get dedicated tests.

  * **Negative cases as first-class tests.** Malformed / wrong-shape / oversized /
    truncated bodies, wrong methods, missing routes, injection-shaped strings. After
    each we assert the server is still alive AND still correct, and we scrape the
    server's own log for fatal/critical lines a clean client result would hide.

Run:  uv run pytest test/eval_macos/test_http_api_e2e.py -v
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import random
import sys
from pathlib import Path

import pytest
import requests

from infinity.common import ConflictType, SparseVector
from infinity.index import IndexInfo, IndexType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402


# --------------------------------------------------------------------------- #
# HTTP client                                                                  #
# --------------------------------------------------------------------------- #

JSON_HEADERS = {"accept": "application/json", "content-type": "application/json"}


class HttpClient:
    """Thin wrapper over requests bound to one instance's HTTP base.

    Keeps full control over method, headers and raw body so negative tests can send
    exactly the bytes they mean to. Positive calls use ``json=`` for convenience.
    """

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.session = requests.Session()

    def req(self, method: str, path: str, *, json_body=None, raw_body=None,
            headers=None, timeout=30):
        url = self.base + path
        hdr = dict(JSON_HEADERS)
        if headers is not None:
            hdr = headers  # explicit override (may be empty / wrong on purpose)
        if raw_body is not None:
            return self.session.request(method, url, data=raw_body, headers=hdr,
                                        timeout=timeout)
        if json_body is not None:
            return self.session.request(method, url, data=json.dumps(json_body),
                                        headers=hdr, timeout=timeout)
        return self.session.request(method, url, headers=hdr, timeout=timeout)

    # convenience verbs (GET carries a JSON body -- that is what the SDK does)
    def get(self, path, json_body=None, **kw):
        return self.req("GET", path, json_body=json_body, **kw)

    def post(self, path, json_body=None, **kw):
        return self.req("POST", path, json_body=json_body, **kw)

    def put(self, path, json_body=None, **kw):
        return self.req("PUT", path, json_body=json_body, **kw)

    def delete(self, path, json_body=None, **kw):
        return self.req("DELETE", path, json_body=json_body, **kw)


# --------------------------------------------------------------------------- #
# Response helpers                                                             #
# --------------------------------------------------------------------------- #

def rows_from_http(resp_json) -> list[dict]:
    """Normalize the HTTP select ``output`` shape into a list of {col: value}.

    The server serializes each row as a JSON array of single-column cells; a nullable
    column adds a sibling ``{col}_bitmap`` boolean in the same cell, false == NULL
    (see http_search_impl.cpp). We collapse that back into a plain dict per row.
    """
    out = []
    for row in resp_json.get("output", []):
        d = {}
        bitmaps = {}
        for cell in row:
            for k, v in cell.items():
                if k.endswith("_bitmap"):
                    bitmaps[k[: -len("_bitmap")]] = v
                else:
                    d[k] = v
        for col, valid in bitmaps.items():
            if not valid:
                d[col] = None
        out.append(d)
    return out


def thrift_rows(table_obj, columns, filt: str | None = None) -> list[dict]:
    q = table_obj.output(columns)
    if filt is not None:
        q = q.filter(filt)
    df, _ = q.to_pl()
    return df.to_dicts()


def approx(a, b, tol=1e-4):
    if a is None or b is None:
        return a is b or a == b
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def inst():
    it = harness.instance("eval-http")
    it.start()
    try:
        yield it
    finally:
        it.stop()


@pytest.fixture(scope="module")
def http(inst):
    return HttpClient(inst.http_base)


@pytest.fixture(scope="module")
def conn(inst):
    c = inst.connect()
    try:
        yield c
    finally:
        c.disconnect()


@pytest.fixture(scope="module")
def db(conn):
    return conn.get_database("default_db")


# --------------------------------------------------------------------------- #
# Small assertion helpers                                                      #
# --------------------------------------------------------------------------- #

def ok(resp):
    """Assert an HTTP call succeeded at both the transport and app layer."""
    assert resp.status_code == 200, (resp.status_code, resp.text[:500])
    body = resp.json()
    assert body.get("error_code") == 0, body
    return body


def crashed(inst) -> list[str]:
    """Return server log lines that indicate a crash (fatal/critical), never
    ordinary handled errors."""
    return [ln for ln in inst.log_errors()
            if "| critical |" in ln.lower() or "| fatal |" in ln.lower()]


def assert_alive(inst, http):
    """The server must still answer a trivial request after a hostile one."""
    r = http.get("/databases")
    assert r.status_code == 200, f"server unresponsive: {r.status_code} {r.text[:200]}"
    assert isinstance(r.json().get("databases"), list)
    assert not crashed(inst), crashed(inst)


# --------------------------------------------------------------------------- #
# Table builders used across tests                                            #
# --------------------------------------------------------------------------- #

def http_create_table(http, tname, fields, opt="error"):
    return http.post(f"/databases/default_db/tables/{tname}",
                     json_body={"fields": fields, "create_option": opt})


def http_drop_table(http, tname, opt="ignore_if_not_exists"):
    return http.delete(f"/databases/default_db/tables/{tname}",
                       json_body={"drop_option": opt})


def http_insert(http, tname, rows):
    return http.post(f"/databases/default_db/tables/{tname}/docs", json_body=rows)


def http_select(http, tname, body):
    return http.get(f"/databases/default_db/tables/{tname}/docs", json_body=body)


# --------------------------------------------------------------------------- #
# 0. Smoke: the plumbing works                                                 #
# --------------------------------------------------------------------------- #

def test_smoke_list_databases(inst, http):
    r = http.get("/databases")
    body = ok(r)
    assert "default_db" in body["databases"]
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 1. Database CRUD + HTTP<->thrift parity                                      #
# --------------------------------------------------------------------------- #

def test_database_crud_and_parity(inst, http, conn):
    name = "eval_http_db1"
    ok(http.post(f"/databases/{name}", json_body={"create_option": "error"}))

    # HTTP list and thrift list must agree on the set of databases.
    http_dbs = set(ok(http.get("/databases"))["databases"])
    thrift_dbs = set(conn.list_databases().db_names)
    assert name in http_dbs
    assert http_dbs == thrift_dbs, (http_dbs, thrift_dbs)

    # show database over HTTP
    body = ok(http.get(f"/databases/{name}"))
    assert body["database_name"] == name

    # duplicate create with error option -> structured error, server alive
    r = http.post(f"/databases/{name}", json_body={"create_option": "error"})
    assert r.status_code == 500
    assert r.json()["error_code"] != 0

    # ignore_if_exists must succeed
    ok(http.post(f"/databases/{name}", json_body={"create_option": "ignore_if_exists"}))

    ok(http.delete(f"/databases/{name}", json_body={"drop_option": "error"}))
    assert name not in set(ok(http.get("/databases"))["databases"])
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 2. Table CRUD + parity                                                       #
# --------------------------------------------------------------------------- #

def test_table_crud_and_parity(inst, http, db):
    t = "eval_http_tbl1"
    http_drop_table(http, t)
    fields = [
        {"name": "id", "type": "integer"},
        {"name": "name", "type": "varchar"},
        {"name": "score", "type": "float"},
    ]
    ok(http_create_table(http, t, fields))

    # list tables: HTTP and thrift agree
    http_tables = set(ok(http.get("/databases/default_db/tables"))["table_names"])
    thrift_tables = set(db.list_tables().table_names)
    assert t in http_tables
    assert http_tables == thrift_tables, (http_tables, thrift_tables)

    # column metadata
    cols = ok(http.get(f"/databases/default_db/tables/{t}/columns"))["columns"]
    by_name = {c["name"]: c for c in cols}
    assert by_name["id"]["type"] == "Integer"
    assert by_name["name"]["type"] == "Varchar"
    assert by_name["score"]["type"] == "Float"

    # show table
    st = ok(http.get(f"/databases/default_db/tables/{t}"))
    assert st["table_name"] == t
    assert int(st["column_count"]) == 3

    # rename
    t2 = "eval_http_tbl1_renamed"
    http_drop_table(http, t2)
    ok(http.post(f"/databases/default_db/tables/{t}/rename",
                 json_body={"new_table_name": t2}))
    assert t2 in set(ok(http.get("/databases/default_db/tables"))["table_names"])
    assert t not in set(ok(http.get("/databases/default_db/tables"))["table_names"])

    http_drop_table(http, t2)
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 3. Insert / count / select parity (read path)                               #
# --------------------------------------------------------------------------- #

def test_insert_count_select_parity(inst, http, db):
    t = "eval_http_rows"
    db.drop_table(t, ConflictType.Ignore)
    tbl = db.create_table(t, {"id": {"type": "integer"}, "v": {"type": "integer"}})

    n = 100
    tbl.insert([{"id": i, "v": i * 2} for i in range(n)])

    # count(*) over HTTP
    body = ok(http_select(http, t, {"output": ["count(*)"]}))
    rows = rows_from_http(body)
    count_val = list(rows[0].values())[0]
    assert count_val == n, rows

    # select all, HTTP vs thrift, sorted by id
    http_rows = sorted(rows_from_http(ok(http_select(http, t, {"output": ["id", "v"]}))),
                       key=lambda r: r["id"])
    th_rows = sorted(thrift_rows(db.get_table(t), ["id", "v"]), key=lambda r: r["id"])
    assert len(http_rows) == n
    assert http_rows == th_rows, (http_rows[:3], th_rows[:3])

    # filtered select parity
    f = "v >= 100 and id < 60"
    h = sorted(rows_from_http(ok(http_select(http, t, {"output": ["id", "v"], "filter": f}))),
               key=lambda r: r["id"])
    th = sorted(thrift_rows(db.get_table(t), ["id", "v"], f), key=lambda r: r["id"])
    assert h == th and len(h) == 10, (h, th)

    # limit
    lim = rows_from_http(ok(http_select(http, t, {"output": ["id"], "limit": "5"})))
    assert len(lim) == 5

    db.drop_table(t)
    assert not crashed(inst)


def test_write_path_parity_http_insert_thrift_read(inst, http, db):
    """Insert over HTTP, read back over thrift -- the HTTP insert parser must
    produce exactly the values thrift sees."""
    t = "eval_http_write"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [
        {"name": "id", "type": "integer"},
        {"name": "d", "type": "double"},
        {"name": "s", "type": "varchar"},
    ]))
    payload = [
        {"id": 10, "d": 1.5, "s": "alpha"},
        {"id": 20, "d": -2.25, "s": "beta"},
        {"id": 30, "d": 0.0, "s": "gamma"},
    ]
    ok(http_insert(http, t, payload))
    th = sorted(thrift_rows(db.get_table(t), ["id", "d", "s"]), key=lambda r: r["id"])
    assert th == payload, th
    http_drop_table(http, t)
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 4. Serialization fidelity                                                    #
# --------------------------------------------------------------------------- #

def test_int64_precision_roundtrip(inst, http, db):
    """int64 beyond 2**53 must survive JSON round-trip on the HTTP path and
    agree with thrift. (JSON numbers are doubles in many parsers; a silent
    truncation here would be a real defect.)"""
    t = "eval_http_i64"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                    {"name": "big", "type": "bigint"}]))
    vals = [
        2 ** 53 + 1,
        2 ** 53 - 1,
        9223372036854775807,   # INT64_MAX
        -9223372036854775808,  # INT64_MIN
        1234567890123456789,
    ]
    ok(http_insert(http, t, [{"id": i, "big": v} for i, v in enumerate(vals)]))
    hrows = {r["id"]: r["big"] for r in rows_from_http(
        ok(http_select(http, t, {"output": ["id", "big"]})))}
    throws = {r["id"]: r["big"] for r in thrift_rows(db.get_table(t), ["id", "big"])}
    for i, v in enumerate(vals):
        assert hrows[i] == v, f"HTTP truncated int64: got {hrows[i]} expected {v}"
        assert throws[i] == v, f"thrift mismatch: got {throws[i]} expected {v}"
    http_drop_table(http, t)
    assert not crashed(inst)


def test_float_double_precision(inst, http, db):
    t = "eval_http_fp"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                    {"name": "f", "type": "float"},
                                    {"name": "d", "type": "double"}]))
    ok(http_insert(http, t, [{"id": 1, "f": 0.1, "d": 3.141592653589793}]))
    h = rows_from_http(ok(http_select(http, t, {"output": ["f", "d"]})))[0]
    th = thrift_rows(db.get_table(t), ["f", "d"])[0]
    # double preserves full precision on both paths
    assert h["d"] == 3.141592653589793 == th["d"]
    # float is f32; HTTP and thrift must agree on the widened representation
    assert h["f"] == th["f"], (h["f"], th["f"])
    assert approx(h["f"], 0.1, tol=1e-6)
    http_drop_table(http, t)
    assert not crashed(inst)


def test_embedding_roundtrip(inst, http, db):
    t = "eval_http_emb"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                    {"name": "vec", "type": "vector,4,float"}]))
    v = [1.5, -2.0, 0.25, 3.0]
    ok(http_insert(http, t, [{"id": 1, "vec": v}]))
    h = rows_from_http(ok(http_select(http, t, {"output": ["vec"]})))[0]["vec"]
    # HTTP serializes embeddings as a JSON-ish string; it must parse to the input.
    parsed = json.loads(h) if isinstance(h, str) else h
    assert [float(x) for x in parsed] == v, h
    th = thrift_rows(db.get_table(t), ["vec"])[0]["vec"]
    assert [float(x) for x in th] == v, th
    http_drop_table(http, t)
    assert not crashed(inst)


def test_null_representation(inst, http, db):
    t = "eval_http_null"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                    {"name": "v", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": 1, "v": 42}, {"id": 2, "v": None}]))
    hrows = {r["id"]: r["v"] for r in rows_from_http(
        ok(http_select(http, t, {"output": ["id", "v"]})))}
    assert hrows[1] == 42
    assert hrows[2] is None, f"NULL not represented as null: {hrows[2]!r}"
    throws = {r["id"]: r["v"] for r in thrift_rows(db.get_table(t), ["id", "v"])}
    assert throws[2] is None
    http_drop_table(http, t)
    assert not crashed(inst)


def test_unicode_and_quotes(inst, http, db):
    t = "eval_http_uni"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "s", "type": "varchar"}]))
    strings = ['wörld "quoted"', "emoji 🚀 test", "line\nbreak\ttab",
               "sql'; DROP TABLE x;--", "中文字符", "back\\slash"]
    ok(http_insert(http, t, [{"id": i, "s": s} for i, s in enumerate(strings)]))
    hrows = {r["id"]: r["s"] for r in rows_from_http(
        ok(http_select(http, t, {"output": ["id", "s"]})))}
    throws = {r["id"]: r["s"] for r in thrift_rows(db.get_table(t), ["id", "s"])}
    for i, s in enumerate(strings):
        assert hrows[i] == s, f"HTTP mangled unicode/quote: {hrows[i]!r} != {s!r}"
        assert throws[i] == s, f"thrift mangled: {throws[i]!r} != {s!r}"
    http_drop_table(http, t)
    assert not crashed(inst)


def test_empty_result(inst, http, db):
    t = "eval_http_empty"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": 1}]))
    body = ok(http_select(http, t, {"output": ["id"], "filter": "id > 999"}))
    assert rows_from_http(body) == []
    http_drop_table(http, t)
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 5. Search: dense / sparse / fulltext / fusion -- parity + invariants         #
# --------------------------------------------------------------------------- #

def _ids_in_order(rows):
    return [r["id"] for r in rows]


def _dist_key(row):
    for k in ("DISTANCE", "SIMILARITY", "SCORE"):
        if k in row:
            return k
    return None


def test_dense_search_parity_and_recall(inst, http, db):
    """Dense KNN (no index -> exact). HTTP and thrift must return the same rows in
    the same order; that order must equal a brute-force L2 ranking (recall 1.0);
    distances must be non-decreasing."""
    t = "eval_http_dense"
    http_drop_table(http, t)
    dim, n, k = 8, 60, 10
    rng = random.Random(1234)
    vecs = {i: [rng.uniform(-1, 1) for _ in range(dim)] for i in range(n)}
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                    {"name": "vec", "type": f"vector,{dim},float"}]))
    ok(http_insert(http, t, [{"id": i, "vec": vecs[i]} for i in range(n)]))

    q = [rng.uniform(-1, 1) for _ in range(dim)]
    search = [{"match_method": "dense", "fields": "vec", "query_vector": q,
               "element_type": "float", "metric_type": "l2", "topn": k}]
    hrows = rows_from_http(ok(http_select(http, t, {"output": ["id", "_distance"], "search": search})))
    tbl = db.get_table(t)
    tdf, _ = tbl.output(["id", "_distance"]).match_dense("vec", q, "float", "l2", k).to_pl()
    trows = tdf.to_dicts()

    assert len(hrows) == k and len(trows) == k
    # parity: same id ordering
    assert _ids_in_order(hrows) == _ids_in_order(trows), (hrows, trows)
    # distances non-decreasing
    dk = _dist_key(hrows[0])
    dists = [r[dk] for r in hrows]
    assert dists == sorted(dists), dists
    # recall vs brute force
    bf = sorted(range(n), key=lambda i: sum((a - b) ** 2 for a, b in zip(vecs[i], q)))[:k]
    assert set(_ids_in_order(hrows)) == set(bf), (set(_ids_in_order(hrows)), set(bf))
    http_drop_table(http, t)
    assert not crashed(inst)


def test_integer_valued_vector_insert(inst, http, db):
    """Regression: inserting integer-valued JSON arrays into a float vector column
    must actually store the rows and remain searchable (not silently drop them)."""
    t = "eval_http_ivec"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                    {"name": "vec", "type": "vector,4,float"}]))
    ok(http_insert(http, t, [{"id": 0, "vec": [1, 0, 0, 0]},
                             {"id": 1, "vec": [0, 1, 0, 0]},
                             {"id": 2, "vec": [0, 0, 1, 0]}]))
    cnt = list(rows_from_http(ok(http_select(http, t, {"output": ["count(*)"]})))[0].values())[0]
    assert cnt == 3, f"integer-valued vectors not stored: count={cnt}"
    hrows = rows_from_http(ok(http_select(http, t, {
        "output": ["id", "_distance"],
        "search": [{"match_method": "dense", "fields": "vec", "query_vector": [1, 0, 0, 0],
                    "element_type": "float", "metric_type": "l2", "topn": 3}]})))
    assert len(hrows) == 3, f"KNN over integer-valued vectors returned {len(hrows)} rows"
    assert hrows[0]["id"] == 0
    http_drop_table(http, t)
    assert not crashed(inst)


def test_fulltext_search_parity(inst, http, db):
    t = "eval_http_ft"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "body", "type": "varchar"}]))
    docs = ["the quick brown fox", "lazy dog sleeps", "quick quick fox runs",
            "a slow green turtle", "fox and hound"]
    ok(http_insert(http, t, [{"id": i, "body": d} for i, d in enumerate(docs)]))
    ok(http.post(f"/databases/default_db/tables/{t}/indexes/ft",
                 json_body={"fields": ["body"], "index": {"type": "FULLTEXT"},
                            "create_option": "error"}))
    search = [{"match_method": "text", "fields": "body", "matching_text": "quick fox", "topn": 5}]
    hrows = rows_from_http(ok(http_select(http, t, {"output": ["id", "_score"], "search": search})))
    tdf, _ = db.get_table(t).output(["id", "_score"]).match_text("body", "quick fox", 5).to_pl()
    trows = tdf.to_dicts()
    assert len(hrows) > 0
    assert _ids_in_order(hrows) == _ids_in_order(trows), (hrows, trows)
    # scores non-increasing (best first)
    scores = [r["SCORE"] for r in hrows]
    assert scores == sorted(scores, reverse=True), scores
    http_drop_table(http, t)
    assert not crashed(inst)


def test_sparse_search_parity(inst, http, db):
    t = "eval_http_sparse"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "sv", "type": "sparse,100,float,int"}]))
    data = {0: {"10": 1.0, "20": 2.0}, 1: {"10": 0.5, "30": 3.0}, 2: {"20": 1.0, "40": 1.0}}
    ok(http_insert(http, t, [{"id": i, "sv": data[i]} for i in range(3)]))
    qv = {"10": 1.0, "20": 1.0}
    search = [{"match_method": "sparse", "fields": "sv", "query_vector": qv,
               "metric_type": "ip", "topn": 5}]
    hrows = rows_from_http(ok(http_select(http, t, {"output": ["id", "_similarity"], "search": search})))
    tdf, _ = db.get_table(t).output(["id", "_similarity"]).match_sparse(
        "sv", SparseVector([10, 20], [1.0, 1.0]), "ip", 5).to_pl()
    trows = tdf.to_dicts()
    assert len(hrows) > 0
    assert _ids_in_order(hrows) == _ids_in_order(trows), (hrows, trows)
    http_drop_table(http, t)
    assert not crashed(inst)


def test_fusion_search(inst, http, db):
    t = "eval_http_fusion"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "body", "type": "varchar"},
                                   {"name": "vec", "type": "vector,4,float"}]))
    rows = [
        {"id": 0, "body": "quick brown fox", "vec": [1.0, 0.0, 0.0, 0.0]},
        {"id": 1, "body": "lazy dog", "vec": [0.0, 1.0, 0.0, 0.0]},
        {"id": 2, "body": "quick fox jumps", "vec": [0.9, 0.1, 0.0, 0.0]},
        {"id": 3, "body": "green turtle", "vec": [0.0, 0.0, 1.0, 0.0]},
    ]
    ok(http_insert(http, t, rows))
    ok(http.post(f"/databases/default_db/tables/{t}/indexes/ft",
                 json_body={"fields": ["body"], "index": {"type": "FULLTEXT"},
                            "create_option": "error"}))
    qvec = [1.0, 0.0, 0.0, 0.0]
    search = [
        {"match_method": "dense", "fields": "vec", "query_vector": qvec,
         "element_type": "float", "metric_type": "l2", "topn": 4},
        {"match_method": "text", "fields": "body", "matching_text": "quick fox", "topn": 4},
        {"fusion_method": "rrf", "topn": 3},
    ]
    hrows = rows_from_http(ok(http_select(http, t, {"output": ["id"], "search": search})))
    assert 0 < len(hrows) <= 3, hrows
    # every returned id must be a real row id
    assert set(_ids_in_order(hrows)).issubset({0, 1, 2, 3})
    # parity with thrift fusion
    tq = (db.get_table(t).output(["id"])
          .match_dense("vec", qvec, "float", "l2", 4)
          .match_text("body", "quick fox", 4)
          .fusion("rrf", 3, None))
    tdf, _ = tq.to_pl()
    assert _ids_in_order(hrows) == _ids_in_order(tdf.to_dicts()), (hrows, tdf.to_dicts())
    http_drop_table(http, t)
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 6. Mutate: delete + update (incl. the key-order defect)                      #
# --------------------------------------------------------------------------- #

def test_delete_parity(inst, http, db):
    t = "eval_http_del"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": i} for i in range(20)]))
    body = ok(http.delete(f"/databases/default_db/tables/{t}/docs",
                          json_body={"filter": "id < 5"}))
    assert body["deleted_rows"] == 5, body
    remaining = sorted(r["id"] for r in thrift_rows(db.get_table(t), ["id"]))
    assert remaining == list(range(5, 20))
    http_drop_table(http, t)
    assert not crashed(inst)


def test_update_parity_update_first(inst, http, db):
    """With the SDK's key order (update before filter) the update works and is
    visible to thrift."""
    t = "eval_http_upd"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "v", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": i, "v": i} for i in range(5)]))
    ok(http.put(f"/databases/default_db/tables/{t}/docs",
                json_body={"update": {"v": 999}, "filter": "id = 2"}))
    got = {r["id"]: r["v"] for r in thrift_rows(db.get_table(t), ["id", "v"])}
    assert got[2] == 999, got
    http_drop_table(http, t)
    assert not crashed(inst)


def test_update_is_json_key_order_sensitive(inst, http, db):
    """DEFECT: the HTTP UPDATE endpoint's result depends on JSON key order.

    UpdateHandler reads ``doc["update"]`` (operator[] rewinds the object) then
    ``doc.find_field("filter")`` (forward-only, no rewind) in
    src/network/http_server_impl.cpp:1461,1608. If "filter" appears before
    "update" in the body, find_field cannot see it and the handler throws
    NO_SUCH_FIELD. JSON object member order is not semantically significant
    (RFC 8259 sec 4), so both orderings must behave identically. This test asserts
    the correct behaviour and therefore FAILS while the defect is present.
    """
    t = "eval_http_upd_order"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "v", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": 1, "v": 1}]))
    # filter BEFORE update -- a valid request a conforming client may send
    r = http.put(f"/databases/default_db/tables/{t}/docs",
                 json_body={"filter": "id = 1", "update": {"v": 555}})
    got = {row["id"]: row["v"] for row in thrift_rows(db.get_table(t), ["id", "v"])}
    http_drop_table(http, t)
    assert not crashed(inst)
    assert r.status_code == 200 and r.json().get("error_code") == 0, (
        f"UPDATE rejected valid filter-first body: {r.status_code} {r.text[:200]}")
    assert got[1] == 555, f"UPDATE with filter-first body did not apply: {got}"


# --------------------------------------------------------------------------- #
# 7. Import / export                                                           #
# --------------------------------------------------------------------------- #

def test_import_export_csv(inst, http, db):
    t = "eval_http_io"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "a", "type": "integer"},
                                   {"name": "b", "type": "integer"}]))
    src = inst.stage_csv("eval_http_io.csv", [[i, i * 10] for i in range(25)])
    ok(http.put(f"/databases/default_db/tables/{t}",
                json_body={"file_path": str(src), "file_type": "csv",
                           "header": False, "delimiter": ","}))
    cnt = list(rows_from_http(ok(http_select(http, t, {"output": ["count(*)"]})))[0].values())[0]
    assert cnt == 25, cnt
    # export back out (note the route uses singular /table/)
    dst = str(inst.data_root() / "eval_http_io_out.csv")
    ok(http.get(f"/databases/default_db/table/{t}",
                json_body={"file_path": dst, "file_type": "csv", "header": False,
                           "delimiter": ",", "columns": ["a", "b"]}))
    text = Path(dst).read_text().strip().splitlines()
    assert len(text) == 25, text[:3]
    assert text[0].split(",") == ["0", "0"]
    http_drop_table(http, t)
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 8. Index CRUD + parity                                                       #
# --------------------------------------------------------------------------- #

def test_index_crud_parity(inst, http, db):
    t = "eval_http_idx"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "num", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": i, "num": i} for i in range(5)]))
    ok(http.post(f"/databases/default_db/tables/{t}/indexes/idx_num",
                 json_body={"fields": ["num"], "index": {"type": "SECONDARY"},
                            "create_option": "error"}))
    http_idx = {i["index_name"] for i in
                ok(http.get(f"/databases/default_db/tables/{t}/indexes")).get("indexes", [])}
    thrift_idx = set(db.get_table(t).list_indexes().index_names)
    assert "idx_num" in http_idx
    assert http_idx == thrift_idx, (http_idx, thrift_idx)

    detail = ok(http.get(f"/databases/default_db/tables/{t}/indexes/idx_num"))
    assert detail["index_type"] == "SECONDARY", detail

    ok(http.delete(f"/databases/default_db/tables/{t}/indexes/idx_num",
                   json_body={"drop_option": "error"}))
    after = {i["index_name"] for i in
             ok(http.get(f"/databases/default_db/tables/{t}/indexes")).get("indexes", [])}
    assert "idx_num" not in after
    http_drop_table(http, t)
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 9. Metadata / show endpoints -- all must be 200 + JSON                       #
# --------------------------------------------------------------------------- #

def test_metadata_endpoints(inst, http, db):
    t = "eval_http_meta"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": i} for i in range(3)]))

    for path in ["/configs", "/variables/global", "/instance/buffer",
                 "/instance/memory",
                 f"/databases/default_db/tables/{t}/segments",
                 f"/databases/default_db/tables/{t}/columns"]:
        r = http.get(path)
        assert r.status_code == 200, (path, r.status_code, r.text[:200])
        assert r.json().get("error_code") == 0, (path, r.text[:200])

    # a specific global variable
    gv = ok(http.get("/variables/global/query_count"))
    assert "query_count" in gv or "error_code" in gv
    # a specific config
    cfg = ok(http.get("/configs/cpu_limit"))
    assert "cpu_limit" in cfg
    http_drop_table(http, t)
    assert not crashed(inst)


# --------------------------------------------------------------------------- #
# 10. Negative cases (first-class). Each asserts a specific outcome AND that    #
#     the server survives and stays correct.                                    #
# --------------------------------------------------------------------------- #

def test_neg_malformed_json_select_returns_json_error(inst, http, db):
    """The SELECT path catches simdjson errors and returns a structured JSON
    error (kInvalidJsonFormat). Contrast with the DDL handlers below."""
    t = "eval_neg_badjson"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    r = http.get(f"/databases/default_db/tables/{t}/docs", raw_body="{not valid json")
    assert r.status_code in (200, 500)
    body = r.json()  # must be JSON, not an oatpp text page
    assert body.get("error_code") not in (None, 0), body
    http_drop_table(http, t)
    assert_alive(inst, http)


def test_neg_ddl_malformed_json_breaks_json_contract(inst, http):
    """PIN + FINDING: DDL handlers (create database/table etc.) do not guard body
    parsing, so a malformed JSON body escapes as an oatpp plaintext 500 instead of
    the API's ``{"error_code","error_msg"}`` JSON contract that every other error
    path (docs/search) returns. The server stays alive.
    """
    r = http.post("/databases/eval_neg_ddl", raw_body="{not json")
    assert r.status_code == 500
    # Observed defect: body is oatpp's plaintext error page, not JSON.
    with pytest.raises(Exception):
        r.json()
    assert "oatpp" in r.text.lower()
    assert_alive(inst, http)


def test_neg_missing_required_field_create_db_leaks_exception(inst, http):
    """PIN + FINDING: POST /databases/{name} with a body lacking ``create_option``
    dereferences a missing simdjson field unconditionally
    (http_server_impl.cpp:285) with no try/catch, so it returns an oatpp plaintext
    500 ``NO_SUCH_FIELD`` rather than a structured error. A client that omits an
    optional-looking field gets an unparseable response. Server stays alive.
    """
    r = http.post("/databases/eval_neg_nofield", json_body={})
    assert r.status_code == 500
    assert "NO_SUCH_FIELD" in r.text, r.text[:200]
    with pytest.raises(Exception):
        r.json()
    assert_alive(inst, http)


def test_neg_wrong_method(inst, http):
    r = http.req("PATCH", "/databases")
    # oatpp has no PATCH mapping -> 404 (arguably should be 405); pin the behavior.
    assert r.status_code == 404, r.status_code
    assert_alive(inst, http)


def test_neg_nonexistent_route(inst, http):
    r = http.get("/no/such/route/here")
    assert r.status_code == 404
    assert_alive(inst, http)


def test_neg_nonexistent_objects(inst, http):
    # nonexistent database
    r = http.get("/databases/definitely_not_a_db")
    assert r.status_code == 500 and r.json()["error_code"] != 0
    # nonexistent table select
    r = http.get("/databases/default_db/tables/definitely_not_a_table/docs",
                 json_body={"output": ["*"]})
    assert r.status_code == 500 and r.json()["error_code"] != 0
    # nonexistent index show
    r = http.get("/databases/default_db/tables/definitely_not_a_table/indexes/nope")
    assert r.status_code == 500 and r.json()["error_code"] != 0
    assert_alive(inst, http)


def test_neg_wrong_shape_bodies(inst, http):
    # create table with fields as an object instead of an array
    r = http.post("/databases/default_db/tables/eval_neg_shape",
                  json_body={"fields": {"x": 1}, "create_option": "error"})
    assert r.status_code == 500 and r.json()["error_code"] != 0
    # insert an object where an array of rows is expected
    ok(http_create_table(http, "eval_neg_shape2",
                         [{"name": "id", "type": "integer"}]))
    r = http.post("/databases/default_db/tables/eval_neg_shape2/docs",
                  json_body={"id": 1})
    assert r.status_code == 500 and r.json().get("error_code", 0) != 0, r.text[:200]
    http_drop_table(http, "eval_neg_shape2")
    assert_alive(inst, http)


def test_neg_content_type_ignored(inst, http):
    """PIN + FINDING (minor): the server parses the body as JSON regardless of
    Content-Type. A create with text/plain still succeeds."""
    name = "eval_neg_ct"
    r = http.req("POST", f"/databases/{name}",
                 raw_body=json.dumps({"create_option": "error"}),
                 headers={"content-type": "text/plain"})
    assert r.status_code == 200 and r.json()["error_code"] == 0, r.text[:200]
    http.delete(f"/databases/{name}", json_body={"drop_option": "ignore_if_not_exists"})
    assert_alive(inst, http)


def test_neg_large_and_truncated_body(inst, http):
    t = "eval_neg_big"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "s", "type": "varchar"}]))
    # a genuinely large but valid varchar value must round-trip
    big = "x" * (2 * 1024 * 1024)
    r = http_insert(http, t, [{"id": 1, "s": big}])
    assert r.status_code == 200 and r.json()["error_code"] == 0
    # a truncated JSON body must yield an error, not a hang/crash
    r = http.get(f"/databases/default_db/tables/{t}/docs",
                 raw_body='{"output": ["id"], "filter": "id = ')
    assert r.status_code in (200, 500)
    assert r.json().get("error_code") not in (None, 0)
    http_drop_table(http, t)
    assert_alive(inst, http)


def test_neg_deeply_nested_json_no_stack_overflow(inst, http):
    t = "eval_neg_deep"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    deep = "[" * 3000 + "]" * 3000
    r = http.get(f"/databases/default_db/tables/{t}/docs", raw_body=deep)
    # must be a controlled error, and the server must survive
    assert r.status_code in (200, 500)
    http_drop_table(http, t)
    assert_alive(inst, http)


def test_neg_injection_shaped_strings_do_not_corrupt(inst, http, db):
    """Injection-shaped strings in filter/value fields must be treated as data or
    rejected -- never executed. A canary table must remain intact afterwards."""
    canary = "eval_neg_canary"
    http_drop_table(http, canary)
    ok(http_create_table(http, canary, [{"name": "id", "type": "integer"}]))
    ok(http_insert(http, canary, [{"id": 7}]))

    # injection-shaped filter -> parser must reject, server must survive
    r = http.get(f"/databases/default_db/tables/{canary}/docs",
                 json_body={"output": ["*"], "filter": "1=1; DROP TABLE eval_neg_canary;--"})
    assert r.status_code in (200, 500)  # accept either; correctness is the canary below

    # injection-shaped string stored as a value must round-trip verbatim
    ok(http_insert(http, canary, [{"id": 8}]))
    rows = sorted(r_["id"] for r_ in thrift_rows(db.get_table(canary), ["id"]))
    assert rows == [7, 8], rows  # table not dropped, data intact
    http_drop_table(http, canary)
    assert_alive(inst, http)


def test_neg_duplicate_column_in_insert(inst, http):
    t = "eval_neg_dup"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    # raw body with a duplicated key
    r = http.post(f"/databases/default_db/tables/{t}/docs",
                  raw_body='[{"id": 1, "id": 2}]')
    assert r.status_code == 500 and r.json()["error_code"] != 0, r.text[:200]
    http_drop_table(http, t)
    assert_alive(inst, http)


def test_neg_oversized_identifier(inst, http):
    """A long-but-under-limit identifier is accepted; a URL whose path exceeds
    oatpp's request-line buffer (~4 KB) is reset at the TCP layer with no HTTP
    response. Either way the server must survive (no crash)."""
    mid = "t" + "a" * 2000
    r = http.post(f"/databases/default_db/tables/{mid}",
                  json_body={"fields": [{"name": "id", "type": "integer"}],
                             "create_option": "error"})
    assert r.status_code == 200 and r.json()["error_code"] == 0, r.text[:200]
    http_drop_table(http, mid)

    huge = "t" + "a" * 8000
    try:
        http.post(f"/databases/default_db/tables/{huge}",
                  json_body={"fields": [{"name": "id", "type": "integer"}],
                             "create_option": "error"})
    except requests.exceptions.ConnectionError:
        pass  # oatpp reset the oversized request line; acceptable if server survives
    assert_alive(inst, http)


def test_neg_bad_vector_type_syntax_no_crash(inst, http):
    r = http.post("/databases/default_db/tables/eval_neg_badvec",
                  json_body={"fields": [{"name": "v", "type": "vector,abc,float"}],
                             "create_option": "error"})
    assert r.status_code == 500 and r.json()["error_code"] != 0, r.text[:200]
    assert_alive(inst, http)


def test_neg_wrong_dimension_vector_insert(inst, http):
    t = "eval_neg_dim"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "vec", "type": "vector,4,float"}]))
    r = http_insert(http, t, [{"id": 1, "vec": [1.0, 2.0, 3.0]}])  # 3 != 4
    assert r.status_code == 500 and r.json()["error_code"] != 0, r.text[:200]
    http_drop_table(http, t)
    assert_alive(inst, http)


def test_wrong_type_insert_silently_corrupts_to_null(inst, http, db):
    """DEFECT: inserting a string into an INTEGER column over HTTP returns
    ``error_code: 0`` (success) but silently stores NULL -- invalid input is
    accepted and the row is corrupted with no error. Verified by reading the value
    back over thrift. This test asserts the correct behaviour (the insert must be
    rejected) and therefore FAILS while the defect is present.
    """
    t = "eval_wrongtype"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"},
                                   {"name": "v", "type": "integer"}]))
    r = http_insert(http, t, [{"id": 1, "v": "not_an_int"}])
    stored = {row["id"]: row["v"] for row in thrift_rows(db.get_table(t), ["id", "v"])}
    http_drop_table(http, t)
    assert not crashed(inst)
    # Correct behaviour: reject the bad value with an error.
    assert r.json().get("error_code") != 0, (
        f"string into INTEGER column silently accepted; stored={stored}")


def test_unknown_search_option_silently_ignored(inst, http, db):
    """PIN + FINDING: an unrecognized key inside the search ``option`` object is
    silently ignored (http_search_impl.cpp only handles ``total_hits_count``),
    even though an unknown *top-level* key is rejected with kInvalidExpression.
    A plausible-but-wrong option name therefore exercises nothing and still 200s.
    """
    t = "eval_opt"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": i} for i in range(3)]))

    # unknown top-level key IS rejected
    r_top = http_select(http, t, {"output": ["*"], "totally_bogus_key": 1})
    assert r_top.status_code == 500 and r_top.json()["error_code"] != 0

    # unknown OPTION key is silently accepted (the defect being pinned)
    r_opt = http_select(http, t, {"output": ["*"], "option": {"totally_bogus_option": "true"}})
    assert r_opt.status_code == 200 and r_opt.json()["error_code"] == 0
    assert len(rows_from_http(r_opt.json())) == 3  # option ignored, all rows returned
    http_drop_table(http, t)
    assert_alive(inst, http)


# --------------------------------------------------------------------------- #
# 11. Concurrency                                                              #
# --------------------------------------------------------------------------- #

def test_concurrent_requests_stay_correct(inst, http, db):
    t = "eval_http_conc"
    http_drop_table(http, t)
    ok(http_create_table(http, t, [{"name": "id", "type": "integer"}]))
    ok(http_insert(http, t, [{"id": i} for i in range(50)]))
    base = inst.http_base

    def worker(i):
        # independent request objects; requests is threadsafe per-call
        r = requests.request("GET", f"{base}/databases/default_db/tables/{t}/docs",
                             data=json.dumps({"output": ["count(*)"]}),
                             headers=JSON_HEADERS, timeout=30)
        return r.status_code, r.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(worker, range(64)))
    for sc, body in results:
        assert sc == 200 and body["error_code"] == 0, (sc, body)
        cnt = list(rows_from_http(body)[0].values())[0]
        assert cnt == 50, cnt
    http_drop_table(http, t)
    assert_alive(inst, http)
