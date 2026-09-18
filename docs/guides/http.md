# HTTP guide

The same operations as the [Python guide](python.md), from any language, over the
REST API on port 23820. The bodies below are the ones in `example/http/*.sh`, which
run against the server as written; the complete endpoint list is in the
[HTTP API reference](../references/http_api_reference.mdx).

Conventions that trip people up:

- A **read is a `GET` with a JSON body**. Some HTTP clients refuse to send one;
  `curl --request GET --data '...'` does, and so do `requests` and `fetch` when you set
  the method explicitly.
- Every response is JSON with an `error_code`. `0` is success. A non-zero code comes
  with an `error_msg`.
- Send `content-type: application/json`.

The examples use a shell variable for brevity:

```sh
API=http://127.0.0.1:23820/databases/default_db
H='content-type: application/json'
```

## Tables

```sh
# create (ignore_if_exists | error)
curl -X POST "$API/tables/articles" -H "$H" -d '{
  "create_option": "ignore_if_exists",
  "fields": [
    {"name": "id",       "type": "integer", "constraints": ["PRIMARY KEY"]},
    {"name": "title",    "type": "varchar"},
    {"name": "body",     "type": "varchar"},
    {"name": "category", "type": "varchar"},
    {"name": "year",     "type": "integer"},
    {"name": "vec",      "type": "vector,4,float"}
  ]}'

# list, show, drop
curl -X GET    "$API/tables"
curl -X GET    "$API/tables/articles"
curl -X DELETE "$API/tables/articles" -H "$H" -d '{"drop_option": "ignore_if_not_exists"}'
```

The type strings are the ones the Python guide lists. A dense vector of 384 dimensions
is `"vector,384,float"`; a sparse vector is `"sparse,30000,float,int"`.

## Insert

```sh
curl -X POST "$API/tables/articles/docs" -H "$H" -d '[
  {"id": 1, "title": "Bloom filters", "body": "a bloom filter tests set membership",
   "category": "storage", "year": 2024, "vec": [1.0, 1.2, 0.8, 0.9]},
  {"id": 2, "title": "HNSW", "body": "hnsw builds a navigable small world graph",
   "category": "index", "year": 2023, "vec": [4.0, 4.2, 4.3, 4.5]}
]'
```

Sparse vectors are objects keyed by index: `{"10": 1.1, "20": 2.2}`. Tensors are lists
of vectors. The same input-validation caveats as in Python apply: a wrong-typed value
is stored as NULL with `error_code: 0` ([known issue 8](../known-issues.md)).

## Indexes

```sh
# HNSW on the vector column. Over HTTP the key is uppercase "M"; the Python SDK wants "m".
curl -X POST "$API/tables/articles/indexes/vec_idx" -H "$H" -d '{
  "fields": ["vec"],
  "index": {"type": "hnsw", "M": "16", "ef_construction": "200", "metric": "cosine"},
  "create_option": "ignore_if_exists"}'

# full text
curl -X POST "$API/tables/articles/indexes/body_idx" -H "$H" -d '{
  "fields": ["body"],
  "index": {"type": "fulltext", "analyzer": "standard"},
  "create_option": "ignore_if_exists"}'

# list and drop
curl -X GET    "$API/tables/articles/indexes"
curl -X DELETE "$API/tables/articles/indexes/vec_idx" -H "$H" -d '{"drop_option": "ignore_if_not_exists"}'
```

## Read

```sh
# rows, with a filter, sort and paging
curl -X GET "$API/tables/articles/docs" -H "$H" -d '{
  "output": ["id", "title", "year"],
  "filter": "year >= 2023",
  "sort":   [{"year": "desc"}, {"id": "asc"}],
  "offset": "0",
  "limit":  "20"}'

# count
curl -X GET "$API/tables/articles/docs" -H "$H" -d '{"output": ["count(*)"]}'
```

## Search

Every search is the same `GET .../docs` with a `search` array. Each element is one
retrieval arm; a final element with `fusion_method` merges them.

```sh
# vector search
curl -X GET "$API/tables/articles/docs" -H "$H" -d '{
  "output": ["id", "title", "similarity()"],
  "search": [
    {"match_method": "dense", "fields": "vec", "query_vector": [3.0, 2.8, 2.7, 3.1],
     "element_type": "float", "metric_type": "cosine", "topn": 10}
  ]}'

# full-text search, with the matching terms highlighted
curl -X GET "$API/tables/articles/docs" -H "$H" -d '{
  "output": ["id", "title", "score()"],
  "highlight": ["body"],
  "search": [
    {"match_method": "text", "fields": "body", "matching_text": "graph OR filter", "topn": 10}
  ]}'

# hybrid: both arms, a structured filter, fused by reciprocal rank
curl -X GET "$API/tables/articles/docs" -H "$H" -d '{
  "output": ["id", "title", "category", "year", "score()"],
  "search": [
    {"match_method": "text",  "fields": "body", "matching_text": "graph", "topn": 20},
    {"match_method": "dense", "fields": "vec",  "query_vector": [3.0, 2.8, 2.7, 3.1],
     "element_type": "float", "metric_type": "cosine", "topn": 20},
    {"fusion_method": "rrf", "topn": 5}
  ],
  "filter": "year >= 2023"}'
```

Search arm fields:

| key | meaning |
|---|---|
| `match_method` | `dense`, `sparse`, `text` or `tensor` |
| `fields` | the column (dense, sparse, tensor) or comma-separated columns with optional `^weight` (text) |
| `query_vector`, `matching_text`, `query_tensor` | the query, per method |
| `element_type`, `metric_type` | for dense: the vector element type and `cosine`, `ip` or `l2` |
| `topn` | candidates this arm contributes |
| `params` | per-method options, e.g. `{"ef": "128"}` for dense, `{"operator": "or"}` for text |

Fusion elements take `fusion_method` (`rrf`, `weighted_sum`, `max`, `match_tensor`),
`topn`, and an optional `params` object with the same keys as the Python `fusion()`
call. `filter` sits beside `search` and applies to every arm inside the search, not
after it.

Output columns: `score()` for fused and full-text results, `similarity()` or
`distance()` for a dense-only query, `row_id()` for the internal row id.

## Update and delete

```sh
curl -X PUT "$API/tables/articles/docs" -H "$H" -d '{
  "update": {"category": "indexes"},
  "filter": "id = 2"}'

curl -X DELETE "$API/tables/articles/docs" -H "$H" -d '{"filter": "year < 2020"}'
```

## Everything else

Databases (`POST /databases/<name>`, `GET /databases`), snapshots, `SHOW`-style
metadata (`GET /tables/<t>/segments`, `.../columns`), server metrics and cluster
administration are all in the [reference](../references/http_api_reference.mdx). The
shell scripts under [`example/http/`](../../example/http/) exercise most of them
end to end.
