# Python guide

Everything you do with Infinity from Python, in the order you will do it. Every call
here is one the demo or the test suites exercise against a natively built server; the
full parameter list for each is in the
[SDK reference](../references/pysdk_api_reference.md).

## Install and connect

```sh
pip install infinity-sdk        # or, inside the repository: make sdk
```

The distribution is `infinity-sdk`; the module is `infinity`.

```python
import infinity
from infinity.common import ConflictType, NetworkAddress

conn = infinity.connect(NetworkAddress("127.0.0.1", 23817))
db = conn.get_database("default_db")
```

`connect()` takes a `NetworkAddress`, never a string. Passing `"127.0.0.1:23817"`
raises `INVALID_SERVER_ADDRESS`, which looks exactly like a server that is down. With
no argument it connects to `127.0.0.1:23817`, which is the default port. There is no
embedded mode in this port: `connect("/some/directory")` does not work, and the
commented-out lines in `example/*.py` that suggest it are dead.

Every database has a `default_db`. `conn.create_database("mine")` and
`conn.get_database("mine")` work the way you expect. Call `conn.disconnect()` when
you are done.

## Create a table

```python
db.drop_table("articles", ConflictType.Ignore)   # ok if it does not exist

table = db.create_table("articles", {
    "id":       {"type": "integer", "constraints": ["primary key"]},
    "title":    {"type": "varchar"},
    "body":     {"type": "varchar"},
    "category": {"type": "varchar"},
    "year":     {"type": "integer"},
    "vec":      {"type": "vector,384,float"},
}, ConflictType.Error)                            # error if it already exists
```

Column types you will use:

| Type string | Holds |
|---|---|
| `integer` (or `int32`), `int64`, `int16`, `int8` | integers of that width |
| `float` (or `float32`), `double`, `float16`, `bfloat16` | floats of that width |
| `varchar` | text of any length |
| `bool` | true or false |
| `vector,N,float` | a dense embedding of N float32 values; `float16`, `bfloat16`, `int8` and `uint8` elements also exist |
| `sparse,N,float,int` | a sparse vector over a vocabulary of size N, with float values and int indices |
| `tensor,N,float` | any number of N-dimensional vectors per row |
| `array,varchar` | an array of the element type |

Dates, times and the quantized vector element types are in the
[reference](../references/pysdk_api_reference.md#create_table).

`ConflictType` is `Error`, `Ignore` or `Replace` and appears on most create and drop
calls. `db.list_tables()` and `table.show_columns()` inspect what exists.

## Insert

```python
table.insert([
    {"id": 1, "title": "Bloom filters", "body": "...", "category": "storage", "year": 2024, "vec": [0.01, ...]},
    {"id": 2, "title": "HNSW", "body": "...", "category": "index", "year": 2023, "vec": [0.02, ...]},
])
```

`insert` takes a dict or a list of dicts. Batches of a few thousand rows per call are a
good default; the server builds the memory index as rows arrive and dumps it to disk
every 65,536 rows (`mem_index_capacity`).

Two things to know before you load real data:

- **Wrong-typed values are not rejected.** A string in an integer column is stored as
  NULL and the call reports success ([known issue 8](../known-issues.md)). Validate
  types on your side.
- **A NaN inside a vector is stored and will outrank real neighbours** in L2 search
  ([known issue 11](../known-issues.md)). Check `numpy.isfinite` before inserting.

For bulk loads, `import_data` reads a file **the server can see**; the path is resolved
by the server process, not the client:

```python
table.import_data("/absolute/path/articles.jsonl", {"file_type": "jsonl"})
table.import_data("/absolute/path/articles.csv",   {"file_type": "csv", "header": True, "delimiter": ","})
```

The SDK accepts `csv`, `json`, `jsonl`, `fvecs`, `csr` and `bvecs`. Parquet appears in
the upstream reference but the SDK's file-type mapping does not include it and
raises `IMPORT_FILE_FORMAT_ERROR` before reaching the server.

## Index

Indexes are created per column and can be added before or after the data.

```python
from infinity.index import IndexInfo, IndexType

# Vector index. Parameter values are strings, and the key is lowercase "m".
table.create_index("vec_idx", IndexInfo("vec", IndexType.Hnsw, {
    "m": "16",
    "ef_construction": "200",
    "metric": "cosine",          # cosine | ip | l2
}), ConflictType.Error)

# Full-text index. The analyzer defaults to "standard" (lowercase + English stemming).
table.create_index("title_idx", IndexInfo("title", IndexType.FullText), ConflictType.Error)
table.create_index("body_idx",  IndexInfo("body",  IndexType.FullText, {"ANALYZER": "standard"}), ConflictType.Error)

# Secondary index: speeds up equality and range filters on a scalar column.
table.create_index("year_idx", IndexInfo("year", IndexType.Secondary), ConflictType.Error)
```

Full-text analyzers: `standard` (lowercase plus English stemming; `standard-german`
and so on for other stemmers), `chinese`, `traditional`, `japanese`, `korean`, `rag`
and `ik` (dictionary-backed, which is what the `resource` submodule is for), `ngram`,
and `keyword` for exact-match columns. The full list is in the
[SDK reference](../references/pysdk_api_reference.md#create_index).

`table.list_indexes()`, `table.show_index("vec_idx")` and
`table.drop_index("vec_idx", ConflictType.Ignore)` manage them. Read
[known issue 1](../known-issues.md) before putting a Secondary index on a column you
filter with `<=` or `>=` at the type's boundaries.

## Query

Every query is a chain that starts with `output()` and ends with one of `to_pl()`,
`to_df()` or `to_arrow()`. Those three return a **tuple**: the result (a Polars
DataFrame, a pandas DataFrame or an Arrow table) and an extra-results dict that is
usually `None`.

```python
rows, _ = table.output(["id", "title"]).filter("year >= 2024").to_pl()
rows, _ = table.output(["*"]).limit(20).offset(40).to_pl()
rows, _ = table.output(["count(*)"]).to_pl()
rows, _ = table.output(["category", "count(*)"]).group_by("category").to_pl()

from infinity.common import SortType
rows, _ = table.output(["id", "year"]).sort([["year", SortType.Desc], ["id", SortType.Asc]]).to_pl()
```

Filters are SQL expressions over the columns: `=`, `!=`, `<`, `<=`, `>`, `>=`,
`in (...)`, `not in (...)`, `and`, `or`, `not`, arithmetic and functions. Strings take
single quotes: `"category = 'storage' and year >= 2024"`.

### Vector search

```python
query_vec = embed("how do I keep the index build fast")   # your model, same as at insert time

rows, _ = (table.output(["id", "title", "_similarity"])
                .match_dense("vec", query_vec, "float", "cosine", 10)
                .to_pl())
```

`match_dense(column, vector, element_type, metric, topn)`. The metric should match the
index; `cosine` and `ip` expose the score as `_similarity` (higher is better), `l2` as
`_distance` (lower is better). Pass `ef_search` through the options argument to spend
more effort per query: `match_dense("vec", q, "float", "cosine", 10, {"ef": "128"})`.

### Full-text search

```python
rows, _ = (table.output(["id", "title", "score()"])
                .match_text("title^2,body", "hnsw prefetch", 10)
                .to_pl())
```

`match_text(fields, text, topn, options)`. `fields` is a comma-separated list, and
`title^2` weights title matches twice as heavily as body matches. The query text
understands `AND`, `OR`, `NOT`, quoted phrases, `field:term`, `term^2` boosts and
`"a phrase"~3` proximity. To search literal text that happens to contain those
characters (a part number, a URL), pass `{"operator": "or"}` or `{"operator": "and"}`
and the syntax is switched off. BM25 parameters go in the same dict as
`bm25_param_k1` and `bm25_param_b`.

### Hybrid search

Chain more than one `match_*` and finish with `fusion`:

```python
rows, _ = (table.output(["id", "title", "category", "year", "score()"])
                .match_text("title^2,body", question, 20)
                .match_dense("vec", embed(question), "float", "cosine", 20)
                .filter("category = 'storage' AND year >= 2024")
                .fusion(method="rrf", topn=5)
                .to_pl())
```

`fusion(method, topn, params)`:

| method | what it does | params |
|---|---|---|
| `rrf` | merges by rank; needs no tuning | `{"rank_constant": 60}` |
| `weighted_sum` | normalizes each arm's scores and sums them with weights | `{"weights": "1,2", "normalize": "minmax"}` |
| `match_tensor` | reranks with a tensor column (late interaction) | `{"field": "tensor", "query_tensor": [[...]], "element_type": "float"}` |

The `filter` applies to every arm and is evaluated inside the search, so `topn=5`
really returns the five best matching rows, not whatever survived a post-filter.
Ask for more candidates per arm (here 20) than you want fused results (5).

`score()` is the column for fused and full-text results. A dense-only query rejects
`score()`; use `_similarity` or `_distance` there.

### Sparse vectors and tensors

```python
from infinity.common import SparseVector
rows, _ = (table.output(["id", "_similarity"])
                .match_sparse("sparse_col", SparseVector([12, 340, 9001], [0.5, 1.2, 0.3]), "ip", 10)
                .to_pl())
```

Tensor columns are searched with `match_tensor` or used as the `match_tensor` fusion
reranker; the upstream [search guide](search_guide.md) covers both.

## Update and delete

```python
table.update("id = 2", {"category": "indexes", "year": 2024})
table.delete("year < 2020")
table.delete()          # every row
```

Updates are versioned: the old row stays on disk as a dead version until the periodic
compaction runs. A table that has been updated tens of thousands of times before a
compaction can return wrong `sum()` and `max()` results ([known issue 6](../known-issues.md));
`count(*)` and plain scans are unaffected. `table.compact()` forces a compaction.

## Snapshots

```python
db.create_table_snapshot("articles_2026_09", "articles")
db.restore_table_snapshot("articles_2026_09")
conn.list_snapshots()
conn.drop_snapshot("articles_2026_09")
```

Snapshots are the backup mechanism. Two open defects matter here: a restored table's
**full-text index returns nothing** until it is rebuilt, and snapshot names are used as
file paths without sanitization, so only ever pass a plain name
([known issues 4 and 7](../known-issues.md)). Database and system snapshots exist too
(`conn.create_database_snapshot`, `conn.create_system_snapshot`).

## Errors

Failures raise `infinity.InfinityException` with `error_code` and `error_msg`.
Calls that return a status object instead expose the same two fields; check
`error_code == 0`. Note that some bad inputs are accepted silently rather than raising
(see the insert section), which is the opposite of what a database should do and is
tracked as a known issue.

## The five things that cost people the most time

1. `infinity.connect()` wants a `NetworkAddress`, not a string.
2. `to_pl()`, `to_df()` and `to_arrow()` return a tuple. Unpack it.
3. Index parameter values are strings (`"m": "16"`, not `"m": 16`), and the HNSW key
   is lowercase `m` over the SDK.
4. `score()` needs a full-text, tensor or fusion arm; a dense-only query exposes
   `_similarity` or `_distance`.
5. `import_data` and `export_data` paths are resolved by the **server**.

## Next

The [RAG tutorial](rag.md) puts this together with an embedding model, and the
[demo](../../demo/README.md) is 150 lines of working code you can copy from.
