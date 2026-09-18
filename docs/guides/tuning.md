# Tuning

The knobs that change speed, recall or memory, with the measured effect of each. All
numbers come from [BENCHMARKS.md](../apple_silicon/BENCHMARKS.md) and the
[evaluation](../apple_silicon/EVALUATION.md) unless a row says otherwise; SIFT1M means
1,000,000 vectors of 128 dimensions on an M4 Mac mini with 16 GB.

## Start from the defaults

| What | Default that works | Change it when |
|---|---|---|
| HNSW `m` | 16 is the engine default; the benchmarks use 32 | recall at your `ef` is short of target and memory allows |
| HNSW `ef_construction` | 50 is the engine default; set 200, as the benchmarks do | you can spend build time to buy recall at every query later |
| query `ef` | equals `topn` unless set; use 64 to 128 | recall matters more than latency; raise it per query |
| metric | `cosine` for text embeddings | the model card says otherwise |
| insert batch | a few thousand rows per `insert()` | never smaller for bulk loads |
| `wal_flush` | `full_fsync` | bulk loading throwaway data: `no_sync`, then switch back |
| `buffer_manager_size` | 4 GB | a machine with more memory and an index that does not fit |

## The HNSW knobs, measured

Build cost and recall at SIFT1M, `m=32`, 10 build threads:

| `ef_construction` | build | recall@10 at `ef`=32 | 64 | 128 | 256 |
|---:|---:|---:|---:|---:|---:|
| 200 | 35.3 s | 0.9410 | 0.9822 | 0.9959 | 0.9993 |
| 235 | 41.4 s | 0.9442 | 0.9835 | 0.9964 | 0.9993 |

Read across a row for what the query-time `ef` buys: from 32 to 128 recall climbs
from 0.94 to 0.996, and each step costs roughly proportional query time because `ef`
is the number of candidates the search keeps in hand. Read down a column for what
build-time effort buys: 17% more build time for a few tenths of a point of recall at
low `ef`, and nothing once `ef` saturates recall near 0.999.

Rules that fall out of that:

- **Pick `ef` per query, not per index.** A search that feeds a reranker can run at
  `ef=32`; one that shows results directly wants 128 or more. The SDK takes it as
  `match_dense(..., {"ef": "128"})`.
- **Spend `ef_construction` when queries outnumber builds**, which is almost always.
  Build once at 200 or above; it costs seconds at build time and nothing per query.
- **`m` trades memory for graph quality.** At `m=32` the graph is about 256 bytes per
  vector on top of the vector itself; at 16 it is half that and recall at a given `ef`
  is a little lower. Both are fine; 32 is what the published numbers use.

## Data shape matters more than any knob

Uniform random vectors are the worst case for a graph index because they have no
structure to navigate. Real embeddings cluster. On the same machine, 200,000 vectors
of 768 dimensions:

| data | build | vectors/s | QPS | p50 |
|---|---:|---:|---:|---:|
| clustered (embedding-like) | 40.2 s | 4,977 | 4,996 | 686 µs |
| uniform random | 132.1 s | 1,514 | 1,536 | 1,945 µs |

If a benchmark on synthetic uniform data looks 3× slower than the README, that is why.
Real embeddings land between the clustered row and the SIFT numbers.

## Dimension

Cost scales with dimension. Same machine, clustered synthetic data:

| dimension | build vectors/s | QPS | p50 |
|---:|---:|---:|---:|
| 128 (SIFT) | 28,300 | 24,700 | 209 µs |
| 768 | 4,977 | 4,996 | 686 µs |
| 1536 | 3,055 | 3,215 | 1,091 µs |

A 384-dimension model such as `bge-small` is several times cheaper than a
1,536-dimension one for most retrieval tasks; check whether the extra dimensions buy
recall on your data before paying for them everywhere.

## Memory

An HNSW index keeps the float32 vectors in memory plus the graph. Roughly:

```
bytes ≈ N × (4 × dimension + 256)      at m=32
```

| index | memory |
|---|---|
| 1M × 384 | 1.8 GB |
| 1M × 768 | 3.3 GB |
| 1M × 1536 | 6.4 GB |
| 3M × 768 | 10 GB |

Leave room for the buffer manager (4 GB by default), the in-memory index segments
(`memindex_memory_quota`, 1 GB) and the OS. A 16 GB Mac comfortably serves about three
million 768-dimension vectors.

## Threads and cores

Index building parallelizes across cores; the published build used all 10 cores of an
M4 (4 performance plus 6 efficiency). The efficiency cores are slower per thread, which
is part of why the M4 mini's matched-recall speedup over FAISS (1.40×) is below the
M3 Pro's (1.52× with 6 performance cores). The server sizes its own pools from the
core count; the benchmark harness takes the count explicitly (`make bench
PARTICIPANTS=8`).

Query throughput in the tables is in-process with 12 concurrent threads. A server in
front of the engine adds serialization; expect fewer queries per second over HTTP than
the harness reports, and measure your own path.

## Filters

A filter inside a search (`.filter("year >= 2024")` chained onto `match_*`) is
evaluated as part of the search, so a selective filter does not hollow out the
result. For equality and range filters on scalar columns, a Secondary index makes the
filter cheap:

```python
table.create_index("year_idx", IndexInfo("year", IndexType.Secondary), ConflictType.Error)
```

Before adding one, read [known issue 1](../known-issues.md): a predicate that folds to
always-true or always-false on a Secondary-indexed integer column (for example
`v <= 127` on an `int8`) currently crashes the server.

## Full-text search

- **Analyzer** is the biggest lever. `standard` stems English; `standard-german` and
  the other language stemmers exist; `chinese`, `japanese`, `korean`, `rag` and `ik`
  are dictionary-based; `ngram` tolerates typos and partial matches at the cost of
  index size; `keyword` matches whole values exactly.
- **Field weights**: `match_text("title^2,body", ...)` counts a title hit twice.
- **BM25 parameters** `bm25_param_k1` (term-frequency saturation, default 1.2) and
  `bm25_param_b` (length normalization, default 0.75) go in the options dict. Lower
  `b` if long documents are being unfairly penalized.
- **Candidates per arm**: in a hybrid query, give each arm two to four times the
  final `topn`; fusion can only promote what an arm returned.

## Writes

- **Batch inserts.** Thousands of rows per `insert()`. The in-memory index segment is
  dumped every 65,536 rows (`mem_index_capacity`); very small batches waste that
  machinery.
- **Durability costs per commit batch**: about 2.3 ms at `full_fsync`, 28 µs at
  `fsync`, 1.5 µs at `no_sync`. A single serial writer sees 338 commits/s against
  1,888; many concurrent writers share one sync per batch. Bulk-load with
  `run_server.sh start --wal-flush no_sync`, then restart with the default.
- **Compaction after churn.** Updates and deletes leave dead row versions until the
  periodic compaction (`compact_interval`, 120 s). After a large update pass call
  `table.compact()`, both for space and because of
  [known issue 6](../known-issues.md).

## What is not tunable yet

Hand-written NEON kernels (today the x86 intrinsics are lowered by SIMDe), batched
distance through Accelerate, and per-chip thread defaults are on the
[roadmap](../apple_silicon/ROADMAP.md) as the next performance levers. The profile
that says where the time goes is in
[SEARCHLAYER_DECOMPOSITION.md](../apple_silicon/SEARCHLAYER_DECOMPOSITION.md).
