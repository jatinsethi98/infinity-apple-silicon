# Concepts

Vector search, full-text search and hybrid retrieval, explained for someone who has
built ordinary databases but not this kind. Nothing here is specific to Infinity until
the last sections; the vocabulary is what every vector database uses.

## The problem

A normal database finds rows by exact values: `WHERE title = 'Bloom filters'`. A
full-text index loosens that to words: `title CONTAINS 'bloom'`. Neither can answer
"find me the article that explains why my search misses documents that clearly answer
the question" when the article never uses the words *miss*, *clearly* or *question*.
That needs search by **meaning**, and search by meaning is what vectors give you.

## Embeddings

An **embedding model** turns a piece of text (or an image, or audio) into a list of
numbers, typically 384 to 1,536 of them. The list is called a **vector** or an
**embedding**, and the number of entries is its **dimension**. The model is trained so
that texts with similar meaning land close together: "how do I speed up my index build"
and "reducing HNSW construction time" produce nearby vectors even though they share no
words.

Infinity does not produce embeddings. You run a model yourself and hand the numbers to
the database. The demo uses [fastembed](https://github.com/qdrant/fastembed) with a
384-dimension model that runs on the CPU without PyTorch; OpenAI, Cohere, Voyage and
sentence-transformers models all work the same way. Two rules of thumb:

- Use **one model** per column. Vectors from different models are not comparable.
- Some models expect a prefix on queries (the demo's model wants
  `"Represent this sentence for searching relevant passages: "` before a question).
  Read the model card; skipping it quietly costs recall.

## Similarity metrics

"Close together" needs a definition. Three are common, and you choose one when you
create an index and again when you query.

| Metric | Meaning | Use it when |
|---|---|---|
| `cosine` | the angle between two vectors, ignoring their length | the default for text embeddings |
| `ip` (inner product) | the dot product; larger is more similar | your vectors are already unit-length, where it equals cosine and is cheaper |
| `l2` | Euclidean distance; smaller is more similar | the model documentation says so, or the vectors are not embeddings |

Most embedding models emit unit-length vectors, in which case `cosine` and `ip` rank
identically.

## Exact search versus approximate search

To find the 10 nearest vectors to a query, the exact answer requires comparing the
query against every stored vector: a million vectors of 768 dimensions is 768 million
multiplications per query. That is fine for ten thousand rows and hopeless for ten
million. An **approximate nearest neighbour** (ANN) index answers the same question by
looking at a small fraction of the vectors and accepting that it will occasionally miss
one of the true top 10. "Occasionally" is measured, not hoped for; see recall below.

## HNSW

**Hierarchical Navigable Small World** is the ANN index Infinity uses for dense
vectors and the one this repository optimized. Picture every vector as a node with
edges to a few of its nearest neighbours, stacked in layers where the top layer is
sparse and the bottom layer holds everything. A search starts at the top, greedily walks
toward the query, drops a layer, and repeats until it reaches the bottom with a list of
candidates. Building the index means inserting every vector by running that same search
and wiring the newcomer to what it found.

Three knobs control the whole thing:

| Knob | Set when | Raising it | Typical |
|---|---|---|---|
| `m` | index creation | more edges per node: better recall, more memory, slower build | 16 to 32 |
| `ef_construction` | index creation | more candidates considered per insert: better recall, slower build, no query cost | 200 |
| `ef_search` | each query | more candidates considered per query: better recall, slower query | 64 to 256 |

Infinity's benchmark table uses `m=32`, `ef_construction=200` and reports several
`ef_search` values, which is why you will see "efC" and "ef" in the engineering docs.
The [tuning guide](guides/tuning.md) has measured numbers for what each knob costs.

## Recall

**Recall@10** is the fraction of the true 10 nearest neighbours that the index actually
returned, averaged over many queries. 0.99 means one true neighbour in a hundred was
missed. It is the one number that makes vector benchmarks different from database
benchmarks: any engine can build faster or answer faster by returning worse results, so
a speed comparison is only meaningful **at matched recall**. That is why this
repository's headline is "1.40× faster at matched recall" and not the larger
equal-parameter number.

Recall is scored against ground truth from an exact search, which is expensive, so
public datasets ship it precomputed. SIFT1M, the dataset behind the benchmarks, comes
with 10,000 queries and their true neighbours.

## Throughput and latency

| Term | Meaning |
|---|---|
| build time, vectors/s | how long inserting N vectors into a fresh index takes; the write side |
| QPS | queries per second with several threads querying at once; the read side |
| p50, p95, p99 | the latency that half, 95% and 99% of single queries finish within |

A number quoted without its dataset, dimension, `m`, `ef_construction`, `ef_search`,
recall and thread count is not comparable to anything. Every table in this repository
states all seven.

## Full-text search and BM25

The other half of hybrid search is the one search engines have done for decades. Text
is split into tokens by an **analyzer** (lowercasing, stemming, and for Chinese,
Japanese and Korean a dictionary-based segmenter), and an **inverted index** records
which rows contain each token. A query is scored with **BM25**, which rewards rows
where the query's terms are frequent and penalizes terms that appear everywhere.

Keyword search is exact where vector search is fuzzy. It finds the one document that
contains `roaring` or `A01-233:BC`, and it is unbeatable for names, identifiers and
rare terms, which embeddings blur into their neighbourhood.

## Hybrid search and fusion

Each method fails in a way the other does not. Keyword search misses a document that
answers the question in different words; vector search returns a plausible-but-wrong
neighbour for a rare exact term. The demo in this repository is built around one
question of each kind.

**Hybrid search** runs both and merges the results. The merge needs care because a BM25
score and a cosine similarity are on unrelated scales. **Reciprocal rank fusion** (RRF)
sidesteps that by combining ranks rather than scores: each result gets
`1 / (k + rank)` from every list it appears in, with `k` conventionally 60, and the sums
are sorted. It needs no tuning and no normalization, which is why it is the default.
Infinity also offers `weighted_sum` (when you know one arm matters more) and a
tensor-based reranker.

## Filters

Real queries have conditions: only documents from 2024, only this tenant, only this
category. The important property is whether the condition is applied **inside** the
search or **after** it. A post-filter takes the top 10 vector hits and discards those
that fail the condition, so a selective filter can return three rows, or none. Infinity
evaluates the filter as part of the search, so "top 5 matching rows where
`category = 'storage' AND year >= 2024`" returns the five best rows that satisfy the
condition. In the SDK this is the `.filter("...")` call chained onto a search.

## One table versus several systems

The alternative to a hybrid engine is a vector store plus a search engine plus a
database, with your application copying data between them and merging results by hand.
Infinity keeps the vectors, the text and the structured columns in one table with one
query planner and one transaction, which is the argument the demo makes with numbers.

## Other data types

Infinity also indexes **sparse vectors** (a vocabulary-sized vector where most entries
are zero, as produced by SPLADE-style models; searched by inner product) and
**tensors** (several vectors per row, for late-interaction models like ColBERT; used as
a reranker). Both are covered in the upstream [search guide](guides/search_guide.md).
They are not benchmarked in this repository.

## How much memory an index needs

An HNSW index keeps the float32 vectors in memory plus roughly 256 bytes of graph per
vector at `m=32`. As a rule of thumb:

| vectors × dimension | memory |
|---|---|
| 1,000,000 × 384 | about 1.8 GB |
| 1,000,000 × 768 | about 3.3 GB |
| 1,000,000 × 1536 | about 6.4 GB |

A 16 GB Mac holds roughly three million 768-dimension vectors or one and a half
million at 1,536, leaving room for the OS and the rest of the server.

## Reading the benchmark table

The README's table has six columns. In order: the engine; the wall time to build a
1,000,000-vector index; that divided into vectors per second; queries per second with
12 threads; the median single-query latency; and recall@10 at `ef_search=256` against
the official ground truth. The rows share dataset, parameters, thread count and
machine, and FAISS was built from source against Apple's BLAS because the Homebrew
bottle is 1.5× slower and would have flattered Infinity. All of that is what makes
the ratios in the text meaningful.

## Glossary

| Term | Short definition |
|---|---|
| ANN | approximate nearest neighbour search; trades a little recall for a lot of speed |
| analyzer | the tokenizer a full-text index runs over text before indexing it |
| BM25 | the standard relevance formula for keyword search |
| dense vector | an ordinary embedding, every entry meaningful |
| efConstruction, efSearch | the HNSW effort knobs at build time and query time |
| embedding | a vector produced by a model to represent meaning |
| fusion | merging the results of several search methods into one ranking |
| ground truth | the exact nearest neighbours a recall score is measured against |
| HNSW | the graph-based ANN index Infinity uses for dense vectors |
| M | edges per node in an HNSW graph |
| recall@k | the fraction of the true k nearest neighbours that were returned |
| RRF | reciprocal rank fusion; merge by rank, not score |
| sparse vector | a mostly-zero vector over a vocabulary, from models like SPLADE |
| tensor | several vectors per row, for late-interaction reranking |
| top-k, topn | the number of results a search returns |
