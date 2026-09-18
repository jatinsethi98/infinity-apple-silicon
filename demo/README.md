# Hybrid search demo

A five-minute demo of the thing Infinity does that a vector-only index cannot:
keyword search, vector search and structured filters over **one** table, in one
query, with the results fused.

## Run it

You need a running Infinity server. From the repository root:

```sh
scripts/apple_silicon/run_server.sh start
uv run --with fastembed demo/hybrid_search_demo.py
```

The first run downloads about 90 MB of embedding model; later runs are offline.
The demo creates its own table, prints its results, and drops the table again
(pass `--keep` to leave it behind).

## What it shows

36 short technical articles are loaded into one table with three indexes on it:
a full-text index over `title` and `body`, an HNSW index over a 384-dimension
embedding, and the ordinary `category` / `year` / `reading_minutes` columns.

Two questions are then asked three ways each. They fail in opposite directions:

| | keyword only (BM25) | vector only (HNSW) | hybrid, fused with RRF |
|---|---:|---:|---:|
| `"why does my search miss documents that clearly answer the question"` | rank 2 | **rank 1** | **rank 1** |
| `"roaring"` | **rank 1** | rank 5 | **rank 1** |
| correct answer ranked first | 1 of 2 | 1 of 2 | **2 of 2** |

*(rank of the article that actually answers the question, out of the top 5)*

The first question is a vocabulary mismatch: the article that answers it does not
contain the words the question uses, so BM25 ranks a different article first
purely because that one repeats the question's vocabulary. Embeddings do not care
about the exact words, so vector search finds it.

The second is the opposite failure. `roaring` appears once in the whole corpus.
The embedding maps it into the neighbourhood of "sparse representations" and
returns a plausible but wrong article; the inverted index knows exactly which
document contains the token.

Reciprocal rank fusion gets both right, and needs no per-query tuning to do it:
it combines the two result *rankings* rather than their scores, so BM25 scores and
cosine similarities never have to be normalized onto a comparable scale.

The demo then runs the fused search again with `category = 'storage' AND year >=
2024` attached. The predicate is evaluated as part of the search rather than as a
pass over its output, which is what keeps the top-k contract honest: ask for 5 and
you get the 5 best matching rows, not however many of some other 5 happened to
survive a post-filter.

These ranks are not staged. They were found by running every candidate question
against all three modes and keeping the pair that disagreed — see the scoreboard
the demo prints for whatever your machine actually produces.

## Ask your own question

```sh
uv run --with fastembed demo/hybrid_search_demo.py --ask "how do I tune recall"
```

## Files

| File | |
|---|---|
| `hybrid_search_demo.py` | The demo. Roughly 150 lines of Python against the Infinity SDK. |
| `corpus.json` | 36 articles: `doc_id`, `title`, `body`, `category`, `year`, `reading_minutes`. |
| `rag_quickstart.py` | The retrieval half of a RAG app over your own files: chunk, embed, store, hybrid-retrieve, print a prompt. Walkthrough in [docs/guides/rag.md](../docs/guides/rag.md). |

## Notes

- Infinity does not generate embeddings. It stores and searches whatever vectors
  you hand it, which is why the demo brings its own model
  ([fastembed](https://github.com/qdrant/fastembed), ONNX, no PyTorch).
- `--port` defaults to 23817, the server's client/thrift port. The HTTP API is on
  23820 and the PostgreSQL wire protocol on 5432.
- Three SDK details that cost people time, all exercised in the demo source:
  `infinity.connect()` takes a `NetworkAddress` and not a string; `to_pl()`
  returns a `(dataframe, extra)` tuple; and index parameter values must be
  strings, with HNSW's `m` in lowercase over the Python SDK.
