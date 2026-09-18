# RAG tutorial

Retrieval-augmented generation is three steps: split your documents into chunks and
embed them, retrieve the chunks most relevant to a question, and hand those chunks to
a language model as context. Infinity is the middle step. This page builds it in about
100 lines of Python, and the finished file is [`demo/rag_quickstart.py`](../../demo/rag_quickstart.py).

## Run it first

With a server running (`make start`):

```sh
uv run --with fastembed demo/rag_quickstart.py docs/*.md --ask "how do I tune recall"
```

It indexes this repository's own documentation and prints the retrieved chunks and a
ready-to-send prompt. Swap in your own files.

## 1. Chunk

Language models have a context budget and embedding models have an input limit, so
documents are split into pieces of a few hundred words. Two decisions matter:

- **Split on structure**, not on a fixed byte count. Paragraphs, headings and list
  items are natural units; a chunk that cuts a sentence in half embeds badly.
- **Overlap** adjacent chunks by 10 to 20 percent so a fact that straddles a boundary
  is whole in at least one of them.

The demo's `chunk_text()` packs paragraphs into roughly 900-character chunks with
150 characters of overlap. For production, a token-aware splitter from your model's
tokenizer is better; the shape of the loop stays the same.

Keep provenance with every chunk: which file, which position. You will want it for
citations and for deleting a document's chunks when it changes.

## 2. Embed

```python
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")            # 384-d, CPU, no PyTorch
vectors = [v.tolist() for v in model.embed(chunk_texts)]
```

Any embedding model works. Use the same one for documents and questions, and check
the model card for a query prefix: `bge-small` wants
`"Represent this sentence for searching relevant passages: "` in front of questions and
nothing in front of documents. The [concepts page](../concepts.md#embeddings) has the
rest of the vocabulary.

## 3. Store

One table holds the chunk text, its embedding and its provenance, with a full-text
index on the text and an HNSW index on the vector:

```python
table = db.create_table("rag_chunks", {
    "chunk_id": {"type": "integer", "constraints": ["primary key"]},
    "source":   {"type": "varchar"},
    "position": {"type": "integer"},
    "text":     {"type": "varchar"},
    "vec":      {"type": "vector,384,float"},
}, ConflictType.Error)

table.insert(rows)                                          # dicts with those five keys
table.create_index("vec_idx",  IndexInfo("vec", IndexType.Hnsw,
                   {"m": "16", "ef_construction": "200", "metric": "cosine"}), ConflictType.Error)
table.create_index("text_idx", IndexInfo("text", IndexType.FullText), ConflictType.Error)
```

Add whatever metadata you will filter on as ordinary columns: a tenant id, a document
type, a date. Filtering happens inside the search, so the filter costs nothing in
recall.

## 4. Retrieve

The question goes through both arms and the results are fused by rank:

```python
q_vec = next(iter(model.embed([QUERY_PREFIX + question]))).tolist()

frame, _ = (table.output(["chunk_id", "source", "position", "text", "score()"])
                 .match_text("text", question, 10, {"operator": "or"})
                 .match_dense("vec", q_vec, "float", "cosine", 10)
                 .fusion(method="rrf", topn=5)
                 .to_pl())
hits = frame.to_dicts()
```

`{"operator": "or"}` turns off the full-text query syntax, so a question containing a
colon or parentheses is searched as words rather than parsed as operators. Ask each arm
for more candidates (10) than you keep after fusion (5): fusion can only promote a
chunk that at least one arm returned.

Why both arms: keyword search finds the chunk that contains the exact identifier or
rare term in the question; vector search finds the chunk that answers it in different
words. The [demo](../../demo/README.md) shows each one failing where the other
succeeds, on real questions.

To restrict retrieval, add `.filter("source = 'handbook.md'")` or
`.filter("tenant_id = 42")` before `.fusion(...)`.

## 5. Generate

Build the prompt and send it to whichever model you use. The demo prints it instead of
calling one, so it runs without an API key:

```python
context = "\n\n---\n\n".join(f"[{h['source']}#{h['position']}]\n{h['text']}" for h in hits)
prompt = (
    "Answer the question using only the context below. If the context does not "
    "contain the answer, say so. Cite sources by their [file#chunk] tag.\n\n"
    f"Context:\n{context}\n\nQuestion: {question}\n"
)
```

## Keeping the index current

- **Re-indexing a changed document:** `table.delete("source = 'handbook.md'")`, then
  insert its new chunks. Deleted rows stay on disk as dead versions until compaction;
  after heavy churn call `table.compact()`. Read [known issue 6](../known-issues.md)
  before relying on `sum()` or `max()` over a heavily updated table.
- **Adding documents:** insert them. Both indexes are maintained incrementally.
- **Backups:** table snapshots, with the caveat in [known issue 7](../known-issues.md)
  that a restored full-text index currently returns nothing until rebuilt.

## Sizing

A 384-dimension model needs about 1.8 GB of memory per million chunks; 768 dimensions
about 3.3 GB. The M4 Mac mini in the benchmarks indexes about 5,000 768-d vectors a
second and answers about 5,000 queries a second in-process, which is far beyond what
one application's users generate. What that costs against a hosted vector database is
worked out in [COST_COMPARISON.md](../apple_silicon/COST_COMPARISON.md).
