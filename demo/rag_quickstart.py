#!/usr/bin/env python3
"""The retrieval half of a RAG app, end to end, in one file.

    uv run --with fastembed demo/rag_quickstart.py docs/*.md --ask "how do I tune recall"

Give it any text or Markdown files. It splits them into overlapping chunks, embeds
every chunk with a small CPU model, stores chunk text + embedding + provenance in ONE
Infinity table with a full-text index and an HNSW index, and answers --ask with a
hybrid search fused by reciprocal rank. The output is the retrieved context and a
ready-to-send prompt; hand that prompt to whichever language model you use.

Infinity does not generate embeddings or call a model. It stores and searches what
you give it, which is why this file brings its own embedding model (fastembed,
BAAI/bge-small-en-v1.5, 384-d, ONNX, no PyTorch; the first run downloads ~90 MB).

Every SDK call here has the same shape as demo/hybrid_search_demo.py, which is run
against a native server as part of verifying this repository.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

TABLE = "rag_chunks"

# bge-small was trained with an asymmetric prefix: documents are embedded bare,
# queries get this instruction. Dropping it measurably costs recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split on paragraph boundaries, then pack paragraphs into ~size-character
    chunks that overlap by ~overlap characters so a sentence cut at a boundary is
    still whole in one of them. Character counts, not tokens: good enough for a
    quickstart and dependency-free."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            chunks.append(current)
        # Carry the tail of the previous chunk forward as overlap.
        tail = current[-overlap:] if current and overlap else ""
        current = f"{tail}\n\n{para}" if tail else para
        while len(current) > size:            # a single huge paragraph
            chunks.append(current[:size])
            current = current[size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def load_model():
    try:
        from fastembed import TextEmbedding
    except ImportError:
        sys.exit("fastembed is not installed. Run this as:\n"
                 "    uv run --with fastembed demo/rag_quickstart.py FILES... --ask QUESTION")
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=pathlib.Path, help="text or Markdown files to index")
    parser.add_argument("--ask", required=True, metavar="QUESTION")
    parser.add_argument("--topn", type=int, default=5, help="chunks to retrieve (default 5)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23817, help="Infinity SDK (thrift) port")
    parser.add_argument("--keep", action="store_true", help="leave the table behind for more questions")
    args = parser.parse_args()

    import infinity
    from infinity.common import ConflictType, NetworkAddress
    from infinity.index import IndexInfo, IndexType

    # 1. Chunk.
    rows = []
    for path in args.files:
        for position, text in enumerate(chunk_text(path.read_text(encoding="utf-8", errors="replace"))):
            rows.append({"chunk_id": len(rows), "source": str(path), "position": position, "text": text})
    if not rows:
        sys.exit("no text found in the given files")
    print(f"{len(rows)} chunks from {len(args.files)} file(s)")

    # 2. Embed.
    model = load_model()
    print("embedding (the first run downloads the model) ...")
    vectors = [v.tolist() for v in model.embed([r["text"] for r in rows])]
    dim = len(vectors[0])
    for row, vec in zip(rows, vectors):
        row["vec"] = vec

    # 3. Store: text, vector and provenance in one table, indexed both ways.
    conn = infinity.connect(NetworkAddress(args.host, args.port))
    try:
        db = conn.get_database("default_db")
        db.drop_table(TABLE, ConflictType.Ignore)
        table = db.create_table(TABLE, {
            "chunk_id": {"type": "integer", "constraints": ["primary key"]},
            "source":   {"type": "varchar"},
            "position": {"type": "integer"},
            "text":     {"type": "varchar"},
            "vec":      {"type": f"vector,{dim},float"},
        }, ConflictType.Error)
        for start in range(0, len(rows), 500):
            table.insert(rows[start:start + 500])
        table.create_index("vec_idx", IndexInfo("vec", IndexType.Hnsw,
                           {"m": "16", "ef_construction": "200", "metric": "cosine"}), ConflictType.Error)
        table.create_index("text_idx", IndexInfo("text", IndexType.FullText), ConflictType.Error)

        # 4. Retrieve: keyword arm + vector arm, fused by rank. {"operator": "or"}
        #    turns off full-text query syntax so a question containing ":" or "(" is
        #    searched as words rather than parsed as operators.
        q_vec = next(iter(model.embed([QUERY_PREFIX + args.ask]))).tolist()
        candidates = max(args.topn, 10)
        frame, _ = (table.output(["chunk_id", "source", "position", "text", "score()"])
                         .match_text("text", args.ask, candidates, {"operator": "or"})
                         .match_dense("vec", q_vec, "float", "cosine", candidates)
                         .fusion(method="rrf", topn=args.topn)
                         .to_pl())
        hits = frame.to_dicts()

        # 5. Build the prompt. This is the hand-off point to your language model.
        print(f"\nTop {len(hits)} chunks for: {args.ask!r}\n")
        for i, h in enumerate(hits, 1):
            preview = h["text"].replace("\n", " ")[:100]
            print(f"  {i}. {h['SCORE']:.4f}  {h['source']}#{h['position']}  {preview}...")

        context = "\n\n---\n\n".join(f"[{h['source']}#{h['position']}]\n{h['text']}" for h in hits)
        prompt = (
            "Answer the question using only the context below. If the context does not\n"
            "contain the answer, say so. Cite sources by their [file#chunk] tag.\n\n"
            f"Context:\n{context}\n\nQuestion: {args.ask}\n"
        )
        print("\n" + "=" * 78 + "\nPROMPT (send this to your model)\n" + "=" * 78 + "\n")
        print(prompt)

        if args.keep:
            print(f"Table '{TABLE}' kept; query it again with the Python guide's hybrid example.")
        else:
            db.drop_table(TABLE, ConflictType.Ignore)
        return 0
    finally:
        conn.disconnect()


if __name__ == "__main__":
    sys.exit(main())
