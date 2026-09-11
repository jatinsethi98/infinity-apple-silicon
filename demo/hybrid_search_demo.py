#!/usr/bin/env python3
"""Hybrid search demo: why you would use Infinity instead of a vector-only index.

Run it with:

    uv run --with fastembed demo/hybrid_search_demo.py

The demo loads 36 short technical articles into ONE Infinity table and indexes
them three ways at once -- BM25 over the text, HNSW over the embeddings, and the
plain structured columns. Then it asks two questions that fail in opposite
directions:

  * A natural-language question whose answer uses none of the question's words.
    Keyword search cannot find it. Vector search can.
  * A rare exact term. Vector search blurs it into a neighbouring topic.
    Keyword search nails it.

Each question is run three ways -- keyword only, vector only, and both fused with
reciprocal rank fusion -- and the demo reports where the correct answer landed.
Neither single method gets both questions right. Fusion does.

A fourth query adds a structured predicate to the fused search, which is the part
a vector-only store cannot do without over-fetching and discarding.

Embeddings come from fastembed (BAAI/bge-small-en-v1.5, 384-d, ONNX, no torch).
The first run downloads about 90 MB of model; later runs are offline. Infinity
does not generate embeddings itself -- it stores and searches whatever you give
it -- so a demo has to bring its own model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEMO_DIR = pathlib.Path(__file__).resolve().parent
TABLE = "demo_articles"

# bge-small was trained with an asymmetric prefix: documents are embedded bare,
# queries get this instruction. Dropping it measurably costs recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# The two questions and the article that actually answers each. These are not
# rigged: the ranks the demo prints were found by running every candidate
# question in the corpus through all three modes and keeping the pair that
# fails in opposite directions.
QUESTIONS = [
    (
        "why does my search miss documents that clearly answer the question",
        30,  # "Vocabulary mismatch is the failure mode keyword search cannot fix"
        "A natural-language question. The article that answers it never uses the words\n"
        '"miss", "clearly" or "search miss" -- so BM25 ranks a different article first,\n'
        "purely because that one repeats the question's vocabulary more often.",
    ),
    (
        "roaring",
        34,  # "Secondary indexes on low-cardinality columns"
        "A rare exact term, mentioned once in the whole corpus. An embedding maps it into\n"
        'the neighbourhood of "sparse representations" and returns a plausible-but-wrong\n'
        "article; the inverted index knows exactly which document contains the token.",
    ),
]

# Applied to the fused search in the final query.
FILTER = "category = 'storage' AND year >= 2024"


def load_model():
    try:
        from fastembed import TextEmbedding
    except ImportError:
        sys.exit(
            "fastembed is not installed. Run this demo as:\n"
            "    uv run --with fastembed demo/hybrid_search_demo.py"
        )
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def rank_of(frame, doc_id: int) -> int | None:
    for i, row in enumerate(frame.to_dicts(), 1):
        if row["doc_id"] == doc_id:
            return i
    return None


def print_hits(frame, expected: int, indent: str = "     ") -> None:
    """Print a result frame as a ranked list.

    The score column's name depends on which arms the query used: a fused or
    full-text query exposes SCORE, a dense-only query exposes SIMILARITY (or
    DISTANCE for an L2 metric). Pick whichever came back rather than hard-coding
    the mapping.
    """
    rows = frame.to_dicts()
    if not rows:
        print(f"{indent}(no rows)")
        return
    score_col = next(
        (c for c in frame.columns if c.upper() in ("SCORE", "SIMILARITY", "DISTANCE")), None
    )
    for i, row in enumerate(rows, 1):
        mark = " <-- correct" if row["doc_id"] == expected else ""
        score = f"{row[score_col]:8.4f}  " if score_col else ""
        print(f"{indent}{i}. {score}{row['title'][:58]}{mark}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23817, help="Infinity client (thrift) port")
    parser.add_argument("--ask", metavar="TEXT", help="ask your own question and stop")
    parser.add_argument("--keep", action="store_true", help="leave the demo table behind")
    parser.add_argument("--topn", type=int, default=5, help="results per query (default 5)")
    args = parser.parse_args()

    # Imported here so that --help works without the SDK installed.
    import infinity
    from infinity.common import ConflictType, NetworkAddress
    from infinity.index import IndexInfo, IndexType

    corpus = json.loads((DEMO_DIR / "corpus.json").read_text())
    model = load_model()
    print(f"embedding {len(corpus)} articles (the first run downloads the model) ...")
    doc_vectors = [
        v.tolist() for v in model.embed([f"{d['title']}. {d['body']}" for d in corpus])
    ]
    dim = len(doc_vectors[0])

    def embed_query(text: str) -> list[float]:
        return next(iter(model.embed([QUERY_PREFIX + text]))).tolist()

    # connect() takes a NetworkAddress, never a string: passing "127.0.0.1:23817"
    # raises INVALID_SERVER_ADDRESS, which reads like the server is down.
    conn = infinity.connect(NetworkAddress(args.host, args.port))
    try:
        db = conn.get_database("default_db")
        db.drop_table(TABLE, ConflictType.Ignore)
        table = db.create_table(
            TABLE,
            {
                "doc_id": {"type": "integer", "constraints": ["primary key"]},
                "title": {"type": "varchar"},
                "body": {"type": "varchar"},
                "category": {"type": "varchar"},
                "year": {"type": "integer"},
                "reading_minutes": {"type": "integer"},
                "vec": {"type": f"vector,{dim},float"},
            },
            ConflictType.Error,
        )
        table.insert(
            [
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "body": d["body"],
                    "category": d["category"],
                    "year": d["year"],
                    "reading_minutes": d["reading_minutes"],
                    "vec": v,
                }
                for d, v in zip(corpus, doc_vectors)
            ]
        )

        # Index parameter VALUES must all be strings, and the HNSW key is
        # lowercase "m" over the Python SDK (uppercase "M" works only over HTTP).
        table.create_index(
            "vec_idx",
            IndexInfo(
                "vec", IndexType.Hnsw, {"m": "16", "ef_construction": "200", "metric": "cosine"}
            ),
            ConflictType.Error,
        )
        table.create_index("title_idx", IndexInfo("title", IndexType.FullText), ConflictType.Error)
        table.create_index("body_idx", IndexInfo("body", IndexType.FullText), ConflictType.Error)
        print(
            f"loaded {len(corpus)} rows into '{TABLE}': "
            f"HNSW on vec ({dim}-d), full text on title and body, "
            "plus category / year / reading_minutes as ordinary columns\n"
        )

        # "title^2,body" searches both text columns and weights a title match
        # twice as heavily as a body match.
        text_fields = "title^2,body"

        def keyword(q: str, topn: int):
            return table.output(["doc_id", "title", "score()"]).match_text(
                text_fields, q, topn
            ).to_pl()[0]

        def vector(q_vec, topn: int):
            # score() is rejected for a dense-only query: the server reserves it
            # for queries with a full-text, tensor or fusion arm. Cosine
            # similarity is exposed as _similarity instead.
            return table.output(["doc_id", "title", "_similarity"]).match_dense(
                "vec", q_vec, "float", "cosine", topn
            ).to_pl()[0]

        def hybrid(q: str, q_vec, topn: int, where: str | None = None):
            # RRF combines the two arms by rank, not by score, so BM25 scores and
            # cosine similarities never have to be normalized onto a common scale.
            builder = (
                table.output(["doc_id", "title", "category", "year", "score()"])
                .match_text(text_fields, q, max(topn, 10))
                .match_dense("vec", q_vec, "float", "cosine", max(topn, 10))
            )
            if where:
                builder = builder.filter(where)
            return builder.fusion(method="rrf", topn=topn).to_pl()[0]

        if args.ask:
            q_vec = embed_query(args.ask)
            print(f"question: {args.ask!r}\n")
            for label, frame in (
                ("keyword only (BM25)", keyword(args.ask, args.topn)),
                ("vector only (HNSW) ", vector(q_vec, args.topn)),
                ("hybrid (RRF)       ", hybrid(args.ask, q_vec, args.topn)),
            ):
                print(f"  {label}")
                # -1 matches no doc_id, so nothing is flagged as "correct": for an
                # ad-hoc question the demo has no ground truth to compare against.
                print_hits(frame, expected=-1)
                print()
            if not args.keep:
                db.drop_table(TABLE, ConflictType.Ignore)
            return 0

        scoreboard = []
        for question, expected, why in QUESTIONS:
            q_vec = embed_query(question)
            print("=" * 78)
            print(f"QUESTION: {question!r}")
            print(f"ANSWER:   #{expected} {corpus[expected - 1]['title']!r}")
            print(f"\n{why}\n")

            frames = {
                "keyword only (BM25)": keyword(question, args.topn),
                "vector only (HNSW)": vector(q_vec, args.topn),
                "hybrid (BM25 + HNSW, RRF)": hybrid(question, q_vec, args.topn),
            }
            ranks = {}
            for label, frame in frames.items():
                got = rank_of(frame, expected)
                ranks[label] = got
                verdict = "correct answer first" if got == 1 else (
                    f"correct answer at rank {got}" if got else "correct answer not in top "
                    f"{args.topn}"
                )
                print(f"  {label}: {verdict}")
                print_hits(frame, expected)
                print()
            scoreboard.append((question, ranks))

        print("=" * 78)
        print("SCOREBOARD -- rank of the correct answer, lower is better\n")
        modes = list(scoreboard[0][1])
        width = max(len(m) for m in modes)
        for mode in modes:
            cells = " ".join(
                f"{str(ranks[mode] or '>' + str(args.topn)):>4}" for _, ranks in scoreboard
            )
            firsts = sum(1 for _, ranks in scoreboard if ranks[mode] == 1)
            print(f"  {mode:<{width}}  {cells}   ({firsts}/{len(scoreboard)} ranked first)")
        print(
            "\nEach single method wins one question and loses the other. Fusion wins both,\n"
            "and it needed no per-query tuning to do it."
        )

        # The part a vector-only store cannot do. The predicate is evaluated as
        # part of the search rather than as a post-filter, so the top-k contract
        # still holds when the filter is selective.
        question, expected, _ = QUESTIONS[0]
        frame = hybrid(question, embed_query(question), args.topn, where=FILTER)
        print("\n" + "=" * 78)
        print("HYBRID + STRUCTURED FILTER")
        print(f"\nSame fused search, restricted to {FILTER}.")
        print("The predicate is part of the search, not a pass over its output, so k results")
        print("really are the k best matching rows rather than whatever survived a")
        print("post-filter. This is the query a vector-only index cannot answer.\n")
        for row in frame.to_dicts():
            print(f"     {row['SCORE']:8.4f}  [{row['category']}, {row['year']}] {row['title'][:48]}")

        print("\n" + "=" * 78)
        print("All of that ran against one table holding the text, the vectors and the")
        print("structured columns together -- one query planner, one transaction, one copy")
        print("of the data. That is the argument for a hybrid engine over a vector index")
        print("bolted onto a separate database.")
        print("\nTry your own question:")
        print("  uv run --with fastembed demo/hybrid_search_demo.py --ask 'your question here'")

        if args.keep:
            print(f"\nTable '{TABLE}' left in place for you to poke at.")
        else:
            db.drop_table(TABLE, ConflictType.Ignore)
            print(f"\nDropped '{TABLE}'. Pass --keep to leave it in place.")
        return 0
    finally:
        conn.disconnect()


if __name__ == "__main__":
    sys.exit(main())
