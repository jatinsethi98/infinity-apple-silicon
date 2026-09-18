# Infinity for Apple Silicon

**A hybrid-search database that runs natively on your Mac.** Vectors, full-text search
and SQL filters live in one table and are answered by one query. No Docker, no Rosetta,
and a faster HNSW build than FAISS on the same machine.

[![macOS arm64 build](https://github.com/jatinsethi98/infinity-apple-silicon/actions/workflows/macos_arm64.yml/badge.svg)](https://github.com/jatinsethi98/infinity-apple-silicon/actions/workflows/macos_arm64.yml)
[![lint](https://github.com/jatinsethi98/infinity-apple-silicon/actions/workflows/lint.yml/badge.svg)](https://github.com/jatinsethi98/infinity-apple-silicon/actions/workflows/lint.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform: macOS 14+ on Apple Silicon](https://img.shields.io/badge/platform-macOS%2014%2B%20arm64-lightgrey.svg)](docs/getting-started.md)

This is a native `arm64-apple-darwin` port of [Infinity](https://github.com/infiniflow/infinity),
the open-source database behind [RAGFlow](https://github.com/infiniflow/ragflow). Upstream
ships only x86-64 Linux Docker images; this repository makes the same engine build, run and
pass its test suites on a Mac, and tunes its vector index for Apple's memory system.

## Why you might want it

- **One table, three kinds of search.** Keyword search (BM25), vector search (HNSW) and
  ordinary `WHERE` filters, fused into one ranked result. You do not need a vector store
  *and* a search engine *and* a database. [See it in 5 minutes.](demo/)
- **Fast on a Mac.** Builds a 1,000,000-vector HNSW index 1.4× faster than FAISS at the
  same recall and answers 1.3× more queries per second, measured on an M4 Mac mini.
  [Numbers and method.](#benchmarks)
- **Local and free.** A Mac mini you already own serves a RAG index for the cost of its
  electricity. [What that costs against Pinecone or AWS.](docs/apple_silicon/COST_COMPARISON.md)
- **Apache-2.0, upstream-compatible API.** The same [Python SDK](docs/guides/python.md)
  and [HTTP API](docs/guides/http.md) as upstream Infinity.

## Quick start

You need an Apple Silicon Mac on macOS 14 or newer, [Homebrew](https://brew.sh) and the
Xcode command line tools (`xcode-select --install`). Everything else is installed for you.

```sh
git clone --recurse-submodules https://github.com/jatinsethi98/infinity-apple-silicon.git
cd infinity-apple-silicon

make doctor   # checks prerequisites and changes nothing
make setup    # toolchain, dependencies and the server build: ~12 min on an M4, 20 GB
make start    # server on 127.0.0.1 (ports 23817 SDK, 23820 HTTP, 5432 Postgres)
make demo     # hybrid search over 36 articles, with the results explained
```

`make` with no arguments lists every target. The build compiles roughly 1,500 C++23
module units plus thirty dependencies; measured at 12 minutes on an M4 Mac mini, longer
on older chips. It is safe to interrupt and re-run, and every step it has already done
is skipped.

**Prebuilt package.** Once the first tagged release is published, `install.sh` downloads
it, verifies its checksum and links `infinity` into `~/.local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/jatinsethi98/infinity-apple-silicon/main/install.sh | bash
```

Until then, the source build above is the way in.

### Your first query

Install the SDK (`make sdk`, or `pip install infinity-sdk`), then:

```python
import infinity
from infinity.common import ConflictType, NetworkAddress
from infinity.index import IndexInfo, IndexType

conn = infinity.connect(NetworkAddress("127.0.0.1", 23817))
db = conn.get_database("default_db")

db.drop_table("quickstart", ConflictType.Ignore)
table = db.create_table("quickstart", {
    "id":   {"type": "integer", "constraints": ["primary key"]},
    "text": {"type": "varchar"},
    "vec":  {"type": "vector,4,float"},
    "year": {"type": "integer"},
}, ConflictType.Error)

table.insert([
    {"id": 1, "text": "a bloom filter tests set membership",       "vec": [1.0, 1.2, 0.8, 0.9], "year": 2024},
    {"id": 2, "text": "hnsw builds a navigable small world graph", "vec": [4.0, 4.2, 4.3, 4.5], "year": 2023},
])
table.create_index("vec_idx",  IndexInfo("vec", IndexType.Hnsw,
                   {"m": "16", "ef_construction": "200", "metric": "cosine"}), ConflictType.Error)
table.create_index("text_idx", IndexInfo("text", IndexType.FullText), ConflictType.Error)

# Keyword search + vector search + a filter, fused by rank, in one round trip.
result, _ = (table.output(["id", "text", "score()"])
                  .match_text("text", "graph", 10)
                  .match_dense("vec", [3.0, 2.8, 2.7, 3.1], "float", "cosine", 10)
                  .filter("year >= 2023")
                  .fusion(method="rrf", topn=5)
                  .to_pl())
print(result)
conn.disconnect()
```

The [Python guide](docs/guides/python.md) walks through every call, and the
[HTTP guide](docs/guides/http.md) does the same with `curl`.

## Benchmarks

SIFT1M (1,000,000 × 128-d), HNSW M=32, efConstruction=200, 10 build threads, k=10,
efSearch=256, recall scored against the official ground truth. Mac mini M4 (16 GB), 2026-09-03.

| engine | index build | vectors/s | QPS | p50 latency | recall@10 |
|---|---:|---:|---:|---:|---:|
| **Infinity, this repo** | **35.3 s** | **28,300** | **24,700** | **209 µs** | 0.9993 |
| FAISS 1.15, Accelerate BLAS, from source | 58.1 s | 17,200 | 19,000 | 237 µs | 0.9994 |
| FAISS 1.15, `faiss-cpu` wheel (Python) | 60.3 s | 16,600 | 12,400 | 467 µs | 0.9989 |
| hnswlib 0.8 (Python) | 86.3 s | 11,600 | 8,600 | 644 µs | 0.9984 |
| usearch 2.26, NEON (Python) | 112.9 s | 8,900 | 6,100 | 1,019 µs | 0.9986 |

At matched recall (Infinity at efConstruction=235) the build is **1.40× faster** than
FAISS; on an M3 Pro the same comparison measured **1.52×** over a six-block campaign. At
768 dimensions the M4 builds about 5,000 vectors/s and serves about 5,000 QPS.

Every number comes from scripts in this repository and reproduces in about fifteen
minutes with `make bench-build bench-datasets bench`. Method, fairness rules and raw
output: [docs/apple_silicon/BENCHMARKS.md](docs/apple_silicon/BENCHMARKS.md). If the
vocabulary is new to you, [Concepts](docs/concepts.md) explains recall, HNSW and the
knobs in plain terms.

## Status

| | |
|---|---|
| Builds and runs natively | Yes. 1,126 of 1,131 unit tests and 205 of 205 SQL logic tests pass on macOS arm64. |
| Verified on macOS | Vector and full-text search, hybrid fusion, filters, update and delete, bulk import and export, crash recovery, a self-tested relocatable package. |
| Durability | Commits are `F_FULLFSYNC`-durable by default; process death is recovered from. |
| Not yet | Trusting it with data you cannot lose. An adversarial evaluation found **eleven engine defects**, four of which let an ordinary client crash the server. [Known issues](docs/known-issues.md) lists each with a workaround. |
| Platforms | macOS 14+ on Apple Silicon only. Linux and x86-64 were removed; use [upstream](https://github.com/infiniflow/infinity) there. |

The honest one-line summary: a fast engine and a complete port, with a validation layer
that is still being finished. What is next is in the [roadmap](docs/apple_silicon/ROADMAP.md).

## Documentation

Start with [docs/README.md](docs/README.md). The short list:

| If you want to | Read |
|---|---|
| Install, start, stop, fix a setup problem | [Getting started](docs/getting-started.md) |
| Understand vectors, HNSW, recall and hybrid search | [Concepts](docs/concepts.md) |
| Use the Python SDK | [Python guide](docs/guides/python.md) |
| Use it from any language over HTTP | [HTTP guide](docs/guides/http.md) |
| Build a retrieval-augmented (RAG) app | [RAG tutorial](docs/guides/rag.md) |
| Run it day to day: instances, ports, config, backups | [Operations](docs/guides/operations.md) |
| Tune index parameters, threads and memory | [Tuning](docs/guides/tuning.md) |
| Reproduce or extend the benchmarks | [Benchmarking](docs/guides/benchmarking.md) |
| Know what is broken before you hit it | [Known issues](docs/known-issues.md) |
| See the evidence behind every claim | [Engineering record](docs/apple_silicon/) |

## Contributing

Issues and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) says what belongs
here and what belongs upstream, and how to run the tests. Questions go in
[Discussions](https://github.com/jatinsethi98/infinity-apple-silicon/discussions).

## License and attribution

Apache-2.0. Infinity is © InfiniFlow; this repository is a derivative work of Infinity
0.7.3 and keeps its license. The changes relative to upstream are listed in
[NOTICE](NOTICE) and [CHANGELOG.md](CHANGELOG.md).
