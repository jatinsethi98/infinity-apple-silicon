# Infinity for Apple Silicon

Native `arm64-apple-darwin` build of [Infinity](https://github.com/infiniflow/infinity), the
open-source hybrid-search database behind [RAGFlow](https://github.com/infiniflow/ragflow), with
an HNSW index build tuned for Apple's memory system.

**No Docker. No Rosetta. Faster than FAISS on the same Mac.**

| | |
|---|---|
| Upstream version | Infinity 0.7.3 (Apache-2.0, © InfiniFlow) |
| Platform | macOS 14+ on Apple Silicon; measured on M3 Pro and M4 |
| Status | Engine core and HNSW benchmark harness working natively; full database port in progress (see [Status](#status)) |

## Why this exists

Upstream Infinity requires an x86-64 CPU with AVX2 and ships only as a Linux Docker image, so
it has never run on a Mac. RAGFlow, which uses Infinity as one of its document engines, tells
Apple Silicon users to build their own images and lists Infinity on ARM64 as unsupported.

This repository is a port of the Infinity engine to Apple Silicon plus a set of measured
optimizations to its HNSW index build: whole-vector prefetch, prefetch that skips already-visited
candidates, a 128-byte prefetch stride matching the M-series cache line, a four-accumulator L2
kernel and a one-query-against-four-candidates batch kernel, all dispatched at runtime. Every
optimization was accepted only after a paired A/B campaign showed a statistically significant
end-to-end improvement at matched recall.

## Benchmarks

SIFT1M (1,000,000 × 128-d), HNSW M=32, efConstruction=200, 10 build threads, k=10, efSearch=256,
recall@10 scored against the official SIFT1M ground truth. Mac mini with Apple M4 (4P+6E), 16 GB,
2026-09-03. Full method, raw output and reproduction commands in
[docs/apple_silicon/BENCHMARKS.md](docs/apple_silicon/BENCHMARKS.md).

| engine | index build | vectors/s | QPS (12 threads) | p50 latency | recall@10 |
|---|---:|---:|---:|---:|---:|
| **Infinity, this repo** | **35.3 s** | **28,300** | **24,700** | **209 µs** | 0.9993 |
| FAISS 1.15, Accelerate BLAS, from source | 58.1 s | 17,200 | 19,000 | 237 µs | 0.9994 |
| FAISS 1.15, `faiss-cpu` pip wheel (Python) | 60.3 s | 16,600 | 12,400 | 467 µs | 0.9989 |
| hnswlib 0.8 (Python) | 86.3 s | 11,600 | 8,600 | 644 µs | 0.9984 |
| usearch 2.26, NEON (Python) | 112.9 s | 8,900 | 6,100 | 1,019 µs | 0.9986 |

- **1.65× faster index build than FAISS at equal parameters, 1.30× higher query throughput**
  (median of two alternating paired runs, Accelerate-linked FAISS built from source).
- **1.40× faster at matched recall.** Raising Infinity to efConstruction=235 puts its recall at or
  above FAISS at every efSearch and builds in 41.4 s.
- On a MacBook Pro with M3 Pro the matched-recall speedup measured **1.52×** (95% CI 1.51–1.53)
  over a six-block randomized campaign.
- Embedding-sized vectors on the M4: 5,000 vectors/s build and 5,000 QPS at 768-d, 3,100 vectors/s
  and 3,200 QPS at 1536-d (clustered synthetic data).

We could not find published build-time or QPS numbers for any of these engines measured on Apple
Silicon; these appear to be the first. Every number here comes from scripts in this repository and
can be re-run in about fifteen minutes.

## Status

| Area | State |
|---|---|
| Native arm64 compile and link of the engine, unit tests and HNSW harness | Working |
| Native server lifecycle: create, insert, flush, HNSW build, indexed query, restart and reload | Verified (M3 Pro, 2026-08) |
| HNSW index build and query performance vs FAISS | Measured, see above |
| Full-text search, update and delete, bulk import, crash recovery on macOS | Not yet verified natively |
| Linux x86-64 and Linux ARM64 behaviour after the port | Not yet re-verified |
| Packaging, installer, Homebrew formula, macOS CI | Not started |

Read this as: a fast engine core and a working port, not yet a shippable database. The plan to
close the gap is in [docs/apple_silicon/ROADMAP.md](docs/apple_silicon/ROADMAP.md).

## Quick start: build and run the benchmark (about 15 minutes)

The benchmark harness compiles Infinity's production HNSW code directly and needs no vcpkg and
no server build.

```sh
brew install llvm@20 cmake ninja libomp faiss simde
export SDKROOT=$(xcrun --show-sdk-path)

scripts/apple_silicon/bootstrap_ctpl.sh                       # patched thread-pool header, no vcpkg needed
cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke
cmake --build build/bench --target infinity_hnsw_d0 faiss_hnsw_d0

python3 scripts/bench/fetch_datasets.py sift1m                # 168 MB download into ./datasets
export HNSW_D0_EXTERNAL_QUERIES=$PWD/datasets/sift1m/query.f32
export HNSW_D0_EXTERNAL_GROUNDTRUTH=$PWD/datasets/sift1m/groundtruth.i32

python3 scripts/bench/run_baseline.py \
  --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 --efc 200 \
  --ef 32,64,128,256 --participants $(sysctl -n hw.ncpu) --pairs 2 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench/faiss_hnsw_d0
```

The Homebrew FAISS bottle links OpenBLAS and is about 1.5× slower than FAISS built against
Apple's Accelerate framework, so it flatters Infinity. For the fair comparison used in the table
above, build the Accelerate FAISS and point `--faiss-bin` at it:

```sh
scripts/apple_silicon/build_faiss_accelerate.sh               # builds build/bench-faiss-src/faiss_hnsw_d0
```

## Full server build

The complete Infinity server builds natively with the `macos-arm64-release` CMake preset. It
needs a bootstrapped vcpkg checkout and takes about an hour. Instructions, platform notes and
known differences from the Linux build are in
[docs/apple_silicon/README.md](docs/apple_silicon/README.md). The server exposes the same
[Python SDK](https://infiniflow.org/docs/dev/pysdk_api_reference) and
[HTTP API](https://infiniflow.org/docs/dev/http_api_reference) as upstream.

## Documentation

| Document | What it covers |
|---|---|
| [docs/apple_silicon/README.md](docs/apple_silicon/README.md) | Building on macOS, platform boundaries, SIMD and allocator notes, HNSW convention differences vs FAISS |
| [docs/apple_silicon/BENCHMARKS.md](docs/apple_silicon/BENCHMARKS.md) | Published results, method, fairness rules, limitations, reproduction |
| [docs/apple_silicon/COST_COMPARISON.md](docs/apple_silicon/COST_COMPARISON.md) | What serving a RAG index costs on a Mac mini vs AWS vs Pinecone serverless |
| [docs/apple_silicon/ROADMAP.md](docs/apple_silicon/ROADMAP.md) | What is done, what is next, and the definition of done |
| [docs/apple_silicon/BASELINE.md](docs/apple_silicon/BASELINE.md) | Lab notebook of the optimization campaign on the M3 Pro, including corrections |
| [docs/apple_silicon/SEARCHLAYER_DECOMPOSITION.md](docs/apple_silicon/SEARCHLAYER_DECOMPOSITION.md) | Profile of the index-build hot path and the hypotheses it ruled out |
| [docs/apple_silicon/PRIOR_EFFORT_AUDIT.md](docs/apple_silicon/PRIOR_EFFORT_AUDIT.md) | Audit of the first porting attempt and what it actually proved |
| [scripts/bench/README.md](scripts/bench/README.md) | The benchmark driver, the fairness contract, and how to A/B a change |
| [tools/apple_silicon/native_hnsw_smoke/README.md](tools/apple_silicon/native_hnsw_smoke/README.md) | The standalone harness that compiles the production HNSW code |
| [docs/apple_silicon/benchmarks/](docs/apple_silicon/benchmarks/) | Raw output of every published run |

## Relationship to upstream

This is a port of [infiniflow/infinity](https://github.com/infiniflow/infinity) at v0.7.3, not a
rewrite. Platform-specific code is gated so that one portable codebase can serve Linux and macOS,
and the intent is to contribute the port upstream as a series of reviewable pull requests. Infinity
is licensed under Apache-2.0 and is © InfiniFlow; this repository keeps that license.

## Community

Issues and pull requests are welcome here. For Infinity itself see the
[upstream repository](https://github.com/infiniflow/infinity),
[documentation](https://infiniflow.org/docs/dev/) and
[Discord](https://discord.gg/jEfRUwEYEV).
