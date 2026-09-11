# Benchmarks

Published results for the Apple Silicon port. Every number on this page was produced by a script
in this repository, on the machine named, on the date given, and the raw engine output is committed
under [`benchmarks/`](benchmarks/). Read [Limitations](#limitations) before quoting anything.

## What is measured

| metric | meaning | database analogue |
|---|---|---|
| index build | wall time to insert N vectors into a fresh HNSW graph, measured inside the engine (`cold_build_ns`), excluding dataset load and the audit phases | bulk load / write throughput |
| vectors/s | N divided by index build time | rows written per second |
| QPS | queries per second, 12 concurrent threads, k=10, after a warm-up, validated against a checksum | read throughput |
| p50 / p95 / p99 | latency of one query at a time on one thread | read latency |
| recall@10 | fraction of the true 10 nearest neighbours the index returned, scored against the official SIFT1M ground truth over its 10,000 published queries | correctness (the index is approximate) |

Recall is what makes vector benchmarks different from database benchmarks: any engine can build or
search faster by returning worse answers. A build-time ratio is only meaningful at matched recall,
so results are reported at equal parameters *and* at the parameter setting where Infinity's recall
meets FAISS's at every efSearch.

## Mac mini, Apple M4, 2026-09-03

Host: Mac mini (Mac16,10), Apple M4 with 4 performance and 6 efficiency cores, 16 GB, 128-byte
cache lines, macOS 26.3.1, Homebrew clang 20.1.8, CMake 4.4.3, AC power, thermal state nominal,
normal desktop load present (see [host state](#limitations)). Full details in
[`benchmarks/2026-09-03-m4-mini/host.txt`](benchmarks/2026-09-03-m4-mini/host.txt).

Workload: SIFT1M base set, 1,000,000 × 128-d float32. HNSW M=32, efConstruction=200,
10 build threads, chunk 8192. Queries: k=10, efSearch=256.

### Infinity vs FAISS, paired

FAISS 1.15.0 built from source against Apple's Accelerate BLAS (Metal backend off), linked into
the same harness binary as Infinity. Two alternating-order pairs, each engine in a fresh process;
medians reported.

| engine | build (median) | min – max | vectors/s | QPS | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Infinity, efC=200 | **35.32 s** | 34.95 – 35.69 | 28,312 | **24,672** | 209 µs | 235 µs | 249 µs |
| FAISS, efC=200 | 58.10 s | 58.07 – 58.13 | 17,212 | 18,958 | 237 µs | 317 µs | 356 µs |
| Infinity, efC=235 (iso-recall point) | 41.45 s | single run | 24,127 | 24,586 | 207 µs | – | – |

Recall@10 on the official ground truth (median of the two runs):

| efSearch | Infinity efC=200 | FAISS efC=200 | Infinity efC=235 |
|---:|---:|---:|---:|
| 32 | 0.9410 | 0.9425 | 0.9442 |
| 64 | 0.9822 | 0.9830 | 0.9835 |
| 128 | 0.9959 | 0.9963 | 0.9964 |
| 256 | 0.9993 | 0.9994 | 0.9993 |

Headline ratios:

| comparison | build | QPS |
|---|---:|---:|
| equal parameters (efC=200 both) | **1.645×** faster | **1.301×** higher |
| matched recall (Infinity efC=235 vs FAISS efC=200) | **1.402×** faster | 1.30× higher |

At equal parameters Infinity's recall is 0.0001 to 0.0015 below FAISS, which is one to fifteen
misses per 100,000 neighbour slots. At efC=235 it is at or above FAISS at every efSearch except
a 0.0001 tie at 256, which is one query in 10,000 and inside build-to-build variance. The harness
driver still prints `RECALL-UNMATCHED` for this run because its gate uses the 64-synthetic-query
self-audit, an instrument this project retired for parity decisions (see
[README.md](README.md#the-recall-difference-was-mostly-the-ruler-and-what-remains-of-it-is-tiny)).

### Other engines, same run

Same dataset, parameters and machine. Python rows were run through each library's Python package
with `uv`, 10 build threads, QPS over the 10,000 published queries at 12 threads, and include the
Python wrapper's overhead. Script: [`scripts/bench/bench_python_libs.py`](../../scripts/bench/bench_python_libs.py).

| engine | how | build | vectors/s | QPS | p50 | recall@10 (ef 256) |
|---|---|---:|---:|---:|---:|---:|
| Infinity, this repo | C++ harness | 35.3 s | 28,312 | 24,672 | 209 µs | 0.9993 |
| FAISS 1.15.0, Accelerate, from source | C++ harness | 58.1 s | 17,212 | 18,958 | 237 µs | 0.9994 |
| FAISS 1.15.0, `faiss-cpu` pip wheel | Python | 60.3 s | 16,575 | 12,379 | 467 µs | 0.9989 |
| FAISS 1.15.0, Homebrew bottle (OpenBLAS) | C++ harness | 88.9 s | 11,252 | 11,116 | 468 µs | 0.9994 |
| hnswlib 0.8.0 | Python | 86.3 s | 11,582 | 8,626 | 644 µs | 0.9984 |
| usearch 2.26.2, `hardware_acceleration=neon` | Python, f32 | 112.9 s | 8,856 | 6,056 | 1,019 µs | 0.9986 |

Every engine builds essentially the same graph at these parameters (edge counts within 2%), so
this table compares distance kernels, prefetching and memory access, not algorithms. The Homebrew
FAISS row is included only to document why it is not the reference: its BLAS backend alone costs
1.5× on build.

### Embedding-sized vectors

Real text embeddings are 768 to 1536 dimensions. SIFT is the only real dataset here, so these rows
use synthetic data: "clustered" is 1,000–2,000 Gaussian clusters, unit-normalised, which has the
low intrinsic dimension of real embeddings; "uniform" is uniform random, which has no structure
and is the worst case for any graph index. Expect real embeddings to land between the clustered
and SIFT numbers. Recall for these rows comes from the harness's 64 uniform probe queries and is a
sanity check only.

| data | vectors × dims | build | vectors/s | QPS | p50 | p99 |
|---|---|---:|---:|---:|---:|---:|
| clustered synthetic | 200,000 × 768 | 40.2 s | 4,977 | 4,996 | 686 µs | 1,102 µs |
| clustered synthetic | 100,000 × 1536 | 32.7 s | 3,055 | 3,215 | 1,091 µs | 1,623 µs |
| uniform random | 200,000 × 768 | 132.1 s | 1,514 | 1,536 | 1,945 µs | 2,432 µs |
| uniform random | 100,000 × 1536 | 117.6 s | 850 | 966 | 3,358 µs | 3,668 µs |

Memory: the HNSW index holds the float32 vectors plus about 256 bytes of graph per vector at M=32,
so 1M × 768-d needs about 3.3 GB and 1M × 1536-d about 6.4 GB. A 16 GB machine holds roughly
3M × 768-d or 1.5M × 1536-d.

## MacBook Pro, Apple M3 Pro, 2026-08 to 2026-09-02

The optimization campaign was run on a MacBook Pro (Mac15,7, M3 Pro, 6P+6E, 36 GB). Its final
paired campaign, six randomized blocks with all arms in the same thermal state, Accelerate FAISS
from source, 12 threads:

| arm | median build | speedup vs FAISS efC=200 | 95% CI |
|---|---:|---:|---|
| FAISS efC=200 | 59.96 s | – | – |
| Infinity efC=235 (matched recall) | 39.37 s | **1.523×** | [1.514×, 1.532×] |
| Infinity efC=240 (recall significantly above FAISS everywhere) | 40.17 s | 1.486× | [1.470×, 1.501×] |
| Infinity efC=250, before the last two fixes | 45.81 s | 1.312× | [1.306×, 1.319×] |

The full notebook, including the corrections that moved the headline from 1.344× to 1.52×, is
[BASELINE.md](BASELINE.md). The M4 mini number (1.40× at matched recall) is lower mainly because
it is a single iso-recall run on a 10-core part with six efficiency cores, against a 12-thread
campaign on a 6P+6E part; the equal-parameter ratio on the M4 (1.645×) is in line with the M3 Pro.

## Reproduce

```sh
brew install llvm@20 cmake ninja libomp faiss simde
export SDKROOT=$(xcrun --show-sdk-path)
scripts/apple_silicon/bootstrap_ctpl.sh
cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke
cmake --build build/bench --target infinity_hnsw_d0 faiss_hnsw_d0
scripts/apple_silicon/build_faiss_accelerate.sh            # fair FAISS -> build/bench-faiss-src/faiss_hnsw_d0

python3 scripts/bench/fetch_datasets.py sift1m --spot-check
export HNSW_D0_EXTERNAL_QUERIES=$PWD/datasets/sift1m/query.f32
export HNSW_D0_EXTERNAL_GROUNDTRUTH=$PWD/datasets/sift1m/groundtruth.i32

# paired Infinity vs Accelerate FAISS, equal parameters
python3 scripts/bench/run_baseline.py \
  --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 --efc 200 \
  --ef 32,64,128,256 --participants 10 --pairs 2 --chunk-size 8192 --query-count 1000 \
  --build-grain 1 --timeout 1800 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0

# iso-recall point (single Infinity run at efC=235)
build/bench/infinity_hnsw_d0 datasets/sift1m/base.f32 1000000 128 32 235 256 8192 1000 10 1 /tmp/sidecar.json

# Python libraries on the same data
uv run --python 3.12 --with numpy,hnswlib   python scripts/bench/bench_python_libs.py hnswlib
uv run --python 3.12 --with numpy,usearch   python scripts/bench/bench_python_libs.py usearch
uv run --python 3.12 --with numpy,faiss-cpu python scripts/bench/bench_python_libs.py faiss
```

The harness binaries take eleven positional arguments:
`DATASET N D M EF_CONSTRUCTION EF_SEARCH CHUNK QUERY_COUNT PARTICIPANTS BUILD_GRAIN AUDIT_SIDECAR`
and print `key=value` lines; `run_baseline.py` parses those. The two external-truth environment
variables are only honoured at n=1,000,000 because the published query ids address the full base set.

## Fairness rules

1. Same dataset file, vector count, dimensions, M, efConstruction and thread count for every engine.
2. FAISS is built from source against Accelerate. The Homebrew bottle is not a valid reference.
3. Build time is the engine's own clock around construction and insert, never the process wall clock.
4. Paired, alternating-order runs in fresh processes; medians with spread; never a single run for a
   headline (the efC=235 point above is marked as single).
5. Recall on the official ground truth for any parity claim; the 64-query self-audit is a sanity
   check only.
6. Every ratio is quoted with its N. Ratios measured at 12,288 vectors do not transfer to 1M.

## Limitations

- **One machine per campaign, desktop load present.** The M4 mini runs were taken on a machine
  running a browser and other desktop processes (`cpu_idle` 79–82% at preflight, load average
  2–6). The engines are compute-bound and got their cores, and FAISS's run-to-run spread of 0.1%
  says the clocks were stable, but a quiet machine would tighten Infinity's 1% spread.
- **Query throughput is in-process.** 24,700 QPS is the engine's ceiling with 12 threads hammering
  it directly; a server in front of it adds HTTP and serialization cost.
- **The iso-recall point on the M4 is a single run.** Treat 1.40× as ±3% until a paired campaign
  at efC=235 is run on that machine.
- **Python rows include wrapper overhead**, which affects QPS and p50 more than build time.
- **Synthetic embedding-sized data.** No real 768-d or 1536-d corpus was used yet; adding one
  (for example a public OpenAI or nomic embedding dump) is the highest-value addition to this page.
- **The 12-thread query concurrency is a harness constant** and oversubscribes the 10-core M4 slightly.
- **Only HNSW build and query are measured.** Full-text search, filtering, updates, deletes,
  persistence and recovery are not benchmarked because they are not yet verified on macOS.
