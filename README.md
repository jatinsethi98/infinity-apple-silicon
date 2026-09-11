# Infinity for Apple Silicon

Native `arm64-apple-darwin` build of [Infinity](https://github.com/infiniflow/infinity), the
open-source hybrid-search database behind [RAGFlow](https://github.com/infiniflow/ragflow), with
an HNSW index build tuned for Apple's memory system.

**No Docker. No Rosetta. Faster than FAISS on the same Mac.**

| | |
|---|---|
| Upstream version | Infinity 0.7.3 (Apache-2.0, © InfiniFlow) |
| Platform | macOS 14+ on Apple Silicon **only** — the build refuses other hosts |
| Status | Server builds, runs and passes its suites natively; **not yet safe for data you care about** (see [Status](#status)) |

## Quick start

Three commands from a fresh clone to a working hybrid-search database.

**You need:** an Apple Silicon Mac on macOS 14 or newer, the Xcode command line tools
(`xcode-select --install`), [Homebrew](https://brew.sh), [uv](https://docs.astral.sh/uv/)
(`brew install uv`), and about 25 GB of free disk. The C++ toolchain, the vcpkg
dependencies and the analyzer dictionaries are installed for you.

```sh
git clone --recurse-submodules https://github.com/jatinsethi98/infinity-apple-silicon.git
cd infinity-apple-silicon

scripts/apple_silicon/setup.sh                       # 1. install the toolchain, build the server
scripts/apple_silicon/run_server.sh start            # 2. start it on 127.0.0.1
uv run --with fastembed demo/hybrid_search_demo.py   # 3. see what it does
```

Step 1 takes about an hour: it installs Homebrew LLVM 20, CMake, Ninja and libomp,
fetches the analyzer dictionaries, bootstraps vcpkg at the pinned baseline, and then
compiles roughly 1500 C++23 module translation units. It is safe to re-run — every
step is skipped if it is already done. Add `--with-tests` to build the unit tests
and the SQL logic test harness too, or `--no-build` to stop after the prerequisites.

Two things worth knowing before you start it: the `resource` submodule in step 0 is
about 500 MB of full-text analyzer dictionaries, and if you forget
`--recurse-submodules`, `setup.sh` fetches it for you rather than failing.

### What the demo shows

[`demo/`](demo/) loads 36 technical articles into one table with a full-text index,
an HNSW vector index and ordinary structured columns, then asks two questions that
break the two search methods in opposite directions:

| | keyword only (BM25) | vector only (HNSW) | both, fused with RRF |
|---|---:|---:|---:|
| a question whose answer shares none of its words | rank 2 | **rank 1** | **rank 1** |
| a rare exact term (`"roaring"`) | **rank 1** | rank 5 | **rank 1** |
| correct answer ranked first | 1 of 2 | 1 of 2 | **2 of 2** |

That is the argument for a hybrid engine in one table: neither method alone is
enough, and you should not have to run two databases to get both. Ask your own
question with `--ask "..."`. Details in [demo/README.md](demo/README.md).

### Talk to it yourself

The server speaks the same [Python SDK](https://infiniflow.org/docs/dev/pysdk_api_reference)
and [HTTP API](https://infiniflow.org/docs/dev/http_api_reference) as upstream
Infinity, on ports 23817 (SDK), 23820 (HTTP) and 5432 (PostgreSQL wire).

The SDK ships in this repository. Install it into a virtualenv with
[uv](https://docs.astral.sh/uv/) — note the import name is `infinity` while the
distribution is `infinity-sdk`:

```sh
uv sync --python 3.11 --all-extras
```

```python
import infinity
from infinity.common import ConflictType, NetworkAddress
from infinity.index import IndexInfo, IndexType

# connect() takes a NetworkAddress, not a string.
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

# Index parameter values are strings, and HNSW's "m" is lowercase over the SDK.
table.create_index("vec_idx", IndexInfo("vec", IndexType.Hnsw,
                   {"m": "16", "ef_construction": "200", "metric": "cosine"}), ConflictType.Error)
table.create_index("text_idx", IndexInfo("text", IndexType.FullText), ConflictType.Error)

# Full text, vector search and a structured filter in one query, fused by rank.
# to_pl() returns a (dataframe, extra) tuple.
result, _ = (table.output(["id", "text", "score()"])
                  .match_text("text", "graph", 10)
                  .match_dense("vec", [3.0, 2.8, 2.7, 3.1], "float", "cosine", 10)
                  .filter("year >= 2023")
                  .fusion(method="rrf", topn=5)
                  .to_pl())
print(result)

conn.disconnect()
```

Or without Python at all:

```sh
curl -X POST 'http://127.0.0.1:23820/databases/default_db/tables/curl_demo' \
  -H 'content-type: application/json' \
  -d '{"create_option": "ignore_if_exists",
       "fields": [{"name": "id", "type": "integer", "constraints": ["PRIMARY KEY"]},
                  {"name": "text", "type": "varchar"}]}'

curl -X POST 'http://127.0.0.1:23820/databases/default_db/tables/curl_demo/docs' \
  -H 'content-type: application/json' \
  -d '[{"id": 1, "text": "a bloom filter tests set membership"}]'

# Note: SELECT is a GET with a JSON body.
curl -X GET 'http://127.0.0.1:23820/databases/default_db/tables/curl_demo/docs' \
  -H 'content-type: application/json' \
  -d '{"output": ["id", "text"], "filter": "id >= 1"}'
```

### Managing the server

```sh
scripts/apple_silicon/run_server.sh status
scripts/apple_silicon/run_server.sh stop
scripts/apple_silicon/run_server.sh start --fresh          # wipe this instance's data first
scripts/apple_silicon/run_server.sh start --instance two --port-offset 100
```

Data, config and logs for an instance live under `build/instances/<name>/`, not in
`/var/infinity` — on macOS `/var` and `/usr/share` are root-owned, so the shipped
`conf/infinity_conf.toml` cannot be used from a checkout. `run_server.sh` renders a
usable config for you; that is why you should start the server through it rather
than by running the binary directly.

### If something goes wrong

| Symptom | Cause |
|---|---|
| `cmake ... is too old` / `import std` errors | CMake must be 4.0.3 or newer, and below 4.5. `brew upgrade cmake`. |
| `no clang++ at /opt/homebrew/opt/llvm@20/bin` | Apple Clang cannot build this. `brew install llvm@20`. |
| vcpkg reports a missing `spdlog`, `thrift` or `zlib` | A shallow vcpkg clone cannot resolve the pinned baseline. `setup.sh` and `build_server.sh` repair this automatically; re-run either. |
| `INVALID_SERVER_ADDRESS` from the SDK | `infinity.connect()` needs `NetworkAddress("127.0.0.1", 23817)`, not a string. It looks like the server is down but is not. |
| Analyzer or CJK full-text tests fail | The `resource` submodule is missing: `git submodule update --init --recursive resource`. |
| `uname -m` says `x86_64` on an M-series Mac | You are in a Rosetta shell. Start a native one: `arch -arm64 zsh`. |

Server logs are at `build/instances/<name>/log/infinity.log`.

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
| Native server lifecycle: create, insert, flush, HNSW build, indexed query, restart and reload | Verified |
| HNSW index build and query performance vs FAISS | Measured, see above |
| Unit tests | 1126 / 1131 pass; 5 known failures, each with a root cause on record |
| SQL logic tests | 205 / 205 pass — full-text, update/delete/drop, import, export, compact, optimize |
| Crash recovery from process death, and power-loss-safe `F_FULLFSYNC` commits | Verified |
| Packaging | Working: a relocatable tarball that is started and queried as part of building it |
| macOS arm64 CI | Written; see the badge-less truth in [MACOS_VERIFICATION.md](docs/apple_silicon/MACOS_VERIFICATION.md) |
| **Data safety** | **Eleven known blockers.** See below. |
| Linux and x86-64 | **Removed.** The build refuses a non-Darwin host by design; use [upstream](https://github.com/infiniflow/infinity). |
| HTTP API, cluster mode | Lightly tested |

**Do not put data you care about in this yet.** A deliberate adversarial evaluation
([docs/apple_silicon/EVALUATION.md](docs/apple_silicon/EVALUATION.md)) found eleven
blockers, and they fall into two patterns. Four are ways an ordinary client can crash
the whole server process with one call — including a plain `WHERE v <= 127` on an
indexed `int8` column. The rest are worse for a retrieval store: wrong-typed values
silently becoming NULL, a NaN in a vector outranking the true nearest neighbour, and a
restored snapshot's full-text index returning nothing — each with `error_code: 0` and
nothing in the log. A wrong answer that reports success is one nothing downstream can
detect.

Read this as: a fast engine and a real port, with a functional surface that is broad
and a validation layer that is not finished. The plan to close the gap is in
[docs/apple_silicon/ROADMAP.md](docs/apple_silicon/ROADMAP.md).

## Reproduce the benchmarks (about 15 minutes)

The benchmark harness compiles Infinity's production HNSW code directly, so it needs
neither vcpkg nor a server build — it is independent of the Quick start above.

```sh
brew install llvm@20 cmake ninja libomp faiss simde   # CMake must be >= 4.0.3 and < 4.5
export SDKROOT=$(xcrun --show-sdk-path)

scripts/apple_silicon/bootstrap_ctpl.sh                       # patched thread-pool header, no vcpkg needed
cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke
cmake --build build/bench --target infinity_hnsw_d0 faiss_hnsw_d0

python3 scripts/bench/fetch_datasets.py sift1m                # 168 MB from a slow academic mirror;
                                                              # this step can take much longer than the rest
export HNSW_D0_EXTERNAL_QUERIES=$PWD/datasets/sift1m/query.f32
export HNSW_D0_EXTERNAL_GROUNDTRUTH=$PWD/datasets/sift1m/groundtruth.i32

python3 scripts/bench/run_baseline.py \
  --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 --efc 200 \
  --ef 32,64,128,256 --participants 10 --pairs 2 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench/faiss_hnsw_d0
```

`--participants 10` is the thread count the published table was measured at, on a
10-core M4. Use `$(sysctl -n hw.ncpu)` instead if you want your machine's numbers
rather than a comparison against the table.

The Homebrew FAISS bottle links OpenBLAS and is about 1.5× slower than FAISS built against
Apple's Accelerate framework, so it flatters Infinity. For the fair comparison used in the table
above, build the Accelerate FAISS and point `--faiss-bin` at it:

```sh
scripts/apple_silicon/build_faiss_accelerate.sh               # builds build/bench-faiss-src/faiss_hnsw_d0
python3 scripts/bench/run_baseline.py ... --faiss-bin build/bench-faiss-src/faiss_hnsw_d0
```

## Building by hand

`scripts/apple_silicon/setup.sh` is a wrapper over steps you can also run yourself:
the CMake presets, the vcpkg bootstrap, the platform notes and the known differences
from the Linux build are documented in
[docs/apple_silicon/README.md](docs/apple_silicon/README.md).

## Documentation

| Document | What it covers |
|---|---|
| [demo/README.md](demo/README.md) | The hybrid-search demo: what it shows and why those ranks |
| [docs/apple_silicon/README.md](docs/apple_silicon/README.md) | Building on macOS, platform boundaries, SIMD and allocator notes, HNSW convention differences vs FAISS |
| [docs/apple_silicon/MACOS_VERIFICATION.md](docs/apple_silicon/MACOS_VERIFICATION.md) | What is verified on macOS and what is not, with the command behind each claim |
| [docs/apple_silicon/EVALUATION.md](docs/apple_silicon/EVALUATION.md) | Adversarial functional, concurrency, recovery and load testing; the eleven blockers |
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
rewrite. This fork targets Apple Silicon only: the Linux and x86-64 build paths were removed
rather than left in place unverified, and the top-level `CMakeLists.txt` refuses a non-Darwin
host. Use upstream for any other platform.

Because the port is a fork rather than a patch series, contributing it upstream means
re-landing the platform work behind portability gates. That is deliberate future work, not
something this tree is shaped for today. Infinity is licensed under Apache-2.0 and is
© InfiniFlow; this repository keeps that license, and its attribution is recorded in
[NOTICE](NOTICE).

## Community

Issues and pull requests for **this port** are welcome here — see
[CONTRIBUTING.md](CONTRIBUTING.md). For Infinity itself see the
[upstream repository](https://github.com/infiniflow/infinity),
[documentation](https://infiniflow.org/docs/dev/) and
[Discord](https://discord.gg/jEfRUwEYEV).
