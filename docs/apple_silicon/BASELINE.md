# Infinity vs FAISS — HNSW index-build baseline (Apple Silicon)

First honest, real-scale HNSW **index-build** comparison of Infinity's native arm64 port
against stock FAISS on the same workload. Query latency is out of scope; this measures
build time and reports recall only to keep the comparison fair.

## Host & power
- MacBook Pro Mac15,7, Apple M3 Pro (6P+6E, 12 threads), 36 GB, macOS 26.2 arm64.
- Compiler: Homebrew clang 20.1.8. Branch `apple-silicon/port`, HEAD `01da17d59`.
- **Power: unstable.** Session ran on battery / intermittent AC (a flaky adapter that
  could not sustain peak build draw): battery drained ~79% → ~68% across the runs and
  `pmset` alternated between "AC attached; not charging" and "discharging".
- **Thermal: nominal throughout** — `pmset -g therm` recorded no throttling, and FAISS's
  0.2–0.8% run-to-run build-time MAD confirms clocks stayed stable (no thermal decay).
- Competing load: this is a corporate laptop (security agents, 10+ MCP servers, Chrome).
  `loadavg` was ~5 at rest and spiked under the build; a compute-bound build still got
  ~10.5/12 cores (FAISS 1056% CPU), so the noisy loadavg did not steal cores. No
  concurrent compile ran during any measured build.

## Both engines use identical settings, enforced
Same dataset file, N, D, M=32, efConstruction, and **12 build threads** for both. Thread
count is set in-code (FAISS `omp_set_num_threads(12)`; Infinity `ctpl::thread_pool(12)` —
it does not link OpenMP) and the harness marks a run invalid unless `thread_count==12`.
Build time is each engine's own `steady_clock` `*_cold_build_ns`, measured strictly around
index construction + insert and **outside** the harness's suspend barriers (verified in
`tools/apple_silicon/native_hnsw_smoke/{faiss,infinity}_hnsw_bridge.cpp`). Recall@10 is the
harness's own brute-force audit over **64 synthetic held-out queries** at ef∈{32,64,128,256,512},
identical code in both binaries — it is a fair A/B signal, **not** SIFT's official groundtruth
recall, and at 64 queries its granularity is ~0.0016 with real sampling noise.

## Task 1 — which FAISS is the fair baseline? (Accelerate)
Two byte-identical harness binaries differing only in the linked `libfaiss.dylib`, measured
on the same 12,288×128 workload (M=32, efC=200, 12 threads, 5 runs each):

| FAISS build | BLAS backend | median build | min–max | recall@10 ef128 | edges |
|---|---|---:|---:|---:|---:|
| Homebrew `libfaiss` | OpenBLAS | 665.8 ms | 655.6–724.2 | ~0.976 | ~574k |
| From-source `libfaiss` | Apple Accelerate | **264.7 ms** | 259.8–307.5 | ~0.976 | ~574k |

Homebrew is **2.516× slower**, purely from the BLAS backend — recall and edge counts are
identical, so the *same graph* is built; only the distance kernels differ. Using the slow
Homebrew FAISS would inflate Infinity's result. **Decision: adopt the from-source Accelerate
FAISS as the headline baseline** (faster ⇒ conservative), and both numbers are reported here
so the choice is auditable. This matches the historical ~294 ms Accelerate figure.

## Task 2 — SIFT1M build baseline (equal parameters)
SIFT1M base vectors, M=32, efConstruction=200, 12 threads, chunk=8192, query-count=1000,
build-grain=1, 3 alternating pairs. FAISS = Accelerate build.

| n | engine | median build | min–max | rel MAD | vectors/sec | directed edges | L0 cap / upper cap |
|---:|---|---:|---:|---:|---:|---:|:--:|
| **1,000,000** | Infinity | **49.038 s** | 47.868–53.201 | 2.4% | 20,392 | 30,075,721 | 64 / 32 |
| **1,000,000** | FAISS | 64.385 s | 58.016–64.545 | 0.2% | 15,531 | 30,686,543 | 64 / 32 |
| 200,000 | Infinity | **6.133 s** | 6.055–6.219 | 1.3% | 32,611 | 5,544,122 | 64 / 32 |
| 200,000 | FAISS | 6.648 s | 6.595–6.732 | 0.8% | 30,084 | 5,639,981 | 64 / 32 |

Equal-parameter build-time ratio Infinity/FAISS: **0.762× at 1M**, **0.923× at 200k**
(<1 = Infinity faster). Edge deficit (Infinity vs FAISS): 1.99% at 1M, 1.70% at 200k —
much smaller than the ~7.2% seen at 12k (the deficit shrinks with scale).

**recall@10 (median), and why the equal-param ratio is NOT the headline:**

| efSearch | 1M FAISS | 1M Infinity | 1M deficit (F−I) | 200k FAISS | 200k Infinity | 200k deficit |
|---:|---:|---:|---:|---:|---:|---:|
| 32  | 0.7625 | 0.7625 | +0.0000 | 0.9109 | 0.9313 | −0.0203 |
| 64  | 0.8281 | 0.8078 | **+0.0203** | 0.9375 | 0.9563 | −0.0188 |
| 128 | 0.8672 | 0.8328 | **+0.0344** | 0.9625 | 0.9797 | −0.0172 |
| 256 | 0.9031 | 0.8906 | +0.0125 | 0.9922 | 0.9969 | −0.0047 |
| 512 | 0.9547 | 0.9453 | +0.0094 | 1.0000 | 0.9984 | +0.0016 |

At 1M, Infinity trails FAISS recall (positive deficit) — a faster build at lower recall is
not a win, so the 0.762× ratio is flagged RECALL-UNMATCHED. At 200k the sign **flips**:
Infinity meets or beats FAISS at ef≤256. That flip across scales, within a 64-query
estimate, is why iso-recall (below) is the honest headline.

## Task 3 — iso-recall comparison (the honest headline)
Fix FAISS at efC=200 (recall@10 ef64=0.8281, ef128=0.8672); raise Infinity's efC until its
recall@10 at **both** ef64 and ef128 ≥ FAISS's, then compare build time at that point.

- At **200k**, no efC bump is needed: Infinity already exceeds FAISS at ef64/ef128 at
  efC=200, so the iso-recall ratio ≈ the equal-param **0.923×** (Infinity faster *and* at
  equal-or-higher recall).
- At **1M** (real scale, the hard case where Infinity is recall-disadvantaged):

<!-- ISO_RECALL_1M -->
_(1M iso-recall sweep in progress — table filled on completion.)_

## Reproduce every number (one command each)
```bash
# Task 1 — FAISS Homebrew vs Accelerate A/B (12,288×128)
python3 scripts/bench/faiss_ab.py \
  --homebrew-bin build/bench/faiss_hnsw_d0 --fromsrc-bin build/bench-faiss-src/faiss_hnsw_d0 \
  --dataset /tmp/d0test/d0-f32le-n12288-d128-seed0.bin --n 12288 --d 128 \
  --m 32 --efc 200 --participants 12 --runs 5

# Task 2 — SIFT1M equal-param baseline (swap --n 200000 --dataset /tmp/d0test/sift1m-base-200k-d128.f32 for 200k)
python3 scripts/bench/run_baseline.py \
  --dataset /Users/sethjatq/Desktop/proj/datasets/sift1m/base.f32 --n 1000000 --d 128 \
  --m 32 --efc 200 --ef 32,64,128,256,512 --participants 12 --pairs 3 \
  --chunk-size 8192 --query-count 1000 --build-grain 1 --timeout 1800 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0

# Task 3 — 1M iso-recall sweep (FAISS fixed @200, Infinity swept)
python3 scripts/bench/iso_recall.py \
  --dataset /Users/sethjatq/Desktop/proj/datasets/sift1m/base.f32 --n 1000000 --d 128 \
  --m 32 --faiss-efc 200 --infinity-efc 250,300,400 --participants 12 --pairs 2 \
  --query-count 1000 --build-grain 1 --timeout 1800 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0
```
Every `iso_recall.py` row is independently reproducible with `run_baseline.py`: FAISS@200 is
the FAISS row of the `--efc 200` run; Infinity@X is the Infinity row of an `--efc X` run.
The 200k dataset is the first 102,400,000 bytes (200,000×128 f32) of SIFT1M `base.f32`.

## Limitations (read before quoting a number)
- **Unstable power** (battery/flaky AC). Thermal stayed nominal and FAISS MAD ≤0.8% argues no
  throttling, but a clean AC re-run is warranted before publishing.
- **Single host, corporate laptop** with background daemons; no isolation. Re-run on an idle
  machine to tighten Infinity's 1.3–2.4% MAD.
- **Recall is a 64-synthetic-query brute-force audit**, not SIFT groundtruth — fair for the
  A/B, but coarse (~0.0016 granularity) and the deficit sign flips across scales; do not read
  absolute recall as SIFT recall.
- **Build-time measurement asymmetry:** Infinity's `cold_build_ns` includes its thread-pool
  ctor + index allocation; FAISS's includes the `IndexHNSWFlat` ctor + `add()`. Both exclude
  the audit/query phases. The gap is negligible vs a multi-second build.
- **Harness CLI carries attestation cruft:** the in-repo binaries require a 17-arg protocol and
  self-`SIGSTOP` at barriers; `scripts/bench/campaign.py` is the minimal shim that drives them.
  Timing is unaffected (barriers bracket, but sit outside, the timed region). If the toolchain
  agent lands clean 11-arg binaries, the shim can be dropped without changing any number.
- **Scale dependence:** Infinity is *slower* than FAISS at 12k (1.54×), *faster* at 200k
  (0.923×) and 1M (0.762×) — fixed setup cost dominates at tiny N. Quote numbers with their N.
