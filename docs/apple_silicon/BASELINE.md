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

## Task 2 — SIFT1M build baseline
SIFT1M base vectors, M=32, 12 threads, chunk=8192, query-count=1000, build-grain=1,
2-3 alternating pairs per point. FAISS = from-source Accelerate build.

| efConstruction | Infinity median | FAISS median | ratio Inf/FAISS | Infinity rel MAD | FAISS rel MAD | recall verdict at matched efSearch |
|---:|---:|---:|---:|---:|---:|---|
| 200 | **49.038 s** | 64.385 s | **0.762x** | 2.4% | 0.2% | Infinity LOWER (-0.020 @ef64, -0.034 @ef128) |
| 225 | 66.210 s | 80.183 s | 0.826x | 1.7% | 0.1% | mixed (+0.006 @ef64, -0.012 @ef128) |
| 250 | 75.834 s | 85.600 s | 0.886x | 5.2% | 0.6% | Infinity HIGHER at every ef (+0.008 .. +0.031) |
| 300 | 94.508 s | 104.270 s | 0.906x | 0.5% | 4.0% | Infinity HIGHER at every ef (+0.010 .. +0.029) |

Ratio < 1 means Infinity builds faster. **At equal parameters Infinity builds
1.10x-1.31x faster than FAISS at 1M scale**, and the margin narrows as
efConstruction rises. A 200,000-vector point gives 6.133 s vs 6.648 s (0.923x).

Directed-edge deficit (Infinity vs FAISS) at efC=200: 30,075,721 vs 30,686,543 =
**1.99% at 1M**, 1.70% at 200k -- far smaller than the 7.2% seen at 12k, because
reciprocal back-edges have more opportunity to fill in at scale.

**Scale dependence matters more than anything else here.** Infinity is *slower*
than FAISS at 12,288 vectors (~1.5x) and *faster* at 200k and 1M. Fixed setup
cost dominates at tiny N. The prior effort's headline "0.902x FAISS" was measured
at 12,288 vectors -- 1.2% of SIFT1M -- which is why it concluded Infinity was
behind. Always quote N.

## Task 3 — iso-recall comparison (the honest headline)
At equal efConstruction the two engines land at different recall@10, and a faster
build at lower recall is not a win — so **comparing build time at equal
efConstruction is not apples-to-apples.** Fix FAISS at efC=200 and raise Infinity's
efC until its recall@10 meets FAISS's at the same efSearch:

> **Correction (2026-09-02):** this section previously attributed the recall gap to
> Infinity's forward-edge budget of M per layer (vs FAISS's 2*M at level 0) making a
> sparser graph. Measurement refutes that mechanism — the two engines build graphs of
> essentially matched density, and the forward budget is not the binding constraint on
> degree. See the convention note in [README.md](README.md). **The iso-recall numbers
> below are unaffected**: they are direct measurements of build time at matched recall
> and do not depend on the explanation. What is no longer claimed is *why* the recall
> differs — that cause is currently unknown.

| point | Infinity build | vs FAISS@200 (64.385 s) | recall@10 ef64 | ef128 | meets FAISS@200 (0.8281 / 0.8672)? |
|---|---:|---:|---:|---:|:--|
| Infinity efC=200 | 49.038 s | 0.762x | 0.8078 | 0.8328 | no (short at both) |
| Infinity efC=225 | 66.210 s | **1.028x** | 0.8320 | 0.8539 | ef64 yes, ef128 short by 0.013 |
| Infinity efC=250 | 75.834 s | **1.178x** | 0.8531 | 0.8781 | **yes at every efSearch** |

**Iso-recall result: Infinity is between 1.03x and 1.18x SLOWER than FAISS at
matched recall** -- effectively at parity, bracketed by the last point that misses
recall parity and the first that achieves it. The 64-query recall estimate
(~0.0016 granularity, larger sampling noise) does not support a tighter claim; a
10,000-query recall harness would.

### Superseded 2026-09-02 by the whole-vector prefetch fix (commit 30dd58cbe)

`PlainVecStoreInnerBase::Prefetch` issued one `__builtin_prefetch` per candidate
vector, covering the first 64-byte cache line of a 512-byte (d=128) embedding and
leaving the other seven lines to stall. Prefetching the whole vector is 1.3796x on
index build (paired A/B, 15 blocks, 95% CI [0.7163, 0.7335]). Re-measured on the same
machine, same day, same Accelerate-linked FAISS reference:

| comparison | before | after | note |
|---|---:|---:|---|
| equal-param build (efC=200) | 0.813x | **0.590x** = 1.694x faster | recall-unmatched, not claimable |
| **iso-recall build** | 1.178x slower | **0.744x** = **1.344x faster** | Infinity efC=250 vs FAISS efC=200 |
| QPS (k=10, ef=256) | 1.092x | **1.364x** | same hint helps the query path |

Iso-recall detail: Infinity efC=250 builds in **47.680 s** against FAISS efC=200's
**64.087 s**, with recall@10 ef64 0.8547 >= 0.8266 and ef128 0.8805 >= 0.8711, so
parity is met at every efSearch. Equal-param detail: Infinity **37.944 s** (rel MAD
0.4%) vs FAISS **64.288 s** (rel MAD 0.4%); QPS 24,271 vs 17,799; p50 latency 275 us
vs 376 us.

**This is now the number to beat: 1.344x at matched recall.** Reaching the project's
>=1.5x goal needs a further ~10.4% off Infinity's build (47.680 s -> <=42.725 s).

Where that is likely to come from, per a 1M/12-thread profile of the current build:
53% of active worker time is in the L2 distance kernels and **38% is SearchLayer's own
control flow** (frontier heap operations, visited tests, neighbour scan), which nothing
has touched yet. Everything else is noise -- all lock traffic ~1%, all malloc ~0.5%,
`__bzero` 0.26%. Note the kernel is latency-bound, not FP-bound: scattered candidate
reads cost 64.8 ns per distance versus 9.2 ns sequential at identical arithmetic, which
is why fusing the multiply-add (48 -> 32 FP ops) measured as nothing. Widening the
kernel to 8 candidates in flight is worth only ~1.07x once whole-vector prefetch is in
place, so it is not the next lever.

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
- **Harness CLI:** the campaign binding is now optional -- current binaries accept a plain 11-arg
  invocation and no longer suspend themselves, so `run_baseline.py` drives them under a bare
  subprocess and only falls back to `campaign.py` for older prebuilt binaries. Verified identical
  results in both modes (same graph sha256).
- **Scale dependence:** see Task 2. Never quote a ratio without its N.
- **Iso-recall is bracketed, not pinned.** The 64-query recall audit is too coarse to locate the
  exact efConstruction where Infinity meets FAISS's recall. Wiring the official SIFT1M 10,000-query
  set and provided groundtruth (already downloaded to datasets/sift1m/query.f32 and
  groundtruth.i32, but not yet used by the harness) is the single highest-value improvement to
  this mechanism.
