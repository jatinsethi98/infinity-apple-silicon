# Infinity vs FAISS — HNSW index-build baseline (Apple Silicon)

> **This is the lab notebook of the optimization campaign on the M3 Pro (August to 2 September
> 2026), kept in full including the sections later corrected.** For the current published numbers,
> the method, and reproduction commands see [BENCHMARKS.md](BENCHMARKS.md). Paths in the commands
> below refer to that machine; on a fresh checkout use `datasets/sift1m/base.f32` from
> `scripts/bench/fetch_datasets.py` and `build/bench-faiss-src/faiss_hnsw_d0` from
> `scripts/apple_silicon/build_faiss_accelerate.sh`.

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

**Superseded 2026-09-02 (later the same day) — see "Iso-recall, on published ground truth" below.
The 1.344x figure is now understood to be an artifact of the recall instrument, not a property of
the two engines. The current matched-recall figure is 1.523x.**

That profile has been re-taken and decomposed; the 53%/38% split above is stale (it predates the
whole-vector prefetch fix) and the decomposition superseded the guesses built on it. See
[SEARCHLAYER_DECOMPOSITION.md](SEARCHLAYER_DECOMPOSITION.md) for the current profile, the exact
traversal counts, the two changes that came out of it, and the five hypotheses it killed.

## Reproduce every number (one command each)

Prerequisites, once. These commands were originally recorded against an absolute
dataset path on the author's machine; they now use the repository-relative layout that
`fetch_datasets.py` produces, so they run from a fresh clone.

```bash
# Binaries: the bench preset gives both arms, and build_faiss_accelerate.sh gives
# the Accelerate-linked FAISS the campaign actually compared against.
cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke
cmake --build build/bench --target infinity_hnsw_d0 faiss_hnsw_d0
scripts/apple_silicon/build_faiss_accelerate.sh

# SIFT1M into ./datasets/sift1m/
python3 scripts/bench/fetch_datasets.py sift1m
export HNSW_D0_EXTERNAL_QUERIES=$PWD/datasets/sift1m/query.f32
export HNSW_D0_EXTERNAL_GROUNDTRUTH=$PWD/datasets/sift1m/groundtruth.i32

# The 200k arm is the first 102,400,000 bytes (200,000 x 128 f32) of base.f32.
mkdir -p datasets/synth
head -c 102400000 datasets/sift1m/base.f32 > datasets/synth/sift1m-base-200k-d128.f32
```

`--participants 12` below is the thread count these runs were measured at, on a 12-core
M3 Pro. Keep it to compare against the numbers in this document; use
`$(sysctl -n hw.ncpu)` to characterise your own machine instead.

```bash
# Task 1 — FAISS Homebrew vs Accelerate A/B (12,288×128)
# The synthetic dataset is generated deterministically by run_baseline.py's
# --dataset default if it is absent, so this path is created on first use.
python3 scripts/bench/faiss_ab.py \
  --homebrew-bin build/bench/faiss_hnsw_d0 --fromsrc-bin build/bench-faiss-src/faiss_hnsw_d0 \
  --dataset datasets/synth/d0-f32le-n12288-d128-seed0.bin --n 12288 --d 128 \
  --m 32 --efc 200 --participants 12 --runs 5

# Task 2 — SIFT1M equal-param baseline
# For the 200k arm: --n 200000 --dataset datasets/synth/sift1m-base-200k-d128.f32
python3 scripts/bench/run_baseline.py \
  --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 \
  --m 32 --efc 200 --ef 32,64,128,256,512 --participants 12 --pairs 3 \
  --chunk-size 8192 --query-count 1000 --build-grain 1 --timeout 1800 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0

# Task 3 — 1M iso-recall sweep (FAISS fixed @200, Infinity swept)
python3 scripts/bench/iso_recall.py \
  --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 \
  --m 32 --faiss-efc 200 --infinity-efc 250,300,400 --participants 12 --pairs 2 \
  --query-count 1000 --build-grain 1 --timeout 1800 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0
```
Every `iso_recall.py` row is independently reproducible with `run_baseline.py`: FAISS@200 is
the FAISS row of the `--efc 200` run; Infinity@X is the Infinity row of an `--efc X` run.

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

## Iso-recall, on published ground truth (2026-09-02, supersedes Task 3)

Two independent things were wrong with the 1.344x figure, and they were worth about the same
amount.

### 1. The recall instrument was wrong, so the operating point was wrong

Every recall figure above comes from the harness self-audit: **64 synthetic queries**, uniform in
`[0,1)` per coordinate, truth derived by exhaustive search. It is exact and deterministic, but it
is out-of-distribution for SIFT (canonical descriptors are integer-valued on a scale of tens) and
it is coarse: recall@10 over 64 queries has only 640 neighbour slots, so its finest step is
1/640 = 0.0016 and one query's full result set is worth 0.0156. `AuditHnswD0RecallExternal`, which scores against the
**published** SIFT1M 10,000-query set and ground truth, had existed and been unit-tested since
`fcd910c5c` but was never called by the benchmark. It is wired in now.

Both engines scored by the same code over the same 10,000 published queries (identical
`queries_sha256` and `groundtruth_sha256` in every run), n=1,000,000, M=32, 12 threads, 3 builds
per point:

| efSearch | Infinity efC=200 | FAISS efC=200 | Infinity − FAISS | the self-audit said |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.94187 | 0.94328 | −0.00141 | — |
| 64 | 0.98231 | 0.98314 | −0.00083 | −0.0172 |
| 128 | 0.99560 | 0.99605 | −0.00045 | −0.0266 |
| 256 | 0.99925 | 0.99930 | −0.00005 | — |
| 512 | 0.99996 | 0.99995 | +0.00001 | — |

The real deficit at equal parameters is **0.0005–0.0014**, twenty to sixty times smaller than the
self-audit reported. Build-to-build variance shrinks the same way: ~0.0005 over the published
queries, against a 0.041 spread over 35 recorded builds on the 64-query audit.

Sweeping Infinity's efConstruction against FAISS@200 on published truth (3 builds per point;
efSearch 512 is excluded from the decision because both engines exceed 0.9999 there, where
0.00001 is one query in 10,000):

| Infinity efC | ef32 | ef64 | ef128 | ef256 | median ≥ FAISS at all deciding points |
| ---: | ---: | ---: | ---: | ---: | :--: |
| 200 | −0.00141 | −0.00083 | −0.00045 | −0.00005 | no |
| 210 | −0.00009 | −0.00037 | −0.00028 | −0.00003 | no |
| 225 | +0.00062 | +0.00011 | −0.00001 | +0.00004 | no (short by 0.1 query per 10,000) |
| 230 | +0.00120 | +0.00043 | −0.00002 | +0.00006 | no |
| **235** | **+0.00129** | **+0.00059** | **+0.00000** | **+0.00004** | **YES** |
| 240 | +0.00196 | +0.00039 | +0.00008 | +0.00008 | YES |

**The iso-recall point is efConstruction=235, not 250** — under the same rule the previous
headline used, namely median recall ≥ FAISS at every deciding efSearch.

That rule is not the only defensible one, and the answer depends on which is chosen, so both are
reported. Re-measured with 5–6 independent builds per point (the differences below are between
independent build draws, so a Welch t on the per-build means is the right test; the same 10,000
queries score both engines, so query-sampling error is common and cancels):

| vs FAISS efC=200 | ef32 | ef64 | ef128 | ef256 |
| --- | ---: | ---: | ---: | ---: |
| Infinity efC=235 | +0.00119 (t=+8.2) | +0.00035 (t=+3.7) | +0.00006 (**t=+1.5, tie**) | +0.00006 (t=+3.9) |
| Infinity efC=240 | +0.00162 (t=+8.2) | +0.00036 (t=+6.4) | +0.00014 (t=+3.7) | +0.00009 (t=+6.1) |

So **efC=235 is never significantly lower** than FAISS, but its ef128 margin is a statistical tie
rather than a win; **efC=240 is significantly higher at every non-saturated point.** A single-build
comparison at efC=235 can therefore land on either side at ef128 — one did, during a verification
run — which is exactly why the medians above are over 3+ builds. The remaining differences at the
crossing are on the order of 10 neighbour slots in 100,000.

### 2. Two bit-identical build-time fixes

Both came out of instrumenting `SearchLayer`'s traversal
([SEARCHLAYER_DECOMPOSITION.md](SEARCHLAYER_DECOMPOSITION.md)), and both leave the graph
bit-identical (`infinity_graph_sha256=50c8ffda...` at the n=100,000 determinism gate):

| change | paired A/B | blocks |
| --- | --- | ---: |
| Do not prefetch a candidate already in the visited set (57.0% of prefetched candidates were being discarded unread) | **0.9626** of previous, 95% CI [0.9577, 0.9676] | 5 |
| `SIMDPrefetchRange` stride 64 → 128 bytes (`sysctl hw.cachelinesize` is 128 on Apple M-series, so half the `prfm` were redundant) | **0.9723** of previous, 95% CI [0.9668, 0.9780] | 15 |

### The result

One paired campaign, 6 randomized blocks, quiet machine, all five arms measured against each other
so every ratio below is paired within the same thermal state. FAISS is the from-source
Accelerate-linked build (`build/bench-faiss-src/faiss_hnsw_d0`).

| arm | median build | rel MAD | ratio vs FAISS@200 | speedup | 95% CI on speedup |
| --- | ---: | ---: | ---: | ---: | --- |
| FAISS efC=200 | 59.955 s | 0.28% | 1.0000 | — | — |
| **Infinity efC=235, both fixes** | **39.366 s** | 0.37% | 0.6565 | **1.523x** | **[1.514x, 1.532x]** |
| Infinity efC=225, both fixes | 37.800 s | 0.49% | 0.6309 | 1.585x | [1.575x, 1.595x] |
| Infinity efC=235, original code | 42.572 s | 0.33% | 0.7109 | 1.407x | [1.396x, 1.418x] |
| Infinity efC=250, original code | 45.809 s | 0.14% | 0.7621 | 1.312x | [1.306x, 1.319x] |

The last row reproduces the previous headline (1.312x here against 1.344x recorded earlier, a
different day and a busier machine), which is what licenses reading the others as a change rather
than as a new measurement.

A second, independent 5-block campaign the same morning adds the efC=240 point and reproduces
efC=235:

| arm | median build | ratio vs FAISS@200 | speedup | 95% CI on speedup |
| --- | ---: | ---: | ---: | --- |
| FAISS efC=200 | 59.705 s | 1.0000 | — | — |
| Infinity efC=235 | 39.236 s | 0.6617 | 1.512x | [1.482x, 1.541x] |
| Infinity efC=240 | 40.169 s | 0.6732 | 1.486x | [1.470x, 1.501x] |

**Iso-recall result, stated against both parity criteria:**

- **Median recall ≥ FAISS at every deciding efSearch** (the rule the previous 1.344x used, so this
  is the like-for-like number): efC=235 → **1.523x**, 95% CI [1.514x, 1.532x], reproduced at
  1.512x [1.482x, 1.541x] in a second campaign. **The >=1.5x goal is met.**
- **Recall significantly higher at every deciding efSearch** (a stricter rule than any previous
  number here was held to): efC=240 → **1.486x**, 95% CI [1.470x, 1.501x]. Essentially at 1.5x but
  not above it.

Quote whichever criterion you state. The honest one-line summary is that Infinity builds SIFT1M
about **1.49x–1.52x** faster than FAISS at matched published-truth recall, up from 1.344x, and which
end of that you land on is a question about the parity rule rather than about either engine.

Where the 13.85% came from, each ratio paired within that one campaign:

| contribution | ratio | improvement | 95% CI |
| --- | ---: | ---: | --- |
| operating point efC 250 → 235 (the recall-instrument fix) | 0.9328 | 6.72% | [5.74%, 7.70%] |
| the two code fixes plus incidental codegen, at fixed efC=235 | 0.9235 | 7.65% | [7.39%, 7.90%] |
| **total, efC=250 original → efC=235 with both fixes** | **0.8615** | **13.85%** | **[13.16%, 14.54%]** |

The two halves are almost exactly equal, which is the summary of the whole exercise: half the win
was in the code and half was in the ruler.

**One caveat on the middle row.** Its "original code" arm is the current source with the knob turned
off, not a build of `c13dedb73`, so it also carries whatever incidental difference the two builds
have — not-taken branches for the retained knobs, object layout, code layout. The two attributions
that are clean are the individual A/Bs above: 0.9626 for skip-visited and 0.9723 for the stride,
which compose to 0.9359 (**6.4%**). Read 6.4% as the attributable code win and 7.65% as the
end-to-end delta between two real builds; the gap between them is the incidental part. Closing it
would need a commit-pinned three-arm campaign against `c13dedb73` itself.

### Reproduce

```bash
# Recall parity on published ground truth (both engines, same audit code, same queries)
export HNSW_D0_EXTERNAL_QUERIES=$PWD/../datasets/sift1m/query.f32
export HNSW_D0_EXTERNAL_GROUNDTRUTH=$PWD/../datasets/sift1m/groundtruth.i32
python3 scripts/bench/iso_recall.py \
  --dataset $PWD/../datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 \
  --faiss-efc 200 --infinity-efc 200,225,235,240 --participants 12 --pairs 3 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0

# The headline build-time campaign (audit OFF: unset the two variables above, so a 10k-query
# audit cannot warm the machine before the next arm's build). All five arms in one campaign, which
# is what makes every ratio in the contribution table paired within one thermal state.
unset HNSW_D0_EXTERNAL_QUERIES HNSW_D0_EXTERNAL_GROUNDTRUTH
python3 scripts/bench/knob_scan.py \
  --dataset $PWD/../datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 \
  --participants 12 --blocks 6 --require-ac --label headline-isorecall \
  --arm "faiss200:bin=build/bench-faiss-src/faiss_hnsw_d0,efc=200" \
  --arm "inf235_new:bin=build/bench/infinity_hnsw_d0,efc=235" \
  --arm "inf225_new:bin=build/bench/infinity_hnsw_d0,efc=225" \
  --arm "inf235_orig:bin=build/bench/infinity_hnsw_d0,efc=235,INFINITY_HNSW_PREFETCH_SKIP_VISITED=0" \
  --arm "inf250_orig:bin=build/bench/infinity_hnsw_d0,efc=250,INFINITY_HNSW_PREFETCH_SKIP_VISITED=0" \
  --reference faiss200
```

Note that reproducing the *stride* row of the contribution table needs two binaries, since the
stride is a compile-time constant: build once with `kCacheLineBytes = 64` in
`src/common/simd/simd_functions.h`, save that executable, restore, rebuild, then
`ab_build.py --control-bin <the 64 build> --candidate-bin <the 128 build>`.

### Limitations specific to these numbers

- **One host, one session.** Every arm above was measured in the same campaign on a quiet machine,
  so the ratios are internally paired and comparable; the absolute seconds are not portable.
- **`participants=12` builds are order-nondeterministic**, so each arm's graph differs run to run.
  Recall is therefore reported as a median over repeated builds, and the bit-identity gate is run
  separately at `participants=1`.
- **Recall parity is decided at efSearch 32/64/128/256** and is a median over 3 builds per point.
  The residual differences at the crossing are on the order of one query in 10,000; a per-query
  paired equivalence test over more build pairs would be the next tightening.
- **The published-truth audit only runs at n=1,000,000**, by design — the published IDs address the
  full canonical base, so a prefix would silently address the wrong rows.
