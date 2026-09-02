# Decomposing `SearchLayer` (Apple Silicon, M3 Pro)

`KnnHnswBase::SearchLayer` is where HNSW index build spends its time, and a sampling profile
attributes most of that to SearchLayer's **own** frame. That number is useless on its own, because
SearchLayer is an inlining blob: `commit_candidate`, the frontier heap's push/pop, the visited bit
test, `GetNeighbors`, and the prefetch issue loop are all inside it. Two prior optimisation attempts
failed by acting on a hypothesis about the blob without opening it.

This document records the decomposition, what it showed, and every hypothesis it killed. Read the
"refuted" section before proposing anything here — most of the obvious ideas are already dead, with
numbers.

## The profile, and what it does and does not license

`sample <pid> 40 1` on the 1M build at efConstruction=250, 12 threads. Top-of-stack, active samples
only (13 threads x 29,060 samples = 377,780 total; waits — `__psynch_cvwait` 44,500,
`__psynch_mutexwait` 496, `__ulock_wait2` 249, `__ulock_wait` 10 — total 45,255; active 332,525):

| frame | samples | % of active |
| --- | ---: | ---: |
| `SearchLayer` (self) | 228,989 | 68.9% |
| `F32L2SSEBatch4` | 37,554 | 11.3% |
| `F32L2SSE` | 29,391 | 8.8% |
| `SelectNeighborsHeuristic<false>` | 12,932 | 3.9% |
| `HeapResultHandler::AddResult` | 12,862 | 3.9% |
| all lock traffic | 2,405 | 0.7% |
| all malloc paths | ~1,500 | 0.5% |
| `__bzero` + `_platform_memset` | 1,083 | 0.3% |

This **inverts** the pre-`30dd58cbe` profile (53% kernels / 38% SearchLayer self). Note the shares
are not directly comparable across the two: `30dd58cbe` cut absolute build time by 1.38x, so a
share can rise while its absolute cost falls.

Two cautions on this profile, both established rather than assumed:

- **`sample` is a wall-clock, all-thread sampler**, not an on-CPU one. Proof: every thread —
  including the main thread, which is blocked for the entire build — has an identical 29,060
  samples. So "SearchLayer self" blends real work, memory-stall cycles smeared across the frame,
  and involuntary off-CPU time from a machine with a corporate-agent load floor. It identifies the
  target surface; it does not measure a recoverable quantity.
- The 40 s window captured ~29 s of a ~47 s build.

## The decomposition: counts, not times

Timing sub-regions from inside the loop is not available — the only cheap clock is `CNTVCT_EL0` at
24 MHz (41.7 ns per tick), coarser than the operations being separated. So the traversal is
instrumented with exact per-thread counters instead, and the costs are reasoned from those.

Build with `-DINFINITY_HNSW_INSTRUMENT=ON` in a **separate** build directory (it perturbs time, so
an instrumented binary must never be used for a timing measurement):

```sh
cmake -G Ninja -S tools/apple_silicon/native_hnsw_smoke -B build/bench-instr \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DINFINITY_NATIVE_CTPL_INCLUDE_DIR=$PWD/vcpkg_installed/arm64-osx/include \
  -DINFINITY_HNSW_INSTRUMENT=ON
cmake --build build/bench-instr --target infinity_hnsw_d0
build/bench-instr/infinity_hnsw_d0 /tmp/d0test/sift-n100000-d128.bin \
  100000 128 32 200 256 8192 256 1 1 /tmp/out.bin | grep infinity_instr_
```

The counters cannot change the graph — nothing they touch feeds a distance, a comparison, or a
branch that decides an edge — and every instrumented run re-proves that by still printing
`infinity_graph_sha256=50c8ffda...`.

Measured at n=100,000, efConstruction=200, `participants=1` (deterministic):

| quantity | value | note |
| --- | ---: | --- |
| `SearchLayer` calls | 103,303 | 99,999 of them at layer 0 |
| pops per call | **199.7** | replaces a guess of ~300; it tracks efConstruction |
| neighbours seen | 521,895,571 | 25.3 per pop, matching mean degree 25.3 |
| skipped, already visited | **297,570,050** | **57.0% of all neighbour visits** |
| skipped, out of range | 0 | |
| prefetches issued for a then-skipped vector | **297,570,050** | **57.0% of prefetch traffic, wasted** |
| distance evaluations | 224,325,521 | |
| ... via the 4-wide kernel | 194,063,636 lanes | 48,515,909 `Batch4` calls |
| ... via the 1-wide kernel | **30,261,885** | **13.5%**, all of it the `<4` pending tail |
| `commit_candidate` rejected | 180,640,332 | 80.5% of distance evals commit nothing |
| frontier pushes / pops | 43,788,492 / 20,632,543 | max heap size 734 |
| sift-down depth summed over pops | 177,580,077 | mean 8.6 levels |
| visited bitmap allocated+zeroed | 1.29 GB total | 12,500 B per call at this `n` |

Three things fall straight out of this table.

1. **The prefetch cursor runs ahead of the visited test**, so 57% of prefetched vectors are
   discarded without a byte being read. Fixed — see below.
2. **The batch tail is not a rare remainder.** Because 57% of neighbours are skipped, four
   unvisited candidates accumulate slowly, and 13.5% of all distance work lands on the 1-wide
   kernel.
3. **`pops_per_call` ~= efConstruction**, so the traversal cost scales with efC roughly linearly —
   which is why the iso-recall operating point (see [BASELINE.md](BASELINE.md)) is worth as much as
   any micro-optimisation here.

## What worked

### Do not prefetch a candidate that is already visited (kept, ON by default)

`prefetch_drain` now tests the visited bit before issuing the hint. The test is read-only and a
prefetch has no semantics, so the graph is unchanged and the determinism gate proves it exactly
rather than approximately. Prefetch calls fall 521,895,571 -> 224,325,521, i.e. exactly the wasted
count, leaving prefetch issue matched one-to-one with demand.

Paired A/B, 1M, efConstruction=250, 12 threads, 5 randomized blocks: **0.9626 of the previous build
time, 95% CI [0.9577, 0.9676]**. An earlier 5-block scan of the same change measured 0.9572,
CI [0.9368, 0.9780] — consistent.

### The prefetch stride was wrong for this machine (kept)

`SIMDPrefetchRange` walked the vector in 64-byte steps. `sysctl hw.cachelinesize` is **128** on
Apple M-series, so a 512-byte embedding issued eight `prfm` covering four physical lines: half of
them were redundant hits on a line fill already in flight.

This is worth recording as a lesson, because the naive estimate said it did not matter. Counting
instructions, halving the `prfm` count saves a few tenths of a percent of build time — below what
the A/B can resolve. Measured, it is far larger, which says the cost of a redundant prefetch is not
its issue slot but the load/store-unit resource it occupies.

It also explains why the `prefetch_step_` sweep below found nothing: that knob varies how many
*vectors* are prefetched ahead, while every vector always cost eight `prfm`. The binding quantity
was `prfm` **per vector**, which no setting of that knob could reach.

## What was refuted (do not re-attempt without new evidence)

Everything here was measured on this machine at the real operating point.

| hypothesis | result |
| --- | --- |
| The 512-`prfm` burst back-pressures the issue stream, so a shallower pipeline wins. `prefetch_step_` swept 64/16/8/4/2/1 at 1M, efC=250, 3 paired blocks. | **Refuted.** Every step from 2 to 64 is indistinguishable (best point estimate 0.9814, CI [0.9332, 1.0322]). Only step=1 differs, at **1.2133x SLOWER** — which also proves the campaign could have resolved a 10% effect, so the null is real and not underpower. |
| A rolling prefetch window (constant lookahead, advance one vector per neighbour) beats the front-loaded burst. Implemented as `prefetch_per_iter_`; arm `step=8, per_iter=1`. | **Refuted.** 1.0349, CI [1.0036, 1.0672] — significantly *slower*. Knob retained to document this. |
| Send the `<4` batch tail through the 4-wide kernel with padded lanes (bit-identical, since `F32L2SSEBatch4` is per-lane bit-identical to `F32L2SSE` — `native_simd_l2_smoke` asserts it over 69,824 outputs). Converts 30.3M 1-wide calls into 15.3M 4-wide calls. | **No effect.** 1.0060, CI [0.9907, 1.0216]. Knob retained, default OFF. |
| `SearchLayerNearest`'s greedy descent has no prefetch at all, so adding one should help. | **Not resolvable.** Stacked on skip-visited it moves the ratio from 0.9626 to 0.9597, i.e. ~0.3%, with overlapping CIs. Knob retained, default OFF. |
| Small-`n` prefetch tuning transfers to 1M. | **Refuted, and a trap.** `step=4` is 4.2% faster at n=100,000/efC=200 (CI [0.9456, 0.9701], significant) and within noise at 1M/efC=250. Validate at 1M or be misled. |

Previously refuted elsewhere and still refuted: hand-written NEON kernels, `Batch8`/`Batch16`
widening, FAISS `prune_headroom`, the level-0 double forward budget, a persistent per-thread visited
table, reserving the frontier vector, re-batching the reciprocal phase, and Accelerate GEMM.

## What is left, and how big it can be

The counters bound the remaining levers, and the arithmetic is unforgiving: to save X% of total
build time, a change to a region that is R% of it must remove X/R of that region.

- `HeapResultHandler::AddResult` is 3.9% of active time **as its own frame** — it is not inlined.
- `SelectNeighborsHeuristic<false>` is 3.9%, plus a share of the 1-wide kernel's 8.8%. It has a
  half-built `template<bool EnableBatch4>` scaffold whose batch branch is unimplemented. Careful:
  its scalar loop breaks at the first `cr_dist < c_dist`, and candidates are nearest-first, so it
  usually breaks early — batching can waste up to three distances and go negative.
- The frontier heap: 20.6M pops at a mean sift-down depth of 8.6 over a ~200-element,
  L1-resident array. A 4-ary heap would halve the depth. Because every `(-dist, idx)` pair in the
  frontier is unique (a vertex is pushed at most once per call, guarded by `visited`), **any**
  correct priority queue pops the same sequence — so a replacement is bit-identity-safe and can be
  gated by the determinism hash rather than the recall gate.
- The 80.5% `commit_candidate` rejection rate is not addressable bit-identically: a vertex is
  visited once, and `SelectNeighborsHeuristic` consumes carried-in distances, so computing fewer
  distances means a different graph.
- The only large remaining memory lever is **reducing scatter** — co-locating vectors with
  adjacency so candidate reads are not random over a 512 MB store. Prefetch cannot hide the
  residual (30.1 ns prefetched-scattered against 9.2 ns sequential). This perturbs layout and needs
  fresh hash and recall gates.

## Measurement rules this work established

- **Never report a build-time win without a paired confidence interval.** A control-arm spread of
  11% across identical code has been observed on this host; "the median improved" is not evidence.
- **Never run agents, editors, or a second campaign during a timing measurement.** A scan whose
  block 0 overlapped background work put the control arm at 48.9 s against 46.6 s in later blocks,
  which alone produced an apparent 4.4% "win" that vanished on a quiet machine.
- **Prefer one binary with two environment arms** over two binaries, when the change can be a
  runtime knob: it removes codegen and code layout as confounds entirely. `ab_build.py` takes
  `--control-env`/`--candidate-env`, and `knob_scan.py` scans many arms in one paired campaign.
- **Gate on the graph hash for anything claimed to be semantics-free.** A prefetch or layout change
  that alters the hash is not what it claims to be.
