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

Measured at the **target** workload, n=1,000,000, efConstruction=235, 12 participants — the
configuration every claim below is about. A second column gives n=100,000 / efC=200 /
`participants=1`, which is the deterministic configuration the graph-hash gate uses; the two differ
enough that the small one must not be quoted as a target-workload fact:

| quantity | 1M / efC=235 / 12 | 100k / efC=200 / 1 |
| --- | ---: | ---: |
| `SearchLayer` calls at layer 0 | 999,999 | 99,999 |
| pops per call | **235.3** | 199.7 |
| neighbours per pop | 31.6 | 25.3 |
| neighbours seen | 7,684,435,223 | 521,895,571 |
| skipped, already visited | 3,522,748,401 = **45.8%** | 297,570,050 = **57.0%** |
| skipped, out of range | 0 | 0 |
| distance evaluations | 4,161,686,822 | 224,325,521 |
| ... via the 1-wide kernel (the `<4` tail) | **8.7%** | **13.5%** |
| max frontier size | 1,087 | 734 |
| sift-down depth summed over pops | 2,195,933,469 | 177,580,077 |

`pops_per_call` tracks efConstruction almost exactly (235.3 at efC=235, 199.7 at efC=200), which is
the single most useful number here: traversal cost is close to linear in efC, so **where the
iso-recall operating point sits is worth as much as any micro-optimisation inside the loop.** That
is what the recall-instrument fix in [BASELINE.md](BASELINE.md) exploited.

Note also that the derived counters confirm the skip-visited fix does what it claims: with it on,
`prefetch_vec_calls` (4,161,686,822) equals `distance_evals` exactly — prefetch issue is matched
one-to-one with demand, with nothing issued speculatively.

Three things fall straight out of this table.

1. **The prefetch cursor runs ahead of the visited test**, so 45.8% of prefetched vectors were
   discarded without a byte being read. Fixed — see below.
2. **The batch tail is not a rare remainder.** Because nearly half of neighbours are skipped, four
   unvisited candidates accumulate slowly, and 8.7% of distance work lands on the 1-wide kernel.
3. **The frontier reaches 1,087 entries**, not the ~235 an efC-sized bound would suggest, and the
   summed sift-down depth is 2.196 billion levels over 243 million pops — a mean depth of 9.0.

## What worked

### Do not prefetch a candidate that is already visited (kept, ON by default)

`prefetch_drain` now tests the visited bit before issuing the hint. The test is read-only and a
prefetch has no semantics, so the graph is unchanged and the determinism gate proves it exactly
rather than approximately. At the target workload it suppresses 3,522,748,401 hints and leaves
4,161,686,822 — exactly the number of distance evaluations.

The filter is **exact**, not merely conservative: entries within one adjacency list are distinct and
`visited` is only written by the scan itself, so a candidate unvisited when its hint is issued
cannot become visited before the scan reaches it; and since `visited` is monotonic within a call, a
suppressed candidate is never subsequently read. It suppresses precisely the wasted hints.

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

Two caveats found in review, both now fixed in the code:

- The walk must start at the line the object BEGINS in. Stepping 128 from an unaligned base misses
  the last line of an object that straddles a boundary. At d=128 the backing allocation is
  page-aligned and the vector stride is 512 bytes, so every vector is already 128-aligned and the
  measured A/B above is unaffected — but it was wrong for any dimension whose vector size is not a
  multiple of the line, so the loop now aligns down.
- The 128 is a compile-time constant on `__APPLE__ && __aarch64__`, verified by `sysctl
  hw.cachelinesize` on this target. If some Apple arm64 core has 64-byte lines, a 128 stride there
  prefetches every other line — a throughput regression on that core, not a correctness problem,
  since prefetch is a hint and the demand loads are unaffected.

Both fixes were then measured to cost nothing at this workload: with the skip-visited knob pinned in
both arms so only the alignment and the bounds guard differ, 1.0037, CI [0.9876, 1.0201] over 4
paired blocks — indistinguishable, as the alignment argument predicts.

## What was refuted (do not re-attempt without new evidence)

Everything here was measured on this machine at the real operating point.

| hypothesis | evidence | verdict |
| --- | --- | --- |
| The `prfm` burst back-pressures the issue stream, so a shallower **vector** pipeline wins. `prefetch_step_` swept 64/16/8/4/2/1 at **1M, efC=250, 3 paired blocks**. | Every step from 2 to 64 indistinguishable; best point estimate 0.9814, CI [0.9332, 1.0322]. step=1 is **1.2133x SLOWER**, CI [1.1144, 1.3209]. | **Not supported at this resolution.** The campaign resolved only ~5% half-width (block 0 was contaminated by concurrent work), so it excludes a large win but *not* a 2-6% one. What it does establish is that lookahead depth is not where the 8-vs-4 `prfm` problem lived. A clean 5-block re-run would be needed to close the 2% door. |
| A rolling prefetch window (constant lookahead, advance one vector per neighbour) beats the front-loaded burst. `prefetch_per_iter_`; arm `step=8, per_iter=1` vs the `step=64` default. | 1.0349, CI [1.0036, 1.0672] — significantly slower. **But: n=100,000, efC=200, only 2 blocks, and both blocks ran the reference first.** It also varies depth and scheduling together. | **Not supported, and not properly tested.** Treat as "no reason to pursue", not as refuted at the target workload. Knob retained so a proper test is one command away: compare `step=8,per_iter=8` against `step=8,per_iter=1` so only scheduling changes. |
| Send the `<4` batch tail through the 4-wide kernel with padded lanes (bit-identical, since `F32L2SSEBatch4` is per-lane bit-identical to `F32L2SSE` — `native_simd_l2_smoke` asserts it over 69,824 outputs). | 1.0060, CI [0.9907, 1.0216] at 1M, efC=250, 5 blocks. | **No effect**, and the interval is tight enough to exclude anything above ~1%. Knob retained, default OFF. |
| `SearchLayerNearest`'s greedy descent has no prefetch at all, so adding one should help. | Stacked on skip-visited, 0.9626 -> 0.9597; the direct arm-to-arm interval is 0.9969, CI [0.9874, 1.0066]. | **Genuinely unresolved** — a ~0.3% effect either way. Knob retained, default OFF. |
| Small-`n` prefetch tuning transfers to 1M. | `step=4` is 4.2% faster at n=100,000/efC=200 (CI [0.9456, 0.9701], significant) and within noise at 1M/efC=250. | **Refuted, and a trap.** Validate at 1M or be misled. |

Previously refuted elsewhere and still refuted: hand-written NEON kernels, `Batch8`/`Batch16`
widening, FAISS `prune_headroom`, the level-0 double forward budget, a persistent per-thread visited
table, reserving the frontier vector, re-batching the reciprocal phase, and Accelerate GEMM.

## What is left, and how big it can be

The counters bound the remaining levers, and the arithmetic is unforgiving: to save X% of total
build time, a change to a region that is R% of it must remove X/R of that region.

- `HeapResultHandler::AddResult` is 3.9% of active time **as its own frame** — it is not inlined.
- The `prefetch_step_`/`prefetch_per_iter_` space is not actually closed; see the two "not properly
  tested" rows above. Two clean 5-block campaigns would close it for good.
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
- **The adjacency-list read at the top of each pop** (`GetNeighbors`, hnsw_alg.cppm) has no
  prefetch, and cannot easily get one: it is serially blocking, because the neighbour vectors
  cannot be prefetched until the list itself lands. The original hypothesis for this whole
  investigation was that this stall was the dominant cost and that prefetching one or two pops
  ahead from the frontier heap would recover 10-15%. That was never measured, and it is still not
  measured — but the counters now size it. There are 243 million pops at the target workload, each
  reading ~256 bytes scattered across a ~120 MB graph store. Even assuming every one is a fully
  exposed miss at ~60 ns and every nanosecond is recoverable, that is ~14.6 s of thread time,
  ~1.2 s of wall time across 12 workers, or **about 3% of build time as a hard upper bound** — a
  legitimate lever, but a third of what the hypothesis assumed, and it needs a speculative prefetch
  off the heap top whose hit rate is unknown because a push can change the top between pops.
  Anyone picking this up should measure the mispredict rate first: the counters make that a
  four-line addition.
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
- **When a DEFAULT changes, an old binary is no longer a valid control.** The first attempt to
  measure the alignment fix compared the new build against a binary saved before skip-visited became
  the default, with no environment set — so it measured skip-visited a third time (0.9669,
  CI [0.9618, 0.9719], which does at least corroborate 0.9572 and 0.9626) rather than the alignment.
  Pin every knob explicitly on both arms; do not let either arm inherit a default.
