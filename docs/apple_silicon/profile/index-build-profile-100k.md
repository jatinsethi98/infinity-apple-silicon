# Current-Code Apple Silicon HNSW Sidecar Profile

## Status

CAPTURE AND INGESTION-ONLY REDUCTION PASS; EXTERNAL REVIEW BLOCKED BY
MANAGED POLICY.

This is a development profile, not a qualification timing campaign or a
FAISS comparison. It identifies where the current native arm64 Infinity HNSW
build spends worker time and how evenly the current scheduler uses 12 workers.

## Capture Identity

The hold was explicitly released for this sidecar. The runner then admitted
one fresh process and captured it with macOS `sample`:

```text
date=2026-08-21
binary=build/native-hnsw-d0-release/infinity_hnsw_d0
binary_sha256=bb71066a00f855dc50d3d9eac733d65a998435ca9f7fc48491c73c416c283ac8
binary_type=Mach-O 64-bit executable arm64
build_freshness=ninja: no work to do.
build_type=Release
compiler=/opt/homebrew/opt/llvm@20/bin/clang++
architecture=arm64
deployment_target=14.0
workload_status=0
sample_status=0
selected_inputs_stable=yes
```

The runner used the shared `HnswBulkBuild` boundary used by production
`HnswHandler::InsertVecs` at
`src/storage/knn_index/knn_hnsw/hnsw_handler.cppm:154-168`. The development
bridge calls that same boundary at
`tools/apple_silicon/native_hnsw_smoke/infinity_hnsw_bridge.cpp:28-46`.

## Workload

```text
vectors=100000
dimensions=128
distance=squared L2 float32
M=32
efConstruction=200
workers=12
submitted_tasks=12
cold_ingestion_ns=11715579708
insert_call_ns=11714923541
graph_valid=1
graph_reachable_count=100000
```

The observed 11.716-second ingestion is from a sampled development run and
must not be mixed into the unsampled timing campaign. Post-build query work
was present in the process but is excluded from every result below.

## Reduction Method

`analyze-sample.py` parses only the call graph section of the raw `sample`
report. It finds the 12 CTPL worker subtrees rooted at
`std::__packaged_task_func` frames containing `HnswRunBuildTasks`.

- Self samples are each node's inclusive count minus its direct children.
- Inclusive symbol counts add each terminal stack once per distinct canonical
  symbol, avoiding recursive-frame double counting.
- The 55,364 terminal samples equal the sum of all 12 task roots exactly.
- The tree has 1,911 nodes, no negative self counts, 12 task roots, and 12
  distinct worker threads.
- Percentages use 55,364 ingestion task-active sample observations as their
  denominator.

These percentages estimate aggregate build-worker active/blocked thread-time
share. They are not percentages of the 11.716-second wall clock and are not
standalone speedup predictions.

## Dominant Stacks

Inclusive stacks, excluding task/build wrappers:

| Rank | Inclusive stack | Samples | Share |
| ---: | --- | ---: | ---: |
| 1 | `SearchLayer<true>` construction traversal | 36,774 | 66.42% |
| 2 | `ConnectNeighbors` reciprocal update/pruning | 16,691 | 30.15% |
| 3 | `SelectNeighborsHeuristic<true>` under reciprocal pruning | 14,446 | 26.09% |
| 4 | `SelectNeighborsHeuristic<false>` for the new node | 1,350 | 2.44% |
| 5 | `SearchLayerNearest<true>` upper-layer entry search | 463 | 0.84% |

`SelectNeighborsHeuristic<true>` is nested inside `ConnectNeighbors`; those
inclusive rows must not be added. The exclusive phase partition does sum to
100%: traversal 66.42%, reciprocal update/pruning 30.15%, new-node selection
2.44%, upper-layer entry search 0.84%, and wrappers/other 0.16%.

Dominant self/top-of-stack symbols:

| Rank | Self symbol | Samples | Share |
| ---: | --- | ---: | ---: |
| 1 | `F32L2SSEBatch4` | 34,135 | 61.66% |
| 2 | `SearchLayer<true>` control and inlined work | 12,465 | 22.51% |
| 3 | `F32L2SSE` scalar distance | 2,966 | 5.36% |
| 4 | `HeapResultHandler::AddResult` | 1,588 | 2.87% |
| 5 | `SelectNeighborsHeuristic<true>` control | 1,341 | 2.42% |
| 6 | `__psynch_cvwait` | 1,067 | 1.93% |
| 7 | `SIMDPrefetch` | 542 | 0.98% |
| 8 | `SelectNeighborsHeuristic<false>` control | 431 | 0.78% |

Batch4 L2 splits into 19,954 samples (36.04%) in construction traversal and
14,181 (25.61%) in reciprocal update/pruning. All scalar L2 paths total 2,966
samples (5.36%). Blocked lock waits total 1,097 samples (1.98%), all lock
paths total 2.58%, and allocator paths total 217 samples (0.39%). Lock removal
and allocator replacement are therefore not first-order experiments.

Source attribution:

- Traversal, visited checks, batching, and heap updates:
  `src/storage/knn_index/knn_hnsw/hnsw_alg.cppm:215-296`.
- New-node selection:
  `src/storage/knn_index/knn_hnsw/hnsw_alg.cppm:329-397`.
- Reciprocal update, candidate distances, pruning, and neighbor locks:
  `src/storage/knn_index/knn_hnsw/hnsw_alg.cppm:399-447`.
- Build pipeline:
  `src/storage/knn_index/knn_hnsw/hnsw_alg.cppm:851-870`.
- Apple arm64 Batch4 kernel:
  `src/common/simd/hnsw_simd_func.cppm:1123-1224`.

## Worker Occupancy

The 12 task-root active sample counts, in sample thread order, are:

```text
3386 3354 3385 3726 4353 4546 4797 5065 5208 5479 5843 6222
```

```text
minimum/maximum=53.91%
mean=4613.67 samples
sample-window active-stack occupancy=55364/(12*6222)=74.15%
tail capacity relative to longest task=19300 sample observations
```

This is sampled active-stack occupancy, not hardware CPU utilization. It
nevertheless exposes material task-lifetime skew. The scheduler computes one
contiguous bucket per worker at
`src/storage/knn_index/knn_hnsw/hnsw_bulk_build.cppm:28-63,160-183`.
Perfect balancing with unchanged work has a mathematical worker-stage ceiling
of `1/0.7415 = 1.348x`; contention, task overhead, and graph-order effects make
that a bound rather than a forecast.

## Reward-Ranked Experiments

1. **Oversubscribe build buckets and let the pool dynamically drain them.**
   Sweep 2x, 4x, and 8x tasks per worker, with a 512/1024-vertex floor. This
   directly targets the 25.85% sampled tail-capacity loss. Expected reward is
   roughly 15-25% ingestion reduction if graph-order effects remain benign;
   the evidence-based ceiling is 25.85% worker-stage wall reduction. Gate on
   graph validity, reachability, recall, task accounting, and paired timing.

2. **A/B a native NEON Batch4 L2 microkernel.** Compare the current
   SSE2-compatibility implementation with direct `arm_neon.h` FMA loads,
   accumulator counts, and `vaddvq_f32` reductions. Microbenchmark first, then
   run the unchanged HNSW contract. The target owns 61.66% of self samples; a
   15-25% kernel gain would imply a 9-15% aggregate worker-sample opportunity,
   with an expected 5-12% ingestion gain after non-kernel work and imbalance.

3. **Reduce `SearchLayer` frontier/control cost without changing semantics.**
   Prototype a fixed-capacity/specialized candidate frontier and result heap,
   preserving tie behavior and exact distance ordering. This targets 22.51%
   `SearchLayer` self plus 2.87% `AddResult` self. Expected ingestion reward is
   3-8%; require graph/recall gates because small ordering changes can alter
   HNSW topology.

## Host And Permission Caveats

The preflight's seven snapshots were 62.28-77.20% idle and passed the
schema-5 development floor of 25%. Persistent corporate services prevented
using an 80% development gate; no competing Infinity/FAISS HNSW process was
present. This limits timing interpretation but does not explain the
within-process stack composition or per-task lifetime spread.

`/usr/bin/sample` succeeded with status 0. `spindump` was neither needed nor
attempted, so there is no profiling-permission blocker to report.

This is one 1 ms sampling run. Sampling overhead, scheduler/core migration,
and background services can perturb both worker timing and stack mix. A repeat
profile should confirm any optimization that changes the ranking.

The runner hashed the binary and selected HNSW, SIMD, bridge, runner, and
driver sources before and after capture. It did not hash every transitive
Ninja input; freshness is additionally supported by the immediate no-op Ninja
check. Claims are scoped to the recorded binary identity rather than to every
unhashed source file.

## Artifacts

Durable checkpoint artifacts:

```text
analysis/build-profile-analysis.json
raw/infinity-current-profile.sample.txt
raw/workload.stdout.txt
raw/workload-command.txt
raw/sample-command.txt
raw/host-before.txt
raw/top-before.txt
raw/processes-before.txt
raw/inputs-before.sha256
raw/inputs-after.sha256
raw/exit-status.txt
raw/sample.stderr.txt
artifact-manifest.sha256
verification.md
```

The complete capture remains at:

```text
/private/tmp/infinity-current-code-hnsw-sidecar-profile-v1
```

The 20.6 MB graph audit sidecar is referenced in place:

```text
infinity-current-profile.audit-v1.bin
sha256=efdb772f3fa3552c7efaf341086b3f04fd8edf791846373f5bdf459ba0b379d0
```

The raw sample SHA-256 is
`40ec158a398b2f43e0a9aa2b65857b01b46e7fba56feda27d7fa1c29d71a3eb`.
The parsed analysis SHA-256 is
`0c4a3177b3a86fe37ae96c9206a82ee0bb9a953162aa63b47de097cfd0a32308`.

## Review

The exact requested invocation used `claude-fable-5`, `--effort max`, and
`--permission-mode plan`. It exited before review because the sandbox denied a
write to `~/.claude/settings.json`. The required escalation was then rejected:
managed tenant policy prohibits transmitting private workspace code and
`/private/tmp` artifacts to an external service. No fallback model was used,
and this checkpoint does not claim a Claude verdict.

The invocation, stdout, stderr, and blocked status are retained under
`review/`. This external-review gate remains unresolved; the local reduction
was independently rerun from the durable raw sample and matched after removing
only the input-path field.
