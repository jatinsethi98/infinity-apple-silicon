# Roadmap

## Where the port stands (September 2026)

| Capability | State | Evidence |
|---|---|---|
| Native arm64 compile and link of the engine, unit tests, HNSW harness | Done | `macos-arm64-*` presets, `tools/apple_silicon/native_hnsw_smoke` |
| Server lifecycle: create table, insert, flush, HNSW build, indexed query, restart, reload | Verified on M3 Pro (2026-08) | [PRIOR_EFFORT_AUDIT.md](PRIOR_EFFORT_AUDIT.md) §2 |
| HNSW index build faster than FAISS at matched recall | Done, measured | [BENCHMARKS.md](BENCHMARKS.md): 1.40× on M4, 1.52× on M3 Pro |
| Query throughput vs FAISS | Done, measured | 1.30× on M4 |
| Reproducible benchmark harness, dataset fetch, paired A/B and iso-recall drivers | Done | `scripts/bench/`, `scripts/apple_silicon/` |
| Full-text search on macOS | Not verified | – |
| Update, delete, drop on macOS | Not verified | – |
| Bulk import on macOS | Not verified | – |
| Crash recovery and WAL replay on macOS | Not verified | – |
| Linux x86-64 and Linux ARM64 unchanged after the port | Not re-verified | upstream CI not yet run on this branch |
| Packaging: tarball, Homebrew formula, arm64 Python wheel | Not started | – |
| macOS CI runner with build cache and smoke tests | Not started | – |
| Hand-written NEON kernels (today the x86 intrinsics are lowered by SIMDe) | Not started | the largest untouched lever, see [SEARCHLAYER_DECOMPOSITION.md](SEARCHLAYER_DECOMPOSITION.md) |

Honest summary: a fast engine core and a working port, not yet a shippable database.

## Next: make it a database people can run

In the order they unblock each other.

1. **Functional parity on macOS.** Run the upstream unit, SQL, Python, HTTP, restart and recovery
   test suites natively. Fix what fails. Document any exclusion that is genuinely Linux-only.
   Covers full-text search, update/delete/drop, bulk import and crash recovery.
2. **Keep Linux green.** Run the existing Linux x86-64 and ARM64 CI on this branch and keep every
   platform-specific change gated, so the port stays upstreamable.
3. **Packaging.** A native tarball, a Homebrew formula, and an arm64 macOS wheel for the embedded
   Python module, so a fresh Mac can install and run without a development checkout.
4. **macOS CI.** An arm64 runner that builds the server and the harness, runs the smoke tests, and
   checks for performance regressions against the numbers in BENCHMARKS.md.
5. **Upstream pull requests.** The toolchain and triplet, the platform gating, the SIMD dispatch
   changes and the HNSW build fixes, as separate reviewable PRs against infiniflow/infinity.
6. **RAGFlow on Apple Silicon.** With Infinity native, an ARM64 RAGFlow setup that uses Infinity
   as its document engine instead of Elasticsearch, which RAGFlow currently lists as unsupported.

## Then: performance

- Hand-written NEON kernels for L2, inner product, cosine and normalization, replacing the
  SIMDe-lowered x86 intrinsics, with scalar reference tests for each.
- Batched distance through Accelerate for the wide candidate lists in `SearchLayer`, benchmarked
  against the current kernels rather than assumed.
- A real 768-d and 1536-d embedding corpus in the benchmark set, with published-truth recall.
- Worker-count and allocator defaults per chip generation (M1 through M4), measured.

## Later: a service

Only after the above. The engine's economics are already favourable (see
[COST_COMPARISON.md](COST_COMPARISON.md)), but a hosted or standalone product needs multi-tenancy,
authentication, backups, monitoring and an operations story that this repository does not have yet.
Cloud deployments run on Linux, where the Apple-specific tuning does not apply, so that work would
build on upstream's Linux ARM64 build rather than on this port.

## Ground rules that stay in force

- Compatibility before optimization; one portable codebase, no permanent macOS fork.
- A generic Apple Silicon binary with runtime-dispatched kernels; no M4-only instructions in releases.
- Documented Apple APIs only; no private AMX instructions, no forced thread placement.
- Metal only if profiling proves a workload benefits.
- No performance claim without dataset, dimensionality, index parameters, recall and thread count
  held constant, and no headline from a single run.

## Definition of done

- Builds natively from a clean checkout on Apple Silicon with no Docker or Rosetta.
- Applicable upstream tests pass on macOS arm64; Linux CI stays green.
- HNSW indexing, full-text indexing, persistence, WAL recovery and the client APIs work natively.
- A packaged artifact installs and runs on a first-generation M1 and on an M4 with 16 GB.
- Benchmarks are reproducible and include recall, memory and dimensionality.
- Every architecture-specific optimization has a scalar correctness comparison.
- The port is a sequence of reviewable upstream PRs, not one monolithic patch.
