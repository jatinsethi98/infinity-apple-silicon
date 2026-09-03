# Roadmap

## Where the port stands (September 2026)

| Capability | State | Evidence |
|---|---|---|
| Native arm64 compile and link of the engine, unit tests, HNSW harness | Done | `macos-arm64-*` presets, `tools/apple_silicon/native_hnsw_smoke` |
| Server lifecycle: create table, insert, flush, HNSW build, indexed query, restart, reload | Verified on M3 Pro (2026-08) | [PRIOR_EFFORT_AUDIT.md](PRIOR_EFFORT_AUDIT.md) §2 |
| HNSW index build faster than FAISS at matched recall | Done, measured | [BENCHMARKS.md](BENCHMARKS.md): 1.40× on M4, 1.52× on M3 Pro |
| Query throughput vs FAISS | Done, measured | 1.30× on M4 |
| Reproducible benchmark harness, dataset fetch, paired A/B and iso-recall drivers | Done | `scripts/bench/`, `scripts/apple_silicon/` |
| Full-text search on macOS | Verified, incl. CJK dictionary analyzers | [MACOS_VERIFICATION.md](MACOS_VERIFICATION.md); `dql/fulltext` 8/8 + `fulltext_chinese_analyzer.slt` |
| Update, delete, drop on macOS | Verified | `dml/delete`, `dml/update`, `ddl/drop`, `dml/compact`, `dml/cleanup` |
| Bulk import on macOS | Verified | `dml/import` 27/27, `dml/export` 5/5 |
| Crash recovery and WAL replay on macOS | Verified against process death only | `scripts/apple_silicon/verify_crash_recovery.sh` |
| Power-loss durability | **Broken upstream, all platforms** | the WAL never fsyncs; see MACOS_VERIFICATION.md |
| Linux x86-64 and Linux ARM64 unchanged after the port | Not re-verified | upstream CI not yet run on this branch |
| Packaging: relocatable arm64 tarball | Done, self-tested | `scripts/apple_silicon/make_package.sh` |
| Packaging: Homebrew formula, arm64 embedded Python wheel | Not started | the shipped wheel is the pure-Python remote SDK |
| macOS CI runner with build cache and smoke tests | Written, never executed | `.github/workflows/macos_arm64.yml` |
| HTTP API and cluster mode on macOS | Not tested | two concurrent standalone instances do work |
| Hand-written NEON kernels (today the x86 intrinsics are lowered by SIMDe) | Not started | the largest untouched lever, see [SEARCHLAYER_DECOMPOSITION.md](SEARCHLAYER_DECOMPOSITION.md) |

Honest summary: the engine's functionality is verified natively and there is a packaged
artifact, so this is a database you can run on a Mac. It is not yet one to trust with
data you cannot lose: power-loss durability is broken upstream on every platform (the
WAL never fsyncs), and neither the HTTP API nor cluster mode is tested here.

## Next: make it a database people can rely on

In the order they unblock each other.

1. **Run the macOS CI workflow.** It exists and has never executed. Until it is green
   once, none of the above is protected against regression.
2. **Keep Linux green.** Run the existing Linux x86-64 and ARM64 CI on this branch. Four
   changes now touch shared code — the float parse, two test-harness fixes and the
   unit-test resource resolver — and none has been run on Linux.
3. **HTTP API parity.** Every server starts an HTTP listener that nothing here
   exercises; the SQL logic tests drive only the PostgreSQL path.
   `python/test_pysdk --http` is the existing harness.
4. **Durability.** Decide whether to fix the WAL commit path (`fsync`, and
   `F_FULLFSYNC` on Darwin) and the un-truncated export writes. Both are upstream
   defects recorded in [MACOS_VERIFICATION.md](MACOS_VERIFICATION.md), and both are
   storage-engine decisions rather than port work.
5. **The two engine-side test failures.** The PGM `long double` overflow needs the
   vendored arithmetic reformulated so it cannot overflow where `long double` is 64-bit;
   the low-cardinality index test needs its key type corrected.
6. **Upstream pull requests.** The toolchain and triplet, the platform gating, the SIMD
   dispatch changes, the HNSW build fixes, and — separately, because they are not
   macOS-specific — the harness portability fixes. As reviewable PRs against
   infiniflow/infinity.
7. **RAGFlow on Apple Silicon.** With Infinity native, an ARM64 RAGFlow setup that uses
   Infinity as its document engine instead of Elasticsearch, which RAGFlow currently
   lists as unsupported.

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
