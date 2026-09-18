# Roadmap

## Where the port stands (September 2026)

| Capability | State | Evidence |
|---|---|---|
| Native arm64 compile and link of the engine, unit tests, HNSW harness | Done | `macos-arm64-*` presets, `tools/apple_silicon/native_hnsw_smoke` |
| One-command setup, run, test and package from a clone | Done | `scripts/apple_silicon/setup.sh` and friends; `make` |
| Server lifecycle: create, insert, flush, HNSW build, indexed query, restart, reload | Verified | [MACOS_VERIFICATION.md](MACOS_VERIFICATION.md) |
| HNSW index build faster than FAISS at matched recall | Done, measured | [BENCHMARKS.md](BENCHMARKS.md): 1.40× on M4, 1.52× on M3 Pro |
| Query throughput vs FAISS | Done, measured | 1.30× on M4 |
| Reproducible benchmark harness, dataset fetch, paired A/B and iso-recall drivers | Done | `scripts/bench/`, `make bench` |
| Full-text search on macOS, incl. CJK dictionary analyzers | Verified | `dql/fulltext` 8/8 plus `fulltext_chinese_analyzer.slt` |
| Update, delete, drop, compact, cleanup on macOS | Verified | `dml/delete`, `dml/update`, `ddl/drop`, `dml/compact`, `dml/cleanup` |
| Bulk import and export on macOS | Verified | `dml/import` 27/27, `dml/export` 5/5 |
| Crash recovery and WAL replay | Verified against process death | `scripts/apple_silicon/verify_crash_recovery.sh` |
| Power-loss durability | Fixed on this fork: `F_FULLFSYNC` per committed batch | commit `b2d59374a`; `test/eval_macos/test_wal_durability_e2e.py` |
| Corrupt-WAL handling | **Broken**: one bad byte discards the whole log | [EVALUATION.md](EVALUATION.md) blocker 5 |
| Input validation | **Eleven blockers**, four crashes and seven silent wrong answers | [EVALUATION.md](EVALUATION.md); summary in [known issues](../known-issues.md) |
| Packaging: relocatable arm64 tarball with checksum, and an installer | Done, self-tested | `make package`, `install.sh` |
| Packaging: Homebrew formula, arm64 embedded Python wheel | Not started | the shipped wheel is the pure-Python remote SDK |
| macOS CI: build, unit tests, SQL suite, recovery, package | **Green** on a hosted `macos-15` runner, 2026-09-18 | `.github/workflows/macos_arm64.yml`, run 35317365990 |
| Lint CI: shell, links, YAML | Done | `.github/workflows/lint.yml` |
| HTTP API on macOS | One end-to-end suite, 38/40 | `test/eval_macos/test_http_api_e2e.py` |
| Cluster mode on macOS | Not tested | two standalone instances side by side do work |
| Linux and x86-64 | Removed by design | use [upstream](https://github.com/infiniflow/infinity) |
| Hand-written NEON kernels (today x86 intrinsics lowered by SIMDe) | Not started | the largest untouched lever, see [SEARCHLAYER_DECOMPOSITION.md](SEARCHLAYER_DECOMPOSITION.md) |

Honest summary: a fast engine, a complete port and a runnable, packaged database, with
a validation layer that is not finished. Until the blockers are fixed it is a database
to run, not one to trust with data you cannot regenerate.

## Next: make it a database people can rely on

In the order they unblock each other.

1. **First tagged release.** CI is green as of 2026-09-18, so everything above is
   protected against regression and a release can be cut: `v0.7.3-apple.1`, the
   tarball and its checksum attached, and `install.sh` pointing at it, so a fresh Mac
   gets a server in a minute instead of an hour.
2. **The eleven blockers**, crashes first. Each has a fix direction in
   [EVALUATION.md](EVALUATION.md) and a failing regression test in
   `test/eval_macos/test_server_crash_regressions.py`. The always-true/always-false
   predicate crash (blocker 1) and the corrupt-WAL deletion (blocker 5) are the two
   that bite ordinary use.
3. **The two engine-side unit-test failures.** The PGM `long double` overflow needs
   the vendored arithmetic reformulated so it cannot overflow where `long double` is
   64-bit; the low-cardinality index test needs its key type corrected.
4. **HTTP API parity.** The SQL logic tests drive only the PostgreSQL path; extend
   the end-to-end HTTP suite to cover what they cover.
5. **Homebrew formula.** Once releases exist, a tap that installs the tarball, so
   `brew install` is the install command.
6. **Upstream pull requests.** The float parse, the test-harness portability fixes,
   the resource resolver, and the two storage fixes are platform-independent and
   belong in infiniflow/infinity now. The toolchain, triplet, platform gating and SIMD
   dispatch follow behind portability gates, because this fork removed Linux and
   upstream must not.
7. **RAGFlow on Apple Silicon.** With Infinity native, an ARM64 RAGFlow setup that
   uses Infinity as its document engine instead of Elasticsearch, which RAGFlow
   currently lists as unsupported.

## Then: performance

- Hand-written NEON kernels for L2, inner product, cosine and normalization,
  replacing the SIMDe-lowered x86 intrinsics, with scalar reference tests for each.
- Batched distance through Accelerate for the wide candidate lists in `SearchLayer`,
  benchmarked against the current kernels rather than assumed.
- A real 768-d and 1536-d embedding corpus in the benchmark set, with
  published-truth recall.
- Worker-count and allocator defaults per chip generation (M1 through M4), measured.

## Later: a service

Only after the above. The engine's economics are already favourable (see
[COST_COMPARISON.md](COST_COMPARISON.md)), but a hosted or standalone product needs
multi-tenancy, authentication, backups, monitoring and an operations story that this
repository does not have yet. Cloud deployments run on Linux, where the Apple-specific
tuning does not apply, so that work would build on upstream's Linux ARM64 build rather
than on this port.

## Ground rules that stay in force

- A generic Apple Silicon binary with runtime-dispatched kernels; no M4-only
  instructions in releases.
- Documented Apple APIs only; no private AMX instructions, no forced thread placement.
- Metal only if profiling proves a workload benefits.
- No performance claim without dataset, dimensionality, index parameters, recall and
  thread count held constant, and no headline from a single run.
- Platform-independent fixes go upstream as they are made, so this fork's divergence
  is only ever the Apple-specific part.

## Definition of done

- Builds natively from a clean checkout on Apple Silicon with no Docker or Rosetta,
  and CI proves it on every change.
- The applicable upstream tests pass on macOS arm64, and the eleven blockers are
  closed with regression tests.
- HNSW indexing, full-text indexing, persistence, WAL recovery, corrupt-WAL handling
  and the client APIs work natively.
- A packaged artifact installs and runs on a first-generation M1 and on an M4 with
  16 GB, from `install.sh` or Homebrew.
- Benchmarks are reproducible and include recall, memory and dimensionality.
- Every architecture-specific optimization has a scalar correctness comparison.
- The platform work is a sequence of reviewable upstream PRs, not one monolithic
  patch.
