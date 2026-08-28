# Apple Silicon Port — Status Audit (2026-08-27)

Audit of the 6-day autonomous Codex run (Aug 18–26) against
`~/Downloads/apple_silicon_port_roadmap.md`.

Machine: MacBook Pro Mac15,7 · Apple M3 Pro (6P+6E, 12 threads) · 36 GB · macOS 26.2 arm64.
(The roadmap targets "M4/16-GB"; the actual host is M3 Pro/36 GB.)

---

## 1. Bottom line

| Claim | Reality |
|---|---|
| "Ported the repo to native Mac code" | **True, and it works** — but it was never committed. It lived for 6 days as uncommitted working-tree state in a scratch checkout. Now committed as `apple-silicon/port-salvage-v1`. |
| "0.9x FAISS performance" | A 12,288-vector synthetic toy (~0.3 s build), at **worse recall than FAISS**, on battery, self-labelled `claim_eligible: false`. Raw data deleted. The one preserved twin run says **0.851x**. |
| "Then it never improved" | It never *tried*. **Zero optimization experiments were ever executed.** All three were blocked by gates the agent itself built. |
| Roadmap completion | **~20%** measured against the checkout; **~5%** measured against the committed repo (which contains zero source changes). |

The committed repo `infinity @ 0f6fc6950` has 5 commits / **+44,139 lines / 0 deletions**, of which
**46 of 47 files** are `docs/apple_silicon/` and `scripts/apple_silicon/`. The 47th is a
one-line `CLAUDE.md`. **No `src/`, `cmake/`, or vcpkg change is committed.**

---

## 2. What is genuinely proven

- **Native arm64 compile + link of the whole engine** — `build/macos-native-server/src/test_main`,
  148 MB Mach-O arm64.
- **Native server boots without Docker or Rosetta** — create table → batched insert → flush →
  HNSW build → forced indexed query → clean shutdown → **restart → reload → identical digest,
  recall@10 = 1.0**. Scale: 1,024 rows × 32 dims. Negative control (missing index → err 3023) correct.
- **FAISS v1.15.0 built natively** (`libfaiss.dylib`, arm64) as a pinned comparison baseline.
- **A real profile of the indexing hot path** — the single most valuable artifact produced (§5).
- Focused native test passes: `test_main` rebuilds, `PersistenceManagerTest` 6/6, Python lifecycle 7/7,
  focused CTest 6/6.

The port itself is competent work: platform-gating of `mold`/GNU-static/crosscompile blocks,
Homebrew libc++ module discovery, `INFINITY_TARGET_PROCESSOR` from `CMAKE_OSX_ARCHITECTURES`,
an `arm64-osx` vcpkg triplet, boost split into components, jemalloc made opt-in,
`ports/roaring` + `ports/vit-vit-ctpl` overlays, and Darwin fixes across
persistence/catalog/txn/`local_file_handle`/config/`infinity_context`.

**Important correction to the roadmap narrative:** the SIMD work is **not hand-written NEON**.
`src/common/simd/hnsw_simd_func.cppm` contains **497 x86 intrinsics and 0 NEON intrinsics**;
`simde/x86/sse.h` + `simde/x86/avx512.h` lower SSE2 to NEON on arm64. Real NEON does appear
in the emitted binary, but it is compiler-generated. Roadmap Phase 7 ("native NEON backends")
is therefore effectively **not started** — which is good news, because it is the largest
untouched lever.

---

## 3. What the 0.9x number actually is

Origin: `native-smoke-v1/docs/apple_silicon/checkpoints/native-hnsw-current-default-schema5-v1/README.md`
(`0.9020287606558678x`, equal-order-weighted geometric mean of Infinity/FAISS build throughput).

- **Dataset:** 12,288 × 128 float32, `python-random` uniform [0,1), seed 0, 6,291,456 bytes.
  Not SIFT1M. Uniform random has no cluster structure — pathological for ANN.
- **What was timed:** cold **index-build** only. Median 327 ms (Infinity) vs 294 ms (FAISS).
  A 0.3-second measurement.
- **Recall parity: FAILED.** Gate ceiling is 0.005 deficit:

  | k | efSearch | Infinity | FAISS | deficit | gate |
  |---:|---:|---:|---:|---:|---|
  | 10 | 32 | 0.697656 | 0.746094 | 0.059375 | FAIL |
  | 10 | 64 | 0.865625 | 0.885938 | 0.037500 | FAIL |
  | 10 | 128 | 0.961719 | 0.977344 | 0.021875 | FAIL |
  | 100 | 128 | 0.905859 | 0.918906 | 0.015781 | FAIL |

- **Infinity also built a sparser graph:** 533,484 directed edges vs FAISS 574,667 at
  identical `M=32`/`efConstruction=200` — a 7.2% edge deficit. So Infinity did **less work**,
  produced a **worse index**, and was still **15% slower**. The true like-for-like deficit is
  worse than 0.85x.
- **Environment:** on battery, with a development 25% CPU-idle gate substituted for the
  default 95%. `claim_eligible: false`, `scope: d0-development-only` in its own JSON.
- **Provenance:** the raw evidence dir (`/private/tmp/infinity-d0-current-default-schema5-v2`,
  3.9 GiB) **no longer exists**. The headline number cannot be re-verified. Its preserved
  sibling (`campaign-v8/summary.json`, same workload, same day) reports **0.8514711503095199**
  with per-pair spread 0.7639–1.0422 and 6.08% relative MAD — i.e. 0.851 vs 0.902 is inside noise.
- **SIFT1M / GIST1M / 768-d Email1M: never downloaded, never run.** Every `.bin` on disk is the
  same 6 MB synthetic file. The Enron corpus was downloaded (443 MB) and never extracted or
  embedded. Roadmap Phase 1 was skipped entirely.

**One under-appreciated fact:** FAISS was built with empty `CMAKE_CXX_FLAGS` and no
`FAISS_OPT_LEVEL` — so on arm64 FAISS has **no hand-written SIMD at all**; its AVX2/AVX512
kernels are x86-only. Infinity, with a hand-tuned batch-4 kernel, is losing to
plain auto-vectorized C++. That means **the gap is structural, not SIMD-width** — consistent
with the profile below.

---

## 4. How the loop happened

The failure was structural, not technical. Codex converted the 9-phase roadmap into its own
10-gate contract system (C0–C9) and then defined **C0** as: *"freeze charter, inputs, evidence
contract, benchmark contract, and adversarial review."* No engineering is in C0. It then tried
to satisfy C0 with a notarization-grade supply-chain proof:

- **Reimplemented `brew install` byte-exactly in Python** (`materialization_v3.py`, 3,859 lines):
  re-audit 7 Homebrew bottles against a lock, extract only locked members (9,886 members,
  1.47 GB), apply 9 text + 54 Mach-O install-name/rpath transforms, ad-hoc re-sign, verify
  CDHashes and `minos 14.0` — all *without executing Homebrew or Ruby*.
- Required a **fresh case-sensitive APFS disk image per attempt** (2.5 GB each; 9 were on disk).
- Required a **two-commit A/B transaction** with an independent verifier that refuses to run
  unless launched as `python -I -S -B`.

Each failure added an invariant instead of relaxing the bar. The retirement suffixes are the
loop, written on the filesystem:

```
c0-capture-v11.retired-v12-symlink-mismatch
c0-capture-v14.retired-prefix-symlink-mismatch
c0-capture-v16-r1.retired-absent-tag-probe
c0-capture-v17-r1.retired-packed-object-fsync
c0-capture-v19-r1.retired-session-interruption
c0-capture-v19-r2.retired-provenance-context-mismatch
```

symlink hashing → prefix-symlink ordering → absent-tag probe → packed-object fsync →
provenance-xattr match → a three-stage 17/14/12 GiB disk gate. **9 attempts, 0 completions.**
`docs/apple_silicon/evidence/c0/` is still an **empty directory** — the C0 deliverable was
never produced, so "Commit B" was never formed.

Then the same pattern recurred in the benchmark phase. Measurement-harness **schema v3 → v16**,
each version rejecting its own runs:

- `Configured executable is missing: /private/tmp/infinity-cmake-4.3.4/…/cmake` (pinned toolchain
  evaporated from `/private/tmp`)
- `control benchmark Ninja commands has conflicting compiler invocations for hnsw_d0_main.cpp.o`
- `control became stale during capture`
- and finally: four clean v5 binaries were frozen, then *"the managed approval broker timed out
  before the read-only execution-policy gate could start"* — **twice**. `native-hnsw-schema16-v5-readiness.md`
  line 155: *"No v5 performance result exists yet."*

Compounding it: the mandated adversarial review (`claude --model claude-fable-5 --effort max`)
**never ran** — the one review on disk used opus-5 at default effort, returned **FAIL**
(2 Critical / 5 High / 11 Medium / 3 Low), and the required passing follow-up round does not exist.
By its own rules the project never left C0.

**Diagnosis: it optimized for provability instead of for the thing being proved.** Verification
cost scaled superlinearly with rigor while the measured number never moved. Notably, the agent's
own self-assessment (`apple-silicon-objective-gap-matrix-v1.md`) is honest and accurate — it
correctly labels 0.902x "a verified negative result." The failure was execution, not candour.

---

## 5. The one thing worth its weight: the profile

`native-smoke-v1/docs/apple_silicon/checkpoints/native-hnsw-current-code-sidecar-profile-v1/`
— macOS `sample` over 100,000 × 128 ingestion, M=32, efC=200, 12 workers, 11.72 s,
55,364 task-active samples reduced across 12 disjoint worker subtrees.

**Exclusive phase partition (sums to 100%):**

| Phase | Share |
|---|---:|
| `SearchLayer<true>` construction traversal | 66.42% |
| `ConnectNeighbors` reciprocal update/pruning | 30.15% |
| new-node selection | 2.44% |
| upper-layer entry search | 0.84% |

**Dominant self symbols:**

| Symbol | Share |
|---|---:|
| `F32L2SSEBatch4` (the SIMDe SSE2→NEON kernel) | **61.66%** |
| `SearchLayer<true>` control + inlined | 22.51% |
| `F32L2SSE` scalar | 5.36% |
| `HeapResultHandler::AddResult` | 2.87% |
| all lock paths | **2.58%** |
| all allocator paths | **0.39%** |

**Worker occupancy: 74.15%** (per-worker sample counts 3386…6222, min/max = 53.91%).
Perfect balance alone is a **1.348x** ceiling.

Two roadmap Phase-8 items are killed by this data: **jemalloc-vs-malloc (0.39%) and
lock removal (2.58%) are noise**, not first-order work.

---

## 6. Is 1.5x reachable?

From ~0.85x at *unmatched* recall, 1.5x needs ~1.77x; after paying for recall parity
(Infinity is 7.2% short on edges) call it **~1.9x needed**.

| Lever | Targets | Realistic | Notes |
|---|---|---|---|
| Bucket oversubscription / dynamic drain | 25.85% tail loss | 1.20–1.35x | ceiling 1/0.7415; pure scheduling, no topology change |
| **Wide batched distance via Accelerate GEMV** | 61.66% | **1.25–1.9x** | batch all M=32…64 neighbours of the popped node; ‖a−b‖²=‖a‖²+‖b‖²−2a·b with precomputed norms → one GEMV. Accelerate reaches AMX through a **public** API. Roadmap explicitly sanctions "MLAS versus Accelerate for sufficiently large batched operations" |
| Native `arm_neon.h` kernel (replacing SIMDe) | same 61.66% | 1.10–1.25x | only if not superseded by batching |
| `SearchLayer` frontier + heap specialization | 22.51% + 2.87% | 1.05–1.10x | fixed-capacity frontier; preserve tie order |
| Distance reuse in `ConnectNeighbors` | 30.15% phase | 1.05–1.15x | reciprocal pruning recomputes distances the traversal already had |
| Vector/adjacency co-location + prefetch | memory-bound share | 1.05–1.15x | 512 B/vector × 64 scattered reads per node |

Codex's own three levers alone compound to only ~1.3x — which matches its "Amdahl ~1.25x"
pessimism. **The batched-distance/Accelerate path is what makes 1.5x credible**, and it was
never on its list. Conservative compound: 1.25 × 1.4 × 1.07 × 1.08 ≈ **2.0x → ~1.7x FAISS**.

**The single most important missing measurement:** distance computations per inserted vector,
for both engines. That decomposes 0.85x into "does more work" vs "slower per unit of work" and
determines the entire strategy. Nobody measured it in 6 days.

---

## 7. Roadmap status

| Phase | Status | Evidence / gap |
|---|---|---|
| 1 Linux baselines | **NOT STARTED** | no SIFT1M/GIST1M/Email1M; only a schema for recording results |
| 2 macOS toolchain | **PARTIAL ~60%** | `arm64-osx` triplet + working vcpkg deps exist; **no `CMakePresets.json` anywhere** |
| 3 First native server | **PARTIAL ~60%** | boots + persists + restarts at 1024 rows; no update/delete/drop, no bulk import, no full-text |
| 4 Functional parity | **~15%** | focused suites pass; no SQL-logic-test results, no HTTP/Postgres, no recovery run, no exclusion list |
| 5 Distribution | **~5%** | `Packaging.cmake` +18 lines; no tarball, no formula, no macOS CI job |
| 6 Native perf baseline | **PARTIAL ~25%** | the profile (§5) is real and good; the *benchmark* basis is a 12k toy |
| 7 NEON optimization | **~15%** | standalone NEON micro + scalar reference exist but are **not integrated**; SIMDe used in-tree; **no end-to-end win for any optimization** |
| 8 macOS runtime tuning | **~5%** | allocator/lock work is ≤3% of time (§5) — deprioritize |
| 9 Harden / handover | **~5%** | crash-recovery harness written (5,685 lines Python, 30/30 synthetic) but **never run against the server**; no upstream PRs |

---

## 8. What was preserved (`/Users/sethjatq/Desktop/proj/_salvage/`)

- `native-smoke-v1-port-salvage-v1.bundle` (79 MB) — the port, as branch
  `apple-silicon/port-salvage-v1` = 244 files / +101,018 / −1,025, also committed in-place
  in `native-smoke-v1` as `4ee449fda`.
- `profile-sidecar-v1/` — the §5 profile with raw `sample` output.
- `d0-matched-v1-faiss-baseline/` — the FAISS-matched D0 campaign + harness.
- `level0-screen-*.patch`, `AppleHnswLevel0Screen.cmake`, `level0_recall_screen_v1/` —
  the level-0 recall experiment from the second checkout (its only unique delta: 92 lines
  in `hnsw_alg.cppm` + a 187-line cmake module; the rest was byte-identical).
- `native-server-hnsw-crash-recovery-v1.patch` — applies to the pristine `infinity` checkout.
- `apple-silicon-objective-gap-matrix-v1.md`, `native-hnsw-schema16-v5-readiness.md`.
- `datasets/enron_mail_20150507.tar.gz` — the only real-world corpus fetched.

Toolchain confirmed intact and reusable **without** the C0 materialization ritual:
Homebrew clang 20.1.8 at `/opt/homebrew/opt/llvm@20`, prebuilt `vcpkg_installed/arm64-osx` (291 MB).

## 9. Disk reclaimed

68 GB → 9 GB in `~/Desktop/proj` (free space 62 GB → 122 GB).

Deleted: 26 `c0-*` clones and 2.5 GB APFS images (20.9 GB); 7 full-engine A/B build trees for
rejected optimizations plus sanitizer/duplicate trees (29.5 GB); 6 level-0 policy-test build arms
(1.1 GB); `infinity/tmp/apple_silicon` C0 scratch (5.8 GB); 3 failed preflight evidence dirs (3.9 GB).

Kept: `faiss-d0-matched{,-v2,-v3}` + `faiss-stats-disabled-v1` (the FAISS baseline),
`native-hnsw-d0-release` and `native-hnsw-d0-matched-v1` (working `infinity_hnsw_d0` +
`faiss_hnsw_d0` harnesses), `macos-native-server` (the proof-of-life build), `profiles`.
