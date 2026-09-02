# Infinity on Apple Silicon

Native `arm64-apple-darwin` build of Infinity. No Docker, no Rosetta.

## Status

| Area | State |
|---|---|
| Native arm64 compile + link | Working (engine, unit tests, HNSW benchmark harness) |
| Native server lifecycle | Create / insert / flush / HNSW build / indexed query / restart + reload verified |
| HNSW index-build benchmark vs FAISS | See [BASELINE.md](BASELINE.md) |
| Full-text, update/delete/drop, bulk import, crash recovery | Not yet verified natively |
| Linux x86-64 / ARM64 preservation | Not yet re-verified after the port |
| Packaging / macOS CI | Not started |

## Prerequisites

```sh
brew install llvm@20 cmake ninja libomp faiss
```

Everything is pinned to Homebrew LLVM 20 (`/opt/homebrew/opt/llvm@20`). Apple Clang is not used —
the project needs C++23 modules support that matches the vcpkg-built dependencies.

`SDKROOT` must be set, or Homebrew Clang will not find the platform headers and configure fails with
`'pthread.h' file not found`:

```sh
export SDKROOT=$(xcrun --show-sdk-path)
```

For the full server build you also need a bootstrapped vcpkg whose history contains the baseline
commit pinned in `vcpkg.json`:

```sh
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=$PWD
```

A shallow clone is not enough — vcpkg must be able to `git show <baseline>:versions/baseline.json`.

## Building

Presets live in `CMakePresets.json` (server) and
`tools/apple_silicon/native_hnsw_smoke/CMakePresets.json` (benchmark harness).

```sh
# Full server, Release
cmake --preset macos-arm64-release
cmake --build --preset macos-arm64-release

# Unit tests
cmake --preset macos-arm64-debug
cmake --build --preset macos-arm64-test     # builds test_main

# HNSW benchmark harness only (fast: no server, no vcpkg toolchain needed)
cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke
cmake --build --preset bench -S tools/apple_silicon/native_hnsw_smoke
```

The benchmark harness is a standalone CMake project. It compiles Infinity's `hnsw_alg.cppm` and
`hnsw_simd_func.cppm` directly and links Homebrew FAISS, so it builds in minutes rather than the
hour a full engine build takes. That makes it the right target for optimization iteration.

## Platform boundaries

Where the port diverges from Linux, and why:

- **Linker/toolchain.** `-fuse-ld=mold`, GNU static-libgcc/libstdc++ flags, and the cross-compile
  blocks in the top-level `CMakeLists.txt` are gated behind `CMAKE_SYSTEM_NAME STREQUAL "Linux"`.
  On Apple the build discovers Homebrew libc++'s `libc++.modules.json` instead.
- **Architecture detection.** `INFINITY_TARGET_PROCESSOR` is derived from `CMAKE_OSX_ARCHITECTURES`
  on Apple, since `CMAKE_SYSTEM_PROCESSOR` is unreliable there.
- **SIMD.** The distance kernels are written in x86 SSE/AVX intrinsics and lowered to NEON by
  [SIMDe](https://github.com/simd-everywhere/simde) (`vcpkg.json` dependency `simde`,
  included via `src/common/simd/simd_common_intrin_include.h`). There is deliberately **no**
  hand-written `arm_neon.h` code in `src/` yet; `tools/apple_silicon/native_hnsw_smoke/neon_l2_micro.cpp`
  is the hand-written NEON L2 kernel plus its scalar reference, kept as the seed for that work.
  Apple arm64 additionally gets a 4-accumulator `F32L2SSE` unroll and a `F32L2SSEBatch4` kernel
  (one query against four candidates), dispatched through function pointers in `simd_init_impl.cpp`.
- **Allocator.** jemalloc is an opt-in vcpkg feature and is **off** on macOS; the hardcoded
  `libjemalloc_pic.a` path is Linux-only. Profiling shows allocator paths are 0.39% of index-build
  time, so this is not a performance concern.
- **Dependencies.** `boost` is split into the components actually used rather than the monolithic
  port. `ports/roaring` and `ports/vit-vit-ctpl` are overlay ports; the latter carries a
  strong-push-guarantee patch that the parallel HNSW build path depends on.

## Benchmarking

See [`scripts/bench/README.md`](../../scripts/bench/README.md) for the harness and the fairness
rules, and [BASELINE.md](BASELINE.md) for measured results.

```sh
python3 scripts/bench/fetch_datasets.py sift1m --spot-check
python3 scripts/bench/run_baseline.py --help
```

## Known HNSW convention difference vs FAISS

Infinity gives a newly inserted node a forward-edge budget of `M` at **every** layer, following the
HNSW paper and hnswlib. FAISS gives it `2*M` at level 0. Per-level *capacity* is identical in both
(`Mmax0 = 2*M`, `Mmax = M`) and the diversity-pruning rule is identical; only the new-node budget
differs.

**This convention difference has no measurable effect on graph density, and does not explain the
recall difference.** An earlier version of this section claimed it made Infinity's graph ~7% sparser
and therefore lower-recall. That was never measured, and it is wrong.

The budget is not the binding constraint. Measured at n=100000, `M=32`, `efConstruction=200`,
`participants=1` (deterministic), level-0 mean degree is **25.3** — below the `M=32` budget and well
below the 64-slot capacity. `SelectNeighborsHeuristic`'s diversity rule is what decides degree: it
rejects the large majority of the 200 `efConstruction` candidates, so the selection loop stops on
running out of acceptable candidates, not on reaching the cap. Two direct consequences, both
measured on level-0 directed edge counts:

| change | level-0 edges | delta |
| --- | --- | --- |
| baseline | 2,531,954 | — |
| forward budget raised to `2*M` at level 0 | 2,532,895 | **+941** (+0.04%) |
| FAISS-style `prune_headroom = 0.2` | 2,531,936 | **−18** (−0.001%) |

Raising the budget to FAISS's convention adds 941 edges out of 2.5M. Both knobs exist behind
`INFINITY_HNSW_LEVEL0_DOUBLE_BUDGET` and `INFINITY_HNSW_PRUNE_HEADROOM` in the native harness
bridge if you want to re-derive this.

Against FAISS at matched parameters the two engines build graphs of **essentially the same
density**, so there is no sparsity deficit to close:

| n | metric | Infinity | FAISS | FAISS denser by |
| --- | --- | --- | --- | --- |
| 12,288 | total directed edges | 533,484 | 574,667 | 7.2% |
| 100,000 | total directed edges | 2,586,903 | 2,595,348 | 0.33% |
| 100,000 | level-0 directed edges | 2,531,954 | 2,541,185 | 0.36% |
| 1,000,000 | level-0 directed edges | 29,437,305 | 30,047,512 | 2.1% |

The n=12,288 row is where the "~7% sparser" claim came from (see
[PRIOR_EFFORT_AUDIT.md](PRIOR_EFFORT_AUDIT.md)). It is a real measurement, and it does not
generalise: the same metric gives 0.33% at n=100,000. **Quote n with any density claim**, exactly as
[BASELINE.md](BASELINE.md) already warns for build-time ratios — 12,288 vectors is 1.2% of SIFT1M and
fixed setup effects dominate there.

Recall at equal `efSearch` still differs by a few points in either direction depending on `n`, and
that difference is **not yet explained** — density is ruled out, but the cause has not been
established, so do not attribute it to a specific mechanism without measuring. Comparing at
iso-recall remains the right practice for build-time claims; treat it as guarding against an
unexplained difference rather than a known one.

The convention difference itself is still a design difference rather than a defect, and changing it
would alter construction semantics for every Infinity index on every platform — but on this
evidence it would also buy essentially nothing.
