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
differs. The consequence is that Infinity builds a ~7% sparser graph and therefore has lower
recall at the same `efSearch`, so **a build-time comparison at equal `efConstruction` is not
apples-to-apples.** Compare at iso-recall instead. This is a design difference, not a defect —
changing it would alter construction semantics for every Infinity index on every platform.
