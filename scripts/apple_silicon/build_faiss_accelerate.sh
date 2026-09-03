#!/usr/bin/env bash
# Build FAISS from source against Apple's Accelerate BLAS and relink the benchmark harness's
# faiss_hnsw_d0 against it. The Homebrew faiss bottle links OpenBLAS and is ~1.5x slower on
# HNSW build, so it must not be used as the reference (see docs/apple_silicon/BENCHMARKS.md).
# Result: build/bench-faiss-src/faiss_hnsw_d0
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FAISS_TAG=${FAISS_TAG:-v1.15.0}
WORK=${FAISS_WORK:-$ROOT/build/faiss-accelerate}
export SDKROOT=${SDKROOT:-$(xcrun --show-sdk-path)}
mkdir -p "$WORK"
[ -d "$WORK/src" ] || git clone --depth 1 --branch "$FAISS_TAG" https://github.com/facebookresearch/faiss.git "$WORK/src"
cmake -S "$WORK/src" -B "$WORK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_METAL=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DBLA_VENDOR=Apple -DOpenMP_ROOT=/opt/homebrew/opt/libomp \
  -DCMAKE_INSTALL_PREFIX="$WORK/install" -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build "$WORK/build" -j"$(sysctl -n hw.ncpu)"
cmake --install "$WORK/build"
otool -L "$WORK/install/lib/libfaiss.dylib" | grep -q Accelerate || { echo "libfaiss is not linked against Accelerate"; exit 1; }
cmake -S "$ROOT/tools/apple_silicon/native_hnsw_smoke" -B "$ROOT/build/bench-faiss-src" -G Ninja -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DINFINITY_NATIVE_CTPL_INCLUDE_DIR="$ROOT/vcpkg_installed/arm64-osx/include" \
  -DINFINITY_NATIVE_SMOKE_WITH_FAISS=ON \
  -DFAISS_INCLUDE_DIR="$WORK/install/include" -DFAISS_LIBRARY="$WORK/install/lib/libfaiss.dylib"
cmake --build "$ROOT/build/bench-faiss-src" --target faiss_hnsw_d0
echo "built $ROOT/build/bench-faiss-src/faiss_hnsw_d0 (Accelerate FAISS $FAISS_TAG)"
