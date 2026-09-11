#!/usr/bin/env bash
# Sweep HNSW build-task granularity (buckets per worker) and record the effect on
# index-build time, QPS and recall. buckets-per-worker = 1 reproduces the old
# one-contiguous-bucket-per-worker behaviour, so it is the control arm.
#
# Usage: scripts/bench/sweep_build_buckets.sh <results-dir> [buckets-per-worker ...]
#
# Environment:
#   DATASET       base vectors (default: datasets/sift1m/base.f32, which
#                 `python3 scripts/bench/fetch_datasets.py sift1m` downloads)
#   INFINITY_BIN  Infinity arm binary (default: build/bench/infinity_hnsw_d0)
#   FAISS_BIN     FAISS arm binary. The default is the Accelerate-linked build from
#                 scripts/apple_silicon/build_faiss_accelerate.sh; the Homebrew
#                 bottle links OpenBLAS and is ~1.5x slower, which flatters Infinity.
#   PARTICIPANTS  build/query threads (default: all cores)
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT="${1:?usage: sweep_build_buckets.sh <results-dir> [bpw ...]}"
shift
BPW_LIST=("$@")
if [ "${#BPW_LIST[@]}" -eq 0 ]; then
  BPW_LIST=(1 4 8 16)
fi

# Repo-relative, matching fetch_datasets.py's DEFAULT_ROOT of <repo>/datasets.
DATASET=${DATASET:-datasets/sift1m/base.f32}
# bench-faiss-src holds only the FAISS target that build_faiss_accelerate.sh builds;
# the Infinity arm comes from the `bench` preset's build directory.
INFINITY_BIN=${INFINITY_BIN:-build/bench/infinity_hnsw_d0}
FAISS_BIN=${FAISS_BIN:-build/bench-faiss-src/faiss_hnsw_d0}   # from-source, Apple Accelerate
PARTICIPANTS=${PARTICIPANTS:-$(sysctl -n hw.ncpu)}

for path in "$DATASET" "$INFINITY_BIN" "$FAISS_BIN"; do
    [[ -e $path ]] || {
        printf 'error: %s does not exist.\n' "$path" >&2
        printf '       See the "Reproduce the benchmarks" section of README.md for how to\n' >&2
        printf '       produce the datasets and binaries this sweep needs.\n' >&2
        exit 1
    }
done

mkdir -p "$OUT"
export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"

for bpw in "${BPW_LIST[@]}"; do
  echo "=============== buckets_per_worker=${bpw} ==============="
  export INFINITY_HNSW_BUILD_BUCKETS_PER_WORKER="$bpw"
  python3 scripts/bench/run_baseline.py \
    --dataset "$DATASET" --n 1000000 --d 128 \
    --m 32 --efc 200 --ef 32,64,128,256,512 \
    --participants "$PARTICIPANTS" --pairs 2 \
    --chunk-size 8192 --query-count 1000 --build-grain 1 --timeout 1800 \
    --infinity-bin "$INFINITY_BIN" --faiss-bin "$FAISS_BIN" \
    > "${OUT}/bpw-${bpw}.log" 2>&1
  echo "  -> ${OUT}/bpw-${bpw}.log"
  grep -E '^\| (Infinity|FAISS) \||ratio|buckets/worker' "${OUT}/bpw-${bpw}.log" | head -12 || true
done

echo
echo "=============== SUMMARY ==============="
python3 scripts/bench/summarize_sweep.py "$OUT"
