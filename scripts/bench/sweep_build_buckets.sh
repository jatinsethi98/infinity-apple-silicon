#!/usr/bin/env bash
# Sweep HNSW build-task granularity (buckets per worker) and record the effect on
# index-build time, QPS and recall. buckets-per-worker = 1 reproduces the old
# one-contiguous-bucket-per-worker behaviour, so it is the control arm.
#
# Usage: scripts/bench/sweep_build_buckets.sh <results-dir> [buckets-per-worker ...]
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT="${1:?usage: sweep_build_buckets.sh <results-dir> [bpw ...]}"
shift
BPW_LIST=("$@")
if [ "${#BPW_LIST[@]}" -eq 0 ]; then
  BPW_LIST=(1 4 8 16)
fi

DATASET=/Users/sethjatq/Desktop/proj/datasets/sift1m/base.f32
INFINITY_BIN=build/bench-faiss-src/infinity_hnsw_d0
FAISS_BIN=build/bench-faiss-src/faiss_hnsw_d0   # from-source, Apple Accelerate

mkdir -p "$OUT"
export SDKROOT="$(xcrun --show-sdk-path)"

for bpw in "${BPW_LIST[@]}"; do
  echo "=============== buckets_per_worker=${bpw} ==============="
  export INFINITY_HNSW_BUILD_BUCKETS_PER_WORKER="$bpw"
  python3 scripts/bench/run_baseline.py \
    --dataset "$DATASET" --n 1000000 --d 128 \
    --m 32 --efc 200 --ef 32,64,128,256,512 \
    --participants 12 --pairs 2 \
    --chunk-size 8192 --query-count 1000 --build-grain 1 --timeout 1800 \
    --infinity-bin "$INFINITY_BIN" --faiss-bin "$FAISS_BIN" \
    > "${OUT}/bpw-${bpw}.log" 2>&1
  echo "  -> ${OUT}/bpw-${bpw}.log"
  grep -E '^\| (Infinity|FAISS) \||ratio|buckets/worker' "${OUT}/bpw-${bpw}.log" | head -12 || true
done

echo
echo "=============== SUMMARY ==============="
python3 scripts/bench/summarize_sweep.py "$OUT"
