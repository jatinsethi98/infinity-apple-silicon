# Benchmarking

How to reproduce the published numbers on your own Mac, how to read what comes out,
and how to measure a change without fooling yourself. The harness compiles the
production HNSW code directly, so none of this needs the server build or vcpkg.

## Reproduce the README table (about 15 minutes)

```sh
make bench-build       # harness: brew install llvm@20 cmake ninja libomp faiss simde, then cmake
make bench-faiss       # the fair FAISS reference, built from source against Accelerate
make bench-datasets    # SIFT1M, 168 MB, from a slow academic mirror; can take a while
make bench             # paired Infinity vs FAISS, two alternating pairs
```

`make bench` runs the same command the published table came from:

```sh
HNSW_D0_EXTERNAL_QUERIES=$PWD/datasets/sift1m/query.f32 \
HNSW_D0_EXTERNAL_GROUNDTRUTH=$PWD/datasets/sift1m/groundtruth.i32 \
python3 scripts/bench/run_baseline.py \
  --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 --efc 200 \
  --ef 32,64,128,256 --participants 10 --pairs 2 \
  --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0
```

`--participants` is the build thread count. `make bench` uses your core count; pass
`PARTICIPANTS=10` to match the published M4 run instead. The two environment
variables score recall against SIFT1M's official 10,000 queries; without them the
harness falls back to a 64-query synthetic self-audit that is fine for A/B signal and
wrong for any published recall number.

Why the FAISS reference is built from source: the Homebrew bottle links OpenBLAS and
builds the same graph about 1.5× slower than FAISS against Apple's Accelerate
framework. Comparing against the bottle inflates Infinity's ratio; `make bench` uses
the Accelerate build automatically when it exists.

## Reading the output

Each run writes `scripts/bench/results/<timestamp>/`:

| file | contents |
|---|---|
| `raw-<engine>-p<pair>-<order>.txt` | the exact command and full output of every process |
| `results.json` | host state at preflight, every run, medians and spread, the recall gate, the ratio |
| stdout | a Markdown table: build medians, vectors/s, ratio, recall per `ef` |

Three things to check before quoting a number:

1. **Spread.** The table gives min, max and relative MAD across pairs. FAISS varies by
   about 0.1% run to run; Infinity by about 1% on a machine with a browser open. A
   large spread means thermal or load noise, not a result.
2. **The recall gate.** The driver prints `RECALL-UNMATCHED` when the engines differ
   by more than 0.005 at any `ef`. That gate uses the synthetic self-audit; for the
   published comparison read the `*_external_recall_at_10_ef_*` keys in the raw
   output, which are the official-ground-truth numbers, and see
   [BENCHMARKS.md](../apple_silicon/BENCHMARKS.md#infinity-vs-faiss-paired) for why the
   two disagree.
3. **Host state.** `results.json` records AC or battery, thermal pressure, CPU idle
   and competing processes. On battery the numbers are not comparable to anything.

## Measuring a change of your own

Two Infinity binaries are **not** compared by pointing `--faiss-bin` at the old one;
the harness derives its output keys from the compiled engine and the comparison
silently breaks. Use the A/B driver, which interleaves both binaries in randomized
blocks inside one campaign and decides on a 95% confidence interval:

```sh
cp build/bench/infinity_hnsw_d0 build/bench/infinity_hnsw_d0.control
# ... make the change, rebuild the harness ...
python3 scripts/bench/ab_build.py \
  --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 --efc 200 \
  --participants 10 --blocks 15 --require-ac --label my-change \
  --control-bin build/bench/infinity_hnsw_d0.control \
  --candidate-bin build/bench/infinity_hnsw_d0
```

Run it once with both binaries pointing at the same file first: that measures the
noise floor and tells you the smallest effect the campaign can resolve. Keep a change
only when the interval excludes 1.0. The optimization campaign behind the current
numbers discarded several changes that looked like wins under a single unpaired run;
[BASELINE.md](../apple_silicon/BASELINE.md) is the notebook.

## Comparing with Python libraries

Same data and parameters, through each library's Python package:

```sh
uv run --python 3.12 --with numpy,hnswlib   python scripts/bench/bench_python_libs.py hnswlib
uv run --python 3.12 --with numpy,usearch   python scripts/bench/bench_python_libs.py usearch
uv run --python 3.12 --with numpy,faiss-cpu python scripts/bench/bench_python_libs.py faiss
```

These rows include the Python wrapper's overhead, which affects QPS and latency more
than build time.

## Embedding-sized vectors

SIFT is 128-dimensional. The README's 768- and 1,536-dimension rows use clustered
synthetic data (Gaussian clusters, unit-normalized) because that has the low intrinsic
dimension real embeddings have; uniform random data is the worst case for a graph
index and understates every engine by about 3×. The harness accepts any little-endian
float32 file with the matching `--n` and `--d`, so a dump of your own embeddings is the
best benchmark of all. Recall for such a run comes from the synthetic self-audit unless
you also supply queries and ground truth in the SIFT format.

## The fairness rules

The contract every published number follows, in one place:

1. Same dataset file, vector count, dimension, `m`, `ef_construction` and thread count
   for every engine.
2. FAISS built from source against Accelerate; never the Homebrew bottle.
3. Build time is the engine's own clock around construction, not the process wall
   clock.
4. Paired, alternating-order runs in fresh processes; medians with spread; never a
   headline from a single run.
5. Recall on the official ground truth for any parity claim.
6. Every ratio quoted with its N. Ratios at 12,288 vectors do not transfer to a million.

The full statement, and the limitations of the current numbers, are in
[BENCHMARKS.md](../apple_silicon/BENCHMARKS.md) and
[scripts/bench/README.md](../../scripts/bench/README.md). A pull request that claims a
speedup is expected to follow them; see [CONTRIBUTING.md](../../CONTRIBUTING.md).
