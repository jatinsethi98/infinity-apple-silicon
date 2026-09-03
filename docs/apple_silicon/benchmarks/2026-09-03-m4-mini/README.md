# Raw output, Mac mini M4, 2026-09-03

Everything summarised in [BENCHMARKS.md](../../BENCHMARKS.md) for the M4 mini comes from these files.
Per-query id lists and per-query checksums were stripped from the raw stdout copies to keep the
repository small; every aggregate, recall, latency and checksum line is intact.

| file | what it is |
|---|---|
| `host.txt` | machine, OS, compiler, FAISS build details, fork commit |
| `results-paired-infinity-vs-faiss-accelerate.json` | `run_baseline.py` output: two alternating pairs, Infinity vs FAISS 1.15 built from source against Accelerate |
| `run_baseline-accelerate.log` | the driver's printed summary for that run |
| `raw/accelerate-raw-*.txt` | engine stdout for each of the four runs in that campaign |
| `results-single-infinity-vs-faiss-homebrew.json`, `run_baseline-homebrew.log`, `raw/homebrew-raw-*.txt` | the first run of the day against the Homebrew (OpenBLAS) FAISS bottle; kept to document why that bottle is not the reference |
| `raw/infinity-sift1m.txt` | Infinity alone on SIFT1M, efC=200, official-truth recall |
| `raw/infinity-efc235.txt` | Infinity alone at efC=235, the iso-recall point |
| `raw/infinity-c768.txt`, `raw/infinity-c1536.txt` | clustered synthetic 768-d (200k) and 1536-d (100k) |
| `raw/infinity-d768.txt`, `raw/infinity-d1536.txt` | uniform-random synthetic, the HNSW worst case |
| `engines.json` | the Python-library rows (faiss-cpu wheel, hnswlib, usearch) as measured by `scripts/bench/bench_python_libs.py` |

Synthetic datasets were generated with numpy: uniform is `default_rng(0).random((n, d), float32)`;
clustered is `k` standard-normal centres (2000 for 768-d with seed 1, 1000 for 1536-d with seed 2),
each point a random centre plus 0.35 × standard-normal noise, unit-normalised. SIFT1M came from
`scripts/bench/fetch_datasets.py sift1m` (sha256 of `base.f32`
`ac8e11ed111b09e882bcf0ef1a77ebdd36c89f495bce8fd596603f69c35148e6`).
