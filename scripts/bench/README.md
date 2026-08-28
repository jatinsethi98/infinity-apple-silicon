# HNSW build baseline: Infinity vs FAISS

`run_baseline.py` is the durable, minimal mechanism for comparing Infinity's HNSW
index-build against FAISS's on the **same** workload. Reuse it for every future
optimization experiment. It is one file; read it before you trust it.

## Run it

```bash
python3 scripts/bench/run_baseline.py \
  --dataset /tmp/d0test/d0-f32le-n12288-d128-seed0.bin \
  --n 12288 --d 128 --m 32 --efc 200 --ef 32,64,128 \
  --participants 12 --pairs 3 --query-count 256 --timeout 600 \
  --infinity-bin /path/to/infinity_hnsw_d0 \
  --faiss-bin    /path/to/faiss_hnsw_d0
```

Both binaries take identical argv
(`DATASET N D M EFC EFSEARCH CHUNK QUERYCOUNT PARTICIPANTS BUILDGRAIN AUDIT_SIDECAR`)
and print `key=value` lines on stdout (keys prefixed `infinity_`/`faiss_`). The driver
parses those; it does not need the whole server built. Prebuilt smoke binaries live at
`.../native-smoke-v1/build/native-hnsw-d0-release/{infinity,faiss}_hnsw_d0`.

Output goes to `scripts/bench/results/<UTC-timestamp>/`:
- `raw-<engine>-p<pair>-<order>.txt` — the exact argv + full stdout/stderr of each run.
- `results.json` — host preflight, every run, aggregates, the recall gate, and the ratio.
- A markdown table is printed to stdout (build medians, vectors/sec, ratio, recall + gate).

If the dataset file is missing it is regenerated deterministically with
`random.Random(seed).random()` float32 (seed 0, n=12288, d=128 reproduces the canonical
sha256 `f2e29c0f…`).

## The fairness contract (this is the whole thing)

1. Same dataset file, vector count, dimensions for both engines.
2. Same M and efConstruction; same thread count (`--participants`).
3. recall@10 compared at matched efSearch. The build-time ratio is **only reportable**
   when every compared efSearch has `|FAISS − Infinity| <= --recall-deficit-threshold`
   (default 0.005). Otherwise the run is flagged **RECALL-UNMATCHED** and the ratio is
   printed but marked *not comparable* — a faster build at lower recall is not a win.
4. `--pairs N` paired runs with **alternating order** (infinity-first, faiss-first, …),
   each engine in a fresh process, to cancel thermal/order drift.
5. Report **median + spread** (min, max, relative MAD). Never a single run.
6. Refuse to run on battery when `--require-ac` is set; otherwise warn loudly and record it.
7. Record host state: AC/battery, thermal pressure (`pmset -g therm`), CPU idle %
   (`top -l 1`), and competing processes (`ps -r`).

## Adding a new experiment arm

The driver compares exactly two binaries that share the argv/stdout contract above.
To test an optimization: build a new variant binary that still emits the same
`infinity_*` keys, then point `--infinity-bin` at it and `--faiss-bin` at the FAISS
reference (or at the previous Infinity binary for an A/B). Change nothing else — same
dataset, M, efC, participants — so the only variable is the code under test. Compare
`results.json` across timestamped runs. Do **not** add verification layers, schemas, or
provenance capture; if a number needs defending, re-run with more `--pairs` on AC power.

## Known limitations (read before quoting a number)

- **Build timing source.** These prebuilt (Aug-21) binaries emit `*_cold_build_ns`
  directly on stdout, so timing is the engine's own measured build time. Newer runner
  source (Aug-27) gates that block behind a 17-arg "campaign" protocol; if you rebuild
  and the driver can't find `*_cold_build_ns`, it falls back to **wall-clock of the
  subprocess** (which includes dataset load, recall audit, and query benchmark, so it is
  an overestimate) and sets `build_timing_used_wall_clock_fallback=true` in results.json
  plus a Notes line. Treat fallback numbers as directional only.
- **recall@10 is deterministic** (fixed seed) and independent of the timing run; the
  engine emits recall for fixed efSearch points {32,64,128,256,512} in a single run, so
  `--ef` only *selects* which of those to report — values outside that set come back n/a.
- **`--participants` sets the engine argv thread count**; the driver runs engines
  sequentially (never concurrently) so they don't contend.
- **Host noise is not eliminated, only exposed.** On battery or under competing load
  (e.g. a concurrent `clang-20` build) the relative MAD rises — that is the signal to
  re-run on AC/idle, not to trust the median. Preflight records both.
- The large per-run audit sidecar the engine writes is deleted after parsing; everything
  the driver reports comes from stdout, which is preserved verbatim in `raw-*.txt`.
