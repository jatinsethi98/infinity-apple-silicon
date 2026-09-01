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

`run_baseline.py` compares Infinity against **FAISS**. To test an optimization, build a
variant binary that still emits the same `infinity_*` keys, point `--infinity-bin` at it,
keep `--faiss-bin` on the FAISS reference, and change nothing else — same dataset, M, efC,
participants. Compare `results.json` across timestamped runs.

### For old-vs-new Infinity, use `ab_build.py` — not this driver

An earlier version of this file said you could A/B two Infinity binaries by pointing
`--faiss-bin` at the previous one. **That silently produces wrong numbers.** The harness
binary derives its stdout key prefix from its own compiled-in engine identity
(`hnsw_d0_runner.cpp`), so an Infinity binary always emits `infinity_*`. As the `faiss`
arm it misses `faiss_cold_build_ns`, misses the unprefixed fallback, and gets downgraded
to `wall_clock_fallback` with no recall values — which also silently defeats the recall
gate. Do not do it.

Comparing two independent `run_baseline.py` campaigns is also statistically too weak for
the effect sizes that matter here. The observed 6-run relative MAD on the 1M build is
0.6–1.4%, and one recorded run held a 57.8 s thermal outlier against a 49.0 s minimum;
with `SE(median) ≈ 1.858·MAD/√n`, a 6-vs-6 difference of medians has a 95% half-width of
1.4–3.0%. A real 1–2% win is not distinguishable that way, and "the median improved" is
not evidence.

`scripts/bench/ab_build.py` does it properly: both binaries interleaved inside one
campaign in randomized per-block order, decided on a two-sided 95% t interval over the
per-block log ratios `ln(t_candidate/t_control)`, with min-of-N as a
contamination-robust cross-check. Keep a change only when the interval excludes 1.0.

```bash
python3 scripts/bench/ab_build.py \
  --dataset /Users/sethjatq/Desktop/proj/datasets/sift1m/base.f32 \
  --n 1000000 --d 128 --m 32 --efc 200 --participants 12 \
  --blocks 15 --require-ac --label my-change \
  --control-bin  build/bench/infinity_hnsw_d0.control \
  --candidate-bin build/bench/infinity_hnsw_d0
```

Run it with `--control-bin` and `--candidate-bin` pointing at the **same** file first: that
measures the harness noise floor and tells you the smallest effect the campaign can
resolve. Do **not** add verification layers, schemas, or provenance capture; if a number
needs defending, add blocks.

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
