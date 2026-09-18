# Changelog

What changed in this repository relative to upstream Infinity 0.7.3, newest first.
Engine defects that are known and not yet fixed are in [docs/known-issues.md](docs/known-issues.md),
not here. Versions will follow `v0.7.3-apple.N` until the port and upstream converge.

## Unreleased

### Developer experience

- A `Makefile` wrapping every script: `make doctor`, `setup`, `start`, `demo`, `test`,
  `slt`, `recovery`, `package`, `bench`, `lint` and more; `make` lists them.
- `scripts/apple_silicon/doctor.sh`: a read-only prerequisite check with the fix for
  each missing item, including the bison requirement below.
- `install.sh`: installs a release tarball (or a local one from `make package`) under
  `~/.infinity`, verifies its SHA-256, and links `infinity` into `~/.local/bin`.
- `make package` now writes a `.sha256` beside the tarball and the release workflow
  publishes both.
- A `demo/rag_quickstart.py` that chunks, embeds, stores and hybrid-retrieves any set
  of text files, printing a ready-to-send prompt.
- `run_server.sh --config FILE` starts an instance from a config you wrote.
  `make_package.sh --version` names the tarball after the release tag, so
  `install.sh` keeps successive releases side by side; `--require-slt` makes the
  package self-test's query check mandatory, and CI passes it. The probe's BM25
  options were misspelled (`bm25_params`) and silently ignored; corrected.
- `make bench` refuses to run without the Accelerate-linked FAISS reference instead
  of silently falling back to the Homebrew build that flatters Infinity.
- The build-time and disk figures were measured on an M4 Mac mini during the local
  integration run: about 12 minutes and 20 GB, not the hour and 25 GB the docs said.

### Documentation

- The README is a front page again: pitch, quick start, one code sample, the
  benchmark table, honest status, and links.
- New human-facing documentation under `docs/`: getting started, concepts (vector
  search explained without jargon), Python and HTTP guides, a RAG tutorial,
  operations, tuning, benchmarking, known issues and an FAQ, with `docs/README.md`
  as the index. The engineering record under `docs/apple_silicon/` is unchanged.
- Removed upstream's Linux and Docker deployment pages, cluster guide, Linux benchmark
  report and 2024 release notes, which described a platform this fork does not build.
- `scripts/check_md_links.py` checks every relative link and anchor; it runs under
  `make lint` and in CI.
- Corrections from an adversarial review of the guides: Parquet import and `max`
  fusion are not in the Python SDK; a tensor search arm over HTTP uses the key
  `field`; browser `fetch` cannot send the GET-with-body reads; the exception class
  is `infinity.InfinityException`; and the configuration section of the operations
  guide now describes what `run_server.sh` actually does.

### Engine

- `Value::StringToValue` parsed float and double with `std::from_chars`, whose
  floating-point overload is implemented inside the libc++ library from LLVM 20 on.
  Compiling against Homebrew LLVM 20's headers but linking the macOS SDK's libc++
  works only when the SDK is recent enough to export it (Xcode 26 does, the macos-15
  CI runner's Xcode 16.4 does not), and a binary linked that way cannot load on an
  older macOS regardless of the 14.0 deployment target. It now uses `strtof`/`strtod`
  with the same strictness, which every supported macOS provides.

### CI

- Fixed the native macOS workflow's first run: the hosted runner has no Homebrew
  bison, so vcpkg's thrift port fell back to Xcode's bison 2.3, which rejects the
  `--file-prefix-map` flag thrift passes. The workflows, `setup.sh`,
  `build_server.sh` and `doctor.sh` now all install or check for bison 3.7+, and for
  `pkg-config`, which vcpkg's ports need and which a fresh Mac does not have (the
  dependency build died on abseil without it during the local integration run). With that
  and the `from_chars` fix above, the workflow went green for the first time on
  2026-09-18 (run 35317365990: build, unit tests, SQL suite, crash recovery and the
  package self-test, 50 minutes cold).
- A `lint` workflow on a Linux runner: shell syntax, shellcheck, Python compilation,
  Markdown links and YAML validity, in a couple of minutes.
- Dependabot for GitHub Actions versions.

### Community

- Issue templates that ask for `make doctor` output and the macOS version, a pull
  request checklist, a code of conduct, a security policy and this changelog.
- Removed the remaining Linux build and Docker release scripts from `scripts/`,
  along with `codecov.yml` and `.dockerignore`, none of which this fork used.

## Fork history, 2026-08-27 to 2026-09-11

Everything below is on `main` and is what the README's numbers and status describe.

### Port and tooling

- Native `arm64-apple-darwin` port of the engine, server, unit tests and packaging
  (`e793a5433`, `78316791e`); CMake presets `macos-arm64-release` and
  `macos-arm64-debug`.
- Linux and x86-64 support removed rather than left unverified; the build refuses a
  non-Darwin host (`136c63a82`). jemalloc, mold, the systemd unit and the RPM/DEB
  packaging went with them.
- `scripts/apple_silicon/setup.sh`, `build_server.sh`, `run_server.sh`, `run_slt.py`,
  `install_test_tools.sh`, `verify_crash_recovery.sh` and `make_package.sh`: one
  command from a clone to a running server, repo-local test tooling, a path-rewritten
  SQL suite that runs without root, a crash-recovery check, and a self-tested
  relocatable tarball (`eeb5ac628`, `886f5a0de`).
- A native macOS arm64 GitHub Actions workflow that builds, tests, checks recovery
  and packages (`cc8831f54`); upstream's Linux workflows and nightly release cron
  removed (`886f5a0de`).
- `demo/`: a hybrid-search demo over 36 articles with the ranks explained.

### Engine fixes (platform-independent unless noted)

- WAL commits are now durable: `fsync` plus `F_FULLFSYNC` per committed batch by
  default, with `fsync` and `no_sync` levels; the WAL directory is synced on create
  and rename; shutdown can no longer report an unsynced transaction as committed
  (`b2d59374a`).
- `RcuMultiMap::range()` iterated a `std::multimap` without the lock while inserts
  mutated it; the lock is now held for the traversal. Exported files are opened with
  `O_TRUNC` so a shorter export no longer leaves the previous file's tail
  (`ccc58a3da`).
- Float and double literals longer than one character were rejected on macOS by an
  Apple-only `sscanf` branch; both platforms now share one parser, which also stops
  silently accepting `1.5abc` (`4f2da91b0`).
- Unit tests find the analyzer dictionaries in a source checkout instead of only
  under `/usr/share/infinity` (`f80c88af7`); two date tests no longer depend on the
  timezone (`098801819`); the test harness honours the staging directory it is given
  (`1dcc31ba8`); the FST byte helpers use `memcpy` instead of an over-reading
  unaligned load (`136c63a82`).
- New tests for a dictionary-backed analyzer and float-valued BM25 options
  (`ba39cd7ae`), plus 19 end-to-end evaluation suites under `test/eval_macos/`.

### HNSW index-build performance on Apple Silicon

Each accepted only after a paired A/B campaign showed a significant improvement at
matched recall; the notebook is `docs/apple_silicon/BASELINE.md`.

- Whole-vector prefetch of candidates instead of their first cache line
  (`0d383c663`); skip prefetching candidates already visited, and a 128-byte prefetch
  stride matching the M-series cache line (`d95c0f7b5`, `ec8542f21`).
- A fused-multiply-add squared-L2 kernel and a four-accumulator unroll for Apple
  arm64, dispatched at runtime (`ff2f1d3d6`).
- Three allocation and call overheads removed from the search loop (`62338a474`);
  the visited set kept bit-packed and cleared by dirty word (`1375f707d`).
- Recall scored against the official SIFT1M ground truth rather than 64 synthetic
  queries, which moved the matched-recall operating point and corrected an earlier
  "sparser graph" explanation (`13969115d`, `4ceb8d975`, `70e4f714d`).
- A paired candidate-versus-control A/B driver with a recall gate, a knob-scan
  driver, and the repeatable Infinity-versus-FAISS baseline mechanism
  (`83015a601`, `ae1b717c6`, `06a8812d3`).

Result: 1.65× faster build than FAISS at equal parameters and 1.40× at matched recall
on an M4 Mac mini, 1.52× at matched recall on an M3 Pro, with 1.30× higher query
throughput. `docs/apple_silicon/BENCHMARKS.md` has the method and raw output.
