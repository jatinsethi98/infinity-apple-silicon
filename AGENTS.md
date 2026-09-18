# Working in this repository

Instructions for coding agents and for people who want the same shortcuts. This is the
native Apple Silicon port of [Infinity](https://github.com/infiniflow/infinity); macOS
on arm64 is the **only** supported platform and the build refuses anything else.

## Layout

| Path | What lives there |
|---|---|
| `src/` | The C++23 engine: `parser/`, `planner/`, `executor/` (query pipeline), `storage/` (buffer manager, WAL, indexes), `network/` (thrift, HTTP, PostgreSQL wire). |
| `python/infinity_sdk/` | The Python SDK (`import infinity`, distribution `infinity-sdk`). |
| `scripts/apple_silicon/` | Setup, build, run, test and package scripts. Every one prints `--help`. |
| `scripts/bench/` | Benchmark drivers and the fairness contract. |
| `tools/apple_silicon/native_hnsw_smoke/` | The standalone harness that compiles the production HNSW code without vcpkg. |
| `test/sql/` | SQL logic tests (`.slt`); `test/eval_macos/` the end-to-end evaluation suites; `src/unit_test/` the C++ unit tests. |
| `demo/` | The hybrid-search demo and the RAG quickstart. |
| `docs/` | Human-facing docs; `docs/apple_silicon/` the engineering record; `docs/references/` upstream's API reference. |
| `resource/` | Submodule: full-text analyzer dictionaries. Required for CJK analyzers and their tests. |

## Commands

`make` lists everything. The ones you will use:

```sh
make doctor        # prerequisites, changes nothing
make setup         # toolchain + vcpkg + server build (10 to 15 minutes on an M4, first time)
make build         # rebuild the server;  make build-tests adds test_main
make start | stop | status | logs
make test          # C++ unit tests
make slt           # SQL logic suite on a fresh instance
make recovery      # crash-recovery check
make package       # relocatable tarball, self-tested
make lint          # shellcheck + Markdown link check
make bench-build bench-datasets bench   # HNSW harness vs FAISS
```

Prefer the `make` targets or the scripts over raw `cmake`, `ninja` or the binary:

- `scripts/apple_silicon/build_server.sh <preset> <targets...>` validates the CMake
  range (4.0.3 to 4.4.x; `import std` is gated on a version-specific key), `SDKROOT`,
  `VCPKG_ROOT`, the vcpkg baseline and bison 3.7+, each of which otherwise fails with
  a message that names the wrong cause. Build output: `build/macos-arm64-release/src/{infinity,test_main}`;
  build log: `build/<preset>-build.log`.
- `scripts/apple_silicon/run_server.sh start [--instance NAME] [--port-offset N] [--fresh]`
  renders a usable config into `build/instances/<name>/` because the shipped one
  points at root-owned `/var/infinity`. Logs are at `build/instances/<name>/log/infinity.log`.
- Toolchain: Homebrew LLVM 20 (Apple Clang cannot build this), CMake 4.0.3 to 4.4.x,
  Ninja, libomp, bison 3.7+. `SDKROOT` must be exported; the scripts and the
  Makefile do it.

## Rules

- **C++23**, formatted with `clang-format-20` against `.clang-format`. Match the
  surrounding code. Comments explain *why*.
- **Python**: always `uv` (`uv run`, `uv sync`), never the system interpreter or `pip`.
- **Tests before a PR**: `make build-tests && make test`, and `make slt` for anything
  touching the engine or SQL. The five documented unit-test exclusions are in
  `.github/workflows/macos_arm64.yml` with their reasons.
- **Performance claims** follow `scripts/bench/README.md`: paired alternating runs
  (`ab_build.py`), recall on the official ground truth, comparison at matched recall,
  result recorded in `docs/apple_silicon/BENCHMARKS.md` with raw output under
  `docs/apple_silicon/benchmarks/`.
- **Do not describe this engine as production-ready.** `docs/known-issues.md` lists
  eleven open defects, four of which crash the server from an ordinary client.
- **Do not re-add Linux or x86-64 paths.** Contribute platform-independent fixes
  upstream instead.
- **Docs**: every Markdown link is checked by `scripts/check_md_links.py` in CI. Keep
  `docs/README.md`, `CHANGELOG.md` and `docs/known-issues.md` current when a change
  affects them.

## SDK details that cost the most time

- `infinity.connect()` takes a `NetworkAddress`, never a string; a string raises
  `INVALID_SERVER_ADDRESS`, which reads like the server is down.
- `to_pl()`, `to_df()` and `to_arrow()` return a `(result, extra)` **tuple**.
- Index parameter *values* are strings; HNSW's key is lowercase `m` over the SDK
  and uppercase `M` over HTTP.
- `score()` needs a full-text, tensor or fusion arm; a dense-only query exposes
  `_similarity` (cosine, ip) or `_distance` (l2).
- Ports: 23817 SDK/thrift, 23820 HTTP, 5432 PostgreSQL wire, 23850 peer.
- There is no embedded mode. `import_data` and `export_data` paths are resolved by
  the server.
