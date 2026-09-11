# Infinity Project Instructions for Coding Agents

This file provides context, build instructions, and coding standards for the Infinity project.
It is structured to follow GitHub Copilot's [customization guidelines](https://docs.github.com/en/copilot/concepts/prompting/response-customization).

> **READ THIS FIRST.** This repository is the **macOS / Apple Silicon fork** of Infinity.
> The build refuses any non-Darwin host. Sections 3 to 6 below are inherited from upstream
> and describe the **Linux** workflow; on this fork they are wrong in specific ways
> (`cmake-build-debug`, `ymake`, `/var/infinity/...`). **Section 7 is the authoritative
> build and test procedure here** — go there first, and treat 3-6 as background on how
> upstream does it.
>
> The single command that works from a fresh clone is:
> ```sh
> scripts/apple_silicon/setup.sh
> ```

## 1. Project Overview
Infinity is an AI-native database built for LLM applications, providing incredibly fast hybrid search of dense embedding, sparse embedding, tensor, and full-text.

- **Core**: C++23 (requires Clang 20+)
- **SDK**: Python 3.11+

- **Architecture**: Single-binary.
  - `src/parser`, `src/planner`, `src/executor`: Query pipeline.
  - `src/storage`: Storage engine.
  - `src/network`: Communication.

## 2. Directory Structure
- `src/`: Core database engine source code.
  - `parser/`, `planner/`, `executor/`: Query processing pipeline.
  - `storage/`: Storage engine (buffer manager, wal, etc.).
  - `network/`: Server-client communication.
- `python/`: Python SDK (`infinity_sdk`) and Python-based tests.
- `test/`:
  - `sql/`: SQL Logic Tests (`.slt` files) for functional verification.
  - `data/`: Test datasets.
- `benchmark/`: Performance benchmarking tools.
- `cmake/`: CMake modules and dependency finders.
- `scripts/`: CI/CD and build helper scripts.

## 3. Build Instructions
The project uses **CMake** with **Ninja** generator and requires **Clang 20+**.

### Core Engine (C++)
- **Standard Build**:
  ```bash
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -S . -B cmake-build-debug
  cmake --build cmake-build-debug --target infinity
  ```
- **Developer Shortcut** (if `ymake` is available):
  - `ymake 1 d infinity`: Build infinity target in debug mode.

### Python SDK
Use `uv` for package management.
```bash
(cd python/infinity_sdk && uv build)
```

## 4. Testing Instructions

### Unit Tests (C++)
- **Build**: `ymake 1 d test_main` or `cmake --build cmake-build-debug --target test_main`
- **Run**: `./cmake-build-debug/src/test_main`

### Python SDK Tests
Located in `python/`. Use `uv` to manage the environment.

1. **Setup Environment**:
   ```bash
   uv sync --python 3.11 --frozen --all-extras
   source .venv/bin/activate
   # uv pip install python/infinity_sdk # This is handled by uv sync if infinity_sdk is in workspace
   ```

2. **Run Tests**:
   - **Pre-requisite**: Start Infinity server first!
     ```bash
     # Default configuration
     # Note: The log of infinity is at /var/infinity/log/infinity.log
     nohup ./cmake-build-debug/src/infinity > /dev/null 2>/dev/null &

     # To enable debug/trace logging:
     # 1. Create a config file (e.g., cp conf/infinity_conf.toml my_conf.toml)
     # 2. Set `log_level = "debug"` or `"trace"` in the config file.
     # 3. Start with -f:
     # Note: The log of infinity is at /var/infinity/log/infinity.log
     nohup ./cmake-build-debug/src/infinity -f my_conf.toml > /dev/null 2>/dev/null &
     ```
   - **Run Test**:
     ```bash
     uv run pytest python/test/cases/test_basic.py::TestInfinity::test_basic
     ```
   - **Note**: `local infinity` mode is deprecated.

### Functional Tests (SQL)
- SQL Logic Tests (`.slt`) are located in `test/sql`.
- **Requirement**: Ensure `sqllogictest` binary is in your system `PATH`.

## 5. Coding Standards & Guidelines
- **C++ Standard**: C++23. Use modern features but ensure compatibility with Clang 20.
- **C++ Formatting**: **MUST** use `clang-format-20` and follow `.clang-format` configuration.
- **Dependencies**: Managed via `vcpkg` (integrated in CMake).
- **Python**: **ALWAYS** use `uv` instead of `pip` for installing Python packages. **ALWAYS** use the python interpreter and pytest from the virtual environment created by `uv` at the project root (e.g. `uv run python`, `uv run pytest`), NOT the system ones.

## 6. Shell Command Execution Guidelines
When executing shell commands:
- NEVER prefix commands with environment variable assignments such as "PYTHONPATH=...", "ENV=...", or "FOO=bar cmd".
- If an environment variable is required, ask the user to configure it permanently (e.g., via pytest.ini, .env, or shell profile) instead of injecting it inline.
- ONLY execute direct commands like "pytest", "python -m pytest", or "uv run pytest".
- NEVER use chained commands with "&&", ";", or "|".

## 7. macOS / Apple Silicon (this fork) — AUTHORITATIVE
This repository is the native `arm64-apple-darwin` port of Infinity, and the **only**
supported platform. Do not use the Linux commands in sections 3-6. Full detail is in
`docs/apple_silicon/README.md`.

**Setup and build.** One command, safe to re-run, skips whatever is already done:
```sh
scripts/apple_silicon/setup.sh [--with-tests] [--no-build] [--skip-brew]
```
It installs the Homebrew toolchain, initialises the `resource` submodule, bootstraps
vcpkg at the pinned baseline, then delegates to `build_server.sh`. Prefer
`scripts/apple_silicon/build_server.sh <preset> <targets...>` over a bare
`cmake --preset`: it validates the CMake version at **both** ends of the supported
range (4.0.3 up to 4.4.x; `import std` is gated behind a version-specific UUID),
resolves `SDKROOT` and `VCPKG_ROOT`, and repairs a shallow vcpkg baseline.

- Toolchain: Homebrew LLVM 20 (Apple Clang cannot build this), CMake 4.0.3–4.4.x, Ninja.
- Build output: `build/macos-arm64-release/src/{infinity,test_main}`. Build logs go to
  `build/<preset>-build.log`, not to stdout.

**Running the server.** Use the wrapper, never the raw binary:
```sh
scripts/apple_silicon/run_server.sh start [--fresh] [--instance NAME] [--port-offset N]
```
The shipped `conf/infinity_conf.toml` points every directory at root-owned
`/var/infinity/*` and `resource_dir` at `/usr/share/infinity/resource`, which cannot
work from a checkout on macOS. The wrapper renders a usable config into
`build/instances/<name>/`. **Logs are at `build/instances/<name>/log/infinity.log`**, not
`/var/infinity/log/infinity.log` as section 4 says.

**Tests.**
```sh
./build/macos-arm64-release/src/test_main        # unit tests
scripts/apple_silicon/install_test_tools.sh      # sqllogictest, checksum-pinned
scripts/apple_silicon/run_server.sh start --instance ci --fresh   # --fresh: the SLT
uv run python scripts/apple_silicon/run_slt.py                    # suite is not idempotent
scripts/apple_silicon/verify_crash_recovery.sh --rows 2000
```
`python/test_pysdk/` is a good API reference but depends on `/var/infinity/test_data/`,
which is not creatable without root here. The suites that actually pass on macOS live in
`test/eval_macos/`.

**Benchmark harness** (independent of the server build, no vcpkg):
`scripts/apple_silicon/bootstrap_ctpl.sh`, then
`cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke` and
`cmake --build build/bench --target infinity_hnsw_d0 faiss_hnsw_d0`.

**Any performance claim** must follow `scripts/bench/README.md` (alternating paired runs,
Accelerate-linked FAISS from `scripts/apple_silicon/build_faiss_accelerate.sh`, recall
against the official ground truth, comparison at matched recall) and be recorded in
`docs/apple_silicon/BENCHMARKS.md` with raw output under `docs/apple_silicon/benchmarks/`.

**Python SDK gotchas** that cost the most time:
- Distribution is `infinity-sdk`; the import name is `infinity`.
- `infinity.connect()` takes a `NetworkAddress`, never a string. A string raises
  `INVALID_SERVER_ADDRESS`, which reads like the server is down.
- `to_pl()` / `to_df()` / `to_arrow()` return a `(dataframe, extra_result)` **tuple**.
- Index parameter *values* must be strings, and HNSW's `m` is lowercase over the SDK
  (uppercase `M` is accepted only over the HTTP API).
- `score()` requires a full-text, tensor or fusion arm; a dense-only query exposes
  `_similarity` (cosine/ip) or `_distance` (l2).
- Ports: 23817 SDK/thrift, 23820 HTTP, 5432 PostgreSQL wire.

**Known state.** `docs/apple_silicon/MACOS_VERIFICATION.md` records what is verified;
`docs/apple_silicon/EVALUATION.md` records eleven open blockers, including four ways a
client can crash the server. Do not describe this engine as production-ready.
