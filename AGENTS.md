# Infinity Project Instructions for GitHub Copilot

This file provides context, build instructions, and coding standards for the Infinity project.
It is structured to follow GitHub Copilot's [customization guidelines](https://docs.github.com/en/copilot/concepts/prompting/response-customization).

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

## 7. macOS / Apple Silicon (this fork)
This repository is the native `arm64-apple-darwin` port of Infinity. On macOS do not use the
Linux build commands above; use the presets and scripts documented in
`docs/apple_silicon/README.md`:
- Toolchain: Homebrew LLVM 20, CMake 4.0.3+, Ninja; `SDKROOT` must be exported.
- Benchmark harness (no vcpkg): `scripts/apple_silicon/bootstrap_ctpl.sh`, then
  `cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke` and
  `cmake --build build/bench --target infinity_hnsw_d0 faiss_hnsw_d0`.
- Full server: `cmake --preset macos-arm64-release` with `VCPKG_ROOT` set.
- Any performance claim must follow `scripts/bench/README.md` (paired runs, Accelerate-linked
  FAISS from `scripts/apple_silicon/build_faiss_accelerate.sh`, recall on official ground truth)
  and be recorded in `docs/apple_silicon/BENCHMARKS.md` with raw output under
  `docs/apple_silicon/benchmarks/`.
