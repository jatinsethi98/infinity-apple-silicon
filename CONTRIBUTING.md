# Contributing

This repository is the native Apple Silicon port of
[Infinity](https://github.com/infiniflow/infinity). Where you should file something
depends on which of the two it belongs to.

**Belongs here** — anything about the port or the Mac:

- It does not build, configure, link or start on an Apple Silicon Mac.
- A test fails here but passes upstream on Linux.
- The macOS tooling (`scripts/apple_silicon/`, `scripts/bench/`, `demo/`) is wrong,
  unclear, or assumes something about your machine that is not true.
- The benchmark numbers do not reproduce, or the method behind them is unsound.
- One of the [known blockers](docs/apple_silicon/EVALUATION.md) — or a new one.

**Belongs upstream** — anything about Infinity itself: SQL semantics, the query
planner, index algorithms, the Python SDK's API, feature requests for the database.
File those at [infiniflow/infinity](https://github.com/infiniflow/infinity/issues/new/choose)
so they reach the people who maintain that code, and so a fix benefits every platform
rather than this fork alone.

## Reporting a bug

Open a [GitHub issue](https://github.com/jatinsethi98/infinity-apple-silicon/issues/new/choose)
and include:

- `sw_vers -productVersion` and `sysctl -n machdep.cpu.brand_string`
- `cmake --version` and `/opt/homebrew/opt/llvm@20/bin/clang++ --version`
- the commit you are on (`git rev-parse --short HEAD`)
- for a build failure, the tail of `build/<preset>-build.log`
- for a runtime failure, the tail of `build/instances/<name>/log/infinity.log`

The single most common report is not a bug: `infinity.connect()` takes a
`NetworkAddress`, not a string, and passing a string raises
`INVALID_SERVER_ADDRESS`, which reads like the server is down.

## Sending a change

```sh
git checkout -b my-change
# ... work ...
scripts/apple_silicon/build_server.sh macos-arm64-release infinity test_main
./build/macos-arm64-release/src/test_main
git push origin my-change
```

Then open a pull request. `.github/workflows/macos_arm64.yml` runs the build, the
unit tests, the SQL logic suite, a crash-recovery check and a packaging self-test on
a native `macos-15` arm64 runner.

### Before you open it

- Build and run the tests locally first. A macOS runner is slow and billed at a
  multiplier; a cold run here is roughly an hour.
- Keep one concern per pull request.
- Add a test for a fix. If the bug was a crash, the test should be the call that
  crashed.
- If you touch a `scripts/apple_silicon/` script, run it. Several of them are the
  only thing standing between a reader and a cryptic toolchain error, so a
  regression there is worse than it looks.

### Performance claims

A change presented as a speedup needs the evidence described in
[scripts/bench/README.md](scripts/bench/README.md): alternating paired runs rather
than all-of-A-then-all-of-B, recall scored against the official ground truth, and a
comparison at matched recall rather than at matched parameters. Record the result in
[docs/apple_silicon/BENCHMARKS.md](docs/apple_silicon/BENCHMARKS.md) with the raw
output under `docs/apple_silicon/benchmarks/`. This is not ceremony — the
optimization campaign behind the current numbers had to discard several changes that
looked like wins under a single unpaired run.

### Style

- C++23, formatted with `clang-format-20` against the checked-in `.clang-format`.
- Comments should explain *why*, not *what*. A comment recording why a workaround
  exists, and what would let it be removed, is worth more than a paragraph
  restating the code.
- Match the surrounding code. This tree deliberately carries a lot of explanatory
  comment in its scripts and platform code, because the failure modes are obscure.

## What is out of scope

- **Restoring Linux or x86-64 support.** It was removed deliberately rather than
  left in place unverified. Contributing the port back upstream behind portability
  gates is the right way to serve those platforms; re-adding half-tested build
  paths here is not.
- **New database features.** Those belong upstream. See above.
