# Contributing

Thanks for looking. This repository is the native Apple Silicon port of
[Infinity](https://github.com/infiniflow/infinity); knowing which of the two a change
belongs to saves everyone time.

**Belongs here:** anything about the port or the Mac. It does not build, start or pass
a test on an Apple Silicon Mac; the scripts, `Makefile`, packaging or CI are wrong or
unclear; the benchmarks do not reproduce or the method is unsound; the docs are
wrong; one of the [known issues](docs/known-issues.md), or a new one.

**Belongs upstream:** Infinity itself. SQL semantics, the planner, index algorithms,
the SDK's API, new database features. File those at
[infiniflow/infinity](https://github.com/infiniflow/infinity/issues/new/choose) so a
fix reaches every platform. If you are not sure, open a discussion here and ask.

## Reporting a bug

Use the [bug template](https://github.com/jatinsethi98/infinity-apple-silicon/issues/new/choose).
It asks for the output of `make doctor`, which captures your macOS, chip and toolchain
in one paste, the commit or release you are on, and the tail of the relevant log:
`build/<preset>-build.log` for a build failure, `build/instances/<name>/log/infinity.log`
for a runtime one.

The most common report is not a bug: `infinity.connect()` takes a `NetworkAddress`,
not a string, and a string raises `INVALID_SERVER_ADDRESS`, which reads like the
server is down.

## Making a change

```sh
git clone --recurse-submodules https://github.com/<you>/infinity-apple-silicon.git
cd infinity-apple-silicon
make doctor && make setup            # once; about an hour
git checkout -b my-change
# ... work ...
make lint                            # shell syntax, shellcheck, Markdown links
make build-tests && make test        # C++ unit tests
make slt                             # SQL logic suite, if you touched the engine or SQL
git push origin my-change
```

Then open a pull request; the template lists what to tick. CI runs the lint job in
minutes and the full native build, unit tests, SQL suite, crash-recovery check and
packaging self-test on a `macos-15` arm64 runner in about an hour.

### What makes a pull request easy to merge

- **One concern per PR.** A fix and a refactor are two PRs.
- **A test for a fix.** If the bug was a crash, the test is the call that crashed.
  `test/eval_macos/test_server_crash_regressions.py` already has one failing test per
  known blocker; fixing a blocker means making its test pass.
- **Run any script you touched.** Several of them are the only thing between a
  reader and a cryptic toolchain error, so a regression there is worse than it looks.
- **Update the docs that the change affects.** `docs/known-issues.md` when a defect
  is fixed, `CHANGELOG.md` for anything user-visible, the relevant guide when
  behaviour changes. `make lint` checks the links.

### Performance claims

A change presented as a speedup needs the evidence described in
[docs/guides/benchmarking.md](docs/guides/benchmarking.md): interleaved paired runs
with `scripts/bench/ab_build.py` rather than all-of-A-then-all-of-B, recall scored
against the official ground truth, and a comparison at matched recall rather than at
matched parameters. Record the result in
[docs/apple_silicon/BENCHMARKS.md](docs/apple_silicon/BENCHMARKS.md) with the raw
output under `docs/apple_silicon/benchmarks/`. This is not ceremony: the optimization
campaign behind the current numbers had to discard several changes that looked like
wins under a single unpaired run.

### Style

- C++23, formatted with `clang-format-20` against the checked-in `.clang-format`.
- Python through `uv`; scripts must pass `shellcheck -S warning` and run under the
  bash 3.2 that macOS ships.
- Comments explain *why*, not *what*. A note on why a workaround exists and what
  would let it go is worth more than a paragraph restating the code.

## Good first contributions

- A real 768-dimension embedding corpus for the benchmarks, with ground truth.
- The two engine-side unit-test failures listed in the [roadmap](docs/apple_silicon/ROADMAP.md).
- Any known issue with a fix direction already written down.
- Anything in the docs that confused you: a PR that fixes the sentence is welcome.

## Out of scope

- **Restoring Linux or x86-64 support.** Removed deliberately; the way to serve those
  platforms is to contribute the port upstream behind portability gates.
- **New database features.** Upstream.

By contributing you agree that your contribution is licensed under the repository's
Apache-2.0 license.
