## What this changes

<!-- One paragraph: the problem, and what this does about it. Link the issue if there is one. -->

## How it was verified

<!-- Tick what you ran. A macOS CI run is about an hour, so local checks first. -->

- [ ] `make lint` (shell syntax, shellcheck, Markdown links)
- [ ] `make build-tests` and `make test` pass, with only the five documented exclusions failing
- [ ] `make slt` passes (if the change touches the engine, SQL, or the test drivers)
- [ ] `make recovery` passes (if the change touches storage, the WAL or startup)
- [ ] `make package` self-test passes (if the change touches packaging or the launcher)
- [ ] Any script under `scripts/apple_silicon/` that I changed, I ran

## If this claims a performance change

- [ ] Measured with `scripts/bench/ab_build.py` in randomized paired blocks, not a single run
- [ ] Recall scored against the official ground truth and compared at matched recall
- [ ] Result recorded in `docs/apple_silicon/BENCHMARKS.md` with raw output under `docs/apple_silicon/benchmarks/`

## Documentation

- [ ] Docs, `CHANGELOG.md` and `docs/known-issues.md` updated where the change affects them, or not applicable
