### Summary

<!-- What changes, and why. If it fixes an issue, link it. -->

### Verification

<!-- Delete what does not apply. Say what you actually ran, not what should pass. -->

- [ ] `scripts/apple_silicon/build_server.sh macos-arm64-release infinity test_main`
- [ ] `./build/macos-arm64-release/src/test_main`
- [ ] `uv run python scripts/apple_silicon/run_slt.py` (against a `--fresh` server)
- [ ] New test covering the change — for a crash fix, the call that used to crash

Machine: <!-- e.g. M3 Pro, macOS 15.3, clang 20.1.8, cmake 4.4.2 -->

### Performance claims

<!-- Only if this PR claims a speedup. Otherwise delete this section.
     scripts/bench/README.md has the contract: alternating paired runs, recall against
     the official ground truth, comparison at MATCHED RECALL rather than matched
     parameters. Record the result in docs/apple_silicon/BENCHMARKS.md with raw output
     under docs/apple_silicon/benchmarks/. -->

### Notes for the reviewer

<!-- Anything non-obvious: a workaround and what would let it be removed, a platform
     assumption, a deliberate omission. This is the part that is expensive to
     reconstruct later. -->
