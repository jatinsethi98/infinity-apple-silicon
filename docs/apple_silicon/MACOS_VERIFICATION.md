# What is verified on macOS arm64, and what is not

Measured on an Apple M-series Mac (macOS 15, Homebrew LLVM 20.1.8, CMake 4.4.2) on
2026-09-03, from a clean configure of `main`. Every number below has the command that
produced it. Where something is unverified it says so; where it is broken it says that
too.

Before this pass the port had no server binary at all: `build/macos-arm64-release/`
held only a `CMakeCache.txt`, because the previous configure had died resolving vcpkg
dependencies against a **shallow** vcpkg clone, where the pinned baseline commit is a
graft without its tree so `git show <baseline>:versions/baseline.json` fails. The error
names `spdlog`/`thrift`/`zlib`, which reads like a dependency problem rather than a
clone-depth one. `scripts/apple_silicon/build_server.sh` now repairs that case itself
rather than documenting it as a gotcha.

## Summary

| Area | State | Evidence |
|---|---|---|
| Native arm64 build of server + unit tests | Working | `build_server.sh macos-arm64-release infinity test_main`, 3448 targets, exit 0 |
| Unit tests | **1126 / 1131 pass** | `test_main`; 5 known failures below |
| SQL logic tests | **205 / 205 pass** | `run_slt.py`; covers full-text, update/delete/drop, import, export, compact, optimize |
| Full-text search, incl. CJK dictionaries | Working | `dql/fulltext` 8/8 plus a new dictionary-backed test |
| Update / delete / drop | Working | `dml/delete` 4/4, `dml/update`, `ddl/drop`, `dml/compact` 7/7, `dml/cleanup` |
| Bulk import | Working | `dml/import` 27/27, `dml/export` 5/5 |
| Crash recovery (process death) | Working | `verify_crash_recovery.sh --rows 2000`; `EVALUATION.md` extends this to power-loss-safe fsync, and to corrupt-WAL handling, which is **broken** |
| Packaging | Working, self-tested | `make_package.sh` → 128 MB relocatable tarball |
| macOS CI | **Green** on a hosted `macos-15` runner, 2026-09-18 (run 35317365990, 50 min): build, unit tests, SQL suite, crash recovery, package self-test | `.github/workflows/macos_arm64.yml` |
| Linux x86-64 / ARM64 | **Removed** | `CMakeLists.txt` refuses a non-Darwin host before `project()`; use [upstream](https://github.com/infiniflow/infinity) |
| HTTP API, cluster mode | **Not tested** | see "Not covered" |

## Reproducing it

```sh
git submodule update --init --recursive resource      # analyzer dictionaries
scripts/apple_silicon/build_server.sh macos-arm64-release infinity test_main
scripts/apple_silicon/install_test_tools.sh           # sqllogictest, checksum-pinned
./build/macos-arm64-release/src/test_main
scripts/apple_silicon/run_server.sh start --fresh     # --fresh: the suite is not
uv run python scripts/apple_silicon/run_slt.py        #   idempotent against old data
scripts/apple_silicon/run_server.sh stop
scripts/apple_silicon/verify_crash_recovery.sh --rows 2000
scripts/apple_silicon/make_package.sh
```

## The defect that mattered

`src/parser/type/data_type.cpp`, in `StringToValue<FloatT>` and `<DoubleT>`, had an
`#if defined(__APPLE__)` branch no other platform took:

```cpp
auto ret = std::sscanf(str.data(), "%a", &value);
ParserAssert((size_t)ret == str.size(), "Error: parse float error");
```

`sscanf` returns the number of **fields assigned** — 1 on success — not characters
consumed. Comparing that to the input length holds only for single-character input.
Measured with a standalone probe on the project's own toolchain: `1` passes, and
`1.5`, `3.25`, `0.0001`, `-2.5` all parse to the correct value and are then rejected
by the assertion.

That one branch accounted for **7 of the 24 original unit-test failures** and reached
two of the six areas this pass was about:

- `src/storage/column_vector/column_vector.cppm` — the string-to-column append path,
  i.e. CSV/JSONL import of any float, double or embedding column.
- `src/planner/bound_select_statement_impl.cpp:237-249` and the index-scan equivalent —
  full-text options `bm25_param_k1`, `bm25_param_b`, `bm25_param_delta*` and
  `score_threshold`. Any full-text query tuning BM25 threw.

It did not affect SQL float literals, which the parser converts on its own path — which
is why `test_simple_agg.slt` inserted `1.0, 2.0, 3.0` into a `FLOAT` column and summed
them correctly while float *import* was broken.

`test/sql/dql/fulltext/fulltext_chinese_analyzer.slt` was added to pin both halves: a
dictionary-backed analyzer actually tokenizing, and non-default BM25 parameters
changing the score. Note that unrecognised search options are silently ignored rather
than rejected, so a test written against a plausible-but-wrong option name passes while
exercising nothing — the option names have to be exact.

## Remaining unit-test failures (5 of 1131)

| Test | Cause | Class |
|---|---|---|
| `TestPGM/9.TestPGMTypeSupport` (`double`) | On arm64-apple-darwin `long double` **is** `double` (verified: `sizeof` 8, `LDBL_MANT_DIG` 53, `LDBL_MAX` 1.797693e308). Upstream PGM computes segment intersections in `long double`; with keys spanning the full `double` range the products reach ~1.15e310, which overflows to `inf` on Darwin and yields a `NaN` intercept, tripping PGM's own guard. x86-64 Linux has 80-bit x87 with headroom to ~1.19e4932. | Engine, Darwin-only |
| `LowCardinalitySecondaryIndexTest.TestAllDataType` | The test builds `std::multimap` keyed by the raw type where the API contract wants `ConvertToOrderedType`. It passes on libstdc++ only because that tree's node layout puts the value at the same offset regardless of alignment; libc++'s does not. Production always passes the ordered type. | Test |
| 3 × `RAGAnalyzerTest.*consistency_with_python`, `test_set_language_dutch` | Compare the C++ tokenizer against a Python reference invoked at test time, which needs an NLTK `punkt_tab` **network download**. The C++ side is correct (it produced the right Dutch stem, `huiz`); the Python side returned nothing. | Environment |

Note on the PGM one: the message it throws ("Change the type of `Segment::intercept`
to uint64") is misleading — the value is `NaN` from overflow, not a legitimately large
intercept, so widening the field does not fix it. The real fix is to reformulate the
arithmetic so it never forms products of that magnitude (divide before multiply, shift
keys relative to the origin). `__float128` is not available on this target, so the
usual "widen the float type" fix is not an option either. Left unfixed deliberately:
it is a numerical change to vendored third-party code and belongs in its own change
with its own validation.

## Findings reported here, then fixed

> **Both findings below were subsequently fixed on this branch, and both fixes are
> verified working.** See `EVALUATION.md` for the verification. The original text is kept
> because the reasoning about *why* each was left alone at the time is still the record of
> that decision — but do not read either as describing current behaviour.

**The WAL never fsyncs.** — *Fixed by commit `b2d5937`.* `full_fsync` is now the shipped
default and uses `F_FULLFSYNC`; the three `wal_flush` modes are distinct and live. Measured:
median single-row commit latency 3.4–3.9 ms under `full_fsync` versus ~0.17 ms under `fsync`
and `no_sync`, a ~10–12× min-latency ratio that confirms a real device flush. Group commit
amortises it (4.2× throughput at 8 concurrent clients). Acknowledged commits survive SIGKILL
and WAL replay in all three modes.

<details><summary>Original finding (historical)</summary>

`src/storage/wal/wal_manager_impl.cpp` — all three `FlushOptionType` branches call only
`ofstream::flush()`, and two carry upstream's own `// FIXME: not flush` comments. There is no
`fsync` anywhere in the WAL write path. Bytes reach the kernel page cache, so **process death
is survivable and power loss is not**, on every platform. This is why the recovery check above
is described as process death only. On Darwin the eventual fix is harder than adding `fsync`:
`fsync` does not flush the drive's write cache on macOS, so it needs `F_FULLFSYNC`.

</details>

Note that fixing the fsync did **not** make the WAL trustworthy end to end: a single corrupted
byte in it causes the engine to delete the entire log and abort startup, losing every commit
since the last checkpoint. See `EVALUATION.md` blocker 5.

**Exported files are not truncated.** — *Fixed by commit `ccc58a3d`.* `COPY TO` now refuses to
overwrite an existing file, and a refused overwrite leaves the original byte-identical.

<details><summary>Original finding (historical)</summary>

`src/storage/io/virtual_store_impl.cpp:127` opens `FileAccessMode::kWrite` with
`O_RDWR | O_CREAT` and no `O_TRUNC`, and writes begin at offset zero. Exporting a short result
over a longer existing file leaves the previous tail in place. 22 production call sites share
that mode, including buffer-manager file workers and snapshot metadata, and some may rely on
writing into an existing file, so a blanket `O_TRUNC` risks data loss. The targeted fix is a
separate `kWriteTruncate` mode used by the export path. `run_slt.py` clears its staging `tmp/`
before every run so the export suites are at least not affected by the previous run's
leftovers.

</details>

CSV export has a *separate*, still-open escaping defect: varchar containing a comma, quote or
newline is written unquoted, so an export → re-import round-trip corrupts data
(`EVALUATION.md` blocker 10).

## A finding that was refuted, and a different one at the same site that was not

`src/storage/common/rcu_multimap.cppm:97` declares `read_map_` as
`InnerMultiMap *volatile` and publishes it with a bare store, which reads exactly like
the classic weak-memory-ordering bug: `volatile` gives no release/acquire ordering, so
on arm64 a reader could observe a published pointer before the pointee's contents. Five
independent reconnaissance passes flagged it, and it is worth recording why that
convergence was not evidence.

**The publication bug is not reachable.** Three adversarial verifications — briefed
respectively to refute, to construct an interleaving, and to establish reachability —
agreed 3/3. The publishing store lives only in `CheckSwapInLock`, whose only caller is
`Get()`; the sole production instance
(`src/storage/secondary_index/secondary_index_in_mem_impl.cpp:56`) never calls
`Get()`/`get()`/`GetWithRcuTime`. `read_map_` is therefore written once in the
constructor and never republished, so there is no publish/subscribe race to lose. On
any platform. Converting it to `std::atomic` with acquire/release would be reasonable
hardening but fixes nothing today.

**The site is not safe, though, and for a different reason.** — *Fixed by commit `ccc58a3d`,
which holds `dirty_lock_` across the whole traversal. Verified: ~49,000 concurrent range
queries against 120,000 concurrent inserts, ~48M rows validated, zero violations, no crash,
no hang, and the state survives SIGKILL + WAL recovery. The serialisation cost is negligible
(median ×1.0 versus quiescent), so it is not a throughput regression. See `EVALUATION.md`.*

<details><summary>Original finding (historical)</summary>

`RcuMultiMap::range` (`rcu_multimap.cppm:372`) acquires `dirty_lock_` only long enough to copy
the `dirty_map_` pointer (`:381-383`), then **releases it and iterates the container
unlocked** — `lower_bound`, `upper_bound` and the merge loop all run outside the lock, while
`Insert` mutates that same `std::multimap` under it. Concurrent access is expected here: an
indexing thread appending to the in-memory secondary index and a query thread in
`RangeQueryInner` (`secondary_index_in_mem_impl.cpp:217-220`) reach it at the same time.
Iterating a `std::multimap` while another thread inserts into it is a data race and undefined
behaviour — the red-black tree rebalances under the reader.

That is platform-independent, not an arm64 issue, though weak ordering makes the consequences
less predictable. It is also genuinely a storage-engine fix (hold the lock across the
iteration, or make this a real RCU with reader reference counting) with throughput
implications, so it is reported rather than changed here — the same call as the WAL and export
findings above. The related concern that readers take no grace period and the old map is freed
without draining them is part of the same defect.

</details>

One caveat on the verification: it ran against the release binary. Testing that the race is
*gone* rather than merely not observed would need a ThreadSanitizer build, which has not been
done.

An earlier revision of this document claimed "all live data sits in `dirty_map_` behind
`dirty_lock_`". That was wrong: the pointer is read under the lock, the container is
not.

## Not covered

Stated plainly so the summary table is not read as more than it is.

- **HTTP API.** Every server starts both a PostgreSQL and an HTTP listener. The SQL
  logic tests drive only the PostgreSQL path, so nothing here exercises HTTP
  serialization or endpoints. `python/test_pysdk` has a `--http` mode that would.
- **Cluster mode.** Untested. Two independent standalone instances *are* verified to
  run concurrently, which found a real blocker: `peer_port` defaults to 23850 bound to
  `0.0.0.0` and does not appear in `conf/infinity_conf.toml` at all, so a second
  instance dies binding a port the first holds — and it dies *after* printing the log
  lines that look like a successful start. `run_server.sh` offsets it.
- **Linux.** No Linux run was made on this branch. The changes to shared code are the
  float parse (previously Apple-only, now shared), the two test-harness fixes, and the
  unit-test resource resolver, which probes the installed path first specifically so
  Linux CI resolves as before.
- **The macOS CI workflow is now a gate.** Its first two hosted runs failed and each
  found a real portability defect: the runner image has no Homebrew bison (vcpkg's
  thrift port needs 3.7+), and `Value::StringToValue` referenced a libc++-library
  `from_chars` symbol the runner's Xcode 16.4 SDK does not export. Both are fixed, and
  run 35317365990 (2026-09-18) passed every step on a `macos-15` runner in 50 minutes.
- **Python wheel.** The configured wheel is the pure-Python remote SDK; the native
  embedded module is outside package discovery. No arm64 embedded wheel is produced,
  and the packaged tarball is the shipping artifact instead.

## Test-harness portability fixed along the way

These were bugs on every platform; macOS only made them visible.

- `tools/sqllogictest.py` accepted `-c/--copy` and then hardcoded
  `/var/infinity/test_data/tmp` anyway, so the flag was silently ineffective.
- `tools/generate_hnsw_with_delete.py` shadowed its `copy_dir` argument with a local
  constant, took the argument as a `bool` where every sibling generator takes a path,
  and embedded the absolute path into the SQL it generates.
- The analyzer and highlighter unit tests hardcoded `/usr/share/infinity/resource` and
  failed when absent. Linux CI passes only because the workflows bind-mount
  `${PWD}/resource` onto that path, so a plain source checkout fails on any platform.
- `python/infinity_sdk/infinity/rag_tokenizer.py` looked for its dictionary only beside
  itself and in `/usr/share/infinity/resource/rag`, never in the repository's own
  `resource/rag`. Its compiled-trie cache was also written next to the dictionary,
  which meant into the `resource/` **git submodule**; it now goes under
  `~/.cache/infinity/`.
- `DateTypeOldTest.TestEqStdChronoForward` built its reference date with `mktime`
  (local time) and `ceil<days>`, so it failed in every timezone west of UTC. Linux CI
  runs UTC containers, which hid it.
- `DateTypeTest.TestNegativeYears` expected `-001-05-04`, the old fmt `{:04d}` output,
  but `DateT` is `DateTypeStd` and renders through `std::chrono`, which the standard
  requires to print `-0001`. It would fail on Linux too.

## The 94-file absolute path

94 of the committed `.slt` files contain the data root inside the SQL:

```
COPY t FROM '/var/infinity/test_data/embedding_float_dim4.csv' WITH (…);
```

The **server** resolves that path, so no runner flag can redirect it. Linux CI runs as
root in a container where `/var/infinity` is writable; on macOS `/var` is root-owned.

`scripts/apple_silicon/run_slt.py` materialises a path-rewritten copy of `test/sql`
under `build/slt/sql` and runs `sqllogictest` over that, rather than creating
`/var/infinity` (a privileged change outside the repo, to run a test suite) or editing
94 upstream files (a permanent diff that conflicts with every upstream change to the
suite). Generated `.slt` files need no rewriting because they embed whatever root the
generators are given.

The honest cost of that choice: Linux CI never exercises the materialiser, so the two
platforms run from different corpora. Making portable staging the shared runner for
both is the better long-term shape.
