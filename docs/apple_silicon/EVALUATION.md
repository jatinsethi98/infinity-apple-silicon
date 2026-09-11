# End-to-end evaluation of the macOS arm64 port

Measured on 2026-09-03 against branch `apple-silicon/macos-functional-verification` at
commit `ccc58a3d`, using the already-built `build/macos-arm64-release/src/infinity`
(macOS 15, 12 cores, 36 GB RAM, Homebrew LLVM 20). Every finding has a command that
reproduces it.

This document covers **functional, negative, concurrency, recovery, durability and load
testing**. It complements `MACOS_VERIFICATION.md`, which covers whether the port builds and
runs at all. Where the two disagree, this document is newer — see
[Corrections to MACOS_VERIFICATION.md](#corrections-to-macos_verificationmd).

The suites live in `test/eval_macos/` and share `test/eval_macos/harness.py`, which gives
each suite its own server on its own ports with its own data directory. That isolation is
not a nicety: several suites deliberately kill the server, and without it one crash cascades
into every later test as a connection error, which reads like a broken harness rather than a
crashed engine. That misreading is how two of the blockers below were nearly missed.

## Bottom line

The port is functionally broad and its concurrency story is genuinely good. 225 of 239 new
functional tests pass. The committed C++ unit suite shows **no regression** from the two
storage commits on this branch. The entire committed `python/parallel_test` corpus passes on
macOS. Both storage fixes on this branch — the WAL fsync (`b2d5937`) and the `rcu_multimap`
range lock (`ccc58a3d`) — were independently verified to work under heavy stress.

But the engine is not currently safe to trust with data. The evaluation confirmed **eleven
blockers**, and they are not eleven unrelated bugs. They are two patterns:

**Pattern A — missing input validation arrives as a crash.** Four separate ways for an
ordinary client to kill the whole server process with one call, no special privileges
needed. One is a boundary condition a person writes by accident (`WHERE v <= 127` on an
indexed `int8` column). One is a documented SDK method (`flush_catalog()`).

**Pattern B — errors are swallowed instead of raised.** Wrong-typed values become NULL,
aggregates read uninitialised memory, a NaN in a vector outranks the true nearest neighbour,
and a restored snapshot's full-text index returns nothing — every one of them with
`error_code: 0` and nothing in the server log.

Pattern B is the more dangerous for a RAG store. Each instance returns a plausible answer
that is wrong, so nothing downstream can detect it. Pattern A at least fails loudly.

## What was run

| Suite | Tests | Result |
|---|---|---|
| `test_ddl_e2e.py` | 34 | 34 pass |
| `test_ingest_export_e2e.py` | 27 | 26 pass |
| `test_dense_vector_e2e.py` | 30 | 29 pass |
| `test_sparse_vector_e2e.py` | 32 | a crash cascaded the module — see blocker 2 |
| `test_fulltext_e2e.py` | 31 | 31 pass |
| `test_hybrid_fusion_e2e.py` | 27 | 27 pass |
| `test_secondary_filter_e2e.py` | 23 | 22 pass |
| `test_type_system_e2e.py` | 37 | 31 pass |
| `test_query_language_e2e.py` | 37 | 33 pass |
| `test_snapshot_e2e.py` | 16 | 14 pass |
| `test_mutate_e2e.py` | 31 | 28 pass |
| `test_http_api_e2e.py` | 40 | 38 pass |
| `test_concurrency_dml_e2e.py` | 17 | 17 pass |
| `test_concurrency_index_e2e.py` | 15 | 14 pass, 1 xfail |
| `test_crash_recovery_e2e.py` | 25 | 22 pass, 3 xfail (confirmed defects) |
| `test_wal_durability_e2e.py` | 8 | 8 pass |
| `run_parallel_suite.py` (committed corpus) | 18 | 18 pass, 5m54s, zero server errors |
| `test_server_crash_regressions.py` | 27 | 10 pass / 17 fail **by design** — each failure pins a confirmed defect |
| Committed C++ unit tests (`test_main`) | 1131 | 1126 pass, 5 fail — **exactly the 5 already documented**, no regression |

Findings were adversarially verified: each significant one was re-checked by independent
agents briefed to *refute* it — one attempting reproduction, one attacking the test rather
than the engine, and for source-level claims one reading the implementation. A finding
survived only if no lens refuted it. **Eight claims were refuted this way and are excluded**,
including a plausible-sounding "insert throughput does not scale past 2 threads" that turned
out to be a client-side measurement artifact. Every blocker below was additionally reproduced
by hand.

## Blockers

### 1. Any predicate that folds to always-false/always-true crashes the server

Only when the column carries a **Secondary index**. On an `int8` column (range −128..127):

```python
t.create_index("idx_v", index.IndexInfo("v", index.IndexType.Secondary))
t.output(["id"]).filter("v > 127").to_pl()      # server process dies, SIGSEGV
```

Confirmed for `int8`/`int16`/`int32`/`int64`, for `>`, `>=`, `<`, `<=` and `=`, and through
an `AND` compound. **Not** reproducible for `!=`, for `OR`, for a float column, or without
the index. `v > 126` is fine.

These are ordinary boundary conditions, not malformed input. `WHERE v <= 127` on an indexed
`int8` column is SQL a person writes by accident, and it takes the server down for every
connected client.

Under lldb: `EXC_BAD_ACCESS` at `ldr x8, [x8, #0x8]` / `blr x8` — a call through a garbage
vtable slot — inside the inlined `InfinityThriftService::Select`. **Nothing is written to
`infinity.log`**; the process dies before reaching any error path.

Root cause: `FilterExpressionPushDownHelper::UnwindCast`
(`src/planner/optimizer/index_scan/filter_expression_push_down_helper_impl.cpp`, the early
return at :357 and the three `IntegralContinueUnwind` false branches at :395/:403/:411)
signals the folded case with a sentinel tuple:

```cpp
return {0, Value::MakeNull(), nullptr, compare_type};   // kAlwaysFalse / kAlwaysTrue
```

— column id `0`, and a **null expression pointer**. The contract is that callers check
`compare_type` before touching the rest, which that function's own early return does. Two
consumers in `filter_expression_push_down_indexscanfilter_impl.cpp` do not:

- `SolveForColVal` (:567-576) calls `new_candidate_column_index_map_.at(column_id)` with the
  sentinel `0` — either a throw, or silently the **wrong column's index**.
- `SolveForFuncVal` (:578-590) dereferences the null pointer outright:
  `static_cast<FunctionExpression *>(base_expression.get())->ExtractFunctionInfo()`.

The switch at `:824-829` *does* handle these compare types correctly, returning
`IndexFilterEvaluatorAllTrue`/`AllFalse`; both lambdas run earlier and never reach it.

**Fix direction:** return early from both lambdas on `kAlwaysFalse`/`kAlwaysTrue` before
using `column_id` or the expression pointer, routing to the existing evaluators at :824-829.
Keep the two cases **distinct** — collapsing always-true onto always-false would convert
this crash into silently wrong results. `test_server_crash_regressions.py` asserts row
counts, not just survival, specifically to catch that.

Reached independently from the snapshot suite by a different route, where the always-true
path returned **wrong results rather than crashing** — so this can corrupt an answer
silently, not only kill the process.

### 2. `match_sparse` does not validate `topn`

```python
t.output(["id"]).match_sparse("v", SparseVector([0,1],[1.0,1.0]), "ip", -1)   # server dies
```

Crashes for `topn = -1` and `topn = 2**63-1`. `match_dense` rejects the same values cleanly
with an `InfinityException` and stays alive, so the validation exists in the engine — the
sparse path does not reach it. `topn = 0` is rejected on both paths.

### 3. `conn.flush_catalog()` segfaults the server, 100% of the time

Reproduced on a fresh empty database and on one holding 100 rows; 3/3 fresh servers in an
independent run. The server logs a checkpoint transaction and then dies:

```
[critical] Txn ID: 2, Text: checkpoint, ... State: Committed
[critical] Error: Segmentation fault: 11
```

`flush_data()` is safe (5/5). This one compounds blocker 5: that defect loses every commit
since the last checkpoint and the shipped `checkpoint_interval` is 86400s, so the obvious
mitigation is to take a checkpoint before risky work — and this is the API for doing that.

### 4. Snapshot names are concatenated into a filesystem path unsanitised

```python
db.create_table_snapshot("../../escape_two", "t")     # error_code = 0
```

Verified by hand; traversal is not limited to one level:

```
snapshots/                 clean_snap          <- correct location
<instance data root>/      escape_one[.json]   <- escaped one level, sibling to wal/ and catalog/
build/instances/           escape_two          <- escaped two levels
```

Every call returned `error_code=0`. This is an arbitrary-file-write primitive for anyone with
query access, bounded only by the server process's own permissions, and it can write next to
or over `catalog/`, `wal/` and `persistence/`. **Fix:** reject any snapshot name that is not
a single path component.

### 5. One corrupt WAL byte deletes the whole WAL and aborts startup

Reproduced by hand with `wal_flush = full_fsync`, so all 200 rows were durably synced first.
Flipping one bit deep inside the first WAL record — past the 24-byte header, leaving both
size fields intact so only the CRC differs — produces:

```
[warning]  Found bad wal entry .../wal/wal.log@0          <- detected correctly
[warning]  Remove wal log .../wal/wal.log                 <- deletes the ENTIRE log
[info]     Find and set checkpoint max commit ts: 0
[critical] WAL replay: No checkpoint found in wal@src/storage/wal/wal_manager_impl.cpp:803
[critical] TerminateHandler: Unhandled Exception: ...
```

Process dies with `Abort trap: 6`; `wal/` is left **empty**; the 200 committed rows are gone
and the database will not open again.

Detection works. The handling is the defect: the engine removes the whole WAL file rather
than truncating at the first bad entry, then `UnrecoverableError` throws an
`UnrecoverableException` that nothing on the startup path catches, so `TerminateHandler`
aborts.

Three things compound it:

- The shipped `checkpoint_interval` is 86400s, so a fresh or lightly-used database has **no
  checkpoint at all** and the loss window is every commit ever made.
- The abort happens **after** the server binds its PostgreSQL, HTTP, thrift and peer
  listeners, so a port-probing health check reports success moments before the process dies.
  `MACOS_VERIFICATION.md` records the same shape for the `peer_port` collision; it is a
  recurring startup pattern.
- It bounds what `b2d5937` bought. The WAL is now genuinely fsynced, but one corrupt byte
  discards the durably-written data anyway.

**Fix direction:** truncate at the first bad entry and replay what precedes it (the usual WAL
contract), or refuse to start with a diagnosable error — never delete the log. And catch
`UnrecoverableException` on the startup path so a recovery failure exits with a message
rather than `std::terminate`.

*Attribution note:* an agent predicted a null dereference here, from `WalEntry::ReadAdv`
building its checksum-mismatch diagnostic with `entry->cmds_[0]` while `cmds_` is still empty
(`src/storage/wal/wal_entry_impl.cpp:2213-2262`, populated at :2252). That reading looks
correct and is worth fixing on its own, but it is **not** the failure observed here — the
stack goes through `WalManager::GetReplayEntries` and the process dies on an unhandled
exception, not a segfault.

### 6. `sum()`/`max()` read outside the live-row set after routine UPDATE churn

Reproduced by hand with a sharp, repeatable threshold. 500 rows, `UPDATE ... SET a = 100`
applied N times, compared against pulling every value to the client and aggregating in
Python:

```
 updates   phys~  count | scan: n/sum/min/max          | server: sum/min/max
      15    8000    500 | n=500 sum=50000 min/max=100  | sum=50000 min/max=100   OK
      16    8500    500 | n=500 sum=50000 min/max=100  | sum=50000 min/max=100   OK
      17    9000    500 | n=500 sum=50000 min/max=100  | sum=50256               *** WRONG ***
      20   10500    500 | n=500 sum=50000 min/max=100  | sum=50256               *** WRONG ***
```

`count(*)` is correct throughout, the scan returns exactly 500 values all equal to 100
(`distinct = [100]`), and the server log is clean. The break is at ~8500→9000 *physical*
row-versions — once live rows plus the dead versions left by UPDATE span more than one
8192-row block.

Independent agents saw `max(a) = 256` and, over the **HTTP** transport on the same table,
`max(a) = 50176` — values no live row ever held, differing between runs. **Under concurrency
it amplifies catastrophically: `sum` returned 6,263,463,996 against a true value of 29,500.**
A pure-insert control spanning two blocks aggregates correctly, and `GROUP BY a` returns the
right answer on the same connection. So this is the ungrouped-aggregate path reading dead or
uninitialised slots, not a client or serialization artifact.

Two consequences. `SELECT sum(x)` silently returns a wrong number on any table that has been
updated enough — which is every table, eventually. And because the values are uninitialised
memory rather than stale row data, an aggregate is a channel through which adjacent process
memory can reach a client.

### 7. Full-text search silently returns nothing after a snapshot restore

One table carrying HNSW + FullText + Secondary indexes; snapshot → drop → restore. Verified
by hand:

```
                       fulltext('apple')      dense(top3)   secondary(num>=30)
BEFORE snapshot        [0, 1, 2, 3, 4]        [0, 1, 2]     [3, 4, 5, 6, 7, 8, 9]
AFTER drop+restore     []                     [0, 1, 2]     [3, 4, 5, 6, 7, 8, 9]
```

Row count is 10, all three indexes are reported present, `show_index` reports
`index_type='FULLTEXT'` with `segment_index_count='1'`, and the server error log is empty.
`optimize()` does not repair it; it reproduces at database-snapshot scope and across a
restart.

A restored table looks entirely healthy while half of hybrid retrieval silently returns
nothing. Dense and secondary restoring correctly is what makes this specific rather than
"restore is broken".

### 8. Wrong-typed and out-of-range values are silently stored as NULL

Verified by hand on **both** transports:

| Inserted into an `INTEGER` column | Result |
|---|---|
| `"abc"` | **accepted** — thrift raises nothing, HTTP returns `200 {"error_code": 0}` — stored as **NULL** |
| `3.7` | **accepted**, silently stored as `4` |
| `True` | correctly rejected: `Can't cast from Boolean to Integer@src/function/cast/bool_cast.cppm:67` |

The cast layer validates some conversions and silently substitutes NULL for others. The
caller is told the write succeeded either way, so a loading pipeline cannot detect that it
just wrote a column of NULLs. An out-of-range `int8` is inconsistent on top of that: NULL via
`INSERT`, `0` via CSV import.

### 9. Aggregates over an all-NULL integer column return sentinels, not NULL

`min()`, `max()` and `sum()` over a column whose values are all NULL return sentinel integers
instead of NULL — silently wrong inside an ordinary `GROUP BY`, where a group with no data
reports a number indistinguishable from a real one.

### 10. CSV export does not quote or escape varchar content

A varchar containing a comma, quote or newline is written unescaped, so export → re-import
corrupts or loses data. Distinct from the `O_TRUNC` issue fixed in `ccc58a3d`: that fix
holds — `COPY TO` now refuses to overwrite an existing file, and a refused overwrite leaves
the original byte-identical.

### 11. A stored NaN silently outranks the true nearest neighbour

In L2 KNN a stored vector containing NaN is returned as top-1 ahead of the genuine nearest
neighbour, so the top result is garbage. The thrift SDK renders a stored NaN as Python `None`,
making NaN indistinguishable from NULL at the client — so an ingestion pipeline that lets one
NaN through poisons every query near it, invisibly.

## Majors

- **A FullText index built while `match_text` queries run concurrently is silently left
  unusable.** `create_index` returns OK and `list_indexes` shows it, but `match_text` errors
  "index doesn't exist". A restart repairs it. Confirmed 3/3.
- **`~20 nested arithmetic operators` make the server drop the connection**, reported to the
  client as `TOO_MANY_CONNECTIONS`.
- **`DATE` accepts year 0 and negative years** and then produces a value that crashes the SDK
  decoder on read.
- **CSV `has_header` is ignored** on both import and export.

## Performance and durability, measured

The WAL fsync fix (`b2d5937`) was verified claim-by-claim against the running binary, and it
holds: the `wal_flush` setting is live, the three modes are distinct, and `full_fsync`
performs a real device flush.

| Measurement | `full_fsync` (shipped default) | `fsync` | `no_sync` |
|---|---|---|---|
| Single-row commit latency, median | 3.4–3.9 ms | ~0.16–0.19 ms | ~0.18–0.19 ms |
| Single-row commit latency, p99 | ~8 ms | — | — |
| Single-row commit throughput | ~288/s idle, 168/s contended | ~5300–5770/s | ~5300–5770/s |

The ~10–12× min-latency ratio between `full_fsync` and the other two is the discriminator
that proves `F_FULLFSYNC` is doing a real device flush rather than a no-op.

- **Group commit works**: `full_fsync` goes from 290 commits/s at one client to 1222/s at 8
  concurrent clients (4.2×), so the per-commit device flush is amortised across concurrent
  transactions.
- **Batching dominates write throughput**: batch-1 ~288 rows/s → batch-1000 ~67,000 rows/s
  (~234×) while commits/s falls only 288→67. Anyone loading data should batch.
- **Recovery is fast and idempotent**: restart+replay 0.69 s for a 4.5 KB WAL and 0.66 s for
  a 1.24 MB / 40,000-row WAL. Four SIGKILLs *during* startup replay still recovered
  40,000/40,000 rows.
- **The 8192-row block boundary matters** — it is the trigger for blocker 6, and worth
  knowing as an operational threshold.

A caveat on scaling numbers: a reported "insert throughput does not scale past 2 threads"
finding was **refuted** on verification as a client-side artifact (Python GIL, synchronous
request/response). Treat single-client SDK throughput as a client measurement unless the load
generator is proven not to be the bottleneck.

## What is verified working

Worth stating plainly, because a list of defects reads worse than the port is.

- **Both storage fixes on this branch work.**
  - `rcu_multimap` range lock (`ccc58a3d`): ~49,000 concurrent range queries against 120,000
    concurrent inserts, ~48M rows validated, **zero violations**, no crash, no hang, no
    server error — and the state survived SIGKILL + WAL recovery. Serialisation cost is
    negligible (median ×1.0 versus quiescent), so the fix is not a throughput regression.
  - WAL fsync (`b2d5937`): acknowledged commits survive SIGKILL and pure WAL replay in all
    three modes, with exact row sets — no lost, duplicated or torn rows.
- **Concurrent DML is correct.** Concurrent inserts lose and duplicate nothing (8 threads ×
  4000 disjoint ids → exactly 32,000 rows). Same-row varchar and embedding updates are never
  torn. Disjoint updates never lose an update. A reader racing an all-row UPDATE saw 0
  half-applied rows and 0 cross-row version mixes across 586 scans — so **a single DML
  statement is atomic and snapshot-isolated to concurrent readers**, and a multi-row insert is
  all-or-nothing (a reader observed only 0 or 8192 rows of an 8192-row batch). This isolation
  behaviour is undocumented and was determined empirically.
- **The committed `python/parallel_test` corpus passes on macOS**: 8 modules, 18 tests, 5m54s,
  zero server-side errors, zero crashes, zero hangs. Its only blocker is portability — the
  thrift port is hardcoded to 23817 in `common/common_values.py` with no override.
- **Crash recovery's positive path is robust.** Acknowledged commits survive SIGKILL and are
  correct through every access path (scan, filter, KNN, full-text, secondary). Mid-flight
  crashes leave a valid contiguous prefix, never a torn row. Torn WAL tails and appended
  garbage are tolerated. Truncated data files fail loud on read.
- **DDL**: 34/34. Correct default-backfill of existing rows on `add_columns`, data preserved
  across `rename` and `drop_columns`, refusal to drop a column backing an index, clean
  rejection of every malformed identifier.
- **Full-text search**: 31/31, including dictionary-backed CJK analyzers, with BM25
  parameters verified against hand-computed scores rather than merely accepted.
- **Hybrid fusion**: 27/27, with RRF and weighted-sum rankings checked against rankings
  computed by hand from the individual arms — so fusion is not silently dropping an arm or
  ignoring its weights.
- **No regression in the committed unit suite**: 1126/1131, failing exactly the five already
  documented (1 Darwin `long double`, 1 libc++ test bug, 3 needing an NLTK network download).
- **Injection safety**: string predicates are treated strictly as data. `"x' OR '1'='1"`
  matches only the row whose stored value is that literal string.

## Notable non-blockers

- **Unrecognised options are silently ignored** across full-text, fusion and BMP index
  parameters — no error, no effect, no log line. `MACOS_VERIFICATION.md` warns about this for
  search options; it is broader. A mistyped tuning parameter looks like it worked, and a test
  written against a wrong option name passes while exercising nothing.
- **Parquet ingest is not implemented** (no SDK parquet type; HTTP returns 500 for a client
  error).
- **float16 / bfloat16 vector columns accept inserts but cannot be queried** via `match_dense`.
- **`ConflictType.Replace` is unimplemented** for `create_table` and `create_database`.
- **`drop_database` cascade-deletes a non-empty database** with no RESTRICT guard.
- **A stale table handle silently rebinds** to a recreated same-name table — handles carry no
  object identity.
- **Table and column names are case-insensitive**; identifiers are ASCII-only.
- **Restore performs no integrity validation** — a corrupted snapshot restores as "Success"
  with a `count(*)` that does not match reality.
- **HTTP `UPDATE` is sensitive to JSON key order**, rejecting a valid body with `NO_SUCH_FIELD`
  when `filter` precedes `update`. JSON objects are unordered by specification.
- **Two concurrent `create_index` calls on one table** surface a raw rocksdb "Resource busy"
  error rather than a modelled one.
- **`Restore of a nonexistent snapshot leaks an absolute server filesystem path** in the
  client error.
- **SDK: `filter('x BETWEEN a AND b')` and `filter('x = NULL')` raise `RecursionError`**
  client-side and never reach the server.
  `python/infinity_sdk/infinity/remote_thrift/utils.py:638-639` falls into a catch-all
  `else: return traverse_conditions(cons[1])`, and `cons[1]` on a sqlglot expression builds a
  `Bracket` rather than descending, so it recurses forever. The wire protocol has a
  first-class `BetweenExpr` and the server binds it, so this is purely a client gap. Use
  `x >= a AND x <= b` and `x IS NULL`.
- **int8 value 127 cannot be carried by a Secondary index** — index creation fails with
  `TApplicationException: The value 127 is reserved as a sentinel`. 127 is a legal `int8`, so
  it must either be indexable or be rejected at insert time.
- **The committed parallel tests are survival-only** — they assert nothing about final state,
  so they would pass against an engine that silently lost rows.

## A defect in this evaluation's own harness

`harness.log_errors()` originally matched `| error |`, but this build's spdlog pattern is
`[HH:MM:SS.mmm] [tid] [level] message` — square brackets. It therefore returned `[]`
unconditionally, which silently turned every suite's server-side error assertion into a
no-op. Two independent agents caught it.

It is fixed (`_LEVEL_RE` matching `[error]`/`[critical]`/`[fatal]`, plus a separate
`log_warnings()` because the engine reports genuinely destructive actions such as deleting a
corrupt WAL at *warning* level). Verified against a provoked error: 6 lines now matched where
0 were before.

**Consequence for reading this document:** any "the server log was clean" statement made by a
suite that ran before the fix is unverified, not verified. The findings themselves do not
depend on it — they rest on wrong return values, dead processes and files on disk — but the
absence-of-error claims should be re-established by rerunning the suites.

Note also that an empty `log_errors()` is not proof of health even now: the index-scan SIGSEGV
(blocker 1) writes nothing to the log at all. Pair it with `pid()`.

## Corrections to MACOS_VERIFICATION.md

Confirmed stale by multiple independent agents:

- It documents the WAL as **never fsyncing** and power loss as unsurvivable, listed under
  "findings reported but deliberately not fixed". Commit `b2d5937` fixed this, and the fix is
  verified working above.
- It describes `RcuMultiMap::range` as an **unfixed data race**. Commit `ccc58a3d` fixed it,
  and the fix is verified holding under ~49,000 concurrent range queries.

## Reproducing

```sh
# The server binary must already be built:
scripts/apple_silicon/build_server.sh macos-arm64-release infinity test_main

uv sync --python 3.11 --all-extras

# The confirmed crashes and silent-corruption defects. 17 of these fail by design.
uv run pytest test/eval_macos/test_server_crash_regressions.py -v

# Everything else. Each suite starts and stops its own server.
uv run pytest test/eval_macos -v
```

Suites are safe to run concurrently: `harness.INSTANCES` assigns each a distinct port offset
and data directory, and they avoid the default ports so they never touch a `dev` instance a
person is using. Do not run load tests concurrently with anything, including each other.

## Not covered

- **Cluster mode.** Untested, as in `MACOS_VERIFICATION.md`.
- **Linux.** Nothing here ran on Linux. The platform-independent findings — the crashes, the
  silent coercion, the snapshot traversal, the aggregate out-of-bounds read, the WAL deletion
  — should reproduce there and are worth confirming, since none of them is arm64-specific.
- **The `.slt` corpus** was not re-run in this pass; `MACOS_VERIFICATION.md` records 205/205.
- **Resource-exhaustion limits and DDL-racing-DML** were written
  (`test_resource_limits_e2e.py`, `test_concurrency_ddl_e2e.py`, both on disk) but their
  agents were lost to an infrastructure auth failure before reporting. The suites exist and
  have not been run to completion.
- **ThreadSanitizer.** All concurrency testing used the release binary. The `rcu_multimap`
  race is documented as fixed and tested sound, but observing the *absence* of UB needs a TSAN
  build.
- **`python/restart_test`** is unrunnable as committed: `infinity_runner.py` hardcodes
  `data_dir = "/var/infinity"` and issues `kill -9` against **every** process whose name
  contains "infinity", which would kill unrelated servers. Recovery was covered through the
  harness instead.
- **`python/test_pysdk`** expects `/var/infinity/test_data` (`common_values.py`:
  `TEST_TMP_DIR`), absent on macOS, so its import/export tests cannot run as written.
- **Power-loss durability.** Only process death was tested. `full_fsync` uses `F_FULLFSYNC`
  and shows a device-flush latency signature, which is strong evidence but not a pulled plug.
