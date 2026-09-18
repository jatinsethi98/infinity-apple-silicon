# Known issues

What is broken today, what triggers it, and how to stay clear of it. Every entry was
reproduced by hand and independently re-verified; the full write-up with stack traces,
root causes and fix directions is [EVALUATION.md](apple_silicon/EVALUATION.md). If you
hit something that is not here, [open an issue](https://github.com/jatinsethi98/infinity-apple-silicon/issues/new/choose).

The one-paragraph summary: the engine's functional surface is broad and its
concurrency is solid (225 of 239 end-to-end tests pass, 205 of 205 SQL logic tests,
zero regressions in the C++ unit suite). What is not finished is input validation.
Some bad inputs crash the server; worse, some are accepted and produce a wrong answer
with `error_code: 0`. Until these are fixed, **do not put data you cannot regenerate
into it**, and validate on the client side what the server does not.

## Blockers

| # | Trigger | What happens | Avoid it by |
|---|---|---|---|
| 1 | A filter that folds to always-true or always-false on a column with a **Secondary index**: `v > 127` or `v <= 127` on an `int8`, and the same at the bounds of `int16`, `int32`, `int64`, with `>`, `>=`, `<`, `<=`, `=` or inside `AND` | server process dies, nothing logged; in one path returns wrong rows instead | not filtering at a type's exact boundary on indexed integer columns, or using a wider type |
| 2 | `match_sparse(..., topn=-1)` or `topn=2**63-1` | server dies | validating `topn` client-side (`match_dense` already rejects these) |
| 3 | `conn.flush_catalog()` | segfault, every time | never calling it; `flush_data()` is safe |
| 4 | A snapshot name containing `/` or `..` | writes files outside the snapshot directory with `error_code: 0` | only ever passing a plain name; treating query access as filesystem access until fixed |
| 5 | One corrupt byte in the WAL | the entire WAL is deleted on startup and the server aborts; committed rows are gone | taking snapshots; keeping copies of anything you cannot regenerate |
| 6 | `sum()`, `min()`, `max()` after roughly 8,500 or more physical row versions on a table that has been `UPDATE`d repeatedly | wrong numbers, sometimes wildly wrong under concurrency; `count(*)`, scans and `GROUP BY` are correct | calling `table.compact()` after heavy update churn; not trusting ungrouped aggregates on churned tables |
| 7 | Restoring a table or database snapshot that had a full-text index | the full-text index reports present but returns nothing; dense and secondary indexes restore correctly | dropping and recreating full-text indexes after every restore |
| 8 | Inserting a wrong-typed value: `"abc"` or `3.7` into an integer column | stored as NULL or rounded, call reports success | validating types before insert |
| 9 | `min()`, `max()`, `sum()` over an all-NULL integer column | a sentinel integer instead of NULL | checking `count(col)` first |
| 10 | CSV export of varchar values containing commas, quotes or newlines | unescaped output; re-import corrupts or loses data | exporting as JSONL or Parquet |
| 11 | A stored vector containing NaN | it is returned as the top-1 L2 result ahead of the true neighbour; the SDK shows it as `None` | checking `numpy.isfinite` on every vector before insert |

Blockers 1, 2 and 3 are pattern A: a crash with nothing in the log. Blockers 4 and 6
through 11 are pattern B: a wrong answer reported as success, which nothing downstream
can detect. Pattern B is the one to design around.

## Majors

- A full-text index created **while `match_text` queries run concurrently** is silently
  unusable ("index doesn't exist") until the server restarts. Create indexes before
  opening the table to query traffic.
- Around twenty nested arithmetic operators in one expression make the server drop the
  connection with `TOO_MANY_CONNECTIONS`.
- `DATE` accepts year 0 and negative years, and then crashes the SDK decoder on read.
- `has_header` is ignored on CSV import and export.

## Test-suite exclusions

Five of 1,131 C++ unit tests fail on macOS arm64, each with a root cause on record in
[MACOS_VERIFICATION.md](apple_silicon/MACOS_VERIFICATION.md): three RAG-analyzer tests
that need a network download of an NLTK tokenizer, a PGM-index test that overflows
because `long double` is 64-bit on arm64, and a low-cardinality index test that passes
the wrong key type. CI excludes exactly those five.

## Not covered by any test here

- **Cluster mode** (leader and follower servers) is untested on macOS.
- **The HTTP API** is covered by one end-to-end suite (38 of 40 tests pass); the SQL
  logic suite drives only the PostgreSQL path.
- **Linux** is not built or tested from this repository at all; it was removed.

## Fixed on this fork

Defects found during the port and already fixed here, so you do not go looking for
them: the WAL never called `fsync` (upstream, all platforms; now `F_FULLFSYNC` by
default), exports opened files without `O_TRUNC` and left the old tail, a secondary
index range scan iterated a map without its lock, float parsing on macOS rejected
every literal longer than one character, and two date tests encoded timezone
assumptions. Each is in [CHANGELOG.md](../CHANGELOG.md) with its commit.

## Status of fixes

None of the eleven blockers has a fix merged yet. Fix directions for each are in
[EVALUATION.md](apple_silicon/EVALUATION.md#blockers), and
[`test/eval_macos/test_server_crash_regressions.py`](../test/eval_macos/test_server_crash_regressions.py)
has one failing test per defect, so a fix comes with its regression test already
written. They are the top of the [roadmap](apple_silicon/ROADMAP.md).
