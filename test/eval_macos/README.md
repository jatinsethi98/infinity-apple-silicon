# macOS arm64 end-to-end evaluation

The suites behind [`docs/apple_silicon/EVALUATION.md`](../../docs/apple_silicon/EVALUATION.md).
They are separate from the committed corpora (`test/sql/`, `python/test_pysdk/`,
`src/unit_test/`) for two reasons: the committed ones hardcode `/var/infinity`, which
is not writable without root on macOS, and these deliberately crash the server, which
a shared instance cannot survive.

## Run them

You need a built server at `build/macos-arm64-release/src/infinity`.

```sh
uv sync --python 3.11 --all-extras

# One suite.
uv run pytest test/eval_macos/test_fulltext_e2e.py -v

# All of them, each on its own server, in parallel.
uv run python test/eval_macos/run_parallel_suite.py

# Regression check against the committed corpora and the documented baseline.
uv run python test/eval_macos/run_committed_suites.py
```

Do not run two suites against one server. Each suite owns an instance name and a port
offset in `harness.py`'s `INSTANCES` table, because several of them kill the server on
purpose; without that isolation one crash cascades into every later test as a
connection error, which reads like a broken harness rather than a crashed engine.

## Layout

| | |
|---|---|
| `harness.py` | Per-suite server lifecycle, a writable data root under `build/eval/`, connection helper, teardown that runs on failure. Read its docstring first. |
| `conftest.py` | Puts this directory on `sys.path` so the suites can `import harness`. |
| `test_*_e2e.py` | The suites: DDL, DML/mutation, type system, dense/sparse vectors, full text, hybrid fusion, query language, ingest/export, secondary filters, snapshots, WAL durability, crash recovery, resource limits, three concurrency suites, and the HTTP API. |
| `test_server_crash_regressions.py` | One test per way a client can crash the server. Each should fail today; each becomes a regression test once fixed. |
| `run_parallel_suite.py` | Runs every suite concurrently, one server each. |
| `run_committed_suites.py` | Runs the *committed* corpora and asserts the result still matches `MACOS_VERIFICATION.md` — exactly the five documented unit-test failures and no more, 205/205 SLT, and every `test_pysdk` failure explained by a hardcoded Linux path rather than a wrong answer. |

One-off probes used while narrowing a finding down are gitignored (see `.gitignore`
here). They are throwaway by construction; whatever they proved is in the suite that
came out of them.

## Reading a failure

A failure here is one of two things, and the distinction matters more than the count:

- **A crash.** The server process is gone; every later assertion in that suite reports
  a connection error. Look at the first failure, not the cascade.
- **A wrong answer with `error_code: 0`.** Nothing in the log, nothing in the status.
  These are the dangerous ones for a retrieval store, because nothing downstream can
  detect them. `EVALUATION.md` calls this "Pattern B".
