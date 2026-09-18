# Python

What is in this directory and how it relates to the server you build from this
repository.

| Path | What it is |
|---|---|
| `infinity_sdk/` | The `infinity` package: the client SDK, published on PyPI as `infinity-sdk`. |
| `test_pysdk/` | Upstream's SDK test corpus. Much of it hardcodes `/var/infinity/test_data`, which is not creatable without root on macOS; the suites that pass natively are in `../test/eval_macos/`. |
| `parallel_test/`, `restart_test/`, `test_cluster/` | Upstream's concurrency, restart and cluster drivers. `parallel_test` passes on macOS; `test_cluster` is untested here. |
| `benchmark/` | Upstream's comparisons against Elasticsearch and Qdrant on Linux. Not maintained in this fork; the Apple Silicon benchmarks live in `../scripts/bench/`. |
| `infinity_embedded/` | Upstream's embedded (in-process) client. **Not functional in this fork**: the native extension it needs is not built, and `infinity.connect()` accepts only a network address. |

## Using the SDK

```sh
pip install infinity-sdk            # the published wheel works against this server
# or, from a checkout, into the repo's virtualenv:
uv sync --python 3.11 --all-extras  # make sdk
```

```python
import infinity
from infinity.common import ConflictType, NetworkAddress

conn = infinity.connect(NetworkAddress("127.0.0.1", 23817))
db = conn.get_database("default_db")
```

The [Python guide](../docs/guides/python.md) walks through tables, indexes and search;
the [SDK reference](../docs/references/pysdk_api_reference.md) lists every call.

## Differences from upstream's copy

One: `infinity/rag_tokenizer.py` finds its dictionary (`huqie.txt`) in the
repository's `resource/rag/` when running from a checkout, and caches the compiled
trie under `~/.cache/infinity` instead of writing next to the dictionary inside the
submodule. Everything else is upstream 0.7.3.

## Publishing

The SDK's version is in the root `pyproject.toml` and the `client_version` handshake
in `infinity/remote_thrift/client.py`; both must move together. This fork does not
publish to PyPI; the upstream wheel is protocol-compatible with the server here.
