# Examples

These are upstream Infinity's API examples, carried over unchanged apart from the
notes below. They are a reference for individual API calls.

**If you want to see what Infinity is for, start with [../demo/](../demo/) instead** —
that is a single worked use case with its results explained, rather than one script
per method.

## Prerequisites

All of these need a running server. On macOS, start it through the wrapper rather
than by running the binary, because the shipped `conf/infinity_conf.toml` points at
root-owned `/var/infinity`:

```shell
scripts/apple_silicon/run_server.sh start
```

The Python SDK ships in this repository. Install it from the repository root:

```shell
uv sync --python 3.11 --all-extras
```

The distribution is named `infinity-sdk` but the import name is `infinity`. A
published wheel would be `pip install infinity-sdk==0.7.3`, but for this fork use the
checkout, since the port's changes are not on PyPI.

> **There is no embedded/local mode in this fork.** Upstream documented an
> `infinity-embedded-sdk` package and an `infinity.connect("/var/infinity")` form for
> opening a local directory. `connect()` here accepts only a `NetworkAddress`, and a
> string argument raises `INVALID_SERVER_ADDRESS`. Most scripts in this directory
> still carry that call as a **commented-out** line; it is dead, and it is the single
> most common reason someone concludes the server is down when it is running.

Once the server is up:

```shell
uv run python example/simple_example.py
```

Good starting points: `simple_example.py`, `hnsw_index.py`, `fulltext_search.py`,
`hybrid_search.py`, `filter_data.py`.

Two caveats about the rest:

- `import_data.py` and `export_data.py` name paths that the **server** resolves, not
  the client. They work from a checkout because both are on the same machine here.
- `cleanup_example2.py` and `cleanup_example3.py` hardcode `/var/infinity/data`, which
  does not exist when the server runs from a checkout. They need their path edited
  to the instance's data directory under `build/instances/<name>/`.

## HTTP examples

Shell scripts in `http/` drive the same operations over the REST API on port 23820.
They need `curl`, which macOS ships.

```shell
bash example/http/create_list_show_database.sh
```

Note that a SELECT is a `GET` with a JSON request body, which some HTTP clients
refuse; `curl --request GET --data '...'` handles it.
