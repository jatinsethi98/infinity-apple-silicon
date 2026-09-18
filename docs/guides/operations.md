# Operations

Running the server day to day: where things live, how to run more than one, what the
configuration keys do, and what "durable" means here.

## Two ways to run it

**From a source checkout**, through the wrapper:

```sh
make start                 # scripts/apple_silicon/run_server.sh start --instance dev
make stop
make status
make restart
make logs
```

**From a package** (a release tarball, or `make package`), through its launcher:

```sh
~/.infinity/current/bin/infinity                 # foreground, Ctrl-C to stop
~/.infinity/current/bin/infinity --data-dir DIR  # data somewhere else
```

Do not run `build/macos-arm64-release/src/infinity` directly from a checkout. The
shipped `conf/infinity_conf.toml` points every directory at `/var/infinity` and the
analyzer dictionaries at `/usr/share/infinity`, both root-owned on macOS, so the bare
binary fails or writes somewhere you did not intend. The wrapper and the launcher each
render a configuration with usable paths first.

## Where things live

| | Checkout (`run_server.sh`) | Package (`bin/infinity`) |
|---|---|---|
| rendered config | `build/instances/<name>/infinity_conf.toml` | `<install>/etc/infinity_conf.toml` (from `.toml.in`) |
| data, catalog, WAL, snapshots | `build/instances/<name>/{data,catalog,wal,snapshots,...}` | `<data-dir>/{data,catalog,wal,snapshots,...}`, default `~/.local/share/infinity` |
| log | `build/instances/<name>/log/infinity.log` | `<data-dir>/log/infinity.log` |
| stdout of the process | `build/instances/<name>/stdout.log` | your terminal |
| analyzer dictionaries | `resource/` in the checkout | `<install>/share/infinity/resource` |
| pid | `build/instances/<name>/infinity.pid` | none; it runs in the foreground |

Everything for an instance is under one directory, so deleting that directory is a
complete reset. `make start` with `--fresh` does exactly that first:

```sh
scripts/apple_silicon/run_server.sh start --fresh
```

## Ports

| Port | Protocol | Clients |
|---|---|---|
| 23817 | Thrift | the Python SDK |
| 23820 | HTTP | `curl`, any language, the HTTP test suite |
| 5432 | PostgreSQL wire | `psql`, SQL clients, the SQL logic tests |
| 23850 | peer | cluster replication; bound even in standalone mode |

The server binds `127.0.0.1` when started through the wrapper or the launcher. Put a
reverse proxy with authentication in front of it before exposing it to a network:
Infinity has no users, passwords or TLS of its own.

## More than one instance

Each instance needs its own name and a port offset so the four listeners do not
collide. The peer port is the one people forget: a second server started without an
offset dies right after logging what looks like a successful start.

```sh
scripts/apple_silicon/run_server.sh start --instance two --port-offset 100   # 23917, 23920, 5532, 23950
scripts/apple_silicon/run_server.sh stop  --instance two
make clean-instances                                                          # stop all, delete all
```

The evaluation suites in `test/eval_macos/` use this to run eighteen servers at once.

## Configuration

The wrapper renders `build/instances/<name>/infinity_conf.toml` from the shipped
`conf/infinity_conf.toml` on every start, so edit the shipped file to change a default
for all instances, or pass `--binary` and a hand-written `-f config.toml` for a one-off.
The keys that matter operationally:

| Key | Default | Meaning |
|---|---|---|
| `[network] server_address` | `0.0.0.0` shipped, `127.0.0.1` rendered | interface to bind |
| `[network] postgres_port`, `http_port`, `client_port` | 5432, 23820, 23817 | the three client ports |
| `[network] peer_ip`, `peer_port` | 127.0.0.1, 23850 | the replication listener; set even standalone |
| `[buffer] buffer_manager_size` | 4GB | cache for column data and index pages; raise it on a 32 GB machine |
| `[buffer] memindex_memory_quota` | 1GB | memory the in-progress (not yet dumped) indexes may use |
| `[storage] mem_index_capacity` | 65536 | rows per in-memory index segment before it is written out |
| `[storage] optimize_interval`, `cleanup_interval`, `compact_interval` | 10s, 60s, 120s | background merge, garbage collection and compaction cadence |
| `[wal] wal_flush` | `full_fsync` | durability level; see below |
| `[wal] checkpoint_interval` | 86400s | how often the catalog is checkpointed so the WAL can be truncated |
| `[log] log_level` | info | `trace`, `debug`, `info`, `warning`, `error`, `critical` |
| `[resource] resource_dir` | rendered | where the analyzer dictionaries are |

The [configuration reference](../references/configurations.mdx) lists every key.

## Durability

The write-ahead log is the engine's only durability mechanism. `wal_flush` chooses
what "committed" means, measured per sync on an Apple internal SSD:

| `wal_flush` | survives | cost per commit batch |
|---|---|---|
| `full_fsync` (default) | power loss: `fsync` plus `F_FULLFSYNC`, which makes the drive flush its own cache | about 2.3 ms |
| `fsync` | an OS crash, not a power cut | about 28 µs |
| `no_sync` | only the process dying | about 1.5 µs |

One sync happens per batch of committed transactions, so under concurrent load the cost
is amortized; a single serial client pays it per commit (338 commits/s at `full_fsync`
against 1,888 at `no_sync` on the test machine). For a benchmark or throwaway data,
`run_server.sh start --wal-flush no_sync`.

Two open defects bound what this buys. A single corrupt byte in the WAL currently
causes the whole log to be discarded on startup ([known issue 5](../known-issues.md)),
and the shipped checkpoint interval of a day means a young database has no checkpoint
to fall back on. Until that is fixed, take snapshots of anything you cannot regenerate.

## Snapshots and backups

```python
db.create_table_snapshot("articles_2026_09_18", "articles")
conn.create_database_snapshot("default_2026_09_18", "default_db")
conn.list_snapshots()
db.restore_table_snapshot("articles_2026_09_18")
```

Snapshots are written under the instance's `snapshots/` directory and are the unit of
backup: copy that directory somewhere safe. Two caveats from the evaluation, both
tracked in [known issues](../known-issues.md): only ever use a plain name (names are
used as file paths unsanitized), and rebuild the full-text index after a restore,
because a restored one returns nothing.

Do not call `conn.flush_catalog()`; it crashes the server ([known issue 3](../known-issues.md)).
`conn.flush_data()` is safe.

## Health and logs

- `make status` reports the pid and the three ports of a running instance. At start,
  the wrapper waits for a real TCP connect on the client port rather than `nc -z`,
  because BSD netcat reports success for a bound-but-not-yet-accepting socket.
- `make logs` follows the engine log. Startup, checkpoints, index dumps and errors are
  all there. Crashes caused by the defects in known issues 1 to 3 leave **nothing** in
  the log; the process is simply gone. Check `stdout.log` and `make status`.
- The HTTP `/instance/*` endpoints expose the internals: `GET /instance/memory`,
  `/instance/buffer`, `/instance/memindex`, `/instance/queries`,
  `/instance/transactions`, `/instance/logs`, and `GET /admin/node/current` for the
  node's role. `example/http/show_metrics.sh` calls each one; the
  [HTTP reference](../references/http_api_reference.mdx#administration) documents them.

## Upgrading

A package install keeps every version under `~/.infinity/versions/` and repoints
`current`; the data directory is separate and is not touched. There is no on-disk
format migration tooling, and the on-disk format follows upstream Infinity 0.7.3.
Take a snapshot before upgrading across a version that changes it.

## Uninstalling

Checkout: `make clean-instances` then delete the clone; vcpkg lives beside it in
`../vcpkg-root`. Package: `rm -rf ~/.infinity ~/.local/bin/infinity` and, if you want
the data gone too, `~/.local/share/infinity`.
