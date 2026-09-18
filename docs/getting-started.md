# Getting started

From nothing to a running server and a first query. The commands here are the ones
`make` wraps; the scripts behind them live in `scripts/apple_silicon/` and every one
prints `--help`.

## What you need

| | |
|---|---|
| Mac | Any Apple Silicon Mac (M1 or later). 16 GB of memory is comfortable; 8 GB builds, slowly. |
| macOS | 14 (Sonoma) or newer. |
| Tools | [Homebrew](https://brew.sh) and the Xcode command line tools: `xcode-select --install`. |
| Disk | About 20 GB free for a source build (the build tree alone is 11 GB, vcpkg 2 GB). A prebuilt package needs about 150 MB. |
| Time | About 12 minutes for a source build on an M4 Mac mini, measured; longer on older chips or slow networks. Under a minute for a package. |

Check all of it in one go. This changes nothing on your machine:

```sh
make doctor
```

Every `MISSING` line comes with the command that fixes it.

## Install

### From source

```sh
git clone --recurse-submodules https://github.com/jatinsethi98/infinity-apple-silicon.git
cd infinity-apple-silicon
make setup
```

`make setup` runs `scripts/apple_silicon/setup.sh`, which:

1. checks the host is arm64 macOS with the Xcode tools,
2. installs the Homebrew toolchain: LLVM 20, CMake, Ninja, libomp, bison and pkg-config,
3. fetches the `resource` submodule (about 500 MB of full-text analyzer dictionaries)
   if `--recurse-submodules` was forgotten,
4. clones [vcpkg](https://github.com/microsoft/vcpkg) next to the repository at the
   exact baseline pinned in `vcpkg.json` and bootstraps it,
5. builds the server. Dependencies come first (about 6 minutes on an M4), then
   roughly 1,500 C++23 module translation units (about 3 minutes on all ten cores).

It is safe to interrupt and re-run: each step is skipped once done. Useful options:

```sh
scripts/apple_silicon/setup.sh --with-tests   # also build the unit tests and install sqllogictest
scripts/apple_silicon/setup.sh --no-build     # prerequisites only
scripts/apple_silicon/setup.sh --skip-brew    # you provide clang 20, cmake, ninja, libomp, bison, pkg-config
scripts/apple_silicon/setup.sh --vcpkg-root ~/src/vcpkg
```

The binary lands at `build/macos-arm64-release/src/infinity`. To rebuild after a code
change, `make build`. To build in a fresh shell later, export `VCPKG_ROOT` first;
`setup.sh` prints the exact line at the end.

Why Homebrew LLVM and not Apple Clang: the engine is written against C++23 modules
and `import std`, which Apple's toolchain does not support. Why the CMake range is
capped (4.0.3 up to 4.4.x): CMake gates `import std` behind a version-specific key that
has to be adopted deliberately, so `setup.sh` never upgrades an existing CMake for you.

### Prebuilt package

Releases attach a relocatable tarball, `infinity-<version>-macos-arm64.tar.gz`, and a
`.sha256` beside it. The installer downloads the latest one, verifies it, unpacks it
under `~/.infinity`, and links `infinity` into `~/.local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/jatinsethi98/infinity-apple-silicon/main/install.sh | bash
```

The same script installs a tarball you built yourself with `make package`:

```sh
./install.sh --from-file build/package/infinity-0.7.3-macos-arm64.tar.gz
```

Inside the package, `bin/infinity` is a launcher that renders a configuration for
wherever the tree was unpacked and starts `libexec/infinity` with it. Data goes to
`~/.local/share/infinity` unless you pass `--data-dir`. There is nothing to
uninstall beyond `rm -rf ~/.infinity ~/.local/bin/infinity`.

Until the first release is tagged there is no package to download, and the installer
says so and points at the source build.

## Start, stop, look

From a source checkout:

```sh
make start      # starts an instance named "dev" on 127.0.0.1
make status
make logs       # follows build/instances/dev/log/infinity.log
make stop
```

From a package:

```sh
infinity                      # foreground; Ctrl-C stops it
infinity --data-dir ~/idx     # somewhere else for the data
```

Either way the server listens on three ports:

| Port | Protocol | Used by |
|---|---|---|
| 23817 | Thrift | the Python SDK |
| 23820 | HTTP | `curl`, any language |
| 5432 | PostgreSQL wire | `psql`, SQL clients, the SQL test suite |

The source checkout never uses the shipped `conf/infinity_conf.toml` directly. It
points every directory at `/var/infinity`, which is root-owned on macOS, so
`run_server.sh` renders a per-instance copy under `build/instances/<name>/` with paths
inside the checkout. Details, including running two instances side by side, are in
[Operations](guides/operations.md).

## Your first query

With the server running, install the SDK into the repository's virtual environment:

```sh
make sdk                                   # uv sync --python 3.11 --all-extras
uv run python example/simple_example.py    # create, insert, filter, drop
```

Or from any Python environment: `pip install infinity-sdk`. The distribution is named
`infinity-sdk`; the import is `infinity`. The [Python guide](guides/python.md) takes it
from there.

Without Python, three `curl` calls do the same thing:

```sh
# create a table
curl -X POST 'http://127.0.0.1:23820/databases/default_db/tables/hello' \
  -H 'content-type: application/json' \
  -d '{"create_option": "ignore_if_exists",
       "fields": [{"name": "id", "type": "integer", "constraints": ["PRIMARY KEY"]},
                  {"name": "text", "type": "varchar"}]}'

# insert a row
curl -X POST 'http://127.0.0.1:23820/databases/default_db/tables/hello/docs' \
  -H 'content-type: application/json' \
  -d '[{"id": 1, "text": "a bloom filter tests set membership"}]'

# read it back (a SELECT is a GET with a JSON body)
curl -X GET 'http://127.0.0.1:23820/databases/default_db/tables/hello/docs' \
  -H 'content-type: application/json' \
  -d '{"output": ["id", "text"], "filter": "id >= 1"}'
```

The [HTTP guide](guides/http.md) covers indexes and search the same way.

## See what it is for

```sh
make demo
```

loads 36 short articles into one table with a full-text index, a vector index and
ordinary columns, then asks questions that keyword search and vector search get wrong
in opposite directions, and shows fusion getting both right. The first run downloads a
90 MB embedding model. [demo/README.md](../demo/README.md) explains the results.

## Run the tests

```sh
make build-tests   # server + unit-test binary
make test          # 1,131 C++ unit tests; 5 known failures are documented
make slt           # 205 SQL logic tests on a fresh instance
make recovery      # kill a server mid-write, restart, check every commit came back
make package       # build a tarball and start it from a scratch directory
```

What each of those proves, and what none of them cover, is recorded in
[MACOS_VERIFICATION.md](apple_silicon/MACOS_VERIFICATION.md).

## When something goes wrong

| Symptom | Cause and fix |
|---|---|
| `make doctor` says `bison 2.3 ... too old` | macOS ships an old bison and the thrift dependency needs 3.7+. `brew install bison`. |
| `Could not find pkg-config` while building abseil | vcpkg's ports need it and macOS ships none. `brew install pkg-config`; `make doctor` checks for it. |
| `cmake ... is too old` or `import std` errors | CMake must be 4.0.3 or newer and below 4.5: `brew upgrade cmake`. |
| `cmake ... newer than this build supports` | Install a supported one alongside: `brew unlink cmake && brew install cmake@4.4`. |
| `no clang++ at /opt/homebrew/opt/llvm@20/bin` | Apple Clang cannot build this. `brew install llvm@20`. |
| `'pthread.h' file not found` at configure | `SDKROOT` is unset in this shell. `export SDKROOT=$(xcrun --show-sdk-path)`; the scripts do this for you. |
| vcpkg reports a missing `spdlog`, `thrift` or `zlib` | A shallow vcpkg clone cannot resolve the pinned baseline. `setup.sh` and `build_server.sh` repair it; re-run either. |
| `unrecognized option --file-prefix-map` while building thrift | Same bison problem as the first row. |
| `INVALID_SERVER_ADDRESS` from the SDK | `infinity.connect()` takes `NetworkAddress("127.0.0.1", 23817)`, not a string. The server is probably fine. |
| Analyzer or CJK full-text tests fail | The `resource` submodule is missing: `git submodule update --init --recursive resource`. |
| `uname -m` says `x86_64` on an M-series Mac | You are in a Rosetta shell. `arch -arm64 zsh`. |
| A second server dies right after "started" | Port collision, including the peer port 23850. Use `make start INSTANCE=two` with `--port-offset`; see [Operations](guides/operations.md). |
| The disk filled up mid-build | The build tree is 11 GB and vcpkg another 2 GB, plus the dictionaries and instances. Free 20 GB and re-run `make setup`; it resumes. |

Server logs are at `build/instances/<name>/log/infinity.log` for a checkout and
`<data-dir>/log/infinity.log` for a package. Build logs are at
`build/<preset>-build.log`. If none of this helps, open an
[issue](https://github.com/jatinsethi98/infinity-apple-silicon/issues/new/choose) with
the output of `make doctor` and the tail of the relevant log.
