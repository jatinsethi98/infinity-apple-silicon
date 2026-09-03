#!/usr/bin/env bash
#
# Build a relocatable macOS arm64 tarball, then prove it runs from the extracted tree.
#
# Why a relocatable tarball rather than CPack's install rules
# ----------------------------------------------------------
# cmake/Packaging.cmake targets a Linux system install: CMAKE_INSTALL_PREFIX is
# /usr, it ships conf/infinity.service (a systemd unit, meaningless on macOS), and
# it installs no resource/ tree at all -- so a package built from it has none of the
# analyzer dictionaries and every CJK/RAG/IK full-text index fails at runtime.
# On macOS /usr is additionally protected by SIP and cannot be written to.
#
# So this produces a tree that runs from wherever it is unpacked, with no
# privileged step and no assumption that /var/infinity or /usr/share/infinity
# exist. The generated launcher resolves its own location and writes a config
# pointing inside the install, which is what makes it relocatable.
#
# The final step extracts the tarball to a scratch directory, starts the server out
# of it, runs a query, and stops it. A package that has not been started is not a
# package that works -- the resource path, the config template, and the launcher are
# all things that can only be validated by running.
#
# Usage:
#   scripts/apple_silicon/make_package.sh [--binary PATH] [--outdir DIR] [--skip-selftest]
#
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

binary="build/macos-arm64-release/src/infinity"
outdir="build/package"
selftest=1

while (( $# )); do
    case "$1" in
        --binary) binary=${2:?--binary needs a path}; shift ;;
        --outdir) outdir=${2:?--outdir needs a path}; shift ;;
        --skip-selftest) selftest=0 ;;
        -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) printf 'error: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# Same probe as run_server.sh: BSD netcat's -z misreports a bound-but-not-accepting
# socket, and /dev/tcp is not available in every shell build.
port_accepts() {
    python3 - "$1" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || die "this packages for macOS arm64; host is $(uname -s)-$(uname -m)"
[[ -x $binary ]] || die "no server binary at $binary — build it with scripts/apple_silicon/build_server.sh"

# The dictionaries live in a git submodule. Packaging without them silently produces
# an artifact whose CJK/RAG/IK full-text is broken, so refuse rather than warn.
[[ -f resource/jieba/dict/jieba.dict.utf8 ]] \
    || die "resource/ is empty or incomplete — run: git submodule update --init resource"

arch=$(lipo -archs "$binary")
[[ $arch == arm64 ]] || die "$binary is '$arch', expected arm64"

version=$(sed -n 's/^ *"version" *: *"\([^"]*\)".*/\1/p' vcpkg.json | head -1)
[[ -n $version ]] || die "could not read version from vcpkg.json"

name="infinity-${version}-macos-arm64"
stage="$outdir/$name"
tarball="$outdir/${name}.tar.gz"

log "staging $name"
rm -rf "$stage"
mkdir -p "$stage"/{bin,libexec,etc,share/infinity}

install -m 0755 "$binary" "$stage/libexec/infinity"
cp -R resource "$stage/share/infinity/resource"

# Config template. @PREFIX@ is substituted by the launcher at first run rather than
# at package time, because the install location is not known until it is unpacked.
cat >"$stage/etc/infinity_conf.toml.in" <<'EOF'
# Rendered from etc/infinity_conf.toml.in by bin/infinity at startup.
# Edit this template, not the rendered infinity_conf.toml, which is overwritten.
[general]
version                  = "@VERSION@"
time_zone                = "utc-8"

[network]
server_address           = "127.0.0.1"
postgres_port            = 5432
http_port                = 23820
client_port              = 23817
connection_pool_size     = 128
peer_ip                  = "127.0.0.1"
peer_port                = 23850

[log]
log_filename             = "infinity.log"
log_dir                  = "@DATA@/log"
log_to_stdout            = false
log_file_max_size        = "1GB"
log_file_rotate_count    = 3
log_level                = "info"

[storage]
persistence_dir          = "@DATA@/persistence"
data_dir                 = "@DATA@/data"
catalog_dir              = "@DATA@/catalog"
optimize_interval        = "10s"
cleanup_interval         = "60s"
compact_interval         = "120s"
storage_type             = "local"
mem_index_capacity       = 65536
snapshot_dir             = "@DATA@/snapshots"

[buffer]
buffer_manager_size      = "4GB"
lru_num                  = 7
temp_dir                 = "@DATA@/tmp"
result_cache             = "off"
memindex_memory_quota    = "1GB"

[wal]
wal_dir                  = "@DATA@/wal"
checkpoint_interval      = "86400s"
wal_compact_threshold    = "1GB"
wal_flush                = "flush_at_once"

[resource]
resource_dir             = "@PREFIX@/share/infinity/resource"
EOF

# Substitute the version now rather than leaving it for the launcher: the launcher
# would have to read it back out of this same template, which is circular.
sed -i '' -e "s|@VERSION@|$version|g" "$stage/etc/infinity_conf.toml.in"

# Launcher. Resolves the install prefix from its own path so the tree can be moved.
cat >"$stage/bin/infinity" <<'EOF'
#!/usr/bin/env bash
#
# Launcher for a relocatable Infinity install. Renders the config template against
# this install's location and the chosen data directory, then execs the server.
#
#   infinity [--data-dir DIR] [--config FILE] [server args...]
#
# --data-dir defaults to ${INFINITY_DATA_DIR:-$HOME/.local/share/infinity}. Nothing
# is written inside the install tree except the rendered config, so the install can
# live on a read-only volume if --config points elsewhere.
#
set -euo pipefail

self=$(cd -- "$(dirname -- "$(readlink "${BASH_SOURCE[0]}" || printf '%s' "${BASH_SOURCE[0]}")")" && pwd)
prefix=$(cd -- "$self/.." && pwd)

data_dir=${INFINITY_DATA_DIR:-$HOME/.local/share/infinity}
config=""
args=()
while (( $# )); do
    case "$1" in
        --data-dir) data_dir=${2:?--data-dir needs a path}; shift ;;
        --config) config=${2:?--config needs a path}; shift ;;
        *) args+=("$1") ;;
    esac
    shift
done

mkdir -p "$data_dir"/{log,data,catalog,persistence,snapshots,tmp,wal}

if [[ -z $config ]]; then
    # Only @PREFIX@ and @DATA@ are deferred to run time; the version is substituted
    # when the package is built, since it is known then.
    config="$prefix/etc/infinity_conf.toml"
    sed -e "s|@PREFIX@|$prefix|g" -e "s|@DATA@|$data_dir|g" \
        "$prefix/etc/infinity_conf.toml.in" >"$config"
fi

# ${args[@]+...} rather than a bare "${args[@]}": macOS ships bash 3.2, where
# expanding an empty array under `set -u` is an unbound-variable error.
exec "$prefix/libexec/infinity" -f "$config" ${args[@]+"${args[@]}"}
EOF
chmod 0755 "$stage/bin/infinity"

cat >"$stage/README.md" <<EOF
# Infinity $version — macOS arm64

A relocatable build. Unpack it anywhere; nothing needs to be installed and no step
needs root.

    ./bin/infinity

Data goes to \`\$HOME/.local/share/infinity\` by default. Override it:

    ./bin/infinity --data-dir /path/to/data

The server listens on the PostgreSQL wire protocol at 127.0.0.1:5432 and HTTP at
127.0.0.1:23820. To change ports or logging, edit \`etc/infinity_conf.toml.in\` —
\`etc/infinity_conf.toml\` is regenerated from it on every start.

To put it on PATH:

    ln -s "\$PWD/bin/infinity" /usr/local/bin/infinity

\`share/infinity/resource\` holds the full-text analyzer dictionaries (jieba, ik,
rag, mecab, opencc). They are required for CJK and RAG analyzers and are found
automatically relative to this install.

Built from Infinity's native Apple Silicon port. See docs/apple_silicon/ in the
source tree for what is and is not verified on macOS.
EOF

log "creating $tarball"
tar -czf "$tarball" -C "$outdir" "$name"
size=$(du -h "$tarball" | cut -f1)
log "packaged $tarball ($size)"

# ------------------------------------------------------------------- self-test

if (( ! selftest )); then
    log "skipping self-test (--skip-selftest)"
    exit 0
fi

slt="$repo_root/build/tools/bin/sqllogictest"
scratch=$(mktemp -d)
# The package is tested on the port it actually ships with, because "runs with the
# default config" is the property being verified. That means the port has to be free.
port=5432
if port_accepts "$port"; then
    die "port $port is already in use; stop other instances first (scripts/apple_silicon/run_server.sh stop) or pass --skip-selftest"
fi
cleanup_selftest() {
    if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -rf "$scratch"
}
trap cleanup_selftest EXIT

log "self-test: extracting to $scratch and starting the server from it"
tar -xzf "$tarball" -C "$scratch"

# Unpack, then MOVE it: the install ends up at a different path than it was staged at,
# so anything that baked in an absolute prefix at package time is caught here.
mv "$scratch/$name" "$scratch/relocated"
extracted="$scratch/relocated"

"$extracted/bin/infinity" --data-dir "$scratch/data" >"$scratch/server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        log "server exited during self-test; log:"
        tail -n 30 "$scratch/server.log" >&2
        die "self-test failed to start"
    fi
    if port_accepts "$port"; then
        ready=1
        break
    fi
    sleep 0.5
done
(( ready )) || die "self-test: server did not accept connections"

log "self-test: server up from the relocated tree; exercising it"

if [[ -x $slt ]]; then
    cat >"$scratch/probe.slt" <<'SLT'
statement ok
DROP TABLE IF EXISTS pkg_probe;

statement ok
CREATE TABLE pkg_probe(id INTEGER, body VARCHAR, vec EMBEDDING(FLOAT, 4));

statement ok
INSERT INTO pkg_probe VALUES (1, '我来到北京清华大学', [1.5, 2.25, 0.125, 3.0]);

statement ok
CREATE INDEX pkg_ft ON pkg_probe(body) USING FULLTEXT WITH (analyzer=chinese);

query I
SELECT COUNT(*) FROM pkg_probe;
----
1

query T
SELECT body FROM pkg_probe SEARCH MATCH TEXT ('body', '清华大学', 'topn=5;bm25_params=1.2,0.75');
----
我来到北京清华大学
SLT
    # The chinese analyzer is the load-bearing check: it only works if the packaged
    # resource/ tree was found through the relocated prefix. bm25_params exercises
    # the float option parsing at the same time.
    "$slt" -p "$port" "$scratch/probe.slt" >&2 || die "self-test query failed"
    log "self-test PASSED: packaged CJK full-text + BM25 options work from the relocated install"
else
    log "self-test: sqllogictest not present, verified startup only"
    log "  run scripts/apple_silicon/install_test_tools.sh for the query check"
fi

exit 0
