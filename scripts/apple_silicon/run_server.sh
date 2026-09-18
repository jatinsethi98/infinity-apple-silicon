#!/usr/bin/env bash
#
# Start a natively-built Infinity server against repo-local directories.
#
# The shipped conf/infinity_conf.toml points every directory at /var/infinity/*
# and resource_dir at /usr/share/infinity/resource. On Linux that matches the
# packaged install and CI runs as root, so it just works. On macOS /var and
# /usr/share are root-owned, so a developer running from a checkout cannot use
# the default config at all.
#
# Rather than ask people to edit a tracked config (and risk committing it), this
# renders a config into the build tree with absolute paths derived from the
# checkout, and points resource_dir at the repository's own resource/ tree so the
# dictionary-backed full-text analyzers (jieba, ik, rag, mecab, opencc) work
# without a system install.
#
# Usage:
#   scripts/apple_silicon/run_server.sh [start|stop|status|restart] [--instance NAME]
#
# Options:
#   --instance NAME   run an independent instance with its own data directory
#                     (default: dev)
#   --port-offset N   add N to every listening port (default 0, i.e. the shipped
#                     defaults 5432/23820/23817). Give a second instance a non-zero
#                     offset to run it alongside the first.
#   --fresh           delete this instance's data directory before starting. Required
#                     when rerunning the SQL logic suite, which is not idempotent
#                     against surviving state (see do_start).
#   --foreground      run in the foreground instead of detaching
#   --binary PATH     server binary (default: build/macos-arm64-release/src/infinity)
#   --config FILE     start with this config instead of the rendered one. The wrapper
#                     still derives ports from --port-offset for its readiness probe
#                     and status, so pass the offset that matches the file.
#   --wal-flush MODE  WAL durability: full_fsync (default, power-loss safe, ~2.3ms/sync)
#                     | fsync (OS-crash safe, ~28us) | no_sync (~1.5us, unsafe)
#
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

action=start
instance=dev
port_offset=0
fresh=0
foreground=0
binary="build/macos-arm64-release/src/infinity"
config=""
# full_fsync matches the shipped default, so a development instance exercises the same
# durability path users get. Override with --wal-flush no_sync when benchmarking, where
# a ~2.3 ms device flush per commit batch would dominate the measurement.
wal_flush=full_fsync

while (( $# )); do
    case "$1" in
        start|stop|status|restart) action=$1 ;;
        --instance) instance=${2:?--instance needs a name}; shift ;;
        --port-offset) port_offset=${2:?--port-offset needs a number}; shift ;;
        --fresh) fresh=1 ;;
        --foreground) foreground=1 ;;
        --binary) binary=${2:?--binary needs a path}; shift ;;
        --config) config=${2:?--config needs a path}; shift ;;
        --wal-flush) wal_flush=${2:?--wal-flush needs a mode}; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) printf 'error: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# $instance becomes a directory that --fresh deletes, so it must not be able to contain
# a path traversal.
[[ $instance =~ ^[A-Za-z0-9_-]+$ ]] \
    || die "--instance must match [A-Za-z0-9_-]+ (it becomes a directory name): '$instance'"
[[ $port_offset =~ ^[0-9]+$ ]] || die "--port-offset must be a non-negative integer: '$port_offset'"

# Give each instance its own root and its own ports, so a second instance can run
# alongside the first without either noticing.
inst_root="$repo_root/build/instances/$instance"
conf="$inst_root/infinity_conf.toml"
pidfile="$inst_root/infinity.pid"

# Ports are the shipped defaults plus an explicit offset. Deriving the offset from
# a hash of the instance name would avoid collisions automatically, but it also
# means the default instance does not listen on 5432 -- and sqllogictest connects
# to 5432 with no way to know otherwise. Predictability matters more here than
# automatic collision avoidance.
pg_port=$(( 5432 + port_offset ))
http_port=$(( 23820 + port_offset ))
client_port=$(( 23817 + port_offset ))
# The peer server also listens, and its default (23850) is NOT in
# conf/infinity_conf.toml -- it only appears in the cluster configs
# (conf/leader.toml). A second instance therefore dies binding a port the first one
# already holds, with the failure appearing AFTER the log lines that look like a
# successful start. It also defaults to 0.0.0.0; a development instance has no
# reason to accept peer connections from off-box, so bind loopback.
peer_port=$(( 23850 + port_offset ))

# Readiness probe. `nc -z` is not usable: macOS ships a BSD netcat whose -z with a
# listening-but-not-yet-accepting socket returns non-zero, so it reported failure
# for a server that had in fact bound the port. Bash's /dev/tcp is also out, since
# this may run under a shell built without it. A tiny python connect() is the one
# check that is both accurate and present on every macOS.
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

server_pid() {
    [[ -f $pidfile ]] || return 1
    local pid
    pid=$(<"$pidfile")
    [[ -n $pid ]] || return 1
    [[ $pid =~ ^[0-9]+$ ]] || { rm -f "$pidfile"; return 1; }
    kill -0 "$pid" 2>/dev/null || { rm -f "$pidfile"; return 1; }

    # A live pid is not proof it is still OUR server: pids are recycled, and a stale
    # pidfile plus a reused pid would make `stop` SIGKILL an unrelated process. Require
    # that the process is actually running this instance's config before signalling it.
    if ! ps -o command= -p "$pid" 2>/dev/null | grep -qF -- "$conf"; then
        rm -f "$pidfile"
        return 1
    fi
    printf '%s' "$pid"
}

do_status() {
    if pid=$(server_pid); then
        log "instance '$instance' running: pid $pid, pg=$pg_port http=$http_port client=$client_port"
        log "root: $inst_root"
        return 0
    fi
    log "instance '$instance' not running"
    return 1
}

do_stop() {
    if ! pid=$(server_pid); then
        log "instance '$instance' not running"
        return 0
    fi
    log "stopping pid $pid"
    # Only ever signal the PID we recorded. Matching on the process name would be
    # able to kill an unrelated 'infinity' process on a developer's machine.
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
        # Re-check identity immediately before SIGKILL. Ten seconds passed while waiting
        # for SIGTERM, which is long enough for the pid to have been freed and reused by
        # an unrelated process; liveness alone is not evidence it is still our server.
        if ps -o command= -p "$pid" 2>/dev/null | grep -qF -- "$conf"; then
            log "did not exit on SIGTERM, sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        else
            log "pid $pid is no longer this instance's server; not signalling it"
        fi
    fi
    rm -f "$pidfile"
    log "stopped"
}

write_config() {
    mkdir -p "$inst_root"/{log,data,catalog,persistence,snapshots,tmp,wal}
    # A sentinel file, not just the directory: resource/ is a submodule, and an
    # uninitialised one is an empty directory that passes an existence check and then
    # fails every dictionary-backed full-text index at runtime.
    [[ -f $repo_root/resource/jieba/dict/jieba.dict.utf8 ]] \
        || die "resource/ is empty or incomplete — run: git submodule update --init --recursive resource"

    # Read the version rather than hardcoding it, so this does not go stale on the next
    # version bump. make_package.sh reads the same source.
    local version
    version=$(sed -n 's/^ *"version" *: *"\([^"]*\)".*/\1/p' "$repo_root/vcpkg.json" | head -1)
    [[ -n $version ]] || die "could not read version from vcpkg.json"

    cat >"$conf" <<EOF
# Generated by scripts/apple_silicon/run_server.sh for instance '$instance'.
# Edits here are overwritten on the next start; change the script instead.
[general]
version                  = "$version"
time_zone                = "utc-8"

[network]
server_address           = "127.0.0.1"
postgres_port            = $pg_port
http_port                = $http_port
client_port              = $client_port
connection_pool_size     = 128
peer_ip                  = "127.0.0.1"
peer_port                = $peer_port

[log]
log_filename             = "infinity.log"
log_dir                  = "$inst_root/log"
log_to_stdout            = false
log_file_max_size        = "1GB"
log_file_rotate_count    = 3
log_level                = "info"

[storage]
persistence_dir          = "$inst_root/persistence"
data_dir                 = "$inst_root/data"
catalog_dir              = "$inst_root/catalog"
optimize_interval        = "10s"
cleanup_interval         = "60s"
compact_interval         = "120s"
storage_type             = "local"
mem_index_capacity       = 65536
snapshot_dir             = "$inst_root/snapshots"

[buffer]
buffer_manager_size      = "4GB"
lru_num                  = 7
temp_dir                 = "$inst_root/tmp"
result_cache             = "off"
memindex_memory_quota    = "1GB"

[wal]
wal_dir                  = "$inst_root/wal"
checkpoint_interval      = "86400s"
wal_compact_threshold    = "1GB"
wal_flush                = "$wal_flush"

[resource]
resource_dir             = "$repo_root/resource"
EOF
    log "wrote $conf"
}

do_start() {
    if pid=$(server_pid); then
        die "instance '$instance' already running as pid $pid (use restart, or stop first)"
    fi
    [[ -x $binary ]] || die "no server binary at $binary — build it with scripts/apple_silicon/build_server.sh"

    if (( fresh )); then
        # The SQL logic suite is not idempotent against a surviving data directory:
        # ddl/drop/test_drop_column.slt does CREATE SNAPSHOT with a fixed name, which
        # fails on a second run because the snapshot from the first is still there.
        # Anything that reruns the suite against the same instance needs this.
        log "removing existing data for instance '$instance'"
        rm -rf "$inst_root"
    fi

    if [[ -n $config ]]; then
        [[ -f $config ]] || die "no config file at $config"
        # Absolute, because the server is started from the repository root and the
        # user may have passed a relative path.
        conf=$(cd -- "$(dirname -- "$config")" && pwd)/$(basename -- "$config")
        mkdir -p "$inst_root"
        log "using $conf as given; --port-offset $port_offset must match the ports in it"
    else
        write_config
    fi

    if (( foreground )); then
        log "starting in foreground: $binary -f $conf"
        exec "$binary" -f "$conf"
    fi

    log "starting $binary (pg=$pg_port http=$http_port client=$client_port)"
    "$binary" -f "$conf" >"$inst_root/stdout.log" 2>&1 &
    local pid=$!
    printf '%s' "$pid" >"$pidfile"

    # Wait for the port to accept a connection rather than sleeping a fixed amount:
    # first start initialises a catalog and is slower than a restart that replays one.
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pidfile"
            log "server exited during startup; last lines of $inst_root/stdout.log:"
            tail -n 20 "$inst_root/stdout.log" >&2 || true
            die "startup failed"
        fi
        if port_accepts "$pg_port"; then
            log "ready: pid $pid, postgres port $pg_port (peer $peer_port, http $http_port)"
            return 0
        fi
        sleep 0.5
    done
    die "server did not accept connections on port $pg_port within 60s"
}

case "$action" in
    start)   do_start ;;
    stop)    do_stop ;;
    status)  do_status ;;
    restart) do_stop; do_start ;;
esac
