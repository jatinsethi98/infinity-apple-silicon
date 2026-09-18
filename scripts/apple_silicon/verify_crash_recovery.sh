#!/usr/bin/env bash
#
# Verify that acknowledged commits survive an abrupt process death on macOS.
#
# What this proves, and what it deliberately does not
# ---------------------------------------------------
# It kills the server with SIGKILL, which the process cannot trap or flush on, then
# restarts it and requires every acknowledged commit to still be there. That covers
# WAL append, catalog persistence, checkpoint, and replay-on-startup.
#
# It does NOT prove durability against machine crash or power loss, because SIGKILL
# leaves the kernel page cache intact: a test that only kills the process cannot tell
# a synced WAL from an unsynced one. Power-loss durability is a separate claim. Since
# commit b2d59374a the WAL syncs each committed batch (F_FULLFSYNC by default, see the
# [wal] section of conf/infinity_conf.toml), and test/eval_macos/test_wal_durability_e2e.py
# checks that the setting is live; what this script adds is the replay-on-restart half.
#
# Usage:
#   scripts/apple_silicon/verify_crash_recovery.sh [--rows N] [--keep]
#
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

rows=2000
keep=0
instance=recovery
port_offset=200

while (( $# )); do
    case "$1" in
        --rows) rows=${2:?--rows needs a number}; shift ;;
        --keep) keep=1 ;;
        --instance) instance=${2:?}; shift ;;
        --port-offset) port_offset=${2:?}; shift ;;
        -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) printf 'error: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

log() { printf '\n==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# $instance is interpolated into a path that is then `rm -rf`'d, so it must not be
# able to contain a traversal. Restrict it to a plain name rather than trying to
# sanitise afterwards.
[[ $instance =~ ^[A-Za-z0-9_-]+$ ]] \
    || die "--instance must match [A-Za-z0-9_-]+ (it becomes a directory that is deleted): '$instance'"
[[ $port_offset =~ ^[0-9]+$ ]] || die "--port-offset must be a non-negative integer: '$port_offset'"

port=$(( 5432 + port_offset ))
inst_root="$repo_root/build/instances/$instance"
work="$repo_root/build/recovery-check"
slt="$repo_root/build/tools/bin/sqllogictest"
runner="$repo_root/scripts/apple_silicon/run_server.sh"

[[ -x $slt ]] || die "sqllogictest not found; run scripts/apple_silicon/install_test_tools.sh"

mkdir -p "$work"

cleanup() {
    if (( keep )); then
        log "leaving instance '$instance' running (--keep); root: $inst_root"
    else
        bash "$runner" stop --instance "$instance" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# Start from an empty data directory so the row count is unambiguous.
bash "$runner" stop --instance "$instance" >/dev/null 2>&1 || true
rm -rf "$inst_root"

log "starting instance '$instance' on port $port"
bash "$runner" start --instance "$instance" --port-offset "$port_offset" >&2

# ---------------------------------------------------------------- write phase

log "writing $rows rows, a full-text index and a vector index, then reading them back"

{
    printf 'statement ok\nDROP TABLE IF EXISTS recovery_probe;\n\n'
    printf 'statement ok\nCREATE TABLE recovery_probe(id INTEGER, body VARCHAR, vec EMBEDDING(FLOAT, 4));\n\n'
    # Float embeddings and a varchar body on purpose: this is the shape that exercises
    # the text->value conversion path, which is where the Darwin float-parse defect lived.
    for (( i = 0; i < rows; i++ )); do
        printf 'statement ok\nINSERT INTO recovery_probe VALUES (%d, '"'"'row %d harmful chemical'"'"', [%d.5, %d.25, 0.125, %d.0]);\n\n' \
            "$i" "$i" "$((i % 7))" "$((i % 11))" "$((i % 3))"
    done
    printf 'statement ok\nCREATE INDEX rp_ft ON recovery_probe(body) USING FULLTEXT;\n\n'
    printf 'statement ok\nCREATE INDEX rp_hnsw ON recovery_probe(vec) USING HNSW WITH (M=16, ef_construction=50, metric=l2);\n\n'
    printf 'query I\nSELECT COUNT(*) FROM recovery_probe;\n----\n%d\n\n' "$rows"
} >"$work/write.slt"

"$slt" -p "$port" "$work/write.slt" >&2 || die "write phase failed"

# ------------------------------------------------------------------ the kill

pid=$(<"$inst_root/infinity.pid")
log "SIGKILL pid $pid (no chance to flush, close files, or checkpoint)"
kill -9 "$pid"

for _ in $(seq 1 100); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
done
kill -0 "$pid" 2>/dev/null && die "pid $pid survived SIGKILL"
rm -f "$inst_root/infinity.pid"
log "process is gone"

# --------------------------------------------------------------- recovery

log "restarting; the server must replay its WAL to get back to $rows rows"
bash "$runner" start --instance "$instance" --port-offset "$port_offset" >&2

{
    # Row count first: the headline assertion.
    printf 'query I\nSELECT COUNT(*) FROM recovery_probe;\n----\n%d\n\n' "$rows"
    # Then prove the recovered table is queryable through both indexes rather than
    # merely present in the catalog -- a count can be satisfied by metadata alone.
    printf 'query I\nSELECT COUNT(*) FROM recovery_probe WHERE id = 0;\n----\n1\n\n'
    printf 'statement ok\nSELECT id FROM recovery_probe SEARCH MATCH TEXT ('"'"'body'"'"', '"'"'harmful chemical'"'"', '"'"'topn=5'"'"');\n\n'
    printf 'statement ok\nSELECT id FROM recovery_probe SEARCH MATCH VECTOR (vec, [0.5, 0.25, 0.125, 1.0], '"'"'float'"'"', '"'"'l2'"'"', 5);\n\n'
    # And that the recovered table still accepts writes.
    printf 'statement ok\nINSERT INTO recovery_probe VALUES (%d, '"'"'post recovery'"'"', [1.5, 2.25, 0.125, 3.0]);\n\n' "$rows"
    printf 'query I\nSELECT COUNT(*) FROM recovery_probe;\n----\n%d\n\n' "$(( rows + 1 ))"
} >"$work/verify.slt"

if "$slt" -p "$port" "$work/verify.slt" >&2; then
    log "PASS: all $rows acknowledged commits survived SIGKILL; both indexes queryable; table still writable"
    log "note: this is process-death durability, NOT power-loss durability (see the header of this script)"
    exit 0
fi

log "FAIL: data did not survive SIGKILL"
log "server log: $inst_root/stdout.log"
exit 1
