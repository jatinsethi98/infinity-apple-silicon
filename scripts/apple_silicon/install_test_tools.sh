#!/usr/bin/env bash
#
# Install the external test tools the suites need, into a repo-local prefix.
#
# Only sqllogictest today. It is a Rust binary that the .slt suites are driven
# through by tools/sqllogictest.py, installed from a release tarball rather than
# built. The version is pinned here so that a local run and a CI run disagree about
# the engine, never about the harness: .github/workflows/macos_arm64.yml calls this
# same script rather than installing its own copy, which is what makes the pin
# single-sourced. (Upstream's tests.yml pinned it separately; that workflow does not
# exist in this fork.)
#
# Installs to build/tools/bin rather than /usr/local/bin: no sudo, and nothing
# outside the repo changes. Add it to PATH:
#
#   export PATH="$PWD/build/tools/bin:$PATH"
#
set -euo pipefail

# The single source of truth for the sqllogictest version, local and CI alike.
SQLLOGICTEST_VERSION=0.28.4

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

prefix="$repo_root/build/tools/bin"
mkdir -p "$prefix"

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# SHA-256 of each release tarball, pinned per platform. A pinned tag and TLS are not
# enough on their own: a GitHub release asset can be replaced after publication without
# the tag changing, and this script makes the result executable and then runs it. Verify
# before extracting.
#
# To add or refresh a platform:
#   curl -fsSLO https://github.com/risinglightdb/sqllogictest-rs/releases/download/v<VER>/<ASSET>
#   shasum -a 256 <ASSET>
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
        target=aarch64-apple-darwin
        sha256=48a1481afaf7fd447351bd3e4ccbea4131adee8d3358ad0bc43623d9677692b2 ;;
    Darwin-x86_64)
        target=x86_64-apple-darwin
        sha256=03011c7459af5f34850e1d1b1c9a242fa62c93060e2d976038391f0f262f4bb4 ;;
    Linux-aarch64)
        target=aarch64-unknown-linux-musl
        sha256=d887690cbda94b8be022945e66df77fd914f73b115a22dd0341d98eadbdf4bac ;;
    Linux-x86_64)
        target=x86_64-unknown-linux-musl
        sha256=2dd992e5fa0d02c224f61b09542512d92bf4036cf942ea6f96a0db646228f8e8 ;;
    *) die "no sqllogictest release asset for $(uname -s)-$(uname -m)" ;;
esac

if [[ -x "$prefix/sqllogictest" ]]; then
    have=$("$prefix/sqllogictest" --version 2>/dev/null | awk '{print $2}' || true)
    if [[ $have == "$SQLLOGICTEST_VERSION" ]]; then
        log "sqllogictest $have already installed at $prefix"
        exit 0
    fi
    log "replacing sqllogictest $have with $SQLLOGICTEST_VERSION"
fi

tarball="sqllogictest-bin-v${SQLLOGICTEST_VERSION}-${target}.tar.gz"
url="https://github.com/risinglightdb/sqllogictest-rs/releases/download/v${SQLLOGICTEST_VERSION}/${tarball}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

log "downloading $tarball"
curl -fsSL "$url" -o "$tmp/$tarball" || die "download failed: $url"

actual=$(shasum -a 256 "$tmp/$tarball" | awk '{print $1}')
if [[ $actual != "$sha256" ]]; then
    die "checksum mismatch for $tarball
  expected $sha256
  actual   $actual
Refusing to install. If the upstream release was legitimately re-published, verify the
new asset yourself and update the pin in this script."
fi
log "checksum verified"

tar -xzf "$tmp/$tarball" -C "$tmp"
[[ -f "$tmp/sqllogictest" ]] || die "tarball did not contain a 'sqllogictest' binary"
install -m 0755 "$tmp/sqllogictest" "$prefix/sqllogictest"

log "installed $("$prefix/sqllogictest" --version) -> $prefix/sqllogictest"
log 'add it to PATH:  export PATH="$PWD/build/tools/bin:$PATH"'
