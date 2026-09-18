#!/usr/bin/env bash
#
# Install a prebuilt Infinity server for Apple Silicon from a GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/jatinsethi98/infinity-apple-silicon/main/install.sh | bash
#
# What it does:
#   1. finds the release (latest, or --version vX.Y.Z) and its macos-arm64 tarball
#   2. downloads it, verifies the SHA-256 the release publishes next to it
#   3. unpacks it under ~/.infinity/versions/<name>, points ~/.infinity/current at it
#   4. links ~/.local/bin/infinity -> ~/.infinity/current/bin/infinity
#
# Nothing needs sudo and nothing outside those two directories is touched. Remove
# an install with `rm -rf ~/.infinity ~/.local/bin/infinity`. Data written by the
# server goes to ~/.local/share/infinity (see `infinity --help`), not the install.
#
# Options (flags or environment variables):
#   --version TAG      INFINITY_VERSION   release tag, e.g. v0.7.3-apple.1 (default: latest)
#   --prefix DIR       INFINITY_PREFIX    install root (default: ~/.infinity)
#   --bin-dir DIR      INFINITY_BIN_DIR   where to link the launcher (default: ~/.local/bin)
#   --from-file PATH                      install a local tarball (what `make package` builds)
#   --no-link                             do not create the bin-dir symlink
#   --repo OWNER/NAME  INFINITY_REPO      GitHub repository (default: jatinsethi98/infinity-apple-silicon)
#
set -euo pipefail

repo=${INFINITY_REPO:-jatinsethi98/infinity-apple-silicon}
version=${INFINITY_VERSION:-latest}
prefix=${INFINITY_PREFIX:-$HOME/.infinity}
bin_dir=${INFINITY_BIN_DIR:-$HOME/.local/bin}
from_file=""
link=1

while (( $# )); do
    case "$1" in
        --version)   version=${2:?--version needs a tag}; shift ;;
        --prefix)    prefix=${2:?--prefix needs a directory}; shift ;;
        --bin-dir)   bin_dir=${2:?--bin-dir needs a directory}; shift ;;
        --from-file) from_file=${2:?--from-file needs a path}; shift ;;
        --repo)      repo=${2:?--repo needs owner/name}; shift ;;
        --no-link)   link=0 ;;
        -h|--help)   sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'error: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ host

[[ $(uname -s) == Darwin ]] || die "these builds are for macOS only; on $(uname -s) use upstream https://github.com/infiniflow/infinity"
if [[ $(uname -m) != arm64 ]]; then
    if [[ $(sysctl -n sysctl.proc_translated 2>/dev/null || printf 0) == 1 ]]; then
        die "this shell runs under Rosetta; start a native one (arch -arm64 zsh) and re-run"
    fi
    die "uname -m reports $(uname -m); an Apple Silicon Mac is required"
fi
macos=$(sw_vers -productVersion)
(( ${macos%%.*} >= 14 )) || die "macOS $macos is too old; the builds target macOS 14 and newer"
command -v curl >/dev/null || die "curl is required"
command -v tar  >/dev/null || die "tar is required"

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# --------------------------------------------------------------- tarball

if [[ -n $from_file ]]; then
    [[ -f $from_file ]] || die "no such file: $from_file"
    tarball=$from_file
    log "installing from $tarball"
else
    api="https://api.github.com/repos/$repo/releases"
    if [[ $version == latest ]]; then
        release_json=$(curl -fsSL "$api/latest") || die "no published release found at github.com/$repo/releases.
       Until the first release is cut, build from source instead:
         git clone --recurse-submodules https://github.com/$repo.git && cd ${repo#*/} && make setup"
    else
        release_json=$(curl -fsSL "$api/tags/$version") || die "no release tagged $version at github.com/$repo/releases"
    fi
    tag=$(printf '%s' "$release_json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
    # Release assets are named infinity-<version>-macos-arm64.tar.gz by make_package.sh,
    # with a .sha256 file beside each tarball.
    asset_url=$(printf '%s' "$release_json" \
        | sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*macos-arm64\.tar\.gz\)".*/\1/p' | head -1)
    sha_url=$(printf '%s' "$release_json" \
        | sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*macos-arm64\.tar\.gz\.sha256\)".*/\1/p' | head -1)
    [[ -n $asset_url ]] || die "release $tag has no macos-arm64 tarball attached"
    log "release $tag: $asset_url"

    tarball="$scratch/$(basename "$asset_url")"
    curl -fSL --progress-bar -o "$tarball" "$asset_url"

    if [[ -n $sha_url ]]; then
        expected=$(curl -fsSL "$sha_url" | awk '{print $1}')
        actual=$(shasum -a 256 "$tarball" | awk '{print $1}')
        [[ $expected == "$actual" ]] || die "SHA-256 mismatch for $(basename "$tarball"): expected $expected, got $actual"
        log "SHA-256 verified"
    else
        log "warning: the release publishes no .sha256 for this asset; skipping verification"
    fi
fi

# --------------------------------------------------------------- unpack

name=$(tar -tzf "$tarball" | head -1 | cut -d/ -f1)
[[ -n $name ]] || die "could not read the top-level directory out of $tarball"
[[ $name == infinity-* ]] || die "unexpected tarball layout (top-level directory is '$name', expected infinity-<version>-macos-arm64)"

versions_dir="$prefix/versions"
target="$versions_dir/$name"
mkdir -p "$versions_dir"
if [[ -d $target ]]; then
    log "replacing existing $target"
    rm -rf "$target"
fi
tar -xzf "$tarball" -C "$versions_dir"
[[ -x $target/bin/infinity && -x $target/libexec/infinity ]] || die "$target does not contain bin/infinity and libexec/infinity; the tarball is not an Infinity package"

archs=$(lipo -archs "$target/libexec/infinity" 2>/dev/null || printf '?')
case "$archs" in
    *arm64*) ;;
    *) die "the server binary in this package is '$archs', not arm64" ;;
esac

ln -sfn "$target" "$prefix/current"
log "installed $name -> $prefix/current"

# ---------------------------------------------------------------- link

if (( link )); then
    mkdir -p "$bin_dir"
    ln -sfn "$prefix/current/bin/infinity" "$bin_dir/infinity"
    log "linked $bin_dir/infinity"
    case ":$PATH:" in
        *":$bin_dir:"*) on_path=1 ;;
        *) on_path=0 ;;
    esac
else
    on_path=0
fi

# -------------------------------------------------------------- summary

cat >&2 <<EOF

Done. $name is installed at:

  $prefix/current

Start the server (Ctrl-C stops it; data goes to ~/.local/share/infinity):

  $( (( link )) && printf '%s' "$bin_dir/infinity" || printf '%s' "$prefix/current/bin/infinity" )

It listens on 127.0.0.1: 23817 (Python SDK), 23820 (HTTP), 5432 (PostgreSQL wire).
Then, from Python:

  pip install infinity-sdk        # or: uv add infinity-sdk
  python -c 'import infinity; print(infinity.connect().list_databases())'

Docs: https://github.com/$repo/tree/main/docs
EOF
if (( link )) && (( ! on_path )); then
    printf '\nNote: %s is not on your PATH. Add it, or call the launcher by its full path.\n' "$bin_dir" >&2
fi
