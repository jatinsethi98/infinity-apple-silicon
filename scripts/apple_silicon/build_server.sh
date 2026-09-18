#!/usr/bin/env bash
#
# Configure and build the Infinity server natively on Apple Silicon.
#
# This exists because the build needs three pieces of environment that are not
# discoverable from CMakePresets.json alone, and getting any of them wrong
# produces a failure that looks like something else:
#
#   SDKROOT      Homebrew clang does not find the platform headers without it.
#                Configure fails with "'pthread.h' file not found", which reads
#                like a broken toolchain rather than a missing SDK path.
#   VCPKG_ROOT   The preset's CMAKE_TOOLCHAIN_FILE interpolates it. Unset, CMake
#                reports a missing toolchain file for a path that is just "/scripts/...".
#   the baseline The vcpkg checkout must be able to `git show <baseline>:versions/baseline.json`
#                for the commit pinned in vcpkg.json. A shallow clone has the commit
#                as a graft without its tree, so every port fails to resolve a version
#                and the error names spdlog/thrift/zlib rather than the clone depth.
#                We repair that here instead of documenting it as a gotcha.
#
# Usage:
#   scripts/apple_silicon/build_server.sh [preset] [target ...]
#
#   preset   macos-arm64-release (default) | macos-arm64-debug
#   target   ninja targets; default is the preset's default target
#
# Environment overrides:
#   VCPKG_ROOT   path to a bootstrapped vcpkg checkout (searched for if unset)
#   LLVM_PREFIX  Homebrew LLVM prefix (default /opt/homebrew/opt/llvm@20)
#   JOBS         parallel build jobs (default: all cores)
#
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

preset=${1:-macos-arm64-release}
shift || true
targets=("$@")

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- host checks

[[ $(uname -s) == Darwin ]] || die "this script is for macOS; on Linux use the top-level CMakeLists directly"
[[ $(uname -m) == arm64 ]] || die "this script targets Apple Silicon; uname -m reports $(uname -m)"

llvm_prefix=${LLVM_PREFIX:-/opt/homebrew/opt/llvm@20}
[[ -x "$llvm_prefix/bin/clang++" ]] || die "no clang++ at $llvm_prefix/bin — install with: brew install llvm@20"

command -v cmake >/dev/null || die "cmake not found — brew install cmake"
command -v ninja >/dev/null || die "ninja not found — brew install ninja"

# vcpkg's thrift port needs bison 3.7+ (it passes --file-prefix-map). macOS ships
# bison 2.3 and vcpkg only warns before using it, so the failure would otherwise
# arrive minutes into the dependency build as "unrecognized option". vcpkg looks in
# the two Homebrew prefixes before PATH; check the same places in the same order.
bison_bin=""
for candidate in /opt/homebrew/opt/bison/bin/bison /usr/local/opt/bison/bin/bison "$(command -v bison || true)"; do
    if [[ -n $candidate && -x $candidate ]]; then bison_bin=$candidate; break; fi
done
[[ -n $bison_bin ]] || die "bison not found — the thrift dependency needs bison 3.7+: brew install bison"
bison_sortable=$("$bison_bin" --version | head -1 | awk '{print $NF}' | awk -F. '{printf "%d%03d", $1, $2}')
(( bison_sortable >= 3007 )) \
    || die "$bison_bin is bison $("$bison_bin" --version | head -1 | awk '{print $NF}'); the thrift dependency needs 3.7+ — brew install bison"

# The presets key `import std` support to the CMake version, so an old CMake
# configures and then fails deep in the build with unresolved std module imports.
cmake_version=$(cmake --version | head -1 | awk '{print $3}')
# Compare all three components: a major-only check accepted 4.0.0, which is below the
# 4.0.3 the presets actually need for their `import std` support.
cmake_sortable=$(printf '%s' "$cmake_version" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')
(( cmake_sortable >= 4000003 )) \
    || die "cmake $cmake_version is too old; 4.0.3 or newer required (brew upgrade cmake)"
# And the upper end: CMakeLists.txt maps the CMAKE_EXPERIMENTAL_CXX_IMPORT_STD UUID
# per version and FATAL_ERRORs at 4.5 or above. Catch that here, where the message
# can say what to do about it, rather than at the configure step where it cannot.
(( cmake_sortable < 4005000 )) \
    || die "cmake $cmake_version is newer than this build supports (4.0.3 up to 4.4.x);
       \`import std\` is gated behind a version-specific UUID that has to be adopted
       deliberately. Install a supported cmake, e.g. brew install cmake@4.4"

# ------------------------------------------------------------------- SDKROOT

if [[ -z ${SDKROOT:-} ]]; then
    SDKROOT=$(xcrun --show-sdk-path) || die "xcrun could not locate the macOS SDK; is Xcode or the CLT installed?"
    export SDKROOT
fi
[[ -d $SDKROOT ]] || die "SDKROOT=$SDKROOT is not a directory"
log "SDKROOT=$SDKROOT"

# ----------------------------------------------------------------- VCPKG_ROOT

if [[ -z ${VCPKG_ROOT:-} ]]; then
    for candidate in "$repo_root/../vcpkg-root" "$repo_root/../vcpkg" "$HOME/vcpkg" /opt/vcpkg; do
        if [[ -f "$candidate/.vcpkg-root" ]]; then
            VCPKG_ROOT=$(cd -- "$candidate" && pwd)
            break
        fi
    done
fi
[[ -n ${VCPKG_ROOT:-} ]] || die "VCPKG_ROOT is unset and no vcpkg checkout was found; see docs/apple_silicon/README.md"
export VCPKG_ROOT
[[ -f "$VCPKG_ROOT/.vcpkg-root" ]] || die "VCPKG_ROOT=$VCPKG_ROOT does not look like a vcpkg checkout (no .vcpkg-root)"
[[ -x "$VCPKG_ROOT/vcpkg" ]] || die "vcpkg is not bootstrapped at $VCPKG_ROOT — run ./bootstrap-vcpkg.sh -disableMetrics there"
log "VCPKG_ROOT=$VCPKG_ROOT"

# ------------------------------------------------- vcpkg baseline reachability

# Manifest mode resolves every port's version through the baseline commit's
# versions/baseline.json. Read the pin out of vcpkg.json rather than hardcoding it
# so this check keeps working when the manifest is bumped.
baseline=$(sed -n 's/.*"builtin-baseline"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' vcpkg.json | head -1)
if [[ -n $baseline ]]; then
    if ! git -C "$VCPKG_ROOT" show "$baseline:versions/baseline.json" >/dev/null 2>&1; then
        log "vcpkg baseline $baseline is not resolvable in $VCPKG_ROOT (shallow clone?); fetching it"
        git -C "$VCPKG_ROOT" fetch --depth=1 origin "$baseline" >&2 \
            || die "could not fetch vcpkg baseline $baseline; check network access to github.com/microsoft/vcpkg"
        git -C "$VCPKG_ROOT" show "$baseline:versions/baseline.json" >/dev/null 2>&1 \
            || die "fetched $baseline but versions/baseline.json is still unreachable"
        log "baseline repaired"
    else
        log "vcpkg baseline $baseline resolves"
    fi
else
    log "warning: could not parse builtin-baseline out of vcpkg.json; skipping the reachability check"
fi

# ---------------------------------------------------------------- build it

jobs=${JOBS:-$(sysctl -n hw.ncpu)}
mkdir -p build
logfile="build/${preset}-build.log"

log "configuring preset $preset"
cmake --preset "$preset"

if (( ${#targets[@]} )); then
    log "building targets: ${targets[*]} (-j$jobs), logging to $logfile"
    cmake --build "build/$preset" -j "$jobs" --target "${targets[@]}" >"$logfile" 2>&1
else
    log "building default target (-j$jobs), logging to $logfile"
    cmake --build "build/$preset" -j "$jobs" >"$logfile" 2>&1
fi

log "build finished; tail of $logfile:"
tail -n 5 "$logfile" >&2
