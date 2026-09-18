#!/usr/bin/env bash
#
# Check whether this Mac can build and run Infinity, without changing anything.
#
# One line per prerequisite: ok, WARN, MISSING or info, each with the command that
# fixes it. Exits 0 when every required check passes and 1 otherwise, so it can gate
# a script. It never installs, upgrades or writes anything; `setup.sh` is the script
# that does.
#
# Usage:
#   scripts/apple_silicon/doctor.sh [--bench]
#
# Options:
#   --bench     also check what the HNSW benchmark harness needs (faiss, simde,
#               the patched ctpl header, the SIFT1M dataset)
#   -h, --help  this message
#
set -uo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root" || exit 1

want_bench=0
for arg in "$@"; do
    case "$arg" in
        --bench) want_bench=1 ;;
        -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'error: unknown argument %s\n' "$arg" >&2; exit 2 ;;
    esac
done

failures=0
row()     { printf '  %-8s %-22s %s\n' "$1" "$2" "$3"; }
ok()      { row "ok"      "$1" "$2"; }
info()    { row "info"    "$1" "$2"; }
warn()    { row "WARN"    "$1" "$2"; }
missing() { row "MISSING" "$1" "$2"; failures=$((failures + 1)); }
section() { printf '\n%s\n' "$1"; }

# "4.4.3" -> 4004003, so versions compare as integers (bash 3.2 has no other way).
vernum() { printf '%s' "$1" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}'; }

brew_prefix=$(brew --prefix 2>/dev/null || printf '/opt/homebrew')

# ------------------------------------------------------------------- host

section "Host"

if [[ $(uname -s) != Darwin ]]; then
    missing "macOS" "this fork builds on macOS only; for $(uname -s) use upstream infiniflow/infinity"
else
    macos=$(sw_vers -productVersion)
    if (( ${macos%%.*} >= 14 )); then
        ok "macOS $macos" "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    else
        missing "macOS $macos" "macOS 14 or newer is required"
    fi
fi

arch=$(uname -m)
translated=$(sysctl -n sysctl.proc_translated 2>/dev/null || printf 0)
if [[ $arch == arm64 ]]; then
    ok "arm64 shell" "native"
elif [[ $translated == 1 ]]; then
    missing "arm64 shell" "this shell runs under Rosetta; start a native one:  arch -arm64 zsh"
else
    missing "arm64 shell" "uname -m reports $arch; an Apple Silicon Mac is required"
fi

mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || printf 0) / 1073741824 ))
if (( mem_gb >= 16 )); then
    ok "memory" "${mem_gb} GB"
else
    warn "memory" "${mem_gb} GB; the full build and a 1M-vector index are comfortable at 16 GB and up"
fi

free_gb=$(df -g . | awk 'NR==2 {print $4}')
if (( free_gb >= 20 )); then
    ok "disk" "${free_gb} GB free on this volume"
else
    warn "disk" "${free_gb} GB free; the full server build needs about 20 GB, the benchmark harness about 2 GB"
fi

# -------------------------------------------------------------- toolchain

section "Toolchain"

if sdk=$(xcrun --show-sdk-path 2>/dev/null); then
    ok "Xcode CLT" "$sdk"
else
    missing "Xcode CLT" "xcode-select --install"
fi

if command -v brew >/dev/null 2>&1; then
    ok "Homebrew" "$brew_prefix"
else
    missing "Homebrew" "install it from https://brew.sh"
fi

llvm_prefix=${LLVM_PREFIX:-$brew_prefix/opt/llvm@20}
if [[ -x $llvm_prefix/bin/clang++ ]]; then
    ok "clang 20" "$("$llvm_prefix/bin/clang++" --version | head -1)"
else
    missing "clang 20" "brew install llvm@20   (Apple Clang cannot build this: it needs C++23 modules)"
fi

if command -v cmake >/dev/null 2>&1; then
    cmake_version=$(cmake --version | head -1 | awk '{print $3}')
    n=$(vernum "$cmake_version")
    if (( n < 4000003 )); then
        missing "cmake $cmake_version" "4.0.3 or newer is required:  brew upgrade cmake"
    elif (( n >= 4005000 )); then
        missing "cmake $cmake_version" "too new; 4.0.3 up to 4.4.x is supported:  brew unlink cmake && brew install cmake@4.4"
    else
        ok "cmake $cmake_version" "within the supported range (4.0.3 to 4.4.x)"
    fi
else
    missing "cmake" "brew install cmake"
fi

if command -v ninja >/dev/null 2>&1; then
    ok "ninja" "$(ninja --version)"
else
    missing "ninja" "brew install ninja"
fi

if [[ -d $brew_prefix/opt/libomp ]]; then
    ok "libomp" "$brew_prefix/opt/libomp"
else
    missing "libomp" "brew install libomp"
fi

if command -v pkg-config >/dev/null 2>&1; then
    ok "pkg-config" "$(pkg-config --version) ($(command -v pkg-config))"
else
    missing "pkg-config" "brew install pkg-config   (vcpkg's ports need it; the dependency build fails on abseil without it)"
fi

# vcpkg's thrift port needs bison 3.7+ (it passes --file-prefix-map). macOS ships
# 2.3 and vcpkg only warns before using it. vcpkg searches the two Homebrew prefixes
# before PATH, so look in the same places in the same order.
bison_bin=""
for candidate in "$brew_prefix/opt/bison/bin/bison" /usr/local/opt/bison/bin/bison "$(command -v bison 2>/dev/null || true)"; do
    if [[ -n $candidate && -x $candidate ]]; then bison_bin=$candidate; break; fi
done
if [[ -z $bison_bin ]]; then
    missing "bison" "brew install bison   (the thrift dependency needs 3.7+)"
else
    bison_version=$("$bison_bin" --version | head -1 | awk '{print $NF}')
    if (( $(vernum "$bison_version") >= 3007000 )); then
        ok "bison $bison_version" "$bison_bin"
    else
        missing "bison $bison_version" "too old for the thrift dependency (needs 3.7+):  brew install bison"
    fi
fi

if command -v uv >/dev/null 2>&1; then
    ok "uv" "$(uv --version | head -1)"
else
    missing "uv" "brew install uv   (runs the demo, the SDK and the Python test drivers)"
fi

# ------------------------------------------------------------- repository

section "Repository"

if [[ -f resource/jieba/dict/jieba.dict.utf8 ]]; then
    ok "resource submodule" "analyzer dictionaries present"
else
    warn "resource submodule" "not fetched; setup.sh does it, or:  git submodule update --init --recursive resource"
fi

vcpkg_root=${VCPKG_ROOT:-}
if [[ -z $vcpkg_root ]]; then
    for candidate in "$repo_root/../vcpkg-root" "$repo_root/../vcpkg" "$HOME/vcpkg" /opt/vcpkg; do
        if [[ -f "$candidate/.vcpkg-root" ]]; then vcpkg_root=$(cd -- "$candidate" && pwd); break; fi
    done
fi
if [[ -n $vcpkg_root && -x $vcpkg_root/vcpkg ]]; then
    ok "vcpkg" "$vcpkg_root"
elif [[ -n $vcpkg_root ]]; then
    warn "vcpkg" "$vcpkg_root exists but is not bootstrapped; setup.sh will do it"
else
    info "vcpkg" "not found yet; setup.sh clones it next to this repository (../vcpkg-root)"
fi

server=build/macos-arm64-release/src/infinity
if [[ -x $server ]]; then
    ok "server binary" "$server ($(lipo -archs "$server" 2>/dev/null || printf '?'))"
else
    info "server binary" "not built yet:  make setup   (10 to 15 minutes on an M4)"
fi

if [[ -x build/macos-arm64-release/src/test_main ]]; then
    ok "unit test binary" "build/macos-arm64-release/src/test_main"
else
    info "unit test binary" "not built:  make build-tests"
fi

if lsof -nP -iTCP:23817 -sTCP:LISTEN >/dev/null 2>&1; then
    info "server" "something is listening on 23817; see:  make status"
else
    info "server" "nothing listening on 23817; start one with:  make start"
fi

# ------------------------------------------------------- benchmark harness

if (( want_bench )); then
    section "Benchmark harness (--bench)"
    for formula in faiss simde; do
        if [[ -d $brew_prefix/opt/$formula ]]; then
            ok "$formula" "$brew_prefix/opt/$formula"
        else
            missing "$formula" "brew install $formula"
        fi
    done
    if [[ -f vcpkg_installed/arm64-osx/include/ctpl_stl.h ]]; then
        ok "ctpl header" "vcpkg_installed/arm64-osx/include/ctpl_stl.h"
    else
        warn "ctpl header" "scripts/apple_silicon/bootstrap_ctpl.sh   (make bench-build does this)"
    fi
    if [[ -x build/bench/infinity_hnsw_d0 ]]; then
        ok "harness binary" "build/bench/infinity_hnsw_d0"
    else
        info "harness binary" "not built:  make bench-build"
    fi
    if [[ -x build/bench-faiss-src/faiss_hnsw_d0 ]]; then
        ok "Accelerate FAISS" "build/bench-faiss-src/faiss_hnsw_d0"
    else
        info "Accelerate FAISS" "not built; the Homebrew FAISS is 1.5x slower and flatters Infinity:  make bench-faiss"
    fi
    if [[ -f datasets/sift1m/base.f32 ]]; then
        ok "SIFT1M dataset" "datasets/sift1m"
    else
        info "SIFT1M dataset" "not downloaded:  make bench-datasets   (168 MB)"
    fi
fi

# ---------------------------------------------------------------- summary

printf '\n'
if (( failures )); then
    printf '%d problem(s). Fix the MISSING rows above and re-run:  scripts/apple_silicon/doctor.sh\n' "$failures"
    exit 1
fi
printf 'Ready. Next:  make setup    (installs what is left and builds the server)\n'
exit 0
