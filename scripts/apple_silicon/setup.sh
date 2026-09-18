#!/usr/bin/env bash
#
# One command from a fresh clone to a running Infinity server on Apple Silicon.
#
# This exists because getting there by hand is ten steps across four tools, and
# every one of them fails in a way that does not name the real cause: a missing
# llvm@20 shows up as a C++23 module error, a CMake older than 4.0.3 configures
# fine and then dies on `import std`, a shallow vcpkg clone reports a missing
# spdlog rather than a missing baseline, and a missing `resource` submodule
# surfaces as a full-text analyzer test failure.
#
# What it does, in order, skipping anything already done:
#   1. checks the host is arm64 macOS with the Xcode command line tools
#   2. installs the Homebrew toolchain (llvm@20, cmake, ninja, libomp, bison, pkg-config)
#   3. initialises the `resource` submodule (full-text analyzer dictionaries)
#   4. clones and bootstraps vcpkg at the baseline pinned in vcpkg.json
#   5. hands off to build_server.sh, which owns the configure and build
#
# It does NOT modify your shell profile. The environment it needs is exported for
# the duration of the run, and the two lines you would want to persist are
# printed at the end.
#
# Usage:
#   scripts/apple_silicon/setup.sh [options]
#
# Options:
#   --no-build          set up prerequisites only; do not compile
#   --skip-brew         do not run Homebrew; only verify the tools are present
#   --vcpkg-root DIR    where to keep the vcpkg checkout
#                       (default: ../vcpkg-root, next to this repository)
#   --with-tests        also build test_main and install sqllogictest
#   -h, --help          this message
#
# The build takes about 12 minutes on an M4 Mac mini (six for the vcpkg dependencies,
# three for about 1500 C++23 module translation units, the rest for the unit tests),
# longer on older chips, and needs about 20 GB of free disk.
#
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

do_build=1
skip_brew=0
with_tests=0
vcpkg_root=${VCPKG_ROOT:-}

while (( $# )); do
    case "$1" in
        --no-build) do_build=0 ;;
        --skip-brew) skip_brew=1 ;;
        --with-tests) with_tests=1 ;;
        --vcpkg-root) vcpkg_root=${2:?--vcpkg-root needs a directory}; shift ;;
        -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'error: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

step() { printf '\n==> %s\n' "$*" >&2; }
info() { printf '    %s\n' "$*" >&2; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------- 1. host checks

step "Checking the host"

[[ $(uname -s) == Darwin ]] \
    || die "this is the Apple-Silicon-only fork of Infinity; it does not build on $(uname -s).
       For Linux or x86-64, use upstream: https://github.com/infiniflow/infinity"
[[ $(uname -m) == arm64 ]] \
    || die "uname -m reports $(uname -m), not arm64.
       If you are on an Apple Silicon Mac inside a Rosetta shell, start a native one:
         arch -arm64 zsh"

xcrun --show-sdk-path >/dev/null 2>&1 \
    || die "the Xcode command line tools are missing. Install them with:
         xcode-select --install"
export SDKROOT=${SDKROOT:-$(xcrun --show-sdk-path)}

info "macOS $(sw_vers -productVersion) on $(sysctl -n machdep.cpu.brand_string)"
info "SDKROOT=$SDKROOT"

# Measured on an M4 mini: 11 GB of build tree, 2 GB of vcpkg checkout and build trees,
# and a few GB for the dictionary submodule, instances and the package; 19 GB in all.
free_gb=$(df -g . | awk 'NR==2 {print $4}')
if (( free_gb < 20 )); then
    info "warning: only ${free_gb} GB free on this volume; the full build needs about 20 GB"
fi

# --------------------------------------------------------- 2. Homebrew toolchain

llvm_prefix=${LLVM_PREFIX:-/opt/homebrew/opt/llvm@20}

if (( skip_brew )); then
    step "Skipping Homebrew (--skip-brew); verifying the toolchain instead"
else
    step "Installing the Homebrew toolchain"
    command -v brew >/dev/null \
        || die "Homebrew is not installed. Install it from https://brew.sh, then re-run this script.
       Or pass --skip-brew if you are providing clang 20, cmake 4.0.3+, ninja and libomp yourself."
    # llvm@20 specifically: the project hard-fails below clang 20 and needs a libc++
    # whose module manifest matches the compiler. Apple Clang cannot build this.
    # bison: vcpkg's thrift port passes --file-prefix-map, which needs bison 3.7+.
    # macOS ships 2.3 at /usr/bin/bison and vcpkg only warns before using it, so a
    # machine without Homebrew bison fails minutes into the dependency build with
    # "unrecognized option". vcpkg looks in /opt/homebrew/opt/bison/bin on its own.
    # pkg-config: vcpkg's ports call vcpkg_fixup_pkgconfig, which needs a pkg-config
    # binary and fails on the very first dependency (abseil) without one. Homebrew's
    # formula is pkgconf; pkg-config is its alias, and `brew list pkg-config` resolves it.
    for formula in llvm@20 ninja libomp bison pkg-config; do
        if brew list --versions "$formula" >/dev/null 2>&1; then
            info "$formula already installed"
        else
            info "installing $formula"
            brew install "$formula"
        fi
    done
    # Deliberately NOT `brew upgrade cmake`. The CMakeLists pins an `import std`
    # feature UUID per CMake version and hard-errors outside 4.0.3 ... 4.4.x, so a
    # blind upgrade to a future 4.5 would break a working install. Only install
    # when cmake is absent; the version range is checked below either way.
    if brew list --versions cmake >/dev/null 2>&1; then
        info "cmake already installed; leaving the version alone"
    else
        info "installing cmake"
        brew install cmake
    fi
fi

[[ -x "$llvm_prefix/bin/clang++" ]] \
    || die "no clang++ at $llvm_prefix/bin.
       Install it with: brew install llvm@20
       Or point LLVM_PREFIX at your own clang 20 install."
command -v cmake >/dev/null || die "cmake not found; brew install cmake"
command -v ninja >/dev/null || die "ninja not found; brew install ninja"
command -v pkg-config >/dev/null || die "pkg-config not found; the vcpkg dependency build needs it:
         brew install pkg-config"

# vcpkg searches /opt/homebrew/opt/bison/bin and /usr/local/opt/bison/bin before PATH,
# so check the same places it will, in the same order.
bison_bin=""
for candidate in /opt/homebrew/opt/bison/bin/bison /usr/local/opt/bison/bin/bison "$(command -v bison || true)"; do
    if [[ -n $candidate && -x $candidate ]]; then bison_bin=$candidate; break; fi
done
[[ -n $bison_bin ]] || die "bison not found. The thrift dependency needs bison 3.7 or newer:
         brew install bison"
bison_version=$("$bison_bin" --version | head -1 | awk '{print $NF}')
bison_sortable=$(printf '%s' "$bison_version" | awk -F. '{printf "%d%03d", $1, $2}')
(( bison_sortable >= 3007 )) \
    || die "bison $bison_version at $bison_bin is too old; the thrift dependency needs 3.7 or newer.
       macOS ships 2.3. Install a current one with: brew install bison"

# CMakeLists.txt selects a CMAKE_EXPERIMENTAL_CXX_IMPORT_STD UUID per CMake
# version and FATAL_ERRORs outside 4.0.3 ... 4.4.x. Check BOTH ends here: too old
# configures cleanly and then dies on `import std`, and too new fails at the
# CMakeLists guard with a message that does not suggest a fix.
cmake_version=$(cmake --version | head -1 | awk '{print $3}')
cmake_sortable=$(printf '%s' "$cmake_version" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')
(( cmake_sortable >= 4000003 )) \
    || die "cmake $cmake_version is too old; this build needs 4.0.3 or newer for its
       \`import std\` support. Upgrade with: brew upgrade cmake"
if (( cmake_sortable >= 4005000 )); then
    die "cmake $cmake_version is newer than this build supports (4.0.3 up to 4.4.x).
       CMake gates \`import std\` behind a version-specific UUID, so a new release
       has to be adopted deliberately rather than picked up automatically.
       Install a supported one alongside it, for example:
         brew unlink cmake && brew install cmake@4.4
       or point this script at your own via PATH."
fi

info "clang: $("$llvm_prefix/bin/clang++" --version | head -1)"
info "cmake: $cmake_version"
info "ninja: $(ninja --version)"
info "bison: $bison_version ($bison_bin)"

# ------------------------------------------------------- 3. resource submodule

step "Checking the resource submodule"

# The full-text analyzers (jieba, ik, rag, mecab, opencc) load their dictionaries
# from resource/. Without it the server starts but CJK full-text search is broken,
# and a package built from this tree would ship that breakage.
sentinel=resource/jieba/dict/jieba.dict.utf8
if [[ -f $sentinel ]]; then
    info "already present"
else
    if [[ ! -d .git ]] && [[ ! -f .git ]]; then
        die "resource/ is missing and this is not a git checkout, so it cannot be fetched.
       Download the dictionaries from https://github.com/infiniflow/resource"
    fi
    info "fetching analyzer dictionaries (about 500 MB)"
    git submodule update --init --recursive resource
    [[ -f $sentinel ]] || die "submodule checkout finished but $sentinel is still missing"
fi

# -------------------------------------------------------------- 4. vcpkg

step "Checking vcpkg"

baseline=$(sed -n 's/.*"builtin-baseline"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' vcpkg.json | head -1)
[[ -n $baseline ]] || die "could not read builtin-baseline out of vcpkg.json"

if [[ -z $vcpkg_root ]]; then
    # Same search order build_server.sh uses, so a checkout it would have found is
    # reused rather than duplicated.
    for candidate in "$repo_root/../vcpkg-root" "$repo_root/../vcpkg" "$HOME/vcpkg" /opt/vcpkg; do
        if [[ -f "$candidate/.vcpkg-root" ]]; then
            vcpkg_root=$(cd -- "$candidate" && pwd)
            info "found an existing checkout at $vcpkg_root"
            break
        fi
    done
fi
# Deliberately outside the repository: build*/ is gitignored but a 2.5 GB checkout
# inside the tree slows down every git and every editor index.
: "${vcpkg_root:=$(cd -- "$repo_root/.." && pwd)/vcpkg-root}"

if [[ ! -f "$vcpkg_root/.vcpkg-root" ]]; then
    info "cloning vcpkg into $vcpkg_root"
    # A blobless partial clone is a fraction of the size of a full one and still
    # lets git resolve the baseline commit's tree, which a --depth=1 clone cannot.
    git clone --filter=blob:none --no-checkout https://github.com/microsoft/vcpkg.git "$vcpkg_root"
    git -C "$vcpkg_root" fetch --depth=1 origin "$baseline"
    git -C "$vcpkg_root" checkout FETCH_HEAD
fi

if [[ ! -x "$vcpkg_root/vcpkg" ]]; then
    info "bootstrapping vcpkg"
    "$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics
fi

# Manifest mode resolves every port version through the baseline commit's
# versions/baseline.json. In a shallow clone that commit is a graft with no tree,
# and the resulting error names spdlog or thrift rather than the clone depth.
if ! git -C "$vcpkg_root" show "$baseline:versions/baseline.json" >/dev/null 2>&1; then
    info "baseline $baseline is not resolvable here; fetching that commit"
    git -C "$vcpkg_root" fetch --depth=1 origin "$baseline"
    git -C "$vcpkg_root" show "$baseline:versions/baseline.json" >/dev/null 2>&1 \
        || die "fetched $baseline but versions/baseline.json is still unreachable.
       Check network access to github.com/microsoft/vcpkg."
fi

export VCPKG_ROOT=$vcpkg_root
info "VCPKG_ROOT=$VCPKG_ROOT (baseline $baseline resolves)"

# ------------------------------------------------------------------ 5. build

if (( ! do_build )); then
    step "Prerequisites are ready (--no-build, stopping here)"
    printf '\nTo build, run:\n\n  export VCPKG_ROOT=%s\n  scripts/apple_silicon/build_server.sh macos-arm64-release infinity\n\n' "$VCPKG_ROOT" >&2
    exit 0
fi

targets=(infinity)
(( with_tests )) && targets+=(test_main)

step "Building ${targets[*]} (about 12 minutes on an M4; longer on older chips)"
info "progress is written to build/macos-arm64-release-build.log"
scripts/apple_silicon/build_server.sh macos-arm64-release "${targets[@]}"

if (( with_tests )); then
    step "Installing sqllogictest"
    scripts/apple_silicon/install_test_tools.sh
fi

# ------------------------------------------------------------------- summary

cat >&2 <<EOF

$(printf '=%.0s' {1..70})
Done. The server binary is at:

  build/macos-arm64-release/src/infinity

Start it, and run the demo:

  scripts/apple_silicon/run_server.sh start
  uv run --with fastembed demo/hybrid_search_demo.py

To build again in a new shell, export this first (or add it to your profile):

  export VCPKG_ROOT=$VCPKG_ROOT
EOF

if (( with_tests )); then
    cat >&2 <<'EOF'

To run the test suites:

  ./build/macos-arm64-release/src/test_main
  scripts/apple_silicon/run_server.sh start --instance ci --fresh
  uv run python scripts/apple_silicon/run_slt.py
EOF
fi
