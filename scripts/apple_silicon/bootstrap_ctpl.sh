#!/usr/bin/env bash
# Fetch the ctpl_stl.h thread-pool header that Infinity's HNSW build uses and apply the
# repo's strong-push-guarantee patch, without needing a vcpkg checkout. Installs it where
# the native_hnsw_smoke `bench` preset already looks: vcpkg_installed/arm64-osx/include.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${1:-$ROOT/vcpkg_installed/arm64-osx/include}
TAG=ctpl_v.0.0.2
mkdir -p "$DEST"
if [ -f "$DEST/ctpl_stl.h" ] && grep -q push_has_strong_exception_guarantee "$DEST/ctpl_stl.h"; then
  echo "patched ctpl_stl.h already present at $DEST"; exit 0
fi
curl -fsSL -o "$DEST/ctpl_stl.h" "https://raw.githubusercontent.com/vit-vit/CTPL/$TAG/ctpl_stl.h"
( cd "$DEST" && patch -p1 --silent < "$ROOT/ports/vit-vit-ctpl/strong-push-guarantee.patch" )
grep -q push_has_strong_exception_guarantee "$DEST/ctpl_stl.h" && echo "installed patched ctpl_stl.h to $DEST"
