#!/bin/sh
set -eu

SOURCE_NAME="Homebrew libomp 22.1.8 arm64 macOS 14 bottle"
SOURCE_SHA256=060c51de1382c489235cbf98b832ddd30f8e7c06dacf83aa59fd131927170631
HEADER_SHA256=5974470842520cea4bc50136e2329bbf4e36ba928d317e86f7def2ba1752d3d4
FORMULA_SHA256=c273f0bd2480778decb9bd5e02fbdc2e0423f77c03f92bb933cc797a97803699
OUTPUT_SHA256=d672d5e16383cc21b12e80b8fa050b5d07811eefa4835e2c8c0c56762cba00be

if [ "$#" -ne 2 ]; then
    echo "usage: $0 SOURCE_PREFIX OUTPUT_PREFIX" >&2
    exit 64
fi

source_prefix=$1
output_prefix=$2
source_library="$source_prefix/lib/libomp.dylib"
source_header="$source_prefix/include/omp.h"
source_formula="$source_prefix/.brew/libomp.rb"
output_library="$output_prefix/lib/libomp.dylib"
output_header="$output_prefix/include/omp.h"

hash_file() {
    /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

if [ ! -f "$source_library" ] || [ ! -f "$source_header" ] || [ ! -f "$source_formula" ]; then
    echo "source prefix does not contain the pinned libomp bottle files: $source_prefix" >&2
    exit 66
fi
if [ -e "$output_library" ] || [ -e "$output_header" ]; then
    echo "refusing to overwrite existing D0 libomp output: $output_prefix" >&2
    exit 73
fi
if [ "$(hash_file "$source_library")" != "$SOURCE_SHA256" ]; then
    echo "source libomp.dylib hash mismatch" >&2
    exit 65
fi
if [ "$(hash_file "$source_header")" != "$HEADER_SHA256" ]; then
    echo "source omp.h hash mismatch" >&2
    exit 65
fi
if [ "$(hash_file "$source_formula")" != "$FORMULA_SHA256" ]; then
    echo "source Homebrew formula hash mismatch" >&2
    exit 65
fi

/bin/mkdir -p "$output_prefix/lib" "$output_prefix/include"
/bin/cp "$source_library" "$output_library"
/bin/chmod u+w "$output_library"
/usr/bin/install_name_tool -id @rpath/libomp.dylib "$output_library"
/usr/bin/codesign --force --sign - "$output_library"
/bin/chmod a-w "$output_library"
/bin/cp "$source_header" "$output_header"
/bin/chmod a-w "$output_header"

if [ "$(hash_file "$output_library")" != "$OUTPUT_SHA256" ]; then
    echo "normalized libomp.dylib hash mismatch" >&2
    exit 65
fi
if [ "$(hash_file "$output_header")" != "$HEADER_SHA256" ]; then
    echo "copied omp.h hash mismatch" >&2
    exit 65
fi

echo "status=PASS"
echo "source=$SOURCE_NAME"
echo "source_libomp_sha256=$SOURCE_SHA256"
echo "source_formula_sha256=$FORMULA_SHA256"
echo "libomp_sha256=$OUTPUT_SHA256"
echo "omp_header_sha256=$HEADER_SHA256"
