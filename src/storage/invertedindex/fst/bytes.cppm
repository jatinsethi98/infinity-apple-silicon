// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

#include <cassert>
#include <cstring>

export module infinity_core:fst.bytes;

import :fst.writer;

import third_party;

namespace infinity {

// These helpers used to select between an unaligned load and a byte-shift fallback via
// HAVE_EFFICIENT_UNALIGNED_ACCESS, which was defined only when a probe of the Linux
// kernel config succeeded. That probe went with Linux support, and it is deliberately
// not replaced by an unconditional definition, for two measured reasons:
//
//   - It bought nothing. On clang 20 targeting arm64 at -O2, the byte-shift form, an
//     unaligned reinterpret_cast load, and memcpy all compile to the SAME single `ldr`.
//     The compiler recognises the idiom.
//   - For UnpackUint below it was actively WRONG: that fast path loads a full 8 bytes
//     and masks, overreading up to 7 bytes past the nbytes the caller asked for.
//
// memcpy is used here instead of a cast because it is well defined for unaligned and
// potentially-aliasing memory and produces that same single instruction. The
// __BIG_ENDIAN__ branches are gone: this fork is arm64 macOS only, which is
// little-endian, and the CMake guard rejects anything else.

/// Read a u32 in little endian format from the beginning of the given slice.
u32 ReadU32LE(const u8 *ptr) {
    u32 result;
    std::memcpy(&result, ptr, sizeof(result));
    return result;
}

/// Read a u64 in little endian format from the beginning of the given slice.
export u64 ReadU64LE(const u8 *ptr) {
    u64 result;
    std::memcpy(&result, ptr, sizeof(result));
    return result;
}

/// Write a u32 in little endian format to the beginning of the given ptr.
void WriteU32LE(u32 n, u8 *ptr) { std::memcpy(ptr, &n, sizeof(n)); }

/// Like WriteU32LE, but to an ostream implementation.
void IoWriteU32LE(u32 n, Writer &wtr) {
    wtr.Write((u8 *)&n, 4);
}

/// Write a u64 in little endian format to the beginning of the given ptr.
void WriteU64LE(u64 n, u8 *ptr) { std::memcpy(ptr, &n, sizeof(n)); }

/// Like WriteU64LE, but to an ostream implementation.
void IoWriteU64LE(u64 n, Writer &wtr) {
    wtr.Write((u8 *)&n, 8);
}

/// PackSize returns the smallest number of bytes that can encode `n`.
u8 PackSize(u64 n) {
    // __builtin_clzl is a bit slower than cascaded if:
    // return n==0 ? 1 : 8 - (__builtin_clzl(n) >> 3);
    if (n < 1ULL << 8) {
        return 1;
    } else if (n < 1ULL << 16) {
        return 2;
    } else if (n < 1ULL << 24) {
        return 3;
    } else if (n < 1ULL << 32) {
        return 4;
    } else if (n < 1ULL << 40) {
        return 5;
    } else if (n < 1ULL << 48) {
        return 6;
    } else if (n < 1ULL << 56) {
        return 7;
    } else {
        return 8;
    }
}

/// PackUintIn is like PackUint, but always uses the number of bytes given
/// to pack the number given.
///
/// `nbytes` must be >= pack_size(n) and <= 8, where `pack_size(n)` is the
/// smallest number of bytes that can store the integer given.
void PackUintIn(Writer &wtr, u64 n, u8 nbytes) {
    assert(nbytes >= 1 && nbytes <= 8);
    wtr.Write((u8 *)&n, nbytes);
}

/// PackUint packs the given integer in the smallest number of bytes possible,
/// and writes it to the given writer. The number of bytes written is returned
/// on success.
u8 PackUint(Writer &wtr, u64 n) {
    u8 nbytes = PackSize(n);
    PackUintIn(wtr, n, nbytes);
    return nbytes;
}

/// UnpackUint is the dual of PackUint. It unpacks the integer at the current
/// position in `ptr` after reading `nbytes` bytes.
///
/// `nbytes` must be >= 1 and <= 8.
u64 UnpackUint(u8 *ptr, u8 nbytes) {
    assert(nbytes >= 1 && nbytes <= 8);
    // Reads exactly nbytes. The previous alternative -- load 8 bytes and mask off the
    // unwanted high ones -- overreads by up to 7 bytes, which is a real out-of-bounds
    // access whenever the encoded value sits at the end of a buffer or mapping. Not a
    // performance trade: it was incorrect.
    u64 n = 0;
    std::memcpy(&n, ptr, nbytes);
    return n;
}

/// Compare two byte slice according to the lexicographically order
int CompareBytes(u8 *bs1_data, size_t bs1_len, u8 *bs2_data, size_t bs2_len) {
    size_t common_len = std::min(bs1_len, bs2_len);
    int ret = std::memcmp(bs1_data, bs2_data, common_len);
    if (ret != 0) {
        return ret;
    }
    if (bs1_len < bs2_len) {
        return -1;
    } else if (bs1_len > bs2_len) {
        return 1;
    } else {
        return 0;
    }
}

} // namespace infinity