// Copyright(C) 2025 InfiniFlow, Inc. All rights reserved.
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

#pragma once

namespace infinity {

// Deliberately defined inline in the header rather than in a .cpp.
//
// This lowers to a single arm64 `prfm pldl1keep, [x0]` (or the x86 equivalent). As an
// out-of-line function it compiled to a two-instruction body reached by `bl`, and the
// HNSW construction traversal calls it once per neighbour of every popped node
// (hnsw_alg.cppm SearchLayer prefetch loop, via DataStore::PrefetchVec ->
// PlainVecStoreInnerBase::Prefetch). On a 1M-vector build that is on the order of 10^10
// calls to issue one instruction, and the call also denies the compiler any freedom to
// schedule the prefetch relative to the loads it is meant to cover.
//
// `__builtin_prefetch` is a compiler builtin and needs no includes.
inline void SIMDPrefetch(const void *ptr) { __builtin_prefetch(ptr, 0 /* rw: read */, 3 /* locality: high */); }

// Prefetch an entire OBJECT, not just its first cache line.
//
// SIMDPrefetch above touches one 64-byte line. That is the right primitive for a pointer-sized
// or single-line target, and the wrong one for a vector: at d=128 float an embedding is 512
// bytes -- EIGHT lines -- so prefetching the head leaves 7/8 of it cold, and the kernel stalls
// on the remainder anyway.
//
// The cost of getting this wrong is large because HNSW's candidate reads are scattered. Measured
// on this M3 Pro over a 512 MB working set with random candidate indices, d=128, four candidates
// in flight: 64.8 ns per distance with no prefetch versus 30.1 ns with the whole vector
// prefetched -- 2.15x. The same kernel over a SEQUENTIAL sweep of the same data costs 9.2 ns, so
// roughly 90% of the scattered cost is exposed memory latency rather than arithmetic. That is
// also why fusing the multiply-add (a 1.5x reduction in FP ops) changed nothing measurable.
//
// `bytes` is rounded up to whole lines. The stride is only a hint's granularity, so getting it
// wrong costs a redundant or a missing prefetch, never correctness -- which is why it can be
// chosen per target rather than probed at runtime.
//
// Apple arm64 lines are 128 bytes, not 64: `sysctl hw.cachelinesize` reports 128 on M1 through
// M4 (verified on this Mac15,7 M3 Pro). An earlier revision of this comment asserted 64 "on every
// arm64 Apple core", and that was simply wrong. At a 64-byte stride a 512-byte embedding issued
// EIGHT `prfm` where four cover the same four physical lines, so half of them were redundant hits
// on a line fill already in flight.
//
// If some Apple arm64 core has 64-byte lines, a 128-byte stride there prefetches every other line.
// That is a throughput regression on such a core, not a correctness problem -- the demand loads are
// unaffected -- so the constant is chosen at compile time rather than paying a runtime query in a
// function called billions of times per build.
//
// The walk must start at the line the object BEGINS in, not at the object's own address. An object
// that straddles a line boundary otherwise loses its tail: a 512-byte vector starting 64 bytes into
// a 128-byte line spans FIVE lines, and stepping 0/128/256/384 from the unaligned base touches only
// four of them, leaving the last 64 bytes to stall. At d=128 the store happens to hand out
// 128-aligned vectors (the backing allocation is page-aligned and the stride is 512 bytes), so this
// was latent rather than active -- but it is wrong for any dimension whose vector size is not a
// multiple of the line, and it costs one AND to fix.
inline void SIMDPrefetchRange(const void *ptr, unsigned long bytes) {
#if defined(__APPLE__) && defined(__aarch64__)
    constexpr unsigned long kCacheLineBytes = 128;
#else
    constexpr unsigned long kCacheLineBytes = 64;
#endif
    const char *const begin = static_cast<const char *>(ptr);
    const char *const end = begin + bytes;
    const char *cursor =
        begin - (reinterpret_cast<unsigned long>(begin) & (kCacheLineBytes - 1));
    for (; cursor < end; cursor += kCacheLineBytes) {
        __builtin_prefetch(cursor, 0 /* rw: read */, 3 /* locality: high */);
    }
}

} // namespace infinity
