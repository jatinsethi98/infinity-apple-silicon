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

} // namespace infinity
