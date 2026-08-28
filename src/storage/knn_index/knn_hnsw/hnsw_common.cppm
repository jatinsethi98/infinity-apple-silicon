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

export module infinity_core:hnsw_common;

import :infinity_exception;
import :sparse_util;
import :default_values;

import std;
import std.compat;

namespace infinity {

export struct HnswConfig {
    size_t lvq_buffer_size_;
};

export constexpr size_t AlignTo(size_t a, size_t b) { return (a + b - 1) / b * b; }

export using MeanType = double;
export using VertexType = i32;
export using VertexListSize = i32;
export using LayerSize = i32;

export constexpr VertexType kInvalidVertex = -1;
export constexpr LayerSize kHnswMaxSupportedLayer = 64;

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)

export struct HnswIncrementalReciprocalExecutionEvidence {
    bool treatment_compiled{};
    bool capture_armed{};
    bool eligible_branch_entered{};
    bool successful_unchanged_observed{};
    bool successful_updated_observed{};
};

#endif

#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)

export struct HnswThresholdBatch4ExecutionEvidence {
    bool treatment_compiled{};
    bool capture_armed{};
    bool eligible_branch_entered{};
    bool rejected_lane_observed{};
    // A surviving lane escaped early rejection and was therefore fully
    // evaluated; it need not pass the snapshot threshold or enter the live heap.
    bool surviving_lane_observed{};
};

#endif

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)

export constexpr size_t kHnswIncrementalReciprocalScratchCapacity = 256;

export enum class HnswIncrementalReciprocalResult {
    kFallbackInvalidState,
    kFallbackScratchCapacity,
    kFallbackDuplicate,
    kFallbackCenterTie,
    kFallbackNonFiniteCenter,
    kFallbackUncertifiedOrder,
    kUnchangedNewFarthest,
    kUnchangedRejected,
    kUpdated,
};

export struct HnswIncrementalReciprocalStats {
    std::uint64_t reciprocal_links{};
    std::uint64_t direct_appends{};
    std::uint64_t full_overflows{};
    std::uint64_t certificate_hits{};
    std::uint64_t certificate_misses{};
    std::uint64_t certificate_sets{};
    std::uint64_t certificate_clears{};
    std::uint64_t unchanged_new_farthest{};
    std::uint64_t unchanged_rejected{};
    std::uint64_t updated_full{};
    std::uint64_t updated_underfull{};
    std::uint64_t fallback_invalid_state{};
    std::uint64_t fallback_scratch_capacity{};
    std::uint64_t fallback_duplicate{};
    std::uint64_t fallback_center_tie{};
    std::uint64_t fallback_nonfinite_center{};
    std::uint64_t fallback_uncertified_order{};
    std::uint64_t baseline_center_distance_evaluations{};
    std::uint64_t baseline_pair_distance_evaluations{};
    std::uint64_t incremental_center_distance_evaluations{};
    std::uint64_t incremental_pair_distance_evaluations{};
    std::uint64_t shadow_comparisons{};
    std::uint64_t shadow_mismatches{};
    std::uint64_t predicted_avoided_distance_evaluations{};
    std::uint64_t predicted_extra_distance_evaluations{};
};

// Applies one scalar diversity-heuristic update to a certified full list
// stored in farthest-to-nearest center-distance order. Every fallback leaves
// the input list unchanged.
export template <typename DistanceType, typename CenterDistance, typename NewToOldDistance, typename OldToNewDistance>
HnswIncrementalReciprocalResult TryIncrementalReciprocalUpdate(VertexType new_vertex,
                                                               DistanceType new_center_distance,
                                                               VertexType *neighbors,
                                                               VertexListSize *neighbor_count,
                                                               size_t capacity,
                                                               std::span<DistanceType> center_distance_scratch,
                                                               CenterDistance &&center_distance,
                                                               NewToOldDistance &&new_to_old_distance,
                                                               OldToNewDistance &&old_to_new_distance) {
    auto is_finite = [](const auto &value) {
        using Value = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_floating_point_v<Value>) {
            return std::isfinite(value);
        }
        return true;
    };

    if (neighbors == nullptr || neighbor_count == nullptr || new_vertex < 0 || capacity == 0 || *neighbor_count < 0 ||
        static_cast<size_t>(*neighbor_count) != capacity) {
        return HnswIncrementalReciprocalResult::kFallbackInvalidState;
    }
    if (center_distance_scratch.size() < capacity) {
        return HnswIncrementalReciprocalResult::kFallbackScratchCapacity;
    }
    if (capacity > kHnswIncrementalReciprocalScratchCapacity) {
        return HnswIncrementalReciprocalResult::kFallbackScratchCapacity;
    }
    if (!is_finite(new_center_distance)) {
        return HnswIncrementalReciprocalResult::kFallbackNonFiniteCenter;
    }

    for (size_t index = 0; index < capacity; ++index) {
        if (neighbors[index] == new_vertex) {
            return HnswIncrementalReciprocalResult::kFallbackDuplicate;
        }
        const DistanceType old_distance = center_distance(neighbors[index]);
        if (!is_finite(old_distance)) {
            return HnswIncrementalReciprocalResult::kFallbackNonFiniteCenter;
        }
        center_distance_scratch[index] = old_distance;
        if (old_distance == new_center_distance) {
            return HnswIncrementalReciprocalResult::kFallbackCenterTie;
        }
        if (index != 0 && !(center_distance_scratch[index - 1] > old_distance)) {
            return HnswIncrementalReciprocalResult::kFallbackUncertifiedOrder;
        }
    }

    size_t farther_count = 0;
    while (farther_count < capacity && center_distance_scratch[farther_count] > new_center_distance) {
        ++farther_count;
    }

    // The full old list reaches capacity before a new farthest candidate is
    // visited by the reference heuristic.
    if (farther_count == 0) {
        return HnswIncrementalReciprocalResult::kUnchangedNewFarthest;
    }

    // Old entries closer to the center are the accepted predecessor set for
    // the new candidate, in nearest-to-farthest processing order.
    for (size_t index = capacity; index > farther_count; --index) {
        if (new_to_old_distance(neighbors[index - 1]) < new_center_distance) {
            return HnswIncrementalReciprocalResult::kUnchangedRejected;
        }
    }

    std::array<VertexType, kHnswIncrementalReciprocalScratchCapacity> updated_neighbors;
    size_t accepted_count = capacity - farther_count + 1;
    size_t write = farther_count;
    for (size_t read = farther_count; read > 0 && accepted_count < capacity;) {
        --read;
        const VertexType old_vertex = neighbors[read];
        if (!(old_to_new_distance(old_vertex) < center_distance_scratch[read])) {
            updated_neighbors[--write] = old_vertex;
            ++accepted_count;
        }
    }

    const size_t accepted_farther_count = farther_count - write;
    const size_t closer_count = capacity - farther_count;
    std::memmove(updated_neighbors.data(), updated_neighbors.data() + write, accepted_farther_count * sizeof(VertexType));
    updated_neighbors[accepted_farther_count] = new_vertex;
    std::copy_n(neighbors + farther_count, closer_count, updated_neighbors.data() + accepted_farther_count + 1);
    std::copy_n(updated_neighbors.data(), accepted_count, neighbors);
    *neighbor_count = static_cast<VertexListSize>(accepted_count);
    return HnswIncrementalReciprocalResult::kUpdated;
}

#endif

export template <typename Iterator, typename RtnType, typename LabelType>
concept DataIteratorConcept = requires(Iterator iter) {
    typename std::decay_t<Iterator>::ValueType;
    { iter.Next() } -> std::same_as<std::optional<std::pair<RtnType, LabelType>>>;
    // HnswIndexInMem::InsertVecs<Iter> needs row count which might be different with the vector count,
    // for example MemIndexInserterIter1<MultiVectorRef<ElementT>>.
    { iter.GetRowCount() } -> std::same_as<size_t>;
};

export template <typename DataType, typename LabelType>
class DenseVectorIter {
    const DataType *ptr_;
    const size_t dim_;
    size_t remaining_;
    LabelType label_;

public:
    using This = DenseVectorIter<DataType, LabelType>;
    using Split = std::vector<This>;
    using ValueType = const DataType *;

    DenseVectorIter(ValueType ptr, size_t dim, size_t vec_num, LabelType offset = 0) : ptr_(ptr), dim_(dim), remaining_(vec_num), label_(offset) {}

    std::optional<std::pair<ValueType, LabelType>> Next() {
        if (remaining_ == 0) {
            return std::nullopt;
        }
        auto ret = ptr_;
        ptr_ += dim_;
        --remaining_;
        return std::make_pair(ret, label_++);
    }

    size_t GetRowCount() const { return remaining_; }

    Split split() && {
        Split res;
        size_t vec_num = 0;
        ValueType head = ptr_;
        while (remaining_ != 0) {
            if (vec_num == DEFAULT_ITER_BATCH_SIZE) {
                res.emplace_back(head, dim_, DEFAULT_ITER_BATCH_SIZE, label_ + res.size() * DEFAULT_ITER_BATCH_SIZE);
                vec_num = 0;
                head = ptr_;
            }
            ptr_ += dim_;
            --remaining_;
            ++vec_num;
        }
        res.emplace_back(head, dim_, vec_num, label_ + res.size() * DEFAULT_ITER_BATCH_SIZE);
        return res;
    }
};

export template <typename DataType, typename IdxType, typename LabelType>
class SparseVectorIter {
    const i64 *indptr_;
    const IdxType *indice_;
    const DataType *data_;
    const i64 *indptr_end_;
    LabelType label_;

public:
    using ValueType = SparseVecRef<DataType, IdxType>;

    SparseVectorIter(const i64 *indptr, const IdxType *indice, const DataType *data, i32 vec_num, LabelType offset = 0)
        : indptr_(indptr), indice_(indice), data_(data), indptr_end_(indptr_ + vec_num + 1), label_(offset) {}

    std::optional<std::pair<SparseVecRef<DataType, IdxType>, LabelType>> Next() {
        if (indptr_ + 1 == indptr_end_) {
            return std::nullopt;
        }
        i64 nnz = indptr_[1] - indptr_[0];
        const IdxType *indice = indice_ + indptr_[0];
        const DataType *data = data_ + indptr_[0];
        ++indptr_;
        return std::make_pair(SparseVecRef<DataType, IdxType>(nnz, indice, data), label_++);
    }

    size_t GetRowCount() const { return indptr_end_ - indptr_ - 1; }
};

export template <typename LabelType>
class FilterBase {
public:
    virtual bool operator()(const LabelType &vertex_i) const = 0;
};

export template <typename Filter, typename LabelType>
concept FilterConcept = requires(LabelType label) { std::is_same_v<Filter, std::nullopt_t> || std::is_base_of_v<FilterBase<LabelType>, Filter>; };

export struct HnswInsertConfig {
    bool optimize_;
};

export constexpr HnswInsertConfig kDefaultHnswInsertConfig = {
    .optimize_ = false,
};

export template <typename Iter>
concept SplitIter = requires(Iter iter) { typename Iter::Split; };

} // namespace infinity
