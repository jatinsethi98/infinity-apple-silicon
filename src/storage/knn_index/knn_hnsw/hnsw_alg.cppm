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

export module infinity_core:hnsw_alg;

import :local_file_handle;
import :infinity_exception;
import :knn_result_handler;
import :multivector_result_handler;
import :hnsw_common;
import :data_store;
import :data_store_util;
import :dist_func_lsg_wrapper;
import :plain_vec_store;
import :default_values;
import :utility;
import :hnsw_lsg_builder;
import :index_hnsw;
#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
import :hnsw_lvq_capture;
#endif

import std;
import third_party;

import logical_type;
import serialize;
import column_def;

// Fixme: some variable has implicit type conversion.
// Fixme: some variable has confusing name.
// Fixme: has no test for different `DataType`.

// Todo: make more embedding type.
// Todo: make module partition.

namespace infinity {

struct HnswCompressionTargetTag {};

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
template <typename VecStoreType, typename = void>
struct HnswPlainDenseVecStore : std::false_type {};

template <typename VecStoreType>
struct HnswPlainDenseVecStore<VecStoreType, std::void_t<decltype(VecStoreType::IsPlainDense)>> : std::bool_constant<VecStoreType::IsPlainDense> {};
#endif

export struct KnnSearchOption {
    size_t ef_ = 0;
    LogicalType column_logical_type_ = LogicalType::kEmbedding;
};

template <typename Distance, typename DataStore>
concept FourCandidateDistance = requires(const Distance &distance,
                                         const typename Distance::QueryType &query,
                                         const std::array<VertexType, 4> &vertices,
                                         const DataStore &data_store,
                                         std::array<typename Distance::DistanceType, 4> &distances) {
    { distance.SupportsBatch4() } noexcept -> std::same_as<bool>;
    { distance.Batch4(query, vertices, data_store, distances) } noexcept -> std::same_as<void>;
};

template <typename Distance, typename DataStore>
concept FourCandidateThresholdDistance =
    FourCandidateDistance<Distance, DataStore> && requires(const Distance &distance,
                                                           const typename Distance::QueryType &query,
                                                           const std::array<VertexType, 4> &vertices,
                                                           const DataStore &data_store,
                                                           typename Distance::DistanceType threshold,
                                                           std::array<typename Distance::DistanceType, 4> &distances) {
        { distance.SupportsBatch4WithinThreshold() } noexcept -> std::same_as<bool>;
        { distance.Batch4WithinThreshold(query, vertices, data_store, threshold, distances) } noexcept -> std::same_as<std::uint8_t>;
    };

export template <typename VecStoreType, typename LabelType, bool OwnMem>
class KnnHnswBase {
public:
    using This = KnnHnswBase<VecStoreType, LabelType, OwnMem>;
    using DataType = typename VecStoreType::DataType;
    using QueryVecType = typename VecStoreType::QueryVecType;
    using QueryType = typename VecStoreType::QueryType;
    using DataStore = DataStore<VecStoreType, LabelType, OwnMem>;
    using Distance = typename VecStoreType::Distance;
    using DistanceType = typename Distance::DistanceType;

    using PDV = std::pair<DistanceType, VertexType>;
    using CMP = CompareByFirst<DistanceType, VertexType>;
    using CMPReverse = CompareByFirstReverse<DistanceType, VertexType>;
    using DistHeap = std::priority_queue<PDV, std::vector<PDV>, CMP>;

    constexpr static bool LSG = IsLSGDistance<Distance>;

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    struct IncrementalReciprocalSelectionMetadata {
        bool heuristic_branch{};
        bool used_batch4{};
        bool finite_strict_order{};
        bool full{};
        std::uint64_t pair_distance_evaluations{};
    };

    static constexpr size_t kIncrementalReciprocalScratchCapacity = kHnswIncrementalReciprocalScratchCapacity;
    static constexpr size_t kIncrementalReciprocalCertificateLayerCount = std::numeric_limits<std::uint64_t>::digits;
    static constexpr bool kIncrementalReciprocalSupported = OwnMem && !LSG && HnswPlainDenseVecStore<VecStoreType>::value;
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
    static constexpr std::uint32_t kIncrementalReciprocalEvidenceArmed = 1U << 0U;
    static constexpr std::uint32_t kIncrementalReciprocalEvidenceEligible = 1U << 1U;
    static constexpr std::uint32_t kIncrementalReciprocalEvidenceUnchanged = 1U << 2U;
    static constexpr std::uint32_t kIncrementalReciprocalEvidenceUpdated = 1U << 3U;
    static constexpr std::uint32_t kIncrementalReciprocalEvidenceSealed = 1U << 31U;
#endif

#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
    static constexpr bool kThresholdBatch4ExecutionEvidenceSupported = OwnMem && FourCandidateThresholdDistance<Distance, DataStore>;
    static constexpr std::uint32_t kThresholdBatch4EvidenceArmed = 1U << 0U;
    static constexpr std::uint32_t kThresholdBatch4EvidenceEligible = 1U << 1U;
    static constexpr std::uint32_t kThresholdBatch4EvidenceRejected = 1U << 2U;
    static constexpr std::uint32_t kThresholdBatch4EvidenceSurviving = 1U << 3U;
    static constexpr std::uint32_t kThresholdBatch4EvidenceSealed = 1U << 31U;
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    enum class IncrementalReciprocalCounter : size_t {
        kReciprocalLinks,
        kDirectAppends,
        kFullOverflows,
        kCertificateHits,
        kCertificateMisses,
        kCertificateSets,
        kCertificateClears,
        kUnchangedNewFarthest,
        kUnchangedRejected,
        kUpdatedFull,
        kUpdatedUnderfull,
        kFallbackInvalidState,
        kFallbackScratchCapacity,
        kFallbackDuplicate,
        kFallbackCenterTie,
        kFallbackNonfiniteCenter,
        kFallbackUncertifiedOrder,
        kBaselineCenterDistanceEvaluations,
        kBaselinePairDistanceEvaluations,
        kIncrementalCenterDistanceEvaluations,
        kIncrementalPairDistanceEvaluations,
        kShadowComparisons,
        kShadowMismatches,
        kPredictedAvoidedDistanceEvaluations,
        kPredictedExtraDistanceEvaluations,
        kCount,
    };
#endif

    static std::pair<size_t, size_t> GetMmax(size_t M) {
        constexpr size_t kMaximumM = static_cast<size_t>(std::numeric_limits<VertexListSize>::max()) / 2;
        if (M < 2 || M > kMaximumM) {
            throw std::invalid_argument("HNSW M must be in [2, INT32_MAX / 2]");
        }
        return {2 * M, M};
    }

public:
    KnnHnswBase() : M_(0), ef_construction_(0), mult_(0), prefetch_step_(DEFAULT_PREFETCH_SIZE) {}
    KnnHnswBase(This &&other) noexcept
        : M_(std::exchange(other.M_, 0)), ef_construction_(std::exchange(other.ef_construction_, 0)), mult_(std::exchange(other.mult_, 0.0)),
          build_failed_(other.build_failed_.exchange(false, std::memory_order_acq_rel)), level_generator_(std::move(other.level_generator_)),
          level_cursor_(std::exchange(other.level_cursor_, 0)),
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
          incremental_reciprocal_certification_disabled_(other.incremental_reciprocal_certification_disabled_),
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
          incremental_reciprocal_enabled_(std::exchange(other.incremental_reciprocal_enabled_, true)),
#endif
#endif
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
          incremental_reciprocal_execution_evidence_(other.incremental_reciprocal_execution_evidence_.exchange(0, std::memory_order_acq_rel)),
#endif
#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
          threshold_batch4_execution_evidence_(other.threshold_batch4_execution_evidence_.exchange(0, std::memory_order_acq_rel)),
#endif
          data_store_(std::move(other.data_store_)), distance_(std::move(other.distance_)),
          prefetch_step_(L1_CACHE_SIZE / data_store_.vec_store_meta().GetVecSizeInBytes()) {
        static_assert(std::is_nothrow_move_constructible_v<DataStore>);
        static_assert(std::is_nothrow_move_constructible_v<Distance>);
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        other.incremental_reciprocal_certificate_masks_.clear();
#endif
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        for (auto &counter : other.reciprocal_counters_) {
            counter.store(0, std::memory_order_relaxed);
        }
#endif
    }
    This &operator=(This &&other) noexcept {
        static_assert(std::is_nothrow_move_assignable_v<DataStore>);
        static_assert(std::is_nothrow_move_assignable_v<Distance>);
        if (this != &other) {
            M_ = std::exchange(other.M_, 0);
            ef_construction_ = std::exchange(other.ef_construction_, 0);
            mult_ = std::exchange(other.mult_, 0.0);
            build_failed_.store(other.build_failed_.exchange(false, std::memory_order_acq_rel), std::memory_order_release);
            level_generator_ = std::move(other.level_generator_);
            level_cursor_ = std::exchange(other.level_cursor_, 0);
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
            incremental_reciprocal_execution_evidence_.store(other.incremental_reciprocal_execution_evidence_.exchange(0, std::memory_order_acq_rel),
                                                             std::memory_order_release);
#endif
#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
            threshold_batch4_execution_evidence_.store(other.threshold_batch4_execution_evidence_.exchange(0, std::memory_order_acq_rel),
                                                       std::memory_order_release);
#endif
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            incremental_reciprocal_certificate_masks_.clear();
            other.incremental_reciprocal_certificate_masks_.clear();
            incremental_reciprocal_certification_disabled_ =
                incremental_reciprocal_certification_disabled_ || other.incremental_reciprocal_certification_disabled_;
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
            incremental_reciprocal_enabled_ = std::exchange(other.incremental_reciprocal_enabled_, true);
#endif
#endif
            data_store_ = std::move(other.data_store_);
            distance_ = std::move(other.distance_);
            prefetch_step_ = L1_CACHE_SIZE / data_store_.vec_store_meta().GetVecSizeInBytes();
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            for (auto &counter : reciprocal_counters_) {
                counter.store(0, std::memory_order_relaxed);
            }
            for (auto &counter : other.reciprocal_counters_) {
                counter.store(0, std::memory_order_relaxed);
            }
#endif
        }
        return *this;
    }

    size_t GetSizeInBytes() const { return sizeof(M_) + sizeof(ef_construction_) + data_store_.GetSizeInBytes(); }

    LabelType GetLabel(VertexType vertex_i) const {
        EnsureBuildUsable();
        return data_store_.GetLabel(vertex_i);
    }

    void Save(LocalFileHandle &file_handle) const {
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
        data_store_.EnsureAllVerticesBuilt();
        data_store_.Check();
        file_handle.Append(&M_, sizeof(M_));
        file_handle.Append(&ef_construction_, sizeof(ef_construction_));
        data_store_.Save(file_handle);
    }

    void SaveToPtr(LocalFileHandle &file_handle) const {
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
        data_store_.EnsureAllVerticesBuilt();
        data_store_.Check();
        file_handle.Append(&M_, sizeof(M_));
        file_handle.Append(&ef_construction_, sizeof(ef_construction_));
        data_store_.SaveToPtr(file_handle);
    }

protected:
    static constexpr LayerSize kMaxSupportedLayer = kHnswMaxSupportedLayer;

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    bool IncrementalReciprocalActive() const {
        if constexpr (!kIncrementalReciprocalSupported) {
            return false;
        }
        if (incremental_reciprocal_certification_disabled_) {
            return false;
        }
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
        return incremental_reciprocal_enabled_;
#else
        return true;
#endif
    }

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    void CountIncrementalReciprocal(IncrementalReciprocalCounter counter, std::uint64_t amount = 1) const {
        reciprocal_counters_[static_cast<size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
    }

    void CountIncrementalReciprocalResult(HnswIncrementalReciprocalResult result, VertexListSize result_size, size_t capacity) const {
        switch (result) {
            case HnswIncrementalReciprocalResult::kFallbackInvalidState:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kFallbackInvalidState);
                break;
            case HnswIncrementalReciprocalResult::kFallbackScratchCapacity:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kFallbackScratchCapacity);
                break;
            case HnswIncrementalReciprocalResult::kFallbackDuplicate:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kFallbackDuplicate);
                break;
            case HnswIncrementalReciprocalResult::kFallbackCenterTie:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kFallbackCenterTie);
                break;
            case HnswIncrementalReciprocalResult::kFallbackNonFiniteCenter:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kFallbackNonfiniteCenter);
                break;
            case HnswIncrementalReciprocalResult::kFallbackUncertifiedOrder:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kFallbackUncertifiedOrder);
                break;
            case HnswIncrementalReciprocalResult::kUnchangedNewFarthest:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kUnchangedNewFarthest);
                break;
            case HnswIncrementalReciprocalResult::kUnchangedRejected:
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kUnchangedRejected);
                break;
            case HnswIncrementalReciprocalResult::kUpdated:
                CountIncrementalReciprocal(result_size == static_cast<VertexListSize>(capacity) ? IncrementalReciprocalCounter::kUpdatedFull
                                                                                                : IncrementalReciprocalCounter::kUpdatedUnderfull);
                break;
        }
    }
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE) && defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL)
    void RecordIncrementalReciprocalExecutionEvidence(HnswIncrementalReciprocalResult result) {
        if constexpr (!kIncrementalReciprocalSupported) {
            return;
        }
        const std::uint32_t state = incremental_reciprocal_execution_evidence_.load(std::memory_order_relaxed);
        constexpr std::uint32_t kComplete = kIncrementalReciprocalEvidenceEligible | kIncrementalReciprocalEvidenceUnchanged |
                                            kIncrementalReciprocalEvidenceUpdated;
        if ((state & kIncrementalReciprocalEvidenceArmed) == 0 || (state & kIncrementalReciprocalEvidenceSealed) != 0 ||
            (state & kComplete) == kComplete) {
            return;
        }
        std::uint32_t observed = kIncrementalReciprocalEvidenceEligible;
        switch (result) {
            case HnswIncrementalReciprocalResult::kUnchangedNewFarthest:
            case HnswIncrementalReciprocalResult::kUnchangedRejected:
                observed |= kIncrementalReciprocalEvidenceUnchanged;
                break;
            case HnswIncrementalReciprocalResult::kUpdated:
                observed |= kIncrementalReciprocalEvidenceUpdated;
                break;
            default:
                break;
        }
        const std::uint32_t missing = observed & ~state;
        if (missing != 0) {
            incremental_reciprocal_execution_evidence_.fetch_or(missing, std::memory_order_relaxed);
        }
    }
#endif

    bool IsIncrementalReciprocalCertified(VertexType vertex, i32 layer) const {
        return IncrementalReciprocalActive() && vertex >= 0 && static_cast<size_t>(vertex) < incremental_reciprocal_certificate_masks_.size() &&
               layer >= 0 && static_cast<size_t>(layer) < kIncrementalReciprocalCertificateLayerCount &&
               (incremental_reciprocal_certificate_masks_[static_cast<size_t>(vertex)] & (std::uint64_t{1} << static_cast<size_t>(layer))) != 0;
    }

    void SetIncrementalReciprocalCertificate(VertexType vertex, i32 layer) {
        if (!IncrementalReciprocalActive() || vertex < 0 || static_cast<size_t>(vertex) >= incremental_reciprocal_certificate_masks_.size() ||
            layer < 0 || static_cast<size_t>(layer) >= kIncrementalReciprocalCertificateLayerCount) {
            return;
        }
        const std::uint64_t bit = std::uint64_t{1} << static_cast<size_t>(layer);
        std::uint64_t &mask = incremental_reciprocal_certificate_masks_[static_cast<size_t>(vertex)];
        if ((mask & bit) == 0) {
            mask |= bit;
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            CountIncrementalReciprocal(IncrementalReciprocalCounter::kCertificateSets);
#endif
        }
    }

    void ClearIncrementalReciprocalCertificate(VertexType vertex, i32 layer) {
        if constexpr (!kIncrementalReciprocalSupported) {
            return;
        }
        if (vertex < 0 || static_cast<size_t>(vertex) >= incremental_reciprocal_certificate_masks_.size() || layer < 0 ||
            static_cast<size_t>(layer) >= kIncrementalReciprocalCertificateLayerCount) {
            return;
        }
        const std::uint64_t bit = std::uint64_t{1} << static_cast<size_t>(layer);
        std::uint64_t &mask = incremental_reciprocal_certificate_masks_[static_cast<size_t>(vertex)];
        if ((mask & bit) != 0) {
            mask &= ~bit;
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            CountIncrementalReciprocal(IncrementalReciprocalCounter::kCertificateClears);
#endif
        }
    }

    void ClearAllIncrementalReciprocalCertificates() {
        if constexpr (!kIncrementalReciprocalSupported) {
            return;
        }
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        std::uint64_t cleared = 0;
#endif
        for (std::uint64_t &mask : incremental_reciprocal_certificate_masks_) {
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            cleared += static_cast<std::uint64_t>(std::popcount(mask));
#endif
            mask = 0;
        }
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        CountIncrementalReciprocal(IncrementalReciprocalCounter::kCertificateClears, cleared);
#endif
    }

    void PrepareIncrementalReciprocalCertificates() {
        if constexpr (!kIncrementalReciprocalSupported) {
            return;
        }
        const size_t vertex_count = data_store_.cur_vec_num();
        if (incremental_reciprocal_certificate_masks_.size() < vertex_count) {
            incremental_reciprocal_certificate_masks_.resize(vertex_count, 0);
            incremental_reciprocal_certificate_bytes_.store(incremental_reciprocal_certificate_masks_.capacity() * sizeof(std::uint64_t),
                                                            std::memory_order_relaxed);
        }
    }

    bool HasValidIncrementalReciprocalIds(VertexType center, const VertexType *neighbors, VertexListSize neighbor_count, size_t capacity) const {
        if constexpr (!kIncrementalReciprocalSupported) {
            return false;
        }
        const size_t vertex_count = data_store_.cur_vec_num();
        if (center < 0 || static_cast<size_t>(center) >= vertex_count || neighbors == nullptr || neighbor_count < 0 ||
            static_cast<size_t>(neighbor_count) != capacity || capacity > kIncrementalReciprocalScratchCapacity) {
            return false;
        }
        std::array<VertexType, kIncrementalReciprocalScratchCapacity> sorted_neighbors;
        for (size_t index = 0; index < capacity; ++index) {
            const VertexType neighbor = neighbors[index];
            if (neighbor < 0 || static_cast<size_t>(neighbor) >= vertex_count || neighbor == center) {
                return false;
            }
            sorted_neighbors[index] = neighbor;
        }
        auto sorted_end = sorted_neighbors.begin() + static_cast<std::ptrdiff_t>(capacity);
        std::sort(sorted_neighbors.begin(), sorted_end);
        return std::adjacent_find(sorted_neighbors.begin(), sorted_end) == sorted_end;
    }

    static bool IsFiniteIncrementalReciprocalDistance(DistanceType distance) {
        if constexpr (std::is_floating_point_v<DistanceType>) {
            return std::isfinite(distance);
        }
        return true;
    }
#endif

#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
    void RecordThresholdBatch4ExecutionEvidence(bool threshold_path_ran, std::uint8_t exact_mask) const {
        const std::uint32_t state = threshold_batch4_execution_evidence_.load(std::memory_order_relaxed);
        constexpr std::uint32_t kComplete =
            kThresholdBatch4EvidenceEligible | kThresholdBatch4EvidenceRejected | kThresholdBatch4EvidenceSurviving;
        if ((state & kThresholdBatch4EvidenceArmed) == 0 || (state & kThresholdBatch4EvidenceSealed) != 0 ||
            (state & kComplete) == kComplete) {
            return;
        }
        constexpr std::uint8_t kFourLaneMask = 0x0fU;
        constexpr std::uint8_t kUpperLaneMask = static_cast<std::uint8_t>(~kFourLaneMask);
        if ((exact_mask & kUpperLaneMask) != 0) {
            throw std::logic_error("thresholded Batch4 execution evidence observed invalid upper mask bits");
        }
#if defined(INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL)
        if constexpr (!kThresholdBatch4ExecutionEvidenceSupported) {
            return;
        }
        if (!threshold_path_ran) {
            return;
        }
        const std::uint8_t surviving_lanes = exact_mask & kFourLaneMask;
        std::uint32_t observed = kThresholdBatch4EvidenceEligible;
        if (surviving_lanes != 0 && surviving_lanes != kFourLaneMask) {
            observed |= kThresholdBatch4EvidenceRejected | kThresholdBatch4EvidenceSurviving;
        }
        const std::uint32_t missing = observed & ~state;
        if (missing != 0) {
            threshold_batch4_execution_evidence_.fetch_or(missing, std::memory_order_relaxed);
        }
#else
        static_cast<void>(threshold_path_ran);
#endif
    }
#endif

    void EnsureBuildUsable() const {
        if (IsBuildFailed()) {
            throw std::logic_error("HNSW index is unusable after a failed build");
        }
    }

    i32 GenerateRandomLayer(std::mt19937 &generator) const {
        constexpr double kMt19937Range = 4'294'967'296.0;
        constexpr double kMt19937PairRange = kMt19937Range * kMt19937Range;
        const double low = static_cast<double>(generator());
        const double high = static_cast<double>(generator());
        const double sample = (low + high * kMt19937Range) / kMt19937PairRange;
        if (sample <= 0.0) {
            return kMaxSupportedLayer;
        }
        return static_cast<i32>(std::min(-std::log(sample) * mult_, static_cast<double>(kMaxSupportedLayer)));
    }

    template <LogicalType ColumnLogicalType>
    using SearchLayerReturnParam3T = std::conditional_t<ColumnLogicalType == LogicalType::kEmbedding, VertexType, LabelType>;

    // return the nearest `ef_construction_` neighbors of `query` in layer `layer_idx`
    template <bool WithLock,
              FilterConcept<LabelType> Filter = std::nullopt_t,
              LogicalType ColumnLogicalType = LogicalType::kEmbedding,
              typename MultiVectorInnerTopnIndexType = void>
    std::tuple<size_t, std::unique_ptr<DistanceType[]>, std::unique_ptr<SearchLayerReturnParam3T<ColumnLogicalType>[]>>
    SearchLayer(VertexType enter_point, const QueryType &query, VertexType query_i, i32 layer_idx, size_t result_n, const Filter &filter) const {
        static_assert(ColumnLogicalType == LogicalType::kEmbedding || ColumnLogicalType == LogicalType::kMultiVector);
#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
        HnswLvqPhaseScope capture_phase(query_i == kInvalidVertex ? HnswLvqPhase::kQuery : HnswLvqPhase::kConstructionBeam, layer_idx);
#endif
        auto d_ptr = std::make_unique_for_overwrite<DistanceType[]>(result_n);
        auto i_ptr = std::make_unique_for_overwrite<SearchLayerReturnParam3T<ColumnLogicalType>[]>(result_n);
        using ResultHandler = std::conditional_t<ColumnLogicalType == LogicalType::kEmbedding,
                                                 HeapResultHandler<CompareMax<DistanceType, VertexType>>,
                                                 MultiVectorResultHandler<DistanceType, LabelType, MultiVectorInnerTopnIndexType>>;
        ResultHandler result_handler(1, result_n, d_ptr.get(), i_ptr.get());
        result_handler.Begin();
        auto add_result = [&](DistanceType add_dist, VertexType add_v) {
            if constexpr (ColumnLogicalType == LogicalType::kEmbedding) {
                if constexpr (!std::is_same_v<Filter, std::nullopt_t>) {
                    if (filter(this->GetLabel(add_v))) {
                        result_handler.AddResult(0, add_dist, add_v);
                    }
                } else {
                    result_handler.AddResult(0, add_dist, add_v);
                }
            } else if constexpr (ColumnLogicalType == LogicalType::kMultiVector) {
                const auto l = this->GetLabel(add_v);
                if constexpr (!std::is_same_v<Filter, std::nullopt_t>) {
                    if (filter(l)) {
                        result_handler.AddResult(add_dist, l);
                    }
                } else {
                    result_handler.AddResult(add_dist, l);
                }
            } else {
                static_assert(false, "Unsupported column logical type");
            }
        };
        DistHeap candidate;

        data_store_.PrefetchVec(enter_point);
        // enter_point will not be added to result_handler, the distance is not used
        {
            auto dist = distance_(query, enter_point, data_store_, query_i);
            candidate.emplace(-dist, enter_point);
            add_result(dist, enter_point);
        }

        size_t cur_vec_num = data_store_.cur_vec_num();
        std::vector<bool> visited(cur_vec_num, false);
        visited[enter_point] = true;
        auto commit_candidate = [&](DistanceType dist, VertexType vertex) {
            if (result_handler.GetSize(0) < result_n || dist <= result_handler.GetDistance0(0)) {
                candidate.emplace(-dist, vertex);
                add_result(dist, vertex);
            }
        };

        while (!candidate.empty()) {
            const auto [minus_c_dist, c_idx] = candidate.top();
            candidate.pop();
            if (result_handler.GetSize(0) == result_n && -minus_c_dist > result_handler.GetDistance0(0)) {
                break;
            }

            HnswVertexSharedLock lock;
            if constexpr (WithLock && OwnMem) {
                lock = data_store_.SharedLock(c_idx);
                EnsureBuildUsable();
            }

            const auto [neighbors_p, neighbor_size] = data_store_.GetNeighbors(c_idx, layer_idx);
            auto visit_neighbors_scalar = [&] {
                i32 prefetch_start = 0;
                for (i32 i = 0; i < neighbor_size; ++i) {
                    for (size_t j = prefetch_step_; prefetch_start < neighbor_size && j > 0; --j) {
                        data_store_.PrefetchVec(neighbors_p[prefetch_start++]);
                    }
                    VertexType n_idx = neighbors_p[i];
                    if (n_idx >= (VertexType)cur_vec_num || visited[n_idx]) {
                        continue;
                    }
                    visited[n_idx] = true;
                    commit_candidate(distance_(query, n_idx, data_store_, query_i), n_idx);
                }
            };

            if constexpr (WithLock && OwnMem && ColumnLogicalType == LogicalType::kEmbedding && std::is_same_v<Filter, std::nullopt_t> &&
                          FourCandidateDistance<Distance, DataStore>) {
                if (
#if defined(__APPLE__) && defined(__aarch64__) && defined(INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL)
                    false &&
#endif
                    query_i != kInvalidVertex && distance_.SupportsBatch4()) {
                    std::array<VertexType, 4> pending_vertices{};
                    std::array<DistanceType, 4> pending_distances{};
                    size_t pending_count = 0;
                    i32 prefetch_start = 0;
                    for (i32 i = 0; i < neighbor_size; ++i) {
                        for (size_t j = prefetch_step_; prefetch_start < neighbor_size && j > 0; --j) {
                            data_store_.PrefetchVec(neighbors_p[prefetch_start++]);
                        }
                        const VertexType n_idx = neighbors_p[i];
                        if (n_idx >= (VertexType)cur_vec_num || visited[n_idx]) {
                            continue;
                        }
                        visited[n_idx] = true;
                        pending_vertices[pending_count++] = n_idx;
                        if (pending_count == pending_vertices.size()) {
                            bool threshold_path_ran = false;
                            std::uint8_t exact_mask = 0;
#if defined(INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL)
                            if constexpr (FourCandidateThresholdDistance<Distance, DataStore>) {
                                if (result_handler.GetSize(0) == result_n && distance_.SupportsBatch4WithinThreshold()) {
                                    const DistanceType threshold = result_handler.GetDistance0(0);
                                    exact_mask = distance_.Batch4WithinThreshold(query, pending_vertices, data_store_, threshold, pending_distances);
                                    threshold_path_ran = true;
                                }
                            }
#endif
                            if (!threshold_path_ran) {
                                distance_.Batch4(query, pending_vertices, data_store_, pending_distances);
                            }
#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
                            RecordThresholdBatch4ExecutionEvidence(threshold_path_ran, exact_mask);
#endif
                            for (size_t lane = 0; lane < pending_count; ++lane) {
                                if (!threshold_path_ran || (exact_mask & (std::uint8_t{1} << lane)) != 0) {
                                    commit_candidate(pending_distances[lane], pending_vertices[lane]);
                                }
                            }
                            pending_count = 0;
                        }
                    }
                    for (size_t lane = 0; lane < pending_count; ++lane) {
                        const VertexType n_idx = pending_vertices[lane];
                        commit_candidate(distance_(query, n_idx, data_store_, query_i), n_idx);
                    }
                } else {
                    visit_neighbors_scalar();
                }
            } else {
                visit_neighbors_scalar();
            }
        }
        result_handler.EndWithoutSort();
        EnsureBuildUsable();
        return {result_handler.GetSize(0), std::move(d_ptr), std::move(i_ptr)};
    }

    template <bool WithLock>
    VertexType SearchLayerNearest(VertexType enter_point, const QueryType &query, VertexType query_i, i32 layer_idx) const {
#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
        HnswLvqPhaseScope capture_phase(query_i == kInvalidVertex ? HnswLvqPhase::kQuery : HnswLvqPhase::kConstructionGreedy, layer_idx);
#endif
        VertexType cur_p = enter_point;
        auto cur_dist = distance_(query, cur_p, data_store_, query_i);
        bool check = true;
        while (check) {
            check = false;

            HnswVertexSharedLock lock;
            if constexpr (WithLock && OwnMem) {
                lock = data_store_.SharedLock(cur_p);
                EnsureBuildUsable();
            }

            const auto [neighbors_p, neighbor_size] = data_store_.GetNeighbors(cur_p, layer_idx);
            for (int i = neighbor_size - 1; i >= 0; --i) {
                VertexType n_idx = neighbors_p[i];
                auto n_dist = distance_(query, n_idx, data_store_, query_i);
                if (n_dist < cur_dist) {
                    cur_p = n_idx;
                    cur_dist = n_dist;
                    check = true;
                }
            }
        }
        EnsureBuildUsable();
        return cur_p;
    }

    // the function does not need mutex because the lock of `result_p` is already acquired
    template <bool EnableBatch4 = false>
    void SelectNeighborsHeuristic(std::vector<PDV> candidates,
                                  size_t M,
                                  VertexType *result_p,
                                  VertexListSize *result_size_p
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                                  ,
                                  IncrementalReciprocalSelectionMetadata *incremental_metadata = nullptr
#endif
    ) const {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        IncrementalReciprocalSelectionMetadata metadata{};
        std::optional<DistanceType> previous_accepted_distance;
#endif
        VertexListSize result_size = 0;
        if (candidates.size() < M) {
            std::sort(candidates.begin(), candidates.end(), CMPReverse());
            for (const auto &[_, idx] : candidates) {
                result_p[result_size++] = idx;
            }
        } else {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            metadata.heuristic_branch = true;
            metadata.finite_strict_order = true;
#endif
            std::make_heap(candidates.begin(), candidates.end(), CMPReverse());
            while (!candidates.empty() && size_t(result_size) < M) {
                std::pop_heap(candidates.begin(), candidates.end(), CMPReverse());
                const auto &[c_dist, c_idx] = candidates.back();
                QueryType c_data = data_store_.GetVecToQuery(c_idx);
                bool check = true;
                auto check_scalar_tail = [&](size_t start) {
                    for (size_t i = start; i < size_t(result_size); ++i) {
                        VertexType r_idx = result_p[i];
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                        ++metadata.pair_distance_evaluations;
#endif
                        auto cr_dist = distance_(c_data, r_idx, data_store_, c_idx);
                        if (cr_dist < c_dist) {
                            check = false;
                            break;
                        }
                    }
                };
                if constexpr (EnableBatch4 && FourCandidateDistance<Distance, DataStore>) {
                    if (
#if defined(__APPLE__) && defined(__aarch64__) && defined(INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING)
                        false &&
#endif
                        distance_.SupportsBatch4()) {
                        size_t i = 0;
                        for (; i + 4 <= size_t(result_size); i += 4) {
                            const std::array<VertexType, 4> result_vertices{
                                result_p[i],
                                result_p[i + 1],
                                result_p[i + 2],
                                result_p[i + 3],
                            };
                            std::array<DistanceType, 4> result_distances{};
                            distance_.Batch4(c_data, result_vertices, data_store_, result_distances);
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                            metadata.used_batch4 = true;
#endif
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                            metadata.pair_distance_evaluations += result_distances.size();
#endif
                            for (DistanceType cr_dist : result_distances) {
                                if (cr_dist < c_dist) {
                                    check = false;
                                    break;
                                }
                            }
                            if (!check) {
                                break;
                            }
                        }
                        if (check) {
                            check_scalar_tail(i);
                        }
                    } else {
                        check_scalar_tail(0);
                    }
                } else {
                    check_scalar_tail(0);
                }
                if (check) {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                    if (!IsFiniteIncrementalReciprocalDistance(c_dist) ||
                        (previous_accepted_distance.has_value() && !(c_dist > *previous_accepted_distance))) {
                        metadata.finite_strict_order = false;
                    }
                    previous_accepted_distance = c_dist;
#endif
                    result_p[result_size++] = c_idx;
                }
                candidates.pop_back();
            }
            std::reverse(result_p, result_p + result_size);
        }
        *result_size_p = result_size;
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        metadata.full = size_t(result_size) == M;
        if (incremental_metadata != nullptr) {
            *incremental_metadata = metadata;
        }
#endif
    }

    void ConnectNeighbors(VertexType vertex_i, const VertexType *q_neighbors_p, VertexListSize q_neighbor_size, i32 layer_idx) {
#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
        HnswLvqPhaseScope capture_phase(HnswLvqPhase::kReciprocalSelection, layer_idx);
#endif
        for (int i = 0; i < q_neighbor_size; ++i) {
            VertexType n_idx = q_neighbors_p[i];

            HnswVertexUniqueLock lock = data_store_.UniqueLock(n_idx);

            auto [n_neighbors_p, n_neighbor_size_p] = data_store_.GetNeighborsMut(n_idx, layer_idx);
            VertexListSize n_neighbor_size = *n_neighbor_size_p;
            size_t Mmax = layer_idx == 0 ? data_store_.Mmax0() : data_store_.Mmax();
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            if constexpr (kIncrementalReciprocalSupported) {
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kReciprocalLinks);
            }
#endif
            if (n_neighbor_size < VertexListSize(Mmax)) {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                ClearIncrementalReciprocalCertificate(n_idx, layer_idx);
#endif
                *(n_neighbors_p + n_neighbor_size) = vertex_i;
                *n_neighbor_size_p = n_neighbor_size + 1;
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                if constexpr (kIncrementalReciprocalSupported) {
                    CountIncrementalReciprocal(IncrementalReciprocalCounter::kDirectAppends);
                }
#endif
                continue;
            }
            QueryType n_data = data_store_.GetVecToQuery(n_idx);
            auto n_dist = distance_(n_data, vertex_i, data_store_, n_idx);

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            const bool certified = IsIncrementalReciprocalCertified(n_idx, layer_idx);
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            if constexpr (kIncrementalReciprocalSupported) {
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kFullOverflows);
                CountIncrementalReciprocal(certified ? IncrementalReciprocalCounter::kCertificateHits
                                                     : IncrementalReciprocalCounter::kCertificateMisses);
            }
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            std::array<VertexType, kIncrementalReciprocalScratchCapacity> shadow_neighbors;
            VertexListSize shadow_neighbor_size = n_neighbor_size;
            HnswIncrementalReciprocalResult shadow_result = HnswIncrementalReciprocalResult::kFallbackInvalidState;
            bool shadow_comparable = false;
            std::uint64_t incremental_center_evaluations = 0;
            std::uint64_t incremental_pair_evaluations = 0;
            if (certified && n_neighbor_size >= 0 && size_t(n_neighbor_size) == Mmax && Mmax <= kIncrementalReciprocalScratchCapacity) {
                std::copy_n(n_neighbors_p, size_t(n_neighbor_size), shadow_neighbors.begin());
                std::array<DistanceType, kIncrementalReciprocalScratchCapacity> center_distance_scratch;
                QueryType vertex_data = data_store_.GetVecToQuery(vertex_i);
                incremental_center_evaluations = 1;
                shadow_result = TryIncrementalReciprocalUpdate(
                    vertex_i,
                    n_dist,
                    shadow_neighbors.data(),
                    &shadow_neighbor_size,
                    Mmax,
                    std::span<DistanceType>(center_distance_scratch.data(), center_distance_scratch.size()),
                    [&](VertexType old_vertex) {
                        ++incremental_center_evaluations;
                        return distance_(n_data, old_vertex, data_store_, n_idx);
                    },
                    [&](VertexType old_vertex) {
                        ++incremental_pair_evaluations;
                        return distance_(vertex_data, old_vertex, data_store_, vertex_i);
                    },
                    [&](VertexType old_vertex) {
                        ++incremental_pair_evaluations;
                        QueryType old_data = data_store_.GetVecToQuery(old_vertex);
                        return distance_(old_data, vertex_i, data_store_, old_vertex);
                    });
                CountIncrementalReciprocalResult(shadow_result, shadow_neighbor_size, Mmax);
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kIncrementalCenterDistanceEvaluations, incremental_center_evaluations);
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kIncrementalPairDistanceEvaluations, incremental_pair_evaluations);
                shadow_comparable = shadow_result == HnswIncrementalReciprocalResult::kUnchangedNewFarthest ||
                                    shadow_result == HnswIncrementalReciprocalResult::kUnchangedRejected ||
                                    shadow_result == HnswIncrementalReciprocalResult::kUpdated;
            }
#elif defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL)
            if (certified && n_neighbor_size >= 0 && size_t(n_neighbor_size) == Mmax && Mmax <= kIncrementalReciprocalScratchCapacity) {
                std::array<VertexType, kIncrementalReciprocalScratchCapacity> incremental_neighbors;
                std::copy_n(n_neighbors_p, size_t(n_neighbor_size), incremental_neighbors.begin());
                VertexListSize incremental_neighbor_size = n_neighbor_size;
                std::array<DistanceType, kIncrementalReciprocalScratchCapacity> center_distance_scratch;
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
                std::uint64_t incremental_center_evaluations = 1;
                std::uint64_t incremental_pair_evaluations = 0;
#endif
                QueryType vertex_data = data_store_.GetVecToQuery(vertex_i);
                const HnswIncrementalReciprocalResult incremental_result = TryIncrementalReciprocalUpdate(
                    vertex_i,
                    n_dist,
                    incremental_neighbors.data(),
                    &incremental_neighbor_size,
                    Mmax,
                    std::span<DistanceType>(center_distance_scratch.data(), center_distance_scratch.size()),
                    [&](VertexType old_vertex) {
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
                        ++incremental_center_evaluations;
#endif
                        return distance_(n_data, old_vertex, data_store_, n_idx);
                    },
                    [&](VertexType old_vertex) {
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
                        ++incremental_pair_evaluations;
#endif
                        return distance_(vertex_data, old_vertex, data_store_, vertex_i);
                    },
                    [&](VertexType old_vertex) {
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
                        ++incremental_pair_evaluations;
#endif
                        QueryType old_data = data_store_.GetVecToQuery(old_vertex);
                        return distance_(old_data, vertex_i, data_store_, old_vertex);
                    });
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
                RecordIncrementalReciprocalExecutionEvidence(incremental_result);
#endif
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
                CountIncrementalReciprocalResult(incremental_result, incremental_neighbor_size, Mmax);
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kIncrementalCenterDistanceEvaluations, incremental_center_evaluations);
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kIncrementalPairDistanceEvaluations, incremental_pair_evaluations);
#endif
                if (incremental_result == HnswIncrementalReciprocalResult::kUnchangedNewFarthest ||
                    incremental_result == HnswIncrementalReciprocalResult::kUnchangedRejected) {
                    continue;
                }
                if (incremental_result == HnswIncrementalReciprocalResult::kUpdated) {
                    std::copy_n(incremental_neighbors.begin(), size_t(incremental_neighbor_size), n_neighbors_p);
                    *n_neighbor_size_p = incremental_neighbor_size;
                    if (size_t(incremental_neighbor_size) != Mmax) {
                        ClearIncrementalReciprocalCertificate(n_idx, layer_idx);
                    }
                    continue;
                }
            }
#endif
#endif

            std::vector<PDV> candidates;
            candidates.reserve(n_neighbor_size + 1);
            candidates.emplace_back(n_dist, vertex_i);
            size_t candidate_index = 0;
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            bool used_batch4_center_distances = false;
#endif
            if constexpr (FourCandidateDistance<Distance, DataStore>) {
                if (
#if defined(__APPLE__) && defined(__aarch64__) && defined(INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING)
                    false &&
#endif
                    distance_.SupportsBatch4()) {
                    for (; candidate_index + 4 <= size_t(n_neighbor_size); candidate_index += 4) {
                        const std::array<VertexType, 4> candidate_vertices{
                            n_neighbors_p[candidate_index],
                            n_neighbors_p[candidate_index + 1],
                            n_neighbors_p[candidate_index + 2],
                            n_neighbors_p[candidate_index + 3],
                        };
                        std::array<DistanceType, 4> candidate_distances{};
                        distance_.Batch4(n_data, candidate_vertices, data_store_, candidate_distances);
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                        used_batch4_center_distances = true;
#endif
                        for (size_t lane = 0; lane < candidate_vertices.size(); ++lane) {
                            candidates.emplace_back(candidate_distances[lane], candidate_vertices[lane]);
                        }
                    }
                }
            }
            for (; candidate_index < size_t(n_neighbor_size); ++candidate_index) {
                const VertexType candidate_vertex = n_neighbors_p[candidate_index];
                candidates.emplace_back(distance_(n_data, candidate_vertex, data_store_, n_idx), candidate_vertex);
            }

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            IncrementalReciprocalSelectionMetadata selection_metadata{};
            SelectNeighborsHeuristic<true>(std::move(candidates), Mmax, n_neighbors_p, n_neighbor_size_p, &selection_metadata); // write in memory
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            CountIncrementalReciprocal(IncrementalReciprocalCounter::kBaselineCenterDistanceEvaluations, std::uint64_t(n_neighbor_size) + 1);
            CountIncrementalReciprocal(IncrementalReciprocalCounter::kBaselinePairDistanceEvaluations, selection_metadata.pair_distance_evaluations);
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            if (certified && shadow_comparable) {
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kShadowComparisons);
                const bool exact = shadow_neighbor_size == *n_neighbor_size_p &&
                                   (*n_neighbor_size_p == 0 ||
                                    std::memcmp(shadow_neighbors.data(), n_neighbors_p, size_t(*n_neighbor_size_p) * sizeof(VertexType)) == 0);
                if (!exact) {
                    CountIncrementalReciprocal(IncrementalReciprocalCounter::kShadowMismatches);
                } else {
                    const std::uint64_t baseline_evaluations = std::uint64_t(n_neighbor_size) + 1 + selection_metadata.pair_distance_evaluations;
                    const std::uint64_t incremental_evaluations = incremental_center_evaluations + incremental_pair_evaluations;
                    CountIncrementalReciprocal(baseline_evaluations >= incremental_evaluations
                                                   ? IncrementalReciprocalCounter::kPredictedAvoidedDistanceEvaluations
                                                   : IncrementalReciprocalCounter::kPredictedExtraDistanceEvaluations,
                                               baseline_evaluations >= incremental_evaluations ? baseline_evaluations - incremental_evaluations
                                                                                               : incremental_evaluations - baseline_evaluations);
                }
            } else if (certified) {
                const std::uint64_t nonshared_incremental_evaluations =
                    incremental_center_evaluations + incremental_pair_evaluations - std::min<std::uint64_t>(incremental_center_evaluations, 1);
                CountIncrementalReciprocal(IncrementalReciprocalCounter::kPredictedExtraDistanceEvaluations, nonshared_incremental_evaluations);
            }
#endif

            if (Mmax <= kIncrementalReciprocalScratchCapacity && !used_batch4_center_distances && selection_metadata.heuristic_branch &&
                !selection_metadata.used_batch4 && selection_metadata.finite_strict_order && selection_metadata.full &&
                HasValidIncrementalReciprocalIds(n_idx, n_neighbors_p, *n_neighbor_size_p, Mmax)) {
                SetIncrementalReciprocalCertificate(n_idx, layer_idx);
            } else {
                ClearIncrementalReciprocalCertificate(n_idx, layer_idx);
            }
#else
            SelectNeighborsHeuristic<true>(std::move(candidates), Mmax, n_neighbors_p, n_neighbor_size_p); // write in memory
#endif
        }
    }

    template <bool WithLock, FilterConcept<LabelType> Filter, LogicalType ColumnLogicalType>
    auto SearchLayerHelper(VertexType enter_point, const QueryType &query, i32 layer_idx, size_t result_n, const Filter &filter) const {
        if constexpr (ColumnLogicalType == LogicalType::kEmbedding) {
            return SearchLayer<WithLock, Filter, ColumnLogicalType>(enter_point, query, kInvalidVertex, layer_idx, result_n, filter);
        } else if constexpr (ColumnLogicalType == LogicalType::kMultiVector) {
            if (result_n <= std::numeric_limits<u8>::max()) {
                return SearchLayer<WithLock, Filter, ColumnLogicalType, u8>(enter_point, query, kInvalidVertex, layer_idx, result_n, filter);
            }
            if (result_n <= std::numeric_limits<u16>::max()) {
                return SearchLayer<WithLock, Filter, ColumnLogicalType, u16>(enter_point, query, kInvalidVertex, layer_idx, result_n, filter);
            }
            if (result_n <= std::numeric_limits<u32>::max()) {
                return SearchLayer<WithLock, Filter, ColumnLogicalType, u32>(enter_point, query, kInvalidVertex, layer_idx, result_n, filter);
            }
            UnrecoverableError(fmt::format("Unsupported result_n : {}, which is larger than u32::max()", result_n));
            return std::tuple<size_t, std::unique_ptr<DistanceType[]>, std::unique_ptr<SearchLayerReturnParam3T<ColumnLogicalType>[]>>{};
        } else {
            static_assert(false, "Unsupported column logical type");
        }
    }

    template <bool WithLock, FilterConcept<LabelType> Filter = std::nullopt_t, LogicalType ColumnLogicalType = LogicalType::kEmbedding>
    std::tuple<size_t, std::unique_ptr<DistanceType[]>, std::unique_ptr<SearchLayerReturnParam3T<ColumnLogicalType>[]>>
    KnnSearchInner(const QueryVecType &q, size_t k, const Filter &filter, const KnnSearchOption &option) const {
        size_t ef = option.ef_;
        if (ef == 0) {
            ef = k;
        }
        QueryType query = data_store_.MakeQuery(q);
        auto [max_layer, ep] = data_store_.GetEnterPoint();
        if (ep == -1) {
            EnsureBuildUsable();
            return {0, nullptr, nullptr};
        }
        for (i32 cur_layer = max_layer; cur_layer > 0; --cur_layer) {
            ep = SearchLayerNearest<WithLock>(ep, query, kInvalidVertex, cur_layer);
        }
        auto result = SearchLayerHelper<WithLock, Filter, ColumnLogicalType>(ep, query, 0, ef, filter);
        EnsureBuildUsable();
        return result;
    }

public:
    void InitLSGBuilder(const IndexHnsw *index_hnsw, std::shared_ptr<ColumnDef> column_def) {
        EnsureBuildUsable();
        if (!LSG) {
            UnrecoverableError("InsertSampleVecs when LSG not use!");
        }
        lsg_builder_ = HnswLSGBuilder<DataType, DistanceType>(index_hnsw, column_def);
    }

    template <typename Iter>
    size_t InsertSampleVecs(Iter iter, size_t sample_num = std::numeric_limits<size_t>::max()) {
        EnsureBuildUsable();
        if (!LSG) {
            UnrecoverableError("InsertSampleVecs when LSG not use!");
        }
        if (!lsg_builder_.has_value()) {
            UnrecoverableError("lsg_builder_ not exist, maybe not Init!");
        }
        return lsg_builder_->InsertSampleVec(std::move(iter), sample_num);
    }

    template <typename Iter>
    void InsertLSAvg(Iter iter, size_t row_count) {
        EnsureBuildUsable();
        if (!LSG) {
            UnrecoverableError("InsertLSAvg when LSG not use!");
        }
        if (!lsg_builder_.has_value()) {
            UnrecoverableError("lsg_builder_ not exist, maybe not Init!");
        }
        lsg_builder_->InsertLSAvg(std::move(iter), row_count);
    }

    void SetLSGParam() {
        EnsureBuildUsable();
        if (!LSG) {
            UnrecoverableError("InsertSampleVecs when LSG not use!");
        }
        if (!lsg_builder_.has_value()) {
            UnrecoverableError("lsg_builder_ not exist, maybe not Init!");
        }
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        ClearAllIncrementalReciprocalCertificates();
#endif
        distance_.SetLSGParam(lsg_builder_->alpha(), lsg_builder_->avg());
    }

public:
    template <DataIteratorConcept<QueryVecType, LabelType> Iterator>
    std::pair<size_t, size_t> InsertVecs(Iterator &&iter, const HnswInsertConfig &config = kDefaultHnswInsertConfig) {
        EnsureBuildUsable();
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
        EnsureAllVerticesBuiltWithOperationLockHeld();
        try {
            auto [start_i, end_i] = StoreDataWithOperationLockHeld(std::forward<Iterator>(iter), config);
            for (VertexType vertex_i = start_i; vertex_i < end_i; ++vertex_i) {
                BuildWithOperationLockHeld(vertex_i);
            }
            static_cast<void>(FinalizeBuildWithOperationLockHeld());
            return {start_i, end_i};
        } catch (...) {
            MarkBuildFailed();
            throw;
        }
    }

    template <DataIteratorConcept<QueryVecType, LabelType> Iterator>
    std::pair<VertexType, VertexType> StoreData(Iterator &&iter, const HnswInsertConfig &config = kDefaultHnswInsertConfig) {
        EnsureBuildUsable();
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
        return StoreDataWithOperationLockHeld(std::forward<Iterator>(iter), config);
    }

    template <DataIteratorConcept<QueryVecType, LabelType> Iterator>
    std::pair<VertexType, VertexType> StoreDataWithOperationLockHeld(Iterator &&iter, const HnswInsertConfig &config = kDefaultHnswInsertConfig) {
        EnsureBuildUsable();
        try {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            if (config.optimize_) {
                ClearAllIncrementalReciprocalCertificates();
            }
#endif
            std::pair<VertexType, VertexType> stored_range;
            if (config.optimize_) {
                stored_range = data_store_.OptAddVec(std::forward<Iterator>(iter));
            } else {
                stored_range = data_store_.AddVec(std::forward<Iterator>(iter));
            }
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            PrepareIncrementalReciprocalCertificates();
#endif
            return stored_range;
        } catch (...) {
            MarkBuildFailed();
            throw;
        }
    }

    void Optimize() {
        EnsureBuildUsable();
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
        try {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            ClearAllIncrementalReciprocalCertificates();
#endif
            data_store_.Optimize();
        } catch (...) {
            MarkBuildFailed();
            throw;
        }
    }

private:
    size_t RepairLevelZeroConnectivityWithOperationLockHeld() {
        static_assert(OwnMem, "Level-zero connectivity repair requires an owning HNSW index");

        const size_t vertex_count = data_store_.cur_vec_num();
        if (vertex_count == 0) {
            return 0;
        }

        const auto [max_layer, entry_point] = data_store_.GetEnterPoint();
        static_cast<void>(max_layer);
        if (entry_point < 0 || static_cast<size_t>(entry_point) >= vertex_count) {
            throw std::logic_error("Cannot finalize an HNSW graph with an invalid entry point");
        }

        const size_t level_zero_capacity = data_store_.Mmax0();
        if (level_zero_capacity == 0 && vertex_count > 1) {
            throw std::logic_error("Cannot connect an HNSW graph with zero level-zero capacity");
        }

        std::vector<std::uint8_t> visited(vertex_count, 0);
        std::vector<VertexType> parent(vertex_count, kInvalidVertex);
        std::vector<VertexType> reachable;
        reachable.reserve(vertex_count);

        auto checked_neighbors = [&](VertexType vertex) -> std::pair<const VertexType *, VertexListSize> {
            const auto neighbors = data_store_.GetNeighbors(vertex, 0);
            if (neighbors.second < 0 || static_cast<size_t>(neighbors.second) > level_zero_capacity) {
                throw std::logic_error("Cannot finalize an HNSW graph with an invalid level-zero degree");
            }
            return neighbors;
        };

        visited[static_cast<size_t>(entry_point)] = 1;
        reachable.push_back(entry_point);
        size_t traversal_cursor = 0;
        auto drain_reachable = [&] {
            while (traversal_cursor < reachable.size()) {
                const VertexType source = reachable[traversal_cursor++];
                const auto [neighbors, degree] = checked_neighbors(source);
                for (VertexListSize index = 0; index < degree; ++index) {
                    const VertexType target = neighbors[index];
                    if (target < 0 || static_cast<size_t>(target) >= vertex_count) {
                        throw std::logic_error("Cannot finalize an HNSW graph with an out-of-range level-zero edge");
                    }
                    if (visited[static_cast<size_t>(target)] != 0) {
                        continue;
                    }
                    visited[static_cast<size_t>(target)] = 1;
                    parent[static_cast<size_t>(target)] = source;
                    reachable.push_back(target);
                }
            }
        };
        drain_reachable();

        struct BridgeSlot {
            VertexType source;
            VertexListSize index;
            bool append;
        };

        auto append_slot = [&](VertexType source) -> std::optional<BridgeSlot> {
            const auto [neighbors, degree] = checked_neighbors(source);
            static_cast<void>(neighbors);
            if (static_cast<size_t>(degree) < level_zero_capacity) {
                return BridgeSlot{.source = source, .index = degree, .append = true};
            }
            return std::nullopt;
        };
        auto non_tree_slot = [&](VertexType source) -> std::optional<BridgeSlot> {
            const auto [neighbors, degree] = checked_neighbors(source);
            for (VertexListSize index = 0; index < degree; ++index) {
                const VertexType target = neighbors[index];
                if (target < 0 || static_cast<size_t>(target) >= vertex_count) {
                    throw std::logic_error("Cannot finalize an HNSW graph with an out-of-range level-zero edge");
                }
                if (parent[static_cast<size_t>(target)] != source) {
                    return BridgeSlot{.source = source, .index = index, .append = false};
                }
            }
            return std::nullopt;
        };

        size_t repair_count = 0;
        size_t next_unreachable = 0;
        while (reachable.size() != vertex_count) {
            while (next_unreachable < vertex_count && visited[next_unreachable] != 0) {
                ++next_unreachable;
            }
            if (next_unreachable == vertex_count) {
                throw std::logic_error("HNSW connectivity finalization lost track of an unreachable vertex");
            }
            const VertexType target = static_cast<VertexType>(next_unreachable);

            std::optional<BridgeSlot> bridge;
            const auto [target_neighbors, target_degree] = checked_neighbors(target);
            for (VertexListSize index = 0; index < target_degree && !bridge; ++index) {
                const VertexType candidate_source = target_neighbors[index];
                if (candidate_source < 0 || static_cast<size_t>(candidate_source) >= vertex_count) {
                    throw std::logic_error("Cannot finalize an HNSW graph with an out-of-range level-zero edge");
                }
                if (visited[static_cast<size_t>(candidate_source)] == 0) {
                    continue;
                }
                bridge = append_slot(candidate_source);
                if (!bridge) {
                    bridge = non_tree_slot(candidate_source);
                }
            }

            for (size_t index = 0; index < reachable.size() && !bridge; ++index) {
                bridge = append_slot(reachable[index]);
            }
            for (size_t index = 0; index < reachable.size() && !bridge; ++index) {
                bridge = non_tree_slot(reachable[index]);
            }
            if (!bridge) {
                throw std::logic_error("HNSW connectivity finalization could not find a reachability-preserving bridge slot");
            }

            {
                auto source_lock = data_store_.UniqueLock(bridge->source);
                auto [neighbors, degree] = data_store_.GetNeighborsMut(bridge->source, 0);
                if (*degree < 0 || static_cast<size_t>(*degree) > level_zero_capacity) {
                    throw std::logic_error("HNSW connectivity finalization observed an invalid level-zero degree");
                }
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                const bool certificate_before = IsIncrementalReciprocalCertified(bridge->source, 0);
#endif
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                ClearIncrementalReciprocalCertificate(bridge->source, 0);
#endif
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                const bool certificate_after = IsIncrementalReciprocalCertified(bridge->source, 0);
#endif
                if (bridge->append) {
                    if (*degree != bridge->index || static_cast<size_t>(*degree) >= level_zero_capacity) {
                        throw std::logic_error("HNSW connectivity finalization observed a concurrent graph mutation");
                    }
                    neighbors[*degree] = target;
                    ++*degree;
                } else {
                    if (bridge->index >= *degree) {
                        throw std::logic_error("HNSW connectivity finalization observed a concurrent graph mutation");
                    }
                    neighbors[bridge->index] = target;
                }
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
                if (connectivity_repair_hook_) {
                    connectivity_repair_hook_(bridge->source, target, bridge->append, certificate_before, certificate_after);
                }
#endif
            }

            visited[next_unreachable] = 1;
            parent[next_unreachable] = bridge->source;
            reachable.push_back(target);
            ++repair_count;
            drain_reachable();
        }
        return repair_count;
    }

public:
    size_t FinalizeBuildWithOperationLockHeld() {
        EnsureBuildUsable();
        if constexpr (!OwnMem) {
            throw std::logic_error("Cannot finalize a non-owning HNSW index");
        } else {
            data_store_.EnsureAllVerticesBuilt();
            try {
                return RepairLevelZeroConnectivityWithOperationLockHeld();
            } catch (...) {
                MarkBuildFailed();
                throw;
            }
        }
    }

    bool FinalizeBuildIfCompleteWithOperationLockHeld() {
        EnsureBuildUsable();
        if constexpr (!OwnMem) {
            return false;
        } else if (!data_store_.AllVerticesBuilt()) {
            return false;
        } else {
            static_cast<void>(FinalizeBuildWithOperationLockHeld());
            return true;
        }
    }

    size_t FinalizeBuild() {
        EnsureBuildUsable();
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
        return FinalizeBuildWithOperationLockHeld();
    }

private:
    void FinalizeBuildIfComplete() {
        if constexpr (OwnMem) {
            if (!data_store_.AllVerticesBuilt()) {
                return;
            }
            auto operation_lock = AcquireExclusiveOperationLock();
            EnsureBuildUsable();
            static_cast<void>(FinalizeBuildIfCompleteWithOperationLockHeld());
        }
    }

public:
    void EnsureAllVerticesBuiltWithOperationLockHeld() const {
        EnsureBuildUsable();
        data_store_.EnsureAllVerticesBuilt();
    }

    std::unique_lock<std::shared_mutex> AcquireExclusiveOperationLock() const { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::shared_mutex> TryAcquireExclusiveOperationLock() const { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    std::shared_lock<std::shared_mutex> AcquireSharedOperationLock() const { return std::shared_lock(operation_mutex_); }

    std::vector<LayerSize> PrepareBuildLevels(size_t start, size_t end) {
        EnsureBuildUsable();
        constexpr size_t kMaxVertexCount = static_cast<size_t>(std::numeric_limits<VertexType>::max()) + 1;
        if (end < start || end > kMaxVertexCount) {
            throw std::invalid_argument("HNSW bulk level range is invalid");
        }

        std::lock_guard lock(level_generator_mutex_);
        std::vector<LayerSize> levels(end - start);
        if (start < level_cursor_) {
            std::mt19937 scratch_generator;
            size_t scratch_cursor = 0;
            while (scratch_cursor < start) {
                static_cast<void>(GenerateRandomLayer(scratch_generator));
                ++scratch_cursor;
            }
            while (scratch_cursor < end) {
                levels[scratch_cursor - start] = GenerateRandomLayer(scratch_generator);
                ++scratch_cursor;
            }
            if (end > level_cursor_) {
                level_generator_ = std::move(scratch_generator);
                level_cursor_ = end;
            }
            return levels;
        }

        while (level_cursor_ < start) {
            static_cast<void>(GenerateRandomLayer(level_generator_));
            ++level_cursor_;
        }
        while (level_cursor_ < end) {
            levels[level_cursor_ - start] = GenerateRandomLayer(level_generator_);
            ++level_cursor_;
        }
        return levels;
    }

private:
    struct StagedBuildConnections {
        i32 top_layer{-1};
        std::array<VertexListSize, static_cast<size_t>(kMaxSupportedLayer) + 1> layer_sizes{};
        std::vector<VertexType> neighbors;
    };

    void ValidateBuildVertex(VertexType vertex_i) const {
        if (vertex_i < 0 || static_cast<size_t>(vertex_i) >= data_store_.cur_vec_num()) {
            throw std::out_of_range("HNSW build vertex is outside the stored range");
        }
    }

    static void ValidateBuildLevel(LayerSize q_layer) {
        if (q_layer < 0 || q_layer > kMaxSupportedLayer) {
            throw std::invalid_argument("HNSW vertex level is outside the supported range");
        }
    }

    bool ClaimInitialEntryPoint(VertexType vertex_i, LayerSize q_layer, i32 &max_layer, VertexType &entry_point) {
        std::tie(max_layer, entry_point) = data_store_.GetEnterPoint();
        if (entry_point != kInvalidVertex) {
            return false;
        }

        std::lock_guard lock(initial_entry_mutex_);
        std::tie(max_layer, entry_point) = data_store_.GetEnterPoint();
        if (entry_point != kInvalidVertex) {
            return false;
        }

        const auto [old_max_layer, old_entry_point] = data_store_.TryUpdateEnterPoint(q_layer, vertex_i);
        if (old_max_layer != -1 || old_entry_point != kInvalidVertex) {
            throw std::logic_error("HNSW initial entry-point claim observed inconsistent graph metadata");
        }
        return true;
    }

    void BuildClaimed(VertexType vertex_i, LayerSize q_layer) {
        QueryType query = data_store_.GetVecToQuery(vertex_i);
        i32 max_layer = -1;
        VertexType ep = kInvalidVertex;
        const bool initial_entry_point = ClaimInitialEntryPoint(vertex_i, q_layer, max_layer, ep);
#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        if (build_entry_hook_) {
            build_entry_hook_(vertex_i);
        }
#endif
        if (initial_entry_point) {
            return;
        }

        for (i32 cur_layer = max_layer; cur_layer > q_layer; --cur_layer) {
            ep = SearchLayerNearest<true>(ep, query, vertex_i, cur_layer);
        }

        StagedBuildConnections staged;
        staged.top_layer = std::min(q_layer, max_layer);
        if (staged.top_layer >= 0) {
            const size_t layer_count = static_cast<size_t>(staged.top_layer) + 1;
            if (M_ <= std::numeric_limits<size_t>::max() / layer_count) {
                staged.neighbors.reserve(layer_count * M_);
            }
        }

        for (i32 cur_layer = staged.top_layer; cur_layer >= 0; --cur_layer) {
            auto [result_n, d_ptr, v_ptr] = SearchLayer<true>(ep, query, vertex_i, cur_layer, ef_construction_, std::nullopt);
            auto search_result = std::vector<PDV>(result_n);
            for (size_t i = 0; i < result_n; ++i) {
                search_result[i] = {d_ptr[i], v_ptr[i]};
            }

            {
                HnswVertexUniqueLock vertex_lock = data_store_.UniqueLock(vertex_i);
                const auto [q_neighbors_p, q_neighbor_size_p] = data_store_.GetNeighborsMut(vertex_i, cur_layer);
#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
                HnswLvqPhaseScope capture_phase(HnswLvqPhase::kNewSelection, cur_layer);
#endif
                SelectNeighborsHeuristic(std::move(search_result), M_, q_neighbors_p, q_neighbor_size_p);
                if (*q_neighbor_size_p <= 0) {
                    throw std::logic_error("HNSW construction selected an empty neighbor list for a non-empty graph");
                }
                staged.layer_sizes[static_cast<size_t>(cur_layer)] = *q_neighbor_size_p;
                staged.neighbors.insert(staged.neighbors.end(), q_neighbors_p, q_neighbors_p + *q_neighbor_size_p);
                ep = q_neighbors_p[0];
            }
        }

        static_cast<void>(data_store_.TryUpdateEnterPoint(q_layer, vertex_i));

        size_t neighbor_offset = 0;
        for (i32 cur_layer = staged.top_layer; cur_layer >= 0; --cur_layer) {
            const VertexListSize layer_size = staged.layer_sizes[static_cast<size_t>(cur_layer)];
            ConnectNeighbors(vertex_i, staged.neighbors.data() + neighbor_offset, layer_size, cur_layer);
            neighbor_offset += static_cast<size_t>(layer_size);
        }
        if (neighbor_offset != staged.neighbors.size()) {
            throw std::logic_error("HNSW staged neighbor publication consumed an inconsistent neighbor count");
        }
    }

public:
    void MarkBuildFailed() noexcept { build_failed_.store(true, std::memory_order_release); }

    bool IsBuildFailed() const noexcept { return build_failed_.load(std::memory_order_acquire); }

    void BuildWithOperationLockHeld(VertexType vertex_i) {
        EnsureBuildUsable();
        ValidateBuildVertex(vertex_i);
        LayerSize level;
        {
            HnswVertexUniqueLock vertex_lock = data_store_.UniqueLock(vertex_i);
            if (data_store_.IsVertexBuilt(vertex_i)) {
                throw std::logic_error("HNSW vertex is already built");
            }
            try {
                const size_t ordinal = static_cast<size_t>(vertex_i);
                level = PrepareBuildLevels(ordinal, ordinal + 1).front();
                data_store_.AddVertex(vertex_i, level);
            } catch (...) {
                MarkBuildFailed();
                throw;
            }
        }
        try {
            BuildClaimed(vertex_i, level);
        } catch (...) {
            MarkBuildFailed();
            throw;
        }
    }

    void BuildWithOperationLockHeld(VertexType vertex_i, LayerSize q_layer) {
        EnsureBuildUsable();
        ValidateBuildLevel(q_layer);
        ValidateBuildVertex(vertex_i);
        {
            HnswVertexUniqueLock vertex_lock = data_store_.UniqueLock(vertex_i);
            if (data_store_.IsVertexBuilt(vertex_i)) {
                throw std::logic_error("HNSW vertex is already built");
            }
            try {
                data_store_.AddVertex(vertex_i, q_layer);
            } catch (...) {
                MarkBuildFailed();
                throw;
            }
        }
        try {
            BuildClaimed(vertex_i, q_layer);
        } catch (...) {
            MarkBuildFailed();
            throw;
        }
    }

    void Build(VertexType vertex_i) {
        EnsureBuildUsable();
        ValidateBuildVertex(vertex_i);
        {
            auto operation_lock = AcquireSharedOperationLock();
            EnsureBuildUsable();
            BuildWithOperationLockHeld(vertex_i);
        }
        FinalizeBuildIfComplete();
    }

    void Build(VertexType vertex_i, LayerSize q_layer) {
        EnsureBuildUsable();
        ValidateBuildLevel(q_layer);
        ValidateBuildVertex(vertex_i);
        {
            auto operation_lock = AcquireSharedOperationLock();
            EnsureBuildUsable();
            BuildWithOperationLockHeld(vertex_i, q_layer);
        }
        FinalizeBuildIfComplete();
    }

    template <FilterConcept<LabelType> Filter = std::nullopt_t, bool WithLock = true>
    std::tuple<size_t, std::unique_ptr<DistanceType[]>, std::unique_ptr<LabelType[]>>
    KnnSearch(const QueryVecType &q, size_t k, const Filter &filter, const KnnSearchOption &option = {}) const {
        EnsureBuildUsable();
        switch (option.column_logical_type_) {
            case LogicalType::kEmbedding: {
                auto [result_n, d_ptr, v_ptr] = KnnSearchInner<WithLock, Filter>(q, k, filter, option);
                auto labels = std::make_unique_for_overwrite<LabelType[]>(result_n);
                for (size_t i = 0; i < result_n; ++i) {
                    labels[i] = GetLabel(v_ptr[i]);
                }
                EnsureBuildUsable();
                return {result_n, std::move(d_ptr), std::move(labels)};
            }
            case LogicalType::kMultiVector: {
                auto result = KnnSearchInner<WithLock, Filter, LogicalType::kMultiVector>(q, k, filter, option);
                EnsureBuildUsable();
                return result;
            }
            default: {
                UnrecoverableError(fmt::format("Unsupported column logical type: {}", LogicalType2Str(option.column_logical_type_)));
            }
        }
        return {};
    }

    template <bool WithLock = true>
    std::tuple<size_t, std::unique_ptr<DistanceType[]>, std::unique_ptr<LabelType[]>>
    KnnSearch(const QueryVecType &q, size_t k, const KnnSearchOption &option = {}) const {
        return KnnSearch<std::nullopt_t, WithLock>(q, k, std::nullopt, option);
    }

    // function for test, add sort for convenience
    template <FilterConcept<LabelType> Filter = std::nullopt_t, bool WithLock = true>
    std::vector<std::pair<DistanceType, LabelType>>
    KnnSearchSorted(const QueryVecType &q, size_t k, const Filter &filter, const KnnSearchOption &option = {}) const {
        EnsureBuildUsable();
        auto [result_n, d_ptr, v_ptr] = KnnSearchInner<WithLock, Filter>(q, k, filter, option);
        std::vector<std::pair<DistanceType, LabelType>> result(result_n);
        for (size_t i = 0; i < result_n; ++i) {
            result[i] = {d_ptr[i], GetLabel(v_ptr[i])};
        }
        std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        EnsureBuildUsable();
        return result;
    }

    // function for test
    std::vector<std::pair<DistanceType, LabelType>> KnnSearchSorted(const QueryVecType &q, size_t k, const KnnSearchOption &option = {}) const {
        return KnnSearchSorted<std::nullopt_t>(q, k, std::nullopt, option);
    }

    size_t GetVecNum() const { return data_store_.cur_vec_num(); }

    size_t mem_usage() const {
        size_t usage = data_store_.mem_usage();
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        usage += incremental_reciprocal_certificate_bytes_.load(std::memory_order_relaxed);
#endif
        return usage;
    }

    Distance &distance() {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        auto operation_lock = AcquireExclusiveOperationLock();
        ClearAllIncrementalReciprocalCertificates();
        incremental_reciprocal_certification_disabled_ = true;
#endif
        return distance_;
    }

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
    bool ArmIncrementalReciprocalExecutionEvidence() {
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL)
        if constexpr (kIncrementalReciprocalSupported) {
            if (data_store_.cur_vec_num() != 0) {
                throw std::invalid_argument("incremental reciprocal execution evidence must be armed before insertion");
            }
            std::uint32_t expected = 0;
            return incremental_reciprocal_execution_evidence_.compare_exchange_strong(expected,
                                                                                      kIncrementalReciprocalEvidenceArmed,
                                                                                      std::memory_order_release,
                                                                                      std::memory_order_relaxed);
        }
#endif
        return false;
    }

    HnswIncrementalReciprocalExecutionEvidence GetIncrementalReciprocalExecutionEvidence() const {
        auto operation_lock = AcquireExclusiveOperationLock();
        const std::uint32_t state = incremental_reciprocal_execution_evidence_.load(std::memory_order_acquire);
        return {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL)
            .treatment_compiled = kIncrementalReciprocalSupported,
#else
            .treatment_compiled = false,
#endif
            .capture_armed = (state & kIncrementalReciprocalEvidenceArmed) != 0,
            .eligible_branch_entered = (state & kIncrementalReciprocalEvidenceEligible) != 0,
            .successful_unchanged_observed = (state & kIncrementalReciprocalEvidenceUnchanged) != 0,
            .successful_updated_observed = (state & kIncrementalReciprocalEvidenceUpdated) != 0,
        };
    }

    HnswIncrementalReciprocalExecutionEvidence SealAndGetIncrementalReciprocalExecutionEvidence() {
        auto operation_lock = AcquireExclusiveOperationLock();
        const std::uint32_t state =
            incremental_reciprocal_execution_evidence_.fetch_or(kIncrementalReciprocalEvidenceSealed, std::memory_order_acq_rel) |
            kIncrementalReciprocalEvidenceSealed;
        return {
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL)
            .treatment_compiled = kIncrementalReciprocalSupported,
#else
            .treatment_compiled = false,
#endif
            .capture_armed = (state & kIncrementalReciprocalEvidenceArmed) != 0,
            .eligible_branch_entered = (state & kIncrementalReciprocalEvidenceEligible) != 0,
            .successful_unchanged_observed = (state & kIncrementalReciprocalEvidenceUnchanged) != 0,
            .successful_updated_observed = (state & kIncrementalReciprocalEvidenceUpdated) != 0,
        };
    }
#endif

#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
    bool ArmThresholdBatch4ExecutionEvidence() {
        auto operation_lock = AcquireExclusiveOperationLock();
        EnsureBuildUsable();
#if defined(INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL)
        if constexpr (kThresholdBatch4ExecutionEvidenceSupported) {
            if (data_store_.cur_vec_num() != 0) {
                throw std::invalid_argument("thresholded Batch4 execution evidence must be armed before insertion");
            }
            std::uint32_t expected = 0;
            return threshold_batch4_execution_evidence_.compare_exchange_strong(expected,
                                                                                kThresholdBatch4EvidenceArmed,
                                                                                std::memory_order_release,
                                                                                std::memory_order_relaxed);
        }
#endif
        return false;
    }

    HnswThresholdBatch4ExecutionEvidence GetThresholdBatch4ExecutionEvidence() const {
        auto operation_lock = AcquireExclusiveOperationLock();
        const std::uint32_t state = threshold_batch4_execution_evidence_.load(std::memory_order_acquire);
        return {
#if defined(INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL)
            .treatment_compiled = kThresholdBatch4ExecutionEvidenceSupported,
#else
            .treatment_compiled = false,
#endif
            .capture_armed = (state & kThresholdBatch4EvidenceArmed) != 0,
            .eligible_branch_entered = (state & kThresholdBatch4EvidenceEligible) != 0,
            .rejected_lane_observed = (state & kThresholdBatch4EvidenceRejected) != 0,
            .surviving_lane_observed = (state & kThresholdBatch4EvidenceSurviving) != 0,
        };
    }

    HnswThresholdBatch4ExecutionEvidence SealAndGetThresholdBatch4ExecutionEvidence() {
        auto operation_lock = AcquireExclusiveOperationLock();
        const std::uint32_t state =
            threshold_batch4_execution_evidence_.fetch_or(kThresholdBatch4EvidenceSealed, std::memory_order_acq_rel) |
            kThresholdBatch4EvidenceSealed;
        return {
#if defined(INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL)
            .treatment_compiled = kThresholdBatch4ExecutionEvidenceSupported,
#else
            .treatment_compiled = false,
#endif
            .capture_armed = (state & kThresholdBatch4EvidenceArmed) != 0,
            .eligible_branch_entered = (state & kThresholdBatch4EvidenceEligible) != 0,
            .rejected_lane_observed = (state & kThresholdBatch4EvidenceRejected) != 0,
            .surviving_lane_observed = (state & kThresholdBatch4EvidenceSurviving) != 0,
        };
    }
#endif

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
    void SetIncrementalReciprocalEnabledForTest(bool enabled) {
        auto operation_lock = AcquireExclusiveOperationLock();
        ClearAllIncrementalReciprocalCertificates();
        incremental_reciprocal_enabled_ = enabled;
    }
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    struct IncrementalReciprocalCertificateStorageForTest {
        size_t mask_count{};
        size_t capacity_bytes{};
        size_t charged_bytes{};
    };

    void SetBuildEntryHookForTest(std::function<void(VertexType)> hook) {
        auto operation_lock = AcquireExclusiveOperationLock();
        build_entry_hook_ = std::move(hook);
    }

    void SetConnectivityRepairHookForTest(std::function<void(VertexType, VertexType, bool, bool, bool)> hook) {
        auto operation_lock = AcquireExclusiveOperationLock();
        connectivity_repair_hook_ = std::move(hook);
    }

    void SetAllLevelZeroIncrementalReciprocalCertificatesForTest() {
        auto operation_lock = AcquireExclusiveOperationLock();
        PrepareIncrementalReciprocalCertificates();
        for (size_t vertex = 0; vertex < data_store_.cur_vec_num(); ++vertex) {
            SetIncrementalReciprocalCertificate(static_cast<VertexType>(vertex), 0);
        }
    }

    HnswIncrementalReciprocalStats GetIncrementalReciprocalStats() const {
        auto load = [&](IncrementalReciprocalCounter counter) {
            return reciprocal_counters_[static_cast<size_t>(counter)].load(std::memory_order_relaxed);
        };
        return HnswIncrementalReciprocalStats{
            .reciprocal_links = load(IncrementalReciprocalCounter::kReciprocalLinks),
            .direct_appends = load(IncrementalReciprocalCounter::kDirectAppends),
            .full_overflows = load(IncrementalReciprocalCounter::kFullOverflows),
            .certificate_hits = load(IncrementalReciprocalCounter::kCertificateHits),
            .certificate_misses = load(IncrementalReciprocalCounter::kCertificateMisses),
            .certificate_sets = load(IncrementalReciprocalCounter::kCertificateSets),
            .certificate_clears = load(IncrementalReciprocalCounter::kCertificateClears),
            .unchanged_new_farthest = load(IncrementalReciprocalCounter::kUnchangedNewFarthest),
            .unchanged_rejected = load(IncrementalReciprocalCounter::kUnchangedRejected),
            .updated_full = load(IncrementalReciprocalCounter::kUpdatedFull),
            .updated_underfull = load(IncrementalReciprocalCounter::kUpdatedUnderfull),
            .fallback_invalid_state = load(IncrementalReciprocalCounter::kFallbackInvalidState),
            .fallback_scratch_capacity = load(IncrementalReciprocalCounter::kFallbackScratchCapacity),
            .fallback_duplicate = load(IncrementalReciprocalCounter::kFallbackDuplicate),
            .fallback_center_tie = load(IncrementalReciprocalCounter::kFallbackCenterTie),
            .fallback_nonfinite_center = load(IncrementalReciprocalCounter::kFallbackNonfiniteCenter),
            .fallback_uncertified_order = load(IncrementalReciprocalCounter::kFallbackUncertifiedOrder),
            .baseline_center_distance_evaluations = load(IncrementalReciprocalCounter::kBaselineCenterDistanceEvaluations),
            .baseline_pair_distance_evaluations = load(IncrementalReciprocalCounter::kBaselinePairDistanceEvaluations),
            .incremental_center_distance_evaluations = load(IncrementalReciprocalCounter::kIncrementalCenterDistanceEvaluations),
            .incremental_pair_distance_evaluations = load(IncrementalReciprocalCounter::kIncrementalPairDistanceEvaluations),
            .shadow_comparisons = load(IncrementalReciprocalCounter::kShadowComparisons),
            .shadow_mismatches = load(IncrementalReciprocalCounter::kShadowMismatches),
            .predicted_avoided_distance_evaluations = load(IncrementalReciprocalCounter::kPredictedAvoidedDistanceEvaluations),
            .predicted_extra_distance_evaluations = load(IncrementalReciprocalCounter::kPredictedExtraDistanceEvaluations),
        };
    }

    size_t GetIncrementalReciprocalLiveCertificateCount() const {
        auto operation_lock = AcquireExclusiveOperationLock();
        size_t count = 0;
        for (std::uint64_t mask : incremental_reciprocal_certificate_masks_) {
            count += static_cast<size_t>(std::popcount(mask));
        }
        return count;
    }

    IncrementalReciprocalCertificateStorageForTest GetIncrementalReciprocalCertificateStorageForTest() const {
        auto operation_lock = AcquireExclusiveOperationLock();
        return {
            .mask_count = incremental_reciprocal_certificate_masks_.size(),
            .capacity_bytes = incremental_reciprocal_certificate_masks_.capacity() * sizeof(std::uint64_t),
            .charged_bytes = incremental_reciprocal_certificate_bytes_.load(std::memory_order_relaxed),
        };
    }

    bool HasIncrementalReciprocalCertificateForTest(VertexType vertex, i32 layer) const {
        auto operation_lock = AcquireExclusiveOperationLock();
        return IsIncrementalReciprocalCertified(vertex, layer);
    }
#endif

    LayerSize GetGraphLevel(VertexType vertex) const { return data_store_.GetLevel(vertex); }

    std::pair<const VertexType *, VertexListSize> GetGraphNeighbors(VertexType vertex, i32 layer) const {
        return data_store_.GetNeighbors(vertex, layer);
    }

    std::pair<i32, VertexType> GetGraphEnterPoint() const { return data_store_.GetEnterPoint(); }

    size_t GetGraphMmax0() const { return data_store_.Mmax0(); }

    size_t GetGraphMmax() const { return data_store_.Mmax(); }

protected:
    size_t M_;
    size_t ef_construction_;

    // 1 / log(1.0 * M_)
    double mult_;

    mutable std::shared_mutex operation_mutex_;
    std::atomic<bool> build_failed_{};
    std::mutex initial_entry_mutex_;
    std::mutex level_generator_mutex_;
    std::mt19937 level_generator_{};
    size_t level_cursor_{};

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    std::vector<std::uint64_t> incremental_reciprocal_certificate_masks_;
    std::atomic<size_t> incremental_reciprocal_certificate_bytes_{};
    bool incremental_reciprocal_certification_disabled_{};
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
    bool incremental_reciprocal_enabled_{true};
#endif
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
    std::atomic<std::uint32_t> incremental_reciprocal_execution_evidence_{};
#endif

#if defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
    mutable std::atomic<std::uint32_t> threshold_batch4_execution_evidence_{};
#endif

    DataStore data_store_;
    Distance distance_;

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    mutable std::array<std::atomic<std::uint64_t>, static_cast<size_t>(IncrementalReciprocalCounter::kCount)> reciprocal_counters_{};
    std::function<void(VertexType)> build_entry_hook_;
    std::function<void(VertexType, VertexType, bool, bool, bool)> connectivity_repair_hook_;
#endif

    size_t prefetch_step_;

    std::optional<HnswLSGBuilder<DataType, DistanceType>> lsg_builder_{};

    // //---------------------------------------------- Following is the tmp debug function. ----------------------------------------------
public:
    void Check() const {
        EnsureBuildUsable();
        data_store_.Check();
    }

    void Dump(std::ostream &os) const {
        EnsureBuildUsable();
        os << std::endl << "---------------------------------------------" << std::endl;
        os << "[CONST] M: " << M_ << ", ef_construction: " << ef_construction_ << ", mult: " << mult_ << std::endl;
        data_store_.Dump(os);
        os << "---------------------------------------------" << std::endl;
    }
};

export template <typename VecStoreType, typename LabelType, bool OwnMem = true>
class KnnHnsw : public KnnHnswBase<VecStoreType, LabelType, OwnMem> {
public:
    using This = KnnHnsw<VecStoreType, LabelType, OwnMem>;
    using DataStore = DataStore<VecStoreType, LabelType, OwnMem>;
    using Distance = typename VecStoreType::Distance;
    using CompressLVQVecStoreType = decltype(VecStoreType::template ToLVQ<i8>());
    using CompressRabitqVecStoreType = decltype(VecStoreType::ToRabitq());
    constexpr static bool kOwnMem = OwnMem;

    KnnHnsw(size_t M, size_t ef_construction, DataStore data_store, Distance distance) {
        const auto [expected_mmax0, expected_mmax] = This::GetMmax(M);
        if (data_store.Mmax0() != expected_mmax0 || data_store.Mmax() != expected_mmax) {
            throw std::invalid_argument("HNSW M does not match the stored graph capacities");
        }
        this->M_ = M;
        this->ef_construction_ = std::max(M, ef_construction);
        this->mult_ = 1 / std::log(1.0 * M);
        this->data_store_ = std::move(data_store);
        this->distance_ = std::move(distance);
    }

private:
    template <typename, typename, bool>
    friend class KnnHnsw;

    KnnHnsw(HnswCompressionTargetTag, size_t M, size_t ef_construction, size_t dim) {
        static_cast<void>(This::GetMmax(M));
        this->M_ = M;
        this->ef_construction_ = std::max(M, ef_construction);
        this->mult_ = 1 / std::log(1.0 * M);
        this->distance_ = Distance(dim);
    }

    void InstallCompressedDataStore(DataStore data_store) noexcept {
        static_assert(std::is_nothrow_move_constructible_v<DataStore>);
        static_assert(std::is_nothrow_move_assignable_v<DataStore>);
        this->data_store_ = std::move(data_store);
        this->prefetch_step_ = L1_CACHE_SIZE / this->data_store_.vec_store_meta().GetVecSizeInBytes();
    }

public:
    static std::unique_ptr<This> Make(size_t chunk_size, size_t max_chunk_n, size_t dim, size_t M, size_t ef_construction) {
        auto [Mmax0, Mmax] = This::GetMmax(M);
        auto data_store = DataStore::Make(chunk_size, max_chunk_n, dim, Mmax0, Mmax);
        Distance distance(data_store.dim());
        return std::make_unique<This>(M, ef_construction, std::move(data_store), std::move(distance));
    }

    static std::unique_ptr<This> Load(LocalFileHandle &file_handle) {
        const size_t M = HnswReadStream<size_t>(file_handle, "HNSW M");
        const auto expected_capacities = This::GetMmax(M);
        const size_t ef_construction = HnswReadStream<size_t>(file_handle, "HNSW ef_construction");
        if (ef_construction > static_cast<size_t>(std::numeric_limits<VertexType>::max())) {
            HnswStreamError("HNSW ef_construction exceeds the vertex representation");
        }

        auto data_store = DataStore::Load(file_handle, 0, expected_capacities);
        HnswRequireStreamEmpty(file_handle);
        Distance distance(data_store.dim());

        auto index = std::make_unique<This>(M, ef_construction, std::move(data_store), std::move(distance));
        index->Check();
        return index;
    }

    static std::unique_ptr<This> LoadFromPtr(LocalFileHandle &file_handle, size_t size) {
        HnswEnsureStreamAvailable(file_handle, size, "HNSW pointer image");
        auto buffer = std::make_unique<char[]>(size);
        HnswReadExact(file_handle, buffer.get(), size, "HNSW pointer image");
        HnswPointerReader reader(buffer.get(), size);
        const size_t M = reader.Read<size_t>("HNSW M");
        static_cast<void>(This::GetMmax(M));
        const size_t ef_construction = reader.Read<size_t>("HNSW ef_construction");
        if (ef_construction > static_cast<size_t>(std::numeric_limits<VertexType>::max())) {
            HnswPointerImageError("HNSW ef_construction exceeds the vertex representation");
        }
        auto data_store = DataStore::LoadFromPtr(reader);
        Distance distance(data_store.dim());
        reader.RequireEmpty();
        auto index = std::make_unique<This>(M, ef_construction, std::move(data_store), std::move(distance));
        index->Check();
        return index;
    }

    std::unique_ptr<KnnHnsw<CompressLVQVecStoreType, LabelType>> CompressToLVQ() && {
        this->EnsureBuildUsable();
        auto operation_lock = this->AcquireExclusiveOperationLock();
        this->EnsureBuildUsable();
        this->EnsureAllVerticesBuiltWithOperationLockHeld();
        this->data_store_.Check();
        if constexpr (std::is_same_v<VecStoreType, CompressLVQVecStoreType>) {
            return std::make_unique<This>(std::move(*this));
        } else {
            using CompressedHnsw = KnnHnsw<CompressLVQVecStoreType, LabelType>;
            auto result = std::unique_ptr<CompressedHnsw>(
                new CompressedHnsw(HnswCompressionTargetTag{}, this->M_, this->ef_construction_, this->data_store_.dim()));
            auto compressed_datastore = std::move(this->data_store_).template CompressToLVQ<CompressLVQVecStoreType>();
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            this->ClearAllIncrementalReciprocalCertificates();
#endif
            result->InstallCompressedDataStore(std::move(compressed_datastore));
            return result;
        }
    }

    std::unique_ptr<KnnHnsw<CompressRabitqVecStoreType, LabelType>> CompressToRabitq() && {
        this->EnsureBuildUsable();
        auto operation_lock = this->AcquireExclusiveOperationLock();
        this->EnsureBuildUsable();
        this->EnsureAllVerticesBuiltWithOperationLockHeld();
        this->data_store_.Check();
        if constexpr (std::is_same_v<VecStoreType, CompressRabitqVecStoreType>) {
            return std::make_unique<This>(std::move(*this));
        } else {
            using CompressedHnsw = KnnHnsw<CompressRabitqVecStoreType, LabelType>;
            auto result = std::unique_ptr<CompressedHnsw>(
                new CompressedHnsw(HnswCompressionTargetTag{}, this->M_, this->ef_construction_, this->data_store_.dim()));
            auto compressed_datastore = std::move(this->data_store_).template CompressToRabitq<CompressRabitqVecStoreType>();
#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
            this->ClearAllIncrementalReciprocalCertificates();
#endif
            result->InstallCompressedDataStore(std::move(compressed_datastore));
            return result;
        }
    }
};

export template <typename VecStoreType, typename LabelType>
class KnnHnsw<VecStoreType, LabelType, false> : public KnnHnswBase<VecStoreType, LabelType, false> {
public:
    using This = KnnHnsw<VecStoreType, LabelType, false>;
    using DataStore = DataStore<VecStoreType, LabelType, false>;
    using Distance = typename VecStoreType::Distance;
    constexpr static bool kOwnMem = false;

    KnnHnsw(size_t M, size_t ef_construction, DataStore data_store, Distance distance) {
        const auto [expected_mmax0, expected_mmax] = This::GetMmax(M);
        if (data_store.Mmax0() != expected_mmax0 || data_store.Mmax() != expected_mmax) {
            throw std::invalid_argument("HNSW M does not match the stored graph capacities");
        }
        this->M_ = M;
        this->ef_construction_ = std::max(M, ef_construction);
        this->mult_ = 1 / std::log(1.0 * M);
        this->data_store_ = std::move(data_store);
        this->distance_ = std::move(distance);
    }
    KnnHnsw(This &&other) noexcept : KnnHnswBase<VecStoreType, LabelType, false>(std::move(other)) {}
    KnnHnsw &operator=(This &&other) noexcept {
        if (this != &other) {
            KnnHnswBase<VecStoreType, LabelType, false>::operator=(std::move(other));
        }
        return *this;
    }

public:
    static std::unique_ptr<This> LoadFromPtr(const char *&ptr, size_t size) {
        HnswPointerReader reader(ptr, size);
        const size_t M = reader.Read<size_t>("HNSW M");
        static_cast<void>(This::GetMmax(M));
        const size_t ef_construction = reader.Read<size_t>("HNSW ef_construction");
        if (ef_construction > static_cast<size_t>(std::numeric_limits<VertexType>::max())) {
            HnswPointerImageError("HNSW ef_construction exceeds the vertex representation");
        }
        auto data_store = DataStore::LoadFromPtr(reader);
        Distance distance(data_store.dim());
        reader.RequireEmpty();
        auto index = std::make_unique<This>(M, ef_construction, std::move(data_store), std::move(distance));
        index->Check();
        ptr = reader.current();
        return index;
    }
};

} // namespace infinity
