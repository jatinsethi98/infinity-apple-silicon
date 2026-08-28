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

export module infinity_core:dist_func_l2;

import :hnsw_common;
import :plain_vec_store;
import :lvq_vec_store;
import :rabitq_vec_store;
import :simd_functions;
#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
import :hnsw_lvq_capture;
#endif

import std;

namespace infinity {

export template <typename DataType, typename CompressType>
class LVQL2Dist;

export template <typename DataType>
class RabitqL2Dist;

export template <typename DataType>
class PlainLSL2Dist;

export template <typename DataType>
class PlainL2Dist {
public:
    using This = PlainL2Dist<DataType>;
    using VecStoreMeta = PlainVecStoreMeta<DataType>;
    using StoreType = typename VecStoreMeta::StoreType;
    using QueryType = typename VecStoreMeta::QueryType;
    using DistanceType = typename VecStoreMeta::DistanceType;
    using LVQDist = LVQL2Dist<DataType, i8>;
    using RabitqDist = RabitqL2Dist<DataType>;

private:
    using SIMDFuncType = std::conditional_t<std::is_same_v<DataType, float>, f32, i32> (*)(const DataType *, const DataType *, size_t);
#if defined(__APPLE__) && defined(__aarch64__)
    using Batch4SIMDFuncType = void (*)(const f32 *, const f32 *, const f32 *, const f32 *, const f32 *, size_t, f32 *);
    using Batch4ThresholdSIMDFuncType = std::uint8_t (*)(const f32 *, const f32 *, const f32 *, const f32 *, const f32 *, size_t, f32, f32 *);
#endif

    SIMDFuncType SIMDFunc = nullptr;
#if defined(__APPLE__) && defined(__aarch64__)
    Batch4SIMDFuncType Batch4SIMDFunc = nullptr;
    Batch4ThresholdSIMDFuncType Batch4ThresholdSIMDFunc = nullptr;
#endif

public:
    PlainL2Dist()
        : SIMDFunc(nullptr)
#if defined(__APPLE__) && defined(__aarch64__)
          ,
          Batch4SIMDFunc(nullptr), Batch4ThresholdSIMDFunc(nullptr)
#endif
    {
    }
    PlainL2Dist(PlainL2Dist &&other) noexcept
        : SIMDFunc(std::exchange(other.SIMDFunc, nullptr))
#if defined(__APPLE__) && defined(__aarch64__)
          ,
          Batch4SIMDFunc(std::exchange(other.Batch4SIMDFunc, nullptr)), Batch4ThresholdSIMDFunc(std::exchange(other.Batch4ThresholdSIMDFunc, nullptr))
#endif
    {
    }
    PlainL2Dist &operator=(PlainL2Dist &&other) noexcept {
        if (this != &other) {
            SIMDFunc = std::exchange(other.SIMDFunc, nullptr);
#if defined(__APPLE__) && defined(__aarch64__)
            Batch4SIMDFunc = std::exchange(other.Batch4SIMDFunc, nullptr);
            Batch4ThresholdSIMDFunc = std::exchange(other.Batch4ThresholdSIMDFunc, nullptr);
#endif
        }
        return *this;
    }
    ~PlainL2Dist() = default;

    PlainL2Dist(size_t dim) {
        if constexpr (std::is_same<DataType, float>()) {
            if (dim % 16 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_F32L2_16_ptr_;
#if defined(__APPLE__) && defined(__aarch64__)
                Batch4SIMDFunc = GetSIMD_FUNCTIONS().HNSW_F32L2_BATCH4_16_ptr_;
                Batch4ThresholdSIMDFunc = GetSIMD_FUNCTIONS().HNSW_F32L2_BATCH4_THRESHOLD_16_ptr_;
#endif
            } else {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_F32L2_ptr_;
#if defined(__APPLE__) && defined(__aarch64__)
                Batch4SIMDFunc = GetSIMD_FUNCTIONS().HNSW_F32L2_BATCH4_ptr_;
                Batch4ThresholdSIMDFunc = GetSIMD_FUNCTIONS().HNSW_F32L2_BATCH4_THRESHOLD_ptr_;
#endif
            }
        } else if constexpr (std::is_same<DataType, i8>()) {
            if (dim % 64 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8L2_64_ptr_;
            } else if (dim % 32 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8L2_32_ptr_;
            } else if (dim % 16 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8L2_16_ptr_;
            } else {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8L2_ptr_;
            }
        } else if constexpr (std::is_same<DataType, u8>()) {
            if (dim % 64 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_U8L2_64_ptr_;
            } else if (dim % 32 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_U8L2_32_ptr_;
            } else if (dim % 16 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_U8L2_16_ptr_;
            } else {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_U8L2_ptr_;
            }
        }
    }

    template <typename DataStore>
    DistanceType operator()(const QueryType &v1, VertexType v2_i, const DataStore &data_store, VertexType v1_i = kInvalidVertex) const {
        return Inner(v1, data_store.GetVec(v2_i), data_store.dim());
    }

#if defined(__APPLE__) && defined(__aarch64__)
    bool SupportsBatch4() const noexcept { return Batch4SIMDFunc != nullptr; }
    bool SupportsBatch4WithinThreshold() const noexcept { return Batch4ThresholdSIMDFunc != nullptr; }

    template <typename DataStore>
        requires std::is_same_v<DataType, f32>
    void Batch4(const QueryType &query,
                const std::array<VertexType, 4> &vertices,
                const DataStore &data_store,
                std::array<DistanceType, 4> &distances) const noexcept {
        const std::array<StoreType, 4> candidates{
            data_store.GetVec(vertices[0]),
            data_store.GetVec(vertices[1]),
            data_store.GetVec(vertices[2]),
            data_store.GetVec(vertices[3]),
        };
        const size_t dim = data_store.dim();
        if (Batch4SIMDFunc != nullptr) {
            Batch4SIMDFunc(query, candidates[0], candidates[1], candidates[2], candidates[3], dim, distances.data());
            return;
        }
        for (size_t lane = 0; lane < candidates.size(); ++lane) {
            DistanceType distance = 0;
            for (size_t index = 0; index < dim; ++index) {
                const DistanceType delta = query[index] - candidates[lane][index];
                distance += delta * delta;
            }
            distances[lane] = distance;
        }
    }

    template <typename DataStore>
        requires std::is_same_v<DataType, f32>
    std::uint8_t Batch4WithinThreshold(const QueryType &query,
                                       const std::array<VertexType, 4> &vertices,
                                       const DataStore &data_store,
                                       DistanceType threshold,
                                       std::array<DistanceType, 4> &distances) const noexcept {
        const std::array<StoreType, 4> candidates{
            data_store.GetVec(vertices[0]),
            data_store.GetVec(vertices[1]),
            data_store.GetVec(vertices[2]),
            data_store.GetVec(vertices[3]),
        };
        if (Batch4ThresholdSIMDFunc != nullptr) {
            return Batch4ThresholdSIMDFunc(query,
                                           candidates[0],
                                           candidates[1],
                                           candidates[2],
                                           candidates[3],
                                           data_store.dim(),
                                           threshold,
                                           distances.data());
        }
        Batch4(query, vertices, data_store, distances);
        return 0x0f;
    }
#endif

    LVQDist ToLVQDistance(size_t dim) &&;

    RabitqDist ToRabitqDistance(size_t dim) &&;

private:
    DistanceType Inner(const QueryType &v1, const StoreType &v2, size_t dim) const { return SIMDFunc(v1, v2, dim); }
};

export template <typename DataType, typename CompressType>
class LVQL2Cache {
public:
    // for l2 distance, const1 = scale * norm1(compress), const2 = scale * scale * norm2(compress)
    using LocalCacheType = std::pair<DataType, DataType>;
    using GlobalCacheType = std::tuple<>;

    static LocalCacheType MakeLocalCache(const CompressType *c, DataType scale, size_t dim, const MeanType *) {
        i64 norm1 = 0;
        i64 norm2 = 0;
        for (size_t i = 0; i < dim; ++i) {
            norm1 += c[i];
            norm2 += c[i] * c[i];
        }
        return {norm1 * scale, norm2 * scale * scale};
    }

    static GlobalCacheType MakeGlobalCache(const MeanType *, size_t) { return {}; }

public:
    static void DumpLocalCache(std::ostream &os, const LocalCacheType &local_cache) {
        os << "norm1_scale: " << local_cache.first << ", norm2sq_scalesq: " << local_cache.second << std::endl;
    }

    static void DumpGlobalCache(std::ostream &, const GlobalCacheType &) {}
};

export template <typename DataType, typename CompressType>
class LVQL2Dist {
public:
    using This = LVQL2Dist<DataType, CompressType>;
    using LVQDist = This;
    using RabitqDist = This;
    using VecStoreMetaType = LVQVecStoreMetaType<DataType, CompressType, LVQL2Cache<DataType, CompressType>>;
    using StoreType = typename VecStoreMetaType::StoreType;
    using QueryType = typename VecStoreMetaType::QueryType;
    using DistanceType = typename VecStoreMetaType::DistanceType;

private:
    using SIMDFuncType = i32 (*)(const CompressType *, const CompressType *, size_t);

    SIMDFuncType SIMDFunc = nullptr;

public:
    LVQL2Dist() : SIMDFunc(nullptr) {}
    LVQL2Dist(LVQL2Dist &&other) noexcept : SIMDFunc(std::exchange(other.SIMDFunc, nullptr)) {}
    LVQL2Dist &operator=(LVQL2Dist &&other) noexcept {
        if (this != &other) {
            SIMDFunc = std::exchange(other.SIMDFunc, nullptr);
        }
        return *this;
    }
    ~LVQL2Dist() = default;
    LVQL2Dist(size_t dim) {
        if constexpr (std::is_same<CompressType, i8>()) {
            if (dim % 64 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8IP_64_ptr_;
            } else if (dim % 32 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8IP_32_ptr_;
            } else if (dim % 16 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8IP_16_ptr_;
            } else {
                SIMDFunc = GetSIMD_FUNCTIONS().HNSW_I8IP_ptr_;
            }
        }
    }

    template <typename DataStore>
    DistanceType operator()(const QueryType &v1, VertexType v2_i, const DataStore &data_store, VertexType v1_i = kInvalidVertex) const {
        const StoreType &v2 = data_store.GetVec(v2_i);
        size_t dim = data_store.dim();
        const DistanceType result = Inner(v1, v2, dim);
#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
        constexpr size_t header_bytes = sizeof(typename VecStoreMetaType::LVQData);
        const auto *query_record = reinterpret_cast<const std::byte *>(v1->compress_vec_) - header_bytes;
        const auto *candidate_record = reinterpret_cast<const std::byte *>(v2->compress_vec_) - header_bytes;
        HnswObserveLvqL2(
            query_record, candidate_record, header_bytes + dim * sizeof(CompressType), dim, result, v1_i, v2_i);
#endif
        return result;
    }

private:
    DistanceType Inner(const QueryType &v1, const StoreType &v2, size_t dim) const {
        i32 c1c2_ip = SIMDFunc(v1->compress_vec_, v2->compress_vec_, dim);
        auto scale1 = v1->scale_;
        auto scale2 = v2->scale_;
        auto beta = v1->bias_ - v2->bias_;

        auto [norm1_scale_1, norm2sq_scalesq_1] = v1->local_cache_;
        auto [norm1_scale_2, norm2sq_scalesq_2] = v2->local_cache_;
        return norm2sq_scalesq_1 + norm2sq_scalesq_2 + beta * beta * dim - 2 * scale1 * scale2 * c1c2_ip + 2 * beta * norm1_scale_1 -
               2 * beta * norm1_scale_2;
    }
};

export template <typename DataType>
class RabitqL2Dist {
public:
    using This = RabitqL2Dist<DataType>;
    using LVQDist = This;
    using RabitqDist = This;
    using MetaType = RabitqVecStoreMetaType<DataType>;
    using StoreType = typename MetaType::StoreType;
    using QueryType = typename MetaType::QueryType;
    using DistanceType = typename MetaType::DistanceType;
    using CompressType = typename MetaType::CompressType;
    using AlignType = typename MetaType::AlignType;

private:
    using SIMDFuncType = i32 (*)(const CompressType *, const AlignType *, size_t);

    SIMDFuncType SIMDFunc = nullptr;

public:
    RabitqL2Dist() : SIMDFunc(nullptr) {}
    RabitqL2Dist(RabitqL2Dist &&other) noexcept : SIMDFunc(std::exchange(other.SIMDFunc, nullptr)) {}
    RabitqL2Dist &operator=(RabitqL2Dist &&other) noexcept {
        if (this != &other) {
            SIMDFunc = std::exchange(other.SIMDFunc, nullptr);
        }
        return *this;
    }
    ~RabitqL2Dist() = default;
    RabitqL2Dist(size_t dim) {
        if constexpr (std::is_same<CompressType, u8>() && std::is_same<AlignType, u8>()) {
            if (dim % 64 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().Rabitq_U8IP_64_ptr_;
            } else if (dim % 32 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().Rabitq_U8IP_32_ptr_;
            } else if (dim % 16 == 0) {
                SIMDFunc = GetSIMD_FUNCTIONS().Rabitq_U8IP_16_ptr_;
            } else {
                SIMDFunc = GetSIMD_FUNCTIONS().Rabitq_U8IP_ptr_;
            }
        }
    }

    template <typename DataStore>
    DistanceType operator()(const QueryType &v1, VertexType v2_i, const DataStore &data_store, VertexType v1_i = kInvalidVertex) const {
        const StoreType &v2 = data_store.GetVec(v2_i);
        size_t dim = data_store.dim();
        return Inner(v1, v2, dim);
    }

private:
    DistanceType Inner(const QueryType &query, const StoreType &base, size_t dim) const {
        // estimate <x, q>
        // DistanceType ip_estimate = MetaType::IpDistanceBetweenQueryAndBinaryCode(query->query_compress_vec_, base->compress_vec_, dim);
        DistanceType ip_estimate = SIMDFunc(query->query_compress_vec_, base->compress_vec_, dim);
        DistanceType ip_recover =
            MetaType::RecoverIpDistance(ip_estimate, dim, base->sum_, query->query_sum_, query->query_lower_bound_, query->query_delta_);

        // estimate ||o_r, q_r||^2
        return MetaType::RecoverL2DistanceSqr(ip_recover / base->error_, base->norm_, query->query_norm_);
    }
};

template <typename DataType>
LVQL2Dist<DataType, i8> PlainL2Dist<DataType>::ToLVQDistance(size_t dim) && {
    return LVQL2Dist<DataType, i8>(dim);
}

template <typename DataType>
RabitqL2Dist<DataType> PlainL2Dist<DataType>::ToRabitqDistance(size_t dim) && {
    return RabitqL2Dist<DataType>(dim);
}

} // namespace infinity
