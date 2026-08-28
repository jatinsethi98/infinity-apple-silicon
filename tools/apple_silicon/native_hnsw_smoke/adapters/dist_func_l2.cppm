export module infinity_core:dist_func_l2;

import :hnsw_common;
import :hnsw_simd_func;
import :plain_vec_store;

import std;

export namespace infinity {

template <typename DataType>
class PlainL2Dist {
public:
    using VecStoreMeta = PlainVecStoreMeta<DataType>;
    using StoreType = typename VecStoreMeta::StoreType;
    using QueryType = typename VecStoreMeta::QueryType;
    using DistanceType = typename VecStoreMeta::DistanceType;

    PlainL2Dist() = default;
    explicit PlainL2Dist(size_t dim) : dim_(dim) {
        if constexpr (std::is_same_v<DataType, float>) {
            simd_func_ = dim % 16 == 0 ? &F32L2SSE : &F32L2SSEResidual;
#if defined(__APPLE__) && defined(__aarch64__)
            batch4_simd_func_ = dim % 16 == 0 ? &F32L2SSEBatch4 : &F32L2SSEResidualBatch4;
            batch4_threshold_simd_func_ = dim % 16 == 0 ? &F32L2SSEBatch4WithinThreshold : &F32L2SSEResidualBatch4WithinThreshold;
#endif
        }
    }
    PlainL2Dist(PlainL2Dist &&) = default;
    PlainL2Dist &operator=(PlainL2Dist &&) = default;

    template <typename DataStore>
    DistanceType operator()(const QueryType &query, VertexType vertex, const DataStore &data_store, VertexType = kInvalidVertex) const {
        const StoreType base = data_store.GetVec(vertex);
        if constexpr (std::is_same_v<DataType, float>) {
            return simd_func_(query, base, dim_);
        } else {
            DistanceType distance = 0;
            for (size_t i = 0; i < dim_; ++i) {
                const DistanceType delta = query[i] - base[i];
                distance += delta * delta;
            }
            return distance;
        }
    }

    bool SupportsBatch4() const noexcept { return batch4_simd_func_ != nullptr; }
    bool SupportsBatch4WithinThreshold() const noexcept { return batch4_threshold_simd_func_ != nullptr; }

    template <typename DataStore>
        requires std::is_same_v<DataType, float>
    void Batch4(const QueryType &query,
                const std::array<VertexType, 4> &vertices,
                const DataStore &data_store,
                std::array<DistanceType, 4> &distances) const noexcept {
        const StoreType candidate0 = data_store.GetVec(vertices[0]);
        const StoreType candidate1 = data_store.GetVec(vertices[1]);
        const StoreType candidate2 = data_store.GetVec(vertices[2]);
        const StoreType candidate3 = data_store.GetVec(vertices[3]);
        batch4_simd_func_(query, candidate0, candidate1, candidate2, candidate3, dim_, distances.data());
    }

    template <typename DataStore>
        requires std::is_same_v<DataType, float>
    std::uint8_t Batch4WithinThreshold(const QueryType &query,
                                       const std::array<VertexType, 4> &vertices,
                                       const DataStore &data_store,
                                       DistanceType threshold,
                                       std::array<DistanceType, 4> &distances) const noexcept {
        const StoreType candidate0 = data_store.GetVec(vertices[0]);
        const StoreType candidate1 = data_store.GetVec(vertices[1]);
        const StoreType candidate2 = data_store.GetVec(vertices[2]);
        const StoreType candidate3 = data_store.GetVec(vertices[3]);
        return batch4_threshold_simd_func_(query, candidate0, candidate1, candidate2, candidate3, dim_, threshold, distances.data());
    }

    PlainL2Dist ToLVQDistance(size_t) && { return PlainL2Dist(dim_); }
    PlainL2Dist ToRabitqDistance(size_t) && { return PlainL2Dist(dim_); }

private:
    using F32SIMDFunc = float (*)(const float *, const float *, size_t);
    using F32Batch4SIMDFunc = void (*)(const float *, const float *, const float *, const float *, const float *, size_t, float *);
    using F32Batch4ThresholdSIMDFunc =
        std::uint8_t (*)(const float *, const float *, const float *, const float *, const float *, size_t, float, float *);

    size_t dim_ = 0;
    F32SIMDFunc simd_func_ = nullptr;
    F32Batch4SIMDFunc batch4_simd_func_ = nullptr;
    F32Batch4ThresholdSIMDFunc batch4_threshold_simd_func_ = nullptr;
};

} // namespace infinity
