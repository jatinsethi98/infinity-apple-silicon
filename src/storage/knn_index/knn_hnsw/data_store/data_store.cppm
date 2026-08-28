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

export module infinity_core:data_store;

import :hnsw_common;
import :local_file_handle;
import :vec_store_type;
import :graph_store;
import :infinity_exception;
import :data_store_util;
import :plain_vec_store;
export import :spinlock;

import std;

import serialize;

namespace infinity {

#if defined(__APPLE__) && defined(__aarch64__) && defined(INFINITY_ENABLE_APPLE_HNSW_COMPACT_VERTEX_LOCKS)
export using HnswVertexMutex = SpinLock;
#else
export using HnswVertexMutex = std::shared_mutex;
#endif
export using HnswVertexSharedLock = std::shared_lock<HnswVertexMutex>;
export using HnswVertexUniqueLock = std::unique_lock<HnswVertexMutex>;

template <typename VecStoreT, typename LabelType, bool OwnMem>
class DataStoreInner;

export template <typename VecStoreT, typename LabelType>
class DataStoreChunkIter;

template <typename VecStoreT, typename LabelType>
class DataStoreIter;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"

export template <typename VecStoreT, typename LabelType, bool OwnMem>
class DataStoreBase {
public:
    using This = DataStoreBase<VecStoreT, LabelType, OwnMem>;
    using QueryVecType = typename VecStoreT::QueryVecType;
    using VecStoreMeta = typename VecStoreT::template Meta<OwnMem>;

public:
    template <typename T, typename = void>
    struct has_compress_type : std::false_type {};

    template <typename T>
    struct has_compress_type<T, std::void_t<typename T::CompressType>> : std::true_type {};

    DataStoreBase() = default;
    DataStoreBase(VecStoreMeta &&vec_store_meta, GraphStoreMeta &&graph_store_meta)
        : vec_store_meta_(std::move(vec_store_meta)), graph_store_meta_(std::move(graph_store_meta)) {}
    DataStoreBase(This &&other) noexcept
        : vec_store_meta_(std::move(other.vec_store_meta_)), graph_store_meta_(std::move(other.graph_store_meta_)) {
        static_assert(std::is_nothrow_move_constructible_v<VecStoreMeta>);
        static_assert(std::is_nothrow_move_constructible_v<GraphStoreMeta>);
    }
    DataStoreBase &operator=(This &&other) noexcept {
        static_assert(std::is_nothrow_move_assignable_v<VecStoreMeta>);
        static_assert(std::is_nothrow_move_assignable_v<GraphStoreMeta>);
        if (this != &other) {
            vec_store_meta_ = std::move(other.vec_store_meta_);
            graph_store_meta_ = std::move(other.graph_store_meta_);
        }
        return *this;
    }
    // ~DataStoreBase() = default;

    typename VecStoreT::QueryType MakeQuery(QueryVecType query) const { return vec_store_meta_.MakeQuery(query); }

    const VecStoreMeta &vec_store_meta() const { return vec_store_meta_; }

    size_t dim() const { return vec_store_meta_.dim(); }

    // Graph store
    std::pair<i32, VertexType> GetEnterPoint() const { return graph_store_meta_.GetEnterPoint(); }

    size_t Mmax0() const { return graph_store_meta_.Mmax0(); }
    size_t Mmax() const { return graph_store_meta_.Mmax(); }

protected:
    static void ValidatePointerImageEntryPoint(const GraphStoreMeta &graph_store_meta, size_t cur_vec_num) {
        const auto [max_layer, entry_point] = graph_store_meta.GetEnterPoint();
        if (cur_vec_num == 0) {
            if (max_layer != -1 || entry_point != kInvalidVertex) {
                HnswPointerImageError("empty graph has an entry point");
            }
            return;
        }
        if (entry_point < 0 || static_cast<size_t>(entry_point) >= cur_vec_num) {
            HnswPointerImageError("graph entry point is out of range");
        }
    }

    template <typename Store>
    static void CheckGraphTopology(const Store &store, size_t cur_vec_num, i32 observed_max_layer) {
        const auto [max_layer, entry_point] = store.GetEnterPoint();
        if (observed_max_layer != max_layer) {
            UnrecoverableError("max_l != max_layer");
        }
        if (cur_vec_num == 0) {
            if (max_layer != -1 || entry_point != kInvalidVertex) {
                UnrecoverableError("Empty HNSW graph has an entry point");
            }
            return;
        }
        if (entry_point < 0 || size_t(entry_point) >= cur_vec_num) {
            UnrecoverableError("HNSW graph entry point is out of range");
        }
        if (store.GetLevel(entry_point) != max_layer) {
            UnrecoverableError("HNSW graph entry point is not on the maximum level");
        }

        for (VertexType vertex = 0; size_t(vertex) < cur_vec_num; ++vertex) {
            const LayerSize level = store.GetLevel(vertex);
            for (i32 layer = 1; layer <= level; ++layer) {
                const auto [neighbors, degree] = store.GetNeighbors(vertex, layer);
                for (VertexListSize index = 0; index < degree; ++index) {
                    if (store.GetLevel(neighbors[index]) < layer) {
                        UnrecoverableError("HNSW upper-layer edge targets a lower-level vertex");
                    }
                }
            }
        }

        std::vector<bool> visited(cur_vec_num, false);
        std::vector<VertexType> pending;
        pending.reserve(cur_vec_num);
        pending.push_back(entry_point);
        visited[size_t(entry_point)] = true;
        for (size_t cursor = 0; cursor < pending.size(); ++cursor) {
            const auto [neighbors, degree] = store.GetNeighbors(pending[cursor], 0);
            for (VertexListSize index = 0; index < degree; ++index) {
                const VertexType neighbor = neighbors[index];
                if (!visited[size_t(neighbor)]) {
                    visited[size_t(neighbor)] = true;
                    pending.push_back(neighbor);
                }
            }
        }
        if (pending.size() == cur_vec_num) {
            return;
        }

        std::vector<size_t> indegrees(cur_vec_num, 0);
        for (VertexType source = 0; size_t(source) < cur_vec_num; ++source) {
            const auto [neighbors, degree] = store.GetNeighbors(source, 0);
            for (VertexListSize index = 0; index < degree; ++index) {
                ++indegrees[size_t(neighbors[index])];
            }
        }
        const auto first_unreachable = std::find(visited.begin(), visited.end(), false) - visited.begin();
        const auto [unreachable_neighbors, unreachable_degree] =
            store.GetNeighbors(static_cast<VertexType>(first_unreachable), 0);
        static_cast<void>(unreachable_neighbors);
        UnrecoverableError("HNSW level-zero graph is not reachable from the entry point: reached " +
                           std::to_string(pending.size()) + "/" + std::to_string(cur_vec_num) +
                           ", first unreachable vertex " + std::to_string(first_unreachable) +
                           " has indegree " + std::to_string(indegrees[first_unreachable]) +
                           " and outdegree " + std::to_string(unreachable_degree));
    }

    VecStoreMeta vec_store_meta_;
    GraphStoreMeta graph_store_meta_;
};

export template <typename VecStoreT, typename LabelType, bool OwnMem = true>
class DataStore : public DataStoreBase<VecStoreT, LabelType, OwnMem> {
public:
    using This = DataStore<VecStoreT, LabelType, OwnMem>;
    using Base = DataStoreBase<VecStoreT, LabelType, OwnMem>;
    using DataType = typename VecStoreT::DataType;
    using QueryVecType = typename VecStoreT::QueryVecType;
    using Inner = DataStoreInner<VecStoreT, LabelType, OwnMem>;
    using VecStoreMeta = typename VecStoreT::template Meta<OwnMem>;
    using VecStoreInner = typename VecStoreT::template Inner<OwnMem>;

    friend class DataStoreChunkIter<VecStoreT, LabelType>;
    friend class DataStoreIter<VecStoreT, LabelType>;

private:
    DataStore(size_t chunk_size, size_t max_chunk_n, VecStoreMeta &&vec_store_meta, GraphStoreMeta &&graph_store_meta)
        : Base(std::move(vec_store_meta), std::move(graph_store_meta)), chunk_size_(chunk_size), max_chunk_n_(max_chunk_n),
          chunk_shift_(__builtin_ctzll(chunk_size)), inners_(std::make_unique<Inner[]>(max_chunk_n)), mem_usage_(0) {
        assert(chunk_size > 0);
        assert((chunk_size & (chunk_size - 1)) == 0);
        cur_vec_num_ = 0;
    }

public:
    DataStore() = default;
    static constexpr size_t OwnedMetadataBytesPerVertex() requires OwnMem {
        return sizeof(LabelType) + sizeof(HnswVertexMutex);
    }

    DataStore(DataStore &&other) noexcept : Base(std::move(other)) {
        chunk_size_ = std::exchange(other.chunk_size_, 0);
        max_chunk_n_ = std::exchange(other.max_chunk_n_, 0);
        chunk_shift_ = std::exchange(other.chunk_shift_, 0);
        cur_vec_num_ = other.cur_vec_num_.exchange(0);
        built_vec_num_ = other.built_vec_num_.exchange(0);
        inners_ = std::exchange(other.inners_, nullptr);
        mem_usage_ = other.mem_usage_.exchange(0);
    }
    DataStore &operator=(DataStore &&other) noexcept {
        if (this != &other) {
            FreeInners();
            Base::operator=(std::move(other));
            chunk_size_ = std::exchange(other.chunk_size_, 0);
            max_chunk_n_ = std::exchange(other.max_chunk_n_, 0);
            chunk_shift_ = std::exchange(other.chunk_shift_, 0);
            cur_vec_num_ = other.cur_vec_num_.exchange(0);
            built_vec_num_ = other.built_vec_num_.exchange(0);
            inners_ = std::exchange(other.inners_, nullptr);
            mem_usage_ = other.mem_usage_.exchange(0);
        }
        return *this;
    }
    ~DataStore() { FreeInners(); }

private:
    void FreeInners() noexcept {
        if (!inners_) {
            return;
        }
        size_t cur_vec_num = this->cur_vec_num();
        auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t chunk_size = (i < chunk_num - 1) ? chunk_size_ : last_chunk_size;
            inners_[i].Free(chunk_size, this->graph_store_meta_);
        }
        inners_.reset();
    }

public:
    static This Make(size_t chunk_size, size_t max_chunk_n, size_t dim, size_t Mmax0, size_t Mmax) {
        bool normalize = false;
        if constexpr (Base::template has_compress_type<VecStoreT>::value) {
            normalize = std::is_same_v<VecStoreMeta, typename LVQCosVecStoreType<DataType, typename VecStoreT::CompressType>::template Meta<OwnMem>>;
        }
        VecStoreMeta vec_store_meta = VecStoreMeta::Make(dim, normalize);
        GraphStoreMeta graph_store_meta = GraphStoreMeta::Make(Mmax0, Mmax);
        This ret(chunk_size, max_chunk_n, std::move(vec_store_meta), std::move(graph_store_meta));
        ret.cur_vec_num_ = 0;
        ret.built_vec_num_ = 0;

        size_t mem_usage = 0;
        ret.inners_[0] = Inner::Make(chunk_size, ret.vec_store_meta_, ret.graph_store_meta_, mem_usage);
        ret.mem_usage_.store(mem_usage);
        return ret;
    }

    void Save(LocalFileHandle &file_handle) const {
        EnsureAllVerticesBuilt();
        size_t cur_vec_num = this->cur_vec_num();
        auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);

        file_handle.Append(&chunk_size_, sizeof(chunk_size_));
        file_handle.Append(&max_chunk_n_, sizeof(max_chunk_n_));

        file_handle.Append(&cur_vec_num, sizeof(cur_vec_num));
        this->vec_store_meta_.Save(file_handle);
        this->graph_store_meta_.Save(file_handle, cur_vec_num);
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t chunk_size = (i < chunk_num - 1) ? chunk_size_ : last_chunk_size;
            inners_[i].Save(file_handle, chunk_size, this->vec_store_meta_, this->graph_store_meta_);
        }
    }

    void SaveToPtr(LocalFileHandle &file_handle) const {
        EnsureAllVerticesBuilt();
        size_t cur_vec_num = this->cur_vec_num();

        file_handle.Append(&cur_vec_num, sizeof(cur_vec_num));
        this->vec_store_meta_.Save(file_handle);
        this->graph_store_meta_.Save(file_handle, cur_vec_num);

        auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);
        Inner::SaveToPtr(file_handle, inners_.get(), this->vec_store_meta_, this->graph_store_meta_, chunk_size_, chunk_num, last_chunk_size);
    }

    static This Load(LocalFileHandle &file_handle,
                     size_t max_chunk_n = 0,
                     std::optional<std::pair<size_t, size_t>> expected_graph_capacities = std::nullopt) {
        const size_t chunk_size = HnswReadStream<size_t>(file_handle, "chunk size");
        const size_t max_chunk_n1 = HnswReadStream<size_t>(file_handle, "chunk count");
        if (chunk_size == 0 || !std::has_single_bit(chunk_size)) {
            HnswStreamError("chunk size must be a nonzero power of two");
        }
        if (max_chunk_n1 == 0) {
            HnswStreamError("chunk count must be nonzero");
        }
        const size_t serialized_capacity = HnswStreamCheckedMultiply(chunk_size, max_chunk_n1, "vector capacity");
        if (serialized_capacity > static_cast<size_t>(std::numeric_limits<VertexType>::max())) {
            HnswStreamError("vector capacity exceeds the vertex representation");
        }
        if (max_chunk_n == 0) {
            max_chunk_n = max_chunk_n1;
        }
        if (max_chunk_n < max_chunk_n1) {
            HnswStreamError("requested chunk count is smaller than the serialized capacity");
        }
        const size_t load_capacity = HnswStreamCheckedMultiply(chunk_size, max_chunk_n, "loaded vector capacity");
        if (load_capacity > static_cast<size_t>(std::numeric_limits<VertexType>::max())) {
            HnswStreamError("loaded vector capacity exceeds the vertex representation");
        }

        const size_t cur_vec_num = HnswReadStream<size_t>(file_handle, "vector count");
        if (cur_vec_num > serialized_capacity) {
            HnswStreamError("vector count exceeds the serialized capacity");
        }
        VecStoreMeta vec_store_meta = VecStoreMeta::Load(file_handle);
        GraphStoreMeta graph_store_meta = GraphStoreMeta::Load(file_handle, expected_graph_capacities);

        This ret = This(chunk_size, max_chunk_n, std::move(vec_store_meta), std::move(graph_store_meta));

        size_t mem_usage = 0;
        auto [chunk_num, last_chunk_size] = ret.ChunkInfo(cur_vec_num);
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t cur_chunk_size = (i < chunk_num - 1) ? chunk_size : last_chunk_size;
            ret.inners_[i] = Inner::Load(file_handle,
                                         cur_chunk_size,
                                         chunk_size,
                                         ret.vec_store_meta_,
                                         ret.graph_store_meta_,
                                         mem_usage,
                                         i * chunk_size,
                                         cur_vec_num);
        }
        ret.mem_usage_.store(mem_usage);
        ret.cur_vec_num_.store(cur_vec_num);
        if (expected_graph_capacities) {
            ret.ValidateLoadedStreamGraph();
        }
        ret.built_vec_num_.store(cur_vec_num);
        return ret;
    }

    static This LoadFromPtr(HnswPointerReader &reader) {
        const size_t cur_vec_num = reader.Read<size_t>("vector count");
        if (cur_vec_num > static_cast<size_t>(std::numeric_limits<VertexType>::max())) {
            HnswPointerImageError("vector count exceeds the vertex representation");
        }
        VecStoreMeta vec_store_meta = VecStoreMeta::LoadFromPtr(reader);
        GraphStoreMeta graph_store_meta = GraphStoreMeta::LoadFromPtr(reader);
        Base::ValidatePointerImageEntryPoint(graph_store_meta, cur_vec_num);

        const size_t chunk_size = cur_vec_num == 0 ? 1 : std::bit_ceil(cur_vec_num);
        This ret = This(chunk_size, 1 /*max_chunk_n*/, std::move(vec_store_meta), std::move(graph_store_meta));

        size_t mem_usage = 0;
        ret.inners_[0] = Inner::LoadFromPtr(reader, cur_vec_num, chunk_size, ret.vec_store_meta_, ret.graph_store_meta_, mem_usage);
        ret.cur_vec_num_ = cur_vec_num;
        ret.mem_usage_.store(mem_usage);
        size_t built_vec_num = 0;
        for (VertexType vertex = 0; size_t(vertex) < cur_vec_num; ++vertex) {
            built_vec_num += ret.IsVertexBuilt(vertex);
        }
        ret.built_vec_num_.store(built_vec_num);
        return ret;
    }

    void ValidateGraphAttachment(size_t graph_inner_count, size_t built_vec_num) const {
        const auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num());
        static_cast<void>(last_chunk_size);
        if (graph_inner_count != chunk_num || built_vec_num > cur_vec_num()) {
            throw std::logic_error("Invalid HNSW graph attachment");
        }
    }

    void SetGraph(GraphStoreMeta &&graph_meta,
                  std::vector<GraphStoreInner<OwnMem>> &&graph_inners,
                  size_t built_vec_num,
                  size_t upper_layer_mem_usage) noexcept {
        static_assert(std::is_nothrow_move_assignable_v<GraphStoreMeta>);
        static_assert(std::is_nothrow_move_assignable_v<GraphStoreInner<OwnMem>>);
        this->graph_store_meta_ = std::move(graph_meta);
        for (size_t i = 0; i < graph_inners.size(); ++i) {
            inners_[i].SetGraphStoreInner(std::move(graph_inners[i]));
        }
        built_vec_num_.store(built_vec_num, std::memory_order_release);
        mem_usage_.fetch_add(upper_layer_mem_usage, std::memory_order_relaxed);
    }

    size_t GetSizeInBytes() const {
        size_t cur_vec_num = this->cur_vec_num();
        auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);

        size_t size = 0;
        size += sizeof(chunk_size_);
        size += sizeof(max_chunk_n_);
        size += sizeof(cur_vec_num_);
        size += sizeof(built_vec_num_);
        size += this->vec_store_meta_.GetSizeInBytes();
        size += this->graph_store_meta_.GetSizeInBytes();
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t chunk_size = (i < chunk_num - 1) ? chunk_size_ : last_chunk_size;
            size += inners_[i].GetSizeInBytes(chunk_size, this->vec_store_meta_, this->graph_store_meta_);
        }
        return size;
    }

    // Vec store
    template <DataIteratorConcept<QueryVecType, LabelType> Iterator>
    std::pair<size_t, size_t> AddVec(Iterator &&query_iter) {
        size_t mem_usage = 0;
        size_t cur_vec_num = this->cur_vec_num();
        size_t start_idx = cur_vec_num;
        auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);
        while (true) {
            size_t remain_size = chunk_size_ - last_chunk_size;
            auto [insert_n, used_up] =
                inners_[chunk_num - 1].AddVec(std::forward<Iterator>(query_iter), last_chunk_size, remain_size, this->vec_store_meta_, mem_usage);
            cur_vec_num += insert_n;
            last_chunk_size += insert_n;
            if (cur_vec_num == max_chunk_n_ * chunk_size_) {
                break;
            }
            if (last_chunk_size == chunk_size_) {
                inners_[chunk_num++] = Inner::Make(chunk_size_, this->vec_store_meta_, this->graph_store_meta_, mem_usage);
                last_chunk_size = 0;
            }
            if (used_up) {
                break;
            }
        }
        cur_vec_num_.store(cur_vec_num);
        mem_usage_.fetch_add(mem_usage);
        return {start_idx, cur_vec_num};
    }

    template <DataIteratorConcept<QueryVecType, LabelType> Iterator>
    std::pair<size_t, size_t> OptAddVec(Iterator &&query_iter) {
        if constexpr (VecStoreT::HasOptimize) {
            size_t mem_usage = 0;
            size_t cur_vec_num = this->cur_vec_num();
            auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);
            if (chunk_num > 0) {
                std::vector<std::pair<VecStoreInner *, size_t>> vec_inners;
                for (size_t i = 0; i < chunk_num; ++i) {
                    size_t chunk_size = (i < chunk_num - 1) ? chunk_size_ : last_chunk_size;
                    vec_inners.emplace_back(inners_[i].vec_store_inner(), chunk_size);
                }
                Iterator query_iter_copy = query_iter;
                this->vec_store_meta_.template Optimize<LabelType, Iterator>(std::move(query_iter_copy), vec_inners, mem_usage);
            }
            mem_usage_.fetch_add(mem_usage);
        }
        return AddVec(std::move(query_iter));
    }

    void Optimize() {
        if constexpr (!VecStoreT::HasOptimize) {
            return;
        }
        DenseVectorIter<DataType, LabelType> empty_iter(nullptr, this->dim(), 0);
        AddVec(std::move(empty_iter));
    }

    void PrefetchVec(size_t vec_i) const {
        const auto &[inner, idx] = GetInner(vec_i);
        inner.PrefetchVec(idx, this->vec_store_meta_);
    }

    typename VecStoreT::StoreType GetVec(size_t vec_i) const {
        const auto &[inner, idx] = GetInner(vec_i);
        return inner.GetVec(idx, this->vec_store_meta_);
    }

    typename VecStoreT::QueryType GetVecToQuery(size_t vec_i) const {
        const auto &[inner, idx] = GetInner(vec_i);
        return inner.GetVecToQuery(idx, this->vec_store_meta_);
    }

    // Graph store
    void AddVertex(VertexType vec_i, i32 layer_n) {
        auto [inner, idx] = GetInner(vec_i);
        size_t mem_usage = 0;
        inner.AddVertex(idx, layer_n, this->graph_store_meta_, mem_usage);
        mem_usage_.fetch_add(mem_usage);
        built_vec_num_.fetch_add(1, std::memory_order_release);
    }

    std::pair<VertexType *, VertexListSize *> GetNeighborsMut(VertexType vertex_i, i32 layer_i) {
        auto [inner, idx] = GetInner(vertex_i);
        return inner.GetNeighborsMut(idx, layer_i, this->graph_store_meta_);
    }

    std::pair<const VertexType *, VertexListSize> GetNeighbors(VertexType vertex_i, i32 layer_i) const {
        const auto &[inner, idx] = GetInner(vertex_i);
        return inner.GetNeighbors(idx, layer_i, this->graph_store_meta_);
    }

    LayerSize GetLevel(VertexType vertex_i) const {
        const auto &[inner, idx] = GetInner(vertex_i);
        return inner.GetLevel(idx, this->graph_store_meta_);
    }

    bool IsVertexBuilt(VertexType vertex_i) const {
        const auto &[inner, idx] = GetInner(vertex_i);
        return inner.IsVertexBuilt(idx, this->graph_store_meta_);
    }

    bool AllVerticesBuilt() const {
        if (this->Mmax0() == 0 && this->Mmax() == 0) {
            return true;
        }
        return built_vec_num_.load(std::memory_order_acquire) == cur_vec_num_.load(std::memory_order_acquire);
    }

    size_t built_vec_num() const { return built_vec_num_.load(std::memory_order_acquire); }

    void EnsureAllVerticesBuilt() const {
        if (!AllVerticesBuilt()) {
            throw std::logic_error("HNSW index contains stored but unbuilt vertices");
        }
    }

    std::pair<i32, VertexType> TryUpdateEnterPoint(i32 layer, VertexType vertex_i) {
        return this->graph_store_meta_.TryUpdateEnterPoint(layer, vertex_i);
    }

    // other
    LabelType GetLabel(size_t vec_i) const {
        const auto &[inner, idx] = GetInner(vec_i);
        return inner.GetLabel(idx);
    }

    HnswVertexSharedLock SharedLock(size_t vec_i) const {
        const auto &[inner, idx] = GetInner(vec_i);
        return inner.SharedLock(idx);
    }

    HnswVertexUniqueLock UniqueLock(size_t vec_i) {
        const auto &[inner, idx] = GetInner(vec_i);
        return inner.UniqueLock(idx);
    }

    size_t cur_vec_num() const { return cur_vec_num_.load(); }

    size_t mem_usage() const { return mem_usage_.load(); }

    template <typename CompressVecStoreType>
    DataStore<CompressVecStoreType, LabelType, OwnMem> CompressToLVQ() &&;

    template <typename CompressVecStoreType>
    DataStore<CompressVecStoreType, LabelType, OwnMem> CompressToRabitq() &&;

private:
    void ValidateLoadedStreamGraph() const {
        const size_t vertex_count = cur_vec_num();
        i32 observed_max_layer = -1;
        for (VertexType vertex = 0; static_cast<size_t>(vertex) < vertex_count; ++vertex) {
            const LayerSize level = GetLevel(vertex);
            if (level < 0 || level > kHnswMaxSupportedLayer) {
                HnswStreamError("graph contains an invalid vertex level");
            }
            observed_max_layer = std::max(observed_max_layer, level);
        }

        const auto [max_layer, entry_point] = this->GetEnterPoint();
        if (max_layer != observed_max_layer) {
            HnswStreamError("graph maximum layer does not match its vertices");
        }
        if (vertex_count == 0) {
            if (max_layer != -1 || entry_point != kInvalidVertex) {
                HnswStreamError("empty graph has an entry point");
            }
            return;
        }
        if (entry_point < 0 || static_cast<size_t>(entry_point) >= vertex_count || GetLevel(entry_point) != max_layer) {
            HnswStreamError("graph entry point is invalid");
        }
        for (VertexType vertex = 0; static_cast<size_t>(vertex) < vertex_count; ++vertex) {
            const LayerSize level = GetLevel(vertex);
            for (LayerSize layer = 1; layer <= level; ++layer) {
                const auto [neighbors, degree] = GetNeighbors(vertex, layer);
                for (VertexListSize index = 0; index < degree; ++index) {
                    if (GetLevel(neighbors[index]) < layer) {
                        HnswStreamError("upper-layer edge targets a lower-level vertex");
                    }
                }
            }
        }

        std::vector<bool> visited(vertex_count, false);
        std::vector<VertexType> pending;
        pending.reserve(vertex_count);
        pending.push_back(entry_point);
        visited[static_cast<size_t>(entry_point)] = true;
        for (size_t cursor = 0; cursor < pending.size(); ++cursor) {
            const auto [neighbors, degree] = GetNeighbors(pending[cursor], 0);
            for (VertexListSize index = 0; index < degree; ++index) {
                const VertexType neighbor = neighbors[index];
                if (!visited[static_cast<size_t>(neighbor)]) {
                    visited[static_cast<size_t>(neighbor)] = true;
                    pending.push_back(neighbor);
                }
            }
        }
        if (pending.size() != vertex_count) {
            HnswStreamError("graph level-zero topology is disconnected from its entry point");
        }
    }

    std::pair<Inner &, size_t> GetInner(size_t vec_i) { return {inners_[vec_i >> chunk_shift_], vec_i & (chunk_size_ - 1)}; }

    std::pair<const Inner &, size_t> GetInner(size_t vec_i) const { return {inners_[vec_i >> chunk_shift_], vec_i & (chunk_size_ - 1)}; }

    // return chunk_num & last chunk size
    std::pair<size_t, size_t> ChunkInfo(size_t cur_vec_num) const {
        size_t chunk_num = std::min(max_chunk_n_, (cur_vec_num >> chunk_shift_) + 1);
        assert(chunk_num > 0);
        size_t last_chunk_size = cur_vec_num - ((chunk_num - 1) << chunk_shift_);
        return {chunk_num, last_chunk_size};
    };

private:
    size_t chunk_size_;
    size_t max_chunk_n_;
    size_t chunk_shift_;

    std::atomic<size_t> cur_vec_num_;
    std::atomic<size_t> built_vec_num_{0};

    std::unique_ptr<Inner[]> inners_;
    std::atomic<size_t> mem_usage_ = 0;

public:
    void Check() const {
        i32 max_l = -1;
        size_t i;
        size_t cur_vec_num = this->cur_vec_num();
        auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);
        for (i = 0; i < chunk_num; ++i) {
            i32 max_l1 = -1;
            size_t chunk_size = i < chunk_num - 1 ? chunk_size_ : last_chunk_size;
            inners_[i].Check(chunk_size, this->graph_store_meta_, i * chunk_size_, cur_vec_num, max_l1);
            max_l = std::max(max_l, max_l1);
        }
        Base::CheckGraphTopology(*this, cur_vec_num, max_l);
    }

    void Dump(std::ostream &os) const {
        size_t cur_vec_num = this->cur_vec_num();
        auto [chunk_num, last_chunk_size] = ChunkInfo(cur_vec_num);

        os << "[CONST] chunk_size: " << chunk_size_ << ", max_chunk_n: " << max_chunk_n_ << ", chunk_shift: " << chunk_shift_ << std::endl;
        os << "cur_vec_num: " << cur_vec_num << std::endl;

        this->vec_store_meta_.Dump(os);
        for (size_t i = 0; i < chunk_num; ++i) {
            os << "chunk " << i << std::endl;
            size_t cur_chunk_size = (i < chunk_num - 1) ? chunk_size_ : last_chunk_size;
            inners_[i].DumpVec(os, i * chunk_size_, cur_chunk_size, this->vec_store_meta_);
        }

        this->graph_store_meta_.Dump(os);
        for (size_t i = 0; i < chunk_num; ++i) {
            os << "chunk " << i << std::endl;
            size_t cur_chunk_size = (i < chunk_num - 1) ? chunk_size_ : last_chunk_size;
            inners_[i].DumpGraph(os, cur_chunk_size, this->graph_store_meta_);
        }
    }
};

export template <typename VecStoreT, typename LabelType>
class DataStore<VecStoreT, LabelType, false> : public DataStoreBase<VecStoreT, LabelType, false> {
public:
    using This = DataStore<VecStoreT, LabelType, false>;
    using VecStoreMeta = typename VecStoreT::template Meta<false>;
    using Base = DataStoreBase<VecStoreT, LabelType, false>;
    using Inner = DataStoreInner<VecStoreT, LabelType, false>;

private:
    DataStore(size_t cur_vec_num, VecStoreMeta vec_store_meta, GraphStoreMeta graph_store_meta)
        : Base(std::move(vec_store_meta), std::move(graph_store_meta)), cur_vec_num_(cur_vec_num) {}

public:
    DataStore() = default;
    // DataStore(DataStore &&other) : Base(std::move(other)), inner_(std::move(other.inner_)), cur_vec_num_(other.cur_vec_num_) {}
    // DataStore &operator=(DataStore &&other) {
    //     if (this != &other) {
    //         Base::operator=(std::move(other));
    //         inner_ = std::move(other.inner_);
    //         cur_vec_num_ = other.cur_vec_num_;
    //     }
    //     return *this;
    // }
    // ~DataStore() = default;

    static This LoadFromPtr(HnswPointerReader &reader) {
        const size_t cur_vec_num = reader.Read<size_t>("vector count");
        if (cur_vec_num > static_cast<size_t>(std::numeric_limits<VertexType>::max())) {
            HnswPointerImageError("vector count exceeds the vertex representation");
        }
        VecStoreMeta vec_store_meta = VecStoreMeta::LoadFromPtr(reader);
        GraphStoreMeta graph_store_meta = GraphStoreMeta::LoadFromPtr(reader);
        Base::ValidatePointerImageEntryPoint(graph_store_meta, cur_vec_num);

        This ret = This(cur_vec_num, std::move(vec_store_meta), std::move(graph_store_meta));
        ret.inner_ = Inner::LoadFromPtr(reader, cur_vec_num, cur_vec_num, ret.vec_store_meta_, ret.graph_store_meta_);
        return ret;
    }

    typename VecStoreT::StoreType GetVec(size_t vec_i) const { return inner_.GetVec(vec_i, this->vec_store_meta_); }

    void PrefetchVec(size_t vec_i) const { inner_.PrefetchVec(vec_i, this->vec_store_meta_); }

    std::pair<const VertexType *, VertexListSize> GetNeighbors(VertexType vertex_i, i32 layer_i) const {
        return inner_.GetNeighbors(vertex_i, layer_i, this->graph_store_meta_);
    }

    LayerSize GetLevel(VertexType vertex_i) const { return inner_.GetLevel(vertex_i, this->graph_store_meta_); }

    bool IsVertexBuilt(VertexType vertex_i) const { return inner_.IsVertexBuilt(vertex_i, this->graph_store_meta_); }

    bool AllVerticesBuilt() const {
        if (this->Mmax0() == 0 && this->Mmax() == 0) {
            return true;
        }
        for (VertexType vertex_i = 0; static_cast<size_t>(vertex_i) < cur_vec_num_; ++vertex_i) {
            if (!IsVertexBuilt(vertex_i)) {
                return false;
            }
        }
        return true;
    }

    void EnsureAllVerticesBuilt() const {
        if (!AllVerticesBuilt()) {
            throw std::logic_error("HNSW index contains stored but unbuilt vertices");
        }
    }

    LabelType GetLabel(size_t vec_i) const { return inner_.GetLabel(vec_i); }

    size_t cur_vec_num() const { return cur_vec_num_; }

    size_t mem_usage() const { return inner_.ExtraMemoryUsage(); }

private:
    Inner inner_;
    size_t cur_vec_num_ = 0;

public:
    void Check() const {
        i32 max_l = -1;
        inner_.Check(cur_vec_num_, this->graph_store_meta_, 0, cur_vec_num_, max_l);
        auto [max_layer, ep] = this->GetEnterPoint();
        static_cast<void>(max_layer);
        static_cast<void>(ep);
        Base::CheckGraphTopology(*this, cur_vec_num_, max_l);
    }

    void Dump() const {
        std::cout << "[CONST] cur_vec_num: " << cur_vec_num_ << std::endl;
        this->vec_store_meta_.Dump();
        inner_.DumpVec(std::cout, 0, cur_vec_num_, this->vec_store_meta_);
        this->graph_store_meta_.Dump();
        inner_.DumpGraph(std::cout, cur_vec_num_, this->graph_store_meta_);
    }
};

#pragma clang diagnostic pop
//----------------------------------------------- Inner -----------------------------------------------

template <typename LabelType, bool OwnMem>
class HnswLabelStore;

template <typename LabelType>
class HnswLabelStore<LabelType, true> {
public:
    HnswLabelStore() = default;
    HnswLabelStore(std::unique_ptr<LabelType[]> labels) : labels_(std::move(labels)) {}

    LabelType &operator[](size_t index) { return labels_[index]; }
    const LabelType &operator[](size_t index) const { return labels_[index]; }
    LabelType *get() const { return labels_.get(); }

private:
    std::unique_ptr<LabelType[]> labels_;
};

template <typename LabelType>
class HnswLabelStore<LabelType, false> {
public:
    HnswLabelStore() = default;
    HnswLabelStore(const char *labels) : labels_(labels) {}

    LabelType operator[](size_t index) const noexcept {
        static_assert(std::is_trivially_copyable_v<LabelType>);
        LabelType label;
        std::memcpy(&label, labels_ + index * sizeof(LabelType), sizeof(LabelType));
        return label;
    }

private:
    const char *labels_ = nullptr;
};

template <typename VecStoreT, typename LabelType, bool OwnMem>
class DataStoreInnerBase {
public:
    using This = DataStoreInner<VecStoreT, LabelType, OwnMem>;
    using DataType = typename VecStoreT::DataType;
    using VecStoreInner = typename VecStoreT::template Inner<OwnMem>;
    using VecStoreMeta = typename VecStoreT::template Meta<OwnMem>;
    using GraphStoreInner = GraphStoreInner<OwnMem>;

    friend class DataStoreIter<VecStoreT, LabelType>;

public:
    DataStoreInnerBase() = default;

    void Save(LocalFileHandle &file_handle, size_t cur_vec_num, const VecStoreMeta &vec_store_meta, const GraphStoreMeta &graph_store_meta) const {
        this->vec_store_inner_.Save(file_handle, cur_vec_num, vec_store_meta);
        this->graph_store_inner_.Save(file_handle, cur_vec_num, graph_store_meta);
        file_handle.Append(this->labels_.get(), sizeof(LabelType) * cur_vec_num);
    }

    static void SaveToPtr(LocalFileHandle &file_handle,
                          const This *inners,
                          const VecStoreMeta &vec_store_meta,
                          const GraphStoreMeta &graph_store_meta,
                          size_t ck_size,
                          size_t chunk_num,
                          size_t last_chunk_size) {
        std::vector<const typename VecStoreInner::Base *> vec_store_inners;
        std::vector<const typename GraphStoreInner::Base *> graph_store_inners;
        for (size_t i = 0; i < chunk_num; ++i) {
            vec_store_inners.emplace_back(&inners[i].vec_store_inner_);
            graph_store_inners.emplace_back(&inners[i].graph_store_inner_);
        }
        VecStoreInner::SaveToPtr(file_handle, vec_store_inners, vec_store_meta, ck_size, chunk_num, last_chunk_size);
        GraphStoreInner::SaveToPtr(file_handle, graph_store_inners, graph_store_meta, ck_size, chunk_num, last_chunk_size);
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t chunk_size = (i < chunk_num - 1) ? ck_size : last_chunk_size;
            file_handle.Append(inners[i].labels_.get(), sizeof(LabelType) * chunk_size);
        }
    }

    void Free(size_t cur_vec_num, const GraphStoreMeta &graph_store_meta) { graph_store_inner_.Free(cur_vec_num, graph_store_meta); }

    size_t GetSizeInBytes(size_t chunk_size, const VecStoreMeta &vec_store_meta, const GraphStoreMeta &graph_store_meta) const {
        size_t size = 0;
        size += vec_store_inner_.GetSizeInBytes(chunk_size, vec_store_meta);
        size += graph_store_inner_.GetSizeInBytes(chunk_size, graph_store_meta);
        size += sizeof(LabelType) * chunk_size;
        return size;
    }

    // vec store
    typename VecStoreT::StoreType GetVec(VertexType vec_i, const VecStoreMeta &meta) const { return vec_store_inner_.GetVec(vec_i, meta); }

    typename VecStoreT::QueryType GetVecToQuery(VertexType vec_i, const VecStoreMeta &meta) const {
        return vec_store_inner_.GetVecToQuery(vec_i, meta);
    }

    void PrefetchVec(VertexType vec_i, const VecStoreMeta &meta) const { vec_store_inner_.Prefetch(vec_i, meta); }

    // graph store
    std::pair<const VertexType *, VertexListSize> GetNeighbors(VertexType vertex_i, i32 layer_i, const GraphStoreMeta &meta) const {
        return graph_store_inner_.GetNeighbors(vertex_i, layer_i, meta);
    }

    LayerSize GetLevel(VertexType vertex_i, const GraphStoreMeta &meta) const { return graph_store_inner_.GetLevel(vertex_i, meta); }

    bool IsVertexBuilt(VertexType vertex_i, const GraphStoreMeta &meta) const { return graph_store_inner_.IsVertexBuilt(vertex_i, meta); }

    LabelType GetLabel(VertexType vec_i) const { return labels_[vec_i]; }

    VecStoreInner *vec_store_inner() { return &vec_store_inner_; }

    GraphStoreInner *graph_store_inner() { return &graph_store_inner_; }
    void SetGraphStoreInner(GraphStoreInner &&graph_store_inner) noexcept { graph_store_inner_ = std::move(graph_store_inner); }
    size_t ExtraMemoryUsage() const noexcept {
        if constexpr (OwnMem) {
            return 0;
        } else {
            return graph_store_inner_.ExtraMemoryUsage();
        }
    }

protected:
    VecStoreInner vec_store_inner_;
    GraphStoreInner graph_store_inner_;
    HnswLabelStore<LabelType, OwnMem> labels_;

public:
    void Check(size_t chunk_size, const GraphStoreMeta &meta, VertexType vertex_i_offset, size_t cur_vec_num, i32 &max_l) const {
        graph_store_inner_.Check(chunk_size, meta, vertex_i_offset, cur_vec_num, max_l);
    }

    void DumpVec(std::ostream &os, size_t offset, size_t chunk_size, const VecStoreMeta &meta) const {
        vec_store_inner_.Dump(os, offset, chunk_size, meta);
        os << "labels: [";
        for (size_t i = 0; i < chunk_size; ++i) {
            os << labels_[i] << ", ";
        }
        os << "]" << std::endl;
    }

    void DumpGraph(std::ostream &os, size_t chunk_size, const GraphStoreMeta &meta) const { graph_store_inner_.Dump(os, chunk_size, meta); }
};

template <typename VecStoreT, typename LabelType, bool OwnMem>
class DataStoreInner : public DataStoreInnerBase<VecStoreT, LabelType, OwnMem> {
private:
    using This = DataStoreInner<VecStoreT, LabelType, OwnMem>;
    using VecStoreInner = typename VecStoreT::template Inner<OwnMem>;
    using VecStoreMeta = typename VecStoreT::template Meta<OwnMem>;
    using GraphStoreInner = GraphStoreInner<OwnMem>;
    using QueryVecType = typename VecStoreT::QueryVecType;

    DataStoreInner(size_t chunk_size, VecStoreInner vec_store_inner, GraphStoreInner graph_store_inner) {
        this->vec_store_inner_ = std::move(vec_store_inner);
        this->graph_store_inner_ = std::move(graph_store_inner);
        this->labels_ = std::make_unique<LabelType[]>(chunk_size);
        vertex_mutex_ = std::make_unique<HnswVertexMutex[]>(chunk_size);
    }

public:
    DataStoreInner() = default;
    static This Make(size_t chunk_size, VecStoreMeta &vec_store_meta, GraphStoreMeta &graph_store_meta, size_t &mem_usage) {
        auto vec_store_inner = VecStoreInner::Make(chunk_size, vec_store_meta, mem_usage);
        auto graph_store_inner = GraphStoreInner::Make(chunk_size, graph_store_meta, mem_usage);
        mem_usage = HnswCheckedAdd(mem_usage, OwnedCapacityMemoryUsage(chunk_size), "HNSW owned inner memory");
        return This(chunk_size, std::move(vec_store_inner), std::move(graph_store_inner));
    }

    static This Load(LocalFileHandle &file_handle,
                     size_t cur_vec_num,
                     size_t chunk_size,
                     VecStoreMeta &vec_store_meta,
                     GraphStoreMeta &graph_store_meta,
                     size_t &mem_usage,
                     size_t vertex_offset = 0,
                     size_t total_vertex_n = std::numeric_limits<size_t>::max()) {
        auto vec_store_inner = VecStoreInner::Load(file_handle, cur_vec_num, chunk_size, vec_store_meta, mem_usage);
        auto graph_store_iner =
            GraphStoreInner::Load(file_handle, cur_vec_num, chunk_size, graph_store_meta, mem_usage, vertex_offset, total_vertex_n);
        mem_usage =
            HnswStreamCheckedAdd(mem_usage, StreamOwnedCapacityMemoryUsage(chunk_size), "HNSW owned inner memory");
        This ret(chunk_size, std::move(vec_store_inner), std::move(graph_store_iner));
        const size_t labels_size = HnswStreamCheckedMultiply(sizeof(LabelType), cur_vec_num, "HNSW labels");
        HnswReadExact(file_handle, ret.labels_.get(), labels_size, "HNSW labels");
        return ret;
    }

    static This LoadFromPtr(HnswPointerReader &reader,
                            size_t cur_vec_num,
                            size_t chunk_size,
                            VecStoreMeta &vec_store_meta,
                            GraphStoreMeta &graph_store_meta,
                            size_t &mem_usage) {
        auto vec_store_inner = VecStoreInner::LoadFromPtr(reader, cur_vec_num, chunk_size, vec_store_meta, mem_usage);
        auto graph_store_inner = GraphStoreInner::LoadFromPtr(reader, cur_vec_num, chunk_size, graph_store_meta, mem_usage);
        mem_usage = HnswCheckedAdd(mem_usage, OwnedCapacityMemoryUsage(chunk_size), "HNSW owned inner memory");
        This ret(chunk_size, std::move(vec_store_inner), std::move(graph_store_inner));
        const size_t labels_size = HnswCheckedMultiply(sizeof(LabelType), cur_vec_num, "HNSW labels");
        reader.CopyTo(ret.labels_.get(), labels_size, "HNSW labels");
        return ret;
    }

    // vec store
    template <DataIteratorConcept<QueryVecType, LabelType> Iterator>
    std::pair<size_t, bool> AddVec(Iterator &&query_iter, VertexType start_idx, size_t remain_num, const VecStoreMeta &meta, size_t &mem_usage) {
        size_t insert_n = 0;
        bool used_up = false;
        while (insert_n < remain_num) {
            if (auto ret = query_iter.Next(); ret) {
                auto &[vec, label] = *ret;
                this->vec_store_inner_.SetVec(start_idx + insert_n, vec, meta, mem_usage);
                this->labels_[start_idx + insert_n] = label;
                ++insert_n;
            } else {
                used_up = true;
                break;
            }
        }
        return {insert_n, used_up};
    }

    // graph store
    void AddVertex(VertexType vec_i, i32 layer_n, const GraphStoreMeta &meta, size_t &mem_usage) {
        this->graph_store_inner_.AddVertex(vec_i, layer_n, meta, mem_usage);
    }
    std::pair<VertexType *, VertexListSize *> GetNeighborsMut(VertexType vertex_i, i32 layer_i, const GraphStoreMeta &meta) {
        return this->graph_store_inner_.GetNeighborsMut(vertex_i, layer_i, meta);
    }

    HnswVertexSharedLock SharedLock(VertexType vec_i) const { return HnswVertexSharedLock(vertex_mutex_[vec_i]); }

    HnswVertexUniqueLock UniqueLock(VertexType vec_i) { return HnswVertexUniqueLock(vertex_mutex_[vec_i]); }

private:
    static size_t OwnedCapacityMemoryUsage(size_t chunk_size) {
        const size_t labels = HnswCheckedMultiply(chunk_size, sizeof(LabelType), "HNSW label capacity");
        const size_t vertex_mutexes = HnswCheckedMultiply(chunk_size, sizeof(HnswVertexMutex), "HNSW vertex mutex capacity");
        return HnswCheckedAdd(labels, vertex_mutexes, "HNSW owned inner capacity");
    }

    static size_t StreamOwnedCapacityMemoryUsage(size_t chunk_size) {
        const size_t labels = HnswStreamCheckedMultiply(chunk_size, sizeof(LabelType), "HNSW label capacity");
        const size_t vertex_mutexes =
            HnswStreamCheckedMultiply(chunk_size, sizeof(HnswVertexMutex), "HNSW vertex mutex capacity");
        return HnswStreamCheckedAdd(labels, vertex_mutexes, "HNSW owned inner capacity");
    }

    mutable std::unique_ptr<HnswVertexMutex[]> vertex_mutex_;
};

template <typename VecStoreT, typename LabelType>
class DataStoreInner<VecStoreT, LabelType, false> : public DataStoreInnerBase<VecStoreT, LabelType, false> {
public:
    using This = DataStoreInner<VecStoreT, LabelType, false>;
    using VecStoreInner = typename VecStoreT::template Inner<false>;
    using VecStoreMeta = typename VecStoreT::template Meta<false>;
    using GraphStoreInner = GraphStoreInner<false>;

private:
    DataStoreInner(size_t chunk_size, VecStoreInner vec_store_inner, GraphStoreInner graph_store_inner, const char *labels) {
        this->vec_store_inner_ = std::move(vec_store_inner);
        this->graph_store_inner_ = std::move(graph_store_inner);
        this->labels_ = labels;
    }

public:
    DataStoreInner() = default;

    static This LoadFromPtr(HnswPointerReader &reader,
                            size_t cur_vec_num,
                            size_t chunk_size,
                            VecStoreMeta &vec_store_meta,
                            const GraphStoreMeta &graph_store_meta) {
        auto vec_store_inner = VecStoreInner::LoadFromPtr(reader, cur_vec_num, vec_store_meta);
        auto graph_store_inner = GraphStoreInner::LoadFromPtr(reader, cur_vec_num, chunk_size, graph_store_meta);
        const size_t labels_size = HnswCheckedMultiply(sizeof(LabelType), cur_vec_num, "HNSW labels");
        const char *labels = reader.ReadBytes(labels_size, "HNSW labels");
        return This(chunk_size, std::move(vec_store_inner), std::move(graph_store_inner), labels);
    }
};

template <typename VecStoreT, typename LabelType>
class DataStoreChunkIter {
public:
    using Inner = typename DataStore<VecStoreT, LabelType, true>::Inner;

    DataStoreChunkIter(const DataStore<VecStoreT, LabelType, true> *data_store) : data_store_(data_store) {
        std::tie(chunk_num_, last_chunk_size_) = data_store_->ChunkInfo(data_store_->cur_vec_num());
    }

    std::optional<std::pair<const Inner *, size_t>> Next() {
        if (cur_chunk_i_ >= chunk_num_) {
            return std::nullopt;
        }
        auto ret = std::pair<const Inner *, size_t>(&data_store_->inners_[cur_chunk_i_],
                                                    (cur_chunk_i_ == chunk_num_ - 1) ? last_chunk_size_ : data_store_->chunk_size_);
        ++cur_chunk_i_;
        return ret;
    }

    const DataStore<VecStoreT, LabelType, true> *data_store_;

private:
    size_t cur_chunk_i_ = 0;
    size_t chunk_num_;
    size_t last_chunk_size_;
};

template <typename VecStoreT, typename LabelType>
class DataStoreInnerIter {
public:
    using VecMeta = typename VecStoreT::template Meta<true>;
    using Inner = DataStoreInner<VecStoreT, LabelType, true>;
    using StoreType = typename VecStoreT::StoreType;

    DataStoreInnerIter(const VecMeta *vec_meta, const Inner *inner, size_t max_vec_num)
        : vec_meta_(vec_meta), inner_(inner), max_vec_num_(max_vec_num), cur_idx_(0) {}

    std::optional<std::pair<StoreType, LabelType>> Next() {
        if (cur_idx_ >= max_vec_num_) {
            return std::nullopt;
        }
        auto ret = std::pair<StoreType, LabelType>(inner_->GetVec(cur_idx_, *vec_meta_), inner_->GetLabel(cur_idx_));
        ++cur_idx_;
        return ret;
    }

private:
    const VecMeta *vec_meta_;
    const Inner *inner_;
    size_t max_vec_num_;

    size_t cur_idx_;
};

template <typename VecStoreT, typename LabelType>
class DataStoreIter {
public:
    using StoreType = typename VecStoreT::StoreType;
    using InnerIter = DataStoreInnerIter<VecStoreT, LabelType>;
    using ValueType = StoreType;

    DataStoreIter(const DataStore<VecStoreT, LabelType, true> *data_store)
        : data_store_iter_(data_store), inner_iter_(std::nullopt), row_count_(data_store->cur_vec_num()) {}

    std::optional<std::pair<StoreType, LabelType>> Next() {
        if (!inner_iter_.has_value()) {
            auto inner_opt = data_store_iter_.Next();
            if (!inner_opt.has_value()) {
                return std::nullopt;
            }
            const auto &[inner, chunk_size] = inner_opt.value();
            inner_iter_ = InnerIter(&data_store_iter_.data_store_->vec_store_meta_, inner, chunk_size);
        }
        auto &inner = inner_iter_.value();
        auto vec_opt = inner.Next();
        if (!vec_opt.has_value()) {
            inner_iter_ = std::nullopt;
            return Next();
        }
        return vec_opt.value();
    }

    size_t GetRowCount() const { return row_count_; }

private:
    DataStoreChunkIter<VecStoreT, LabelType> data_store_iter_;
    std::optional<InnerIter> inner_iter_;
    size_t row_count_ = 0;
};

template <typename VecStoreT, typename LabelType, bool OwnMem>
template <typename CompressVecStoreType>
DataStore<CompressVecStoreType, LabelType, OwnMem> DataStore<VecStoreT, LabelType, OwnMem>::CompressToLVQ() && {
    if constexpr (std::is_same_v<CompressVecStoreType, VecStoreT>) {
        return std::move(*this);
    } else {
        const auto [chunk_num, last_chunk_size] = this->ChunkInfo(this->cur_vec_num());
        static_cast<void>(last_chunk_size);
        const size_t built_vec_num = this->built_vec_num();
        auto ret = DataStore<CompressVecStoreType, LabelType, OwnMem>::Make(this->chunk_size_,
                                                                            this->max_chunk_n_,
                                                                            this->vec_store_meta_.dim(),
                                                                            this->Mmax0(),
                                                                            this->Mmax());
        ret.OptAddVec(DataStoreIter<VecStoreT, LabelType>(this));
        if (ret.cur_vec_num() != this->cur_vec_num()) {
            throw std::logic_error("HNSW LVQ compression did not preserve the vector count");
        }
        ret.ValidateGraphAttachment(chunk_num, built_vec_num);

        size_t upper_layer_mem_usage = 0;
        for (size_t i = 0; i < chunk_num; ++i) {
            const size_t current_chunk_size = i + 1 < chunk_num ? this->chunk_size_ : last_chunk_size;
            upper_layer_mem_usage =
                HnswCheckedAdd(upper_layer_mem_usage,
                               this->inners_[i].graph_store_inner()->UpperLayerMemUsage(current_chunk_size, this->graph_store_meta_),
                               "HNSW transferred upper-layer memory");
        }
        std::vector<GraphStoreInner<OwnMem>> graph_inners;
        graph_inners.reserve(chunk_num);
        static_assert(std::is_nothrow_move_constructible_v<GraphStoreInner<OwnMem>>);
        for (size_t i = 0; i < chunk_num; ++i) {
            graph_inners.emplace_back(std::move(*this->inners_[i].graph_store_inner()));
        }
        ret.SetGraph(std::move(this->graph_store_meta_), std::move(graph_inners), built_vec_num, upper_layer_mem_usage);
        this->inners_ = nullptr;
        this->cur_vec_num_.store(0, std::memory_order_release);
        this->built_vec_num_.store(0, std::memory_order_release);
        this->mem_usage_.store(0, std::memory_order_release);
        return ret;
    }
}

template <typename VecStoreT, typename LabelType, bool OwnMem>
template <typename CompressVecStoreType>
DataStore<CompressVecStoreType, LabelType, OwnMem> DataStore<VecStoreT, LabelType, OwnMem>::CompressToRabitq() && {
    if constexpr (std::is_same_v<CompressVecStoreType, VecStoreT>) {
        return std::move(*this);
    } else {
        const auto [chunk_num, last_chunk_size] = this->ChunkInfo(this->cur_vec_num());
        static_cast<void>(last_chunk_size);
        const size_t built_vec_num = this->built_vec_num();
        auto ret = DataStore<CompressVecStoreType, LabelType, OwnMem>::Make(this->chunk_size_,
                                                                            this->max_chunk_n_,
                                                                            this->vec_store_meta_.dim(),
                                                                            this->Mmax0(),
                                                                            this->Mmax());
        ret.OptAddVec(DataStoreIter<VecStoreT, LabelType>(this));
        if (ret.cur_vec_num() != this->cur_vec_num()) {
            throw std::logic_error("HNSW Rabitq compression did not preserve the vector count");
        }
        ret.ValidateGraphAttachment(chunk_num, built_vec_num);

        size_t upper_layer_mem_usage = 0;
        for (size_t i = 0; i < chunk_num; ++i) {
            const size_t current_chunk_size = i + 1 < chunk_num ? this->chunk_size_ : last_chunk_size;
            upper_layer_mem_usage =
                HnswCheckedAdd(upper_layer_mem_usage,
                               this->inners_[i].graph_store_inner()->UpperLayerMemUsage(current_chunk_size, this->graph_store_meta_),
                               "HNSW transferred upper-layer memory");
        }
        std::vector<GraphStoreInner<OwnMem>> graph_inners;
        graph_inners.reserve(chunk_num);
        static_assert(std::is_nothrow_move_constructible_v<GraphStoreInner<OwnMem>>);
        for (size_t i = 0; i < chunk_num; ++i) {
            graph_inners.emplace_back(std::move(*this->inners_[i].graph_store_inner()));
        }
        ret.SetGraph(std::move(this->graph_store_meta_), std::move(graph_inners), built_vec_num, upper_layer_mem_usage);
        this->inners_ = nullptr;
        this->cur_vec_num_.store(0, std::memory_order_release);
        this->built_vec_num_.store(0, std::memory_order_release);
        this->mem_usage_.store(0, std::memory_order_release);
        return ret;
    }
}

} // namespace infinity
