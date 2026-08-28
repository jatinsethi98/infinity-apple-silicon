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
#include <new>

export module infinity_core:graph_store;

import :hnsw_common;
import :local_file_handle;
import :data_store_util;
import :infinity_exception;

import std;

import serialize;

namespace infinity {

struct VertexL0 {
    LayerSize layer_n_;
    char *layers_p_;
    VertexListSize neighbor_n_;
    VertexType neighbors_[];
};
struct VertexLX {
    VertexListSize neighbor_n_;
    VertexType neighbors_[];
};

void ValidatePointerGraphRecordAlignment(size_t level0_size, size_t levelx_size) {
    if (level0_size % alignof(VertexL0) != 0) {
        HnswPointerImageError("graph level-zero record stride is misaligned");
    }
    if (levelx_size % alignof(VertexLX) != 0) {
        HnswPointerImageError("graph upper-layer record stride is misaligned");
    }
}

size_t ValidatePointerGraphRecords(const char *graph,
                                   size_t cur_vertex_n,
                                   size_t level0_size,
                                   size_t levelx_size,
                                   size_t Mmax0,
                                   size_t serialized_layer_sum) {
    ValidatePointerGraphRecordAlignment(level0_size, levelx_size);
    if (cur_vertex_n != 0 && reinterpret_cast<std::uintptr_t>(graph) % alignof(VertexL0) != 0) {
        HnswPointerImageError("graph level-zero data is misaligned");
    }
    size_t expected_layer_sum = 0;
    for (size_t vertex_index = 0; vertex_index < cur_vertex_n; ++vertex_index) {
        const auto *vertex = reinterpret_cast<const VertexL0 *>(graph + vertex_index * level0_size);
        if (vertex->layer_n_ < 0 || vertex->layer_n_ > kHnswMaxSupportedLayer) {
            HnswPointerImageError("graph contains an invalid vertex level");
        }
        if (vertex->neighbor_n_ < 0 || static_cast<size_t>(vertex->neighbor_n_) > Mmax0) {
            HnswPointerImageError("graph level-zero degree exceeds capacity");
        }
        for (VertexListSize neighbor_index = 0; neighbor_index < vertex->neighbor_n_; ++neighbor_index) {
            const VertexType neighbor = vertex->neighbors_[neighbor_index];
            if (neighbor < 0 || static_cast<size_t>(neighbor) >= cur_vertex_n) {
                HnswPointerImageError("graph contains an out-of-range level-zero edge");
            }
            if (static_cast<size_t>(neighbor) == vertex_index) {
                HnswPointerImageError("graph contains a level-zero self edge");
            }
            for (VertexListSize previous = 0; previous < neighbor_index; ++previous) {
                if (vertex->neighbors_[previous] == neighbor) {
                    HnswPointerImageError("graph contains a duplicate level-zero edge");
                }
            }
        }

        std::uintptr_t encoded_offset = 0;
        std::memcpy(&encoded_offset, &vertex->layers_p_, sizeof(encoded_offset));
        const size_t expected_offset = HnswCheckedMultiply(expected_layer_sum, levelx_size, "graph upper-layer offset");
        if ((vertex->layer_n_ == 0 && encoded_offset != 0) ||
            (vertex->layer_n_ > 0 && encoded_offset != expected_offset)) {
            HnswPointerImageError("graph contains an invalid upper-layer offset");
        }
        expected_layer_sum =
            HnswCheckedAdd(expected_layer_sum, static_cast<size_t>(vertex->layer_n_), "graph upper-layer count");
    }
    if (expected_layer_sum != serialized_layer_sum) {
        HnswPointerImageError("graph upper-layer count does not match vertex levels");
    }
    return HnswCheckedMultiply(expected_layer_sum, levelx_size, "graph upper-layer data");
}

void ValidatePointerUpperGraphRecords(const char *graph,
                                      const char *layers,
                                      size_t cur_vertex_n,
                                      size_t level0_size,
                                      size_t levelx_size,
                                      size_t Mmax,
                                      size_t serialized_layer_sum) {
    if (serialized_layer_sum != 0 &&
        (layers == nullptr || reinterpret_cast<std::uintptr_t>(layers) % alignof(VertexLX) != 0)) {
        HnswPointerImageError("graph upper-layer data is misaligned");
    }

    const char *next_layer = layers;
    for (size_t vertex_index = 0; vertex_index < cur_vertex_n; ++vertex_index) {
        const auto *vertex = reinterpret_cast<const VertexL0 *>(graph + vertex_index * level0_size);
        for (LayerSize layer = 1; layer <= vertex->layer_n_; ++layer) {
            const auto *record = reinterpret_cast<const VertexLX *>(next_layer);
            if (record->neighbor_n_ < 0 || static_cast<size_t>(record->neighbor_n_) > Mmax) {
                HnswPointerImageError("graph upper-layer degree exceeds capacity");
            }
            for (VertexListSize neighbor_index = 0; neighbor_index < record->neighbor_n_; ++neighbor_index) {
                const VertexType neighbor = record->neighbors_[neighbor_index];
                if (neighbor < 0 || static_cast<size_t>(neighbor) >= cur_vertex_n) {
                    HnswPointerImageError("graph contains an out-of-range upper-layer edge");
                }
                if (static_cast<size_t>(neighbor) == vertex_index) {
                    HnswPointerImageError("graph contains an upper-layer self edge");
                }
                for (VertexListSize previous = 0; previous < neighbor_index; ++previous) {
                    if (record->neighbors_[previous] == neighbor) {
                        HnswPointerImageError("graph contains a duplicate upper-layer edge");
                    }
                }
                const auto *target = reinterpret_cast<const VertexL0 *>(graph + static_cast<size_t>(neighbor) * level0_size);
                if (target->layer_n_ < layer) {
                    HnswPointerImageError("graph upper-layer edge targets a lower-level vertex");
                }
            }
            next_layer += levelx_size;
        }
    }
}

size_t ValidateStreamGraphRecords(const char *graph,
                                  size_t cur_vertex_n,
                                  size_t vertex_offset,
                                  size_t total_vertex_n,
                                  size_t level0_size,
                                  size_t levelx_size,
                                  size_t Mmax0,
                                  size_t serialized_layer_sum) {
    if (vertex_offset > total_vertex_n || cur_vertex_n > total_vertex_n - vertex_offset) {
        HnswStreamError("graph chunk range exceeds the vector count");
    }
    if (level0_size % alignof(VertexL0) != 0 || levelx_size % alignof(VertexLX) != 0) {
        HnswStreamError("graph record stride is misaligned");
    }
    if (cur_vertex_n != 0 && reinterpret_cast<std::uintptr_t>(graph) % alignof(VertexL0) != 0) {
        HnswStreamError("graph level-zero data is misaligned");
    }

    size_t expected_layer_sum = 0;
    for (size_t vertex_index = 0; vertex_index < cur_vertex_n; ++vertex_index) {
        const auto *vertex = reinterpret_cast<const VertexL0 *>(graph + vertex_index * level0_size);
        if (vertex->layer_n_ < 0 || vertex->layer_n_ > kHnswMaxSupportedLayer) {
            HnswStreamError("graph contains an invalid vertex level");
        }
        if (vertex->neighbor_n_ < 0 || static_cast<size_t>(vertex->neighbor_n_) > Mmax0) {
            HnswStreamError("graph level-zero degree exceeds capacity");
        }
        const size_t global_vertex = vertex_offset + vertex_index;
        for (VertexListSize neighbor_index = 0; neighbor_index < vertex->neighbor_n_; ++neighbor_index) {
            const VertexType neighbor = vertex->neighbors_[neighbor_index];
            if (neighbor < 0 || static_cast<size_t>(neighbor) >= total_vertex_n) {
                HnswStreamError("graph contains an out-of-range level-zero edge");
            }
            if (static_cast<size_t>(neighbor) == global_vertex) {
                HnswStreamError("graph contains a level-zero self edge");
            }
            for (VertexListSize previous = 0; previous < neighbor_index; ++previous) {
                if (vertex->neighbors_[previous] == neighbor) {
                    HnswStreamError("graph contains a duplicate level-zero edge");
                }
            }
        }
        expected_layer_sum =
            HnswStreamCheckedAdd(expected_layer_sum, static_cast<size_t>(vertex->layer_n_), "graph upper-layer count");
    }
    if (expected_layer_sum != serialized_layer_sum) {
        HnswStreamError("graph upper-layer count does not match vertex levels");
    }
    return HnswStreamCheckedMultiply(expected_layer_sum, levelx_size, "graph upper-layer data");
}

void ValidateStreamUpperGraphRecords(char *graph,
                                     char *layers,
                                     size_t cur_vertex_n,
                                     size_t vertex_offset,
                                     size_t total_vertex_n,
                                     size_t level0_size,
                                     size_t levelx_size,
                                     size_t Mmax) {
    if (layers != nullptr && reinterpret_cast<std::uintptr_t>(layers) % alignof(VertexLX) != 0) {
        HnswStreamError("graph upper-layer data is misaligned");
    }
    char *next_layer = layers;
    for (size_t vertex_index = 0; vertex_index < cur_vertex_n; ++vertex_index) {
        auto *vertex = reinterpret_cast<VertexL0 *>(graph + vertex_index * level0_size);
        const size_t global_vertex = vertex_offset + vertex_index;
        char *const vertex_layers = next_layer;
        for (LayerSize layer = 1; layer <= vertex->layer_n_; ++layer) {
            const auto *record = reinterpret_cast<const VertexLX *>(next_layer);
            if (record->neighbor_n_ < 0 || static_cast<size_t>(record->neighbor_n_) > Mmax) {
                HnswStreamError("graph upper-layer degree exceeds capacity");
            }
            for (VertexListSize neighbor_index = 0; neighbor_index < record->neighbor_n_; ++neighbor_index) {
                const VertexType neighbor = record->neighbors_[neighbor_index];
                if (neighbor < 0 || static_cast<size_t>(neighbor) >= total_vertex_n) {
                    HnswStreamError("graph contains an out-of-range upper-layer edge");
                }
                if (static_cast<size_t>(neighbor) == global_vertex) {
                    HnswStreamError("graph contains an upper-layer self edge");
                }
                for (VertexListSize previous = 0; previous < neighbor_index; ++previous) {
                    if (record->neighbors_[previous] == neighbor) {
                        HnswStreamError("graph contains a duplicate upper-layer edge");
                    }
                }
            }
            next_layer += levelx_size;
        }
        vertex->layers_p_ = vertex->layer_n_ == 0 ? nullptr : vertex_layers;
    }
}

template <typename Record>
struct GraphAlignedByteDeleter {
    static constexpr size_t kAlignment = std::max(alignof(Record), alignof(std::max_align_t));

    void operator()(char *ptr) const noexcept {
        ::operator delete(ptr, std::align_val_t{kAlignment});
    }
};

template <typename Record>
using GraphAlignedBytePtr = std::unique_ptr<char, GraphAlignedByteDeleter<Record>>;

template <typename Record>
GraphAlignedBytePtr<Record> CopyGraphBytesIfMisaligned(const char *source, size_t size) {
    if (size == 0 || reinterpret_cast<std::uintptr_t>(source) % alignof(Record) == 0) {
        return {};
    }
    auto *copy = static_cast<char *>(::operator new(size, std::align_val_t{GraphAlignedByteDeleter<Record>::kAlignment}));
    GraphAlignedBytePtr<Record> owner(copy);
    std::memcpy(copy, source, size);
    return owner;
}

export class GraphStoreMeta {
private:
    GraphStoreMeta(size_t Mmax0, size_t Mmax)
        : Mmax0_(Mmax0), Mmax_(Mmax), level0_size_(sizeof(VertexL0) + sizeof(VertexType) * Mmax0),
          levelx_size_(sizeof(VertexLX) + sizeof(VertexType) * Mmax) {}

public:
    GraphStoreMeta() : Mmax0_(0), Mmax_(0), level0_size_(0), levelx_size_(0), max_layer_(-1), enterpoint_(-1) {}
    GraphStoreMeta(GraphStoreMeta &&other) noexcept
        : Mmax0_(std::exchange(other.Mmax0_, 0)), Mmax_(std::exchange(other.Mmax_, 0)), level0_size_(std::exchange(other.level0_size_, 0)),
          levelx_size_(std::exchange(other.levelx_size_, 0)), max_layer_(std::exchange(other.max_layer_, -1)),
          enterpoint_(std::exchange(other.enterpoint_, -1)) {}
    GraphStoreMeta &operator=(GraphStoreMeta &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        Mmax0_ = std::exchange(other.Mmax0_, 0);
        Mmax_ = std::exchange(other.Mmax_, 0);
        level0_size_ = std::exchange(other.level0_size_, 0);
        levelx_size_ = std::exchange(other.levelx_size_, 0);
        max_layer_ = std::exchange(other.max_layer_, -1);
        enterpoint_ = std::exchange(other.enterpoint_, -1);
        return *this;
    }
    ~GraphStoreMeta() = default;

    static GraphStoreMeta Make(size_t Mmax0, size_t Mmax) {
        GraphStoreMeta meta(Mmax0, Mmax);
        meta.max_layer_ = -1;
        meta.enterpoint_ = -1;
        return meta;
    }

    size_t GetSizeInBytes() const { return sizeof(Mmax0_) + sizeof(Mmax_) + sizeof(max_layer_) + sizeof(enterpoint_); }

    void Save(LocalFileHandle &file_handle, size_t cur_vec_num) const {
        file_handle.Append(&Mmax0_, sizeof(Mmax0_));
        file_handle.Append(&Mmax_, sizeof(Mmax_));

        file_handle.Append(&max_layer_, sizeof(max_layer_));
        file_handle.Append(&enterpoint_, sizeof(enterpoint_));
    }

    static GraphStoreMeta
    Load(LocalFileHandle &file_handle, std::optional<std::pair<size_t, size_t>> expected_capacities = std::nullopt) {
        const size_t Mmax0 = HnswReadStream<size_t>(file_handle, "graph level-zero capacity");
        const size_t Mmax = HnswReadStream<size_t>(file_handle, "graph upper-layer capacity");
        if (Mmax0 > static_cast<size_t>(std::numeric_limits<VertexListSize>::max()) ||
            Mmax > static_cast<size_t>(std::numeric_limits<VertexListSize>::max())) {
            HnswStreamError("graph capacity exceeds the degree representation");
        }
        if (expected_capacities && *expected_capacities != std::pair{Mmax0, Mmax}) {
            HnswStreamError("graph capacities do not match HNSW M");
        }
        static_cast<void>(HnswStreamCheckedAdd(sizeof(VertexL0),
                                               HnswStreamCheckedMultiply(sizeof(VertexType), Mmax0, "graph level-zero record"),
                                               "graph level-zero record"));
        static_cast<void>(HnswStreamCheckedAdd(sizeof(VertexLX),
                                               HnswStreamCheckedMultiply(sizeof(VertexType), Mmax, "graph upper-layer record"),
                                               "graph upper-layer record"));

        GraphStoreMeta meta(Mmax0, Mmax);
        const i32 max_layer = HnswReadStream<i32>(file_handle, "graph maximum layer");
        const VertexType enterpoint = HnswReadStream<VertexType>(file_handle, "graph entry point");
        if (max_layer < -1 || max_layer > kHnswMaxSupportedLayer || enterpoint < kInvalidVertex ||
            ((max_layer == -1) != (enterpoint == kInvalidVertex))) {
            HnswStreamError("graph metadata contains an invalid entry point");
        }
        meta.max_layer_ = max_layer;
        meta.enterpoint_ = enterpoint;
        return meta;
    }

    static GraphStoreMeta LoadFromPtr(HnswPointerReader &reader) {
        const size_t Mmax0 = reader.Read<size_t>("graph level-zero capacity");
        const size_t Mmax = reader.Read<size_t>("graph upper-layer capacity");
        if (Mmax0 > static_cast<size_t>(std::numeric_limits<VertexListSize>::max()) ||
            Mmax > static_cast<size_t>(std::numeric_limits<VertexListSize>::max())) {
            HnswPointerImageError("graph capacity exceeds the degree representation");
        }
        static_cast<void>(
            HnswCheckedAdd(sizeof(VertexL0), HnswCheckedMultiply(sizeof(VertexType), Mmax0, "graph level-zero record"), "graph level-zero record"));
        static_cast<void>(
            HnswCheckedAdd(sizeof(VertexLX), HnswCheckedMultiply(sizeof(VertexType), Mmax, "graph upper-layer record"), "graph upper-layer record"));
        GraphStoreMeta meta(Mmax0, Mmax);
        const i32 max_layer = reader.Read<i32>("graph maximum layer");
        const VertexType enterpoint = reader.Read<VertexType>("graph entry point");
        if (max_layer < -1 || max_layer > kHnswMaxSupportedLayer || enterpoint < kInvalidVertex ||
            ((max_layer == -1) != (enterpoint == kInvalidVertex))) {
            HnswPointerImageError("graph metadata contains an invalid entry point");
        }
        meta.max_layer_ = max_layer;
        meta.enterpoint_ = enterpoint;
        return meta;
    }

    size_t Mmax0() const { return Mmax0_; }
    size_t Mmax() const { return Mmax_; }
    size_t level0_size() const { return level0_size_; }
    size_t levelx_size() const { return levelx_size_; }

    std::pair<i32, VertexType> GetEnterPoint() const {
        std::unique_lock lck(mtx_);
        return {max_layer_, enterpoint_};
    }

    std::pair<i32, VertexType> TryUpdateEnterPoint(i32 layer, VertexType vertex_i) {
        std::unique_lock lck(mtx_);
        if (layer > max_layer_) {
            i32 old_max_layer = max_layer_;
            VertexType old_enterpoint = enterpoint_;
            max_layer_ = layer;
            enterpoint_ = vertex_i;
            return {old_max_layer, old_enterpoint};
        } else {
            return {max_layer_, enterpoint_};
        }
    }

private:
    size_t Mmax0_;
    size_t Mmax_;
    size_t level0_size_;
    size_t levelx_size_;

    mutable std::mutex mtx_;
    i32 max_layer_;
    VertexType enterpoint_;

public:
    void Dump(std::ostream &os) const {
        auto [max_layer, enterpoint] = GetEnterPoint();
        os << "[CONST] Mmax0: " << Mmax0_ << ", Mmax: " << Mmax_ << ", level0_size: " << level0_size_ << ", levelx_size: " << levelx_size_
           << std::endl;
        os << "max_layer: " << max_layer << ", enterpoint: " << enterpoint << std::endl;
    }
};

template <bool OwnMem>
class GraphStoreInnerBase {
public:
    using This = GraphStoreInnerBase<OwnMem>;

    GraphStoreInnerBase() = default;

    void Save(LocalFileHandle &file_handle, size_t cur_vertex_n, const GraphStoreMeta &meta) const {
        size_t layer_sum = 0;
        for (VertexType vertex_i = 0; vertex_i < (VertexType)cur_vertex_n; ++vertex_i) {
            layer_sum += GetLevel0(vertex_i, meta)->layer_n_;
        }
        file_handle.Append(&layer_sum, sizeof(layer_sum));
        file_handle.Append(graph_.get(), cur_vertex_n * meta.level0_size());
        for (VertexType vertex_i = 0; vertex_i < (VertexType)cur_vertex_n; ++vertex_i) {
            const VertexL0 *v = GetLevel0(vertex_i, meta);
            if (v->layer_n_) {
                file_handle.Append(v->layers_p_, meta.levelx_size() * v->layer_n_);
            }
        }
    }

    static void SaveToPtr(LocalFileHandle &file_handle,
                          const std::vector<const This *> &inners,
                          const GraphStoreMeta &meta,
                          size_t ck_size,
                          size_t chunk_num,
                          size_t last_chunk_size) {
        size_t layer_sum = 0;
        std::vector<std::vector<std::pair<size_t, size_t>>> layers_ptrs_off_vec;
        for (size_t i = 0; i < chunk_num; ++i) {
            std::vector<std::pair<size_t, size_t>> layers_ptrs_off;
            size_t chunk_size = (i < chunk_num - 1) ? ck_size : last_chunk_size;
            const auto &inner = inners[i];
            for (VertexType vertex_i = 0; vertex_i < (VertexType)chunk_size; ++vertex_i) {
                const VertexL0 *v = inner->GetLevel0(vertex_i, meta);
                if (!v->layer_n_) {
                    continue;
                }
                size_t offset = layer_sum * meta.levelx_size();
                size_t ptr_off = reinterpret_cast<const char *>(&v->layers_p_) - inner->graph_.get();
                layers_ptrs_off.emplace_back(ptr_off, offset);
                layer_sum += v->layer_n_;
            }
            layers_ptrs_off_vec.emplace_back(std::move(layers_ptrs_off));
        }
        file_handle.Append(&layer_sum, sizeof(layer_sum));
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t chunk_size = (i < chunk_num - 1) ? ck_size : last_chunk_size;
            const auto &inner = inners[i];
            auto buffer = std::make_unique<char[]>(chunk_size * meta.level0_size());
            std::copy(inner->graph_.get(), inner->graph_.get() + chunk_size * meta.level0_size(), buffer.get());
            for (const auto &[ptr_off, offset] : layers_ptrs_off_vec[i]) {
                char *ptr = buffer.get() + ptr_off;
                *reinterpret_cast<size_t *>(ptr) = offset;
            }
            file_handle.Append(buffer.get(), chunk_size * meta.level0_size());
        }
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t chunk_size = (i < chunk_num - 1) ? ck_size : last_chunk_size;
            const auto &inner = inners[i];
            for (VertexType vertex_i = 0; vertex_i < (VertexType)chunk_size; ++vertex_i) {
                const VertexL0 *v = inner->GetLevel0(vertex_i, meta);
                if (v->layer_n_) {
                    char *ptr = v->layers_p_;
                    file_handle.Append(ptr, meta.levelx_size() * v->layer_n_);
                }
            }
        }
    }

    size_t GetSizeInBytes(size_t cur_vertex_n, const GraphStoreMeta &meta) const {
        size_t size = 0;
        for (VertexType vertex_i = 0; vertex_i < (VertexType)cur_vertex_n; ++vertex_i) {
            const VertexL0 *v = GetLevel0(vertex_i, meta);
            size += sizeof(v->layer_n_) + sizeof(v->neighbor_n_) + sizeof(VertexType) * v->neighbor_n_;
            for (i32 layer_i = 1; layer_i <= v->layer_n_; ++layer_i) {
                const VertexLX *vx = GetLevelX(v->layers_p_, layer_i, meta);
                size += sizeof(vx->neighbor_n_) + sizeof(VertexType) * vx->neighbor_n_;
            }
        }
        return size;
    }

    size_t UpperLayerMemUsage(size_t cur_vertex_n, const GraphStoreMeta &meta) const {
        size_t size = 0;
        for (VertexType vertex_i = 0; static_cast<size_t>(vertex_i) < cur_vertex_n; ++vertex_i) {
            const LayerSize level = GetLevel(vertex_i, meta);
            if (level > 0) {
                size = HnswCheckedAdd(
                    size,
                    HnswCheckedMultiply(static_cast<size_t>(level), meta.levelx_size(), "HNSW upper-layer memory"),
                    "HNSW upper-layer memory");
            }
        }
        return size;
    }

    std::pair<const VertexType *, VertexListSize> GetNeighbors(VertexType vertex_i, i32 layer_i, const GraphStoreMeta &meta) const {
        const VertexL0 *v = GetLevel0(vertex_i, meta);
        if (layer_i == 0) {
            return {v->neighbors_, v->neighbor_n_};
        }
        const VertexLX *vx = GetLevelX(v->layers_p_, layer_i, meta);
        return {vx->neighbors_, vx->neighbor_n_};
    }

    LayerSize GetLevel(VertexType vertex_i, const GraphStoreMeta &meta) const { return GetLevel0(vertex_i, meta)->layer_n_; }

    bool IsVertexBuilt(VertexType vertex_i, const GraphStoreMeta &meta) const { return GetLevel(vertex_i, meta) >= 0; }

protected:
    const VertexL0 *GetLevel0(VertexType vertex_i, const GraphStoreMeta &meta) const {
        return reinterpret_cast<const VertexL0 *>(graph_.get() + vertex_i * meta.level0_size());
    }

    const VertexLX *GetLevelX(const char *layer_p, i32 layer_i, const GraphStoreMeta &meta) const {
        assert(layer_i > 0);
        if constexpr (OwnMem) {
            return reinterpret_cast<const VertexLX *>(layer_p + (layer_i - 1) * meta.levelx_size());
        } else {
            size_t offset = reinterpret_cast<std::uintptr_t>(layer_p) + (layer_i - 1) * meta.levelx_size();
            return reinterpret_cast<const VertexLX *>(layer_start_.get() + offset);
        }
    }

protected:
    ArrayPtr<char, OwnMem> graph_;
    PPtr<OwnMem> layer_start_;

    //---------------------------------------------- Following is the tmp debug function. ----------------------------------------------

public:
    void Check(VertexType cur_vertex_n, const GraphStoreMeta &meta, VertexType vertex_i_offset, size_t cur_vec_num, i32 &max_l) const {
        i32 max_layer = -1;
        for (VertexType vertex_i = 0; vertex_i < cur_vertex_n; ++vertex_i) {
            const VertexL0 *v = GetLevel0(vertex_i, meta);
            const VertexType out_vertex_i = vertex_i + vertex_i_offset;
            if (v->layer_n_ < 0) {
                UnrecoverableError("HNSW graph has a negative vertex level");
            }
            if constexpr (OwnMem) {
                if (v->layer_n_ > 0 && v->layers_p_ == nullptr) {
                    UnrecoverableError("HNSW graph has missing upper-layer storage");
                }
            }
            max_layer = std::max(v->layer_n_, max_layer);

            auto check_neighbors = [&](const VertexType *neighbors, VertexListSize neighbor_n, size_t capacity) {
                if (neighbor_n < 0 || size_t(neighbor_n) > capacity) {
                    UnrecoverableError("HNSW graph degree exceeds its layer capacity");
                }
                for (VertexListSize i = 0; i < neighbor_n; ++i) {
                    const VertexType neighbor_idx = neighbors[i];
                    if (neighbor_idx < 0 || size_t(neighbor_idx) >= cur_vec_num) {
                        UnrecoverableError("HNSW graph contains an out-of-range edge");
                    }
                    if (neighbor_idx == out_vertex_i) {
                        UnrecoverableError("HNSW graph contains a self edge");
                    }
                    for (VertexListSize j = 0; j < i; ++j) {
                        if (neighbors[j] == neighbor_idx) {
                            UnrecoverableError("HNSW graph contains a duplicate edge");
                        }
                    }
                }
            };

            check_neighbors(v->neighbors_, v->neighbor_n_, meta.Mmax0());
            for (int layer_i = 1; layer_i <= v->layer_n_; ++layer_i) {
                const VertexLX *vx = GetLevelX(v->layers_p_, layer_i, meta);
                check_neighbors(vx->neighbors_, vx->neighbor_n_, meta.Mmax());
            }
        }
        max_l = max_layer;
    }

    void Dump(std::ostream &os, VertexType cur_vertex_n, const GraphStoreMeta &meta) const {
        if (cur_vertex_n == 0) {
            return;
        }
        i32 max_layer = 0;
        std::vector<std::vector<VertexType>> layer2vertex;
        for (VertexType vertex_i = 0; vertex_i < cur_vertex_n; ++vertex_i) {
            const VertexL0 *v = GetLevel0(vertex_i, meta);
            max_layer = std::max(max_layer, v->layer_n_);
            if (max_layer >= i32(layer2vertex.size())) {
                layer2vertex.resize(max_layer + 1);
            }
            for (i32 layer_i = 0; layer_i <= v->layer_n_; ++layer_i) {
                layer2vertex[layer_i].emplace_back(vertex_i);
            }
        }
        for (i32 layer = max_layer; layer >= 0; --layer) {
            os << "layer " << layer << std::endl;
            for (VertexType vertex_i : layer2vertex[layer]) {
                os << vertex_i << ": ";
                const int *neighbors = nullptr;
                int neighbor_n = 0;

                const VertexL0 *v = GetLevel0(vertex_i, meta);
                if (layer == 0) {
                    neighbors = v->neighbors_;
                    neighbor_n = v->neighbor_n_;
                } else {
                    const VertexLX *vx = GetLevelX(v->layers_p_, layer, meta);
                    neighbors = vx->neighbors_;
                    neighbor_n = vx->neighbor_n_;
                }
                for (int i = 0; i < neighbor_n; ++i) {
                    os << neighbors[i] << ", ";
                }
                os << std::endl;
            }
        }
    }
};

export template <bool OwnMem>
class GraphStoreInner : public GraphStoreInnerBase<OwnMem> {
public:
    using Base = GraphStoreInnerBase<OwnMem>;

private:
    GraphStoreInner(size_t max_vertex, const GraphStoreMeta &meta, size_t loaded_vertex_n) : loaded_vertex_n_(loaded_vertex_n) {
        this->graph_ = std::make_unique<char[]>(max_vertex * meta.level0_size());
    }

public:
    GraphStoreInner() = default;
    GraphStoreInner(GraphStoreInner &&) noexcept = default;
    GraphStoreInner &operator=(GraphStoreInner &&) noexcept = default;

    void Free(size_t current_vertex_num, const GraphStoreMeta &meta) {
        if (this->graph_.get() == nullptr) {
            return;
        }
        for (VertexType vertex_i = loaded_vertex_n_; vertex_i < VertexType(current_vertex_num); ++vertex_i) {
            delete[] GetLevel0(vertex_i, meta)->layers_p_;
        }
    }

    static GraphStoreInner Make(size_t max_vertex, const GraphStoreMeta &meta, size_t &mem_usage) {
        GraphStoreInner graph_store(max_vertex, meta, 0);
        std::fill(graph_store.graph_.get(), graph_store.graph_.get() + max_vertex * meta.level0_size(), 0);
        if (meta.Mmax0() != 0 || meta.Mmax() != 0) {
            graph_store.InitializeUnused(0, max_vertex, meta);
        }
        mem_usage += max_vertex * meta.level0_size();
        return graph_store;
    }

    static GraphStoreInner Load(LocalFileHandle &file_handle,
                                size_t cur_vertex_n,
                                size_t max_vertex,
                                const GraphStoreMeta &meta,
                                size_t &mem_usage,
                                size_t vertex_offset = 0,
                                size_t total_vertex_n = std::numeric_limits<size_t>::max()) {
        if (cur_vertex_n > max_vertex) {
            HnswStreamError("graph vertex count exceeds capacity");
        }
        if (total_vertex_n == std::numeric_limits<size_t>::max()) {
            total_vertex_n = cur_vertex_n;
        }
        const size_t layer_sum = HnswReadStream<size_t>(file_handle, "graph upper-layer count");
        const size_t graph_size = HnswStreamCheckedMultiply(cur_vertex_n, meta.level0_size(), "graph level-zero data");
        const size_t graph_capacity = HnswStreamCheckedMultiply(max_vertex, meta.level0_size(), "graph level-zero capacity");
        HnswEnsureStreamAvailable(file_handle, graph_size, "graph level-zero data");
        GraphStoreInner graph_store(max_vertex, meta, cur_vertex_n);
        std::fill(graph_store.graph_.get(), graph_store.graph_.get() + max_vertex * meta.level0_size(), 0);
        if (meta.Mmax0() != 0 || meta.Mmax() != 0) {
            graph_store.InitializeUnused(cur_vertex_n, max_vertex, meta);
        }
        HnswReadExact(file_handle, graph_store.graph_.get(), graph_size, "graph level-zero data");

        const size_t layers_size = ValidateStreamGraphRecords(graph_store.graph_.get(),
                                                              cur_vertex_n,
                                                              vertex_offset,
                                                              total_vertex_n,
                                                              meta.level0_size(),
                                                              meta.levelx_size(),
                                                              meta.Mmax0(),
                                                              layer_sum);
        HnswEnsureStreamAvailable(file_handle, layers_size, "graph upper-layer data");
        auto loaded_layers = std::make_unique<char[]>(layers_size);
        HnswReadExact(file_handle, loaded_layers.get(), layers_size, "graph upper-layer data");
        ValidateStreamUpperGraphRecords(graph_store.graph_.get(),
                                        loaded_layers.get(),
                                        cur_vertex_n,
                                        vertex_offset,
                                        total_vertex_n,
                                        meta.level0_size(),
                                        meta.levelx_size(),
                                        meta.Mmax());
        graph_store.loaded_layers_ = std::move(loaded_layers);

        mem_usage = HnswStreamCheckedAdd(
            mem_usage, HnswStreamCheckedAdd(graph_capacity, layers_size, "graph memory usage"), "graph memory usage");
        return graph_store;
    }

    static GraphStoreInner
    LoadFromPtr(HnswPointerReader &reader, size_t cur_vertex_n, size_t max_vertex, const GraphStoreMeta &meta, size_t &mem_usage) {
        if (cur_vertex_n > max_vertex) {
            HnswPointerImageError("graph vertex count exceeds capacity");
        }
        const size_t layer_sum = reader.Read<size_t>("graph upper-layer count");
        const size_t graph_size = HnswCheckedMultiply(cur_vertex_n, meta.level0_size(), "graph level-zero data");
        const size_t graph_capacity = HnswCheckedMultiply(max_vertex, meta.level0_size(), "graph level-zero capacity");
        reader.EnsureAvailable(graph_size, "graph level-zero data");
        GraphStoreInner graph_store(max_vertex, meta, cur_vertex_n);
        std::fill(graph_store.graph_.get(), graph_store.graph_.get() + max_vertex * meta.level0_size(), 0);
        if (meta.Mmax0() != 0 || meta.Mmax() != 0) {
            graph_store.InitializeUnused(cur_vertex_n, max_vertex, meta);
        }
        reader.CopyTo(graph_store.graph_.get(), graph_size, "graph level-zero data");
        const size_t layers_size = ValidatePointerGraphRecords(graph_store.graph_.get(),
                                                               cur_vertex_n,
                                                               meta.level0_size(),
                                                               meta.levelx_size(),
                                                               meta.Mmax0(),
                                                               layer_sum);
        const char *layers_source = reader.ReadBytes(layers_size, "graph upper-layer data");

        std::unique_ptr<char[]> loaded_layers;
        if (layers_size != 0) {
            loaded_layers = std::make_unique<char[]>(layers_size);
            std::memcpy(loaded_layers.get(), layers_source, layers_size);
        }
        ValidatePointerUpperGraphRecords(graph_store.graph_.get(),
                                         loaded_layers.get(),
                                         cur_vertex_n,
                                         meta.level0_size(),
                                         meta.levelx_size(),
                                         meta.Mmax(),
                                         layer_sum);
        char *loaded_layers_p = loaded_layers.get();
        for (VertexType vertex_i = 0; vertex_i < (VertexType)cur_vertex_n; ++vertex_i) {
            VertexL0 *v = graph_store.GetLevel0(vertex_i, meta);
            if (v->layer_n_) {
                const size_t vertex_layers_size =
                    HnswCheckedMultiply(meta.levelx_size(), static_cast<size_t>(v->layer_n_), "graph vertex upper layers");
                v->layers_p_ = loaded_layers_p;
                loaded_layers_p += vertex_layers_size;
            } else {
                v->layers_p_ = nullptr;
            }
        }
        graph_store.loaded_layers_ = std::move(loaded_layers);

        mem_usage = HnswCheckedAdd(mem_usage, HnswCheckedAdd(graph_capacity, layers_size, "graph memory usage"), "graph memory usage");
        return graph_store;
    }

    void AddVertex(VertexType vertex_i, i32 layer_n, const GraphStoreMeta &meta, size_t &mem_usage) {
        VertexL0 *v = GetLevel0(vertex_i, meta);
        if (v->layer_n_ >= 0) {
            throw std::logic_error("HNSW vertex is already built");
        }

        std::unique_ptr<char[]> layers;
        if (layer_n) {
            layers = std::make_unique<char[]>(meta.levelx_size() * static_cast<size_t>(layer_n));
            for (i32 layer_i = 1; layer_i <= layer_n; ++layer_i) {
                VertexLX *vx = GetLevelX(layers.get(), layer_i, meta);
                vx->neighbor_n_ = 0;
            }
        }

        v->neighbor_n_ = 0;
        v->layers_p_ = layers.release();
        v->layer_n_ = layer_n;
        mem_usage += meta.levelx_size() * static_cast<size_t>(layer_n);
    }

    std::pair<VertexType *, VertexListSize *> GetNeighborsMut(VertexType vertex_i, i32 layer_i, const GraphStoreMeta &meta) {
        VertexL0 *v = GetLevel0(vertex_i, meta);
        if (layer_i == 0) {
            return {v->neighbors_, &v->neighbor_n_};
        }
        VertexLX *vx = GetLevelX(v->layers_p_, layer_i, meta);
        return {vx->neighbors_, &vx->neighbor_n_};
    }

private:
    void InitializeUnused(size_t begin, size_t end, const GraphStoreMeta &meta) {
        for (size_t vertex_i = begin; vertex_i < end; ++vertex_i) {
            VertexL0 *v = GetLevel0(static_cast<VertexType>(vertex_i), meta);
            v->layer_n_ = -1;
            v->layers_p_ = nullptr;
            v->neighbor_n_ = 0;
        }
    }

    VertexL0 *GetLevel0(VertexType vertex_i, const GraphStoreMeta &meta) {
        return reinterpret_cast<VertexL0 *>(this->graph_.get() + vertex_i * meta.level0_size());
    }
    VertexLX *GetLevelX(char *layer_p, i32 layer_i, const GraphStoreMeta &meta) {
        assert(layer_i > 0);
        return reinterpret_cast<VertexLX *>(layer_p + (layer_i - 1) * meta.levelx_size());
    }

private:
    ArrayPtr<char, OwnMem> loaded_layers_;
    size_t loaded_vertex_n_{};
};

export template <>
class GraphStoreInner<false> : public GraphStoreInnerBase<false> {
public:
    using Base = GraphStoreInnerBase<false>;
    GraphStoreInner() = default;
    GraphStoreInner(GraphStoreInner &&other) noexcept
        : Base(std::move(other)), level0_sidecar_(std::move(other.level0_sidecar_)),
          upper_layers_sidecar_(std::move(other.upper_layers_sidecar_)),
          extra_memory_usage_(std::exchange(other.extra_memory_usage_, 0)) {
        RebindSidecars();
        other.ResetAliases();
    }
    GraphStoreInner &operator=(GraphStoreInner &&other) noexcept {
        if (this != &other) {
            Base::operator=(std::move(other));
            level0_sidecar_ = std::move(other.level0_sidecar_);
            upper_layers_sidecar_ = std::move(other.upper_layers_sidecar_);
            extra_memory_usage_ = std::exchange(other.extra_memory_usage_, 0);
            RebindSidecars();
            other.ResetAliases();
        }
        return *this;
    }

    static GraphStoreInner LoadFromPtr(HnswPointerReader &reader, size_t cur_vertex_n, size_t max_vertex, const GraphStoreMeta &meta) {
        if (cur_vertex_n > max_vertex) {
            HnswPointerImageError("graph vertex count exceeds capacity");
        }
        ValidatePointerGraphRecordAlignment(meta.level0_size(), meta.levelx_size());
        const size_t layer_sum = reader.Read<size_t>("graph upper-layer count");
        const size_t graph_size = HnswCheckedMultiply(cur_vertex_n, meta.level0_size(), "graph level-zero data");
        const char *serialized_graph = reader.ReadBytes(graph_size, "graph level-zero data");

        GraphStoreInner graph_store;
        graph_store.level0_sidecar_ = CopyGraphBytesIfMisaligned<VertexL0>(serialized_graph, graph_size);
        const char *graph = graph_store.level0_sidecar_ ? graph_store.level0_sidecar_.get() : serialized_graph;
        const size_t layers_size =
            ValidatePointerGraphRecords(graph, cur_vertex_n, meta.level0_size(), meta.levelx_size(), meta.Mmax0(), layer_sum);
        const char *serialized_layers = reader.ReadBytes(layers_size, "graph upper-layer data");
        graph_store.upper_layers_sidecar_ = CopyGraphBytesIfMisaligned<VertexLX>(serialized_layers, layers_size);
        const char *layers = graph_store.upper_layers_sidecar_ ? graph_store.upper_layers_sidecar_.get() : serialized_layers;
        ValidatePointerUpperGraphRecords(
            graph, layers, cur_vertex_n, meta.level0_size(), meta.levelx_size(), meta.Mmax(), layer_sum);

        graph_store.graph_ = graph;
        graph_store.layer_start_.set(layers);
        graph_store.extra_memory_usage_ =
            HnswCheckedAdd(graph_store.level0_sidecar_ ? graph_size : 0,
                           graph_store.upper_layers_sidecar_ ? layers_size : 0,
                           "mapped graph sidecar memory usage");
        return graph_store;
    }

    size_t ExtraMemoryUsage() const noexcept { return extra_memory_usage_; }

private:
    void RebindSidecars() noexcept {
        if (level0_sidecar_) {
            this->graph_ = level0_sidecar_.get();
        }
        if (upper_layers_sidecar_) {
            this->layer_start_.set(upper_layers_sidecar_.get());
        }
    }

    void ResetAliases() noexcept {
        this->graph_ = ArrayPtr<char, false>{};
        this->layer_start_ = PPtr<false>{};
    }

private:
    GraphAlignedBytePtr<VertexL0> level0_sidecar_;
    GraphAlignedBytePtr<VertexLX> upper_layers_sidecar_;
    size_t extra_memory_usage_ = 0;
};

} // namespace infinity
