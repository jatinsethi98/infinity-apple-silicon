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

#include <common/simd/simd_functions.h>

export module infinity_core:sparse_vec_store;

import :local_file_handle;
import :hnsw_common;
import :data_store_util;
import :sparse_util;

import std;

namespace infinity {

export template <typename DataType, typename IdxType>
class SparseVecStoreMeta {
public:
    using This = SparseVecStoreMeta<DataType, IdxType>;
    using QueryVecType = SparseVecRef<DataType, IdxType>;
    using StoreType = SparseVecRef<DataType, IdxType>;
    using QueryType = SparseVecRef<DataType, IdxType>;
    using DistanceType = std::conditional_t<std::is_same_v<DataType, bool>, IdxType, std::conditional_t<std::is_same_v<DataType, f64>, f64, f32>>;

private:
    SparseVecStoreMeta(size_t max_dim) : max_dim_(max_dim) {}

public:
    SparseVecStoreMeta() = default;
    static This Make(size_t max_dim) { return This(max_dim); }
    static This Make(size_t max_dim, bool) { return This(max_dim); }

    void Save(LocalFileHandle &file_handle) const { file_handle.Append(&max_dim_, sizeof(max_dim_)); }

    static This Load(LocalFileHandle &file_handle) {
        const size_t max_dim = HnswReadStream<size_t>(file_handle, "sparse vector dimension");
        if (max_dim == 0 || max_dim > static_cast<size_t>(std::numeric_limits<IdxType>::max())) {
            HnswStreamError("sparse vector dimension is outside the index representation");
        }
        static_cast<void>(HnswStreamCheckedMultiply(sizeof(DataType), max_dim, "sparse vector dimension"));
        return This(max_dim);
    }

    // Get size of vector in search
    size_t GetVecSizeInBytes() const { return sizeof(DataType) * max_dim_; }

    QueryType MakeQuery(QueryVecType vec) const { return vec; }

    size_t dim() const { return max_dim_; }

private:
    size_t max_dim_;

public:
    void Dump(std::ostream &os) const { os << "[CONST] max dim: " << max_dim_ << std::endl; }
};

export template <typename DataType, typename IdxType>
class SparseVecStoreInner {
public:
    using This = SparseVecStoreInner<DataType, IdxType>;
    using Meta = SparseVecStoreMeta<DataType, IdxType>;
    using SparseVecRef = SparseVecRef<DataType, IdxType>;
    using SparseVecEle = SparseVecEle<DataType, IdxType>;

private:
    SparseVecStoreInner(size_t max_vec_num, const Meta &meta) : vecs_(std::make_unique_for_overwrite<SparseVecEle[]>(max_vec_num)) {}

public:
    SparseVecStoreInner() = default;

    static This Make(size_t max_vec_num, const Meta &meta, size_t &mem_usage) {
        auto ret = This(max_vec_num, meta);
        mem_usage += sizeof(SparseVecEle) * max_vec_num;
        return ret;
    }

    void Save(LocalFileHandle &file_handle, size_t cur_vec_num, const Meta &meta) const {
        size_t nnz = 0;
        for (size_t i = 0; i < cur_vec_num; ++i) {
            nnz += vecs_[i].nnz_;
        }
        file_handle.Append(&nnz, sizeof(nnz));
        auto indptr = std::make_unique_for_overwrite<i32[]>(cur_vec_num + 1);
        indptr[0] = 0;
        auto indice = std::make_unique_for_overwrite<IdxType[]>(nnz);
        auto data = std::make_unique_for_overwrite<DataType[]>(nnz);
        for (size_t i = 0; i < cur_vec_num; ++i) {
            const SparseVecEle &vec = vecs_[i];
            std::copy(vec.indices_.get(), vec.indices_.get() + vec.nnz_, indice.get() + indptr[i]);
            std::copy(vec.data_.get(), vec.data_.get() + vec.nnz_, data.get() + indptr[i]);
            indptr[i + 1] = indptr[i] + vec.nnz_;
        }
        file_handle.Append(indptr.get(), sizeof(i32) * (cur_vec_num + 1));
        file_handle.Append(indice.get(), sizeof(IdxType) * nnz);
        file_handle.Append(data.get(), sizeof(DataType) * nnz);
    }

    static This Load(LocalFileHandle &file_handle, size_t cur_vec_num, size_t max_vec_num, const Meta &meta, size_t &mem_usage) {
        if (cur_vec_num > max_vec_num) {
            HnswStreamError("sparse vector count exceeds capacity");
        }
        const size_t nnz = HnswReadStream<size_t>(file_handle, "sparse nonzero count");
        if (nnz > static_cast<size_t>(std::numeric_limits<i32>::max())) {
            HnswStreamError("sparse nonzero count exceeds the row-offset representation");
        }
        const size_t indptr_count = HnswStreamCheckedAdd(cur_vec_num, 1, "sparse row offsets");
        const size_t indptr_size = HnswStreamCheckedMultiply(sizeof(i32), indptr_count, "sparse row offsets");
        const size_t indices_size = HnswStreamCheckedMultiply(sizeof(IdxType), nnz, "sparse indices");
        const size_t data_size = HnswStreamCheckedMultiply(sizeof(DataType), nnz, "sparse values");
        const size_t payload_size =
            HnswStreamCheckedAdd(indptr_size, HnswStreamCheckedAdd(indices_size, data_size, "sparse payload"), "sparse payload");
        HnswEnsureStreamAvailable(file_handle, payload_size, "sparse payload");

        auto indptr = std::make_unique_for_overwrite<i32[]>(indptr_count);
        HnswReadExact(file_handle, indptr.get(), indptr_size, "sparse row offsets");
        auto indice = std::make_unique_for_overwrite<IdxType[]>(nnz);
        HnswReadExact(file_handle, indice.get(), indices_size, "sparse indices");
        auto data = std::make_unique_for_overwrite<DataType[]>(nnz);
        HnswReadExact(file_handle, data.get(), data_size, "sparse values");

        if (indptr[0] != 0) {
            HnswStreamError("sparse row offsets must start at zero");
        }
        for (size_t row = 0; row < cur_vec_num; ++row) {
            if (indptr[row] < 0 || indptr[row + 1] < indptr[row] ||
                static_cast<size_t>(indptr[row + 1]) > nnz) {
                HnswStreamError("sparse row offsets are not monotonic and bounded");
            }
            for (i32 position = indptr[row]; position < indptr[row + 1]; ++position) {
                const IdxType index = indice[position];
                if constexpr (std::is_signed_v<IdxType>) {
                    if (index < 0) {
                        HnswStreamError("sparse index is negative");
                    }
                }
                if (static_cast<size_t>(index) >= meta.dim() ||
                    (position > indptr[row] && indice[position - 1] >= index)) {
                    HnswStreamError("sparse row indices must be in range and strictly increasing");
                }
            }
        }
        if (static_cast<size_t>(indptr[cur_vec_num]) != nnz) {
            HnswStreamError("sparse final row offset does not match the nonzero count");
        }

        const size_t vec_array_size = HnswStreamCheckedMultiply(sizeof(SparseVecEle), max_vec_num, "sparse vector capacity");
        const size_t element_size = HnswStreamCheckedAdd(sizeof(IdxType), sizeof(DataType), "sparse element");
        const size_t element_memory = HnswStreamCheckedMultiply(element_size, nnz, "sparse elements");
        const size_t loaded_memory = HnswStreamCheckedAdd(vec_array_size, element_memory, "sparse vector memory usage");
        This ret(max_vec_num, meta);
        for (size_t i = 0; i < cur_vec_num; ++i) { // todo: optimize it
            SparseVecEle &vec = ret.vecs_[i];
            vec.nnz_ = indptr[i + 1] - indptr[i];
            vec.indices_ = std::make_unique_for_overwrite<IdxType[]>(vec.nnz_);
            vec.data_ = std::make_unique_for_overwrite<DataType[]>(vec.nnz_);

            std::copy(indice.get() + indptr[i], indice.get() + indptr[i + 1], vec.indices_.get());
            std::copy(data.get() + indptr[i], data.get() + indptr[i + 1], vec.data_.get());
        }
        mem_usage = HnswStreamCheckedAdd(mem_usage, loaded_memory, "sparse vector memory usage");
        return ret;
    }

    void SetVec(size_t idx, const SparseVecRef &vec, const Meta &meta, size_t &mem_usage) {
        SparseVecEle &dst = vecs_[idx];
        dst.nnz_ = vec.nnz_;
        dst.indices_ = std::make_unique_for_overwrite<IdxType[]>(vec.nnz_);
        dst.data_ = std::make_unique_for_overwrite<DataType[]>(vec.nnz_);
        mem_usage += sizeof(IdxType) * vec.nnz_ + sizeof(DataType) * vec.nnz_;

        std::copy(vec.indices_, vec.indices_ + vec.nnz_, dst.indices_.get());
        std::copy(vec.data_, vec.data_ + vec.nnz_, dst.data_.get());
    }

    SparseVecRef GetVec(size_t idx, const Meta &meta) const {
        const SparseVecEle &vec = vecs_[idx];
        return SparseVecRef(vec.nnz_, vec.indices_.get(), vec.data_.get());
    }

    SparseVecRef GetVecToQuery(size_t idx, const Meta &meta) const { return GetVec(idx, meta); }

    void Prefetch(size_t idx, const Meta &meta) const {
        const SparseVecEle &vec = vecs_[idx];
        SIMDPrefetch(vec.indices_.get());
        SIMDPrefetch(vec.data_.get());
    }

private:
    std::unique_ptr<SparseVecEle[]> vecs_;

public:
    void Dump(std::ostream &os, size_t offset, size_t chunk_size, const Meta &meta) const {
        for (int i = 0; i < (int)chunk_size; ++i) {
            os << "vec " << i << "(" << i + offset << "): ";
            const SparseVecEle &vec = vecs_[i];
            for (i32 j = 0; j < vec.nnz_; ++j) {
                os << vec.indices_[j] << ":" << vec.data_[j] << " ";
            }
            os << std::endl;
        }
    }
};

} // namespace infinity
