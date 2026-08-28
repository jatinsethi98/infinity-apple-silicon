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
#include <cstddef>

#include <common/simd/simd_functions.h>

export module infinity_core:lvq_vec_store;

import :local_file_handle;
import :hnsw_common;
import :data_store_util;

import std;

import serialize;

namespace infinity {

export template <typename DataType, typename LocalCacheType, typename CompressType>
struct LVQData {
    DataType scale_;
    DataType bias_;
    LocalCacheType local_cache_;
    CompressType compress_vec_[];
};

export template <typename DataType, typename LocalCacheType, typename CompressType>
class LVQDataView {
    using This = LVQDataView<DataType, LocalCacheType, CompressType>;
    using Layout = LVQData<DataType, LocalCacheType, CompressType>;
    using CacheFirst = typename LocalCacheType::first_type;
    using CacheSecond = typename LocalCacheType::second_type;

    static_assert(std::is_standard_layout_v<Layout>);
    static_assert(std::is_standard_layout_v<LocalCacheType>);
    static_assert(sizeof(Layout) == offsetof(Layout, compress_vec_));
    static_assert(alignof(CompressType) == 1);

public:
    LVQDataView() = default;
    explicit LVQDataView(const char *data)
        : scale_(HnswLoadUnaligned<DataType>(data + offsetof(Layout, scale_))), bias_(HnswLoadUnaligned<DataType>(data + offsetof(Layout, bias_))),
          local_cache_{HnswLoadUnaligned<CacheFirst>(data + offsetof(Layout, local_cache_) + offsetof(LocalCacheType, first)),
                       HnswLoadUnaligned<CacheSecond>(data + offsetof(Layout, local_cache_) + offsetof(LocalCacheType, second))},
          compress_vec_(reinterpret_cast<const CompressType *>(data + sizeof(Layout))) {}

    const This *operator->() const noexcept { return this; }

    DataType scale_{};
    DataType bias_{};
    LocalCacheType local_cache_{};
    const CompressType *compress_vec_{nullptr};
};

export template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreInner;

export template <typename DataType, typename CompressType, typename LVQCache>
class LVQVecStoreMetaType {
public:
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    using StoredDataView = LVQDataView<DataType, LocalCacheType, CompressType>;
    struct LVQQuery {
        std::unique_ptr<char[]> inner_;
        StoredDataView view_;

        const StoredDataView *operator->() const noexcept { return &view_; }
        char *data() const noexcept { return inner_.get(); }
        void Refresh() { view_ = StoredDataView(inner_.get()); }

        explicit LVQQuery(size_t compress_data_size) : inner_(std::make_unique_for_overwrite<char[]>(compress_data_size)) {}
        LVQQuery(size_t compress_data_size, const char *data) : LVQQuery(compress_data_size) {
            std::memcpy(inner_.get(), data, compress_data_size);
            Refresh();
        }
        LVQQuery(LVQQuery &&other) = default;
        LVQQuery &operator=(LVQQuery &&other) = default;
    };

    using StoreType = StoredDataView;
    using QueryType = LVQQuery;
    using DistanceType = f32;
};

template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreMetaBase {
public:
    // Compress type must be i8 temporarily
    static_assert(std::is_same<CompressType, i8>() || std::is_same<CompressType, void>());
    constexpr static size_t max_bucket_idx_ = std::numeric_limits<CompressType>::max() - std::numeric_limits<CompressType>::min(); // 255 for i8

    using This = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem>;
    using Inner = LVQVecStoreInner<DataType, CompressType, LVQCache, OwnMem>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using GlobalCacheType = LVQCache::GlobalCacheType;
    using LVQData = LVQVecStoreMetaType<DataType, CompressType, LVQCache>::LVQData;
    using LVQQuery = LVQVecStoreMetaType<DataType, CompressType, LVQCache>::LVQQuery;
    using StoreType = LVQVecStoreMetaType<DataType, CompressType, LVQCache>::StoreType;
    using QueryType = LVQVecStoreMetaType<DataType, CompressType, LVQCache>::QueryType;
    using DistanceType = f32;

public:
    LVQVecStoreMetaBase() : dim_(0), compress_data_size_(0), normalize_(false) {}
    LVQVecStoreMetaBase(This &&other) noexcept
        : dim_(std::exchange(other.dim_, 0)), compress_data_size_(std::exchange(other.compress_data_size_, 0)), mean_(std::move(other.mean_)),
          global_cache_(std::exchange(other.global_cache_, GlobalCacheType())), normalize_(other.normalize_) {}
    LVQVecStoreMetaBase &operator=(This &&other) noexcept {
        static_assert(std::is_nothrow_move_assignable_v<decltype(mean_)>);
        static_assert(std::is_nothrow_move_assignable_v<GlobalCacheType>);
        if (this != &other) {
            dim_ = std::exchange(other.dim_, 0);
            compress_data_size_ = std::exchange(other.compress_data_size_, 0);
            mean_ = std::move(other.mean_);
            global_cache_ = std::exchange(other.global_cache_, GlobalCacheType());
            normalize_ = other.normalize_;
        }
        return *this;
    }

    size_t GetSizeInBytes() const { return sizeof(dim_) + sizeof(MeanType) * dim_ + sizeof(GlobalCacheType); }

    // Get size of vector in search
    size_t GetVecSizeInBytes() const { return compress_data_size_; }

    void Save(LocalFileHandle &file_handle) const {
        file_handle.Append(&dim_, sizeof(dim_));
        file_handle.Append(mean_.get(), sizeof(MeanType) * dim_);
        if constexpr (!std::same_as<GlobalCacheType, std::tuple<>>) {
            file_handle.Append(&global_cache_, sizeof(GlobalCacheType));
        }
    }

    LVQQuery MakeQuery(const DataType *vec) const {
        LVQQuery query(compress_data_size_);
        CompressTo(vec, query.data());
        query.Refresh();
        return query;
    }

    void CompressTo(const DataType *src, char *dest) const {
        std::unique_ptr<DataType[]> normalized;
        if (normalize_) {
            normalized = std::make_unique_for_overwrite<DataType[]>(this->dim_);
            DataType norm = 0;
            for (size_t j = 0; j < this->dim_; ++j) {
                DataType x = src[j];
                norm += x * x;
            }
            norm = std::sqrt(norm);
            if (norm == 0) {
                std::fill(normalized.get(), normalized.get() + this->dim_, 0);
            } else {
                for (size_t j = 0; j < this->dim_; ++j) {
                    normalized[j] = src[j] / norm;
                }
            }
            src = normalized.get();
        }

        static_assert(sizeof(LVQData) == offsetof(LVQData, compress_vec_));
        static_assert(alignof(CompressType) == 1);
        std::memset(dest, 0, sizeof(LVQData));
        CompressType *compress = reinterpret_cast<CompressType *>(dest + sizeof(LVQData));

        DataType lower = std::numeric_limits<DataType>::max();
        DataType upper = -std::numeric_limits<DataType>::max();
        for (size_t j = 0; j < dim_; ++j) {
            auto x = static_cast<DataType>(src[j] - mean_[j]);
            lower = std::min(lower, x);
            upper = std::max(upper, x);
        }
        DataType scale = (upper - lower) / max_bucket_idx_;
        DataType bias = lower - std::numeric_limits<CompressType>::min() * scale;
        if (scale == 0) {
            std::fill(compress, compress + dim_, 0);
        } else {
            DataType scale_inv = 1 / scale;
            for (size_t j = 0; j < dim_; ++j) {
                auto c = std::floor((src[j] - mean_[j] - bias) * scale_inv + 0.5);
                assert(c <= std::numeric_limits<CompressType>::max() && c >= std::numeric_limits<CompressType>::min());
                compress[j] = c;
            }
        }
        const LocalCacheType local_cache = LVQCache::MakeLocalCache(compress, scale, dim_, mean_.get());
        using CacheFirst = typename LocalCacheType::first_type;
        using CacheSecond = typename LocalCacheType::second_type;
        static_assert(std::is_standard_layout_v<LVQData>);
        static_assert(std::is_standard_layout_v<LocalCacheType>);
        HnswStoreUnaligned(dest + offsetof(LVQData, scale_), scale);
        HnswStoreUnaligned(dest + offsetof(LVQData, bias_), bias);
        HnswStoreUnaligned(dest + offsetof(LVQData, local_cache_) + offsetof(LocalCacheType, first),
                           static_cast<const CacheFirst &>(local_cache.first));
        HnswStoreUnaligned(dest + offsetof(LVQData, local_cache_) + offsetof(LocalCacheType, second),
                           static_cast<const CacheSecond &>(local_cache.second));
    }

    size_t dim() const { return dim_; }
    size_t compress_data_size() const { return compress_data_size_; }

    const GlobalCacheType &global_cache() const { return global_cache_; }

    // for unit test
    const MeanType *mean() const { return mean_.get(); }

protected:
    void DecompressByMeanTo(const StoreType &src, const MeanType *mean, DataType *dest) const {
        const CompressType *compress = src->compress_vec_;
        DataType scale = src->scale_;
        DataType bias = src->bias_;
        for (size_t i = 0; i < dim_; ++i) {
            dest[i] = scale * compress[i] + bias + mean[i];
        }
    }

    void DecompressTo(const StoreType &src, DataType *dest) const { DecompressByMeanTo(src, mean_.get(), dest); };

protected:
    size_t dim_;
    size_t compress_data_size_;

    ArrayPtr<MeanType, OwnMem> mean_;
    GlobalCacheType global_cache_;

    bool normalize_{false};

public:
    void Dump(std::ostream &os) const {
        os << "[CONST] dim: " << dim_ << ", compress_data_size: " << compress_data_size_ << std::endl;
        os << "mean: ";
        for (size_t i = 0; i < dim_; ++i) {
            os << mean_[i] << " ";
        }
        os << std::endl;
        LVQCache::DumpGlobalCache(os, global_cache_);
    }
};

export template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreMeta : public LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem> {
    using This = LVQVecStoreMeta<DataType, CompressType, LVQCache, OwnMem>;
    using Inner = LVQVecStoreInner<DataType, CompressType, LVQCache, OwnMem>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    using GlobalCacheType = LVQCache::GlobalCacheType;

private:
    LVQVecStoreMeta(size_t dim) {
        this->dim_ = dim;
        this->compress_data_size_ = sizeof(LVQData) + sizeof(CompressType) * dim;
        this->mean_ = std::make_unique<MeanType[]>(dim);
        std::fill(this->mean_.get(), this->mean_.get() + dim, 0);
        this->global_cache_ = LVQCache::MakeGlobalCache(this->mean_.get(), dim);
    }

public:
    LVQVecStoreMeta() = default;
    static This Make(size_t dim) { return This(dim); }
    static This Make(size_t dim, bool normalize) {
        This ret(dim);
        ret.normalize_ = normalize;
        return ret;
    }

    static This Load(LocalFileHandle &file_handle) {
        const size_t dim = HnswReadStream<size_t>(file_handle, "LVQ dimension");
        if (dim == 0) {
            HnswStreamError("LVQ dimension must be nonzero");
        }
        static_cast<void>(HnswStreamCheckedAdd(sizeof(LVQData),
                                               HnswStreamCheckedMultiply(sizeof(CompressType), dim, "LVQ compressed vector"),
                                               "LVQ compressed vector"));
        const size_t mean_size = HnswStreamCheckedMultiply(sizeof(MeanType), dim, "LVQ mean");
        const size_t cache_size = std::is_same_v<GlobalCacheType, std::tuple<>> ? 0 : sizeof(GlobalCacheType);
        HnswEnsureStreamAvailable(
            file_handle, HnswStreamCheckedAdd(mean_size, cache_size, "LVQ metadata"), "LVQ metadata");
        This meta(dim);
        HnswReadExact(file_handle, meta.mean_.get(), mean_size, "LVQ mean");
        if constexpr (!std::is_same_v<GlobalCacheType, std::tuple<>>) {
            HnswReadExact(file_handle, &meta.global_cache_, sizeof(GlobalCacheType), "LVQ global cache");
        }
        return meta;
    }

    static This LoadFromPtr(HnswPointerReader &reader) {
        const size_t dim = reader.Read<size_t>("LVQ dimension");
        if (dim == 0) {
            HnswPointerImageError("LVQ dimension must be nonzero");
        }
        static_cast<void>(
            HnswCheckedAdd(sizeof(LVQData), HnswCheckedMultiply(sizeof(CompressType), dim, "LVQ compressed vector"), "LVQ compressed vector"));
        const size_t mean_size = HnswCheckedMultiply(sizeof(MeanType), dim, "LVQ mean");
        const size_t cache_size = std::is_same_v<GlobalCacheType, std::tuple<>> ? 0 : sizeof(GlobalCacheType);
        reader.EnsureAvailable(HnswCheckedAdd(mean_size, cache_size, "LVQ metadata"), "LVQ metadata");
        This meta(dim);
        reader.CopyTo(meta.mean_.get(), mean_size, "LVQ mean");
        if constexpr (!std::is_same_v<GlobalCacheType, std::tuple<>>) {
            using CacheFirst = typename GlobalCacheType::first_type;
            using CacheSecond = typename GlobalCacheType::second_type;
            static_assert(sizeof(GlobalCacheType) == sizeof(CacheFirst) + sizeof(CacheSecond));
            meta.global_cache_ =
                GlobalCacheType{reader.Read<CacheFirst>("LVQ global cache first value"), reader.Read<CacheSecond>("LVQ global cache second value")};
        }
        return meta;
    }

    template <typename LabelType, DataIteratorConcept<const DataType *, LabelType> Iterator>
    void Optimize(Iterator &&query_iter, const std::vector<std::pair<Inner *, size_t>> &inners, size_t &mem_usage) {
        auto new_mean = std::make_unique<MeanType[]>(this->dim_);
        auto temp_decompress = std::make_unique<DataType[]>(this->dim_);
        size_t cur_vec_num = 0;
        for (const auto [inner, size] : inners) {
            for (size_t i = 0; i < size; ++i) {
                this->DecompressTo(inner->GetVec(i, *this), temp_decompress.get());
                for (size_t j = 0; j < this->dim_; ++j) {
                    new_mean[j] += temp_decompress[j];
                }
            }
            cur_vec_num += size;
        }
        while (true) {
            if (auto ret = query_iter.Next(); ret) {
                auto &[vec, _] = *ret;
                for (size_t i = 0; i < this->dim_; ++i) {
                    new_mean[i] += vec[i];
                }
                ++cur_vec_num;
            } else {
                break;
            }
        }
        for (size_t i = 0; i < this->dim_; ++i) {
            new_mean[i] /= cur_vec_num;
        }
        new_mean = this->mean_.exchange(std::move(new_mean)); //

        for (auto [inner, size] : inners) {
            for (size_t i = 0; i < size; ++i) {
                this->DecompressByMeanTo(inner->GetVec(i, *this), new_mean.get(), temp_decompress.get());
                inner->SetVec(i, temp_decompress.get(), *this, mem_usage);
            }
        }
        this->global_cache_ = LVQCache::MakeGlobalCache(this->mean_.get(), this->dim_);
    }
};

export template <typename DataType, typename CompressType, typename LVQCache>
class LVQVecStoreMeta<DataType, CompressType, LVQCache, false> : public LVQVecStoreMetaBase<DataType, CompressType, LVQCache, false> {
    using This = LVQVecStoreMeta<DataType, CompressType, LVQCache, false>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    using GlobalCacheType = LVQCache::GlobalCacheType;

private:
    LVQVecStoreMeta(size_t dim, MeanType *mean, GlobalCacheType global_cache) {
        this->dim_ = dim;
        this->compress_data_size_ = sizeof(LVQData) + sizeof(CompressType) * dim;
        this->mean_ = mean;
        this->global_cache_ = global_cache;
    }

public:
    LVQVecStoreMeta() = default;

    static This LoadFromPtr(HnswPointerReader &reader) {
        const size_t dim = reader.Read<size_t>("LVQ dimension");
        if (dim == 0) {
            HnswPointerImageError("LVQ dimension must be nonzero");
        }
        static_cast<void>(
            HnswCheckedAdd(sizeof(LVQData), HnswCheckedMultiply(sizeof(CompressType), dim, "LVQ compressed vector"), "LVQ compressed vector"));
        auto *mean = const_cast<MeanType *>(reader.ReadArray<MeanType>(dim, "LVQ mean"));
        GlobalCacheType global_cache{};
        if constexpr (!std::is_same_v<GlobalCacheType, std::tuple<>>) {
            using CacheFirst = typename GlobalCacheType::first_type;
            using CacheSecond = typename GlobalCacheType::second_type;
            static_assert(sizeof(GlobalCacheType) == sizeof(CacheFirst) + sizeof(CacheSecond));
            global_cache =
                GlobalCacheType{reader.Read<CacheFirst>("LVQ global cache first value"), reader.Read<CacheSecond>("LVQ global cache second value")};
        }
        This meta(dim, mean, global_cache);
        return meta;
    }
};

template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreInnerBase {
public:
    using This = LVQVecStoreInnerBase<DataType, CompressType, LVQCache, OwnMem>;
    using Meta = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem>;
    // Decompress: Q = scale * C + bias + Mean
    using StoreType = Meta::StoreType;
    using QueryType = Meta::QueryType;

public:
    LVQVecStoreInnerBase() = default;

    size_t GetSizeInBytes(size_t cur_vec_num, const Meta &meta) const { return cur_vec_num * meta.compress_data_size(); }

    void Save(LocalFileHandle &file_handle, size_t cur_vec_num, const Meta &meta) const {
        file_handle.Append(ptr_.get(), cur_vec_num * meta.compress_data_size());
    }

    static void SaveToPtr(LocalFileHandle &file_handle,
                          const std::vector<const This *> &inners,
                          const Meta &meta,
                          size_t ck_size,
                          size_t chunk_num,
                          size_t last_chunk_size) {
        for (size_t i = 0; i < chunk_num; ++i) {
            size_t chunk_size = (i < chunk_num - 1) ? ck_size : last_chunk_size;
            file_handle.Append(inners[i]->ptr_.get(), chunk_size * meta.compress_data_size());
        }
    }

    StoreType GetVec(size_t idx, const Meta &meta) const { return StoreType(ptr_.get() + idx * meta.compress_data_size()); }

    QueryType GetVecToQuery(size_t idx, const Meta &meta) const {
        return QueryType(meta.compress_data_size(), ptr_.get() + idx * meta.compress_data_size());
    }

    void Prefetch(VertexType vec_i, const Meta &meta) const {
        SIMDPrefetch(static_cast<const void *>(ptr_.get() + vec_i * meta.compress_data_size()));
    }

protected:
    ArrayPtr<char, OwnMem> ptr_;

public:
    void Dump(std::ostream &os, size_t offset, size_t chunk_size, const Meta &meta) const {
        for (int i = 0; i < (int)chunk_size; ++i) {
            os << "vec " << i << "(" << offset + i << "): ";
            StoreType vec = GetVec(i, meta);
            os << "scale: " << vec->scale_ << ", bias: " << vec->bias_ << std::endl;
            os << "compress_vec: ";
            for (size_t j = 0; j < meta.dim(); ++j) {
                os << static_cast<int>(vec->compress_vec_[j]) << " ";
            }
            os << std::endl;
            LVQCache::DumpLocalCache(os, vec->local_cache_);
        }
    }
};

export template <typename DataType, typename CompressType, typename LVQCache, bool OwnMem>
class LVQVecStoreInner : public LVQVecStoreInnerBase<DataType, CompressType, LVQCache, OwnMem> {
public:
    using This = LVQVecStoreInner<DataType, CompressType, LVQCache, OwnMem>;
    using Meta = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, OwnMem>;
    using LocalCacheType = LVQCache::LocalCacheType;
    using LVQData = LVQData<DataType, LocalCacheType, CompressType>;
    using Base = LVQVecStoreInnerBase<DataType, CompressType, LVQCache, OwnMem>;

private:
    LVQVecStoreInner(size_t max_vec_num, const Meta &meta) { this->ptr_ = std::make_unique<char[]>(max_vec_num * meta.compress_data_size()); }

public:
    LVQVecStoreInner() = default;

    static This Make(size_t max_vec_num, const Meta &meta, size_t &mem_usage) {
        auto ret = This(max_vec_num, meta);
        mem_usage += max_vec_num * meta.compress_data_size();
        return ret;
    }

    static This Load(LocalFileHandle &file_handle, size_t cur_vec_num, size_t max_vec_num, const Meta &meta, size_t &mem_usage) {
        if (cur_vec_num > max_vec_num) {
            HnswStreamError("LVQ vector count exceeds capacity");
        }
        const size_t stored_size = HnswStreamCheckedMultiply(cur_vec_num, meta.compress_data_size(), "LVQ vector data");
        const size_t max_size = HnswStreamCheckedMultiply(max_vec_num, meta.compress_data_size(), "LVQ vector capacity");
        HnswEnsureStreamAvailable(file_handle, stored_size, "LVQ vector data");
        This ret(max_vec_num, meta);
        HnswReadExact(file_handle, ret.ptr_.get(), stored_size, "LVQ vector data");
        mem_usage = HnswStreamCheckedAdd(mem_usage, max_size, "LVQ vector memory usage");
        return ret;
    }

    static This LoadFromPtr(HnswPointerReader &reader, size_t cur_vec_num, size_t max_vec_num, const Meta &meta, size_t &mem_usage) {
        if (cur_vec_num > max_vec_num) {
            HnswPointerImageError("LVQ vector count exceeds capacity");
        }
        const size_t stored_size = HnswCheckedMultiply(cur_vec_num, meta.compress_data_size(), "LVQ vector data");
        const size_t max_size = HnswCheckedMultiply(max_vec_num, meta.compress_data_size(), "LVQ vector capacity");
        reader.EnsureAvailable(stored_size, "LVQ vector data");
        This ret(max_vec_num, meta);
        reader.CopyTo(ret.ptr_.get(), stored_size, "LVQ vector data");
        mem_usage = HnswCheckedAdd(mem_usage, max_size, "LVQ vector memory usage");
        return ret;
    }

    void SetVec(size_t idx, const DataType *vec, const Meta &meta, size_t &mem_usage) { meta.CompressTo(vec, GetVecMut(idx, meta)); }

private:
    char *GetVecMut(size_t idx, const Meta &meta) { return this->ptr_.get() + idx * meta.compress_data_size(); }
};

export template <typename DataType, typename CompressType, typename LVQCache>
class LVQVecStoreInner<DataType, CompressType, LVQCache, false> : public LVQVecStoreInnerBase<DataType, CompressType, LVQCache, false> {
public:
    using This = LVQVecStoreInner<DataType, CompressType, LVQCache, false>;
    using Meta = LVQVecStoreMetaBase<DataType, CompressType, LVQCache, false>;
    using Base = LVQVecStoreInnerBase<DataType, CompressType, LVQCache, false>;

private:
    LVQVecStoreInner(const char *ptr) { this->ptr_ = ptr; }

public:
    LVQVecStoreInner() = default;

    static This LoadFromPtr(HnswPointerReader &reader, size_t cur_vec_num, const Meta &meta) {
        const size_t size = HnswCheckedMultiply(cur_vec_num, meta.compress_data_size(), "LVQ vector data");
        return This(reader.ReadBytes(size, "LVQ vector data"));
    }
};

} // namespace infinity
