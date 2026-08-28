//
// Created by infiniflow on 25-5-13.
//

module;

module infinity_core:hnsw_handler_core.impl;

import :hnsw_handler;
import :default_values;
import :infinity_exception;

import embedding_info;
import embedding_type;
import column_def;

namespace infinity {

template <typename DataType, bool OwnMem>
AbstractHnsw InitAbstractIndexT(const IndexHnsw *index_hnsw) {
    switch (index_hnsw->encode_type_) {
        case HnswEncodeType::kPlain: {
            if (index_hnsw->build_type_ == HnswBuildType::kLSG) {
                switch (index_hnsw->metric_type_) {
                    case MetricType::kMetricL2: {
                        using HnswIndex = KnnHnsw<PlainL2VecStoreType<DataType, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricInnerProduct: {
                        using HnswIndex = KnnHnsw<PlainIPVecStoreType<DataType, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricCosine: {
                        using HnswIndex = KnnHnsw<PlainCosVecStoreType<DataType, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    default: {
                        return nullptr;
                    }
                }
            } else if (index_hnsw->build_type_ == HnswBuildType::kPlain) {
                switch (index_hnsw->metric_type_) {
                    case MetricType::kMetricL2: {
                        using HnswIndex = KnnHnsw<PlainL2VecStoreType<DataType>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricInnerProduct: {
                        using HnswIndex = KnnHnsw<PlainIPVecStoreType<DataType>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricCosine: {
                        using HnswIndex = KnnHnsw<PlainCosVecStoreType<DataType>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    default: {
                        return nullptr;
                    }
                }
            } else {
                return nullptr;
            }
        }
        case HnswEncodeType::kLVQ: {
            if constexpr (std::is_same_v<DataType, u8> || std::is_same_v<DataType, i8>) {
                return nullptr;
            } else if (index_hnsw->build_type_ == HnswBuildType::kLSG) {
                switch (index_hnsw->metric_type_) {
                    case MetricType::kMetricL2: {
                        using HnswIndex = KnnHnsw<LVQL2VecStoreType<DataType, i8, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricInnerProduct: {
                        using HnswIndex = KnnHnsw<LVQIPVecStoreType<DataType, i8, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricCosine: {
                        using HnswIndex = KnnHnsw<LVQCosVecStoreType<DataType, i8, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    default: {
                        return nullptr;
                    }
                }
            } else if (index_hnsw->build_type_ == HnswBuildType::kPlain) {
                switch (index_hnsw->metric_type_) {
                    case MetricType::kMetricL2: {
                        using HnswIndex = KnnHnsw<LVQL2VecStoreType<DataType, i8>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricInnerProduct: {
                        using HnswIndex = KnnHnsw<LVQIPVecStoreType<DataType, i8>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricCosine: {
                        using HnswIndex = KnnHnsw<LVQCosVecStoreType<DataType, i8>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    default: {
                        return nullptr;
                    }
                }
            } else {
                return nullptr;
            }
        }
        case HnswEncodeType::kRabitq: {
            if constexpr (!std::is_same_v<DataType, f32>) {
                return nullptr;
            } else if (index_hnsw->build_type_ == HnswBuildType::kLSG) {
                switch (index_hnsw->metric_type_) {
                    case MetricType::kMetricL2: {
                        using HnswIndex = KnnHnsw<RabitqL2VecStoreType<DataType, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricInnerProduct: {
                        using HnswIndex = KnnHnsw<RabitqIPVecStoreType<DataType, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricCosine: {
                        using HnswIndex = KnnHnsw<RabitqCosVecStoreType<DataType, true>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    default: {
                        return nullptr;
                    }
                }
            } else if (index_hnsw->build_type_ == HnswBuildType::kPlain) {
                switch (index_hnsw->metric_type_) {
                    case MetricType::kMetricL2: {
                        using HnswIndex = KnnHnsw<RabitqL2VecStoreType<DataType>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricInnerProduct: {
                        using HnswIndex = KnnHnsw<RabitqIPVecStoreType<DataType>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    case MetricType::kMetricCosine: {
                        using HnswIndex = KnnHnsw<RabitqCosVecStoreType<DataType>, SegmentOffset, OwnMem>;
                        return std::unique_ptr<HnswIndex>();
                    }
                    default: {
                        return nullptr;
                    }
                }
            } else {
                return nullptr;
            }
        }
        default: {
            return nullptr;
        }
    }
}

template <bool OwnMem>
AbstractHnsw InitAbstractIndexT(const IndexBase *index_base, std::shared_ptr<ColumnDef> column_def) {
    const auto *index_hnsw = static_cast<const IndexHnsw *>(index_base);
    const auto *embedding_info = static_cast<const EmbeddingInfo *>(column_def->type()->type_info().get());

    switch (embedding_info->Type()) {
        case EmbeddingDataType::kElemFloat: {
            return InitAbstractIndexT<float, OwnMem>(index_hnsw);
        }
        case EmbeddingDataType::kElemUInt8: {
            return InitAbstractIndexT<u8, OwnMem>(index_hnsw);
        }
        case EmbeddingDataType::kElemInt8: {
            return InitAbstractIndexT<i8, OwnMem>(index_hnsw);
        }
        default: {
            return nullptr;
        }
    }
}

AbstractHnsw HnswHandler::InitAbstractIndex(const IndexBase *index_base, std::shared_ptr<ColumnDef> column_def, bool own_mem) {
    if (own_mem) {
        return InitAbstractIndexT<true>(index_base, column_def);
    } else {
        return InitAbstractIndexT<false>(index_base, column_def);
    }
}

HnswHandler::HnswHandler(const IndexBase *index_base, std::shared_ptr<ColumnDef> column_def, bool own_mem)
    : hnsw_(InitAbstractIndex(index_base, column_def, own_mem)) {
    if (!own_mem)
        return;
    const auto *index_hnsw = static_cast<const IndexHnsw *>(index_base);
    const auto *embedding_info = static_cast<const EmbeddingInfo *>(column_def->type()->type_info().get());

    size_t chunk_size = index_hnsw->block_size_;
    size_t max_chunk_num = (DEFAULT_SEGMENT_CAPACITY - 1) / chunk_size + 1;

    size_t dim = embedding_info->Dimension();
    size_t M = index_hnsw->M_;
    size_t ef_construction = index_hnsw->ef_construction_;
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (!std::is_same_v<T, std::nullptr_t>) {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    index = IndexT::Make(chunk_size, max_chunk_num, dim, M, ef_construction);
                    if constexpr (IndexT::LSG) {
                        index->InitLSGBuilder(index_hnsw, column_def);
                    }
                } else {
                    UnrecoverableError("HnswHandler::HnswHandler: index does not own memory");
                }
            }
        },
        hnsw_);
}

std::unique_ptr<HnswHandler> HnswHandler::Make(const IndexBase *index_base, std::shared_ptr<ColumnDef> column_def, bool own_mem) {
    return std::make_unique<HnswHandler>(index_base, column_def, own_mem);
}

size_t HnswHandler::MemUsage() const {
    std::shared_lock handler_lock(hnsw_mutex_);
    return std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                return size_t(0);
            } else {
                return index->mem_usage();
            }
        },
        hnsw_);
}

size_t HnswHandler::GetRowCount() const {
    std::shared_lock handler_lock(hnsw_mutex_);
    return std::visit(
        [](auto &&index) {
            using IndexType = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<IndexType, std::nullptr_t>) {
                return size_t(0);
            } else {
                return index->GetVecNum();
            }
        },
        hnsw_);
}

size_t HnswHandler::GetSizeInBytes() const {
    std::shared_lock handler_lock(hnsw_mutex_);
    return std::visit(
        [](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                return size_t(0);
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    return index->GetSizeInBytes();
                } else {
                    return size_t(0);
                }
            }
        },
        hnsw_);
}

std::pair<size_t, size_t> HnswHandler::GetInfo() const {
    std::shared_lock handler_lock(hnsw_mutex_);
    return std::visit(
        [](auto &&index) -> std::pair<size_t, size_t> {
            using IndexType = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<IndexType, std::nullptr_t>) {
                return {};
            } else {
                return {index->mem_usage(), index->GetVecNum()};
            }
        },
        hnsw_);
}

void HnswHandler::MarkBuildFailed() {
    std::shared_lock handler_lock(hnsw_mutex_);
    std::visit(
        [](auto &&index) {
            using IndexType = std::decay_t<decltype(index)>;
            if constexpr (!std::is_same_v<IndexType, std::nullptr_t>) {
                index->MarkBuildFailed();
            }
        },
        hnsw_);
}

bool HnswHandler::IsBuildFailed() const {
    std::shared_lock handler_lock(hnsw_mutex_);
    return std::visit(
        [](auto &&index) {
            using IndexType = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<IndexType, std::nullptr_t>) {
                return true;
            } else {
                return index->IsBuildFailed();
            }
        },
        hnsw_);
}

void HnswHandler::Check() const {
    std::shared_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                index->Check();
            }
        },
        hnsw_);
}

void HnswHandler::CompressToLVQ() {
    std::unique_lock handler_lock(hnsw_mutex_);
    AbstractHnsw replacement = std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
                return AbstractHnsw{nullptr};
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    using HnswIndexDataType = IndexT::DataType;
                    if constexpr (std::is_same_v<HnswIndexDataType, i8> || std::is_same_v<HnswIndexDataType, u8>) {
                        UnrecoverableError("Invalid index type.");
                        return AbstractHnsw{nullptr};
                    } else {
                        using Replacement = decltype(std::move(*index).CompressToLVQ());
                        static_assert(std::is_nothrow_constructible_v<AbstractHnsw, Replacement>);
                        return AbstractHnsw{std::move(*index).CompressToLVQ()};
                    }
                } else {
                    UnrecoverableError("Invalid index type.");
                    return AbstractHnsw{nullptr};
                }
            }
        },
        hnsw_);
    static_assert(std::is_nothrow_swappable_v<AbstractHnsw>);
    hnsw_.swap(replacement);
}

void HnswHandler::CompressToRabitq() {
    std::unique_lock handler_lock(hnsw_mutex_);
    AbstractHnsw replacement = std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
                return AbstractHnsw{nullptr};
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    using HnswIndexDataType = IndexT::DataType;
                    if constexpr (std::is_same_v<HnswIndexDataType, i8> || std::is_same_v<HnswIndexDataType, u8>) {
                        UnrecoverableError("Invalid index type.");
                        return AbstractHnsw{nullptr};
                    } else {
                        using Replacement = decltype(std::move(*index).CompressToRabitq());
                        static_assert(std::is_nothrow_constructible_v<AbstractHnsw, Replacement>);
                        return AbstractHnsw{std::move(*index).CompressToRabitq()};
                    }
                } else {
                    UnrecoverableError("Invalid index type.");
                    return AbstractHnsw{nullptr};
                }
            }
        },
        hnsw_);
    static_assert(std::is_nothrow_swappable_v<AbstractHnsw>);
    hnsw_.swap(replacement);
}

} // namespace infinity
