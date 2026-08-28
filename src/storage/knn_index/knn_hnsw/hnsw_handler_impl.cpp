//
// Created by infiniflow on 25-5-13.
//

module;

module infinity_core:hnsw_handler.impl;

import :hnsw_handler;
import :buffer_manager;
import :buffer_handle;
import :block_column_iter;
import :memindex_tracer;
import :default_values;
import :multivector_util;
import :infinity_exception;
import :column_vector;
import :local_file_handle;
import :chunk_index_meta;
import :plain_vec_store;

import third_party;

import embedding_info;
import embedding_type;
import column_def;
import row_id;
import logical_type;

namespace infinity {

size_t HnswHandler::InsertVecs(SegmentOffset block_offset,
                               const ColumnVector &col,
                               BlockOffset offset,
                               BlockOffset row_count,
                               const HnswInsertConfig &config,
                               size_t kBuildBucketSize) {
    std::shared_lock handler_lock(hnsw_mutex_);
    size_t mem_usage{};
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (!std::is_same_v<T, std::nullptr_t>) {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    using DataType = typename IndexT::DataType;
                    switch (const auto &column_data_type = col.data_type(); column_data_type->type()) {
                        case LogicalType::kEmbedding: {
                            MemIndexInserterIter1<DataType> iter(block_offset, col, offset, row_count);
                            HnswHandler::InsertVecs(index, std::move(iter), config, mem_usage, kBuildBucketSize);
                            break;
                        }
                        case LogicalType::kMultiVector: {
                            MemIndexInserterIter1<MultiVectorRef<DataType>> iter(block_offset, col, offset, row_count);
                            HnswHandler::InsertVecs(index, std::move(iter), config, mem_usage, kBuildBucketSize);
                            break;
                        }
                        default: {
                            UnrecoverableError(fmt::format("Unsupported column type for HNSW index: {}", column_data_type->ToString()));
                            break;
                        }
                    }
                } else {
                    UnrecoverableError("HnswHandler::InsertVecs: index does not own memory");
                }
            }
        },
        hnsw_);
    return mem_usage;
}

size_t
HnswHandler::InsertSampleVecs(size_t sample_num, SegmentOffset block_offset, BlockOffset offset, const ColumnVector &col, BlockOffset row_count) {
    std::unique_lock handler_lock(hnsw_mutex_);
    size_t insert_num = 0;
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (!std::is_same_v<T, std::nullptr_t>) {
                using IndexT = std::decay_t<decltype(*index)>;
                using DataType = typename IndexT::DataType;
                switch (const auto &column_data_type = col.data_type(); column_data_type->type()) {
                    case LogicalType::kEmbedding: {
                        MemIndexInserterIter1<DataType> iter(block_offset, col, offset, row_count);
                        insert_num = index->InsertSampleVecs(iter, sample_num);
                        break;
                    }
                    case LogicalType::kMultiVector: {
                        MemIndexInserterIter1<MultiVectorRef<DataType>> iter(block_offset, col, offset, row_count);
                        insert_num = index->InsertSampleVecs(iter, sample_num);
                        break;
                    }
                    default: {
                        UnrecoverableError(fmt::format("Unsupported column type for HNSW index: {}", column_data_type->ToString()));
                        break;
                    }
                }
            }
        },
        hnsw_);
    return insert_num;
}

void HnswHandler::InsertLSAvg(SegmentOffset block_offset, BlockOffset offset, const ColumnVector &col, BlockOffset row_count) {
    std::unique_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (!std::is_same_v<T, std::nullptr_t>) {
                using IndexT = std::decay_t<decltype(*index)>;
                using DataType = typename IndexT::DataType;
                switch (const auto &column_data_type = col.data_type(); column_data_type->type()) {
                    case LogicalType::kEmbedding: {
                        MemIndexInserterIter1<DataType> iter(block_offset, col, offset, row_count);
                        index->InsertLSAvg(iter, row_count);
                        break;
                    }
                    case LogicalType::kMultiVector: {
                        MemIndexInserterIter1<MultiVectorRef<DataType>> iter(block_offset, col, offset, row_count);
                        index->InsertLSAvg(iter, row_count);
                        break;
                    }
                    default: {
                        UnrecoverableError(fmt::format("Unsupported column type for HNSW index: {}", column_data_type->ToString()));
                        break;
                    }
                }
            }
        },
        hnsw_);
}

void HnswHandler::SetLSGParam() {
    std::unique_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::LSG) {
                    index->SetLSGParam();
                } else {
                    UnrecoverableError("Invalid index type.");
                }
            }
        },
        hnsw_);
}

void HnswHandler::Save(LocalFileHandle &file_handle) const {
    std::shared_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    index->Save(file_handle);
                } else {
                    UnrecoverableError("Invalid index type.");
                }
            }
        },
        hnsw_);
}

void HnswHandler::SaveToPtr(LocalFileHandle &file_handle) const {
    std::shared_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    index->SaveToPtr(file_handle);
                } else {
                    UnrecoverableError("Invalid index type.");
                }
            }
        },
        hnsw_);
}

void HnswHandler::Load(LocalFileHandle &file_handle) {
    std::unique_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    index = IndexT::Load(file_handle);
                } else {
                    UnrecoverableError("Invalid index type.");
                }
            }
        },
        hnsw_);
}

void HnswHandler::LoadFromPtr(LocalFileHandle &file_handle, size_t file_size) {
    std::unique_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    index = IndexT::LoadFromPtr(file_handle, file_size);
                } else {
                    UnrecoverableError("Invalid index type.");
                }
            }
        },
        hnsw_);
}

void HnswHandler::LoadFromPtr(const char *ptr, size_t size) {
    std::unique_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    UnrecoverableError("Invalid index type.");
                } else {
                    index = IndexT::LoadFromPtr(ptr, size);
                }
            }
        },
        hnsw_);
}

void HnswHandler::Build(VertexType vertex_i) {
    std::shared_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    index->Build(vertex_i);
                } else {
                    UnrecoverableError("Invalid index type.");
                }
            }
        },
        hnsw_);
}

void HnswHandler::Optimize() {
    std::shared_lock handler_lock(hnsw_mutex_);
    std::visit(
        [&](auto &&index) {
            using T = std::decay_t<decltype(index)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                UnrecoverableError("Invalid index type.");
            } else {
                using IndexT = std::decay_t<decltype(*index)>;
                if constexpr (IndexT::kOwnMem) {
                    index->Optimize();
                } else {
                    UnrecoverableError("Invalid index type.");
                }
            }
        },
        hnsw_);
}

HnswIndexInMem::~HnswIndexInMem() {
    size_t mem_usage = hnsw_handler_->MemUsage();
    if (own_memory_ && hnsw_handler_ != nullptr) {
        delete hnsw_handler_;
    }

    auto *storage = InfinityContext::instance().storage();
    if (storage == nullptr) {
        return;
    }
    auto *memindex_tracer = storage->memindex_tracer();
    if (memindex_tracer != nullptr) {
        memindex_tracer->DecreaseMemUsed(mem_usage);
    }
}

std::unique_ptr<HnswIndexInMem> HnswIndexInMem::Make(RowID begin_row_id, const IndexBase *index_base, std::shared_ptr<ColumnDef> column_def) {
    auto memidx = std::make_unique<HnswIndexInMem>(begin_row_id, index_base, column_def);

    auto *storage = InfinityContext::instance().storage();
    if (storage != nullptr) {
        auto *memindex_tracer = storage->memindex_tracer();
        if (memindex_tracer != nullptr) {
            memindex_tracer->IncreaseMemoryUsage(memidx->hnsw_handler_->MemUsage());
        }
    }
    return memidx;
}

std::unique_ptr<HnswIndexInMem> HnswIndexInMem::Make(const IndexBase *index_base, std::shared_ptr<ColumnDef> column_def) {
    RowID begin_row_id{0, 0};
    auto memidx = std::make_unique<HnswIndexInMem>(begin_row_id, index_base, column_def);

    auto *storage = InfinityContext::instance().storage();
    if (storage != nullptr) {
        auto *memindex_tracer = storage->memindex_tracer();
        if (memindex_tracer != nullptr) {
            memindex_tracer->IncreaseMemoryUsage(memidx->hnsw_handler_->MemUsage());
        }
    }
    return memidx;
}

MemIndexTracerInfo HnswIndexInMem::GetInfo() const {
    auto [mem_used, row_cnt] = hnsw_handler_->GetInfo();
    return MemIndexTracerInfo(std::make_shared<std::string>(index_name_),
                              std::make_shared<std::string>(table_name_),
                              std::make_shared<std::string>(db_name_),
                              mem_used,
                              row_cnt);
}

void HnswIndexInMem::InsertVecs(SegmentOffset block_offset,
                                const ColumnVector &col,
                                BlockOffset offset,
                                BlockOffset row_count,
                                const HnswInsertConfig &config) {
    const size_t mem_before = hnsw_handler_->MemUsage();
    try {
        size_t mem_usage = hnsw_handler_->InsertVecs(block_offset, col, offset, row_count, config, kBuildBucketSize);
        row_count_ += row_count;
        IncreaseMemoryUsageBase(mem_usage);
    } catch (...) {
        const size_t mem_after = hnsw_handler_->MemUsage();
        IncreaseMemoryUsageBase(mem_after > mem_before ? mem_after - mem_before : 0);
        throw;
    }
}

void HnswIndexInMem::Dump(BufferObj *buffer_obj, size_t *dump_size_ptr) {
    if (IsBuildFailed()) {
        throw std::logic_error("cannot dump an HNSW index after a failed build");
    }
    if (dump_size_ptr != nullptr) {
        size_t dump_size = hnsw_handler_->MemUsage();
        *dump_size_ptr = dump_size;
    }

    BufferHandle handle = buffer_obj->Load();
    auto *data_ptr = static_cast<HnswHandlerPtr *>(handle.GetDataMut());
    *data_ptr = hnsw_handler_;
    own_memory_ = false;
    chunk_handle_ = std::move(handle);
}

size_t
HnswIndexInMem::InsertSampleVecs(size_t sample_num, SegmentOffset block_offset, BlockOffset offset, const ColumnVector &col, BlockOffset row_count) {
    return hnsw_handler_->InsertSampleVecs(sample_num, block_offset, offset, col, row_count);
}

void HnswIndexInMem::InsertLSAvg(SegmentOffset block_offset, BlockOffset offset, const ColumnVector &col, BlockOffset row_count) {
    hnsw_handler_->InsertLSAvg(block_offset, offset, col, row_count);
}

void HnswIndexInMem::SetLSGParam() { hnsw_handler_->SetLSGParam(); }

size_t HnswIndexInMem::GetRowCount() const { return row_count_; }

size_t HnswIndexInMem::GetSizeInBytes() const { return hnsw_handler_->GetSizeInBytes(); }

bool HnswIndexInMem::IsBuildFailed() const { return hnsw_handler_->IsBuildFailed(); }

const ChunkIndexMetaInfo HnswIndexInMem::GetChunkIndexMetaInfo() const {
    return ChunkIndexMetaInfo{"", begin_row_id_, GetRowCount(), 0, GetSizeInBytes()};
}

} // namespace infinity
