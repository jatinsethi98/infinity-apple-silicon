module infinity_core:hnsw_handler_diagnostic_stubs.impl;

import :hnsw_handler;
import :chunk_index_meta;
import :memindex_tracer;
import std.compat;

namespace infinity {

HnswIndexInMem::~HnswIndexInMem() {
    if (own_memory_) {
        delete hnsw_handler_;
    }
}

MemIndexTracerInfo HnswIndexInMem::GetInfo() const {
    const auto [mem_used, row_count] = hnsw_handler_ == nullptr ? std::pair<size_t, size_t>{} : hnsw_handler_->GetInfo();
    return MemIndexTracerInfo(std::make_shared<std::string>(),
                              std::make_shared<std::string>(),
                              std::make_shared<std::string>(),
                              mem_used,
                              row_count);
}

const ChunkIndexMetaInfo HnswIndexInMem::GetChunkIndexMetaInfo() const {
    return ChunkIndexMetaInfo{"", begin_row_id_, row_count_, 0, hnsw_handler_ == nullptr ? 0 : hnsw_handler_->GetSizeInBytes()};
}

} // namespace infinity
