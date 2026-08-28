export module infinity_core:chunk_index_meta;

import internal_types;
import std.compat;

export namespace infinity {

struct ChunkIndexMetaInfo {
    ChunkIndexMetaInfo() = default;
    ChunkIndexMetaInfo(std::string base_name, RowID base_row_id, size_t row_count, size_t term_count, size_t index_size)
        : base_name_(std::move(base_name)), base_row_id_(base_row_id), row_count_(row_count), term_count_(term_count),
          index_size_(index_size) {}

    std::string base_name_;
    RowID base_row_id_;
    size_t row_count_{};
    size_t term_count_{};
    size_t index_size_{};
};

} // namespace infinity
