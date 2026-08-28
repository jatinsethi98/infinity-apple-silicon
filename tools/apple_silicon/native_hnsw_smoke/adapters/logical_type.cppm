export module logical_type;

export import std;

export namespace infinity {

enum class LogicalType {
    kEmbedding,
    kMultiVector,
};

inline std::string_view LogicalType2Str(LogicalType type) {
    return type == LogicalType::kEmbedding ? "Embedding" : "MultiVector";
}

} // namespace infinity
