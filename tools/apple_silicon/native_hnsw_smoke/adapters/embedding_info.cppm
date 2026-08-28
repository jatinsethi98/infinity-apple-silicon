export module embedding_info;

import embedding_type;
import std.compat;

export namespace infinity {

class TypeInfo {
public:
    virtual ~TypeInfo() = default;
};

class EmbeddingInfo final : public TypeInfo {
public:
    EmbeddingInfo(EmbeddingDataType type, size_t dimension) : type_(type), dimension_(dimension) {}

    EmbeddingDataType Type() const { return type_; }
    size_t Dimension() const { return dimension_; }

private:
    EmbeddingDataType type_;
    size_t dimension_;
};

} // namespace infinity
