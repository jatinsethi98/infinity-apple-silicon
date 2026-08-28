export module data_type;

import embedding_info;
import std.compat;

export namespace infinity {

class DataType {
public:
    explicit DataType(std::shared_ptr<TypeInfo> type_info) : type_info_(std::move(type_info)) {}

    const std::shared_ptr<TypeInfo> &type_info() const { return type_info_; }

private:
    std::shared_ptr<TypeInfo> type_info_;
};

} // namespace infinity
