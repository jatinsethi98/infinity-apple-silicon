export module column_def;

import data_type;
import std.compat;

export namespace infinity {

class ColumnDef {
public:
    explicit ColumnDef(std::shared_ptr<DataType> type) : type_(std::move(type)) {}

    const std::shared_ptr<DataType> &type() const { return type_; }

private:
    std::shared_ptr<DataType> type_;
};

} // namespace infinity
