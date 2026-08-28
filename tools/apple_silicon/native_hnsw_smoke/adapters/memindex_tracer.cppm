export module infinity_core:memindex_tracer;

import std.compat;

export namespace infinity {

struct MemIndexTracerInfo {
    MemIndexTracerInfo(std::shared_ptr<std::string> index_name,
                       std::shared_ptr<std::string> table_name,
                       std::shared_ptr<std::string> db_name,
                       size_t mem_used,
                       size_t row_count)
        : index_name_(std::move(index_name)), table_name_(std::move(table_name)), db_name_(std::move(db_name)), mem_used_(mem_used),
          row_count_(row_count) {}

    std::shared_ptr<std::string> index_name_;
    std::shared_ptr<std::string> table_name_;
    std::shared_ptr<std::string> db_name_;
    size_t mem_used_;
    size_t row_count_;
};

} // namespace infinity
