export module infinity_core:index_hnsw;

import :index_base;

export namespace infinity {

enum class HnswEncodeType {
    kPlain,
    kLVQ,
    kRabitq,
    kInvalid,
};

enum class HnswBuildType {
    kPlain,
    kLSG,
    kInvalid,
};

class IndexHnsw final : public IndexBase {
public:
    IndexHnsw(MetricType metric_type,
              HnswEncodeType encode_type,
              HnswBuildType build_type,
              size_t M,
              size_t ef_construction,
              size_t block_size)
        : metric_type_(metric_type), encode_type_(encode_type), build_type_(build_type), M_(M), ef_construction_(ef_construction),
          block_size_(block_size) {}

    const MetricType metric_type_;
    HnswEncodeType encode_type_;
    HnswBuildType build_type_;
    const size_t M_;
    const size_t ef_construction_;
    const size_t block_size_;
};

} // namespace infinity
