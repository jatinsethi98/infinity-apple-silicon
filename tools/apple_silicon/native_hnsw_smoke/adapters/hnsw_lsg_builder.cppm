export module infinity_core:hnsw_lsg_builder;

import :infinity_type;

export namespace infinity {

template <typename DataType, typename DistanceType>
class HnswLSGBuilder {
public:
    HnswLSGBuilder() = default;

    template <typename... Args>
    explicit HnswLSGBuilder(Args &&...) {}

    template <typename Iter>
    size_t InsertSampleVec(Iter &&, size_t) {
        return 0;
    }

    template <typename Iter>
    void InsertLSAvg(Iter &&, size_t) {}

    DistanceType alpha() const { return DistanceType{1}; }
    const double *avg() const { return nullptr; }
};

} // namespace infinity
