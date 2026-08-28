export module infinity_core:index_base;

import :infinity_type;

export namespace infinity {

enum class MetricType : i8 {
    kMetricCosine,
    kMetricInnerProduct,
    kMetricL2,
    kInvalid,
};

class IndexBase {
public:
    virtual ~IndexBase() = default;
};

} // namespace infinity
