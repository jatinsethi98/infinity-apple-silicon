export module infinity_core:multivector_result_handler;

import std.compat;

export namespace infinity {

template <typename DistanceType, typename LabelType, typename InnerTopnIndexType>
class MultiVectorResultHandler {
public:
    MultiVectorResultHandler(int query_count, size_t top_k, DistanceType *distances, LabelType *labels)
        : top_k_(top_k), distances_(distances), labels_(labels) {
        if (query_count != 1) {
            throw std::invalid_argument("MultiVectorResultHandler supports one query");
        }
    }

    void Begin() {}
    void EndWithoutSort() {}

    size_t GetSize(int) const { return size_; }

    DistanceType GetDistance0(int) const {
        DistanceType maximum = std::numeric_limits<DistanceType>::lowest();
        for (size_t index = 0; index < size_; ++index) {
            maximum = std::max(maximum, distances_[index]);
        }
        return maximum;
    }

    void AddResult(DistanceType distance, LabelType label) {
        for (size_t index = 0; index < size_; ++index) {
            if (labels_[index] == label) {
                distances_[index] = std::min(distances_[index], distance);
                return;
            }
        }
        if (size_ < top_k_) {
            distances_[size_] = distance;
            labels_[size_] = label;
            ++size_;
            return;
        }

        size_t worst = 0;
        for (size_t index = 1; index < size_; ++index) {
            if (distances_[index] > distances_[worst]) {
                worst = index;
            }
        }
        if (distance < distances_[worst]) {
            distances_[worst] = distance;
            labels_[worst] = label;
        }
    }

private:
    size_t top_k_;
    DistanceType *distances_;
    LabelType *labels_;
    size_t size_{};
};

} // namespace infinity
