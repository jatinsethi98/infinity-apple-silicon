#include "hnsw_kernel_replay_adapter.h"

namespace hnsw_kernel_replay {

extern "C" __attribute__((noinline, used)) std::int32_t
HnswKernelReplayI8ForcedScalar(const std::int8_t *left, const std::int8_t *right, std::size_t dimension) {
    std::int32_t sum = 0;
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#pragma clang loop unroll(disable)
    for (std::size_t index = 0; index < dimension; ++index) {
        sum += static_cast<std::int32_t>(left[index]) * static_cast<std::int32_t>(right[index]);
    }
    return sum;
}

} // namespace hnsw_kernel_replay
