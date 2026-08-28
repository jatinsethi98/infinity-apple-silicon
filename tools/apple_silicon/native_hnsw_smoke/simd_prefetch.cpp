#include <common/simd/simd_functions.h>

namespace infinity {

void SIMDPrefetch(const void *ptr) {
    __builtin_prefetch(ptr, 0, 3);
}

} // namespace infinity
