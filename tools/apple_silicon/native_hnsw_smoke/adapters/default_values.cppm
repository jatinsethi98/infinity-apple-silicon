export module infinity_core:default_values;

import :infinity_type;

export namespace infinity {

inline constexpr size_t KB = 1024;
inline constexpr size_t DEFAULT_PREFETCH_SIZE = 4;
inline constexpr size_t DEFAULT_ITER_BATCH_SIZE = 1024;
inline constexpr size_t L1_CACHE_SIZE = 32 * KB;
inline constexpr size_t DEFAULT_SEGMENT_CAPACITY = 1024 * 8192;

} // namespace infinity
