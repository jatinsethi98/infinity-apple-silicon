#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__clang__) || defined(__GNUC__)
#define HNSW_KERNEL_REPLAY_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define HNSW_KERNEL_REPLAY_NOINLINE __declspec(noinline)
#else
#define HNSW_KERNEL_REPLAY_NOINLINE
#endif

namespace hnsw_kernel_replay {

using F32Kernel = float (*)(const float *, const float *, std::size_t);
using F32Batch4Kernel = void (*)(const float *,
                                 const float *,
                                 const float *,
                                 const float *,
                                 const float *,
                                 std::size_t,
                                 float[4]);
using I8Kernel = std::int32_t (*)(const std::int8_t *, const std::int8_t *, std::size_t);
using OpaqueFunctionAddress = void (*)();

enum class Metric {
    kL2,
    kInnerProduct,
};

enum class Boundary {
    kDirect,
    kSelected,
    kPublicDispatch,
};

struct KernelDescriptor {
    const char *id;
    const char *expected_symbol;
    F32Kernel function;
    F32Kernel runtime_selected_function;
};

enum class RuntimeSelectionExpectation {
    kMustMatch,
    kMustDiffer,
};

struct Batch4KernelDescriptor {
    const char *id;
    const char *expected_production_symbol;
    F32Batch4Kernel wrapper_address;
    OpaqueFunctionAddress production_function_address;
    OpaqueFunctionAddress runtime_selected_production_address;
    RuntimeSelectionExpectation runtime_selection_expectation;

    [[nodiscard]] bool MatchesRuntimeSelection() const noexcept {
        return production_function_address == runtime_selected_production_address;
    }

    [[nodiscard]] bool SatisfiesRuntimeSelectionExpectation() const noexcept {
        const bool matches = MatchesRuntimeSelection();
        return runtime_selection_expectation == RuntimeSelectionExpectation::kMustMatch ? matches : !matches;
    }
};

enum class I8KernelVariant {
    kSelected,
    kDirectBF,
    kDirectSIMDeSSE,
    kForcedScalar,
};

struct I8KernelDescriptor {
    const char *id;
    const char *expected_symbol;
    I8Kernel function;
    I8Kernel runtime_selected_function;
    RuntimeSelectionExpectation runtime_selection_expectation;

    [[nodiscard]] bool MatchesRuntimeSelection() const noexcept { return function == runtime_selected_function; }

    [[nodiscard]] bool SatisfiesRuntimeSelectionExpectation() const noexcept {
        const bool matches = MatchesRuntimeSelection();
        return runtime_selection_expectation == RuntimeSelectionExpectation::kMustMatch ? matches : !matches;
    }
};

extern "C" std::int32_t HnswKernelReplayI8ForcedScalar(const std::int8_t *left, const std::int8_t *right, std::size_t dimension);

KernelDescriptor GetInfinityKernel(Metric metric, Boundary boundary, std::size_t dimension);
Batch4KernelDescriptor GetInfinityBatch4Kernel(Metric metric, Boundary boundary, std::size_t dimension);
I8KernelDescriptor GetInfinityI8Kernel(I8KernelVariant variant, std::size_t dimension);
KernelDescriptor GetFaissKernel(Metric metric, Boundary boundary, std::size_t dimension);
Batch4KernelDescriptor GetFaissBatch4Kernel(Metric metric, Boundary boundary, std::size_t dimension);

} // namespace hnsw_kernel_replay
