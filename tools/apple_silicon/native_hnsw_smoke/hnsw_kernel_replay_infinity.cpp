#include "hnsw_kernel_replay_adapter.h"

import std;
import std.compat;
import infinity_core;

namespace hnsw_kernel_replay {

extern "C" HNSW_KERNEL_REPLAY_NOINLINE void HnswKernelReplayInfinityF32L2Batch4Aligned(const float *query,
                                                                                       const float *candidate0,
                                                                                       const float *candidate1,
                                                                                       const float *candidate2,
                                                                                       const float *candidate3,
                                                                                       std::size_t dimension,
                                                                                       float output[4]) {
    infinity::F32L2SSEBatch4(query, candidate0, candidate1, candidate2, candidate3, dimension, output);
}

extern "C" HNSW_KERNEL_REPLAY_NOINLINE void HnswKernelReplayInfinityF32L2Batch4Residual(const float *query,
                                                                                        const float *candidate0,
                                                                                        const float *candidate1,
                                                                                        const float *candidate2,
                                                                                        const float *candidate3,
                                                                                        std::size_t dimension,
                                                                                        float output[4]) {
    infinity::F32L2SSEResidualBatch4(query, candidate0, candidate1, candidate2, candidate3, dimension, output);
}

namespace {

static_assert(std::is_same_v<decltype(&infinity::I8IPBF), I8Kernel>);
static_assert(std::is_same_v<decltype(&infinity::I8IPSSE), I8Kernel>);
static_assert(std::is_same_v<decltype(&infinity::I8IPSSEResidual), I8Kernel>);
static_assert(std::is_same_v<infinity::F32DistanceBatch4FuncType, F32Batch4Kernel>);
static_assert(std::is_same_v<decltype(&infinity::F32L2SSEBatch4), F32Batch4Kernel>);
static_assert(std::is_same_v<decltype(&infinity::F32L2SSEResidualBatch4), F32Batch4Kernel>);
static_assert(std::is_same_v<decltype(&HnswKernelReplayInfinityF32L2Batch4Aligned), F32Batch4Kernel>);
static_assert(std::is_same_v<decltype(&HnswKernelReplayInfinityF32L2Batch4Residual), F32Batch4Kernel>);

template <typename FunctionPointer>
OpaqueFunctionAddress EraseFunctionAddress(FunctionPointer function) noexcept {
    static_assert(std::is_pointer_v<FunctionPointer>);
    static_assert(std::is_function_v<std::remove_pointer_t<FunctionPointer>>);
    return reinterpret_cast<OpaqueFunctionAddress>(function);
}

F32Batch4Kernel InfinityBatch4Wrapper(F32Batch4Kernel production) {
    if (production == &infinity::F32L2SSEBatch4) {
        return &HnswKernelReplayInfinityF32L2Batch4Aligned;
    }
    if (production == &infinity::F32L2SSEResidualBatch4) {
        return &HnswKernelReplayInfinityF32L2Batch4Residual;
    }
    throw std::logic_error("Infinity selected an unsupported F32 L2 Batch4 production function");
}

F32Batch4Kernel DirectBatch4Kernel(std::size_t dimension) {
    return dimension % 16 == 0 ? &infinity::F32L2SSEBatch4 : &infinity::F32L2SSEResidualBatch4;
}

F32Batch4Kernel SelectedBatch4Kernel(std::size_t dimension) {
    const infinity::SIMD_FUNCTIONS &functions = infinity::GetSIMD_FUNCTIONS();
    return dimension % 16 == 0 ? functions.HNSW_F32L2_BATCH4_16_ptr_ : functions.HNSW_F32L2_BATCH4_ptr_;
}

const char *Batch4KernelId(Boundary boundary, std::size_t dimension) {
    const bool residual = dimension % 16 != 0;
    if (boundary == Boundary::kDirect) {
        return residual ? "infinity-direct-f32-l2-batch4-residual" : "infinity-direct-f32-l2-batch4";
    }
    return residual ? "infinity-selected-f32-l2-batch4-residual" : "infinity-selected-f32-l2-batch4";
}

const char *Batch4ProductionSymbol(F32Batch4Kernel production) {
    if (production == &infinity::F32L2SSEBatch4) {
        return "infinity::F32L2SSEBatch4";
    }
    if (production == &infinity::F32L2SSEResidualBatch4) {
        return "infinity::F32L2SSEResidualBatch4";
    }
    return "infinity::<unexpected-selected-f32-l2-batch4-kernel>";
}

F32Kernel DirectKernel(Metric metric, std::size_t dimension) {
    const bool aligned = dimension % 16 == 0;
    if (metric == Metric::kL2) {
        return aligned ? &infinity::F32L2SSE : &infinity::F32L2SSEResidual;
    }
    return aligned ? &infinity::F32IPSSE : &infinity::F32IPSSEResidual;
}

F32Kernel SelectedKernel(Metric metric, std::size_t dimension) {
    const infinity::SIMD_FUNCTIONS &functions = infinity::GetSIMD_FUNCTIONS();
    const bool aligned = dimension % 16 == 0;
    if (metric == Metric::kL2) {
        return aligned ? functions.HNSW_F32L2_16_ptr_ : functions.HNSW_F32L2_ptr_;
    }
    return aligned ? functions.HNSW_F32IP_16_ptr_ : functions.HNSW_F32IP_ptr_;
}

const char *KernelId(Metric metric, Boundary boundary, std::size_t dimension) {
    const bool residual = dimension % 16 != 0;
    if (metric == Metric::kL2) {
        if (boundary == Boundary::kDirect) {
            return residual ? "infinity-direct-f32-l2-residual" : "infinity-direct-f32-l2";
        }
        return residual ? "infinity-selected-f32-l2-residual" : "infinity-selected-f32-l2";
    }
    if (boundary == Boundary::kDirect) {
        return residual ? "infinity-direct-f32-ip-residual" : "infinity-direct-f32-ip";
    }
    return residual ? "infinity-selected-f32-ip-residual" : "infinity-selected-f32-ip";
}

const char *DirectSymbol(Metric metric, std::size_t dimension) {
    const bool residual = dimension % 16 != 0;
    if (metric == Metric::kL2) {
        return residual ? "infinity::F32L2SSEResidual" : "infinity::F32L2SSE";
    }
    return residual ? "infinity::F32IPSSEResidual" : "infinity::F32IPSSE";
}

const char *SelectedSymbol(Metric metric, std::size_t dimension, F32Kernel selected) {
    if (selected == DirectKernel(metric, dimension)) {
        return DirectSymbol(metric, dimension);
    }
    if (metric == Metric::kL2 && selected == &infinity::F32L2BF) {
        return "infinity::F32L2BF";
    }
    if (metric == Metric::kInnerProduct && selected == &infinity::F32IPBF) {
        return "infinity::F32IPBF";
    }
    return "infinity::<unexpected-selected-f32-kernel>";
}

I8Kernel SelectedI8Kernel(std::size_t dimension) {
    const infinity::SIMD_FUNCTIONS &functions = infinity::GetSIMD_FUNCTIONS();
    if (dimension % 64 == 0) {
        return functions.HNSW_I8IP_64_ptr_;
    }
    if (dimension % 32 == 0) {
        return functions.HNSW_I8IP_32_ptr_;
    }
    if (dimension % 16 == 0) {
        return functions.HNSW_I8IP_16_ptr_;
    }
    return functions.HNSW_I8IP_ptr_;
}

I8Kernel DirectSIMDeSSEI8Kernel(std::size_t dimension) { return dimension % 16 == 0 ? &infinity::I8IPSSE : &infinity::I8IPSSEResidual; }

const char *SelectedI8Symbol(I8Kernel selected) {
    if (selected == &infinity::I8IPBF) {
        return "infinity::I8IPBF";
    }
    if (selected == &infinity::I8IPSSE) {
        return "infinity::I8IPSSE";
    }
    if (selected == &infinity::I8IPSSEResidual) {
        return "infinity::I8IPSSEResidual";
    }
    return "infinity::<unexpected-selected-i8-kernel>";
}

const char *I8KernelId(I8KernelVariant variant, std::size_t dimension) {
    switch (variant) {
        case I8KernelVariant::kSelected:
            return "infinity-selected-i8-ip";
        case I8KernelVariant::kDirectBF:
            return "infinity-direct-i8-ip-bf";
        case I8KernelVariant::kDirectSIMDeSSE:
            return dimension % 16 == 0 ? "infinity-control-i8-ip-simde-sse" : "infinity-control-i8-ip-simde-sse-residual";
        case I8KernelVariant::kForcedScalar:
            return "replay-control-i8-ip-forced-scalar";
    }
    std::abort();
}

I8Kernel VariantI8Kernel(I8KernelVariant variant, std::size_t dimension, I8Kernel selected) {
    switch (variant) {
        case I8KernelVariant::kSelected:
            return selected;
        case I8KernelVariant::kDirectBF:
            return &infinity::I8IPBF;
        case I8KernelVariant::kDirectSIMDeSSE:
            return DirectSIMDeSSEI8Kernel(dimension);
        case I8KernelVariant::kForcedScalar:
            return &HnswKernelReplayI8ForcedScalar;
    }
    std::abort();
}

const char *I8ExpectedSymbol(I8KernelVariant variant, std::size_t dimension, I8Kernel selected) {
    switch (variant) {
        case I8KernelVariant::kSelected:
            return SelectedI8Symbol(selected);
        case I8KernelVariant::kDirectBF:
            return "infinity::I8IPBF";
        case I8KernelVariant::kDirectSIMDeSSE:
            return dimension % 16 == 0 ? "infinity::I8IPSSE" : "infinity::I8IPSSEResidual";
        case I8KernelVariant::kForcedScalar:
            return "HnswKernelReplayI8ForcedScalar";
    }
    std::abort();
}

RuntimeSelectionExpectation I8SelectionExpectation(I8KernelVariant variant) {
    switch (variant) {
        case I8KernelVariant::kSelected:
        case I8KernelVariant::kDirectBF:
            return RuntimeSelectionExpectation::kMustMatch;
        case I8KernelVariant::kDirectSIMDeSSE:
        case I8KernelVariant::kForcedScalar:
            return RuntimeSelectionExpectation::kMustDiffer;
    }
    std::abort();
}

} // namespace

KernelDescriptor GetInfinityKernel(Metric metric, Boundary boundary, std::size_t dimension) {
    if (boundary == Boundary::kPublicDispatch) {
        throw std::invalid_argument("Infinity has no public-dispatch replay boundary");
    }
    const F32Kernel selected = SelectedKernel(metric, dimension);
    return KernelDescriptor{
        .id = KernelId(metric, boundary, dimension),
        .expected_symbol = boundary == Boundary::kDirect ? DirectSymbol(metric, dimension) : SelectedSymbol(metric, dimension, selected),
        .function = boundary == Boundary::kDirect ? DirectKernel(metric, dimension) : selected,
        .runtime_selected_function = selected,
    };
}

Batch4KernelDescriptor GetInfinityBatch4Kernel(Metric metric, Boundary boundary, std::size_t dimension) {
    if (metric != Metric::kL2) {
        throw std::invalid_argument("Infinity F32 Batch4 replay supports only L2");
    }
    if (boundary == Boundary::kPublicDispatch) {
        throw std::invalid_argument("Infinity has no public-dispatch Batch4 replay boundary");
    }

    const F32Batch4Kernel selected = SelectedBatch4Kernel(dimension);
    if (selected == nullptr) {
        throw std::logic_error("Infinity selected a null F32 L2 Batch4 production function");
    }
    const F32Batch4Kernel production = boundary == Boundary::kDirect ? DirectBatch4Kernel(dimension) : selected;
    const Batch4KernelDescriptor descriptor{
        .id = Batch4KernelId(boundary, dimension),
        .expected_production_symbol = Batch4ProductionSymbol(production),
        .wrapper_address = InfinityBatch4Wrapper(production),
        .production_function_address = EraseFunctionAddress(production),
        .runtime_selected_production_address = EraseFunctionAddress(selected),
        .runtime_selection_expectation = RuntimeSelectionExpectation::kMustMatch,
    };
    if (!descriptor.SatisfiesRuntimeSelectionExpectation()) {
        throw std::logic_error(std::string("Infinity F32 L2 Batch4 runtime-selection contract failed for ") + descriptor.id +
                               ": selected symbol is " + Batch4ProductionSymbol(selected));
    }
    return descriptor;
}

I8KernelDescriptor GetInfinityI8Kernel(I8KernelVariant variant, std::size_t dimension) {
    if (dimension == 0) {
        throw std::invalid_argument("I8 replay dimension must be positive");
    }

    const I8Kernel selected = SelectedI8Kernel(dimension);
    const I8KernelDescriptor descriptor{
        .id = I8KernelId(variant, dimension),
        .expected_symbol = I8ExpectedSymbol(variant, dimension, selected),
        .function = VariantI8Kernel(variant, dimension, selected),
        .runtime_selected_function = selected,
        .runtime_selection_expectation = I8SelectionExpectation(variant),
    };
    if (!descriptor.SatisfiesRuntimeSelectionExpectation()) {
        throw std::logic_error(std::string("I8 replay runtime-selection contract failed for ") + descriptor.id + ": selected symbol is " +
                               SelectedI8Symbol(selected));
    }
    return descriptor;
}

} // namespace hnsw_kernel_replay
