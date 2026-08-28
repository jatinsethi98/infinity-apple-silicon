#include "hnsw_kernel_replay_adapter.h"

#include <faiss/utils/distances.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace hnsw_kernel_replay {

using FaissBatch4ProductionKernel = void (*)(const float *,
                                             const float *,
                                             const float *,
                                             const float *,
                                             const float *,
                                             std::size_t,
                                             float &,
                                             float &,
                                             float &,
                                             float &);

extern "C" HNSW_KERNEL_REPLAY_NOINLINE void HnswKernelReplayFaissF32L2Batch4ArmNeon(const float *query,
                                                                                    const float *candidate0,
                                                                                    const float *candidate1,
                                                                                    const float *candidate2,
                                                                                    const float *candidate3,
                                                                                    std::size_t dimension,
                                                                                    float output[4]) {
    faiss::fvec_L2sqr_batch_4<faiss::SIMDLevel::ARM_NEON>(
        query, candidate0, candidate1, candidate2, candidate3, dimension, output[0], output[1], output[2], output[3]);
}

extern "C" HNSW_KERNEL_REPLAY_NOINLINE void HnswKernelReplayFaissF32IPBatch4ArmNeon(const float *query,
                                                                                    const float *candidate0,
                                                                                    const float *candidate1,
                                                                                    const float *candidate2,
                                                                                    const float *candidate3,
                                                                                    std::size_t dimension,
                                                                                    float output[4]) {
    faiss::fvec_inner_product_batch_4<faiss::SIMDLevel::ARM_NEON>(
        query, candidate0, candidate1, candidate2, candidate3, dimension, output[0], output[1], output[2], output[3]);
}

extern "C" HNSW_KERNEL_REPLAY_NOINLINE void HnswKernelReplayFaissF32L2Batch4Public(const float *query,
                                                                                  const float *candidate0,
                                                                                  const float *candidate1,
                                                                                  const float *candidate2,
                                                                                  const float *candidate3,
                                                                                  std::size_t dimension,
                                                                                  float output[4]) {
    faiss::fvec_L2sqr_batch_4(
        query, candidate0, candidate1, candidate2, candidate3, dimension, output[0], output[1], output[2], output[3]);
}

extern "C" HNSW_KERNEL_REPLAY_NOINLINE void HnswKernelReplayFaissF32IPBatch4Public(const float *query,
                                                                                  const float *candidate0,
                                                                                  const float *candidate1,
                                                                                  const float *candidate2,
                                                                                  const float *candidate3,
                                                                                  std::size_t dimension,
                                                                                  float output[4]) {
    faiss::fvec_inner_product_batch_4(
        query, candidate0, candidate1, candidate2, candidate3, dimension, output[0], output[1], output[2], output[3]);
}

namespace {

static_assert(std::is_same_v<decltype(static_cast<FaissBatch4ProductionKernel>(
                                 &faiss::fvec_L2sqr_batch_4<faiss::SIMDLevel::ARM_NEON>)),
                             FaissBatch4ProductionKernel>);
static_assert(std::is_same_v<decltype(static_cast<FaissBatch4ProductionKernel>(
                                 &faiss::fvec_inner_product_batch_4<faiss::SIMDLevel::ARM_NEON>)),
                             FaissBatch4ProductionKernel>);
static_assert(std::is_same_v<decltype(static_cast<FaissBatch4ProductionKernel>(&faiss::fvec_L2sqr_batch_4)),
                             FaissBatch4ProductionKernel>);
static_assert(std::is_same_v<decltype(static_cast<FaissBatch4ProductionKernel>(&faiss::fvec_inner_product_batch_4)),
                             FaissBatch4ProductionKernel>);
static_assert(std::is_same_v<decltype(&HnswKernelReplayFaissF32L2Batch4ArmNeon), F32Batch4Kernel>);
static_assert(std::is_same_v<decltype(&HnswKernelReplayFaissF32IPBatch4ArmNeon), F32Batch4Kernel>);
static_assert(std::is_same_v<decltype(&HnswKernelReplayFaissF32L2Batch4Public), F32Batch4Kernel>);
static_assert(std::is_same_v<decltype(&HnswKernelReplayFaissF32IPBatch4Public), F32Batch4Kernel>);

template <typename FunctionPointer>
OpaqueFunctionAddress EraseFunctionAddress(FunctionPointer function) noexcept {
    static_assert(std::is_pointer_v<FunctionPointer>);
    static_assert(std::is_function_v<std::remove_pointer_t<FunctionPointer>>);
    return reinterpret_cast<OpaqueFunctionAddress>(function);
}

F32Kernel DirectKernel(Metric metric) {
    return metric == Metric::kL2 ? &faiss::fvec_L2sqr<faiss::SIMDLevel::ARM_NEON>
                                 : &faiss::fvec_inner_product<faiss::SIMDLevel::ARM_NEON>;
}

F32Kernel PublicKernel(Metric metric) {
    return metric == Metric::kL2 ? static_cast<F32Kernel>(&faiss::fvec_L2sqr)
                                 : static_cast<F32Kernel>(&faiss::fvec_inner_product);
}

FaissBatch4ProductionKernel DirectBatch4Kernel(Metric metric) {
    return metric == Metric::kL2
        ? static_cast<FaissBatch4ProductionKernel>(&faiss::fvec_L2sqr_batch_4<faiss::SIMDLevel::ARM_NEON>)
        : static_cast<FaissBatch4ProductionKernel>(&faiss::fvec_inner_product_batch_4<faiss::SIMDLevel::ARM_NEON>);
}

FaissBatch4ProductionKernel SelectedBatch4Kernel(Metric metric) {
    const faiss::SIMDLevel selected_level = faiss::SIMDConfig::get_dispatched_level();
    if (selected_level != faiss::SIMDLevel::ARM_NEON) {
        throw std::logic_error("FAISS F32 Batch4 replay expected ARM_NEON runtime dispatch, got " +
                               faiss::to_string(selected_level));
    }
    return DirectBatch4Kernel(metric);
}

FaissBatch4ProductionKernel PublicBatch4Kernel(Metric metric) {
    return metric == Metric::kL2 ? static_cast<FaissBatch4ProductionKernel>(&faiss::fvec_L2sqr_batch_4)
                                 : static_cast<FaissBatch4ProductionKernel>(&faiss::fvec_inner_product_batch_4);
}

F32Batch4Kernel Batch4Wrapper(Metric metric, Boundary boundary) {
    if (metric == Metric::kL2) {
        return boundary == Boundary::kPublicDispatch ? &HnswKernelReplayFaissF32L2Batch4Public
                                                     : &HnswKernelReplayFaissF32L2Batch4ArmNeon;
    }
    return boundary == Boundary::kPublicDispatch ? &HnswKernelReplayFaissF32IPBatch4Public
                                                 : &HnswKernelReplayFaissF32IPBatch4ArmNeon;
}

const char *KernelId(Metric metric, Boundary boundary) {
    if (metric == Metric::kL2) {
        return boundary == Boundary::kPublicDispatch ? "faiss-public-f32-l2" : "faiss-direct-f32-l2-arm-neon";
    }
    return boundary == Boundary::kPublicDispatch ? "faiss-public-f32-ip" : "faiss-direct-f32-ip-arm-neon";
}

const char *Batch4KernelId(Metric metric, Boundary boundary) {
    if (metric == Metric::kL2) {
        return boundary == Boundary::kPublicDispatch ? "faiss-public-f32-l2-batch4" : "faiss-direct-f32-l2-batch4-arm-neon";
    }
    return boundary == Boundary::kPublicDispatch ? "faiss-public-f32-ip-batch4" : "faiss-direct-f32-ip-batch4-arm-neon";
}

const char *ExpectedSymbol(Metric metric, Boundary boundary) {
    if (boundary == Boundary::kPublicDispatch) {
        return metric == Metric::kL2 ? "faiss::fvec_L2sqr" : "faiss::fvec_inner_product";
    }
    return metric == Metric::kL2 ? "faiss::fvec_L2sqr<ARM_NEON>" : "faiss::fvec_inner_product<ARM_NEON>";
}

const char *ExpectedBatch4ProductionSymbol(Metric metric, Boundary boundary) {
    if (boundary == Boundary::kPublicDispatch) {
        return metric == Metric::kL2 ? "faiss::fvec_L2sqr_batch_4" : "faiss::fvec_inner_product_batch_4";
    }
    return metric == Metric::kL2 ? "faiss::fvec_L2sqr_batch_4<ARM_NEON>"
                                 : "faiss::fvec_inner_product_batch_4<ARM_NEON>";
}

} // namespace

KernelDescriptor GetFaissKernel(Metric metric, Boundary boundary, std::size_t) {
    if (boundary == Boundary::kSelected) {
        boundary = Boundary::kDirect;
    }
    const F32Kernel direct = DirectKernel(metric);
    return KernelDescriptor{
        .id = KernelId(metric, boundary),
        .expected_symbol = ExpectedSymbol(metric, boundary),
        .function = boundary == Boundary::kPublicDispatch ? PublicKernel(metric) : direct,
        .runtime_selected_function = direct,
    };
}

Batch4KernelDescriptor GetFaissBatch4Kernel(Metric metric, Boundary boundary, std::size_t) {
    if (boundary == Boundary::kSelected) {
        boundary = Boundary::kDirect;
    }

    const FaissBatch4ProductionKernel selected = SelectedBatch4Kernel(metric);
    const FaissBatch4ProductionKernel production =
        boundary == Boundary::kPublicDispatch ? PublicBatch4Kernel(metric) : selected;
    const Batch4KernelDescriptor descriptor{
        .id = Batch4KernelId(metric, boundary),
        .expected_production_symbol = ExpectedBatch4ProductionSymbol(metric, boundary),
        .wrapper_address = Batch4Wrapper(metric, boundary),
        .production_function_address = EraseFunctionAddress(production),
        .runtime_selected_production_address = EraseFunctionAddress(selected),
        .runtime_selection_expectation = boundary == Boundary::kPublicDispatch ? RuntimeSelectionExpectation::kMustDiffer
                                                                               : RuntimeSelectionExpectation::kMustMatch,
    };
    if (!descriptor.SatisfiesRuntimeSelectionExpectation()) {
        throw std::logic_error(std::string("FAISS F32 Batch4 runtime-selection contract failed for ") + descriptor.id);
    }
    return descriptor;
}

} // namespace hnsw_kernel_replay
