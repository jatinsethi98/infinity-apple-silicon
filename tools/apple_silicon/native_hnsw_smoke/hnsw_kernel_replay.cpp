#include "hnsw_kernel_replay_adapter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <os/signpost.h>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/sysctl.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using hnsw_kernel_replay::Boundary;
using hnsw_kernel_replay::F32Kernel;
using hnsw_kernel_replay::KernelDescriptor;
using hnsw_kernel_replay::Metric;

constexpr std::uint64_t kSeed = 0x8ddfea08eb382d69ULL;
constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kDefaultEvictionBytes = 64ULL << 20;
constexpr std::size_t kValidationLimit = 4096;
constexpr std::size_t kMinimumPairedRounds = 32;
constexpr std::size_t kBootstrapReplicates = 50'000;
constexpr double kDefaultMaximumRelativeError = 1e-5;
constexpr double kDefaultMinimumArmSeconds = 0.20;

volatile double g_result_sink = 0.0;
volatile std::uint64_t g_eviction_sink = 0;

enum class Mode {
    kPaired,
    kAttach,
    kValidate,
};

enum class Regime {
    kHot,
    kL1,
    kL2,
    kTlb,
    kDram,
};

enum class Engine {
    kInfinity,
    kFaiss,
};

struct KernelChoice {
    Engine engine;
    Boundary boundary;
};

struct Options {
    Mode mode = Mode::kPaired;
    Metric metric = Metric::kL2;
    Regime regime = Regime::kHot;
    KernelChoice left{Engine::kInfinity, Boundary::kSelected};
    KernelChoice right{Engine::kFaiss, Boundary::kDirect};
    KernelChoice attach_kernel{Engine::kInfinity, Boundary::kSelected};
    std::size_t dimension = 128;
    std::size_t calls = 0;
    std::size_t rounds = kMinimumPairedRounds;
    std::size_t warmup_calls = 0;
    std::size_t working_set_bytes = 0;
    std::size_t eviction_bytes = kDefaultEvictionBytes;
    std::size_t duration_seconds = 20;
    double maximum_relative_error = kDefaultMaximumRelativeError;
    double minimum_arm_seconds = kDefaultMinimumArmSeconds;
    bool wait_for_signal = false;
    bool require_bit_identical = false;
};

struct HardwareGeometry {
    std::uint64_t memory_bytes = 0;
    std::uint64_t page_bytes = 0;
    std::uint64_t performance_l1d_bytes = 0;
    std::uint64_t performance_l2_bytes = 0;
    std::uint64_t performance_core_count = 0;
};

struct FreeDeleter {
    void operator()(std::byte *pointer) const noexcept { std::free(pointer); }
};

struct ReplayData {
    std::unique_ptr<std::byte, FreeDeleter> storage;
    std::vector<float> query;
    std::vector<std::uint32_t> tape;
    std::size_t stride_bytes = 0;
    std::size_t candidate_count = 0;
    std::size_t allocation_bytes = 0;
    std::uint64_t input_digest = kFnvOffset;

    const float *Candidate(std::size_t index) const noexcept {
        return reinterpret_cast<const float *>(storage.get() + index * stride_bytes);
    }
};

struct Validation {
    std::size_t samples = 0;
    std::size_t finite_failures = 0;
    std::size_t bit_mismatches = 0;
    std::uint32_t maximum_ulp_difference = 0;
    double maximum_left_relative_error = 0;
    double maximum_right_relative_error = 0;
    std::uint64_t left_output_digest = kFnvOffset;
    std::uint64_t right_output_digest = kFnvOffset;
};

struct Timing {
    double seconds = 0;
    double thread_cpu_seconds = 0;
    double checksum = 0;
};

struct RuntimePolicy {
    int set_qos_status = 0;
    int get_qos_status = 0;
    qos_class_t qos_class = QOS_CLASS_UNSPECIFIED;
    int relative_priority = 0;
};

struct RawArm {
    std::size_t block = 0;
    const char *order = nullptr;
    const KernelDescriptor *kernel = nullptr;
    Timing timing;
};

std::string JsonEscape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (byte < 0x20) {
                    constexpr char digits[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped += digits[byte >> 4];
                    escaped += digits[byte & 0xf];
                } else {
                    escaped += static_cast<char>(byte);
                }
        }
    }
    return escaped;
}

std::uint64_t ParseUnsigned(std::string_view value, const char *name) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
    }
    std::string text(value);
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    return parsed;
}

double ParsePositiveDouble(std::string_view value, const char *name) {
    std::string text(value);
    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(parsed) || parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive finite number");
    }
    return parsed;
}

std::string_view RequireValue(int argc, char **argv, int &index) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index - 1]);
    }
    return argv[index];
}

KernelChoice ParseKernel(std::string_view value) {
    if (value == "infinity-direct") {
        return {Engine::kInfinity, Boundary::kDirect};
    }
    if (value == "infinity-selected") {
        return {Engine::kInfinity, Boundary::kSelected};
    }
    if (value == "faiss-direct" || value == "faiss-selected") {
        return {Engine::kFaiss, Boundary::kDirect};
    }
    if (value == "faiss-public") {
        return {Engine::kFaiss, Boundary::kPublicDispatch};
    }
    throw std::invalid_argument("unknown kernel: " + std::string(value));
}

Regime ParseRegime(std::string_view value) {
    if (value == "hot") {
        return Regime::kHot;
    }
    if (value == "l1") {
        return Regime::kL1;
    }
    if (value == "l2") {
        return Regime::kL2;
    }
    if (value == "tlb") {
        return Regime::kTlb;
    }
    if (value == "dram") {
        return Regime::kDram;
    }
    throw std::invalid_argument("unknown regime: " + std::string(value));
}

const char *RegimeName(Regime regime) {
    switch (regime) {
        case Regime::kHot:
            return "hot";
        case Regime::kL1:
            return "l1";
        case Regime::kL2:
            return "l2";
        case Regime::kTlb:
            return "tlb";
        case Regime::kDram:
            return "dram";
    }
    std::abort();
}

const char *MetricName(Metric metric) { return metric == Metric::kL2 ? "l2" : "inner_product"; }

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--mode") {
            const std::string_view value = RequireValue(argc, argv, index);
            if (value == "paired") {
                options.mode = Mode::kPaired;
            } else if (value == "attach") {
                options.mode = Mode::kAttach;
            } else if (value == "validate") {
                options.mode = Mode::kValidate;
            } else {
                throw std::invalid_argument("unknown mode: " + std::string(value));
            }
        } else if (argument == "--metric") {
            const std::string_view value = RequireValue(argc, argv, index);
            if (value == "l2") {
                options.metric = Metric::kL2;
            } else if (value == "ip" || value == "inner-product") {
                options.metric = Metric::kInnerProduct;
            } else {
                throw std::invalid_argument("unknown metric: " + std::string(value));
            }
        } else if (argument == "--regime") {
            options.regime = ParseRegime(RequireValue(argc, argv, index));
        } else if (argument == "--left") {
            options.left = ParseKernel(RequireValue(argc, argv, index));
        } else if (argument == "--right") {
            options.right = ParseKernel(RequireValue(argc, argv, index));
        } else if (argument == "--kernel") {
            options.attach_kernel = ParseKernel(RequireValue(argc, argv, index));
        } else if (argument == "--dimension") {
            options.dimension = ParseUnsigned(RequireValue(argc, argv, index), "dimension");
        } else if (argument == "--calls") {
            options.calls = ParseUnsigned(RequireValue(argc, argv, index), "calls");
        } else if (argument == "--rounds") {
            options.rounds = ParseUnsigned(RequireValue(argc, argv, index), "rounds");
        } else if (argument == "--warmup-calls") {
            options.warmup_calls = ParseUnsigned(RequireValue(argc, argv, index), "warmup calls");
        } else if (argument == "--working-set-bytes") {
            options.working_set_bytes = ParseUnsigned(RequireValue(argc, argv, index), "working-set bytes");
        } else if (argument == "--eviction-bytes") {
            options.eviction_bytes = ParseUnsigned(RequireValue(argc, argv, index), "eviction bytes");
        } else if (argument == "--duration-seconds") {
            options.duration_seconds = ParseUnsigned(RequireValue(argc, argv, index), "duration seconds");
        } else if (argument == "--maximum-relative-error") {
            options.maximum_relative_error =
                ParsePositiveDouble(RequireValue(argc, argv, index), "maximum relative error");
        } else if (argument == "--minimum-arm-seconds") {
            options.minimum_arm_seconds = ParsePositiveDouble(RequireValue(argc, argv, index), "minimum arm seconds");
        } else if (argument == "--wait-for-signal") {
            options.wait_for_signal = true;
        } else if (argument == "--require-bit-identical") {
            options.require_bit_identical = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: hnsw_kernel_replay [options]\n"
                << "  --mode paired|validate|attach\n"
                << "  --metric l2|ip --dimension N --regime hot|l1|l2|tlb|dram\n"
                << "  --left K --right K       paired kernels\n"
                << "  --kernel K               attach-mode kernel\n"
                << "  K: infinity-direct|infinity-selected|faiss-direct|faiss-public\n"
                << "  --calls N --rounds N --warmup-calls N\n"
                << "  --working-set-bytes N --eviction-bytes N\n"
                << "  --duration-seconds N --wait-for-signal\n"
                << "  --maximum-relative-error X --minimum-arm-seconds X\n"
                << "  --require-bit-identical\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.dimension == 0 || options.dimension > (1ULL << 20)) {
        throw std::invalid_argument("dimension must be in [1, 1048576]");
    }
    if (options.duration_seconds == 0) {
        throw std::invalid_argument("duration must be positive");
    }
    if (options.mode == Mode::kPaired &&
        (options.rounds < kMinimumPairedRounds || options.rounds % 2 != 0)) {
        throw std::invalid_argument(
            "paired mode requires an even --rounds value of at least " +
            std::to_string(kMinimumPairedRounds));
    }
    return options;
}

std::uint64_t SplitMix64(std::uint64_t &state) {
    std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

void HashBytes(std::uint64_t &hash, const void *data, std::size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

float DeterministicFloat(std::uint64_t &state) {
    const std::uint32_t mantissa = static_cast<std::uint32_t>(SplitMix64(state) >> 41);
    return static_cast<float>(mantissa) * (2.0f / 8388607.0f) - 1.0f;
}

std::size_t RoundUp(std::size_t value, std::size_t alignment) {
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("size overflow");
    }
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t ReadRequiredSysctlU64(const char *name) {
    std::array<std::byte, sizeof(std::uint64_t)> bytes{};
    std::size_t size = bytes.size();
    if (sysctlbyname(name, bytes.data(), &size, nullptr, 0) != 0) {
        throw std::runtime_error(std::string("failed to read required sysctl ") + name);
    }
    std::uint64_t value = 0;
    if (size == sizeof(std::uint32_t)) {
        std::uint32_t value32 = 0;
        std::memcpy(&value32, bytes.data(), sizeof(value32));
        value = value32;
    } else if (size == sizeof(value)) {
        std::memcpy(&value, bytes.data(), sizeof(value));
    } else {
        throw std::runtime_error(
            std::string("unsupported integer width for required sysctl ") + name);
    }
    if (value == 0) {
        throw std::runtime_error(std::string("required sysctl is zero: ") + name);
    }
    return value;
}

HardwareGeometry ReadHardwareGeometry() {
    return {
        .memory_bytes = ReadRequiredSysctlU64("hw.memsize"),
        .page_bytes = ReadRequiredSysctlU64("hw.pagesize"),
        .performance_l1d_bytes = ReadRequiredSysctlU64("hw.perflevel0.l1dcachesize"),
        .performance_l2_bytes = ReadRequiredSysctlU64("hw.perflevel0.l2cachesize"),
        .performance_core_count = ReadRequiredSysctlU64("hw.perflevel0.physicalcpu"),
    };
}

std::size_t CheckedMultiply(std::uint64_t left, std::uint64_t right, const char *name) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string(name) + " overflow");
    }
    return static_cast<std::size_t>(left * right);
}

std::size_t DefaultWorkingSet(Regime regime, const HardwareGeometry &geometry) {
    switch (regime) {
        case Regime::kHot:
            return 0;
        case Regime::kL1:
            return static_cast<std::size_t>(geometry.performance_l1d_bytes / 2);
        case Regime::kL2:
            return static_cast<std::size_t>(geometry.performance_l2_bytes / 2);
        case Regime::kTlb:
            return CheckedMultiply(geometry.performance_l2_bytes, 8, "TLB geometry");
        case Regime::kDram:
            return CheckedMultiply(geometry.performance_l2_bytes, 16, "DRAM geometry");
    }
    std::abort();
}

void ValidateRequestedGeometry(
    const Options &options,
    const HardwareGeometry &geometry,
    std::size_t vector_bytes,
    std::size_t target_bytes) {
    const std::uint64_t l1 = geometry.performance_l1d_bytes;
    const std::uint64_t l2 = geometry.performance_l2_bytes;
    if (options.regime == Regime::kHot) {
        if (options.working_set_bytes != 0) {
            throw std::invalid_argument("hot geometry does not accept --working-set-bytes");
        }
        return;
    }
    if (target_bytes < vector_bytes) {
        throw std::invalid_argument("working set is smaller than one vector");
    }
    switch (options.regime) {
        case Regime::kHot:
            break;
        case Regime::kL1:
            if (target_bytes > l1) {
                throw std::invalid_argument("L1 geometry exceeds the performance-core L1D size");
            }
            break;
        case Regime::kL2:
            if (target_bytes <= l1 || target_bytes > l2) {
                throw std::invalid_argument("L2 geometry must exceed L1D and fit in performance-core L2");
            }
            break;
        case Regime::kTlb:
            if (target_bytes < CheckedMultiply(l2, 4, "TLB minimum geometry")) {
                throw std::invalid_argument("TLB geometry must be at least four performance-core L2 caches");
            }
            break;
        case Regime::kDram:
            if (target_bytes < CheckedMultiply(l2, 8, "DRAM minimum geometry")) {
                throw std::invalid_argument("DRAM geometry must be at least eight performance-core L2 caches");
            }
            if (target_bytes > geometry.memory_bytes / 4) {
                throw std::invalid_argument("DRAM geometry may not reserve more than one quarter of physical memory");
            }
            break;
    }
    if (options.eviction_bytes < CheckedMultiply(l2, 2, "eviction minimum geometry")) {
        throw std::invalid_argument("eviction buffer must be at least twice the performance-core L2 size");
    }
}

ReplayData MakeReplayData(
    const Options &options,
    const HardwareGeometry &geometry,
    bool enforce_named_geometry = true) {
    ReplayData data;
    const std::size_t vector_bytes = options.dimension * sizeof(float);
    const std::size_t page_bytes = static_cast<std::size_t>(geometry.page_bytes);
    data.stride_bytes =
        options.regime == Regime::kTlb ? std::max(page_bytes, RoundUp(vector_bytes, page_bytes)) : RoundUp(vector_bytes, kCacheLineBytes);

    const std::size_t target_bytes =
        options.working_set_bytes == 0 ? DefaultWorkingSet(options.regime, geometry) : options.working_set_bytes;
    if (enforce_named_geometry) {
        ValidateRequestedGeometry(options, geometry, vector_bytes, target_bytes);
    }
    if (options.regime == Regime::kHot) {
        data.candidate_count = 1;
    } else {
        const std::size_t possible_count = std::max<std::size_t>(1, target_bytes / data.stride_bytes);
        data.candidate_count = std::bit_floor(possible_count);
    }
    if (data.candidate_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("candidate count exceeds replay tape width");
    }
    if (data.candidate_count > std::numeric_limits<std::size_t>::max() / data.stride_bytes) {
        throw std::overflow_error("candidate allocation overflow");
    }
    data.allocation_bytes = data.candidate_count * data.stride_bytes;

    void *allocation = nullptr;
    const std::size_t allocation_alignment = std::max(page_bytes, kCacheLineBytes);
    const int allocation_status = posix_memalign(&allocation, allocation_alignment, data.allocation_bytes);
    if (allocation_status != 0 || allocation == nullptr) {
        throw std::runtime_error("candidate allocation failed: " + std::string(std::strerror(allocation_status)));
    }
    data.storage.reset(static_cast<std::byte *>(allocation));
    data.query.resize(options.dimension);
    data.tape.resize(data.candidate_count);

    std::uint64_t state = kSeed ^ options.dimension;
    for (float &value : data.query) {
        value = DeterministicFloat(state);
        HashBytes(data.input_digest, &value, sizeof(value));
    }
    for (std::size_t candidate = 0; candidate < data.candidate_count; ++candidate) {
        float *values = reinterpret_cast<float *>(data.storage.get() + candidate * data.stride_bytes);
        for (std::size_t component = 0; component < options.dimension; ++component) {
            values[component] = DeterministicFloat(state);
            HashBytes(data.input_digest, &values[component], sizeof(values[component]));
        }
        data.tape[candidate] = static_cast<std::uint32_t>(candidate);
    }
    for (std::size_t index = data.tape.size(); index > 1; --index) {
        const std::size_t swap_index = SplitMix64(state) % index;
        std::swap(data.tape[index - 1], data.tape[swap_index]);
    }
    HashBytes(data.input_digest, data.tape.data(), data.tape.size() * sizeof(data.tape.front()));
    return data;
}

ReplayData MakeValidationData(const Options &options, const HardwareGeometry &geometry) {
    Options validation_options = options;
    validation_options.regime = Regime::kL2;
    const std::size_t stride_bytes = RoundUp(options.dimension * sizeof(float), kCacheLineBytes);
    validation_options.working_set_bytes = stride_bytes * kValidationLimit;
    validation_options.eviction_bytes = std::max<std::size_t>(
        validation_options.eviction_bytes,
        CheckedMultiply(geometry.performance_l2_bytes, 2, "validation eviction geometry"));
    return MakeReplayData(validation_options, geometry, false);
}

KernelDescriptor ResolveKernel(KernelChoice choice, Metric metric, std::size_t dimension) {
    if (choice.engine == Engine::kInfinity) {
        return hnsw_kernel_replay::GetInfinityKernel(metric, choice.boundary, dimension);
    }
    return hnsw_kernel_replay::GetFaissKernel(metric, choice.boundary, dimension);
}

std::uintptr_t FunctionAddress(F32Kernel function) {
    static_assert(sizeof(F32Kernel) == sizeof(std::uintptr_t));
    return std::bit_cast<std::uintptr_t>(function);
}

bool PrintKernelRecord(const char *role, const KernelDescriptor &kernel) {
    Dl_info information{};
    const std::uintptr_t address = FunctionAddress(kernel.function);
    const bool resolved = dladdr(reinterpret_cast<const void *>(address), &information) != 0;
    const std::uintptr_t selected_address = FunctionAddress(kernel.runtime_selected_function);
    const bool starts_at_resolved_symbol =
        resolved && reinterpret_cast<std::uintptr_t>(information.dli_saddr) == address;
    const bool expected_symbol_known =
        std::string_view(kernel.expected_symbol).find("unexpected") == std::string_view::npos;
    const bool selected_boundary = std::string_view(kernel.id).find("-selected-") != std::string_view::npos;
    const bool selection_valid = !selected_boundary || address == selected_address;
    const bool inspection_passed =
        resolved && starts_at_resolved_symbol && expected_symbol_known && selection_valid;
    std::cout << "{\"record\":\"kernel\",\"role\":\"" << role << "\",\"id\":\"" << JsonEscape(kernel.id)
              << "\",\"expected_symbol\":\"" << JsonEscape(kernel.expected_symbol) << "\",\"address\":" << address
              << ",\"runtime_selected_address\":" << selected_address << ",\"matches_runtime_selection\":"
              << (address == selected_address ? "true" : "false") << ",\"dladdr_resolved\":" << (resolved ? "true" : "false")
              << ",\"starts_at_resolved_symbol\":" << (starts_at_resolved_symbol ? "true" : "false")
              << ",\"inspection_passed\":" << (inspection_passed ? "true" : "false");
    if (resolved) {
        std::cout << ",\"image\":\"" << JsonEscape(information.dli_fname == nullptr ? "" : information.dli_fname) << "\",\"symbol\":\""
                  << JsonEscape(information.dli_sname == nullptr ? "" : information.dli_sname) << "\",\"symbol_address\":"
                  << reinterpret_cast<std::uintptr_t>(information.dli_saddr);
    }
    std::cout << "}\n";
    return inspection_passed;
}

double Reference(const ReplayData &data, std::size_t candidate, Metric metric, std::size_t dimension) {
    const float *right = data.Candidate(candidate);
    double result = 0;
    if (metric == Metric::kL2) {
        for (std::size_t component = 0; component < dimension; ++component) {
            const double difference = static_cast<double>(data.query[component]) - static_cast<double>(right[component]);
            result += difference * difference;
        }
    } else {
        for (std::size_t component = 0; component < dimension; ++component) {
            result += static_cast<double>(data.query[component]) * static_cast<double>(right[component]);
        }
    }
    return result;
}

std::uint32_t OrderedFloatBits(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

std::uint32_t UlpDifference(float left, float right) {
    const std::uint32_t ordered_left = OrderedFloatBits(left);
    const std::uint32_t ordered_right = OrderedFloatBits(right);
    return ordered_left > ordered_right ? ordered_left - ordered_right : ordered_right - ordered_left;
}

double RelativeError(float actual, double reference) {
    return std::abs(static_cast<double>(actual) - reference) / std::max(1.0, std::abs(reference));
}

Validation Validate(
    const ReplayData &data,
    const KernelDescriptor &left,
    const KernelDescriptor &right,
    Metric metric,
    std::size_t dimension) {
    Validation validation;
    validation.samples = std::min(kValidationLimit, data.tape.size());
    for (std::size_t sample = 0; sample < validation.samples; ++sample) {
        const std::size_t candidate = data.tape[sample];
        const float left_value = left.function(data.query.data(), data.Candidate(candidate), dimension);
        const float right_value = right.function(data.query.data(), data.Candidate(candidate), dimension);
        const double reference = Reference(data, candidate, metric, dimension);
        validation.finite_failures += !std::isfinite(left_value) || !std::isfinite(right_value);
        validation.bit_mismatches += std::bit_cast<std::uint32_t>(left_value) != std::bit_cast<std::uint32_t>(right_value);
        validation.maximum_ulp_difference = std::max(validation.maximum_ulp_difference, UlpDifference(left_value, right_value));
        validation.maximum_left_relative_error =
            std::max(validation.maximum_left_relative_error, RelativeError(left_value, reference));
        validation.maximum_right_relative_error =
            std::max(validation.maximum_right_relative_error, RelativeError(right_value, reference));
        const std::uint32_t left_bits = std::bit_cast<std::uint32_t>(left_value);
        const std::uint32_t right_bits = std::bit_cast<std::uint32_t>(right_value);
        HashBytes(validation.left_output_digest, &left_bits, sizeof(left_bits));
        HashBytes(validation.right_output_digest, &right_bits, sizeof(right_bits));
    }
    return validation;
}

bool ValidationPassed(const Validation &validation, const Options &options) {
    const bool reference_error_passed =
        validation.maximum_left_relative_error <= options.maximum_relative_error &&
        validation.maximum_right_relative_error <= options.maximum_relative_error;
    return validation.finite_failures == 0 && reference_error_passed &&
        (!options.require_bit_identical || validation.bit_mismatches == 0);
}

const char *TreatmentClass(const Validation &validation) {
    return validation.bit_mismatches == 0 ? "bit_identical_whole_kernel" : "numeric_whole_kernel";
}

void PrintValidation(
    const Validation &validation,
    const ReplayData &validation_data,
    const Options &options) {
    const bool reference_error_passed =
        validation.maximum_left_relative_error <= options.maximum_relative_error &&
        validation.maximum_right_relative_error <= options.maximum_relative_error;
    std::cout << "{\"record\":\"validation\",\"samples\":" << validation.samples
              << ",\"validation_candidate_count\":" << validation_data.candidate_count
              << ",\"validation_input_digest_fnv1a64\":" << validation_data.input_digest << ",\"finite_failures\":"
              << validation.finite_failures << ",\"bit_mismatches\":" << validation.bit_mismatches << ",\"bit_identical\":"
              << (validation.bit_mismatches == 0 ? "true" : "false") << ",\"maximum_ulp_difference\":"
              << validation.maximum_ulp_difference << ",\"maximum_left_relative_error\":" << validation.maximum_left_relative_error
              << ",\"maximum_right_relative_error\":" << validation.maximum_right_relative_error << ",\"left_output_digest\":"
              << validation.left_output_digest << ",\"right_output_digest\":" << validation.right_output_digest
              << ",\"maximum_relative_error_gate\":" << options.maximum_relative_error
              << ",\"reference_error_passed\":" << (reference_error_passed ? "true" : "false")
              << ",\"treatment_class\":\"" << TreatmentClass(validation) << "\",\"validation_passed\":"
              << (ValidationPassed(validation, options) ? "true" : "false") << "}\n";
}

void Evict(std::vector<std::byte> &buffer) {
    std::uint64_t sum = 0;
    for (std::size_t offset = 0; offset < buffer.size(); offset += kCacheLineBytes) {
        sum += static_cast<unsigned char>(buffer[offset]);
    }
    g_eviction_sink = sum;
}

void Warm(const ReplayData &data, Regime regime) {
    if (regime == Regime::kTlb || regime == Regime::kDram) {
        return;
    }
    double sum = std::accumulate(data.query.begin(), data.query.end(), 0.0);
    for (std::size_t candidate = 0; candidate < data.candidate_count; ++candidate) {
        const float *values = data.Candidate(candidate);
        for (std::size_t component = 0; component < data.query.size(); component += 16) {
            sum += values[component];
        }
    }
    g_result_sink = sum;
}

void Condition(const ReplayData &data, Regime regime, std::vector<std::byte> &eviction) {
    Evict(eviction);
    Warm(data, regime);
}

double TimespecSeconds(const timespec &value) {
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) * 1e-9;
}

extern "C" __attribute__((noinline)) float HnswKernelReplayNullF32(
    const float *,
    const float *,
    std::size_t) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return 0.0f;
}

__attribute__((noinline)) double RunKernelBody(
    F32Kernel function,
    const ReplayData &data,
    std::size_t dimension,
    std::size_t calls,
    std::size_t rotation) {
    std::array<double, 8> checksums{};
    const std::size_t mask = data.tape.size() - 1;
    for (std::size_t call = 0; call < calls; ++call) {
        const std::size_t candidate = data.tape[(call + rotation) & mask];
        checksums[call & (checksums.size() - 1)] += function(data.query.data(), data.Candidate(candidate), dimension);
    }
    return std::accumulate(checksums.begin(), checksums.end(), 0.0);
}

__attribute__((noinline)) Timing RunKernel(
    F32Kernel function,
    const ReplayData &data,
    std::size_t dimension,
    std::size_t calls,
    std::size_t rotation) {
    timespec cpu_begin{};
    timespec cpu_end{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_begin) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
    }
    const auto begin = std::chrono::steady_clock::now();
    const double checksum = RunKernelBody(function, data, dimension, calls, rotation);
    const auto end = std::chrono::steady_clock::now();
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
    }
    g_result_sink = checksum;
    return {
        .seconds = std::chrono::duration<double>(end - begin).count(),
        .thread_cpu_seconds = TimespecSeconds(cpu_end) - TimespecSeconds(cpu_begin),
        .checksum = checksum,
    };
}

void PrintRawArm(const RawArm &arm, std::size_t calls, std::size_t dimension) {
    const double distances_per_second = static_cast<double>(calls) / arm.timing.seconds;
    std::cout << "{\"record\":\"timing\",\"block\":" << arm.block << ",\"order\":\"" << arm.order << "\",\"kernel\":\""
              << JsonEscape(arm.kernel->id) << "\",\"calls\":" << calls << ",\"components\":" << calls * dimension
              << ",\"seconds\":" << arm.timing.seconds << ",\"distances_per_second\":" << distances_per_second
              << ",\"thread_cpu_seconds\":" << arm.timing.thread_cpu_seconds << ",\"thread_cpu_over_wall\":"
              << arm.timing.thread_cpu_seconds / arm.timing.seconds
              << ",\"nanoseconds_per_distance\":" << arm.timing.seconds * 1e9 / static_cast<double>(calls)
              << ",\"nanoseconds_per_component\":" << arm.timing.seconds * 1e9 / static_cast<double>(calls * dimension)
              << ",\"checksum\":" << arm.timing.checksum << "}\n";
}

double Mean(const std::vector<double> &values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

std::pair<double, double> StratifiedBootstrap95(
    const std::vector<double> &ab_log_ratios,
    const std::vector<double> &ba_log_ratios) {
    std::vector<double> estimates;
    estimates.reserve(kBootstrapReplicates);
    std::uint64_t state = kSeed ^ 0xa42f9b1637d31e85ULL;
    for (std::size_t replicate = 0; replicate < kBootstrapReplicates; ++replicate) {
        double ab_sum = 0;
        double ba_sum = 0;
        for (std::size_t index = 0; index < ab_log_ratios.size(); ++index) {
            ab_sum += ab_log_ratios[SplitMix64(state) % ab_log_ratios.size()];
        }
        for (std::size_t index = 0; index < ba_log_ratios.size(); ++index) {
            ba_sum += ba_log_ratios[SplitMix64(state) % ba_log_ratios.size()];
        }
        estimates.push_back(
            0.5 *
            (ab_sum / static_cast<double>(ab_log_ratios.size()) +
             ba_sum / static_cast<double>(ba_log_ratios.size())));
    }
    std::sort(estimates.begin(), estimates.end());
    const std::size_t low_index =
        static_cast<std::size_t>(std::floor(0.025 * static_cast<double>(estimates.size() - 1)));
    const std::size_t high_index =
        static_cast<std::size_t>(std::ceil(0.975 * static_cast<double>(estimates.size() - 1)));
    return {estimates[low_index], estimates[high_index]};
}

std::size_t CalibratePairedCalls(
    const Options &options,
    const ReplayData &data,
    const KernelDescriptor &left,
    const KernelDescriptor &right,
    std::vector<std::byte> &eviction,
    std::size_t initial_calls) {
    std::size_t calls = initial_calls;
    for (;;) {
        Condition(data, options.regime, eviction);
        const Timing left_timing = RunKernel(left.function, data, options.dimension, calls, 0);
        Condition(data, options.regime, eviction);
        const Timing right_timing = RunKernel(right.function, data, options.dimension, calls, 0);
        const double minimum_seconds = std::min(left_timing.seconds, right_timing.seconds);
        std::cout << "{\"record\":\"calibration\",\"calls\":" << calls
                  << ",\"left_seconds\":" << left_timing.seconds
                  << ",\"right_seconds\":" << right_timing.seconds
                  << ",\"minimum_arm_seconds_gate\":" << options.minimum_arm_seconds
                  << ",\"passed\":" << (minimum_seconds >= options.minimum_arm_seconds ? "true" : "false")
                  << "}\n";
        if (minimum_seconds >= options.minimum_arm_seconds) {
            return calls;
        }
        if (calls > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error("call calibration overflow");
        }
        calls *= 2;
    }
}

int RunPaired(
    const Options &options,
    const ReplayData &data,
    const KernelDescriptor &left,
    const KernelDescriptor &right,
    std::vector<std::byte> &eviction,
    std::size_t calls,
    std::size_t warmup_calls) {
    calls = CalibratePairedCalls(options, data, left, right, eviction, calls);
    warmup_calls = std::max<std::size_t>(
        warmup_calls,
        std::max<std::size_t>(1, std::min<std::size_t>(calls / 8, std::size_t{1} << 20)));

    Condition(data, options.regime, eviction);
    RunKernel(left.function, data, options.dimension, warmup_calls, 0);
    Condition(data, options.regime, eviction);
    RunKernel(right.function, data, options.dimension, warmup_calls, 0);
    Condition(data, options.regime, eviction);
    const Timing harness_floor =
        RunKernel(&HnswKernelReplayNullF32, data, options.dimension, calls, 0);
    std::cout << "{\"record\":\"harness_floor\",\"calls\":" << calls
              << ",\"seconds\":" << harness_floor.seconds
              << ",\"nanoseconds_per_call\":"
              << harness_floor.seconds * 1e9 / static_cast<double>(calls)
              << ",\"scope\":\"tape_lookup_addressing_indirect_call_checksum\"}\n";

    std::vector<double> ab_log_ratios;
    std::vector<double> ba_log_ratios;
    for (std::size_t block = 0; block < options.rounds; ++block) {
        const bool ab = block % 2 == 0;
        const std::array<const KernelDescriptor *, 2> order = ab
            ? std::array<const KernelDescriptor *, 2>{&left, &right}
            : std::array<const KernelDescriptor *, 2>{&right, &left};
        std::array<Timing, 2> timings;
        for (std::size_t arm = 0; arm < order.size(); ++arm) {
            Condition(data, options.regime, eviction);
            timings[arm] = RunKernel(order[arm]->function, data, options.dimension, calls, block * 7919);
            if (timings[arm].seconds < options.minimum_arm_seconds) {
                throw std::runtime_error("timed arm fell below the minimum duration gate");
            }
            PrintRawArm(
                RawArm{
                    .block = block,
                    .order = ab ? "AB" : "BA",
                    .kernel = order[arm],
                    .timing = timings[arm],
                },
                calls,
                options.dimension);
        }
        const double left_seconds = ab ? timings[0].seconds : timings[1].seconds;
        const double right_seconds = ab ? timings[1].seconds : timings[0].seconds;
        const double log_rate_ratio = std::log(right_seconds / left_seconds);
        (ab ? ab_log_ratios : ba_log_ratios).push_back(log_rate_ratio);
        std::cout << "{\"record\":\"pair\",\"block\":" << block << ",\"order\":\"" << (ab ? "AB" : "BA")
                  << "\",\"left_over_right_rate_ratio\":" << std::exp(log_rate_ratio) << "}\n";
    }

    if (ab_log_ratios.empty() || ba_log_ratios.empty()) {
        std::cout << "{\"record\":\"summary\",\"status\":\"INSUFFICIENT_ORDERS\"}\n";
        return 1;
    }
    const double ab_mean = Mean(ab_log_ratios);
    const double ba_mean = Mean(ba_log_ratios);
    const double theta = 0.5 * (ab_mean + ba_mean);
    const auto [bootstrap_low, bootstrap_high] =
        StratifiedBootstrap95(ab_log_ratios, ba_log_ratios);
    std::cout << "{\"record\":\"summary\",\"status\":\"VALIDATED_DIAGNOSTIC\",\"estimand\":\"stratified_paired_log_rate_ratio\""
              << ",\"left\":\"" << JsonEscape(left.id) << "\",\"right\":\"" << JsonEscape(right.id) << "\",\"ab_blocks\":"
              << ab_log_ratios.size() << ",\"ba_blocks\":" << ba_log_ratios.size() << ",\"left_over_right_rate_ratio\":"
              << std::exp(theta) << ",\"stratified_bootstrap_replicates\":" << kBootstrapReplicates
              << ",\"stratified_bootstrap_95_low\":" << std::exp(bootstrap_low)
              << ",\"stratified_bootstrap_95_high\":" << std::exp(bootstrap_high) << "}\n";
    return 0;
}

extern "C" __attribute__((noinline)) void HnswKernelReplayRoiBegin() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

extern "C" __attribute__((noinline)) void HnswKernelReplayRoiEnd() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

void WaitForStartSignal() {
    sigset_t wait_set;
    sigemptyset(&wait_set);
    sigaddset(&wait_set, SIGUSR1);
    sigset_t previous_set;
    const int block_status = pthread_sigmask(SIG_BLOCK, &wait_set, &previous_set);
    if (block_status != 0) {
        throw std::runtime_error(
            "pthread_sigmask(SIG_BLOCK) failed: " + std::string(std::strerror(block_status)));
    }
    std::cout << "{\"record\":\"ready\",\"pid\":" << getpid()
              << ",\"start_signal\":\"SIGUSR1\",\"signal_wait\":\"sigwait\"}\n"
              << std::flush;
    int received_signal = 0;
    const int wait_status = sigwait(&wait_set, &received_signal);
    const int restore_status = pthread_sigmask(SIG_SETMASK, &previous_set, nullptr);
    if (wait_status != 0 || received_signal != SIGUSR1) {
        throw std::runtime_error(
            "sigwait(SIGUSR1) failed: " + std::string(std::strerror(wait_status)));
    }
    if (restore_status != 0) {
        throw std::runtime_error(
            "pthread_sigmask(SIG_SETMASK) failed: " + std::string(std::strerror(restore_status)));
    }
}

std::pair<std::size_t, Timing> CalibrateSingleCalls(
    const Options &options,
    const ReplayData &data,
    const KernelDescriptor &kernel,
    std::vector<std::byte> &eviction,
    std::size_t initial_calls) {
    std::size_t calls = initial_calls;
    for (;;) {
        Condition(data, options.regime, eviction);
        const Timing timing =
            RunKernel(kernel.function, data, options.dimension, calls, 0);
        std::cout << "{\"record\":\"attach_calibration\",\"calls\":" << calls
                  << ",\"seconds\":" << timing.seconds
                  << ",\"minimum_arm_seconds_gate\":" << options.minimum_arm_seconds
                  << ",\"passed\":" << (timing.seconds >= options.minimum_arm_seconds ? "true" : "false")
                  << "}\n";
        if (timing.seconds >= options.minimum_arm_seconds) {
            return {calls, timing};
        }
        if (calls > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error("attach call calibration overflow");
        }
        calls *= 2;
    }
}

int RunAttach(
    const Options &options,
    const ReplayData &data,
    const KernelDescriptor &kernel,
    std::vector<std::byte> &eviction,
    std::size_t calls) {
    const auto [calibrated_calls, calibration] =
        CalibrateSingleCalls(options, data, kernel, eviction, calls);
    calls = calibrated_calls;
    const std::size_t batches = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(
            std::ceil(static_cast<double>(options.duration_seconds) / calibration.seconds)));
    if (batches > std::numeric_limits<std::size_t>::max() / calls) {
        throw std::overflow_error("attach total call count overflow");
    }
    if (options.wait_for_signal) {
        WaitForStartSignal();
    }

    Condition(data, options.regime, eviction);
    const os_log_t signpost_log =
        os_log_create("com.infiniflow.infinity", "hnsw-kernel-replay");
    const os_signpost_id_t signpost_id = os_signpost_id_generate(signpost_log);
    timespec cpu_begin{};
    timespec cpu_end{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_begin) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
    }
    const auto roi_begin = std::chrono::steady_clock::now();
    os_signpost_interval_begin(
        signpost_log,
        signpost_id,
        "hnsw-kernel-replay-roi",
        "kernel=%{public}s calls=%{public}zu batches=%{public}zu",
        kernel.id,
        calls,
        batches);
    HnswKernelReplayRoiBegin();
    double checksum = 0;
    for (std::size_t batch = 0; batch < batches; ++batch) {
        checksum += RunKernelBody(
            kernel.function,
            data,
            options.dimension,
            calls,
            batch * 7919);
    }
    HnswKernelReplayRoiEnd();
    os_signpost_interval_end(
        signpost_log,
        signpost_id,
        "hnsw-kernel-replay-roi",
        "checksum=%{public}f",
        checksum);
    const auto roi_end = std::chrono::steady_clock::now();
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
    }
    const double seconds = std::chrono::duration<double>(roi_end - roi_begin).count();
    const double thread_cpu_seconds =
        TimespecSeconds(cpu_end) - TimespecSeconds(cpu_begin);
    const std::size_t total_calls = batches * calls;
    g_result_sink = checksum;
    std::cout << "{\"record\":\"attach_result\",\"kernel\":\"" << JsonEscape(kernel.id) << "\",\"batches\":" << batches
              << ",\"calls\":" << total_calls << ",\"seconds\":" << seconds << ",\"distances_per_second\":"
              << static_cast<double>(total_calls) / seconds
              << ",\"thread_cpu_seconds\":" << thread_cpu_seconds
              << ",\"thread_cpu_over_wall\":" << thread_cpu_seconds / seconds
              << ",\"roi_clock_polling\":false,\"roi_signpost\":true,\"checksum\":" << checksum << "}\n";
    return 0;
}

RuntimePolicy ConfigureRuntimePolicy() {
    RuntimePolicy policy;
    policy.set_qos_status = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    policy.get_qos_status = pthread_get_qos_class_np(pthread_self(), &policy.qos_class, &policy.relative_priority);
    return policy;
}

void PrintConfiguration(
    const Options &options,
    const ReplayData &data,
    std::size_t calls,
    std::size_t warmup_calls,
    const RuntimePolicy &policy,
    const HardwareGeometry &geometry) {
    std::cout << "{\"record\":\"configuration\",\"schema\":2,\"pid\":" << getpid() << ",\"architecture\":\"arm64-apple\""
              << ",\"metric\":\"" << MetricName(options.metric) << "\",\"dimension\":" << options.dimension
              << ",\"main_loop_components\":" << (options.dimension / 16) * 16 << ",\"residual_components\":"
              << options.dimension % 16 << ",\"requested_geometry\":\"" << RegimeName(options.regime) << "\",\"calls\":" << calls
              << ",\"warmup_calls\":" << warmup_calls << ",\"rounds\":" << options.rounds << ",\"candidate_count\":"
              << data.candidate_count << ",\"vector_bytes\":" << options.dimension * sizeof(float) << ",\"stride_bytes\":"
              << data.stride_bytes << ",\"allocation_bytes\":" << data.allocation_bytes << ",\"tape_entries\":"
              << data.tape.size() << ",\"input_digest_fnv1a64\":" << data.input_digest << ",\"page_bytes\":" << geometry.page_bytes
              << ",\"performance_l1d_bytes\":" << geometry.performance_l1d_bytes
              << ",\"performance_l2_bytes\":" << geometry.performance_l2_bytes
              << ",\"performance_core_count\":" << geometry.performance_core_count
              << ",\"physical_memory_bytes\":" << geometry.memory_bytes
              << ",\"custom_working_set\":" << (options.working_set_bytes == 0 ? "false" : "true")
              << ",\"eviction_bytes\":" << options.eviction_bytes
              << ",\"minimum_arm_seconds\":" << options.minimum_arm_seconds
              << ",\"maximum_relative_error\":" << options.maximum_relative_error
              << ",\"set_qos_status\":" << policy.set_qos_status << ",\"get_qos_status\":" << policy.get_qos_status
              << ",\"qos_class\":"
              << static_cast<unsigned int>(policy.qos_class) << ",\"qos_relative_priority\":" << policy.relative_priority
              << ",\"cache_state_claim\":\"geometry_only_unverified_by_pmu\""
              << ",\"claim_scope\":\"synthetic_same-pointer_microbenchmark\"}\n";
}

std::size_t DefaultCalls(const Options &options, const ReplayData &data) {
    if (options.calls != 0) {
        return options.calls;
    }
    switch (options.regime) {
        case Regime::kHot:
        case Regime::kL1:
            return 1ULL << 20;
        case Regime::kL2:
            return 1ULL << 19;
        case Regime::kTlb:
        case Regime::kDram:
            return std::max<std::size_t>(data.candidate_count, 1ULL << 18);
    }
    std::abort();
}

} // namespace

int main(int argc, char **argv) {
    try {
#if !defined(__APPLE__) || !defined(__aarch64__)
        std::cerr << "hnsw_kernel_replay requires native arm64 macOS\n";
        return 64;
#else
        std::cout << std::setprecision(17);
        const Options options = ParseOptions(argc, argv);
        const RuntimePolicy runtime_policy = ConfigureRuntimePolicy();
        if (runtime_policy.set_qos_status != 0 || runtime_policy.get_qos_status != 0) {
            throw std::runtime_error("pthread QoS configuration failed");
        }
        const HardwareGeometry hardware_geometry = ReadHardwareGeometry();
        const ReplayData data = MakeReplayData(options, hardware_geometry);
        const std::size_t calls = DefaultCalls(options, data);
        const std::size_t warmup_calls =
            options.warmup_calls == 0
            ? std::max<std::size_t>(1, std::min<std::size_t>(calls / 8, std::size_t{1} << 16))
            : options.warmup_calls;
        if (calls == 0) {
            throw std::invalid_argument("calls must be positive");
        }
        std::vector<std::byte> eviction(options.eviction_bytes);
        for (std::size_t offset = 0; offset < eviction.size(); offset += kCacheLineBytes) {
            eviction[offset] = static_cast<std::byte>((offset / kCacheLineBytes) & 0xff);
        }

        PrintConfiguration(
            options,
            data,
            calls,
            warmup_calls,
            runtime_policy,
            hardware_geometry);
        const ReplayData validation_data =
            MakeValidationData(options, hardware_geometry);
        if (options.mode == Mode::kAttach) {
            const KernelDescriptor kernel = ResolveKernel(options.attach_kernel, options.metric, options.dimension);
            if (!PrintKernelRecord("attach", kernel)) {
                std::cout << "{\"record\":\"final\",\"status\":\"KERNEL_INSPECTION_FAILED\"}\n";
                return 2;
            }
            const Validation validation =
                Validate(validation_data, kernel, kernel, options.metric, options.dimension);
            PrintValidation(validation, validation_data, options);
            if (!ValidationPassed(validation, options)) {
                std::cout << "{\"record\":\"final\",\"status\":\"VALIDATION_FAILED\"}\n";
                return 2;
            }
            return RunAttach(options, data, kernel, eviction, calls);
        }

        const KernelDescriptor left = ResolveKernel(options.left, options.metric, options.dimension);
        const KernelDescriptor right = ResolveKernel(options.right, options.metric, options.dimension);
        const bool left_inspected = PrintKernelRecord("left", left);
        const bool right_inspected = PrintKernelRecord("right", right);
        if (!left_inspected || !right_inspected) {
            std::cout << "{\"record\":\"final\",\"status\":\"KERNEL_INSPECTION_FAILED\"}\n";
            return 2;
        }
        const Validation validation = Validate(validation_data, left, right, options.metric, options.dimension);
        PrintValidation(validation, validation_data, options);
        if (!ValidationPassed(validation, options)) {
            std::cout << "{\"record\":\"final\",\"status\":\"VALIDATION_FAILED\"}\n";
            return 2;
        }
        if (options.mode == Mode::kValidate) {
            std::cout << "{\"record\":\"final\",\"status\":\""
                      << (validation.bit_mismatches == 0
                              ? "VALIDATED_BIT_IDENTICAL_TREATMENT"
                              : "VALIDATED_NUMERIC_TREATMENT")
                      << "\"}\n";
            return 0;
        }
        const int status = RunPaired(options, data, left, right, eviction, calls, warmup_calls);
        std::cout << "{\"record\":\"final\",\"status\":\""
                  << (status == 0 ? "VALIDATED_DIAGNOSTIC_COMPLETE" : "FAIL")
                  << "\",\"treatment_class\":\"" << TreatmentClass(validation) << "\"}\n";
        return status;
#endif
    } catch (const std::exception &error) {
        std::cerr << "hnsw_kernel_replay: " << error.what() << '\n';
        return 64;
    }
}
