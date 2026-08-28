#include "hnsw_kernel_replay_adapter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
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
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using hnsw_kernel_replay::I8Kernel;
using hnsw_kernel_replay::I8KernelDescriptor;
using hnsw_kernel_replay::I8KernelVariant;

constexpr std::uint64_t kSeed = 0x65daef390cd27437ULL;
constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::size_t kLvqHeaderBytes = 16;
constexpr std::size_t kMinimumPairedRounds = 32;
constexpr std::size_t kBootstrapReplicates = 99'999;
constexpr std::size_t kValidationCandidateLimit = 4096;
constexpr double kDefaultMinimumArmSeconds = 0.20;
constexpr std::size_t kDefaultEvictionBytes = 64ULL << 20;
constexpr double kMinimumThreadCpuOverWall = 0.995;
constexpr double kMaximumThreadCpuOverWall = 1.02;
constexpr std::size_t kMaximumSafeDimension =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / (128ULL * 128ULL);

volatile std::int64_t g_result_sink = 0;
volatile std::uint64_t g_eviction_sink = 0;

enum class Mode {
    kValidate,
    kKernelPaired,
    kLayoutPaired,
};

enum class Regime {
    kHot,
    kL1,
    kL2,
    kTlb,
    kDram,
};

enum class Layout {
    kPackedLvq,
    kF32PaddedLvq,
};

struct Options {
    Mode mode = Mode::kValidate;
    Regime regime = Regime::kHot;
    Layout layout = Layout::kPackedLvq;
    Layout left_layout = Layout::kPackedLvq;
    Layout right_layout = Layout::kF32PaddedLvq;
    I8KernelVariant left = I8KernelVariant::kSelected;
    I8KernelVariant right = I8KernelVariant::kForcedScalar;
    I8KernelVariant layout_kernel = I8KernelVariant::kSelected;
    std::size_t dimension = 128;
    std::size_t candidate_count = 0;
    std::size_t working_set_bytes = 0;
    std::size_t calls = 0;
    std::size_t rounds = kMinimumPairedRounds;
    std::size_t eviction_bytes = kDefaultEvictionBytes;
    double minimum_arm_seconds = kDefaultMinimumArmSeconds;
};

struct HardwareGeometry {
    std::uint64_t memory_bytes = 0;
    std::uint64_t page_bytes = 0;
    std::uint64_t cache_line_bytes = 0;
    std::uint64_t performance_l1d_bytes = 0;
    std::uint64_t performance_l2_bytes = 0;
};

struct LvqHeader {
    float scale = 0;
    float bias = 0;
    float norm1_scale = 0;
    float norm2sq_scalesq = 0;
};

static_assert(sizeof(LvqHeader) == kLvqHeaderBytes);
static_assert(std::is_trivially_copyable_v<LvqHeader>);

struct FreeDeleter {
    void operator()(std::byte *pointer) const noexcept { std::free(pointer); }
};

struct ReplayData {
    std::unique_ptr<std::byte, FreeDeleter> storage;
    std::vector<std::int8_t> query;
    LvqHeader query_header;
    std::vector<std::uint32_t> tape;
    std::size_t useful_record_bytes = 0;
    std::size_t stride_bytes = 0;
    std::size_t candidate_count = 0;
    std::size_t allocation_bytes = 0;
    std::uint64_t operand_digest = kFnvOffset;
    std::uint64_t layout_digest = kFnvOffset;

    const std::int8_t *Candidate(std::size_t index) const noexcept {
        return reinterpret_cast<const std::int8_t *>(
            storage.get() + index * stride_bytes + kLvqHeaderBytes);
    }

    const std::byte *Record(std::size_t index) const noexcept {
        return storage.get() + index * stride_bytes;
    }
};

struct Timing {
    double seconds = 0;
    double thread_cpu_seconds = 0;
    std::int64_t checksum = 0;
    std::size_t cpu_start = std::numeric_limits<std::size_t>::max();
    std::size_t cpu_end = std::numeric_limits<std::size_t>::max();
    int cpu_start_status = 0;
    int cpu_end_status = 0;
    long minor_page_faults = 0;
    long major_page_faults = 0;

    [[nodiscard]] double ThreadCpuOverWall() const noexcept {
        return thread_cpu_seconds / seconds;
    }

    [[nodiscard]] bool EndpointCpuStable() const noexcept {
        return cpu_start_status == 0 && cpu_end_status == 0 && cpu_start == cpu_end;
    }

    [[nodiscard]] bool OnCpuGatePassed() const noexcept {
        const double ratio = ThreadCpuOverWall();
        return ratio >= kMinimumThreadCpuOverWall && ratio <= kMaximumThreadCpuOverWall;
    }

    [[nodiscard]] bool PageFaultGatePassed() const noexcept {
        return minor_page_faults == 0 && major_page_faults == 0;
    }
};

struct StorageCoverage {
    std::size_t useful_bytes = 0;
    std::size_t padding_bytes = 0;
    std::size_t cache_lines = 0;
    std::size_t pages = 0;
};

struct RuntimePolicy {
    int set_qos_status = 0;
    int get_qos_status = 0;
    qos_class_t qos_class = QOS_CLASS_UNSPECIFIED;
    int relative_priority = 0;
};

std::uint64_t ParseUnsigned(std::string_view value, const char *name) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
    }
    const std::string text(value);
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    return parsed;
}

double ParsePositiveDouble(std::string_view value, const char *name) {
    const std::string text(value);
    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        !std::isfinite(parsed) || parsed <= 0) {
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

I8KernelVariant ParseVariant(std::string_view value) {
    if (value == "selected") {
        return I8KernelVariant::kSelected;
    }
    if (value == "bf" || value == "autovec") {
        return I8KernelVariant::kDirectBF;
    }
    if (value == "simde-sse") {
        return I8KernelVariant::kDirectSIMDeSSE;
    }
    if (value == "scalar") {
        return I8KernelVariant::kForcedScalar;
    }
    throw std::invalid_argument("unknown I8 kernel: " + std::string(value));
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

Layout ParseLayout(std::string_view value) {
    if (value == "packed-lvq") {
        return Layout::kPackedLvq;
    }
    if (value == "f32-padded-lvq") {
        return Layout::kF32PaddedLvq;
    }
    throw std::invalid_argument("unknown layout: " + std::string(value));
}

const char *LayoutName(Layout layout) {
    return layout == Layout::kPackedLvq ? "packed-lvq" : "f32-padded-lvq";
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--mode") {
            const std::string_view value = RequireValue(argc, argv, index);
            if (value == "validate") {
                options.mode = Mode::kValidate;
            } else if (value == "paired") {
                options.mode = Mode::kKernelPaired;
            } else if (value == "layout-paired") {
                options.mode = Mode::kLayoutPaired;
            } else {
                throw std::invalid_argument("unknown mode: " + std::string(value));
            }
        } else if (argument == "--left") {
            options.left = ParseVariant(RequireValue(argc, argv, index));
        } else if (argument == "--right") {
            options.right = ParseVariant(RequireValue(argc, argv, index));
        } else if (argument == "--dimension") {
            options.dimension = ParseUnsigned(RequireValue(argc, argv, index), "dimension");
        } else if (argument == "--regime") {
            options.regime = ParseRegime(RequireValue(argc, argv, index));
        } else if (argument == "--layout") {
            options.layout = ParseLayout(RequireValue(argc, argv, index));
        } else if (argument == "--left-layout") {
            options.left_layout = ParseLayout(RequireValue(argc, argv, index));
        } else if (argument == "--right-layout") {
            options.right_layout = ParseLayout(RequireValue(argc, argv, index));
        } else if (argument == "--kernel") {
            options.layout_kernel = ParseVariant(RequireValue(argc, argv, index));
        } else if (argument == "--candidate-count") {
            options.candidate_count =
                ParseUnsigned(RequireValue(argc, argv, index), "candidate count");
        } else if (argument == "--working-set-bytes") {
            options.working_set_bytes =
                ParseUnsigned(RequireValue(argc, argv, index), "working-set bytes");
        } else if (argument == "--calls") {
            options.calls = ParseUnsigned(RequireValue(argc, argv, index), "calls");
        } else if (argument == "--rounds") {
            options.rounds = ParseUnsigned(RequireValue(argc, argv, index), "rounds");
        } else if (argument == "--eviction-bytes") {
            options.eviction_bytes =
                ParseUnsigned(RequireValue(argc, argv, index), "eviction bytes");
        } else if (argument == "--minimum-arm-seconds") {
            options.minimum_arm_seconds =
                ParsePositiveDouble(RequireValue(argc, argv, index), "minimum arm seconds");
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: hnsw_kernel_replay_i8 [options]\n"
                << "  --mode validate|paired|layout-paired\n"
                << "  --left K --right K, K=selected|bf|autovec|simde-sse|scalar\n"
                << "  --kernel K               shared kernel for layout-paired mode\n"
                << "  --dimension N --regime hot|l1|l2|tlb|dram\n"
                << "  --layout packed-lvq|f32-padded-lvq\n"
                << "  --left-layout L --right-layout L\n"
                << "  --candidate-count N --working-set-bytes N\n"
                << "  --calls N --rounds N --eviction-bytes N\n"
                << "  --minimum-arm-seconds X\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    if (options.dimension == 0 || options.dimension > kMaximumSafeDimension) {
        throw std::invalid_argument(
            "dimension must be in [1, " + std::to_string(kMaximumSafeDimension) +
            "] so every signed-int8 dot product fits int32");
    }
    if (options.mode != Mode::kValidate &&
        (options.rounds < kMinimumPairedRounds || options.rounds % 2 != 0)) {
        throw std::invalid_argument(
            "paired mode requires an even --rounds value of at least " +
            std::to_string(kMinimumPairedRounds));
    }
    if (options.mode == Mode::kLayoutPaired && options.candidate_count == 0) {
        throw std::invalid_argument(
            "layout-paired mode requires an explicit --candidate-count");
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

std::int8_t DeterministicI8(std::uint64_t &state) {
    return static_cast<std::int8_t>(SplitMix64(state) & 0xffU);
}

float DeterministicUnitFloat(std::uint64_t &state) {
    constexpr double denominator = static_cast<double>(std::uint32_t{1} << 24);
    const auto bits = static_cast<std::uint32_t>(SplitMix64(state) >> 40);
    return static_cast<float>(static_cast<double>(bits) / denominator);
}

LvqHeader MakeLvqHeader(
    const std::int8_t *codes,
    std::size_t dimension,
    std::uint64_t &state) {
    std::int64_t norm1 = 0;
    std::int64_t norm2 = 0;
    for (std::size_t component = 0; component < dimension; ++component) {
        const std::int64_t value = codes[component];
        norm1 += value;
        norm2 += value * value;
    }
    const float scale = 0.001f + DeterministicUnitFloat(state) * 0.019f;
    const float bias = DeterministicUnitFloat(state) * 2.0f - 1.0f;
    return {
        .scale = scale,
        .bias = bias,
        .norm1_scale = static_cast<float>(norm1) * scale,
        .norm2sq_scalesq = static_cast<float>(norm2) * scale * scale,
    };
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
        .cache_line_bytes = ReadRequiredSysctlU64("hw.cachelinesize"),
        .performance_l1d_bytes = ReadRequiredSysctlU64("hw.perflevel0.l1dcachesize"),
        .performance_l2_bytes = ReadRequiredSysctlU64("hw.perflevel0.l2cachesize"),
    };
}

std::size_t CheckedMultiply(std::uint64_t left, std::uint64_t right, const char *name) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string(name) + " overflow");
    }
    return static_cast<std::size_t>(left * right);
}

std::size_t LayoutStride(Layout layout, std::size_t dimension) {
    const std::size_t packed = kLvqHeaderBytes + dimension;
    if (layout == Layout::kPackedLvq) {
        return packed;
    }
    const std::size_t padded =
        CheckedMultiply(dimension, sizeof(float), "padded stride");
    if (padded < packed) {
        throw std::invalid_argument(
            "f32-padded LVQ requires 4 * dimension >= dimension + 16");
    }
    return padded;
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

ReplayData MakeReplayData(const Options &options, const HardwareGeometry &geometry) {
    ReplayData data;
    data.useful_record_bytes = kLvqHeaderBytes + options.dimension;
    data.stride_bytes = LayoutStride(options.layout, options.dimension);
    if (options.candidate_count != 0) {
        data.candidate_count = options.candidate_count;
    } else if (options.regime == Regime::kHot) {
        data.candidate_count = 1;
    } else {
        const std::size_t target = options.working_set_bytes == 0
            ? DefaultWorkingSet(options.regime, geometry)
            : options.working_set_bytes;
        if (target < data.stride_bytes) {
            throw std::invalid_argument("working set is smaller than one record");
        }
        data.candidate_count = std::bit_floor(target / data.stride_bytes);
    }
    if (data.candidate_count == 0 ||
        data.candidate_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid candidate count");
    }
    if (data.candidate_count >
        std::numeric_limits<std::size_t>::max() / data.stride_bytes) {
        throw std::overflow_error("candidate allocation overflow");
    }
    data.allocation_bytes = data.candidate_count * data.stride_bytes;
    if (options.candidate_count == 0 &&
        (options.regime == Regime::kTlb || options.regime == Regime::kDram) &&
        data.allocation_bytes < geometry.performance_l2_bytes * 8) {
        throw std::invalid_argument(
            "TLB/DRAM geometry must be at least eight P-cluster L2 caches");
    }
    if (data.allocation_bytes > geometry.memory_bytes / 4) {
        throw std::invalid_argument("replay allocation may not exceed one quarter of memory");
    }

    void *allocation = nullptr;
    const int status = posix_memalign(
        &allocation,
        std::max<std::size_t>(geometry.page_bytes, geometry.cache_line_bytes),
        data.allocation_bytes);
    if (status != 0 || allocation == nullptr) {
        throw std::runtime_error(
            "candidate allocation failed: " + std::string(std::strerror(status)));
    }
    data.storage.reset(static_cast<std::byte *>(allocation));
    std::memset(data.storage.get(), 0xa5, data.allocation_bytes);
    data.query.resize(options.dimension);
    data.tape.resize(data.candidate_count);

    std::uint64_t state = kSeed ^ options.dimension;
    for (std::int8_t &value : data.query) {
        value = DeterministicI8(state);
    }
    data.query_header = MakeLvqHeader(data.query.data(), options.dimension, state);
    HashBytes(data.operand_digest, &data.query_header, sizeof(data.query_header));
    HashBytes(
        data.operand_digest,
        data.query.data(),
        data.query.size() * sizeof(data.query.front()));
    for (std::size_t candidate = 0; candidate < data.candidate_count; ++candidate) {
        auto *values = const_cast<std::int8_t *>(data.Candidate(candidate));
        for (std::size_t component = 0; component < options.dimension; ++component) {
            values[component] = DeterministicI8(state);
        }
        const LvqHeader header = MakeLvqHeader(values, options.dimension, state);
        std::memcpy(
            data.storage.get() + candidate * data.stride_bytes,
            &header,
            sizeof(header));
        HashBytes(data.operand_digest, &header, sizeof(header));
        HashBytes(data.operand_digest, values, options.dimension);
        data.tape[candidate] = static_cast<std::uint32_t>(candidate);
    }
    for (std::size_t index = data.tape.size(); index > 1; --index) {
        const std::size_t swap_index = SplitMix64(state) % index;
        std::swap(data.tape[index - 1], data.tape[swap_index]);
    }
    HashBytes(data.operand_digest, data.tape.data(), data.tape.size() * sizeof(data.tape.front()));
    HashBytes(data.layout_digest, data.storage.get(), data.allocation_bytes);
    HashBytes(data.layout_digest, &data.stride_bytes, sizeof(data.stride_bytes));
    return data;
}

LvqHeader LoadHeader(const ReplayData &data, std::size_t candidate) {
    LvqHeader header;
    std::memcpy(&header, data.Record(candidate), sizeof(header));
    return header;
}

std::size_t CountCoveredUnits(const ReplayData &data, std::size_t unit_bytes) {
    if (unit_bytes == 0) {
        throw std::invalid_argument("coverage unit must be positive");
    }
    std::size_t covered = 0;
    std::size_t previous_last = 0;
    bool have_previous = false;
    for (std::size_t candidate = 0; candidate < data.candidate_count; ++candidate) {
        const std::size_t begin = candidate * data.stride_bytes;
        const std::size_t end = begin + data.useful_record_bytes - 1;
        const std::size_t first_unit = begin / unit_bytes;
        const std::size_t last_unit = end / unit_bytes;
        if (!have_previous || first_unit > previous_last) {
            covered += last_unit - first_unit + 1;
        } else if (last_unit > previous_last) {
            covered += last_unit - previous_last;
        }
        previous_last = std::max(previous_last, last_unit);
        have_previous = true;
    }
    return covered;
}

StorageCoverage MeasureCoverage(
    const ReplayData &data,
    const HardwareGeometry &geometry) {
    return {
        .useful_bytes = CheckedMultiply(
            data.candidate_count, data.useful_record_bytes, "useful bytes"),
        .padding_bytes = CheckedMultiply(
            data.candidate_count,
            data.stride_bytes - data.useful_record_bytes,
            "padding bytes"),
        .cache_lines = CountCoveredUnits(
            data, static_cast<std::size_t>(geometry.cache_line_bytes)),
        .pages = CountCoveredUnits(
            data, static_cast<std::size_t>(geometry.page_bytes)),
    };
}

void RequireEquivalentOperands(
    const ReplayData &left,
    const ReplayData &right,
    std::size_t dimension) {
    if (left.candidate_count != right.candidate_count ||
        left.useful_record_bytes != right.useful_record_bytes ||
        left.query_header.scale != right.query_header.scale ||
        left.query_header.bias != right.query_header.bias ||
        left.query_header.norm1_scale != right.query_header.norm1_scale ||
        left.query_header.norm2sq_scalesq != right.query_header.norm2sq_scalesq ||
        left.query != right.query ||
        left.tape != right.tape ||
        left.operand_digest != right.operand_digest) {
        throw std::runtime_error("layout arms do not share identical logical operands");
    }
    for (std::size_t candidate = 0; candidate < left.candidate_count; ++candidate) {
        if (std::memcmp(
                left.Record(candidate),
                right.Record(candidate),
                kLvqHeaderBytes) != 0 ||
            std::memcmp(
                left.Candidate(candidate),
                right.Candidate(candidate),
                dimension) != 0) {
            throw std::runtime_error(
                "layout arms differ in a useful record at candidate " +
                std::to_string(candidate));
        }
    }
}

void RequirePaddingCanary(const ReplayData &data, const char *phase) {
    constexpr std::byte canary{0xa5};
    for (std::size_t candidate = 0; candidate < data.candidate_count; ++candidate) {
        const std::byte *record = data.Record(candidate);
        for (std::size_t offset = data.useful_record_bytes;
             offset < data.stride_bytes;
             ++offset) {
            if (record[offset] != canary) {
                throw std::runtime_error(
                    std::string("LVQ padding canary changed during ") + phase +
                    " at candidate " + std::to_string(candidate) +
                    ", offset " + std::to_string(offset));
            }
        }
    }
    std::cout << "{\"record\":\"padding_canary\",\"phase\":\"" << phase
              << "\",\"stride_bytes\":" << data.stride_bytes
              << ",\"useful_record_bytes\":" << data.useful_record_bytes
              << ",\"passed\":true}\n";
}

std::uintptr_t FunctionAddress(I8Kernel function) {
    static_assert(sizeof(I8Kernel) == sizeof(std::uintptr_t));
    return std::bit_cast<std::uintptr_t>(function);
}

bool PrintKernelRecord(const char *role, const I8KernelDescriptor &kernel) {
    Dl_info information{};
    const std::uintptr_t address = FunctionAddress(kernel.function);
    const bool resolved =
        dladdr(reinterpret_cast<const void *>(address), &information) != 0;
    const bool starts_at_symbol =
        resolved && reinterpret_cast<std::uintptr_t>(information.dli_saddr) == address;
    const bool selection_valid = kernel.SatisfiesRuntimeSelectionExpectation();
    const bool passed = resolved && starts_at_symbol && selection_valid;
    std::cout << "{\"record\":\"kernel\",\"role\":\"" << role
              << "\",\"id\":\"" << kernel.id
              << "\",\"expected_symbol\":\"" << kernel.expected_symbol
              << "\",\"address\":" << address
              << ",\"runtime_selected_address\":"
              << FunctionAddress(kernel.runtime_selected_function)
              << ",\"matches_runtime_selection\":"
              << (kernel.MatchesRuntimeSelection() ? "true" : "false")
              << ",\"selection_expectation_passed\":"
              << (selection_valid ? "true" : "false")
              << ",\"dladdr_resolved\":" << (resolved ? "true" : "false")
              << ",\"starts_at_resolved_symbol\":"
              << (starts_at_symbol ? "true" : "false")
              << ",\"inspection_passed\":" << (passed ? "true" : "false");
    if (resolved) {
        std::cout << ",\"image\":\""
                  << (information.dli_fname == nullptr ? "" : information.dli_fname)
                  << "\",\"symbol\":\""
                  << (information.dli_sname == nullptr ? "" : information.dli_sname)
                  << "\"";
    }
    std::cout << "}\n";
    return passed;
}

std::int64_t ExactDot(
    const std::int8_t *left,
    const std::int8_t *right,
    std::size_t dimension) {
    std::int64_t sum = 0;
    for (std::size_t component = 0; component < dimension; ++component) {
        sum += static_cast<std::int64_t>(left[component]) *
            static_cast<std::int64_t>(right[component]);
    }
    return sum;
}

__attribute__((noinline)) float FullLvqL2(
    I8Kernel function,
    const ReplayData &data,
    std::size_t candidate,
    std::size_t dimension) {
    const std::int32_t dot =
        function(data.query.data(), data.Candidate(candidate), dimension);
    const LvqHeader candidate_header = LoadHeader(data, candidate);
    const float scale1 = data.query_header.scale;
    const float scale2 = candidate_header.scale;
    const float beta = data.query_header.bias - candidate_header.bias;
    return data.query_header.norm2sq_scalesq +
        candidate_header.norm2sq_scalesq +
        beta * beta * static_cast<float>(dimension) -
        2.0f * scale1 * scale2 * static_cast<float>(dot) +
        2.0f * beta * data.query_header.norm1_scale -
        2.0f * beta * candidate_header.norm1_scale;
}

void RequireInt32(std::int64_t value, std::size_t dimension, const char *scope) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error(
            std::string(scope) + " exact dot overflows int32 at dimension " +
            std::to_string(dimension));
    }
}

std::array<I8KernelDescriptor, 4> AllKernels(std::size_t dimension) {
    return {
        hnsw_kernel_replay::GetInfinityI8Kernel(I8KernelVariant::kSelected, dimension),
        hnsw_kernel_replay::GetInfinityI8Kernel(I8KernelVariant::kDirectBF, dimension),
        hnsw_kernel_replay::GetInfinityI8Kernel(I8KernelVariant::kDirectSIMDeSSE, dimension),
        hnsw_kernel_replay::GetInfinityI8Kernel(I8KernelVariant::kForcedScalar, dimension),
    };
}

std::vector<std::size_t> ValidationDimensions(std::size_t requested) {
    std::vector<std::size_t> dimensions{
        1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        127, 128, 129, 959, 960, 961, requested,
    };
    dimensions.erase(
        std::remove_if(
            dimensions.begin(),
            dimensions.end(),
            [](std::size_t dimension) { return dimension > kMaximumSafeDimension; }),
        dimensions.end());
    std::sort(dimensions.begin(), dimensions.end());
    dimensions.erase(std::unique(dimensions.begin(), dimensions.end()), dimensions.end());
    return dimensions;
}

void FillAdversarial(
    std::vector<std::int8_t> &left,
    std::vector<std::int8_t> &right,
    std::size_t pattern,
    std::uint64_t &state) {
    for (std::size_t index = 0; index < left.size(); ++index) {
        switch (pattern) {
            case 0:
                left[index] = -128;
                right[index] = -128;
                break;
            case 1:
                left[index] = -128;
                right[index] = 127;
                break;
            case 2: {
                constexpr std::array<std::int8_t, 4> left_values{-128, 127, -1, 1};
                constexpr std::array<std::int8_t, 4> right_values{127, -128, 1, -1};
                left[index] = left_values[index & 3U];
                right[index] = right_values[index & 3U];
                break;
            }
            case 3:
                left[index] = DeterministicI8(state);
                right[index] = DeterministicI8(state);
                break;
            default:
                std::abort();
        }
    }
}

std::uint64_t ValidateAdversarial(std::size_t requested_dimension) {
    std::uint64_t output_digest = kFnvOffset;
    std::size_t cases = 0;
    std::uint64_t state = kSeed ^ 0x93bb3ef6ee2bf145ULL;
    for (const std::size_t dimension : ValidationDimensions(requested_dimension)) {
        std::vector<std::int8_t> left(dimension);
        std::vector<std::int8_t> right(dimension);
        const auto kernels = AllKernels(dimension);
        for (std::size_t pattern = 0; pattern < 4; ++pattern) {
            FillAdversarial(left, right, pattern, state);
            const std::int64_t exact = ExactDot(left.data(), right.data(), dimension);
            RequireInt32(exact, dimension, "adversarial validation");
            const auto expected = static_cast<std::int32_t>(exact);
            for (const I8KernelDescriptor &kernel : kernels) {
                const std::int32_t actual =
                    kernel.function(left.data(), right.data(), dimension);
                if (actual != expected) {
                    throw std::runtime_error(
                        std::string("exact I8 validation failed for ") + kernel.id +
                        " at dimension " + std::to_string(dimension) +
                        ", pattern " + std::to_string(pattern) +
                        ": expected " + std::to_string(expected) +
                        ", actual " + std::to_string(actual));
                }
                HashBytes(output_digest, &actual, sizeof(actual));
            }
            ++cases;
        }
    }
    std::cout << "{\"record\":\"adversarial_validation\",\"dimensions\":"
              << ValidationDimensions(requested_dimension).size()
              << ",\"patterns_per_dimension\":4,\"cases\":" << cases
              << ",\"kernels_per_case\":4,\"exact_int64_oracle\":true"
              << ",\"int32_overflow_rejected\":true,\"output_digest_fnv1a64\":"
              << output_digest << ",\"passed\":true}\n";
    return output_digest;
}

std::uint64_t ValidateReplayData(const ReplayData &data, std::size_t dimension) {
    const auto kernels = AllKernels(dimension);
    const std::size_t samples =
        std::min(kValidationCandidateLimit, data.candidate_count);
    std::uint64_t output_digest = kFnvOffset;
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const std::size_t candidate = data.tape[sample];
        const std::int64_t exact =
            ExactDot(data.query.data(), data.Candidate(candidate), dimension);
        RequireInt32(exact, dimension, "replay validation");
        const auto expected = static_cast<std::int32_t>(exact);
        for (const I8KernelDescriptor &kernel : kernels) {
            const std::int32_t actual =
                kernel.function(data.query.data(), data.Candidate(candidate), dimension);
            if (actual != expected) {
                throw std::runtime_error(
                    std::string("replay I8 validation failed for ") + kernel.id);
            }
            HashBytes(output_digest, &actual, sizeof(actual));
        }
    }
    std::cout << "{\"record\":\"replay_validation\",\"samples\":" << samples
              << ",\"kernels_per_sample\":4,\"exact_int64_oracle\":true"
              << ",\"exact_int32_equality\":true,\"output_digest_fnv1a64\":"
              << output_digest << ",\"passed\":true}\n";
    return output_digest;
}

std::uint64_t ValidateFullLvqReplayData(
    const ReplayData &data,
    const I8KernelDescriptor &kernel,
    std::size_t dimension) {
    const std::size_t samples =
        std::min(kValidationCandidateLimit, data.candidate_count);
    std::uint64_t output_digest = kFnvOffset;
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const std::size_t candidate = data.tape[sample];
        const float actual =
            FullLvqL2(kernel.function, data, candidate, dimension);
        if (!std::isfinite(actual)) {
            throw std::runtime_error("full LVQ replay produced a non-finite distance");
        }
        HashBytes(output_digest, &actual, sizeof(actual));
    }
    std::cout << "{\"record\":\"full_lvq_validation\",\"samples\":" << samples
              << ",\"kernel\":\"" << kernel.id
              << "\",\"full_header_and_payload\":true"
              << ",\"finite_outputs\":true,\"output_digest_fnv1a64\":"
              << output_digest << ",\"passed\":true}\n";
    return output_digest;
}

void Evict(std::vector<std::byte> &buffer, std::size_t cache_line_bytes) {
    std::uint64_t sum = 0;
    for (std::size_t offset = 0; offset < buffer.size(); offset += cache_line_bytes) {
        sum += static_cast<unsigned char>(buffer[offset]);
    }
    g_eviction_sink = sum;
}

void Warm(
    const ReplayData &data,
    std::size_t cache_line_bytes) {
    std::int64_t sum = 0;
    const auto *query_header =
        reinterpret_cast<const unsigned char *>(&data.query_header);
    for (std::size_t offset = 0; offset < sizeof(data.query_header); ++offset) {
        sum += query_header[offset];
    }
    for (const std::int8_t value : data.query) {
        sum += value;
    }
    for (const std::uint32_t candidate : data.tape) {
        const std::byte *record = data.Record(candidate);
        for (std::size_t offset = 0; offset < data.useful_record_bytes;
             offset += cache_line_bytes) {
            sum += static_cast<unsigned char>(record[offset]);
        }
    }
    g_result_sink = sum;
}

void Condition(
    const ReplayData &data,
    Regime regime,
    std::size_t cache_line_bytes,
    std::vector<std::byte> &eviction) {
    Evict(eviction, cache_line_bytes);
    if (regime == Regime::kDram) {
        return;
    }
    const std::size_t prime_passes =
        regime == Regime::kHot || regime == Regime::kL1 ? 3 : 1;
    for (std::size_t pass = 0; pass < prime_passes; ++pass) {
        Warm(data, cache_line_bytes);
    }
}

double TimespecSeconds(const timespec &value) {
    return static_cast<double>(value.tv_sec) +
        static_cast<double>(value.tv_nsec) * 1e-9;
}

extern "C" __attribute__((noinline)) std::int32_t HnswKernelReplayNullI8(
    const std::int8_t *,
    const std::int8_t *,
    std::size_t) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return 0;
}

__attribute__((noinline)) std::int64_t RunKernelBody(
    I8Kernel function,
    const ReplayData &data,
    std::size_t dimension,
    std::size_t calls,
    std::size_t rotation) {
    std::array<std::int64_t, 8> checksums{};
    std::size_t tape_index = rotation % data.tape.size();
    for (std::size_t call = 0; call < calls; ++call) {
        const std::size_t candidate = data.tape[tape_index];
        checksums[call & (checksums.size() - 1)] +=
            function(data.query.data(), data.Candidate(candidate), dimension);
        if (++tape_index == data.tape.size()) {
            tape_index = 0;
        }
    }
    return std::accumulate(checksums.begin(), checksums.end(), std::int64_t{0});
}

Timing RunKernel(
    I8Kernel function,
    const ReplayData &data,
    std::size_t dimension,
    std::size_t calls,
    std::size_t rotation) {
    std::size_t cpu_start = std::numeric_limits<std::size_t>::max();
    std::size_t cpu_end_number = std::numeric_limits<std::size_t>::max();
    const int cpu_start_status = pthread_cpu_number_np(&cpu_start);
    rusage usage_begin{};
    rusage usage_end{};
    if (getrusage(RUSAGE_SELF, &usage_begin) != 0) {
        throw std::runtime_error("getrusage begin failed");
    }
    timespec cpu_begin{};
    timespec cpu_end{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_begin) != 0) {
        throw std::runtime_error("clock_gettime begin failed");
    }
    const auto begin = std::chrono::steady_clock::now();
    const std::int64_t checksum =
        RunKernelBody(function, data, dimension, calls, rotation);
    const auto end = std::chrono::steady_clock::now();
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
        throw std::runtime_error("clock_gettime end failed");
    }
    if (getrusage(RUSAGE_SELF, &usage_end) != 0) {
        throw std::runtime_error("getrusage end failed");
    }
    const int cpu_end_status = pthread_cpu_number_np(&cpu_end_number);
    g_result_sink = checksum;
    return {
        .seconds = std::chrono::duration<double>(end - begin).count(),
        .thread_cpu_seconds =
            TimespecSeconds(cpu_end) - TimespecSeconds(cpu_begin),
        .checksum = checksum,
        .cpu_start = cpu_start,
        .cpu_end = cpu_end_number,
        .cpu_start_status = cpu_start_status,
        .cpu_end_status = cpu_end_status,
        .minor_page_faults = usage_end.ru_minflt - usage_begin.ru_minflt,
        .major_page_faults = usage_end.ru_majflt - usage_begin.ru_majflt,
    };
}

__attribute__((noinline)) std::int64_t RunFullLvqBody(
    I8Kernel function,
    const ReplayData &data,
    std::size_t dimension,
    std::size_t calls,
    std::size_t rotation) {
    std::array<double, 8> checksums{};
    std::size_t tape_index = rotation % data.tape.size();
    for (std::size_t call = 0; call < calls; ++call) {
        const std::size_t candidate = data.tape[tape_index];
        checksums[call & (checksums.size() - 1)] +=
            FullLvqL2(function, data, candidate, dimension);
        if (++tape_index == data.tape.size()) {
            tape_index = 0;
        }
    }
    return static_cast<std::int64_t>(
        std::accumulate(checksums.begin(), checksums.end(), 0.0));
}

Timing RunFullLvq(
    I8Kernel function,
    const ReplayData &data,
    std::size_t dimension,
    std::size_t calls,
    std::size_t rotation) {
    std::size_t cpu_start = std::numeric_limits<std::size_t>::max();
    std::size_t cpu_end_number = std::numeric_limits<std::size_t>::max();
    const int cpu_start_status = pthread_cpu_number_np(&cpu_start);
    rusage usage_begin{};
    rusage usage_end{};
    if (getrusage(RUSAGE_SELF, &usage_begin) != 0) {
        throw std::runtime_error("getrusage begin failed");
    }
    timespec cpu_begin{};
    timespec cpu_end{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_begin) != 0) {
        throw std::runtime_error("clock_gettime begin failed");
    }
    const auto begin = std::chrono::steady_clock::now();
    const std::int64_t checksum =
        RunFullLvqBody(function, data, dimension, calls, rotation);
    const auto end = std::chrono::steady_clock::now();
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
        throw std::runtime_error("clock_gettime end failed");
    }
    if (getrusage(RUSAGE_SELF, &usage_end) != 0) {
        throw std::runtime_error("getrusage end failed");
    }
    const int cpu_end_status = pthread_cpu_number_np(&cpu_end_number);
    g_result_sink = checksum;
    return {
        .seconds = std::chrono::duration<double>(end - begin).count(),
        .thread_cpu_seconds =
            TimespecSeconds(cpu_end) - TimespecSeconds(cpu_begin),
        .checksum = checksum,
        .cpu_start = cpu_start,
        .cpu_end = cpu_end_number,
        .cpu_start_status = cpu_start_status,
        .cpu_end_status = cpu_end_status,
        .minor_page_faults = usage_end.ru_minflt - usage_begin.ru_minflt,
        .major_page_faults = usage_end.ru_majflt - usage_begin.ru_majflt,
    };
}

void RequireEligibleTiming(const Timing &timing, const char *scope) {
    if (!timing.OnCpuGatePassed() || !timing.PageFaultGatePassed()) {
        throw std::runtime_error(
            std::string(scope) + " failed the on-CPU or page-fault gate");
    }
}

bool TimingEligible(const Timing &timing) {
    return timing.OnCpuGatePassed() && timing.PageFaultGatePassed();
}

double Mean(const std::vector<double> &values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
}

std::pair<double, double> StratifiedBootstrap95(
    const std::vector<double> &ab,
    const std::vector<double> &ba) {
    std::vector<double> estimates;
    estimates.reserve(kBootstrapReplicates);
    std::uint64_t state = kSeed ^ 0x44a7a2458b3b2e31ULL;
    for (std::size_t replicate = 0; replicate < kBootstrapReplicates; ++replicate) {
        double ab_sum = 0;
        double ba_sum = 0;
        for (std::size_t index = 0; index < ab.size(); ++index) {
            ab_sum += ab[SplitMix64(state) % ab.size()];
        }
        for (std::size_t index = 0; index < ba.size(); ++index) {
            ba_sum += ba[SplitMix64(state) % ba.size()];
        }
        estimates.push_back(
            0.5 * (ab_sum / static_cast<double>(ab.size()) +
                   ba_sum / static_cast<double>(ba.size())));
    }
    std::sort(estimates.begin(), estimates.end());
    const std::size_t low =
        static_cast<std::size_t>(std::floor(0.025 * (estimates.size() - 1)));
    const std::size_t high =
        static_cast<std::size_t>(std::ceil(0.975 * (estimates.size() - 1)));
    return {estimates[low], estimates[high]};
}

std::size_t DefaultCalls(const Options &options, const ReplayData &data) {
    if (options.calls != 0) {
        return options.calls;
    }
    if (options.regime == Regime::kDram) {
        return std::max<std::size_t>(data.candidate_count, 1ULL << 18);
    }
    return 1ULL << 20;
}

std::size_t RoundUpToMultiple(
    std::size_t value,
    std::size_t multiple,
    const char *name) {
    if (multiple == 0) {
        throw std::invalid_argument(std::string(name) + " multiple is zero");
    }
    const std::size_t remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }
    if (value > std::numeric_limits<std::size_t>::max() - (multiple - remainder)) {
        throw std::overflow_error(std::string(name) + " overflow");
    }
    return value + multiple - remainder;
}

std::size_t CalibrateCalls(
    const Options &options,
    const ReplayData &data,
    const I8KernelDescriptor &left,
    const I8KernelDescriptor &right,
    std::vector<std::byte> &eviction,
    std::size_t calls,
    std::size_t cache_line_bytes) {
    for (;;) {
        Condition(data, options.regime, cache_line_bytes, eviction);
        const Timing left_timing =
            RunKernel(left.function, data, options.dimension, calls, 0);
        Condition(data, options.regime, cache_line_bytes, eviction);
        const Timing right_timing =
            RunKernel(right.function, data, options.dimension, calls, 0);
        const double minimum_seconds =
            std::min(left_timing.seconds, right_timing.seconds);
        std::cout << "{\"record\":\"calibration\",\"calls\":" << calls
                  << ",\"left_seconds\":" << left_timing.seconds
                  << ",\"right_seconds\":" << right_timing.seconds
                  << ",\"minimum_arm_seconds_gate\":" << options.minimum_arm_seconds
                  << ",\"passed\":"
                  << (minimum_seconds >= options.minimum_arm_seconds ? "true" : "false")
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
    const I8KernelDescriptor &left,
    const I8KernelDescriptor &right,
    std::vector<std::byte> &eviction,
    std::size_t calls,
    std::size_t cache_line_bytes) {
    calls = CalibrateCalls(
        options, data, left, right, eviction, calls, cache_line_bytes);
    const std::size_t warmup_calls =
        std::min(calls, std::size_t{1} << 20);
    Condition(data, options.regime, cache_line_bytes, eviction);
    RunKernel(left.function, data, options.dimension, warmup_calls, 0);
    Condition(data, options.regime, cache_line_bytes, eviction);
    RunKernel(right.function, data, options.dimension, warmup_calls, 0);

    Condition(data, options.regime, cache_line_bytes, eviction);
    const Timing floor =
        RunKernel(&HnswKernelReplayNullI8, data, options.dimension, calls, 0);
    std::cout << "{\"record\":\"harness_floor\",\"calls\":" << calls
              << ",\"seconds\":" << floor.seconds
              << ",\"nanoseconds_per_call\":"
              << floor.seconds * 1e9 / static_cast<double>(calls)
              << "}\n";

    std::vector<double> ab;
    std::vector<double> ba;
    for (std::size_t block = 0; block < options.rounds; ++block) {
        const bool order_ab = block % 2 == 0;
        const std::array<const I8KernelDescriptor *, 2> order = order_ab
            ? std::array<const I8KernelDescriptor *, 2>{&left, &right}
            : std::array<const I8KernelDescriptor *, 2>{&right, &left};
        std::array<Timing, 2> timings;
        for (std::size_t arm = 0; arm < order.size(); ++arm) {
            Condition(data, options.regime, cache_line_bytes, eviction);
            timings[arm] = RunKernel(
                order[arm]->function,
                data,
                options.dimension,
                calls,
                block * 7919);
            if (timings[arm].seconds < options.minimum_arm_seconds) {
                throw std::runtime_error("timed arm fell below minimum duration");
            }
            RequireEligibleTiming(timings[arm], "kernel-paired timed arm");
            std::cout << "{\"record\":\"timing\",\"block\":" << block
                      << ",\"order\":\"" << (order_ab ? "AB" : "BA")
                      << "\",\"kernel\":\"" << order[arm]->id
                      << "\",\"calls\":" << calls
                      << ",\"components\":" << calls * options.dimension
                      << ",\"seconds\":" << timings[arm].seconds
                      << ",\"distances_per_second\":"
                      << static_cast<double>(calls) / timings[arm].seconds
                      << ",\"nanoseconds_per_component\":"
                      << timings[arm].seconds * 1e9 /
                          static_cast<double>(calls * options.dimension)
                      << ",\"thread_cpu_over_wall\":"
                      << timings[arm].ThreadCpuOverWall()
                      << ",\"cpu_start\":" << timings[arm].cpu_start
                      << ",\"cpu_end\":" << timings[arm].cpu_end
                      << ",\"endpoint_cpu_stable\":"
                      << (timings[arm].EndpointCpuStable() ? "true" : "false")
                      << ",\"minor_page_faults\":"
                      << timings[arm].minor_page_faults
                      << ",\"major_page_faults\":"
                      << timings[arm].major_page_faults
                      << ",\"checksum\":" << timings[arm].checksum << "}\n";
        }
        const double left_seconds =
            order_ab ? timings[0].seconds : timings[1].seconds;
        const double right_seconds =
            order_ab ? timings[1].seconds : timings[0].seconds;
        const double log_rate_ratio = std::log(right_seconds / left_seconds);
        (order_ab ? ab : ba).push_back(log_rate_ratio);
        std::cout << "{\"record\":\"pair\",\"block\":" << block
                  << ",\"order\":\"" << (order_ab ? "AB" : "BA")
                  << "\",\"left_over_right_rate_ratio\":"
                  << std::exp(log_rate_ratio) << "}\n";
    }

    const double theta = 0.5 * (Mean(ab) + Mean(ba));
    const auto [low, high] = StratifiedBootstrap95(ab, ba);
    std::cout << "{\"record\":\"summary\",\"status\":\"VALIDATED_DIAGNOSTIC\""
              << ",\"estimand\":\"stratified_paired_log_rate_ratio\""
              << ",\"left\":\"" << left.id << "\",\"right\":\"" << right.id
              << "\",\"ab_blocks\":" << ab.size() << ",\"ba_blocks\":" << ba.size()
              << ",\"left_over_right_rate_ratio\":" << std::exp(theta)
              << ",\"stratified_bootstrap_replicates\":" << kBootstrapReplicates
              << ",\"stratified_bootstrap_95_low\":" << std::exp(low)
              << ",\"stratified_bootstrap_95_high\":" << std::exp(high)
              << ",\"claim_scope\":\"synthetic_exact_operand_whole_kernel_only\"}\n";
    return 0;
}

std::size_t CalibrateLayoutCalls(
    const Options &options,
    const ReplayData &left,
    const ReplayData &right,
    const I8KernelDescriptor &kernel,
    std::vector<std::byte> &eviction,
    std::size_t calls,
    std::size_t cache_line_bytes) {
    for (;;) {
        Condition(left, options.regime, cache_line_bytes, eviction);
        const Timing left_timing =
            RunFullLvq(kernel.function, left, options.dimension, calls, 0);
        Condition(right, options.regime, cache_line_bytes, eviction);
        const Timing right_timing =
            RunFullLvq(kernel.function, right, options.dimension, calls, 0);
        const double minimum_seconds =
            std::min(left_timing.seconds, right_timing.seconds);
        std::cout << "{\"record\":\"layout_calibration\",\"calls\":" << calls
                  << ",\"left_seconds\":" << left_timing.seconds
                  << ",\"right_seconds\":" << right_timing.seconds
                  << ",\"minimum_arm_seconds_gate\":" << options.minimum_arm_seconds
                  << ",\"passed\":"
                  << (minimum_seconds >= options.minimum_arm_seconds ? "true" : "false")
                  << "}\n";
        if (minimum_seconds >= options.minimum_arm_seconds) {
            return calls;
        }
        if (calls > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error("layout call calibration overflow");
        }
        calls *= 2;
    }
}

int RunLayoutPaired(
    const Options &options,
    const ReplayData &left,
    const ReplayData &right,
    const I8KernelDescriptor &kernel,
    std::vector<std::byte> &eviction,
    std::size_t calls,
    std::size_t cache_line_bytes) {
    calls = CalibrateLayoutCalls(
        options,
        left,
        right,
        kernel,
        eviction,
        calls,
        cache_line_bytes);
    const std::size_t warmup_calls =
        std::min(calls, std::size_t{1} << 20);
    Condition(left, options.regime, cache_line_bytes, eviction);
    RunFullLvq(kernel.function, left, options.dimension, warmup_calls, 0);
    Condition(right, options.regime, cache_line_bytes, eviction);
    RunFullLvq(kernel.function, right, options.dimension, warmup_calls, 0);

    std::vector<double> ab;
    std::vector<double> ba;
    const std::size_t target_per_order = options.rounds / 2;
    const std::size_t maximum_invalid_blocks =
        std::max<std::size_t>(1, options.rounds / 10);
    std::size_t invalid_blocks = 0;
    std::size_t endpoint_cpu_changed_arms = 0;
    std::size_t attempt = 0;
    while (ab.size() < target_per_order || ba.size() < target_per_order) {
        if (invalid_blocks > maximum_invalid_blocks) {
            throw std::runtime_error(
                "layout-paired invalid-block fraction exceeded 10%");
        }
        const bool order_ab = ab.size() >= target_per_order
            ? false
            : ba.size() >= target_per_order
                ? true
                : attempt % 2 == 0;
        const std::array<const ReplayData *, 2> order = order_ab
            ? std::array<const ReplayData *, 2>{&left, &right}
            : std::array<const ReplayData *, 2>{&right, &left};
        const std::array<const char *, 2> role = order_ab
            ? std::array<const char *, 2>{"left", "right"}
            : std::array<const char *, 2>{"right", "left"};
        const std::array<Layout, 2> layout = order_ab
            ? std::array<Layout, 2>{options.left_layout, options.right_layout}
            : std::array<Layout, 2>{options.right_layout, options.left_layout};
        std::array<Timing, 2> timings;
        for (std::size_t arm = 0; arm < order.size(); ++arm) {
            Condition(*order[arm], options.regime, cache_line_bytes, eviction);
            timings[arm] = RunFullLvq(
                kernel.function,
                *order[arm],
                options.dimension,
                calls,
                attempt * 7919);
            if (timings[arm].seconds < options.minimum_arm_seconds) {
                throw std::runtime_error(
                    "layout-paired timed arm fell below minimum duration");
            }
            if (!timings[arm].EndpointCpuStable()) {
                ++endpoint_cpu_changed_arms;
            }
            std::cout << "{\"record\":\"layout_timing\",\"attempt\":" << attempt
                      << ",\"order\":\"" << (order_ab ? "AB" : "BA")
                      << "\",\"role\":\"" << role[arm]
                      << "\",\"layout\":\"" << LayoutName(layout[arm])
                      << "\",\"kernel\":\"" << kernel.id
                      << "\",\"calls\":" << calls
                      << ",\"components\":" << calls * options.dimension
                      << ",\"seconds\":" << timings[arm].seconds
                      << ",\"distances_per_second\":"
                      << static_cast<double>(calls) / timings[arm].seconds
                      << ",\"nanoseconds_per_distance\":"
                      << timings[arm].seconds * 1e9 / static_cast<double>(calls)
                      << ",\"thread_cpu_over_wall\":"
                      << timings[arm].ThreadCpuOverWall()
                      << ",\"cpu_start\":" << timings[arm].cpu_start
                      << ",\"cpu_end\":" << timings[arm].cpu_end
                      << ",\"endpoint_cpu_stable\":"
                      << (timings[arm].EndpointCpuStable() ? "true" : "false")
                      << ",\"on_cpu_gate_passed\":"
                      << (timings[arm].OnCpuGatePassed() ? "true" : "false")
                      << ",\"minor_page_faults\":"
                      << timings[arm].minor_page_faults
                      << ",\"major_page_faults\":"
                      << timings[arm].major_page_faults
                      << ",\"page_fault_gate_passed\":"
                      << (timings[arm].PageFaultGatePassed() ? "true" : "false")
                      << ",\"eligible\":"
                      << (TimingEligible(timings[arm]) ? "true" : "false")
                      << ",\"checksum\":" << timings[arm].checksum << "}\n";
        }
        if (timings[0].checksum != timings[1].checksum) {
            throw std::runtime_error(
                "layout arms produced different full-LVQ checksums");
        }
        if (!TimingEligible(timings[0]) || !TimingEligible(timings[1])) {
            ++invalid_blocks;
            std::cout << "{\"record\":\"layout_invalid_block\",\"attempt\":"
                      << attempt << ",\"order\":\""
                      << (order_ab ? "AB" : "BA")
                      << "\",\"reason\":\"on_cpu_or_page_fault_gate\""
                      << ",\"invalid_blocks\":" << invalid_blocks
                      << ",\"maximum_invalid_blocks\":"
                      << maximum_invalid_blocks << "}\n";
            ++attempt;
            continue;
        }
        const double left_seconds =
            order_ab ? timings[0].seconds : timings[1].seconds;
        const double right_seconds =
            order_ab ? timings[1].seconds : timings[0].seconds;
        const double log_rate_ratio = std::log(right_seconds / left_seconds);
        (order_ab ? ab : ba).push_back(log_rate_ratio);
        std::cout << "{\"record\":\"layout_pair\",\"attempt\":" << attempt
                  << ",\"valid_block\":" << (ab.size() + ba.size() - 1)
                  << ",\"order\":\"" << (order_ab ? "AB" : "BA")
                  << "\",\"left_over_right_rate_ratio\":"
                  << std::exp(log_rate_ratio)
                  << ",\"output_checksum_equal\":true}\n";
        ++attempt;
    }

    const double theta = 0.5 * (Mean(ab) + Mean(ba));
    const auto [low, high] = StratifiedBootstrap95(ab, ba);
    std::cout << "{\"record\":\"layout_summary\""
              << ",\"status\":\"VALIDATED_DIAGNOSTIC\""
              << ",\"estimand\":\"stratified_paired_log_rate_ratio\""
              << ",\"left_layout\":\"" << LayoutName(options.left_layout)
              << "\",\"right_layout\":\"" << LayoutName(options.right_layout)
              << "\",\"shared_kernel\":\"" << kernel.id
              << "\",\"ab_blocks\":" << ab.size()
              << ",\"ba_blocks\":" << ba.size()
              << ",\"invalid_blocks\":" << invalid_blocks
              << ",\"attempted_blocks\":" << attempt
              << ",\"endpoint_cpu_changed_arms\":"
              << endpoint_cpu_changed_arms
              << ",\"invalid_block_fraction\":"
              << static_cast<double>(invalid_blocks) /
                     static_cast<double>(attempt)
              << ",\"left_over_right_rate_ratio\":" << std::exp(theta)
              << ",\"stratified_bootstrap_replicates\":" << kBootstrapReplicates
              << ",\"stratified_bootstrap_95_low\":" << std::exp(low)
              << ",\"stratified_bootstrap_95_high\":" << std::exp(high)
              << ",\"claim_scope\":\"synthetic_full_lvq_stride_footprint_treatment_only\"}\n";
    return 0;
}

RuntimePolicy ConfigureRuntimePolicy() {
    RuntimePolicy policy;
    policy.set_qos_status =
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    policy.get_qos_status = pthread_get_qos_class_np(
        pthread_self(), &policy.qos_class, &policy.relative_priority);
    return policy;
}

void PrintConfiguration(
    const Options &options,
    const ReplayData &data,
    const HardwareGeometry &geometry,
    const RuntimePolicy &policy,
    std::size_t calls,
    const char *role) {
    const StorageCoverage coverage = MeasureCoverage(data, geometry);
    std::cout << "{\"record\":\"configuration\",\"schema\":2"
              << ",\"role\":\"" << role << "\""
              << ",\"architecture\":\"arm64-apple\",\"data_type\":\"signed-int8\""
              << ",\"metric\":\"lvq_l2\",\"dimension\":" << options.dimension
              << ",\"maximum_overflow_safe_dimension\":" << kMaximumSafeDimension
              << ",\"regime\":\"" << RegimeName(options.regime)
              << "\",\"layout\":\"" << LayoutName(options.layout)
              << "\",\"lvq_header_bytes\":" << kLvqHeaderBytes
              << ",\"useful_record_bytes\":" << data.useful_record_bytes
              << ",\"stride_bytes\":" << data.stride_bytes
              << ",\"candidate_count\":" << data.candidate_count
              << ",\"allocation_bytes\":" << data.allocation_bytes
              << ",\"candidate_useful_bytes\":" << coverage.useful_bytes
              << ",\"candidate_padding_bytes\":" << coverage.padding_bytes
              << ",\"candidate_useful_cache_lines\":" << coverage.cache_lines
              << ",\"candidate_useful_pages\":" << coverage.pages
              << ",\"operand_digest_fnv1a64\":" << data.operand_digest
              << ",\"layout_digest_fnv1a64\":" << data.layout_digest
              << ",\"calls\":" << calls << ",\"rounds\":" << options.rounds
              << ",\"eviction_bytes\":" << options.eviction_bytes
              << ",\"minimum_arm_seconds\":" << options.minimum_arm_seconds
              << ",\"page_bytes\":" << geometry.page_bytes
              << ",\"cache_line_bytes\":" << geometry.cache_line_bytes
              << ",\"performance_l1d_bytes\":" << geometry.performance_l1d_bytes
              << ",\"performance_l2_bytes\":" << geometry.performance_l2_bytes
              << ",\"physical_memory_bytes\":" << geometry.memory_bytes
              << ",\"set_qos_status\":" << policy.set_qos_status
              << ",\"get_qos_status\":" << policy.get_qos_status
              << ",\"qos_class\":" << static_cast<unsigned int>(policy.qos_class)
              << ",\"qos_relative_priority\":" << policy.relative_priority
              << ",\"endpoint_cpu_check\":\"record_only\""
              << ",\"in_arm_migration_observation\":false"
              << ",\"thread_cpu_over_wall_min\":" << kMinimumThreadCpuOverWall
              << ",\"thread_cpu_over_wall_max\":" << kMaximumThreadCpuOverWall
              << ",\"zero_timed_page_faults_required\":true"
              << ",\"cache_state_claim\":\"geometry_only_unverified_by_pmu\""
              << ",\"claim_scope\":\"synthetic_exact_operand_diagnostic_only\"}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
#if !defined(__APPLE__) || !defined(__aarch64__)
        std::cerr << "hnsw_kernel_replay_i8 requires native arm64 macOS\n";
        return 64;
#else
        std::cout << std::setprecision(17);
        Options options = ParseOptions(argc, argv);
        const RuntimePolicy policy = ConfigureRuntimePolicy();
        if (policy.set_qos_status != 0 || policy.get_qos_status != 0) {
            throw std::runtime_error("pthread QoS configuration failed");
        }
        const HardwareGeometry geometry = ReadHardwareGeometry();
        options.eviction_bytes = std::max<std::size_t>(
            options.eviction_bytes,
            CheckedMultiply(
                geometry.performance_l2_bytes, 2, "minimum eviction bytes"));
        std::vector<std::byte> eviction(options.eviction_bytes);
        for (std::size_t offset = 0; offset < eviction.size();
             offset += geometry.cache_line_bytes) {
            eviction[offset] =
                static_cast<std::byte>(
                    (offset / geometry.cache_line_bytes) & 0xffU);
        }

        bool inspection_passed = true;
        for (const I8KernelDescriptor &kernel : AllKernels(options.dimension)) {
            inspection_passed &= PrintKernelRecord("validation", kernel);
        }
        if (!inspection_passed) {
            std::cout << "{\"record\":\"final\",\"status\":\"KERNEL_INSPECTION_FAILED\"}\n";
            return 2;
        }

        ValidateAdversarial(options.dimension);
        if (options.mode == Mode::kLayoutPaired) {
            Options left_options = options;
            left_options.layout = options.left_layout;
            Options right_options = options;
            right_options.layout = options.right_layout;
            const ReplayData left_data = MakeReplayData(left_options, geometry);
            const ReplayData right_data = MakeReplayData(right_options, geometry);
            RequireEquivalentOperands(
                left_data, right_data, options.dimension);
            RequirePaddingCanary(left_data, "before_timing_left");
            RequirePaddingCanary(right_data, "before_timing_right");
            const std::size_t calls = RoundUpToMultiple(
                DefaultCalls(options, left_data),
                left_data.candidate_count,
                "layout calls");
            if (calls == 0) {
                throw std::invalid_argument("calls must be positive");
            }
            PrintConfiguration(
                left_options, left_data, geometry, policy, calls, "left");
            PrintConfiguration(
                right_options, right_data, geometry, policy, calls, "right");
            std::cout << "{\"record\":\"layout_operand_equivalence\""
                      << ",\"candidate_count\":" << left_data.candidate_count
                      << ",\"useful_record_bytes\":"
                      << left_data.useful_record_bytes
                      << ",\"operand_digest_fnv1a64\":"
                      << left_data.operand_digest
                      << ",\"query_header_equal\":true"
                      << ",\"query_codes_equal\":true"
                      << ",\"candidate_headers_equal\":true"
                      << ",\"candidate_codes_equal\":true"
                      << ",\"tape_equal\":true"
                      << ",\"padding_unread_by_full_lvq_kernel\":true"
                      << ",\"passed\":true}\n";

            const I8KernelDescriptor kernel =
                hnsw_kernel_replay::GetInfinityI8Kernel(
                    options.layout_kernel, options.dimension);
            inspection_passed = PrintKernelRecord("layout-shared", kernel);
            if (!inspection_passed) {
                std::cout
                    << "{\"record\":\"final\",\"status\":\"KERNEL_INSPECTION_FAILED\"}\n";
                return 2;
            }
            const std::uint64_t left_dot_digest =
                ValidateReplayData(left_data, options.dimension);
            const std::uint64_t right_dot_digest =
                ValidateReplayData(right_data, options.dimension);
            const std::uint64_t left_full_digest =
                ValidateFullLvqReplayData(
                    left_data, kernel, options.dimension);
            const std::uint64_t right_full_digest =
                ValidateFullLvqReplayData(
                    right_data, kernel, options.dimension);
            if (left_dot_digest != right_dot_digest ||
                left_full_digest != right_full_digest) {
                throw std::runtime_error(
                    "layout validation output digests differ");
            }
            const int status = RunLayoutPaired(
                options,
                left_data,
                right_data,
                kernel,
                eviction,
                calls,
                static_cast<std::size_t>(geometry.cache_line_bytes));
            RequirePaddingCanary(left_data, "after_timing_left");
            RequirePaddingCanary(right_data, "after_timing_right");
            std::cout << "{\"record\":\"final\",\"status\":\""
                      << (status == 0
                              ? "VALIDATED_LAYOUT_DIAGNOSTIC_COMPLETE"
                              : "FAIL")
                      << "\"}\n";
            return status;
        }

        const ReplayData data = MakeReplayData(options, geometry);
        const std::size_t calls = DefaultCalls(options, data);
        if (calls == 0) {
            throw std::invalid_argument("calls must be positive");
        }
        PrintConfiguration(
            options, data, geometry, policy, calls, "single");
        const I8KernelDescriptor left =
            hnsw_kernel_replay::GetInfinityI8Kernel(
                options.left, options.dimension);
        const I8KernelDescriptor right =
            hnsw_kernel_replay::GetInfinityI8Kernel(
                options.right, options.dimension);
        inspection_passed = PrintKernelRecord("left", left) &&
            PrintKernelRecord("right", right);
        if (!inspection_passed) {
            std::cout
                << "{\"record\":\"final\",\"status\":\"KERNEL_INSPECTION_FAILED\"}\n";
            return 2;
        }
        ValidateReplayData(data, options.dimension);
        ValidateFullLvqReplayData(data, left, options.dimension);
        if (options.mode == Mode::kValidate) {
            std::cout
                << "{\"record\":\"final\",\"status\":\"VALIDATED_EXACT_INT32_AND_FULL_LVQ_TREATMENT\"}\n";
            return 0;
        }
        const int status = RunPaired(
            options,
            data,
            left,
            right,
            eviction,
            calls,
            static_cast<std::size_t>(geometry.cache_line_bytes));
        std::cout << "{\"record\":\"final\",\"status\":\""
                  << (status == 0 ? "VALIDATED_DIAGNOSTIC_COMPLETE" : "FAIL")
                  << "\"}\n";
        return status;
#endif
    } catch (const std::exception &error) {
        std::cerr << "hnsw_kernel_replay_i8: " << error.what() << '\n';
        return 64;
    }
}
