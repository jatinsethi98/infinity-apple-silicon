#if defined(__APPLE__)
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

import std;
import std.compat;
import infinity_core;

#if defined(__APPLE__) && defined(__aarch64__)
namespace {

constexpr std::size_t kLaneCount = 4;
constexpr std::size_t kMaximumDimension = 1025;
constexpr std::size_t kAlignmentCount = 4;
constexpr std::size_t kStorageSize = kMaximumDimension + kAlignmentCount;
constexpr std::size_t kRandomCaseCount = 384;
constexpr std::uint8_t kLaneMask = 0x0F;

static_assert(sizeof(float) == sizeof(std::uint32_t));
static_assert(std::numeric_limits<float>::is_iec559);

using ExactBatch4Kernel = void (*)(const float *, const float *, const float *, const float *, const float *, std::size_t, float *);
using ThresholdedBatch4Kernel =
    std::uint8_t (*)(const float *, const float *, const float *, const float *, const float *, std::size_t, float, float *);

static_assert(std::is_same_v<decltype(&infinity::F32L2SSEBatch4), ExactBatch4Kernel>);
static_assert(std::is_same_v<decltype(&infinity::F32L2SSEResidualBatch4), ExactBatch4Kernel>);
static_assert(std::is_same_v<decltype(&infinity::F32L2SSEBatch4WithinThreshold), ThresholdedBatch4Kernel>);
static_assert(std::is_same_v<decltype(&infinity::F32L2SSEResidualBatch4WithinThreshold), ThresholdedBatch4Kernel>);

struct KernelPair {
    std::string_view name;
    ExactBatch4Kernel exact;
    ThresholdedBatch4Kernel thresholded;
};

constexpr KernelPair kAlignedKernel{
    .name = "F32L2SSEBatch4WithinThreshold",
    .exact = &infinity::F32L2SSEBatch4,
    .thresholded = &infinity::F32L2SSEBatch4WithinThreshold,
};

constexpr KernelPair kResidualKernel{
    .name = "F32L2SSEResidualBatch4WithinThreshold",
    .exact = &infinity::F32L2SSEResidualBatch4,
    .thresholded = &infinity::F32L2SSEResidualBatch4WithinThreshold,
};

struct Inputs {
    const float *query;
    std::array<const float *, kLaneCount> candidates;
};

struct Statistics {
    std::uint64_t exact_calls = 0;
    std::uint64_t thresholded_calls = 0;
    std::uint64_t lane_checks = 0;
    std::uint64_t exact_lanes = 0;
    std::uint64_t rejected_lanes = 0;
    std::uint64_t qualifying_lanes = 0;
    std::uint64_t boundary_thresholds = 0;
    std::uint64_t randomized_cases = 0;
    std::uint64_t guard_page_cases = 0;
};

struct alignas(64) FloatStorage {
    std::array<float, kStorageSize> values{};
};

struct DataSet {
    FloatStorage query;
    std::array<FloatStorage, kLaneCount> candidates;
};

[[noreturn]] void Fail(std::string_view suite, const KernelPair &kernel, std::size_t case_id, std::size_t dimension, std::string_view detail) {
    std::ostringstream stream;
    stream << suite << ": kernel=" << kernel.name << " case=" << case_id << " dimension=" << dimension << ": " << detail;
    throw std::runtime_error(stream.str());
}

std::uint32_t FloatBits(float value) { return std::bit_cast<std::uint32_t>(value); }

std::string HexBits(float value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setw(8) << std::setfill('0') << FloatBits(value);
    return stream.str();
}

std::array<float, kLaneCount> ComputeExact(const KernelPair &kernel, const Inputs &inputs, std::size_t dimension, Statistics &statistics) {
    std::array<float, kLaneCount> exact{};
    kernel.exact(inputs.query, inputs.candidates[0], inputs.candidates[1], inputs.candidates[2], inputs.candidates[3], dimension, exact.data());
    ++statistics.exact_calls;
    return exact;
}

std::uint8_t CheckThreshold(const KernelPair &kernel,
                            const Inputs &inputs,
                            std::size_t dimension,
                            float threshold,
                            const std::array<float, kLaneCount> &exact,
                            std::string_view suite,
                            std::size_t case_id,
                            Statistics &statistics) {
    std::array<float, kLaneCount> actual{
        std::bit_cast<float>(std::uint32_t{0xDEADBEEF}),
        std::bit_cast<float>(std::uint32_t{0xDFADBEEF}),
        std::bit_cast<float>(std::uint32_t{0xE0ADBEEF}),
        std::bit_cast<float>(std::uint32_t{0xE1ADBEEF}),
    };
    const std::uint8_t exact_mask = kernel.thresholded(inputs.query,
                                                       inputs.candidates[0],
                                                       inputs.candidates[1],
                                                       inputs.candidates[2],
                                                       inputs.candidates[3],
                                                       dimension,
                                                       threshold,
                                                       actual.data());
    ++statistics.thresholded_calls;

    if ((exact_mask & static_cast<std::uint8_t>(~kLaneMask)) != 0) {
        std::ostringstream detail;
        detail << "returned bits outside the four-lane mask: mask=0x" << std::hex << static_cast<unsigned>(exact_mask);
        Fail(suite, kernel, case_id, dimension, detail.str());
    }

    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        const std::uint8_t bit = static_cast<std::uint8_t>(std::uint8_t{1} << lane);
        const bool is_exact = (exact_mask & bit) != 0;
        const bool qualifies = exact[lane] <= threshold;
        ++statistics.lane_checks;
        statistics.exact_lanes += is_exact;
        statistics.rejected_lanes += !is_exact;
        statistics.qualifying_lanes += qualifies;

        if (is_exact && FloatBits(actual[lane]) != FloatBits(exact[lane])) {
            std::ostringstream detail;
            detail << "lane " << lane << " exact result differs: threshold=" << HexBits(threshold) << " expected=" << HexBits(exact[lane])
                   << " actual=" << HexBits(actual[lane]) << " mask=0x" << std::hex << static_cast<unsigned>(exact_mask);
            Fail(suite, kernel, case_id, dimension, detail.str());
        }
        if (!is_exact && qualifies) {
            std::ostringstream detail;
            detail << "lane " << lane << " rejected despite exact D<=threshold: threshold=" << HexBits(threshold) << " exact=" << HexBits(exact[lane])
                   << " mask=0x" << std::hex << static_cast<unsigned>(exact_mask);
            Fail(suite, kernel, case_id, dimension, detail.str());
        }
    }
    return exact_mask;
}

void CheckBoundaryThresholds(const KernelPair &kernel,
                             const Inputs &inputs,
                             std::size_t dimension,
                             std::string_view suite,
                             std::size_t case_id,
                             Statistics &statistics) {
    const std::array<float, kLaneCount> exact = ComputeExact(kernel, inputs, dimension, statistics);
    constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
    constexpr float kPositiveInfinity = std::numeric_limits<float>::infinity();

    for (float distance : exact) {
        const std::array<float, 3> thresholds{
            std::nextafter(distance, kNegativeInfinity),
            distance,
            std::nextafter(distance, kPositiveInfinity),
        };
        for (float threshold : thresholds) {
            CheckThreshold(kernel, inputs, dimension, threshold, exact, suite, case_id, statistics);
            ++statistics.boundary_thresholds;
        }
    }
}

void CheckExplicitThresholds(const KernelPair &kernel,
                             const Inputs &inputs,
                             std::size_t dimension,
                             std::span<const float> thresholds,
                             std::string_view suite,
                             std::size_t case_id,
                             Statistics &statistics) {
    const std::array<float, kLaneCount> exact = ComputeExact(kernel, inputs, dimension, statistics);
    for (float threshold : thresholds) {
        CheckThreshold(kernel, inputs, dimension, threshold, exact, suite, case_id, statistics);
    }
}

Inputs AlignedInputs(const DataSet &data, std::size_t dimension, std::size_t alignment_case) {
    const std::size_t query_offset = alignment_case % kAlignmentCount;
    Inputs inputs{
        .query = data.query.values.data() + query_offset,
        .candidates = {},
    };
    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        const std::size_t candidate_offset = (alignment_case + dimension + lane) % kAlignmentCount;
        inputs.candidates[lane] = data.candidates[lane].values.data() + candidate_offset;
    }
    return inputs;
}

void CheckStorageAlignment(const DataSet &data) {
    if (reinterpret_cast<std::uintptr_t>(data.query.values.data()) % 64 != 0) {
        throw std::runtime_error("query test storage is not 64-byte aligned");
    }
    for (const FloatStorage &candidate : data.candidates) {
        if (reinterpret_cast<std::uintptr_t>(candidate.values.data()) % 64 != 0) {
            throw std::runtime_error("candidate test storage is not 64-byte aligned");
        }
    }
}

void FillFiniteData(DataSet &data, std::uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> values(-2048, 2048);
    for (float &value : data.query.values) {
        value = std::ldexp(static_cast<float>(values(random)), -7);
    }
    for (FloatStorage &candidate : data.candidates) {
        for (float &value : candidate.values) {
            value = std::ldexp(static_cast<float>(values(random)), -7);
        }
    }
}

void RunDimensionAndAlignmentCases(Statistics &statistics) {
    DataSet data;
    CheckStorageAlignment(data);
    FillFiniteData(data, 0x5EED1234U);

    for (std::size_t dimension = 0; dimension <= kMaximumDimension; ++dimension) {
        for (std::size_t alignment_case = 0; alignment_case < kAlignmentCount; ++alignment_case) {
            const Inputs inputs = AlignedInputs(data, dimension, alignment_case);
            CheckBoundaryThresholds(kResidualKernel, inputs, dimension, "dimension-alignment", alignment_case, statistics);
            if (dimension % 16 == 0) {
                CheckBoundaryThresholds(kAlignedKernel, inputs, dimension, "dimension-alignment", alignment_case, statistics);
            }
        }
    }
}

void FillCheckpointData(DataSet &data) {
    data.query.values.fill(0.0F);
    for (FloatStorage &candidate : data.candidates) {
        candidate.values.fill(0.0F);
    }
    std::fill_n(data.candidates[0].values.begin(), 32, 1.0F);
    std::fill_n(data.candidates[2].values.begin(), 32, 0.5F);
    std::fill_n(data.candidates[3].values.begin(), 64, 0.5F);
}

void RunCheckpointBoundaryCases(Statistics &statistics) {
    DataSet data;
    FillCheckpointData(data);
    const Inputs inputs{
        .query = data.query.values.data(),
        .candidates =
            {
                data.candidates[0].values.data(),
                data.candidates[1].values.data(),
                data.candidates[2].values.data(),
                data.candidates[3].values.data(),
            },
    };
    constexpr float kThreshold = 16.0F;

    for (const auto &[kernel, dimension] : std::array<std::pair<KernelPair, std::size_t>, 2>{
             std::pair{kAlignedKernel, std::size_t{64}},
             std::pair{kResidualKernel, std::size_t{65}},
         }) {
        CheckBoundaryThresholds(kernel, inputs, dimension, "checkpoint-boundary", 0, statistics);
        const std::array<float, kLaneCount> exact = ComputeExact(kernel, inputs, dimension, statistics);
        const std::uint8_t exact_mask = CheckThreshold(kernel, inputs, dimension, kThreshold, exact, "checkpoint-boundary", 1, statistics);
        if ((exact_mask & std::uint8_t{0x01}) != 0) {
            Fail("checkpoint-boundary", kernel, 1, dimension, "lane 0 was not rejected at the 32-component checkpoint");
        }
        if ((exact_mask & std::uint8_t{0x0E}) != std::uint8_t{0x0E}) {
            Fail("checkpoint-boundary", kernel, 1, dimension, "a D<=threshold control lane was not exact");
        }
    }
}

void FillSpecialData(DataSet &data) {
    data.query.values.fill(0.0F);
    for (FloatStorage &candidate : data.candidates) {
        candidate.values.fill(0.0F);
    }

    const float denormal = std::numeric_limits<float>::denorm_min();
    data.candidates[0].values[3] = std::numeric_limits<float>::quiet_NaN();
    data.candidates[1].values[17] = std::numeric_limits<float>::infinity();
    data.candidates[2].values[31] = -std::numeric_limits<float>::infinity();
    data.candidates[3].values[0] = denormal;
    data.candidates[3].values[32] = -denormal;
    data.candidates[3].values[63] = 1.0e-20F;
    data.candidates[3].values[95] = -1.0e-20F;
}

void FillSpecialQueryData(DataSet &data) {
    data.query.values.fill(0.0F);
    for (FloatStorage &candidate : data.candidates) {
        candidate.values.fill(0.0F);
    }

    const float denormal = std::numeric_limits<float>::denorm_min();
    data.query.values[0] = denormal;
    data.query.values[16] = -denormal;
    data.query.values[35] = std::numeric_limits<float>::infinity();
    data.query.values[67] = std::numeric_limits<float>::quiet_NaN();
    data.query.values[100] = -std::numeric_limits<float>::infinity();
    data.candidates[0].values[35] = std::numeric_limits<float>::infinity();
    data.candidates[0].values[100] = -std::numeric_limits<float>::infinity();
    data.candidates[2].values[35] = -std::numeric_limits<float>::infinity();
    data.candidates[2].values[100] = std::numeric_limits<float>::infinity();
}

void RunSpecialValueDataSet(const DataSet &data, std::string_view suite, Statistics &statistics) {
    constexpr std::array<std::size_t, 18> kDimensions{
        0,
        1,
        4,
        15,
        16,
        17,
        31,
        32,
        33,
        36,
        63,
        64,
        65,
        68,
        96,
        101,
        128,
        129,
    };
    const std::array<float, 9> thresholds{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::max(),
        -0.0F,
        0.0F,
        std::numeric_limits<float>::denorm_min(),
        1.0F,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };

    for (std::size_t case_id = 0; case_id < kDimensions.size(); ++case_id) {
        const std::size_t dimension = kDimensions[case_id];
        const Inputs inputs{
            .query = data.query.values.data(),
            .candidates =
                {
                    data.candidates[0].values.data(),
                    data.candidates[1].values.data(),
                    data.candidates[2].values.data(),
                    data.candidates[3].values.data(),
                },
        };
        CheckBoundaryThresholds(kResidualKernel, inputs, dimension, suite, case_id, statistics);
        CheckExplicitThresholds(kResidualKernel, inputs, dimension, thresholds, suite, case_id, statistics);
        if (dimension % 16 == 0) {
            CheckBoundaryThresholds(kAlignedKernel, inputs, dimension, suite, case_id, statistics);
            CheckExplicitThresholds(kAlignedKernel, inputs, dimension, thresholds, suite, case_id, statistics);
        }
    }
}

void RunSpecialValueCases(Statistics &statistics) {
    DataSet candidate_specials;
    FillSpecialData(candidate_specials);
    RunSpecialValueDataSet(candidate_specials, "candidate-special-values", statistics);

    DataSet query_specials;
    FillSpecialQueryData(query_specials);
    RunSpecialValueDataSet(query_specials, "query-special-values", statistics);
}

void RunRandomizedCases(Statistics &statistics) {
    DataSet data;
    std::mt19937 random(0xC001D00DU);
    std::uniform_int_distribution<std::size_t> dimensions(0, kMaximumDimension);
    std::uniform_int_distribution<std::size_t> alignments(0, kAlignmentCount - 1);
    std::uniform_int_distribution<int> values(-8192, 8192);
    std::uniform_int_distribution<int> exponents(-10, 2);
    std::uniform_int_distribution<std::size_t> lanes(0, kLaneCount - 1);

    for (std::size_t case_id = 0; case_id < kRandomCaseCount; ++case_id) {
        const std::size_t dimension = dimensions(random);
        const std::size_t extent = dimension + kAlignmentCount;
        for (std::size_t index = 0; index < extent; ++index) {
            data.query.values[index] = std::ldexp(static_cast<float>(values(random)), exponents(random));
            for (FloatStorage &candidate : data.candidates) {
                candidate.values[index] = std::ldexp(static_cast<float>(values(random)), exponents(random));
            }
        }

        Inputs inputs{
            .query = data.query.values.data() + alignments(random),
            .candidates = {},
        };
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            inputs.candidates[lane] = data.candidates[lane].values.data() + alignments(random);
        }

        CheckBoundaryThresholds(kResidualKernel, inputs, dimension, "randomized", case_id, statistics);
        const std::array<float, kLaneCount> residual_exact = ComputeExact(kResidualKernel, inputs, dimension, statistics);
        const float arbitrary_threshold = residual_exact[lanes(random)] * 0.625F;
        CheckThreshold(kResidualKernel, inputs, dimension, arbitrary_threshold, residual_exact, "randomized", case_id, statistics);

        if (dimension % 16 == 0) {
            CheckBoundaryThresholds(kAlignedKernel, inputs, dimension, "randomized", case_id, statistics);
            const std::array<float, kLaneCount> aligned_exact = ComputeExact(kAlignedKernel, inputs, dimension, statistics);
            CheckThreshold(kAlignedKernel, inputs, dimension, arbitrary_threshold, aligned_exact, "randomized", case_id, statistics);
        }
        ++statistics.randomized_cases;
    }
}

class GuardedPrefix {
public:
    GuardedPrefix(std::size_t readable_float_count, std::size_t logical_float_count) {
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size <= 0 || readable_float_count > logical_float_count ||
            static_cast<std::size_t>(page_size) < logical_float_count * sizeof(float)) {
            throw std::runtime_error("invalid Darwin page size for guard-page test");
        }
        page_size_ = static_cast<std::size_t>(page_size);
        mapping_size_ = 2 * page_size_;
        mapping_ = ::mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (mapping_ == MAP_FAILED) {
            throw std::runtime_error("mmap failed in guard-page test: errno=" + std::to_string(errno));
        }

        auto *guard_page = static_cast<std::byte *>(mapping_) + page_size_;
        values_ = reinterpret_cast<float *>(guard_page) - readable_float_count;
        std::fill_n(values_, logical_float_count, 0.0F);
        std::fill_n(values_, readable_float_count, 1.0F);
        if (::mprotect(guard_page, page_size_, PROT_NONE) != 0) {
            const int error = errno;
            static_cast<void>(::munmap(mapping_, mapping_size_));
            mapping_ = MAP_FAILED;
            throw std::runtime_error("mprotect failed in guard-page test: errno=" + std::to_string(error));
        }
    }

    GuardedPrefix(const GuardedPrefix &) = delete;
    GuardedPrefix &operator=(const GuardedPrefix &) = delete;

    ~GuardedPrefix() {
        if (mapping_ != MAP_FAILED) {
            static_cast<void>(::munmap(mapping_, mapping_size_));
        }
    }

    const float *values() const { return values_; }

private:
    void *mapping_ = MAP_FAILED;
    std::size_t page_size_ = 0;
    std::size_t mapping_size_ = 0;
    float *values_ = nullptr;
};

void RunOneGuardPageCase(const KernelPair &kernel, std::size_t dimension, std::size_t case_id, Statistics &statistics) {
    constexpr std::size_t kReadableComponents = 32;
    constexpr float kThreshold = 16.0F;
    alignas(64) std::array<float, 65> query{};
    alignas(64) std::array<float, 65> candidate0_mirror{};
    alignas(64) std::array<float, 65> candidate1{};
    alignas(64) std::array<float, 65> candidate2{};
    alignas(64) std::array<float, 65> candidate3{};
    std::fill_n(candidate0_mirror.begin(), kReadableComponents, 1.0F);
    std::fill_n(candidate2.begin(), 64, 0.25F);
    std::fill_n(candidate3.begin(), 64, 0.5F);

    const Inputs mirror_inputs{
        .query = query.data(),
        .candidates =
            {
                candidate0_mirror.data(),
                candidate1.data(),
                candidate2.data(),
                candidate3.data(),
            },
    };
    const std::array<float, kLaneCount> exact = ComputeExact(kernel, mirror_inputs, dimension, statistics);

    GuardedPrefix guarded_candidate(kReadableComponents, dimension);
    const Inputs guarded_inputs{
        .query = query.data(),
        .candidates =
            {
                guarded_candidate.values(),
                candidate1.data(),
                candidate2.data(),
                candidate3.data(),
            },
    };
    const std::uint8_t exact_mask = CheckThreshold(kernel, guarded_inputs, dimension, kThreshold, exact, "guard-page", case_id, statistics);
    if ((exact_mask & std::uint8_t{0x01}) != 0) {
        Fail("guard-page", kernel, case_id, dimension, "lane 0 was not rejected after its readable 32-component prefix");
    }
    if ((exact_mask & std::uint8_t{0x0E}) != std::uint8_t{0x0E}) {
        Fail("guard-page", kernel, case_id, dimension, "an unguarded D<=threshold control lane was not exact");
    }
    ++statistics.guard_page_cases;
}

void RunGuardPageCases(Statistics &statistics) {
    RunOneGuardPageCase(kAlignedKernel, 64, 0, statistics);
    RunOneGuardPageCase(kResidualKernel, 65, 1, statistics);
}

} // namespace
#endif

int main() {
#if !defined(__APPLE__) || !defined(__aarch64__)
    std::cerr << "thresholded Batch4 L2 test requires arm64 macOS\n";
    return 64;
#else
    try {
        Statistics statistics;
        RunDimensionAndAlignmentCases(statistics);
        RunCheckpointBoundaryCases(statistics);
        RunSpecialValueCases(statistics);
        RunRandomizedCases(statistics);
        RunGuardPageCases(statistics);

        if (statistics.exact_lanes == 0 || statistics.rejected_lanes == 0 || statistics.qualifying_lanes == 0 || statistics.guard_page_cases != 2) {
            throw std::runtime_error("test campaign did not exercise every mask outcome");
        }

        std::cout << "status=PASS\n";
        std::cout << "dimension_range=0:" << kMaximumDimension << '\n';
        std::cout << "alignment_offsets_bytes=0,4,8,12\n";
        std::cout << "randomized_cases=" << statistics.randomized_cases << '\n';
        std::cout << "exact_calls=" << statistics.exact_calls << '\n';
        std::cout << "thresholded_calls=" << statistics.thresholded_calls << '\n';
        std::cout << "boundary_thresholds=" << statistics.boundary_thresholds << '\n';
        std::cout << "lane_checks=" << statistics.lane_checks << '\n';
        std::cout << "exact_lanes=" << statistics.exact_lanes << '\n';
        std::cout << "rejected_lanes=" << statistics.rejected_lanes << '\n';
        std::cout << "qualifying_lanes=" << statistics.qualifying_lanes << '\n';
        std::cout << "guard_page_cases=" << statistics.guard_page_cases << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "status=FAIL\n";
        std::cerr << "error=" << error.what() << '\n';
        return 1;
    }
#endif
}
