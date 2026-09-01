import std;
import std.compat;
import infinity_core;

#if defined(__APPLE__) && defined(__aarch64__)
namespace {

constexpr size_t kMaximumDimension = 1025;
constexpr size_t kCandidateCount = 4;
constexpr size_t kAlignmentCount = 4;
constexpr double kMaximumRelativeError = 5e-6;

double ReferenceL2(const float *left, const float *right, size_t dimension) {
    double result = 0;
    for (size_t index = 0; index < dimension; ++index) {
        const double delta = static_cast<double>(left[index]) - static_cast<double>(right[index]);
        result += delta * delta;
    }
    return result;
}

double RelativeError(float actual, double expected) {
    return std::abs(static_cast<double>(actual) - expected) / std::max(1.0, expected);
}

bool SameBits(float left, float right) { return std::bit_cast<uint32_t>(left) == std::bit_cast<uint32_t>(right); }

// Sentinel written into the threshold kernel's output before each call. The kernel writes
// only surviving lanes, and its caller relies on that by consuming the returned mask; if an
// aborted lane's slot were overwritten with a partial sum, a caller that trusted the array
// instead of the mask would silently admit a wrong distance. Asserting the sentinel
// survives pins that contract.
constexpr float kUntouched = -12345.0f;

// Contract checks for the threshold kernels, which are otherwise untested. `plain` holds
// the corresponding non-threshold batch distances, already known bit-identical to scalar.
//
// Three properties, in order of importance:
//   1. With threshold = +inf nothing may abort: the mask is all lanes and every distance is
//      BIT-IDENTICAL to the plain kernel. This is what lets a caller pass +inf whenever it
//      has no cutoff yet (e.g. while the result heap is not full) and get exact results.
//   2. A surviving lane's distance is bit-identical to the plain kernel's.
//   3. An aborted lane's TRUE distance is strictly greater than the threshold. This is the
//      soundness property: aborting must never discard a candidate that was within the
//      cutoff. Partial sums of squares only grow, so `partial > t` implies `final > t`.
// The converse of 3 is deliberately NOT required -- the kernel only tests at 32-component
// checkpoints, so a lane may survive with a distance above the threshold. That is
// conservative and harmless.
template <typename BatchThresholdFn>
size_t CheckThresholdKernel(BatchThresholdFn &&kernel,
                            const float *query,
                            const std::array<const float *, kCandidateCount> &candidates,
                            size_t dimension,
                            const std::array<float, kCandidateCount> &plain,
                            size_t &cases,
                            size_t &untouched_violations,
                            size_t &unsound_aborts) {
    size_t mismatches = 0;
    const float infinite_threshold = std::numeric_limits<float>::infinity();
    // Using the smallest true distance as the finite threshold guarantees the nearest lane
    // survives while leaving the others eligible to abort, so both paths get exercised.
    const float tight_threshold = *std::min_element(plain.begin(), plain.end());
    for (const float threshold : {infinite_threshold, tight_threshold, 0.0f}) {
        std::array<float, kCandidateCount> out{};
        out.fill(kUntouched);
        const std::uint8_t mask = kernel(query, candidates, dimension, threshold, out.data());
        for (size_t lane = 0; lane < kCandidateCount; ++lane) {
            const bool survived = (mask & (std::uint8_t{1} << lane)) != 0;
            ++cases;
            if (survived) {
                mismatches += !SameBits(out[lane], plain[lane]);
            } else {
                untouched_violations += !SameBits(out[lane], kUntouched);
                unsound_aborts += !(plain[lane] > threshold);
            }
            if (std::isinf(threshold) && !survived) {
                ++mismatches; // property 1: +inf must never abort
            }
        }
    }
    return mismatches;
}

} // namespace
#endif

int main() {
#if !defined(__APPLE__) || !defined(__aarch64__)
    std::cerr << "native SIMD smoke requires arm64 macOS\n";
    return 64;
#else

    std::vector<float> query_storage(kMaximumDimension + kAlignmentCount);
    std::array<std::vector<float>, kCandidateCount> candidate_storage;
    for (auto &storage : candidate_storage) {
        storage.resize(kMaximumDimension + kAlignmentCount);
    }
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
    for (float &value : query_storage) {
        value = distribution(rng);
    }
    for (auto &storage : candidate_storage) {
        for (float &value : storage) {
            value = distribution(rng);
        }
    }

    double maximum_relative_error = 0;
    size_t failed_cases = 0;
    size_t batch_cases = 0;
    size_t batch_bit_mismatches = 0;
    size_t threshold_cases = 0;
    size_t threshold_bit_mismatches = 0;
    size_t threshold_untouched_violations = 0;
    size_t threshold_unsound_aborts = 0;
    for (size_t query_offset = 0; query_offset < kAlignmentCount; ++query_offset) {
        const float *query = query_storage.data() + query_offset;
        for (size_t candidate_offset = 0; candidate_offset < kAlignmentCount; ++candidate_offset) {
            std::array<const float *, kCandidateCount> candidates{};
            for (size_t lane = 0; lane < kCandidateCount; ++lane) {
                candidates[lane] = candidate_storage[lane].data() + ((candidate_offset + lane) % kAlignmentCount);
            }
            for (size_t dimension = 0; dimension <= kMaximumDimension; ++dimension) {
                std::array<float, kCandidateCount> scalar{};
                std::array<float, kCandidateCount> batch{};
                for (size_t lane = 0; lane < kCandidateCount; ++lane) {
                    const double reference = ReferenceL2(query, candidates[lane], dimension);
                    scalar[lane] = infinity::F32L2SSEResidual(query, candidates[lane], dimension);
                    const double residual_error = RelativeError(scalar[lane], reference);
                    maximum_relative_error = std::max(maximum_relative_error, residual_error);
                    failed_cases += residual_error > kMaximumRelativeError;
                }
                infinity::F32L2SSEResidualBatch4(query,
                                                candidates[0],
                                                candidates[1],
                                                candidates[2],
                                                candidates[3],
                                                dimension,
                                                batch.data());
                for (size_t lane = 0; lane < kCandidateCount; ++lane) {
                    batch_bit_mismatches += !SameBits(batch[lane], scalar[lane]);
                    ++batch_cases;
                }

                threshold_bit_mismatches += CheckThresholdKernel(
                    [](const float *q,
                       const std::array<const float *, kCandidateCount> &c,
                       size_t dim,
                       float threshold,
                       float *out) {
                        return infinity::F32L2SSEResidualBatch4WithinThreshold(q, c[0], c[1], c[2], c[3], dim, threshold, out);
                    },
                    query, candidates, dimension, batch,
                    threshold_cases, threshold_untouched_violations, threshold_unsound_aborts);

                if (dimension % 16 == 0) {
                    for (size_t lane = 0; lane < kCandidateCount; ++lane) {
                        const double reference = ReferenceL2(query, candidates[lane], dimension);
                        scalar[lane] = infinity::F32L2SSE(query, candidates[lane], dimension);
                        const double aligned_error = RelativeError(scalar[lane], reference);
                        maximum_relative_error = std::max(maximum_relative_error, aligned_error);
                        failed_cases += aligned_error > kMaximumRelativeError;
                    }
                    infinity::F32L2SSEBatch4(query,
                                            candidates[0],
                                            candidates[1],
                                            candidates[2],
                                            candidates[3],
                                            dimension,
                                            batch.data());
                    for (size_t lane = 0; lane < kCandidateCount; ++lane) {
                        batch_bit_mismatches += !SameBits(batch[lane], scalar[lane]);
                        ++batch_cases;
                    }

                    threshold_bit_mismatches += CheckThresholdKernel(
                        [](const float *q,
                           const std::array<const float *, kCandidateCount> &c,
                           size_t dim,
                           float threshold,
                           float *out) {
                            return infinity::F32L2SSEBatch4WithinThreshold(q, c[0], c[1], c[2], c[3], dim, threshold, out);
                        },
                        query, candidates, dimension, batch,
                        threshold_cases, threshold_untouched_violations, threshold_unsound_aborts);
                }
            }
        }
    }

    failed_cases += batch_bit_mismatches;
    failed_cases += threshold_bit_mismatches + threshold_untouched_violations + threshold_unsound_aborts;
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "status=" << (failed_cases == 0 ? "PASS" : "FAIL") << '\n';
    std::cout << "architecture=arm64-apple\n";
    std::cout << "kernel=F32L2SSE_SIMDe_NEON_batch4\n";
    std::cout << "dimensions_tested=0:" << kMaximumDimension << '\n';
    std::cout << "pointer_alignment_combinations=" << kAlignmentCount * kAlignmentCount << '\n';
    std::cout << "batch_outputs_tested=" << batch_cases << '\n';
    std::cout << "batch_bit_mismatches=" << batch_bit_mismatches << '\n';
    std::cout << "threshold_lane_outcomes_tested=" << threshold_cases << '\n';
    std::cout << "threshold_bit_mismatches=" << threshold_bit_mismatches << '\n';
    std::cout << "threshold_untouched_violations=" << threshold_untouched_violations << '\n';
    std::cout << "threshold_unsound_aborts=" << threshold_unsound_aborts << '\n';
    std::cout << "maximum_relative_error=" << maximum_relative_error << '\n';
    std::cout << "failed_cases=" << failed_cases << '\n';

    return failed_cases == 0 ? 0 : 1;
#endif
}
