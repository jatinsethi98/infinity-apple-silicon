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
                }
            }
        }
    }

    failed_cases += batch_bit_mismatches;
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "status=" << (failed_cases == 0 ? "PASS" : "FAIL") << '\n';
    std::cout << "architecture=arm64-apple\n";
    std::cout << "kernel=F32L2SSE_SIMDe_NEON_batch4\n";
    std::cout << "dimensions_tested=0:" << kMaximumDimension << '\n';
    std::cout << "pointer_alignment_combinations=" << kAlignmentCount * kAlignmentCount << '\n';
    std::cout << "batch_outputs_tested=" << batch_cases << '\n';
    std::cout << "batch_bit_mismatches=" << batch_bit_mismatches << '\n';
    std::cout << "maximum_relative_error=" << maximum_relative_error << '\n';
    std::cout << "failed_cases=" << failed_cases << '\n';

    return failed_cases == 0 ? 0 : 1;
#endif
}
