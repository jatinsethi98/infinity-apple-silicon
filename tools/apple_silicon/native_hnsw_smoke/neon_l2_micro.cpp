#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <vector>

#if !defined(__aarch64__)
#error "This prototype requires AArch64 NEON"
#endif

namespace {

using DistanceFunction = float (*)(const float *, const float *, std::size_t);

volatile double result_sink = 0.0;

float Reduce(float32x4_t sum) {
    alignas(16) float lanes[4];
    vst1q_f32(lanes, sum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

__attribute__((noinline)) float
L2MulAdd4(const float *left, const float *right, std::size_t dimension) {
    const std::size_t aligned = dimension & ~std::size_t{15};
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    for (std::size_t offset = 0; offset < aligned; offset += 16) {
        const float32x4_t diff0 =
            vsubq_f32(vld1q_f32(left + offset), vld1q_f32(right + offset));
        const float32x4_t diff1 = vsubq_f32(
            vld1q_f32(left + offset + 4),
            vld1q_f32(right + offset + 4));
        const float32x4_t diff2 = vsubq_f32(
            vld1q_f32(left + offset + 8),
            vld1q_f32(right + offset + 8));
        const float32x4_t diff3 = vsubq_f32(
            vld1q_f32(left + offset + 12),
            vld1q_f32(right + offset + 12));
        sum0 = vaddq_f32(sum0, vmulq_f32(diff0, diff0));
        sum1 = vaddq_f32(sum1, vmulq_f32(diff1, diff1));
        sum2 = vaddq_f32(sum2, vmulq_f32(diff2, diff2));
        sum3 = vaddq_f32(sum3, vmulq_f32(diff3, diff3));
    }
    float result =
        Reduce(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    for (std::size_t offset = aligned; offset < dimension; ++offset) {
        const float diff = left[offset] - right[offset];
        result += diff * diff;
    }
    return result;
}

__attribute__((noinline)) float
L2Fma4(const float *left, const float *right, std::size_t dimension) {
    const std::size_t aligned = dimension & ~std::size_t{15};
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    for (std::size_t offset = 0; offset < aligned; offset += 16) {
        const float32x4_t diff0 =
            vsubq_f32(vld1q_f32(left + offset), vld1q_f32(right + offset));
        const float32x4_t diff1 = vsubq_f32(
            vld1q_f32(left + offset + 4),
            vld1q_f32(right + offset + 4));
        const float32x4_t diff2 = vsubq_f32(
            vld1q_f32(left + offset + 8),
            vld1q_f32(right + offset + 8));
        const float32x4_t diff3 = vsubq_f32(
            vld1q_f32(left + offset + 12),
            vld1q_f32(right + offset + 12));
        sum0 = vfmaq_f32(sum0, diff0, diff0);
        sum1 = vfmaq_f32(sum1, diff1, diff1);
        sum2 = vfmaq_f32(sum2, diff2, diff2);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
    }
    float result =
        Reduce(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    for (std::size_t offset = aligned; offset < dimension; ++offset) {
        const float diff = left[offset] - right[offset];
        result += diff * diff;
    }
    return result;
}

__attribute__((noinline)) float
L2Fma8(const float *left, const float *right, std::size_t dimension) {
    const std::size_t aligned32 = dimension & ~std::size_t{31};
    std::array<float32x4_t, 8> sums;
    std::fill(sums.begin(), sums.end(), vdupq_n_f32(0.0f));
    for (std::size_t offset = 0; offset < aligned32; offset += 32) {
        for (std::size_t lane = 0; lane < sums.size(); ++lane) {
            const std::size_t vector_offset = offset + lane * 4;
            const float32x4_t diff = vsubq_f32(
                vld1q_f32(left + vector_offset),
                vld1q_f32(right + vector_offset));
            sums[lane] = vfmaq_f32(sums[lane], diff, diff);
        }
    }
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (float32x4_t partial : sums) {
        sum = vaddq_f32(sum, partial);
    }
    float result = Reduce(sum);
    for (std::size_t offset = aligned32; offset < dimension; ++offset) {
        const float diff = left[offset] - right[offset];
        result += diff * diff;
    }
    return result;
}

double ScalarL2(
    const float *left,
    const float *right,
    std::size_t dimension) {
    double result = 0.0;
    for (std::size_t offset = 0; offset < dimension; ++offset) {
        const double diff =
            static_cast<double>(left[offset]) - static_cast<double>(right[offset]);
        result += diff * diff;
    }
    return result;
}

struct Measurement {
    double median_seconds{};
    double nanoseconds_per_element{};
    double checksum{};
};

std::pair<double, double> MeasureOnce(
    DistanceFunction function,
    const std::vector<float> &data,
    std::span<const float> query,
    std::size_t vector_count,
    std::size_t dimension,
    std::size_t calls,
    std::size_t round) {
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t call = 0; call < calls; ++call) {
        const std::size_t row =
            (call * 4051 + round * 7919) & (vector_count - 1);
        checksum += function(
            query.data(),
            data.data() + row * dimension,
            dimension);
    }
    const auto end = std::chrono::steady_clock::now();
    result_sink = checksum;
    return {
        std::chrono::duration<double>(end - start).count(),
        checksum,
    };
}

Measurement Summarize(
    std::vector<double> durations,
    double checksum,
    std::size_t dimension,
    std::size_t calls) {
    std::sort(durations.begin(), durations.end());
    const double median = durations[durations.size() / 2];
    return {
        .median_seconds = median,
        .nanoseconds_per_element =
            median * 1e9 / static_cast<double>(calls * dimension),
        .checksum = checksum,
    };
}

} // namespace

int main(int argc, char **argv) {
    const std::size_t calls =
        argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1'000'000;
    const std::size_t rounds =
        argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 5;
    constexpr std::size_t vector_count = 8192;
    constexpr std::array<std::size_t, 3> dimensions = {128, 768, 960};
    const std::array<std::pair<const char *, DistanceFunction>, 3> functions = {
        std::pair{"mul_add_4", &L2MulAdd4},
        std::pair{"fma_4", &L2Fma4},
        std::pair{"fma_8", &L2Fma8},
    };
    constexpr std::array<std::array<std::size_t, 3>, 6> run_orders = {{
        {0, 1, 2},
        {1, 2, 0},
        {2, 0, 1},
        {2, 1, 0},
        {1, 0, 2},
        {0, 2, 1},
    }};

    if (calls == 0 || rounds == 0) {
        std::cerr << "calls and rounds must be positive\n";
        return 64;
    }

    std::mt19937 generator(20260820);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    bool valid = true;
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "scope=development-only\n";
    std::cout << "calls=" << calls << '\n';
    std::cout << "rounds=" << rounds << '\n';

    for (std::size_t dimension : dimensions) {
        std::vector<float> query(dimension);
        std::vector<float> data(vector_count * dimension);
        std::generate(query.begin(), query.end(), [&] { return distribution(generator); });
        std::generate(data.begin(), data.end(), [&] { return distribution(generator); });

        for (std::size_t row = 0; row < 128; ++row) {
            const float *candidate = data.data() + row * dimension;
            const double reference = ScalarL2(query.data(), candidate, dimension);
            for (const auto &[name, function] : functions) {
                const double actual = function(query.data(), candidate, dimension);
                const double tolerance =
                    8.0 * std::numeric_limits<float>::epsilon() * reference;
                if (std::abs(actual - reference) > tolerance) {
                    std::cerr << "validation failure: dimension=" << dimension
                              << " row=" << row << " function=" << name
                              << " expected=" << reference
                              << " actual=" << actual << '\n';
                    valid = false;
                }
            }
        }

        std::array<std::vector<double>, 3> durations;
        std::array<double, 3> checksums{};
        for (std::size_t index = 0; index < functions.size(); ++index) {
            MeasureOnce(
                functions[index].second,
                data,
                query,
                vector_count,
                dimension,
                std::max<std::size_t>(calls / 10, 1),
                index);
            durations[index].reserve(rounds);
        }
        for (std::size_t round = 0; round < rounds; ++round) {
            for (std::size_t index : run_orders[round % run_orders.size()]) {
                const auto [seconds, checksum] = MeasureOnce(
                    functions[index].second,
                    data,
                    query,
                    vector_count,
                    dimension,
                    calls,
                    round);
                durations[index].push_back(seconds);
                checksums[index] = checksum;
                std::cout << "dimension=" << dimension
                          << " round=" << round
                          << " kernel=" << functions[index].first
                          << " seconds=" << seconds << '\n';
            }
        }

        std::array<Measurement, 3> measurements;
        for (std::size_t index = 0; index < functions.size(); ++index) {
            measurements[index] = Summarize(
                std::move(durations[index]),
                checksums[index],
                dimension,
                calls);
            std::cout << "dimension=" << dimension
                      << " kernel=" << functions[index].first
                      << " median_seconds=" << measurements[index].median_seconds
                      << " ns_per_element="
                      << measurements[index].nanoseconds_per_element
                      << " checksum=" << measurements[index].checksum << '\n';
        }
        std::cout << "dimension=" << dimension
                  << " fma4_speedup="
                  << measurements[0].median_seconds /
                         measurements[1].median_seconds
                  << " fma8_speedup="
                  << measurements[0].median_seconds /
                         measurements[2].median_seconds
                  << '\n';
    }

    std::cout << "status=" << (valid ? "PASS" : "FAIL") << '\n';
    return valid ? 0 : 1;
}
