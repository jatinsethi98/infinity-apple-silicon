// Is Infinity's HNSW distance kernel latency-bound on scattered candidate reads, or
// FP-throughput-bound?
//
// This question decides the next optimisation. The 1M/12-thread profile puts 53% of build time
// in the L2 kernels, and FMA (which cut the inner loop from 48 to 32 FP ops, a 1.5x bound on
// arithmetic) produced no measurable speedup. That is the signature of a kernel waiting on
// memory rather than on the FP pipes -- but "signature of" is not evidence, and the existing
// neon_l2_micro cannot settle it: it walks 8192 vectors sequentially over a 4 MB working set,
// which is L2-resident and prefetch-friendly, i.e. the opposite of graph traversal.
//
// So reproduce the build's access pattern and vary only the number of INDEPENDENT candidate
// streams:
//
//   * working set 1,000,000 x 128 floats = 512 MB, far beyond the 16 MB shared L2, as in a
//     real SIFT1M build;
//   * candidate indices drawn at random, as a graph traversal reaches them, with the index
//     list precomputed so RNG cost stays out of the timed loop;
//   * kernels differing ONLY in how many candidates are in flight: 1, 4 (what Infinity ships),
//     8 (what more memory-level parallelism would look like). Accumulators per candidate are
//     held at 2 so that 8 candidates need 16 vector registers of the 32 available and cannot
//     spill -- register pressure was the stated reason the 8-wide option was dismissed.
//
// Plus a SEQUENTIAL control at the same working-set size. That is the actual hypothesis test:
// if scattered is much slower than sequential at equal FP work, the kernel is latency-bound and
// more streams should help; if they are close, it is throughput-bound and widening is pointless.
//
// Development-only diagnostic. Reports ns per distance evaluation; correctness of every kernel
// is checked against a double-precision reference before timing.

#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kDim = 128;
constexpr std::size_t kVectors = 1'000'000; // 512 MB at d=128 float
constexpr std::size_t kChunks = kDim / 4;   // 32 NEON lanes-of-4 per vector

double ScalarL2(const float *a, const float *b, std::size_t d) {
    double total = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        const double delta = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        total += delta * delta;
    }
    return total;
}

inline float Reduce(float32x4_t a, float32x4_t b) { return vaddvq_f32(vaddq_f32(a, b)); }

// One candidate, 2 accumulators.
inline void L2x1(const float *q, const float *const *cands, float *out) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    const float *c = cands[0];
    for (std::size_t i = 0; i < kChunks; i += 2) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(q + i * 4), vld1q_f32(c + i * 4));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(q + i * 4 + 4), vld1q_f32(c + i * 4 + 4));
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
    }
    out[0] = Reduce(s0, s1);
}

// Four candidates in flight, 2 accumulators each = 8 vector registers.
inline void L2x4(const float *q, const float *const *cands, float *out) {
    float32x4_t s[4][2];
    for (int k = 0; k < 4; ++k) { s[k][0] = vdupq_n_f32(0.0f); s[k][1] = vdupq_n_f32(0.0f); }
    for (std::size_t i = 0; i < kChunks; i += 2) {
        const float32x4_t q0 = vld1q_f32(q + i * 4);
        const float32x4_t q1 = vld1q_f32(q + i * 4 + 4);
        for (int k = 0; k < 4; ++k) {
            const float32x4_t d0 = vsubq_f32(q0, vld1q_f32(cands[k] + i * 4));
            const float32x4_t d1 = vsubq_f32(q1, vld1q_f32(cands[k] + i * 4 + 4));
            s[k][0] = vfmaq_f32(s[k][0], d0, d0);
            s[k][1] = vfmaq_f32(s[k][1], d1, d1);
        }
    }
    for (int k = 0; k < 4; ++k) { out[k] = Reduce(s[k][0], s[k][1]); }
}

// Eight candidates in flight, 2 accumulators each = 16 vector registers of 32.
inline void L2x8(const float *q, const float *const *cands, float *out) {
    float32x4_t s[8][2];
    for (int k = 0; k < 8; ++k) { s[k][0] = vdupq_n_f32(0.0f); s[k][1] = vdupq_n_f32(0.0f); }
    for (std::size_t i = 0; i < kChunks; i += 2) {
        const float32x4_t q0 = vld1q_f32(q + i * 4);
        const float32x4_t q1 = vld1q_f32(q + i * 4 + 4);
        for (int k = 0; k < 8; ++k) {
            const float32x4_t d0 = vsubq_f32(q0, vld1q_f32(cands[k] + i * 4));
            const float32x4_t d1 = vsubq_f32(q1, vld1q_f32(cands[k] + i * 4 + 4));
            s[k][0] = vfmaq_f32(s[k][0], d0, d0);
            s[k][1] = vfmaq_f32(s[k][1], d1, d1);
        }
    }
    for (int k = 0; k < 8; ++k) { out[k] = Reduce(s[k][0], s[k][1]); }
}

struct Kernel {
    const char *name;
    std::size_t width;
    void (*fn)(const float *, const float *const *, float *);
};

} // namespace

int main(int argc, char **argv) {
    const std::size_t distances = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8'000'000;
    const std::size_t rounds = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 5;
    if (distances == 0 || rounds == 0) {
        std::fprintf(stderr, "distances and rounds must be positive\n");
        return 64;
    }

    std::printf("scope=development-only\n");
    std::printf("working_set_bytes=%zu\n", kVectors * kDim * sizeof(float));
    std::printf("dim=%zu distances_per_measurement=%zu rounds=%zu\n", kDim, distances, rounds);

    std::vector<float> data(kVectors * kDim);
    std::vector<float> query(kDim);
    {
        std::mt19937 gen(20260902);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float &v : query) { v = dist(gen); }
        for (float &v : data) { v = dist(gen); }
    }

    // Scattered index list, and a sequential control of the same length. Both precomputed so
    // neither RNG nor modulo cost lands inside a timed region.
    std::vector<std::uint32_t> scattered(distances);
    std::vector<std::uint32_t> sequential(distances);
    {
        std::mt19937 gen(777);
        std::uniform_int_distribution<std::uint32_t> pick(0, kVectors - 1);
        for (std::size_t i = 0; i < distances; ++i) {
            scattered[i] = pick(gen);
            sequential[i] = static_cast<std::uint32_t>(i % kVectors);
        }
    }

    const Kernel kernels[] = {
        {"width1", 1, &L2x1},
        {"width4", 4, &L2x4},
        {"width8", 8, &L2x8},
    };

    // Correctness before timing: a fast wrong kernel is worthless.
    for (const Kernel &k : kernels) {
        const float *cands[8];
        float out[8];
        for (std::size_t base = 0; base + k.width <= 64; base += k.width) {
            for (std::size_t j = 0; j < k.width; ++j) { cands[j] = data.data() + (base + j) * kDim; }
            k.fn(query.data(), cands, out);
            for (std::size_t j = 0; j < k.width; ++j) {
                const double ref = ScalarL2(query.data(), cands[j], kDim);
                const double tol = 1e-4 * std::max(1.0, ref);
                if (std::abs(static_cast<double>(out[j]) - ref) > tol) {
                    std::fprintf(stderr, "validation failure: %s row=%zu ref=%f got=%f\n",
                                 k.name, base + j, ref, static_cast<double>(out[j]));
                    return 1;
                }
            }
        }
    }
    std::printf("validation=PASS\n");

    struct Result { double ns[3]; };
    // Rotate kernel order across rounds so a warming or thermal trend cannot systematically
    // favour whichever kernel would otherwise always run first.
    const std::size_t order[6][3] = {{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};

    for (int pattern = 0; pattern < 2; ++pattern) {
        const std::vector<std::uint32_t> &idx = pattern == 0 ? scattered : sequential;
        const char *pattern_name = pattern == 0 ? "scattered" : "sequential";
        std::vector<std::vector<double>> samples(3);

        for (std::size_t r = 0; r < rounds; ++r) {
            for (std::size_t slot = 0; slot < 3; ++slot) {
                const std::size_t ki = order[r % 6][slot];
                const Kernel &k = kernels[ki];
                const float *cands[8];
                float out[8];
                float sink = 0.0f;
                const auto t0 = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i + k.width <= distances; i += k.width) {
                    for (std::size_t j = 0; j < k.width; ++j) {
                        cands[j] = data.data() + static_cast<std::size_t>(idx[i + j]) * kDim;
                    }
                    k.fn(query.data(), cands, out);
                    for (std::size_t j = 0; j < k.width; ++j) { sink += out[j]; }
                }
                const auto t1 = std::chrono::steady_clock::now();
                const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                samples[ki].push_back(ns / static_cast<double>(distances));
                // Consume sink so the optimiser cannot delete the whole loop.
                if (!std::isfinite(sink)) { std::fprintf(stderr, "non-finite sink\n"); return 1; }
            }
        }

        std::printf("\npattern=%s\n", pattern_name);
        double base = 0.0;
        for (std::size_t ki = 0; ki < 3; ++ki) {
            std::vector<double> &s = samples[ki];
            std::sort(s.begin(), s.end());
            const double med = s[s.size() / 2];
            if (ki == 0) { base = med; }
            std::printf("  %-8s ns_per_distance=%7.3f  min=%7.3f  vs_width1=%5.3fx\n",
                        kernels[ki].name, med, s.front(), base / med);
        }
    }
    return 0;
}
