#include <faiss/IndexHNSW.h>
#include <omp.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kDimension = 16;
constexpr std::size_t kChunkSize = 128;
constexpr std::size_t kChunkCount = 10;
constexpr std::size_t kVectorCount = kChunkSize * kChunkCount;
constexpr int kM = 8;
constexpr int kEfConstruction = 200;
constexpr int kEfSearch = 10;
constexpr double kMinimumSelfRecall = 0.95;

} // namespace

int main() {
#if !defined(__APPLE__) || !defined(__aarch64__)
    std::cerr << "native smoke requires arm64 macOS\n";
    return 64;
#endif

    std::vector<float> data(kVectorCount * kDimension);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> distribution;
    for (float &value : data) {
        value = distribution(rng);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(1);

    faiss::IndexHNSWFlat index(static_cast<int>(kDimension), kM, faiss::METRIC_L2);
    index.hnsw.efConstruction = kEfConstruction;
    index.hnsw.efSearch = kEfSearch;

    const auto insert_begin = std::chrono::steady_clock::now();
    index.add(static_cast<faiss::idx_t>(kVectorCount), data.data());
    const auto insert_end = std::chrono::steady_clock::now();

    std::vector<float> distances(kVectorCount);
    std::vector<faiss::idx_t> labels(kVectorCount);
    index.search(
        static_cast<faiss::idx_t>(kVectorCount),
        data.data(),
        1,
        distances.data(),
        labels.data());

    std::size_t correct = 0;
    double distance_checksum = 0;
    for (std::size_t i = 0; i < kVectorCount; ++i) {
        correct += labels[i] == static_cast<faiss::idx_t>(i);
        distance_checksum += distances[i];
    }

    const double recall = static_cast<double>(correct) / static_cast<double>(kVectorCount);
    const auto insert_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_begin).count();
    const double vectors_per_second = static_cast<double>(kVectorCount) * 1'000'000'000.0 / static_cast<double>(insert_ns);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "status=" << (recall >= kMinimumSelfRecall ? "PASS" : "FAIL") << '\n';
    std::cout << "architecture=arm64-apple\n";
    std::cout << "algorithm=faiss::IndexHNSWFlat\n";
    std::cout << "faiss_version=1.15.0\n";
    std::cout << "threads=" << omp_get_max_threads() << '\n';
    std::cout << "vectors=" << kVectorCount << '\n';
    std::cout << "dimensions=" << kDimension << '\n';
    std::cout << "index_size=" << index.ntotal << '\n';
    std::cout << "insert_nanoseconds=" << insert_ns << '\n';
    std::cout << "insert_vectors_per_second=" << vectors_per_second << '\n';
    std::cout << "self_recall_at_1=" << recall << '\n';
    std::cout << "distance_checksum=" << distance_checksum << '\n';

    return recall >= kMinimumSelfRecall && index.ntotal == static_cast<faiss::idx_t>(kVectorCount) ? 0 : 1;
}
