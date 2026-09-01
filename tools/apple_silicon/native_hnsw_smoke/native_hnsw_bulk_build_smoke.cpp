import std;
import std.compat;
import infinity_core;

namespace {

constexpr std::size_t kVectorCount = 12'288;
constexpr std::size_t kDimension = 128;
constexpr std::size_t kChunkSize = 8'192;
constexpr std::size_t kChunkCount = 2;
constexpr std::size_t kM = 32;
constexpr std::size_t kEfConstruction = 200;
constexpr std::size_t kEfSearch = 32;
constexpr std::size_t kParticipantCount = 12;
// Bucket size is max(minimum_bucket_size, stored/(workers*buckets_per_worker)).
// The production floor of 1024 would dominate at this vector count and collapse the
// build to one task per worker, so this smoke would never exercise the multi-bucket
// scheduling it is meant to cover. Lower the floor instead of inflating the vector
// count: keeping 12,288 vectors keeps the smoke fast, and the scheduling path is the
// thing under test, not the scale.
constexpr std::size_t kMinimumBucketSize = 128;
// Opt in explicitly rather than relying on kHnswBuildBucketsPerWorker, whose
// shipping default is 1. This smoke exists to cover the multi-bucket scheduling
// path, so it must request it regardless of what the default happens to be.
constexpr std::size_t kBucketsPerWorker = 8;
// Derived from the builder's own helper so it cannot drift from the real formula.
constexpr std::size_t kExpectedTaskCount =
    (kVectorCount - 1)
        / infinity::HnswBuildBucketSize(kVectorCount, kParticipantCount, kMinimumBucketSize, kBucketsPerWorker)
    + 1;
static_assert(kExpectedTaskCount > kParticipantCount, "smoke must exercise more than one bucket per worker");
constexpr std::size_t kQueryCount = 1'000;
constexpr double kMinimumSelfRecall = 0.95;

} // namespace

int main() {
#if !defined(__APPLE__) || !defined(__aarch64__)
    std::cerr << "native bulk-build smoke requires arm64 macOS\n";
    return 64;
#endif

    using Label = infinity::i32;
    using Hnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, Label>;

    std::vector<float> data(kVectorCount * kDimension);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> distribution;
    for (float &value : data) {
        value = distribution(rng);
    }

    const auto cold_begin = std::chrono::steady_clock::now();
    infinity::ctpl::thread_pool build_pool(kParticipantCount);
    auto index = Hnsw::Make(kChunkSize, kChunkCount, kDimension, kM, kEfConstruction);
    infinity::DenseVectorIter<float, Label> iterator(data.data(), kDimension, kVectorCount);
    const auto bulk_begin = std::chrono::steady_clock::now();
    const infinity::HnswBulkBuildResult build_result =
        infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, build_pool, kMinimumBucketSize, kBucketsPerWorker);
    const auto bulk_end = std::chrono::steady_clock::now();

    index->Check();

    const infinity::KnnSearchOption search_option{.ef_ = kEfSearch};
    std::size_t correct = 0;
    double distance_checksum = 0;
    for (std::size_t query_index = 0; query_index < kQueryCount; ++query_index) {
        const auto result = index->KnnSearchSorted(data.data() + query_index * kDimension, 1, search_option);
        if (!result.empty()) {
            correct += result.front().second == static_cast<Label>(query_index);
            distance_checksum += result.front().first;
        }
    }

    const double recall = static_cast<double>(correct) / static_cast<double>(kQueryCount);
    const auto cold_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_end - cold_begin).count();
    const auto bulk_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_end - bulk_begin).count();
    const bool passed = build_pool.size() == static_cast<int>(kParticipantCount) && build_result.start_ == 0 && build_result.end_ == kVectorCount &&
                        build_result.submitted_task_count_ == kExpectedTaskCount && build_result.mem_usage_ > 0 &&
                        index->GetVecNum() == kVectorCount && recall >= kMinimumSelfRecall && std::isfinite(distance_checksum);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "status=" << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "scope=contract-approved-production-bulk-build-path\n";
    std::cout << "architecture=arm64-apple\n";
    std::cout << "vectors=" << kVectorCount << '\n';
    std::cout << "dimensions=" << kDimension << '\n';
    std::cout << "M=" << kM << '\n';
    std::cout << "ef_construction=" << kEfConstruction << '\n';
    std::cout << "requested_participants=" << kParticipantCount << '\n';
    std::cout << "created_participants=" << build_pool.size() << '\n';
    std::cout << "submitted_tasks=" << build_result.submitted_task_count_ << '\n';
    std::cout << "insert_range=" << build_result.start_ << ':' << build_result.end_ << '\n';
    std::cout << "index_size=" << index->GetVecNum() << '\n';
    std::cout << "memory_delta_bytes=" << build_result.mem_usage_ << '\n';
    std::cout << "cold_build_nanoseconds=" << cold_ns << '\n';
    std::cout << "bulk_build_nanoseconds=" << bulk_ns << '\n';
    std::cout << "self_recall_at_1=" << recall << '\n';
    std::cout << "distance_checksum=" << distance_checksum << '\n';

    return passed ? 0 : 1;
}
