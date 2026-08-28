import std;
import std.compat;
import infinity_core;

namespace {

constexpr size_t kDimension = 19;
constexpr size_t kChunkSize = 128;
constexpr size_t kChunkCount = 10;
constexpr size_t kVectorCount = kChunkSize * kChunkCount;
constexpr size_t kM = 8;
constexpr size_t kEfConstruction = 200;
constexpr size_t kEfSearch = 10;
constexpr double kMinimumSelfRecall = 0.95;

uint64_t HashBytes(std::string_view bytes) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

int main() {
#if !defined(__APPLE__) || !defined(__aarch64__)
    std::cerr << "native smoke requires arm64 macOS\n";
    return 64;
#endif

    using Label = infinity::u64;
    using Hnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, Label>;

    std::vector<float> data(kVectorCount * kDimension);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> distribution;
    for (float &value : data) {
        value = distribution(rng);
    }

    auto index = Hnsw::Make(kChunkSize, kChunkCount, kDimension, kM, kEfConstruction);
    infinity::DenseVectorIter<float, Label> iterator(data.data(), kDimension, kVectorCount);

    const auto insert_begin = std::chrono::steady_clock::now();
    const auto [first, last] = index->InsertVecs(std::move(iterator), {.optimize_ = true});
    const auto insert_end = std::chrono::steady_clock::now();

    index->Check();

    const infinity::KnnSearchOption search_option{.ef_ = kEfSearch};
    size_t correct = 0;
    double distance_checksum = 0;
    for (size_t i = 0; i < kVectorCount; ++i) {
        const float *query = data.data() + i * kDimension;
        const auto result = index->KnnSearchSorted(query, 1, search_option);
        if (!result.empty()) {
            distance_checksum += result.front().first;
            correct += result.front().second == static_cast<Label>(i);
        }
    }

    const double recall = static_cast<double>(correct) / static_cast<double>(kVectorCount);
    const auto insert_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_begin).count();
    const double vectors_per_second = static_cast<double>(kVectorCount) * 1'000'000'000.0 / static_cast<double>(insert_ns);
    std::ostringstream dump;
    index->Dump(dump);
    const uint64_t index_dump_hash = HashBytes(dump.view());

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "status=" << (recall >= kMinimumSelfRecall ? "PASS" : "FAIL") << '\n';
    std::cout << "architecture=arm64-apple\n";
    std::cout << "algorithm=infinity::KnnHnsw<PlainL2VecStoreType<float>,uint64_t>\n";
    std::cout << "vectors=" << kVectorCount << '\n';
    std::cout << "dimensions=" << kDimension << '\n';
    std::cout << "insert_range=" << first << ':' << last << '\n';
    std::cout << "index_size=" << index->GetVecNum() << '\n';
    std::cout << "insert_nanoseconds=" << insert_ns << '\n';
    std::cout << "insert_vectors_per_second=" << vectors_per_second << '\n';
    std::cout << "self_recall_at_1=" << recall << '\n';
    std::cout << "distance_checksum=" << distance_checksum << '\n';
    std::cout << "index_dump_fnv1a64=" << index_dump_hash << '\n';

    return recall >= kMinimumSelfRecall && first == 0 && last == kVectorCount && index->GetVecNum() == kVectorCount ? 0 : 1;
}
