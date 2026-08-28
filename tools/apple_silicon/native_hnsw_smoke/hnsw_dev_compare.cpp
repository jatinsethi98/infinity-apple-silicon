#include "hnsw_dev_bridge.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

constexpr unsigned long long kDefaultVectors = 10'000;
constexpr unsigned long long kDefaultDimensions = 16;
constexpr int kDefaultM = 32;
constexpr int kDefaultEfConstruction = 200;
constexpr int kDefaultEfSearch = 32;
constexpr int kDefaultParticipantCount = 1;
constexpr unsigned long long kDefaultChunkSize = 8192;
constexpr unsigned long long kDefaultQueryCount = 1000;

unsigned long long ParseUnsigned(const char *value) { return std::strtoull(value, nullptr, 10); }

int ParseInt(const char *value) { return static_cast<int>(std::strtol(value, nullptr, 10)); }

bool IsPowerOfTwo(unsigned long long value) { return value != 0 && (value & (value - 1)) == 0; }

std::uint64_t HashData(const std::vector<float> &data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (float value : data) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= (bits >> shift) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

void PrintResult(std::string_view name, const HnswDevResult &result) {
    std::cout << name << "_valid=" << result.valid << '\n';
    std::cout << name << "_threads=" << result.thread_count << '\n';
    std::cout << name << "_index_size=" << result.index_size << '\n';
    std::cout << name << "_cold_build_ns=" << result.cold_build_ns << '\n';
    std::cout << name << "_insert_call_ns=" << result.insert_call_ns << '\n';
    std::cout << name << "_self_recall_at_1=" << result.self_recall_at_1 << '\n';
    std::cout << name << "_distance_checksum=" << result.distance_checksum << '\n';
}

} // namespace

int main(int argc, char **argv) {
    HnswDevConfig config{
        .vector_count = argc > 1 ? ParseUnsigned(argv[1]) : kDefaultVectors,
        .dimension = argc > 2 ? ParseUnsigned(argv[2]) : kDefaultDimensions,
        .chunk_size = argc > 6 ? ParseUnsigned(argv[6]) : kDefaultChunkSize,
        .query_count = argc > 7 ? ParseUnsigned(argv[7]) : kDefaultQueryCount,
        .m = argc > 3 ? ParseInt(argv[3]) : kDefaultM,
        .ef_construction = argc > 4 ? ParseInt(argv[4]) : kDefaultEfConstruction,
        .ef_search = argc > 5 ? ParseInt(argv[5]) : kDefaultEfSearch,
        .participant_count = argc > 9 ? ParseInt(argv[9]) : kDefaultParticipantCount,
        .build_grain = argc > 10 ? ParseUnsigned(argv[10]) : 0,
    };
    const std::string_view order = argc > 8 ? argv[8] : "infinity-first";

    if (config.vector_count == 0 || config.dimension == 0 || config.m < 2 || config.ef_construction < config.m || config.ef_search < 1 ||
        config.participant_count < 1 || config.build_grain != 0 || !IsPowerOfTwo(config.chunk_size) ||
        (order != "infinity-first" && order != "faiss-first" && order != "infinity-only" && order != "faiss-only")) {
        std::cerr << "usage: hnsw_native_dev_compare [vectors] [dimensions] [M] [efConstruction] "
                     "[efSearch] [chunkSize] [queryCount] "
                     "[infinity-first|faiss-first|infinity-only|faiss-only] "
                     "[participants] [reservedBuildGrainMustBe0]\n";
        return 64;
    }

    std::vector<float> data(static_cast<std::size_t>(config.vector_count * config.dimension));
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> distribution;
    for (float &value : data) {
        value = distribution(rng);
    }

    HnswDevResult infinity_result{};
    HnswDevResult faiss_result{};
    int infinity_exit = 0;
    int faiss_exit = 0;
    const bool run_infinity = order != "faiss-only";
    const bool run_faiss = order != "infinity-only";
    if (order == "infinity-first") {
        infinity_exit = RunInfinityHnsw(data.data(), &config, &infinity_result);
        faiss_exit = RunFaissHnsw(data.data(), &config, &faiss_result);
    } else if (order == "faiss-first") {
        faiss_exit = RunFaissHnsw(data.data(), &config, &faiss_result);
        infinity_exit = RunInfinityHnsw(data.data(), &config, &infinity_result);
    } else if (run_infinity) {
        infinity_exit = RunInfinityHnsw(data.data(), &config, &infinity_result);
    } else {
        faiss_exit = RunFaissHnsw(data.data(), &config, &faiss_result);
    }

    const bool passed = (!run_infinity || infinity_exit == 0) && (!run_faiss || faiss_exit == 0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "status=" << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "scope=development-only\n";
    std::cout << "order=" << order << '\n';
    std::cout << "seed=0\n";
    std::cout << "data_fnv1a64=" << HashData(data) << '\n';
    std::cout << "vectors=" << config.vector_count << '\n';
    std::cout << "dimensions=" << config.dimension << '\n';
    std::cout << "M=" << config.m << '\n';
    std::cout << "ef_construction=" << config.ef_construction << '\n';
    std::cout << "ef_search=" << config.ef_search << '\n';
    std::cout << "participants=" << config.participant_count << '\n';
    std::cout << "build_grain=" << config.build_grain << '\n';
    if (run_infinity) {
        PrintResult("infinity", infinity_result);
    }
    if (run_faiss) {
        PrintResult("faiss", faiss_result);
    }
    if (run_infinity && run_faiss) {
        const double cold_ratio = static_cast<double>(faiss_result.cold_build_ns) / static_cast<double>(infinity_result.cold_build_ns);
        const double insert_ratio = static_cast<double>(faiss_result.insert_call_ns) / static_cast<double>(infinity_result.insert_call_ns);
        std::cout << "cold_build_ratio=" << cold_ratio << '\n';
        std::cout << "insert_call_ratio=" << insert_ratio << '\n';
    }

    return passed ? 0 : 1;
}
