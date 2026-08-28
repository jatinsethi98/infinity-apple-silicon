#pragma once

#include "hnsw_d0_audit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

inline constexpr std::uint32_t kHnswD0QueryBenchmarkSchema = 2;
inline constexpr std::size_t kHnswD0QueryK = 10;
inline constexpr std::size_t kHnswD0QueryEf = 256;
inline constexpr double kHnswD0QueryRecallFloor = 0.99;
inline constexpr double kHnswD0QueryMaximumAbsoluteRecallGap = 0.005;
inline constexpr std::size_t kHnswD0QueryWarmupPasses = 16;
inline constexpr std::size_t kHnswD0QueryMeasuredPasses = 160;
inline constexpr std::size_t kHnswD0QueryLatencyConcurrency = 1;
inline constexpr std::size_t kHnswD0QueryThroughputConcurrency = 12;
inline constexpr std::size_t kHnswD0QueryLatencySampleCount =
    kHnswD0HeldOutQueryCount * kHnswD0QueryMeasuredPasses;
inline constexpr std::size_t kHnswD0QueryThroughputOperations =
    kHnswD0HeldOutQueryCount * kHnswD0QueryMeasuredPasses;
inline constexpr std::uint32_t kHnswD0NearestRankMethod = 1;
inline constexpr std::uint32_t kHnswD0ProcessLocalTopKTransaction = 1;

struct HnswD0QueryBenchmark {
    bool valid = false;
    std::string timed_query_corpus_sha256;
    std::vector<std::uint64_t> latency_samples_ns;
    std::uint64_t latency_validated_operations = 0;
    std::uint64_t latency_validated_result_checksum = 0;
    std::uint64_t throughput_operations = 0;
    std::uint64_t throughput_wall_ns = 0;
    std::uint64_t throughput_validated_operations = 0;
    std::uint64_t throughput_validated_result_checksum = 0;
    std::uint64_t result_checksum = 0;
    std::array<std::uint64_t, kHnswD0HeldOutQueryCount> per_query_checksums{};
};

std::vector<float> GenerateHnswD0HeldOutQueries(std::size_t dimension);

HnswD0QueryBenchmark BenchmarkHnswD0Queries(const std::vector<float> &queries,
                                            std::size_t dimension,
                                            std::size_t vector_count,
                                            const HnswD0Search &search);

bool HnswD0QueryBenchmarkMatchesRecall(const HnswD0QueryBenchmark &benchmark,
                                       const HnswD0RecallAudit &recall,
                                       std::size_t vector_count);
