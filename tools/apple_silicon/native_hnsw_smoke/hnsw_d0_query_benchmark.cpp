#include "hnsw_d0_query_benchmark.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <exception>
#include <latch>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace {

constexpr std::uint64_t kFnv1a64Offset = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1a64Prime = 1099511628211ULL;

void Require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::uint64_t SplitMix64(std::uint64_t &state) {
    std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

float NextQueryValue(std::uint64_t &state) {
    const std::uint32_t numerator = static_cast<std::uint32_t>(SplitMix64(state) >> 40U);
    return static_cast<float>(numerator) * (1.0F / 16777216.0F);
}

void HashByte(std::uint64_t &hash, std::uint8_t value) {
    hash ^= value;
    hash *= kFnv1a64Prime;
}

void HashText(std::uint64_t &hash, std::string_view value) {
    for (const char byte : value) {
        HashByte(hash, static_cast<std::uint8_t>(byte));
    }
}

template <typename Integer>
void HashLittleEndian(std::uint64_t &hash, Integer value) {
    using Unsigned = std::make_unsigned_t<Integer>;
    const Unsigned bits = static_cast<Unsigned>(value);
    for (std::size_t offset = 0; offset < sizeof(bits); ++offset) {
        HashByte(hash, static_cast<std::uint8_t>(bits >> (offset * 8U)));
    }
}

std::uint64_t QueryResultChecksum(std::size_t query_index,
                                  std::size_t vector_count,
                                  const std::vector<std::pair<float, std::int64_t>> &results) {
    Require(results.size() == kHnswD0QueryK, "HNSW query benchmark search returned the wrong result count");
    float previous_distance = -std::numeric_limits<float>::infinity();
    std::uint64_t checksum = kFnv1a64Offset;
    HashText(checksum, "infinity-hnsw-d0-query-result-v1");
    HashLittleEndian(checksum, static_cast<std::uint64_t>(query_index));
    HashLittleEndian(checksum, static_cast<std::uint64_t>(results.size()));
    for (std::size_t rank = 0; rank < results.size(); ++rank) {
        const auto [distance, label] = results[rank];
        Require(std::isfinite(distance) && distance >= 0.0F, "HNSW query benchmark search returned an invalid distance");
        Require(distance >= previous_distance, "HNSW query benchmark search results are not sorted");
        Require(label >= 0 && static_cast<std::size_t>(label) < vector_count,
                "HNSW query benchmark search returned an out-of-range label");
        Require(std::none_of(results.begin(),
                             results.begin() + static_cast<std::ptrdiff_t>(rank),
                             [label](const auto &previous) { return previous.second == label; }),
                "HNSW query benchmark search returned a duplicate label");
        previous_distance = distance;
        HashLittleEndian(checksum, label);
        HashLittleEndian(checksum, std::bit_cast<std::uint32_t>(distance));
    }
    return checksum;
}

std::uint64_t AggregateChecksum(const std::array<std::uint64_t, kHnswD0HeldOutQueryCount> &per_query) {
    std::uint64_t checksum = kFnv1a64Offset;
    HashText(checksum, "infinity-hnsw-d0-query-benchmark-v1");
    HashLittleEndian(checksum, static_cast<std::uint64_t>(kHnswD0HeldOutQueryCount));
    HashLittleEndian(checksum, static_cast<std::uint64_t>(kHnswD0QueryK));
    HashLittleEndian(checksum, static_cast<std::uint64_t>(kHnswD0QueryEf));
    for (std::size_t query_index = 0; query_index < per_query.size(); ++query_index) {
        HashLittleEndian(checksum, static_cast<std::uint64_t>(query_index));
        HashLittleEndian(checksum, per_query[query_index]);
    }
    return checksum;
}

void RecordChecksum(std::array<std::uint64_t, kHnswD0HeldOutQueryCount> &checksums,
                    std::array<bool, kHnswD0HeldOutQueryCount> &observed,
                    std::size_t query_index,
                    std::uint64_t checksum) {
    if (!observed[query_index]) {
        checksums[query_index] = checksum;
        observed[query_index] = true;
        return;
    }
    Require(checksums[query_index] == checksum, "HNSW query benchmark returned nondeterministic results");
}

std::uint64_t ValidatedPhaseChecksum(
    std::uint64_t validated_operations,
    std::uint64_t expected_operations,
    const std::array<std::uint64_t, kHnswD0HeldOutQueryCount> &per_query_checksums,
    std::string_view operation_count_error) {
    Require(validated_operations == expected_operations, operation_count_error);
    return AggregateChecksum(per_query_checksums);
}

struct ThroughputMeasurement {
    std::uint64_t wall_ns = 0;
    std::uint64_t validated_operations = 0;
};

ThroughputMeasurement RunConcurrentThroughputQueries(
    const std::vector<float> &queries,
    std::size_t dimension,
    std::size_t vector_count,
    std::size_t warmup_operation_count,
    std::size_t measured_operation_count,
    const std::array<std::uint64_t, kHnswD0HeldOutQueryCount> &expected_checksums,
    const HnswD0Search &search) {
    std::latch ready(kHnswD0QueryThroughputConcurrency);
    std::latch warmup_start(1);
    std::latch warmup_completed(kHnswD0QueryThroughputConcurrency);
    std::latch measured_ready(kHnswD0QueryThroughputConcurrency);
    std::latch measured_start(1);
    std::latch measured_completed(kHnswD0QueryThroughputConcurrency);
    std::latch workers_may_exit(1);
    std::array<std::uint64_t, kHnswD0QueryThroughputConcurrency> warmup_validated{};
    std::array<std::uint64_t, kHnswD0QueryThroughputConcurrency> measured_validated{};
    std::mutex error_mutex;
    std::exception_ptr error;
    std::atomic<bool> cancel_workers{false};
    std::vector<std::thread> workers;
    workers.reserve(kHnswD0QueryThroughputConcurrency);
    try {
        for (std::size_t worker = 0; worker < kHnswD0QueryThroughputConcurrency; ++worker) {
            workers.emplace_back([&, worker] {
                ready.count_down();
                const auto run_phase = [&](std::size_t operation_count, std::uint64_t &worker_operations) {
                    for (std::size_t operation = worker; operation < operation_count;
                         operation += kHnswD0QueryThroughputConcurrency) {
                        const std::size_t query_index = operation % kHnswD0HeldOutQueryCount;
                        const float *query = queries.data() + query_index * dimension;
                        const auto returned = search(query, kHnswD0QueryK, kHnswD0QueryEf);
                        const std::uint64_t checksum = QueryResultChecksum(query_index, vector_count, returned);
                        Require(checksum == expected_checksums[query_index],
                                "HNSW throughput query returned results that differ from latency measurement");
                        ++worker_operations;
                    }
                };

                warmup_start.wait();
                if (cancel_workers.load(std::memory_order_acquire)) {
                    return;
                }
                try {
                    run_phase(warmup_operation_count, warmup_validated[worker]);
                } catch (...) {
                    std::lock_guard lock(error_mutex);
                    if (error == nullptr) {
                        error = std::current_exception();
                    }
                }
                warmup_completed.count_down();

                measured_ready.count_down();
                measured_start.wait();
                try {
                    run_phase(measured_operation_count, measured_validated[worker]);
                } catch (...) {
                    std::lock_guard lock(error_mutex);
                    if (error == nullptr) {
                        error = std::current_exception();
                    }
                }
                measured_completed.count_down();
                workers_may_exit.wait();
            });
        }
    } catch (...) {
        cancel_workers.store(true, std::memory_order_release);
        warmup_start.count_down();
        measured_start.count_down();
        workers_may_exit.count_down();
        for (std::thread &worker : workers) {
            worker.join();
        }
        throw;
    }

    ready.wait();
    warmup_start.count_down();
    warmup_completed.wait();
    measured_ready.wait();

    const auto begin = std::chrono::steady_clock::now();
    measured_start.count_down();
    measured_completed.wait();
    const auto end = std::chrono::steady_clock::now();
    workers_may_exit.count_down();
    for (std::thread &worker : workers) {
        worker.join();
    }
    if (error != nullptr) {
        std::rethrow_exception(error);
    }
    const std::uint64_t warmup_validated_total =
        std::accumulate(warmup_validated.begin(), warmup_validated.end(), std::uint64_t{0});
    Require(warmup_validated_total == warmup_operation_count,
            "HNSW throughput benchmark validated the wrong warmup operation count");
    const std::uint64_t measured_validated_total =
        std::accumulate(measured_validated.begin(), measured_validated.end(), std::uint64_t{0});
    Require(measured_validated_total == measured_operation_count,
            "HNSW throughput benchmark validated the wrong measured operation count");
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    Require(elapsed > 0, "HNSW throughput benchmark wall duration is not positive");
    return ThroughputMeasurement{
        .wall_ns = static_cast<std::uint64_t>(elapsed),
        .validated_operations = measured_validated_total,
    };
}

} // namespace

std::vector<float> GenerateHnswD0HeldOutQueries(std::size_t dimension) {
    Require(dimension > 0, "HNSW query benchmark dimension is invalid");
    Require(dimension <= std::numeric_limits<std::size_t>::max() / kHnswD0HeldOutQueryCount,
            "HNSW query benchmark query storage overflows");
    std::vector<float> queries(kHnswD0HeldOutQueryCount * dimension);
    std::uint64_t state = kHnswD0HeldOutQuerySeed;
    for (float &value : queries) {
        value = NextQueryValue(state);
    }
    return queries;
}

HnswD0QueryBenchmark BenchmarkHnswD0Queries(const std::vector<float> &queries,
                                            std::size_t dimension,
                                            std::size_t vector_count,
                                            const HnswD0Search &search) {
    Require(dimension > 0, "HNSW query benchmark dimension is invalid");
    Require(vector_count >= kHnswD0QueryK, "HNSW query benchmark vector count is too small");
    Require(queries.size() == kHnswD0HeldOutQueryCount * dimension, "HNSW query benchmark query corpus has the wrong size");
    Require(bool(search), "HNSW query benchmark search callback is missing");

    HnswD0QueryBenchmark result;
    result.timed_query_corpus_sha256 = HnswD0QueryCorpusSha256(queries, dimension);
    std::array<bool, kHnswD0HeldOutQueryCount> observed{};
    for (std::size_t pass = 0; pass < kHnswD0QueryWarmupPasses; ++pass) {
        for (std::size_t query_index = 0; query_index < kHnswD0HeldOutQueryCount; ++query_index) {
            const auto returned = search(queries.data() + query_index * dimension, kHnswD0QueryK, kHnswD0QueryEf);
            RecordChecksum(result.per_query_checksums,
                           observed,
                           query_index,
                           QueryResultChecksum(query_index, vector_count, returned));
        }
    }
    Require(std::all_of(observed.begin(), observed.end(), [](bool value) { return value; }),
            "HNSW query benchmark warmup did not observe every query");

    result.latency_samples_ns.reserve(kHnswD0QueryLatencySampleCount);
    for (std::size_t pass = 0; pass < kHnswD0QueryMeasuredPasses; ++pass) {
        for (std::size_t query_index = 0; query_index < kHnswD0HeldOutQueryCount; ++query_index) {
            const float *query = queries.data() + query_index * dimension;
            const auto begin = std::chrono::steady_clock::now();
            const auto returned = search(query, kHnswD0QueryK, kHnswD0QueryEf);
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            Require(elapsed >= 0, "HNSW query benchmark latency clock moved backwards");
            result.latency_samples_ns.push_back(static_cast<std::uint64_t>(elapsed));
            RecordChecksum(result.per_query_checksums,
                           observed,
                           query_index,
                           QueryResultChecksum(query_index, vector_count, returned));
            ++result.latency_validated_operations;
        }
    }
    Require(result.latency_samples_ns.size() == kHnswD0QueryLatencySampleCount,
            "HNSW query benchmark retained the wrong latency sample count");
    result.latency_validated_result_checksum =
        ValidatedPhaseChecksum(result.latency_validated_operations,
                               kHnswD0QueryLatencySampleCount,
                               result.per_query_checksums,
                               "HNSW query benchmark validated the wrong latency operation count");

    result.throughput_operations = kHnswD0QueryThroughputOperations;
    const ThroughputMeasurement throughput =
        RunConcurrentThroughputQueries(queries,
                                       dimension,
                                       vector_count,
                                       kHnswD0HeldOutQueryCount * kHnswD0QueryWarmupPasses,
                                       kHnswD0QueryThroughputOperations,
                                       result.per_query_checksums,
                                       search);
    result.throughput_wall_ns = throughput.wall_ns;
    result.throughput_validated_operations = throughput.validated_operations;
    Require(HnswD0QueryCorpusSha256(queries, dimension) == result.timed_query_corpus_sha256,
            "HNSW query benchmark corpus changed during measurement");
    result.throughput_validated_result_checksum =
        ValidatedPhaseChecksum(result.throughput_validated_operations,
                               result.throughput_operations,
                               result.per_query_checksums,
                               "HNSW query benchmark validated the wrong throughput operation count");
    result.result_checksum = AggregateChecksum(result.per_query_checksums);
    Require(result.latency_validated_result_checksum == result.result_checksum &&
                result.throughput_validated_result_checksum == result.result_checksum,
            "HNSW query benchmark phase result checksums differ");
    result.valid = true;
    return result;
}

bool HnswD0QueryBenchmarkMatchesRecall(const HnswD0QueryBenchmark &benchmark,
                                       const HnswD0RecallAudit &recall,
                                       std::size_t vector_count) {
    if (!benchmark.valid || !recall.valid || vector_count < kHnswD0QueryK ||
        benchmark.timed_query_corpus_sha256 != recall.queries_sha256 ||
        benchmark.latency_samples_ns.size() != kHnswD0QueryLatencySampleCount ||
        benchmark.latency_validated_operations != kHnswD0QueryLatencySampleCount ||
        benchmark.throughput_operations != kHnswD0QueryThroughputOperations ||
        benchmark.throughput_validated_operations != benchmark.throughput_operations) {
        return false;
    }
    std::size_t point = 0;
    while (point < kHnswD0RecallK.size() &&
           (kHnswD0RecallK[point] != kHnswD0QueryK || kHnswD0RecallEf[point] != kHnswD0QueryEf)) {
        ++point;
    }
    if (point == kHnswD0RecallK.size() ||
        recall.returned_results[point].size() != kHnswD0HeldOutQueryCount * kHnswD0QueryK) {
        return false;
    }
    try {
        for (std::size_t query_index = 0; query_index < kHnswD0HeldOutQueryCount; ++query_index) {
            std::vector<std::pair<float, std::int64_t>> returned;
            returned.reserve(kHnswD0QueryK);
            const std::size_t begin = query_index * kHnswD0QueryK;
            for (std::size_t rank = 0; rank < kHnswD0QueryK; ++rank) {
                const HnswD0ReturnedResult &item = recall.returned_results[point][begin + rank];
                returned.emplace_back(item.distance, item.label);
            }
            if (QueryResultChecksum(query_index, vector_count, returned) !=
                benchmark.per_query_checksums[query_index]) {
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    const std::uint64_t aggregate_checksum = AggregateChecksum(benchmark.per_query_checksums);
    return aggregate_checksum == benchmark.result_checksum &&
           benchmark.latency_validated_result_checksum == aggregate_checksum &&
           benchmark.throughput_validated_result_checksum == aggregate_checksum;
}
