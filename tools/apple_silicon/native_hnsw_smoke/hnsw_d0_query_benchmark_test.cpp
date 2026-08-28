#include "hnsw_d0_query_benchmark.h"

#include <atomic>
#include <bit>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct WorkerExitRecorder {
    std::atomic<std::size_t> *exit_count;

    ~WorkerExitRecorder() {
        exit_count->fetch_add(1, std::memory_order_relaxed);
    }
};

} // namespace

int main() {
    constexpr std::size_t kDimension = 2;
    constexpr std::size_t kVectorCount = 128;
    static_assert(kHnswD0QueryBenchmarkSchema == 2);
    const std::vector<float> first = GenerateHnswD0HeldOutQueries(kDimension);
    const std::vector<float> second = GenerateHnswD0HeldOutQueries(kDimension);
    if (first != second || first.size() != kHnswD0HeldOutQueryCount * kDimension ||
        std::bit_cast<std::uint32_t>(first.front()) != 0x3ec79f8cU) {
        std::cerr << "held-out query generation is not deterministic\n";
        return 1;
    }

    std::atomic<std::size_t> throughput_worker_exits{};
    std::atomic<std::size_t> active_workers{};
    std::atomic<std::size_t> maximum_active_workers{};
    std::atomic<std::size_t> search_calls{};
    const std::thread::id benchmark_thread = std::this_thread::get_id();
    const HnswD0Search search = [&](const float *, std::size_t k, std::size_t ef) {
        search_calls.fetch_add(1, std::memory_order_relaxed);
        if (k != kHnswD0QueryK || ef != kHnswD0QueryEf) {
            return std::vector<std::pair<float, std::int64_t>>{};
        }
        if (std::this_thread::get_id() != benchmark_thread) {
            thread_local WorkerExitRecorder exit_recorder{&throughput_worker_exits};
            if (throughput_worker_exits.load(std::memory_order_relaxed) != 0) {
                throw std::runtime_error("throughput worker exited between warmup and measurement");
            }
            const std::size_t active = active_workers.fetch_add(1, std::memory_order_acq_rel) + 1;
            std::size_t maximum = maximum_active_workers.load(std::memory_order_relaxed);
            while (maximum < active &&
                   !maximum_active_workers.compare_exchange_weak(
                       maximum, active, std::memory_order_relaxed, std::memory_order_relaxed)) {
            }
            std::this_thread::yield();
            active_workers.fetch_sub(1, std::memory_order_acq_rel);
        }
        std::vector<std::pair<float, std::int64_t>> result;
        result.reserve(k);
        for (std::size_t rank = 0; rank < k; ++rank) {
            result.emplace_back(static_cast<float>(rank), static_cast<std::int64_t>(rank));
        }
        return result;
    };
    const HnswD0QueryBenchmark benchmark = BenchmarkHnswD0Queries(first, kDimension, kVectorCount, search);
    if (!benchmark.valid || benchmark.latency_samples_ns.size() != kHnswD0QueryLatencySampleCount ||
        benchmark.timed_query_corpus_sha256 != HnswD0QueryCorpusSha256(first, kDimension) ||
        benchmark.latency_validated_operations != kHnswD0QueryLatencySampleCount ||
        benchmark.latency_validated_result_checksum != benchmark.result_checksum ||
        benchmark.throughput_operations != kHnswD0QueryThroughputOperations || benchmark.throughput_wall_ns == 0 ||
        benchmark.throughput_validated_operations != kHnswD0QueryThroughputOperations ||
        benchmark.throughput_validated_result_checksum != benchmark.result_checksum ||
        benchmark.result_checksum != 2528166704761894621ULL) {
        std::cerr << "query benchmark protocol fields are invalid\n";
        return 1;
    }
    constexpr std::size_t kExpectedSearchCalls =
        kHnswD0HeldOutQueryCount * kHnswD0QueryWarmupPasses +
        kHnswD0QueryLatencySampleCount +
        kHnswD0HeldOutQueryCount * kHnswD0QueryWarmupPasses +
        kHnswD0QueryThroughputOperations;
    if (search_calls.load(std::memory_order_relaxed) != kExpectedSearchCalls) {
        std::cerr << "query benchmark executed an unexpected search count\n";
        return 1;
    }
    if (throughput_worker_exits.load(std::memory_order_relaxed) != kHnswD0QueryThroughputConcurrency) {
        std::cerr << "query benchmark did not retain one worker cohort across throughput phases\n";
        return 1;
    }
    if (maximum_active_workers.load(std::memory_order_relaxed) < 2) {
        std::cerr << "query benchmark did not execute throughput workers concurrently\n";
        return 1;
    }

    constexpr std::size_t kCallsBeforeMeasuredLatency =
        kHnswD0HeldOutQueryCount * kHnswD0QueryWarmupPasses;
    std::atomic<std::size_t> latency_calls{};
    const HnswD0Search corrupt_measured_latency =
        [&latency_calls](const float *, std::size_t k, std::size_t) {
            const bool corrupt =
                latency_calls.fetch_add(1, std::memory_order_relaxed) == kCallsBeforeMeasuredLatency;
            std::vector<std::pair<float, std::int64_t>> result;
            result.reserve(k);
            for (std::size_t rank = 0; rank < k; ++rank) {
                const std::int64_t label =
                    corrupt && rank == 0 ? static_cast<std::int64_t>(k) : static_cast<std::int64_t>(rank);
                result.emplace_back(static_cast<float>(rank), label);
            }
            return result;
        };
    bool rejected_corrupt_measured_latency = false;
    try {
        static_cast<void>(
            BenchmarkHnswD0Queries(first, kDimension, kVectorCount, corrupt_measured_latency));
    } catch (const std::runtime_error &error) {
        rejected_corrupt_measured_latency =
            std::string_view(error.what()).find("nondeterministic results") != std::string_view::npos;
    }
    if (!rejected_corrupt_measured_latency) {
        std::cerr << "timed latency accepted a corrupt query result\n";
        return 1;
    }

    constexpr std::size_t kCallsBeforeConcurrentWarmup =
        kHnswD0HeldOutQueryCount * kHnswD0QueryWarmupPasses +
        kHnswD0QueryLatencySampleCount;
    constexpr std::size_t kCallsBeforeMeasuredThroughput =
        kCallsBeforeConcurrentWarmup +
        kHnswD0HeldOutQueryCount * kHnswD0QueryWarmupPasses;
    std::atomic<std::size_t> calls{};
    const HnswD0Search corrupt_measured_throughput =
        [&calls](const float *, std::size_t k, std::size_t) {
            const bool corrupt = calls.fetch_add(1, std::memory_order_relaxed) == kCallsBeforeMeasuredThroughput;
            std::vector<std::pair<float, std::int64_t>> result;
            result.reserve(k);
            for (std::size_t rank = 0; rank < k; ++rank) {
                const std::int64_t label =
                    corrupt && rank == 0 ? static_cast<std::int64_t>(k) : static_cast<std::int64_t>(rank);
                result.emplace_back(static_cast<float>(rank), label);
            }
            return result;
        };
    bool rejected_corrupt_measured_throughput = false;
    try {
        static_cast<void>(
            BenchmarkHnswD0Queries(first, kDimension, kVectorCount, corrupt_measured_throughput));
    } catch (const std::runtime_error &error) {
        rejected_corrupt_measured_throughput =
            std::string_view(error.what()).find("throughput query returned results") != std::string_view::npos;
    }
    if (!rejected_corrupt_measured_throughput) {
        std::cerr << "timed throughput accepted a corrupt query result\n";
        return 1;
    }

    HnswD0RecallAudit recall;
    recall.valid = true;
    recall.queries_sha256 = benchmark.timed_query_corpus_sha256;
    std::size_t query_point = 0;
    while (query_point < kHnswD0RecallK.size() &&
           (kHnswD0RecallK[query_point] != kHnswD0QueryK ||
            kHnswD0RecallEf[query_point] != kHnswD0QueryEf)) {
        ++query_point;
    }
    if (query_point == kHnswD0RecallK.size()) {
        std::cerr << "query benchmark recall point is missing\n";
        return 1;
    }
    auto &replay = recall.returned_results[query_point];
    replay.reserve(kHnswD0HeldOutQueryCount * kHnswD0QueryK);
    for (std::size_t query_index = 0; query_index < kHnswD0HeldOutQueryCount; ++query_index) {
        for (std::size_t rank = 0; rank < kHnswD0QueryK; ++rank) {
            replay.push_back(HnswD0ReturnedResult{
                .distance = static_cast<float>(rank),
                .label = static_cast<std::int64_t>(rank),
            });
        }
    }
    if (!HnswD0QueryBenchmarkMatchesRecall(benchmark, recall, kVectorCount)) {
        std::cerr << "valid query benchmark does not match its replay\n";
        return 1;
    }
    const auto rejects_attestation_mutation = [&](auto mutate) {
        HnswD0QueryBenchmark changed = benchmark;
        mutate(changed);
        return !HnswD0QueryBenchmarkMatchesRecall(changed, recall, kVectorCount);
    };
    if (!rejects_attestation_mutation(
            [](HnswD0QueryBenchmark &changed) { --changed.latency_validated_operations; }) ||
        !rejects_attestation_mutation(
            [](HnswD0QueryBenchmark &changed) { changed.latency_validated_result_checksum ^= 1; }) ||
        !rejects_attestation_mutation(
            [](HnswD0QueryBenchmark &changed) { --changed.throughput_validated_operations; }) ||
        !rejects_attestation_mutation(
            [](HnswD0QueryBenchmark &changed) { changed.throughput_validated_result_checksum ^= 1; }) ||
        !rejects_attestation_mutation([](HnswD0QueryBenchmark &changed) {
            changed.timed_query_corpus_sha256[0] = changed.timed_query_corpus_sha256[0] == '0' ? '1' : '0';
        })) {
        std::cerr << "query replay accepted a mutated phase attestation\n";
        return 1;
    }

    std::atomic<std::size_t> failing_calls{};
    std::atomic<std::size_t> failing_worker_exits{};
    const HnswD0Search fail_concurrent_warmup =
        [&](const float *, std::size_t k, std::size_t) {
            if (std::this_thread::get_id() != benchmark_thread) {
                thread_local WorkerExitRecorder exit_recorder{&failing_worker_exits};
            }
            const std::size_t call = failing_calls.fetch_add(1, std::memory_order_relaxed);
            if (call == kCallsBeforeConcurrentWarmup) {
                throw std::runtime_error("injected concurrent warmup failure");
            }
            std::vector<std::pair<float, std::int64_t>> result;
            result.reserve(k);
            for (std::size_t rank = 0; rank < k; ++rank) {
                result.emplace_back(static_cast<float>(rank), static_cast<std::int64_t>(rank));
            }
            return result;
        };
    bool propagated_concurrent_warmup_failure = false;
    try {
        static_cast<void>(BenchmarkHnswD0Queries(first, kDimension, kVectorCount, fail_concurrent_warmup));
    } catch (const std::runtime_error &error) {
        propagated_concurrent_warmup_failure =
            std::string_view(error.what()).find("injected concurrent warmup failure") != std::string_view::npos;
    }
    if (!propagated_concurrent_warmup_failure ||
        failing_worker_exits.load(std::memory_order_relaxed) != kHnswD0QueryThroughputConcurrency) {
        std::cerr << "concurrent warmup failure did not drain and join every worker\n";
        return 1;
    }
    return 0;
}
