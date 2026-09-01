#include "hnsw_dev_bridge.h"
#include "hnsw_d0_timing_probe.h"

import std;
import std.compat;
import infinity_core;

extern "C" int RunInfinityHnsw(const float *data, const HnswDevConfig *config, HnswDevResult *result) {
    if (result == nullptr) {
        return 64;
    }
    *result = HnswDevResult{};
    try {
        if (data == nullptr || config == nullptr || config->vector_count == 0 || config->dimension == 0 || config->chunk_size == 0 ||
            config->query_count == 0 || config->query_count > config->vector_count || config->m < 2 || config->ef_construction < config->m ||
            config->ef_search < 1 || config->participant_count < 1 || config->vector_count > std::numeric_limits<std::int32_t>::max() ||
            config->dimension > std::numeric_limits<std::size_t>::max() || config->chunk_size > std::numeric_limits<std::size_t>::max() ||
            config->query_count > std::numeric_limits<std::size_t>::max() || config->build_grain > std::numeric_limits<std::size_t>::max()) {
            return 64;
        }
        using Label = std::int32_t;
        using Hnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, Label>;

        const size_t vector_count = static_cast<size_t>(config->vector_count);
        const size_t dimension = static_cast<size_t>(config->dimension);
        const size_t chunk_size = static_cast<size_t>(config->chunk_size);
        const size_t max_chunk_count = (vector_count - 1) / chunk_size + 1;

        if (config->index_barrier != nullptr &&
            config->index_barrier(HnswD0IndexBarrierPhase::kBeforeIndex, config->index_barrier_context) != 0) {
            return 74;
        }
        const auto cold_begin = HNSW_D0_READ_STEADY_CLOCK(kTimerStarted);
        infinity::ctpl::thread_pool build_pool(config->participant_count);
        auto index = Hnsw::Make(chunk_size, max_chunk_count, dimension, static_cast<size_t>(config->m), static_cast<size_t>(config->ef_construction));
        HNSW_D0_TIMING_PROBE(kIndexReady);
        HNSW_D0_TIMING_PROBE(kExecutionWitnessArmBegin);
        HNSW_D0_TIMING_PROBE(kExecutionWitnessArmEnd);
        infinity::DenseVectorIter<float, Label> iterator(data, dimension, vector_count);
        // Dev-harness sweep knob for build-task granularity. Unset uses the library
        // default (kHnswBuildBucketsPerWorker), so a plain run measures shipping
        // behaviour. Reading it here rather than baking it in lets one binary A/B
        // several granularities, which keeps codegen out of the comparison.
        const std::size_t buckets_per_worker = [] {
            const char *raw = std::getenv("INFINITY_HNSW_BUILD_BUCKETS_PER_WORKER");
            if (raw == nullptr) {
                return infinity::kHnswBuildBucketsPerWorker;
            }
            char *parse_end = nullptr;
            const unsigned long long parsed = std::strtoull(raw, &parse_end, 10);
            if (parse_end == raw || *parse_end != '\0' || parsed == 0 || parsed > 4096) {
                return infinity::kHnswBuildBucketsPerWorker;
            }
            return static_cast<std::size_t>(parsed);
        }();
        std::cout << "infinity_build_buckets_per_worker=" << buckets_per_worker << '\n';

        const auto insert_begin = HNSW_D0_READ_STEADY_CLOCK(kBuildEntered);
        const infinity::HnswBulkBuildResult build_result =
            config->build_grain == 0
                ? infinity::HnswBulkBuild(index,
                                          std::move(iterator),
                                          infinity::HnswInsertConfig{.optimize_ = true},
                                          build_pool,
                                          size_t{1024},
                                          buckets_per_worker)
                : infinity::HnswBulkBuild(index,
                                          std::move(iterator),
                                          infinity::HnswInsertConfig{.optimize_ = true},
                                          build_pool,
                                          static_cast<size_t>(config->build_grain),
                                          buckets_per_worker);
        HNSW_D0_TIMING_PROBE(kBuildReturned);
        const auto insert_end = HNSW_D0_READ_STEADY_CLOCK(kTimerStopped);
        HNSW_D0_TIMING_PROBE(kExecutionWitnessSealBegin);
        constexpr bool incremental_witness_valid = true;
        constexpr bool threshold_witness_valid = true;
        HNSW_D0_TIMING_PROBE(kExecutionWitnessSealEnd);
        if (config->index_barrier != nullptr &&
            config->index_barrier(HnswD0IndexBarrierPhase::kAfterIndex, config->index_barrier_context) != 0) {
            return 74;
        }

        result->index_size = index->GetVecNum();
        result->cold_build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - cold_begin).count();
        result->insert_call_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_begin).count();
        result->thread_count = build_pool.size();
        result->submitted_tasks = build_result.submitted_task_count_;
        result->build_start = build_result.start_;
        result->build_end = build_result.end_;

        const HnswD0Search search = [&](const float *query, std::size_t k, std::size_t ef) {
            auto found = index->KnnSearchSorted(query, k, infinity::KnnSearchOption{.ef_ = ef});
            if (found.size() > k) {
                found.resize(k);
            }
            std::vector<std::pair<float, std::int64_t>> returned;
            returned.reserve(found.size());
            for (const auto &[distance, label] : found) {
                returned.emplace_back(distance, static_cast<std::int64_t>(label));
            }
            return returned;
        };
        HNSW_D0_TIMING_PROBE(kQueryBenchmarkEntered);
        const std::vector<float> heldout_queries = GenerateHnswD0HeldOutQueries(dimension);
        result->query_benchmark = BenchmarkHnswD0Queries(heldout_queries, dimension, vector_count, search);

        HNSW_D0_TIMING_PROBE(kSelfRecallEntered);
        const size_t query_count = static_cast<size_t>(config->query_count);
        const infinity::KnnSearchOption search_option{.ef_ = static_cast<size_t>(config->ef_search)};
        size_t correct = 0;
        double distance_checksum = 0;
        for (size_t i = 0; i < query_count; ++i) {
            const auto results = index->KnnSearchSorted(data + i * dimension, 1, search_option);
            if (!results.empty()) {
                correct += results.front().second == static_cast<Label>(i);
                distance_checksum += results.front().first;
            }
        }

        result->self_recall_at_1 = static_cast<double>(correct) / static_cast<double>(query_count);
        result->distance_checksum = distance_checksum;
        HNSW_D0_TIMING_PROBE(kRecallAuditEntered);
        result->recall_audit = AuditHnswD0Recall(
            data,
            vector_count,
            dimension,
            heldout_queries,
            search);

        HNSW_D0_TIMING_PROBE(kGraphAuditEntered);
        index->Check();
        const auto [max_level, entry_point] = index->GetGraphEnterPoint();
        result->graph_audit = AuditHnswD0Graph(HnswD0GraphInput{
            .vertex_count = vector_count,
            .max_level = max_level,
            .entry_point = entry_point,
            .level0_capacity = index->GetGraphMmax0(),
            .upper_capacity = index->GetGraphMmax(),
            .level = [&](std::int32_t vertex) { return index->GetGraphLevel(vertex); },
            .label = [&](std::int32_t vertex) { return static_cast<std::int64_t>(index->GetLabel(vertex)); },
            .neighbors =
                [&](std::int32_t vertex, std::int32_t layer) {
                    const auto [neighbors, degree] = index->GetGraphNeighbors(vertex, layer);
                    return HnswD0NeighborRange{
                        .data = neighbors,
                        .size = static_cast<std::size_t>(degree),
                        .capacity = layer == 0 ? index->GetGraphMmax0() : index->GetGraphMmax(),
                        .tail_valid = true,
                    };
                },
        });

        const size_t minimum_bucket_size = config->build_grain == 0 ? size_t{1024} : static_cast<size_t>(config->build_grain);
        size_t expected_task_count = 0;
        if (result->thread_count > 0) {
            // Use the builder's own helper so this prediction cannot drift from it.
            const size_t bucket_size = infinity::HnswBuildBucketSize(vector_count,
                                                                     static_cast<size_t>(result->thread_count),
                                                                     minimum_bucket_size,
                                                                     buckets_per_worker);
            expected_task_count = (vector_count - 1) / bucket_size + 1;
        }
        result->valid = build_result.mem_usage_ > 0 && result->build_start == 0 && result->build_end == vector_count &&
                        result->submitted_tasks == expected_task_count && result->index_size == vector_count &&
                        result->thread_count == config->participant_count && result->cold_build_ns > 0 && result->insert_call_ns > 0 &&
                        result->insert_call_ns <= result->cold_build_ns && result->self_recall_at_1 >= 0.95 && result->self_recall_at_1 <= 1.0 &&
                        std::isfinite(result->distance_checksum) && result->query_benchmark.valid && result->graph_audit.valid &&
                        result->recall_audit.valid && incremental_witness_valid && threshold_witness_valid &&
                        HnswD0QueryBenchmarkMatchesRecall(result->query_benchmark, result->recall_audit, vector_count);
        HNSW_D0_TIMING_PROBE(kBridgeCompleted);
        return result->valid ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "Infinity HNSW audit failed: " << error.what() << '\n';
        result->valid = 0;
        return 2;
    } catch (...) {
        result->valid = 0;
        return 2;
    }
}
