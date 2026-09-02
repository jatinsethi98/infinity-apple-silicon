#include "hnsw_dev_bridge.h"
#include "hnsw_d0_timing_probe.h"
#include "hnsw_d0_external_truth.h"

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
        // Two graph-shape sweep knobs, read here rather than baked in so one binary can run
        // every arm and codegen stays out of the comparison -- same rationale as the build
        // granularity knob below. Both default to the library's historical behaviour, so an
        // unset environment measures shipping behaviour and reproduces the determinism gate.
        const float prune_headroom = [] {
            const char *raw = std::getenv("INFINITY_HNSW_PRUNE_HEADROOM");
            if (raw == nullptr) {
                return 0.0F;
            }
            char *parse_end = nullptr;
            const double parsed = std::strtod(raw, &parse_end);
            if (parse_end == raw || *parse_end != '\0' || !(parsed >= 0.0) || parsed > 0.5) {
                return 0.0F;
            }
            return static_cast<float>(parsed);
        }();
        const bool level0_double_budget = [] {
            const char *raw = std::getenv("INFINITY_HNSW_LEVEL0_DOUBLE_BUDGET");
            return raw != nullptr && std::strcmp(raw, "1") == 0;
        }();
        // Prefetch pipeline depth, in candidate vectors run ahead. 0/unset keeps the computed
        // default. Purely a memory hint, so no value can change the graph -- which makes the
        // graph-hash gate an exact check on this knob rather than merely a strong one.
        const std::size_t prefetch_step = [] {
            const char *raw = std::getenv("INFINITY_HNSW_PREFETCH_STEP");
            if (raw == nullptr) {
                return std::size_t{0};
            }
            char *parse_end = nullptr;
            const unsigned long long parsed = std::strtoull(raw, &parse_end, 10);
            if (parse_end == raw || *parse_end != '\0' || parsed > 4096) {
                return std::size_t{0};
            }
            return static_cast<std::size_t>(parsed);
        }();
        // How many candidate vectors the neighbour scan prefetches per iteration AFTER the first.
        // 0/unset means "same as the step", which is the historical burst shape: the step is
        // drained on every iteration, so the whole neighbour list is prefetched up front. Setting
        // this to 1 turns the same code into a rolling window of `step` vectors advanced one per
        // neighbour. Also purely a memory hint, so the graph hash gates it exactly.
        const std::size_t prefetch_per_iter = [] {
            const char *raw = std::getenv("INFINITY_HNSW_PREFETCH_PER_ITER");
            if (raw == nullptr) {
                return std::size_t{0};
            }
            char *parse_end = nullptr;
            const unsigned long long parsed = std::strtoull(raw, &parse_end, 10);
            if (parse_end == raw || *parse_end != '\0' || parsed > 4096) {
                return std::size_t{0};
            }
            return static_cast<std::size_t>(parsed);
        }();
        // Two bit-identical traversal knobs found by the SearchLayer decomposition. Both are
        // OFF by default so an unset environment measures shipping behaviour, and both are
        // provably graph-neutral, so the determinism gate is an exact check on them:
        //   PREFETCH_SKIP_VISITED -- do not issue a prefetch for a candidate already visited
        //     (57.0% of prefetched candidates at n=100000/efC=200 were then skipped).
        //   BATCH_TAIL_PADDING    -- run the <4 remainder of a neighbour scan through the
        //     4-wide kernel with padded lanes instead of the 1-wide kernel (13.5% of all
        //     distance evaluations were on that tail).
        // Defaults ON in the library now that it is measured, so this knob exists to turn it
        // OFF for a control arm. Unset therefore means ON, unlike the other knobs here.
        const bool prefetch_skip_visited = [] {
            const char *raw = std::getenv("INFINITY_HNSW_PREFETCH_SKIP_VISITED");
            return raw == nullptr || std::strcmp(raw, "0") != 0;
        }();
        const bool batch_tail_padding = [] {
            const char *raw = std::getenv("INFINITY_HNSW_BATCH_TAIL_PADDING");
            return raw != nullptr && std::strcmp(raw, "1") == 0;
        }();
        const bool nearest_prefetch = [] {
            const char *raw = std::getenv("INFINITY_HNSW_NEAREST_PREFETCH");
            return raw != nullptr && std::strcmp(raw, "1") == 0;
        }();
        index->SetNearestPrefetch(nearest_prefetch);
        index->SetPrefetchSkipVisited(prefetch_skip_visited);
        index->SetBatchTailPadding(batch_tail_padding);
        index->SetPruneHeadroom(prune_headroom);
        index->SetLevel0DoubleBudget(level0_double_budget);
        index->SetPrefetchStep(prefetch_step);
        index->SetPrefetchPerIter(prefetch_per_iter);
        std::cout << "infinity_prefetch_step=" << index->GetPrefetchStep() << '\n';
        std::cout << "infinity_prefetch_per_iter=" << index->GetPrefetchPerIter() << '\n';
        std::cout << "infinity_prefetch_skip_visited=" << (index->GetPrefetchSkipVisited() ? 1 : 0) << '\n';
        std::cout << "infinity_batch_tail_padding=" << (index->GetBatchTailPadding() ? 1 : 0) << '\n';
        std::cout << "infinity_nearest_prefetch=" << (index->GetNearestPrefetch() ? 1 : 0) << '\n';
        // Echo what was actually applied, not what was requested: SetPruneHeadroom clamps, so
        // a run's record must come from the index or it can disagree with the graph produced.
        std::cout << "infinity_prune_headroom=" << index->GetPruneHeadroom() << '\n';
        std::cout << "infinity_level0_double_budget=" << (index->GetLevel0DoubleBudget() ? 1 : 0) << '\n';
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

#ifdef INFINITY_HNSW_INSTRUMENT
        // Zero the traversal counters here so the dump below covers the BUILD only. The
        // query and audit phases that follow also call SearchLayer, and folding them in
        // would corrupt every per-call statistic.
        infinity::HnswInstrumentationReset();
#endif
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

#ifdef INFINITY_HNSW_INSTRUMENT
        // Read back while the worker threads still exist -- their counter blocks are
        // thread_local storage, so this must happen before the pool is destroyed.
        {
            const infinity::HnswSearchLayerCounters c = infinity::HnswInstrumentationTotals();
            const auto rate = [](std::uint64_t num, std::uint64_t den) {
                return den == 0 ? 0.0 : static_cast<double>(num) / static_cast<double>(den);
            };
            std::cout << "infinity_instr_present=1\n";
            std::cout << "infinity_instr_calls=" << c.calls << '\n';
            std::cout << "infinity_instr_calls_layer0=" << c.calls_layer0 << '\n';
            std::cout << "infinity_instr_pops=" << c.pops << '\n';
            std::cout << "infinity_instr_pops_break=" << c.pops_break << '\n';
            std::cout << "infinity_instr_pushes=" << c.pushes << '\n';
            std::cout << "infinity_instr_commit_rejected=" << c.commit_rejected << '\n';
            std::cout << "infinity_instr_neighbors_seen=" << c.neighbors_seen << '\n';
            std::cout << "infinity_instr_skip_out_of_range=" << c.skip_out_of_range << '\n';
            std::cout << "infinity_instr_skip_visited=" << c.skip_visited << '\n';
            std::cout << "infinity_instr_dist_batch4_calls=" << c.dist_batch4_calls << '\n';
            std::cout << "infinity_instr_dist_scalar_tail=" << c.dist_scalar_tail << '\n';
            std::cout << "infinity_instr_dist_scalar_path=" << c.dist_scalar_path << '\n';
            std::cout << "infinity_instr_prefetch_vec_calls=" << c.prefetch_vec_calls << '\n';
            std::cout << "infinity_instr_prefetch_wasted=" << c.prefetch_wasted << '\n';
            std::cout << "infinity_instr_prefetch_suppressed=" << c.prefetch_suppressed << '\n';
            std::cout << "infinity_instr_dist_batch4_tail_calls=" << c.dist_batch4_tail_calls << '\n';
            std::cout << "infinity_instr_heap_size_sum_at_push=" << c.heap_size_sum_at_push << '\n';
            std::cout << "infinity_instr_heap_size_sum_at_pop=" << c.heap_size_sum_at_pop << '\n';
            std::cout << "infinity_instr_heap_log2_sum_at_push=" << c.heap_log2_sum_at_push << '\n';
            std::cout << "infinity_instr_heap_log2_sum_at_pop=" << c.heap_log2_sum_at_pop << '\n';
            std::cout << "infinity_instr_heap_size_max=" << c.heap_size_max << '\n';
            std::cout << "infinity_instr_visited_alloc_bytes=" << c.visited_alloc_bytes << '\n';
            // Derived rates, so a reader does not have to divide by hand.
            std::cout << "infinity_instr_pops_per_call=" << rate(c.pops, c.calls) << '\n';
            std::cout << "infinity_instr_neighbors_per_pop=" << rate(c.neighbors_seen, c.pops) << '\n';
            std::cout << "infinity_instr_visited_skip_fraction="
                      << rate(c.skip_visited, c.neighbors_seen) << '\n';
            // Padded tail calls execute 4 lanes but commit fewer, so they are counted as the
            // 4 lanes the kernel actually computed -- the point of the counter is kernel work.
            std::cout << "infinity_instr_distance_evals="
                      << ((c.dist_batch4_calls + c.dist_batch4_tail_calls) * 4
                          + c.dist_scalar_tail + c.dist_scalar_path) << '\n';
            std::cout << "infinity_instr_scalar_distance_fraction="
                      << rate(c.dist_scalar_tail + c.dist_scalar_path,
                              (c.dist_batch4_calls + c.dist_batch4_tail_calls) * 4
                              + c.dist_scalar_tail + c.dist_scalar_path)
                      << '\n';
            std::cout << "infinity_instr_prefetch_waste_fraction="
                      << rate(c.prefetch_wasted, c.prefetch_vec_calls) << '\n';
            std::cout << "infinity_instr_pops_per_call_hist=";
            for (std::size_t i = 0; i < c.pops_per_call.size(); ++i) {
                std::cout << (i ? "," : "") << i << ':' << c.pops_per_call[i];
            }
            std::cout << '\n';
            std::cout << "infinity_instr_neighbor_size_hist=";
            for (std::size_t i = 0; i < c.neighbor_size_hist.size(); ++i) {
                std::cout << (i ? "," : "") << i << ':' << c.neighbor_size_hist[i];
            }
            std::cout << '\n';
        }
#endif

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

        // Published-ground-truth recall, when requested. Runs AFTER the self-audit and outside
        // every timed region, so enabling it cannot move a build-time number.
        {
            HnswD0ExternalTruthRequest external_request = HnswD0ReadExternalTruthRequest();
            result->external_recall_audit =
                HnswD0RunExternalTruthAudit(data, vector_count, dimension, search, external_request);
            result->external_recall_skip_reason =
                result->external_recall_audit.valid ? std::string{} : external_request.skip_reason;
        }

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
