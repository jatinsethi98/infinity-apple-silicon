#include "hnsw_dev_bridge.h"
#include "hnsw_d0_timing_probe.h"
#include "hnsw_d0_external_truth.h"

#include <faiss/IndexHNSW.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

extern "C" int RunFaissHnsw(const float *data, const HnswDevConfig *config, HnswDevResult *result) {
    if (result == nullptr) {
        return 64;
    }
    *result = HnswDevResult{};
    try {
        if (data == nullptr || config == nullptr || config->vector_count == 0 || config->dimension == 0 || config->query_count == 0 ||
            config->query_count > config->vector_count || config->m < 2 || config->ef_construction < config->m || config->ef_search < 1 ||
            config->participant_count < 1 || config->vector_count > static_cast<unsigned long long>(std::numeric_limits<faiss::idx_t>::max()) ||
            config->dimension > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
            config->query_count > static_cast<unsigned long long>(std::numeric_limits<faiss::idx_t>::max())) {
            return 64;
        }
        const std::size_t vector_count = static_cast<std::size_t>(config->vector_count);
        const std::size_t dimension = static_cast<std::size_t>(config->dimension);

        omp_set_dynamic(0);
        omp_set_num_threads(config->participant_count);

        if (config->index_barrier != nullptr &&
            config->index_barrier(HnswD0IndexBarrierPhase::kBeforeIndex, config->index_barrier_context) != 0) {
            return 74;
        }
        const auto cold_begin = HNSW_D0_READ_STEADY_CLOCK(kTimerStarted);
        faiss::IndexHNSWFlat index(static_cast<int>(dimension), config->m, faiss::METRIC_L2);
        index.hnsw.efConstruction = config->ef_construction;
        index.hnsw.efSearch = config->ef_search;
        HNSW_D0_TIMING_PROBE(kIndexReady);
        HNSW_D0_TIMING_PROBE(kExecutionWitnessArmBegin);
        HNSW_D0_TIMING_PROBE(kExecutionWitnessArmEnd);
        const auto insert_begin = HNSW_D0_READ_STEADY_CLOCK(kBuildEntered);
        index.add(static_cast<faiss::idx_t>(vector_count), data);
        HNSW_D0_TIMING_PROBE(kBuildReturned);
        const auto insert_end = HNSW_D0_READ_STEADY_CLOCK(kTimerStopped);
        HNSW_D0_TIMING_PROBE(kExecutionWitnessSealBegin);
        HNSW_D0_TIMING_PROBE(kExecutionWitnessSealEnd);
        if (config->index_barrier != nullptr &&
            config->index_barrier(HnswD0IndexBarrierPhase::kAfterIndex, config->index_barrier_context) != 0) {
            return 74;
        }

        result->index_size = index.ntotal;
        result->cold_build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - cold_begin).count();
        result->insert_call_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_begin).count();
        result->thread_count = omp_get_max_threads();

        const HnswD0Search search = [&](const float *query, std::size_t k, std::size_t ef) {
            std::vector<float> found_distances(k);
            std::vector<faiss::idx_t> found_labels(k);
            faiss::SearchParametersHNSW parameters;
            parameters.efSearch = static_cast<int>(ef);
            parameters.check_relative_distance = index.hnsw.check_relative_distance;
            parameters.bounded_queue = index.hnsw.search_bounded_queue;
            index.search(1, query, static_cast<faiss::idx_t>(k), found_distances.data(), found_labels.data(), &parameters);
            std::vector<std::pair<float, std::int64_t>> returned;
            returned.reserve(k);
            for (std::size_t rank = 0; rank < k; ++rank) {
                returned.emplace_back(found_distances[rank], static_cast<std::int64_t>(found_labels[rank]));
            }
            return returned;
        };
        HNSW_D0_TIMING_PROBE(kQueryBenchmarkEntered);
        const std::vector<float> heldout_queries = GenerateHnswD0HeldOutQueries(dimension);
        result->query_benchmark = BenchmarkHnswD0Queries(heldout_queries, dimension, vector_count, search);

        HNSW_D0_TIMING_PROBE(kSelfRecallEntered);
        const std::size_t query_count = static_cast<std::size_t>(config->query_count);
        std::vector<float> distances(query_count);
        std::vector<faiss::idx_t> labels(query_count);
        index.search(static_cast<faiss::idx_t>(query_count), data, 1, distances.data(), labels.data());

        std::size_t correct = 0;
        double distance_checksum = 0;
        for (std::size_t i = 0; i < query_count; ++i) {
            correct += labels[i] == static_cast<faiss::idx_t>(i);
            distance_checksum += distances[i];
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
        // every timed region, so enabling it cannot move a build-time number. Identical code in
        // both bridges on purpose: the two engines must be scored by the same audit or the
        // comparison means nothing.
        {
            HnswD0ExternalTruthRequest external_request = HnswD0ReadExternalTruthRequest();
            result->external_recall_audit =
                HnswD0RunExternalTruthAudit(data, vector_count, dimension, search, external_request);
            result->external_recall_skip_reason =
                result->external_recall_audit.valid ? std::string{} : external_request.skip_reason;
        }

        HNSW_D0_TIMING_PROBE(kGraphAuditEntered);
        if (index.storage == nullptr || index.storage->ntotal != index.ntotal || index.hnsw.levels.size() != vector_count ||
            index.hnsw.offsets.size() != vector_count + 1 || index.hnsw.offsets.front() != 0 ||
            index.hnsw.neighbors.size() != index.hnsw.offsets.back()) {
            throw std::runtime_error("FAISS graph container metadata is inconsistent");
        }
        for (std::size_t vertex = 0; vertex < vector_count; ++vertex) {
            const int level_count = index.hnsw.levels[vertex];
            if (level_count < 1 || static_cast<std::size_t>(level_count) >= index.hnsw.cum_nneighbor_per_level.size() ||
                index.hnsw.offsets[vertex] > index.hnsw.offsets[vertex + 1] ||
                index.hnsw.offsets[vertex + 1] - index.hnsw.offsets[vertex] !=
                    static_cast<std::size_t>(index.hnsw.cum_nb_neighbors(level_count))) {
                throw std::runtime_error("FAISS graph level or offset metadata is inconsistent");
            }
        }
        result->graph_audit = AuditHnswD0Graph(HnswD0GraphInput{
            .vertex_count = vector_count,
            .max_level = index.hnsw.max_level,
            .entry_point = index.hnsw.entry_point,
            .level0_capacity = static_cast<std::size_t>(index.hnsw.nb_neighbors(0)),
            .upper_capacity = static_cast<std::size_t>(index.hnsw.nb_neighbors(1)),
            .level = [&](std::int32_t vertex) { return index.hnsw.levels[static_cast<std::size_t>(vertex)] - 1; },
            .label = [](std::int32_t vertex) { return static_cast<std::int64_t>(vertex); },
            .neighbors =
                [&](std::int32_t vertex, std::int32_t layer) {
                    std::size_t begin = 0;
                    std::size_t end = 0;
                    index.hnsw.neighbor_range(vertex, layer, &begin, &end);
                    std::size_t occupied = 0;
                    bool saw_sentinel = false;
                    bool tail_valid = true;
                    for (std::size_t offset = begin; offset < end; ++offset) {
                        if (index.hnsw.neighbors[offset] == -1) {
                            saw_sentinel = true;
                        } else if (index.hnsw.neighbors[offset] < -1) {
                            tail_valid = false;
                        } else if (saw_sentinel) {
                            tail_valid = false;
                        } else {
                            ++occupied;
                        }
                    }
                    return HnswD0NeighborRange{
                        .data = index.hnsw.neighbors.data() + begin,
                        .size = occupied,
                        .capacity = end - begin,
                        .tail_valid = tail_valid,
                    };
                },
        });
        result->valid = index.ntotal == static_cast<faiss::idx_t>(vector_count) && result->thread_count == config->participant_count &&
                        result->cold_build_ns > 0 && result->insert_call_ns > 0 && result->insert_call_ns <= result->cold_build_ns &&
                        result->self_recall_at_1 >= 0.95 && result->self_recall_at_1 <= 1.0 && std::isfinite(result->distance_checksum) &&
                        result->query_benchmark.valid && result->graph_audit.valid && result->recall_audit.valid &&
                        HnswD0QueryBenchmarkMatchesRecall(result->query_benchmark, result->recall_audit, vector_count);
        HNSW_D0_TIMING_PROBE(kBridgeCompleted);
        return result->valid ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "FAISS HNSW audit failed: " << error.what() << '\n';
        result->valid = 0;
        return 2;
    } catch (...) {
        result->valid = 0;
        return 2;
    }
}
