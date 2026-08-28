#include "hnsw_dev_bridge.h"

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
        using Hnsw = infinity::KnnHnsw<infinity::LVQL2VecStoreType<float, i8>, Label>;

        const size_t vector_count = static_cast<size_t>(config->vector_count);
        const size_t dimension = static_cast<size_t>(config->dimension);
        const size_t chunk_size = static_cast<size_t>(config->chunk_size);
        const size_t max_chunk_count = (vector_count - 1) / chunk_size + 1;

        const auto cold_begin = std::chrono::steady_clock::now();
        infinity::ctpl::thread_pool build_pool(config->participant_count);
        auto index = Hnsw::Make(chunk_size, max_chunk_count, dimension, static_cast<size_t>(config->m), static_cast<size_t>(config->ef_construction));
        infinity::DenseVectorIter<float, Label> iterator(data, dimension, vector_count);
        const auto insert_begin = std::chrono::steady_clock::now();
        const infinity::HnswBulkBuildResult build_result =
            config->build_grain == 0 ? infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, build_pool)
                                     : infinity::HnswBulkBuild(index,
                                                               std::move(iterator),
                                                               infinity::HnswInsertConfig{.optimize_ = true},
                                                               build_pool,
                                                               static_cast<size_t>(config->build_grain));
        const auto insert_end = std::chrono::steady_clock::now();

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
        const std::vector<float> heldout_queries = GenerateHnswD0HeldOutQueries(dimension);
        result->query_benchmark = BenchmarkHnswD0Queries(heldout_queries, dimension, vector_count, search);

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
        result->recall_audit = AuditHnswD0Recall(
            data,
            vector_count,
            dimension,
            heldout_queries,
            search);

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
            const size_t average_bucket_size = (vector_count - 1) / static_cast<size_t>(result->thread_count) + 1;
            const size_t bucket_size = std::max(minimum_bucket_size, average_bucket_size);
            expected_task_count = (vector_count - 1) / bucket_size + 1;
        }
        result->valid = build_result.mem_usage_ > 0 && result->build_start == 0 && result->build_end == vector_count &&
                        result->submitted_tasks == expected_task_count && result->index_size == vector_count &&
                        result->thread_count == config->participant_count && result->cold_build_ns > 0 && result->insert_call_ns > 0 &&
                        result->insert_call_ns <= result->cold_build_ns && result->self_recall_at_1 >= 0.95 && result->self_recall_at_1 <= 1.0 &&
                        std::isfinite(result->distance_checksum) && result->query_benchmark.valid && result->graph_audit.valid &&
                        result->recall_audit.valid &&
                        HnswD0QueryBenchmarkMatchesRecall(result->query_benchmark, result->recall_audit, vector_count);
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
