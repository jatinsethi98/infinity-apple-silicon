#pragma once

#include "hnsw_d0_audit.h"
#include "hnsw_d0_query_benchmark.h"

#include <cstdint>
#include <string>

enum class HnswD0IndexBarrierPhase : int {
    kBeforeIndex = 1,
    kAfterIndex = 2,
};

using HnswD0IndexBarrier = int (*)(HnswD0IndexBarrierPhase phase, void *context);

struct HnswDevConfig {
    unsigned long long vector_count;
    unsigned long long dimension;
    unsigned long long chunk_size;
    unsigned long long query_count;
    int m;
    int ef_construction;
    int ef_search;
    int participant_count;
    unsigned long long build_grain;
    HnswD0IndexBarrier index_barrier;
    void *index_barrier_context;
};

struct HnswD0ExecutionWitness {
    std::uint32_t incremental_treatment_compiled{};
    std::uint32_t incremental_capture_armed{};
    std::uint32_t incremental_eligible_branch_entered{};
    std::uint32_t incremental_successful_unchanged_observed{};
    std::uint32_t incremental_successful_updated_observed{};
    std::uint32_t threshold_treatment_compiled{};
    std::uint32_t threshold_capture_armed{};
    std::uint32_t threshold_eligible_branch_entered{};
    std::uint32_t threshold_rejected_lane_observed{};
    std::uint32_t threshold_surviving_lane_observed{};
};

struct HnswDevResult {
    unsigned long long index_size;
    unsigned long long cold_build_ns;
    unsigned long long insert_call_ns;
    double self_recall_at_1;
    double distance_checksum;
    int thread_count;
    int valid;
    unsigned long long submitted_tasks;
    unsigned long long build_start;
    unsigned long long build_end;
    HnswD0QueryBenchmark query_benchmark;
    HnswD0ExecutionWitness execution_witness;
    HnswD0GraphAudit graph_audit;
    HnswD0RecallAudit recall_audit;
    // Recall against the PUBLISHED SIFT1M ground truth, when the run asked for it. Optional
    // and therefore never part of the validity predicate: `valid` stays false and
    // external_recall_skip_reason says why whenever it did not run. See
    // hnsw_d0_external_truth.h for why the 64-query self-audit above is not a substitute.
    HnswD0ExternalRecallAudit external_recall_audit;
    std::string external_recall_skip_reason;
};

extern "C" int RunInfinityHnsw(const float *data, const HnswDevConfig *config, HnswDevResult *result);
extern "C" int RunFaissHnsw(const float *data, const HnswDevConfig *config, HnswDevResult *result);
