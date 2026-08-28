#include "hnsw_d0_timing_probe.h"

#include "hnsw_dev_bridge.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace hnsw_d0_timing_probe {

std::vector<Phase> phases;
std::vector<std::pair<Phase, std::chrono::steady_clock::time_point>> clock_reads;
std::vector<HnswD0IndexBarrierPhase> index_barriers;

void Record(Phase phase) { phases.push_back(phase); }

std::chrono::steady_clock::time_point ReadSteadyClock(Phase phase) {
    const auto timestamp = std::chrono::steady_clock::now();
    phases.push_back(phase);
    clock_reads.emplace_back(phase, timestamp);
    return timestamp;
}

} // namespace hnsw_d0_timing_probe

namespace {

using Bridge = int (*)(const float *, const HnswDevConfig *, HnswDevResult *);
using hnsw_d0_timing_probe::Phase;

constexpr std::array kExpectedPhases{
    Phase::kTimerStarted,
    Phase::kIndexReady,
    Phase::kExecutionWitnessArmBegin,
    Phase::kExecutionWitnessArmEnd,
    Phase::kBuildEntered,
    Phase::kBuildReturned,
    Phase::kTimerStopped,
    Phase::kExecutionWitnessSealBegin,
    Phase::kExecutionWitnessSealEnd,
    Phase::kQueryBenchmarkEntered,
    Phase::kSelfRecallEntered,
    Phase::kRecallAuditEntered,
    Phase::kGraphAuditEntered,
    Phase::kBridgeCompleted,
};

int RecordIndexBarrier(HnswD0IndexBarrierPhase phase, void *) {
    hnsw_d0_timing_probe::index_barriers.push_back(phase);
    return 0;
}

std::vector<float> MakeData(std::size_t count, std::size_t dimension) {
    std::vector<float> data(count * dimension);
    std::uint64_t state = 0x243f6a8885a308d3ULL;
    for (float &value : data) {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t bits = state;
        bits = (bits ^ (bits >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        bits = (bits ^ (bits >> 27U)) * 0x94d049bb133111ebULL;
        bits ^= bits >> 31U;
        value = static_cast<float>(bits >> 40U) * (1.0F / 16'777'216.0F);
    }
    return data;
}

bool RunBoundaryCase(std::string_view name, Bridge bridge, const std::vector<float> &data, const HnswDevConfig &config, bool require_bulk_metadata) {
    hnsw_d0_timing_probe::phases.clear();
    hnsw_d0_timing_probe::clock_reads.clear();
    hnsw_d0_timing_probe::index_barriers.clear();
    HnswDevResult result{};
    const int exit_code = bridge(data.data(), &config, &result);
    const bool exact_phase_order = hnsw_d0_timing_probe::phases.size() == kExpectedPhases.size() &&
                                   std::equal(kExpectedPhases.begin(), kExpectedPhases.end(), hnsw_d0_timing_probe::phases.begin());
    const bool build_before_stop =
        exact_phase_order && std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kBuildReturned) <
                                 std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kTimerStopped);
    const auto arm_begin =
        std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kExecutionWitnessArmBegin);
    const auto arm_end =
        std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kExecutionWitnessArmEnd);
    const auto timer_stop = std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kTimerStopped);
    const auto seal_begin =
        std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kExecutionWitnessSealBegin);
    const auto seal_end =
        std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kExecutionWitnessSealEnd);
    const auto query_begin =
        std::find(hnsw_d0_timing_probe::phases.begin(), hnsw_d0_timing_probe::phases.end(), Phase::kQueryBenchmarkEntered);
    const bool exact_witness_boundary =
        exact_phase_order && arm_begin < arm_end && arm_end < std::find(hnsw_d0_timing_probe::phases.begin(),
                                                                        hnsw_d0_timing_probe::phases.end(),
                                                                        Phase::kBuildEntered) &&
        timer_stop < seal_begin && seal_begin < seal_end && seal_end < query_begin;
    const bool post_timing_work =
        exact_phase_order &&
        std::all_of(timer_stop + 1, hnsw_d0_timing_probe::phases.end(), [timer_stop, &phases = hnsw_d0_timing_probe::phases](Phase phase) {
            return std::find(phases.begin(), phases.end(), phase) > timer_stop;
        });
    const auto clock_timestamp = [](Phase phase) {
        const auto found =
            std::find_if(hnsw_d0_timing_probe::clock_reads.begin(),
                         hnsw_d0_timing_probe::clock_reads.end(),
                         [phase](const auto &entry) { return entry.first == phase; });
        return found == hnsw_d0_timing_probe::clock_reads.end()
                   ? std::chrono::steady_clock::time_point{}
                   : found->second;
    };
    const bool exact_clock_reads =
        hnsw_d0_timing_probe::clock_reads.size() == 3 &&
        hnsw_d0_timing_probe::clock_reads[0].first == Phase::kTimerStarted &&
        hnsw_d0_timing_probe::clock_reads[1].first == Phase::kBuildEntered &&
        hnsw_d0_timing_probe::clock_reads[2].first == Phase::kTimerStopped;
    const auto clock_start = clock_timestamp(Phase::kTimerStarted);
    const auto insert_start = clock_timestamp(Phase::kBuildEntered);
    const auto clock_stop = clock_timestamp(Phase::kTimerStopped);
    const auto recorded_cold_build_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_stop - clock_start).count();
    const auto recorded_insert_call_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_stop - insert_start).count();
    const bool exact_reported_durations =
        exact_clock_reads && recorded_cold_build_ns >= 0 && recorded_insert_call_ns >= 0 &&
        static_cast<std::uint64_t>(recorded_cold_build_ns) == result.cold_build_ns &&
        static_cast<std::uint64_t>(recorded_insert_call_ns) == result.insert_call_ns;
    const bool exact_index_barriers =
        hnsw_d0_timing_probe::index_barriers ==
        std::vector{
            HnswD0IndexBarrierPhase::kBeforeIndex,
            HnswD0IndexBarrierPhase::kAfterIndex,
        };

    const std::size_t vector_count = static_cast<std::size_t>(config.vector_count);
    const std::size_t workers = static_cast<std::size_t>(config.participant_count);
    const std::size_t average_bucket = (vector_count - 1) / workers + 1;
    const std::size_t bucket = std::max(static_cast<std::size_t>(config.build_grain), average_bucket);
    const std::size_t expected_tasks = (vector_count - 1) / bucket + 1;
    const bool bulk_metadata =
        !require_bulk_metadata || (result.build_start == 0 && result.build_end == config.vector_count && result.submitted_tasks == expected_tasks);
    const bool complete_result = exit_code == 0 && result.valid == 1 && result.index_size == config.vector_count &&
                                 result.thread_count == config.participant_count && result.cold_build_ns > 0 && result.insert_call_ns > 0 &&
                                 result.insert_call_ns <= result.cold_build_ns && result.query_benchmark.valid && result.recall_audit.valid &&
                                 result.graph_audit.valid;
    const bool passed =
        exact_phase_order && build_before_stop && exact_witness_boundary && post_timing_work && exact_reported_durations &&
        exact_index_barriers && bulk_metadata && complete_result;

    std::cout << name << "_exit_code=" << exit_code << '\n';
    std::cout << name << "_exact_phase_order=" << exact_phase_order << '\n';
    std::cout << name << "_build_before_timer_stop=" << build_before_stop << '\n';
    std::cout << name << "_execution_witness_boundary=" << exact_witness_boundary << '\n';
    std::cout << name << "_queries_and_audits_after_timer=" << post_timing_work << '\n';
    std::cout << name << "_reported_durations_match_clock_reads=" << exact_reported_durations << '\n';
    std::cout << name << "_index_barriers_outside_timer=" << exact_index_barriers << '\n';
    std::cout << name << "_complete_stored_range_and_tasks=" << bulk_metadata << '\n';
    std::cout << name << "_valid_result=" << complete_result << '\n';
    return passed;
}

} // namespace

int main() {
    constexpr std::size_t kVectorCount = 2'048;
    constexpr std::size_t kDimension = 32;
    const std::vector<float> data = MakeData(kVectorCount, kDimension);
    const HnswDevConfig config{
        .vector_count = kVectorCount,
        .dimension = kDimension,
        .chunk_size = kVectorCount,
        .query_count = 128,
        .m = 16,
        .ef_construction = 96,
        .ef_search = 128,
        .participant_count = 4,
        .build_grain = 128,
        .index_barrier = RecordIndexBarrier,
        .index_barrier_context = nullptr,
    };

    const bool infinity_passed = RunBoundaryCase("infinity", RunInfinityHnsw, data, config, true);
    const bool faiss_passed = RunBoundaryCase("faiss", RunFaissHnsw, data, config, false);
    const bool passed = infinity_passed && faiss_passed;
    std::cout << "status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
