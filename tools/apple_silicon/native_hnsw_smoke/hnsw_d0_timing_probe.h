#pragma once

#if defined(HNSW_D0_TIMING_PROBE_ENABLED)

#include <chrono>

namespace hnsw_d0_timing_probe {

enum class Phase {
    kTimerStarted,
    kIndexReady,
    kExecutionWitnessArmBegin,
    kExecutionWitnessArmEnd,
    kBuildEntered,
    kBuildReturned,
    kTimerStopped,
    kExecutionWitnessSealBegin,
    kExecutionWitnessSealEnd,
    kQueryBenchmarkEntered,
    kSelfRecallEntered,
    kRecallAuditEntered,
    kGraphAuditEntered,
    kBridgeCompleted,
};

void Record(Phase phase);
std::chrono::steady_clock::time_point ReadSteadyClock(Phase phase);

} // namespace hnsw_d0_timing_probe

#define HNSW_D0_TIMING_PROBE(phase) ::hnsw_d0_timing_probe::Record(::hnsw_d0_timing_probe::Phase::phase)
#define HNSW_D0_READ_STEADY_CLOCK(phase) \
    ::hnsw_d0_timing_probe::ReadSteadyClock(::hnsw_d0_timing_probe::Phase::phase)

#else

#define HNSW_D0_TIMING_PROBE(phase)
#define HNSW_D0_READ_STEADY_CLOCK(phase) std::chrono::steady_clock::now()

#endif
