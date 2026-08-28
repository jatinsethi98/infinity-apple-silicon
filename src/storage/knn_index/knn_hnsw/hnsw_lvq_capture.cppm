export module infinity_core:hnsw_lvq_capture;

import :hnsw_common;

import std;

namespace infinity {

export enum class HnswLvqPhase : u8 {
    kUnknown = 0,
    kConstructionGreedy = 1,
    kConstructionBeam = 2,
    kNewSelection = 3,
    kReciprocalSelection = 4,
    kQuery = 5,
};

export struct HnswLvqCallContext {
    HnswLvqPhase phase{HnswLvqPhase::kUnknown};
    LayerSize layer{-1};
    VertexType query_vertex{kInvalidVertex};
    VertexType candidate_vertex{kInvalidVertex};
};

export class HnswLvqCaptureSink {
public:
    virtual ~HnswLvqCaptureSink() = default;

    virtual void Observe(std::span<const std::byte> query_record,
                         std::span<const std::byte> candidate_record,
                         size_t dimension,
                         f32 distance,
                         const HnswLvqCallContext &context) = 0;
};

#if defined(INFINITY_ENABLE_HNSW_LVQ_CAPTURE)

namespace {

std::atomic<HnswLvqCaptureSink *> hnsw_lvq_capture_sink{};
thread_local HnswLvqPhase hnsw_lvq_phase = HnswLvqPhase::kUnknown;
thread_local LayerSize hnsw_lvq_layer = -1;

} // namespace

export inline constexpr bool kHnswLvqCaptureEnabled = true;

export class HnswLvqCaptureSession {
public:
    explicit HnswLvqCaptureSession(HnswLvqCaptureSink &sink) {
        HnswLvqCaptureSink *expected = nullptr;
        if (!hnsw_lvq_capture_sink.compare_exchange_strong(expected, &sink, std::memory_order_release, std::memory_order_relaxed)) {
            throw std::logic_error("Only one HNSW LVQ capture session may be active");
        }
        sink_ = &sink;
    }

    HnswLvqCaptureSession(const HnswLvqCaptureSession &) = delete;
    HnswLvqCaptureSession &operator=(const HnswLvqCaptureSession &) = delete;
    HnswLvqCaptureSession(HnswLvqCaptureSession &&) = delete;
    HnswLvqCaptureSession &operator=(HnswLvqCaptureSession &&) = delete;

    ~HnswLvqCaptureSession() {
        HnswLvqCaptureSink *expected = sink_;
        static_cast<void>(
            hnsw_lvq_capture_sink.compare_exchange_strong(expected, nullptr, std::memory_order_release, std::memory_order_relaxed));
    }

private:
    HnswLvqCaptureSink *sink_{};
};

export class HnswLvqPhaseScope {
public:
    HnswLvqPhaseScope(HnswLvqPhase phase, LayerSize layer) noexcept
        : previous_phase_(std::exchange(hnsw_lvq_phase, phase)), previous_layer_(std::exchange(hnsw_lvq_layer, layer)) {}

    HnswLvqPhaseScope(const HnswLvqPhaseScope &) = delete;
    HnswLvqPhaseScope &operator=(const HnswLvqPhaseScope &) = delete;

    ~HnswLvqPhaseScope() {
        hnsw_lvq_phase = previous_phase_;
        hnsw_lvq_layer = previous_layer_;
    }

private:
    HnswLvqPhase previous_phase_;
    LayerSize previous_layer_;
};

export inline void HnswObserveLvqL2(const std::byte *query_record,
                                    const std::byte *candidate_record,
                                    size_t record_bytes,
                                    size_t dimension,
                                    f32 distance,
                                    VertexType query_vertex,
                                    VertexType candidate_vertex) {
    HnswLvqCaptureSink *sink = hnsw_lvq_capture_sink.load(std::memory_order_acquire);
    if (sink == nullptr) {
        return;
    }
    sink->Observe(std::span(query_record, record_bytes),
                  std::span(candidate_record, record_bytes),
                  dimension,
                  distance,
                  HnswLvqCallContext{
                      .phase = hnsw_lvq_phase,
                      .layer = hnsw_lvq_layer,
                      .query_vertex = query_vertex,
                      .candidate_vertex = candidate_vertex,
                  });
}

#else

export inline constexpr bool kHnswLvqCaptureEnabled = false;

export class HnswLvqCaptureSession {
public:
    explicit HnswLvqCaptureSession(HnswLvqCaptureSink &) {
        throw std::logic_error("HNSW LVQ capture is not enabled in this build");
    }
};

export class HnswLvqPhaseScope {
public:
    HnswLvqPhaseScope(HnswLvqPhase, LayerSize) noexcept {}
};

export inline void HnswObserveLvqL2(const std::byte *,
                                    const std::byte *,
                                    size_t,
                                    size_t,
                                    f32,
                                    VertexType,
                                    VertexType) noexcept {}

#endif

} // namespace infinity
