#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

inline constexpr std::size_t kHnswD0HeldOutQueryCount = 64;
inline constexpr std::uint64_t kHnswD0HeldOutQuerySeed = 0x6a09e667f3bcc909ULL;
inline constexpr std::array<std::size_t, 8> kHnswD0RecallEf{32, 64, 128, 256, 512, 128, 256, 512};
inline constexpr std::array<std::size_t, 8> kHnswD0RecallK{10, 10, 10, 10, 10, 100, 100, 100};

struct HnswD0NeighborRange {
    const std::int32_t *data = nullptr;
    std::size_t size = 0;
    std::size_t capacity = 0;
    bool tail_valid = false;
};

struct HnswD0GraphInput {
    std::size_t vertex_count = 0;
    std::int32_t max_level = -1;
    std::int32_t entry_point = -1;
    std::size_t level0_capacity = 0;
    std::size_t upper_capacity = 0;
    std::function<std::int32_t(std::int32_t)> level;
    std::function<std::int64_t(std::int32_t)> label;
    std::function<HnswD0NeighborRange(std::int32_t, std::int32_t)> neighbors;
};

struct HnswD0GraphLayerAudit {
    std::size_t capacity = 0;
    std::vector<std::int32_t> neighbors;
};

struct HnswD0GraphVertexAudit {
    std::int32_t level = -1;
    std::int64_t label = -1;
    std::vector<HnswD0GraphLayerAudit> layers;
};

struct HnswD0GraphAudit {
    bool valid = false;
    std::size_t vertex_count = 0;
    std::size_t reachable_count = 0;
    std::size_t directed_edges = 0;
    std::size_t level0_directed_edges = 0;
    std::int32_t max_level = -1;
    std::int32_t entry_point = -1;
    std::size_t level0_capacity = 0;
    std::size_t upper_capacity = 0;
    std::string levels_sha256;
    std::string graph_sha256;
    std::string level_histogram;
    std::string degree_histograms;
    std::vector<HnswD0GraphVertexAudit> vertices;
};

using HnswD0Search = std::function<std::vector<std::pair<float, std::int64_t>>(const float *, std::size_t, std::size_t)>;

struct HnswD0ReturnedResult {
    float distance = 0.0F;
    std::int64_t label = -1;
};

struct HnswD0RecallAudit {
    bool valid = false;
    std::size_t query_count = 0;
    std::uint64_t query_seed = 0;
    std::string queries_sha256;
    std::string truth_sha256;
    std::array<double, 5> recall_at_10{};
    std::array<double, 3> recall_at_100{};
    std::array<std::string, 8> returned_ids;
    std::array<std::string, 8> returned_distances_sha256;
    std::string truth_tie_counts_at_10;
    std::string truth_tie_counts_at_100;
    std::array<std::vector<HnswD0ReturnedResult>, 8> returned_results;
};

struct HnswD0AttestationImage {
    std::uintptr_t header = 0;
    std::intptr_t slide = 0;
    std::string path;
    std::string uuid;
    bool shared_cache = false;

    bool operator==(const HnswD0AttestationImage &) const = default;
};

struct HnswD0FileIdentity {
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::uint64_t size = 0;

    bool operator==(const HnswD0FileIdentity &) const = default;
};

struct HnswD0AttestationSnapshot {
    HnswD0FileIdentity held_executable;
    HnswD0FileIdentity mapped_executable;
    std::string dynamic_cdhash;
    std::vector<HnswD0AttestationImage> images;
};

struct HnswD0AttestationSummary {
    std::uint64_t initial_image_adds = 0;
    std::uint64_t later_image_adds = 0;
    std::uint64_t image_removes = 0;
    bool image_set_unchanged = false;
};

bool HnswD0MappedExecutableMatchesHeldFile(const HnswD0AttestationSnapshot &snapshot);

class HnswD0LiveAttestation {
public:
    HnswD0LiveAttestation();
    ~HnswD0LiveAttestation();

    HnswD0LiveAttestation(const HnswD0LiveAttestation &) = delete;
    HnswD0LiveAttestation &operator=(const HnswD0LiveAttestation &) = delete;
    HnswD0LiveAttestation(HnswD0LiveAttestation &&) noexcept;
    HnswD0LiveAttestation &operator=(HnswD0LiveAttestation &&) noexcept;

    bool Begin(std::string &diagnostic);
    bool Capture(HnswD0AttestationSnapshot &snapshot, std::string &diagnostic) const;
    bool Validate(const HnswD0AttestationSnapshot &before,
                  const HnswD0AttestationSnapshot &after,
                  HnswD0AttestationSummary &summary,
                  std::string &diagnostic) const;
    int executable_fd() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

HnswD0GraphAudit AuditHnswD0Graph(const HnswD0GraphInput &input);

std::string HnswD0QueryCorpusSha256(const std::vector<float> &queries, std::size_t dimension);

HnswD0RecallAudit AuditHnswD0Recall(const float *base,
                                    std::size_t vector_count,
                                    std::size_t dimension,
                                    const std::vector<float> &queries,
                                    const HnswD0Search &search);
