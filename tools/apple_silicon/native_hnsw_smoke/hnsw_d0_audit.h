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

// The canonical SIFT1M base cardinality. The published ground-truth IDs address this exact
// corpus, so external-truth recall is only defined here (see HnswD0ExternalRecallAudit).
inline constexpr std::size_t kHnswD0ExternalTruthVectorCount = 1000000;

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

// Recall against the CANONICAL SIFT1M ground truth, as distinct from HnswD0RecallAudit's
// self-audit.
//
// Why a separate entity rather than a mode of AuditHnswD0Recall: the two differ in what
// they can prove and what they cost.
//
//   * The self-audit derives truth itself, in double precision, over every indexed row.
//     That works at any n, which is what makes it usable in the small smokes, but the
//     "recall" it reports is against a corpus it invented -- it cannot be compared to a
//     number anyone else published.
//   * This one reads the published truth. That makes it comparable to FAISS and to the
//     literature, but it is only meaningful at n=1,000,000: the published IDs index the
//     full canonical base (query 0's true nearest neighbour is ID 932085), so against a
//     prefix they address the wrong rows -- silently, since the IDs are still in range for
//     a large enough prefix. Hence the exact-n requirement, enforced not assumed.
//
// It is deliberately CHEAP. Recomputing truth at n=1e6 x 10,000 queries would be ~1.3e12
// distance evaluations plus 10,000 sorts of a million elements. But recall@10 does not need
// the full distance array -- only the k-th true neighbour's distance, as the acceptance
// cutoff, plus one distance per returned label. That is ~20 distance evaluations per query
// instead of a million, so this can run over all 10,000 official queries.
//
// Both a tie-tolerant and a strict-ID recall are reported. Tie-tolerant is the correct
// figure (a vector at exactly the cutoff distance is an equally valid answer, and SIFT has
// such ties), but publishing only it invites the suspicion that ties are inflating it, so
// the strict count travels alongside as the pessimistic bound.
struct HnswD0ExternalRecallAudit {
    bool valid = false;
    std::string truth_source;
    std::size_t query_count = 0;
    std::size_t groundtruth_columns = 0;
    std::string queries_sha256;
    std::string groundtruth_sha256;
    // Indexed to match the first five entries of kHnswD0RecallEf: ef 32/64/128/256/512.
    std::array<double, 5> recall_at_10{};
    std::array<double, 5> strict_id_recall_at_10{};
    // Queries whose 10th true neighbour shares its distance with a further row, i.e. those
    // for which the two recall figures above can legitimately disagree.
    std::size_t queries_with_cutoff_ties = 0;
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

// `vector_count` MUST be exactly kHnswD0ExternalTruthVectorCount; see the struct comment.
// `queries` is nq*dimension row-major, `groundtruth` is nq*groundtruth_columns row-major
// ascending-by-distance IDs. `query_limit` caps how many queries are evaluated (0 = all);
// the cap exists so a cheaper gate can run a prefix, and it is recorded in the result.
HnswD0ExternalRecallAudit AuditHnswD0RecallExternal(const float *base,
                                                    std::size_t vector_count,
                                                    std::size_t dimension,
                                                    const std::vector<float> &queries,
                                                    const std::vector<std::int32_t> &groundtruth,
                                                    std::size_t groundtruth_columns,
                                                    std::size_t query_limit,
                                                    const HnswD0Search &search);
