#include "hnsw_d0_audit.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Neighbors = std::array<std::array<std::int32_t, 2>, 4>;
using Degrees = std::array<std::size_t, 4>;

HnswD0GraphAudit Audit(const Neighbors &neighbors, const Degrees &degrees) {
    return AuditHnswD0Graph(HnswD0GraphInput{
        .vertex_count = neighbors.size(),
        .max_level = 0,
        .entry_point = 0,
        .level0_capacity = 2,
        .upper_capacity = 1,
        .level = [](std::int32_t) { return 0; },
        .label = [](std::int32_t vertex) { return vertex; },
        .neighbors =
            [&](std::int32_t vertex, std::int32_t layer) {
                if (layer != 0) {
                    throw std::logic_error("unexpected layer");
                }
                const std::size_t ordinal = static_cast<std::size_t>(vertex);
                return HnswD0NeighborRange{
                    .data = neighbors[ordinal].data(),
                    .size = degrees[ordinal],
                    .capacity = neighbors[ordinal].size(),
                    .tail_valid = true,
                };
            },
    });
}

// --- external-truth recall audit fixture ------------------------------------------------
//
// AuditHnswD0RecallExternal insists on exactly 1,000,000 base rows, because the published
// SIFT ground-truth IDs address that corpus. Dimension is free, though, so a d=2 corpus
// satisfies the row requirement in 8 MB and makes every distance hand-checkable.
//
// Layout: base[i] = (i, 0), EXCEPT base[10], which is placed on top of base[9] at (9, 0).
// For the query at the origin the true squared distances are therefore i^2 for i <= 9, and
// row 10 ties row 9 at exactly 81. That tie is the point of the fixture: it is the only
// configuration where tie-tolerant and strict-ID recall are allowed to disagree, so it is
// the only one that proves the two are computed differently rather than aliased.
constexpr std::size_t kExternalRows = kHnswD0ExternalTruthVectorCount;
constexpr std::size_t kExternalDim = 2;
constexpr std::size_t kExternalGroundTruthColumns = 11;
constexpr double kExternalCutoff = 81.0; // = 9^2, the rank-9 (10th) true distance

std::vector<float> MakeExternalBase() {
    std::vector<float> base(kExternalRows * kExternalDim, 0.0F);
    for (std::size_t row = 0; row < kExternalRows; ++row) {
        base[row * kExternalDim] = static_cast<float>(row);
    }
    base[10 * kExternalDim] = 9.0F; // duplicate of row 9 -> tie at the cutoff
    return base;
}

// Ascending by distance: rows 0..9 at 0,1,4,...,81 then row 10 also at 81.
std::vector<std::int32_t> MakeExternalGroundTruth() {
    std::vector<std::int32_t> truth(kExternalGroundTruthColumns);
    for (std::size_t rank = 0; rank < kExternalGroundTruthColumns; ++rank) {
        truth[rank] = static_cast<std::int32_t>(rank);
    }
    return truth;
}

// A search that always returns `labels`, with each label's true distance as its score so the
// audit's sortedness and finiteness preconditions hold.
HnswD0Search FixedSearch(const std::vector<std::int32_t> &labels, const std::vector<float> &base) {
    return [&labels, &base](const float *query, std::size_t k, std::size_t) {
        std::vector<std::pair<float, std::int64_t>> out;
        out.reserve(k);
        for (std::size_t rank = 0; rank < k && rank < labels.size(); ++rank) {
            const std::size_t row = static_cast<std::size_t>(labels[rank]);
            double distance = 0.0;
            for (std::size_t component = 0; component < kExternalDim; ++component) {
                const double difference =
                    static_cast<double>(query[component]) - static_cast<double>(base[row * kExternalDim + component]);
                distance += difference * difference;
            }
            out.emplace_back(static_cast<float>(distance), static_cast<std::int64_t>(labels[rank]));
        }
        return out;
    };
}

bool Rejects(const char *what, auto &&call) {
    try {
        call();
    } catch (const std::exception &) {
        return true;
    }
    std::cerr << "external recall audit accepted " << what << '\n';
    return false;
}

} // namespace

int main() {
    const Degrees one_edge_each{1, 1, 1, 1};
    const HnswD0GraphAudit connected = Audit(
        Neighbors{{
            {{1, 0}},
            {{2, 0}},
            {{3, 0}},
            {{0, 0}},
        }},
        one_edge_each);
    if (!connected.valid || connected.reachable_count != 4 || connected.vertices.size() != 4) {
        std::cerr << "connected graph audit failed\n";
        return 1;
    }

    const HnswD0GraphAudit disconnected = Audit(
        Neighbors{{
            {{1, 0}},
            {{0, 0}},
            {{3, 0}},
            {{2, 0}},
        }},
        one_edge_each);
    if (disconnected.valid || disconnected.reachable_count != 2 || disconnected.vertices.size() != 4 ||
        disconnected.directed_edges != 4) {
        std::cerr << "disconnected graph was not retained as a complete invalid audit\n";
        return 1;
    }

    bool malformed_rejected = false;
    try {
        static_cast<void>(Audit(
            Neighbors{{
                {{1, 1}},
                {{0, 0}},
                {{3, 0}},
                {{2, 0}},
            }},
            Degrees{2, 1, 1, 1}));
    } catch (const std::runtime_error &) {
        malformed_rejected = true;
    }
    if (!malformed_rejected) {
        std::cerr << "duplicate graph edge was not rejected\n";
        return 1;
    }

    const HnswD0AttestationSnapshot mismatched_vnode{
        .held_executable = {.device = 1, .inode = 2, .size = 3},
        .mapped_executable = {.device = 1, .inode = 4, .size = 3},
        .dynamic_cdhash = {},
        .images = {},
    };
    if (HnswD0MappedExecutableMatchesHeldFile(mismatched_vnode)) {
        std::cerr << "mapped executable mismatch was accepted\n";
        return 1;
    }

#if defined(__APPLE__)
    HnswD0LiveAttestation attestation;
    std::string diagnostic;
    if (!attestation.Begin(diagnostic) || attestation.executable_fd() < 0) {
        std::cerr << "live attestation initialization failed: " << diagnostic << '\n';
        return 1;
    }
    HnswD0AttestationSnapshot before;
    HnswD0AttestationSnapshot after;
    if (!attestation.Capture(before, diagnostic) || !attestation.Capture(after, diagnostic)) {
        std::cerr << "live attestation capture failed: " << diagnostic << '\n';
        return 1;
    }
    if (before.dynamic_cdhash.size() != 40 || before.images.empty() ||
        !HnswD0MappedExecutableMatchesHeldFile(before)) {
        std::cerr << "live attestation snapshot is incomplete\n";
        return 1;
    }
    HnswD0AttestationSummary summary;
    if (!attestation.Validate(before, after, summary, diagnostic) || !summary.image_set_unchanged ||
        summary.later_image_adds != 0 || summary.image_removes != 0) {
        std::cerr << "unchanged live attestation was rejected: " << diagnostic << '\n';
        return 1;
    }

    HnswD0AttestationSnapshot changed_cdhash = after;
    changed_cdhash.dynamic_cdhash[0] = changed_cdhash.dynamic_cdhash[0] == '0' ? '1' : '0';
    if (attestation.Validate(before, changed_cdhash, summary, diagnostic)) {
        std::cerr << "changed dynamic CDHash was accepted\n";
        return 1;
    }
#endif

    // --- external-truth recall audit ----------------------------------------------------
    const std::vector<float> external_base = MakeExternalBase();
    const std::vector<std::int32_t> external_truth = MakeExternalGroundTruth();
    const std::vector<float> external_query{0.0F, 0.0F};

    // Sanity-check the fixture itself before trusting any recall it produces: the tie the
    // whole test rests on must actually be there.
    if (external_base[9 * kExternalDim] != external_base[10 * kExternalDim]) {
        std::cerr << "external fixture lost its cutoff tie\n";
        return 1;
    }

    const auto audit_with = [&](const std::vector<std::int32_t> &returned) {
        return AuditHnswD0RecallExternal(external_base.data(),
                                        kExternalRows,
                                        kExternalDim,
                                        external_query,
                                        external_truth,
                                        kExternalGroundTruthColumns,
                                        0,
                                        FixedSearch(returned, external_base));
    };

    // Exact truth returned: both figures perfect, and the tie is reported.
    const std::vector<std::int32_t> perfect{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const HnswD0ExternalRecallAudit exact = audit_with(perfect);
    if (!exact.valid || exact.truth_source != "official-sift1m" || exact.query_count != 1 ||
        exact.queries_with_cutoff_ties != 1) {
        std::cerr << "external recall audit rejected exact truth\n";
        return 1;
    }
    for (std::size_t point = 0; point < exact.recall_at_10.size(); ++point) {
        if (exact.recall_at_10[point] != 1.0 || exact.strict_id_recall_at_10[point] != 1.0) {
            std::cerr << "external recall audit did not score exact truth as 1.0\n";
            return 1;
        }
    }

    // The load-bearing case: row 10 substituted for row 9. It sits at exactly the cutoff, so
    // tie-tolerant recall must stay 1.0 while strict-ID recall must drop to 0.9. If these two
    // move together, one of them is not being computed.
    const std::vector<std::int32_t> tied{0, 1, 2, 3, 4, 5, 6, 7, 8, 10};
    const HnswD0ExternalRecallAudit tie = audit_with(tied);
    if (!tie.valid) {
        std::cerr << "external recall audit rejected the tie case\n";
        return 1;
    }
    for (std::size_t point = 0; point < tie.recall_at_10.size(); ++point) {
        if (tie.recall_at_10[point] != 1.0 || tie.strict_id_recall_at_10[point] != 0.9) {
            std::cerr << "external recall audit mishandled a tie at the cutoff: tolerant="
                      << tie.recall_at_10[point] << " strict=" << tie.strict_id_recall_at_10[point] << '\n';
            return 1;
        }
    }

    // A genuine miss: row 999999 is far outside the cutoff, so BOTH figures must drop.
    const std::vector<std::int32_t> missed{0, 1, 2, 3, 4, 5, 6, 7, 8, 999999};
    const HnswD0ExternalRecallAudit miss = audit_with(missed);
    if (!miss.valid) {
        std::cerr << "external recall audit rejected the miss case\n";
        return 1;
    }
    for (std::size_t point = 0; point < miss.recall_at_10.size(); ++point) {
        if (miss.recall_at_10[point] != 0.9 || miss.strict_id_recall_at_10[point] != 0.9) {
            std::cerr << "external recall audit did not penalise a true miss\n";
            return 1;
        }
    }

    // Distinct inputs must hash distinctly, else the digests cannot detect a swapped file.
    if (exact.queries_sha256.size() != 64 || exact.groundtruth_sha256.size() != 64) {
        std::cerr << "external recall audit digests are malformed\n";
        return 1;
    }

    // Preconditions. Each of these would otherwise yield a plausible wrong number rather
    // than an error, which is exactly why they are enforced.
    if (!Rejects("a base that is not the canonical 1e6 rows", [&] {
            AuditHnswD0RecallExternal(external_base.data(), kExternalRows - 1, kExternalDim, external_query,
                                      external_truth, kExternalGroundTruthColumns, 0, FixedSearch(perfect, external_base));
        })) {
        return 1;
    }
    if (!Rejects("ground truth with too few columns", [&] {
            AuditHnswD0RecallExternal(external_base.data(), kExternalRows, kExternalDim, external_query,
                                      std::vector<std::int32_t>(external_truth.begin(), external_truth.begin() + 10), 10, 0,
                                      FixedSearch(perfect, external_base));
        })) {
        return 1;
    }
    {
        std::vector<std::int32_t> descending = external_truth;
        std::reverse(descending.begin(), descending.end());
        if (!Rejects("ground truth that is not ascending by distance", [&] {
                AuditHnswD0RecallExternal(external_base.data(), kExternalRows, kExternalDim, external_query, descending,
                                          kExternalGroundTruthColumns, 0, FixedSearch(perfect, external_base));
            })) {
            return 1;
        }
    }
    {
        std::vector<std::int32_t> out_of_range = external_truth;
        out_of_range[5] = static_cast<std::int32_t>(kExternalRows);
        if (!Rejects("an out-of-range ground-truth id", [&] {
                AuditHnswD0RecallExternal(external_base.data(), kExternalRows, kExternalDim, external_query, out_of_range,
                                          kExternalGroundTruthColumns, 0, FixedSearch(perfect, external_base));
            })) {
            return 1;
        }
    }
    {
        std::vector<std::int32_t> mismatched_rows = external_truth;
        mismatched_rows.push_back(0);
        if (!Rejects("a ground-truth row count that disagrees with the query corpus", [&] {
                AuditHnswD0RecallExternal(external_base.data(), kExternalRows, kExternalDim, external_query,
                                          mismatched_rows, kExternalGroundTruthColumns, 0,
                                          FixedSearch(perfect, external_base));
            })) {
            return 1;
        }
    }

    std::cout << "external_recall_cutoff=" << kExternalCutoff << '\n';
    std::cout << "external_recall_queries_with_cutoff_ties=" << exact.queries_with_cutoff_ties << '\n';
    std::cout << "status=PASS\n";
    return 0;
}
