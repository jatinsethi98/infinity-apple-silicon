#include "hnsw_d0_audit.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

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

    std::cout << "status=PASS\n";
    return 0;
}
