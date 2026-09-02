#pragma once

// Loading the PUBLISHED SIFT1M query set and ground truth, and running the external-truth
// recall audit from either engine's bridge.
//
// WHY THIS EXISTS. AuditHnswD0RecallExternal has been implemented and unit-tested since
// commit fcd910c5c, but nothing ever called it: every recorded benchmark number came from
// AuditHnswD0Recall, a self-audit over 64 synthetic held-out queries whose truth it derives
// itself by exhaustive double-precision search. That audit is not buggy and its queries are
// not resampled -- they are a fixed SplitMix64 draw (hnsw_d0_query_benchmark.cpp) and its
// truth is exact -- but it is unfit for LOCATING an iso-recall point, for two separate
// reasons:
//
//   1. WRONG DISTRIBUTION. The synthetic queries are uniform in [0,1) per coordinate, while
//      canonical SIFT descriptors are integer-valued on a scale of tens. They are therefore a
//      tightly clustered, near-origin, out-of-distribution workload. Recall measured on them
//      is a real quantity, but it is not SIFT recall and its Infinity-vs-FAISS gap does not
//      transfer: measured here at n=1,000,000, efC=200, the self-audit reports a deficit of
//      0.0172 at ef=64 and 0.0266 at ef=128, while the published queries put the same two
//      deficits at 0.00083 and 0.00045 -- a factor of ~20 and ~59.
//   2. COARSE GRANULARITY. recall@10 over 64 queries has 640 neighbour slots, so its finest
//      step is 1/640 = 0.0016 and one query's whole result set is worth 0.0156. Across 35
//      recorded efC=200 1M builds the self-audit's ef=128 recall ranges 0.8297..0.8703 -- a
//      spread of 0.041 that looks like enormous build-to-build variance but is 26 of those 640
//      slots, i.e. 2.6 queries' worth. Measured over the 10,000 published queries instead, the
//      same build-to-build variance is ~0.0005. The variance was mostly in the ruler, not the
//      graph.
//
// The consequence was not academic. A single draw from that 0.041-wide distribution is what
// placed the project's "iso-recall" operating point at efConstruction=250; on published truth
// the crossing is at efC 225-235, which is ~7 s of build time.
//
// The published truth fixes both problems at once: the right query distribution, and 10,000
// queries instead of 64. It also makes the number comparable to published SIFT1M recall,
// which the self-audit never was.
//
// Enabled by environment variable so the default run keeps its existing cost and output, and
// so a campaign can turn it on for the runs where it matters:
//   HNSW_D0_EXTERNAL_QUERIES      path to query.f32      (nq x dimension, f32 row-major)
//   HNSW_D0_EXTERNAL_GROUNDTRUTH  path to groundtruth.i32 (nq x cols, i32 row-major)
//   HNSW_D0_EXTERNAL_QUERY_LIMIT  evaluate only the first N queries (0/unset = all 10,000)
// Both paths must be set, or the audit is skipped. It is only meaningful at n=1,000,000
// (the published IDs address the full canonical base); AuditHnswD0RecallExternal enforces
// that itself, so this loader simply declines to run below that size rather than producing
// a plausible wrong number.

#include "hnsw_d0_audit.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct HnswD0ExternalTruthRequest {
    bool requested = false;      // both env paths were set
    std::string queries_path;
    std::string groundtruth_path;
    std::size_t query_limit = 0; // 0 == all
    std::string skip_reason;     // why the audit did not run, when it did not
};

inline HnswD0ExternalTruthRequest HnswD0ReadExternalTruthRequest() {
    HnswD0ExternalTruthRequest request;
    const char *queries = std::getenv("HNSW_D0_EXTERNAL_QUERIES");
    const char *truth = std::getenv("HNSW_D0_EXTERNAL_GROUNDTRUTH");
    if (queries == nullptr || truth == nullptr || *queries == '\0' || *truth == '\0') {
        request.skip_reason = "not requested";
        return request;
    }
    request.requested = true;
    request.queries_path = queries;
    request.groundtruth_path = truth;
    if (const char *limit = std::getenv("HNSW_D0_EXTERNAL_QUERY_LIMIT"); limit != nullptr) {
        char *parse_end = nullptr;
        const unsigned long long parsed = std::strtoull(limit, &parse_end, 10);
        if (parse_end != limit && *parse_end == '\0') {
            request.query_limit = static_cast<std::size_t>(parsed);
        }
    }
    return request;
}

// Read a whole file of fixed-width little-endian records. Returns false and leaves `out`
// untouched on any failure, including a size that is not a whole multiple of the row width --
// a truncated file would otherwise silently shorten the query set.
template <typename T>
inline bool HnswD0ReadRawRows(const std::string &path, std::size_t columns, std::vector<T> &out, std::string &error) {
    if (columns == 0) {
        error = "zero columns";
        return false;
    }
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open " + path;
        return false;
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        error = "cannot seek " + path;
        return false;
    }
    const long size = std::ftell(file);
    if (size < 0) {
        std::fclose(file);
        error = "cannot size " + path;
        return false;
    }
    std::rewind(file);
    const std::size_t bytes = static_cast<std::size_t>(size);
    if (bytes == 0 || bytes % (sizeof(T) * columns) != 0) {
        std::fclose(file);
        error = path + " is not a whole number of " + std::to_string(columns) + "-column rows";
        return false;
    }
    std::vector<T> values(bytes / sizeof(T));
    const std::size_t read = std::fread(values.data(), sizeof(T), values.size(), file);
    std::fclose(file);
    if (read != values.size()) {
        error = "short read on " + path;
        return false;
    }
    out = std::move(values);
    return true;
}

// Run the audit if it was requested and is meaningful here. On any problem the result stays
// invalid and `skip_reason` explains why, so a caller emits a reason rather than silence.
inline HnswD0ExternalRecallAudit HnswD0RunExternalTruthAudit(const float *base,
                                                            std::size_t vector_count,
                                                            std::size_t dimension,
                                                            const HnswD0Search &search,
                                                            HnswD0ExternalTruthRequest &request) {
    HnswD0ExternalRecallAudit audit;
    if (!request.requested) {
        return audit;
    }
    if (vector_count != kHnswD0ExternalTruthVectorCount) {
        request.skip_reason = "n != " + std::to_string(kHnswD0ExternalTruthVectorCount)
                              + "; published ground-truth IDs address the full canonical base";
        return audit;
    }
    std::vector<float> queries;
    std::string error;
    if (!HnswD0ReadRawRows<float>(request.queries_path, dimension, queries, error)) {
        request.skip_reason = error;
        return audit;
    }
    const std::size_t query_rows = queries.size() / dimension;
    std::vector<std::int32_t> truth;
    // The ground-truth column count is whatever the file's row count implies once the query
    // count is known, so a 100-column SIFT file and a 10-column one both work without a knob.
    if (query_rows == 0) {
        request.skip_reason = "empty query file";
        return audit;
    }
    if (!HnswD0ReadRawRows<std::int32_t>(request.groundtruth_path, query_rows, truth, error)) {
        request.skip_reason = error;
        return audit;
    }
    const std::size_t columns = truth.size() / query_rows;
    if (columns <= 10) {
        request.skip_reason = "ground truth has " + std::to_string(columns)
                              + " columns; recall@10 with ties needs more than 10";
        return audit;
    }
    audit = AuditHnswD0RecallExternal(base, vector_count, dimension, queries, truth, columns,
                                      request.query_limit, search);
    if (!audit.valid) {
        request.skip_reason = "AuditHnswD0RecallExternal reported invalid";
    }
    return audit;
}
