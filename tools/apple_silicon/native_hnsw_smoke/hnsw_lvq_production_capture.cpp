#include "hnsw_d0_audit.h"
#include "hnsw_lvq_replay_format.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

import std;
import std.compat;
import infinity_core;

namespace {

constexpr int kUsageExitCode = 64;
constexpr std::uint32_t kLvqHeaderBytes = 16;

struct Options {
    std::filesystem::path dataset_path;
    std::filesystem::path tape_path;
    std::size_t vector_count{};
    std::size_t dimension{};
    std::size_t m{};
    std::size_t ef_construction{};
    std::size_t chunk_size{};
    std::size_t capture_limit{};
    std::array<std::byte, 32> dataset_sha256{};
    std::array<std::byte, 32> capture_request_sha256{};
};

template <typename Integer>
Integer ParseInteger(const char *text, const char *name) {
    const std::string_view input(text);
    Integer parsed{};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), parsed, 10);
    if (input.empty() || error != std::errc{} || end != input.data() + input.size()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return parsed;
}

Options ParseOptions(int argc, char **argv) {
    if (argc != 11) {
        throw std::invalid_argument(
            "usage: infinity_hnsw_lvq_capture DATASET TAPE|- VECTORS DIMENSION M EF_CONSTRUCTION CHUNK_SIZE CAPTURE_LIMIT "
            "DATASET_SHA256 REQUEST_SHA256");
    }
    Options options{
        .dataset_path = argv[1],
        .tape_path = argv[2],
        .vector_count = ParseInteger<std::size_t>(argv[3], "vector count"),
        .dimension = ParseInteger<std::size_t>(argv[4], "dimension"),
        .m = ParseInteger<std::size_t>(argv[5], "M"),
        .ef_construction = ParseInteger<std::size_t>(argv[6], "efConstruction"),
        .chunk_size = ParseInteger<std::size_t>(argv[7], "chunk size"),
        .capture_limit = ParseInteger<std::size_t>(argv[8], "capture limit"),
        .dataset_sha256 = hnsw_lvq_replay::ParseSha256Hex(argv[9]),
        .capture_request_sha256 = hnsw_lvq_replay::ParseSha256Hex(argv[10]),
    };
    if (options.vector_count == 0 || options.dimension == 0 || options.m < 2 || options.ef_construction < options.m ||
        options.chunk_size == 0 || !std::has_single_bit(options.chunk_size) ||
        options.vector_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        options.dimension > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("capture configuration is out of range");
    }
    if (options.capture_limit == 0) {
        if (options.tape_path != "-") {
            throw std::invalid_argument("zero-limit control must use '-' as its tape path");
        }
    } else {
        if (!infinity::kHnswLvqCaptureEnabled) {
            throw std::invalid_argument("capture limit is nonzero but this binary has capture disabled");
        }
        if (options.tape_path == "-") {
            throw std::invalid_argument("capture requires a tape output path");
        }
    }
    return options;
}

class Dataset {
public:
    Dataset(const Dataset &) = delete;
    Dataset &operator=(const Dataset &) = delete;

    Dataset(const std::filesystem::path &path, std::size_t expected_bytes) {
        int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        fd_ = open(path.c_str(), flags);
        if (fd_ < 0) {
            ThrowSystemError("open", path);
        }
        struct stat before {};
        if (fstat(fd_, &before) != 0) {
            ThrowSystemError("fstat", path);
        }
        if (!S_ISREG(before.st_mode) || before.st_size < 0 || static_cast<std::uintmax_t>(before.st_size) != expected_bytes) {
            throw std::runtime_error("dataset is not a regular file of the expected size");
        }
        values_.resize(expected_bytes / sizeof(float));
        std::size_t offset = 0;
        auto *destination = reinterpret_cast<std::byte *>(values_.data());
        while (offset < expected_bytes) {
            const std::size_t request =
                std::min(expected_bytes - offset, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t count = pread(fd_, destination + offset, request, static_cast<off_t>(offset));
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                ThrowSystemError("pread", path);
            }
            offset += static_cast<std::size_t>(count);
        }
        struct stat after {};
        if (fstat(fd_, &after) != 0) {
            ThrowSystemError("fstat", path);
        }
        if (before.st_dev != after.st_dev || before.st_ino != after.st_ino || before.st_size != after.st_size ||
            before.st_mtimespec.tv_sec != after.st_mtimespec.tv_sec || before.st_mtimespec.tv_nsec != after.st_mtimespec.tv_nsec ||
            before.st_ctimespec.tv_sec != after.st_ctimespec.tv_sec || before.st_ctimespec.tv_nsec != after.st_ctimespec.tv_nsec) {
            throw std::runtime_error("dataset changed while being read");
        }
        if (close(fd_) != 0) {
            fd_ = -1;
            ThrowSystemError("close", path);
        }
        fd_ = -1;
        if (!std::all_of(values_.begin(), values_.end(), [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("dataset contains a non-finite value");
        }
    }

    ~Dataset() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    const float *data() const noexcept { return values_.data(); }

private:
    [[noreturn]] static void ThrowSystemError(const char *operation, const std::filesystem::path &path) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), std::string(operation) + " " + path.string());
    }

    int fd_{-1};
    std::vector<float> values_;
};

class CaptureSink final : public infinity::HnswLvqCaptureSink {
public:
    CaptureSink(std::size_t capture_limit, std::size_t dimension)
        : capture_limit_(capture_limit), dimension_(dimension), record_bytes_(kLvqHeaderBytes + dimension) {
        events_.reserve(capture_limit_);
        operands_.reserve(std::min(capture_limit_ * 2, std::size_t{32'768}));
    }

    void Observe(std::span<const std::byte> query_record,
                 std::span<const std::byte> candidate_record,
                 std::size_t dimension,
                 f32 distance,
                 const infinity::HnswLvqCallContext &context) override {
        ++observed_call_count_;
        const std::thread::id current_thread = std::this_thread::get_id();
        if (!thread_seen_) {
            thread_seen_ = true;
            capture_thread_ = current_thread;
        } else if (capture_thread_ != current_thread) {
            throw std::runtime_error("canonical HNSW LVQ capture observed more than one worker thread");
        }
        if (dimension != dimension_ || query_record.size() != record_bytes_ || candidate_record.size() != record_bytes_) {
            throw std::runtime_error("HNSW LVQ observer received a malformed record");
        }
        if (context.phase == infinity::HnswLvqPhase::kUnknown || context.layer < 0 || context.layer > 64 ||
            context.candidate_vertex < 0 || !std::isfinite(distance)) {
            throw std::runtime_error("HNSW LVQ observer received invalid context or distance");
        }
        if (events_.size() == capture_limit_) {
            return;
        }

        const std::uint64_t ordinal = events_.size();
        const std::uint64_t query_id =
            FindOrAddOperand(query_record, hnsw_lvq_replay::kQueryOperandRole, ordinal);
        const std::uint64_t candidate_id =
            FindOrAddOperand(candidate_record, hnsw_lvq_replay::kCandidateOperandRole, ordinal);
        events_.push_back(hnsw_lvq_replay::Event{
            .ordinal = ordinal,
            .query_operand_id = query_id,
            .candidate_operand_id = candidate_id,
            .query_vertex = context.query_vertex,
            .candidate_vertex = context.candidate_vertex,
            .layer = context.layer,
            .phase = static_cast<std::uint32_t>(context.phase),
            .distance_bits = std::bit_cast<std::uint32_t>(distance),
            .flags = 0,
        });
    }

    hnsw_lvq_replay::Tape Finish(const Options &options, const std::string &graph_sha256) && {
        if (!thread_seen_ || events_.size() != capture_limit_ || observed_call_count_ < capture_limit_) {
            throw std::runtime_error("HNSW LVQ capture did not produce the exact requested prefix");
        }
        return hnsw_lvq_replay::Tape{
            .dimension = static_cast<std::uint32_t>(dimension_),
            .record_bytes = static_cast<std::uint32_t>(record_bytes_),
            .worker_count = 1,
            .observed_call_count = observed_call_count_,
            .capture_limit = capture_limit_,
            .dataset_sha256 = options.dataset_sha256,
            .capture_request_sha256 = options.capture_request_sha256,
            .graph_sha256 = hnsw_lvq_replay::ParseSha256Hex(graph_sha256),
            .operands = std::move(operands_),
            .events = std::move(events_),
        };
    }

    std::size_t observed_call_count() const noexcept { return observed_call_count_; }
    std::size_t event_count() const noexcept { return events_.size(); }
    std::size_t operand_count() const noexcept { return operands_.size(); }

private:
    std::uint64_t FindOrAddOperand(std::span<const std::byte> record, std::uint32_t role, std::uint64_t event_ordinal) {
        const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(record.data()));
        auto &candidates = operands_by_address_[address];
        for (std::uint64_t id : candidates) {
            auto &operand = operands_[static_cast<std::size_t>(id)];
            if (std::equal(operand.record.begin(), operand.record.end(), record.begin(), record.end())) {
                operand.role_mask |= role;
                return id;
            }
        }
        const std::uint64_t id = operands_.size();
        operands_.push_back(hnsw_lvq_replay::Operand{
            .id = id,
            .captured_address = address,
            .first_event_ordinal = event_ordinal,
            .role_mask = role,
            .record = std::vector<std::byte>(record.begin(), record.end()),
        });
        candidates.push_back(id);
        return id;
    }

    std::size_t capture_limit_;
    std::size_t dimension_;
    std::size_t record_bytes_;
    std::size_t observed_call_count_{};
    bool thread_seen_{};
    std::thread::id capture_thread_{};
    std::vector<hnsw_lvq_replay::Operand> operands_;
    std::vector<hnsw_lvq_replay::Event> events_;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> operands_by_address_;
};

std::size_t CheckedDatasetBytes(const Options &options) {
    if (options.dimension > std::numeric_limits<std::size_t>::max() / options.vector_count ||
        options.dimension * options.vector_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("dataset byte count overflows");
    }
    return options.dimension * options.vector_count * sizeof(float);
}

HnswD0GraphAudit AuditGraph(const auto &index, std::size_t vector_count) {
    const auto [max_level, entry_point] = index.GetGraphEnterPoint();
    return AuditHnswD0Graph(HnswD0GraphInput{
        .vertex_count = vector_count,
        .max_level = max_level,
        .entry_point = entry_point,
        .level0_capacity = index.GetGraphMmax0(),
        .upper_capacity = index.GetGraphMmax(),
        .level = [&](std::int32_t vertex) { return index.GetGraphLevel(vertex); },
        .label = [&](std::int32_t vertex) { return static_cast<std::int64_t>(index.GetLabel(vertex)); },
        .neighbors =
            [&](std::int32_t vertex, std::int32_t layer) {
                const auto [neighbors, degree] = index.GetGraphNeighbors(vertex, layer);
                return HnswD0NeighborRange{
                    .data = neighbors,
                    .size = static_cast<std::size_t>(degree),
                    .capacity = layer == 0 ? index.GetGraphMmax0() : index.GetGraphMmax(),
                    .tail_valid = true,
                };
            },
    });
}

int Run(const Options &options) {
    if (std::endian::native != std::endian::little) {
        throw std::runtime_error("HNSW LVQ capture requires a little-endian host");
    }
    Dataset dataset(options.dataset_path, CheckedDatasetBytes(options));
    using Label = std::int32_t;
    using Hnsw = infinity::KnnHnsw<infinity::LVQL2VecStoreType<float, i8>, Label>;
    const std::size_t max_chunk_count = (options.vector_count - 1) / options.chunk_size + 1;
    auto index = Hnsw::Make(
        options.chunk_size, max_chunk_count, options.dimension, options.m, options.ef_construction);
    infinity::ctpl::thread_pool build_pool(1);
    CaptureSink sink(options.capture_limit, options.dimension);
    infinity::DenseVectorIter<float, Label> iterator(dataset.data(), options.dimension, options.vector_count);
    infinity::HnswBulkBuildResult build_result;
    if (options.capture_limit == 0) {
        build_result =
            infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, build_pool);
    } else {
        infinity::HnswLvqCaptureSession session(sink);
        build_result =
            infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, build_pool);
    }
    index->Check();
    const HnswD0GraphAudit graph = AuditGraph(*index, options.vector_count);
    if (!graph.valid || graph.vertex_count != options.vector_count || graph.reachable_count != options.vector_count) {
        throw std::runtime_error("captured HNSW graph audit failed");
    }

    std::uintmax_t tape_bytes = 0;
    std::size_t event_count = 0;
    std::size_t operand_count = 0;
    std::size_t observed_call_count = 0;
    if (options.capture_limit != 0) {
        event_count = sink.event_count();
        operand_count = sink.operand_count();
        observed_call_count = sink.observed_call_count();
        hnsw_lvq_replay::Tape tape = std::move(sink).Finish(options, graph.graph_sha256);
        hnsw_lvq_replay::WriteTape(options.tape_path, tape);
        static_cast<void>(hnsw_lvq_replay::ReadTape(options.tape_path));
        tape_bytes = std::filesystem::file_size(options.tape_path);
    }

    std::cout << "status=PASS\n";
    std::cout << "scope=production-origin-lvq-capture-preflight\n";
    std::cout << "capture_compiled=" << (infinity::kHnswLvqCaptureEnabled ? 1 : 0) << '\n';
    std::cout << "capture_active=" << (options.capture_limit != 0 ? 1 : 0) << '\n';
    std::cout << "workers=1\n";
    std::cout << "vectors=" << options.vector_count << '\n';
    std::cout << "dimension=" << options.dimension << '\n';
    std::cout << "M=" << options.m << '\n';
    std::cout << "ef_construction=" << options.ef_construction << '\n';
    std::cout << "capture_limit=" << options.capture_limit << '\n';
    std::cout << "captured_events=" << event_count << '\n';
    std::cout << "observed_calls=" << observed_call_count << '\n';
    std::cout << "operands=" << operand_count << '\n';
    std::cout << "tape_bytes=" << tape_bytes << '\n';
    std::cout << "dataset_sha256=" << hnsw_lvq_replay::Sha256Hex(options.dataset_sha256) << '\n';
    std::cout << "capture_request_sha256=" << hnsw_lvq_replay::Sha256Hex(options.capture_request_sha256) << '\n';
    std::cout << "graph_sha256=" << graph.graph_sha256 << '\n';
    std::cout << "graph_levels_sha256=" << graph.levels_sha256 << '\n';
    std::cout << "graph_directed_edges=" << graph.directed_edges << '\n';
    std::cout << "graph_level0_directed_edges=" << graph.level0_directed_edges << '\n';
    std::cout << "memory_delta=" << build_result.mem_usage_ << '\n';
    std::cout << "submitted_tasks=" << build_result.submitted_task_count_ << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        return Run(ParseOptions(argc, argv));
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << '\n';
        return kUsageExitCode;
    } catch (const std::exception &error) {
        std::cerr << "HNSW LVQ production capture failed: " << error.what() << '\n';
        return 1;
    }
}
