#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

import std;
import std.compat;
import infinity_core;

#if !defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
#error "production exactness emitter requires incremental reciprocal execution evidence"
#endif

#if !defined(INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
#error "production exactness emitter requires thresholded Batch4 execution evidence"
#endif

namespace {

constexpr int kUsageExitCode = 64;
constexpr std::size_t kVectorCount = 12'288;
constexpr std::size_t kDimension = 128;
constexpr std::size_t kChunkSize = 8'192;
constexpr std::size_t kMaxChunks = 2;
constexpr std::size_t kM = 32;
constexpr std::size_t kEfConstruction = 200;
constexpr std::size_t kWorkerCount = 1;
constexpr std::size_t kDatasetBytes = kVectorCount * kDimension * sizeof(float);
constexpr std::uint32_t kGraphSchemaVersion = 1;
constexpr std::uint32_t kJsonSchemaVersion = 1;
constexpr std::uint32_t kExecutionEvidenceSchemaVersion = 2;
constexpr std::uint32_t kGraphHeaderBytes = 184;
constexpr std::uint32_t kExecutionEvidenceBytes = 56;
constexpr std::string_view kDatasetSha256 = "f2e29c0f1a64d48a81e2adba0213c53d3015f459ef9936668a7dabb6a025f32a";
constexpr std::array<std::uint8_t, 32> kDatasetDigest{
    0xf2, 0xe2, 0x9c, 0x0f, 0x1a, 0x64, 0xd4, 0x8a, 0x81, 0xe2, 0xad, 0xba, 0x02, 0x13, 0xc5, 0x3d,
    0x30, 0x15, 0xf4, 0x59, 0xef, 0x99, 0x36, 0x66, 0x8a, 0x7d, 0xab, 0xb6, 0xa0, 0x25, 0xf3, 0x2a,
};
constexpr std::array<std::uint8_t, 8> kGraphMagic{'I', 'F', 'H', 'X', 'G', 'R', '0', '1'};
constexpr std::array<std::uint8_t, 8> kExecutionEvidenceMagic{'I', 'F', 'H', 'X', 'W', 'T', '0', '1'};

using Label = std::int32_t;
using Hnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, Label>;
using Digest = std::array<std::uint8_t, 32>;

static_assert(kDatasetBytes == 6'291'456);
static_assert((kVectorCount - 1) / kChunkSize + 1 == kMaxChunks);

struct Options {
    int dataset_fd{-1};
    int graph_fd{-1};
    int save_to_ptr_fd{-1};
    int execution_evidence_fd{-1};
};

struct DescriptorIdentity {
    dev_t device{};
    ino_t inode{};
};

struct DescriptorSnapshot {
    dev_t device{};
    ino_t inode{};
    mode_t mode{};
    nlink_t link_count{};
    off_t size{};
    timespec modification_time{};
    timespec status_change_time{};
};

struct DescriptorSet {
    DescriptorSnapshot dataset;
    DescriptorSnapshot graph;
    DescriptorSnapshot save_to_ptr;
    DescriptorSnapshot execution_evidence;
};

struct FrozenDataset {
    std::vector<float> values;
    DescriptorSnapshot snapshot;
};

struct Build {
    std::unique_ptr<Hnsw> index;
    infinity::HnswBulkBuildResult result;
    infinity::HnswIncrementalReciprocalExecutionEvidence incremental_execution_evidence;
    infinity::HnswThresholdBatch4ExecutionEvidence threshold_execution_evidence;
};

class StandardStreamAliasError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU,
    0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU,
    0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U,
    0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

[[noreturn]] void ThrowSystemError(std::string_view operation) {
    const int error = errno;
    throw std::system_error(error, std::generic_category(), std::string(operation));
}

void Require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::uint32_t RotateRight(std::uint32_t value, unsigned int count) { return (value >> count) | (value << (32U - count)); }

class Sha256 {
public:
    void Update(const void *input, std::size_t size) {
        if (size > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
            throw std::length_error("SHA-256 input is too large");
        }
        const auto *data = static_cast<const std::uint8_t *>(input);
        total_bytes_ += size;
        while (size > 0) {
            const std::size_t copied = std::min(size, block_.size() - block_size_);
            std::copy_n(data, copied, block_.data() + block_size_);
            block_size_ += copied;
            data += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                Compress(block_.data());
                block_size_ = 0;
            }
        }
    }

    Digest Finalize() const {
        Sha256 finalized = *this;
        const std::uint64_t bit_count = finalized.total_bytes_ * 8U;
        std::array<std::uint8_t, 128> padding{};
        padding[0] = 0x80;
        const std::size_t padding_size = finalized.block_size_ < 56 ? 56 - finalized.block_size_ : 120 - finalized.block_size_;
        finalized.Update(padding.data(), padding_size);
        std::array<std::uint8_t, 8> length{};
        for (std::size_t index = 0; index < length.size(); ++index) {
            length[length.size() - 1 - index] = static_cast<std::uint8_t>(bit_count >> (index * 8U));
        }
        finalized.Update(length.data(), length.size());

        Digest digest{};
        for (std::size_t word = 0; word < finalized.state_.size(); ++word) {
            for (std::size_t byte = 0; byte < 4; ++byte) {
                digest[word * 4 + byte] = static_cast<std::uint8_t>(finalized.state_[word] >> ((3 - byte) * 8U));
            }
        }
        return digest;
    }

private:
    void Compress(const std::uint8_t *block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = index * 4;
            words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) | (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                           (static_cast<std::uint32_t>(block[offset + 2]) << 8U) | static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const std::uint32_t previous_15 = words[index - 15];
            const std::uint32_t previous_2 = words[index - 2];
            const std::uint32_t sigma0 = RotateRight(previous_15, 7) ^ RotateRight(previous_15, 18) ^ (previous_15 >> 3U);
            const std::uint32_t sigma1 = RotateRight(previous_2, 17) ^ RotateRight(previous_2, 19) ^ (previous_2 >> 10U);
            words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t temporary1 = h + sum1 + choice + kSha256RoundConstants[index] + words[index];
            const std::uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_{};
    std::uint64_t total_bytes_{};
};

Digest Hash(std::span<const std::uint8_t> bytes) {
    Sha256 hash;
    hash.Update(bytes.data(), bytes.size());
    return hash.Finalize();
}

std::string Hex(const Digest &digest) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (const std::uint8_t byte : digest) {
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

int ParseDescriptor(const char *text, std::string_view name) {
    const std::string_view input(text == nullptr ? "" : text);
    int descriptor = -1;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), descriptor, 10);
    if (input.empty() || error != std::errc{} || end != input.data() + input.size() || descriptor <= STDERR_FILENO) {
        throw std::invalid_argument("invalid " + std::string(name));
    }
    return descriptor;
}

Options ParseOptions(int argc, char **argv) {
    if (argc != 9 || std::string_view(argv[1]) != "--dataset-fd" || std::string_view(argv[3]) != "--graph-fd" ||
        std::string_view(argv[5]) != "--save-to-ptr-fd" || std::string_view(argv[7]) != "--execution-evidence-fd") {
        throw std::invalid_argument("usage: infinity_hnsw_exactness_emitter_production --dataset-fd N --graph-fd N --save-to-ptr-fd N "
                                    "--execution-evidence-fd N");
    }
    Options options{
        .dataset_fd = ParseDescriptor(argv[2], "dataset descriptor"),
        .graph_fd = ParseDescriptor(argv[4], "graph descriptor"),
        .save_to_ptr_fd = ParseDescriptor(argv[6], "SaveToPtr descriptor"),
        .execution_evidence_fd = ParseDescriptor(argv[8], "execution evidence descriptor"),
    };
    const std::array descriptors{
        options.dataset_fd,
        options.graph_fd,
        options.save_to_ptr_fd,
        options.execution_evidence_fd,
    };
    for (std::size_t left = 0; left < descriptors.size(); ++left) {
        for (std::size_t right = left + 1; right < descriptors.size(); ++right) {
            if (descriptors[left] == descriptors[right]) {
                throw std::invalid_argument("descriptors must be distinct");
            }
        }
    }
    return options;
}

DescriptorIdentity IdentifyDescriptor(int descriptor, std::string_view name) {
    struct stat status{};
    if (fstat(descriptor, &status) != 0) {
        ThrowSystemError("fstat " + std::string(name));
    }
    return DescriptorIdentity{
        .device = status.st_dev,
        .inode = status.st_ino,
    };
}

std::optional<DescriptorIdentity> IdentifyStandardStream(int descriptor, std::string_view name) {
    struct stat status{};
    if (fstat(descriptor, &status) != 0) {
        if (errno == EBADF) {
            return std::nullopt;
        }
        ThrowSystemError("fstat " + std::string(name));
    }
    return DescriptorIdentity{
        .device = status.st_dev,
        .inode = status.st_ino,
    };
}

bool SameObject(const DescriptorIdentity &left, const DescriptorIdentity &right) { return left.device == right.device && left.inode == right.inode; }

void RejectStandardStreamAliases(const Options &options) {
    const std::array artifacts{
        std::pair{IdentifyDescriptor(options.dataset_fd, "dataset descriptor"), std::string_view("dataset descriptor")},
        std::pair{IdentifyDescriptor(options.graph_fd, "graph descriptor"), std::string_view("graph descriptor")},
        std::pair{IdentifyDescriptor(options.save_to_ptr_fd, "SaveToPtr descriptor"), std::string_view("SaveToPtr descriptor")},
        std::pair{IdentifyDescriptor(options.execution_evidence_fd, "execution evidence descriptor"),
                  std::string_view("execution evidence descriptor")},
    };
    const std::array streams{
        std::pair{IdentifyStandardStream(STDIN_FILENO, "stdin"), std::string_view("stdin")},
        std::pair{IdentifyStandardStream(STDOUT_FILENO, "stdout"), std::string_view("stdout")},
        std::pair{IdentifyStandardStream(STDERR_FILENO, "stderr"), std::string_view("stderr")},
    };
    for (const auto &[artifact, artifact_name] : artifacts) {
        for (const auto &[stream, stream_name] : streams) {
            if (stream.has_value() && SameObject(artifact, *stream)) {
                throw StandardStreamAliasError(std::string(artifact_name) + " aliases " + std::string(stream_name));
            }
        }
    }
}

DescriptorSnapshot SnapshotDescriptor(int descriptor, std::string_view name) {
    struct stat status{};
    if (fstat(descriptor, &status) != 0) {
        ThrowSystemError("fstat " + std::string(name));
    }
    Require(S_ISREG(status.st_mode), std::string(name) + " is not a regular file");
    Require(status.st_nlink == 1, std::string(name) + " does not have exactly one link");
    Require(status.st_size >= 0, std::string(name) + " has a negative size");
    return DescriptorSnapshot{
        .device = status.st_dev,
        .inode = status.st_ino,
        .mode = status.st_mode,
        .link_count = status.st_nlink,
        .size = status.st_size,
        .modification_time = status.st_mtimespec,
        .status_change_time = status.st_ctimespec,
    };
}

bool SameObject(const DescriptorSnapshot &left, const DescriptorSnapshot &right) { return left.device == right.device && left.inode == right.inode; }

bool SameStableMetadata(const DescriptorSnapshot &left, const DescriptorSnapshot &right) {
    return SameObject(left, right) && left.mode == right.mode && left.link_count == right.link_count && left.size == right.size &&
           left.modification_time.tv_sec == right.modification_time.tv_sec && left.modification_time.tv_nsec == right.modification_time.tv_nsec &&
           left.status_change_time.tv_sec == right.status_change_time.tv_sec && left.status_change_time.tv_nsec == right.status_change_time.tv_nsec;
}

void RequireOffset(int descriptor, off_t expected, std::string_view name) {
    const off_t offset = lseek(descriptor, 0, SEEK_CUR);
    if (offset < 0) {
        ThrowSystemError("lseek " + std::string(name));
    }
    Require(offset == expected, std::string(name) + " has an unexpected offset");
}

void Seek(int descriptor, off_t offset, std::string_view name) {
    if (lseek(descriptor, offset, SEEK_SET) != offset) {
        ThrowSystemError("lseek " + std::string(name));
    }
}

int DescriptorFlags(int descriptor, std::string_view name) {
    const int flags = fcntl(descriptor, F_GETFL);
    if (flags < 0) {
        ThrowSystemError("fcntl " + std::string(name));
    }
    return flags;
}

DescriptorSet ValidateDescriptors(const Options &options) {
    RejectStandardStreamAliases(options);
    const DescriptorSnapshot dataset = SnapshotDescriptor(options.dataset_fd, "dataset descriptor");
    const DescriptorSnapshot graph = SnapshotDescriptor(options.graph_fd, "graph descriptor");
    const DescriptorSnapshot save_to_ptr = SnapshotDescriptor(options.save_to_ptr_fd, "SaveToPtr descriptor");
    const DescriptorSnapshot execution_evidence = SnapshotDescriptor(options.execution_evidence_fd, "execution evidence descriptor");

    const std::array outputs{graph, save_to_ptr, execution_evidence};
    for (const DescriptorSnapshot &output : outputs) {
        Require(!SameObject(dataset, output), "descriptor files must be distinct");
    }
    for (std::size_t left = 0; left < outputs.size(); ++left) {
        for (std::size_t right = left + 1; right < outputs.size(); ++right) {
            Require(!SameObject(outputs[left], outputs[right]), "descriptor files must be distinct");
        }
    }
    Require(dataset.size == static_cast<off_t>(kDatasetBytes), "dataset has the wrong byte size");
    Require(graph.size == 0, "graph output is not empty");
    Require(save_to_ptr.size == 0, "SaveToPtr output is not empty");
    Require(execution_evidence.size == 0, "execution evidence output is not empty");

    const int dataset_flags = DescriptorFlags(options.dataset_fd, "dataset descriptor");
    const int graph_flags = DescriptorFlags(options.graph_fd, "graph descriptor");
    const int save_to_ptr_flags = DescriptorFlags(options.save_to_ptr_fd, "SaveToPtr descriptor");
    const int execution_evidence_flags = DescriptorFlags(options.execution_evidence_fd, "execution evidence descriptor");
    Require((dataset_flags & O_ACCMODE) != O_WRONLY, "dataset descriptor is not readable");
    Require((graph_flags & O_ACCMODE) == O_RDWR && (graph_flags & O_APPEND) == 0, "graph descriptor must be non-append O_RDWR");
    Require((save_to_ptr_flags & O_ACCMODE) == O_RDWR && (save_to_ptr_flags & O_APPEND) == 0, "SaveToPtr descriptor must be non-append O_RDWR");
    Require((execution_evidence_flags & O_ACCMODE) == O_RDWR && (execution_evidence_flags & O_APPEND) == 0,
            "execution evidence descriptor must be non-append O_RDWR");

    RequireOffset(options.dataset_fd, 0, "dataset descriptor");
    RequireOffset(options.graph_fd, 0, "graph descriptor");
    RequireOffset(options.save_to_ptr_fd, 0, "SaveToPtr descriptor");
    RequireOffset(options.execution_evidence_fd, 0, "execution evidence descriptor");
    return DescriptorSet{
        .dataset = dataset,
        .graph = graph,
        .save_to_ptr = save_to_ptr,
        .execution_evidence = execution_evidence,
    };
}

void ReadExactAt(int descriptor, std::span<std::uint8_t> destination, std::string_view name) {
    std::size_t offset = 0;
    while (offset < destination.size()) {
        const std::size_t request = std::min(destination.size() - offset, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t count = pread(descriptor, destination.data() + offset, request, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            ThrowSystemError("pread " + std::string(name));
        }
        Require(count > 0, std::string(name) + " ended before its recorded size");
        offset += static_cast<std::size_t>(count);
    }
}

std::vector<std::uint8_t> ReadStableBytes(int descriptor, std::size_t size, const DescriptorSnapshot &expected, std::string_view name) {
    const DescriptorSnapshot before = SnapshotDescriptor(descriptor, name);
    Require(SameObject(before, expected), std::string(name) + " was replaced");
    Require(before.size == static_cast<off_t>(size), std::string(name) + " has the wrong byte size");
    std::vector<std::uint8_t> bytes(size);
    ReadExactAt(descriptor, bytes, name);
    const DescriptorSnapshot after = SnapshotDescriptor(descriptor, name);
    Require(SameStableMetadata(before, after), std::string(name) + " changed while being read");
    return bytes;
}

FrozenDataset ReadFrozenDataset(int descriptor, const DescriptorSnapshot &expected) {
    const DescriptorSnapshot before = SnapshotDescriptor(descriptor, "dataset descriptor");
    Require(SameStableMetadata(before, expected), "dataset metadata changed before reading");
    std::vector<float> values(kVectorCount * kDimension);
    auto bytes = std::span<std::uint8_t>(reinterpret_cast<std::uint8_t *>(values.data()), kDatasetBytes);
    ReadExactAt(descriptor, bytes, "dataset descriptor");
    const DescriptorSnapshot after = SnapshotDescriptor(descriptor, "dataset descriptor");
    Require(SameStableMetadata(before, after), "dataset changed while being read");
    Require(Hash(bytes) == kDatasetDigest, "dataset SHA-256 does not match the frozen corpus");
    Require(std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }), "dataset contains a non-finite value");
    return FrozenDataset{
        .values = std::move(values),
        .snapshot = after,
    };
}

void VerifyFrozenDatasetUnchanged(int descriptor, const FrozenDataset &dataset) {
    const std::vector<std::uint8_t> bytes = ReadStableBytes(descriptor, kDatasetBytes, dataset.snapshot, "dataset descriptor");
    Require(std::memcmp(bytes.data(), dataset.values.data(), bytes.size()) == 0, "dataset bytes changed during exactness emission");
    Require(Hash(bytes) == kDatasetDigest, "dataset SHA-256 changed during exactness emission");
}

Build BuildExactIndex(const FrozenDataset &dataset) {
    infinity::ctpl::thread_pool build_pool(kWorkerCount);
    auto index = Hnsw::Make(kChunkSize, kMaxChunks, kDimension, kM, kEfConstruction);
    const bool incremental_armed = index->ArmIncrementalReciprocalExecutionEvidence();
    const bool threshold_armed = index->ArmThresholdBatch4ExecutionEvidence();
    infinity::DenseVectorIter<float, Label> iterator(dataset.values.data(), kDimension, kVectorCount);
    const infinity::HnswBulkBuildResult result =
        infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, build_pool);

    Require(build_pool.size() == static_cast<int>(kWorkerCount), "HNSW build did not use exactly one worker");
    Require(result.start_ == 0, "HNSW build started at the wrong ordinal");
    Require(result.end_ == kVectorCount, "HNSW build did not cover the frozen corpus");
    Require(result.submitted_task_count_ == 1, "HNSW build did not submit exactly one task");
    Require(result.mem_usage_ > 0, "HNSW build reported no memory growth");
    Require(index->GetVecNum() == kVectorCount, "HNSW index has the wrong vector count");
    Require(index->GetGraphMmax0() == 2 * kM, "HNSW level-zero capacity is wrong");
    Require(index->GetGraphMmax() == kM, "HNSW upper-level capacity is wrong");
    index->Check();
    const infinity::HnswIncrementalReciprocalExecutionEvidence incremental_execution_evidence = index->GetIncrementalReciprocalExecutionEvidence();
    if (incremental_execution_evidence.treatment_compiled) {
        Require(incremental_armed, "incremental treatment execution evidence was not armed");
        Require(incremental_execution_evidence.capture_armed, "incremental treatment execution evidence lost its armed state");
        Require(incremental_execution_evidence.eligible_branch_entered, "incremental treatment did not enter the eligible reciprocal branch");
        Require(incremental_execution_evidence.successful_unchanged_observed, "incremental treatment did not observe a successful unchanged result");
        Require(incremental_execution_evidence.successful_updated_observed, "incremental treatment did not observe a successful updated result");
    } else {
        Require(!incremental_armed, "incremental control unexpectedly armed treatment execution evidence");
        Require(!incremental_execution_evidence.capture_armed && !incremental_execution_evidence.eligible_branch_entered &&
                    !incremental_execution_evidence.successful_unchanged_observed && !incremental_execution_evidence.successful_updated_observed,
                "incremental control reported treatment execution evidence");
    }
    const infinity::HnswThresholdBatch4ExecutionEvidence threshold_execution_evidence = index->GetThresholdBatch4ExecutionEvidence();
    if (threshold_execution_evidence.treatment_compiled) {
        Require(threshold_armed, "threshold treatment execution evidence was not armed");
        Require(threshold_execution_evidence.capture_armed, "threshold treatment execution evidence lost its armed state");
        Require(threshold_execution_evidence.eligible_branch_entered, "threshold treatment did not enter the eligible Batch4 branch");
        Require(threshold_execution_evidence.rejected_lane_observed, "threshold treatment did not observe a rejected lane");
        Require(threshold_execution_evidence.surviving_lane_observed,
                "threshold treatment did not fully evaluate a lane that escaped early rejection");
    } else {
        Require(!threshold_armed, "threshold control unexpectedly armed treatment execution evidence");
        Require(!threshold_execution_evidence.capture_armed && !threshold_execution_evidence.eligible_branch_entered &&
                    !threshold_execution_evidence.rejected_lane_observed && !threshold_execution_evidence.surviving_lane_observed,
                "threshold control reported treatment execution evidence");
    }
    return Build{
        .index = std::move(index),
        .result = result,
        .incremental_execution_evidence = incremental_execution_evidence,
        .threshold_execution_evidence = threshold_execution_evidence,
    };
}

void AppendU32(std::vector<std::uint8_t> &output, std::uint32_t value) {
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        output.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
    }
}

void AppendI32(std::vector<std::uint8_t> &output, std::int32_t value) { AppendU32(output, static_cast<std::uint32_t>(value)); }

void AppendU64(std::vector<std::uint8_t> &output, std::uint64_t value) {
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        output.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
    }
}

std::vector<std::uint8_t> CanonicalExecutionEvidence(const infinity::HnswIncrementalReciprocalExecutionEvidence &incremental_evidence,
                                                     const infinity::HnswThresholdBatch4ExecutionEvidence &threshold_evidence) {
    std::vector<std::uint8_t> output;
    output.reserve(kExecutionEvidenceBytes);
    output.insert(output.end(), kExecutionEvidenceMagic.begin(), kExecutionEvidenceMagic.end());
    AppendU32(output, kExecutionEvidenceSchemaVersion);
    AppendU32(output, kExecutionEvidenceBytes);
    AppendU32(output, incremental_evidence.treatment_compiled ? 1U : 0U);
    AppendU32(output, incremental_evidence.capture_armed ? 1U : 0U);
    AppendU32(output, incremental_evidence.eligible_branch_entered ? 1U : 0U);
    AppendU32(output, incremental_evidence.successful_unchanged_observed ? 1U : 0U);
    AppendU32(output, incremental_evidence.successful_updated_observed ? 1U : 0U);
    AppendU32(output, threshold_evidence.treatment_compiled ? 1U : 0U);
    AppendU32(output, threshold_evidence.capture_armed ? 1U : 0U);
    AppendU32(output, threshold_evidence.eligible_branch_entered ? 1U : 0U);
    AppendU32(output, threshold_evidence.rejected_lane_observed ? 1U : 0U);
    AppendU32(output, threshold_evidence.surviving_lane_observed ? 1U : 0U);
    Require(output.size() == kExecutionEvidenceBytes, "execution evidence has the wrong byte size");
    return output;
}

template <typename Index>
std::vector<std::uint8_t> CanonicalGraph(const Index &index, const infinity::HnswBulkBuildResult &build_result) {
    Require(index.GetVecNum() == kVectorCount, "canonical graph has the wrong vector count");
    Require(index.GetGraphMmax0() == 2 * kM && index.GetGraphMmax() == kM, "canonical graph has the wrong capacities");
    const auto [max_level_raw, entry_point_raw] = index.GetGraphEnterPoint();
    const std::int32_t max_level = static_cast<std::int32_t>(max_level_raw);
    const std::int32_t entry_point = static_cast<std::int32_t>(entry_point_raw);
    Require(max_level >= 0, "canonical graph has a negative maximum level");
    Require(entry_point >= 0 && static_cast<std::size_t>(entry_point) < kVectorCount, "canonical graph has an invalid entry point");

    std::vector<std::uint8_t> output;
    output.reserve(kGraphHeaderBytes + kVectorCount * 256);
    output.insert(output.end(), kGraphMagic.begin(), kGraphMagic.end());
    AppendU32(output, kGraphSchemaVersion);
    AppendU32(output, kGraphHeaderBytes);
    AppendU64(output, kVectorCount);
    AppendU64(output, kDimension);
    AppendU64(output, kM);
    AppendU64(output, kEfConstruction);
    AppendU64(output, kChunkSize);
    AppendU64(output, kMaxChunks);
    AppendU64(output, kWorkerCount);
    AppendU64(output, build_result.start_);
    AppendU64(output, build_result.end_);
    AppendU64(output, build_result.submitted_task_count_);
    AppendU64(output, index.GetVecNum());
    AppendU64(output, index.GetGraphMmax0());
    AppendU64(output, index.GetGraphMmax());
    AppendI32(output, max_level);
    AppendI32(output, entry_point);
    AppendU64(output, kVectorCount);
    AppendU64(output, 1);
    output.insert(output.end(), kDatasetDigest.begin(), kDatasetDigest.end());
    AppendU64(output, kGraphHeaderBytes);
    Require(output.size() == kGraphHeaderBytes, "canonical graph header has the wrong size");

    std::int32_t observed_max_level = -1;
    for (std::size_t ordinal = 0; ordinal < kVectorCount; ++ordinal) {
        const auto vertex = static_cast<infinity::VertexType>(ordinal);
        const std::int64_t label = static_cast<std::int64_t>(index.GetLabel(vertex));
        const std::int32_t level = static_cast<std::int32_t>(index.GetGraphLevel(vertex));
        Require(label == static_cast<std::int64_t>(ordinal), "canonical graph label does not match its ordinal");
        Require(level >= 0 && level <= max_level, "canonical graph vertex has an invalid level");
        observed_max_level = std::max(observed_max_level, level);

        AppendU32(output, static_cast<std::uint32_t>(ordinal));
        AppendI32(output, static_cast<std::int32_t>(label));
        AppendI32(output, level);
        AppendU32(output, static_cast<std::uint32_t>(level) + 1);

        for (std::int32_t layer = 0; layer <= level; ++layer) {
            const auto [neighbors, degree_raw] = index.GetGraphNeighbors(vertex, layer);
            const std::int64_t degree = static_cast<std::int64_t>(degree_raw);
            const std::size_t capacity = layer == 0 ? index.GetGraphMmax0() : index.GetGraphMmax();
            Require(degree >= 0 && static_cast<std::size_t>(degree) <= capacity, "canonical graph degree exceeds its layer capacity");
            Require(degree == 0 || neighbors != nullptr, "canonical graph has a null nonempty neighbor list");

            AppendU32(output, static_cast<std::uint32_t>(layer));
            AppendU32(output, static_cast<std::uint32_t>(degree));
            for (std::int64_t neighbor_index = 0; neighbor_index < degree; ++neighbor_index) {
                const std::int32_t neighbor = static_cast<std::int32_t>(neighbors[neighbor_index]);
                Require(neighbor >= 0 && static_cast<std::size_t>(neighbor) < kVectorCount, "canonical graph has an out-of-range neighbor");
                Require(neighbor != static_cast<std::int32_t>(ordinal), "canonical graph contains a self edge");
                for (std::int64_t previous = 0; previous < neighbor_index; ++previous) {
                    Require(neighbors[previous] != neighbors[neighbor_index], "canonical graph contains a duplicate neighbor");
                }
                Require(static_cast<std::int32_t>(index.GetGraphLevel(neighbor)) >= layer,
                        "canonical graph neighbor does not exist at the recorded layer");
                AppendI32(output, neighbor);
            }
        }
    }
    Require(observed_max_level == max_level, "canonical graph maximum level is inconsistent");
    Require(static_cast<std::int32_t>(index.GetGraphLevel(entry_point)) == max_level, "canonical graph entry point is not at the maximum level");
    return output;
}

void Fsync(int descriptor, std::string_view name) {
    for (;;) {
        if (fsync(descriptor) == 0) {
            return;
        }
        if (errno != EINTR) {
            ThrowSystemError("fsync " + std::string(name));
        }
    }
}

void WriteAllAt(int descriptor, std::span<const std::uint8_t> bytes, std::string_view name) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t request = std::min(bytes.size() - offset, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t count = pwrite(descriptor, bytes.data() + offset, request, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            ThrowSystemError("pwrite " + std::string(name));
        }
        Require(count > 0, std::string(name) + " made no write progress");
        offset += static_cast<std::size_t>(count);
    }
}

Digest WriteCanonicalGraph(int descriptor, const DescriptorSnapshot &initial, std::span<const std::uint8_t> graph_bytes) {
    const DescriptorSnapshot before = SnapshotDescriptor(descriptor, "graph descriptor");
    Require(SameStableMetadata(before, initial), "graph descriptor changed before writing");
    RequireOffset(descriptor, 0, "graph descriptor");
    WriteAllAt(descriptor, graph_bytes, "graph descriptor");
    Fsync(descriptor, "graph descriptor");
    const DescriptorSnapshot written = SnapshotDescriptor(descriptor, "graph descriptor");
    Require(SameObject(written, initial), "graph descriptor was replaced");
    Require(written.size == static_cast<off_t>(graph_bytes.size()), "graph descriptor has the wrong final size");
    const std::vector<std::uint8_t> persisted = ReadStableBytes(descriptor, graph_bytes.size(), written, "graph descriptor");
    Require(std::equal(persisted.begin(), persisted.end(), graph_bytes.begin(), graph_bytes.end()),
            "persisted canonical graph differs from the emitted bytes");
    Seek(descriptor, 0, "graph descriptor");
    return Hash(persisted);
}

Digest WriteExecutionEvidence(int descriptor, const DescriptorSnapshot &initial, std::span<const std::uint8_t> evidence_bytes) {
    const DescriptorSnapshot before = SnapshotDescriptor(descriptor, "execution evidence descriptor");
    Require(SameStableMetadata(before, initial), "execution evidence descriptor changed before writing");
    RequireOffset(descriptor, 0, "execution evidence descriptor");
    WriteAllAt(descriptor, evidence_bytes, "execution evidence descriptor");
    Fsync(descriptor, "execution evidence descriptor");
    const DescriptorSnapshot written = SnapshotDescriptor(descriptor, "execution evidence descriptor");
    Require(SameObject(written, initial), "execution evidence descriptor was replaced");
    Require(written.size == static_cast<off_t>(evidence_bytes.size()), "execution evidence descriptor has the wrong final size");
    const std::vector<std::uint8_t> persisted = ReadStableBytes(descriptor, evidence_bytes.size(), written, "execution evidence descriptor");
    Require(std::equal(persisted.begin(), persisted.end(), evidence_bytes.begin(), evidence_bytes.end()),
            "persisted execution evidence differs from the emitted bytes");
    Seek(descriptor, 0, "execution evidence descriptor");
    return Hash(persisted);
}

int DuplicateDescriptor(int descriptor, std::string_view name) {
    int duplicate;
    do {
        duplicate = fcntl(descriptor, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    } while (duplicate < 0 && errno == EINTR);
    if (duplicate < 0) {
        ThrowSystemError("fcntl duplicate " + std::string(name));
    }
    return duplicate;
}

std::unique_ptr<infinity::LocalFileHandle> MakeLocalFileHandle(int descriptor, const std::string &name, infinity::FileAccessMode mode) {
    try {
        return std::make_unique<infinity::LocalFileHandle>(descriptor, name, mode);
    } catch (...) {
        const int saved_error = errno;
        static_cast<void>(close(descriptor));
        errno = saved_error;
        throw;
    }
}

std::vector<std::uint8_t> SavePointerImage(const Hnsw &index, int descriptor, const DescriptorSnapshot &initial) {
    const DescriptorSnapshot before = SnapshotDescriptor(descriptor, "SaveToPtr descriptor");
    Require(SameStableMetadata(before, initial), "SaveToPtr descriptor changed before serialization");
    Seek(descriptor, 0, "SaveToPtr descriptor");

    const int duplicate = DuplicateDescriptor(descriptor, "SaveToPtr descriptor");
    auto file = MakeLocalFileHandle(duplicate, "schema12-production-exactness-save-to-ptr", infinity::FileAccessMode::kWrite);
    index.SaveToPtr(*file);
    Require(file->Sync().ok(), "LocalFileHandle failed to sync SaveToPtr output");
    file.reset();
    Fsync(descriptor, "SaveToPtr descriptor");

    const DescriptorSnapshot written = SnapshotDescriptor(descriptor, "SaveToPtr descriptor");
    Require(SameObject(written, initial), "SaveToPtr descriptor was replaced");
    Require(written.size > 0, "SaveToPtr produced an empty image");
    Require(static_cast<std::uintmax_t>(written.size) <= std::numeric_limits<std::size_t>::max(), "SaveToPtr image is too large");
    const auto size = static_cast<std::size_t>(written.size);
    std::vector<std::uint8_t> bytes = ReadStableBytes(descriptor, size, written, "SaveToPtr descriptor");
    return bytes;
}

void RevalidateRetainedOutput(int descriptor,
                              const DescriptorSnapshot &initial,
                              std::span<const std::uint8_t> expected_bytes,
                              const Digest &expected_digest,
                              std::string_view name) {
    const DescriptorSnapshot terminal = SnapshotDescriptor(descriptor, name);
    Require(SameObject(terminal, initial), std::string(name) + " was replaced");
    Require(terminal.mode == initial.mode, std::string(name) + " mode changed");
    Require(terminal.link_count == initial.link_count, std::string(name) + " link count changed");
    Require(terminal.size == static_cast<off_t>(expected_bytes.size()), std::string(name) + " has the wrong terminal size");

    const int flags = DescriptorFlags(descriptor, name);
    Require((flags & O_ACCMODE) == O_RDWR && (flags & O_APPEND) == 0, std::string(name) + " flags changed");
    RequireOffset(descriptor, 0, name);

    const std::vector<std::uint8_t> persisted = ReadStableBytes(descriptor, expected_bytes.size(), terminal, name);
    Require(std::equal(persisted.begin(), persisted.end(), expected_bytes.begin(), expected_bytes.end()),
            std::string(name) + " bytes changed before terminal validation");
    Require(Hash(persisted) == expected_digest, std::string(name) + " SHA-256 changed before terminal validation");
}

std::unique_ptr<Hnsw> LoadPointerImage(int descriptor, std::size_t size) {
    Seek(descriptor, 0, "SaveToPtr descriptor");
    const int duplicate = DuplicateDescriptor(descriptor, "SaveToPtr descriptor");
    auto file = MakeLocalFileHandle(duplicate, "schema12-production-exactness-save-to-ptr", infinity::FileAccessMode::kRead);
    auto loaded = Hnsw::LoadFromPtr(*file, size);
    Require(file->RemainingBytes() == 0, "LoadFromPtr did not consume the complete image");
    loaded->Check();
    file.reset();
    Seek(descriptor, 0, "SaveToPtr descriptor");
    return loaded;
}

std::string CanonicalJson(const infinity::HnswBulkBuildResult &build_result,
                          std::int32_t max_level,
                          std::int32_t entry_point,
                          std::size_t graph_bytes,
                          const Digest &graph_digest,
                          std::size_t pointer_bytes,
                          const Digest &pointer_digest) {
    std::string json;
    json.reserve(768);
    json += R"({"build":{"end":)";
    json += std::to_string(build_result.end_);
    json += R"(,"memory_delta_positive":true,"start":)";
    json += std::to_string(build_result.start_);
    json += R"(,"submitted_tasks":)";
    json += std::to_string(build_result.submitted_task_count_);
    json += R"(},"configuration":{"chunk_size":)";
    json += std::to_string(kChunkSize);
    json += R"(,"dimension":)";
    json += std::to_string(kDimension);
    json += R"(,"ef_construction":)";
    json += std::to_string(kEfConstruction);
    json += R"(,"m":)";
    json += std::to_string(kM);
    json += R"(,"max_chunks":)";
    json += std::to_string(kMaxChunks);
    json += R"(,"optimize":true,"vectors":)";
    json += std::to_string(kVectorCount);
    json += R"(,"workers":)";
    json += std::to_string(kWorkerCount);
    json += R"(},"dataset":{"bytes":)";
    json += std::to_string(kDatasetBytes);
    json += R"(,"sha256":")";
    json += kDatasetSha256;
    json += R"("},"graph":{"bytes":)";
    json += std::to_string(graph_bytes);
    json += R"(,"entry_point":)";
    json += std::to_string(entry_point);
    json += R"(,"header_bytes":)";
    json += std::to_string(kGraphHeaderBytes);
    json += R"(,"max_level":)";
    json += std::to_string(max_level);
    json += R"(,"mmax":)";
    json += std::to_string(kM);
    json += R"(,"mmax0":)";
    json += std::to_string(2 * kM);
    json += R"(,"sha256":")";
    json += Hex(graph_digest);
    json += R"("},"save_to_ptr":{"bytes":)";
    json += std::to_string(pointer_bytes);
    json += R"(,"roundtrip_graph_equal":true,"sha256":")";
    json += Hex(pointer_digest);
    json += R"("},"schema_version":)";
    json += std::to_string(kJsonSchemaVersion);
    json += R"(,"status":"PASS"})";
    return json;
}

int Run(const Options &options) {
#if !defined(__APPLE__) || !defined(__aarch64__)
    throw std::runtime_error("production exactness emitter requires native Apple arm64");
#endif
    Require(std::endian::native == std::endian::little, "production exactness emitter requires little endian");
    const DescriptorSet descriptors = ValidateDescriptors(options);
    const FrozenDataset dataset = ReadFrozenDataset(options.dataset_fd, descriptors.dataset);
    const Build build = BuildExactIndex(dataset);

    const std::vector<std::uint8_t> execution_evidence =
        CanonicalExecutionEvidence(build.incremental_execution_evidence, build.threshold_execution_evidence);
    const Digest execution_evidence_digest =
        WriteExecutionEvidence(options.execution_evidence_fd, descriptors.execution_evidence, execution_evidence);
    const std::vector<std::uint8_t> graph = CanonicalGraph(*build.index, build.result);
    const Digest graph_digest = WriteCanonicalGraph(options.graph_fd, descriptors.graph, graph);
    const std::vector<std::uint8_t> pointer_image = SavePointerImage(*build.index, options.save_to_ptr_fd, descriptors.save_to_ptr);
    const Digest pointer_digest = Hash(pointer_image);

    std::unique_ptr<Hnsw> loaded = LoadPointerImage(options.save_to_ptr_fd, pointer_image.size());
    const std::vector<std::uint8_t> roundtrip_graph = CanonicalGraph(*loaded, build.result);
    Require(roundtrip_graph == graph, "LoadFromPtr roundtrip changed the canonical graph");
    const std::vector<std::uint8_t> persisted_pointer = ReadStableBytes(options.save_to_ptr_fd,
                                                                        pointer_image.size(),
                                                                        SnapshotDescriptor(options.save_to_ptr_fd, "SaveToPtr descriptor"),
                                                                        "SaveToPtr descriptor");
    Require(persisted_pointer == pointer_image, "SaveToPtr image changed during roundtrip validation");
    Require(Hash(persisted_pointer) == pointer_digest, "SaveToPtr image SHA-256 changed during validation");
    VerifyFrozenDatasetUnchanged(options.dataset_fd, dataset);

    const auto [max_level, entry_point] = build.index->GetGraphEnterPoint();
    const std::string json = CanonicalJson(build.result,
                                           static_cast<std::int32_t>(max_level),
                                           static_cast<std::int32_t>(entry_point),
                                           graph.size(),
                                           graph_digest,
                                           pointer_image.size(),
                                           pointer_digest);
    RevalidateRetainedOutput(options.graph_fd, descriptors.graph, graph, graph_digest, "graph descriptor");
    RevalidateRetainedOutput(options.save_to_ptr_fd, descriptors.save_to_ptr, pointer_image, pointer_digest, "SaveToPtr descriptor");
    RevalidateRetainedOutput(options.execution_evidence_fd,
                             descriptors.execution_evidence,
                             execution_evidence,
                             execution_evidence_digest,
                             "execution evidence descriptor");
    std::cout << json << '\n';
    std::cout.flush();
    Require(std::cout.good(), "failed to write canonical JSON");
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        return Run(ParseOptions(argc, argv));
    } catch (const StandardStreamAliasError &) {
        return 1;
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << '\n';
        return kUsageExitCode;
    } catch (const std::exception &error) {
        std::cerr << "production HNSW exactness emission failed: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "production HNSW exactness emission failed with a non-standard exception\n";
        return 1;
    }
}
