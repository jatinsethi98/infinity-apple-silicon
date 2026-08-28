#include "hnsw_d0_runner.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr int kUsageExitCode = 64;
constexpr int kSidecarExitCode = 74;
constexpr std::uint32_t kAuditSidecarSchema = 4;
constexpr std::uint32_t kAuditSearchCount = 8;
constexpr std::uint64_t kFnv1a64Offset = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1a64Prime = 1099511628211ULL;
constexpr std::size_t kSha256Bytes = 32;

using Sha256Digest = std::array<std::uint8_t, kSha256Bytes>;

std::string HexBytes(const std::uint8_t *bytes, std::size_t size) {
    constexpr char kHexDigits[] = "0123456789abcdef";
    std::string result;
    result.reserve(size * 2);
    for (std::size_t index = 0; index < size; ++index) {
        result.push_back(kHexDigits[bytes[index] >> 4U]);
        result.push_back(kHexDigits[bytes[index] & 0x0fU]);
    }
    return result;
}

void PrintAttestationSnapshot(std::string_view phase, const HnswD0AttestationSnapshot &snapshot) {
    std::cout << "attestation_" << phase << "_dynamic_cdhash=" << snapshot.dynamic_cdhash << '\n';
    std::cout << "attestation_" << phase << "_held_executable_dev=" << snapshot.held_executable.device << '\n';
    std::cout << "attestation_" << phase << "_held_executable_ino=" << snapshot.held_executable.inode << '\n';
    std::cout << "attestation_" << phase << "_held_executable_size=" << snapshot.held_executable.size << '\n';
    std::cout << "attestation_" << phase << "_mapped_executable_dev=" << snapshot.mapped_executable.device << '\n';
    std::cout << "attestation_" << phase << "_mapped_executable_ino=" << snapshot.mapped_executable.inode << '\n';
    std::cout << "attestation_" << phase << "_mapped_executable_size=" << snapshot.mapped_executable.size << '\n';
    std::cout << "attestation_" << phase << "_mapped_vnode_matches_held_fd="
              << (HnswD0MappedExecutableMatchesHeldFile(snapshot) ? 1 : 0) << '\n';
    std::cout << "attestation_" << phase << "_image_count=" << snapshot.images.size() << '\n';
    for (std::size_t index = 0; index < snapshot.images.size(); ++index) {
        const HnswD0AttestationImage &image = snapshot.images[index];
        const auto *path_bytes = reinterpret_cast<const std::uint8_t *>(image.path.data());
        std::cout << "attestation_" << phase << "_image_" << index << "_path_hex=" << HexBytes(path_bytes, image.path.size()) << '\n';
        std::cout << "attestation_" << phase << "_image_" << index << "_uuid=" << image.uuid << '\n';
        std::cout << "attestation_" << phase << "_image_" << index << "_shared_cache=" << (image.shared_cache ? 1 : 0) << '\n';
    }
}

// Campaign mode is opt-in. Without it the runner never suspends itself, so the
// binary can be driven directly by a plain 11-positional-argument invocation with no
// external supervisor. See RunHnswD0 for the two accepted argument counts.
bool g_campaign_mode = false;

bool StopAtAttestationBarrier(std::string_view phase) {
    std::cout << "attestation_barrier=" << phase << '\n';
    std::cout.flush();
    std::cerr.flush();
    if (!g_campaign_mode) {
        return true;
    }
    if (raise(SIGSTOP) != 0) {
        std::cerr << "attestation barrier failed: " << std::strerror(errno) << '\n';
        return false;
    }
    return true;
}

int StopAtIndexBarrier(HnswD0IndexBarrierPhase phase, void *) {
    switch (phase) {
        case HnswD0IndexBarrierPhase::kBeforeIndex:
            return StopAtAttestationBarrier("before-index") ? 0 : 1;
        case HnswD0IndexBarrierPhase::kAfterIndex:
            return StopAtAttestationBarrier("after-index") ? 0 : 1;
    }
    std::cerr << "invalid index attestation barrier phase\n";
    return 1;
}

bool StopAtBeforeWorkBarrier(HnswD0LiveAttestation &attestation,
                             HnswD0AttestationSnapshot &before,
                             std::string &diagnostic) {
    if (!attestation.Capture(before, diagnostic)) {
        return false;
    }
    std::cout << "attestation_schema=3\n";
    PrintAttestationSnapshot("before", before);
    if (!StopAtAttestationBarrier("before-work")) {
        diagnostic = "before-work attestation barrier failed";
        return false;
    }
    return true;
}

bool StopAtAfterWorkBarrier(HnswD0LiveAttestation &attestation,
                            const HnswD0AttestationSnapshot &before,
                            HnswD0AttestationSnapshot &after,
                            HnswD0AttestationSummary &summary,
                            std::string &diagnostic) {
    if (!attestation.Capture(after, diagnostic)) {
        return false;
    }
    const bool passed = attestation.Validate(before, after, summary, diagnostic);
    PrintAttestationSnapshot("after", after);
    std::cout << "attestation_initial_image_adds=" << summary.initial_image_adds << '\n';
    std::cout << "attestation_later_image_adds=" << summary.later_image_adds << '\n';
    std::cout << "attestation_image_removes=" << summary.image_removes << '\n';
    std::cout << "attestation_image_set_unchanged=" << (summary.image_set_unchanged ? 1 : 0) << '\n';
    if (!StopAtAttestationBarrier("after-work")) {
        diagnostic = "after-work attestation barrier failed";
        return false;
    }
    std::cout << "attestation_status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

int RunAttestationOnly() {
    HnswD0LiveAttestation attestation;
    std::string diagnostic;
    if (!attestation.Begin(diagnostic)) {
        std::cerr << "attestation initialization failed: " << diagnostic << '\n';
        return kSidecarExitCode;
    }
    HnswD0AttestationSnapshot before;
    if (!StopAtBeforeWorkBarrier(attestation, before, diagnostic)) {
        std::cerr << "attestation failed: " << diagnostic << '\n';
        return kSidecarExitCode;
    }
    HnswD0AttestationSnapshot after;
    HnswD0AttestationSummary summary;
    if (!StopAtAfterWorkBarrier(attestation, before, after, summary, diagnostic)) {
        std::cerr << "attestation failed: " << diagnostic << '\n';
        return kSidecarExitCode;
    }
    return 0;
}

struct AuditBinding {
    Sha256Digest campaign_nonce{};
    std::uint64_t schedule_sequence = 0;
    std::uint32_t role_id = 0;
    Sha256Digest executable_sha256{};
    Sha256Digest dataset_sha256{};
};

class OwnedDataset {
public:
    OwnedDataset() = default;
    OwnedDataset(const OwnedDataset &) = delete;
    OwnedDataset &operator=(const OwnedDataset &) = delete;

    ~OwnedDataset() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    bool Open(const char *path, std::size_t expected_size) {
        int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        fd_ = open(path, flags);
        if (fd_ < 0) {
            ReportSystemError("open", path);
            return false;
        }

        struct stat file_status{};
        if (fstat(fd_, &file_status) != 0) {
            ReportSystemError("fstat", path);
            return false;
        }
        if (!S_ISREG(file_status.st_mode)) {
            std::cerr << "dataset is not a regular file: " << path << '\n';
            return false;
        }
        if (file_status.st_size < 0 || static_cast<std::uintmax_t>(file_status.st_size) != static_cast<std::uintmax_t>(expected_size)) {
            std::cerr << "dataset size mismatch: expected " << expected_size << " bytes, got " << file_status.st_size << '\n';
            return false;
        }

        if (expected_size % sizeof(float) != 0) {
            std::cerr << "dataset size is not float-aligned: " << expected_size << " bytes\n";
            return false;
        }
        data_.resize(expected_size / sizeof(float));
        auto *destination = reinterpret_cast<std::uint8_t *>(data_.data());
        std::size_t offset = 0;
        while (offset < expected_size) {
            const std::size_t remaining = expected_size - offset;
            const std::size_t request = std::min(remaining, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t count = pread(fd_, destination + offset, request, static_cast<off_t>(offset));
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count < 0) {
                ReportSystemError("pread", path);
                return false;
            }
            if (count == 0) {
                std::cerr << "dataset became truncated while reading: " << path << '\n';
                return false;
            }
            offset += static_cast<std::size_t>(count);
        }

        struct stat after_status{};
        if (fstat(fd_, &after_status) != 0) {
            ReportSystemError("fstat", path);
            return false;
        }
        struct stat path_status{};
        if (lstat(path, &path_status) != 0) {
            ReportSystemError("lstat", path);
            return false;
        }
        if (!SameSnapshot(file_status, after_status) || !S_ISREG(path_status.st_mode) || path_status.st_dev != after_status.st_dev ||
            path_status.st_ino != after_status.st_ino || path_status.st_size != after_status.st_size) {
            std::cerr << "dataset changed while being read: " << path << '\n';
            return false;
        }

        const int owned_fd = fd_;
        fd_ = -1;
        if (close(owned_fd) != 0) {
            ReportSystemError("close", path);
            return false;
        }
        return true;
    }

    const std::uint8_t *bytes() const { return reinterpret_cast<const std::uint8_t *>(data_.data()); }
    const float *floats() const { return data_.data(); }

private:
    static bool SameSnapshot(const struct stat &left, const struct stat &right) {
        if (left.st_dev != right.st_dev || left.st_ino != right.st_ino || left.st_size != right.st_size) {
            return false;
        }
#if defined(__APPLE__)
        return left.st_mtimespec.tv_sec == right.st_mtimespec.tv_sec && left.st_mtimespec.tv_nsec == right.st_mtimespec.tv_nsec &&
               left.st_ctimespec.tv_sec == right.st_ctimespec.tv_sec && left.st_ctimespec.tv_nsec == right.st_ctimespec.tv_nsec;
#else
        return left.st_mtim.tv_sec == right.st_mtim.tv_sec && left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
               left.st_ctim.tv_sec == right.st_ctim.tv_sec && left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
#endif
    }

    static void ReportSystemError(std::string_view operation, const char *path) {
        const int error = errno;
        std::cerr << operation << " failed for " << path << ": " << std::strerror(error) << '\n';
    }

    int fd_ = -1;
    std::vector<float> data_;
};

template <typename Integer>
bool ParseInteger(const char *text, Integer &value) {
    const std::string_view input(text);
    if (input.empty()) {
        return false;
    }
    Integer parsed{};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), parsed, 10);
    if (error != std::errc{} || end != input.data() + input.size()) {
        return false;
    }
    value = parsed;
    return true;
}

int HexDigit(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

bool ParseSha256Hex(const char *text, Sha256Digest &digest) {
    const std::string_view input(text);
    if (input.size() != kSha256Bytes * 2) {
        return false;
    }
    for (std::size_t index = 0; index < digest.size(); ++index) {
        const int upper = HexDigit(input[index * 2]);
        const int lower = HexDigit(input[index * 2 + 1]);
        if (upper < 0 || lower < 0) {
            return false;
        }
        digest[index] = static_cast<std::uint8_t>((upper << 4) | lower);
    }
    return true;
}

std::string Sha256Hex(const Sha256Digest &digest) {
    std::string result;
    result.reserve(digest.size() * 2);
    constexpr char kHexDigits[] = "0123456789abcdef";
    for (const std::uint8_t byte : digest) {
        result.push_back(kHexDigits[byte >> 4U]);
        result.push_back(kHexDigits[byte & 0x0fU]);
    }
    return result;
}

bool IsPowerOfTwo(unsigned long long value) { return value != 0 && (value & (value - 1)) == 0; }

bool ComputeDataBytes(const HnswDevConfig &config, std::size_t &data_bytes) {
    if (config.vector_count > std::numeric_limits<std::size_t>::max() || config.dimension > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    const auto vectors = static_cast<std::size_t>(config.vector_count);
    const auto dimensions = static_cast<std::size_t>(config.dimension);
    if (dimensions != 0 && vectors > std::numeric_limits<std::size_t>::max() / dimensions) {
        return false;
    }
    const std::size_t element_count = vectors * dimensions;
    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    data_bytes = element_count * sizeof(float);
    return data_bytes <= static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max());
}

bool ValidateConfig(const HnswDevConfig &config, std::size_t &data_bytes) {
    if (config.vector_count == 0 || config.dimension == 0 || config.chunk_size == 0 || config.query_count == 0 ||
        config.query_count > config.vector_count || config.m < 2 || config.ef_construction < config.m || config.ef_search < 1 ||
        config.participant_count < 1 || !IsPowerOfTwo(config.chunk_size)) {
        return false;
    }
    if (config.vector_count > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
        config.dimension > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
        config.build_grain > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    return ComputeDataBytes(config, data_bytes);
}

std::uint64_t Fnv1a64(const std::uint8_t *data, std::size_t size) {
    std::uint64_t hash = kFnv1a64Offset;
    for (std::size_t offset = 0; offset < size; ++offset) {
        hash ^= data[offset];
        hash *= kFnv1a64Prime;
    }
    return hash;
}

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU,
    0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU,
    0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U,
    0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

std::uint32_t RotateRight(std::uint32_t value, unsigned int count) { return (value >> count) | (value << (32U - count)); }

void Sha256Compress(const std::uint8_t *block, std::array<std::uint32_t, 8> &state) {
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

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
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
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

class Sha256Hasher {
public:
    void Update(const std::uint8_t *data, std::size_t size) {
        constexpr std::uint64_t kMaximumBytes = std::numeric_limits<std::uint64_t>::max() / 8U;
        if (size > kMaximumBytes - byte_count_) {
            throw std::length_error("SHA-256 input is too large");
        }
        byte_count_ += static_cast<std::uint64_t>(size);

        if (pending_size_ != 0) {
            const std::size_t copied = std::min(size, pending_.size() - pending_size_);
            std::memcpy(pending_.data() + pending_size_, data, copied);
            pending_size_ += copied;
            data += copied;
            size -= copied;
            if (pending_size_ == pending_.size()) {
                Sha256Compress(pending_.data(), state_);
                pending_size_ = 0;
            }
        }
        while (size >= pending_.size()) {
            Sha256Compress(data, state_);
            data += pending_.size();
            size -= pending_.size();
        }
        if (size != 0) {
            std::memcpy(pending_.data(), data, size);
            pending_size_ = size;
        }
    }

    Sha256Digest Finalize() const {
        std::array<std::uint32_t, 8> state = state_;
        std::array<std::uint8_t, 128> tail{};
        std::memcpy(tail.data(), pending_.data(), pending_size_);
        tail[pending_size_] = 0x80;
        const std::size_t padded_size = pending_size_ < 56 ? 64 : 128;
        const std::uint64_t bit_length = byte_count_ * 8U;
        for (std::size_t index = 0; index < 8; ++index) {
            tail[padded_size - 1 - index] = static_cast<std::uint8_t>(bit_length >> (index * 8U));
        }
        Sha256Compress(tail.data(), state);
        if (padded_size == 128) {
            Sha256Compress(tail.data() + 64, state);
        }

        Sha256Digest digest{};
        for (std::size_t word = 0; word < state.size(); ++word) {
            for (std::size_t byte = 0; byte < 4; ++byte) {
                digest[word * 4 + byte] = static_cast<std::uint8_t>(state[word] >> ((3U - byte) * 8U));
            }
        }
        return digest;
    }

private:
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
    std::array<std::uint8_t, 64> pending_{};
    std::size_t pending_size_ = 0;
    std::uint64_t byte_count_ = 0;
};

Sha256Digest ComputeSha256(const std::uint8_t *data, std::size_t size) {
    Sha256Hasher hasher;
    hasher.Update(data, size);
    return hasher.Finalize();
}

std::string Sha256Hex(const std::uint8_t *data, std::size_t size) { return Sha256Hex(ComputeSha256(data, size)); }

bool SameFileSnapshot(const struct stat &left, const struct stat &right) {
    if (left.st_dev != right.st_dev || left.st_ino != right.st_ino || left.st_size != right.st_size) {
        return false;
    }
#if defined(__APPLE__)
    return left.st_mtimespec.tv_sec == right.st_mtimespec.tv_sec && left.st_mtimespec.tv_nsec == right.st_mtimespec.tv_nsec &&
           left.st_ctimespec.tv_sec == right.st_ctimespec.tv_sec && left.st_ctimespec.tv_nsec == right.st_ctimespec.tv_nsec;
#else
    return left.st_mtim.tv_sec == right.st_mtim.tv_sec && left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
           left.st_ctim.tv_sec == right.st_ctim.tv_sec && left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
#endif
}

bool HashHeldExecutable(int descriptor, Sha256Digest &digest, std::string &diagnostic) {
    if (descriptor < 0) {
        diagnostic = "held executable descriptor is invalid";
        return false;
    }

    struct stat before_status{};
    if (fstat(descriptor, &before_status) != 0) {
        const int error = errno;
        diagnostic = "fstat failed for held executable: " + std::string(std::strerror(error));
        return false;
    }
    if (!S_ISREG(before_status.st_mode) || before_status.st_size < 0) {
        diagnostic = "held executable is not a regular file";
        return false;
    }

    Sha256Hasher hasher;
    std::array<std::uint8_t, 64 * 1024> buffer{};
    off_t offset = 0;
    while (offset < before_status.st_size) {
        const auto remaining = static_cast<std::uintmax_t>(before_status.st_size - offset);
        const std::size_t request = static_cast<std::size_t>(std::min<std::uintmax_t>(remaining, buffer.size()));
        const ssize_t count = pread(descriptor, buffer.data(), request, offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            const int error = errno;
            diagnostic = "pread failed for held executable: " + std::string(std::strerror(error));
            return false;
        }
        if (count == 0) {
            diagnostic = "held executable became truncated while hashing";
            return false;
        }
        hasher.Update(buffer.data(), static_cast<std::size_t>(count));
        offset += count;
    }

    struct stat after_status{};
    if (fstat(descriptor, &after_status) != 0) {
        const int error = errno;
        diagnostic = "fstat failed after hashing held executable: " + std::string(std::strerror(error));
        return false;
    }
    if (!SameFileSnapshot(before_status, after_status)) {
        diagnostic = "held executable changed while being hashed";
        return false;
    }
    digest = hasher.Finalize();
    return true;
}

class LittleEndianBuffer {
public:
    void Append(const void *data, std::size_t size) {
        const auto *begin = static_cast<const std::uint8_t *>(data);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

    void AppendU32(std::uint32_t value) {
        std::array<std::uint8_t, 4> bytes{};
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
        }
        Append(bytes.data(), bytes.size());
    }

    void AppendI32(std::int32_t value) { AppendU32(static_cast<std::uint32_t>(value)); }

    void AppendU64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
        }
        Append(bytes.data(), bytes.size());
    }

    void AppendI64(std::int64_t value) { AppendU64(static_cast<std::uint64_t>(value)); }

    std::size_t Size() const { return bytes_.size(); }

    std::vector<std::uint8_t> TakeBytes() { return std::move(bytes_); }

private:
    std::vector<std::uint8_t> bytes_;
};

struct AuditSidecarEvidence {
    std::uint64_t bytes = 0;
    std::string sha256;
};

void RequireSidecar(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::uint32_t SidecarEngine(HnswD0Engine engine) {
    switch (engine) {
        case HnswD0Engine::kInfinity:
            return 1;
        case HnswD0Engine::kFaiss:
            return 2;
    }
    throw std::runtime_error("audit sidecar engine is invalid");
}

// Role id used when no campaign binding is supplied. Must satisfy RoleMatchesEngine.
std::uint32_t DefaultRoleForEngine(HnswD0Engine engine) { return engine == HnswD0Engine::kFaiss ? 4u : 1u; }

bool RoleMatchesEngine(std::uint32_t role_id, HnswD0Engine engine) {
    switch (role_id) {
        case 1:
        case 2:
        case 3:
            return engine == HnswD0Engine::kInfinity;
        case 4:
            return engine == HnswD0Engine::kFaiss;
        default:
            return false;
    }
}

std::uint32_t PackExecutionWitness(const HnswD0ExecutionWitness &witness) {
    const std::array fields{
        witness.incremental_treatment_compiled,
        witness.incremental_capture_armed,
        witness.incremental_eligible_branch_entered,
        witness.incremental_successful_unchanged_observed,
        witness.incremental_successful_updated_observed,
        witness.threshold_treatment_compiled,
        witness.threshold_capture_armed,
        witness.threshold_eligible_branch_entered,
        witness.threshold_rejected_lane_observed,
        witness.threshold_surviving_lane_observed,
    };
    std::uint32_t bits = 0;
    for (std::size_t index = 0; index < fields.size(); ++index) {
        RequireSidecar(fields[index] <= 1, "audit sidecar execution witness is not binary");
        bits |= fields[index] << index;
    }
    return bits;
}

void ValidateAuditSidecar(const HnswDevConfig &config, const HnswDevResult &result) {
    static_assert(kHnswD0RecallK.size() == kAuditSearchCount);
    static_assert(kHnswD0RecallEf.size() == kAuditSearchCount);
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);

    RequireSidecar(result.query_benchmark.valid, "audit sidecar query benchmark is invalid");
    RequireSidecar(result.recall_audit.valid, "audit sidecar recall audit is invalid");
    RequireSidecar(config.vector_count <= std::numeric_limits<std::size_t>::max(), "audit sidecar vector count exceeds size_t");
    const std::size_t vertex_count = static_cast<std::size_t>(config.vector_count);
    const HnswD0GraphAudit &graph = result.graph_audit;
    RequireSidecar(graph.vertex_count == vertex_count, "audit sidecar graph vertex count mismatch");
    RequireSidecar(graph.vertices.size() == vertex_count, "audit sidecar retained vertex count mismatch");
    RequireSidecar(graph.max_level >= 0, "audit sidecar graph maximum level is invalid");
    RequireSidecar(graph.entry_point >= 0 && static_cast<std::size_t>(graph.entry_point) < vertex_count,
                   "audit sidecar graph entry point is out of range");
    RequireSidecar(graph.level0_capacity > 0 && graph.level0_capacity <= std::numeric_limits<std::uint32_t>::max(),
                   "audit sidecar level-zero capacity is out of range");
    RequireSidecar(graph.upper_capacity > 0 && graph.upper_capacity <= std::numeric_limits<std::uint32_t>::max(),
                   "audit sidecar upper capacity is out of range");
    RequireSidecar(config.m >= 0 && static_cast<unsigned long long>(config.m) <= std::numeric_limits<std::uint32_t>::max(),
                   "audit sidecar M is out of range");
    RequireSidecar(config.ef_construction >= 0 &&
                       static_cast<unsigned long long>(config.ef_construction) <= std::numeric_limits<std::uint32_t>::max(),
                   "audit sidecar ef_construction is out of range");
    RequireSidecar(result.query_benchmark.latency_samples_ns.size() == kHnswD0QueryLatencySampleCount,
                   "audit sidecar query latency sample count mismatch");
    RequireSidecar(result.query_benchmark.latency_validated_operations == kHnswD0QueryLatencySampleCount,
                   "audit sidecar query latency validated operation count mismatch");
    RequireSidecar(result.query_benchmark.throughput_operations == kHnswD0QueryThroughputOperations,
                   "audit sidecar query throughput operation count mismatch");
    RequireSidecar(result.query_benchmark.throughput_validated_operations == kHnswD0QueryThroughputOperations,
                   "audit sidecar query throughput validated operation count mismatch");
    RequireSidecar(result.query_benchmark.throughput_wall_ns > 0, "audit sidecar query throughput duration is not positive");
    RequireSidecar(result.query_benchmark.latency_validated_result_checksum == result.query_benchmark.result_checksum &&
                       result.query_benchmark.throughput_validated_result_checksum == result.query_benchmark.result_checksum,
                   "audit sidecar query phase checksums differ");
    Sha256Digest timed_query_corpus_sha256{};
    RequireSidecar(ParseSha256Hex(result.query_benchmark.timed_query_corpus_sha256.c_str(), timed_query_corpus_sha256) &&
                       Sha256Hex(timed_query_corpus_sha256) == result.query_benchmark.timed_query_corpus_sha256,
                   "audit sidecar timed query corpus SHA-256 is not canonical lowercase hex");
    RequireSidecar(result.query_benchmark.timed_query_corpus_sha256 == result.recall_audit.queries_sha256,
                   "audit sidecar timed query corpus differs from recall replay");
    RequireSidecar(HnswD0QueryBenchmarkMatchesRecall(result.query_benchmark, result.recall_audit, vertex_count),
                   "audit sidecar query results differ from the recall audit");
    static_cast<void>(PackExecutionWitness(result.execution_witness));

    std::vector<bool> labels(vertex_count, false);
    std::int32_t observed_max_level = -1;
    for (std::size_t ordinal = 0; ordinal < vertex_count; ++ordinal) {
        const HnswD0GraphVertexAudit &vertex = graph.vertices[ordinal];
        RequireSidecar(vertex.level >= 0 && vertex.level <= graph.max_level, "audit sidecar retained vertex level is out of range");
        RequireSidecar(vertex.layers.size() == static_cast<std::size_t>(vertex.level) + 1, "audit sidecar retained layer count mismatch");
        RequireSidecar(vertex.label >= 0 && static_cast<std::size_t>(vertex.label) < vertex_count, "audit sidecar retained label is out of range");
        RequireSidecar(!labels[static_cast<std::size_t>(vertex.label)], "audit sidecar retained duplicate label");
        labels[static_cast<std::size_t>(vertex.label)] = true;
        observed_max_level = std::max(observed_max_level, vertex.level);
    }
    RequireSidecar(observed_max_level == graph.max_level, "audit sidecar retained maximum level mismatch");
    RequireSidecar(graph.vertices[static_cast<std::size_t>(graph.entry_point)].level == graph.max_level,
                   "audit sidecar entry point is not on the maximum level");

    std::size_t directed_edges = 0;
    std::size_t level0_directed_edges = 0;
    for (std::size_t ordinal = 0; ordinal < vertex_count; ++ordinal) {
        const HnswD0GraphVertexAudit &vertex = graph.vertices[ordinal];
        for (std::int32_t layer = 0; layer <= vertex.level; ++layer) {
            const HnswD0GraphLayerAudit &retained_layer = vertex.layers[static_cast<std::size_t>(layer)];
            const std::size_t expected_capacity = layer == 0 ? graph.level0_capacity : graph.upper_capacity;
            RequireSidecar(retained_layer.capacity == expected_capacity, "audit sidecar retained layer capacity mismatch");
            RequireSidecar(retained_layer.neighbors.size() <= retained_layer.capacity, "audit sidecar retained degree exceeds capacity");
            RequireSidecar(retained_layer.neighbors.size() <= std::numeric_limits<std::uint32_t>::max(),
                           "audit sidecar retained degree is out of range");
            RequireSidecar(directed_edges <= std::numeric_limits<std::size_t>::max() - retained_layer.neighbors.size(),
                           "audit sidecar directed edge count overflow");
            directed_edges += retained_layer.neighbors.size();
            if (layer == 0) {
                RequireSidecar(level0_directed_edges <= std::numeric_limits<std::size_t>::max() - retained_layer.neighbors.size(),
                               "audit sidecar level-zero edge count overflow");
                level0_directed_edges += retained_layer.neighbors.size();
            }
            for (std::size_t rank = 0; rank < retained_layer.neighbors.size(); ++rank) {
                const std::int32_t neighbor = retained_layer.neighbors[rank];
                RequireSidecar(neighbor >= 0 && static_cast<std::size_t>(neighbor) < vertex_count, "audit sidecar retained neighbor is out of range");
                RequireSidecar(static_cast<std::size_t>(neighbor) != ordinal, "audit sidecar retained self edge");
                RequireSidecar(graph.vertices[static_cast<std::size_t>(neighbor)].level >= layer,
                               "audit sidecar retained upper-layer edge targets a lower-level vertex");
                RequireSidecar(std::find(retained_layer.neighbors.begin(),
                                         retained_layer.neighbors.begin() + static_cast<std::ptrdiff_t>(rank),
                                         neighbor) == retained_layer.neighbors.begin() + static_cast<std::ptrdiff_t>(rank),
                               "audit sidecar retained duplicate edge");
            }
        }
    }
    RequireSidecar(directed_edges == graph.directed_edges, "audit sidecar retained directed edge count mismatch");
    RequireSidecar(level0_directed_edges == graph.level0_directed_edges, "audit sidecar retained level-zero edge count mismatch");

    std::vector<bool> visited(vertex_count, false);
    std::vector<std::int32_t> pending;
    pending.reserve(vertex_count);
    pending.push_back(graph.entry_point);
    visited[static_cast<std::size_t>(graph.entry_point)] = true;
    for (std::size_t cursor = 0; cursor < pending.size(); ++cursor) {
        for (const std::int32_t neighbor : graph.vertices[static_cast<std::size_t>(pending[cursor])].layers[0].neighbors) {
            if (!visited[static_cast<std::size_t>(neighbor)]) {
                visited[static_cast<std::size_t>(neighbor)] = true;
                pending.push_back(neighbor);
            }
        }
    }
    RequireSidecar(pending.size() == graph.reachable_count, "audit sidecar retained level-zero reachability count is inconsistent");
    RequireSidecar(graph.valid == (pending.size() == vertex_count), "audit sidecar graph-valid flag disagrees with level-zero reachability");

    const HnswD0RecallAudit &recall = result.recall_audit;
    RequireSidecar(recall.query_count == kHnswD0HeldOutQueryCount, "audit sidecar retained query count mismatch");
    RequireSidecar(recall.query_seed == kHnswD0HeldOutQuerySeed, "audit sidecar retained query seed mismatch");
    for (std::size_t point = 0; point < kAuditSearchCount; ++point) {
        const std::size_t k = kHnswD0RecallK[point];
        RequireSidecar(k <= std::numeric_limits<std::uint32_t>::max() && kHnswD0RecallEf[point] <= std::numeric_limits<std::uint32_t>::max(),
                       "audit sidecar search parameter is out of range");
        RequireSidecar(recall.query_count <= std::numeric_limits<std::size_t>::max() / k, "audit sidecar result count overflow");
        const std::size_t expected_count = recall.query_count * k;
        const auto &returned = recall.returned_results[point];
        RequireSidecar(returned.size() == expected_count, "audit sidecar retained search result count mismatch");
        for (std::size_t query = 0; query < recall.query_count; ++query) {
            std::vector<bool> seen(vertex_count, false);
            float previous_distance = -std::numeric_limits<float>::infinity();
            for (std::size_t rank = 0; rank < k; ++rank) {
                const HnswD0ReturnedResult &item = returned[query * k + rank];
                RequireSidecar(std::isfinite(item.distance) && item.distance >= 0.0F, "audit sidecar retained distance is invalid");
                RequireSidecar(item.distance >= previous_distance, "audit sidecar retained search results are not sorted");
                RequireSidecar(item.label >= 0 && static_cast<std::size_t>(item.label) < vertex_count,
                               "audit sidecar retained result label is out of range");
                RequireSidecar(!seen[static_cast<std::size_t>(item.label)], "audit sidecar retained duplicate result label");
                previous_distance = item.distance;
                seen[static_cast<std::size_t>(item.label)] = true;
            }
        }
    }
}

std::vector<std::uint8_t>
SerializeAuditSidecar(HnswD0Engine engine, const AuditBinding &binding, const HnswDevConfig &config, const HnswDevResult &result) {
    ValidateAuditSidecar(config, result);
    RequireSidecar(RoleMatchesEngine(binding.role_id, engine), "audit sidecar role does not match engine");

    LittleEndianBuffer output;
    constexpr std::array<char, 8> kMagic{'I', 'F', 'D', '0', 'A', 'U', 'D', '4'};
    output.Append(kMagic.data(), kMagic.size());
    output.AppendU32(kAuditSidecarSchema);
    output.AppendU32(SidecarEngine(engine));
    RequireSidecar(output.Size() == 16, "audit sidecar engine prefix offset is invalid");
    output.Append(binding.campaign_nonce.data(), binding.campaign_nonce.size());
    RequireSidecar(output.Size() == 48, "audit sidecar campaign nonce offset is invalid");
    output.AppendU64(binding.schedule_sequence);
    output.AppendU32(binding.role_id);
    output.AppendU32(PackExecutionWitness(result.execution_witness));
    RequireSidecar(output.Size() == 64, "audit sidecar schedule and role offset is invalid");
    output.Append(binding.executable_sha256.data(), binding.executable_sha256.size());
    RequireSidecar(output.Size() == 96, "audit sidecar executable hash offset is invalid");
    output.Append(binding.dataset_sha256.data(), binding.dataset_sha256.size());
    RequireSidecar(output.Size() == 128, "audit sidecar binding prefix size is invalid");

    output.AppendU64(static_cast<std::uint64_t>(config.vector_count));
    output.AppendU64(static_cast<std::uint64_t>(config.dimension));
    output.AppendU32(static_cast<std::uint32_t>(config.m));
    output.AppendU32(static_cast<std::uint32_t>(config.ef_construction));
    output.AppendI32(result.graph_audit.max_level);
    output.AppendI32(result.graph_audit.entry_point);
    output.AppendU32(static_cast<std::uint32_t>(result.graph_audit.level0_capacity));
    output.AppendU32(static_cast<std::uint32_t>(result.graph_audit.upper_capacity));
    output.AppendU64(static_cast<std::uint64_t>(result.recall_audit.query_count));
    output.AppendU64(result.recall_audit.query_seed);
    output.AppendU32(kAuditSearchCount);
    output.AppendU32(0);
    output.AppendU32(kHnswD0QueryBenchmarkSchema);
    output.AppendU32(static_cast<std::uint32_t>(kHnswD0HeldOutQueryCount));
    output.AppendU32(static_cast<std::uint32_t>(kHnswD0QueryK));
    output.AppendU32(static_cast<std::uint32_t>(kHnswD0QueryEf));
    output.AppendU32(static_cast<std::uint32_t>(kHnswD0QueryWarmupPasses));
    output.AppendU32(static_cast<std::uint32_t>(kHnswD0QueryMeasuredPasses));
    output.AppendU32(static_cast<std::uint32_t>(kHnswD0QueryLatencyConcurrency));
    output.AppendU32(static_cast<std::uint32_t>(kHnswD0QueryThroughputConcurrency));
    output.AppendU32(kHnswD0NearestRankMethod);
    output.AppendU32(kHnswD0ProcessLocalTopKTransaction);
    output.AppendU64(std::bit_cast<std::uint64_t>(kHnswD0QueryRecallFloor));
    output.AppendU64(std::bit_cast<std::uint64_t>(kHnswD0QueryMaximumAbsoluteRecallGap));
    Sha256Digest timed_query_corpus_sha256{};
    RequireSidecar(ParseSha256Hex(result.query_benchmark.timed_query_corpus_sha256.c_str(), timed_query_corpus_sha256),
                   "audit sidecar timed query corpus SHA-256 is invalid");
    output.Append(timed_query_corpus_sha256.data(), timed_query_corpus_sha256.size());
    output.AppendU64(static_cast<std::uint64_t>(result.query_benchmark.latency_samples_ns.size()));
    output.AppendU64(result.query_benchmark.latency_validated_operations);
    output.AppendU64(result.query_benchmark.latency_validated_result_checksum);
    output.AppendU64(result.query_benchmark.throughput_operations);
    output.AppendU64(result.query_benchmark.throughput_validated_operations);
    output.AppendU64(result.query_benchmark.throughput_wall_ns);
    output.AppendU64(result.query_benchmark.throughput_validated_result_checksum);
    output.AppendU64(result.query_benchmark.result_checksum);
    output.AppendU64(static_cast<std::uint64_t>(result.query_benchmark.per_query_checksums.size()));
    for (const std::uint64_t checksum : result.query_benchmark.per_query_checksums) {
        output.AppendU64(checksum);
    }
    for (const std::uint64_t sample : result.query_benchmark.latency_samples_ns) {
        output.AppendU64(sample);
    }

    for (const HnswD0GraphVertexAudit &vertex : result.graph_audit.vertices) {
        output.AppendI32(vertex.level);
        output.AppendI64(vertex.label);
        for (const HnswD0GraphLayerAudit &layer : vertex.layers) {
            output.AppendU32(static_cast<std::uint32_t>(layer.capacity));
            output.AppendU32(static_cast<std::uint32_t>(layer.neighbors.size()));
            for (const std::int32_t neighbor : layer.neighbors) {
                output.AppendI32(neighbor);
            }
        }
    }

    for (std::size_t point = 0; point < kAuditSearchCount; ++point) {
        const auto &returned = result.recall_audit.returned_results[point];
        output.AppendU32(static_cast<std::uint32_t>(kHnswD0RecallK[point]));
        output.AppendU32(static_cast<std::uint32_t>(kHnswD0RecallEf[point]));
        output.AppendU64(static_cast<std::uint64_t>(returned.size()));
        for (const HnswD0ReturnedResult &item : returned) {
            output.AppendI64(item.label);
            output.AppendU32(std::bit_cast<std::uint32_t>(item.distance));
        }
    }
    return output.TakeBytes();
}

bool CloseSidecar(int descriptor, int &error) {
    if (close(descriptor) != 0) {
        error = errno;
        return false;
    }
    return true;
}

bool WriteAuditSidecarExclusive(const char *path, const std::vector<std::uint8_t> &bytes, std::string &diagnostic) {
    if (path == nullptr || path[0] == '\0') {
        diagnostic = "audit sidecar path is empty";
        return false;
    }

    int descriptor = -1;
    do {
        descriptor = open(path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, S_IRUSR | S_IWUSR);
    } while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0) {
        const int error = errno;
        diagnostic = "open failed for " + std::string(path) + ": " + std::strerror(error);
        return false;
    }

    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t request = std::min(bytes.size() - offset, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t written = write(descriptor, bytes.data() + offset, request);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            const int write_error = errno;
            int close_error = 0;
            const bool closed = CloseSidecar(descriptor, close_error);
            diagnostic = "write failed for " + std::string(path) + ": " + std::strerror(write_error);
            if (!closed) {
                diagnostic += "; close also failed: ";
                diagnostic += std::strerror(close_error);
            }
            return false;
        }
        if (written == 0) {
            int close_error = 0;
            const bool closed = CloseSidecar(descriptor, close_error);
            diagnostic = "write made no progress for " + std::string(path);
            if (!closed) {
                diagnostic += "; close also failed: ";
                diagnostic += std::strerror(close_error);
            }
            return false;
        }
        offset += static_cast<std::size_t>(written);
    }

    int close_error = 0;
    if (!CloseSidecar(descriptor, close_error)) {
        diagnostic = "close failed for " + std::string(path) + ": " + std::strerror(close_error);
        return false;
    }
    return true;
}

bool CreateAuditSidecar(const char *path,
                        HnswD0Engine engine,
                        const AuditBinding &binding,
                        const HnswDevConfig &config,
                        const HnswDevResult &result,
                        AuditSidecarEvidence &evidence,
                        std::string &diagnostic) {
    try {
        const std::vector<std::uint8_t> bytes = SerializeAuditSidecar(engine, binding, config, result);
        RequireSidecar(bytes.size() <= std::numeric_limits<std::uint64_t>::max(), "audit sidecar byte count is out of range");
        evidence.bytes = static_cast<std::uint64_t>(bytes.size());
        evidence.sha256 = Sha256Hex(bytes.data(), bytes.size());
        if (!WriteAuditSidecarExclusive(path, bytes, diagnostic)) {
            evidence = AuditSidecarEvidence{};
            return false;
        }
        return true;
    } catch (const std::exception &error) {
        diagnostic = error.what();
        evidence = AuditSidecarEvidence{};
        return false;
    } catch (...) {
        diagnostic = "unknown audit sidecar failure";
        evidence = AuditSidecarEvidence{};
        return false;
    }
}

void Prefault(const std::uint8_t *data, std::size_t size) {
    const long configured_page_size = sysconf(_SC_PAGESIZE);
    const std::size_t page_size = configured_page_size > 0 ? static_cast<std::size_t>(configured_page_size) : std::size_t{4096};
    const volatile std::uint8_t *pages = data;
    std::uint8_t checksum = 0;
    for (std::size_t offset = 0; offset < size; offset += page_size) {
        checksum = static_cast<std::uint8_t>(checksum ^ pages[offset]);
    }
    checksum = static_cast<std::uint8_t>(checksum ^ pages[size - 1]);
    (void)checksum;
}

void PrintUsage(const char *binary) {
    std::cerr << "usage: " << binary
              << " DATASET_PATH VECTORS DIMENSIONS M EF_CONSTRUCTION EF_SEARCH CHUNK_SIZE "
                 "QUERY_COUNT PARTICIPANTS BUILD_GRAIN AUDIT_SIDECAR_PATH\n"
              << "       (optional campaign mode appends: CAMPAIGN_NONCE_HEX SCHEDULE_SEQUENCE "
                 "ROLE_ID EXPECTED_EXECUTABLE_SHA256 EXPECTED_DATASET_SHA256; it verifies the "
                 "executable and dataset digests and suspends at each attestation barrier, so it "
                 "requires an external supervisor to send SIGCONT)\n";
}

void PrintResult(HnswD0Engine engine,
                 const AuditBinding &binding,
                 std::uint64_t data_hash,
                 const std::string &data_sha256,
                 std::size_t data_bytes,
                 const HnswDevConfig &config,
                 const HnswDevResult &result,
                 const AuditSidecarEvidence &sidecar,
                 bool passed) {
    const std::string_view engine_name = engine == HnswD0Engine::kInfinity ? "infinity" : "faiss";
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "status=" << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "scope=d0-development-only\n";
    std::cout << "order=" << engine_name << "-only\n";
    std::cout << "campaign_nonce=" << Sha256Hex(binding.campaign_nonce) << '\n';
    std::cout << "schedule_sequence=" << binding.schedule_sequence << '\n';
    std::cout << "role_id=" << binding.role_id << '\n';
    std::cout << "binary_sha256=" << Sha256Hex(binding.executable_sha256) << '\n';
    std::cout << "dataset_sha256=" << Sha256Hex(binding.dataset_sha256) << '\n';
    std::cout << "data_fnv1a64=" << data_hash << '\n';
    std::cout << "data_sha256=" << data_sha256 << '\n';
    std::cout << "data_bytes=" << data_bytes << '\n';
    std::cout << "audit_sidecar_schema=" << kAuditSidecarSchema << '\n';
    std::cout << "audit_sidecar_bytes=" << sidecar.bytes << '\n';
    std::cout << "audit_sidecar_sha256=" << sidecar.sha256 << '\n';
    std::cout << "vectors=" << config.vector_count << '\n';
    std::cout << "dimensions=" << config.dimension << '\n';
    std::cout << "M=" << config.m << '\n';
    std::cout << "ef_construction=" << config.ef_construction << '\n';
    std::cout << "ef_search=" << config.ef_search << '\n';
    std::cout << "participants=" << config.participant_count << '\n';
    std::cout << "build_grain=" << config.build_grain << '\n';
    std::cout << "query_benchmark_schema=" << kHnswD0QueryBenchmarkSchema << '\n';
    std::cout << "query_unique_queries=" << kHnswD0HeldOutQueryCount << '\n';
    std::cout << "query_k=" << kHnswD0QueryK << '\n';
    std::cout << "query_ef_search=" << kHnswD0QueryEf << '\n';
    std::cout << "query_recall_floor=" << kHnswD0QueryRecallFloor << '\n';
    std::cout << "query_maximum_absolute_recall_gap=" << kHnswD0QueryMaximumAbsoluteRecallGap << '\n';
    std::cout << "query_warmup_passes=" << kHnswD0QueryWarmupPasses << '\n';
    std::cout << "query_measured_passes=" << kHnswD0QueryMeasuredPasses << '\n';
    std::cout << "query_latency_concurrency=" << kHnswD0QueryLatencyConcurrency << '\n';
    std::cout << "query_throughput_concurrency=" << kHnswD0QueryThroughputConcurrency << '\n';
    std::cout << "query_percentile_method=nearest-rank\n";
    std::cout << "query_transaction_definition=one-process-local-top-k-call\n";
    std::cout << "query_timed_corpus_sha256=" << result.query_benchmark.timed_query_corpus_sha256 << '\n';
    std::cout << engine_name << "_valid=" << result.valid << '\n';
    std::cout << engine_name << "_threads=" << result.thread_count << '\n';
    std::cout << engine_name << "_index_size=" << result.index_size << '\n';
    std::cout << engine_name << "_cold_build_ns=" << result.cold_build_ns << '\n';
    std::cout << engine_name << "_insert_call_ns=" << result.insert_call_ns << '\n';
    std::cout << engine_name << "_self_recall_at_1=" << result.self_recall_at_1 << '\n';
    std::cout << engine_name << "_distance_checksum=" << result.distance_checksum << '\n';
    std::cout << engine_name << "_query_latency_sample_count=" << result.query_benchmark.latency_samples_ns.size() << '\n';
    std::cout << engine_name << "_query_latency_validated_operations=" << result.query_benchmark.latency_validated_operations << '\n';
    std::cout << engine_name << "_query_latency_validated_result_checksum="
              << result.query_benchmark.latency_validated_result_checksum << '\n';
    std::cout << engine_name << "_query_throughput_operations=" << result.query_benchmark.throughput_operations << '\n';
    std::cout << engine_name << "_query_throughput_validated_operations=" << result.query_benchmark.throughput_validated_operations << '\n';
    std::cout << engine_name << "_query_throughput_wall_ns=" << result.query_benchmark.throughput_wall_ns << '\n';
    std::cout << engine_name << "_query_throughput_validated_result_checksum="
              << result.query_benchmark.throughput_validated_result_checksum << '\n';
    std::cout << engine_name << "_query_result_checksum=" << result.query_benchmark.result_checksum << '\n';
    std::cout << engine_name << "_query_per_query_checksum_count=" << result.query_benchmark.per_query_checksums.size() << '\n';
    std::cout << engine_name << "_query_per_query_checksums=";
    for (std::size_t query_index = 0; query_index < result.query_benchmark.per_query_checksums.size(); ++query_index) {
        if (query_index != 0) {
            std::cout << ',';
        }
        std::cout << result.query_benchmark.per_query_checksums[query_index];
    }
    std::cout << '\n';
    std::cout << engine_name << "_incremental_treatment_compiled=" << result.execution_witness.incremental_treatment_compiled << '\n';
    std::cout << engine_name << "_incremental_capture_armed=" << result.execution_witness.incremental_capture_armed << '\n';
    std::cout << engine_name << "_incremental_eligible_branch_entered=" << result.execution_witness.incremental_eligible_branch_entered << '\n';
    std::cout << engine_name << "_incremental_successful_unchanged_observed="
              << result.execution_witness.incremental_successful_unchanged_observed << '\n';
    std::cout << engine_name << "_incremental_successful_updated_observed="
              << result.execution_witness.incremental_successful_updated_observed << '\n';
    std::cout << engine_name << "_threshold_treatment_compiled=" << result.execution_witness.threshold_treatment_compiled << '\n';
    std::cout << engine_name << "_threshold_capture_armed=" << result.execution_witness.threshold_capture_armed << '\n';
    std::cout << engine_name << "_threshold_eligible_branch_entered=" << result.execution_witness.threshold_eligible_branch_entered << '\n';
    std::cout << engine_name << "_threshold_rejected_lane_observed=" << result.execution_witness.threshold_rejected_lane_observed << '\n';
    std::cout << engine_name << "_threshold_surviving_lane_observed=" << result.execution_witness.threshold_surviving_lane_observed << '\n';
    std::cout << "heldout_query_count=" << result.recall_audit.query_count << '\n';
    std::cout << "heldout_query_seed=" << result.recall_audit.query_seed << '\n';
    std::cout << "heldout_queries_sha256=" << result.recall_audit.queries_sha256 << '\n';
    std::cout << "heldout_truth_sha256=" << result.recall_audit.truth_sha256 << '\n';
    std::cout << "heldout_truth_tie_counts_at_10=" << result.recall_audit.truth_tie_counts_at_10 << '\n';
    std::cout << "heldout_truth_tie_counts_at_100=" << result.recall_audit.truth_tie_counts_at_100 << '\n';
    std::cout << engine_name << "_graph_valid=" << result.graph_audit.valid << '\n';
    std::cout << engine_name << "_graph_vertex_count=" << result.graph_audit.vertex_count << '\n';
    std::cout << engine_name << "_graph_reachable_count=" << result.graph_audit.reachable_count << '\n';
    std::cout << engine_name << "_graph_directed_edges=" << result.graph_audit.directed_edges << '\n';
    std::cout << engine_name << "_graph_level0_directed_edges=" << result.graph_audit.level0_directed_edges << '\n';
    std::cout << engine_name << "_graph_max_level=" << result.graph_audit.max_level << '\n';
    std::cout << engine_name << "_graph_entry_point=" << result.graph_audit.entry_point << '\n';
    std::cout << engine_name << "_graph_level0_capacity=" << result.graph_audit.level0_capacity << '\n';
    std::cout << engine_name << "_graph_upper_capacity=" << result.graph_audit.upper_capacity << '\n';
    std::cout << engine_name << "_graph_levels_sha256=" << result.graph_audit.levels_sha256 << '\n';
    std::cout << engine_name << "_graph_sha256=" << result.graph_audit.graph_sha256 << '\n';
    std::cout << engine_name << "_graph_level_histogram=" << result.graph_audit.level_histogram << '\n';
    std::cout << engine_name << "_graph_degree_histograms=" << result.graph_audit.degree_histograms << '\n';
    for (std::size_t point = 0; point < kHnswD0RecallEf.size(); ++point) {
        if (point < 5) {
            std::cout << engine_name << "_recall_at_10_ef_" << kHnswD0RecallEf[point] << '=' << result.recall_audit.recall_at_10[point] << '\n';
        } else {
            std::cout << engine_name << "_recall_at_100_ef_" << kHnswD0RecallEf[point] << '=' << result.recall_audit.recall_at_100[point - 5] << '\n';
        }
        std::cout << engine_name << "_returned_ids_ef_" << kHnswD0RecallEf[point] << "_k_" << kHnswD0RecallK[point] << '='
                  << result.recall_audit.returned_ids[point] << '\n';
        std::cout << engine_name << "_returned_distances_sha256_ef_" << kHnswD0RecallEf[point] << "_k_" << kHnswD0RecallK[point] << '='
                  << result.recall_audit.returned_distances_sha256[point] << '\n';
    }
    if (engine == HnswD0Engine::kInfinity) {
        std::cout << "infinity_submitted_tasks=" << result.submitted_tasks << '\n';
        std::cout << "infinity_build_start=" << result.build_start << '\n';
        std::cout << "infinity_build_end=" << result.build_end << '\n';
    }
}

} // namespace

int RunHnswD0(int argc, char **argv, HnswD0Engine engine, HnswD0Bridge bridge) {
    if (argc == 2 && std::strcmp(argv[1], "--attest-only") == 0) {
        return RunAttestationOnly();
    }
    if (argc != 12 && argc != 17) {
        PrintUsage(argc > 0 ? argv[0] : "hnsw_d0");
        return kUsageExitCode;
    }
    g_campaign_mode = (argc == 17);  // argc 12 = plain form, argc 17 = plain + 5 binding args

    HnswDevConfig config{};
    AuditBinding binding;
    Sha256Digest expected_executable_sha256{};
    Sha256Digest expected_dataset_sha256{};
    if (!ParseInteger(argv[2], config.vector_count) || !ParseInteger(argv[3], config.dimension) || !ParseInteger(argv[4], config.m) ||
        !ParseInteger(argv[5], config.ef_construction) || !ParseInteger(argv[6], config.ef_search) || !ParseInteger(argv[7], config.chunk_size) ||
        !ParseInteger(argv[8], config.query_count) || !ParseInteger(argv[9], config.participant_count) ||
        !ParseInteger(argv[10], config.build_grain)) {
        PrintUsage(argv[0]);
        return kUsageExitCode;
    }
    if (g_campaign_mode) {
        if (!ParseSha256Hex(argv[12], binding.campaign_nonce) || !ParseInteger(argv[13], binding.schedule_sequence) ||
            !ParseInteger(argv[14], binding.role_id) || !ParseSha256Hex(argv[15], expected_executable_sha256) ||
            !ParseSha256Hex(argv[16], expected_dataset_sha256) || !RoleMatchesEngine(binding.role_id, engine)) {
            PrintUsage(argv[0]);
            return kUsageExitCode;
        }
    } else {
        binding.role_id = DefaultRoleForEngine(engine);
    }

    std::size_t data_bytes = 0;
    if (!ValidateConfig(config, data_bytes)) {
        PrintUsage(argv[0]);
        return kUsageExitCode;
    }
    config.index_barrier = StopAtIndexBarrier;
    config.index_barrier_context = nullptr;

    HnswD0LiveAttestation attestation;
    std::string executable_diagnostic;
    if (!attestation.Begin(executable_diagnostic)) {
        std::cerr << "evidence binding attestation failed: " << executable_diagnostic << '\n';
        return kSidecarExitCode;
    }
    if (!HashHeldExecutable(attestation.executable_fd(), binding.executable_sha256, executable_diagnostic)) {
        std::cerr << "evidence binding failed: " << executable_diagnostic << '\n';
        return kSidecarExitCode;
    }
    if (g_campaign_mode && binding.executable_sha256 != expected_executable_sha256) {
        std::cerr << "evidence binding failed: executable SHA-256 mismatch: expected " << Sha256Hex(expected_executable_sha256) << ", got "
                  << Sha256Hex(binding.executable_sha256) << '\n';
        return kSidecarExitCode;
    }

    OwnedDataset dataset;
    if (!dataset.Open(argv[1], data_bytes)) {
        return 66;
    }
    const std::uint64_t data_hash = Fnv1a64(dataset.bytes(), data_bytes);
    binding.dataset_sha256 = ComputeSha256(dataset.bytes(), data_bytes);
    const std::string data_sha256 = Sha256Hex(binding.dataset_sha256);
    if (g_campaign_mode && binding.dataset_sha256 != expected_dataset_sha256) {
        std::cerr << "evidence binding failed: dataset SHA-256 mismatch: expected " << Sha256Hex(expected_dataset_sha256) << ", got " << data_sha256
                  << '\n';
        return kSidecarExitCode;
    }
    Prefault(dataset.bytes(), data_bytes);

    HnswD0AttestationSnapshot attestation_before;
    if (!StopAtBeforeWorkBarrier(attestation, attestation_before, executable_diagnostic)) {
        std::cerr << "before-work attestation failed: " << executable_diagnostic << '\n';
        return kSidecarExitCode;
    }

    HnswDevResult result{};
    const int bridge_status = bridge(dataset.floats(), &config, &result);
    Sha256Digest executable_sha256_after_bridge{};
    const bool executable_hash_valid =
        HashHeldExecutable(attestation.executable_fd(), executable_sha256_after_bridge, executable_diagnostic);
    const bool executable_hash_unchanged =
        !g_campaign_mode || (executable_hash_valid && executable_sha256_after_bridge == binding.executable_sha256);
    HnswD0AttestationSnapshot attestation_after;
    HnswD0AttestationSummary attestation_summary;
    std::string attestation_diagnostic;
    const bool live_attestation_valid =
        StopAtAfterWorkBarrier(attestation, attestation_before, attestation_after, attestation_summary, attestation_diagnostic);
    if (!executable_hash_valid) {
        std::cerr << "evidence binding failed after benchmark: " << executable_diagnostic << '\n';
        return kSidecarExitCode;
    }
    if (!executable_hash_unchanged) {
        std::cerr << "evidence binding failed: executable SHA-256 changed during benchmark: expected " << Sha256Hex(binding.executable_sha256)
                  << ", got " << Sha256Hex(executable_sha256_after_bridge) << '\n';
        return kSidecarExitCode;
    }
    if (!live_attestation_valid) {
        std::cerr << "after-work attestation failed: " << attestation_diagnostic << '\n';
        return kSidecarExitCode;
    }
    bool passed = bridge_status == 0 && result.valid == 1;
    const bool complete_failure = bridge_status == 1 && result.valid == 0;
    bool sidecar_failed = false;
    AuditSidecarEvidence sidecar;
    if (passed || complete_failure) {
        std::string diagnostic;
        if (!CreateAuditSidecar(argv[11], engine, binding, config, result, sidecar, diagnostic)) {
            std::cerr << "audit sidecar failed: " << diagnostic << '\n';
            passed = false;
            sidecar_failed = true;
        }
    }
    PrintResult(engine, binding, data_hash, data_sha256, data_bytes, config, result, sidecar, passed);
    if (sidecar_failed) {
        return kSidecarExitCode;
    }
    if (passed) {
        return 0;
    }
    return bridge_status != 0 ? bridge_status : 1;
}
