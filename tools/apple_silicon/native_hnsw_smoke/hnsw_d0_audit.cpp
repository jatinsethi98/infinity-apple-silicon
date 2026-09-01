#include "hnsw_d0_audit.h"

#if defined(__APPLE__)
#include <libproc.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <sys/proc_info.h>
#endif
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

#if defined(__APPLE__)

constexpr unsigned int kCsOpsCdHash = 5;
constexpr std::size_t kCdHashBytes = 20;

extern "C" int csops(pid_t pid, unsigned int operations, void *user_address, std::size_t user_size);

std::atomic<std::uint64_t> g_attestation_image_adds{0};
std::atomic<std::uint64_t> g_attestation_image_removes{0};
std::atomic<bool> g_attestation_callbacks_registered{false};

void AttestationImageAdded(const mach_header *, std::intptr_t) noexcept {
    g_attestation_image_adds.fetch_add(1, std::memory_order_relaxed);
}

void AttestationImageRemoved(const mach_header *, std::intptr_t) noexcept {
    g_attestation_image_removes.fetch_add(1, std::memory_order_relaxed);
}

#endif

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU,
    0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU,
    0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U,
    0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

std::uint32_t RotateRight(std::uint32_t value, unsigned int count) { return (value >> count) | (value << (32U - count)); }

class Sha256 {
public:
    void Update(const void *input, std::size_t size) {
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

    void Update(std::string_view text) { Update(text.data(), text.size()); }

    void UpdateU32(std::uint32_t value) {
        std::array<std::uint8_t, 4> bytes{};
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
        }
        Update(bytes.data(), bytes.size());
    }

    void UpdateU64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
        }
        Update(bytes.data(), bytes.size());
    }

    std::string HexDigest() const {
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

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint32_t word : finalized.state_) {
            output << std::setw(8) << word;
        }
        return output.str();
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
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

void Require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::string EncodeHistogram(const std::vector<std::size_t> &histogram) {
    std::string encoded;
    for (std::size_t index = 0; index < histogram.size(); ++index) {
        if (index != 0) {
            encoded.push_back(',');
        }
        encoded += std::to_string(index);
        encoded.push_back(':');
        encoded += std::to_string(histogram[index]);
    }
    return encoded;
}

std::string EncodeDegreeHistograms(const std::vector<std::vector<std::size_t>> &histograms) {
    std::string encoded;
    for (std::size_t layer = 0; layer < histograms.size(); ++layer) {
        if (layer != 0) {
            encoded.push_back(';');
        }
        encoded.push_back('L');
        encoded += std::to_string(layer);
        encoded.push_back(':');
        for (std::size_t degree = 0; degree < histograms[layer].size(); ++degree) {
            if (degree != 0) {
                encoded.push_back(',');
            }
            encoded += std::to_string(histograms[layer][degree]);
        }
    }
    return encoded;
}

void AppendCommaSeparated(std::string &output, std::int64_t value) {
    if (!output.empty()) {
        output.push_back(',');
    }
    output += std::to_string(value);
}

} // namespace

class HnswD0LiveAttestation::Impl {
public:
    ~Impl() {
        if (executable_fd >= 0) {
            close(executable_fd);
        }
    }

    int executable_fd = -1;
    HnswD0FileIdentity executable_identity;
    std::uint64_t initial_image_adds = 0;
    bool begun = false;
};

namespace {

HnswD0FileIdentity FileIdentity(const struct stat &status) {
    return HnswD0FileIdentity{
        .device = static_cast<std::uint64_t>(status.st_dev),
        .inode = static_cast<std::uint64_t>(status.st_ino),
        .size = static_cast<std::uint64_t>(status.st_size),
    };
}

#if defined(__APPLE__)

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

bool MachOUuid(const mach_header *header, std::string &uuid) {
    if (header == nullptr || header->magic != MH_MAGIC_64) {
        return false;
    }
    const auto *header64 = reinterpret_cast<const mach_header_64 *>(header);
    const auto *cursor = reinterpret_cast<const std::uint8_t *>(header64 + 1);
    std::size_t remaining = header64->sizeofcmds;
    for (std::uint32_t index = 0; index < header64->ncmds; ++index) {
        if (remaining < sizeof(load_command)) {
            return false;
        }
        const auto *command = reinterpret_cast<const load_command *>(cursor);
        if (command->cmdsize < sizeof(load_command) || command->cmdsize > remaining) {
            return false;
        }
        if (command->cmd == LC_UUID) {
            if (command->cmdsize < sizeof(uuid_command)) {
                return false;
            }
            const auto *uuid_record = reinterpret_cast<const uuid_command *>(command);
            uuid = HexBytes(uuid_record->uuid, sizeof(uuid_record->uuid));
            return true;
        }
        cursor += command->cmdsize;
        remaining -= command->cmdsize;
    }
    return false;
}

bool CaptureImages(std::vector<HnswD0AttestationImage> &images, std::string &diagnostic) {
    const std::uint32_t count = _dyld_image_count();
    images.clear();
    images.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        const mach_header *header = _dyld_get_image_header(index);
        const char *path = _dyld_get_image_name(index);
        if (header == nullptr || path == nullptr || path[0] == '\0') {
            diagnostic = "dyld returned an incomplete image record";
            return false;
        }
        HnswD0AttestationImage image;
        image.header = reinterpret_cast<std::uintptr_t>(header);
        image.slide = _dyld_get_image_vmaddr_slide(index);
        image.path = path;
        if (!MachOUuid(header, image.uuid)) {
            diagnostic = "loaded image has no valid LC_UUID: " + image.path;
            return false;
        }
        image.shared_cache = _dyld_shared_cache_contains_path(path);
        images.push_back(std::move(image));
    }
    return true;
}

bool CaptureDynamicCdHash(std::string &cdhash, std::string &diagnostic) {
    std::array<std::uint8_t, kCdHashBytes> bytes{};
    if (csops(getpid(), kCsOpsCdHash, bytes.data(), bytes.size()) != 0) {
        const int error = errno;
        diagnostic = "csops(CS_OPS_CDHASH) failed: " + std::string(std::strerror(error));
        return false;
    }
    cdhash = HexBytes(bytes.data(), bytes.size());
    return true;
}

bool CaptureMappedExecutable(HnswD0FileIdentity &identity, std::string &diagnostic) {
    const mach_header *header = _dyld_get_image_header(0);
    if (header == nullptr) {
        diagnostic = "dyld returned no main executable header";
        return false;
    }
    const std::uint64_t header_address = reinterpret_cast<std::uintptr_t>(header);
    proc_regionwithpathinfo region{};
    const int bytes = proc_pidinfo(getpid(),
                                   PROC_PIDREGIONPATHINFO,
                                   header_address,
                                   &region,
                                   static_cast<int>(sizeof(region)));
    if (bytes != static_cast<int>(sizeof(region))) {
        const int error = errno;
        diagnostic = bytes < 0 ? "proc_pidinfo(PROC_PIDREGIONPATHINFO) failed: " + std::string(std::strerror(error))
                               : "proc_pidinfo(PROC_PIDREGIONPATHINFO) returned a short record";
        return false;
    }
    const std::uint64_t base = region.prp_prinfo.pri_address;
    const std::uint64_t size = region.prp_prinfo.pri_size;
    if (size == 0 || base > header_address || header_address - base >= size) {
        diagnostic = "main executable header is outside the reported mapped region";
        return false;
    }
    const vinfo_stat &status = region.prp_vip.vip_vi.vi_stat;
    if (status.vst_ino == 0 || status.vst_size < 0) {
        diagnostic = "main executable mapping has no valid vnode identity";
        return false;
    }
    identity = HnswD0FileIdentity{
        .device = static_cast<std::uint64_t>(status.vst_dev),
        .inode = status.vst_ino,
        .size = static_cast<std::uint64_t>(status.vst_size),
    };
    return true;
}

#endif

} // namespace

bool HnswD0MappedExecutableMatchesHeldFile(const HnswD0AttestationSnapshot &snapshot) {
    return snapshot.held_executable.device != 0 && snapshot.held_executable.inode != 0 &&
           snapshot.held_executable == snapshot.mapped_executable;
}

HnswD0LiveAttestation::HnswD0LiveAttestation() : impl_(std::make_unique<Impl>()) {}

HnswD0LiveAttestation::~HnswD0LiveAttestation() = default;

HnswD0LiveAttestation::HnswD0LiveAttestation(HnswD0LiveAttestation &&) noexcept = default;

HnswD0LiveAttestation &HnswD0LiveAttestation::operator=(HnswD0LiveAttestation &&) noexcept = default;

bool HnswD0LiveAttestation::Begin(std::string &diagnostic) {
    if (impl_->begun) {
        diagnostic = "live attestation session was already started";
        return false;
    }
#if defined(__APPLE__)
    bool expected = false;
    if (g_attestation_callbacks_registered.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        g_attestation_image_adds.store(0, std::memory_order_relaxed);
        g_attestation_image_removes.store(0, std::memory_order_relaxed);
        _dyld_register_func_for_add_image(AttestationImageAdded);
        _dyld_register_func_for_remove_image(AttestationImageRemoved);
    }
    impl_->initial_image_adds = g_attestation_image_adds.load(std::memory_order_acquire);

    std::vector<char> process_path(1024);
    std::uint32_t process_path_size = static_cast<std::uint32_t>(process_path.size());
    if (_NSGetExecutablePath(process_path.data(), &process_path_size) != 0) {
        process_path.resize(process_path_size);
        if (_NSGetExecutablePath(process_path.data(), &process_path_size) != 0) {
            diagnostic = "_NSGetExecutablePath failed for the running executable";
            return false;
        }
    }
    char *canonical_path = realpath(process_path.data(), nullptr);
    if (canonical_path == nullptr) {
        const int error = errno;
        diagnostic = "realpath failed for the running executable: " + std::string(std::strerror(error));
        return false;
    }
    int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    impl_->executable_fd = open(canonical_path, flags);
    const int open_error = errno;
    std::free(canonical_path);
    if (impl_->executable_fd < 0) {
        diagnostic = "open failed for the running executable: " + std::string(std::strerror(open_error));
        return false;
    }
    struct stat status {};
    if (fstat(impl_->executable_fd, &status) != 0) {
        const int error = errno;
        diagnostic = "fstat failed for the held executable: " + std::string(std::strerror(error));
        return false;
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0) {
        diagnostic = "held executable is not a regular file";
        return false;
    }
    impl_->executable_identity = FileIdentity(status);
    impl_->begun = true;
    return true;
#else
    diagnostic = "live executable attestation requires macOS";
    return false;
#endif
}

bool HnswD0LiveAttestation::Capture(HnswD0AttestationSnapshot &snapshot, std::string &diagnostic) const {
    if (!impl_->begun || impl_->executable_fd < 0) {
        diagnostic = "live attestation session was not started";
        return false;
    }
#if defined(__APPLE__)
    struct stat status {};
    if (fstat(impl_->executable_fd, &status) != 0) {
        const int error = errno;
        diagnostic = "fstat failed for the held executable: " + std::string(std::strerror(error));
        return false;
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0) {
        diagnostic = "held executable is no longer a regular file";
        return false;
    }
    snapshot.held_executable = FileIdentity(status);
    if (snapshot.held_executable != impl_->executable_identity) {
        diagnostic = "held executable identity changed during attestation";
        return false;
    }
    if (!CaptureMappedExecutable(snapshot.mapped_executable, diagnostic)) {
        return false;
    }
    if (!HnswD0MappedExecutableMatchesHeldFile(snapshot)) {
        diagnostic = "mapped executable vnode does not match the held executable descriptor";
        return false;
    }
    if (!CaptureDynamicCdHash(snapshot.dynamic_cdhash, diagnostic)) {
        return false;
    }
    if (!CaptureImages(snapshot.images, diagnostic)) {
        return false;
    }
    return true;
#else
    static_cast<void>(snapshot);
    diagnostic = "live executable attestation requires macOS";
    return false;
#endif
}

bool HnswD0LiveAttestation::Validate(const HnswD0AttestationSnapshot &before,
                                     const HnswD0AttestationSnapshot &after,
                                     HnswD0AttestationSummary &summary,
                                     std::string &diagnostic) const {
    if (!impl_->begun) {
        diagnostic = "live attestation session was not started";
        return false;
    }
#if defined(__APPLE__)
    const std::uint64_t final_adds = g_attestation_image_adds.load(std::memory_order_acquire);
    const std::uint64_t final_removes = g_attestation_image_removes.load(std::memory_order_acquire);
    if (final_adds < impl_->initial_image_adds) {
        diagnostic = "dyld image-add counter regressed";
        return false;
    }
    summary.initial_image_adds = impl_->initial_image_adds;
    summary.later_image_adds = final_adds - impl_->initial_image_adds;
    summary.image_removes = final_removes;
    summary.image_set_unchanged = before.images == after.images;

    if (!HnswD0MappedExecutableMatchesHeldFile(before) || !HnswD0MappedExecutableMatchesHeldFile(after)) {
        diagnostic = "mapped executable vnode does not match the held executable descriptor";
        return false;
    }
    if (before.held_executable != impl_->executable_identity || after.held_executable != impl_->executable_identity) {
        diagnostic = "held executable identity changed during attestation";
        return false;
    }
    if (before.dynamic_cdhash.empty() || before.dynamic_cdhash != after.dynamic_cdhash) {
        diagnostic = "dynamic CDHash changed during attestation";
        return false;
    }
    if (!summary.image_set_unchanged || summary.later_image_adds != 0 || summary.image_removes != 0) {
        diagnostic = "dynamic image set changed during attestation";
        return false;
    }
    return true;
#else
    static_cast<void>(before);
    static_cast<void>(after);
    static_cast<void>(summary);
    diagnostic = "live executable attestation requires macOS";
    return false;
#endif
}

int HnswD0LiveAttestation::executable_fd() const {
    return impl_->begun ? impl_->executable_fd : -1;
}

HnswD0GraphAudit AuditHnswD0Graph(const HnswD0GraphInput &input) {
    Require(input.vertex_count > 0, "HNSW graph audit requires at least one vertex");
    Require(input.vertex_count <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()), "HNSW graph vertex count exceeds int32");
    Require(input.max_level >= 0, "HNSW graph maximum level is negative");
    Require(input.entry_point >= 0 && static_cast<std::size_t>(input.entry_point) < input.vertex_count, "HNSW graph entry point is out of range");
    Require(input.level0_capacity > 0 && input.upper_capacity > 0, "HNSW graph capacities are invalid");
    Require(input.level && input.label && input.neighbors, "HNSW graph audit callbacks are missing");

    HnswD0GraphAudit result;
    result.vertex_count = input.vertex_count;
    result.max_level = input.max_level;
    result.entry_point = input.entry_point;
    result.level0_capacity = input.level0_capacity;
    result.upper_capacity = input.upper_capacity;
    result.vertices.resize(input.vertex_count);

    std::vector<std::int32_t> levels(input.vertex_count);
    std::vector<bool> labels(input.vertex_count, false);
    std::vector<std::size_t> level_histogram(static_cast<std::size_t>(input.max_level) + 1, 0);
    Sha256 levels_hash;
    levels_hash.Update("infinity-hnsw-levels-v1");
    levels_hash.UpdateU64(input.vertex_count);
    std::int32_t observed_max_level = -1;
    for (std::size_t vertex = 0; vertex < input.vertex_count; ++vertex) {
        const auto ordinal = static_cast<std::int32_t>(vertex);
        const std::int32_t level = input.level(ordinal);
        Require(level >= 0 && level <= input.max_level, "HNSW graph vertex level is invalid");
        levels[vertex] = level;
        result.vertices[vertex].level = level;
        result.vertices[vertex].layers.resize(static_cast<std::size_t>(level) + 1);
        ++level_histogram[static_cast<std::size_t>(level)];
        observed_max_level = std::max(observed_max_level, level);
        levels_hash.UpdateU32(static_cast<std::uint32_t>(ordinal));
        levels_hash.UpdateU32(static_cast<std::uint32_t>(level));

        const std::int64_t label = input.label(ordinal);
        Require(label >= 0 && static_cast<std::size_t>(label) < input.vertex_count, "HNSW graph label is out of range");
        Require(!labels[static_cast<std::size_t>(label)], "HNSW graph contains a duplicate label");
        labels[static_cast<std::size_t>(label)] = true;
        result.vertices[vertex].label = label;
    }
    Require(observed_max_level == input.max_level, "HNSW graph maximum level disagrees with its vertices");
    Require(levels[static_cast<std::size_t>(input.entry_point)] == input.max_level, "HNSW graph entry point is not on the maximum level");
    result.levels_sha256 = levels_hash.HexDigest();
    result.level_histogram = EncodeHistogram(level_histogram);

    std::vector<std::vector<std::size_t>> degree_histograms;
    degree_histograms.reserve(static_cast<std::size_t>(input.max_level) + 1);
    for (std::int32_t layer = 0; layer <= input.max_level; ++layer) {
        const std::size_t capacity = layer == 0 ? input.level0_capacity : input.upper_capacity;
        degree_histograms.emplace_back(capacity + 1, 0);
    }

    Sha256 graph_hash;
    graph_hash.Update("infinity-hnsw-graph-v1");
    graph_hash.UpdateU64(input.vertex_count);
    graph_hash.UpdateU32(static_cast<std::uint32_t>(input.max_level));
    graph_hash.UpdateU32(static_cast<std::uint32_t>(input.entry_point));
    graph_hash.UpdateU64(input.level0_capacity);
    graph_hash.UpdateU64(input.upper_capacity);
    for (std::size_t vertex = 0; vertex < input.vertex_count; ++vertex) {
        const auto ordinal = static_cast<std::int32_t>(vertex);
        graph_hash.UpdateU32(static_cast<std::uint32_t>(ordinal));
        graph_hash.UpdateU32(static_cast<std::uint32_t>(levels[vertex]));
        graph_hash.UpdateU64(static_cast<std::uint64_t>(result.vertices[vertex].label));
        for (std::int32_t layer = 0; layer <= levels[vertex]; ++layer) {
            const HnswD0NeighborRange range = input.neighbors(ordinal, layer);
            const std::size_t expected_capacity = layer == 0 ? input.level0_capacity : input.upper_capacity;
            Require(range.capacity == expected_capacity, "HNSW graph layer capacity is inconsistent");
            Require(range.size <= range.capacity, "HNSW graph degree exceeds capacity");
            Require(range.size == 0 || range.data != nullptr, "HNSW graph neighbor storage is missing");
            Require(range.tail_valid, "HNSW graph has a non-sentinel entry after an unused slot");
            ++degree_histograms[static_cast<std::size_t>(layer)][range.size];
            result.directed_edges += range.size;
            if (layer == 0) {
                result.level0_directed_edges += range.size;
            }
            graph_hash.UpdateU32(static_cast<std::uint32_t>(layer));
            graph_hash.UpdateU64(range.size);
            HnswD0GraphLayerAudit &retained_layer = result.vertices[vertex].layers[static_cast<std::size_t>(layer)];
            retained_layer.capacity = range.capacity;
            retained_layer.neighbors.reserve(range.size);
            for (std::size_t index = 0; index < range.size; ++index) {
                const std::int32_t neighbor = range.data[index];
                Require(neighbor >= 0 && static_cast<std::size_t>(neighbor) < input.vertex_count, "HNSW graph contains an out-of-range edge");
                Require(neighbor != ordinal, "HNSW graph contains a self edge");
                Require(levels[static_cast<std::size_t>(neighbor)] >= layer, "HNSW graph upper-layer edge targets a lower-level vertex");
                for (std::size_t previous = 0; previous < index; ++previous) {
                    Require(range.data[previous] != neighbor, "HNSW graph contains a duplicate edge");
                }
                graph_hash.UpdateU32(static_cast<std::uint32_t>(neighbor));
                retained_layer.neighbors.push_back(neighbor);
            }
        }
    }
    result.graph_sha256 = graph_hash.HexDigest();
    result.degree_histograms = EncodeDegreeHistograms(degree_histograms);

    std::vector<bool> visited(input.vertex_count, false);
    std::vector<std::int32_t> pending;
    pending.reserve(input.vertex_count);
    pending.push_back(input.entry_point);
    visited[static_cast<std::size_t>(input.entry_point)] = true;
    for (std::size_t cursor = 0; cursor < pending.size(); ++cursor) {
        const auto &neighbors = result.vertices[static_cast<std::size_t>(pending[cursor])].layers[0].neighbors;
        for (const std::int32_t neighbor : neighbors) {
            if (!visited[static_cast<std::size_t>(neighbor)]) {
                visited[static_cast<std::size_t>(neighbor)] = true;
                pending.push_back(neighbor);
            }
        }
    }
    result.reachable_count = pending.size();
    result.valid = result.reachable_count == input.vertex_count;
    return result;
}

std::string HnswD0QueryCorpusSha256(const std::vector<float> &queries, std::size_t dimension) {
    Require(dimension > 0, "HNSW query corpus dimension is invalid");
    Require(
        dimension <= std::numeric_limits<std::size_t>::max() / kHnswD0HeldOutQueryCount &&
            queries.size() == kHnswD0HeldOutQueryCount * dimension,
        "HNSW query corpus has the wrong size");

    Sha256 query_hash;
    query_hash.Update("infinity-hnsw-d0-queries-v1");
    query_hash.UpdateU64(kHnswD0HeldOutQueryCount);
    query_hash.UpdateU64(dimension);
    for (const float value : queries) {
        query_hash.UpdateU32(std::bit_cast<std::uint32_t>(value));
    }
    return query_hash.HexDigest();
}

HnswD0RecallAudit AuditHnswD0Recall(const float *base,
                                    std::size_t vector_count,
                                    std::size_t dimension,
                                    const std::vector<float> &queries,
                                    const HnswD0Search &search) {
    Require(base != nullptr, "HNSW recall audit base pointer is null");
    Require(vector_count >= 100 && vector_count <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "HNSW recall audit vector count is invalid");
    Require(dimension > 0, "HNSW recall audit dimension is invalid");
    Require(queries.size() == kHnswD0HeldOutQueryCount * dimension, "HNSW recall audit query corpus has the wrong size");
    Require(bool(search), "HNSW recall audit search callback is missing");

    HnswD0RecallAudit result;
    result.query_count = kHnswD0HeldOutQueryCount;
    result.query_seed = kHnswD0HeldOutQuerySeed;
    result.queries_sha256 = HnswD0QueryCorpusSha256(queries, dimension);

    Sha256 truth_hash;
    truth_hash.Update("infinity-hnsw-d0-truth-v1");
    truth_hash.UpdateU64(kHnswD0HeldOutQueryCount);
    truth_hash.UpdateU64(vector_count);
    truth_hash.UpdateU64(dimension);
    std::array<Sha256, kHnswD0RecallEf.size()> distance_hashes;
    std::array<std::size_t, 5> recall10_hits{};
    std::array<std::size_t, 3> recall100_hits{};
    std::vector<double> exact_distances(vector_count);
    std::vector<std::int32_t> exact_order(vector_count);
    std::iota(exact_order.begin(), exact_order.end(), 0);
    for (std::size_t point = 0; point < kHnswD0RecallK.size(); ++point) {
        result.returned_results[point].reserve(kHnswD0HeldOutQueryCount * kHnswD0RecallK[point]);
    }

    for (std::size_t query_index = 0; query_index < kHnswD0HeldOutQueryCount; ++query_index) {
        const float *query = queries.data() + query_index * dimension;
        bool matches_base_row = false;
        for (std::size_t vertex = 0; vertex < vector_count; ++vertex) {
            const float *vector = base + vertex * dimension;
            double distance = 0.0;
            bool equal = true;
            for (std::size_t component = 0; component < dimension; ++component) {
                const double difference = static_cast<double>(query[component]) - static_cast<double>(vector[component]);
                distance += difference * difference;
                equal = equal && query[component] == vector[component];
            }
            exact_distances[vertex] = distance;
            matches_base_row = matches_base_row || equal;
        }
        Require(!matches_base_row, "HNSW held-out query duplicates an indexed row");
        std::sort(exact_order.begin(), exact_order.end(), [&](std::int32_t left, std::int32_t right) {
            const double left_distance = exact_distances[static_cast<std::size_t>(left)];
            const double right_distance = exact_distances[static_cast<std::size_t>(right)];
            return left_distance < right_distance || (left_distance == right_distance && left < right);
        });
        const double cutoff10 = exact_distances[static_cast<std::size_t>(exact_order[9])];
        const double cutoff100 = exact_distances[static_cast<std::size_t>(exact_order[99])];
        const std::size_t ties10 =
            static_cast<std::size_t>(std::count_if(exact_distances.begin(), exact_distances.end(), [&](double value) { return value <= cutoff10; }));
        const std::size_t ties100 =
            static_cast<std::size_t>(std::count_if(exact_distances.begin(), exact_distances.end(), [&](double value) { return value <= cutoff100; }));
        AppendCommaSeparated(result.truth_tie_counts_at_10, static_cast<std::int64_t>(ties10));
        AppendCommaSeparated(result.truth_tie_counts_at_100, static_cast<std::int64_t>(ties100));
        truth_hash.UpdateU64(query_index);
        truth_hash.UpdateU64(ties10);
        truth_hash.UpdateU64(ties100);
        for (std::size_t rank = 0; rank < 100; ++rank) {
            truth_hash.UpdateU32(static_cast<std::uint32_t>(exact_order[rank]));
        }

        for (std::size_t point = 0; point < kHnswD0RecallEf.size(); ++point) {
            const std::size_t requested_k = kHnswD0RecallK[point];
            const auto returned = search(query, requested_k, kHnswD0RecallEf[point]);
            Require(returned.size() == requested_k, "HNSW recall audit search returned the wrong result count");
            std::vector<bool> seen(vector_count, false);
            float previous_distance = -std::numeric_limits<float>::infinity();
            for (std::size_t rank = 0; rank < returned.size(); ++rank) {
                const auto [distance, label] = returned[rank];
                Require(std::isfinite(distance), "HNSW recall audit search returned a non-finite distance");
                Require(distance >= previous_distance, "HNSW recall audit search results are not sorted");
                Require(label >= 0 && static_cast<std::size_t>(label) < vector_count, "HNSW recall audit search returned an out-of-range label");
                Require(!seen[static_cast<std::size_t>(label)], "HNSW recall audit search returned a duplicate label");
                previous_distance = distance;
                seen[static_cast<std::size_t>(label)] = true;
                AppendCommaSeparated(result.returned_ids[point], label);
                distance_hashes[point].UpdateU32(std::bit_cast<std::uint32_t>(distance));
                result.returned_results[point].push_back(HnswD0ReturnedResult{.distance = distance, .label = label});
                if (requested_k == 10 && exact_distances[static_cast<std::size_t>(label)] <= cutoff10) {
                    ++recall10_hits[point];
                }
                if (requested_k == 100 && exact_distances[static_cast<std::size_t>(label)] <= cutoff100) {
                    ++recall100_hits[point - 5];
                }
            }
        }
    }

    result.truth_sha256 = truth_hash.HexDigest();
    const double recall10_denominator = static_cast<double>(kHnswD0HeldOutQueryCount * 10);
    const double recall100_denominator = static_cast<double>(kHnswD0HeldOutQueryCount * 100);
    for (std::size_t point = 0; point < kHnswD0RecallEf.size(); ++point) {
        Require(result.returned_results[point].size() == kHnswD0HeldOutQueryCount * kHnswD0RecallK[point],
                "HNSW recall audit retained the wrong result count");
        result.returned_distances_sha256[point] = distance_hashes[point].HexDigest();
        if (point < 5) {
            result.recall_at_10[point] = static_cast<double>(recall10_hits[point]) / recall10_denominator;
        } else {
            result.recall_at_100[point - 5] = static_cast<double>(recall100_hits[point - 5]) / recall100_denominator;
        }
    }
    result.valid = true;
    return result;
}

namespace {

// Exact squared L2 in double precision. This TU is compiled with -ffp-contract=off (see the
// CMakeLists), which matters here: an fused multiply-add would change the low bit of a sum
// and could flip the `<= cutoff` comparison for a candidate sitting exactly on the cutoff --
// precisely the tie case this audit is built to count correctly.
double ExactSquaredL2(const float *left, const float *right, std::size_t dimension) {
    double total = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        const double difference = static_cast<double>(left[index]) - static_cast<double>(right[index]);
        total += difference * difference;
    }
    return total;
}

} // namespace

HnswD0ExternalRecallAudit AuditHnswD0RecallExternal(const float *base,
                                                    std::size_t vector_count,
                                                    std::size_t dimension,
                                                    const std::vector<float> &queries,
                                                    const std::vector<std::int32_t> &groundtruth,
                                                    std::size_t groundtruth_columns,
                                                    std::size_t query_limit,
                                                    const HnswD0Search &search) {
    constexpr std::size_t kK = 10;

    Require(base != nullptr, "external recall audit base pointer is null");
    // Enforced, not documented: the published IDs index the full canonical corpus, and
    // against a prefix they would still be in range for a large enough prefix -- so the
    // failure mode is a plausible-looking wrong number, not a crash.
    Require(vector_count == kHnswD0ExternalTruthVectorCount,
            "external recall audit requires the full canonical base (n=1000000); published "
            "ground-truth IDs address that corpus and are meaningless against a prefix");
    Require(dimension > 0, "external recall audit dimension is invalid");
    Require(bool(search), "external recall audit search callback is missing");
    // kK for the cutoff, plus one more rank to detect a tie sitting on it.
    Require(groundtruth_columns > kK, "external recall audit needs more than 10 ground-truth columns");
    Require(queries.size() % dimension == 0, "external recall audit query corpus is not a whole number of rows");

    const std::size_t available_queries = queries.size() / dimension;
    Require(available_queries > 0, "external recall audit query corpus is empty");
    Require(groundtruth.size() == available_queries * groundtruth_columns,
            "external recall audit ground truth row count does not match the query corpus");

    const std::size_t query_count = (query_limit == 0 || query_limit > available_queries) ? available_queries : query_limit;

    HnswD0ExternalRecallAudit result;
    result.truth_source = "official-sift1m";
    result.query_count = query_count;
    result.groundtruth_columns = groundtruth_columns;

    // Hash the exact bytes consumed, so a swapped or truncated dataset cannot masquerade as
    // this one. Only the evaluated prefix is covered -- that is what the numbers depend on.
    Sha256 query_hash;
    query_hash.Update("infinity-hnsw-d0-external-queries-v1");
    query_hash.UpdateU64(query_count);
    query_hash.UpdateU64(dimension);
    for (std::size_t index = 0; index < query_count * dimension; ++index) {
        query_hash.UpdateU32(std::bit_cast<std::uint32_t>(queries[index]));
    }
    result.queries_sha256 = query_hash.HexDigest();

    Sha256 truth_hash;
    truth_hash.Update("infinity-hnsw-d0-external-truth-v1");
    truth_hash.UpdateU64(query_count);
    truth_hash.UpdateU64(groundtruth_columns);
    for (std::size_t index = 0; index < query_count * groundtruth_columns; ++index) {
        truth_hash.UpdateU32(static_cast<std::uint32_t>(groundtruth[index]));
    }
    result.groundtruth_sha256 = truth_hash.HexDigest();

    std::array<std::size_t, 5> tolerant_hits{};
    std::array<std::size_t, 5> strict_hits{};

    for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
        const float *query = queries.data() + query_index * dimension;
        const std::int32_t *truth_row = groundtruth.data() + query_index * groundtruth_columns;

        // Distances to the first kK+1 published neighbours. Two jobs: the rank-(kK-1)
        // distance is the acceptance cutoff, and rank kK tells us whether a further row ties
        // with it. Verify the published order really is ascending rather than trusting it --
        // a descending or shuffled file would otherwise yield a quietly wrong cutoff.
        std::array<double, kK + 1> truth_distances{};
        for (std::size_t rank = 0; rank <= kK; ++rank) {
            const std::int32_t truth_id = truth_row[rank];
            Require(truth_id >= 0 && static_cast<std::size_t>(truth_id) < vector_count,
                    "external recall audit ground-truth id is out of range for the base");
            truth_distances[rank] = ExactSquaredL2(query, base + static_cast<std::size_t>(truth_id) * dimension, dimension);
            Require(rank == 0 || truth_distances[rank] >= truth_distances[rank - 1],
                    "external recall audit ground-truth row is not ascending by distance");
        }
        const double cutoff = truth_distances[kK - 1];
        if (truth_distances[kK] == cutoff) {
            ++result.queries_with_cutoff_ties;
        }

        for (std::size_t point = 0; point < tolerant_hits.size(); ++point) {
            const auto returned = search(query, kK, kHnswD0RecallEf[point]);
            Require(returned.size() == kK, "external recall audit search returned the wrong result count");
            for (const auto &[distance, label] : returned) {
                Require(std::isfinite(distance), "external recall audit search returned a non-finite distance");
                Require(label >= 0 && static_cast<std::size_t>(label) < vector_count,
                        "external recall audit search returned an out-of-range label");
                // Tie-tolerant: any row within the cutoff distance is an equally correct
                // answer, whether or not it is one of the kK the file happens to list.
                if (ExactSquaredL2(query, base + static_cast<std::size_t>(label) * dimension, dimension) <= cutoff) {
                    ++tolerant_hits[point];
                }
                // Strict: the label must be one of the kK listed IDs. Pessimistic, and the
                // figure most other harnesses report.
                if (std::find(truth_row, truth_row + kK, static_cast<std::int32_t>(label)) != truth_row + kK) {
                    ++strict_hits[point];
                }
            }
        }
    }

    const double denominator = static_cast<double>(query_count * kK);
    for (std::size_t point = 0; point < tolerant_hits.size(); ++point) {
        result.recall_at_10[point] = static_cast<double>(tolerant_hits[point]) / denominator;
        result.strict_id_recall_at_10[point] = static_cast<double>(strict_hits[point]) / denominator;
        // Every strictly-matched label is within the cutoff by construction, so the tolerant
        // count can never be the smaller of the two. If it is, the cutoff is wrong.
        Require(tolerant_hits[point] >= strict_hits[point],
                "external recall audit tie-tolerant recall fell below strict recall");
    }
    result.valid = true;
    return result;
}
