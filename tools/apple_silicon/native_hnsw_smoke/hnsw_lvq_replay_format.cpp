#include "hnsw_lvq_replay_format.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace hnsw_lvq_replay {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'I'},
    std::byte{'F'},
    std::byte{'L'},
    std::byte{'V'},
    std::byte{'Q'},
    std::byte{'T'},
    std::byte{'0'},
    std::byte{'1'},
};

[[noreturn]] void Fail(std::string_view message) {
    throw std::runtime_error("Invalid HNSW LVQ replay tape: " + std::string(message));
}

std::size_t CheckedAdd(std::size_t left, std::size_t right, std::string_view context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        Fail(std::string(context) + " overflows");
    }
    return left + right;
}

std::size_t CheckedMultiply(std::size_t left, std::size_t right, std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        Fail(std::string(context) + " overflows");
    }
    return left * right;
}

void StoreU32(std::vector<std::byte> &bytes, std::size_t offset, std::uint32_t value) {
    if (offset > bytes.size() || bytes.size() - offset < 4) {
        Fail("internal u32 write is out of range");
    }
    for (std::size_t index = 0; index < 4; ++index) {
        bytes[offset + index] = static_cast<std::byte>(value >> (8U * index));
    }
}

void StoreI32(std::vector<std::byte> &bytes, std::size_t offset, std::int32_t value) {
    StoreU32(bytes, offset, std::bit_cast<std::uint32_t>(value));
}

void StoreU64(std::vector<std::byte> &bytes, std::size_t offset, std::uint64_t value) {
    if (offset > bytes.size() || bytes.size() - offset < 8) {
        Fail("internal u64 write is out of range");
    }
    for (std::size_t index = 0; index < 8; ++index) {
        bytes[offset + index] = static_cast<std::byte>(value >> (8U * index));
    }
}

std::uint32_t LoadU32(const std::vector<std::byte> &bytes, std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4) {
        Fail("u32 read is out of range");
    }
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        value |= std::to_integer<std::uint32_t>(bytes[offset + index]) << (8U * index);
    }
    return value;
}

std::int32_t LoadI32(const std::vector<std::byte> &bytes, std::size_t offset) {
    return std::bit_cast<std::int32_t>(LoadU32(bytes, offset));
}

std::uint64_t LoadU64(const std::vector<std::byte> &bytes, std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 8) {
        Fail("u64 read is out of range");
    }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value |= std::to_integer<std::uint64_t>(bytes[offset + index]) << (8U * index);
    }
    return value;
}

std::size_t ToSize(std::uint64_t value, std::string_view context) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        Fail(std::string(context) + " exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

void RequireZero(const std::vector<std::byte> &bytes, std::size_t begin, std::size_t end, std::string_view context) {
    if (begin > end || end > bytes.size()) {
        Fail(std::string(context) + " range is invalid");
    }
    if (std::any_of(bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                    bytes.begin() + static_cast<std::ptrdiff_t>(end),
                    [](std::byte value) { return value != std::byte{}; })) {
        Fail(std::string(context) + " is nonzero");
    }
}

void RequireDigest(const std::array<std::byte, 32> &digest, std::string_view context) {
    if (std::all_of(digest.begin(), digest.end(), [](std::byte value) { return value == std::byte{}; })) {
        Fail(std::string(context) + " is all zero");
    }
}

std::vector<std::byte> ReadAll(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Could not open HNSW LVQ replay tape: " + path.string());
    }
    const std::streamoff end = input.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) > std::numeric_limits<std::size_t>::max()) {
        Fail("file size is invalid");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!input || input.peek() != std::char_traits<char>::eof()) {
        Fail("file read was incomplete");
    }
    return bytes;
}

} // namespace

std::array<std::byte, 32> ParseSha256Hex(const std::string &text) {
    if (text.size() != 64) {
        throw std::invalid_argument("SHA-256 text must contain 64 lowercase hexadecimal characters");
    }
    auto hex_value = [](char value) -> unsigned int {
        if (value >= '0' && value <= '9') {
            return static_cast<unsigned int>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<unsigned int>(value - 'a' + 10);
        }
        throw std::invalid_argument("SHA-256 text must contain 64 lowercase hexadecimal characters");
    };
    std::array<std::byte, 32> digest{};
    for (std::size_t index = 0; index < digest.size(); ++index) {
        digest[index] = static_cast<std::byte>((hex_value(text[2 * index]) << 4U) | hex_value(text[2 * index + 1]));
    }
    return digest;
}

std::string Sha256Hex(const std::array<std::byte, 32> &digest) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (std::byte value : digest) {
        const unsigned int byte = std::to_integer<unsigned int>(value);
        result.push_back(kHex[byte >> 4U]);
        result.push_back(kHex[byte & 0x0fU]);
    }
    return result;
}

std::size_t OperandStride(std::size_t record_bytes) {
    const std::size_t unaligned = CheckedAdd(kOperandHeaderBytes, record_bytes, "operand stride");
    return CheckedAdd(unaligned, 15, "operand alignment") & ~std::size_t{15};
}

void ValidateTape(const Tape &tape) {
    if (tape.dimension == 0 || tape.dimension > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        Fail("dimension is out of range");
    }
    if (tape.record_bytes != tape.dimension + 16U) {
        Fail("record size is not 16 + dimension");
    }
    if (tape.worker_count != 1) {
        Fail("canonical capture must use exactly one worker");
    }
    if (tape.events.empty() || tape.events.size() != tape.capture_limit) {
        Fail("event count does not equal the nonzero capture limit");
    }
    if (tape.observed_call_count < tape.events.size()) {
        Fail("observed call count is smaller than the event count");
    }
    if (tape.operands.empty()) {
        Fail("operand table is empty");
    }
    RequireDigest(tape.dataset_sha256, "dataset SHA-256");
    RequireDigest(tape.capture_request_sha256, "capture request SHA-256");
    RequireDigest(tape.graph_sha256, "graph SHA-256");

    std::vector<std::uint32_t> roles(tape.operands.size(), 0);
    std::vector<std::uint64_t> first_events(tape.operands.size(), std::numeric_limits<std::uint64_t>::max());
    for (std::size_t index = 0; index < tape.operands.size(); ++index) {
        const Operand &operand = tape.operands[index];
        if (operand.id != index) {
            Fail("operand IDs are not contiguous");
        }
        if (operand.captured_address == 0) {
            Fail("operand captured address is zero");
        }
        if (operand.record.size() != tape.record_bytes) {
            Fail("operand record size differs from the header");
        }
        if (operand.role_mask == 0 || (operand.role_mask & ~(kQueryOperandRole | kCandidateOperandRole)) != 0) {
            Fail("operand role mask is invalid");
        }
    }

    for (std::size_t index = 0; index < tape.events.size(); ++index) {
        const Event &event = tape.events[index];
        if (event.ordinal != index) {
            Fail("event ordinals are not contiguous");
        }
        if (event.query_operand_id >= tape.operands.size() || event.candidate_operand_id >= tape.operands.size()) {
            Fail("event operand ID is out of range");
        }
        if (event.candidate_vertex < 0) {
            Fail("candidate vertex is invalid");
        }
        if (event.layer < 0 || event.layer > 64) {
            Fail("event layer is out of range");
        }
        if (event.phase < 1 || event.phase > 5) {
            Fail("event phase is unknown");
        }
        if (event.flags != 0) {
            Fail("event flags are nonzero");
        }
        if (!std::isfinite(std::bit_cast<float>(event.distance_bits))) {
            Fail("event distance is not finite");
        }
        const std::size_t query = static_cast<std::size_t>(event.query_operand_id);
        const std::size_t candidate = static_cast<std::size_t>(event.candidate_operand_id);
        roles[query] |= kQueryOperandRole;
        roles[candidate] |= kCandidateOperandRole;
        first_events[query] = std::min(first_events[query], event.ordinal);
        first_events[candidate] = std::min(first_events[candidate], event.ordinal);
    }

    for (std::size_t index = 0; index < tape.operands.size(); ++index) {
        if (tape.operands[index].role_mask != roles[index]) {
            Fail("operand role mask does not reconcile with events");
        }
        if (tape.operands[index].first_event_ordinal != first_events[index]) {
            Fail("operand first-event ordinal does not reconcile with events");
        }
    }
}

void WriteTape(const std::filesystem::path &path, const Tape &tape) {
    ValidateTape(tape);
    const std::size_t operand_stride = OperandStride(tape.record_bytes);
    const std::size_t operand_section_bytes = CheckedMultiply(tape.operands.size(), operand_stride, "operand section");
    const std::size_t event_section_bytes = CheckedMultiply(tape.events.size(), kEventBytes, "event section");
    const std::size_t event_offset = CheckedAdd(kTapeHeaderBytes, operand_section_bytes, "event offset");
    const std::size_t total_bytes = CheckedAdd(event_offset, event_section_bytes, "tape size");
    std::vector<std::byte> bytes(total_bytes);

    std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
    StoreU32(bytes, 8, kTapeVersion);
    StoreU32(bytes, 12, kTapeHeaderBytes);
    StoreU32(bytes, 16, kEndianTag);
    StoreU32(bytes, 20, tape.dimension);
    StoreU32(bytes, 24, tape.record_bytes);
    StoreU32(bytes, 28, tape.worker_count);
    StoreU32(bytes, 32, kProductionOrigin);
    StoreU32(bytes, 36, kEventBytes);
    StoreU32(bytes, 40, kOperandHeaderBytes);
    StoreU32(bytes, 44, static_cast<std::uint32_t>(operand_stride));
    StoreU32(bytes, 48, kCanonicalSingleWorkerFlag);
    StoreU64(bytes, 56, tape.operands.size());
    StoreU64(bytes, 64, tape.events.size());
    StoreU64(bytes, 72, kTapeHeaderBytes);
    StoreU64(bytes, 80, operand_section_bytes);
    StoreU64(bytes, 88, event_offset);
    StoreU64(bytes, 96, event_section_bytes);
    StoreU64(bytes, 104, tape.observed_call_count);
    StoreU64(bytes, 112, tape.capture_limit);
    std::copy(tape.dataset_sha256.begin(), tape.dataset_sha256.end(), bytes.begin() + 120);
    std::copy(tape.capture_request_sha256.begin(), tape.capture_request_sha256.end(), bytes.begin() + 152);
    std::copy(tape.graph_sha256.begin(), tape.graph_sha256.end(), bytes.begin() + 184);

    for (std::size_t index = 0; index < tape.operands.size(); ++index) {
        const Operand &operand = tape.operands[index];
        const std::size_t offset = kTapeHeaderBytes + index * operand_stride;
        StoreU64(bytes, offset, operand.id);
        StoreU64(bytes, offset + 8, operand.captured_address);
        StoreU64(bytes, offset + 16, operand.first_event_ordinal);
        StoreU32(bytes, offset + 24, operand.role_mask);
        StoreU32(bytes, offset + 28, tape.record_bytes);
        std::copy(operand.record.begin(), operand.record.end(), bytes.begin() + static_cast<std::ptrdiff_t>(offset + kOperandHeaderBytes));
    }

    for (std::size_t index = 0; index < tape.events.size(); ++index) {
        const Event &event = tape.events[index];
        const std::size_t offset = event_offset + index * kEventBytes;
        StoreU64(bytes, offset, event.ordinal);
        StoreU64(bytes, offset + 8, event.query_operand_id);
        StoreU64(bytes, offset + 16, event.candidate_operand_id);
        StoreI32(bytes, offset + 24, event.query_vertex);
        StoreI32(bytes, offset + 28, event.candidate_vertex);
        StoreI32(bytes, offset + 32, event.layer);
        StoreU32(bytes, offset + 36, event.phase);
        StoreU32(bytes, offset + 40, event.distance_bits);
        StoreU32(bytes, offset + 44, event.flags);
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not create HNSW LVQ replay tape: " + path.string());
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) {
        throw std::runtime_error("Could not write HNSW LVQ replay tape: " + path.string());
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Could not close HNSW LVQ replay tape: " + path.string());
    }
}

Tape ReadTape(const std::filesystem::path &path) {
    const std::vector<std::byte> bytes = ReadAll(path);
    if (bytes.size() < kTapeHeaderBytes) {
        Fail("file is smaller than its header");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        Fail("magic does not match");
    }
    if (LoadU32(bytes, 8) != kTapeVersion || LoadU32(bytes, 12) != kTapeHeaderBytes || LoadU32(bytes, 16) != kEndianTag) {
        Fail("version, header size, or endian tag does not match");
    }
    const std::uint32_t dimension = LoadU32(bytes, 20);
    const std::uint32_t record_bytes = LoadU32(bytes, 24);
    const std::uint32_t worker_count = LoadU32(bytes, 28);
    if (LoadU32(bytes, 32) != kProductionOrigin || LoadU32(bytes, 36) != kEventBytes ||
        LoadU32(bytes, 40) != kOperandHeaderBytes || LoadU32(bytes, 48) != kCanonicalSingleWorkerFlag) {
        Fail("canonical format constants do not match");
    }
    RequireZero(bytes, 52, 56, "header reserved field");
    RequireZero(bytes, 216, kTapeHeaderBytes, "header reserved tail");

    const std::size_t operand_count = ToSize(LoadU64(bytes, 56), "operand count");
    const std::size_t event_count = ToSize(LoadU64(bytes, 64), "event count");
    const std::size_t operand_offset = ToSize(LoadU64(bytes, 72), "operand offset");
    const std::size_t operand_section_bytes = ToSize(LoadU64(bytes, 80), "operand section size");
    const std::size_t event_offset = ToSize(LoadU64(bytes, 88), "event offset");
    const std::size_t event_section_bytes = ToSize(LoadU64(bytes, 96), "event section size");
    const std::size_t operand_stride = OperandStride(record_bytes);
    if (LoadU32(bytes, 44) != operand_stride || operand_offset != kTapeHeaderBytes ||
        operand_section_bytes != CheckedMultiply(operand_count, operand_stride, "operand section") ||
        event_offset != CheckedAdd(operand_offset, operand_section_bytes, "event offset") ||
        event_section_bytes != CheckedMultiply(event_count, kEventBytes, "event section") ||
        bytes.size() != CheckedAdd(event_offset, event_section_bytes, "tape size")) {
        Fail("section layout is not canonical");
    }

    Tape tape;
    tape.dimension = dimension;
    tape.record_bytes = record_bytes;
    tape.worker_count = worker_count;
    tape.observed_call_count = LoadU64(bytes, 104);
    tape.capture_limit = LoadU64(bytes, 112);
    std::copy_n(bytes.begin() + 120, 32, tape.dataset_sha256.begin());
    std::copy_n(bytes.begin() + 152, 32, tape.capture_request_sha256.begin());
    std::copy_n(bytes.begin() + 184, 32, tape.graph_sha256.begin());

    tape.operands.reserve(operand_count);
    for (std::size_t index = 0; index < operand_count; ++index) {
        const std::size_t offset = operand_offset + index * operand_stride;
        if (LoadU32(bytes, offset + 28) != record_bytes) {
            Fail("operand record size does not match");
        }
        Operand operand{
            .id = LoadU64(bytes, offset),
            .captured_address = LoadU64(bytes, offset + 8),
            .first_event_ordinal = LoadU64(bytes, offset + 16),
            .role_mask = LoadU32(bytes, offset + 24),
            .record = std::vector<std::byte>(record_bytes),
        };
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset + kOperandHeaderBytes),
                    record_bytes,
                    operand.record.begin());
        RequireZero(bytes,
                    offset + kOperandHeaderBytes + record_bytes,
                    offset + operand_stride,
                    "operand padding");
        tape.operands.push_back(std::move(operand));
    }

    tape.events.reserve(event_count);
    for (std::size_t index = 0; index < event_count; ++index) {
        const std::size_t offset = event_offset + index * kEventBytes;
        tape.events.push_back(Event{
            .ordinal = LoadU64(bytes, offset),
            .query_operand_id = LoadU64(bytes, offset + 8),
            .candidate_operand_id = LoadU64(bytes, offset + 16),
            .query_vertex = LoadI32(bytes, offset + 24),
            .candidate_vertex = LoadI32(bytes, offset + 28),
            .layer = LoadI32(bytes, offset + 32),
            .phase = LoadU32(bytes, offset + 36),
            .distance_bits = LoadU32(bytes, offset + 40),
            .flags = LoadU32(bytes, offset + 44),
        });
    }
    ValidateTape(tape);
    return tape;
}

} // namespace hnsw_lvq_replay
