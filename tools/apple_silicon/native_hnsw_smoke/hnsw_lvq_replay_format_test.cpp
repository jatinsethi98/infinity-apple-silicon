#include "hnsw_lvq_replay_format.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using hnsw_lvq_replay::Event;
using hnsw_lvq_replay::Operand;
using hnsw_lvq_replay::Tape;

std::filesystem::path TestPath(const char *name) {
    return std::filesystem::temp_directory_path() / ("infinity-hnsw-lvq-format-" + std::to_string(::getpid()) + "-" + name);
}

void WriteBytes(const std::filesystem::path &path, const std::vector<std::byte> &bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("test could not write mutated tape");
    }
}

std::vector<std::byte> ReadBytes(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("test could not size tape");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!input) {
        throw std::runtime_error("test could not read tape");
    }
    return bytes;
}

template <typename Mutate>
void RequireRejected(const std::filesystem::path &valid_path, const char *name, Mutate mutate) {
    std::vector<std::byte> bytes = ReadBytes(valid_path);
    mutate(bytes);
    const std::filesystem::path path = TestPath(name);
    WriteBytes(path, bytes);
    bool rejected = false;
    try {
        static_cast<void>(hnsw_lvq_replay::ReadTape(path));
    } catch (const std::exception &) {
        rejected = true;
    }
    std::filesystem::remove(path);
    if (!rejected) {
        throw std::runtime_error(std::string("malformed tape was accepted: ") + name);
    }
}

Tape MakeTape() {
    Tape tape;
    tape.dimension = 4;
    tape.record_bytes = 20;
    tape.worker_count = 1;
    tape.observed_call_count = 3;
    tape.capture_limit = 1;
    tape.dataset_sha256.fill(std::byte{0x11});
    tape.capture_request_sha256.fill(std::byte{0x22});
    tape.graph_sha256.fill(std::byte{0x33});
    tape.operands = {
        Operand{
            .id = 0,
            .captured_address = 0x1000,
            .first_event_ordinal = 0,
            .role_mask = hnsw_lvq_replay::kQueryOperandRole,
            .record = std::vector<std::byte>(20, std::byte{0x44}),
        },
        Operand{
            .id = 1,
            .captured_address = 0x2000,
            .first_event_ordinal = 0,
            .role_mask = hnsw_lvq_replay::kCandidateOperandRole,
            .record = std::vector<std::byte>(20, std::byte{0x55}),
        },
    };
    tape.events = {
        Event{
            .ordinal = 0,
            .query_operand_id = 0,
            .candidate_operand_id = 1,
            .query_vertex = 7,
            .candidate_vertex = 3,
            .layer = 0,
            .phase = 2,
            .distance_bits = std::bit_cast<std::uint32_t>(1.25F),
            .flags = 0,
        },
    };
    return tape;
}

} // namespace

int main() {
    try {
        const std::filesystem::path valid_path = TestPath("valid.bin");
        const Tape expected = MakeTape();
        hnsw_lvq_replay::WriteTape(valid_path, expected);
        const Tape actual = hnsw_lvq_replay::ReadTape(valid_path);
        if (actual.dimension != expected.dimension || actual.record_bytes != expected.record_bytes ||
            actual.observed_call_count != expected.observed_call_count || actual.operands.size() != expected.operands.size() ||
            actual.events.size() != expected.events.size() || actual.events.front().distance_bits != expected.events.front().distance_bits) {
            throw std::runtime_error("round-trip tape differs");
        }

        RequireRejected(valid_path, "magic.bin", [](auto &bytes) { bytes[0] ^= std::byte{1}; });
        RequireRejected(valid_path, "reserved.bin", [](auto &bytes) { bytes[216] = std::byte{1}; });
        RequireRejected(valid_path, "truncate.bin", [](auto &bytes) { bytes.pop_back(); });
        RequireRejected(valid_path, "append.bin", [](auto &bytes) { bytes.push_back(std::byte{}); });
        RequireRejected(valid_path, "phase.bin", [](auto &bytes) {
            const std::size_t event_offset = hnsw_lvq_replay::kTapeHeaderBytes +
                                             2 * hnsw_lvq_replay::OperandStride(20);
            bytes[event_offset + 36] = std::byte{};
            bytes[event_offset + 37] = std::byte{};
            bytes[event_offset + 38] = std::byte{};
            bytes[event_offset + 39] = std::byte{};
        });

        std::filesystem::remove(valid_path);
        std::cout << "status=PASS\n";
        std::cout << "schema=hnsw-lvq-replay-tape-v1\n";
        std::cout << "tamper_cases=5\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HNSW LVQ replay format test failed: " << error.what() << '\n';
        return 1;
    }
}
