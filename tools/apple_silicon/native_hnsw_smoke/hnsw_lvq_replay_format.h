#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace hnsw_lvq_replay {

inline constexpr std::size_t kTapeHeaderBytes = 256;
inline constexpr std::size_t kOperandHeaderBytes = 32;
inline constexpr std::size_t kEventBytes = 48;
inline constexpr std::uint32_t kTapeVersion = 1;
inline constexpr std::uint32_t kEndianTag = 0x01020304U;
inline constexpr std::uint32_t kProductionOrigin = 1;
inline constexpr std::uint32_t kCanonicalSingleWorkerFlag = 1;
inline constexpr std::uint32_t kQueryOperandRole = 1;
inline constexpr std::uint32_t kCandidateOperandRole = 2;

struct Operand {
    std::uint64_t id{};
    std::uint64_t captured_address{};
    std::uint64_t first_event_ordinal{};
    std::uint32_t role_mask{};
    std::vector<std::byte> record;
};

struct Event {
    std::uint64_t ordinal{};
    std::uint64_t query_operand_id{};
    std::uint64_t candidate_operand_id{};
    std::int32_t query_vertex{};
    std::int32_t candidate_vertex{};
    std::int32_t layer{};
    std::uint32_t phase{};
    std::uint32_t distance_bits{};
    std::uint32_t flags{};
};

struct Tape {
    std::uint32_t dimension{};
    std::uint32_t record_bytes{};
    std::uint32_t worker_count{1};
    std::uint64_t observed_call_count{};
    std::uint64_t capture_limit{};
    std::array<std::byte, 32> dataset_sha256{};
    std::array<std::byte, 32> capture_request_sha256{};
    std::array<std::byte, 32> graph_sha256{};
    std::vector<Operand> operands;
    std::vector<Event> events;
};

std::array<std::byte, 32> ParseSha256Hex(const std::string &text);
std::string Sha256Hex(const std::array<std::byte, 32> &digest);
std::size_t OperandStride(std::size_t record_bytes);

void ValidateTape(const Tape &tape);
void WriteTape(const std::filesystem::path &path, const Tape &tape);
Tape ReadTape(const std::filesystem::path &path);

} // namespace hnsw_lvq_replay
