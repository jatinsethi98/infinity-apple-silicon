export module internal_types;

export import std;

export namespace infinity {

class RowID {
public:
    RowID() = default;
    RowID(std::uint32_t segment_id, std::uint32_t segment_offset)
        : value_((std::uint64_t{segment_id} << 32U) | segment_offset) {}

    friend bool operator==(const RowID &, const RowID &) = default;
    friend auto operator<=>(const RowID &, const RowID &) = default;

private:
    std::uint64_t value_{};
};

} // namespace infinity
