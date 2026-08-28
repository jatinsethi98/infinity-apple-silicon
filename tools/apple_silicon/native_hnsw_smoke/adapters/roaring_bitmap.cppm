export module infinity_core:roaring_bitmap;

export import std;

export namespace infinity {

class Bitmask {
public:
    bool IsTrue(std::size_t) const { return true; }
};

} // namespace infinity
