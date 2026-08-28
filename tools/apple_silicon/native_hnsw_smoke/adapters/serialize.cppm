export module serialize;

export import std;

export namespace infinity {

template <typename T>
T ReadBufAdv(const char *&ptr) {
    T value;
    std::memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

} // namespace infinity
