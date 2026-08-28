export module infinity_core:local_file_handle;

import :infinity_exception;
import std;

export namespace infinity {

class LocalFileHandle {
public:
    void Append(const void *data, size_t size) {
        const auto *bytes = static_cast<const char *>(data);
        bytes_.insert(bytes_.end(), bytes, bytes + size);
    }

    void Read(void *data, size_t size) {
        if (size > bytes_.size() - read_offset_) {
            UnrecoverableError("native HNSW smoke persistence read exceeds the buffer");
        }
        std::memcpy(data, bytes_.data() + read_offset_, size);
        read_offset_ += size;
    }

    void Rewind() { read_offset_ = 0; }

    size_t Size() const { return bytes_.size(); }
    size_t RemainingBytes() const { return bytes_.size() - read_offset_; }

private:
    std::vector<char> bytes_;
    size_t read_offset_{};
};

} // namespace infinity
