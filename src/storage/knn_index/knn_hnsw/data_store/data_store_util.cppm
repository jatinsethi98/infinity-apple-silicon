// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

export module infinity_core:data_store_util;

import :local_file_handle;

import std.compat;

namespace infinity {

export [[noreturn]] inline void HnswStreamError(std::string_view detail) {
    throw std::invalid_argument("Invalid HNSW stream: " + std::string(detail));
}

export [[noreturn]] inline void HnswPointerImageError(std::string_view detail) {
    throw std::invalid_argument("Invalid HNSW pointer image: " + std::string(detail));
}

export inline size_t HnswStreamCheckedAdd(size_t left, size_t right, std::string_view context) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        HnswStreamError(std::string(context) + " size addition overflows");
    }
    return left + right;
}

export inline size_t HnswStreamCheckedMultiply(size_t left, size_t right, std::string_view context) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        HnswStreamError(std::string(context) + " size multiplication overflows");
    }
    return left * right;
}

export inline size_t HnswCheckedAdd(size_t left, size_t right, std::string_view context) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        HnswPointerImageError(std::string(context) + " size addition overflows");
    }
    return left + right;
}

export inline size_t HnswCheckedMultiply(size_t left, size_t right, std::string_view context) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        HnswPointerImageError(std::string(context) + " size multiplication overflows");
    }
    return left * right;
}

export inline size_t HnswStreamRemaining(const LocalFileHandle &file_handle) {
    const auto remaining = file_handle.RemainingBytes();
    if constexpr (std::is_signed_v<decltype(remaining)>) {
        if (remaining < 0) {
            HnswStreamError("cannot determine the remaining file size");
        }
    }
    return static_cast<size_t>(remaining);
}

export inline void HnswEnsureStreamAvailable(const LocalFileHandle &file_handle, size_t size, std::string_view context) {
    if (size > HnswStreamRemaining(file_handle)) {
        HnswStreamError(std::string(context) + " is truncated");
    }
}

template <typename FileHandle>
void HnswReadExactImpl(FileHandle &file_handle, void *destination, size_t size, std::string_view context) {
    HnswEnsureStreamAvailable(file_handle, size, context);
    if (size == 0) {
        return;
    }
    if constexpr (std::is_void_v<decltype(file_handle.Read(destination, size))>) {
        file_handle.Read(destination, size);
    } else {
        auto [read_size, status] = file_handle.Read(destination, size);
        if (!status.ok() || read_size != size) {
            HnswStreamError(std::string(context) + " could not be read exactly");
        }
    }
}

export inline void HnswReadExact(LocalFileHandle &file_handle, void *destination, size_t size, std::string_view context) {
    HnswReadExactImpl(file_handle, destination, size, context);
}

export template <typename T>
T HnswReadStream(LocalFileHandle &file_handle, std::string_view context) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    HnswReadExact(file_handle, &value, sizeof(value), context);
    return value;
}

export inline void HnswRequireStreamEmpty(const LocalFileHandle &file_handle) {
    if (HnswStreamRemaining(file_handle) != 0) {
        HnswStreamError("stream contains trailing bytes");
    }
}

export template <typename T>
T HnswLoadUnaligned(const char *source) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

export template <typename T>
void HnswStoreUnaligned(char *destination, const T &value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(destination, &value, sizeof(value));
}

export class HnswPointerReader {
public:
    HnswPointerReader(const char *data, size_t size) : current_(data), remaining_(size) {
        if (data == nullptr && size != 0) {
            HnswPointerImageError("non-empty image has a null base pointer");
        }
    }

    template <typename T>
    T Read(std::string_view context) {
        static_assert(std::is_trivially_copyable_v<T>);
        T value;
        CopyTo(&value, sizeof(value), context);
        return value;
    }

    void EnsureAvailable(size_t size, std::string_view context) const {
        if (size > remaining_) {
            HnswPointerImageError(std::string(context) + " is truncated");
        }
    }

    const char *ReadBytes(size_t size, std::string_view context) {
        EnsureAvailable(size, context);
        const char *result = current_;
        if (size != 0) {
            current_ += size;
            remaining_ -= size;
        }
        return result;
    }

    void CopyTo(void *destination, size_t size, std::string_view context) {
        const char *source = ReadBytes(size, context);
        if (size != 0) {
            std::memcpy(destination, source, size);
        }
    }

    template <typename T>
    const T *ReadArray(size_t count, std::string_view context) {
        const size_t size = HnswCheckedMultiply(count, sizeof(T), context);
        const char *bytes = ReadBytes(size, context);
        if (size != 0 && reinterpret_cast<std::uintptr_t>(bytes) % alignof(T) != 0) {
            HnswPointerImageError(std::string(context) + " is misaligned");
        }
        return reinterpret_cast<const T *>(bytes);
    }

    const char *current() const { return current_; }
    size_t remaining() const { return remaining_; }

    void RequireEmpty() const {
        if (remaining_ != 0) {
            HnswPointerImageError("image contains trailing bytes");
        }
    }

private:
    const char *current_;
    size_t remaining_;
};

export template <typename T, bool OwnMem>
class ArrayPtr {
public:
    ArrayPtr() = default;
    ArrayPtr(std::unique_ptr<T[]> ptr) : ptr_(std::move(ptr)) {}

    T &operator[](size_t idx) { return ptr_[idx]; }
    const T &operator[](size_t idx) const { return ptr_[idx]; }

    T *get() const { return ptr_.get(); }

    std::unique_ptr<T[]> exchange(std::unique_ptr<T[]> ptr) { return std::exchange(ptr_, std::move(ptr)); }

private:
    std::unique_ptr<T[]> ptr_;
};

export template <typename T>
class ArrayPtr<T, false> {
public:
    ArrayPtr() = default;
    ArrayPtr(const T *ptr) : ptr_(ptr) {}

    const T &operator[](size_t idx) const { return ptr_[idx]; }

    const T *get() const { return ptr_; }

private:
    const T *ptr_ = nullptr;
};

export template <bool OwnMem>
class PPtr {
public:
    PPtr() = default;
    void set(char *ptr) { ptr_ = ptr; }
    char *get() const { return ptr_; }

private:
    char *ptr_;
};

export template <>
class PPtr<false> {
public:
    PPtr() = default;
    void set(const char *ptr) { ptr_ = ptr; }
    const char *get() const { return ptr_; }

private:
    const char *ptr_ = nullptr;
};

} // namespace infinity
