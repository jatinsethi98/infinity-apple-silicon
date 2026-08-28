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

module;

#include <cassert>
#include <cerrno>
#include <limits>

module infinity_core:persistence_manager.impl;

import :persistence_manager;
import :uuid;
import :infinity_exception;
import :virtual_store;
import :logger;
import :kv_store;
import :kv_code;
import :infinity_context;
import :object_stats;

import std.compat;
import third_party;

import serialize;
import global_resource_usage;

namespace fs = std::filesystem;

namespace infinity {
constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1 MB

namespace {

size_t CheckedAdd(size_t left, size_t right, std::string_view operation) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        UnrecoverableError(fmt::format("{} size overflow: {} + {}", operation, left, right));
    }
    return left + right;
}

size_t CheckedAlignUp(size_t value, size_t alignment, std::string_view operation) {
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0);
    return CheckedAdd(value, alignment - 1, operation) & ~(alignment - 1);
}

} // namespace

nlohmann::json ObjAddr::Serialize() const {
    nlohmann::json obj;
    obj["obj_key"] = obj_key_;
    obj["part_offset"] = part_offset_;
    obj["part_size"] = part_size_;
    return obj;
}

void ObjAddr::Deserialize(std::string_view obj_str) {
    simdjson::padded_string obj_json(obj_str);
    simdjson::parser parser;
    simdjson::document doc = parser.iterate(obj_json);
    obj_key_ = doc["obj_key"].get<std::string_view>().value();
    part_offset_ = doc["part_offset"].get<size_t>();
    part_size_ = doc["part_size"].get<size_t>();
}

// size_t ObjAddr::GetSizeInBytes() const { return sizeof(int32_t) + obj_key_.size() + sizeof(size_t) + sizeof(size_t); }
//
// void ObjAddr::WriteBufAdv(char *&buf) const {
//     ::infinity::WriteBufAdv(buf, obj_key_);
//     ::infinity::WriteBufAdv(buf, part_offset_);
//     ::infinity::WriteBufAdv(buf, part_size_);
// }
//
// ObjAddr ObjAddr::ReadBufAdv(const char *&buf) {
//     ObjAddr ret;
//     ret.obj_key_ = ::infinity::ReadBufAdv<std::string>(buf);
//     ret.part_offset_ = ::infinity::ReadBufAdv<size_t>(buf);
//     ret.part_size_ = ::infinity::ReadBufAdv<size_t>(buf);
//     return ret;
// }

PersistenceManager::PersistenceManager(Storage *storage,
                                       const std::string &workspace,
                                       const std::string &data_dir,
                                       size_t object_size_limit,
                                       bool local_storage)
    : storage_(storage), workspace_(workspace), local_data_dir_(data_dir), object_size_limit_(object_size_limit) {
    if (local_storage) {
        object_stats_ = std::make_shared<ObjectStats>(storage);
    } else {
        UnrecoverableError("Remote storage is not supported yet.");
    }
    current_object_key_ = ObjCreate();
    current_object_size_ = 0;
    current_object_parts_ = 0;
    current_object_ref_count_ = 0;

    if (local_data_dir_.empty() || local_data_dir_.back() != '/') {
        local_data_dir_ += '/';
    }

    if (!VirtualStore::Exists(workspace_)) {
        VirtualStore::MakeDirectory(workspace_);
    }
    std::string read_path_empty = GetObjPath(ObjAddr::KeyEmpty);
    VirtualStore::Truncate(read_path_empty, 0);
#ifdef INFINITY_DEBUG
    GlobalResourceUsage::IncrObjectCount("PersistenceManager");
#endif
}

PersistenceManager::~PersistenceManager() {
#ifdef INFINITY_DEBUG
    GlobalResourceUsage::DecrObjectCount("PersistenceManager");
#endif
}

PersistWriteResult PersistenceManager::Persist(const std::string &file_path, const std::string &tmp_file_path, bool try_compose) {
    PersistWriteResult result;
    Status status;

    std::error_code ec;
    fs::path src_fp = tmp_file_path;

    std::string local_path = RemovePrefix(file_path);
    if (local_path.empty()) {
        UnrecoverableError(fmt::format("Failed to find local path of {}", local_path));
    }
    std::string pm_fp_key = KeyEncode::PMObjectKey(local_path);

    // Check if the object for local path exists in the KV.
    // If it exists, we will delete the object later after we persist the local path to new object.
    std::string pm_fp_value{};
    kv_store_->Get(pm_fp_key, pm_fp_value);

    size_t src_size = fs::file_size(src_fp, ec);
    if (ec) {
        UnrecoverableError(fmt::format("Failed to get file size of {}", file_path));
    }
    if (src_size == 0) {
        LOG_WARN(fmt::format("Persist empty local path {}", file_path));
        ObjAddr obj_addr(ObjAddr::KeyEmpty, 0, 0);
        fs::remove(tmp_file_path, ec); // This may cause the issue
        if (ec) {
            UnrecoverableError(fmt::format("Failed to remove {}", tmp_file_path));
        }
        status = kv_store_->Put(pm_fp_key, obj_addr.Serialize().dump(), false);
        if (!status.ok()) {
            UnrecoverableError(status.message());
        }

        result.persist_keys_.push_back(ObjAddr::KeyEmpty);
        result.obj_addr_ = obj_addr;
    } else if (!try_compose || src_size >= object_size_limit_) {
        std::string obj_key = ObjCreate();
        fs::path dst_fp = workspace_;
        dst_fp.append(obj_key);
        fs::rename(src_fp, dst_fp, ec);
        if (ec) {
            UnrecoverableError(fmt::format("Failed to rename {} to {}", src_fp.string(), dst_fp.string()));
        }
        ObjAddr obj_addr(obj_key, 0, src_size);
        std::lock_guard<std::mutex> lock(mtx_);
        object_stats_->PutNew(obj_key, std::make_shared<ObjStat>(src_size, 1, 0));
        LOG_TRACE(fmt::format("Persist added dedicated object {}", obj_key));

        status = kv_store_->Put(pm_fp_key, obj_addr.Serialize().dump(), false);
        if (!status.ok()) {
            UnrecoverableError(status.message());
        }

        LOG_TRACE(fmt::format("Persist local path {} to dedicated ObjAddr ({}, {}, {})",
                              local_path,
                              obj_addr.obj_key_,
                              obj_addr.part_offset_,
                              obj_addr.part_size_));
        result.persist_keys_.push_back(obj_key);
        result.obj_addr_ = obj_addr;
    } else {
        std::lock_guard<std::mutex> lock(mtx_);
        if (src_size >= CurrentObjRoomNoLock()) {
            CurrentObjFinalizeNoLock(result.persist_keys_);
        }
        const size_t part_offset = CurrentObjAppendNoLock(tmp_file_path, src_size);
        ObjAddr obj_addr(current_object_key_, part_offset, src_size);
        fs::remove(tmp_file_path, ec);
        if (ec) {
            UnrecoverableError(fmt::format("Failed to remove {}", tmp_file_path));
        }

        object_stats_->PutNew(current_object_key_, std::make_shared<ObjStat>(current_object_size_, current_object_parts_, current_object_ref_count_));
        LOG_TRACE(fmt::format("Persist current object {}", current_object_key_));

        status = kv_store_->Put(pm_fp_key, obj_addr.Serialize().dump(), false);
        if (!status.ok()) {
            UnrecoverableError(status.message());
        }

        LOG_TRACE(fmt::format("Persist local path {} to composed ObjAddr ({}, {}, {})",
                              local_path,
                              obj_addr.obj_key_,
                              obj_addr.part_offset_,
                              obj_addr.part_size_));
        result.obj_addr_ = obj_addr;
    }

    //  Cleanup the old object for local path.
    if (!pm_fp_value.empty()) {
        ObjAddr obj_addr;
        obj_addr.Deserialize(pm_fp_value);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            CleanupNoLock(obj_addr, result.persist_keys_, result.drop_from_remote_keys_);
        }
        LOG_TRACE(fmt::format("Persist deleted mapping from local path {} to ObjAddr({}, {}, {})",
                              local_path,
                              obj_addr.obj_key_,
                              obj_addr.part_offset_,
                              obj_addr.part_size_));
    }

    return result;
}

// TODO:
// - Upload the finalized object to object store in background.
PersistWriteResult PersistenceManager::CurrentObjFinalize(bool validate) {
    PersistWriteResult result;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        CurrentObjFinalizeNoLock(result.persist_keys_);
    }

    if (validate) {
        CheckValid();
    }

    return result;
}

void PersistenceManager::CheckValid() {
    using TimeDurationType = std::chrono::duration<float, std::milli>;
    const auto part1_begin = std::chrono::high_resolution_clock::now();
    std::string pm_object_prefix = KeyEncode::PMObjectPrefix();
    size_t prefix_len = pm_object_prefix.size();
    std::unique_ptr<KVInstance> kv_instance = kv_store_->GetInstance();
    std::unique_ptr<KVIterator> iter = kv_instance->GetIterator();
    iter->Seek(pm_object_prefix);
    for (; iter->Valid() && iter->Key().starts_with(pm_object_prefix); iter->Next()) {
        std::string local_path = iter->Key().ToString().substr(prefix_len);
        ObjAddr obj_addr;
        obj_addr.Deserialize(iter->Value().ToString());
        if (obj_addr.obj_key_ == ObjAddr::KeyEmpty) {
            continue;
        }
        std::shared_ptr<ObjStat> obj_stat = object_stats_->GetNoCount(obj_addr.obj_key_);
        if (obj_stat == nullptr) {
            LOG_ERROR(fmt::format("CheckValid Failed to find object for local path {}", local_path));
        }
    }
    const auto part2_begin = std::chrono::high_resolution_clock::now();

    // Protect access to current_object_size_ with mutex
    size_t current_size;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        current_size = current_object_size_;
    }
    object_stats_->CheckValid(current_size);

    const auto part2_end = std::chrono::high_resolution_clock::now();
    LOG_INFO(fmt::format("PersistenceManager::CheckValid part 1: {} ms, part2: {} ms",
                         static_cast<TimeDurationType>(part2_begin - part1_begin).count(),
                         static_cast<TimeDurationType>(part2_end - part2_begin).count()));
}

void PersistenceManager::CurrentObjFinalizeNoLock(std::vector<std::string> &persist_keys) {
    if (current_object_size_ > 0) {
        if (current_object_parts_ > 1) {
            // Add footer to composed object -- format version 1
            fs::path dst_fp = workspace_;
            dst_fp.append(current_object_key_);
            std::error_code ec;
            const size_t original_size = fs::file_size(dst_fp, ec);
            if (ec) {
                UnrecoverableError(fmt::format("Failed to get size of composed object {}: {}", dst_fp.string(), ec.message()));
            }
            if (original_size != current_object_size_) {
                UnrecoverableError(fmt::format("Composed object {} physical size {} differs from logical size {} before finalization",
                                               dst_fp.string(),
                                               original_size,
                                               current_object_size_));
            }

            try {
                std::ofstream out_file(dst_fp, std::ios::binary | std::ios::app);
                if (!out_file.is_open()) {
                    throw std::runtime_error(fmt::format("Failed to open {}", dst_fp.string()));
                }
                const u32 compose_format = 1;
                out_file.write(reinterpret_cast<const char *>(&compose_format), sizeof(compose_format));
                if (!out_file.good()) {
                    throw std::runtime_error(fmt::format("Failed to write footer to {}", dst_fp.string()));
                }
                out_file.flush();
                if (!out_file.good()) {
                    throw std::runtime_error(fmt::format("Failed to flush footer to {}", dst_fp.string()));
                }
                out_file.close();
                if (out_file.fail()) {
                    throw std::runtime_error(fmt::format("Failed to close {}", dst_fp.string()));
                }
                const size_t expected_size = CheckedAdd(original_size, sizeof(compose_format), "Finalize composed object");
                const size_t actual_size = fs::file_size(dst_fp, ec);
                if (ec || actual_size != expected_size) {
                    throw std::runtime_error(
                        fmt::format("Finalized object {} size mismatch: expected {}, got {} ({})",
                                    dst_fp.string(),
                                    expected_size,
                                    actual_size,
                                    ec ? ec.message() : "no filesystem error"));
                }
            } catch (const std::exception &error) {
                std::error_code rollback_ec;
                fs::resize_file(dst_fp, original_size, rollback_ec);
                if (rollback_ec) {
                    UnrecoverableError(fmt::format("Failed to finalize composed object {}: {}; rollback to {} bytes also failed: {}",
                                                   dst_fp.string(),
                                                   error.what(),
                                                   original_size,
                                                   rollback_ec.message()));
                }
                UnrecoverableError(fmt::format("Failed to finalize composed object {}: {}", dst_fp.string(), error.what()));
            }
        }

        persist_keys.push_back(current_object_key_);
        object_stats_->PutNew(current_object_key_, std::make_shared<ObjStat>(current_object_size_, current_object_parts_, current_object_ref_count_));
        LOG_TRACE(fmt::format("CurrentObjFinalizeNoLock added composed object {}", current_object_key_));
        current_object_key_ = ObjCreate();
        current_object_size_ = 0;
        current_object_parts_ = 0;
        current_object_ref_count_ = 0;
    } else {
        LOG_TRACE(fmt::format("CurrentObjFinalizeNoLock added empty object {}", current_object_key_));
    }
}

PersistReadResult PersistenceManager::GetObjCache(const std::string &file_path) {
    PersistReadResult result;

    std::string local_path = RemovePrefix(file_path);
    if (local_path.empty()) {
        UnrecoverableError(fmt::format("Failed to find local path of {}", local_path));
    }

    std::string pm_fp_key = KeyEncode::PMObjectKey(local_path);
    std::string value;
    Status status = kv_store_->Get(pm_fp_key, value);
    if (!status.ok()) {
        LOG_WARN(fmt::format("GetObjCache Failed to find object for local path {}: {}", local_path, status.message()));
        // LOG_TRACE(fmt::format("All key-value pairs in kv_store: \n{}", kv_store_->ToString()));
        return result;
    }
    ObjAddr obj_addr;
    obj_addr.Deserialize(value);

    std::lock_guard<std::mutex> lock(mtx_);
    result.obj_addr_ = obj_addr;
    if (obj_addr.part_size_ == 0) {
        LOG_TRACE(fmt::format("GetObjCache empty object {} for local path {}", obj_addr.obj_key_, local_path));
        if (obj_addr.obj_key_ != ObjAddr::KeyEmpty) {
            UnrecoverableError(fmt::format("GetObjCache object {} is empty", obj_addr.obj_key_));
        }
    } else if (obj_addr.obj_key_ == current_object_key_) {
        current_object_ref_count_++;
        object_stats_->Get(obj_addr.obj_key_);
        LOG_TRACE(fmt::format("GetObjCache current object {} ref count {}", obj_addr.obj_key_, current_object_ref_count_));
    } else {
        std::shared_ptr<ObjStat> obj_stat = object_stats_->Get(obj_addr.obj_key_);
        LOG_TRACE(fmt::format("GetObjCache object {}, file_path: {}, ref count {}", obj_addr.obj_key_, file_path, obj_stat->ref_count_));
        std::string read_path = GetObjPath(result.obj_addr_.obj_key_);
        if (!VirtualStore::Exists(read_path)) {
            auto expect = ObjCached::kCached;
            obj_stat->cached_.compare_exchange_strong(expect, ObjCached::kNotCached);
            result.obj_stat_ = obj_stat;
        }
    }
    return result;
}

// std::tuple<size_t, Status> PersistenceManager::GetDirectorySize(const std::string &path_str) {
//     size_t total_size = 0;
//
//     if (!VirtualStore::Exists(path_str)) {
//         return {0, Status::IOError(fmt::format("{} doesn't exist.", path_str))};
//     }
//
//     const fs::path path(path_str);
//     for (const auto &entry : fs::recursive_directory_iterator(path)) {
//         if (fs::is_regular_file(entry.status())) {
//             total_size += fs::file_size(entry.path());
//         }
//     }
//
//     return {total_size, Status::OK()};
// }

std::tuple<size_t, Status> PersistenceManager::GetFileSize(const std::string &file_path) {
    PersistReadResult result;

    std::string local_path = RemovePrefix(file_path);
    if (local_path.empty()) {
        UnrecoverableError(fmt::format("Failed to find local path of {}", local_path));
    }

    std::string pm_fp_key = KeyEncode::PMObjectKey(local_path);
    std::string value;
    if (auto status = kv_store_->Get(pm_fp_key, value); !status.ok()) {
        LOG_WARN(fmt::format("GetFileSize Failed to find object for local path {}: {}", local_path, status.message()));
        return {0, Status::NotFound(fmt::format("Can't find {}", local_path))};
    }
    ObjAddr obj_addr;
    obj_addr.Deserialize(value);
    return {obj_addr.part_size_, Status::OK()};
}

// ObjAddr PersistenceManager::GetObjCacheWithoutCnt(const std::string &local_path) {
//     std::string lock_path = RemovePrefix(local_path);
//     if (lock_path.empty()) {
//         UnrecoverableError(fmt::format("Failed to find local path of {}", local_path));
//     }
//
//     std::string pm_fp_key = KeyEncode::PMObjectKey(local_path);
//     std::string value;
//     Status status = kv_store_->Get(pm_fp_key, value);
//     if (!status.ok()) {
//         LOG_WARN(fmt::format("GetFileSize Failed to find object for local path {}: {}", local_path, status.message()));
//         return ObjAddr();
//     }
//     ObjAddr obj_addr;
//     obj_addr.Deserialize(value);
//     return obj_addr;
// }

PersistWriteResult PersistenceManager::PutObjCache(const std::string &file_path) {
    PersistWriteResult result;
    std::string local_path = RemovePrefix(file_path);
    if (local_path.empty()) {
        UnrecoverableError(fmt::format("Failed to find file path of {}", file_path));
    }
    std::string pm_fp_key = KeyEncode::PMObjectKey(local_path);
    std::string value;
    Status status = kv_store_->Get(pm_fp_key, value);
    if (!status.ok()) {
        UnrecoverableError(fmt::format("Failed to find file_path: {} stored object", local_path));
    }
    ObjAddr obj_addr;
    obj_addr.Deserialize(value);
    if (obj_addr.part_size_ == 0) {
        LOG_TRACE(fmt::format("PutObjCache empty object {} for local path {}", obj_addr.obj_key_, local_path));
        return result;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    if (obj_addr.obj_key_ == current_object_key_) {
        if (current_object_ref_count_ <= 0) {
            UnrecoverableError(fmt::format("PutObjCache current object {} ref count is {}", obj_addr.obj_key_, current_object_ref_count_));
        }
        current_object_ref_count_--;
        LOG_TRACE(fmt::format("PutObjCache current object {} ref count {}", obj_addr.obj_key_, current_object_ref_count_));
    } else {
        std::shared_ptr<ObjStat> obj_stat = object_stats_->Release(obj_addr.obj_key_);
        if (obj_stat != nullptr) {
            LOG_TRACE(fmt::format("PutObjCache object {} ref count {}", obj_addr.obj_key_, obj_stat->ref_count_));
        } else {
            LOG_WARN(fmt::format("PutObjCache object {} unknown ref count", obj_addr.obj_key_));
        }
    }
    return result;
}

std::string PersistenceManager::ObjCreate() { return UUID().to_string(); }

size_t PersistenceManager::CurrentObjRoomNoLock() {
    const size_t aligned_size = CheckedAlignUp(current_object_size_, ObjAlignment, "Align current persistence object");
    if (aligned_size >= object_size_limit_) {
        return 0;
    }
    return object_size_limit_ - aligned_size;
}

size_t PersistenceManager::CurrentObjAppendNoLock(const std::string &tmp_file_path, size_t file_size) {
    fs::path src_fp = tmp_file_path;
    fs::path dst_fp = fs::path(workspace_) / current_object_key_;
    const size_t original_logical_size = current_object_size_;
    const size_t original_parts = current_object_parts_;
    const size_t part_offset = CheckedAlignUp(original_logical_size, ObjAlignment, "Align persistence part");
    if (part_offset >= object_size_limit_ || file_size >= object_size_limit_ - part_offset) {
        UnrecoverableError(fmt::format("CurrentObjAppendNoLock object {} cannot fit {} bytes at offset {} below limit {}",
                                       current_object_key_,
                                       file_size,
                                       part_offset,
                                       object_size_limit_));
    }
    const size_t expected_final_size = CheckedAdd(part_offset, file_size, "Append persistence part");

    // Debug: Check if this is a dictionary file
    bool is_dict_file = tmp_file_path.find(".dic") != std::string::npos;
    if (is_dict_file) {
        LOG_DEBUG(fmt::format("CurrentObjAppendNoLock: Processing dictionary file {} (size: {})", tmp_file_path, file_size));
    }

    std::error_code ec;
    const size_t observed_source_size = fs::file_size(src_fp, ec);
    if (ec) {
        UnrecoverableError(fmt::format("Failed to get source file size {}: {}", tmp_file_path, ec.message()));
    }
    if (observed_source_size != file_size) {
        UnrecoverableError(fmt::format("Source file {} size changed before append: expected {}, got {}",
                                       tmp_file_path,
                                       file_size,
                                       observed_source_size));
    }

    const bool destination_existed = fs::exists(dst_fp, ec);
    if (ec) {
        UnrecoverableError(fmt::format("Failed to inspect destination file {}: {}", dst_fp.string(), ec.message()));
    }
    const size_t original_physical_size = destination_existed ? fs::file_size(dst_fp, ec) : 0;
    if (ec) {
        UnrecoverableError(fmt::format("Failed to get destination file size {}: {}", dst_fp.string(), ec.message()));
    }
    if (original_physical_size != original_logical_size) {
        UnrecoverableError(fmt::format("Destination object {} physical size {} differs from logical size {}",
                                       dst_fp.string(),
                                       original_physical_size,
                                       original_logical_size));
    }

    auto buffer = std::make_unique_for_overwrite<char[]>(BUFFER_SIZE);
    std::ifstream src_file;
    std::ofstream dst_file;
    bool destination_opened = false;
    try {
        src_file.open(src_fp, std::ios::binary);
        if (!src_file.is_open()) {
            throw std::runtime_error(fmt::format("Failed to open source file {}", tmp_file_path));
        }
        dst_file.open(dst_fp, std::ios::binary | std::ios::app);
        if (!dst_file.is_open()) {
            throw std::runtime_error(fmt::format("Failed to open destination file {}: {}", dst_fp.string(), strerror(errno)));
        }
        destination_opened = true;

        const std::array<char, ObjAlignment> zero_padding{};
        const size_t padding_size = part_offset - original_logical_size;
        if (padding_size > 0) {
            dst_file.write(zero_padding.data(), static_cast<std::streamsize>(padding_size));
            if (!dst_file.good()) {
                throw std::runtime_error(fmt::format("Failed to write {} padding bytes to {}", padding_size, dst_fp.string()));
            }
        }

        size_t copied = 0;
        while (copied < file_size) {
            const size_t requested = std::min(BUFFER_SIZE, file_size - copied);
            src_file.read(buffer.get(), static_cast<std::streamsize>(requested));
            const std::streamsize read_count = src_file.gcount();
            if (read_count != static_cast<std::streamsize>(requested)) {
                throw std::runtime_error(
                    fmt::format("Short read from {} at offset {}: expected {}, got {}",
                                tmp_file_path,
                                copied,
                                requested,
                                read_count));
            }
            dst_file.write(buffer.get(), read_count);
            if (!dst_file.good()) {
                throw std::runtime_error(
                    fmt::format("Failed to write {} bytes to {} at offset {}",
                                read_count,
                                dst_fp.string(),
                                part_offset + copied));
            }
            copied += static_cast<size_t>(read_count);
        }

        char extra_byte{};
        src_file.read(&extra_byte, 1);
        if (src_file.gcount() != 0) {
            throw std::runtime_error(fmt::format("Source file {} grew during append", tmp_file_path));
        }
        if (src_file.bad()) {
            throw std::runtime_error(fmt::format("Failed to check EOF for source file {}", tmp_file_path));
        }
        src_file.clear();
        const size_t final_source_size = fs::file_size(src_fp, ec);
        if (ec || final_source_size != file_size) {
            throw std::runtime_error(
                fmt::format("Source file {} size changed during append: expected {}, got {} ({})",
                            tmp_file_path,
                            file_size,
                            final_source_size,
                            ec ? ec.message() : "no filesystem error"));
        }

        src_file.close();
        if (src_file.fail()) {
            throw std::runtime_error(fmt::format("Failed to close source file {}", tmp_file_path));
        }
        dst_file.flush();
        if (!dst_file.good()) {
            throw std::runtime_error(fmt::format("Failed to flush destination file {}", dst_fp.string()));
        }
        dst_file.close();
        if (dst_file.fail()) {
            throw std::runtime_error(fmt::format("Failed to close destination file {}", dst_fp.string()));
        }
        const size_t actual_final_size = fs::file_size(dst_fp, ec);
        if (ec || actual_final_size != expected_final_size) {
            throw std::runtime_error(
                fmt::format("Destination object {} size mismatch: expected {}, got {} ({})",
                            dst_fp.string(),
                            expected_final_size,
                            actual_final_size,
                            ec ? ec.message() : "no filesystem error"));
        }
    } catch (const std::exception &error) {
        if (src_file.is_open()) {
            src_file.clear();
            src_file.close();
        }
        if (dst_file.is_open()) {
            dst_file.clear();
            dst_file.close();
        }
        std::error_code rollback_ec;
        if (destination_existed) {
            fs::resize_file(dst_fp, original_physical_size, rollback_ec);
        } else if (destination_opened || fs::exists(dst_fp)) {
            fs::remove(dst_fp, rollback_ec);
        }
        current_object_size_ = original_logical_size;
        current_object_parts_ = original_parts;
        if (rollback_ec) {
            UnrecoverableError(fmt::format("Failed to append {} to object {}: {}; rollback to {} bytes also failed: {}",
                                           tmp_file_path,
                                           current_object_key_,
                                           error.what(),
                                           original_physical_size,
                                           rollback_ec.message()));
        }
        UnrecoverableError(fmt::format("Failed to append {} to object {}: {}", tmp_file_path, current_object_key_, error.what()));
    }

    current_object_size_ = expected_final_size;
    current_object_parts_ = original_parts + 1;

    // Debug: Log completion for dictionary files
    if (is_dict_file) {
        LOG_DEBUG(fmt::format("CurrentObjAppendNoLock: Completed processing dictionary file {} -> object {} (offset: {}, size: {})",
                              tmp_file_path,
                              current_object_key_,
                              part_offset,
                              file_size));
    }
    return part_offset;
}

void PersistenceManager::CleanupNoLock(const ObjAddr &object_addr,
                                       std::vector<std::string> &persist_keys,
                                       std::vector<std::string> &drop_from_remote_keys,
                                       bool check_ref_count) {
    std::shared_ptr<ObjStat> obj_stat = object_stats_->GetNoCount(object_addr.obj_key_);
    if (obj_stat == nullptr) {
        if (object_addr.obj_key_ == ObjAddr::KeyEmpty) {
            assert(object_addr.part_size_ == 0);
            return;
        } else if (object_addr.obj_key_ == current_object_key_) {
            CurrentObjFinalizeNoLock(persist_keys);
            obj_stat = object_stats_->GetNoCount(object_addr.obj_key_);
            assert(obj_stat != nullptr);
        } else {
            UnrecoverableError(fmt::format("CleanupNoLock Failed to find object {}", object_addr.obj_key_));
            return;
        }
    }
    size_t range_end = object_addr.part_offset_ + object_addr.part_size_;
    range_end = (range_end + ObjAlignment - 1) & ~(ObjAlignment - 1);
    Range orig_range(object_addr.part_offset_, range_end);
    Range range(orig_range);
    auto inst_it = obj_stat->deleted_ranges_.lower_bound(range);

    if (inst_it != obj_stat->deleted_ranges_.begin()) {
        auto inst_it_prev = std::prev(inst_it);
        if (inst_it_prev->Cover(orig_range)) {
            // Check duplication. Cleanup could delete a local file multiple times.
            LOG_WARN(fmt::format("CleanupNoLock delete [{}, {}) more than once", range.start_, range.end_));
            return;
        } else if (inst_it_prev->Intersect(orig_range)) {
            // Check intersection with prev range
            UnrecoverableError(fmt::format("ObjAddr {} range to delete [{}, {}) intersects with prev one [{}, {})",
                                           object_addr.obj_key_,
                                           orig_range.start_,
                                           orig_range.end_,
                                           inst_it_prev->start_,
                                           inst_it_prev->end_));
        } else if (orig_range.start_ == inst_it_prev->end_) {
            // Try merge with prev range
            range.start_ = inst_it_prev->start_;
            inst_it = obj_stat->deleted_ranges_.erase(inst_it_prev);
        }
    }

    if (inst_it != obj_stat->deleted_ranges_.end()) {
        if (inst_it->Cover(orig_range)) {
            // Check duplication. Cleanup could delete a local file multiple times.
            LOG_WARN(fmt::format("CleanupNoLock delete [{}, {}) more than once", orig_range.start_, orig_range.end_));
            return;
        } else if (inst_it->Intersect(orig_range)) {
            // Check intersection with next range
            UnrecoverableError(fmt::format("ObjAddr {} range to delete [{}, {}) intersects with next one [{}, {})",
                                           object_addr.obj_key_,
                                           orig_range.start_,
                                           orig_range.end_,
                                           inst_it->start_,
                                           inst_it->end_));
        } else if (orig_range.end_ == inst_it->start_) {
            // Try merge with next range
            range.end_ = inst_it->end_;
            obj_stat->deleted_ranges_.erase(inst_it);
        }
    }

    auto [_, ok] = obj_stat->deleted_ranges_.insert(range);
    if (!ok) {
        UnrecoverableError(fmt::format("Failed to delete ObjAddr {} range [{}, {})", object_addr.obj_key_, range.start_, range.end_));
    }

    LOG_TRACE(fmt::format("Deleted object {} range [{}, {})", object_addr.obj_key_, orig_range.start_, orig_range.end_));
    size_t obj_size = (obj_stat->obj_size_ + ObjAlignment - 1) & ~(ObjAlignment - 1);
    if (range.start_ == 0 && range.end_ == obj_size && object_addr.obj_key_ != current_object_key_) {
        if (object_addr.obj_key_.empty()) {
            UnrecoverableError(fmt::format("Failed to find object key"));
        }
        if (check_ref_count) {
            if (obj_stat->ref_count_ > 0) {
                UnrecoverableError(fmt::format("CleanupNoLock object {} ref count is {}", object_addr.obj_key_, obj_stat->ref_count_));
            }
        }
        drop_from_remote_keys.emplace_back(object_addr.obj_key_);
        object_stats_->Invalidate(object_addr.obj_key_);
        LOG_TRACE(fmt::format("Deleted object {}", object_addr.obj_key_));
    } else {
        object_stats_->PutNoCount(object_addr.obj_key_, obj_stat);
    }
}

// ObjStat PersistenceManager::GetObjStatByObjAddr(const ObjAddr &obj_addr) {
//     std::lock_guard<std::mutex> lock(mtx_);
//     std::shared_ptr<ObjStat> obj_stat = object_stats_->GetNoCount(obj_addr.obj_key_);
//     if (obj_stat == nullptr) {
//         return ObjStat();
//     }
//     return *obj_stat;
// }
//
// void PersistenceManager::SaveLocalPath(const std::string &file_path, const ObjAddr &object_addr) { AddObjAddrToKVStore(file_path, object_addr); }
//
// void PersistenceManager::SaveObjStat(const std::string &obj_key, const std::shared_ptr<ObjStat> &obj_stat) {
//     std::lock_guard<std::mutex> lock(mtx_);
//     object_stats_->PutNoCount(obj_key, obj_stat);
// }
//
// void PersistenceManager::AddObjAddrToKVStore(const std::string &path, const ObjAddr &obj_addr) {
//     std::string key = KeyEncode::PMObjectKey(RemovePrefix(path));
//     std::string value = obj_addr.Serialize().dump();
//     Status status = kv_store_->Put(key, value, false);
//     if (!status.ok()) {
//         UnrecoverableError(status.message());
//     }
// }

std::string PersistenceManager::RemovePrefix(const std::string &path) {
    if (path.starts_with(local_data_dir_)) {
        return path.substr(local_data_dir_.length());
    }
    if (!path.starts_with("/")) {
        return path;
    }
    return "";
}

PersistWriteResult PersistenceManager::Cleanup(const std::string &file_path) {
    PersistWriteResult result;

    std::string local_path = RemovePrefix(file_path);
    if (local_path.empty()) {
        UnrecoverableError(fmt::format("Failed to find local path of {}", local_path));
    }

    std::string pm_fp_key = KeyEncode::PMObjectKey(local_path);
    std::string value;
    Status status = kv_store_->Get(pm_fp_key, value);
    if (!status.ok()) {
        LOG_WARN(fmt::format("Failed to find object for local path {}", local_path));
        return result;
    }
    status = kv_store_->Delete(pm_fp_key, false);
    if (!status.ok()) {
        LOG_CRITICAL(fmt::format("Failed to delete object for local path {}", local_path));
        return result;
    }
    ObjAddr obj_addr;
    obj_addr.Deserialize(value);

    CleanupNoLock(obj_addr, result.persist_keys_, result.drop_from_remote_keys_, true);
    LOG_TRACE(fmt::format("Deleted mapping from local path {} to ObjAddr({}, {}, {})",
                          local_path,
                          obj_addr.obj_key_,
                          obj_addr.part_offset_,
                          obj_addr.part_size_));
    return result;
}

void PersistenceManager::SetKvStore(KVStore *kv_store) {
    kv_store_ = kv_store;
    std::unique_ptr<KVInstance> kv_instance = kv_store_->GetInstance();
    object_stats_->Deserialize(kv_instance.get());
}

std::unordered_map<std::string, std::shared_ptr<ObjStat>> PersistenceManager::GetAllObjects() const { return object_stats_->GetAllObjects(); }

std::unordered_map<std::string, ObjAddr> PersistenceManager::GetAllFiles() const {
    std::unordered_map<std::string, ObjAddr> local_path_obj;
    const std::string &obj_prefix = KeyEncode::PMObjectPrefix();
    size_t obj_prefix_len = obj_prefix.size();

    std::unique_ptr<KVInstance> kv_instance = kv_store_->GetInstance();
    auto iter = kv_instance->GetIterator();
    iter->Seek(obj_prefix);
    while (iter->Valid() && iter->Key().starts_with(obj_prefix)) {
        std::string path = iter->Key().ToString().substr(obj_prefix_len);
        ObjAddr obj_addr;
        obj_addr.Deserialize(iter->Value().ToString());
        local_path_obj.emplace(path, obj_addr);
        iter->Next();
    }
    return local_path_obj;
}

} // namespace infinity
