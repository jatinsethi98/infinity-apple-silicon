module;

#include "unit_test/gtest_expand.h"

#if defined(__APPLE__)
#include <pthread.h>
#endif

module infinity_core:ut.persistence_manager;

import :ut.base_test;

import :persistence_manager;
import :virtual_store;
import third_party;
import :persist_result_handler;
import :local_file_handle;
import :kv_store;
import :status;
import :infinity_exception;

using namespace infinity;
namespace fs = std::filesystem;

class PersistenceManagerTest : public BaseTest {
public:
    void SetUp() override {
        BaseTest::RemoveDbDirs();
        workspace_ = std::string(GetFullTmpDir()) + "/persistence";
        file_dir_ = std::string(GetFullTmpDir()) + "/persistence_src";
        catalog_dir_ = std::string(GetFullTmpDir()) + "/catalog";

        system(("mkdir -p " + workspace_).c_str());
        system(("mkdir -p " + file_dir_).c_str());
        system(("mkdir -p " + catalog_dir_).c_str());

        kv_store_ = std::make_unique<KVStore>();
        Status status = kv_store_->Init(catalog_dir_);
        EXPECT_TRUE(status.ok());
        pm_ = std::make_unique<PersistenceManager>(infinity::InfinityContext::instance().storage(), workspace_, file_dir_, ObjSizeLimit);
        pm_->SetKvStore(kv_store_.get());
        handler_ = std::make_unique<PersistResultHandler>(pm_.get());
    }

    void CheckObjData(const std::string &obj_addr, const std::string &data);

protected:
    std::string workspace_{};
    std::string file_dir_{};
    std::string catalog_dir_{};
    std::unique_ptr<KVStore> kv_store_{};
    std::unique_ptr<PersistenceManager> pm_{};
    static constexpr int ObjSizeLimit = 128;
    std::unique_ptr<PersistResultHandler> handler_;
};

void PersistenceManagerTest::CheckObjData(const std::string &local_file_path, const std::string &data) {
    PersistReadResult result = pm_->GetObjCache(local_file_path);
    const ObjAddr &obj_addr = handler_->HandleReadResult(result);
    std::string obj_path = pm_->GetObjPath(obj_addr.obj_key_);
    fs::path obj_fp(obj_path);
    ASSERT_TRUE(fs::exists(obj_fp));
    ASSERT_EQ(obj_addr.part_size_, data.size());
    size_t obj_file_size = fs::file_size(obj_fp);
    ASSERT_LE(obj_file_size, ObjSizeLimit);

    auto [pm_file_handle, status] = VirtualStore::Open(obj_path, FileAccessMode::kRead);
    EXPECT_TRUE(status.ok());
    status = pm_file_handle->Seek(obj_addr.part_offset_);
    EXPECT_TRUE(status.ok());
    auto file_size = obj_addr.part_size_;
    auto buffer = std::make_unique<char[]>(file_size);
    auto [nread, read_status] = pm_file_handle->Read(buffer.get(), file_size);
    EXPECT_TRUE(read_status.ok());
    ASSERT_EQ(nread, file_size);
    ASSERT_EQ(std::string(buffer.get(), file_size), data);

    PersistWriteResult res = pm_->PutObjCache(local_file_path);
    handler_->HandleWriteResult(res);
}

TEST_F(PersistenceManagerTest, PersistFileBasic) {
    std::string file_path = file_dir_ + "/persist_file";
    std::ofstream out_file(file_path);
    std::string persist_str = "Persistence Manager Test";
    out_file << persist_str;
    out_file.close();
    PersistWriteResult result = pm_->Persist(file_path, file_path);
    handler_->HandleWriteResult(result);
    const ObjAddr &obj_addr = result.obj_addr_;
    ASSERT_TRUE(obj_addr.Valid());
    ASSERT_EQ(obj_addr.part_size_, persist_str.size());
    PersistWriteResult result2 = pm_->CurrentObjFinalize();
    handler_->HandleWriteResult(result2);

    CheckObjData(file_path, persist_str);
}

TEST_F(PersistenceManagerTest, PersistMultiFile) {
    std::string file_path_base = file_dir_ + "/persist_file";
    std::vector<std::string> file_paths;
    std::vector<std::string> persist_strs;
    std::vector<ObjAddr> obj_addrs;
    for (size_t i = 0; i < 10; ++i) {
        std::string file_path = file_path_base + std::to_string(i);
        std::ofstream out_file(file_path);
        std::string persist_str = "Persistence Manager Test " + std::to_string(i);
        out_file << persist_str;
        out_file.close();
        file_paths.push_back(file_path);
        persist_strs.push_back(persist_str);

        PersistWriteResult result = pm_->Persist(file_path, file_path);
        handler_->HandleWriteResult(result);
        const ObjAddr &obj_addr = result.obj_addr_;
        ASSERT_TRUE(obj_addr.Valid());
        ASSERT_EQ(obj_addr.part_size_, persist_str.size());
        obj_addrs.push_back(obj_addr);
    }
    ASSERT_EQ(file_paths.size(), persist_strs.size());
    ASSERT_EQ(file_paths.size(), obj_addrs.size());
    PersistWriteResult result = pm_->CurrentObjFinalize();
    handler_->HandleWriteResult(result);

    for (size_t i = 0; i < file_paths.size(); ++i) {
        CheckObjData(file_paths[i], persist_strs[i]);
    }
}

TEST_F(PersistenceManagerTest, PersistFileMultiThread) {
    std::string file_path_base = file_dir_ + "/persist_file";
    std::vector<std::string> file_paths;
    std::vector<std::string> persist_strs;
    std::unordered_map<std::string, ObjAddr> obj_addrs;
    std::vector<std::thread> threads;
    std::mutex obj_mutex;

    for (size_t i = 0; i < 10; ++i) {
        std::string file_path = file_path_base + std::to_string(i);
        std::ofstream out_file(file_path);
        std::string persist_str = "Persistence Manager Test " + std::to_string(i);
        out_file << persist_str;
        out_file.close();
        file_paths.push_back(file_path);
        persist_strs.push_back(persist_str);

        threads.emplace_back([this, file_path, persist_str, &obj_addrs, &obj_mutex]() {
            PersistWriteResult result = pm_->Persist(file_path, file_path);
            handler_->HandleWriteResult(result);
            const ObjAddr &obj_addr = result.obj_addr_;
            ASSERT_TRUE(obj_addr.Valid());
            ASSERT_EQ(obj_addr.part_size_, persist_str.size());
            std::unique_lock<std::mutex> lock(obj_mutex);
            obj_addrs[file_path] = obj_addr;
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    ASSERT_EQ(file_paths.size(), persist_strs.size());
    ASSERT_EQ(file_paths.size(), obj_addrs.size());
    PersistWriteResult result = pm_->CurrentObjFinalize();
    handler_->HandleWriteResult(result);

    for (size_t i = 0; i < file_paths.size(); ++i) {
        CheckObjData(file_paths[i], persist_strs[i]);
    }
}

TEST_F(PersistenceManagerTest, PersistLargeComposedFileFromBackgroundThread) {
    constexpr size_t kCopyBufferSize = 1024 * 1024;
    constexpr size_t kFileSize = kCopyBufferSize + 1048;
    constexpr size_t kObjectSizeLimit = 2 * kCopyBufferSize;

    handler_.reset();
    pm_ = std::make_unique<PersistenceManager>(
        infinity::InfinityContext::instance().storage(), workspace_, file_dir_, kObjectSizeLimit);
    pm_->SetKvStore(kv_store_.get());
    handler_ = std::make_unique<PersistResultHandler>(pm_.get());

    const std::string file_path = file_dir_ + "/large_composed_file";
    std::string expected(kFileSize, '\0');
    for (size_t idx = 0; idx < expected.size(); ++idx) {
        expected[idx] = static_cast<char>(idx % 251);
    }
    {
        std::ofstream out_file(file_path, std::ios::binary);
        ASSERT_TRUE(out_file.is_open());
        out_file.write(expected.data(), static_cast<std::streamsize>(expected.size()));
        ASSERT_TRUE(out_file.good());
    }

    std::optional<PersistWriteResult> write_result;
    std::exception_ptr worker_error;
    size_t worker_stack_size = 0;
    std::thread worker([&] {
#if defined(__APPLE__)
        worker_stack_size = pthread_get_stacksize_np(pthread_self());
#endif
        try {
            write_result = pm_->Persist(file_path, file_path);
        } catch (...) {
            worker_error = std::current_exception();
        }
    });
    worker.join();

#if defined(__APPLE__)
    EXPECT_LT(worker_stack_size, kCopyBufferSize);
#endif
    ASSERT_FALSE(worker_error);
    ASSERT_TRUE(write_result.has_value());
    handler_->HandleWriteResult(*write_result);
    ASSERT_EQ(write_result->obj_addr_.part_size_, expected.size());

    PersistWriteResult finalize_result = pm_->CurrentObjFinalize();
    handler_->HandleWriteResult(finalize_result);

    const std::string object_path = pm_->GetObjPath(write_result->obj_addr_.obj_key_);
    std::ifstream persisted_file(object_path, std::ios::binary);
    ASSERT_TRUE(persisted_file.is_open());
    persisted_file.seekg(static_cast<std::streamoff>(write_result->obj_addr_.part_offset_));
    std::string actual(expected.size(), '\0');
    persisted_file.read(actual.data(), static_cast<std::streamsize>(actual.size()));
    ASSERT_EQ(persisted_file.gcount(), static_cast<std::streamsize>(actual.size()));
    EXPECT_EQ(actual, expected);
}

TEST_F(PersistenceManagerTest, RejectPhysicalObjectSizeDriftWithoutAdvancingState) {
    const std::string first_path = file_dir_ + "/first";
    const std::string second_path = file_dir_ + "/second";
    const std::string first_data = "first";
    const std::string second_data = "second";
    {
        std::ofstream source(first_path, std::ios::binary);
        source.write(first_data.data(), static_cast<std::streamsize>(first_data.size()));
    }
    PersistWriteResult first_result = pm_->Persist(first_path, first_path);
    handler_->HandleWriteResult(first_result);
    const std::string object_path = pm_->GetObjPath(first_result.obj_addr_.obj_key_);
    ASSERT_EQ(fs::file_size(object_path), first_data.size());

    {
        std::ofstream object(object_path, std::ios::binary | std::ios::app);
        object.put('x');
    }
    {
        std::ofstream source(second_path, std::ios::binary);
        source.write(second_data.data(), static_cast<std::streamsize>(second_data.size()));
    }
    EXPECT_THROW_WITHOUT_STACKTRACE((void)pm_->Persist(second_path, second_path), UnrecoverableException);
    ASSERT_TRUE(fs::exists(second_path));
    ASSERT_EQ(fs::file_size(object_path), first_data.size() + 1);

    fs::resize_file(object_path, first_data.size());
    PersistWriteResult second_result = pm_->Persist(second_path, second_path);
    handler_->HandleWriteResult(second_result);
    EXPECT_EQ(second_result.obj_addr_.obj_key_, first_result.obj_addr_.obj_key_);
    EXPECT_EQ(second_result.obj_addr_.part_offset_, PersistenceManager::ObjAlignment);

    PersistWriteResult finalize_result = pm_->CurrentObjFinalize();
    handler_->HandleWriteResult(finalize_result);
    CheckObjData(first_path, first_data);
    CheckObjData(second_path, second_data);
}

TEST_F(PersistenceManagerTest, CleanupBasic) {
    std::string file_path_base = file_dir_ + "/persist_file";
    std::vector<std::string> file_paths;
    std::vector<std::string> persist_strs;
    std::vector<ObjAddr> obj_addrs;
    std::set<std::string> obj_paths;

    for (size_t i = 0; i < 10; ++i) {
        std::string file_path = file_path_base + std::to_string(i);
        std::ofstream out_file(file_path);
        std::string persist_str = "Persistence Manager Test " + std::to_string(i);
        out_file << persist_str;
        out_file.close();
        file_paths.push_back(file_path);
        persist_strs.push_back(persist_str);

        PersistWriteResult result = pm_->Persist(file_path, file_path);
        handler_->HandleWriteResult(result);
        const ObjAddr &obj_addr = result.obj_addr_;
        ASSERT_TRUE(obj_addr.Valid());
        ASSERT_EQ(obj_addr.part_size_, persist_str.size());
        obj_addrs.push_back(obj_addr);
        obj_paths.insert(workspace_ + "/" + obj_addr.obj_key_);
    }
    ASSERT_EQ(file_paths.size(), persist_strs.size());
    ASSERT_EQ(file_paths.size(), obj_addrs.size());
    PersistWriteResult result = pm_->CurrentObjFinalize();
    handler_->HandleWriteResult(result);

    for (size_t i = 0; i < file_paths.size(); ++i) {
        CheckObjData(file_paths[i], persist_strs[i]);
    }

    for (const auto &obj_path : obj_paths) {
        ASSERT_TRUE(fs::exists(obj_path));
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(file_paths.begin(), file_paths.end(), g);

    for (auto &file_path : file_paths) {
        PersistWriteResult result = pm_->Cleanup(file_path);
        handler_->HandleWriteResult(result);
    }
    for (const auto &obj_path : obj_paths) {
        ASSERT_FALSE(fs::exists(obj_path));
    }
}
