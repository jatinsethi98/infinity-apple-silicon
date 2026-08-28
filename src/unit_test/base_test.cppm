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

#include <cstdlib>
#include "unit_test/gtest_expand.h"

export module infinity_core:ut.base_test;

import :infinity_context;
import :infinity_exception;

import global_resource_usage;

namespace fs = std::filesystem;

namespace infinity {

export template <typename T>
class BaseTestWithParam : public std::conditional_t<std::is_same_v<T, void>, ::testing::Test, ::testing::TestWithParam<T>> {
public:
    BaseTestWithParam() {
        const char *infinity_home_ = GetHomeDir();
        if (bool ok = ValidateDirPermission(infinity_home_); !ok) {
            std::cerr << "FATAL: Please ensure directory " << infinity_home_ << " exists and current user has RWX permission of it." << std::endl;
            abort();
        }
        const char *RESOURCE_DIR = GetResourceDir();
        if (!fs::exists(RESOURCE_DIR)) {
            std::cerr << "WARN: Resource directory doesn't exist: " << RESOURCE_DIR << std::endl;
        } else if (bool ok = ValidateDirPermission(RESOURCE_DIR); !ok) {
            std::cerr << "FATAL: Please ensure directory " << RESOURCE_DIR << " exists and current user has RWX permission of it." << std::endl;
            abort();
        }

        CleanupTmpDir();
    }

    ~BaseTestWithParam() override = default;

    void SetUp() override {
        // SetPrintStacktrace(false);
    }
    void TearDown() override {}

public:
    static constexpr const char *NULL_CONFIG_PATH = "";

    static constexpr const char *CONFIG_PATH = "test/data/config/test.toml";

    static constexpr const char *VFS_OFF_CONFIG_PATH = "test/data/config/test_vfs_off.toml";

    static constexpr const char *NEW_CONFIG_PATH = "test/data/config/test_new.toml";

    static constexpr const char *NEW_VFS_OFF_CONFIG_PATH = "test/data/config/test_new_vfs_off.toml";

    static constexpr const char *NEW_BG_ON_CONFIG_PATH = "test/data/config/test_new_bg_on.toml";

    static constexpr const char *NEW_VFS_OFF_BG_ON_CONFIG_PATH = "test/data/config/test_new_vfs_off_bg_on.toml";

    static constexpr const char *NEW_BG_ON_CONFIG_PATH2 = "test/data/config/test_new_bg_on_2.toml";

    static constexpr const char *NEW_VFS_OFF_BG_ON_CONFIG_PATH2 = "test/data/config/test_new_vfs_off_bg_on_2.toml";

    static constexpr const char *NEW_CONFIG_NOWAL_PATH = "test/data/config/test_new_nowal.toml";

    static constexpr const char *NEW_VFS_OFF_CONFIG_NOWAL_PATH = "test/data/config/test_vfs_off_nowal.toml";

    static constexpr const char *NEW_VFS_OFF_BG_OFF_PATH = "test/data/config/test_vfs_off_bg_off.toml";

    static constexpr const char *S3_STORAGE = "test/data/config/test_minio_s3_storage.toml";

protected:
    const char *GetHomeDir() { return ResolvedHomeDir().c_str(); }

    const char *GetFullDataDir() {
        static const std::string path = (fs::path(ResolvedHomeDir()) / "data").string();
        return path.c_str();
    }

    const char *GetFullWalDir() {
        static const std::string path = (fs::path(ResolvedHomeDir()) / "wal").string();
        return path.c_str();
    }

    const char *GetFullLogDir() {
        static const std::string path = (fs::path(ResolvedHomeDir()) / "log").string();
        return path.c_str();
    }

    const char *GetFullTmpDir() {
        static const std::string path = (fs::path(ResolvedHomeDir()) / "tmp").string();
        return path.c_str();
    }

    const char *GetCatalogDir() {
        static const std::string path = (fs::path(ResolvedHomeDir()) / "catalog").string();
        return path.c_str();
    }

    const char *GetFullPersistDir() {
        static const std::string path = (fs::path(ResolvedHomeDir()) / "persistence").string();
        return path.c_str();
    }

    const char *GetTmpDir() { return "tmp"; }

    const char *GetResourceDir() { return "/usr/share/infinity/resource"; }

    const char *GetSnapshotDir() {
        static const std::string path = (fs::path(ResolvedHomeDir()) / "snapshot").string();
        return path.c_str();
    }

    void CleanupDbDirs() {
        const char *infinity_db_dirs[] =
            {GetFullDataDir(), GetFullWalDir(), GetFullLogDir(), GetFullTmpDir(), GetFullPersistDir(), GetCatalogDir(), GetSnapshotDir()};
        for (auto &dir : infinity_db_dirs) {
            CleanupDirectory(dir);
        }
    }

    void CleanupTmpDir() { CleanupDirectory(GetFullTmpDir()); }

    void RemoveDbDirs() {
        const char *infinity_db_dirs[] =
            {GetFullDataDir(), GetFullWalDir(), GetFullLogDir(), GetFullTmpDir(), GetFullPersistDir(), GetCatalogDir(), GetSnapshotDir()};
        for (auto &dir : infinity_db_dirs) {
            RemoveDirectory(dir);
        }
    }

    // Create a data block with two columns, each with the specified row count.
    std::shared_ptr<DataBlock> MakeInputBlock(const Value &v1, const Value &v2, size_t row_cnt);
    // Create a data block with two columns, each with the specified row count (data is specified in the function).
    std::shared_ptr<DataBlock> MakeInputBlock1(size_t row_cnt);
    // Create a data block with two columns, each with the specified row count (data is specified in the function).
    std::shared_ptr<DataBlock> MakeInputBlock2(size_t row_cnt);

    // Check if the file paths exist or not.
    void CheckFilePaths(std::vector<std::string> &delete_file_paths, std::vector<std::string> &exist_file_paths);

private:
    static const std::string &ResolvedHomeDir() {
        static const std::string home_dir = [] {
#ifdef __APPLE__
            const char *test_home = std::getenv("INFINITY_TEST_HOME");
            if (test_home != nullptr && test_home[0] != '\0') {
                return std::string(test_home);
            }
#endif
            return std::string("/var/infinity");
        }();
        return home_dir;
    }

    // Validate if given path satisfy all of following:
    // - The path is a directory or symlink to a directory.
    // - Current user has read, write, and execute permission of the path.
    bool ValidateDirPermission(const char *path_str) {
        fs::path path(path_str);
        std::error_code ec;

        // Check if the path exists and is a directory or symlink to a directory
        if (!fs::exists(path, ec) || ec)
            return false;
        if (!fs::is_directory(path, ec) && !(fs::is_symlink(path, ec) && fs::is_directory(fs::read_symlink(path), ec)))
            return false;

        // Check read and execute permission
        fs::directory_iterator it(path, fs::directory_options::skip_permission_denied, ec);
        if (ec)
            return false;

        // Check write permission
        fs::path temp_file = path / "temp_file.txt";
        std::ofstream ofs(temp_file, std::ios::out | std::ios::app);
        if (!ofs)
            return false;
        ofs.close();
        fs::remove(temp_file, ec);
        if (ec)
            return false;

        return true;
    }

    void CleanupDirectory(const char *dir) {
        std::error_code error_code;
        fs::path p(dir);
        if (!fs::exists(dir)) {
            std::filesystem::create_directories(p, error_code);
            if (error_code.value() != 0) {
                UnrecoverableError(fmt::format("Failed to create directory {}", dir));
            }
        }
        try {
            for (const auto &dir_entry : std::filesystem::directory_iterator{dir}) {
                std::filesystem::remove_all(dir_entry.path());
            };
        } catch (const std::filesystem::filesystem_error &e) {
            UnrecoverableError(fmt::format("Failed to cleanup {}, exception: {}", dir, e.what()));
        }
    }

    void RemoveDirectory(const char *dir) {
        std::error_code error_code;
        fs::path p(dir);
        try {
            std::filesystem::remove_all(p, error_code);
        } catch (const std::filesystem::filesystem_error &e) {
            UnrecoverableError(fmt::format("Failed to remove {}, exception: {}", dir, e.what()));
        }
    }
};

export using BaseTest = BaseTestWithParam<void>;

export class BaseTestNoParam : public BaseTestWithParam<void> {
public:
    void SetUp() override {
        // Earlier cases may leave a dirty infinity instance. Destroy it first.
        infinity::InfinityContext::instance().UnInit();
        CleanupDbDirs();
#ifdef INFINITY_DEBUG
        infinity::GlobalResourceUsage::Init();
#endif
        auto config_path = std::make_shared<std::string>(BaseTestNoParam::NULL_CONFIG_PATH);
        infinity::InfinityContext::instance().InitPhase1(config_path);
        infinity::InfinityContext::instance().InitPhase2();
    }

    void TearDown() override {
        infinity::InfinityContext::instance().UnInit();
        CleanupDbDirs();
#ifdef INFINITY_DEBUG
        EXPECT_EQ(infinity::GlobalResourceUsage::GetObjectCount(), 0);
        EXPECT_EQ(infinity::GlobalResourceUsage::GetRawMemoryCount(), 0);
        infinity::GlobalResourceUsage::UnInit();
#endif
    }
};

export class NewBaseTestNoParam : public BaseTestWithParam<void> {
public:
    void SetUp() override {
        // Earlier cases may leave a dirty infinity instance. Destroy it first.
        infinity::InfinityContext::instance().UnInit();
        CleanupDbDirs();
#ifdef INFINITY_DEBUG
        infinity::GlobalResourceUsage::Init();
#endif
        auto config_path = std::make_shared<std::string>(BaseTestNoParam::NEW_CONFIG_PATH);
        infinity::InfinityContext::instance().InitPhase1(config_path);
        infinity::InfinityContext::instance().InitPhase2();
    }

    void TearDown() override {
        infinity::InfinityContext::instance().UnInit();
        CleanupDbDirs();
#ifdef INFINITY_DEBUG
        EXPECT_EQ(infinity::GlobalResourceUsage::GetObjectCount(), 0);
        EXPECT_EQ(infinity::GlobalResourceUsage::GetRawMemoryCount(), 0);
        infinity::GlobalResourceUsage::UnInit();
#endif
    }
};

export class BaseTestParamStr : public BaseTestWithParam<std::string> {
public:
    void SetUp() override {
        // Earlier cases may leave a dirty infinity instance. Destroy it first.
        infinity::InfinityContext::instance().UnInit();
        CleanupDbDirs();
#ifdef INFINITY_DEBUG
        infinity::GlobalResourceUsage::Init();
#endif
        std::string config_path_str = GetParam();
        config_path = nullptr;
        if (config_path_str != BaseTestParamStr::NULL_CONFIG_PATH) {
            config_path = std::make_shared<std::string>(std::filesystem::absolute(config_path_str));
        }
        infinity::InfinityContext::instance().InitPhase1(config_path);
        infinity::InfinityContext::instance().InitPhase2();
    }

    void TearDown() override {
        infinity::InfinityContext::instance().UnInit();
        CleanupDbDirs();
#ifdef INFINITY_DEBUG
        EXPECT_EQ(infinity::GlobalResourceUsage::GetObjectCount(), 0);
        EXPECT_EQ(infinity::GlobalResourceUsage::GetRawMemoryCount(), 0);
        infinity::GlobalResourceUsage::UnInit();
#endif
    }

protected:
    std::shared_ptr<std::string> config_path;
};

} // namespace infinity
