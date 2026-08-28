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

#include <gtest/gtest.h>

#ifdef __APPLE__
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#endif

int main(int argc, char **argv) {
#ifdef __APPLE__
    std::error_code error_code;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path(error_code);
    if (error_code) {
        std::cerr << "Failed to locate the temporary directory: " << error_code.message() << std::endl;
        return EXIT_FAILURE;
    }

    std::string test_home_template = (temp_dir / "infinity-unit-test.XXXXXX").string();
    std::vector<char> writable_template(test_home_template.begin(), test_home_template.end());
    writable_template.push_back('\0');
    char *test_home_ptr = ::mkdtemp(writable_template.data());
    if (test_home_ptr == nullptr) {
        std::perror("Failed to create INFINITY_TEST_HOME");
        return EXIT_FAILURE;
    }

    const std::filesystem::path test_home(test_home_ptr);
    if (::setenv("INFINITY_TEST_HOME", test_home.c_str(), 1) != 0) {
        std::perror("Failed to set INFINITY_TEST_HOME");
        std::filesystem::remove_all(test_home, error_code);
        return EXIT_FAILURE;
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __APPLE__
    std::filesystem::remove_all(test_home, error_code);
    if (error_code) {
        std::cerr << "Failed to remove INFINITY_TEST_HOME " << test_home << ": " << error_code.message() << std::endl;
        if (result == EXIT_SUCCESS) {
            result = EXIT_FAILURE;
        }
    }
#endif

    return result;
}
