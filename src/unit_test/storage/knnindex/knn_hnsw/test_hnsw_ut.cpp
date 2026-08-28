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

#include "unit_test/gtest_expand.h"

module infinity_core:ut.test_hnsw;

import :ut.base_test;
import :hnsw_alg;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
import :data_store;
#pragma clang diagnostic pop
import :dist_func_l2;
import :dist_func_ip;
import :dist_func_cos;
import :vec_store_type;
import :hnsw_common;
import :infinity_exception;
import :virtual_store;
import :local_file_handle;

using namespace infinity;

class HnswAlgTest : public BaseTest {
public:
    using LabelT = u64;

    const std::string save_dir_ = GetFullTmpDir();

    constexpr static i32 ef_search_ = 10;

    template <typename Hnsw>
    void TestSimple() {

        int dim = 16;
        int M = 8;
        int ef_construction = 200;
        int chunk_size = 128;
        int max_chunk_n = 10;
        int element_size = max_chunk_n * chunk_size;

        std::mt19937 rng;
        rng.seed(0);
        std::uniform_real_distribution<float> distrib_real;

        auto data = std::make_unique<float[]>(dim * element_size);
        for (int i = 0; i < dim * element_size; ++i) {
            data[i] = distrib_real(rng);
        }

        auto test_func = [&](auto &hnsw_index) {
            // std::fstream os("./tmp/dump.txt");
            // hnsw_index->Dump(os);
            // os.flush();
            hnsw_index->Check();

            KnnSearchOption search_option{.ef_ = ef_search_};
            int correct = 0;
            for (int i = 0; i < element_size; ++i) {
                const float *query = data.get() + i * dim;
                auto result = hnsw_index->KnnSearchSorted(query, ef_search_, search_option);
                for (auto item : result) {
                    if (item.second == (LabelT)i) {
                        ++correct;
                    }
                }
            }
            float correct_rate = float(correct) / element_size;
            std::printf("correct rate: %f\n", correct_rate);
            EXPECT_GE(correct_rate, 0.95);
        };

        std::string filepath = save_dir_ + "/test_hnsw.bin";
        {
            auto hnsw_index = Hnsw::Make(chunk_size, max_chunk_n, dim, M, ef_construction);
            auto iter = DenseVectorIter<float, LabelT>(data.get(), dim, element_size);
            hnsw_index->InsertVecs(std::move(iter), {true});
            // hnsw_index->Dump(std::cout);

            test_func(hnsw_index);

            auto [file_handle, status] = VirtualStore::Open(filepath, FileAccessMode::kWrite);
            if (!status.ok()) {
                UnrecoverableError(status.message());
            }
            hnsw_index->Save(*file_handle);
        }

        {
            auto [file_handle, status] = VirtualStore::Open(filepath, FileAccessMode::kRead);
            if (!status.ok()) {
                UnrecoverableError(status.message());
            }

            auto hnsw_index = Hnsw::Load(*file_handle);

            test_func(hnsw_index);
        }
    }

    template <typename Hnsw, typename LoadHnsw>
    void TestLoad() {
        int dim = 16;
        int M = 8;
        int ef_construction = 200;
        int chunk_size = 128;
        int max_chunk_n = 10;
        int element_size = max_chunk_n * chunk_size;

        std::mt19937 rng;
        rng.seed(0);
        std::uniform_real_distribution<float> distrib_real;

        auto data = std::make_unique<float[]>(dim * element_size);
        for (int i = 0; i < dim * element_size; ++i) {
            data[i] = distrib_real(rng);
        }

        auto test_func = [&](auto &hnsw_index) {
            hnsw_index->Check();

            KnnSearchOption search_option{.ef_ = ef_search_};
            int correct = 0;
            for (int i = 0; i < element_size; ++i) {
                const float *query = data.get() + i * dim;
                auto result = hnsw_index->KnnSearchSorted(query, 1, search_option);
                for (auto item : result) {
                    if (item.second == (LabelT)i) {
                        ++correct;
                    }
                }
            }
            float correct_rate = float(correct) / element_size;
            std::printf("correct rate: %f\n", correct_rate);
            EXPECT_GE(correct_rate, 0.95);
        };

        std::string filepath = save_dir_ + "/test_hnsw.bin";
        {
            auto hnsw_index = Hnsw::Make(chunk_size, max_chunk_n, dim, M, ef_construction);
            auto iter = DenseVectorIter<float, LabelT>(data.get(), dim, element_size);
            hnsw_index->InsertVecs(std::move(iter), {true});

            auto [file_handle, status] = VirtualStore::Open(filepath, FileAccessMode::kWrite);
            if (!status.ok()) {
                UnrecoverableError(status.message());
            }
            hnsw_index->SaveToPtr(*file_handle);
        }
        {
            size_t file_size = VirtualStore::GetFileSize(filepath);
#define USE_MMAP
#ifdef USE_MMAP
            unsigned char *data_ptr = nullptr;
            int ret = VirtualStore::MmapFile(filepath, data_ptr, file_size);
            if (ret < 0) {
                UnrecoverableError("mmap failed");
            }
            const char *ptr = reinterpret_cast<const char *>(data_ptr);
#else
            auto [file_handle, status] = VirtualStore::Open(filepath, FileAccessMode::kRead);
            if (!status.ok()) {
                UnrecoverableError(status.message());
            }
            auto buffer = std::make_unique<char[]>(file_size);
            file_handle->Read(buffer.get(), file_size);
            const char *ptr = buffer.get();
#endif
            auto hnsw_index = LoadHnsw::LoadFromPtr(ptr, file_size);

            test_func(hnsw_index);

#ifdef USE_MMAP
            VirtualStore::MunmapFile(filepath);
#endif
        }
        {
            size_t file_size = VirtualStore::GetFileSize(filepath);
            auto [file_handle, status] = VirtualStore::Open(filepath, FileAccessMode::kRead);
            auto hnsw_index = Hnsw::LoadFromPtr(*file_handle, file_size);

            test_func(hnsw_index);
        }
    }

    template <typename Hnsw, typename CompressedHnsw>
    void TestCompress() {
        int dim = 16;
        int M = 8;
        int ef_construction = 200;
        int chunk_size = 128;
        int max_chunk_n = 10;
        int element_size = max_chunk_n * chunk_size;

        std::mt19937 rng;
        rng.seed(0);
        std::uniform_real_distribution<float> distrib_real;

        auto data = std::make_unique<float[]>(dim * element_size);
        for (int i = 0; i < dim * element_size; ++i) {
            data[i] = distrib_real(rng);
        }

        auto test_func = [&](auto &hnsw_index) {
            hnsw_index->Check();

            KnnSearchOption search_option{.ef_ = ef_search_};

            int correct = 0;
            for (int i = 0; i < element_size; ++i) {
                const float *query = data.get() + i * dim;
                auto result = hnsw_index->KnnSearchSorted(query, 1, search_option);
                for (auto item : result) {
                    if (item.second == (LabelT)i) {
                        ++correct;
                    }
                }
            }
            float correct_rate = float(correct) / element_size;
            std::printf("correct rate: %f\n", correct_rate);
            EXPECT_GE(correct_rate, 0.95);
        };

        {
            auto hnsw_index = Hnsw::Make(chunk_size, max_chunk_n, dim, M, ef_construction);

            auto iter = DenseVectorIter<float, LabelT>(data.get(), dim, element_size);
            hnsw_index->InsertVecs(std::move(iter), {true});
            {
                // std::fstream os("./tmp/dump_1.txt", std::fstream::out);
                // hnsw_index->Dump(os);
            }
            auto compress_hnsw = std::move(*hnsw_index).CompressToLVQ();
            {
                // std::fstream os("./tmp/dump_2.txt", std::fstream::out);
                // compress_hnsw->Dump(os);
            }
            test_func(compress_hnsw);

            auto [file_handle, status] = VirtualStore::Open(save_dir_ + "/test_hnsw.bin", FileAccessMode::kWrite);
            if (!status.ok()) {
                UnrecoverableError(status.message());
            }
            compress_hnsw->Save(*file_handle);
        }
        {
            auto [file_handle, status] = VirtualStore::Open(save_dir_ + "/test_hnsw.bin", FileAccessMode::kRead);
            if (!status.ok()) {
                UnrecoverableError(status.message());
            }

            auto compress_hnsw = CompressedHnsw::Load(*file_handle);

            test_func(compress_hnsw);
        }
    }

    template <typename CompressedStoreType, typename Compress>
    void TestCompactCompressedPointerImage(const std::string &filename, Compress compress) {
        constexpr size_t dim = 9;
        constexpr size_t element_size = 3;
        constexpr size_t chunk_size = 2;
        constexpr size_t max_chunk_n = 2;
        constexpr size_t M = 2;
        constexpr size_t ef_construction = 8;
        constexpr LabelT label_base = 0x0102030405060708ULL;
        using PlainHnsw = KnnHnsw<PlainL2VecStoreType<float>, LabelT>;
        using CompressedHnsw = KnnHnsw<CompressedStoreType, LabelT>;
        using MappedHnsw = KnnHnsw<CompressedStoreType, LabelT, false>;
        using MappedMeta = typename CompressedStoreType::template Meta<false>;

        const std::array<float, dim * element_size> data{
            -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
            7.0f,  6.0f,  5.0f,  4.0f,  3.0f, 2.0f, 1.0f, 0.0f, -1.0f,
            0.5f,  1.5f,  2.5f,  3.5f,  4.5f, 5.5f, 6.5f, 7.5f, 8.5f,
        };

        auto plain = PlainHnsw::Make(chunk_size, max_chunk_n, dim, M, ef_construction);
        auto iter = DenseVectorIter<float, LabelT>(data.data(), dim, element_size, label_base);
        ASSERT_EQ(plain->StoreData(std::move(iter)), (std::pair<VertexType, VertexType>{0, element_size}));
        plain->Build(0, 1);
        plain->Build(1, 0);
        plain->Build(2, 0);
        plain->Check();

        std::unique_ptr<CompressedHnsw> compressed = compress(std::move(*plain));
        ASSERT_NE(compressed, nullptr);

        auto verify = [&](const auto &index) {
            index->Check();
            ASSERT_EQ(index->GetVecNum(), element_size);
            for (VertexType vertex = 0; vertex < static_cast<VertexType>(element_size); ++vertex) {
                EXPECT_EQ(index->GetLabel(vertex), label_base + static_cast<LabelT>(vertex));
            }
            for (size_t query = 0; query < element_size; ++query) {
                const auto result =
                    index->KnnSearchSorted(data.data() + query * dim, element_size, KnnSearchOption{.ef_ = element_size});
                ASSERT_EQ(result.size(), element_size);
                std::array<bool, element_size> labels_seen{};
                for (const auto &[distance, label] : result) {
                    EXPECT_TRUE(std::isfinite(distance));
                    ASSERT_GE(label, label_base);
                    ASSERT_LT(label, label_base + element_size);
                    labels_seen[static_cast<size_t>(label - label_base)] = true;
                }
                EXPECT_TRUE(std::ranges::all_of(labels_seen, std::identity{}));
            }
        };
        verify(compressed);

        const std::string filepath = save_dir_ + "/" + filename;
        {
            auto [file_handle, status] = VirtualStore::Open(filepath, FileAccessMode::kWrite);
            ASSERT_TRUE(status.ok()) << status.message();
            compressed->SaveToPtr(*file_handle);
        }
        size_t file_size = VirtualStore::GetFileSize(filepath);
        ASSERT_GT(file_size, 0u);

        {
            auto [file_handle, status] = VirtualStore::Open(filepath, FileAccessMode::kRead);
            ASSERT_TRUE(status.ok()) << status.message();
            auto loaded = CompressedHnsw::LoadFromPtr(*file_handle, file_size);
            verify(loaded);
        }

        u8 *mapped_bytes = nullptr;
        ASSERT_EQ(VirtualStore::MmapFile(filepath, mapped_bytes, file_size), 0);
        {
            const char *const image = reinterpret_cast<const char *>(mapped_bytes);
            HnswPointerReader section_reader(image, file_size);
            EXPECT_EQ(section_reader.Read<size_t>("test HNSW M"), M);
            EXPECT_EQ(section_reader.Read<size_t>("test HNSW ef_construction"), ef_construction);
            EXPECT_EQ(section_reader.Read<size_t>("test vector count"), element_size);
            MappedMeta mapped_meta = MappedMeta::LoadFromPtr(section_reader);
            GraphStoreMeta graph_meta = GraphStoreMeta::LoadFromPtr(section_reader);
            section_reader.ReadBytes(element_size * mapped_meta.compress_data_size(), "test compressed vectors");
            const size_t layer_sum = section_reader.Read<size_t>("test upper-layer count");
            ASSERT_EQ(layer_sum, 1u);
            const char *const serialized_graph = section_reader.current();
            ASSERT_NE(reinterpret_cast<std::uintptr_t>(serialized_graph) % alignof(void *), 0u);
            const size_t graph_size = element_size * graph_meta.level0_size();
            const size_t layers_size = layer_sum * graph_meta.levelx_size();
            section_reader.ReadBytes(graph_size, "test graph");
            const char *const serialized_layers = section_reader.current();
            ASSERT_NE(reinterpret_cast<std::uintptr_t>(serialized_layers) % alignof(VertexType), 0u);
            section_reader.ReadBytes(layers_size, "test upper layers");
            section_reader.ReadBytes(element_size * sizeof(LabelT), "test labels");
            section_reader.RequireEmpty();

            const char *cursor = image;
            auto mapped = MappedHnsw::LoadFromPtr(cursor, file_size);
            ASSERT_EQ(cursor, image + file_size);
            EXPECT_EQ(mapped->mem_usage(), graph_size + layers_size);
            verify(mapped);

            std::vector<char> image_with_trailing_byte(image, image + file_size);
            image_with_trailing_byte.push_back('\0');
            const char *trailing_cursor = image_with_trailing_byte.data();
            const char *const original_cursor = trailing_cursor;
            EXPECT_THROW(static_cast<void>(MappedHnsw::LoadFromPtr(trailing_cursor, image_with_trailing_byte.size())), std::invalid_argument);
            EXPECT_EQ(trailing_cursor, original_cursor);
        }
        ASSERT_EQ(VirtualStore::MunmapFile(filepath), 0);
    }

    template <typename Hnsw>
    void TestParallel() {
        int dim = 16;
        int M = 8;
        int ef_construction = 200;
        int chunk_size = 128;
        int max_chunk_n = 10;

        int element_size = max_chunk_n * chunk_size;

        std::mt19937 rng;
        rng.seed(0);
        std::uniform_real_distribution<float> distrib_real;

        auto data = std::make_unique<float[]>(dim * element_size);
        for (int i = 0; i < dim * element_size; ++i) {
            data[i] = distrib_real(rng);
        }

        auto hnsw_index = Hnsw::Make(chunk_size, max_chunk_n, dim, M, ef_construction);

        std::atomic<bool> stop = false;

        std::atomic<bool> starve = false;
        std::shared_mutex opt_mtx;

        auto SharedOptLck = [&]() {
            if (starve.load()) {
                starve.wait(true);
            }
            return std::shared_lock(opt_mtx);
        };
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
        auto UniqueOptLck = [&]() {
            bool old = false;
            bool success = starve.compare_exchange_strong(old, true);
            assert(success);
            auto ret = std::unique_lock(opt_mtx);
            starve.store(false);
            starve.notify_all();
            return ret;
        };
#pragma clang diagnostic pop

        auto write_thread = std::thread([&] {
            int start_i = 0, end_i = 0;
            {
                auto w_lck = UniqueOptLck();
                auto iter = DenseVectorIter<float, LabelT>(data.get(), dim, element_size / 2);
                std::tie(start_i, end_i) = hnsw_index->StoreData(std::move(iter), {true});
            }
            std::atomic<i32> idx = start_i;
            std::vector<std::thread> worker_threads;
            for (int i = 0; i < 4; ++i) {
                worker_threads.emplace_back([&] {
                    while (true) {
                        i32 i = idx.fetch_add(1);
                        if (i >= end_i) {
                            break;
                        }
                        auto r_lck = SharedOptLck();
                        hnsw_index->Build(i);
                    }
                });
            }
            for (auto &worker_thread : worker_threads) {
                worker_thread.join();
            }

            const int append_n = element_size - end_i;
            auto append_iter =
                DenseVectorIter<float, LabelT>(data.get() + end_i * dim, dim, append_n, static_cast<LabelT>(end_i));
            hnsw_index->InsertVecs(std::move(append_iter));
            {
                auto w_lck = UniqueOptLck();
                hnsw_index->Optimize();
            }
            stop.store(true);
        });
        std::vector<std::thread> read_threads;
        for (int j = 0; j < 4; ++j) {
            read_threads.emplace_back([&] {
                while (stop.load() == false) {
                    for (int i = 0; i < element_size; ++i) {
                        const float *query = data.get() + i * dim;
                        auto r_lck = SharedOptLck();
                        auto result = hnsw_index->KnnSearchSorted(query, 1);
                        // if (!result.empty()) {
                        //     EXPECT_EQ(result[0].second, (LabelT)i);
                        // }
                    }
                }
            });
        }
        write_thread.join();
        for (auto &t : read_threads) {
            t.join();
        }
        EXPECT_EQ(hnsw_index->GetVecNum(), static_cast<size_t>(element_size));
        EXPECT_FALSE(hnsw_index->IsBuildFailed());
        hnsw_index->Check();
    }
};

TEST_F(HnswAlgTest, test_plain_1) {
    // NOTE: inner product correct rate is not 1. (the vector and itself's distance is not the smallest)
    using Hnsw = KnnHnsw<PlainL2VecStoreType<float>, LabelT>;
    TestSimple<Hnsw>();
}

TEST_F(HnswAlgTest, test_plain_2) {
    using Hnsw = KnnHnsw<PlainCosVecStoreType<float>, LabelT>;
    TestSimple<Hnsw>();
}

TEST_F(HnswAlgTest, test_plain_3) {
    using Hnsw = KnnHnsw<PlainL2VecStoreType<float>, LabelT>;
    TestParallel<Hnsw>();
}

TEST_F(HnswAlgTest, test_plain_4) {
    using Hnsw = KnnHnsw<PlainL2VecStoreType<float>, LabelT>;
    using HnswLoad = KnnHnsw<PlainL2VecStoreType<float>, LabelT, false>;
    TestLoad<Hnsw, HnswLoad>();
}

TEST_F(HnswAlgTest, test_lvq_1) {
    using Hnsw = KnnHnsw<LVQL2VecStoreType<float, int8_t>, LabelT>;
    TestSimple<Hnsw>();
}

TEST_F(HnswAlgTest, test_lvq_2) {
    using Hnsw = KnnHnsw<LVQL2VecStoreType<float, int8_t>, LabelT>;
    TestParallel<Hnsw>();
}

TEST_F(HnswAlgTest, test_lvq_3) {
    using Hnsw = KnnHnsw<LVQL2VecStoreType<float, int8_t>, LabelT>;
    using HnswLoad = KnnHnsw<LVQL2VecStoreType<float, int8_t>, LabelT, false>;
    TestLoad<Hnsw, HnswLoad>();
}

TEST_F(HnswAlgTest, test_lvq_4) {
    using Hnsw = KnnHnsw<PlainL2VecStoreType<float>, LabelT>;
    using CompressedHnsw = KnnHnsw<LVQL2VecStoreType<float, int8_t>, LabelT>;
    TestCompress<Hnsw, CompressedHnsw>();
}

TEST_F(HnswAlgTest, test_rabitq_1) {
    using Hnsw = KnnHnsw<RabitqL2VecStoreType<float>, LabelT>;
    TestSimple<Hnsw>();
}

TEST_F(HnswAlgTest, test_rabitq_2) {
    using Hnsw = KnnHnsw<RabitqL2VecStoreType<float>, LabelT>;
    using HnswLoad = KnnHnsw<RabitqL2VecStoreType<float>, LabelT, false>;
    TestLoad<Hnsw, HnswLoad>();
}

TEST_F(HnswAlgTest, test_rabitq_3) {
    using Hnsw = KnnHnsw<RabitqL2VecStoreType<float>, LabelT>;
    TestParallel<Hnsw>();
}

TEST_F(HnswAlgTest, compact_lvq_pointer_image_aligns_graph_sidecars) {
    using Store = LVQL2VecStoreType<float, i8>;
    TestCompactCompressedPointerImage<Store>(
        "compact_lvq_pointer_image.bin",
        [](auto &&plain) { return std::move(plain).CompressToLVQ(); });
}

TEST_F(HnswAlgTest, compact_rabitq_pointer_image_aligns_graph_sidecars) {
    using Store = RabitqL2VecStoreType<float>;
    TestCompactCompressedPointerImage<Store>(
        "compact_rabitq_pointer_image.bin",
        [](auto &&plain) { return std::move(plain).CompressToRabitq(); });
}
