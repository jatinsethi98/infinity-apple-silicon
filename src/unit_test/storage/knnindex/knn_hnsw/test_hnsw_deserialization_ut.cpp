// Copyright(C) 2026 InfiniFlow, Inc. All rights reserved.
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

module infinity_core:ut.test_hnsw_deserialization;

import :ut.base_test;
import :data_store;
import :graph_store;
import :hnsw_alg;
import :hnsw_common;
import :local_file_handle;
import :lvq_vec_store;
import :rabitq_vec_store;
import :sparse_util;
import :sparse_vec_store;
import :vec_store_type;
import :virtual_store;

import std;

using namespace infinity;

class HnswDeserializationTest : public BaseTest {
protected:
    using PlainHnsw = KnnHnsw<PlainL2VecStoreType<f32>, u64>;
    using MappedPlainHnsw = KnnHnsw<PlainL2VecStoreType<f32>, u64, false>;

    const std::string test_dir_ = GetFullTmpDir();

    void SetUp() override { ASSERT_TRUE(VirtualStore::CleanupDirectory(test_dir_).ok()); }

    void TearDown() override { EXPECT_TRUE(VirtualStore::CleanupDirectory(test_dir_).ok()); }

    static void WriteFile(const std::string &path, const std::vector<char> &bytes, size_t size) {
        ASSERT_LE(size, bytes.size());
        ASSERT_TRUE(VirtualStore::Truncate(path, 0).ok());
        auto [file_handle, status] = VirtualStore::Open(path, FileAccessMode::kWrite);
        ASSERT_TRUE(status.ok()) << status.message();
        ASSERT_TRUE(file_handle->Append(bytes.data(), size).ok());
    }

    static std::vector<char> ReadFile(const std::string &path) {
        const size_t size = VirtualStore::GetFileSize(path);
        std::vector<char> bytes(size);
        auto [file_handle, status] = VirtualStore::Open(path, FileAccessMode::kRead);
        if (!status.ok()) {
            throw std::runtime_error(status.message());
        }
        auto [read_size, read_status] = file_handle->Read(bytes.data(), size);
        if (!read_status.ok() || read_size != size) {
            throw std::runtime_error("Failed to read complete HNSW test fixture");
        }
        return bytes;
    }

    template <typename T>
    static void Patch(std::vector<char> &bytes, size_t offset, const T &value) {
        ASSERT_LE(offset + sizeof(T), bytes.size());
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }

    template <typename T>
    static T Read(const std::vector<char> &bytes, size_t offset) {
        if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
            throw std::out_of_range("HNSW test fixture offset is out of range");
        }
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    }

    template <typename Callback>
    static void ExpectInvalidArgument(Callback &&callback, std::string_view expected_detail) {
        try {
            callback();
            FAIL() << "Expected std::invalid_argument containing: " << expected_detail;
        } catch (const std::invalid_argument &error) {
            EXPECT_NE(std::string_view(error.what()).find(expected_detail), std::string_view::npos) << error.what();
        } catch (const std::exception &error) {
            FAIL() << "Expected std::invalid_argument, got std::exception: " << error.what();
        } catch (...) {
            FAIL() << "Expected std::invalid_argument, got a non-standard exception";
        }
    }

    template <typename Loader>
    static void ExpectInvalidStream(const std::string &path, const std::vector<char> &bytes, Loader &&loader) {
        WriteFile(path, bytes, bytes.size());
        EXPECT_THROW(loader(path), std::invalid_argument);
    }

    template <typename Meta>
    static void ExpectPointerZeroDimensionRejected(std::string_view expected_detail) {
        std::array<char, sizeof(size_t)> bytes{};
        HnswPointerReader reader(bytes.data(), bytes.size());
        ExpectInvalidArgument([&] { static_cast<void>(Meta::LoadFromPtr(reader)); }, expected_detail);
    }

    static std::vector<char>
    SavePlainIndex(const std::string &path, const std::array<LayerSize, 3> &levels, bool pointer_image) {
        constexpr size_t dim = 3;
        constexpr size_t vec_n = levels.size();
        const std::array<f32, dim * vec_n> vectors = {
            0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
        };

        auto index = PlainHnsw::Make(4, 1, dim, 2, 8);
        static_cast<void>(index->StoreData(DenseVectorIter<f32, u64>(vectors.data(), dim, vec_n)));
        for (VertexType vertex = 0; static_cast<size_t>(vertex) < vec_n; ++vertex) {
            index->Build(vertex, levels[static_cast<size_t>(vertex)]);
        }
        index->Check();

        if (!VirtualStore::Truncate(path, 0).ok()) {
            throw std::runtime_error("Failed to truncate HNSW test fixture");
        }
        auto [file_handle, status] = VirtualStore::Open(path, FileAccessMode::kWrite);
        if (!status.ok()) {
            throw std::runtime_error(status.message());
        }
        if (pointer_image) {
            index->SaveToPtr(*file_handle);
        } else {
            index->Save(*file_handle);
        }
        return ReadFile(path);
    }

    struct PlainImageLayout {
        size_t vector_count;
        size_t graph_offset;
        size_t upper_layers_offset;
        size_t level0_size;
        size_t levelx_size;
        size_t mmax0;
        size_t mmax;
        size_t upper_layer_count;
        VertexType entry_point;
    };

    static PlainImageLayout ParsePlainImage(const std::vector<char> &bytes, bool conventional) {
        HnswPointerReader reader(bytes.data(), bytes.size());
        static_cast<void>(reader.Read<size_t>("test HNSW M"));
        static_cast<void>(reader.Read<size_t>("test HNSW ef_construction"));
        if (conventional) {
            static_cast<void>(reader.Read<size_t>("test chunk size"));
            static_cast<void>(reader.Read<size_t>("test chunk count"));
        }
        const size_t vector_count = reader.Read<size_t>("test vector count");
        using Meta = PlainL2VecStoreType<f32>::Meta<true>;
        const Meta vec_meta = Meta::LoadFromPtr(reader);
        GraphStoreMeta graph_meta = GraphStoreMeta::LoadFromPtr(reader);
        const auto [max_layer, entry_point] = graph_meta.GetEnterPoint();
        static_cast<void>(max_layer);
        reader.ReadBytes(vector_count * vec_meta.GetVecSizeInBytes(), "test vectors");
        const size_t upper_layer_count = reader.Read<size_t>("test upper-layer count");
        const size_t graph_offset = static_cast<size_t>(reader.current() - bytes.data());
        reader.ReadBytes(vector_count * graph_meta.level0_size(), "test level-zero graph");
        const size_t upper_layers_offset = static_cast<size_t>(reader.current() - bytes.data());
        reader.ReadBytes(upper_layer_count * graph_meta.levelx_size(), "test upper-layer graph");
        reader.ReadBytes(vector_count * sizeof(u64), "test labels");
        reader.RequireEmpty();
        return {
            .vector_count = vector_count,
            .graph_offset = graph_offset,
            .upper_layers_offset = upper_layers_offset,
            .level0_size = graph_meta.level0_size(),
            .levelx_size = graph_meta.levelx_size(),
            .mmax0 = graph_meta.Mmax0(),
            .mmax = graph_meta.Mmax(),
            .upper_layer_count = upper_layer_count,
            .entry_point = entry_point,
        };
    }

    static void
    ExpectInvalidPointerImage(const std::string &path, const std::vector<char> &bytes, std::string_view expected_detail) {
        WriteFile(path, bytes, bytes.size());
        auto [file_handle, status] = VirtualStore::Open(path, FileAccessMode::kRead);
        ASSERT_TRUE(status.ok()) << status.message();
        ExpectInvalidArgument(
            [&] { static_cast<void>(PlainHnsw::LoadFromPtr(*file_handle, bytes.size())); }, expected_detail);

        const char *cursor = bytes.data();
        const char *const original = cursor;
        ExpectInvalidArgument(
            [&] { static_cast<void>(MappedPlainHnsw::LoadFromPtr(cursor, bytes.size())); }, expected_detail);
        EXPECT_EQ(cursor, original);
    }
};

TEST_F(HnswDeserializationTest, ConventionalRejectsDisconnectedPersistedTopology) {
    std::vector<char> bytes =
        SavePlainIndex(test_dir_ + "/conventional-connected.bin", std::array<LayerSize, 3>{0, 0, 0}, false);
    const PlainImageLayout layout = ParsePlainImage(bytes, true);
    ASSERT_EQ(layout.vector_count, 3u);
    constexpr VertexType isolated = 2;
    ASSERT_NE(layout.entry_point, isolated);

    struct LevelZeroRecordPrefix {
        LayerSize layer;
        char *layers;
        VertexListSize degree;
    };
    constexpr size_t degree_offset = offsetof(LevelZeroRecordPrefix, degree);
    constexpr size_t neighbors_offset = degree_offset + sizeof(VertexListSize);

    for (VertexType source = 0; static_cast<size_t>(source) < layout.vector_count; ++source) {
        const size_t record_offset = layout.graph_offset + static_cast<size_t>(source) * layout.level0_size;
        const VertexListSize degree = Read<VertexListSize>(bytes, record_offset + degree_offset);
        ASSERT_GE(degree, 0);
        ASSERT_LE(static_cast<size_t>(degree), layout.mmax0);

        std::vector<VertexType> retained;
        if (source != isolated) {
            for (VertexListSize index = 0; index < degree; ++index) {
                const VertexType neighbor =
                    Read<VertexType>(bytes, record_offset + neighbors_offset + static_cast<size_t>(index) * sizeof(VertexType));
                if (neighbor != isolated) {
                    retained.push_back(neighbor);
                }
            }
        }
        Patch(bytes, record_offset + degree_offset, static_cast<VertexListSize>(retained.size()));
        for (size_t index = 0; index < retained.size(); ++index) {
            Patch(bytes, record_offset + neighbors_offset + index * sizeof(VertexType), retained[index]);
        }
    }

    const std::string invalid_path = test_dir_ + "/conventional-disconnected.bin";
    WriteFile(invalid_path, bytes, bytes.size());
    auto [file_handle, status] = VirtualStore::Open(invalid_path, FileAccessMode::kRead);
    ASSERT_TRUE(status.ok()) << status.message();
    ExpectInvalidArgument(
        [&] { static_cast<void>(PlainHnsw::Load(*file_handle)); },
        "graph level-zero topology is disconnected from its entry point");
}

TEST_F(HnswDeserializationTest, ConventionalTopLevelRejectsTrailingBytes) {
    std::vector<char> bytes =
        SavePlainIndex(test_dir_ + "/conventional-no-trailing.bin", std::array<LayerSize, 3>{0, 0, 0}, false);
    bytes.push_back('\0');

    const std::string invalid_path = test_dir_ + "/conventional-trailing.bin";
    WriteFile(invalid_path, bytes, bytes.size());
    auto [file_handle, status] = VirtualStore::Open(invalid_path, FileAccessMode::kRead);
    ASSERT_TRUE(status.ok()) << status.message();
    ExpectInvalidArgument(
        [&] { static_cast<void>(PlainHnsw::Load(*file_handle)); }, "stream contains trailing bytes");
}

TEST_F(HnswDeserializationTest, PointerMetadataRejectsZeroDimensions) {
    using PlainMeta = PlainL2VecStoreType<f32>::Meta<true>;
    using OwnedLVQMeta = LVQL2VecStoreType<f32, i8>::Meta<true>;
    using MappedLVQMeta = LVQL2VecStoreType<f32, i8>::Meta<false>;
    using OwnedRabitqMeta = RabitqL2VecStoreType<f32>::Meta<true>;
    using MappedRabitqMeta = RabitqL2VecStoreType<f32>::Meta<false>;

    ExpectPointerZeroDimensionRejected<PlainMeta>("plain vector dimension must be nonzero");
    ExpectPointerZeroDimensionRejected<OwnedLVQMeta>("LVQ dimension must be nonzero");
    ExpectPointerZeroDimensionRejected<MappedLVQMeta>("LVQ dimension must be nonzero");
    ExpectPointerZeroDimensionRejected<OwnedRabitqMeta>("Rabitq dimension must be nonzero");
    ExpectPointerZeroDimensionRejected<MappedRabitqMeta>("Rabitq dimension must be nonzero");
}

TEST_F(HnswDeserializationTest, PointerRejectsMalformedUpperLayerRecordsBeforeCheck) {
    const std::vector<char> valid =
        SavePlainIndex(test_dir_ + "/pointer-upper-valid.bin", std::array<LayerSize, 3>{1, 1, 0}, true);
    const PlainImageLayout layout = ParsePlainImage(valid, false);
    ASSERT_GE(layout.upper_layer_count, 2u);

    constexpr size_t upper_degree_offset = 0;
    constexpr size_t upper_neighbors_offset = sizeof(VertexListSize);
    const VertexListSize valid_degree = Read<VertexListSize>(valid, layout.upper_layers_offset + upper_degree_offset);
    ASSERT_GT(valid_degree, 0);
    ASSERT_LE(static_cast<size_t>(valid_degree), layout.mmax);

    {
        std::vector<char> malformed = valid;
        Patch(malformed,
              layout.upper_layers_offset + upper_degree_offset,
              static_cast<VertexListSize>(layout.mmax + 1));
        ExpectInvalidPointerImage(
            test_dir_ + "/pointer-upper-degree.bin", malformed, "graph upper-layer degree exceeds capacity");
    }
    {
        std::vector<char> malformed = valid;
        Patch(malformed,
              layout.upper_layers_offset + upper_neighbors_offset,
              static_cast<VertexType>(layout.vector_count));
        ExpectInvalidPointerImage(
            test_dir_ + "/pointer-upper-edge.bin", malformed, "graph contains an out-of-range upper-layer edge");
    }
    {
        std::vector<char> malformed = valid;
        Patch(malformed, layout.upper_layers_offset + upper_neighbors_offset, VertexType{2});
        ExpectInvalidPointerImage(
            test_dir_ + "/pointer-upper-target-level.bin",
            malformed,
            "graph upper-layer edge targets a lower-level vertex");
    }
}

TEST_F(HnswDeserializationTest, ConventionalRejectsEveryTruncatedPrefixAndRecovers) {
    using Hnsw = KnnHnsw<PlainL2VecStoreType<f32>, u64>;

    constexpr size_t dim = 3;
    constexpr size_t vec_n = 3;
    const std::array<f32, dim * vec_n> vectors = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };

    const std::string valid_path = test_dir_ + "/conventional-valid.bin";
    {
        auto index = Hnsw::Make(2, 2, dim, 2, 8);
        index->InsertVecs(DenseVectorIter<f32, u64>(vectors.data(), dim, vec_n), {true});
        auto [file_handle, status] = VirtualStore::Open(valid_path, FileAccessMode::kWrite);
        ASSERT_TRUE(status.ok()) << status.message();
        index->Save(*file_handle);
    }

    const std::vector<char> valid = ReadFile(valid_path);
    ASSERT_GT(valid.size(), 1u);
    const std::string truncated_path = test_dir_ + "/conventional-truncated.bin";
    for (size_t prefix = 0; prefix < valid.size(); ++prefix) {
        SCOPED_TRACE(prefix);
        WriteFile(truncated_path, valid, prefix);
        auto [file_handle, status] = VirtualStore::Open(truncated_path, FileAccessMode::kRead);
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_THROW(static_cast<void>(Hnsw::Load(*file_handle)), std::invalid_argument);
    }

    WriteFile(truncated_path, valid, valid.size());
    auto [file_handle, status] = VirtualStore::Open(truncated_path, FileAccessMode::kRead);
    ASSERT_TRUE(status.ok()) << status.message();
    auto loaded = Hnsw::Load(*file_handle);
    ASSERT_NE(loaded, nullptr);
    loaded->Check();
}

TEST_F(HnswDeserializationTest, LVQRejectsInvalidMetadataAndTruncatedPayload) {
    using VecStore = LVQL2VecStoreType<f32, i8>;
    using Meta = VecStore::Meta<true>;
    using Inner = VecStore::Inner<true>;

    constexpr size_t dim = 5;
    constexpr size_t vec_n = 2;
    Meta meta = Meta::Make(dim);
    size_t mem_usage = 0;
    Inner inner = Inner::Make(vec_n, meta, mem_usage);
    const std::array<f32, dim> first = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const std::array<f32, dim> second = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    inner.SetVec(0, first.data(), meta, mem_usage);
    inner.SetVec(1, second.data(), meta, mem_usage);

    const std::string fixture_path = test_dir_ + "/lvq-fixture.bin";
    {
        auto [file_handle, status] = VirtualStore::Open(fixture_path, FileAccessMode::kWrite);
        ASSERT_TRUE(status.ok()) << status.message();
        meta.Save(*file_handle);
        inner.Save(*file_handle, vec_n, meta);
    }
    const std::vector<char> fixture = ReadFile(fixture_path);
    const auto load = [&](const std::string &path) {
        auto [file_handle, status] = VirtualStore::Open(path, FileAccessMode::kRead);
        if (!status.ok()) {
            throw std::runtime_error(status.message());
        }
        Meta loaded_meta = Meta::Load(*file_handle);
        size_t loaded_mem_usage = 0;
        static_cast<void>(Inner::Load(*file_handle, vec_n, vec_n, loaded_meta, loaded_mem_usage));
    };

    {
        std::vector<char> invalid = fixture;
        Patch(invalid, 0, size_t{0});
        ExpectInvalidStream(test_dir_ + "/lvq-zero-dimension.bin", invalid, load);
    }
    {
        std::vector<char> truncated_mean(fixture.begin(), fixture.begin() + sizeof(size_t) + dim * sizeof(f32) - 1);
        ExpectInvalidStream(test_dir_ + "/lvq-truncated-mean.bin", truncated_mean, load);
    }
    {
        std::vector<char> truncated_payload(fixture.begin(), fixture.end() - 1);
        ExpectInvalidStream(test_dir_ + "/lvq-truncated-payload.bin", truncated_payload, load);
    }
}

TEST_F(HnswDeserializationTest, RabitQRejectsAlignmentOverflowAndTruncatedPayload) {
    using VecStore = RabitqL2VecStoreType<f32>;
    using Meta = VecStore::Meta<true>;
    using Inner = VecStore::Inner<true>;

    constexpr size_t origin_dim = 9;
    constexpr size_t vec_n = 2;
    Meta meta = Meta::Make(origin_dim);
    size_t mem_usage = 0;
    Inner inner = Inner::Make(vec_n, meta, mem_usage);
    std::array<f32, origin_dim> vector{};
    std::iota(vector.begin(), vector.end(), 1.0f);
    inner.SetVec(0, vector.data(), meta, mem_usage);
    std::ranges::reverse(vector);
    inner.SetVec(1, vector.data(), meta, mem_usage);

    const std::string fixture_path = test_dir_ + "/rabitq-fixture.bin";
    {
        auto [file_handle, status] = VirtualStore::Open(fixture_path, FileAccessMode::kWrite);
        ASSERT_TRUE(status.ok()) << status.message();
        meta.Save(*file_handle);
        inner.Save(*file_handle, vec_n, meta);
    }
    const std::vector<char> fixture = ReadFile(fixture_path);
    const auto load = [&](const std::string &path) {
        auto [file_handle, status] = VirtualStore::Open(path, FileAccessMode::kRead);
        if (!status.ok()) {
            throw std::runtime_error(status.message());
        }
        Meta loaded_meta = Meta::Load(*file_handle);
        size_t loaded_mem_usage = 0;
        static_cast<void>(Inner::Load(*file_handle, vec_n, vec_n, loaded_meta, loaded_mem_usage));
    };

    {
        std::vector<char> invalid = fixture;
        Patch(invalid, 0, std::numeric_limits<size_t>::max());
        ExpectInvalidStream(test_dir_ + "/rabitq-alignment-overflow.bin", invalid, load);
    }
    {
        const size_t metadata_size = sizeof(size_t) + meta.dim() * meta.dim() * sizeof(f32) + meta.dim() * sizeof(f32);
        std::vector<char> truncated_matrix(fixture.begin(), fixture.begin() + metadata_size - 1);
        ExpectInvalidStream(test_dir_ + "/rabitq-truncated-matrix.bin", truncated_matrix, load);
    }
    {
        std::vector<char> truncated_payload(fixture.begin(), fixture.end() - 1);
        ExpectInvalidStream(test_dir_ + "/rabitq-truncated-payload.bin", truncated_payload, load);
    }
}

TEST_F(HnswDeserializationTest, SparseRejectsInvalidMetadataOffsetsAndIndices) {
    using Meta = SparseVecStoreMeta<f32, i32>;
    using Inner = SparseVecStoreInner<f32, i32>;

    constexpr size_t dim = 8;
    constexpr size_t vec_n = 2;
    const std::array<i32, 2> first_indices = {1, 3};
    const std::array<f32, 2> first_values = {1.0f, 3.0f};
    const std::array<i32, 1> second_indices = {2};
    const std::array<f32, 1> second_values = {2.0f};

    Meta meta = Meta::Make(dim);
    size_t mem_usage = 0;
    Inner inner = Inner::Make(vec_n, meta, mem_usage);
    inner.SetVec(0, SparseVecRef<f32, i32>(first_indices.size(), first_indices.data(), first_values.data()), meta, mem_usage);
    inner.SetVec(1, SparseVecRef<f32, i32>(second_indices.size(), second_indices.data(), second_values.data()), meta, mem_usage);

    const std::string fixture_path = test_dir_ + "/sparse-fixture.bin";
    {
        auto [file_handle, status] = VirtualStore::Open(fixture_path, FileAccessMode::kWrite);
        ASSERT_TRUE(status.ok()) << status.message();
        meta.Save(*file_handle);
        inner.Save(*file_handle, vec_n, meta);
    }
    const std::vector<char> fixture = ReadFile(fixture_path);
    const auto load = [&](const std::string &path) {
        auto [file_handle, status] = VirtualStore::Open(path, FileAccessMode::kRead);
        if (!status.ok()) {
            throw std::runtime_error(status.message());
        }
        Meta loaded_meta = Meta::Load(*file_handle);
        size_t loaded_mem_usage = 0;
        static_cast<void>(Inner::Load(*file_handle, vec_n, vec_n, loaded_meta, loaded_mem_usage));
    };

    constexpr size_t nnz_offset = sizeof(size_t);
    constexpr size_t indptr_offset = nnz_offset + sizeof(size_t);
    constexpr size_t indices_offset = indptr_offset + (vec_n + 1) * sizeof(i32);

    {
        std::vector<char> invalid = fixture;
        Patch(invalid, 0, size_t{0});
        ExpectInvalidStream(test_dir_ + "/sparse-zero-dimension.bin", invalid, load);
    }
    {
        std::vector<char> invalid = fixture;
        Patch(invalid, nnz_offset, static_cast<size_t>(std::numeric_limits<i32>::max()) + 1);
        ExpectInvalidStream(test_dir_ + "/sparse-nnz-overflow.bin", invalid, load);
    }
    {
        std::vector<char> invalid = fixture;
        Patch(invalid, indptr_offset, i32{1});
        ExpectInvalidStream(test_dir_ + "/sparse-indptr-origin.bin", invalid, load);
    }
    {
        std::vector<char> invalid = fixture;
        Patch(invalid, indptr_offset + sizeof(i32), i32{4});
        ExpectInvalidStream(test_dir_ + "/sparse-indptr-bounds.bin", invalid, load);
    }
    {
        std::vector<char> invalid = fixture;
        Patch(invalid, indices_offset, i32{static_cast<i32>(dim)});
        ExpectInvalidStream(test_dir_ + "/sparse-index-bounds.bin", invalid, load);
    }
    {
        std::vector<char> invalid = fixture;
        Patch(invalid, indices_offset + sizeof(i32), first_indices.front());
        ExpectInvalidStream(test_dir_ + "/sparse-index-order.bin", invalid, load);
    }
    {
        std::vector<char> truncated(fixture.begin(), fixture.end() - 1);
        ExpectInvalidStream(test_dir_ + "/sparse-truncated-payload.bin", truncated, load);
    }
}
