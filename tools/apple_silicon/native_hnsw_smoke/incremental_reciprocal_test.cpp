import std;
import std.compat;
import infinity_core;

namespace {

using Label = infinity::u64;
using PlainStore = infinity::PlainL2VecStoreType<float>;
using Hnsw = infinity::KnnHnsw<PlainStore, Label>;
using MappedHnsw = infinity::KnnHnsw<PlainStore, Label, false>;

constexpr std::size_t kDimension = 4;
constexpr std::size_t kGraphM = 32;
constexpr std::size_t kGraphEfConstruction = 64;

#if (defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)) || \
    defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)

using CampaignLabel = infinity::i32;
using CampaignHnsw = infinity::KnnHnsw<PlainStore, CampaignLabel>;

constexpr std::size_t kCampaignCount = 12'288;
constexpr std::size_t kCampaignDimension = 128;
constexpr std::size_t kCampaignChunkSize = 8'192;
constexpr std::size_t kCampaignMaxChunks = 2;
constexpr std::size_t kCampaignM = 32;
constexpr std::size_t kCampaignEfConstruction = 200;

static_assert((kCampaignCount - 1) / kCampaignChunkSize + 1 == kCampaignMaxChunks);

#endif

std::uint64_t HashByte(std::uint64_t hash, std::uint8_t byte) { return (hash ^ byte) * 1'099'511'628'211ULL; }

template <typename Integer>
std::uint64_t HashInteger(std::uint64_t hash, Integer value) {
    using Unsigned = std::make_unsigned_t<Integer>;
    const Unsigned bits = static_cast<Unsigned>(value);
    for (std::size_t shift = 0; shift < sizeof(Unsigned) * 8; shift += 8) {
        hash = HashByte(hash, static_cast<std::uint8_t>((bits >> shift) & 0xffU));
    }
    return hash;
}

std::uint64_t HashBytes(std::span<const char> bytes) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (char byte : bytes) {
        hash = HashByte(hash, static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    return hash;
}

template <typename Index>
std::uint64_t GraphHash(const Index &index) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    const auto [max_layer, entry_point] = index.GetGraphEnterPoint();
    hash = HashInteger(hash, max_layer);
    hash = HashInteger(hash, entry_point);
    hash = HashInteger(hash, index.GetVecNum());
    for (std::size_t vertex = 0; vertex < index.GetVecNum(); ++vertex) {
        const auto vertex_id = static_cast<infinity::VertexType>(vertex);
        const infinity::LayerSize level = index.GetGraphLevel(vertex_id);
        hash = HashInteger(hash, level);
        for (infinity::LayerSize layer = 0; layer <= level; ++layer) {
            const auto [neighbors, degree] = index.GetGraphNeighbors(vertex_id, layer);
            hash = HashInteger(hash, degree);
            for (infinity::VertexListSize i = 0; i < degree; ++i) {
                hash = HashInteger(hash, neighbors[i]);
            }
        }
    }
    return hash;
}

template <typename Index>
std::vector<char> SerializePointerImage(const Index &index) {
    infinity::LocalFileHandle file;
    index.SaveToPtr(file);
    std::vector<char> bytes(file.Size());
    file.Rewind();
    if (!bytes.empty()) {
        file.Read(bytes.data(), bytes.size());
    }
    return bytes;
}

template <typename Left, typename Right>
bool OrderedGraphsEqual(const Left &left, const Right &right) {
    if (left.GetVecNum() != right.GetVecNum() || left.GetGraphEnterPoint() != right.GetGraphEnterPoint()) {
        return false;
    }
    for (std::size_t vertex = 0; vertex < left.GetVecNum(); ++vertex) {
        const auto id = static_cast<infinity::VertexType>(vertex);
        const infinity::LayerSize level = left.GetGraphLevel(id);
        if (level != right.GetGraphLevel(id)) {
            return false;
        }
        for (infinity::LayerSize layer = 0; layer <= level; ++layer) {
            const auto [left_neighbors, left_count] = left.GetGraphNeighbors(id, layer);
            const auto [right_neighbors, right_count] = right.GetGraphNeighbors(id, layer);
            if (left_count != right_count ||
                !std::equal(left_neighbors, left_neighbors + left_count, right_neighbors, right_neighbors + right_count)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<float> MakeData(std::size_t count, std::size_t dimension = kDimension) {
    std::vector<float> data(count * dimension);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            data[row * dimension + column] = static_cast<float>((row * 37 + column * 17) % 251) / 251.0F;
        }
    }
    return data;
}

std::vector<float> MakeCertificateData(std::size_t count, std::size_t dimension = kDimension) {
    std::vector<float> data(count * dimension);
    std::uint64_t state = 0xd1b54a32d192ed03ULL;
    for (float &value : data) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const std::uint64_t bits = state * 0x2545f4914f6cdd1dULL;
        value = static_cast<float>(bits >> 40) * (1.0F / 16'777'216.0F);
    }
    return data;
}

void InsertRange(std::unique_ptr<Hnsw> &index, const std::vector<float> &data, std::size_t dimension, std::size_t start, std::size_t count) {
    infinity::DenseVectorIter<float, Label> iterator(data.data() + start * dimension, dimension, count, static_cast<Label>(start));
    static_cast<void>(index->InsertVecs(std::move(iterator)));
}

std::unique_ptr<Hnsw> MakeDeterministicGraph(std::size_t count = 192) {
    const auto data = MakeData(count);
    auto index = Hnsw::Make(std::bit_ceil(std::max<std::size_t>(256, count)), 1, kDimension, kGraphM, kGraphEfConstruction);
    InsertRange(index, data, kDimension, 0, count);
    index->Check();
    return index;
}

class Reporter {
public:
    void Check(std::string_view name, bool passed) {
        std::cout << "check_" << name << '=' << (passed ? "true" : "false") << '\n';
        passed_ = passed_ && passed;
    }

    bool passed() const { return passed_; }

private:
    bool passed_{true};
};

struct GraphLevelZeroAccountingLayout {
    infinity::LayerSize layer;
    char *upper_layers;
    infinity::VertexListSize neighbor_count;
};

std::size_t ExpectedEmptyOwnedMemory(std::size_t capacity, std::size_t dimension, std::size_t graph_m) {
    const std::size_t vector_bytes = sizeof(float) * dimension;
    const std::size_t level_zero_bytes =
        sizeof(GraphLevelZeroAccountingLayout) + sizeof(infinity::VertexType) * 2 * graph_m;
    const std::size_t owned_metadata_bytes = Hnsw::DataStore::OwnedMetadataBytesPerVertex();
    return capacity * (vector_bytes + level_zero_bytes + owned_metadata_bytes);
}

bool CheckOwnedCapacityAccounting() {
    constexpr std::size_t capacity = 64;
    constexpr std::size_t dimension = 7;
    constexpr std::size_t graph_m = 4;
    auto index = Hnsw::Make(capacity, 1, dimension, graph_m, 16);
    const std::size_t make_expected = ExpectedEmptyOwnedMemory(capacity, dimension, graph_m);

    infinity::LocalFileHandle stream;
    index->Save(stream);
    stream.Rewind();
    auto stream_loaded = Hnsw::Load(stream);

    const std::vector<char> image = SerializePointerImage(*index);
    infinity::LocalFileHandle pointer_file;
    pointer_file.Append(image.data(), image.size());
    pointer_file.Rewind();
    auto pointer_loaded = Hnsw::LoadFromPtr(pointer_file, image.size());
    const std::size_t pointer_expected = ExpectedEmptyOwnedMemory(1, dimension, graph_m);

    const bool make_exact = index->mem_usage() == make_expected;
    const bool stream_exact = stream_loaded->mem_usage() == make_expected;
    const bool pointer_exact = pointer_loaded->mem_usage() == pointer_expected;
    std::cout << "owned_accounting_make_bytes=" << index->mem_usage() << '\n';
    std::cout << "owned_accounting_stream_bytes=" << stream_loaded->mem_usage() << '\n';
    std::cout << "owned_accounting_pointer_bytes=" << pointer_loaded->mem_usage() << '\n';
    std::cout << "owned_accounting_make_exact=" << make_exact << '\n';
    std::cout << "owned_accounting_stream_exact=" << stream_exact << '\n';
    std::cout << "owned_accounting_pointer_exact=" << pointer_exact << '\n';
    return make_exact && stream_exact && pointer_exact;
}

bool CheckNullEmptyDenseVectorIterator() {
    infinity::DenseVectorIter<float, Label> iterator(nullptr, kDimension, 0, 17);
    const bool row_count_empty = iterator.GetRowCount() == 0;
    const bool next_empty = !iterator.Next();
    auto splits = std::move(iterator).split();
    const bool split_empty = splits.size() == 1 && splits.front().GetRowCount() == 0 && !splits.front().Next();
    return row_count_empty && next_empty && split_empty;
}

bool CheckDenseVectorSplitWithNarrowLabel() {
    constexpr std::size_t row_count = 1300;
    std::vector<float> data(row_count);
    std::iota(data.begin(), data.end(), 0.0F);

    infinity::DenseVectorIter<float, infinity::u8> iterator(data.data(), 1, row_count);
    auto splits = std::move(iterator).split();
    if (splits.size() != 2 || splits[0].GetRowCount() != 1024 || splits[1].GetRowCount() != 276) {
        return false;
    }

    std::size_t observed = 0;
    for (auto &split : splits) {
        for (auto value = split.Next(); value; value = split.Next()) {
            const auto [vector, label] = *value;
            if (vector != data.data() + observed || label != static_cast<infinity::u8>(observed)) {
                return false;
            }
            ++observed;
        }
    }
    return observed == row_count;
}

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)

using LsgStore = infinity::PlainL2VecStoreType<float, true>;
using LvqStore = infinity::LVQL2VecStoreType<float, infinity::i8>;
using RabitqStore = infinity::RabitqL2VecStoreType<float>;
using LsgHnsw = infinity::KnnHnsw<LsgStore, Label>;
using LvqHnsw = infinity::KnnHnsw<LvqStore, Label>;
using RabitqHnsw = infinity::KnnHnsw<RabitqStore, Label>;

template <typename Index>
concept PublicCompressedDataStoreInstaller =
    requires(Index &index, typename Index::DataStore data_store) { index.InstallCompressedDataStore(std::move(data_store)); };

static_assert(Hnsw::kIncrementalReciprocalSupported);
static_assert(!MappedHnsw::kIncrementalReciprocalSupported);
static_assert(!LsgHnsw::kIncrementalReciprocalSupported);
static_assert(!LvqHnsw::kIncrementalReciprocalSupported);
static_assert(!RabitqHnsw::kIncrementalReciprocalSupported);
static_assert(!PublicCompressedDataStoreInstaller<LvqHnsw>);
static_assert(!PublicCompressedDataStoreInstaller<RabitqHnsw>);

struct CompressionOracle {
    bool labels_preserved{};
    bool compressed_queries_valid{};
    bool serialized_bytes_exact{};
};

template <typename CompressedIndex, typename PlainIndex>
CompressionOracle CheckCompressionOracle(CompressedIndex &destination, const PlainIndex &plain_control) {
    const std::vector<char> serialized = SerializePointerImage(destination);
    infinity::LocalFileHandle pointer_file;
    pointer_file.Append(serialized.data(), serialized.size());
    pointer_file.Rewind();
    auto loaded = CompressedIndex::LoadFromPtr(pointer_file, serialized.size());

    std::vector<Label> expected_labels;
    expected_labels.reserve(plain_control.GetVecNum());
    bool labels_preserved = destination.GetVecNum() == plain_control.GetVecNum() && loaded->GetVecNum() == plain_control.GetVecNum();
    for (std::size_t vertex = 0; vertex < plain_control.GetVecNum(); ++vertex) {
        const auto vertex_id = static_cast<infinity::VertexType>(vertex);
        const Label expected = plain_control.GetLabel(vertex_id);
        expected_labels.push_back(expected);
        labels_preserved = labels_preserved && destination.GetLabel(vertex_id) == expected && loaded->GetLabel(vertex_id) == expected;
    }
    std::sort(expected_labels.begin(), expected_labels.end());
    labels_preserved = labels_preserved && std::adjacent_find(expected_labels.begin(), expected_labels.end()) == expected_labels.end();

    constexpr std::array<std::array<float, kDimension>, 2> queries{{
        {0.125F, 0.375F, 0.625F, 0.875F},
        {0.9375F, 0.6875F, 0.4375F, 0.1875F},
    }};
    const infinity::KnnSearchOption search_option{.ef_ = plain_control.GetVecNum()};
    bool compressed_queries_valid = true;
    for (const auto &query : queries) {
        auto destination_results = destination.KnnSearchSorted(query.data(), plain_control.GetVecNum(), search_option);
        auto loaded_results = loaded->KnnSearchSorted(query.data(), plain_control.GetVecNum(), search_option);
        auto validate_and_sort = [&](auto &results) {
            bool valid = results.size() == plain_control.GetVecNum();
            for (const auto &[distance, label] : results) {
                static_cast<void>(label);
                valid = valid && std::isfinite(distance);
            }
            std::sort(results.begin(), results.end(), [](const auto &left, const auto &right) {
                return left.second < right.second || (left.second == right.second && left.first < right.first);
            });
            std::vector<Label> labels;
            labels.reserve(results.size());
            for (const auto &[distance, label] : results) {
                static_cast<void>(distance);
                labels.push_back(label);
            }
            return valid && labels == expected_labels && std::adjacent_find(labels.begin(), labels.end()) == labels.end();
        };
        const bool destination_valid = validate_and_sort(destination_results);
        const bool loaded_valid = validate_and_sort(loaded_results);
        compressed_queries_valid = compressed_queries_valid && destination_valid && loaded_valid && destination_results == loaded_results;
    }

    const bool serialized_bytes_exact = SerializePointerImage(destination) == serialized && SerializePointerImage(*loaded) == serialized;
    return {
        .labels_preserved = labels_preserved,
        .compressed_queries_valid = compressed_queries_valid,
        .serialized_bytes_exact = serialized_bytes_exact,
    };
}

struct ReferenceSelection {
    std::vector<infinity::VertexType> neighbors;
    infinity::VertexListSize count{};
};

template <typename CenterDistance, typename PairDistance>
ReferenceSelection SelectNeighborsHeuristicReference(infinity::VertexType new_vertex,
                                                     float new_center_distance,
                                                     std::span<const infinity::VertexType> old_neighbors,
                                                     std::size_t capacity,
                                                     CenterDistance &&center_distance,
                                                     PairDistance &&pair_distance) {
    using Pair = std::pair<float, infinity::VertexType>;
    std::vector<Pair> candidates;
    candidates.reserve(old_neighbors.size() + 1);
    candidates.emplace_back(new_center_distance, new_vertex);
    for (infinity::VertexType old_vertex : old_neighbors) {
        candidates.emplace_back(center_distance(old_vertex), old_vertex);
    }

    const auto compare_by_first_reverse = [](const Pair &left, const Pair &right) { return left.first > right.first; };
    std::make_heap(candidates.begin(), candidates.end(), compare_by_first_reverse);

    ReferenceSelection selected;
    selected.neighbors.reserve(capacity);
    while (!candidates.empty() && selected.neighbors.size() < capacity) {
        std::pop_heap(candidates.begin(), candidates.end(), compare_by_first_reverse);
        const Pair candidate = candidates.back();
        bool accepted = true;
        for (infinity::VertexType prior : selected.neighbors) {
            if (pair_distance(candidate.second, prior) < candidate.first) {
                accepted = false;
                break;
            }
        }
        if (accepted) {
            selected.neighbors.push_back(candidate.second);
        }
        candidates.pop_back();
    }
    std::reverse(selected.neighbors.begin(), selected.neighbors.end());
    selected.count = static_cast<infinity::VertexListSize>(selected.neighbors.size());
    return selected;
}

struct MatrixCase {
    std::size_t capacity{};
    infinity::VertexType new_vertex{};
    std::vector<infinity::VertexType> old_neighbors;
    std::vector<float> center_distance;
    std::vector<float> pair_distance;

    float Center(infinity::VertexType vertex) const { return center_distance[static_cast<std::size_t>(vertex)]; }

    float Pair(infinity::VertexType from, infinity::VertexType to) const {
        const std::size_t width = capacity + 1;
        return pair_distance[static_cast<std::size_t>(from) * width + static_cast<std::size_t>(to)];
    }

    float &Pair(infinity::VertexType from, infinity::VertexType to) {
        const std::size_t width = capacity + 1;
        return pair_distance[static_cast<std::size_t>(from) * width + static_cast<std::size_t>(to)];
    }
};

MatrixCase MakeCertifiedCase(std::size_t capacity, std::size_t farther_count, std::uint64_t mask, bool equality_thresholds) {
    MatrixCase test;
    test.capacity = capacity;
    test.new_vertex = static_cast<infinity::VertexType>(capacity);
    test.old_neighbors.resize(capacity);
    test.center_distance.resize(capacity + 1);
    test.pair_distance.resize((capacity + 1) * (capacity + 1), std::numeric_limits<float>::infinity());

    for (std::size_t index = 0; index < capacity; ++index) {
        test.old_neighbors[index] = static_cast<infinity::VertexType>(index);
        test.center_distance[index] = static_cast<float>(capacity - index);
    }
    test.center_distance[capacity] = static_cast<float>(capacity - farther_count) + 0.5F;

    for (std::size_t candidate = 0; candidate < capacity; ++candidate) {
        for (std::size_t prior = candidate + 1; prior < capacity; ++prior) {
            test.Pair(static_cast<infinity::VertexType>(candidate), static_cast<infinity::VertexType>(prior)) =
                equality_thresholds ? test.center_distance[candidate] : test.center_distance[candidate] + 1.0F;
        }
    }

    for (std::size_t old_index = 0; old_index < capacity; ++old_index) {
        const bool reject = (mask & (std::uint64_t{1} << old_index)) != 0;
        const infinity::VertexType old = static_cast<infinity::VertexType>(old_index);
        if (old_index >= farther_count) {
            test.Pair(test.new_vertex, old) = reject ? test.center_distance[capacity] - 0.25F : test.center_distance[capacity] + 1.0F;
        } else {
            test.Pair(old, test.new_vertex) = reject ? test.center_distance[old_index] - 0.25F : test.center_distance[old_index] + 1.0F;
        }
    }
    return test;
}

bool CompareIncrementalToReference(const MatrixCase &test) {
    const ReferenceSelection reference = SelectNeighborsHeuristicReference(
        test.new_vertex,
        test.Center(test.new_vertex),
        test.old_neighbors,
        test.capacity,
        [&](infinity::VertexType vertex) { return test.Center(vertex); },
        [&](infinity::VertexType from, infinity::VertexType to) { return test.Pair(from, to); });

    std::vector<infinity::VertexType> actual = test.old_neighbors;
    actual.resize(test.capacity + 3, -9'999);
    const std::vector<infinity::VertexType> original = actual;
    infinity::VertexListSize actual_count = static_cast<infinity::VertexListSize>(test.capacity);
    std::vector<float> scratch(test.capacity);
    const auto result = infinity::TryIncrementalReciprocalUpdate(
        test.new_vertex,
        test.Center(test.new_vertex),
        actual.data(),
        &actual_count,
        test.capacity,
        std::span<float>(scratch),
        [&](infinity::VertexType vertex) { return test.Center(vertex); },
        [&](infinity::VertexType old) { return test.Pair(test.new_vertex, old); },
        [&](infinity::VertexType old) { return test.Pair(old, test.new_vertex); });

    infinity::HnswIncrementalReciprocalResult expected = infinity::HnswIncrementalReciprocalResult::kUpdated;
    if (test.Center(test.old_neighbors.front()) < test.Center(test.new_vertex)) {
        expected = infinity::HnswIncrementalReciprocalResult::kUnchangedNewFarthest;
    } else if (reference.count == static_cast<infinity::VertexListSize>(test.capacity) &&
               std::equal(reference.neighbors.begin(), reference.neighbors.end(), test.old_neighbors.begin())) {
        expected = infinity::HnswIncrementalReciprocalResult::kUnchangedRejected;
    }

    const bool same_neighbors = actual_count == reference.count && std::equal(reference.neighbors.begin(), reference.neighbors.end(), actual.begin());
    const bool unchanged_tail = std::equal(original.begin() + static_cast<std::ptrdiff_t>(test.capacity),
                                           original.end(),
                                           actual.begin() + static_cast<std::ptrdiff_t>(test.capacity));
    return result == expected && same_neighbors && unchanged_tail;
}

bool CheckExhaustiveCertifiedDifferential() {
    for (std::size_t capacity = 2; capacity <= 8; ++capacity) {
        const std::uint64_t mask_limit = std::uint64_t{1} << capacity;
        for (std::size_t rank = 0; rank <= capacity; ++rank) {
            for (std::uint64_t mask = 0; mask < mask_limit; ++mask) {
                if (!CompareIncrementalToReference(MakeCertifiedCase(capacity, rank, mask, false))) {
                    return false;
                }
            }
            if (!CompareIncrementalToReference(MakeCertifiedCase(capacity, rank, 0, true))) {
                return false;
            }
        }
    }
    return true;
}

bool CheckRandomizedCertifiedDifferential() {
    std::mt19937_64 generator(0x31e5c7edULL);
    for (std::size_t capacity = 2; capacity <= 64; ++capacity) {
        for (std::size_t iteration = 0; iteration < 24; ++iteration) {
            const std::size_t rank = static_cast<std::size_t>(generator() % (capacity + 1));
            MatrixCase test = MakeCertifiedCase(capacity, rank, generator(), false);
            for (std::size_t old_index = 0; old_index < capacity; ++old_index) {
                const infinity::VertexType old = static_cast<infinity::VertexType>(old_index);
                if (old_index >= rank) {
                    const bool reject = (generator() & 1U) != 0;
                    test.Pair(test.new_vertex, old) = reject ? test.Center(test.new_vertex) - 0.125F : test.Center(test.new_vertex) + 0.125F;
                } else {
                    const bool reject = (generator() & 1U) != 0;
                    test.Pair(old, test.new_vertex) = reject ? test.Center(old) - 0.125F : test.Center(old) + 0.125F;
                }
            }
            if (!CompareIncrementalToReference(test)) {
                return false;
            }
        }
    }
    return true;
}

template <typename Center, typename NewToOld, typename OldToNew>
bool ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult expected,
                             infinity::VertexType new_vertex,
                             float new_center,
                             std::vector<infinity::VertexType> neighbors,
                             infinity::VertexListSize count,
                             std::size_t capacity,
                             std::span<float> scratch,
                             Center &&center,
                             NewToOld &&new_to_old,
                             OldToNew &&old_to_new) {
    const auto original_neighbors = neighbors;
    const infinity::VertexListSize original_count = count;
    const auto actual = infinity::TryIncrementalReciprocalUpdate(new_vertex,
                                                                 new_center,
                                                                 neighbors.data(),
                                                                 &count,
                                                                 capacity,
                                                                 scratch,
                                                                 std::forward<Center>(center),
                                                                 std::forward<NewToOld>(new_to_old),
                                                                 std::forward<OldToNew>(old_to_new));
    return actual == expected && count == original_count && neighbors == original_neighbors;
}

bool CheckFallbackImmutability() {
    constexpr std::size_t capacity = 3;
    const std::vector<infinity::VertexType> valid_neighbors{0, 1, 2, -777, -777};
    const auto center = [](infinity::VertexType vertex) { return 3.0F - static_cast<float>(vertex); };
    const auto pass_new_to_old = [](infinity::VertexType) { return 10.0F; };
    const auto pass_old_to_new = [](infinity::VertexType) { return 10.0F; };
    std::array<float, capacity> scratch{};

    const bool invalid_negative_count = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackInvalidState,
                                                                3,
                                                                1.5F,
                                                                valid_neighbors,
                                                                -1,
                                                                capacity,
                                                                scratch,
                                                                center,
                                                                pass_new_to_old,
                                                                pass_old_to_new);
    const bool invalid_vertex = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackInvalidState,
                                                        -1,
                                                        1.5F,
                                                        valid_neighbors,
                                                        static_cast<infinity::VertexListSize>(capacity),
                                                        capacity,
                                                        scratch,
                                                        center,
                                                        pass_new_to_old,
                                                        pass_old_to_new);
    const bool insufficient_scratch = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackScratchCapacity,
                                                              3,
                                                              1.5F,
                                                              valid_neighbors,
                                                              static_cast<infinity::VertexListSize>(capacity),
                                                              capacity,
                                                              std::span<float>(scratch.data(), capacity - 1),
                                                              center,
                                                              pass_new_to_old,
                                                              pass_old_to_new);
    auto duplicate_neighbors = valid_neighbors;
    duplicate_neighbors[1] = 3;
    const bool duplicate = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackDuplicate,
                                                   3,
                                                   1.5F,
                                                   duplicate_neighbors,
                                                   static_cast<infinity::VertexListSize>(capacity),
                                                   capacity,
                                                   scratch,
                                                   center,
                                                   pass_new_to_old,
                                                   pass_old_to_new);
    const bool tie = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackCenterTie,
                                             3,
                                             2.0F,
                                             valid_neighbors,
                                             static_cast<infinity::VertexListSize>(capacity),
                                             capacity,
                                             scratch,
                                             center,
                                             pass_new_to_old,
                                             pass_old_to_new);
    const auto signed_zero_center = [](infinity::VertexType vertex) { return vertex == 1 ? -0.0F : 3.0F - static_cast<float>(vertex); };
    const bool signed_zero_tie = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackCenterTie,
                                                         3,
                                                         0.0F,
                                                         valid_neighbors,
                                                         static_cast<infinity::VertexListSize>(capacity),
                                                         capacity,
                                                         scratch,
                                                         signed_zero_center,
                                                         pass_new_to_old,
                                                         pass_old_to_new);
    const bool nan_center = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackNonFiniteCenter,
                                                    3,
                                                    std::numeric_limits<float>::quiet_NaN(),
                                                    valid_neighbors,
                                                    static_cast<infinity::VertexListSize>(capacity),
                                                    capacity,
                                                    scratch,
                                                    center,
                                                    pass_new_to_old,
                                                    pass_old_to_new);
    const bool infinity_center = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackNonFiniteCenter,
                                                         3,
                                                         std::numeric_limits<float>::infinity(),
                                                         valid_neighbors,
                                                         static_cast<infinity::VertexListSize>(capacity),
                                                         capacity,
                                                         scratch,
                                                         center,
                                                         pass_new_to_old,
                                                         pass_old_to_new);
    const auto unordered_center = [](infinity::VertexType vertex) {
        constexpr std::array<float, capacity> values{3.0F, 1.0F, 2.0F};
        return values[static_cast<std::size_t>(vertex)];
    };
    const bool unordered = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackUncertifiedOrder,
                                                   3,
                                                   1.5F,
                                                   valid_neighbors,
                                                   static_cast<infinity::VertexListSize>(capacity),
                                                   capacity,
                                                   scratch,
                                                   unordered_center,
                                                   pass_new_to_old,
                                                   pass_old_to_new);
    const auto nonfinite_old_center = [](infinity::VertexType vertex) {
        return vertex == 1 ? std::numeric_limits<float>::infinity() : 3.0F - static_cast<float>(vertex);
    };
    const bool nonfinite_old = ExpectFallbackImmutable(infinity::HnswIncrementalReciprocalResult::kFallbackNonFiniteCenter,
                                                       3,
                                                       1.5F,
                                                       valid_neighbors,
                                                       static_cast<infinity::VertexListSize>(capacity),
                                                       capacity,
                                                       scratch,
                                                       nonfinite_old_center,
                                                       pass_new_to_old,
                                                       pass_old_to_new);
    return invalid_negative_count && invalid_vertex && insufficient_scratch && duplicate && tie && signed_zero_tie && nan_center && infinity_center &&
           unordered && nonfinite_old;
}

bool CheckThrowingCallbacksImmutable() {
    constexpr std::size_t capacity = 3;
    const std::vector<infinity::VertexType> neighbors{0, 1, 2, -1234, -1234};
    std::array<float, capacity> scratch{};
    const auto attempt = [&](auto &&center, auto &&new_to_old, auto &&old_to_new) {
        auto values = neighbors;
        infinity::VertexListSize count = static_cast<infinity::VertexListSize>(capacity);
        try {
            static_cast<void>(infinity::TryIncrementalReciprocalUpdate(3,
                                                                       1.5F,
                                                                       values.data(),
                                                                       &count,
                                                                       capacity,
                                                                       std::span<float>(scratch),
                                                                       std::forward<decltype(center)>(center),
                                                                       std::forward<decltype(new_to_old)>(new_to_old),
                                                                       std::forward<decltype(old_to_new)>(old_to_new)));
        } catch (const std::runtime_error &) {
            return count == static_cast<infinity::VertexListSize>(capacity) && values == neighbors;
        }
        return false;
    };
    const bool center_throws = attempt([](infinity::VertexType) -> float { throw std::runtime_error("center"); },
                                       [](infinity::VertexType) { return 10.0F; },
                                       [](infinity::VertexType) { return 10.0F; });
    const bool new_to_old_throws = attempt([](infinity::VertexType vertex) { return 3.0F - static_cast<float>(vertex); },
                                           [](infinity::VertexType) -> float { throw std::runtime_error("new-to-old"); },
                                           [](infinity::VertexType) { return 10.0F; });
    const bool old_to_new_throws = attempt([](infinity::VertexType vertex) { return 3.0F - static_cast<float>(vertex); },
                                           [](infinity::VertexType) { return 10.0F; },
                                           [](infinity::VertexType) -> float { throw std::runtime_error("old-to-new"); });
    return center_throws && new_to_old_throws && old_to_new_throws;
}

bool CheckNonfinitePairSemantics() {
    constexpr std::size_t capacity = 2;
    const std::array<float, 3> values{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float pair_value : values) {
        MatrixCase test = MakeCertifiedCase(capacity, 1, 0, false);
        test.Pair(test.new_vertex, 1) = pair_value;
        test.Pair(0, test.new_vertex) = pair_value;
        if (!CompareIncrementalToReference(test)) {
            return false;
        }
    }
    return true;
}

bool CheckIncrementalHelper() {
    return CheckExhaustiveCertifiedDifferential() && CheckRandomizedCertifiedDifferential() && CheckFallbackImmutability() &&
           CheckThrowingCallbacksImmutable() && CheckNonfinitePairSemantics();
}

bool CheckProductionLifecycleSemantics() {
    constexpr std::size_t count = 4'096;
    constexpr std::size_t dimension = 32;
    constexpr std::size_t split = count / 2;
    const auto data = MakeCertificateData(count, dimension);

    const auto make_partial = [&] {
        auto index = Hnsw::Make(std::bit_ceil(count), 1, dimension, 16, 96);
        InsertRange(index, data, dimension, 0, split);
        return index;
    };
    const auto load_stream = [](const Hnsw &index) {
        infinity::LocalFileHandle stream;
        index.Save(stream);
        stream.Rewind();
        return Hnsw::Load(stream);
    };
    const auto append_tail = [&](Hnsw &index) {
        infinity::DenseVectorIter<float, Label> iterator(data.data() + split * dimension, dimension, count - split, static_cast<Label>(split));
        static_cast<void>(index.InsertVecs(std::move(iterator)));
    };

    auto move_source = make_partial();
    auto move_control = load_stream(*move_source);
    const bool source_accounts_for_sidecar = move_source->mem_usage() > move_control->mem_usage();
    Hnsw moved(std::move(*move_source));
    const bool move_clears_sidecar_accounting = moved.mem_usage() == move_control->mem_usage();
    append_tail(moved);
    append_tail(*move_control);
    const bool move_exact = OrderedGraphsEqual(moved, *move_control) && SerializePointerImage(moved) == SerializePointerImage(*move_control);

    auto assignment_source = make_partial();
    auto assignment_control = load_stream(*assignment_source);
    auto assignment_target = make_partial();
    bool assignment_target_had_upper_layers = false;
    for (std::size_t vertex = 0; vertex < assignment_target->GetVecNum(); ++vertex) {
        assignment_target_had_upper_layers =
            assignment_target_had_upper_layers ||
            assignment_target->GetGraphLevel(static_cast<infinity::VertexType>(vertex)) > 0;
    }
    *assignment_target = std::move(*assignment_source);
    append_tail(*assignment_target);
    append_tail(*assignment_control);
    const bool assignment_exact = OrderedGraphsEqual(*assignment_target, *assignment_control) &&
                                  SerializePointerImage(*assignment_target) == SerializePointerImage(*assignment_control);

    auto mutable_index = make_partial();
    auto mutable_control = load_stream(*mutable_index);
    static_cast<void>(mutable_index->distance());
    append_tail(*mutable_index);
    append_tail(*mutable_control);
    const bool mutable_exact =
        OrderedGraphsEqual(*mutable_index, *mutable_control) && SerializePointerImage(*mutable_index) == SerializePointerImage(*mutable_control);

    auto optimized = make_partial();
    auto optimize_control = load_stream(*optimized);
    optimized->Optimize();
    append_tail(*optimized);
    append_tail(*optimize_control);
    const bool optimize_exact =
        OrderedGraphsEqual(*optimized, *optimize_control) && SerializePointerImage(*optimized) == SerializePointerImage(*optimize_control);

    auto lvq_source = MakeDeterministicGraph(256);
    auto lvq_control = load_stream(*lvq_source);
    const bool lvq_source_has_sidecar = lvq_source->mem_usage() > lvq_control->mem_usage();
    const std::size_t lvq_sidecar_bytes = lvq_source_has_sidecar ? lvq_source->mem_usage() - lvq_control->mem_usage() : 0;
    const std::uint64_t lvq_graph_before = GraphHash(*lvq_source);
    auto lvq = std::move(*lvq_source).CompressToLVQ();
    const bool lvq_topology_exact = GraphHash(*lvq) == lvq_graph_before && OrderedGraphsEqual(*lvq, *lvq_control);
    const bool lvq_source_accounting_exact = lvq_source_has_sidecar && lvq_source->GetVecNum() == 0 && lvq_source->mem_usage() == lvq_sidecar_bytes;

    auto rabitq_source = MakeDeterministicGraph(256);
    auto rabitq_control = load_stream(*rabitq_source);
    const bool rabitq_source_has_sidecar = rabitq_source->mem_usage() > rabitq_control->mem_usage();
    const std::size_t rabitq_sidecar_bytes = rabitq_source_has_sidecar ? rabitq_source->mem_usage() - rabitq_control->mem_usage() : 0;
    const std::uint64_t rabitq_graph_before = GraphHash(*rabitq_source);
    auto rabitq = std::move(*rabitq_source).CompressToRabitq();
    const bool rabitq_topology_exact = GraphHash(*rabitq) == rabitq_graph_before && OrderedGraphsEqual(*rabitq, *rabitq_control);
    const bool rabitq_source_accounting_exact =
        rabitq_source_has_sidecar && rabitq_source->GetVecNum() == 0 && rabitq_source->mem_usage() == rabitq_sidecar_bytes;

    std::cout << "production_source_accounts_for_sidecar=" << source_accounts_for_sidecar << '\n';
    std::cout << "production_move_clears_sidecar_accounting=" << move_clears_sidecar_accounting << '\n';
    std::cout << "production_move_exact=" << move_exact << '\n';
    std::cout << "production_assignment_target_had_upper_layers=" << assignment_target_had_upper_layers << '\n';
    std::cout << "production_assignment_exact=" << assignment_exact << '\n';
    std::cout << "production_mutable_exact=" << mutable_exact << '\n';
    std::cout << "production_optimize_exact=" << optimize_exact << '\n';
    std::cout << "production_lvq_topology_exact=" << lvq_topology_exact << '\n';
    std::cout << "production_lvq_source_accounting_exact=" << lvq_source_accounting_exact << '\n';
    std::cout << "production_rabitq_topology_exact=" << rabitq_topology_exact << '\n';
    std::cout << "production_rabitq_source_accounting_exact=" << rabitq_source_accounting_exact << '\n';
    return source_accounts_for_sidecar && move_clears_sidecar_accounting && move_exact && assignment_target_had_upper_layers &&
           assignment_exact && mutable_exact && optimize_exact && lvq_topology_exact && lvq_source_accounting_exact &&
           rabitq_topology_exact && rabitq_source_accounting_exact;
}

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)

bool IncrementalReciprocalStatsAreZero(const infinity::HnswIncrementalReciprocalStats &stats) {
    const std::array counters{
        stats.reciprocal_links,
        stats.direct_appends,
        stats.full_overflows,
        stats.certificate_hits,
        stats.certificate_misses,
        stats.certificate_sets,
        stats.certificate_clears,
        stats.unchanged_new_farthest,
        stats.unchanged_rejected,
        stats.updated_full,
        stats.updated_underfull,
        stats.fallback_invalid_state,
        stats.fallback_scratch_capacity,
        stats.fallback_duplicate,
        stats.fallback_center_tie,
        stats.fallback_nonfinite_center,
        stats.fallback_uncertified_order,
        stats.baseline_center_distance_evaluations,
        stats.baseline_pair_distance_evaluations,
        stats.incremental_center_distance_evaluations,
        stats.incremental_pair_distance_evaluations,
        stats.shadow_comparisons,
        stats.shadow_mismatches,
        stats.predicted_avoided_distance_evaluations,
        stats.predicted_extra_distance_evaluations,
    };
    return std::ranges::all_of(counters, [](std::uint64_t value) { return value == 0; });
}

bool CheckLVQCompressionCleanup() {
    auto source = MakeDeterministicGraph(256);
    infinity::LocalFileHandle topology_stream;
    source->Save(topology_stream);
    topology_stream.Rewind();
    auto topology_control = Hnsw::Load(topology_stream);
    source->SetAllLevelZeroIncrementalReciprocalCertificatesForTest();
    const std::size_t source_vector_count = source->GetVecNum();
    const std::uint64_t topology_before = GraphHash(*source);
    const auto storage_before = source->GetIncrementalReciprocalCertificateStorageForTest();
    const std::size_t live_before = source->GetIncrementalReciprocalLiveCertificateCount();
    const auto stats_before = source->GetIncrementalReciprocalStats();

    auto destination = std::move(*source).CompressToLVQ();
    const CompressionOracle compression_oracle = CheckCompressionOracle(*destination, *topology_control);

    const auto storage_after = source->GetIncrementalReciprocalCertificateStorageForTest();
    const std::size_t live_after = source->GetIncrementalReciprocalLiveCertificateCount();
    const auto stats_after = source->GetIncrementalReciprocalStats();
    const auto destination_storage = destination->GetIncrementalReciprocalCertificateStorageForTest();
    const auto destination_stats = destination->GetIncrementalReciprocalStats();
    const bool clear_counter_monotonic = stats_after.certificate_clears >= stats_before.certificate_clears;
    const std::uint64_t clear_count = clear_counter_monotonic ? stats_after.certificate_clears - stats_before.certificate_clears : 0;
    const bool source_datastore_consumed = source_vector_count > 0 && source->GetVecNum() == 0;
    const bool retained_capacity_accurately_charged =
        storage_before.mask_count > 0 && storage_before.capacity_bytes > 0 && storage_before.charged_bytes == storage_before.capacity_bytes &&
        storage_after.mask_count == storage_before.mask_count && storage_after.capacity_bytes == storage_before.capacity_bytes &&
        storage_after.charged_bytes == storage_after.capacity_bytes && source->mem_usage() == storage_after.charged_bytes;
    const bool all_source_bits_cleared = live_before > 0 && !IncrementalReciprocalStatsAreZero(stats_before) && live_after == 0;
    const bool exact_clear_count = clear_counter_monotonic && clear_count == live_before;
    const bool destination_has_no_certificate_state =
        destination_storage.mask_count == 0 && destination_storage.capacity_bytes == 0 && destination_storage.charged_bytes == 0 &&
        destination->GetIncrementalReciprocalLiveCertificateCount() == 0 && IncrementalReciprocalStatsAreZero(destination_stats);
    const bool topology_preserved = destination->GetVecNum() == source_vector_count && GraphHash(*destination) == topology_before &&
                                    OrderedGraphsEqual(*destination, *topology_control);

    std::cout << "lvq_compression_source_datastore_consumed=" << source_datastore_consumed << '\n';
    std::cout << "lvq_compression_mask_count_before=" << storage_before.mask_count << '\n';
    std::cout << "lvq_compression_mask_count_after=" << storage_after.mask_count << '\n';
    std::cout << "lvq_compression_capacity_bytes_before=" << storage_before.capacity_bytes << '\n';
    std::cout << "lvq_compression_capacity_bytes_after=" << storage_after.capacity_bytes << '\n';
    std::cout << "lvq_compression_charged_bytes_after=" << storage_after.charged_bytes << '\n';
    std::cout << "lvq_compression_retained_capacity_accurately_charged=" << retained_capacity_accurately_charged << '\n';
    std::cout << "lvq_compression_live_certificates_before=" << live_before << '\n';
    std::cout << "lvq_compression_live_certificates_after=" << live_after << '\n';
    std::cout << "lvq_compression_certificate_clear_delta=" << clear_count << '\n';
    std::cout << "lvq_compression_all_source_bits_cleared=" << all_source_bits_cleared << '\n';
    std::cout << "lvq_compression_exact_clear_count=" << exact_clear_count << '\n';
    std::cout << "lvq_compression_destination_has_no_certificate_state=" << destination_has_no_certificate_state << '\n';
    std::cout << "lvq_compression_topology_preserved=" << topology_preserved << '\n';
    std::cout << "lvq_compression_labels_preserved=" << compression_oracle.labels_preserved << '\n';
    std::cout << "lvq_compression_queries_valid=" << compression_oracle.compressed_queries_valid << '\n';
    std::cout << "lvq_compression_serialized_bytes_exact=" << compression_oracle.serialized_bytes_exact << '\n';
    return source_datastore_consumed && retained_capacity_accurately_charged && all_source_bits_cleared && exact_clear_count &&
           destination_has_no_certificate_state && topology_preserved && compression_oracle.labels_preserved &&
           compression_oracle.compressed_queries_valid && compression_oracle.serialized_bytes_exact;
}

bool CheckRabitqCompressionCleanup() {
    auto source = MakeDeterministicGraph(256);
    infinity::LocalFileHandle topology_stream;
    source->Save(topology_stream);
    topology_stream.Rewind();
    auto topology_control = Hnsw::Load(topology_stream);
    source->SetAllLevelZeroIncrementalReciprocalCertificatesForTest();
    const std::size_t source_vector_count = source->GetVecNum();
    const std::uint64_t topology_before = GraphHash(*source);
    const auto storage_before = source->GetIncrementalReciprocalCertificateStorageForTest();
    const std::size_t live_before = source->GetIncrementalReciprocalLiveCertificateCount();
    const auto stats_before = source->GetIncrementalReciprocalStats();

    auto destination = std::move(*source).CompressToRabitq();
    const CompressionOracle compression_oracle = CheckCompressionOracle(*destination, *topology_control);

    const auto storage_after = source->GetIncrementalReciprocalCertificateStorageForTest();
    const std::size_t live_after = source->GetIncrementalReciprocalLiveCertificateCount();
    const auto stats_after = source->GetIncrementalReciprocalStats();
    const auto destination_storage = destination->GetIncrementalReciprocalCertificateStorageForTest();
    const auto destination_stats = destination->GetIncrementalReciprocalStats();
    const bool clear_counter_monotonic = stats_after.certificate_clears >= stats_before.certificate_clears;
    const std::uint64_t clear_count = clear_counter_monotonic ? stats_after.certificate_clears - stats_before.certificate_clears : 0;
    const bool source_datastore_consumed = source_vector_count > 0 && source->GetVecNum() == 0;
    const bool retained_capacity_accurately_charged =
        storage_before.mask_count > 0 && storage_before.capacity_bytes > 0 && storage_before.charged_bytes == storage_before.capacity_bytes &&
        storage_after.mask_count == storage_before.mask_count && storage_after.capacity_bytes == storage_before.capacity_bytes &&
        storage_after.charged_bytes == storage_after.capacity_bytes && source->mem_usage() == storage_after.charged_bytes;
    const bool all_source_bits_cleared = live_before > 0 && !IncrementalReciprocalStatsAreZero(stats_before) && live_after == 0;
    const bool exact_clear_count = clear_counter_monotonic && clear_count == live_before;
    const bool destination_has_no_certificate_state =
        destination_storage.mask_count == 0 && destination_storage.capacity_bytes == 0 && destination_storage.charged_bytes == 0 &&
        destination->GetIncrementalReciprocalLiveCertificateCount() == 0 && IncrementalReciprocalStatsAreZero(destination_stats);
    const bool topology_preserved = destination->GetVecNum() == source_vector_count && GraphHash(*destination) == topology_before &&
                                    OrderedGraphsEqual(*destination, *topology_control);

    std::cout << "rabitq_compression_source_datastore_consumed=" << source_datastore_consumed << '\n';
    std::cout << "rabitq_compression_mask_count_before=" << storage_before.mask_count << '\n';
    std::cout << "rabitq_compression_mask_count_after=" << storage_after.mask_count << '\n';
    std::cout << "rabitq_compression_capacity_bytes_before=" << storage_before.capacity_bytes << '\n';
    std::cout << "rabitq_compression_capacity_bytes_after=" << storage_after.capacity_bytes << '\n';
    std::cout << "rabitq_compression_charged_bytes_after=" << storage_after.charged_bytes << '\n';
    std::cout << "rabitq_compression_retained_capacity_accurately_charged=" << retained_capacity_accurately_charged << '\n';
    std::cout << "rabitq_compression_live_certificates_before=" << live_before << '\n';
    std::cout << "rabitq_compression_live_certificates_after=" << live_after << '\n';
    std::cout << "rabitq_compression_certificate_clear_delta=" << clear_count << '\n';
    std::cout << "rabitq_compression_all_source_bits_cleared=" << all_source_bits_cleared << '\n';
    std::cout << "rabitq_compression_exact_clear_count=" << exact_clear_count << '\n';
    std::cout << "rabitq_compression_destination_has_no_certificate_state=" << destination_has_no_certificate_state << '\n';
    std::cout << "rabitq_compression_topology_preserved=" << topology_preserved << '\n';
    std::cout << "rabitq_compression_labels_preserved=" << compression_oracle.labels_preserved << '\n';
    std::cout << "rabitq_compression_queries_valid=" << compression_oracle.compressed_queries_valid << '\n';
    std::cout << "rabitq_compression_serialized_bytes_exact=" << compression_oracle.serialized_bytes_exact << '\n';
    return source_datastore_consumed && retained_capacity_accurately_charged && all_source_bits_cleared && exact_clear_count &&
           destination_has_no_certificate_state && topology_preserved && compression_oracle.labels_preserved &&
           compression_oracle.compressed_queries_valid && compression_oracle.serialized_bytes_exact;
}

class BuildOverlapGate {
public:
    void Arrive() {
        std::unique_lock lock(mutex_);
        if (released_) {
            return;
        }
        worker_threads_.insert(std::this_thread::get_id());
        if (worker_threads_.size() >= 2) {
            released_ = true;
            condition_.notify_all();
            return;
        }
        if (!condition_.wait_for(lock, std::chrono::seconds(10), [&] { return released_; })) {
            timed_out_ = true;
            released_ = true;
            condition_.notify_all();
        }
    }

    bool Passed() const {
        std::lock_guard lock(mutex_);
        return !timed_out_ && worker_threads_.size() >= 2;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::set<std::thread::id> worker_threads_;
    bool released_{};
    bool timed_out_{};
};

class StagedPublicationGate {
public:
    StagedPublicationGate(Hnsw &index, std::pair<infinity::i32, infinity::VertexType> expected_entry_point)
        : index_(index), expected_entry_point_(expected_entry_point) {}

    void Arrive() {
        std::unique_lock lock(mutex_);
        entry_point_unchanged_ = entry_point_unchanged_ && index_.GetGraphEnterPoint() == expected_entry_point_;
        worker_threads_.insert(std::this_thread::get_id());
        if (worker_threads_.size() >= 2) {
            entry_point_unchanged_ = entry_point_unchanged_ && index_.GetGraphEnterPoint() == expected_entry_point_;
            released_ = true;
            condition_.notify_all();
            return;
        }
        if (!condition_.wait_for(lock, std::chrono::seconds(10), [&] { return released_; })) {
            timed_out_ = true;
            released_ = true;
            condition_.notify_all();
        }
    }

    bool Passed() const {
        std::lock_guard lock(mutex_);
        return !timed_out_ && entry_point_unchanged_ && worker_threads_.size() >= 2;
    }

private:
    Hnsw &index_;
    std::pair<infinity::i32, infinity::VertexType> expected_entry_point_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::set<std::thread::id> worker_threads_;
    bool released_{};
    bool timed_out_{};
    bool entry_point_unchanged_{true};
};

bool CheckStagedPublicationLockHierarchy() {
    constexpr std::size_t count = 3;
    const auto data = MakeCertificateData(count);
    auto index = Hnsw::Make(4, 1, kDimension, 2, 16);
    infinity::DenseVectorIter<float, Label> iterator(data.data(), kDimension, count);
    static_cast<void>(index->StoreData(std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}));
    index->Build(0, 0);

    auto gate = std::make_shared<StagedPublicationGate>(*index, std::pair<infinity::i32, infinity::VertexType>{0, 0});
    index->SetBuildEntryHookForTest([gate](infinity::VertexType) { gate->Arrive(); });

    std::array<std::exception_ptr, 2> errors;
    bool finalized = false;
    {
        auto operation_lock = index->AcquireExclusiveOperationLock();
        std::array<std::thread, 2> workers{
            std::thread([&] {
                try {
                    index->BuildWithOperationLockHeld(1, 1);
                } catch (...) {
                    errors[0] = std::current_exception();
                }
            }),
            std::thread([&] {
                try {
                    index->BuildWithOperationLockHeld(2, 1);
                } catch (...) {
                    errors[1] = std::current_exception();
                }
            }),
        };
        for (std::thread &worker : workers) {
            worker.join();
        }
        if (!errors[0] && !errors[1]) {
            try {
                static_cast<void>(index->FinalizeBuildWithOperationLockHeld());
                finalized = true;
            } catch (...) {
            }
        }
    }
    index->SetBuildEntryHookForTest({});

    const bool completed = !errors[0] && !errors[1] && finalized && !index->IsBuildFailed();
    if (completed) {
        index->Check();
    }
    const bool levels_preserved = completed && index->GetGraphLevel(0) == 0 && index->GetGraphLevel(1) == 1 && index->GetGraphLevel(2) == 1;
    std::cout << "staged_publication_gate_passed=" << gate->Passed() << '\n';
    std::cout << "staged_publication_builds_completed=" << completed << '\n';
    std::cout << "staged_publication_levels_preserved=" << levels_preserved << '\n';
    return gate->Passed() && completed && levels_preserved;
}

bool CheckPersistenceAndLifecycle() {
    constexpr std::size_t count = 4'096;
    constexpr std::size_t dimension = 32;
    constexpr std::size_t chunk_size = std::bit_ceil(count);
    const auto data = MakeCertificateData(count, dimension);
    const auto load_stream = [](const Hnsw &source) {
        infinity::LocalFileHandle stream;
        source.Save(stream);
        stream.Rewind();
        return Hnsw::Load(stream);
    };
    const auto append_tail = [&](Hnsw &target) {
        infinity::DenseVectorIter<float, Label> iterator(data.data() + count / 2 * dimension, dimension, count / 2, static_cast<Label>(count / 2));
        static_cast<void>(target.InsertVecs(std::move(iterator)));
    };
    auto index = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    InsertRange(index, data, dimension, 0, count);
    const bool source_has_certificates = index->GetIncrementalReciprocalLiveCertificateCount() > 0;
    const auto source_stats = index->GetIncrementalReciprocalStats();

    infinity::LocalFileHandle stream;
    index->Save(stream);
    stream.Rewind();
    auto stream_loaded = Hnsw::Load(stream);

    const std::vector<char> image = SerializePointerImage(*index);
    infinity::LocalFileHandle pointer_file;
    pointer_file.Append(image.data(), image.size());
    pointer_file.Rewind();
    auto pointer_loaded = Hnsw::LoadFromPtr(pointer_file, image.size());

    const char *cursor = image.data();
    auto mapped_loaded = MappedHnsw::LoadFromPtr(cursor, image.size());

    Hnsw moved(std::move(*index));
    const bool loaded_empty = stream_loaded->GetIncrementalReciprocalLiveCertificateCount() == 0 &&
                              pointer_loaded->GetIncrementalReciprocalLiveCertificateCount() == 0 &&
                              mapped_loaded->GetIncrementalReciprocalLiveCertificateCount() == 0 && cursor == image.data() + image.size();
    const bool move_empty = moved.GetIncrementalReciprocalLiveCertificateCount() == 0;

    auto mutable_index = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    InsertRange(mutable_index, data, dimension, 0, count / 2);
    const bool mutable_before = mutable_index->GetIncrementalReciprocalLiveCertificateCount() > 0;
    static_cast<void>(mutable_index->distance());
    const bool mutable_cleared = mutable_index->GetIncrementalReciprocalLiveCertificateCount() == 0;
    InsertRange(mutable_index, data, dimension, count / 2, count / 2);
    const bool mutable_sticky = mutable_index->GetIncrementalReciprocalLiveCertificateCount() == 0;

    auto disabled_move_source = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    InsertRange(disabled_move_source, data, dimension, 0, count / 2);
    auto disabled_move_control = load_stream(*disabled_move_source);
    auto &disabled_move_distance = disabled_move_source->distance();
    Hnsw disabled_moved(std::move(*disabled_move_source));
    append_tail(disabled_moved);
    append_tail(*disabled_move_control);
    const bool move_construct_sticky = disabled_moved.GetIncrementalReciprocalLiveCertificateCount() == 0 &&
                                       OrderedGraphsEqual(disabled_moved, *disabled_move_control) &&
                                       SerializePointerImage(disabled_moved) == SerializePointerImage(*disabled_move_control);

    auto reuse_replacement = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    InsertRange(reuse_replacement, data, dimension, 0, count / 2);
    *disabled_move_source = std::move(*reuse_replacement);
    auto moved_from_control = load_stream(*disabled_move_source);
    disabled_move_distance = Hnsw::Distance(dimension);
    append_tail(*disabled_move_source);
    append_tail(*moved_from_control);
    const bool moved_from_sticky = disabled_move_source->GetIncrementalReciprocalLiveCertificateCount() == 0 &&
                                   OrderedGraphsEqual(*disabled_move_source, *moved_from_control) &&
                                   SerializePointerImage(*disabled_move_source) == SerializePointerImage(*moved_from_control);

    auto disabled_assignment_source = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    InsertRange(disabled_assignment_source, data, dimension, 0, count / 2);
    auto disabled_assignment_control = load_stream(*disabled_assignment_source);
    auto &disabled_assignment_distance = disabled_assignment_source->distance();
    auto disabled_assignment_target = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    *disabled_assignment_target = std::move(*disabled_assignment_source);
    disabled_assignment_distance = Hnsw::Distance(dimension);
    append_tail(*disabled_assignment_target);
    append_tail(*disabled_assignment_control);
    const bool move_assignment_source_sticky =
        disabled_assignment_target->GetIncrementalReciprocalLiveCertificateCount() == 0 &&
        OrderedGraphsEqual(*disabled_assignment_target, *disabled_assignment_control) &&
        SerializePointerImage(*disabled_assignment_target) == SerializePointerImage(*disabled_assignment_control);

    auto clean_assignment_source = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    InsertRange(clean_assignment_source, data, dimension, 0, count / 2);
    auto clean_assignment_control = load_stream(*clean_assignment_source);
    auto sticky_assignment_target = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    auto &sticky_assignment_distance = sticky_assignment_target->distance();
    *sticky_assignment_target = std::move(*clean_assignment_source);
    sticky_assignment_distance = Hnsw::Distance(dimension);
    append_tail(*sticky_assignment_target);
    append_tail(*clean_assignment_control);
    const bool move_assignment_target_sticky = sticky_assignment_target->GetIncrementalReciprocalLiveCertificateCount() == 0 &&
                                               OrderedGraphsEqual(*sticky_assignment_target, *clean_assignment_control) &&
                                               SerializePointerImage(*sticky_assignment_target) == SerializePointerImage(*clean_assignment_control);

    auto optimized = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    InsertRange(optimized, data, dimension, 0, count);
    const bool optimize_before = optimized->GetIncrementalReciprocalLiveCertificateCount() > 0;
    optimized->Optimize();
    const bool optimize_cleared = optimized->GetIncrementalReciprocalLiveCertificateCount() == 0;

    auto layer64 = Hnsw::Make(1, 1, kDimension, 2, 8);
    const auto one = MakeData(1);
    infinity::DenseVectorIter<float, Label> one_iterator(one.data(), kDimension, 1);
    static_cast<void>(layer64->StoreData(std::move(one_iterator)));
    layer64->Build(0, 64);
    const bool layer64_empty = !layer64->HasIncrementalReciprocalCertificateForTest(0, 64);

    std::cout << "lifecycle_source_has_certificates=" << source_has_certificates << '\n';
    std::cout << "lifecycle_certificate_sets=" << source_stats.certificate_sets << '\n';
    std::cout << "lifecycle_certificate_hits=" << source_stats.certificate_hits << '\n';
    std::cout << "lifecycle_loaded_empty=" << loaded_empty << '\n';
    std::cout << "lifecycle_move_empty=" << move_empty << '\n';
    std::cout << "lifecycle_mutable_before=" << mutable_before << '\n';
    std::cout << "lifecycle_mutable_cleared=" << mutable_cleared << '\n';
    std::cout << "lifecycle_mutable_sticky=" << mutable_sticky << '\n';
    std::cout << "lifecycle_move_construct_sticky=" << move_construct_sticky << '\n';
    std::cout << "lifecycle_moved_from_sticky=" << moved_from_sticky << '\n';
    std::cout << "lifecycle_move_assignment_source_sticky=" << move_assignment_source_sticky << '\n';
    std::cout << "lifecycle_move_assignment_target_sticky=" << move_assignment_target_sticky << '\n';
    std::cout << "lifecycle_optimize_before=" << optimize_before << '\n';
    std::cout << "lifecycle_optimize_cleared=" << optimize_cleared << '\n';
    std::cout << "lifecycle_layer64_empty=" << layer64_empty << '\n';
    return source_has_certificates && loaded_empty && move_empty && mutable_before && mutable_cleared && mutable_sticky && move_construct_sticky &&
           moved_from_sticky && move_assignment_source_sticky && move_assignment_target_sticky && optimize_before && optimize_cleared &&
           layer64_empty;
}

bool CheckRepairRetainsCertificates() {
    constexpr std::size_t count = 2'048;
    constexpr std::size_t dimension = 4;
    const auto data = MakeData(count, dimension);
    auto index = Hnsw::Make(count, 1, dimension, 2, 20);
    infinity::DenseVectorIter<float, Label> iterator(data.data(), dimension, count);
    static_cast<void>(index->StoreData(std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}));
    const auto levels = index->PrepareBuildLevels(0, count);
    struct RepairEvent {
        infinity::VertexType source;
        infinity::VertexType target;
        bool append;
        bool certificate_before;
        bool certificate_after;
    };
    std::vector<RepairEvent> repair_events;
    index->SetConnectivityRepairHookForTest([&](infinity::VertexType source, infinity::VertexType target, bool append, bool before, bool after) {
        repair_events.push_back(RepairEvent{
            .source = source,
            .target = target,
            .append = append,
            .certificate_before = before,
            .certificate_after = after,
        });
    });
    const auto snapshot_level_zero = [&] {
        std::vector<std::vector<infinity::VertexType>> adjacency(count);
        for (std::size_t vertex = 0; vertex < count; ++vertex) {
            const auto [neighbors, degree] = index->GetGraphNeighbors(static_cast<infinity::VertexType>(vertex), 0);
            adjacency[vertex].assign(neighbors, neighbors + degree);
        }
        return adjacency;
    };
    std::size_t repair_count = 0;
    std::vector<std::vector<infinity::VertexType>> before_adjacency;
    infinity::HnswIncrementalReciprocalStats before_stats;
    std::size_t before_live_certificates = 0;
    {
        auto lock = index->AcquireExclusiveOperationLock();
        for (std::size_t vertex = 0; vertex < count; ++vertex) {
            index->BuildWithOperationLockHeld(static_cast<infinity::VertexType>(vertex), levels[vertex]);
        }
    }
    index->SetAllLevelZeroIncrementalReciprocalCertificatesForTest();
    before_adjacency = snapshot_level_zero();
    before_stats = index->GetIncrementalReciprocalStats();
    before_live_certificates = index->GetIncrementalReciprocalLiveCertificateCount();
    {
        auto lock = index->AcquireExclusiveOperationLock();
        repair_count = index->FinalizeBuildWithOperationLockHeld();
    }
    index->SetConnectivityRepairHookForTest({});
    index->Check();
    const auto after_adjacency = snapshot_level_zero();
    const auto after_stats = index->GetIncrementalReciprocalStats();
    const std::size_t after_live_certificates = index->GetIncrementalReciprocalLiveCertificateCount();
    std::set<infinity::VertexType> changed_sources;
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        if (before_adjacency[vertex] != after_adjacency[vertex]) {
            changed_sources.insert(static_cast<infinity::VertexType>(vertex));
        }
    }
    std::set<infinity::VertexType> observed_sources;
    std::size_t observed_certificate_clears = 0;
    bool every_event_valid = true;
    for (const RepairEvent &event : repair_events) {
        observed_sources.insert(event.source);
        observed_certificate_clears += event.certificate_before && !event.certificate_after;
        every_event_valid = every_event_valid && event.source >= 0 && event.target >= 0 && static_cast<std::size_t>(event.source) < count &&
                            static_cast<std::size_t>(event.target) < count && event.source != event.target && !event.certificate_after;
    }
    const std::uint64_t clear_counter_delta = after_stats.certificate_clears - before_stats.certificate_clears;
    const bool live_count_monotonic = before_live_certificates >= after_live_certificates;
    const std::size_t live_count_delta = live_count_monotonic ? before_live_certificates - after_live_certificates : 0;
    const bool per_bridge_invalidation =
        repair_count > 0 && repair_events.size() == repair_count && every_event_valid && observed_sources == changed_sources &&
        observed_certificate_clears > 0 && observed_certificate_clears == observed_sources.size() &&
        clear_counter_delta == observed_certificate_clears && live_count_monotonic && live_count_delta == observed_certificate_clears;
    const bool certificates_retained = after_live_certificates > 0;
    std::cout << "repair_count=" << repair_count << " repair_events=" << repair_events.size() << " repair_changed_sources=" << changed_sources.size()
              << " repair_observed_sources=" << observed_sources.size() << " repair_certificate_clears=" << observed_certificate_clears
              << " repair_clear_counter_delta=" << clear_counter_delta << " repair_live_count_delta=" << live_count_delta
              << " repair_every_event_valid=" << every_event_valid << " repair_per_bridge_invalidation=" << per_bridge_invalidation << '\n';
    return per_bridge_invalidation && certificates_retained;
}

#endif

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)

bool CheckCampaignParameterControlTreatmentExactness() {
    const auto data = MakeCertificateData(kCampaignCount, kCampaignDimension);
    auto control =
        CampaignHnsw::Make(kCampaignChunkSize, kCampaignMaxChunks, kCampaignDimension, kCampaignM, kCampaignEfConstruction);
    auto treatment =
        CampaignHnsw::Make(kCampaignChunkSize, kCampaignMaxChunks, kCampaignDimension, kCampaignM, kCampaignEfConstruction);
    control->SetIncrementalReciprocalEnabledForTest(false);
    treatment->SetIncrementalReciprocalEnabledForTest(true);

    const auto build = [&](std::unique_ptr<CampaignHnsw> &index) {
        infinity::DenseVectorIter<float, CampaignLabel> iterator(data.data(), kCampaignDimension, kCampaignCount);
        infinity::ctpl::thread_pool pool(1);
        return infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, pool);
    };

    const std::size_t control_memory_before = control->mem_usage();
    const infinity::HnswBulkBuildResult control_build = build(control);
    const bool control_memory_monotonic = control->mem_usage() >= control_memory_before;
    const bool control_owned_accounting =
        control_memory_monotonic && control_build.mem_usage_ == control->mem_usage() - control_memory_before;

    const std::size_t treatment_memory_before = treatment->mem_usage();
    const infinity::HnswBulkBuildResult treatment_build = build(treatment);
    const bool treatment_memory_monotonic = treatment->mem_usage() >= treatment_memory_before;
    const bool treatment_owned_accounting =
        treatment_memory_monotonic && treatment_build.mem_usage_ == treatment->mem_usage() - treatment_memory_before;

    control->Check();
    treatment->Check();

    const auto complete_build = [](const infinity::HnswBulkBuildResult &result, const CampaignHnsw &index) {
        return result.start_ == 0 && result.end_ == kCampaignCount && result.submitted_task_count_ == 1 && result.mem_usage_ > 0 &&
               index.GetVecNum() == kCampaignCount;
    };
    const bool builds_complete = complete_build(control_build, *control) && complete_build(treatment_build, *treatment);

    const auto stats = treatment->GetIncrementalReciprocalStats();
    const std::uint64_t unchanged = stats.unchanged_new_farthest + stats.unchanged_rejected;
    const std::uint64_t updated = stats.updated_full + stats.updated_underfull;
    const std::uint64_t successful = unchanged + updated;
    const std::uint64_t fallback = stats.fallback_invalid_state + stats.fallback_scratch_capacity + stats.fallback_duplicate +
                                   stats.fallback_center_tie + stats.fallback_nonfinite_center + stats.fallback_uncertified_order;
    const bool counter_accounting = stats.reciprocal_links == stats.direct_appends + stats.full_overflows &&
                                    stats.full_overflows == stats.certificate_hits + stats.certificate_misses &&
                                    stats.certificate_hits == successful + fallback;
    const bool treatment_executed =
        stats.certificate_sets > 0 && stats.certificate_hits > 0 && successful > 0 && updated > 0;

    const bool ordered_graph_exact = OrderedGraphsEqual(*control, *treatment);
    const std::uint64_t control_graph_hash = GraphHash(*control);
    const std::uint64_t treatment_graph_hash = GraphHash(*treatment);
    const bool graph_hash_exact = control_graph_hash == treatment_graph_hash;
    const std::vector<char> control_image = SerializePointerImage(*control);
    const std::vector<char> treatment_image = SerializePointerImage(*treatment);
    const std::uint64_t control_serialized_hash = HashBytes(control_image);
    const std::uint64_t treatment_serialized_hash = HashBytes(treatment_image);
    const bool serialized_size_exact = !control_image.empty() && control_image.size() == treatment_image.size();
    const bool serialized_hash_exact = control_serialized_hash == treatment_serialized_hash;
    const bool serialized_bytes_exact = control_image == treatment_image;
    const bool owned_accounting = control_owned_accounting && treatment_owned_accounting;

    std::cout << "campaign_exactness_vectors=" << kCampaignCount << '\n';
    std::cout << "campaign_exactness_dimensions=" << kCampaignDimension << '\n';
    std::cout << "campaign_exactness_chunk_size=" << kCampaignChunkSize << '\n';
    std::cout << "campaign_exactness_max_chunks=" << kCampaignMaxChunks << '\n';
    std::cout << "campaign_exactness_M=" << kCampaignM << '\n';
    std::cout << "campaign_exactness_ef_construction=" << kCampaignEfConstruction << '\n';
    std::cout << "campaign_exactness_builds_complete=" << builds_complete << '\n';
    std::cout << "campaign_exactness_owned_accounting=" << owned_accounting << '\n';
    std::cout << "campaign_exactness_certificate_sets=" << stats.certificate_sets << '\n';
    std::cout << "campaign_exactness_certificate_hits=" << stats.certificate_hits << '\n';
    std::cout << "campaign_exactness_successful=" << successful << '\n';
    std::cout << "campaign_exactness_updated=" << updated << '\n';
    std::cout << "campaign_exactness_fallback=" << fallback << '\n';
    std::cout << "campaign_exactness_counter_accounting=" << counter_accounting << '\n';
    std::cout << "campaign_exactness_treatment_executed=" << treatment_executed << '\n';
    std::cout << "campaign_exactness_ordered_graph=" << ordered_graph_exact << '\n';
    std::cout << "campaign_exactness_control_graph_hash=" << control_graph_hash << '\n';
    std::cout << "campaign_exactness_treatment_graph_hash=" << treatment_graph_hash << '\n';
    std::cout << "campaign_exactness_graph_hash=" << graph_hash_exact << '\n';
    std::cout << "campaign_exactness_control_serialized_bytes=" << control_image.size() << '\n';
    std::cout << "campaign_exactness_treatment_serialized_bytes=" << treatment_image.size() << '\n';
    std::cout << "campaign_exactness_control_serialized_hash=" << control_serialized_hash << '\n';
    std::cout << "campaign_exactness_treatment_serialized_hash=" << treatment_serialized_hash << '\n';
    std::cout << "campaign_exactness_serialized_size=" << serialized_size_exact << '\n';
    std::cout << "campaign_exactness_serialized_hash=" << serialized_hash_exact << '\n';
    std::cout << "campaign_exactness_serialized_bytes=" << serialized_bytes_exact << '\n';

    return builds_complete && owned_accounting && counter_accounting && treatment_executed && ordered_graph_exact && graph_hash_exact &&
           serialized_size_exact && serialized_hash_exact && serialized_bytes_exact;
}

bool CheckMutableDistanceAccessorIsInertAndExact() {
    constexpr std::size_t count = 4'096;
    constexpr std::size_t dimension = 32;
    constexpr std::size_t split = count / 2;
    constexpr std::size_t chunk_size = std::bit_ceil(count);
    const auto data = MakeCertificateData(count, dimension);

    auto baseline = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    auto treatment = Hnsw::Make(chunk_size, 1, dimension, 16, 96);
    baseline->SetIncrementalReciprocalEnabledForTest(false);
    treatment->SetIncrementalReciprocalEnabledForTest(true);
    InsertRange(baseline, data, dimension, 0, split);
    InsertRange(treatment, data, dimension, 0, split);

    const auto stats_before_accessor = treatment->GetIncrementalReciprocalStats();
    const std::size_t certificates_before_accessor = treatment->GetIncrementalReciprocalLiveCertificateCount();
    const bool treatment_established =
        certificates_before_accessor > 0 && stats_before_accessor.certificate_sets > 0 && stats_before_accessor.certificate_hits > 0;

    static_cast<void>(treatment->distance());

    const auto stats_after_accessor = treatment->GetIncrementalReciprocalStats();
    const std::size_t certificates_after_accessor = treatment->GetIncrementalReciprocalLiveCertificateCount();
    const bool clear_counter_monotonic = stats_after_accessor.certificate_clears >= stats_before_accessor.certificate_clears;
    const std::uint64_t clear_delta =
        clear_counter_monotonic ? stats_after_accessor.certificate_clears - stats_before_accessor.certificate_clears : 0;
    const bool accessor_cleared_exactly =
        certificates_after_accessor == 0 && clear_counter_monotonic && clear_delta == certificates_before_accessor;

    InsertRange(baseline, data, dimension, split, count - split);
    InsertRange(treatment, data, dimension, split, count - split);
    baseline->Check();
    treatment->Check();

    const auto stats_after_tail = treatment->GetIncrementalReciprocalStats();
    const std::size_t certificates_after_tail = treatment->GetIncrementalReciprocalLiveCertificateCount();
    const bool tail_exercised = stats_after_tail.reciprocal_links > stats_after_accessor.reciprocal_links &&
                                stats_after_tail.full_overflows > stats_after_accessor.full_overflows &&
                                stats_after_tail.certificate_misses > stats_after_accessor.certificate_misses;
    const bool treatment_remained_disabled =
        certificates_after_tail == 0 && stats_after_tail.certificate_sets == stats_after_accessor.certificate_sets &&
        stats_after_tail.certificate_hits == stats_after_accessor.certificate_hits &&
        stats_after_tail.certificate_clears == stats_after_accessor.certificate_clears;

    const bool ordered_graph_exact = OrderedGraphsEqual(*baseline, *treatment);
    const std::uint64_t baseline_graph_hash = GraphHash(*baseline);
    const std::uint64_t treatment_graph_hash = GraphHash(*treatment);
    const bool graph_hash_exact = baseline_graph_hash == treatment_graph_hash;
    const std::vector<char> baseline_image = SerializePointerImage(*baseline);
    const std::vector<char> treatment_image = SerializePointerImage(*treatment);
    const bool serialized_bytes_exact = !baseline_image.empty() && baseline_image == treatment_image;

    std::cout << "mutable_distance_certificates_before=" << certificates_before_accessor << '\n';
    std::cout << "mutable_distance_certificates_after_accessor=" << certificates_after_accessor << '\n';
    std::cout << "mutable_distance_certificates_after_tail=" << certificates_after_tail << '\n';
    std::cout << "mutable_distance_clear_delta=" << clear_delta << '\n';
    std::cout << "mutable_distance_sets_after_accessor=" << stats_after_accessor.certificate_sets << '\n';
    std::cout << "mutable_distance_sets_after_tail=" << stats_after_tail.certificate_sets << '\n';
    std::cout << "mutable_distance_hits_after_accessor=" << stats_after_accessor.certificate_hits << '\n';
    std::cout << "mutable_distance_hits_after_tail=" << stats_after_tail.certificate_hits << '\n';
    std::cout << "mutable_distance_treatment_established=" << treatment_established << '\n';
    std::cout << "mutable_distance_accessor_cleared_exactly=" << accessor_cleared_exactly << '\n';
    std::cout << "mutable_distance_tail_exercised=" << tail_exercised << '\n';
    std::cout << "mutable_distance_treatment_remained_disabled=" << treatment_remained_disabled << '\n';
    std::cout << "mutable_distance_ordered_graph_exact=" << ordered_graph_exact << '\n';
    std::cout << "mutable_distance_baseline_graph_hash=" << baseline_graph_hash << '\n';
    std::cout << "mutable_distance_treatment_graph_hash=" << treatment_graph_hash << '\n';
    std::cout << "mutable_distance_graph_hash_exact=" << graph_hash_exact << '\n';
    std::cout << "mutable_distance_serialized_bytes_exact=" << serialized_bytes_exact << '\n';

    return treatment_established && accessor_cleared_exactly && tail_exercised && treatment_remained_disabled && ordered_graph_exact &&
           graph_hash_exact && serialized_bytes_exact;
}

bool CheckDiagnosticControlTreatment() {
    constexpr std::size_t count = 8'192;
    constexpr std::size_t dimension = 32;
    constexpr std::size_t split = count / 2;
    const auto data = MakeCertificateData(count, dimension);
    auto control = Hnsw::Make(count, 1, dimension, 16, 96);
    auto treatment = Hnsw::Make(count, 1, dimension, 16, 96);
    control->SetIncrementalReciprocalEnabledForTest(false);
    treatment->SetIncrementalReciprocalEnabledForTest(true);
    InsertRange(control, data, dimension, 0, split);
    InsertRange(treatment, data, dimension, 0, split);
    const bool first_batch_certificates = treatment->GetIncrementalReciprocalLiveCertificateCount() > 0;
    InsertRange(control, data, dimension, split, count - split);
    InsertRange(treatment, data, dimension, split, count - split);
    control->Check();
    treatment->Check();
    const auto stats = treatment->GetIncrementalReciprocalStats();
    const bool same_graph = OrderedGraphsEqual(*control, *treatment);
    const bool same_bytes = SerializePointerImage(*control) == SerializePointerImage(*treatment);
    const bool retained_certificates = treatment->GetIncrementalReciprocalLiveCertificateCount() > 0;
    std::cout << "diagnostic_first_batch_certificates=" << first_batch_certificates << '\n';
    std::cout << "diagnostic_retained_certificates=" << retained_certificates << '\n';
    std::cout << "diagnostic_same_graph=" << same_graph << '\n';
    std::cout << "diagnostic_same_bytes=" << same_bytes << '\n';
    std::cout << "diagnostic_certificate_sets=" << stats.certificate_sets << '\n';
    std::cout << "diagnostic_certificate_hits=" << stats.certificate_hits << '\n';
    std::cout << "diagnostic_certificate_misses=" << stats.certificate_misses << '\n';
    return first_batch_certificates && retained_certificates && same_graph && same_bytes && stats.certificate_hits > 0;
}

bool CheckConcurrentTreatmentMutation() {
    constexpr std::size_t count = 12'288;
    constexpr std::size_t dimension = 32;
    constexpr int worker_count = 12;
    constexpr std::size_t run_count = 3;
    const auto data = MakeCertificateData(count, dimension);
    bool all_passed = true;

    for (std::size_t run = 0; run < run_count; ++run) {
        auto index = Hnsw::Make(std::bit_ceil(count), 1, dimension, 16, 96);
        index->SetIncrementalReciprocalEnabledForTest(true);
        auto overlap_gate = std::make_shared<BuildOverlapGate>();
        index->SetBuildEntryHookForTest([overlap_gate](infinity::VertexType) { overlap_gate->Arrive(); });
        infinity::DenseVectorIter<float, Label> iterator(data.data(), dimension, count);
        infinity::ctpl::thread_pool pool(worker_count);
        const infinity::HnswBulkBuildResult build_result =
            infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, pool);
        index->SetBuildEntryHookForTest({});
        index->Check();
        const bool worker_overlap = overlap_gate->Passed();

        const auto stats = index->GetIncrementalReciprocalStats();
        const std::uint64_t unchanged = stats.unchanged_new_farthest + stats.unchanged_rejected;
        const std::uint64_t updated = stats.updated_full + stats.updated_underfull;
        const std::uint64_t successful = unchanged + updated;
        const std::uint64_t fallback = stats.fallback_invalid_state + stats.fallback_scratch_capacity + stats.fallback_duplicate +
                                       stats.fallback_center_tie + stats.fallback_nonfinite_center + stats.fallback_uncertified_order;
        const bool accounting = stats.reciprocal_links == stats.direct_appends + stats.full_overflows &&
                                stats.full_overflows == stats.certificate_hits + stats.certificate_misses &&
                                stats.certificate_hits == successful + fallback &&
                                stats.incremental_center_distance_evaluations >= stats.certificate_hits;
        const bool exercised_treatment = stats.certificate_sets > 0 && stats.certificate_hits > 0 && successful > 0 && updated > 0;
        const bool complete_build = build_result.start_ == 0 && build_result.end_ == count &&
                                    build_result.submitted_task_count_ == static_cast<std::size_t>(worker_count) && build_result.mem_usage_ > 0 &&
                                    index->GetVecNum() == count;

        const std::vector<char> source_image = SerializePointerImage(*index);
        infinity::LocalFileHandle stream;
        index->Save(stream);
        stream.Rewind();
        auto loaded = Hnsw::Load(stream);
        loaded->Check();
        const bool round_trip = loaded->GetVecNum() == count && OrderedGraphsEqual(*index, *loaded) && SerializePointerImage(*loaded) == source_image;
        const bool run_passed = worker_overlap && accounting && exercised_treatment && complete_build && round_trip;
        all_passed = all_passed && run_passed;

        std::cout << "concurrent_treatment_run=" << run << " reciprocal_links=" << stats.reciprocal_links
                  << " full_overflows=" << stats.full_overflows << " certificate_sets=" << stats.certificate_sets
                  << " certificate_hits=" << stats.certificate_hits << " successful=" << successful << " updated=" << updated
                  << " fallback=" << fallback << " worker_overlap=" << worker_overlap << " accounting=" << accounting
                  << " complete_build=" << complete_build << " semantic_graph_round_trip=" << round_trip << " passed=" << run_passed << '\n';
    }
    return all_passed;
}

#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)

struct ShadowScreen {
    bool passed{};
    double coverage{};
    double net_predicted_work{};
};

ShadowScreen RunShadowScreen(int worker_count) {
    const auto data = MakeCertificateData(kCampaignCount, kCampaignDimension);
    auto index =
        CampaignHnsw::Make(kCampaignChunkSize, kCampaignMaxChunks, kCampaignDimension, kCampaignM, kCampaignEfConstruction);
    std::shared_ptr<BuildOverlapGate> overlap_gate;
    if (worker_count > 1) {
        overlap_gate = std::make_shared<BuildOverlapGate>();
        index->SetBuildEntryHookForTest([overlap_gate](infinity::VertexType) { overlap_gate->Arrive(); });
    }
    const std::size_t memory_before = index->mem_usage();
    infinity::DenseVectorIter<float, CampaignLabel> iterator(data.data(), kCampaignDimension, kCampaignCount);
    infinity::ctpl::thread_pool pool(worker_count);
    const infinity::HnswBulkBuildResult build_result =
        infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, pool);
    index->SetBuildEntryHookForTest({});
    index->Check();
    const bool worker_overlap = worker_count == 1 || overlap_gate->Passed();
    const bool memory_monotonic = index->mem_usage() >= memory_before;
    const bool owned_accounting = memory_monotonic && build_result.mem_usage_ == index->mem_usage() - memory_before;
    const bool complete_build =
        pool.size() == worker_count && build_result.start_ == 0 && build_result.end_ == kCampaignCount &&
        build_result.submitted_task_count_ == static_cast<std::size_t>(worker_count) && build_result.mem_usage_ > 0 &&
        index->GetVecNum() == kCampaignCount;
    const auto stats = index->GetIncrementalReciprocalStats();
    const std::uint64_t successful = stats.unchanged_new_farthest + stats.unchanged_rejected + stats.updated_full + stats.updated_underfull;
    const std::uint64_t fallback = stats.fallback_invalid_state + stats.fallback_scratch_capacity + stats.fallback_duplicate +
                                   stats.fallback_center_tie + stats.fallback_nonfinite_center + stats.fallback_uncertified_order;
    const std::uint64_t baseline = stats.baseline_center_distance_evaluations + stats.baseline_pair_distance_evaluations;
    const bool accounting = stats.reciprocal_links == stats.direct_appends + stats.full_overflows &&
                            stats.full_overflows == stats.certificate_hits + stats.certificate_misses &&
                            stats.certificate_hits == successful + fallback;
    const double coverage =
        stats.full_overflows == 0 ? 0.0 : static_cast<double>(stats.shadow_comparisons) / static_cast<double>(stats.full_overflows);
    const double net =
        baseline == 0
            ? 0.0
            : (static_cast<double>(stats.predicted_avoided_distance_evaluations) - static_cast<double>(stats.predicted_extra_distance_evaluations)) /
                  static_cast<double>(baseline);
    std::cout << "shadow_workers=" << worker_count << " vectors=" << kCampaignCount << " dimensions=" << kCampaignDimension
              << " chunk_size=" << kCampaignChunkSize << " max_chunks=" << kCampaignMaxChunks << " M=" << kCampaignM
              << " ef_construction=" << kCampaignEfConstruction << " submitted_tasks=" << build_result.submitted_task_count_
              << " complete_build=" << complete_build << " owned_accounting=" << owned_accounting << " worker_overlap=" << worker_overlap
              << " reciprocal_links=" << stats.reciprocal_links << " full_overflows=" << stats.full_overflows
              << " certificate_sets=" << stats.certificate_sets << " certificate_hits=" << stats.certificate_hits
              << " certificate_misses=" << stats.certificate_misses << " comparisons=" << stats.shadow_comparisons
              << " mismatches=" << stats.shadow_mismatches << '\n';
    return {
        .passed =
            complete_build && owned_accounting && worker_overlap && accounting && stats.shadow_mismatches == 0 && stats.shadow_comparisons > 0,
        .coverage = coverage,
        .net_predicted_work = net,
    };
}

bool CheckShadowScreens() {
    const ShadowScreen one_worker = RunShadowScreen(1);
    const ShadowScreen twelve_workers = RunShadowScreen(12);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "shadow_one_worker_coverage=" << one_worker.coverage << '\n';
    std::cout << "shadow_one_worker_net_predicted_work=" << one_worker.net_predicted_work << '\n';
    std::cout << "shadow_twelve_worker_coverage=" << twelve_workers.coverage << '\n';
    std::cout << "shadow_twelve_worker_net_predicted_work=" << twelve_workers.net_predicted_work << '\n';
    return one_worker.passed && twelve_workers.passed;
}

#endif

#endif

} // namespace

int main() {
    try {
        Reporter reporter;
        auto graph = MakeDeterministicGraph();
        graph->Check();
        const std::vector<char> serialized = SerializePointerImage(*graph);
        const std::uint64_t graph_hash = GraphHash(*graph);
        const std::uint64_t serialized_hash = HashBytes(serialized);
        reporter.Check("deterministic_graph", !serialized.empty() && graph->GetVecNum() == 192);
        reporter.Check("owned_capacity_accounting", CheckOwnedCapacityAccounting());
        reporter.Check("null_empty_dense_vector_iterator", CheckNullEmptyDenseVectorIterator());
        reporter.Check("dense_vector_split_with_narrow_label", CheckDenseVectorSplitWithNarrowLabel());
        std::cout << "graph_hash=" << graph_hash << '\n';
        std::cout << "serialized_hash=" << serialized_hash << '\n';

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        reporter.Check("incremental_helper_differential", CheckIncrementalHelper());
        reporter.Check("production_lifecycle_semantics", CheckProductionLifecycleSemantics());
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS) || defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        reporter.Check("staged_publication_lock_hierarchy", CheckStagedPublicationLockHierarchy());
        reporter.Check("lvq_compression_cleanup", CheckLVQCompressionCleanup());
        reporter.Check("rabitq_compression_cleanup", CheckRabitqCompressionCleanup());
        reporter.Check("incremental_lifecycle", CheckPersistenceAndLifecycle());
        reporter.Check("repair_retains_certificates", CheckRepairRetainsCertificates());
#endif

#if defined(INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL) && defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
        reporter.Check("campaign_parameter_control_treatment_exactness", CheckCampaignParameterControlTreatmentExactness());
        reporter.Check("mutable_distance_accessor_inert_exactness", CheckMutableDistanceAccessorIsInertAndExact());
        reporter.Check("diagnostic_control_treatment", CheckDiagnosticControlTreatment());
        reporter.Check("concurrent_treatment_mutation", CheckConcurrentTreatmentMutation());
#endif

#if defined(INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        reporter.Check("shadow_screens", CheckShadowScreens());
#endif

        std::cout << "status=" << (reporter.passed() ? "PASS" : "FAIL") << '\n';
        return reporter.passed() ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "incremental reciprocal test failed: " << error.what() << '\n';
        return 2;
    } catch (...) {
        std::cerr << "incremental reciprocal test failed with a non-standard exception\n";
        return 2;
    }
}
