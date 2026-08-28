import std;
import std.compat;
import infinity_core;

namespace {

struct AllocationPause {
    std::binary_semaphore started{0};
    std::binary_semaphore release{0};
};

thread_local bool fail_next_array_allocation{};
thread_local AllocationPause *pause_next_array_allocation{};
thread_local std::optional<std::size_t> fail_allocation_ordinal{};
thread_local std::size_t allocation_ordinal{};

void MaybeFailAllocation() {
    if (!fail_allocation_ordinal) {
        return;
    }
    if (allocation_ordinal++ == *fail_allocation_ordinal) {
        fail_allocation_ordinal.reset();
        throw std::bad_alloc();
    }
}

void FailAllocationAt(std::size_t ordinal) {
    allocation_ordinal = 0;
    fail_allocation_ordinal = ordinal;
}

void DisableAllocationFailure() { fail_allocation_ordinal.reset(); }

} // namespace

void *operator new(std::size_t size) {
    MaybeFailAllocation();
    if (void *ptr = std::malloc(size == 0 ? 1 : size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
    if (auto *pause = std::exchange(pause_next_array_allocation, nullptr)) {
        pause->started.release();
        pause->release.acquire();
    }
    if (std::exchange(fail_next_array_allocation, false)) {
        throw std::bad_alloc();
    }
    MaybeFailAllocation();
    if (void *ptr = std::malloc(size == 0 ? 1 : size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete(void *ptr) noexcept { std::free(ptr); }

void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }

void operator delete[](void *ptr) noexcept { std::free(ptr); }

void operator delete[](void *ptr, std::size_t) noexcept { std::free(ptr); }

namespace {

struct MockIterator {};
struct MockConfig {};

class MockIndex {
public:
    MockIndex(std::size_t start, std::size_t end) : start_(start), end_(end) {}

    std::size_t mem_usage() const {
        std::lock_guard lock(mutex_);
        return built_.size();
    }

    std::unique_lock<std::shared_mutex> AcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::shared_mutex> TryAcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    std::pair<std::size_t, std::size_t> StoreData(MockIterator &&, const MockConfig &) {
        ++store_calls_;
        return {start_, end_};
    }

    void Build(std::size_t vertex) {
        std::lock_guard lock(mutex_);
        built_.push_back(vertex);
    }

    std::size_t store_calls() const { return store_calls_; }

    std::vector<std::size_t> built() const {
        std::lock_guard lock(mutex_);
        return built_;
    }

private:
    std::size_t start_;
    std::size_t end_;
    std::size_t store_calls_{};
    std::shared_mutex operation_mutex_;
    mutable std::mutex mutex_;
    std::vector<std::size_t> built_;
};

class PlannedMockIndex {
public:
    PlannedMockIndex(std::size_t start, std::size_t end) : start_(start), end_(end) {}

    std::size_t mem_usage() const {
        std::lock_guard lock(mutex_);
        return built_.size();
    }

    std::unique_lock<std::shared_mutex> AcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::shared_mutex> TryAcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    std::pair<std::size_t, std::size_t> StoreData(MockIterator &&, const MockConfig &) { return {start_, end_}; }

    std::vector<infinity::LayerSize> PrepareBuildLevels(std::size_t start, std::size_t end) {
        std::vector<infinity::LayerSize> levels(end - start);
        for (std::size_t ordinal = start; ordinal < end; ++ordinal) {
            levels[ordinal - start] = static_cast<infinity::LayerSize>((ordinal * 7) % 5);
        }
        return levels;
    }

    void Build(std::size_t vertex, infinity::LayerSize level) {
        std::lock_guard lock(mutex_);
        built_.emplace_back(vertex, level);
    }

    std::vector<std::pair<std::size_t, infinity::LayerSize>> built() const {
        std::lock_guard lock(mutex_);
        return built_;
    }

private:
    std::size_t start_;
    std::size_t end_;
    std::shared_mutex operation_mutex_;
    mutable std::mutex mutex_;
    std::vector<std::pair<std::size_t, infinity::LayerSize>> built_;
};

class PlannedFailureIndex {
public:
    std::size_t mem_usage() const { return built_.load(); }

    std::unique_lock<std::shared_mutex> AcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::shared_mutex> TryAcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    std::pair<std::size_t, std::size_t> StoreData(MockIterator &&, const MockConfig &) { return {0, 2}; }

    std::vector<infinity::LayerSize> PrepareBuildLevels(std::size_t start, std::size_t end) {
        return std::vector<infinity::LayerSize>(end - start, 0);
    }

    void Build(std::size_t vertex, infinity::LayerSize) {
        tasks_started_.count_down();
        tasks_started_.wait();
        if (vertex == 0) {
            throw std::runtime_error("injected planned build failure");
        }
        release_delayed_task_.acquire();
        ++built_;
    }

    void WaitUntilBothTasksStarted() { tasks_started_.wait(); }

    void ReleaseDelayedTask() { release_delayed_task_.release(); }

    std::size_t built() const { return built_.load(); }

private:
    std::shared_mutex operation_mutex_;
    std::latch tasks_started_{2};
    std::binary_semaphore release_delayed_task_{0};
    std::atomic<std::size_t> built_{};
};

class SubmissionFailureIndex {
public:
    std::size_t mem_usage() const { return built_.load(); }

    std::unique_lock<std::shared_mutex> AcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::shared_mutex> TryAcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    std::pair<std::size_t, std::size_t> StoreData(MockIterator &&, const MockConfig &) { return {0, 2}; }

    void Build(std::size_t vertex) {
        if (vertex == 0) {
            first_task_started_.release();
            release_first_task_.acquire();
        }
        ++built_;
    }

    void WaitUntilFirstTaskStarted() { first_task_started_.acquire(); }

    void ReleaseFirstTask() { release_first_task_.release(); }

    std::size_t built() const { return built_.load(); }

    void MarkBuildFailed() noexcept {
        poisoned_.store(true);
        poisoned_.notify_all();
    }

    bool IsBuildFailed() const { return poisoned_.load(); }

    void WaitUntilBuildFailed() const {
        while (!poisoned_.load()) {
            poisoned_.wait(false);
        }
    }

private:
    std::shared_mutex operation_mutex_;
    std::binary_semaphore first_task_started_{0};
    std::binary_semaphore release_first_task_{0};
    std::atomic<std::size_t> built_{};
    std::atomic<bool> poisoned_{};
};

class StrongThrowingPool {
public:
    static constexpr bool push_has_strong_exception_guarantee = true;

    StrongThrowingPool(int worker_count, std::size_t throw_on_push) : worker_count_(worker_count), throw_on_push_(throw_on_push) {}

    ~StrongThrowingPool() {
        for (auto &thread : threads_) {
            thread.join();
        }
    }

    int size() const { return worker_count_; }
    bool owns_current_thread() const noexcept { return false; }

    template <typename Function>
    std::future<void> push(Function function) {
        if (push_count_++ == throw_on_push_) {
            throw std::runtime_error("injected pre-acceptance submission failure");
        }
        std::packaged_task<void()> task([function = std::move(function)] { function(0); });
        std::future<void> future = task.get_future();
        threads_.emplace_back(std::move(task));
        return future;
    }

private:
    int worker_count_;
    std::size_t throw_on_push_;
    std::size_t push_count_{};
    std::vector<std::thread> threads_;
};

class InlineFailureIndex {
public:
    std::size_t mem_usage() const { return attempt_mask_.load(); }

    std::unique_lock<std::shared_mutex> AcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::shared_mutex> TryAcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    std::pair<std::size_t, std::size_t> StoreData(MockIterator &&, const MockConfig &) { return {0, 4}; }

    void Build(std::size_t vertex) {
        attempt_mask_.fetch_or(std::size_t{1} << vertex);
        if (vertex == 0) {
            throw std::runtime_error("first inline build failure");
        }
        if (vertex == 2) {
            throw std::runtime_error("second inline build failure");
        }
    }

    std::size_t attempt_mask() const { return attempt_mask_.load(); }

private:
    std::shared_mutex operation_mutex_;
    std::atomic<std::size_t> attempt_mask_{};
};

class SameIndexContentionIndex {
public:
    std::size_t mem_usage() const { return built_.load(); }

    std::unique_lock<std::shared_mutex> AcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::shared_mutex> TryAcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    void EnsureAllVerticesBuiltWithOperationLockHeld() const {}

    std::pair<std::size_t, std::size_t> StoreData(MockIterator &&, const MockConfig &) {
        ++store_calls_;
        external_store_entered_.release();
        nested_attempt_finished_.acquire();
        return {0, 1};
    }

    std::pair<std::size_t, std::size_t> StoreDataWithOperationLockHeld(MockIterator &&iter, const MockConfig &config) {
        return StoreData(std::move(iter), config);
    }

    void Build(std::size_t) { ++built_; }

    void BuildWithOperationLockHeld(std::size_t vertex) { Build(vertex); }

    std::size_t FinalizeBuildWithOperationLockHeld() { return 0; }

    void WaitUntilExternalStoreEntered() { external_store_entered_.acquire(); }

    void NotifyNestedAttemptFinished() { nested_attempt_finished_.release(); }

    void MarkBuildFailed() noexcept { poisoned_.store(true); }

    std::size_t store_calls() const { return store_calls_.load(); }

    std::size_t built() const { return built_.load(); }

    bool IsBuildFailed() const { return poisoned_.load(); }

private:
    std::shared_mutex operation_mutex_;
    std::binary_semaphore external_store_entered_{0};
    std::binary_semaphore nested_attempt_finished_{0};
    std::atomic<std::size_t> store_calls_{};
    std::atomic<std::size_t> built_{};
    std::atomic<bool> poisoned_{};
};

class PartialHeldLegacyLockingIndex {
public:
    std::size_t mem_usage() const { return built_.load(); }

    std::unique_lock<std::timed_mutex> AcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_); }

    std::unique_lock<std::timed_mutex> TryAcquireExclusiveOperationLock() { return std::unique_lock(operation_mutex_, std::try_to_lock); }

    void EnsureAllVerticesBuiltWithOperationLockHeld() const {}

    std::pair<std::size_t, std::size_t> StoreDataWithOperationLockHeld(MockIterator &&, const MockConfig &) {
        ++held_store_calls_;
        return {0, 4};
    }

    std::pair<std::size_t, std::size_t> StoreData(MockIterator &&, const MockConfig &) {
        std::unique_lock lock(operation_mutex_, std::defer_lock);
        if (!lock.try_lock_for(std::chrono::milliseconds(100))) {
            throw std::logic_error("legacy StoreData was called while an outer operation lock was held");
        }
        ++public_store_calls_;
        return {0, 4};
    }

    void Build(std::size_t) {
        std::unique_lock lock(operation_mutex_, std::defer_lock);
        if (!lock.try_lock_for(std::chrono::milliseconds(100))) {
            throw std::logic_error("legacy Build was called while an outer operation lock was held");
        }
        ++built_;
    }

    std::size_t held_store_calls() const { return held_store_calls_.load(); }
    std::size_t public_store_calls() const { return public_store_calls_.load(); }
    std::size_t built() const { return built_.load(); }

private:
    std::timed_mutex operation_mutex_;
    std::atomic<std::size_t> held_store_calls_{};
    std::atomic<std::size_t> public_store_calls_{};
    std::atomic<std::size_t> built_{};
};

using Hnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, infinity::i32>;
using MappedHnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, infinity::i32, false>;
using Label64Hnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, infinity::u64>;
using MappedLabel64Hnsw = infinity::KnnHnsw<infinity::PlainL2VecStoreType<float>, infinity::u64, false>;

constexpr std::size_t kDimension = 4;

std::vector<float> MakeData(std::size_t count) {
    std::vector<float> data(count * kDimension);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < kDimension; ++column) {
            data[row * kDimension + column] = static_cast<float>((row * 37 + column * 17) % 251) / 251.0F;
        }
    }
    return data;
}

void InsertSequential(std::unique_ptr<Hnsw> &index, const std::vector<float> &data, std::size_t start, std::size_t count) {
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data() + start * kDimension, kDimension, count, static_cast<infinity::i32>(start));
    static_cast<void>(index->InsertVecs(std::move(iterator)));
}

void InsertBulk(std::unique_ptr<Hnsw> &index,
                const std::vector<float> &data,
                std::size_t start,
                std::size_t count,
                infinity::ctpl::thread_pool &pool) {
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data() + start * kDimension, kDimension, count, static_cast<infinity::i32>(start));
    static_cast<void>(infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{}, pool, 1));
}

bool LevelsEqual(const Hnsw &index, const std::vector<infinity::LayerSize> &expected) {
    if (index.GetVecNum() != expected.size()) {
        return false;
    }
    for (std::size_t vertex = 0; vertex < expected.size(); ++vertex) {
        if (index.GetGraphLevel(static_cast<infinity::VertexType>(vertex)) != expected[vertex]) {
            return false;
        }
    }
    return true;
}

template <typename LeftHnsw, typename RightHnsw>
bool GraphsEqual(const LeftHnsw &left, const RightHnsw &right) {
    if (left.GetVecNum() != right.GetVecNum() || left.GetGraphEnterPoint() != right.GetGraphEnterPoint()) {
        return false;
    }
    for (std::size_t vertex = 0; vertex < left.GetVecNum(); ++vertex) {
        const auto vertex_id = static_cast<infinity::VertexType>(vertex);
        const infinity::LayerSize level = left.GetGraphLevel(vertex_id);
        if (level != right.GetGraphLevel(vertex_id)) {
            return false;
        }
        for (infinity::LayerSize layer = 0; layer <= level; ++layer) {
            const auto [left_neighbors, left_count] = left.GetGraphNeighbors(vertex_id, layer);
            const auto [right_neighbors, right_count] = right.GetGraphNeighbors(vertex_id, layer);
            if (left_count != right_count ||
                !std::equal(left_neighbors, left_neighbors + left_count, right_neighbors, right_neighbors + right_count)) {
                return false;
            }
        }
    }
    return true;
}

std::uint64_t HashLevels(const std::vector<infinity::LayerSize> &levels) {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (infinity::LayerSize level : levels) {
        const std::uint32_t value = static_cast<std::uint32_t>(level);
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 0xffU;
            hash *= 1'099'511'628'211ULL;
        }
    }
    return hash;
}

bool CheckEmpty(std::size_t start, int worker_count) {
    auto index = std::make_unique<MockIndex>(start, start);
    infinity::ctpl::thread_pool pool(worker_count);
    const auto result = infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 4);
    return result.start_ == start && result.end_ == start && result.submitted_task_count_ == 0 && result.mem_usage_ == 0 &&
           index->store_calls() == 1 && index->built().empty();
}

bool CheckZeroWorkers() {
    auto index = std::make_unique<MockIndex>(0, 1);
    infinity::ctpl::thread_pool pool(0);
    try {
        static_cast<void>(infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1));
    } catch (const std::invalid_argument &) {
        return index->store_calls() == 0;
    }
    return false;
}

bool CheckBuckets() {
    auto index = std::make_unique<MockIndex>(3, 13);
    infinity::ctpl::thread_pool pool(2);
    const auto result = infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 4);
    auto built = index->built();
    std::sort(built.begin(), built.end());
    std::vector<std::size_t> expected(10);
    std::iota(expected.begin(), expected.end(), 3);
    return result.start_ == 3 && result.end_ == 13 && result.submitted_task_count_ == 2 && built == expected;
}

bool CheckOversizedBucket() {
    auto index = std::make_unique<MockIndex>(3, 13);
    infinity::ctpl::thread_pool pool(2);
    const auto result = infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, std::numeric_limits<std::size_t>::max());
    auto built = index->built();
    std::sort(built.begin(), built.end());
    std::vector<std::size_t> expected(10);
    std::iota(expected.begin(), expected.end(), 3);
    return result.start_ == 3 && result.end_ == 13 && result.submitted_task_count_ == 1 && built == expected;
}

bool CheckNestedOneWorker() {
    auto index = std::make_unique<MockIndex>(0, 10);
    infinity::ctpl::thread_pool pool(1);
    auto outer = pool.push([&](int) { return infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 2); });
    if (outer.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        return false;
    }
    const auto result = outer.get();
    auto built = index->built();
    std::sort(built.begin(), built.end());
    std::vector<std::size_t> expected(10);
    std::iota(expected.begin(), expected.end(), 0);
    return result.submitted_task_count_ == 0 && built == expected;
}

bool CheckNestedSaturatedPool() {
    auto left = std::make_unique<MockIndex>(0, 8);
    auto right = std::make_unique<MockIndex>(8, 16);
    infinity::ctpl::thread_pool pool(2);
    std::latch workers_started(2);
    auto run_nested = [&](auto &index) {
        workers_started.count_down();
        workers_started.wait();
        return infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 2);
    };
    auto left_result = pool.push([&](int) { return run_nested(left); });
    auto right_result = pool.push([&](int) { return run_nested(right); });
    if (left_result.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
        right_result.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        return false;
    }
    const auto left_build = left_result.get();
    const auto right_build = right_result.get();
    return left_build.submitted_task_count_ == 0 && right_build.submitted_task_count_ == 0 && left->built().size() == 8 && right->built().size() == 8;
}

bool CheckNestedInlineExceptionDraining() {
    auto index = std::make_unique<InlineFailureIndex>();
    infinity::ctpl::thread_pool pool(4);
    auto outer = pool.push([&](int) {
        try {
            static_cast<void>(infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1));
        } catch (const std::runtime_error &error) {
            return std::string_view(error.what()) == "first inline build failure" && index->attempt_mask() == 0b1111;
        }
        return false;
    });
    return outer.wait_for(std::chrono::seconds(2)) == std::future_status::ready && outer.get();
}

bool CheckBucketFailureMask(int worker_count, std::size_t expected_mask) {
    auto index = std::make_unique<InlineFailureIndex>();
    infinity::ctpl::thread_pool pool(worker_count);
    try {
        static_cast<void>(infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1));
    } catch (const std::runtime_error &) {
        return index->attempt_mask() == expected_mask;
    }
    return false;
}

bool CheckSameIndexNestedContention() {
    auto index = std::make_unique<SameIndexContentionIndex>();
    infinity::ctpl::thread_pool pool(1);
    auto nested = pool.push([&](int) {
        index->WaitUntilExternalStoreEntered();
        bool rejected = false;
        try {
            static_cast<void>(infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1));
        } catch (const std::logic_error &error) {
            rejected = std::string_view(error.what()) == "HNSW bulk build rejected same-index contention from a pool worker";
        }
        index->NotifyNestedAttemptFinished();
        return rejected;
    });

    auto external = std::async(std::launch::async, [&] { return infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1); });
    if (external.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
        nested.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        return false;
    }
    const auto result = external.get();
    return nested.get() && result.submitted_task_count_ == 1 && index->store_calls() == 1 && index->built() == 1 && !index->IsBuildFailed();
}

bool CheckPartialHeldApiUsesLegacyLocking() {
    auto index = std::make_unique<PartialHeldLegacyLockingIndex>();
    infinity::ctpl::thread_pool pool(2);
    const auto result = infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1);
    return result.start_ == 0 && result.end_ == 4 && result.submitted_task_count_ == 2 && index->held_store_calls() == 0 &&
           index->public_store_calls() == 1 && index->built() == 4;
}

bool CheckTaskExceptionDraining() {
    auto index = std::make_unique<PlannedFailureIndex>();
    infinity::ctpl::thread_pool pool(2);
    auto call = std::async(std::launch::async, [&] { return infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1); });
    index->WaitUntilBothTasksStarted();
    const bool blocked = call.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
    index->ReleaseDelayedTask();
    try {
        static_cast<void>(call.get());
    } catch (const std::runtime_error &) {
        return blocked && index->built() == 1;
    }
    return false;
}

bool CheckSubmissionExceptionDraining() {
    auto index = std::make_unique<SubmissionFailureIndex>();
    StrongThrowingPool pool(2, 1);
    auto call = std::async(std::launch::async, [&] { return infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1); });
    index->WaitUntilFirstTaskStarted();
    index->WaitUntilBuildFailed();
    const bool blocked = call.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
    index->ReleaseFirstTask();
    try {
        static_cast<void>(call.get());
    } catch (const std::runtime_error &) {
        return blocked && index->IsBuildFailed() && index->built() == 1;
    }
    return false;
}

bool CheckPlannedLevels(int worker_count) {
    auto index = std::make_unique<PlannedMockIndex>(3, 27);
    infinity::ctpl::thread_pool pool(worker_count);
    const auto result = infinity::HnswBulkBuild(index, MockIterator{}, MockConfig{}, pool, 1);
    auto built = index->built();
    std::sort(built.begin(), built.end());
    std::vector<std::pair<std::size_t, infinity::LayerSize>> expected;
    for (std::size_t vertex = 3; vertex < 27; ++vertex) {
        expected.emplace_back(vertex, static_cast<infinity::LayerSize>((vertex * 7) % 5));
    }
    return result.submitted_task_count_ == static_cast<std::size_t>(worker_count) && built == expected;
}

struct CanonicalPlanningResult {
    bool passed;
    std::uint64_t hash;
};

CanonicalPlanningResult CheckCanonicalLevelPlanning() {
    constexpr infinity::VertexType kCount = 12'288;
    constexpr infinity::VertexType kSplit = 4'096;
    constexpr std::uint64_t kExpectedHash = 17'633'620'441'343'456'183ULL;

    auto whole_index = Hnsw::Make(kSplit, 3, 1, 32, 200);
    const auto whole = whole_index->PrepareBuildLevels(0, kCount);

    auto split_index = Hnsw::Make(kSplit, 3, 1, 32, 200);
    const auto prefix = split_index->PrepareBuildLevels(0, kSplit);
    const auto suffix = split_index->PrepareBuildLevels(kSplit, kCount);
    std::vector<infinity::LayerSize> split;
    split.reserve(kCount);
    split.insert(split.end(), prefix.begin(), prefix.end());
    split.insert(split.end(), suffix.begin(), suffix.end());

    const auto old_range = split_index->PrepareBuildLevels(17, 113);
    const std::vector<infinity::LayerSize> canonical_old_range(whole.begin() + 17, whole.begin() + 113);
    const auto extension = split_index->PrepareBuildLevels(kCount, kCount + 128);
    const auto whole_extension = whole_index->PrepareBuildLevels(kCount, kCount + 128);

    std::array<std::size_t, 4> histogram{};
    for (infinity::LayerSize level : whole) {
        if (level < 0 || static_cast<std::size_t>(level) >= histogram.size()) {
            return {.passed = false, .hash = HashLevels(whole)};
        }
        ++histogram[static_cast<std::size_t>(level)];
    }
    const std::uint64_t hash = HashLevels(whole);
    const bool passed = whole == split && old_range == canonical_old_range && extension == whole_extension &&
                        histogram == std::array<std::size_t, 4>{11'896, 381, 10, 1} && hash == kExpectedHash;
    return {.passed = passed, .hash = hash};
}

struct ConnectivityRepairResult {
    bool passed;
    std::size_t reachable_before;
    std::size_t repair_count;
    std::size_t reachable_after;
    std::size_t bulk_reachable;
    std::size_t appended_sources;
    std::size_t replacement_sources;
};

ConnectivityRepairResult CheckDeterministicConnectivityRepair() {
    constexpr std::size_t kCount = 5'000;
    constexpr std::size_t kRegressionDimension = 8;
    constexpr std::size_t kChunkSize = 8'192;
    std::vector<float> data(kCount * kRegressionDimension);
    std::mt19937 generator(0);
    std::uniform_real_distribution<float> distribution;
    for (float &value : data) {
        value = distribution(generator);
    }

    auto reachable_count = [](const Hnsw &index) {
        const std::size_t count = index.GetVecNum();
        std::vector<std::uint8_t> visited(count, 0);
        std::vector<infinity::VertexType> pending;
        pending.reserve(count);
        const auto [max_layer, entry_point] = index.GetGraphEnterPoint();
        static_cast<void>(max_layer);
        if (entry_point < 0 || static_cast<std::size_t>(entry_point) >= count) {
            return std::size_t{0};
        }
        visited[static_cast<std::size_t>(entry_point)] = 1;
        pending.push_back(entry_point);
        for (std::size_t cursor = 0; cursor < pending.size(); ++cursor) {
            const auto [neighbors, degree] = index.GetGraphNeighbors(pending[cursor], 0);
            for (infinity::VertexListSize neighbor_index = 0; neighbor_index < degree; ++neighbor_index) {
                const infinity::VertexType neighbor = neighbors[neighbor_index];
                if (neighbor < 0 || static_cast<std::size_t>(neighbor) >= count) {
                    return std::size_t{0};
                }
                if (visited[static_cast<std::size_t>(neighbor)] == 0) {
                    visited[static_cast<std::size_t>(neighbor)] = 1;
                    pending.push_back(neighbor);
                }
            }
        }
        return pending.size();
    };

    auto explicit_index = Hnsw::Make(kChunkSize, 1, kRegressionDimension, 2, 20);
    infinity::DenseVectorIter<float, infinity::i32> explicit_iterator(data.data(), kRegressionDimension, kCount);
    static_cast<void>(explicit_index->StoreData(std::move(explicit_iterator), infinity::HnswInsertConfig{.optimize_ = true}));
    const auto levels = explicit_index->PrepareBuildLevels(0, kCount);
    std::size_t reachable_before;
    std::size_t repair_count;
    std::size_t reachable_after;
    std::size_t appended_sources = 0;
    std::size_t replacement_sources = 0;
    {
        auto operation_lock = explicit_index->AcquireExclusiveOperationLock();
        for (std::size_t vertex = 0; vertex < kCount; ++vertex) {
            explicit_index->BuildWithOperationLockHeld(static_cast<infinity::VertexType>(vertex), levels[vertex]);
        }
        std::vector<std::vector<infinity::VertexType>> neighbors_before(kCount);
        for (std::size_t vertex = 0; vertex < kCount; ++vertex) {
            const auto [neighbors, degree] =
                explicit_index->GetGraphNeighbors(static_cast<infinity::VertexType>(vertex), 0);
            neighbors_before[vertex].assign(neighbors, neighbors + degree);
        }
        reachable_before = reachable_count(*explicit_index);
        repair_count = explicit_index->FinalizeBuildWithOperationLockHeld();
        reachable_after = reachable_count(*explicit_index);
        for (std::size_t vertex = 0; vertex < kCount; ++vertex) {
            const auto [neighbors, degree] =
                explicit_index->GetGraphNeighbors(static_cast<infinity::VertexType>(vertex), 0);
            const std::vector<infinity::VertexType> neighbors_after(neighbors, neighbors + degree);
            if (neighbors_after == neighbors_before[vertex]) {
                continue;
            }
            if (neighbors_after.size() > neighbors_before[vertex].size()) {
                ++appended_sources;
            } else if (neighbors_after.size() == neighbors_before[vertex].size()) {
                ++replacement_sources;
            }
        }
    }
    explicit_index->Check();

    auto bulk_index = Hnsw::Make(kChunkSize, 1, kRegressionDimension, 2, 20);
    infinity::DenseVectorIter<float, infinity::i32> bulk_iterator(data.data(), kRegressionDimension, kCount);
    infinity::ctpl::thread_pool pool(1);
    static_cast<void>(infinity::HnswBulkBuild(bulk_index, std::move(bulk_iterator), infinity::HnswInsertConfig{.optimize_ = true}, pool));
    const std::size_t bulk_reachable = reachable_count(*bulk_index);
    bulk_index->Check();

    return {
        .passed = reachable_before < kCount && repair_count > 0 && reachable_after == kCount && bulk_reachable == kCount &&
                  appended_sources > 0 && replacement_sources > 0,
        .reachable_before = reachable_before,
        .repair_count = repair_count,
        .reachable_after = reachable_after,
        .bulk_reachable = bulk_reachable,
        .appended_sources = appended_sources,
        .replacement_sources = replacement_sources,
    };
}

bool CheckMixedStagedAppendRejection() {
    constexpr std::size_t kCount = 96;
    constexpr std::size_t kSplit = 48;
    constexpr std::size_t kChunkSize = 128;
    const auto data = MakeData(kCount);
    auto index = Hnsw::Make(kChunkSize, 1, kDimension, 8, 32);

    infinity::DenseVectorIter<float, infinity::i32> stored_iterator(data.data(), kDimension, kSplit);
    const auto [stored_begin, stored_end] = index->StoreData(std::move(stored_iterator));
    if (stored_begin != 0 || stored_end != kSplit || index->GetVecNum() != kSplit || index->IsBuildFailed()) {
        return false;
    }

    bool sequential_rejected = false;
    infinity::DenseVectorIter<float, infinity::i32> inserted_iterator(
        data.data() + kSplit * kDimension, kDimension, kCount - kSplit, static_cast<infinity::i32>(kSplit));
    try {
        static_cast<void>(index->InsertVecs(std::move(inserted_iterator)));
    } catch (const std::logic_error &) {
        sequential_rejected = true;
    }

    bool bulk_rejected = false;
    infinity::DenseVectorIter<float, infinity::i32> bulk_iterator(
        data.data() + kSplit * kDimension, kDimension, kCount - kSplit, static_cast<infinity::i32>(kSplit));
    infinity::ctpl::thread_pool pool(2);
    try {
        static_cast<void>(infinity::HnswBulkBuild(index, std::move(bulk_iterator), infinity::HnswInsertConfig{}, pool, 1));
    } catch (const std::logic_error &) {
        bulk_rejected = true;
    }

    if (!sequential_rejected || !bulk_rejected || index->GetVecNum() != kSplit || index->IsBuildFailed() ||
        index->GetGraphEnterPoint() != std::pair<infinity::i32, infinity::VertexType>{-1, -1}) {
        return false;
    }

    for (std::size_t vertex = 0; vertex < kSplit; ++vertex) {
        if (index->GetGraphLevel(static_cast<infinity::VertexType>(vertex)) != -1) {
            return false;
        }
    }
    for (std::size_t vertex = 0; vertex < kSplit; ++vertex) {
        index->Build(static_cast<infinity::VertexType>(vertex));
    }
    infinity::DenseVectorIter<float, infinity::i32> retry_iterator(
        data.data() + kSplit * kDimension, kDimension, kCount - kSplit, static_cast<infinity::i32>(kSplit));
    const auto [inserted_begin, inserted_end] = index->InsertVecs(std::move(retry_iterator));
    index->Check();
    return inserted_begin == kSplit && inserted_end == kCount && !index->IsBuildFailed() && index->GetVecNum() == kCount &&
           index->FinalizeBuild() == 0;
}

bool CheckInvalidMRejection() {
    auto rejected = [](std::size_t m) {
        try {
            static_cast<void>(Hnsw::Make(1, 1, 1, m, 2));
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    constexpr std::size_t kTooLarge = static_cast<std::size_t>(std::numeric_limits<infinity::VertexListSize>::max()) / 2 + 1;
    return rejected(0) && rejected(1) && rejected(kTooLarge);
}

bool CheckMixedBuildLevels() {
    constexpr std::size_t kCount = 384;
    constexpr std::size_t kChunkSize = 512;
    constexpr std::size_t kSplit = 127;
    const auto data = MakeData(kCount);

    auto planner = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    const auto canonical = planner->PrepareBuildLevels(0, static_cast<infinity::VertexType>(kCount));

    auto sequential_then_bulk = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(sequential_then_bulk, data, 0, kSplit);
    infinity::ctpl::thread_pool suffix_pool(4);
    InsertBulk(sequential_then_bulk, data, kSplit, kCount - kSplit, suffix_pool);

    auto bulk_then_sequential = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    infinity::ctpl::thread_pool prefix_pool(4);
    InsertBulk(bulk_then_sequential, data, 0, kSplit, prefix_pool);
    InsertSequential(bulk_then_sequential, data, kSplit, kCount - kSplit);

    auto contaminator = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(contaminator, data, 0, 73);
    auto independent_sequential = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(independent_sequential, data, 0, kCount);

    return LevelsEqual(*sequential_then_bulk, canonical) && LevelsEqual(*bulk_then_sequential, canonical) &&
           LevelsEqual(*independent_sequential, canonical);
}

bool CheckSaveLoadAppend() {
    constexpr std::size_t kCount = 192;
    constexpr std::size_t kChunkSize = 256;
    constexpr std::size_t kSplit = 71;
    const auto data = MakeData(kCount);

    auto uninterrupted = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(uninterrupted, data, 0, kCount);

    auto persisted = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(persisted, data, 0, kSplit);
    infinity::LocalFileHandle file;
    persisted->Save(file);
    file.Rewind();
    auto loaded = Hnsw::Load(file);
    InsertSequential(loaded, data, kSplit, kCount - kSplit);

    return GraphsEqual(*uninterrupted, *loaded);
}

bool CheckPointerRoundTrip() {
    constexpr std::size_t kCount = 128;
    constexpr std::size_t kChunkSize = 256;
    const auto data = MakeData(kCount);

    auto index = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(index, data, 0, kCount);

    infinity::LocalFileHandle file;
    index->SaveToPtr(file);
    const std::size_t size = file.Size();
    file.Rewind();
    auto loaded = Hnsw::LoadFromPtr(file, size);
    loaded->Check();
    return index->FinalizeBuild() == 0 && GraphsEqual(*index, *loaded);
}

bool CheckPackedMappedLabels() {
    constexpr std::size_t kCount = 3;
    const auto data = MakeData(kCount);
    auto index = Label64Hnsw::Make(4, 1, kDimension, 2, 8);
    infinity::DenseVectorIter<float, infinity::u64> iterator(data.data(), kDimension, kCount);
    static_cast<void>(index->StoreData(std::move(iterator)));
    index->Build(0, 1);
    index->Build(1, 0);
    index->Build(2, 0);

    infinity::LocalFileHandle serialized;
    index->SaveToPtr(serialized);
    std::vector<char> image(serialized.Size());
    serialized.Rewind();
    serialized.Read(image.data(), image.size());

    const char *cursor = image.data();
    auto mapped = MappedLabel64Hnsw::LoadFromPtr(cursor, image.size());
    mapped->Check();
    for (std::size_t vertex = 0; vertex < kCount; ++vertex) {
        if (mapped->GetLabel(static_cast<infinity::VertexType>(vertex)) != vertex) {
            return false;
        }
    }
    return cursor == image.data() + image.size() && GraphsEqual(*index, *mapped);
}

bool CheckPointerImageValidation() {
    constexpr std::size_t kCount = 3;
    constexpr std::size_t kChunkSize = 4;
    const auto data = MakeData(kCount);
    auto index = Hnsw::Make(kChunkSize, 1, kDimension, 2, 8);
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), kDimension, kCount);
    static_cast<void>(index->StoreData(std::move(iterator)));
    index->Build(0, 1);
    index->Build(1, 0);
    index->Build(2, 0);

    infinity::LocalFileHandle serialized;
    index->SaveToPtr(serialized);
    std::vector<char> image(serialized.Size());
    serialized.Rewind();
    serialized.Read(image.data(), image.size());

    auto owning_rejects = [](const std::vector<char> &candidate, std::string_view case_name) {
        infinity::LocalFileHandle file;
        file.Append(candidate.data(), candidate.size());
        file.Rewind();
        try {
            static_cast<void>(Hnsw::LoadFromPtr(file, candidate.size()));
        } catch (const std::invalid_argument &) {
            return true;
        } catch (const std::exception &error) {
            std::cerr << "pointer-image owning loader threw an unexpected exception for '" << case_name << "' ("
                      << candidate.size() << " bytes): " << error.what() << '\n';
            return false;
        } catch (...) {
            std::cerr << "pointer-image owning loader threw an unexpected non-standard exception for '" << case_name << "' ("
                      << candidate.size() << " bytes)\n";
            return false;
        }
        std::cerr << "pointer-image owning loader accepted '" << case_name << "' (" << candidate.size() << " bytes)\n";
        return false;
    };
    auto mapped_rejects = [](const std::vector<char> &candidate, std::string_view case_name) {
        const char *cursor = candidate.data();
        const char *const original = cursor;
        try {
            static_cast<void>(MappedHnsw::LoadFromPtr(cursor, candidate.size()));
        } catch (const std::invalid_argument &) {
            if (cursor == original) {
                return true;
            }
            std::cerr << "pointer-image mapped loader advanced its cursor before rejecting '" << case_name << "' ("
                      << candidate.size() << " bytes)\n";
            return false;
        } catch (const std::exception &error) {
            std::cerr << "pointer-image mapped loader threw an unexpected exception for '" << case_name << "' ("
                      << candidate.size() << " bytes): " << error.what() << '\n';
            return false;
        } catch (...) {
            std::cerr << "pointer-image mapped loader threw an unexpected non-standard exception for '" << case_name << "' ("
                      << candidate.size() << " bytes)\n";
            return false;
        }
        std::cerr << "pointer-image mapped loader accepted '" << case_name << "' (" << candidate.size() << " bytes)\n";
        return false;
    };
    auto both_reject = [&](const std::vector<char> &candidate, std::string_view case_name) {
        return owning_rejects(candidate, case_name) && mapped_rejects(candidate, case_name);
    };
    auto overwrite = [](std::vector<char> &candidate, std::size_t offset, auto value) {
        if (offset + sizeof(value) > candidate.size()) {
            return false;
        }
        std::memcpy(candidate.data() + offset, &value, sizeof(value));
        return true;
    };

    for (std::size_t size = 0; size < image.size(); ++size) {
        const std::vector<char> truncated(image.begin(), image.begin() + size);
        if (!both_reject(truncated, "truncated image")) {
            return false;
        }
    }

    std::vector<char> trailing = image;
    trailing.push_back('\0');
    if (!both_reject(trailing, "trailing byte")) {
        return false;
    }

    constexpr std::size_t kMOffset = 0;
    constexpr std::size_t kVectorCountOffset = 2 * sizeof(std::size_t);
    constexpr std::size_t kDimensionOffset = 3 * sizeof(std::size_t);
    constexpr std::size_t kMmax0Offset = 4 * sizeof(std::size_t);
    constexpr std::size_t kEntryPointOffset = 6 * sizeof(std::size_t) + sizeof(infinity::i32);
    constexpr std::size_t kVectorDataOffset = 6 * sizeof(std::size_t) + 2 * sizeof(infinity::i32);
    constexpr std::size_t kLayerSumOffset = kVectorDataOffset + kCount * kDimension * sizeof(float);
    constexpr std::size_t kGraphOffset = kLayerSumOffset + sizeof(std::size_t);
    constexpr std::size_t kLayerPointerOffset = 2 * sizeof(infinity::i32);
    constexpr std::size_t kNeighborCountOffset = kLayerPointerOffset + sizeof(void *);

    std::vector<std::vector<char>> malformed;
    malformed.push_back(image);
    overwrite(malformed.back(), kMOffset, std::size_t{1});
    malformed.push_back(image);
    overwrite(malformed.back(), kVectorCountOffset, std::numeric_limits<std::size_t>::max());
    malformed.push_back(image);
    overwrite(malformed.back(), kDimensionOffset, std::numeric_limits<std::size_t>::max());
    malformed.push_back(image);
    overwrite(malformed.back(), kMmax0Offset, std::numeric_limits<std::size_t>::max());
    malformed.push_back(image);
    overwrite(malformed.back(), kEntryPointOffset, infinity::VertexType{kCount});
    malformed.push_back(image);
    overwrite(malformed.back(), kLayerSumOffset, std::numeric_limits<std::size_t>::max());
    malformed.push_back(image);
    overwrite(malformed.back(), kGraphOffset, infinity::LayerSize{infinity::kHnswMaxSupportedLayer + 1});
    malformed.push_back(image);
    overwrite(malformed.back(), kGraphOffset + kLayerPointerOffset, std::uintptr_t{1});
    malformed.push_back(image);
    overwrite(malformed.back(), kGraphOffset + kNeighborCountOffset, infinity::VertexListSize{5});

    constexpr std::array<std::string_view, 9> kMalformedCaseNames{
        "M below minimum",
        "vector count overflow",
        "dimension overflow",
        "level-zero capacity overflow",
        "entry point out of range",
        "upper-layer count overflow",
        "maximum layer out of range",
        "serialized graph pointer is non-null",
        "level-zero degree exceeds capacity",
    };
    for (std::size_t case_index = 0; case_index < malformed.size(); ++case_index) {
        if (!both_reject(malformed[case_index], kMalformedCaseNames[case_index])) {
            return false;
        }
    }

    const char *mapped_cursor = image.data();
    auto mapped = MappedHnsw::LoadFromPtr(mapped_cursor, image.size());
    mapped->Check();
    infinity::LocalFileHandle owning_file;
    owning_file.Append(image.data(), image.size());
    owning_file.Rewind();
    auto owning = Hnsw::LoadFromPtr(owning_file, image.size());
    return mapped_cursor == image.data() + image.size() && GraphsEqual(*index, *owning) && owning->GetVecNum() == mapped->GetVecNum();
}

bool CheckConventionalStreamImageValidation() {
    constexpr std::size_t kCount = 3;
    constexpr std::size_t kChunkSize = 4;
    constexpr std::size_t kM = 2;
    const auto data = MakeData(kCount);
    auto index = Hnsw::Make(kChunkSize, 1, kDimension, kM, 8);
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), kDimension, kCount);
    static_cast<void>(index->StoreData(std::move(iterator)));
    index->Build(0, 1);
    index->Build(1, 1);
    index->Build(2, 0);

    infinity::LocalFileHandle serialized;
    index->Save(serialized);
    std::vector<char> image(serialized.Size());
    serialized.Rewind();
    serialized.Read(image.data(), image.size());

    auto rejects = [](const std::vector<char> &candidate, std::string_view case_name) {
        infinity::LocalFileHandle file;
        if (!candidate.empty()) {
            file.Append(candidate.data(), candidate.size());
        }
        file.Rewind();
        try {
            static_cast<void>(Hnsw::Load(file));
        } catch (const std::invalid_argument &) {
            return true;
        } catch (const std::exception &error) {
            std::cerr << "conventional HNSW loader threw an unexpected exception for '" << case_name << "' ("
                      << candidate.size() << " bytes): " << error.what() << '\n';
            return false;
        } catch (...) {
            std::cerr << "conventional HNSW loader threw an unexpected non-standard exception for '" << case_name << "' ("
                      << candidate.size() << " bytes)\n";
            return false;
        }
        std::cerr << "conventional HNSW loader accepted '" << case_name << "' (" << candidate.size() << " bytes)\n";
        return false;
    };
    auto read = []<typename Value>(const std::vector<char> &candidate, std::size_t offset) {
        if (offset > candidate.size() || sizeof(Value) > candidate.size() - offset) {
            throw std::out_of_range("conventional HNSW test image offset is out of range");
        }
        Value value;
        std::memcpy(&value, candidate.data() + offset, sizeof(value));
        return value;
    };
    auto overwrite = []<typename Value>(std::vector<char> &candidate, std::size_t offset, Value value) {
        if (offset > candidate.size() || sizeof(value) > candidate.size() - offset) {
            return false;
        }
        std::memcpy(candidate.data() + offset, &value, sizeof(value));
        return true;
    };
    auto align_up = [](std::size_t value, std::size_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    };

    for (std::size_t size = 0; size < image.size(); ++size) {
        const std::vector<char> truncated(image.begin(), image.begin() + size);
        if (!rejects(truncated, "truncated stream")) {
            return false;
        }
    }

    constexpr std::size_t kMOffset = 0;
    constexpr std::size_t kEfConstructionOffset = kMOffset + sizeof(std::size_t);
    constexpr std::size_t kChunkSizeOffset = kEfConstructionOffset + sizeof(std::size_t);
    constexpr std::size_t kMaxChunkCountOffset = kChunkSizeOffset + sizeof(std::size_t);
    constexpr std::size_t kVectorCountOffset = kMaxChunkCountOffset + sizeof(std::size_t);
    constexpr std::size_t kDimensionOffset = kVectorCountOffset + sizeof(std::size_t);
    constexpr std::size_t kMmax0Offset = kDimensionOffset + sizeof(std::size_t);
    constexpr std::size_t kMmaxOffset = kMmax0Offset + sizeof(std::size_t);
    constexpr std::size_t kMaxLayerOffset = kMmaxOffset + sizeof(std::size_t);
    constexpr std::size_t kEntryPointOffset = kMaxLayerOffset + sizeof(infinity::i32);
    constexpr std::size_t kVectorDataOffset = kEntryPointOffset + sizeof(infinity::VertexType);

    const std::size_t vector_count = read.template operator()<std::size_t>(image, kVectorCountOffset);
    const std::size_t dimension = read.template operator()<std::size_t>(image, kDimensionOffset);
    const std::size_t mmax0 = read.template operator()<std::size_t>(image, kMmax0Offset);
    const std::size_t mmax = read.template operator()<std::size_t>(image, kMmaxOffset);
    const std::size_t vector_data_size = vector_count * dimension * sizeof(float);
    const std::size_t layer_sum_offset = kVectorDataOffset + vector_data_size;
    const std::size_t graph_offset = layer_sum_offset + sizeof(std::size_t);
    const std::size_t layer_pointer_offset = align_up(sizeof(infinity::LayerSize), alignof(char *));
    const std::size_t level0_degree_offset = layer_pointer_offset + sizeof(char *);
    const std::size_t level0_neighbors_offset = level0_degree_offset + sizeof(infinity::VertexListSize);
    const std::size_t level0_header_size = align_up(level0_neighbors_offset, alignof(char *));
    const std::size_t level0_size = level0_header_size + mmax0 * sizeof(infinity::VertexType);
    const std::size_t upper_neighbors_offset = sizeof(infinity::VertexListSize);
    const std::size_t upper_size = upper_neighbors_offset + mmax * sizeof(infinity::VertexType);
    const std::size_t layer_sum = read.template operator()<std::size_t>(image, layer_sum_offset);
    const std::size_t upper_graph_offset = graph_offset + vector_count * level0_size;
    const std::size_t labels_offset = upper_graph_offset + layer_sum * upper_size;
    if (vector_count != kCount || dimension != kDimension || mmax0 != 2 * kM || mmax != kM ||
        labels_offset + vector_count * sizeof(infinity::i32) != image.size()) {
        return false;
    }

    std::optional<std::size_t> level0_edge_offset;
    std::optional<std::size_t> upper_degree_offset;
    std::optional<std::size_t> upper_edge_offset;
    std::optional<infinity::VertexType> lower_level_vertex;
    std::size_t upper_ordinal = 0;
    for (std::size_t vertex = 0; vertex < vector_count; ++vertex) {
        const std::size_t record_offset = graph_offset + vertex * level0_size;
        const auto level = read.template operator()<infinity::LayerSize>(image, record_offset);
        const auto degree =
            read.template operator()<infinity::VertexListSize>(image, record_offset + level0_degree_offset);
        if (level == 0 && !lower_level_vertex) {
            lower_level_vertex = static_cast<infinity::VertexType>(vertex);
        }
        if (degree > 0 && !level0_edge_offset) {
            level0_edge_offset = record_offset + level0_neighbors_offset;
        }
        for (infinity::LayerSize layer = 0; layer < level; ++layer, ++upper_ordinal) {
            const std::size_t upper_offset = upper_graph_offset + upper_ordinal * upper_size;
            const auto upper_degree = read.template operator()<infinity::VertexListSize>(image, upper_offset);
            if (upper_degree > 0 && !upper_edge_offset) {
                upper_degree_offset = upper_offset;
                upper_edge_offset = upper_offset + upper_neighbors_offset;
            }
        }
    }
    if (upper_ordinal != layer_sum || !level0_edge_offset || !upper_degree_offset || !upper_edge_offset ||
        !lower_level_vertex) {
        return false;
    }

    std::vector<std::vector<char>> malformed;
    auto add_mutation = [&](std::size_t offset, auto value) {
        malformed.push_back(image);
        return overwrite(malformed.back(), offset, value);
    };
    if (!add_mutation(kChunkSizeOffset, std::size_t{0}) || !add_mutation(kChunkSizeOffset, std::size_t{3}) ||
        !add_mutation(kMaxChunkCountOffset, std::size_t{0}) ||
        !add_mutation(kVectorCountOffset, kChunkSize + 1) ||
        !add_mutation(kDimensionOffset, std::numeric_limits<std::size_t>::max() / sizeof(float) + 1) ||
        !add_mutation(kMOffset, kM + 1) ||
        !add_mutation(kMaxLayerOffset, infinity::i32{infinity::kHnswMaxSupportedLayer + 1}) ||
        !add_mutation(kEntryPointOffset, infinity::VertexType{kCount}) ||
        !add_mutation(layer_sum_offset, layer_sum + 1) ||
        !add_mutation(graph_offset, infinity::LayerSize{infinity::kHnswMaxSupportedLayer + 1}) ||
        !add_mutation(graph_offset + level0_degree_offset,
                      static_cast<infinity::VertexListSize>(mmax0 + 1)) ||
        !add_mutation(*level0_edge_offset, infinity::VertexType{kCount}) ||
        !add_mutation(*upper_degree_offset, static_cast<infinity::VertexListSize>(mmax + 1)) ||
        !add_mutation(*upper_edge_offset, infinity::VertexType{kCount}) ||
        !add_mutation(*upper_edge_offset, *lower_level_vertex)) {
        return false;
    }
    constexpr std::array<std::string_view, 15> kMalformedCaseNames{
        "zero chunk size",
        "non-power-of-two chunk size",
        "zero maximum chunk count",
        "vector count exceeds capacity",
        "dimension byte size overflow",
        "M and graph capacities disagree",
        "maximum layer out of range",
        "entry point out of range",
        "upper-layer count mismatch",
        "vertex layer out of range",
        "level-zero degree exceeds capacity",
        "level-zero edge out of range",
        "upper-layer degree exceeds capacity",
        "upper-layer edge out of range",
        "upper-layer edge targets a lower-level vertex",
    };
    for (std::size_t case_index = 0; case_index < malformed.size(); ++case_index) {
        if (!rejects(malformed[case_index], kMalformedCaseNames[case_index])) {
            return false;
        }
    }

    infinity::LocalFileHandle valid_file;
    valid_file.Append(image.data(), image.size());
    valid_file.Rewind();
    auto loaded = Hnsw::Load(valid_file);
    loaded->Check();

    const auto append_data = MakeData(kCount + 1);
    InsertSequential(index, append_data, kCount, 1);
    InsertSequential(loaded, append_data, kCount, 1);
    loaded->Check();
    return GraphsEqual(*index, *loaded);
}

bool CheckBulkAppendAfterLoad() {
    constexpr std::size_t kCount = 192;
    constexpr std::size_t kChunkSize = 256;
    constexpr std::size_t kSplit = 73;
    const auto data = MakeData(kCount);

    auto uninterrupted = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(uninterrupted, data, 0, kCount);

    auto persisted = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    InsertSequential(persisted, data, 0, kSplit);
    infinity::LocalFileHandle file;
    persisted->Save(file);
    file.Rewind();
    auto loaded = Hnsw::Load(file);
    infinity::ctpl::thread_pool pool(1);
    InsertBulk(loaded, data, kSplit, kCount - kSplit, pool);
    loaded->Check();
    return GraphsEqual(*uninterrupted, *loaded);
}

bool CheckUnbuiltSerializationRejection() {
    auto index = Hnsw::Make(2, 1, kDimension, 2, 2);
    const auto data = MakeData(2);
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), kDimension, 2);
    static_cast<void>(index->StoreData(std::move(iterator)));

    infinity::LocalFileHandle stream_file;
    infinity::LocalFileHandle pointer_file;
    bool stream_rejected = false;
    bool pointer_rejected = false;
    try {
        index->Save(stream_file);
    } catch (const std::logic_error &) {
        stream_rejected = true;
    }
    try {
        index->SaveToPtr(pointer_file);
    } catch (const std::logic_error &) {
        pointer_rejected = true;
    }
    if (!stream_rejected || !pointer_rejected || stream_file.Size() != 0 || pointer_file.Size() != 0 || index->IsBuildFailed() ||
        index->GetGraphLevel(0) != -1 || index->GetGraphLevel(1) != -1) {
        return false;
    }

    index->Build(0, 0);
    index->Build(1, 0);
    index->Save(stream_file);
    index->SaveToPtr(pointer_file);
    return stream_file.Size() > 0 && pointer_file.Size() > 0 && !index->IsBuildFailed();
}

template <typename Compress>
bool CheckIncompleteCompressionRejectionOne(Compress compress) {
    constexpr std::size_t kCount = 8;
    const auto data = MakeData(kCount);
    auto index = Hnsw::Make(kCount, 1, kDimension, 4, 16);
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), kDimension, kCount);
    static_cast<void>(index->StoreData(std::move(iterator)));
    index->Build(0, 0);

    const auto enter_point = index->GetGraphEnterPoint();
    const auto memory = index->mem_usage();
    const auto [neighbors, degree] = index->GetGraphNeighbors(0, 0);
    const std::vector<infinity::VertexType> neighbor_snapshot(neighbors, neighbors + degree);

    bool rejected = false;
    try {
        static_cast<void>(compress(*index));
    } catch (const std::logic_error &error) {
        rejected = std::string_view(error.what()) == "HNSW index contains stored but unbuilt vertices";
    }
    const auto [neighbors_after, degree_after] = index->GetGraphNeighbors(0, 0);
    if (!rejected || index->IsBuildFailed() || index->GetVecNum() != kCount || index->GetGraphEnterPoint() != enter_point ||
        index->GetGraphLevel(0) != 0 || index->mem_usage() != memory || degree_after != degree ||
        !std::equal(neighbor_snapshot.begin(), neighbor_snapshot.end(), neighbors_after)) {
        return false;
    }
    for (std::size_t vertex = 1; vertex < kCount; ++vertex) {
        if (index->GetGraphLevel(static_cast<infinity::VertexType>(vertex)) != -1) {
            return false;
        }
        index->Build(static_cast<infinity::VertexType>(vertex), 0);
    }

    index->Check();
    infinity::LocalFileHandle file;
    index->Save(file);
    file.Rewind();
    auto loaded = Hnsw::Load(file);
    return GraphsEqual(*index, *loaded);
}

bool CheckIncompleteCompressionRejection() {
    const bool lvq = CheckIncompleteCompressionRejectionOne([](Hnsw &index) { return std::move(index).CompressToLVQ(); });
    const bool rabitq = CheckIncompleteCompressionRejectionOne([](Hnsw &index) { return std::move(index).CompressToRabitq(); });
    return lvq && rabitq;
}

bool SourceUsableAfterCompressionFailure(Hnsw &index,
                                         const Hnsw &baseline,
                                         const std::vector<float> &data,
                                         const std::vector<std::pair<float, infinity::i32>> &expected_search) {
    try {
        index.Check();
        if (index.IsBuildFailed() || !GraphsEqual(index, baseline) ||
            index.KnnSearchSorted(data.data() + 5 * kDimension, 8) != expected_search) {
            return false;
        }
        infinity::LocalFileHandle file;
        index.Save(file);
        file.Rewind();
        auto loaded = Hnsw::Load(file);
        return GraphsEqual(index, *loaded);
    } catch (...) {
        return false;
    }
}

struct CompressionAllocationResult {
    bool passed{};
    bool success_preserved{};
    std::size_t failed_allocation_count{};
};

template <typename Compress>
CompressionAllocationResult CheckCompressionAllocationFailures(Compress compress) {
    constexpr std::size_t kCount = 24;
    constexpr std::size_t kMaximumAllocationPoints = 128;
    const auto data = MakeData(kCount);
    auto baseline = Hnsw::Make(32, 1, kDimension, 4, 16);
    InsertSequential(baseline, data, 0, kCount);
    const auto expected_search = baseline->KnnSearchSorted(data.data() + 5 * kDimension, 8);

    std::size_t failed_allocation_count = 0;
    for (std::size_t ordinal = 0; ordinal < kMaximumAllocationPoints; ++ordinal) {
        auto index = Hnsw::Make(32, 1, kDimension, 4, 16);
        InsertSequential(index, data, 0, kCount);
        FailAllocationAt(ordinal);
        try {
            auto compressed = compress(*index);
            DisableAllocationFailure();

            bool success_preserved = false;
            try {
                compressed->Check();
                infinity::LocalFileHandle file;
                compressed->Save(file);
                file.Rewind();
                using CompressedHnsw = typename decltype(compressed)::element_type;
                auto loaded = CompressedHnsw::Load(file);
                success_preserved =
                    compressed->GetVecNum() == kCount && GraphsEqual(*baseline, *compressed) &&
                    compressed->KnnSearchSorted(data.data() + 5 * kDimension, 8) == expected_search &&
                    compressed->mem_usage() == loaded->mem_usage() && GraphsEqual(*compressed, *loaded) && file.Size() > 0;
            } catch (...) {
                success_preserved = false;
            }
            return {
                .passed = ordinal > 0 && failed_allocation_count == ordinal && success_preserved,
                .success_preserved = success_preserved,
                .failed_allocation_count = failed_allocation_count,
            };
        } catch (const std::bad_alloc &) {
            DisableAllocationFailure();
            ++failed_allocation_count;
            if (!SourceUsableAfterCompressionFailure(*index, *baseline, data, expected_search)) {
                return {.failed_allocation_count = failed_allocation_count};
            }
        } catch (...) {
            DisableAllocationFailure();
            return {.failed_allocation_count = failed_allocation_count};
        }
    }
    DisableAllocationFailure();
    return {.failed_allocation_count = failed_allocation_count};
}

bool CheckSerializationWaitsForBuild(bool save_to_ptr) {
    auto index = Hnsw::Make(1, 1, kDimension, 2, 2);
    const auto data = MakeData(1);
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), kDimension, 1);
    static_cast<void>(index->StoreData(std::move(iterator)));

    AllocationPause pause;
    auto build = std::async(std::launch::async, [&] {
        pause_next_array_allocation = &pause;
        index->Build(0, 1);
    });
    pause.started.acquire();

    infinity::LocalFileHandle file;
    auto save = std::async(std::launch::async, [&] {
        if (save_to_ptr) {
            index->SaveToPtr(file);
        } else {
            index->Save(file);
        }
    });
    const bool blocked = save.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
    pause.release.release();
    build.get();
    save.get();

    const std::size_t size = file.Size();
    file.Rewind();
    auto loaded = save_to_ptr ? Hnsw::LoadFromPtr(file, size) : Hnsw::Load(file);
    return blocked && GraphsEqual(*index, *loaded);
}

bool CheckDuplicateBuildRejection() {
    const auto data = MakeData(1);
    auto index = Hnsw::Make(1, 1, kDimension, 2, 2);
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), kDimension, 1);
    static_cast<void>(index->StoreData(std::move(iterator)));
    if (index->GetGraphLevel(0) != -1) {
        return false;
    }
    index->Build(0, 0);
    const auto enter_point = index->GetGraphEnterPoint();
    const auto memory = index->mem_usage();
    const auto [neighbors, neighbor_count] = index->GetGraphNeighbors(0, 0);
    const std::vector<infinity::VertexType> neighbor_snapshot(neighbors, neighbors + neighbor_count);

    bool sequential_rejected = false;
    try {
        index->Build(0);
    } catch (const std::logic_error &) {
        sequential_rejected = true;
    }
    const auto [neighbors_after, neighbor_count_after] = index->GetGraphNeighbors(0, 0);
    if (!sequential_rejected || index->IsBuildFailed() || index->GetGraphLevel(0) != 0 || index->GetGraphEnterPoint() != enter_point ||
        index->mem_usage() != memory || neighbor_count_after != neighbor_count ||
        !std::equal(neighbor_snapshot.begin(), neighbor_snapshot.end(), neighbors_after)) {
        return false;
    }

    infinity::LocalFileHandle file;
    index->Save(file);
    file.Rewind();
    auto loaded = Hnsw::Load(file);
    bool loaded_rejected = false;
    try {
        loaded->Build(0, 1);
    } catch (const std::logic_error &) {
        loaded_rejected = true;
    }
    if (!loaded_rejected || loaded->IsBuildFailed() || !GraphsEqual(*index, *loaded)) {
        return false;
    }

    auto concurrent = Hnsw::Make(1, 1, kDimension, 2, 2);
    infinity::DenseVectorIter<float, infinity::i32> concurrent_iterator(data.data(), kDimension, 1);
    static_cast<void>(concurrent->StoreData(std::move(concurrent_iterator)));
    std::latch start(2);
    auto build_once = [&] {
        start.count_down();
        start.wait();
        try {
            concurrent->Build(0, 0);
            return true;
        } catch (const std::logic_error &) {
            return false;
        }
    };
    auto first = std::async(std::launch::async, build_once);
    auto second = std::async(std::launch::async, build_once);
    const int success_count = static_cast<int>(first.get()) + static_cast<int>(second.get());
    concurrent->Check();
    return success_count == 1 && !concurrent->IsBuildFailed() && concurrent->GetGraphLevel(0) == 0;
}

bool CheckConcurrentBulkSerialization() {
    constexpr std::size_t kCount = 160;
    constexpr std::size_t kChunkSize = 256;
    constexpr std::size_t kSplit = 80;
    const auto data = MakeData(kCount);
    auto index = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    infinity::ctpl::thread_pool pool(4);

    auto first = std::async(std::launch::async, [&] { InsertBulk(index, data, 0, kSplit, pool); });
    auto second = std::async(std::launch::async, [&] { InsertBulk(index, data, kSplit, kCount - kSplit, pool); });
    first.get();
    second.get();

    auto planner = Hnsw::Make(kChunkSize, 1, kDimension, 32, 64);
    const auto canonical = planner->PrepareBuildLevels(0, static_cast<infinity::VertexType>(kCount));
    return LevelsEqual(*index, canonical);
}

bool CheckAllocationFailureBeforePublication() {
    auto index = Hnsw::Make(1, 1, 1, 2, 2);
    const std::array<float, 1> data{1.0F};
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), 1, 1);
    static_cast<void>(index->StoreData(std::move(iterator)));

    fail_next_array_allocation = true;
    try {
        index->Build(0, 1);
    } catch (const std::bad_alloc &) {
        const bool untouched = index->GetGraphLevel(0) == -1 && index->GetGraphEnterPoint() == std::pair<infinity::i32, infinity::VertexType>{-1, -1};
        bool retry_rejected = false;
        try {
            index->Build(0, 1);
        } catch (const std::logic_error &) {
            retry_rejected = true;
        }
        return untouched && index->IsBuildFailed() && retry_rejected;
    }
    return false;
}

bool CheckAllocationFailureAfterPublication() {
    auto index = Hnsw::Make(2, 1, 1, 2, 2);
    const std::array<float, 2> data{1.0F, 2.0F};
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), 1, 2);
    static_cast<void>(index->StoreData(std::move(iterator)));
    index->Build(0, 0);

    fail_next_array_allocation = true;
    try {
        index->Build(1, 0);
    } catch (const std::bad_alloc &) {
        auto rejects_failed_index = [](auto &&operation) {
            try {
                operation();
            } catch (const std::logic_error &error) {
                return std::string_view(error.what()) == "HNSW index is unusable after a failed build";
            }
            return false;
        };
        const bool retry_rejected = rejects_failed_index([&] { index->Build(1, 0); });
        const bool search_rejected = rejects_failed_index([&] { static_cast<void>(index->KnnSearchSorted(data.data(), 1)); });
        const bool save_rejected = rejects_failed_index([&] {
            infinity::LocalFileHandle file;
            index->Save(file);
        });
        const bool optimize_rejected = rejects_failed_index([&] { index->Optimize(); });
        const bool check_rejected = rejects_failed_index([&] { index->Check(); });
        const bool label_rejected = rejects_failed_index([&] { static_cast<void>(index->GetLabel(0)); });
        return index->IsBuildFailed() && index->GetGraphEnterPoint() == std::pair<infinity::i32, infinity::VertexType>{0, 0} && retry_rejected &&
               search_rejected && save_rejected && optimize_rejected && check_rejected && label_rejected;
    }
    return false;
}

bool CheckInFlightSearchRejectsPostPublicationFailure() {
    auto index = Hnsw::Make(2, 1, 1, 2, 2);
    const std::array<float, 2> data{1.0F, 2.0F};
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), 1, 2);
    static_cast<void>(index->StoreData(std::move(iterator)));
    index->Build(0, 0);

    AllocationPause pause;
    auto search = std::async(std::launch::async, [&] {
        pause_next_array_allocation = &pause;
        try {
            static_cast<void>(index->KnnSearchSorted(data.data(), 1));
        } catch (const std::logic_error &error) {
            return std::string_view(error.what()) == "HNSW index is unusable after a failed build";
        }
        return false;
    });
    pause.started.acquire();

    bool build_failed = false;
    fail_next_array_allocation = true;
    try {
        index->Build(1, 0);
    } catch (const std::bad_alloc &) {
        build_failed = true;
    }
    const bool published_before_poison = index->GetGraphLevel(1) == 0;
    pause.release.release();
    return build_failed && published_before_poison && index->IsBuildFailed() && search.get();
}

bool CheckSubmissionFailurePoisonsIndex() {
    auto index = Hnsw::Make(2, 1, kDimension, 2, 2);
    const auto data = MakeData(2);
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), kDimension, 2);
    StrongThrowingPool pool(2, 1);
    try {
        static_cast<void>(infinity::HnswBulkBuild(index, std::move(iterator), infinity::HnswInsertConfig{}, pool, 1));
    } catch (const std::runtime_error &) {
        bool search_rejected = false;
        try {
            static_cast<void>(index->KnnSearchSorted(data.data(), 1));
        } catch (const std::logic_error &) {
            search_rejected = true;
        }
        return index->IsBuildFailed() && index->GetVecNum() == 2 && search_rejected;
    }
    return false;
}

bool CheckInvalidLevels() {
    auto index = Hnsw::Make(1, 1, 1, 2, 2);
    bool negative_rejected = false;
    bool oversized_rejected = false;
    try {
        index->Build(0, -1);
    } catch (const std::invalid_argument &) {
        negative_rejected = true;
    }
    try {
        index->Build(0, std::numeric_limits<infinity::LayerSize>::max());
    } catch (const std::invalid_argument &) {
        oversized_rejected = true;
    }
    return negative_rejected && oversized_rejected && !index->IsBuildFailed() && index->GetVecNum() == 0 &&
           index->GetGraphEnterPoint() == std::pair<infinity::i32, infinity::VertexType>{-1, -1};
}

bool CheckInvalidVertices() {
    auto empty_index = Hnsw::Make(1, 1, 1, 2, 2);
    bool empty_rejected = false;
    bool max_rejected = false;
    try {
        empty_index->Build(0, 0);
    } catch (const std::out_of_range &) {
        empty_rejected = true;
    }
    try {
        empty_index->Build(std::numeric_limits<infinity::VertexType>::max());
    } catch (const std::out_of_range &) {
        max_rejected = true;
    }

    const std::array<float, 1> data{1.0F};
    infinity::DenseVectorIter<float, infinity::i32> iterator(data.data(), 1, 1);
    static_cast<void>(empty_index->StoreData(std::move(iterator)));
    bool negative_rejected = false;
    bool upper_rejected = false;
    try {
        empty_index->Build(-1, 0);
    } catch (const std::out_of_range &) {
        negative_rejected = true;
    }
    try {
        empty_index->Build(1, 0);
    } catch (const std::out_of_range &) {
        upper_rejected = true;
    }

    return empty_rejected && max_rejected && negative_rejected && upper_rejected && !empty_index->IsBuildFailed() &&
           empty_index->GetGraphEnterPoint() == std::pair<infinity::i32, infinity::VertexType>{-1, -1};
}

} // namespace

int main() {
    const bool empty_initial_one_worker = CheckEmpty(0, 1);
    const bool empty_initial_twelve_workers = CheckEmpty(0, 12);
    const bool empty_append_one_worker = CheckEmpty(7, 1);
    const bool empty_append_twelve_workers = CheckEmpty(7, 12);
    const bool zero_workers = CheckZeroWorkers();
    const bool bucket_boundaries = CheckBuckets();
    const bool oversized_bucket = CheckOversizedBucket();
    const bool nested_one_worker = CheckNestedOneWorker();
    const bool nested_saturated_pool = CheckNestedSaturatedPool();
    const bool nested_inline_exception = CheckNestedInlineExceptionDraining();
    const bool one_worker_failure_mask = CheckBucketFailureMask(1, 0b0001);
    const bool two_worker_failure_mask = CheckBucketFailureMask(2, 0b0101);
    const bool four_worker_failure_mask = CheckBucketFailureMask(4, 0b1111);
    const bool same_index_nested_contention = CheckSameIndexNestedContention();
    const bool partial_held_api_uses_legacy_locking = CheckPartialHeldApiUsesLegacyLocking();
    const bool task_exception = CheckTaskExceptionDraining();
    const bool submission_exception = CheckSubmissionExceptionDraining();
    const bool planned_levels_one_worker = CheckPlannedLevels(1);
    const bool planned_levels_two_workers = CheckPlannedLevels(2);
    const bool planned_levels_four_workers = CheckPlannedLevels(4);
    const bool planned_levels_six_workers = CheckPlannedLevels(6);
    const bool planned_levels_eight_workers = CheckPlannedLevels(8);
    const bool planned_levels_twelve_workers = CheckPlannedLevels(12);
    const CanonicalPlanningResult canonical = CheckCanonicalLevelPlanning();
    const ConnectivityRepairResult connectivity_repair = CheckDeterministicConnectivityRepair();
    const bool mixed_staged_append_rejection = CheckMixedStagedAppendRejection();
    const bool invalid_m_rejection = CheckInvalidMRejection();
    const bool mixed_build_levels = CheckMixedBuildLevels();
    const bool save_load_append = CheckSaveLoadAppend();
    const bool pointer_round_trip = CheckPointerRoundTrip();
    const bool packed_mapped_labels = CheckPackedMappedLabels();
    const bool pointer_image_validation = CheckPointerImageValidation();
    const bool conventional_stream_image_validation = CheckConventionalStreamImageValidation();
    const bool bulk_append_after_load = CheckBulkAppendAfterLoad();
    const bool unbuilt_serialization_rejection = CheckUnbuiltSerializationRejection();
    const bool incomplete_compression_rejection = CheckIncompleteCompressionRejection();
    const CompressionAllocationResult lvq_compression_allocations =
        CheckCompressionAllocationFailures([](Hnsw &index) { return std::move(index).CompressToLVQ(); });
    const CompressionAllocationResult rabitq_compression_allocations =
        CheckCompressionAllocationFailures([](Hnsw &index) { return std::move(index).CompressToRabitq(); });
    const bool save_waits_for_build = CheckSerializationWaitsForBuild(false);
    const bool save_to_ptr_waits_for_build = CheckSerializationWaitsForBuild(true);
    const bool duplicate_build_rejection = CheckDuplicateBuildRejection();
    const bool concurrent_bulk_serialization = CheckConcurrentBulkSerialization();
    const bool allocation_failure_pre_publication = CheckAllocationFailureBeforePublication();
    const bool allocation_failure_post_publication = CheckAllocationFailureAfterPublication();
    const bool in_flight_search_rejection = CheckInFlightSearchRejectsPostPublicationFailure();
    const bool submission_failure_poison = CheckSubmissionFailurePoisonsIndex();
    const bool invalid_levels = CheckInvalidLevels();
    const bool invalid_vertices = CheckInvalidVertices();

    const bool passed =
        empty_initial_one_worker && empty_initial_twelve_workers && empty_append_one_worker && empty_append_twelve_workers && zero_workers &&
        bucket_boundaries && oversized_bucket && nested_one_worker && nested_saturated_pool && nested_inline_exception && one_worker_failure_mask &&
        two_worker_failure_mask && four_worker_failure_mask && same_index_nested_contention && partial_held_api_uses_legacy_locking &&
        task_exception && submission_exception &&
        planned_levels_one_worker && planned_levels_two_workers && planned_levels_four_workers && planned_levels_six_workers &&
        planned_levels_eight_workers && planned_levels_twelve_workers && canonical.passed && connectivity_repair.passed &&
        mixed_staged_append_rejection && invalid_m_rejection && mixed_build_levels && save_load_append && pointer_round_trip &&
        packed_mapped_labels && pointer_image_validation && conventional_stream_image_validation && bulk_append_after_load &&
        unbuilt_serialization_rejection && save_waits_for_build &&
        incomplete_compression_rejection && lvq_compression_allocations.passed && rabitq_compression_allocations.passed &&
        save_to_ptr_waits_for_build &&
        duplicate_build_rejection && concurrent_bulk_serialization && allocation_failure_pre_publication &&
        allocation_failure_post_publication && in_flight_search_rejection && submission_failure_poison && invalid_levels &&
        invalid_vertices;

    std::cout << "status=" << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "empty_initial_one_worker=" << empty_initial_one_worker << '\n';
    std::cout << "empty_initial_twelve_workers=" << empty_initial_twelve_workers << '\n';
    std::cout << "empty_append_one_worker=" << empty_append_one_worker << '\n';
    std::cout << "empty_append_twelve_workers=" << empty_append_twelve_workers << '\n';
    std::cout << "zero_workers_pre_store_rejection=" << zero_workers << '\n';
    std::cout << "bucket_boundaries=" << bucket_boundaries << '\n';
    std::cout << "oversized_bucket=" << oversized_bucket << '\n';
    std::cout << "nested_one_worker_inline=" << nested_one_worker << '\n';
    std::cout << "nested_saturated_pool_inline=" << nested_saturated_pool << '\n';
    std::cout << "nested_inline_exception_draining=" << nested_inline_exception << '\n';
    std::cout << "one_worker_bucket_failure_mask=" << one_worker_failure_mask << '\n';
    std::cout << "two_worker_bucket_failure_mask=" << two_worker_failure_mask << '\n';
    std::cout << "four_worker_bucket_failure_mask=" << four_worker_failure_mask << '\n';
    std::cout << "same_index_nested_contention_rejected=" << same_index_nested_contention << '\n';
    std::cout << "partial_held_api_uses_legacy_locking=" << partial_held_api_uses_legacy_locking << '\n';
    std::cout << "task_exception_draining=" << task_exception << '\n';
    std::cout << "submission_exception_draining=" << submission_exception << '\n';
    std::cout << "planned_levels_one_worker=" << planned_levels_one_worker << '\n';
    std::cout << "planned_levels_two_workers=" << planned_levels_two_workers << '\n';
    std::cout << "planned_levels_four_workers=" << planned_levels_four_workers << '\n';
    std::cout << "planned_levels_six_workers=" << planned_levels_six_workers << '\n';
    std::cout << "planned_levels_eight_workers=" << planned_levels_eight_workers << '\n';
    std::cout << "planned_levels_twelve_workers=" << planned_levels_twelve_workers << '\n';
    std::cout << "canonical_level_planning=" << canonical.passed << '\n';
    std::cout << "canonical_level_hash=" << canonical.hash << '\n';
    std::cout << "deterministic_connectivity_repair=" << connectivity_repair.passed << '\n';
    std::cout << "deterministic_reachable_before=" << connectivity_repair.reachable_before << '\n';
    std::cout << "deterministic_repair_count=" << connectivity_repair.repair_count << '\n';
    std::cout << "deterministic_reachable_after=" << connectivity_repair.reachable_after << '\n';
    std::cout << "deterministic_bulk_reachable=" << connectivity_repair.bulk_reachable << '\n';
    std::cout << "deterministic_appended_sources=" << connectivity_repair.appended_sources << '\n';
    std::cout << "deterministic_replacement_sources=" << connectivity_repair.replacement_sources << '\n';
    std::cout << "mixed_staged_append_rejection=" << mixed_staged_append_rejection << '\n';
    std::cout << "invalid_m_rejection=" << invalid_m_rejection << '\n';
    std::cout << "mixed_sequential_bulk_levels=" << mixed_build_levels << '\n';
    std::cout << "save_load_append_graph_equivalence=" << save_load_append << '\n';
    std::cout << "pointer_round_trip_graph_equivalence=" << pointer_round_trip << '\n';
    std::cout << "packed_mapped_labels=" << packed_mapped_labels << '\n';
    std::cout << "pointer_image_validation=" << pointer_image_validation << '\n';
    std::cout << "conventional_stream_image_validation=" << conventional_stream_image_validation << '\n';
    std::cout << "bulk_append_after_load_graph_equivalence=" << bulk_append_after_load << '\n';
    std::cout << "unbuilt_serialization_rejection=" << unbuilt_serialization_rejection << '\n';
    std::cout << "incomplete_compression_rejection=" << incomplete_compression_rejection << '\n';
    std::cout << "lvq_compression_allocation_rollback=" << lvq_compression_allocations.passed << '\n';
    std::cout << "lvq_compression_failed_allocation_points=" << lvq_compression_allocations.failed_allocation_count << '\n';
    std::cout << "lvq_compression_topology_and_built_count=" << lvq_compression_allocations.success_preserved << '\n';
    std::cout << "rabitq_compression_allocation_rollback=" << rabitq_compression_allocations.passed << '\n';
    std::cout << "rabitq_compression_failed_allocation_points=" << rabitq_compression_allocations.failed_allocation_count << '\n';
    std::cout << "rabitq_compression_topology_and_built_count=" << rabitq_compression_allocations.success_preserved << '\n';
    std::cout << "save_waits_for_standalone_build=" << save_waits_for_build << '\n';
    std::cout << "save_to_ptr_waits_for_standalone_build=" << save_to_ptr_waits_for_build << '\n';
    std::cout << "duplicate_build_rejection=" << duplicate_build_rejection << '\n';
    std::cout << "concurrent_bulk_serialization=" << concurrent_bulk_serialization << '\n';
    std::cout << "allocation_failure_pre_publication=" << allocation_failure_pre_publication << '\n';
    std::cout << "allocation_failure_post_publication_poison=" << allocation_failure_post_publication << '\n';
    std::cout << "in_flight_search_post_publication_rejection=" << in_flight_search_rejection << '\n';
    std::cout << "submission_failure_poison=" << submission_failure_poison << '\n';
    std::cout << "invalid_level_pre_mutation_rejection=" << invalid_levels << '\n';
    std::cout << "invalid_vertex_pre_mutation_rejection=" << invalid_vertices << '\n';

    return passed ? 0 : 1;
}
