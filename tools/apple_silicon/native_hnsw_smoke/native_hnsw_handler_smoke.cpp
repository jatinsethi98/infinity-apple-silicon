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

} // namespace

void *operator new[](std::size_t size) {
    if (auto *pause = std::exchange(pause_next_array_allocation, nullptr)) {
        pause->started.release();
        pause->release.acquire();
    }
    if (std::exchange(fail_next_array_allocation, false)) {
        throw std::bad_alloc();
    }
    if (void *ptr = std::malloc(size == 0 ? 1 : size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete[](void *ptr) noexcept { std::free(ptr); }

void operator delete[](void *ptr, std::size_t) noexcept { std::free(ptr); }

namespace {

constexpr std::size_t kVectorCount = 12'288;
constexpr std::size_t kDimension = 128;
constexpr std::size_t kChunkSize = 8'192;
constexpr std::size_t kM = 32;
constexpr std::size_t kEfConstruction = 200;
constexpr std::size_t kEfSearch = 32;
constexpr std::size_t kExpectedWorkerCount = 12;
constexpr std::size_t kBuildBucketSize = 1'024;
constexpr std::size_t kQueryCount = 1'000;
constexpr std::size_t kTransactionVectorCount = 256;
constexpr double kMinimumSelfRecall = 0.95;
constexpr auto kBlockedObservation = std::chrono::milliseconds(100);
constexpr auto kCompletionTimeout = std::chrono::seconds(5);
constexpr std::string_view kWorkerResizeRejection = "HNSW build thread pool worker cannot resize its own pool";

struct PoolLifecycleResult {
    bool worker_resize_completed{};
    bool worker_resize_rejected{};
    bool worker_resize_preserved_count{};
    bool task_started{};
    bool resize_started{};
    bool resize_blocked{};
    bool task_completed{};
    bool resize_completed{};
    std::size_t initial_worker_count{};
    std::size_t final_worker_count{};
};

struct CompressionLockResult {
    bool allocation_paused{};
    bool search_started{};
    bool search_blocked{};
    bool compression_completed{};
    bool search_completed{};
    bool search_valid{};
};

PoolLifecycleResult CheckPoolLifecycle() {
    auto &context = infinity::InfinityContext::instance();
    PoolLifecycleResult result;
    std::future<bool> worker_resize_future;
    {
        auto thread_pool_lease = context.AcquireHnswBuildThreadPool();
        auto &thread_pool = thread_pool_lease.Get();
        result.initial_worker_count = static_cast<std::size_t>(thread_pool.size());
        const int requested_worker_count =
            result.initial_worker_count > 1 ? static_cast<int>(result.initial_worker_count - 1) : 2;
        worker_resize_future = thread_pool.push([&context, requested_worker_count](int) {
            try {
                context.ResizeHnswBuildThreadPool(requested_worker_count);
            } catch (const std::logic_error &error) {
                if (std::string_view(error.what()) == kWorkerResizeRejection) {
                    return true;
                }
                std::cerr << "worker-initiated HNSW pool resize returned the wrong rejection: " << error.what() << '\n';
                return false;
            } catch (const std::exception &error) {
                std::cerr << "worker-initiated HNSW pool resize threw an unexpected exception: " << error.what() << '\n';
                return false;
            } catch (...) {
                std::cerr << "worker-initiated HNSW pool resize threw an unexpected non-standard exception\n";
                return false;
            }
            std::cerr << "worker-initiated HNSW pool resize was not rejected\n";
            return false;
        });
        result.worker_resize_completed = worker_resize_future.wait_for(kCompletionTimeout) == std::future_status::ready;
        if (result.worker_resize_completed) {
            result.worker_resize_rejected = worker_resize_future.get();
        } else {
            std::cerr << "worker-initiated HNSW pool resize did not reject before the shared lease timeout\n";
        }
        result.worker_resize_preserved_count =
            static_cast<std::size_t>(thread_pool.size()) == result.initial_worker_count;
    }
    if (!result.worker_resize_completed && worker_resize_future.wait_for(kCompletionTimeout) == std::future_status::ready) {
        static_cast<void>(worker_resize_future.get());
    }

    std::promise<void> task_started_promise;
    auto task_started_future = task_started_promise.get_future();
    std::promise<void> release_task_promise;
    auto release_task_future = release_task_promise.get_future().share();
    std::promise<void> resize_started_promise;
    auto resize_started_future = resize_started_promise.get_future();
    std::future<void> resize_future;

    {
        auto thread_pool_lease = context.AcquireHnswBuildThreadPool();
        auto &thread_pool = thread_pool_lease.Get();
        auto task_future = thread_pool.push([&task_started_promise, release_task_future](int) {
            task_started_promise.set_value();
            release_task_future.wait();
        });

        result.task_started = task_started_future.wait_for(kCompletionTimeout) == std::future_status::ready;
        resize_future = std::async(std::launch::async, [&context, &resize_started_promise] {
            resize_started_promise.set_value();
            context.ResizeHnswBuildThreadPool(static_cast<int>(kExpectedWorkerCount));
        });
        result.resize_started = resize_started_future.wait_for(kCompletionTimeout) == std::future_status::ready;
        result.resize_blocked = result.resize_started && resize_future.wait_for(kBlockedObservation) == std::future_status::timeout;

        release_task_promise.set_value();
        result.task_completed = task_future.wait_for(kCompletionTimeout) == std::future_status::ready;
        if (result.task_completed) {
            task_future.get();
        }
    }

    result.resize_completed = resize_future.wait_for(kCompletionTimeout) == std::future_status::ready;
    if (result.resize_completed) {
        resize_future.get();
    }
    auto final_lease = context.AcquireHnswBuildThreadPool();
    result.final_worker_count = static_cast<std::size_t>(final_lease.Get().size());
    return result;
}

std::unique_ptr<infinity::HnswHandler> MakePopulatedHandler(const infinity::IndexHnsw &index_definition,
                                                            const std::shared_ptr<infinity::ColumnDef> &column_definition,
                                                            const std::vector<float> &data) {
    auto handler = infinity::HnswHandler::Make(&index_definition, column_definition);
    infinity::DenseVectorIter<float, infinity::SegmentOffset> iterator(data.data(), kDimension, kTransactionVectorCount);
    static_cast<void>(handler->InsertVecs(std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, 64));
    return handler;
}

bool QueryReturnsExpectedLabel(const infinity::HnswHandler &handler, const std::vector<float> &data, std::size_t query_index) {
    const infinity::KnnSearchOption search_option{.ef_ = kEfSearch};
    auto [result_count, distances, labels] =
        handler.SearchIndex<float, infinity::SegmentOffset>(data.data() + query_index * kDimension, 1, search_option);
    if (result_count == 0) {
        return false;
    }
    std::size_t nearest = 0;
    for (std::size_t candidate = 1; candidate < result_count; ++candidate) {
        if (distances[candidate] < distances[nearest]) {
            nearest = candidate;
        }
    }
    return labels[nearest] == static_cast<infinity::SegmentOffset>(query_index) && std::isfinite(distances[nearest]);
}

template <typename Compress>
bool CheckCompressionAllocationRollback(const infinity::IndexHnsw &index_definition,
                                        const std::shared_ptr<infinity::ColumnDef> &column_definition,
                                        const std::vector<float> &data,
                                        Compress compress) {
    auto handler = MakePopulatedHandler(index_definition, column_definition, data);
    const std::size_t memory_before = handler->MemUsage();
    const std::size_t size_before = handler->GetSizeInBytes();

    fail_next_array_allocation = true;
    bool rejected = false;
    try {
        compress(*handler);
    } catch (const std::bad_alloc &) {
        rejected = true;
    } catch (...) {
        fail_next_array_allocation = false;
        return false;
    }
    fail_next_array_allocation = false;

    try {
        handler->Check();
        return rejected && !handler->IsBuildFailed() && handler->GetRowCount() == kTransactionVectorCount &&
               handler->MemUsage() == memory_before && handler->GetSizeInBytes() == size_before &&
               QueryReturnsExpectedLabel(*handler, data, 7);
    } catch (...) {
        return false;
    }
}

CompressionLockResult CheckCompressionExcludesSearch(const infinity::IndexHnsw &index_definition,
                                                     const std::shared_ptr<infinity::ColumnDef> &column_definition,
                                                     const std::vector<float> &data) {
    auto handler = MakePopulatedHandler(index_definition, column_definition, data);
    AllocationPause pause;
    auto compression = std::async(std::launch::async, [&] {
        pause_next_array_allocation = &pause;
        handler->CompressToLVQ();
    });

    CompressionLockResult result;
    result.allocation_paused = pause.started.try_acquire_for(kCompletionTimeout);
    if (!result.allocation_paused) {
        pause.release.release();
        if (compression.wait_for(kCompletionTimeout) == std::future_status::ready) {
            compression.get();
        }
        return result;
    }

    std::promise<void> search_started_promise;
    auto search_started = search_started_promise.get_future();
    auto search = std::async(std::launch::async, [&] {
        search_started_promise.set_value();
        return QueryReturnsExpectedLabel(*handler, data, 11);
    });
    result.search_started = search_started.wait_for(kCompletionTimeout) == std::future_status::ready;
    result.search_blocked = result.search_started && search.wait_for(kBlockedObservation) == std::future_status::timeout;

    pause.release.release();
    result.compression_completed = compression.wait_for(kCompletionTimeout) == std::future_status::ready;
    if (result.compression_completed) {
        try {
            compression.get();
        } catch (...) {
            result.compression_completed = false;
        }
    }
    result.search_completed = search.wait_for(kCompletionTimeout) == std::future_status::ready;
    if (result.search_completed) {
        result.search_valid = search.get();
    }
    handler->Check();
    return result;
}

} // namespace

int main() {
#if !defined(__APPLE__) || !defined(__aarch64__)
    std::cerr << "native HnswHandler smoke requires arm64 macOS\n";
    return 64;
#endif

    const PoolLifecycleResult lifecycle = CheckPoolLifecycle();

    std::vector<float> data(kVectorCount * kDimension);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> distribution;
    for (float &value : data) {
        value = distribution(rng);
    }

    const infinity::IndexHnsw index_definition(infinity::MetricType::kMetricL2,
                                               infinity::HnswEncodeType::kPlain,
                                               infinity::HnswBuildType::kPlain,
                                               kM,
                                               kEfConstruction,
                                               kChunkSize);
    auto embedding_info = std::make_shared<infinity::EmbeddingInfo>(infinity::EmbeddingDataType::kElemFloat, kDimension);
    auto data_type = std::make_shared<infinity::DataType>(embedding_info);
    auto column_definition = std::make_shared<infinity::ColumnDef>(data_type);

    const bool lvq_allocation_rollback =
        CheckCompressionAllocationRollback(index_definition, column_definition, data, [](infinity::HnswHandler &candidate) {
            candidate.CompressToLVQ();
        });
    const bool rabitq_allocation_rollback =
        CheckCompressionAllocationRollback(index_definition, column_definition, data, [](infinity::HnswHandler &candidate) {
            candidate.CompressToRabitq();
        });
    const CompressionLockResult compression_lock = CheckCompressionExcludesSearch(index_definition, column_definition, data);

    const auto cold_begin = std::chrono::steady_clock::now();
    auto handler = infinity::HnswHandler::Make(&index_definition, column_definition);
    const auto factory_end = std::chrono::steady_clock::now();

    infinity::DenseVectorIter<float, infinity::SegmentOffset> iterator(data.data(), kDimension, kVectorCount);
    const auto insert_begin = std::chrono::steady_clock::now();
    const std::size_t memory_delta = handler->InsertVecs(std::move(iterator), infinity::HnswInsertConfig{.optimize_ = true}, kBuildBucketSize);
    const auto insert_end = std::chrono::steady_clock::now();

    handler->Check();

    const infinity::KnnSearchOption search_option{.ef_ = kEfSearch};
    std::size_t correct = 0;
    std::size_t returned_candidate_count = 0;
    double distance_checksum = 0;
    for (std::size_t query_index = 0; query_index < kQueryCount; ++query_index) {
        const float *query = data.data() + query_index * kDimension;
        auto [result_count, distances, labels] = handler->SearchIndex<float, infinity::SegmentOffset>(query, 1, search_option);
        returned_candidate_count = result_count;
        if (result_count > 0) {
            std::size_t nearest = 0;
            for (std::size_t candidate = 1; candidate < result_count; ++candidate) {
                if (distances[candidate] < distances[nearest]) {
                    nearest = candidate;
                }
            }
            correct += labels[nearest] == static_cast<infinity::SegmentOffset>(query_index);
            distance_checksum += distances[nearest];
        }
    }

    auto thread_pool_lease = infinity::InfinityContext::instance().AcquireHnswBuildThreadPool();
    const std::size_t worker_count = static_cast<std::size_t>(thread_pool_lease.Get().size());
    const double recall = static_cast<double>(correct) / static_cast<double>(kQueryCount);
    const auto factory_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(factory_end - cold_begin).count();
    const auto insert_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_begin).count();
    const auto cold_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - cold_begin).count();
    const bool lifecycle_passed = lifecycle.worker_resize_completed && lifecycle.worker_resize_rejected &&
                                  lifecycle.worker_resize_preserved_count && lifecycle.task_started && lifecycle.resize_started &&
                                  lifecycle.resize_blocked && lifecycle.task_completed && lifecycle.resize_completed &&
                                  lifecycle.initial_worker_count == 2 &&
                                  lifecycle.final_worker_count == kExpectedWorkerCount;
    const bool compression_lock_passed =
        compression_lock.allocation_paused && compression_lock.search_started && compression_lock.search_blocked &&
        compression_lock.compression_completed && compression_lock.search_completed && compression_lock.search_valid;
    const bool passed = lifecycle_passed && worker_count == kExpectedWorkerCount && memory_delta > 0 && handler->MemUsage() > 0 &&
                        handler->GetRowCount() == kVectorCount && recall >= kMinimumSelfRecall && std::isfinite(distance_checksum) &&
                        lvq_allocation_rollback && rabitq_allocation_rollback && compression_lock_passed;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "status=" << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "scope=production-hnsw-handler-public-path\n";
    std::cout << "architecture=arm64-apple\n";
    std::cout << "entrypoint=HnswHandler::Make->InsertVecs->Check->SearchIndex\n";
    std::cout << "vectors=" << kVectorCount << '\n';
    std::cout << "dimensions=" << kDimension << '\n';
    std::cout << "M=" << kM << '\n';
    std::cout << "ef_construction=" << kEfConstruction << '\n';
    std::cout << "pool_lifecycle=" << (lifecycle_passed ? "PASS" : "FAIL") << '\n';
    std::cout << "pool_lifecycle_initial_workers=" << lifecycle.initial_worker_count << '\n';
    std::cout << "pool_worker_resize_completed=" << lifecycle.worker_resize_completed << '\n';
    std::cout << "pool_worker_resize_rejected=" << lifecycle.worker_resize_rejected << '\n';
    std::cout << "pool_worker_resize_preserved_count=" << lifecycle.worker_resize_preserved_count << '\n';
    std::cout << "pool_lifecycle_resize_blocked=" << lifecycle.resize_blocked << '\n';
    std::cout << "pool_lifecycle_resize_completed=" << lifecycle.resize_completed << '\n';
    std::cout << "pool_lifecycle_final_workers=" << lifecycle.final_worker_count << '\n';
    std::cout << "handler_lvq_allocation_rollback=" << lvq_allocation_rollback << '\n';
    std::cout << "handler_rabitq_allocation_rollback=" << rabitq_allocation_rollback << '\n';
    std::cout << "handler_compression_allocation_paused=" << compression_lock.allocation_paused << '\n';
    std::cout << "handler_compression_search_started=" << compression_lock.search_started << '\n';
    std::cout << "handler_compression_search_blocked=" << compression_lock.search_blocked << '\n';
    std::cout << "handler_compression_completed=" << compression_lock.compression_completed << '\n';
    std::cout << "handler_compression_search_completed=" << compression_lock.search_completed << '\n';
    std::cout << "handler_compression_search_valid=" << compression_lock.search_valid << '\n';
    std::cout << "hnsw_build_workers=" << worker_count << '\n';
    std::cout << "search_candidates=" << returned_candidate_count << '\n';
    std::cout << "index_size=" << handler->GetRowCount() << '\n';
    std::cout << "memory_delta_bytes=" << memory_delta << '\n';
    std::cout << "memory_total_bytes=" << handler->MemUsage() << '\n';
    std::cout << "factory_nanoseconds=" << factory_ns << '\n';
    std::cout << "insert_nanoseconds=" << insert_ns << '\n';
    std::cout << "cold_factory_and_insert_nanoseconds=" << cold_ns << '\n';
    std::cout << "self_recall_at_1=" << recall << '\n';
    std::cout << "distance_checksum=" << distance_checksum << '\n';

    return passed ? 0 : 1;
}
