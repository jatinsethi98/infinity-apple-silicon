export module infinity_core:hnsw_bulk_build;

import std;

namespace infinity {

export struct HnswBulkBuildResult {
    std::size_t mem_usage_{};
    std::size_t start_{};
    std::size_t end_{};
    std::size_t submitted_task_count_{};
};

// Default number of build tasks handed to each pool worker.
//
// Inserting a vertex costs more as the graph grows, so with one contiguous bucket
// per worker the worker holding the newest vertices does the most work. A profile
// of a 100k x 128 build on 12 workers measured per-worker active samples of
// 3386 3354 3385 3726 4353 4546 4797 5065 5208 5479 5843 6222 -- monotonically
// increasing, 1.84x from first to last. That looked like a scheduling problem
// worth fixing. Measurement says otherwise; see below.
//
// Cutting the range into several tasks per worker lets the pool's shared queue
// hand out the expensive tail across all workers, since each worker pulls another
// task as soon as it finishes one.
//
// MEASURED, and the reason the default is 1: this recovers almost nothing. On
// SIFT1M (1M x 128, M=32, efC=200, 12 threads, AC power, 6 paired runs per arm)
// the median build moved 49.879 s -> 49.424 s, a 1.009x gain. The 74.15%
// "occupancy" in the profile was sampled active-stack share, not CPU utilization:
// a worker that finishes early stops accruing samples, so the 3386..6222 ramp
// measures work done, not time spent idle. There was no 26% of wasted time to
// reclaim. A thread sweep confirms the build is compute-bound and already scales
// well -- 3 -> 6 threads is 1.816x (91% of linear) and 6 -> 12 is 1.319x against a
// 1.54x ceiling for adding 6 efficiency cores to 6 performance cores.
//
// What it does change is graph quality, by altering how insertions interleave
// across workers. At 8 buckets/worker, recall@10 shifts materially and in both
// directions (6 paired runs, medians): ef32 0.7633 -> 0.7352, ef64 0.8070 ->
// 0.8008, ef128 0.8344 -> 0.8617, ef256 0.8922 -> 0.9422, ef512 0.9461 -> 0.9828.
// Better above ef128, worse below it. That is a real accuracy trade, not noise,
// and a 1% build gain does not justify making it silently -- so the shipping
// default preserves the previous one-bucket-per-worker behaviour and the trade is
// left to a deliberate, properly powered recall study. Raise this (or pass
// buckets_per_worker explicitly) to opt in.
export constexpr std::size_t kHnswBuildBucketsPerWorker = 1;

// Vertices per build task. Shared by the builder and by callers that need to
// predict the task count, so the two can never disagree.
export constexpr std::size_t HnswBuildBucketSize(std::size_t stored_count,
                                                 std::size_t worker_count,
                                                 std::size_t minimum_bucket_size,
                                                 std::size_t buckets_per_worker = kHnswBuildBucketsPerWorker) {
    if (stored_count == 0 || worker_count == 0) {
        return std::max<std::size_t>(minimum_bucket_size, 1);
    }
    // Cap the requested granularity so worker_count * buckets_per_worker cannot
    // overflow, and so a caller cannot ask for more tasks than there are vertices.
    const std::size_t requested = std::max<std::size_t>(buckets_per_worker, 1);
    const std::size_t affordable = std::max<std::size_t>(stored_count / worker_count, 1);
    const std::size_t target_bucket_count = worker_count * std::min(requested, affordable);
    const std::size_t balanced_bucket_size = (stored_count - 1) / target_bucket_count + 1;
    return std::max<std::size_t>({minimum_bucket_size, balanced_bucket_size, 1});
}

template <typename ThreadPool, typename BuildVertex, typename OnSubmissionFailure>
std::size_t HnswRunBuildTasks(ThreadPool &thread_pool,
                              std::size_t start,
                              std::size_t end,
                              std::size_t bucket_size,
                              BuildVertex build_vertex,
                              OnSubmissionFailure on_submission_failure) {
    static_assert(ThreadPool::push_has_strong_exception_guarantee, "HNSW bulk build requires push() to either return a future or retain no callable");
    static_assert(noexcept(on_submission_failure()), "HNSW submission-failure callback must not throw");
    static_assert(
        requires(const ThreadPool &pool) {
            { pool.owns_current_thread() } noexcept -> std::same_as<bool>;
        },
        "HNSW bulk build requires worker-thread ownership detection");
    const std::size_t stored_count = end - start;
    const std::size_t bucket_count = (stored_count - 1) / bucket_size + 1;

    if (thread_pool.owns_current_thread()) {
        std::exception_ptr first_task_error;
        for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
            const std::size_t begin = start + bucket * bucket_size;
            const std::size_t finish = begin + std::min(bucket_size, end - begin);
            try {
                for (std::size_t vertex = begin; vertex < finish; ++vertex) {
                    build_vertex(vertex);
                }
            } catch (...) {
                if (!first_task_error) {
                    first_task_error = std::current_exception();
                }
            }
        }
        if (first_task_error) {
            std::rethrow_exception(first_task_error);
        }
        return 0;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(bucket_count);

    try {
        for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
            const std::size_t begin = start + bucket * bucket_size;
            const std::size_t finish = begin + std::min(bucket_size, end - begin);
            futures.emplace_back(thread_pool.push([build_vertex, begin, finish](int) {
                for (std::size_t vertex = begin; vertex < finish; ++vertex) {
                    build_vertex(vertex);
                }
            }));
        }
    } catch (...) {
        const std::exception_ptr submission_error = std::current_exception();
        on_submission_failure();
        for (auto &future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
        std::rethrow_exception(submission_error);
    }

    std::exception_ptr first_task_error;
    for (auto &future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!first_task_error) {
                first_task_error = std::current_exception();
            }
        }
    }
    if (first_task_error) {
        std::rethrow_exception(first_task_error);
    }
    return bucket_count;
}

struct HnswHeldLockOperations {
    template <typename IndexPtr, typename Iter, typename Config>
    static auto Store(IndexPtr &index, Iter &&iter, const Config &config) {
        return index->StoreDataWithOperationLockHeld(std::forward<Iter>(iter), config);
    }

    template <typename Index>
    static void Build(Index *index, std::size_t vertex) {
        index->BuildWithOperationLockHeld(vertex);
    }

    template <typename Index, typename Level>
    static void Build(Index *index, std::size_t vertex, Level level) {
        index->BuildWithOperationLockHeld(vertex, level);
    }

    template <typename IndexPtr>
    static void Finalize(IndexPtr &index) {
        static_cast<void>(index->FinalizeBuildWithOperationLockHeld());
    }
};

struct HnswPublicOperations {
    template <typename IndexPtr, typename Iter, typename Config>
    static auto Store(IndexPtr &index, Iter &&iter, const Config &config) {
        return index->StoreData(std::forward<Iter>(iter), config);
    }

    template <typename Index>
    static void Build(Index *index, std::size_t vertex) {
        index->Build(vertex);
    }

    template <typename Index, typename Level>
    static void Build(Index *index, std::size_t vertex, Level level) {
        index->Build(vertex, level);
    }

    template <typename IndexPtr>
    static void Finalize(IndexPtr &index) {
        if constexpr (requires { index->FinalizeBuild(); }) {
            static_cast<void>(index->FinalizeBuild());
        }
    }
};

template <typename Operations, typename IndexPtr, typename Iter, typename Config, typename ThreadPool>
HnswBulkBuildResult HnswBulkBuildRun(IndexPtr &index,
                                     Iter &&iter,
                                     const Config &config,
                                     ThreadPool &thread_pool,
                                     std::size_t minimum_bucket_size,
                                     std::size_t buckets_per_worker) {
    try {
        const std::size_t mem_before = index->mem_usage();
        const auto [start, end] = Operations::Store(index, std::forward<Iter>(iter), config);
        if (end < start) {
            throw std::logic_error("HNSW bulk build received an invalid stored range");
        }
        const std::size_t stored_count = std::size_t(end - start);
        if (stored_count == 0) {
            return HnswBulkBuildResult{
                .mem_usage_ = index->mem_usage() - mem_before,
                .start_ = std::size_t(start),
                .end_ = std::size_t(end),
                .submitted_task_count_ = 0,
            };
        }

        const std::size_t bucket_size =
            HnswBuildBucketSize(stored_count, std::size_t(thread_pool.size()), minimum_bucket_size, buckets_per_worker);
        auto *const index_ptr = index.get();
        const auto mark_submission_failure = [index_ptr]() noexcept {
            static_cast<void>(index_ptr);
            if constexpr (requires { index_ptr->MarkBuildFailed(); }) {
                index_ptr->MarkBuildFailed();
            }
        };
        std::size_t submitted_task_count;
        if constexpr (requires { index->PrepareBuildLevels(start, end); }) {
            auto level_plan = index->PrepareBuildLevels(start, end);
            const auto levels = std::make_shared<const std::decay_t<decltype(level_plan)>>(std::move(level_plan));
            if (levels->size() != stored_count) {
                throw std::logic_error("HNSW bulk level plan has the wrong size");
            }
            submitted_task_count = HnswRunBuildTasks(
                thread_pool,
                std::size_t(start),
                std::size_t(end),
                bucket_size,
                [index_ptr, levels, start](std::size_t vertex) {
                    Operations::Build(index_ptr, vertex, (*levels)[vertex - std::size_t(start)]);
                },
                mark_submission_failure);
        } else {
            submitted_task_count = HnswRunBuildTasks(
                thread_pool,
                std::size_t(start),
                std::size_t(end),
                bucket_size,
                [index_ptr](std::size_t vertex) { Operations::Build(index_ptr, vertex); },
                mark_submission_failure);
        }
        Operations::Finalize(index);
        return HnswBulkBuildResult{
            .mem_usage_ = index->mem_usage() - mem_before,
            .start_ = std::size_t(start),
            .end_ = std::size_t(end),
            .submitted_task_count_ = submitted_task_count,
        };
    } catch (...) {
        if constexpr (requires { index->MarkBuildFailed(); }) {
            index->MarkBuildFailed();
        }
        throw;
    }
}

export template <typename IndexPtr, typename Iter, typename Config, typename ThreadPool>
HnswBulkBuildResult
HnswBulkBuild(IndexPtr &index,
              Iter &&iter,
              const Config &config,
              ThreadPool &thread_pool,
              std::size_t minimum_bucket_size = 1024,
              std::size_t buckets_per_worker = kHnswBuildBucketsPerWorker) {
    if (thread_pool.size() <= 0) {
        throw std::invalid_argument("HNSW bulk build requires a non-empty thread pool");
    }

    constexpr bool use_held_lock_api = requires {
        index->AcquireExclusiveOperationLock();
        index->TryAcquireExclusiveOperationLock();
        index->EnsureAllVerticesBuiltWithOperationLockHeld();
        index->StoreDataWithOperationLockHeld(std::forward<Iter>(iter), config);
        index->BuildWithOperationLockHeld(std::size_t{});
        index->FinalizeBuildWithOperationLockHeld();
    };

    if constexpr (use_held_lock_api) {
        auto operation_lock =
            thread_pool.owns_current_thread() ? index->TryAcquireExclusiveOperationLock() : index->AcquireExclusiveOperationLock();
        if (!operation_lock.owns_lock()) {
            throw std::logic_error("HNSW bulk build rejected same-index contention from a pool worker");
        }
        index->EnsureAllVerticesBuiltWithOperationLockHeld();
        return HnswBulkBuildRun<HnswHeldLockOperations>(
            index, std::forward<Iter>(iter), config, thread_pool, minimum_bucket_size, buckets_per_worker);
    } else {
        return HnswBulkBuildRun<HnswPublicOperations>(
            index, std::forward<Iter>(iter), config, thread_pool, minimum_bucket_size, buckets_per_worker);
    }
}

} // namespace infinity
