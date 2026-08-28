export module infinity_core:hnsw_bulk_build;

import std;

namespace infinity {

export struct HnswBulkBuildResult {
    std::size_t mem_usage_{};
    std::size_t start_{};
    std::size_t end_{};
    std::size_t submitted_task_count_{};
};

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
                                     std::size_t minimum_bucket_size) {
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

        const std::size_t bucket_size = std::max(minimum_bucket_size, (stored_count - 1) / std::size_t(thread_pool.size()) + 1);
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
HnswBulkBuild(IndexPtr &index, Iter &&iter, const Config &config, ThreadPool &thread_pool, std::size_t minimum_bucket_size = 1024) {
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
            index, std::forward<Iter>(iter), config, thread_pool, minimum_bucket_size);
    } else {
        return HnswBulkBuildRun<HnswPublicOperations>(index, std::forward<Iter>(iter), config, thread_pool, minimum_bucket_size);
    }
}

} // namespace infinity
