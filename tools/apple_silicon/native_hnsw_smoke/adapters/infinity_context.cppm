export module infinity_core:infinity_context;

import :config;
import third_party;

export namespace infinity {

class InfinityContext {
public:
    class HnswBuildThreadPoolLease {
    public:
        HnswBuildThreadPoolLease(const HnswBuildThreadPoolLease &) = delete;
        HnswBuildThreadPoolLease &operator=(const HnswBuildThreadPoolLease &) = delete;
        HnswBuildThreadPoolLease(HnswBuildThreadPoolLease &&) noexcept = default;
        HnswBuildThreadPoolLease &operator=(HnswBuildThreadPoolLease &&) = delete;

        [[nodiscard]] ctpl::thread_pool &Get() & noexcept { return *thread_pool_; }
        ctpl::thread_pool &Get() && = delete;

    private:
        friend class InfinityContext;

        HnswBuildThreadPoolLease(std::shared_mutex &lifecycle_mutex, ctpl::thread_pool &thread_pool)
            : lifecycle_lock_(lifecycle_mutex), thread_pool_(&thread_pool) {}

        std::shared_lock<std::shared_mutex> lifecycle_lock_;
        ctpl::thread_pool *thread_pool_;
    };

    static InfinityContext &instance() {
        static InfinityContext context;
        return context;
    }

    [[nodiscard]] HnswBuildThreadPoolLease AcquireHnswBuildThreadPool() {
        return HnswBuildThreadPoolLease(hnsw_build_thread_pool_lifecycle_mutex_, hnsw_build_thread_pool_);
    }

    void ResizeHnswBuildThreadPool(int worker_count) {
        if (hnsw_build_thread_pool_.owns_current_thread()) {
            throw std::logic_error("HNSW build thread pool worker cannot resize its own pool");
        }
        std::unique_lock lifecycle_lock(hnsw_build_thread_pool_lifecycle_mutex_);
        hnsw_build_thread_pool_.resize(worker_count);
    }

    Config *config() { return &config_; }

private:
    InfinityContext() : hnsw_build_thread_pool_(2) {}

    Config config_;
    std::shared_mutex hnsw_build_thread_pool_lifecycle_mutex_;
    ctpl::thread_pool hnsw_build_thread_pool_;
};

} // namespace infinity
