#include <cstdlib>
#include <new>

void CtplTestHookAfterEmptyPop();
void CtplTestHookAfterCancelQueueExtract();
void CtplTestHookAfterGrowthGateWait(int worker_id, bool should_start);
void CtplTestHookAfterPushValidation();
void CtplTestHookAfterResizeLifecycleLock(int thread_count);
void CtplTestHookAfterStopLifecycleLock(bool is_wait);
void CtplTestHookBeforeCancelQueueExtract();
void CtplTestHookBeforeGrowthFlag(int worker_id);
void CtplTestHookBeforeGrowthGateWait(int worker_id);
void CtplTestHookBeforeGrowthThread(int worker_id);
void CtplTestHookBeforePoolLock();
void CtplTestHookBeforeQueueCommit();
void CtplTestHookBeforeResizeLifecycleLock(int thread_count);
void CtplTestHookBeforeRetiredThreadJoin();
void CtplTestHookBeforeStagedThreadJoin(int worker_id);
void CtplTestHookBeforeStopLifecycleLock(bool is_wait);
void CtplTestHookBeforeStopLifecycleRelease(bool is_wait);
void CtplTestHookBeforeStopLock();
void CtplTestHookBeforeWorkerDequeueLock(int worker_id);

#define CTPL_STL_TEST_HOOK_AFTER_EMPTY_POP() ::CtplTestHookAfterEmptyPop()
#define CTPL_STL_TEST_HOOK_AFTER_CANCEL_QUEUE_EXTRACT() ::CtplTestHookAfterCancelQueueExtract()
#define CTPL_STL_TEST_HOOK_AFTER_GROWTH_GATE_WAIT(worker_id, should_start) ::CtplTestHookAfterGrowthGateWait(worker_id, should_start)
#define CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION() ::CtplTestHookAfterPushValidation()
#define CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK(thread_count) ::CtplTestHookAfterResizeLifecycleLock(thread_count)
#define CTPL_STL_TEST_HOOK_AFTER_STOP_LIFECYCLE_LOCK(is_wait) ::CtplTestHookAfterStopLifecycleLock(is_wait)
#define CTPL_STL_TEST_HOOK_BEFORE_CANCEL_QUEUE_EXTRACT() ::CtplTestHookBeforeCancelQueueExtract()
#define CTPL_STL_TEST_HOOK_BEFORE_GROWTH_FLAG(worker_id) ::CtplTestHookBeforeGrowthFlag(worker_id)
#define CTPL_STL_TEST_HOOK_BEFORE_GROWTH_GATE_WAIT(worker_id) ::CtplTestHookBeforeGrowthGateWait(worker_id)
#define CTPL_STL_TEST_HOOK_BEFORE_GROWTH_THREAD(worker_id) ::CtplTestHookBeforeGrowthThread(worker_id)
#define CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK() ::CtplTestHookBeforePoolLock()
#define CTPL_STL_TEST_HOOK_BEFORE_QUEUE_COMMIT() ::CtplTestHookBeforeQueueCommit()
#define CTPL_STL_TEST_HOOK_BEFORE_RESIZE_LIFECYCLE_LOCK(thread_count) ::CtplTestHookBeforeResizeLifecycleLock(thread_count)
#define CTPL_STL_TEST_HOOK_BEFORE_RETIRED_THREAD_JOIN() ::CtplTestHookBeforeRetiredThreadJoin()
#define CTPL_STL_TEST_HOOK_BEFORE_STAGED_THREAD_JOIN(worker_id) ::CtplTestHookBeforeStagedThreadJoin(worker_id)
#define CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_LOCK(is_wait) ::CtplTestHookBeforeStopLifecycleLock(is_wait)
#define CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_RELEASE(is_wait) ::CtplTestHookBeforeStopLifecycleRelease(is_wait)
#define CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK() ::CtplTestHookBeforeStopLock()
#define CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK(worker_id) ::CtplTestHookBeforeWorkerDequeueLock(worker_id)
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_EMPTY_POP
#undef CTPL_STL_TEST_HOOK_AFTER_CANCEL_QUEUE_EXTRACT
#undef CTPL_STL_TEST_HOOK_AFTER_GROWTH_GATE_WAIT
#undef CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION
#undef CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK
#undef CTPL_STL_TEST_HOOK_AFTER_STOP_LIFECYCLE_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_CANCEL_QUEUE_EXTRACT
#undef CTPL_STL_TEST_HOOK_BEFORE_GROWTH_FLAG
#undef CTPL_STL_TEST_HOOK_BEFORE_GROWTH_GATE_WAIT
#undef CTPL_STL_TEST_HOOK_BEFORE_GROWTH_THREAD
#undef CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_QUEUE_COMMIT
#undef CTPL_STL_TEST_HOOK_BEFORE_RESIZE_LIFECYCLE_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_RETIRED_THREAD_JOIN
#undef CTPL_STL_TEST_HOOK_BEFORE_STAGED_THREAD_JOIN
#undef CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_RELEASE
#undef CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono;

std::atomic<bool> wait_hook_armed(false);
std::mutex wait_hook_mutex;
std::condition_variable wait_hook_cv;
bool wait_hook_entered = false;
bool wait_hook_released = false;

std::atomic<bool> publication_hook_armed(false);
std::mutex publication_hook_mutex;
std::condition_variable publication_hook_cv;
bool before_pool_lock_entered = false;
bool queue_commit_entered = false;

std::atomic<bool> queue_commit_failure(false);

std::atomic<bool> cancellation_hooks_armed(false);
std::mutex cancellation_hook_mutex;
std::condition_variable cancellation_hook_cv;
bool push_validation_entered = false;
bool push_validation_released = false;
bool stop_lock_entered = false;
int cancellation_event = 0;
int queue_commit_event = 0;
int cancel_extract_event = 0;

std::atomic<bool> resize_join_hook_armed(false);
std::mutex resize_join_hook_mutex;
std::condition_variable resize_join_hook_cv;
bool resize_join_hook_entered = false;

enum GrowthFailure {
    kNoGrowthFailure = 0,
    kGrowthFlagFailure = 1,
    kGrowthThreadFailure = 2,
};

std::atomic<int> growth_failure(kNoGrowthFailure);
std::atomic<bool> growth_hooks_armed(false);
std::mutex growth_hook_mutex;
std::condition_variable growth_hook_cv;
int growth_gate_waits = 0;
int growth_rollback_releases = 0;
int growth_rollback_joins = 0;

std::atomic<bool> lifecycle_hooks_armed(false);
std::mutex lifecycle_hook_mutex;
std::condition_variable lifecycle_hook_cv;
bool draining_stop_active = false;
bool cancelling_stop_before_lock = false;
bool cancelling_stop_after_lock = false;
bool mixed_stop_overlap = false;

std::atomic<bool> resize_serialization_hooks_armed(false);
std::mutex resize_serialization_mutex;
std::condition_variable resize_serialization_cv;
int resize_before_lock_count = 0;
int resize_after_lock_count = 0;
bool release_first_resize = false;
bool stop_before_lifecycle_lock = false;
bool stop_after_lifecycle_lock = false;

std::atomic<bool> worker_cancel_hooks_armed(false);
std::atomic<bool> worker_transition_hook_armed(false);
std::mutex worker_cancel_mutex;
std::condition_variable worker_cancel_cv;
bool worker_transition_entered = false;
bool release_worker_transition = false;
bool worker_cancel_queue_extracted = false;

void Require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Predicate>
bool WaitUntil(Predicate predicate, milliseconds timeout) {
    const steady_clock::time_point deadline = steady_clock::now() + timeout;
    while (!predicate()) {
        if (steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(milliseconds(1));
    }
    return true;
}

void TestWorkerOwnership() {
    ctpl::thread_pool pool(1);
    Require(!pool.owns_current_thread(), "external thread was reported as a pool worker");
    std::future<bool> result = pool.push([&pool](int) { return pool.owns_current_thread(); });
    Require(result.wait_for(seconds(2)) == std::future_status::ready, "worker-identity task did not finish");
    Require(result.get(), "pool worker was not recognized");
}

void TestPublicationOrdering() {
    {
        std::lock_guard<std::mutex> lock(wait_hook_mutex);
        wait_hook_entered = false;
        wait_hook_released = false;
    }
    {
        std::lock_guard<std::mutex> lock(publication_hook_mutex);
        before_pool_lock_entered = false;
        queue_commit_entered = false;
    }
    wait_hook_armed.store(true);
    publication_hook_armed.store(true);
    ctpl::thread_pool pool(1);

    {
        std::unique_lock<std::mutex> lock(wait_hook_mutex);
        Require(wait_hook_cv.wait_for(lock, seconds(2), [] { return wait_hook_entered; }), "worker did not reach the controlled empty-queue wait");
    }

    std::future<int> task_result;
    std::exception_ptr producer_error;
    std::atomic<bool> producer_started(false);
    std::thread producer([&] {
        producer_started.store(true);
        try {
            task_result = pool.push([](int) { return 7; });
        } catch (...) {
            producer_error = std::current_exception();
        }
    });

    Require(WaitUntil([&] { return producer_started.load(); }, seconds(2)), "producer did not start");
    bool reached_pool_lock = false;
    bool queue_published_early = false;
    {
        std::unique_lock<std::mutex> lock(publication_hook_mutex);
        reached_pool_lock = publication_hook_cv.wait_for(lock, seconds(2), [] { return before_pool_lock_entered; });
        queue_published_early = queue_commit_entered;
    }

    {
        std::lock_guard<std::mutex> lock(wait_hook_mutex);
        wait_hook_released = true;
    }
    wait_hook_cv.notify_all();
    producer.join();
    publication_hook_armed.store(false);
    if (producer_error) {
        std::rethrow_exception(producer_error);
    }

    Require(reached_pool_lock, "producer did not reach the condition-mutex acquisition");
    Require(!queue_published_early, "queue publication preceded condition-mutex acquisition");
    Require(task_result.wait_for(seconds(2)) == std::future_status::ready, "controlled handoff lost its wakeup");
    Require(task_result.get() == 7, "controlled handoff returned the wrong value");
}

void TestPushCancelLinearization() {
    {
        std::lock_guard<std::mutex> lock(cancellation_hook_mutex);
        push_validation_entered = false;
        push_validation_released = false;
        stop_lock_entered = false;
        cancellation_event = 0;
        queue_commit_event = 0;
        cancel_extract_event = 0;
    }
    cancellation_hooks_armed.store(true);
    ctpl::thread_pool pool(0);

    std::future<int> accepted;
    std::exception_ptr producer_error;
    std::thread producer([&] {
        try {
            accepted = pool.push([](int) { return 31; });
        } catch (...) {
            producer_error = std::current_exception();
        }
    });

    {
        std::unique_lock<std::mutex> lock(cancellation_hook_mutex);
        Require(cancellation_hook_cv.wait_for(lock, seconds(2), [] { return push_validation_entered; }),
                "producer did not pause after push validation");
    }

    std::exception_ptr stop_error;
    std::thread stopper([&] {
        try {
            pool.stop(false);
        } catch (...) {
            stop_error = std::current_exception();
        }
    });

    {
        std::unique_lock<std::mutex> lock(cancellation_hook_mutex);
        Require(cancellation_hook_cv.wait_for(lock, seconds(2), [] { return stop_lock_entered; }), "cancelling stop did not reach the pool mutex");
        push_validation_released = true;
    }
    cancellation_hook_cv.notify_all();

    producer.join();
    stopper.join();
    cancellation_hooks_armed.store(false);
    if (producer_error) {
        std::rethrow_exception(producer_error);
    }
    if (stop_error) {
        std::rethrow_exception(stop_error);
    }

    int committed_at = 0;
    int extracted_at = 0;
    {
        std::lock_guard<std::mutex> lock(cancellation_hook_mutex);
        committed_at = queue_commit_event;
        extracted_at = cancel_extract_event;
    }
    Require(committed_at > 0, "accepted push did not publish its task");
    Require(extracted_at > committed_at, "cancellation extracted the queue before accepted push publication");
    Require(accepted.valid(), "accepted push did not return a future");
    Require(accepted.wait_for(seconds(2)) == std::future_status::ready, "cancelled task future did not become ready");

    bool broken_promise = false;
    try {
        static_cast<void>(accepted.get());
    } catch (const std::future_error &error) {
        broken_promise = error.code() == std::make_error_code(std::future_errc::broken_promise);
    }
    Require(broken_promise, "cancelled accepted task did not report broken_promise");
}

void TestQueueCommitFailure() {
    ctpl::thread_pool pool(1);
    std::atomic<int> executions(0);

    queue_commit_failure.store(true);
    bool rejected = false;
    try {
        std::future<void> ignored = pool.push([&executions](int) { ++executions; });
        static_cast<void>(ignored);
    } catch (const std::bad_alloc &) {
        rejected = true;
    }
    Require(rejected, "queue-commit fault was not propagated");
    std::this_thread::sleep_for(milliseconds(20));
    Require(executions.load() == 0, "rejected unary task was retained");

    queue_commit_failure.store(true);
    rejected = false;
    try {
        std::future<int> ignored = pool.push([](int, int value) { return value; }, 11);
        static_cast<void>(ignored);
    } catch (const std::bad_alloc &) {
        rejected = true;
    }
    Require(rejected, "variadic queue-commit fault was not propagated");

    std::future<int> probe = pool.push([&executions](int) {
        ++executions;
        return 19;
    });
    Require(probe.wait_for(seconds(2)) == std::future_status::ready, "pool did not recover after queue fault");
    Require(probe.get() == 19 && executions.load() == 1, "queue fault retained a ghost task");
}

struct ThrowOnCopy {
    explicit ThrowOnCopy(std::atomic<int> &executions) : executions_(&executions) {}
    ThrowOnCopy(const ThrowOnCopy &) { throw std::runtime_error("injected callable copy failure"); }
    void operator()(int) const { ++*executions_; }
    std::atomic<int> *executions_;
};

void TestCallableFailures() {
    ctpl::thread_pool pool(1);
    std::atomic<int> executions(0);
    ThrowOnCopy callable(executions);
    bool rejected = false;
    try {
        std::future<void> ignored = pool.push(callable);
        static_cast<void>(ignored);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    Require(rejected, "throwing callable copy was accepted");
    Require(executions.load() == 0, "throwing callable copy executed");

    std::future<void> invocation = pool.push([](int) { throw std::runtime_error("injected invocation failure"); });
    Require(invocation.wait_for(seconds(2)) == std::future_status::ready, "throwing invocation future did not become ready");
    bool propagated = false;
    try {
        invocation.get();
    } catch (const std::runtime_error &) {
        propagated = true;
    }
    Require(propagated, "task invocation exception was not propagated through the future");
}

template <typename Push>
void RequireStoppedPushRejected(Push push) {
    bool rejected = false;
    try {
        push();
    } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "cannot push to a stopped thread_pool";
    }
    Require(rejected, "post-stop push did not synchronously report the contract error");
}

void TestPostStopRejection() {
    ctpl::thread_pool drained(1);
    drained.stop(true);
    RequireStoppedPushRejected([&] { static_cast<void>(drained.push([](int) {})); });
    RequireStoppedPushRejected([&] { static_cast<void>(drained.push([](int, int) {}, 1)); });
    drained.stop(true);

    ctpl::thread_pool cancelled(0);
    cancelled.stop(false);
    RequireStoppedPushRejected([&] { static_cast<void>(cancelled.push([](int) {})); });
    cancelled.stop(false);
}

void TestNegativeResizeRejected() {
    bool constructor_rejected = false;
    try {
        ctpl::thread_pool invalid_pool(-1);
    } catch (const std::invalid_argument &error) {
        constructor_rejected = std::string(error.what()) == "thread_pool size cannot be negative";
    }
    Require(constructor_rejected, "negative constructor size did not report the contract error");

    ctpl::thread_pool pool(1);
    bool resize_rejected = false;
    try {
        pool.resize(-1);
    } catch (const std::invalid_argument &error) {
        resize_rejected = std::string(error.what()) == "thread_pool size cannot be negative";
    }
    Require(resize_rejected, "negative resize did not report the contract error");
    Require(pool.size() == 1, "negative resize changed the pool size");
}

void TestWorkerInitiatedResizeRejected() {
    ctpl::thread_pool pool(2);
    const int requested_sizes[] = {2, 3, 1};
    for (std::size_t index = 0; index < sizeof(requested_sizes) / sizeof(requested_sizes[0]); ++index) {
        const int requested_size = requested_sizes[index];
        std::future<void> resize_result = pool.push([&pool, requested_size](int) { pool.resize(requested_size); });
        Require(resize_result.wait_for(seconds(2)) == std::future_status::ready, "worker-initiated resize did not finish");

        bool rejected = false;
        try {
            resize_result.get();
        } catch (const std::logic_error &error) {
            rejected = std::string(error.what()) == "thread_pool worker cannot downsize its own pool";
        }
        Require(rejected, "worker-initiated resize did not report the contract error");
        Require(pool.size() == 2, "worker-initiated resize changed the pool size");
    }
}

struct ReentrantCaptureState {
    explicit ReentrantCaptureState(ctpl::thread_pool &owner)
        : pool(&owner), armed(false), entered(false), returned(false) {}

    ctpl::thread_pool *pool;
    std::atomic<bool> armed;
    std::atomic<bool> entered;
    std::atomic<bool> returned;
};

struct ReentrantStopCapture {
    explicit ReentrantStopCapture(const std::shared_ptr<ReentrantCaptureState> &capture_state)
        : state(capture_state) {}

    ~ReentrantStopCapture() {
        if (state && state->armed.load() && !state->entered.exchange(true)) {
            state->pool->stop(false);
            state->returned.store(true);
        }
    }

    void operator()(int) const {}

    std::shared_ptr<ReentrantCaptureState> state;
};

void TestReentrantCaptureDestruction() {
    ctpl::thread_pool pool(0);
    std::shared_ptr<ReentrantCaptureState> state(new ReentrantCaptureState(pool));
    std::future<void> cancelled = pool.push(ReentrantStopCapture(state));
    state->armed.store(true);

    std::promise<void> stop_finished_promise;
    std::future<void> stop_finished = stop_finished_promise.get_future();
    std::exception_ptr stop_error;
    std::thread stopper([&] {
        try {
            pool.stop(false);
        } catch (...) {
            stop_error = std::current_exception();
        }
        stop_finished_promise.set_value();
    });

    if (stop_finished.wait_for(seconds(2)) != std::future_status::ready) {
        std::cerr << "status=FAIL\n";
        std::cerr << "error=reentrant capture destruction deadlocked stop\n";
        std::cerr.flush();
        std::_Exit(1);
    }
    stopper.join();
    if (stop_error) {
        std::rethrow_exception(stop_error);
    }

    Require(state->entered.load(), "cancelled capture destructor did not re-enter stop");
    Require(state->returned.load(), "reentrant stop did not return");
    Require(cancelled.wait_for(seconds(2)) == std::future_status::ready, "cancelled reentrant task future did not become ready");

    bool broken_promise = false;
    try {
        cancelled.get();
    } catch (const std::future_error &error) {
        broken_promise = error.code() == std::make_error_code(std::future_errc::broken_promise);
    }
    Require(broken_promise, "cancelled reentrant task did not report broken_promise");
}

void TestConcurrentResizeSerialization() {
    ctpl::thread_pool pool(0);
    {
        std::lock_guard<std::mutex> lock(resize_serialization_mutex);
        resize_before_lock_count = 0;
        resize_after_lock_count = 0;
        release_first_resize = false;
        stop_before_lifecycle_lock = false;
        stop_after_lifecycle_lock = false;
    }
    resize_serialization_hooks_armed.store(true);

    std::exception_ptr first_error;
    std::thread first([&] {
        try {
            pool.resize(1);
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    {
        std::unique_lock<std::mutex> lock(resize_serialization_mutex);
        Require(resize_serialization_cv.wait_for(lock, seconds(2), [] { return resize_after_lock_count == 1; }),
                "first resize did not acquire lifecycle ownership");
    }

    std::exception_ptr second_error;
    std::thread second([&] {
        try {
            pool.resize(2);
        } catch (...) {
            second_error = std::current_exception();
        }
    });

    bool second_attempted = false;
    bool overlapped = false;
    {
        std::unique_lock<std::mutex> lock(resize_serialization_mutex);
        second_attempted = resize_serialization_cv.wait_for(lock, seconds(2), [] { return resize_before_lock_count == 2; });
        overlapped = resize_serialization_cv.wait_for(lock, milliseconds(100), [] { return resize_after_lock_count > 1; });
        release_first_resize = true;
    }
    resize_serialization_cv.notify_all();
    first.join();
    second.join();
    resize_serialization_hooks_armed.store(false);

    if (first_error) {
        std::rethrow_exception(first_error);
    }
    if (second_error) {
        std::rethrow_exception(second_error);
    }
    Require(second_attempted, "second resize did not contend for lifecycle ownership");
    Require(!overlapped, "concurrent resizes entered lifecycle together");
    Require(pool.size() == 2, "serialized resizes produced the wrong pool size");
}

void TestResizeStopSerialization() {
    ctpl::thread_pool pool(0);
    {
        std::lock_guard<std::mutex> lock(resize_serialization_mutex);
        resize_before_lock_count = 0;
        resize_after_lock_count = 0;
        release_first_resize = false;
        stop_before_lifecycle_lock = false;
        stop_after_lifecycle_lock = false;
    }
    resize_serialization_hooks_armed.store(true);

    std::exception_ptr resize_error;
    std::thread resize_thread([&] {
        try {
            pool.resize(1);
        } catch (...) {
            resize_error = std::current_exception();
        }
    });
    {
        std::unique_lock<std::mutex> lock(resize_serialization_mutex);
        Require(resize_serialization_cv.wait_for(lock, seconds(2), [] { return resize_after_lock_count == 1; }),
                "resize did not acquire lifecycle ownership before stop");
    }

    std::exception_ptr stop_error;
    std::thread stop_thread([&] {
        try {
            pool.stop(false);
        } catch (...) {
            stop_error = std::current_exception();
        }
    });

    bool stop_attempted = false;
    bool stop_overlapped = false;
    {
        std::unique_lock<std::mutex> lock(resize_serialization_mutex);
        stop_attempted = resize_serialization_cv.wait_for(lock, seconds(2), [] { return stop_before_lifecycle_lock; });
        stop_overlapped = resize_serialization_cv.wait_for(lock, milliseconds(100), [] { return stop_after_lifecycle_lock; });
        release_first_resize = true;
    }
    resize_serialization_cv.notify_all();
    resize_thread.join();
    stop_thread.join();
    resize_serialization_hooks_armed.store(false);

    if (resize_error) {
        std::rethrow_exception(resize_error);
    }
    if (stop_error) {
        std::rethrow_exception(stop_error);
    }
    Require(stop_attempted, "stop did not contend for resize lifecycle ownership");
    Require(!stop_overlapped, "stop entered lifecycle while resize owned it");
    Require(stop_after_lifecycle_lock, "stop never acquired lifecycle ownership after resize");
}

void TestActiveWorkerCancellationBoundary() {
    {
        std::lock_guard<std::mutex> lock(worker_cancel_mutex);
        worker_transition_entered = false;
        release_worker_transition = false;
        worker_cancel_queue_extracted = false;
    }
    ctpl::thread_pool pool(1);
    std::promise<void> active_entered_promise;
    std::future<void> active_entered = active_entered_promise.get_future();
    std::promise<void> release_active_promise;
    std::shared_future<void> release_active = release_active_promise.get_future().share();
    std::future<void> active = pool.push([&](int) {
        active_entered_promise.set_value();
        release_active.wait();
    });
    Require(active_entered.wait_for(seconds(2)) == std::future_status::ready, "active cancellation worker did not start");

    std::atomic<int> queued_executions(0);
    std::future<void> queued = pool.push([&](int) { ++queued_executions; });
    worker_cancel_hooks_armed.store(true);
    worker_transition_hook_armed.store(true);
    release_active_promise.set_value();
    {
        std::unique_lock<std::mutex> lock(worker_cancel_mutex);
        Require(worker_cancel_cv.wait_for(lock, seconds(2), [] { return worker_transition_entered; }),
                "active worker did not pause at the dequeue boundary");
    }

    std::exception_ptr stop_error;
    std::thread stopper([&] {
        try {
            pool.stop(false);
        } catch (...) {
            stop_error = std::current_exception();
        }
    });
    {
        std::unique_lock<std::mutex> lock(worker_cancel_mutex);
        Require(worker_cancel_cv.wait_for(lock, seconds(2), [] { return worker_cancel_queue_extracted; }),
                "cancellation did not reach its queue boundary");
        release_worker_transition = true;
    }
    worker_cancel_cv.notify_all();
    stopper.join();
    worker_cancel_hooks_armed.store(false);
    worker_transition_hook_armed.store(false);

    if (stop_error) {
        std::rethrow_exception(stop_error);
    }
    Require(active.wait_for(seconds(2)) == std::future_status::ready, "active task did not finish during cancellation");
    active.get();
    Require(queued_executions.load() == 0, "task queued at cancellation boundary started afterward");
    Require(queued.wait_for(seconds(2)) == std::future_status::ready, "cancelled boundary task future did not become ready");

    bool broken_promise = false;
    try {
        queued.get();
    } catch (const std::future_error &error) {
        broken_promise = error.code() == std::make_error_code(std::future_errc::broken_promise);
    }
    Require(broken_promise, "cancelled boundary task did not report broken_promise");
}

void TestDrainSemantics() {
    ctpl::thread_pool pool(2);
    std::atomic<int> executions(0);
    std::vector<std::future<void>> futures;
    for (int index = 0; index < 100; ++index) {
        futures.push_back(pool.push([&](int) { ++executions; }));
    }
    pool.stop(true);
    Require(executions.load() == 100, "draining stop did not execute every queued task");
    for (std::size_t index = 0; index < futures.size(); ++index) {
        futures[index].get();
    }
}

void RunWorkerStopRejection(bool is_wait) {
    ctpl::thread_pool pool(1);
    std::promise<void> stop_entered_promise;
    std::future<void> stop_entered = stop_entered_promise.get_future();
    std::promise<void> release_stop_promise;
    std::shared_future<void> release_stop = release_stop_promise.get_future().share();
    std::future<void> stop_result = pool.push([&](int) {
        stop_entered_promise.set_value();
        release_stop.wait();
        pool.stop(is_wait);
    });
    Require(stop_entered.wait_for(seconds(2)) == std::future_status::ready, "worker stop task did not start");

    std::atomic<int> sentinel_executions(0);
    std::future<void> sentinel = pool.push([&](int) { ++sentinel_executions; });
    release_stop_promise.set_value();
    Require(stop_result.wait_for(seconds(2)) == std::future_status::ready, "worker-initiated stop did not finish");

    bool rejected = false;
    try {
        stop_result.get();
    } catch (const std::logic_error &error) {
        rejected = std::string(error.what()) == "thread_pool worker cannot stop its own pool";
    }
    Require(rejected, "worker-initiated stop did not report the contract error");
    Require(pool.size() == 1, "worker-initiated stop changed the pool size");
    Require(sentinel.wait_for(seconds(2)) == std::future_status::ready, "queued sentinel did not run after rejected worker stop");
    sentinel.get();
    Require(sentinel_executions.load() == 1, "queued sentinel ran an unexpected number of times");

    std::future<int> probe = pool.push([](int) { return 37; });
    Require(probe.wait_for(seconds(2)) == std::future_status::ready, "pool rejected work after worker stop rejection");
    Require(probe.get() == 37, "post-rejection probe returned the wrong value");
}

void TestWorkerInitiatedStopRejected() {
    RunWorkerStopRejection(false);
    RunWorkerStopRejection(true);
}

void TestMixedStopSerialization() {
    {
        std::lock_guard<std::mutex> lock(lifecycle_hook_mutex);
        draining_stop_active = false;
        cancelling_stop_before_lock = false;
        cancelling_stop_after_lock = false;
        mixed_stop_overlap = false;
    }
    lifecycle_hooks_armed.store(true);

    ctpl::thread_pool pool(1);
    std::promise<void> blocker_entered_promise;
    std::future<void> blocker_entered = blocker_entered_promise.get_future();
    std::promise<void> release_blocker_promise;
    std::shared_future<void> release_blocker = release_blocker_promise.get_future().share();
    std::future<void> blocker = pool.push([&](int) {
        blocker_entered_promise.set_value();
        release_blocker.wait();
    });
    Require(blocker_entered.wait_for(seconds(2)) == std::future_status::ready, "mixed-stop blocker did not start");

    std::exception_ptr draining_error;
    std::thread draining_stop([&] {
        try {
            pool.stop(true);
        } catch (...) {
            draining_error = std::current_exception();
        }
    });

    bool drain_acquired = false;
    {
        std::unique_lock<std::mutex> lock(lifecycle_hook_mutex);
        drain_acquired = lifecycle_hook_cv.wait_for(lock, seconds(2), [] { return draining_stop_active; });
    }
    if (!drain_acquired) {
        release_blocker_promise.set_value();
        draining_stop.join();
        lifecycle_hooks_armed.store(false);
        Require(false, "draining stop did not acquire lifecycle ownership");
    }

    std::exception_ptr cancelling_error;
    std::thread cancelling_stop([&] {
        try {
            pool.stop(false);
        } catch (...) {
            cancelling_error = std::current_exception();
        }
    });

    bool cancel_reached_lock = false;
    {
        std::unique_lock<std::mutex> lock(lifecycle_hook_mutex);
        cancel_reached_lock = lifecycle_hook_cv.wait_for(lock, seconds(2), [] { return cancelling_stop_before_lock; });
    }
    release_blocker_promise.set_value();
    draining_stop.join();
    cancelling_stop.join();
    lifecycle_hooks_armed.store(false);

    if (draining_error) {
        std::rethrow_exception(draining_error);
    }
    if (cancelling_error) {
        std::rethrow_exception(cancelling_error);
    }
    Require(cancel_reached_lock, "cancelling stop did not contend for lifecycle ownership");
    Require(blocker.wait_for(seconds(2)) == std::future_status::ready, "mixed-stop blocker did not finish");
    blocker.get();

    bool cancel_acquired = false;
    bool overlap = false;
    {
        std::lock_guard<std::mutex> lock(lifecycle_hook_mutex);
        cancel_acquired = cancelling_stop_after_lock;
        overlap = mixed_stop_overlap;
    }
    Require(cancel_acquired, "cancelling stop did not acquire lifecycle ownership after drain");
    Require(!overlap, "mixed stop operations entered lifecycle concurrently");
}

void RunGrowthFailureRollback(GrowthFailure failure) {
    ctpl::thread_pool pool(1);
    std::promise<void> blocker_entered_promise;
    std::future<void> blocker_entered = blocker_entered_promise.get_future();
    std::promise<void> release_blocker_promise;
    std::shared_future<void> release_blocker = release_blocker_promise.get_future().share();
    std::future<void> blocker = pool.push([&](int) {
        blocker_entered_promise.set_value();
        release_blocker.wait();
    });
    Require(blocker_entered.wait_for(seconds(2)) == std::future_status::ready, "existing worker blocker did not start");

    std::atomic<int> sentinel_executions(0);
    std::future<void> sentinel = pool.push([&](int) { ++sentinel_executions; });
    {
        std::lock_guard<std::mutex> lock(growth_hook_mutex);
        growth_gate_waits = 0;
        growth_rollback_releases = 0;
        growth_rollback_joins = 0;
    }
    growth_failure.store(failure);
    growth_hooks_armed.store(true);

    bool rejected = false;
    try {
        pool.resize(3);
    } catch (const std::bad_alloc &) {
        rejected = failure == kGrowthFlagFailure;
    } catch (const std::system_error &error) {
        rejected = failure == kGrowthThreadFailure && error.code() == std::make_error_code(std::errc::resource_unavailable_try_again);
    }
    growth_hooks_armed.store(false);
    growth_failure.store(kNoGrowthFailure);

    Require(rejected, "injected growth failure was not propagated");
    Require(pool.size() == 1, "failed growth changed the published pool size");
    Require(sentinel_executions.load() == 0, "staged worker consumed queued work before growth commit");

    int gate_waits = 0;
    int rollback_releases = 0;
    int rollback_joins = 0;
    {
        std::lock_guard<std::mutex> lock(growth_hook_mutex);
        gate_waits = growth_gate_waits;
        rollback_releases = growth_rollback_releases;
        rollback_joins = growth_rollback_joins;
    }
    if (failure == kGrowthFlagFailure) {
        Require(gate_waits == 0, "flag-allocation failure started a staged worker");
        Require(rollback_joins == 0, "flag-allocation failure attempted a staged-worker join");
    } else {
        Require(gate_waits == 1, "thread-creation failure did not stage exactly one worker");
        Require(rollback_releases == 1, "failed growth did not release its staged worker for rollback");
        Require(rollback_joins == 1, "failed growth did not join its staged worker");
    }

    release_blocker_promise.set_value();
    Require(blocker.wait_for(seconds(2)) == std::future_status::ready, "existing worker blocker did not finish");
    blocker.get();
    Require(sentinel.wait_for(seconds(2)) == std::future_status::ready, "queued work was lost by failed growth");
    sentinel.get();
    Require(sentinel_executions.load() == 1, "queued work changed during failed growth");

    pool.resize(3);
    Require(pool.size() == 3, "pool did not recover after failed growth");
    std::future<int> probe = pool.push([](int) { return 41; });
    Require(probe.wait_for(seconds(2)) == std::future_status::ready, "recovered pool did not execute work");
    Require(probe.get() == 41, "recovered pool returned the wrong value");
}

void TestGrowthFailureRollback() {
    RunGrowthFailureRollback(kGrowthFlagFailure);
    RunGrowthFailureRollback(kGrowthThreadFailure);
}

void TestExternalDownsizeJoinsRetiringWorker() {
    ctpl::thread_pool pool(1);
    std::promise<void> task_entered_promise;
    std::future<void> task_entered = task_entered_promise.get_future();
    std::promise<void> release_task_promise;
    std::shared_future<void> release_task = release_task_promise.get_future().share();
    std::future<int> task_result = pool.push([&](int) {
        task_entered_promise.set_value();
        release_task.wait();
        return 23;
    });
    Require(task_entered.wait_for(seconds(2)) == std::future_status::ready, "retiring worker task did not start");

    {
        std::lock_guard<std::mutex> lock(resize_join_hook_mutex);
        resize_join_hook_entered = false;
    }
    resize_join_hook_armed.store(true);
    std::future<void> resize_result = std::async(std::launch::async, [&pool] { pool.resize(0); });

    bool join_started = false;
    {
        std::unique_lock<std::mutex> lock(resize_join_hook_mutex);
        join_started = resize_join_hook_cv.wait_for(lock, seconds(2), [] { return resize_join_hook_entered; });
    }
    const bool resize_returned_while_active = resize_result.wait_for(milliseconds(0)) == std::future_status::ready;

    release_task_promise.set_value();
    Require(task_result.wait_for(seconds(2)) == std::future_status::ready, "retiring worker task did not finish");
    Require(task_result.get() == 23, "retiring worker task returned the wrong value");
    Require(resize_result.wait_for(seconds(2)) == std::future_status::ready, "external downsize did not join the retiring worker");
    resize_result.get();
    resize_join_hook_armed.store(false);

    Require(join_started, "external downsize did not attempt to join the retiring worker");
    Require(!resize_returned_while_active, "resize returned while a retiring worker was active");
    Require(pool.size() == 0, "external downsize did not reach size zero");

    pool.resize(1);
    std::future<int> probe = pool.push([](int) { return 29; });
    Require(probe.wait_for(seconds(2)) == std::future_status::ready, "resized pool did not accept new work");
    Require(probe.get() == 29, "resized pool returned the wrong value");
}

void TestUninstrumentedHandoffs() {
    ctpl::thread_pool pool(2);
    std::atomic<int> executions(0);
    const int batch_count = 200;
    const int batch_size = 100;
    for (int batch = 0; batch < batch_count; ++batch) {
        std::vector<std::future<void>> futures;
        futures.reserve(batch_size);
        for (int task = 0; task < batch_size; ++task) {
            futures.push_back(pool.push([&executions](int) { ++executions; }));
        }
        for (std::size_t task = 0; task < futures.size(); ++task) {
            Require(futures[task].wait_for(seconds(2)) == std::future_status::ready, "uninstrumented handoff stalled");
            futures[task].get();
        }
    }
    Require(executions.load() == batch_count * batch_size, "uninstrumented handoff lost or duplicated work");
}

} // namespace

void CtplTestHookAfterEmptyPop() {
    if (!wait_hook_armed.exchange(false)) {
        return;
    }
    std::unique_lock<std::mutex> lock(wait_hook_mutex);
    wait_hook_entered = true;
    wait_hook_cv.notify_all();
    wait_hook_cv.wait(lock, [] { return wait_hook_released; });
}

void CtplTestHookAfterCancelQueueExtract() {
    if (!worker_cancel_hooks_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(worker_cancel_mutex);
    worker_cancel_queue_extracted = true;
    worker_cancel_cv.notify_all();
}

void CtplTestHookAfterGrowthGateWait(int worker_id, bool should_start) {
    if (!growth_hooks_armed.load() || worker_id < 1 || should_start) {
        return;
    }
    std::lock_guard<std::mutex> lock(growth_hook_mutex);
    ++growth_rollback_releases;
    growth_hook_cv.notify_all();
}

void CtplTestHookAfterPushValidation() {
    if (!cancellation_hooks_armed.load()) {
        return;
    }
    std::unique_lock<std::mutex> lock(cancellation_hook_mutex);
    push_validation_entered = true;
    cancellation_hook_cv.notify_all();
    cancellation_hook_cv.wait(lock, [] { return push_validation_released; });
}

void CtplTestHookAfterResizeLifecycleLock(int) {
    if (!resize_serialization_hooks_armed.load()) {
        return;
    }
    std::unique_lock<std::mutex> lock(resize_serialization_mutex);
    ++resize_after_lock_count;
    resize_serialization_cv.notify_all();
    if (resize_after_lock_count == 1) {
        resize_serialization_cv.wait(lock, [] { return release_first_resize; });
    }
}

void CtplTestHookAfterStopLifecycleLock(bool is_wait) {
    if (lifecycle_hooks_armed.load()) {
        std::lock_guard<std::mutex> lock(lifecycle_hook_mutex);
        if (is_wait) {
            draining_stop_active = true;
        } else {
            mixed_stop_overlap = draining_stop_active;
            cancelling_stop_after_lock = true;
        }
        lifecycle_hook_cv.notify_all();
    }
    if (resize_serialization_hooks_armed.load()) {
        std::lock_guard<std::mutex> lock(resize_serialization_mutex);
        stop_after_lifecycle_lock = true;
        resize_serialization_cv.notify_all();
    }
}

void CtplTestHookBeforeCancelQueueExtract() {
    if (!cancellation_hooks_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(cancellation_hook_mutex);
    cancel_extract_event = ++cancellation_event;
    cancellation_hook_cv.notify_all();
}

void CtplTestHookBeforeGrowthFlag(int worker_id) {
    if (growth_hooks_armed.load() && worker_id == 1 && growth_failure.load() == kGrowthFlagFailure) {
        throw std::bad_alloc();
    }
}

void CtplTestHookBeforeGrowthGateWait(int worker_id) {
    if (!growth_hooks_armed.load() || worker_id < 1) {
        return;
    }
    std::lock_guard<std::mutex> lock(growth_hook_mutex);
    ++growth_gate_waits;
    growth_hook_cv.notify_all();
}

void CtplTestHookBeforeGrowthThread(int worker_id) {
    if (!growth_hooks_armed.load() || worker_id != 2 || growth_failure.load() != kGrowthThreadFailure) {
        return;
    }
    std::unique_lock<std::mutex> lock(growth_hook_mutex);
    if (!growth_hook_cv.wait_for(lock, seconds(2), [] { return growth_gate_waits == 1; })) {
        throw std::runtime_error("staged worker did not reach its startup gate");
    }
    throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again), "injected thread creation failure");
}

void CtplTestHookBeforePoolLock() {
    if (!publication_hook_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(publication_hook_mutex);
    before_pool_lock_entered = true;
    publication_hook_cv.notify_all();
}

void CtplTestHookBeforeQueueCommit() {
    if (queue_commit_failure.exchange(false)) {
        throw std::bad_alloc();
    }
    if (cancellation_hooks_armed.load()) {
        std::lock_guard<std::mutex> lock(cancellation_hook_mutex);
        queue_commit_event = ++cancellation_event;
        cancellation_hook_cv.notify_all();
    }
    if (!publication_hook_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(publication_hook_mutex);
    queue_commit_entered = true;
    publication_hook_cv.notify_all();
}

void CtplTestHookBeforeResizeLifecycleLock(int) {
    if (!resize_serialization_hooks_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(resize_serialization_mutex);
    ++resize_before_lock_count;
    resize_serialization_cv.notify_all();
}

void CtplTestHookBeforeRetiredThreadJoin() {
    if (!resize_join_hook_armed.exchange(false)) {
        return;
    }
    std::lock_guard<std::mutex> lock(resize_join_hook_mutex);
    resize_join_hook_entered = true;
    resize_join_hook_cv.notify_all();
}

void CtplTestHookBeforeStagedThreadJoin(int worker_id) {
    if (!growth_hooks_armed.load() || worker_id < 1) {
        return;
    }
    std::lock_guard<std::mutex> lock(growth_hook_mutex);
    ++growth_rollback_joins;
    growth_hook_cv.notify_all();
}

void CtplTestHookBeforeStopLifecycleLock(bool is_wait) {
    if (lifecycle_hooks_armed.load() && !is_wait) {
        std::lock_guard<std::mutex> lock(lifecycle_hook_mutex);
        cancelling_stop_before_lock = true;
        lifecycle_hook_cv.notify_all();
    }
    if (resize_serialization_hooks_armed.load()) {
        std::lock_guard<std::mutex> lock(resize_serialization_mutex);
        stop_before_lifecycle_lock = true;
        resize_serialization_cv.notify_all();
    }
}

void CtplTestHookBeforeStopLifecycleRelease(bool is_wait) {
    if (!lifecycle_hooks_armed.load() || !is_wait) {
        return;
    }
    std::lock_guard<std::mutex> lock(lifecycle_hook_mutex);
    draining_stop_active = false;
    lifecycle_hook_cv.notify_all();
}

void CtplTestHookBeforeStopLock() {
    if (!cancellation_hooks_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(cancellation_hook_mutex);
    stop_lock_entered = true;
    cancellation_hook_cv.notify_all();
}

void CtplTestHookBeforeWorkerDequeueLock(int) {
    if (!worker_cancel_hooks_armed.load() || !worker_transition_hook_armed.exchange(false)) {
        return;
    }
    std::unique_lock<std::mutex> lock(worker_cancel_mutex);
    worker_transition_entered = true;
    worker_cancel_cv.notify_all();
    worker_cancel_cv.wait(lock, [] { return release_worker_transition; });
}

int main() {
    static_assert(ctpl::thread_pool::push_has_strong_exception_guarantee, "reviewed CTPL overlay is required");
    try {
        const bool guarantee_marker = ctpl::thread_pool::push_has_strong_exception_guarantee;
        Require(guarantee_marker, "strong-push guarantee marker returned false");
        TestWorkerOwnership();
        TestPublicationOrdering();
        TestPushCancelLinearization();
        TestQueueCommitFailure();
        TestCallableFailures();
        TestPostStopRejection();
        TestNegativeResizeRejected();
        TestWorkerInitiatedResizeRejected();
        TestReentrantCaptureDestruction();
        TestConcurrentResizeSerialization();
        TestResizeStopSerialization();
        TestActiveWorkerCancellationBoundary();
        TestDrainSemantics();
        TestWorkerInitiatedStopRejected();
        TestMixedStopSerialization();
        TestGrowthFailureRollback();
        TestExternalDownsizeJoinsRetiringWorker();
        TestUninstrumentedHandoffs();
        std::cout << "status=PASS\n";
        std::cout << "publication_order=1\n";
        std::cout << "push_cancel_linearization=1\n";
        std::cout << "queue_commit_failure=1\n";
        std::cout << "post_stop_rejection=1\n";
        std::cout << "worker_ownership=1\n";
        std::cout << "negative_resize_rejection=1\n";
        std::cout << "worker_resize_rejection=3\n";
        std::cout << "reentrant_capture_destruction=1\n";
        std::cout << "concurrent_resize_serialization=1\n";
        std::cout << "resize_stop_serialization=1\n";
        std::cout << "active_worker_cancellation_boundary=1\n";
        std::cout << "drain_semantics=100\n";
        std::cout << "worker_stop_rejection=2\n";
        std::cout << "mixed_stop_serialization=1\n";
        std::cout << "growth_failure_rollback=2\n";
        std::cout << "retiring_worker_join=1\n";
        std::cout << "uninstrumented_handoffs=20000\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "status=FAIL\n";
        std::cerr << "error=" << error.what() << '\n';
        return 1;
    }
}
