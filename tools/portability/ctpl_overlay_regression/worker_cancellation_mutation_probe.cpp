#include <cstdlib>

void CtplTestHookAfterCancelQueueExtract();
void CtplTestHookBeforeWorkerDequeueLock(int worker_id);

#define CTPL_STL_TEST_HOOK_AFTER_CANCEL_QUEUE_EXTRACT() ::CtplTestHookAfterCancelQueueExtract()
#define CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK(worker_id) ::CtplTestHookBeforeWorkerDequeueLock(worker_id)
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_CANCEL_QUEUE_EXTRACT
#undef CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

namespace {

using namespace std::chrono;

std::atomic<bool> transition_armed(false);
std::mutex hook_mutex;
std::condition_variable hook_cv;
bool transition_entered = false;
bool release_transition = false;
bool queue_extracted = false;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=worker dequeued outside the cancellation boundary\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookAfterCancelQueueExtract() {
    std::lock_guard<std::mutex> lock(hook_mutex);
    queue_extracted = true;
    hook_cv.notify_all();
}

void CtplTestHookBeforeWorkerDequeueLock(int) {
    if (!transition_armed.exchange(false)) {
        return;
    }
    std::unique_lock<std::mutex> lock(hook_mutex);
    transition_entered = true;
    hook_cv.notify_all();
    hook_cv.wait(lock, [] { return release_transition; });
}

int main() {
    ctpl::thread_pool pool(1);
    std::promise<void> active_entered_promise;
    std::future<void> active_entered = active_entered_promise.get_future();
    std::promise<void> release_active_promise;
    std::shared_future<void> release_active = release_active_promise.get_future().share();
    std::future<void> active = pool.push([&](int) {
        active_entered_promise.set_value();
        release_active.wait();
    });
    if (active_entered.wait_for(seconds(2)) != std::future_status::ready) {
        return 2;
    }

    std::atomic<int> queued_executions(0);
    std::future<void> queued = pool.push([&](int) { ++queued_executions; });
    transition_armed.store(true);
    release_active_promise.set_value();
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return transition_entered; })) {
            return 3;
        }
    }

    std::thread stopper([&] { pool.stop(false); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return queue_extracted; })) {
            return 4;
        }
        release_transition = true;
    }
    hook_cv.notify_all();
    stopper.join();
    active.get();
    if (queued_executions.load() != 0) {
        FailWithExpectedDiagnostic();
    }

    bool broken_promise = false;
    try {
        queued.get();
    } catch (const std::future_error &error) {
        broken_promise = error.code() == std::make_error_code(std::future_errc::broken_promise);
    }
    return broken_promise ? 0 : 5;
}
