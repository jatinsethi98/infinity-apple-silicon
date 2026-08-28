#include <cstdlib>

void CtplTestHookAfterStopLifecycleLock(bool is_wait);
void CtplTestHookBeforeStopLifecycleRelease(bool is_wait);

#define CTPL_STL_TEST_HOOK_AFTER_STOP_LIFECYCLE_LOCK(is_wait) ::CtplTestHookAfterStopLifecycleLock(is_wait)
#define CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_RELEASE(is_wait) ::CtplTestHookBeforeStopLifecycleRelease(is_wait)
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_STOP_LIFECYCLE_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_RELEASE

#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

namespace {

using namespace std::chrono;

std::mutex hook_mutex;
std::condition_variable hook_cv;
bool draining_stop_active = false;
bool cancelling_stop_observed = false;
bool lifecycle_overlap = false;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=mixed stop operations entered lifecycle concurrently\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookAfterStopLifecycleLock(bool is_wait) {
    std::unique_lock<std::mutex> lock(hook_mutex);
    if (is_wait) {
        draining_stop_active = true;
        hook_cv.notify_all();
        return;
    }

    lifecycle_overlap = draining_stop_active;
    cancelling_stop_observed = true;
    hook_cv.notify_all();
    hook_cv.wait(lock, [] { return false; });
}

void CtplTestHookBeforeStopLifecycleRelease(bool is_wait) {
    if (!is_wait) {
        return;
    }
    std::lock_guard<std::mutex> lock(hook_mutex);
    draining_stop_active = false;
    hook_cv.notify_all();
}

int main() {
    ctpl::thread_pool pool(1);
    std::promise<void> blocker_entered_promise;
    std::future<void> blocker_entered = blocker_entered_promise.get_future();
    std::promise<void> release_blocker_promise;
    std::shared_future<void> release_blocker = release_blocker_promise.get_future().share();
    static_cast<void>(pool.push([&](int) {
        blocker_entered_promise.set_value();
        release_blocker.wait();
    }));
    if (blocker_entered.wait_for(seconds(2)) != std::future_status::ready) {
        std::_Exit(2);
    }

    std::thread draining_stop([&] { pool.stop(true); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return draining_stop_active; })) {
            std::_Exit(3);
        }
    }

    std::thread cancelling_stop([&] { pool.stop(false); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return cancelling_stop_observed; })) {
            std::_Exit(4);
        }
        if (lifecycle_overlap) {
            FailWithExpectedDiagnostic();
        }
    }
    std::_Exit(5);
}
