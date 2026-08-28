#include <cstdlib>

void CtplTestHookAfterResizeLifecycleLock(int thread_count);
void CtplTestHookAfterStopLifecycleLock(bool is_wait);
void CtplTestHookBeforeStopLifecycleLock(bool is_wait);

#define CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK(thread_count) ::CtplTestHookAfterResizeLifecycleLock(thread_count)
#define CTPL_STL_TEST_HOOK_AFTER_STOP_LIFECYCLE_LOCK(is_wait) ::CtplTestHookAfterStopLifecycleLock(is_wait)
#define CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_LOCK(is_wait) ::CtplTestHookBeforeStopLifecycleLock(is_wait)
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK
#undef CTPL_STL_TEST_HOOK_AFTER_STOP_LIFECYCLE_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_STOP_LIFECYCLE_LOCK

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace {

using namespace std::chrono;

std::mutex hook_mutex;
std::condition_variable hook_cv;
std::atomic<bool> hooks_armed(false);
bool resize_active = false;
bool release_resize = false;
bool stop_before_lock = false;
bool stop_after_lock = false;
bool lifecycle_overlap = false;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=resize overlapped another lifecycle operation\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookAfterResizeLifecycleLock(int) {
    if (!hooks_armed.load()) {
        return;
    }
    std::unique_lock<std::mutex> lock(hook_mutex);
    resize_active = true;
    hook_cv.notify_all();
    hook_cv.wait(lock, [] { return release_resize; });
}

void CtplTestHookAfterStopLifecycleLock(bool) {
    if (!hooks_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(hook_mutex);
    lifecycle_overlap = resize_active;
    stop_after_lock = true;
    hook_cv.notify_all();
}

void CtplTestHookBeforeStopLifecycleLock(bool) {
    if (!hooks_armed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(hook_mutex);
    stop_before_lock = true;
    hook_cv.notify_all();
}

int main() {
    ctpl::thread_pool pool(0);
    hooks_armed.store(true);
    std::thread resize_thread([&] { pool.resize(1); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return resize_active; })) {
            std::_Exit(2);
        }
    }

    std::thread stop_thread([&] { pool.stop(false); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return stop_before_lock; })) {
            std::_Exit(3);
        }
        hook_cv.wait_for(lock, milliseconds(200), [] { return stop_after_lock; });
        if (lifecycle_overlap) {
            FailWithExpectedDiagnostic();
        }
        release_resize = true;
        resize_active = false;
    }
    hook_cv.notify_all();
    resize_thread.join();
    stop_thread.join();
    return 0;
}
