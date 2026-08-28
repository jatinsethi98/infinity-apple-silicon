#include <cstdlib>

void CtplTestHookAfterEmptyPop();
void CtplTestHookBeforePoolLock();
void CtplTestHookBeforeQueueCommit();

#define CTPL_STL_TEST_HOOK_AFTER_EMPTY_POP() ::CtplTestHookAfterEmptyPop()
#define CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK() ::CtplTestHookBeforePoolLock()
#define CTPL_STL_TEST_HOOK_BEFORE_QUEUE_COMMIT() ::CtplTestHookBeforeQueueCommit()
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_EMPTY_POP
#undef CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK
#undef CTPL_STL_TEST_HOOK_BEFORE_QUEUE_COMMIT

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
bool worker_waiting = false;
bool release_worker = false;
bool before_pool_lock = false;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=queue publication preceded condition-mutex acquisition\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookAfterEmptyPop() {
    std::unique_lock<std::mutex> lock(hook_mutex);
    if (worker_waiting) {
        return;
    }
    worker_waiting = true;
    hook_cv.notify_all();
    hook_cv.wait(lock, [] { return release_worker; });
}

void CtplTestHookBeforePoolLock() {
    std::lock_guard<std::mutex> lock(hook_mutex);
    before_pool_lock = true;
    hook_cv.notify_all();
}

void CtplTestHookBeforeQueueCommit() {
    std::lock_guard<std::mutex> lock(hook_mutex);
    if (!before_pool_lock) {
        FailWithExpectedDiagnostic();
    }
}

int main() {
    ctpl::thread_pool pool(1);
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return worker_waiting; })) {
            return 2;
        }
    }

    std::future<int> result;
    std::thread producer([&] { result = pool.push([](int) { return 7; }); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return before_pool_lock; })) {
            return 3;
        }
        release_worker = true;
    }
    hook_cv.notify_all();
    producer.join();
    if (result.wait_for(seconds(2)) != std::future_status::ready || result.get() != 7) {
        return 4;
    }
    return 0;
}
