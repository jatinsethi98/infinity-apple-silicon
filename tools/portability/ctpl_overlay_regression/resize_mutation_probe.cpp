#include <cstdlib>

void CtplTestHookAfterRetireRequest();

#define CTPL_STL_TEST_HOOK_AFTER_RETIRE_REQUEST() ::CtplTestHookAfterRetireRequest()
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_RETIRE_REQUEST

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
bool retire_requested = false;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=resize returned while a retiring worker was active\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookAfterRetireRequest() {
    std::lock_guard<std::mutex> lock(hook_mutex);
    retire_requested = true;
    hook_cv.notify_all();
}

int main() {
    ctpl::thread_pool pool(1);
    std::promise<void> task_entered_promise;
    std::future<void> task_entered = task_entered_promise.get_future();
    std::promise<void> release_task_promise;
    std::shared_future<void> release_task = release_task_promise.get_future().share();
    std::future<void> task = pool.push([&](int) {
        task_entered_promise.set_value();
        release_task.wait();
    });
    if (task_entered.wait_for(seconds(2)) != std::future_status::ready) {
        return 2;
    }

    std::future<void> resize = std::async(std::launch::async, [&pool] { pool.resize(0); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return retire_requested; })) {
            release_task_promise.set_value();
            return 3;
        }
    }

    if (resize.wait_for(seconds(2)) == std::future_status::ready) {
        FailWithExpectedDiagnostic();
    }

    release_task_promise.set_value();
    if (task.wait_for(seconds(2)) != std::future_status::ready) {
        return 4;
    }
    task.get();
    if (resize.wait_for(seconds(2)) != std::future_status::ready) {
        return 5;
    }
    resize.get();
    return 0;
}
