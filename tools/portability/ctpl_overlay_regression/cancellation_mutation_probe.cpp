#include <cstdlib>

void CtplTestHookAfterPushValidation();
void CtplTestHookBeforeCancelQueueExtract();
void CtplTestHookBeforeStopLock();

#define CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION() ::CtplTestHookAfterPushValidation()
#define CTPL_STL_TEST_HOOK_BEFORE_CANCEL_QUEUE_EXTRACT() ::CtplTestHookBeforeCancelQueueExtract()
#define CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK() ::CtplTestHookBeforeStopLock()
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION
#undef CTPL_STL_TEST_HOOK_BEFORE_CANCEL_QUEUE_EXTRACT
#undef CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK

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
bool push_validated = false;
bool stop_started = false;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=cancellation extracted the queue outside the push linearization lock\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookAfterPushValidation() {
    std::unique_lock<std::mutex> lock(hook_mutex);
    push_validated = true;
    hook_cv.notify_all();
    hook_cv.wait(lock, [] { return false; });
}

void CtplTestHookBeforeCancelQueueExtract() {
    std::lock_guard<std::mutex> lock(hook_mutex);
    if (push_validated) {
        FailWithExpectedDiagnostic();
    }
}

void CtplTestHookBeforeStopLock() {
    std::lock_guard<std::mutex> lock(hook_mutex);
    stop_started = true;
    hook_cv.notify_all();
}

int main() {
    ctpl::thread_pool pool(0);
    std::thread producer([&] { static_cast<void>(pool.push([](int) { return 43; })); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return push_validated; })) {
            return 2;
        }
    }

    std::thread stopper([&] { pool.stop(false); });
    {
        std::unique_lock<std::mutex> lock(hook_mutex);
        if (!hook_cv.wait_for(lock, seconds(2), [] { return stop_started; })) {
            return 3;
        }
    }

    producer.join();
    stopper.join();
    return 4;
}
