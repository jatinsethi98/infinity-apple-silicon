#include <cstdlib>

void CtplTestHookAfterGrowthGateWait(int worker_id, bool should_start);
void CtplTestHookBeforeGrowthGateWait(int worker_id);
void CtplTestHookBeforeGrowthThread(int worker_id);

#define CTPL_STL_TEST_HOOK_AFTER_GROWTH_GATE_WAIT(worker_id, should_start) ::CtplTestHookAfterGrowthGateWait(worker_id, should_start)
#define CTPL_STL_TEST_HOOK_BEFORE_GROWTH_GATE_WAIT(worker_id) ::CtplTestHookBeforeGrowthGateWait(worker_id)
#define CTPL_STL_TEST_HOOK_BEFORE_GROWTH_THREAD(worker_id) ::CtplTestHookBeforeGrowthThread(worker_id)
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_AFTER_GROWTH_GATE_WAIT
#undef CTPL_STL_TEST_HOOK_BEFORE_GROWTH_GATE_WAIT
#undef CTPL_STL_TEST_HOOK_BEFORE_GROWTH_THREAD

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <system_error>
#include <thread>

namespace {

using namespace std::chrono;

std::mutex hook_mutex;
std::condition_variable hook_cv;
bool staged_worker_waiting = false;
bool rollback_worker_blocked = false;
bool resize_finished = false;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=failed growth returned before its staged worker exited\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookAfterGrowthGateWait(int worker_id, bool should_start) {
    if (worker_id != 1 || should_start) {
        return;
    }
    std::unique_lock<std::mutex> lock(hook_mutex);
    rollback_worker_blocked = true;
    hook_cv.notify_all();
    hook_cv.wait(lock, [] { return false; });
}

void CtplTestHookBeforeGrowthGateWait(int worker_id) {
    if (worker_id != 1) {
        return;
    }
    std::lock_guard<std::mutex> lock(hook_mutex);
    staged_worker_waiting = true;
    hook_cv.notify_all();
}

void CtplTestHookBeforeGrowthThread(int worker_id) {
    if (worker_id != 2) {
        return;
    }
    std::unique_lock<std::mutex> lock(hook_mutex);
    if (!hook_cv.wait_for(lock, seconds(2), [] { return staged_worker_waiting; })) {
        throw std::runtime_error("staged worker did not reach its startup gate");
    }
    throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again), "injected thread creation failure");
}

int main() {
    ctpl::thread_pool pool(1);
    std::thread resize([&] {
        try {
            pool.resize(3);
        } catch (const std::system_error &) {
            std::lock_guard<std::mutex> lock(hook_mutex);
            resize_finished = true;
            hook_cv.notify_all();
        }
    });

    std::unique_lock<std::mutex> lock(hook_mutex);
    if (!hook_cv.wait_for(lock, seconds(2), [] { return rollback_worker_blocked; })) {
        return 2;
    }
    if (!hook_cv.wait_for(lock, seconds(2), [] { return resize_finished; })) {
        return 3;
    }
    FailWithExpectedDiagnostic();
}
