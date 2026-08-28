#include <ctpl_stl.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <thread>

namespace {

using namespace std::chrono;

struct CaptureState {
    explicit CaptureState(ctpl::thread_pool &owner)
        : pool(&owner), armed(false), entered(false) {}

    ctpl::thread_pool *pool;
    std::atomic<bool> armed;
    std::atomic<bool> entered;
};

struct ReentrantStopCapture {
    explicit ReentrantStopCapture(const std::shared_ptr<CaptureState> &capture_state)
        : state(capture_state) {}

    ~ReentrantStopCapture() {
        if (state && state->armed.load() && !state->entered.exchange(true)) {
            state->pool->stop(false);
        }
    }

    void operator()(int) const {}

    std::shared_ptr<CaptureState> state;
};

void FailWithExpectedDiagnostic() {
    std::cerr << "error=cancelled capture destruction retained lifecycle ownership\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

int main() {
    ctpl::thread_pool pool(0);
    std::shared_ptr<CaptureState> state(new CaptureState(pool));
    std::future<void> cancelled = pool.push(ReentrantStopCapture(state));
    static_cast<void>(cancelled);
    state->armed.store(true);

    std::promise<void> stop_finished_promise;
    std::future<void> stop_finished = stop_finished_promise.get_future();
    std::thread stopper([&] {
        pool.stop(false);
        stop_finished_promise.set_value();
    });

    if (stop_finished.wait_for(milliseconds(500)) != std::future_status::ready) {
        FailWithExpectedDiagnostic();
    }
    stopper.join();
    return 0;
}
