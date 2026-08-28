#include <ctpl_stl.h>

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

using namespace std::chrono;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=worker resize was not rejected before lifecycle acquisition\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

int main() {
    ctpl::thread_pool pool(1);
    std::future<void> result = pool.push([&](int) { pool.resize(1); });
    if (result.wait_for(seconds(2)) != std::future_status::ready) {
        return 2;
    }
    try {
        result.get();
    } catch (const std::logic_error &) {
        return 0;
    }
    FailWithExpectedDiagnostic();
}
