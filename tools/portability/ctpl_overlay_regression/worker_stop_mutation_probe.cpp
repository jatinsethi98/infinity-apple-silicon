#include <cstdlib>

#include <ctpl_stl.h>

#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace std::chrono;

void FailWithExpectedDiagnostic() {
    std::cerr << "error=worker stop mutated pool state before rejection\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    const std::string mode(argv[1]);
    const bool is_wait = mode == "wait";
    if (!is_wait && mode != "cancel") {
        return 3;
    }

    ctpl::thread_pool pool(1);
    std::future<void> result = pool.push([&](int) { pool.stop(is_wait); });
    if (result.wait_for(seconds(2)) != std::future_status::ready) {
        return 4;
    }
    try {
        result.get();
    } catch (const std::logic_error &) {
        return 5;
    } catch (...) {
        FailWithExpectedDiagnostic();
    }
    FailWithExpectedDiagnostic();
}
