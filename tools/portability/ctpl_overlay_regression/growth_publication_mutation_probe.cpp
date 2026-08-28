#include <cstdlib>

void CtplTestHookBeforeGrowthThread(int worker_id);

#define CTPL_STL_TEST_HOOK_BEFORE_GROWTH_THREAD(worker_id) ::CtplTestHookBeforeGrowthThread(worker_id)
#include <ctpl_stl.h>
#undef CTPL_STL_TEST_HOOK_BEFORE_GROWTH_THREAD

#include <iostream>
#include <system_error>

namespace {

void FailWithExpectedDiagnostic() {
    std::cerr << "error=failed growth published partially constructed worker slots\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

void CtplTestHookBeforeGrowthThread(int worker_id) {
    if (worker_id == 2) {
        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again), "injected thread creation failure");
    }
}

int main() {
    ctpl::thread_pool pool(1);
    try {
        pool.resize(3);
    } catch (const std::system_error &) {
        if (pool.size() != 1) {
            FailWithExpectedDiagnostic();
        }
        return 2;
    }
    return 3;
}
