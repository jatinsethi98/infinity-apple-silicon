#include <cstdlib>

#include <ctpl_stl.h>

#include <iostream>

namespace {

void FailWithExpectedDiagnostic() {
    std::cerr << "error=strong-push guarantee marker mutation survived\n";
    std::cerr.flush();
    std::_Exit(1);
}

} // namespace

int main() {
    if (!ctpl::thread_pool::push_has_strong_exception_guarantee) {
        FailWithExpectedDiagnostic();
    }
    return 2;
}
