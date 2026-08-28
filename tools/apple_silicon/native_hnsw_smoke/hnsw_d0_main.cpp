#include "hnsw_d0_runner.h"

#if defined(HNSW_D0_INFINITY) == defined(HNSW_D0_FAISS)
#error "Define exactly one D0 engine"
#endif

int main(int argc, char **argv) {
#if defined(HNSW_D0_INFINITY)
    return RunHnswD0(argc, argv, HnswD0Engine::kInfinity, RunInfinityHnsw);
#else
    return RunHnswD0(argc, argv, HnswD0Engine::kFaiss, RunFaissHnsw);
#endif
}
