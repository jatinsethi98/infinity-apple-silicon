#pragma once

#include "hnsw_dev_bridge.h"

enum class HnswD0Engine {
    kInfinity,
    kFaiss,
};

using HnswD0Bridge = int (*)(const float *, const HnswDevConfig *, HnswDevResult *);

int RunHnswD0(int argc, char **argv, HnswD0Engine engine, HnswD0Bridge bridge);
