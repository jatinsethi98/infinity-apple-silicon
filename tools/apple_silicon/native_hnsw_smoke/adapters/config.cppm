export module infinity_core:config;

import :infinity_type;

export namespace infinity {

struct Config {
    i64 DenseIndexBuildingWorker() const { return 12; }
};

} // namespace infinity
