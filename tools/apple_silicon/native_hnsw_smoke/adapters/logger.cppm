export module infinity_core:logger;

import std;

export namespace infinity {

inline void LOG_CRITICAL(const std::string &message) {
    std::cerr << "[critical] " << message << '\n';
}

} // namespace infinity
