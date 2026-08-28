export module infinity_core:infinity_exception;

import :infinity_type;

export namespace infinity {

class UnrecoverableException final : public std::runtime_error {
public:
    explicit UnrecoverableException(const std::string &message) : std::runtime_error(message) {}
};

[[noreturn]] inline void UnrecoverableError(const std::string &message) {
    throw UnrecoverableException(message);
}

} // namespace infinity
