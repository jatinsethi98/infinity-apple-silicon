module;

#include <ctpl_stl.h>

export module third_party;

export import std;

export namespace fmt {

template <typename... Args>
std::string format(std::string_view text, Args &&...) {
    return std::string(text);
}

} // namespace fmt

export namespace infinity::ctpl {

using ::ctpl::thread_pool;

} // namespace infinity::ctpl
