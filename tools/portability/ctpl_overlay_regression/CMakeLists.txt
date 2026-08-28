cmake_minimum_required(VERSION 3.20)

project(ctpl_overlay_regression LANGUAGES CXX)

set(CTPL_HEADER "" CACHE FILEPATH "Absolute path to the reviewed ctpl_stl.h")
if(NOT IS_ABSOLUTE "${CTPL_HEADER}" OR NOT EXISTS "${CTPL_HEADER}")
    message(FATAL_ERROR "CTPL_HEADER must name an existing absolute ctpl_stl.h path")
endif()

get_filename_component(CTPL_HEADER_NAME "${CTPL_HEADER}" NAME)
if(NOT CTPL_HEADER_NAME STREQUAL "ctpl_stl.h")
    message(FATAL_ERROR "CTPL_HEADER must name ctpl_stl.h")
endif()

file(READ "${CMAKE_CURRENT_LIST_DIR}/expected_header.sha256" EXPECTED_CTPL_SHA256)
string(STRIP "${EXPECTED_CTPL_SHA256}" EXPECTED_CTPL_SHA256)
file(SHA256 "${CTPL_HEADER}" ACTUAL_CTPL_SHA256)
if(NOT ACTUAL_CTPL_SHA256 STREQUAL EXPECTED_CTPL_SHA256)
    message(FATAL_ERROR "Unexpected ctpl_stl.h SHA-256: ${ACTUAL_CTPL_SHA256}")
endif()

get_filename_component(CTPL_INCLUDE_DIR "${CTPL_HEADER}" DIRECTORY)
find_package(Threads REQUIRED)

file(READ "${CTPL_HEADER}" CTPL_HEADER_CONTENTS)

set(PUBLICATION_BLOCK [==[#if defined(CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK)
            CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK();
#endif
            std::unique_lock<std::mutex> lock(this->mutex);
            if (this->isStop || this->isDone) {
                throw std::runtime_error("cannot push to a stopped thread_pool");
            }
#if defined(CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION)
            CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION();
#endif
            this->q.push(task.get());
            task.release();]==])
set(PUBLICATION_MUTATION [==[            this->q.push(task.get());
            task.release();
#if defined(CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK)
            CTPL_STL_TEST_HOOK_BEFORE_POOL_LOCK();
#endif
            std::unique_lock<std::mutex> lock(this->mutex);
            if (this->isStop || this->isDone) {
                throw std::runtime_error("cannot push to a stopped thread_pool");
            }
#if defined(CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION)
            CTPL_STL_TEST_HOOK_AFTER_PUSH_VALIDATION();
#endif]==])
string(REPLACE "${PUBLICATION_BLOCK}" "${PUBLICATION_MUTATION}" PUBLICATION_MUTANT "${CTPL_HEADER_CONTENTS}")
if(PUBLICATION_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate publication-before-pool-lock mutation")
endif()
string(FIND "${PUBLICATION_MUTANT}" "${PUBLICATION_BLOCK}" REMAINING_PUBLICATION_BLOCK)
if(NOT REMAINING_PUBLICATION_BLOCK EQUAL -1)
    message(FATAL_ERROR "Publication mutation did not replace every push overload")
endif()

set(JOIN_BLOCK [==[                    for (int i = nThreads; i < oldNThreads; ++i) {
#if defined(CTPL_STL_TEST_HOOK_BEFORE_RETIRED_THREAD_JOIN)
                        CTPL_STL_TEST_HOOK_BEFORE_RETIRED_THREAD_JOIN();
#endif
                        if (this->threads[i]->joinable())
                            this->threads[i]->join();
                    }]==])
set(DETACH_MUTATION [==[                    for (int i = oldNThreads - 1; i >= nThreads; --i) {
                        this->threads[i]->detach();
                    }]==])
string(REPLACE "${JOIN_BLOCK}" "${DETACH_MUTATION}" RESIZE_MUTANT "${CTPL_HEADER_CONTENTS}")
if(RESIZE_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate detach-and-shrink mutation")
endif()

set(CANCELLATION_LOCK_BLOCK [==[#if defined(CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK)
            CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK();
#endif
            std::unique_lock<std::mutex> lock(this->mutex);
            if (!isWait) {]==])
set(CANCELLATION_LOCK_MUTATION [==[#if defined(CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK)
            CTPL_STL_TEST_HOOK_BEFORE_STOP_LOCK();
#endif
            std::unique_lock<std::mutex> lock(this->mutex, std::defer_lock);
            if (!isWait) {]==])
string(
    REPLACE
        "${CANCELLATION_LOCK_BLOCK}"
        "${CANCELLATION_LOCK_MUTATION}"
        CANCELLATION_MUTANT
        "${CTPL_HEADER_CONTENTS}"
)
if(CANCELLATION_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate unlocked-cancellation mutation")
endif()

set(
    LIFECYCLE_LOCK_BLOCK
    "std::unique_lock<std::mutex> lifecycle_lock(this->lifecycle_mutex);"
)
set(
    LIFECYCLE_LOCK_MUTATION
    "std::unique_lock<std::mutex> lifecycle_lock(this->lifecycle_mutex, std::defer_lock);"
)
string(
    REPLACE
        "${LIFECYCLE_LOCK_BLOCK}"
        "${LIFECYCLE_LOCK_MUTATION}"
        LIFECYCLE_MUTANT
        "${CTPL_HEADER_CONTENTS}"
)
if(LIFECYCLE_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate unserialized-mixed-stop mutation")
endif()
string(FIND "${LIFECYCLE_MUTANT}" "${LIFECYCLE_LOCK_BLOCK}" REMAINING_LIFECYCLE_LOCK)
if(NOT REMAINING_LIFECYCLE_LOCK EQUAL -1)
    message(FATAL_ERROR "Lifecycle mutation did not replace the stop serialization lock")
endif()

set(WORKER_STOP_GUARD [==[            if (this->owns_current_thread()) {
                throw std::logic_error("thread_pool worker cannot stop its own pool");
            }
]==])
string(REPLACE "${WORKER_STOP_GUARD}" "" WORKER_STOP_MUTANT "${CTPL_HEADER_CONTENTS}")
if(WORKER_STOP_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate missing-worker-stop-guard mutation")
endif()

set(WORKER_RESIZE_GUARD [==[            if (this->owns_current_thread()) {
                throw std::logic_error("thread_pool worker cannot downsize its own pool");
            }
]==])
string(REPLACE "${WORKER_RESIZE_GUARD}" "" WORKER_RESIZE_MUTANT "${CTPL_HEADER_CONTENTS}")
if(WORKER_RESIZE_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate missing-worker-resize-guard mutation")
endif()

set(RESIZE_LIFECYCLE_LOCK_BLOCK [==[#if defined(CTPL_STL_TEST_HOOK_BEFORE_RESIZE_LIFECYCLE_LOCK)
            CTPL_STL_TEST_HOOK_BEFORE_RESIZE_LIFECYCLE_LOCK(nThreads);
#endif
            std::unique_lock<std::mutex> lifecycle_lock(this->lifecycle_mutex);
#if defined(CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK)
            CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK(nThreads);
#endif]==])
set(RESIZE_LIFECYCLE_LOCK_MUTATION [==[#if defined(CTPL_STL_TEST_HOOK_BEFORE_RESIZE_LIFECYCLE_LOCK)
            CTPL_STL_TEST_HOOK_BEFORE_RESIZE_LIFECYCLE_LOCK(nThreads);
#endif
            std::unique_lock<std::mutex> lifecycle_lock(this->lifecycle_mutex, std::defer_lock);
#if defined(CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK)
            CTPL_STL_TEST_HOOK_AFTER_RESIZE_LIFECYCLE_LOCK(nThreads);
#endif]==])
string(
    REPLACE
        "${RESIZE_LIFECYCLE_LOCK_BLOCK}"
        "${RESIZE_LIFECYCLE_LOCK_MUTATION}"
        RESIZE_LIFECYCLE_MUTANT
        "${CTPL_HEADER_CONTENTS}"
)
if(RESIZE_LIFECYCLE_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate unserialized-resize mutation")
endif()

set(REENTRANT_DESTRUCTION_BLOCK [==[            lifecycle_lock.unlock();
            this->delete_tasks(cancelled);
            this->delete_tasks(abandoned);]==])
set(REENTRANT_DESTRUCTION_MUTATION [==[            this->delete_tasks(cancelled);
            this->delete_tasks(abandoned);
            lifecycle_lock.unlock();]==])
string(
    REPLACE
        "${REENTRANT_DESTRUCTION_BLOCK}"
        "${REENTRANT_DESTRUCTION_MUTATION}"
        REENTRANT_DESTRUCTION_MUTANT
        "${CTPL_HEADER_CONTENTS}"
)
if(REENTRANT_DESTRUCTION_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate destruction-under-lifecycle-lock mutation")
endif()

set(WORKER_DEQUEUE_BLOCK [==[#if defined(CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK)
                    CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK(i);
#endif
                    std::function<void(int id)> * _f = nullptr;
                    {
                        std::unique_lock<std::mutex> lock(this->mutex);
                        if (!_flag) {
                            this->q.pop(_f);
                        }]==])
set(WORKER_DEQUEUE_MUTATION [==[                    std::function<void(int id)> * _f = nullptr;
                    if (!_flag) {
                        this->q.pop(_f);
                    }
#if defined(CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK)
                    CTPL_STL_TEST_HOOK_BEFORE_WORKER_DEQUEUE_LOCK(i);
#endif
                    {
                        std::unique_lock<std::mutex> lock(this->mutex);
                        if (!_f && !_flag) {
                            this->q.pop(_f);
                        }]==])
string(
    REPLACE
        "${WORKER_DEQUEUE_BLOCK}"
        "${WORKER_DEQUEUE_MUTATION}"
        WORKER_DEQUEUE_MUTANT
        "${CTPL_HEADER_CONTENTS}"
)
if(WORKER_DEQUEUE_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate dequeue-outside-cancellation-boundary mutation")
endif()

set(GROWTH_RESERVE_BLOCK [==[                        this->threads.reserve(static_cast<std::size_t>(nThreads));
                        this->flags.reserve(static_cast<std::size_t>(nThreads));]==])
set(GROWTH_PUBLICATION_MUTATION [==[                        this->threads.resize(static_cast<std::size_t>(nThreads));
                        this->flags.resize(static_cast<std::size_t>(nThreads));]==])
string(
    REPLACE
        "${GROWTH_RESERVE_BLOCK}"
        "${GROWTH_PUBLICATION_MUTATION}"
        GROWTH_PUBLICATION_MUTANT
        "${CTPL_HEADER_CONTENTS}"
)
if(GROWTH_PUBLICATION_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate eager-growth-publication mutation")
endif()

set(GROWTH_JOIN_BLOCK [==[                            if (staged_threads[i]->joinable())
                                staged_threads[i]->join();]==])
set(GROWTH_JOIN_MUTATION [==[                            if (staged_threads[i]->joinable())
                                staged_threads[i]->detach();]==])
string(
    REPLACE
        "${GROWTH_JOIN_BLOCK}"
        "${GROWTH_JOIN_MUTATION}"
        GROWTH_JOIN_MUTANT
        "${CTPL_HEADER_CONTENTS}"
)
if(GROWTH_JOIN_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate missing-growth-rollback-join mutation")
endif()

set(MARKER_BLOCK
    "static constexpr bool push_has_strong_exception_guarantee = true;"
)
set(MARKER_MUTATION
    "static constexpr bool push_has_strong_exception_guarantee = false;"
)
string(REPLACE "${MARKER_BLOCK}" "${MARKER_MUTATION}" MARKER_MUTANT "${CTPL_HEADER_CONTENTS}")
if(MARKER_MUTANT STREQUAL CTPL_HEADER_CONTENTS)
    message(FATAL_ERROR "Could not generate strong-guarantee-marker mutation")
endif()

set(MUTATION_ROOT "${CMAKE_CURRENT_BINARY_DIR}/mutations")
set(PUBLICATION_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/publication-before-pool-lock")
set(RESIZE_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/detach-and-shrink")
set(CANCELLATION_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/unlocked-cancellation")
set(LIFECYCLE_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/unserialized-mixed-stop")
set(WORKER_STOP_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/missing-worker-stop-guard")
set(WORKER_RESIZE_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/missing-worker-resize-guard")
set(RESIZE_LIFECYCLE_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/unserialized-resize")
set(REENTRANT_DESTRUCTION_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/destruction-under-lifecycle-lock")
set(WORKER_DEQUEUE_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/dequeue-outside-cancellation-boundary")
set(GROWTH_PUBLICATION_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/eager-growth-publication")
set(GROWTH_JOIN_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/missing-growth-rollback-join")
set(MARKER_MUTANT_INCLUDE_DIR "${MUTATION_ROOT}/false-strong-guarantee-marker")
file(
    MAKE_DIRECTORY
        "${PUBLICATION_MUTANT_INCLUDE_DIR}"
        "${RESIZE_MUTANT_INCLUDE_DIR}"
        "${CANCELLATION_MUTANT_INCLUDE_DIR}"
        "${LIFECYCLE_MUTANT_INCLUDE_DIR}"
        "${WORKER_STOP_MUTANT_INCLUDE_DIR}"
        "${WORKER_RESIZE_MUTANT_INCLUDE_DIR}"
        "${RESIZE_LIFECYCLE_MUTANT_INCLUDE_DIR}"
        "${REENTRANT_DESTRUCTION_MUTANT_INCLUDE_DIR}"
        "${WORKER_DEQUEUE_MUTANT_INCLUDE_DIR}"
        "${GROWTH_PUBLICATION_MUTANT_INCLUDE_DIR}"
        "${GROWTH_JOIN_MUTANT_INCLUDE_DIR}"
        "${MARKER_MUTANT_INCLUDE_DIR}"
)
file(WRITE "${PUBLICATION_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${PUBLICATION_MUTANT}")
file(WRITE "${RESIZE_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${RESIZE_MUTANT}")
file(WRITE "${CANCELLATION_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${CANCELLATION_MUTANT}")
file(WRITE "${LIFECYCLE_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${LIFECYCLE_MUTANT}")
file(WRITE "${WORKER_STOP_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${WORKER_STOP_MUTANT}")
file(WRITE "${WORKER_RESIZE_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${WORKER_RESIZE_MUTANT}")
file(WRITE "${RESIZE_LIFECYCLE_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${RESIZE_LIFECYCLE_MUTANT}")
file(WRITE "${REENTRANT_DESTRUCTION_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${REENTRANT_DESTRUCTION_MUTANT}")
file(WRITE "${WORKER_DEQUEUE_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${WORKER_DEQUEUE_MUTANT}")
file(WRITE "${GROWTH_PUBLICATION_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${GROWTH_PUBLICATION_MUTANT}")
file(WRITE "${GROWTH_JOIN_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${GROWTH_JOIN_MUTANT}")
file(WRITE "${MARKER_MUTANT_INCLUDE_DIR}/ctpl_stl.h" "${MARKER_MUTANT}")

enable_testing()

function(configure_probe_target TARGET_NAME SOURCE_NAME INCLUDE_DIR CXX_STANDARD)
    add_executable("${TARGET_NAME}" "${SOURCE_NAME}")
    target_include_directories("${TARGET_NAME}" PRIVATE "${INCLUDE_DIR}")
    target_link_libraries("${TARGET_NAME}" PRIVATE Threads::Threads)
    set_target_properties(
        "${TARGET_NAME}"
        PROPERTIES
            CXX_STANDARD "${CXX_STANDARD}"
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
    )
endfunction()

function(add_expected_failure_test TEST_NAME TARGET_NAME EXPECTED_DIAGNOSTIC)
    set(TEST_COMMAND
        "${CMAKE_COMMAND}"
        "-DTEST_PROGRAM=$<TARGET_FILE:${TARGET_NAME}>"
        "-DEXPECTED_DIAGNOSTIC=${EXPECTED_DIAGNOSTIC}"
    )
    if(ARGC GREATER 3)
        list(APPEND TEST_COMMAND "-DTEST_ARGUMENT=${ARGV3}")
    endif()
    list(APPEND TEST_COMMAND -P "${CMAKE_CURRENT_LIST_DIR}/expect_exact_failure.cmake")
    add_test(NAME "${TEST_NAME}" COMMAND ${TEST_COMMAND})
    set_tests_properties("${TEST_NAME}" PROPERTIES TIMEOUT 30)
endfunction()

foreach(CXX_STANDARD IN ITEMS 11 23)
    set(TARGET_NAME "ctpl_overlay_regression_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${TARGET_NAME}"
        ctpl_overlay_regression.cpp
        "${CTPL_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_test(NAME "${TARGET_NAME}" COMMAND "${TARGET_NAME}")
    set_tests_properties("${TARGET_NAME}" PROPERTIES TIMEOUT 30)

    set(PUBLICATION_TARGET "ctpl_publication_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${PUBLICATION_TARGET}"
        publication_mutation_probe.cpp
        "${PUBLICATION_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${PUBLICATION_TARGET}"
        "${PUBLICATION_TARGET}"
        "error=queue publication preceded condition-mutex acquisition"
    )

    set(RESIZE_TARGET "ctpl_resize_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${RESIZE_TARGET}"
        resize_mutation_probe.cpp
        "${RESIZE_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${RESIZE_TARGET}"
        "${RESIZE_TARGET}"
        "error=resize returned while a retiring worker was active"
    )

    set(CANCELLATION_TARGET "ctpl_cancellation_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${CANCELLATION_TARGET}"
        cancellation_mutation_probe.cpp
        "${CANCELLATION_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${CANCELLATION_TARGET}"
        "${CANCELLATION_TARGET}"
        "error=cancellation extracted the queue outside the push linearization lock"
    )

    set(LIFECYCLE_TARGET "ctpl_lifecycle_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${LIFECYCLE_TARGET}"
        lifecycle_mutation_probe.cpp
        "${LIFECYCLE_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${LIFECYCLE_TARGET}"
        "${LIFECYCLE_TARGET}"
        "error=mixed stop operations entered lifecycle concurrently"
    )

    set(WORKER_STOP_TARGET "ctpl_worker_stop_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${WORKER_STOP_TARGET}"
        worker_stop_mutation_probe.cpp
        "${WORKER_STOP_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${WORKER_STOP_TARGET}_cancel"
        "${WORKER_STOP_TARGET}"
        "error=worker stop mutated pool state before rejection"
        cancel
    )
    add_expected_failure_test(
        "${WORKER_STOP_TARGET}_wait"
        "${WORKER_STOP_TARGET}"
        "error=worker stop mutated pool state before rejection"
        wait
    )

    set(WORKER_RESIZE_TARGET "ctpl_worker_resize_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${WORKER_RESIZE_TARGET}"
        worker_resize_mutation_probe.cpp
        "${WORKER_RESIZE_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${WORKER_RESIZE_TARGET}"
        "${WORKER_RESIZE_TARGET}"
        "error=worker resize was not rejected before lifecycle acquisition"
    )

    set(RESIZE_LIFECYCLE_TARGET "ctpl_resize_lifecycle_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${RESIZE_LIFECYCLE_TARGET}"
        resize_lifecycle_mutation_probe.cpp
        "${RESIZE_LIFECYCLE_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${RESIZE_LIFECYCLE_TARGET}"
        "${RESIZE_LIFECYCLE_TARGET}"
        "error=resize overlapped another lifecycle operation"
    )

    set(REENTRANT_DESTRUCTION_TARGET "ctpl_reentrant_destruction_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${REENTRANT_DESTRUCTION_TARGET}"
        reentrant_destruction_mutation_probe.cpp
        "${REENTRANT_DESTRUCTION_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${REENTRANT_DESTRUCTION_TARGET}"
        "${REENTRANT_DESTRUCTION_TARGET}"
        "error=cancelled capture destruction retained lifecycle ownership"
    )

    set(WORKER_DEQUEUE_TARGET "ctpl_worker_cancellation_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${WORKER_DEQUEUE_TARGET}"
        worker_cancellation_mutation_probe.cpp
        "${WORKER_DEQUEUE_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${WORKER_DEQUEUE_TARGET}"
        "${WORKER_DEQUEUE_TARGET}"
        "error=worker dequeued outside the cancellation boundary"
    )

    set(GROWTH_PUBLICATION_TARGET "ctpl_growth_publication_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${GROWTH_PUBLICATION_TARGET}"
        growth_publication_mutation_probe.cpp
        "${GROWTH_PUBLICATION_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${GROWTH_PUBLICATION_TARGET}"
        "${GROWTH_PUBLICATION_TARGET}"
        "error=failed growth published partially constructed worker slots"
    )

    set(GROWTH_JOIN_TARGET "ctpl_growth_join_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${GROWTH_JOIN_TARGET}"
        growth_join_mutation_probe.cpp
        "${GROWTH_JOIN_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${GROWTH_JOIN_TARGET}"
        "${GROWTH_JOIN_TARGET}"
        "error=failed growth returned before its staged worker exited"
    )

    set(MARKER_TARGET "ctpl_marker_mutation_cxx${CXX_STANDARD}")
    configure_probe_target(
        "${MARKER_TARGET}"
        marker_mutation_probe.cpp
        "${MARKER_MUTANT_INCLUDE_DIR}"
        "${CXX_STANDARD}"
    )
    add_expected_failure_test(
        "${MARKER_TARGET}"
        "${MARKER_TARGET}"
        "error=strong-push guarantee marker mutation survived"
    )
endforeach()
