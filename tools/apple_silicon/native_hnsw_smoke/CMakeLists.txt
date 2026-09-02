cmake_minimum_required(VERSION 4.0.3...4.4.2)

if(CMAKE_VERSION VERSION_LESS_EQUAL 4.2.5)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
elseif(CMAKE_VERSION VERSION_LESS 4.4)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
elseif(CMAKE_VERSION VERSION_LESS 4.5)
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
else()
    message(FATAL_ERROR "Unsupported CMake version for experimental import std: ${CMAKE_VERSION}")
endif()

project(infinity_native_hnsw_smoke LANGUAGES CXX)

if(NOT APPLE)
    message(FATAL_ERROR "This diagnostic target is for native Apple Silicon bring-up")
endif()

get_filename_component(CXX_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(CXX_COMPILER_PREFIX "${CXX_COMPILER_BIN_DIR}" DIRECTORY)
set(HOMEBREW_LIBCXX_MODULES_JSON "${CXX_COMPILER_PREFIX}/lib/c++/libc++.modules.json")
if(CMAKE_CXX_STDLIB_MODULES_JSON STREQUAL "libc++.modules.json" AND EXISTS "${HOMEBREW_LIBCXX_MODULES_JSON}")
    set(CMAKE_CXX_STDLIB_MODULES_JSON "${HOMEBREW_LIBCXX_MODULES_JSON}")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_MODULE_STD ON)

get_filename_component(INFINITY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." REALPATH)
set(ADAPTER_DIR "${CMAKE_CURRENT_LIST_DIR}/adapters")
set(
    INFINITY_NATIVE_CTPL_INCLUDE_DIR
    ""
    CACHE PATH
    "Directory containing the ctpl_stl.h used by Infinity's production HNSW thread pool"
)
if(NOT EXISTS "${INFINITY_NATIVE_CTPL_INCLUDE_DIR}/ctpl_stl.h")
    message(FATAL_ERROR "Set INFINITY_NATIVE_CTPL_INCLUDE_DIR to a directory containing ctpl_stl.h")
endif()
# The Batch4 reciprocal-neighbor pruning treatment failed the paired recall
# gate on Apple arm64, so the scalar reciprocal tail is the retained path. The
# option/def is kept only until the guarded Batch4 body is removed from
# hnsw_alg.cppm (see commit that reduces the engine sources); it must stay ON
# so the retained code path is exercised until then.
option(
    INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING
    "Disable Batch4 reciprocal-neighbor pruning for scalar-matched causal controls"
    ON
)
if(INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING AND
   NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    message(FATAL_ERROR "The scalar reciprocal-pruning control requires an arm64-only build")
endif()
# SearchLayer decomposition counters. A sampling profiler puts ~69% of active worker time
# in SearchLayer's own frame, but everything interesting is inlined into it, so that number
# does not say which part costs the time. Turning this on compiles per-thread counters into
# the traversal and makes the bridge emit them as infinity_instr_* keys. It perturbs TIME,
# so an instrumented binary must never be used for a timing measurement -- configure a
# SEPARATE build directory with it ON and keep the timing build clean.
option(
    INFINITY_HNSW_INSTRUMENT
    "Compile SearchLayer traversal counters (slower; for decomposition, never for timing)"
    OFF
)
set(
    INFINITY_NATIVE_HNSW_ALG_SOURCE
    "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/hnsw_alg.cppm"
    CACHE FILEPATH
    "HNSW algorithm module used by the native diagnostic"
)
set(
    INFINITY_NATIVE_HNSW_SIMD_SOURCE
    "${INFINITY_ROOT}/src/common/simd/hnsw_simd_func.cppm"
    CACHE FILEPATH
    "HNSW SIMD module used by the native diagnostic"
)
set(NATIVE_HNSW_MODULE_BASE_DIRS "${INFINITY_ROOT}")
if(NOT INFINITY_NATIVE_HNSW_ALG_SOURCE MATCHES "^${INFINITY_ROOT}/")
    get_filename_component(INFINITY_NATIVE_HNSW_ALG_BASE_DIR "${INFINITY_NATIVE_HNSW_ALG_SOURCE}" DIRECTORY)
    list(APPEND NATIVE_HNSW_MODULE_BASE_DIRS "${INFINITY_NATIVE_HNSW_ALG_BASE_DIR}")
endif()
if(NOT INFINITY_NATIVE_HNSW_SIMD_SOURCE MATCHES "^${INFINITY_ROOT}/")
    get_filename_component(INFINITY_NATIVE_HNSW_SIMD_BASE_DIR "${INFINITY_NATIVE_HNSW_SIMD_SOURCE}" DIRECTORY)
    list(APPEND NATIVE_HNSW_MODULE_BASE_DIRS "${INFINITY_NATIVE_HNSW_SIMD_BASE_DIR}")
endif()

add_library(infinity_native_hnsw STATIC)
target_sources(
    infinity_native_hnsw
    PUBLIC
    FILE_SET native_hnsw_modules TYPE CXX_MODULES
    BASE_DIRS ${NATIVE_HNSW_MODULE_BASE_DIRS}
    FILES
        "${ADAPTER_DIR}/primary.cppm"
        "${ADAPTER_DIR}/infinity_type.cppm"
        "${ADAPTER_DIR}/default_values.cppm"
        "${ADAPTER_DIR}/infinity_exception.cppm"
        "${ADAPTER_DIR}/sparse_util.cppm"
        "${ADAPTER_DIR}/local_file_handle.cppm"
        "${ADAPTER_DIR}/serialize.cppm"
        "${ADAPTER_DIR}/embedding_type.cppm"
        "${ADAPTER_DIR}/embedding_info.cppm"
        "${ADAPTER_DIR}/data_type.cppm"
        "${ADAPTER_DIR}/vec_store_type.cppm"
        "${ADAPTER_DIR}/dist_func_l2.cppm"
        "${ADAPTER_DIR}/dist_func_ip.cppm"
        "${ADAPTER_DIR}/dist_func_lsg_wrapper.cppm"
        "${ADAPTER_DIR}/multivector_result_handler.cppm"
        "${ADAPTER_DIR}/hnsw_lsg_builder.cppm"
        "${ADAPTER_DIR}/index_base.cppm"
        "${ADAPTER_DIR}/index_hnsw.cppm"
        "${ADAPTER_DIR}/config.cppm"
        "${ADAPTER_DIR}/infinity_context.cppm"
        "${ADAPTER_DIR}/logger.cppm"
        "${ADAPTER_DIR}/memindex_tracer.cppm"
        "${ADAPTER_DIR}/chunk_index_meta.cppm"
        "${ADAPTER_DIR}/base_memindex.cppm"
        "${ADAPTER_DIR}/buffer_handle.cppm"
        "${ADAPTER_DIR}/logical_type.cppm"
        "${ADAPTER_DIR}/column_def.cppm"
        "${ADAPTER_DIR}/third_party.cppm"
        "${ADAPTER_DIR}/internal_types.cppm"
        "${ADAPTER_DIR}/roaring_bitmap.cppm"
        "${INFINITY_ROOT}/src/common/utility/utility.cppm"
        "${INFINITY_ROOT}/src/common/utility/spinlock.cppm"
        "${INFINITY_ROOT}/src/common/simd/simd_common_tools.cppm"
        "${INFINITY_NATIVE_HNSW_SIMD_SOURCE}"
        "${INFINITY_ROOT}/src/storage/knn_index/result_handler.cppm"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/hnsw_common.cppm"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/data_store/data_store_util.cppm"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/data_store/plain_vec_store.cppm"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/data_store/graph_store.cppm"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/data_store/data_store.cppm"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/hnsw_bulk_build.cppm"
        "${INFINITY_NATIVE_HNSW_ALG_SOURCE}"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/hnsw_handler.cppm"
        "${INFINITY_ROOT}/src/storage/knn_index/knn_hnsw/hnsw_handler_core_impl.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/hnsw_handler_diagnostic_stubs.cpp"
)
# simd_prefetch.cpp used to live here to satisfy the SIMDPrefetch reference. It is gone:
# SIMDPrefetch is now an inline function in src/common/simd/simd_functions.h, so there is
# nothing to link. It had to be removed in the same change as the header -- keeping either
# out-of-line definition alongside the inline one is a redefinition error in this TU.
target_include_directories(
    infinity_native_hnsw
    PRIVATE
        "${INFINITY_ROOT}/src"
        "${INFINITY_ROOT}/src/common/simd"
        "${INFINITY_NATIVE_CTPL_INCLUDE_DIR}"
        "/opt/homebrew/opt/simde/include"
)
target_compile_definitions(infinity_native_hnsw PRIVATE SIMDE_ENABLE_NATIVE_ALIASES)
if(INFINITY_HNSW_INSTRUMENT)
    # PUBLIC: the bridge needs the same definition to see the exported dump entry points.
    target_compile_definitions(infinity_native_hnsw PUBLIC INFINITY_HNSW_INSTRUMENT)
endif()
if(INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING)
    target_compile_definitions(
        infinity_native_hnsw
        PUBLIC INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING
    )
endif()
target_compile_options(
    infinity_native_hnsw
    PRIVATE
        -O3
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-unused-private-field
        -Wno-reserved-module-identifier
)

add_executable(infinity_native_hnsw_smoke native_hnsw_smoke.cpp)
target_link_libraries(infinity_native_hnsw_smoke PRIVATE infinity_native_hnsw)
target_compile_options(
    infinity_native_hnsw_smoke
    PRIVATE
        -O3
        -Wall
        -Wextra
        -Wno-reserved-module-identifier
)

add_executable(infinity_native_simd_l2_smoke native_simd_l2_smoke.cpp)
target_link_libraries(infinity_native_simd_l2_smoke PRIVATE infinity_native_hnsw)
target_compile_options(
    infinity_native_simd_l2_smoke
    PRIVATE
        -O3
        -Wall
        -Wextra
        -Wno-reserved-module-identifier
)

add_executable(infinity_native_hnsw_bulk_build_smoke native_hnsw_bulk_build_smoke.cpp)
target_link_libraries(infinity_native_hnsw_bulk_build_smoke PRIVATE infinity_native_hnsw)
target_compile_options(
    infinity_native_hnsw_bulk_build_smoke
    PRIVATE
        -O3
        -Wall
        -Wextra
        -Wno-reserved-module-identifier
)

add_executable(infinity_native_hnsw_handler_smoke native_hnsw_handler_smoke.cpp)
target_link_libraries(infinity_native_hnsw_handler_smoke PRIVATE infinity_native_hnsw)
target_compile_options(
    infinity_native_hnsw_handler_smoke
    PRIVATE
        -O3
        -Wall
        -Wextra
        -Wno-reserved-module-identifier
)

# Hand-written NEON L2 kernel with a scalar reference correctness check; seed of
# the planned NEON optimization work. Wired so it is built and cannot rot.
add_executable(neon_l2_micro neon_l2_micro.cpp)
target_compile_options(
    neon_l2_micro
    PRIVATE
        -O3
        -Wall
        -Wextra
)

add_executable(
    infinity_hnsw_d0
    hnsw_d0_main.cpp
    hnsw_d0_runner.cpp
    hnsw_d0_audit.cpp
    hnsw_d0_query_benchmark.cpp
    infinity_hnsw_bridge.cpp
)
target_include_directories(infinity_hnsw_d0 PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(infinity_hnsw_d0 PRIVATE HNSW_D0_INFINITY)
target_link_libraries(infinity_hnsw_d0 PRIVATE infinity_native_hnsw)
target_compile_options(
    infinity_hnsw_d0
    PRIVATE
        -O3
        -Wall
        -Wextra
        -Wno-reserved-module-identifier
)

enable_testing()
add_test(NAME infinity_native_hnsw_smoke COMMAND infinity_native_hnsw_smoke)
add_test(NAME infinity_native_simd_l2_smoke COMMAND infinity_native_simd_l2_smoke)
add_test(NAME infinity_native_hnsw_bulk_build_smoke COMMAND infinity_native_hnsw_bulk_build_smoke)
add_test(NAME infinity_native_hnsw_handler_smoke COMMAND infinity_native_hnsw_handler_smoke)

add_executable(
    hnsw_d0_query_benchmark_test
    hnsw_d0_query_benchmark_test.cpp
    hnsw_d0_query_benchmark.cpp
)
target_include_directories(hnsw_d0_query_benchmark_test PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_options(hnsw_d0_query_benchmark_test PRIVATE -O3 -Wall -Wextra)
add_test(NAME hnsw_d0_query_benchmark_test COMMAND hnsw_d0_query_benchmark_test)
set_tests_properties(hnsw_d0_query_benchmark_test PROPERTIES TIMEOUT 60)

add_executable(
    hnsw_d0_audit_test
    hnsw_d0_audit_test.cpp
    hnsw_d0_audit.cpp
)
target_include_directories(hnsw_d0_audit_test PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_options(hnsw_d0_audit_test PRIVATE -O3 -Wall -Wextra)
add_test(NAME hnsw_d0_audit_test COMMAND hnsw_d0_audit_test)
set_tests_properties(hnsw_d0_audit_test PROPERTIES TIMEOUT 60)

option(INFINITY_NATIVE_SMOKE_WITH_FAISS "Build the matching stock FAISS diagnostic" OFF)
if(INFINITY_NATIVE_SMOKE_WITH_FAISS)
    set(
        FAISS_INCLUDE_DIR
        "/opt/homebrew/opt/faiss/include"
        CACHE PATH
        "FAISS source or install include directory containing faiss/IndexHNSW.h"
    )
    set(
        FAISS_LIBRARY
        "/opt/homebrew/opt/faiss/lib/libfaiss.dylib"
        CACHE FILEPATH
        "Source-built or installed FAISS library"
    )
    set(
        LIBOMP_INCLUDE_DIR
        "/opt/homebrew/opt/libomp/include"
        CACHE PATH
        "libomp include directory containing omp.h"
    )
    set(
        LIBOMP_LIBRARY
        "/opt/homebrew/opt/libomp/lib/libomp.dylib"
        CACHE FILEPATH
        "libomp library"
    )
    if(NOT EXISTS "${FAISS_INCLUDE_DIR}/faiss/IndexHNSW.h")
        message(FATAL_ERROR "FAISS_INCLUDE_DIR does not contain faiss/IndexHNSW.h: ${FAISS_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${FAISS_LIBRARY}")
        message(FATAL_ERROR "FAISS_LIBRARY does not exist: ${FAISS_LIBRARY}")
    endif()
    if(NOT EXISTS "${LIBOMP_INCLUDE_DIR}/omp.h")
        message(FATAL_ERROR "LIBOMP_INCLUDE_DIR does not contain omp.h: ${LIBOMP_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${LIBOMP_LIBRARY}")
        message(FATAL_ERROR "LIBOMP_LIBRARY does not exist: ${LIBOMP_LIBRARY}")
    endif()

    add_executable(faiss_native_hnsw_smoke faiss_hnsw_smoke.cpp)
    target_include_directories(faiss_native_hnsw_smoke PRIVATE "${FAISS_INCLUDE_DIR}" "${LIBOMP_INCLUDE_DIR}")
    target_link_libraries(faiss_native_hnsw_smoke PRIVATE "${FAISS_LIBRARY}" "${LIBOMP_LIBRARY}")
    target_compile_options(faiss_native_hnsw_smoke PRIVATE -O3 -Wall -Wextra)

    add_executable(
        hnsw_native_dev_compare
        hnsw_dev_compare.cpp
        hnsw_d0_audit.cpp
        hnsw_d0_query_benchmark.cpp
        infinity_hnsw_bridge.cpp
        faiss_hnsw_bridge.cpp
    )
    target_include_directories(
        hnsw_native_dev_compare
        PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FAISS_INCLUDE_DIR}"
            "${LIBOMP_INCLUDE_DIR}"
    )
    target_link_libraries(
        hnsw_native_dev_compare
        PRIVATE
            infinity_native_hnsw
            "${FAISS_LIBRARY}"
            "${LIBOMP_LIBRARY}"
    )
    target_compile_options(
        hnsw_native_dev_compare
        PRIVATE
            -O3
            -Wall
            -Wextra
        -Wno-reserved-module-identifier
    )

    add_executable(
        hnsw_d0_timing_boundary_test
        hnsw_d0_timing_boundary_test.cpp
        hnsw_d0_audit.cpp
        hnsw_d0_query_benchmark.cpp
        infinity_hnsw_bridge.cpp
        faiss_hnsw_bridge.cpp
    )
    target_include_directories(
        hnsw_d0_timing_boundary_test
        PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FAISS_INCLUDE_DIR}"
            "${LIBOMP_INCLUDE_DIR}"
    )
    target_compile_definitions(hnsw_d0_timing_boundary_test PRIVATE HNSW_D0_TIMING_PROBE_ENABLED)
    target_link_libraries(
        hnsw_d0_timing_boundary_test
        PRIVATE
            infinity_native_hnsw
            "${FAISS_LIBRARY}"
            "${LIBOMP_LIBRARY}"
    )
    target_compile_options(
        hnsw_d0_timing_boundary_test
        PRIVATE
            -O2
            -Wall
            -Wextra
            -Wpedantic
            -Wno-reserved-module-identifier
    )

    add_executable(
        faiss_hnsw_d0
        hnsw_d0_main.cpp
        hnsw_d0_runner.cpp
        hnsw_d0_audit.cpp
        hnsw_d0_query_benchmark.cpp
        faiss_hnsw_bridge.cpp
    )
    target_include_directories(
        faiss_hnsw_d0
        PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FAISS_INCLUDE_DIR}"
            "${LIBOMP_INCLUDE_DIR}"
    )
    target_compile_definitions(faiss_hnsw_d0 PRIVATE HNSW_D0_FAISS)
    target_link_libraries(faiss_hnsw_d0 PRIVATE "${FAISS_LIBRARY}" "${LIBOMP_LIBRARY}")
    target_compile_options(faiss_hnsw_d0 PRIVATE -O3 -Wall -Wextra)

    add_test(NAME faiss_native_hnsw_smoke COMMAND faiss_native_hnsw_smoke)
    add_test(NAME hnsw_d0_timing_boundary_test COMMAND hnsw_d0_timing_boundary_test)
    set_tests_properties(hnsw_d0_timing_boundary_test PROPERTIES TIMEOUT 180)
endif()

set_source_files_properties(hnsw_d0_audit.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
