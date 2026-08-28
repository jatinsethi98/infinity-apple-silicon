set(INFINITY_GENERATED_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(INFINITY_COMPILATION_CONFIG_MODULE
        "${INFINITY_GENERATED_MODULE_DIR}/compilation_config.cppm")
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/compilation_config.cppm.in"
        "${INFINITY_COMPILATION_CONFIG_MODULE}"
        @ONLY
)

### Infinity

file(GLOB_RECURSE
        main_cpp
        CONFIGURE_DEPENDS
        bin/infinity_main.cpp
)

file(GLOB_RECURSE
        main_cppm
        CONFIGURE_DEPENDS
        bin/*.cppm
)
list(REMOVE_ITEM main_cppm
        "${CMAKE_CURRENT_SOURCE_DIR}/bin/compilation_config.cppm")

file(GLOB_RECURSE
        infinity_cpp
        CONFIGURE_DEPENDS
        main/*.cpp
)
list(FILTER infinity_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        infinity_impl_cpp
        CONFIGURE_DEPENDS
        main/*_impl.cpp
)

file(GLOB_RECURSE
        infinity_cppm
        CONFIGURE_DEPENDS
        main/*.cppm
)

file(GLOB_RECURSE
        network_cpp
        CONFIGURE_DEPENDS
        network/*.cpp
)
list(FILTER network_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        network_impl_cpp
        CONFIGURE_DEPENDS
        network/*_impl.cpp
)

file(GLOB_RECURSE
        network_cppm
        CONFIGURE_DEPENDS
        network/*.cppm
)

file(GLOB_RECURSE
        function_cpp
        CONFIGURE_DEPENDS
        function/*.cpp
)
list(FILTER function_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        function_impl_cpp
        CONFIGURE_DEPENDS
        function/*_impl.cpp
)

file(GLOB_RECURSE
        function_cppm
        CONFIGURE_DEPENDS
        function/*.cppm
)

file(GLOB_RECURSE
        expression_cpp
        CONFIGURE_DEPENDS
        expression/*.cpp
)
list(FILTER expression_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        expression_impl_cpp
        CONFIGURE_DEPENDS
        expression/*_impl.cpp
)

file(GLOB_RECURSE
        expression_cppm
        CONFIGURE_DEPENDS
        expression/*.cppm
)

file(GLOB_RECURSE
        executor_cpp
        CONFIGURE_DEPENDS
        executor/*.cpp
)
list(FILTER executor_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        executor_impl_cpp
        CONFIGURE_DEPENDS
        executor/*_impl.cpp
)

file(GLOB_RECURSE
        executor_cppm
        CONFIGURE_DEPENDS
        executor/*.cppm
)

file(GLOB_RECURSE
        metrics_cpp
        CONFIGURE_DEPENDS
        metrics/*.cpp
)
list(FILTER metrics_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        metrics_impl_cpp
        CONFIGURE_DEPENDS
        metrics/*_impl.cpp
)

file(GLOB_RECURSE
        metrics_cppm
        CONFIGURE_DEPENDS
        metrics/*.cppm
)

file(GLOB_RECURSE
        planner_cpp
        CONFIGURE_DEPENDS
        planner/*.cpp
)
list(FILTER planner_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        planner_impl_cpp
        CONFIGURE_DEPENDS
        planner/*_impl.cpp
)

file(GLOB_RECURSE
        planner_cppm
        CONFIGURE_DEPENDS
        planner/*.cppm
)

file(GLOB_RECURSE
        scheduler_cpp
        CONFIGURE_DEPENDS
        scheduler/*.cpp
)
list(FILTER scheduler_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        scheduler_impl_cpp
        CONFIGURE_DEPENDS
        scheduler/*_impl.cpp
)

file(GLOB_RECURSE
        scheduler_cppm
        CONFIGURE_DEPENDS
        scheduler/*.cppm
)

file(GLOB_RECURSE
        storage_cpp
        CONFIGURE_DEPENDS
        storage/*.cpp
)
list(FILTER storage_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        storage_impl_cpp
        CONFIGURE_DEPENDS
        storage/*_impl.cpp
)

file(GLOB_RECURSE
        storage_cppm
        CONFIGURE_DEPENDS
        storage/*.cppm
)

file(GLOB_RECURSE
        common_cpp
        CONFIGURE_DEPENDS
        common/*.cpp
)
list(FILTER common_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        common_impl_cpp
        CONFIGURE_DEPENDS
        common/*_impl.cpp
)

file(GLOB_RECURSE
        common_cppm
        CONFIGURE_DEPENDS
        common/*.cppm
)

file(GLOB_RECURSE
        parser_cppm
        CONFIGURE_DEPENDS
        parser/*.cppm
)

#file(GLOB_RECURSE
#        parser_impl_cpp
#        CONFIGURE_DEPENDS
#        parser/*_impl.cpp
#)

file(GLOB_RECURSE
        admin_cpp
        CONFIGURE_DEPENDS
        admin/*.cpp
)
list(FILTER admin_cpp EXCLUDE REGEX .*_impl.cpp)

file(GLOB_RECURSE
        admin_impl_cpp
        CONFIGURE_DEPENDS
        admin/*_impl.cpp
)

file(GLOB_RECURSE
        admin_cppm
        CONFIGURE_DEPENDS
        admin/*.cppm
)

add_library(infinity_core
        ${infinity_cpp}
        ${planner_cpp}
        ${scheduler_cpp}
        ${storage_cpp}
        ${function_cpp}
        ${expression_cpp}
        ${executor_cpp}
        ${metrics_cpp}
        ${common_cpp}
        ${admin_cpp}
        ${network_cpp}
)

option(
        INFINITY_ENABLE_HNSW_LVQ_CAPTURE
        "Compile capture-only observers into the production HNSW LVQ path"
        OFF
)
if (INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "The HNSW LVQ production capture is Apple arm64-only")
    endif ()
    foreach (capture_source
            storage/knn_index/knn_hnsw/hnsw_lvq_capture.cppm
            storage/knn_index/knn_hnsw/hnsw_alg.cppm
            storage/knn_index/knn_hnsw/dist_func_l2.cppm)
        set_property(
                SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${capture_source}"
                APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_HNSW_LVQ_CAPTURE
        )
    endforeach ()
endif ()

option(
        INFINITY_ENABLE_APPLE_HNSW_COMPACT_VERTEX_LOCKS
        "Use compact per-vertex HNSW read-write locks on Apple arm64"
        OFF
)
if (INFINITY_ENABLE_APPLE_HNSW_COMPACT_VERTEX_LOCKS)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "Compact HNSW vertex locks are Apple arm64-only")
    endif ()
    set_property(
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/storage/knn_index/knn_hnsw/data_store/data_store.cppm"
            APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_APPLE_HNSW_COMPACT_VERTEX_LOCKS
    )
endif ()

option(
        INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL
        "Disable Apple HNSW Batch4 construction traversal for causal controls"
        OFF
)
if (INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "The HNSW Batch4 traversal control is Apple arm64-only")
    endif ()
    set_property(
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/storage/knn_index/knn_hnsw/hnsw_alg.cppm"
            APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL
    )
endif ()

option(
        INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL
        "Enable exact threshold-based lane dropping in Apple HNSW Batch4 construction traversal"
        OFF
)
if (INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "Thresholded HNSW Batch4 traversal is Apple arm64-only")
    endif ()
    if (INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL)
        message(FATAL_ERROR "Thresholded HNSW Batch4 traversal requires Batch4 traversal")
    endif ()
    set_property(
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/storage/knn_index/knn_hnsw/hnsw_alg.cppm"
            APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_APPLE_HNSW_THRESHOLD_BATCH4_TRAVERSAL
    )
endif ()

option(
        INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE
        "Compile an armable execution witness for thresholded HNSW Batch4 traversal"
        OFF
)
if (INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "Thresholded HNSW Batch4 execution evidence is Apple arm64-only")
    endif ()
    if (INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL)
        message(FATAL_ERROR "Thresholded HNSW Batch4 execution evidence requires Batch4 traversal")
    endif ()
    foreach (threshold_batch4_evidence_source
            storage/knn_index/knn_hnsw/hnsw_common.cppm
            storage/knn_index/knn_hnsw/hnsw_alg.cppm)
        set_property(
                SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${threshold_batch4_evidence_source}"
                APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE
        )
    endforeach ()
endif ()

set(INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING_DEFAULT OFF)
if (APPLE AND ARM64)
    # The Batch4 treatment failed the paired recall gate.
    set(INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING_DEFAULT ON)
endif ()
option(
        INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING
        "Disable Apple HNSW Batch4 reciprocal-neighbor pruning for causal controls"
        ${INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING_DEFAULT}
)
if (INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "The HNSW Batch4 reciprocal-pruning control is Apple arm64-only")
    endif ()
    set_property(
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/storage/knn_index/knn_hnsw/hnsw_alg.cppm"
            APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING
    )
endif ()

option(
        INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL
        "Enable the experimental exact incremental reciprocal-list update on Apple arm64"
        OFF
)
option(
        INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW
        "Compute experimental incremental reciprocal updates in shadow mode while publishing the reference result"
        OFF
)
option(
        INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS
        "Add counters and a runtime control to the incremental reciprocal treatment"
        OFF
)
option(
        INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE
        "Compile an armable execution witness for the incremental reciprocal exactness gate"
        OFF
)
if (INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "Incremental HNSW reciprocal execution evidence is Apple arm64-only")
    endif ()
    if (INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        message(FATAL_ERROR "Incremental HNSW reciprocal execution evidence is incompatible with shadow mode")
    endif ()
    foreach (incremental_reciprocal_evidence_source
            storage/knn_index/knn_hnsw/hnsw_common.cppm
            storage/knn_index/knn_hnsw/hnsw_alg.cppm)
        set_property(
                SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${incremental_reciprocal_evidence_source}"
                APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE
        )
    endforeach ()
endif ()
if (INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL OR INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "The incremental HNSW reciprocal-update experiment is Apple arm64-only")
    endif ()
    if (INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL AND INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW)
        message(FATAL_ERROR "The incremental HNSW reciprocal treatment and shadow verifier are mutually exclusive")
    endif ()
    if (INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS AND
            NOT INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL)
        message(FATAL_ERROR "Incremental HNSW reciprocal diagnostics require the treatment option")
    endif ()
    if (NOT INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING)
        message(FATAL_ERROR "The incremental HNSW reciprocal experiment requires scalar reciprocal pruning")
    endif ()
    if (INFINITY_ENABLE_HNSW_LVQ_CAPTURE)
        message(FATAL_ERROR "The incremental HNSW reciprocal experiment is incompatible with LVQ distance-call capture")
    endif ()

    foreach (incremental_reciprocal_source
            storage/knn_index/knn_hnsw/hnsw_common.cppm
            storage/knn_index/knn_hnsw/hnsw_alg.cppm)
        if (INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL)
            set_property(
                    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${incremental_reciprocal_source}"
                    APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_APPLE_HNSW_INCREMENTAL_RECIPROCAL
            )
        else ()
            set_property(
                    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${incremental_reciprocal_source}"
                    APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_SHADOW
            )
        endif ()
        if (INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
            set_property(
                    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${incremental_reciprocal_source}"
                    APPEND PROPERTY COMPILE_DEFINITIONS INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS
            )
        endif ()
    endforeach ()
elseif (INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_DIAGNOSTICS)
    message(FATAL_ERROR "Incremental HNSW reciprocal diagnostics require the treatment option")
endif ()

target_sources(infinity_core
        PUBLIC
        FILE_SET core_cxx_modules TYPE CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES
        ${main_cppm}
        ${infinity_cppm}
        ${common_cppm}
        ${planner_cppm}
        ${scheduler_cppm}
        ${storage_cppm}
        ${parser_cppm}
        ${function_cppm}
        ${expression_cppm}
        ${executor_cppm}
        ${metrics_cppm}
        ${admin_cppm}
        ${network_cppm}

        # for cmake bug
        ${infinity_impl_cpp}
        ${planner_impl_cpp}
        ${scheduler_impl_cpp}
        ${storage_impl_cpp}
        ${function_impl_cpp}
        ${expression_impl_cpp}
        ${executor_impl_cpp}
        ${metrics_impl_cpp}
        ${common_impl_cpp}
        ${admin_impl_cpp}
        ${network_impl_cpp}
        #        ${parser_impl_cpp}

        FILE_SET generated_cxx_modules TYPE CXX_MODULES
        BASE_DIRS "${INFINITY_GENERATED_MODULE_DIR}"
        FILES "${INFINITY_COMPILATION_CONFIG_MODULE}"
)

#if (ARCH_AARCH64)
#    set(SSE2NEON_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/sse2neon)
#
#    target_include_directories(
#            sse2neon
#            SYSTEM
#            INTERFACE ${SSE2NEON_ROOT}
#    )
#endif ()

target_include_directories(infinity_core PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(infinity_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(infinity_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/parser")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/zsv/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/tomlplusplus")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/newpfor")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cppjieba/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps/limonp/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/ijma/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/pgm/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/mlas")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/xor_singleheader/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/base64/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/opencc")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/minio-cpp/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/curlpp/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/curl/include")
target_include_directories(infinity_core PUBLIC "${CMAKE_SOURCE_DIR}/third_party/darts/")

add_executable(infinity
        ${main_cpp}
)

#target_sources(infinity
#        PUBLIC
#        FILE_SET core_cxx_modules TYPE CXX_MODULES FILES
#        ${network_cppm}
#)

### Parser (built after infinity_core so it can access modules)
add_subdirectory(parser)

add_dependencies(infinity miniocpp curlpp_static libcurl_static)

set(INFINITY_PLATFORM_LIBRARIES
        Threads::Threads
        ${CMAKE_DL_LIBS}
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND INFINITY_PLATFORM_LIBRARIES
            resolv.a
            atomic.a
            libstdc++exp.a
    )
endif ()

target_link_libraries(infinity
        PUBLIC
        infinity_core
        sql_parser
        onnxruntime_mlas
        zsv_parser
        newpfor
        ${FASTPFOR_STATIC_LIB}
        jma
        opencc
        #        $<IF:$<BOOL:${ARM64}>,opencc,>
        ${INFINITY_PLATFORM_LIBRARIES}
        Boost::asio
        thrift::thrift
        # thriftnb::thriftnb
        libevent::core
        oatpp::oatpp
        Parquet::parquet_static
        Arrow::arrow_static
        ${JEMALLOC_STATIC_LIB}
        miniocpp::miniocpp
        pugixml::static
        curlpp_static
        unofficial::inih::libinih
        unofficial::inih::inireader
        libcurl_static
        OpenSSL::SSL
        OpenSSL::Crypto
        re2::re2
        absl::any absl::log absl::base absl::bits
        PCRE2::8BIT
        RocksDB::rocksdb
        simdjson::simdjson
        nlohmann_json::nlohmann_json
        CLI11::CLI11
        magic_enum::magic_enum
        roaring::roaring
        spdlog::spdlog
        fmt::fmt
        # turbo::base64
        ZLIB::ZLIB
)
target_link_directories(infinity PUBLIC "${CMAKE_BINARY_DIR}/lib")
target_link_directories(infinity PUBLIC "${CMAKE_BINARY_DIR}/third_party/minio-cpp/")
target_link_directories(infinity PUBLIC "${CMAKE_BINARY_DIR}/third_party/curlpp/")
target_link_directories(infinity PUBLIC "${CMAKE_BINARY_DIR}/third_party/curl/")
target_link_directories(infinity PUBLIC "${CMAKE_BINARY_DIR}/third_party/")
target_link_directories(infinity PUBLIC "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib")

target_include_directories(infinity PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

option(
        INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION
        "Build the Apple Silicon D0 benchmark against production infinity_core"
        OFF
)

if (INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "Production Apple HNSW D0 requires native Apple arm64")
    endif ()

    set(INFINITY_APPLE_HNSW_D0_DIR "${CMAKE_SOURCE_DIR}/tools/apple_silicon/native_hnsw_smoke")
    add_executable(infinity_hnsw_d0_production EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_main.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_runner.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_audit.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_query_benchmark.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/infinity_hnsw_bridge.cpp"
    )
    target_include_directories(infinity_hnsw_d0_production PRIVATE "${INFINITY_APPLE_HNSW_D0_DIR}")
    target_compile_definitions(infinity_hnsw_d0_production PRIVATE HNSW_D0_INFINITY)
    target_compile_options(infinity_hnsw_d0_production PRIVATE
            -Wall
            -Wextra
            -Wno-reserved-module-identifier
    )
    set_source_files_properties(
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_audit.cpp"
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
    )

    get_target_property(INFINITY_RUNTIME_LINK_LIBRARIES infinity LINK_LIBRARIES)
    get_target_property(INFINITY_RUNTIME_LINK_DIRECTORIES infinity LINK_DIRECTORIES)
    target_link_libraries(infinity_hnsw_d0_production PRIVATE ${INFINITY_RUNTIME_LINK_LIBRARIES})
    target_link_directories(infinity_hnsw_d0_production PRIVATE ${INFINITY_RUNTIME_LINK_DIRECTORIES})
    add_dependencies(infinity_hnsw_d0_production miniocpp curlpp_static libcurl_static)

    add_executable(infinity_hnsw_exactness_emitter_production EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_exactness_emitter.cpp"
    )
    target_include_directories(infinity_hnsw_exactness_emitter_production PRIVATE "${INFINITY_APPLE_HNSW_D0_DIR}")
    target_compile_options(infinity_hnsw_exactness_emitter_production PRIVATE
            -Wall
            -Wextra
            -Wno-reserved-module-identifier
    )
    if (INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE)
        target_compile_definitions(
                infinity_hnsw_d0_production
                PRIVATE INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE
        )
        target_compile_definitions(
                infinity_hnsw_exactness_emitter_production
                PRIVATE INFINITY_ENABLE_HNSW_INCREMENTAL_RECIPROCAL_EXECUTION_EVIDENCE
        )
    endif ()
    if (INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE)
        target_compile_definitions(
                infinity_hnsw_d0_production
                PRIVATE INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE
        )
        target_compile_definitions(
                infinity_hnsw_exactness_emitter_production
                PRIVATE INFINITY_ENABLE_HNSW_THRESHOLD_BATCH4_EXECUTION_EVIDENCE
        )
    endif ()
    target_link_libraries(infinity_hnsw_exactness_emitter_production PRIVATE ${INFINITY_RUNTIME_LINK_LIBRARIES})
    target_link_directories(infinity_hnsw_exactness_emitter_production PRIVATE ${INFINITY_RUNTIME_LINK_DIRECTORIES})
    add_dependencies(infinity_hnsw_exactness_emitter_production miniocpp curlpp_static libcurl_static)

    add_custom_target(infinity_hnsw_d0_exactness_bundle)
    add_dependencies(infinity_hnsw_d0_exactness_bundle
            infinity_hnsw_d0_production
            infinity_hnsw_exactness_emitter_production
    )

    add_executable(infinity_hnsw_d0_lvq_production EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_main.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_runner.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_audit.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/hnsw_d0_query_benchmark.cpp"
            "${INFINITY_APPLE_HNSW_D0_DIR}/infinity_hnsw_lvq_bridge.cpp"
    )
    target_include_directories(infinity_hnsw_d0_lvq_production PRIVATE "${INFINITY_APPLE_HNSW_D0_DIR}")
    target_compile_definitions(infinity_hnsw_d0_lvq_production PRIVATE HNSW_D0_INFINITY)
    target_compile_options(infinity_hnsw_d0_lvq_production PRIVATE
            -Wall
            -Wextra
            -Wno-reserved-module-identifier
    )
    target_link_libraries(infinity_hnsw_d0_lvq_production PRIVATE ${INFINITY_RUNTIME_LINK_LIBRARIES})
    target_link_directories(infinity_hnsw_d0_lvq_production PRIVATE ${INFINITY_RUNTIME_LINK_DIRECTORIES})
    add_dependencies(infinity_hnsw_d0_lvq_production miniocpp curlpp_static libcurl_static)
endif ()

option(
        INFINITY_BUILD_APPLE_HNSW_KERNEL_REPLAY
        "Build equal-address Apple HNSW kernel replay against production infinity_core and pinned FAISS"
        OFF
)

if (INFINITY_BUILD_APPLE_HNSW_KERNEL_REPLAY)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "Production Apple HNSW kernel replay requires native Apple arm64")
    endif ()

    set(
            INFINITY_APPLE_REPLAY_FAISS_INCLUDE_DIR
            ""
            CACHE PATH
            "Pinned FAISS source directory containing faiss/utils/distances.h"
    )
    set(
            INFINITY_APPLE_REPLAY_FAISS_LIBRARY
            ""
            CACHE FILEPATH
            "Pinned generic-arm64 FAISS library"
    )
    set(
            INFINITY_APPLE_REPLAY_LIBOMP_INCLUDE_DIR
            ""
            CACHE PATH
            "Matched libomp include directory containing omp.h"
    )
    set(
            INFINITY_APPLE_REPLAY_LIBOMP_LIBRARY
            ""
            CACHE FILEPATH
            "Matched libomp library"
    )
    if (NOT EXISTS "${INFINITY_APPLE_REPLAY_FAISS_INCLUDE_DIR}/faiss/utils/distances.h")
        message(FATAL_ERROR "Replay FAISS include directory does not contain faiss/utils/distances.h")
    endif ()
    if (NOT EXISTS "${INFINITY_APPLE_REPLAY_FAISS_LIBRARY}")
        message(FATAL_ERROR "Replay FAISS library does not exist")
    endif ()
    if (NOT EXISTS "${INFINITY_APPLE_REPLAY_LIBOMP_INCLUDE_DIR}/omp.h")
        message(FATAL_ERROR "Replay libomp include directory does not contain omp.h")
    endif ()
    if (NOT EXISTS "${INFINITY_APPLE_REPLAY_LIBOMP_LIBRARY}")
        message(FATAL_ERROR "Replay libomp library does not exist")
    endif ()

    set(INFINITY_APPLE_HNSW_REPLAY_DIR "${CMAKE_SOURCE_DIR}/tools/apple_silicon/native_hnsw_smoke")
    add_library(infinity_hnsw_kernel_replay_i8_scalar OBJECT EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}/hnsw_kernel_replay_i8_scalar.cpp"
    )
    target_include_directories(infinity_hnsw_kernel_replay_i8_scalar PRIVATE
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}"
    )
    target_compile_options(infinity_hnsw_kernel_replay_i8_scalar PRIVATE
            -O3
            -fno-vectorize
            -fno-slp-vectorize
            -fno-unroll-loops
            -fno-lto
            -Wall
            -Wextra
            -Wpedantic
    )
    set_target_properties(infinity_hnsw_kernel_replay_i8_scalar PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION FALSE
            UNITY_BUILD OFF
    )

    add_executable(infinity_hnsw_kernel_replay EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}/hnsw_kernel_replay.cpp"
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}/hnsw_kernel_replay_infinity.cpp"
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}/hnsw_kernel_replay_faiss.cpp"
            $<TARGET_OBJECTS:infinity_hnsw_kernel_replay_i8_scalar>
    )
    target_include_directories(infinity_hnsw_kernel_replay PRIVATE
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}"
            "${INFINITY_APPLE_REPLAY_FAISS_INCLUDE_DIR}"
            "${INFINITY_APPLE_REPLAY_LIBOMP_INCLUDE_DIR}"
    )
    target_compile_options(infinity_hnsw_kernel_replay PRIVATE
            -O3
            -Wall
            -Wextra
            -Wpedantic
            -Wno-reserved-module-identifier
    )
    get_target_property(INFINITY_RUNTIME_LINK_LIBRARIES infinity LINK_LIBRARIES)
    get_target_property(INFINITY_RUNTIME_LINK_DIRECTORIES infinity LINK_DIRECTORIES)
    target_link_libraries(infinity_hnsw_kernel_replay PRIVATE
            ${INFINITY_RUNTIME_LINK_LIBRARIES}
            "${INFINITY_APPLE_REPLAY_FAISS_LIBRARY}"
            "${INFINITY_APPLE_REPLAY_LIBOMP_LIBRARY}"
    )
    target_link_directories(infinity_hnsw_kernel_replay PRIVATE ${INFINITY_RUNTIME_LINK_DIRECTORIES})
    add_dependencies(infinity_hnsw_kernel_replay miniocpp curlpp_static libcurl_static)

    add_executable(infinity_hnsw_kernel_replay_i8 EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}/hnsw_kernel_replay_i8.cpp"
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}/hnsw_kernel_replay_infinity.cpp"
            $<TARGET_OBJECTS:infinity_hnsw_kernel_replay_i8_scalar>
    )
    target_include_directories(infinity_hnsw_kernel_replay_i8 PRIVATE
            "${INFINITY_APPLE_HNSW_REPLAY_DIR}"
    )
    target_compile_options(infinity_hnsw_kernel_replay_i8 PRIVATE
            -O3
            -Wall
            -Wextra
            -Wpedantic
            -Wno-reserved-module-identifier
    )
    target_link_libraries(infinity_hnsw_kernel_replay_i8 PRIVATE
            ${INFINITY_RUNTIME_LINK_LIBRARIES}
    )
    target_link_directories(infinity_hnsw_kernel_replay_i8 PRIVATE
            ${INFINITY_RUNTIME_LINK_DIRECTORIES}
    )
    add_dependencies(infinity_hnsw_kernel_replay_i8
            miniocpp
            curlpp_static
            libcurl_static
    )
endif ()

option(
        INFINITY_BUILD_APPLE_HNSW_LVQ_REPLAY
        "Build the production-origin Apple HNSW LVQ capture and replay format tools"
        OFF
)

if (INFINITY_BUILD_APPLE_HNSW_LVQ_REPLAY)
    if (NOT APPLE OR NOT ARM64)
        message(FATAL_ERROR "Production-origin HNSW LVQ replay requires native Apple arm64")
    endif ()

    set(INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR "${CMAKE_SOURCE_DIR}/tools/apple_silicon/native_hnsw_smoke")
    add_executable(infinity_hnsw_lvq_replay_format_test EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}/hnsw_lvq_replay_format.cpp"
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}/hnsw_lvq_replay_format_test.cpp"
    )
    target_include_directories(infinity_hnsw_lvq_replay_format_test PRIVATE
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}"
    )
    target_compile_options(infinity_hnsw_lvq_replay_format_test PRIVATE
            -O2
            -Wall
            -Wextra
            -Wpedantic
    )

    add_executable(infinity_hnsw_lvq_production_capture EXCLUDE_FROM_ALL
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}/hnsw_lvq_production_capture.cpp"
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}/hnsw_lvq_replay_format.cpp"
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}/hnsw_d0_audit.cpp"
    )
    target_include_directories(infinity_hnsw_lvq_production_capture PRIVATE
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}"
    )
    target_compile_options(infinity_hnsw_lvq_production_capture PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wno-reserved-module-identifier
    )
    set_source_files_properties(
            "${INFINITY_APPLE_HNSW_LVQ_REPLAY_DIR}/hnsw_d0_audit.cpp"
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
    )
    get_target_property(INFINITY_RUNTIME_LINK_LIBRARIES infinity LINK_LIBRARIES)
    get_target_property(INFINITY_RUNTIME_LINK_DIRECTORIES infinity LINK_DIRECTORIES)
    target_link_libraries(infinity_hnsw_lvq_production_capture PRIVATE ${INFINITY_RUNTIME_LINK_LIBRARIES})
    target_link_directories(infinity_hnsw_lvq_production_capture PRIVATE ${INFINITY_RUNTIME_LINK_DIRECTORIES})
    add_dependencies(infinity_hnsw_lvq_production_capture miniocpp curlpp_static libcurl_static)
endif ()

### unit test

file(GLOB_RECURSE
        test_main_cpp
        CONFIGURE_DEPENDS
        unit_test/main.cpp
)

file(GLOB_RECURSE
        ut_ut
        CONFIGURE_DEPENDS
        unit_test/*_ut.cpp
)

file(GLOB_RECURSE
        ut_impl
        CONFIGURE_DEPENDS
        unit_test/*_impl.cpp
)

file(GLOB_RECURSE
        ut_mod
        CONFIGURE_DEPENDS
        unit_test/*.cppm
        unit_test/*_impl.cpp
)
add_executable(unit_test
        ${test_main_cpp}
)
target_sources(unit_test
        PRIVATE
        FILE_SET ut_cxx_modules TYPE CXX_MODULES FILES
        ${ut_mod}

        ${ut_ut}
)

set_target_properties(unit_test PROPERTIES OUTPUT_NAME test_main)
add_dependencies(unit_test miniocpp curlpp_static libcurl_static)

target_link_libraries(unit_test
        infinity_core
        GTest::gtest GTest::gtest_main
        sql_parser
        onnxruntime_mlas
        zsv_parser
        newpfor
        ${FASTPFOR_STATIC_LIB}
        jma
        opencc
        ${INFINITY_PLATFORM_LIBRARIES}
        Boost::asio
        thrift::thrift
        # thriftnb::thriftnb
        libevent::core
        oatpp::oatpp
        Parquet::parquet_static
        Arrow::arrow_static
        ${JEMALLOC_STATIC_LIB}
        miniocpp::miniocpp
        pugixml::static
        curlpp_static
        unofficial::inih::libinih
        unofficial::inih::inireader
        libcurl_static
        OpenSSL::SSL
        OpenSSL::Crypto
        re2::re2
        absl::any absl::log absl::base absl::bits
        PCRE2::8BIT
        RocksDB::rocksdb
        simdjson::simdjson
        nlohmann_json::nlohmann_json
        CLI11::CLI11
        magic_enum::magic_enum
        roaring::roaring
        spdlog::spdlog
        fmt::fmt
        ZLIB::ZLIB
)

target_compile_options(unit_test PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-sign-compare
)

target_link_directories(unit_test PUBLIC "${CMAKE_BINARY_DIR}/lib")
target_link_directories(unit_test PUBLIC "${CMAKE_BINARY_DIR}/third_party/minio-cpp/")
target_link_directories(unit_test PUBLIC "${CMAKE_BINARY_DIR}/third_party/curlpp/")
target_link_directories(unit_test PUBLIC "${CMAKE_BINARY_DIR}/third_party/curl/")
target_link_directories(unit_test PUBLIC "${CMAKE_BINARY_DIR}/third_party/")
target_link_directories(unit_test PUBLIC "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib")

target_include_directories(unit_test PUBLIC "${CMAKE_SOURCE_DIR}/unit_test")
