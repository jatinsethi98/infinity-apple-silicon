set(CPACK_PACKAGE_NAME "infinity")

string(REGEX REPLACE "^[^0-9]+" "" CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")

if (CPACK_PACKAGE_VERSION STREQUAL "")
    string(TIMESTAMP CURRENT_DATE "%Y%m%d")
    math(EXPR NEXT_PATCH "${PROJECT_VERSION_PATCH} + 1")
    set(NIGHTLY_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${NEXT_PATCH}")
    set(CPACK_PACKAGE_VERSION "${NIGHTLY_VERSION}~nightly.${CURRENT_DATE}")
endif ()

set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "Zhichang Yu <yuzhichang@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "The AI-native database built for LLM applications, offering incredibly fast vector and full-text search.")
set(CPACK_PACKAGE_VENDOR "infiniflow")

if (DEFINED PACKAGE_SUFFIX AND NOT PACKAGE_SUFFIX STREQUAL "")
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${PACKAGE_SUFFIX}")
else ()
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# ---------------------------------------------------------------------------
# Install rules
# https://cmake.org/cmake/help/latest/command/install.html
# WARNING: If an absolute path is given, cpack will install the specific files
# on the host system (requires root permission) and then include them in the package.
# If a relative path (interpreted relative to CMAKE_INSTALL_PREFIX) is given,
# cpack includes specific files in the package without actually installing them.
# CMAKE_INSTALL_PREFIX defaults to "/usr/local".
# ---------------------------------------------------------------------------
# Relocatable layout, not a system install. /usr is protected by SIP on macOS and
# cannot be written to at all, and there is no systemd unit to install any more.
# scripts/apple_silicon/make_package.sh produces the artifact that is actually shipped
# (and self-tests it by running the server from a relocated copy); these rules exist so
# `cmake --install` still lays down something sane.
install(TARGETS infinity DESTINATION libexec)
install(FILES conf/infinity_conf.toml DESTINATION etc)
# The full-text analyzers load their dictionaries from resource_dir at runtime, so a
# package without this tree has broken CJK/RAG/IK search. Upstream's install rules
# omitted it entirely.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/resource/" DESTINATION share/infinity/resource)

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
# TGZ only. The RPM and DEB generators went with Linux support, along with the
# conf/postinst maintainer script and the systemd unit they installed.
set(CPACK_GENERATOR "TGZ")

# Enable CPack debug output
set(CPACK_PACKAGE_DEBUG True)

# https://cmake.org/cmake/help/latest/variable/CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION.html
set(CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION "ON")

include(CPack)
