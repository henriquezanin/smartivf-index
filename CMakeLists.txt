cmake_minimum_required(VERSION 3.28)
project(smartivf
        VERSION 0.1.0
        DESCRIPTION "Range-filtered approximate nearest-neighbor index"
        LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Language: C++26 with no GNU extensions
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# GCC >= 14 or Clang >= 19; earlier releases lack the C++23/26 library features
# used here.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "GCC ≥ 14 required for C++26 (you have ${CMAKE_CXX_COMPILER_VERSION})")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
    message(FATAL_ERROR "Clang ≥ 19 required for C++26 (you have ${CMAKE_CXX_COMPILER_VERSION})")
endif()

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(SMARTIVF_BUILD_TESTS "Build the GoogleTest suite" ON)
option(SMARTIVF_USE_NATIVE  "Compile with -march=native (BMI2, AVX2/AVX-512 if available)" ON)
option(SMARTIVF_USE_LTO     "Enable LTO in Release builds" ON)

# ------------------------------------------------------------------------------
# Warnings & optimisation flags
# ------------------------------------------------------------------------------
add_library(smartivf_warnings INTERFACE)
target_compile_options(smartivf_warnings INTERFACE
    -Wall -Wextra -Wpedantic
    -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
    -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion
    -Wnull-dereference -Wdouble-promotion -Wformat=2)

add_library(smartivf_arch INTERFACE)
if(SMARTIVF_USE_NATIVE)
    target_compile_options(smartivf_arch INTERFACE -march=native -mtune=native)
endif()

if(SMARTIVF_USE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
    if(lto_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    else()
        message(WARNING "LTO not supported: ${lto_error}")
    endif()
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# FAISS provides k-means and the SIMD distance kernels.
# The system install is preferred (faiss-cpu AUR or libfaiss-dev); a manual
# lookup is used as a fallback. Upstream's CMake config installs the imported
# target `faiss` (no namespace).
find_package(faiss QUIET CONFIG
             HINTS /usr/share/faiss /usr/lib/cmake/faiss /usr/local/lib/cmake/faiss)
if(TARGET faiss_avx2)
    set(SMARTIVF_FAISS_TARGET faiss_avx2)
    message(STATUS "Linking FAISS AVX2 variant (faiss_avx2)")
elseif(TARGET faiss)
    set(SMARTIVF_FAISS_TARGET faiss)
    message(STATUS "Linking baseline FAISS (faiss)")
else()
    # Fall back to manual lookup if upstream CMake config was not installed.
    find_library(FAISS_LIB NAMES faiss libfaiss
                 HINTS /usr/lib /usr/lib64 /usr/local/lib
                       ${CMAKE_PREFIX_PATH}/lib
                 PATH_SUFFIXES x86_64-linux-gnu)
    find_path(FAISS_INCLUDE_DIR faiss/Index.h
              HINTS /usr/include /usr/local/include
                    ${CMAKE_PREFIX_PATH}/include)
    if(NOT FAISS_LIB OR NOT FAISS_INCLUDE_DIR)
        message(FATAL_ERROR
            "FAISS not found. Install via:\n"
            "    yay -S faiss-cpu   # Arch / AUR\n"
            "    apt install libfaiss-dev   # Debian trixie+\n"
            "or build from source and pass -DCMAKE_PREFIX_PATH=...")
    endif()
    add_library(faiss UNKNOWN IMPORTED)
    set_target_properties(faiss PROPERTIES
        IMPORTED_LOCATION             "${FAISS_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}")
    set(SMARTIVF_FAISS_TARGET faiss)
    message(STATUS "Found FAISS (manual): ${FAISS_LIB} (include: ${FAISS_INCLUDE_DIR})")
endif()

# CLI11 — header-only CLI parser, fetched at configure time.
include(FetchContent)
FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(cli11)

# ------------------------------------------------------------------------------
# Core library
# ------------------------------------------------------------------------------
add_library(smartivf_core STATIC
    src/quantize.cpp
    src/compute_quantile.cpp
    src/range_decompose.cpp
    src/build_partition.cpp
    src/smartivf.cpp
    src/store.cpp
    src/sfc/z_order.cpp
    src/utils/fvecs.cpp
    src/utils/ivecs.cpp
    src/utils/ranges.cpp
    src/utils/scalar.cpp)

target_include_directories(smartivf_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(smartivf_core
    PUBLIC  ${SMARTIVF_FAISS_TARGET} OpenMP::OpenMP_CXX
    PRIVATE smartivf_warnings smartivf_arch)

# ------------------------------------------------------------------------------
# CLI binary
# ------------------------------------------------------------------------------
add_executable(smartivf src/main.cpp)
target_link_libraries(smartivf PRIVATE smartivf_core CLI11::CLI11 smartivf_warnings smartivf_arch)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
if(SMARTIVF_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(googletest)
    endif()

    enable_testing()
    include(GoogleTest)

    add_executable(smartivf_tests
        tests/test_sfc.cpp
        tests/test_quantize.cpp
        tests/test_compute_quantile.cpp
        tests/test_range_decompose.cpp
        tests/test_build_partition.cpp
        tests/test_smartivf.cpp
        tests/test_store.cpp
        tests/test_utils.cpp)

    target_link_libraries(smartivf_tests PRIVATE
        smartivf_core GTest::gtest GTest::gtest_main
        smartivf_warnings smartivf_arch)

    gtest_discover_tests(smartivf_tests)
endif()
