cmake_minimum_required(VERSION 3.16)
project(EMS-DEM VERSION 0.1.0 LANGUAGES CXX)

# Force hipcc as compiler for GPU-first approach
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(HIPCC_EXECUTABLE hipcc)
    if(HIPCC_EXECUTABLE)
        set(CMAKE_CXX_COMPILER ${HIPCC_EXECUTABLE})
        message(STATUS "Using hipcc: ${HIPCC_EXECUTABLE}")
    else()
        message(FATAL_ERROR "hipcc not found. Install ROCm for GPU support.")
    endif()
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(USE_DOUBLE_PRECISION "Use double precision (default: single)" OFF)
option(BUILD_TESTS "Build test cases" ON)
option(BUILD_EXAMPLES "Build example simulations" ON)

# Precision
if(USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE_PRECISION)
    message(STATUS "Using double precision")
else()
    message(STATUS "Using single precision (float)")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# HIP compiler flags
# Target AMD Radeon 780M (gfx1103) architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math --offload-arch=gfx1103")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -D__HIP_PLATFORM_AMD__")

# Core + GPU library (everything compiled with hipcc)
file(GLOB_RECURSE ALL_SOURCES
    src/core/*.cpp
    src/geometry/*.cpp
    src/io/*.cpp
    src/gpu/*.cpp
)

# Note: Triangle closestPoint needs to be GPU-compatible
# Already in triangle.hpp as inline functions

add_library(emsdem STATIC ${ALL_SOURCES})
target_include_directories(emsdem PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "========== EMS-DEM Configuration ==========")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "HIP compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "GPU target:       gfx1103 (AMD Radeon 780M)")
message(STATUS "Double precision: ${USE_DOUBLE_PRECISION}")
message(STATUS "Build tests:      ${BUILD_TESTS}")
message(STATUS "Build examples:   ${BUILD_EXAMPLES}")
message(STATUS "==========================================")
message(STATUS "")
