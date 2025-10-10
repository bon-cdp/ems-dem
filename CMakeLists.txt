cmake_minimum_required(VERSION 3.21)
project(EMS-DEM VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(USE_GPU "Enable GPU acceleration with HIP" ON)
option(USE_DOUBLE_PRECISION "Use double precision (default: single)" OFF)
option(BUILD_TESTS "Build test cases" ON)
option(BUILD_EXAMPLES "Build example simulations" ON)

# Find HIP if GPU enabled
if(USE_GPU)
    # Try to find HIP
    find_package(HIP QUIET)

    if(HIP_FOUND)
        message(STATUS "HIP found: ${HIP_VERSION}")
        enable_language(HIP)
        add_definitions(-DUSE_HIP)
    else()
        message(WARNING "HIP not found. GPU acceleration disabled. Install ROCm for GPU support.")
        set(USE_GPU OFF)
    endif()
endif()

# Precision
if(USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE_PRECISION)
    message(STATUS "Using double precision")
else()
    message(STATUS "Using single precision (float)")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "HIP")
    # hipcc flags
    add_compile_options(-O3 -ffast-math)
endif()

# Core library
file(GLOB_RECURSE CORE_SOURCES
    src/core/*.cpp
    src/cpu/*.cpp
    src/geometry/*.cpp
    src/io/*.cpp
)

add_library(emsdem_core STATIC ${CORE_SOURCES})
target_include_directories(emsdem_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# GPU library (if enabled)
if(USE_GPU AND HIP_FOUND)
    file(GLOB_RECURSE GPU_SOURCES src/gpu/*.hip)

    if(GPU_SOURCES)
        add_library(emsdem_gpu STATIC ${GPU_SOURCES})
        target_include_directories(emsdem_gpu PUBLIC
            ${CMAKE_SOURCE_DIR}/include
        )
        set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()
endif()

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
message(STATUS "C++ compiler:     ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ standard:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "GPU acceleration: ${USE_GPU}")
if(USE_GPU AND HIP_FOUND)
    message(STATUS "HIP version:      ${HIP_VERSION}")
endif()
message(STATUS "Double precision: ${USE_DOUBLE_PRECISION}")
message(STATUS "Build tests:      ${BUILD_TESTS}")
message(STATUS "Build examples:   ${BUILD_EXAMPLES}")
message(STATUS "==========================================")
message(STATUS "")
