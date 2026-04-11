cmake_minimum_required(VERSION 3.20)
project(inferbit C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release optimizations
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")

# Source files
set(INFERBIT_SOURCES
    src/version.c
    src/error.c
    src/config.c
    src/model.c
    src/generate.c
    src/kv_cache.c
    src/speculative.c
    src/simd.c
    src/kernels/scalar.c
    src/kernels/avx2.c
    src/kernels/neon.c
    src/kernels/ternary.c
    src/forward.c
    src/threading.c
    src/ibf_loader.c
    src/gguf.c
    src/safetensors.c
    src/safetensors_multi.c
    src/tensor_source.c
    src/config_json.c
    src/quantize.c
    src/convert.c
    src/convert_gguf.c
    src/eval.c
    src/calibrate.c
    src/cJSON.c
)

# AVX2 compile flags (x86_64 only)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(src/kernels/avx2.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma"
    )
endif()

# Shared library
add_library(inferbit SHARED ${INFERBIT_SOURCES})
target_include_directories(inferbit PUBLIC include)
target_include_directories(inferbit PRIVATE src)
target_link_libraries(inferbit m)

# Static library
add_library(inferbit_static STATIC ${INFERBIT_SOURCES})
target_include_directories(inferbit_static PUBLIC include)
target_include_directories(inferbit_static PRIVATE src)
target_link_libraries(inferbit_static m)
set_target_properties(inferbit_static PROPERTIES OUTPUT_NAME inferbit)

# Test executables
add_executable(test_inferbit tests/test_basic.c)
target_link_libraries(test_inferbit inferbit_static m)
target_include_directories(test_inferbit PRIVATE include)

add_executable(test_ibf_loader tests/test_ibf_loader.c)
target_link_libraries(test_ibf_loader inferbit_static m)
target_include_directories(test_ibf_loader PRIVATE include)

add_executable(test_forward tests/test_forward.c)
target_link_libraries(test_forward inferbit_static m)
target_include_directories(test_forward PRIVATE include)

add_executable(test_convert tests/test_convert.c)
target_link_libraries(test_convert inferbit_static m)
target_include_directories(test_convert PRIVATE include)

enable_testing()
add_test(NAME basic COMMAND test_inferbit)
add_test(NAME ibf_loader COMMAND test_ibf_loader)
add_test(NAME forward COMMAND test_forward)
add_test(NAME convert COMMAND test_convert)

# Install
install(TARGETS inferbit
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES include/inferbit.h DESTINATION include)
