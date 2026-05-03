cmake_minimum_required(VERSION 3.20)
project(inferbit C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings (platform-specific)
if(MSVC)
    add_compile_options(/W3 /std:c17 /D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")

    # Tune for the host microarchitecture on Apple Silicon (enables dotprod
    # scheduling and register pressure assumptions that the kernels rely on
    # to keep batch accumulators in NEON registers instead of spilling).
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        add_compile_options(-mcpu=apple-m1)
    endif()
endif()

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

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
    src/pq_decode.c
    src/pq_kmeans.c
)

# Determine target architecture
# CMAKE_OSX_ARCHITECTURES overrides CMAKE_SYSTEM_PROCESSOR for macOS cross-compile
set(TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
    set(TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
endif()

# AVX2 compile flags — only when targeting x86_64
if(TARGET_ARCH MATCHES "x86_64|AMD64")
    if(MSVC)
        set_source_files_properties(src/kernels/avx2.c PROPERTIES
            COMPILE_FLAGS "/arch:AVX2"
        )
        # Let pq_decode use AVX2 via the auto-vectoriser. The PQ matmul
        # is gather-bound but the LUT-table build and the per-row scale
        # multiply both auto-vec cleanly.
        set_source_files_properties(src/pq_decode.c PROPERTIES
            COMPILE_FLAGS "/arch:AVX2"
        )
    else()
        set_source_files_properties(src/kernels/avx2.c PROPERTIES
            COMPILE_FLAGS "-mavx2 -mfma"
        )
        set_source_files_properties(src/pq_decode.c PROPERTIES
            COMPILE_FLAGS "-mavx2 -mfma"
        )
        set_source_files_properties(src/pq_kmeans.c PROPERTIES
            COMPILE_FLAGS "-mavx2 -mfma"
        )
    endif()
endif()

# Platform libraries
if(WIN32)
    add_compile_definitions(INFERBIT_BUILD_DLL)
    set(PLATFORM_LIBS "")
else()
    set(PLATFORM_LIBS m)
endif()

# Shared library
add_library(inferbit SHARED ${INFERBIT_SOURCES})
target_include_directories(inferbit PUBLIC include)
target_include_directories(inferbit PRIVATE src)
target_link_libraries(inferbit ${PLATFORM_LIBS})
# On Windows, MSVC requires explicit dllexport for symbols to be visible
# from the DLL. Internal headers (pq_decode.h, etc.) declare functions
# without IB_API; auto-export all global symbols so Python ctypes /
# dlsym can resolve them. POSIX is unaffected (no-op there).
set_target_properties(inferbit PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Static library
add_library(inferbit_static STATIC ${INFERBIT_SOURCES})
target_include_directories(inferbit_static PUBLIC include)
target_include_directories(inferbit_static PRIVATE src)
target_link_libraries(inferbit_static ${PLATFORM_LIBS})
set_target_properties(inferbit_static PROPERTIES OUTPUT_NAME inferbit)

# Test executables
add_executable(test_inferbit tests/test_basic.c)
target_link_libraries(test_inferbit inferbit_static ${PLATFORM_LIBS})
target_include_directories(test_inferbit PRIVATE include)

add_executable(test_ibf_loader tests/test_ibf_loader.c)
target_link_libraries(test_ibf_loader inferbit_static ${PLATFORM_LIBS})
target_include_directories(test_ibf_loader PRIVATE include)

add_executable(test_convert tests/test_convert.c)
target_link_libraries(test_convert inferbit_static ${PLATFORM_LIBS})
target_include_directories(test_convert PRIVATE include)

# POSIX-only tests/benches: clock_gettime(CLOCK_MONOTONIC),
# sys/resource.h, and some internal-symbol linkage. Skipped on Windows.
if(NOT WIN32)
    add_executable(test_forward tests/test_forward.c)
    target_link_libraries(test_forward inferbit_static ${PLATFORM_LIBS})
    target_include_directories(test_forward PRIVATE include)

    add_executable(test_pq_decode tests/test_pq_decode.c)
    target_link_libraries(test_pq_decode inferbit_static ${PLATFORM_LIBS})
    target_include_directories(test_pq_decode PRIVATE include src)

    foreach(b bench_pq_decode bench_pq_ffn bench_pq_model_mlp
              bench_pq_real_model bench_pq_pathd bench_pq_q8lut
              bench_pq_streaming bench_pq_dejavu bench_pq_bounded)
        add_executable(${b} tests/${b}.c)
        target_link_libraries(${b} inferbit_static ${PLATFORM_LIBS})
        target_include_directories(${b} PRIVATE include src)
    endforeach()
endif()

enable_testing()
add_test(NAME basic COMMAND test_inferbit)
add_test(NAME ibf_loader COMMAND test_ibf_loader)
add_test(NAME convert COMMAND test_convert)
if(NOT WIN32)
    add_test(NAME forward COMMAND test_forward)
endif()

# Install
install(TARGETS inferbit
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/inferbit.h DESTINATION include)
