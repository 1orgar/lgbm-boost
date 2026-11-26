cmake_minimum_required(VERSION 3.20)
project(lgbm_boost LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FAST_SCORER_BUILD_PYTHON "Build nanobind Python extension" ON)
option(FAST_SCORER_BUILD_CPP_TESTS "Build standalone C++ tests/cli" OFF)
option(FAST_SCORER_USE_OPENMP "Enable OpenMP instead of built-in C++17 ThreadPool" OFF)

# Detect target CPU architecture (respecting CMAKE_OSX_ARCHITECTURES for universal/Rosetta builds)
set(FAST_SCORER_IS_X86_64 OFF)
if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(FAST_SCORER_IS_X86_64 ON)
elseif(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(FAST_SCORER_IS_X86_64 ON)
endif()

set(CORE_SOURCES
    src/model_parser.cpp
    src/compiler.cpp
    src/kernels_scalar.cpp
    src/scorer.cpp
)

if(FAST_SCORER_IS_X86_64)
    list(APPEND CORE_SOURCES
        src/kernels_avx2.cpp
        src/kernels_avx512.cpp
    )
    if(MSVC)
        set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/O2;/arch:AVX2")
        set_source_files_properties(src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/O2;/arch:AVX512")
    else()
        set_source_files_properties(src/kernels_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-O3;-mavx2;-mfma;-mbmi;-mbmi2;-fno-math-errno;-fno-trapping-math"
        )
        set_source_files_properties(src/kernels_avx512.cpp PROPERTIES
            COMPILE_OPTIONS "-O3;-mavx512f;-mavx512cd;-mavx512bw;-mavx512vl;-mavx512dq;-mfma;-mbmi;-mbmi2;-fno-math-errno;-fno-trapping-math"
        )
    endif()
endif()

add_library(fast_scorer_core STATIC ${CORE_SOURCES})
target_include_directories(fast_scorer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(fast_scorer_core PRIVATE /O2)
else()
    target_compile_options(fast_scorer_core PRIVATE -O3 -fno-math-errno -fno-trapping-math -Wall -Wextra -Wno-unused-parameter)
endif()

if(FAST_SCORER_IS_X86_64)
    target_compile_definitions(fast_scorer_core PUBLIC FAST_SCORER_ARCH_X86_64=1)
endif()

# Optional OpenMP detection (with Homebrew libomp hint on macOS arm64)
if(FAST_SCORER_USE_OPENMP)
    if(APPLE AND NOT FAST_SCORER_IS_X86_64 AND EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    endif()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "fast_scorer: OpenMP enabled")
        target_link_libraries(fast_scorer_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(fast_scorer_core PUBLIC FAST_SCORER_HAVE_OPENMP=1)
    else()
        message(STATUS "fast_scorer: OpenMP not found, using built-in C++17 atomic ThreadPool")
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(fast_scorer_core PUBLIC Threads::Threads)

if(FAST_SCORER_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
    )
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
        _lgbm_boost_ext
        STABLE_ABI
        NB_STATIC
        python/bindings.cpp
    )
    target_link_libraries(_lgbm_boost_ext PRIVATE fast_scorer_core)
    install(TARGETS _lgbm_boost_ext LIBRARY DESTINATION lgbm_boost)
endif()

if(FAST_SCORER_BUILD_CPP_TESTS)
    add_executable(fast_scorer_cli src/cli_verify.cpp)
    target_link_libraries(fast_scorer_cli PRIVATE fast_scorer_core)
endif()
