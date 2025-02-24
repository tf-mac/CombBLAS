cmake_minimum_required(VERSION 3.10)

# Test OpenMP Threads Numbers
add_executable(test_openmp test_openmp.cpp)
find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
    target_compile_definitions(test_openmp PUBLIC THREADED)
    target_link_libraries(test_openmp PUBLIC OpenMP::OpenMP_CXX)
elseif (OPENMP_FOUND)
    target_compile_definitions(test_openmp PUBLIC THREADED)
    target_compile_options(test_openmp PUBLIC "${OpenMP_CXX_FLAGS}")
    target_link_libraries(test_openmp PUBLIC "${OpenMP_CXX_FLAGS}")
endif ()

# Test CUDA Aware MPI
add_executable(test_cudampi test_cudaawarempi.cpp)
find_package(MPI REQUIRED)
if (TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
    target_link_libraries(test_cudampi PUBLIC MPI::MPI_CXX)
else ()
    target_compile_options(test_cudampi PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
    target_link_libraries(test_cudampi PUBLIC "${MPI_CXX_LIBRARIES}"
            "${MPI_CXX_LINKFLAGS}")
    target_include_directories(test_cudampi PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif ()


if (USE_CUDA)

    # Test CUDA Memory Manager
    CPMAddPackage(
            NAME GTest
            GITHUB_REPOSITORY google/googletest
            GIT_TAG release-1.12.1
    )
    CPMAddPackage(
            NAME CCCL
            GITHUB_REPOSITORY NVIDIA/cccl
            GIT_TAG v2.5.0
    )
    CPMAddPackage(
            NAME RMM
            GITHUB_REPOSITORY rapidsai/rmm
            GIT_TAG branch-24.10
    )

    #    find_package(GTest REQUIRED)
    #    find_package(CCCL REQUIRED)
    #    find_package(rmm REQUIRED)
    # Add test source files
    set(TEST_SOURCES
            ${CMAKE_SOURCE_DIR}/src/cuda/CudaMemoryManager.cpp
            test_cuda_memory_manager.cpp
    )

    # Add the executable for the tests
    add_executable(test_cmm ${TEST_SOURCES})

    target_include_directories(test_cmm PRIVATE ${CMAKE_SOURCE_DIR}/include)
    # Link the test executable with GoogleTest and the main library
    target_link_libraries(test_cmm GTest::gtest_main pthread rmm::rmm)

    # Enable CTest if not
    enable_testing()
    add_test(NAME test_cmm COMMAND test_cmm)

    # Add Benchmark for CudaMemoryManager
    CPMAddPackage(
            NAME benchmark
            GITHUB_REPOSITORY google/benchmark
            GIT_TAG v1.6.1
            OPTIONS "BENCHMARK_ENABLE_TESTING Off"
    )
    #-------------------------------------------------------------------
    # Benchmark target (using GoogleBenchmark)
    #-------------------------------------------------------------------
    add_executable(bm_cmm
            benchmark_cuda_memory_manager.cpp
            ${CMAKE_SOURCE_DIR}/src/cuda/CudaMemoryManager.cpp
    )
    target_include_directories(bm_cmm PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(bm_cmm PRIVATE benchmark::benchmark rmm::rmm)

    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    add_executable(bm_nsparse
            benchmark_nsparse_hash.cpp
            ${CMAKE_SOURCE_DIR}/src/nsparse/nsparse.cu
    )
    target_include_directories(bm_nsparse PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(bm_nsparse PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(bm_nsparse PRIVATE benchmark::benchmark CUDA::cudart)

    add_executable(nsparse_main
            nsparse_main.cpp
            ${CMAKE_SOURCE_DIR}/src/nsparse/nsparse.cu
    )
    target_include_directories(nsparse_main PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(nsparse_main PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(nsparse_main PRIVATE CUDA::cudart)

    add_executable(acspgemm_main
            acspgemm_main.cpp
            ${CMAKE_SOURCE_DIR}/src/nsparse/nsparse.cu
    )
    target_include_directories(acspgemm_main PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(acspgemm_main PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(acspgemm_main PRIVATE CUDA::cudart)

    add_executable(cusparse_main
            cusparse_main.cpp
            ${CMAKE_SOURCE_DIR}/src/nsparse/nsparse.cu # I need getflops and read csr matrix
    )
    target_include_directories(cusparse_main PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(cusparse_main PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(cusparse_main PRIVATE CUDA::cudart CUDA::cusparse)

endif (USE_CUDA)