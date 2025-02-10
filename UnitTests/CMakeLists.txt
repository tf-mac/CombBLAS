cmake_minimum_required(VERSION 3.10)


# Add GoogleTest
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
# Add RMM
CPMAddPackage(
    NAME RMM
    GITHUB_REPOSITORY rapidsai/rmm
    GIT_TAG branch-24.10
)


# Add GoogleTest
find_package(GTest REQUIRED)
find_package(CCCL REQUIRED)
find_package(rmm REQUIRED)
# Add test source files
set(TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/CudaMemoryManager.cpp
    test_cuda_memory_manager.cpp
)

# Add the executable for the tests
add_executable(runUnitTests ${TEST_SOURCES})

target_include_directories(runUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/include)
# Link the test executable with GoogleTest and the main library
target_link_libraries(runUnitTests GTest::gtest_main pthread rmm::rmm)

# Enable CTest
enable_testing()

# Add a test to CTest
add_test(NAME runUnitTests COMMAND runUnitTests)

# Add Benchmark
CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  GIT_TAG        v1.6.1
  OPTIONS        "BENCHMARK_ENABLE_TESTING Off"
)
#-------------------------------------------------------------------
# Benchmark target (using GoogleBenchmark)
#-------------------------------------------------------------------
add_executable(bm-cmm
  benchmark_cuda_memory_manager.cpp
  ${CMAKE_SOURCE_DIR}/src/CudaMemoryManager.cpp
)
target_include_directories(bm-cmm PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bm-cmm PRIVATE benchmark::benchmark rmm::rmm)