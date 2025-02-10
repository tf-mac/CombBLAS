// benchmark_cuda_memory_manager.cpp

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "CombBLAS/CudaMemoryManager.h"  // Include your custom memory manager header

using namespace combblas;

//-------------------------------------------------------------------------------
// Benchmark for standard CUDA memory allocation using cudaMalloc and cudaFree.
//-------------------------------------------------------------------------------
static void BM_CudaMallocFree(benchmark::State& state)
{
    // Use the size provided as the benchmark parameter (in bytes)
    const std::size_t allocationSize = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        // Allocate device memory using cudaMalloc.
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, allocationSize);
        if (err != cudaSuccess) {
            state.SkipWithError("cudaMalloc failed");
        }
        // Free the memory using cudaFree.
        err = cudaFree(ptr);
        if (err != cudaSuccess) {
            state.SkipWithError("cudaFree failed");
        }
    }
    // Record total bytes processed.
    state.SetBytesProcessed(state.iterations() * allocationSize);
}
BENCHMARK(BM_CudaMallocFree)
    ->Arg(1024)                // 1K
    ->Arg(10 * 1024)           // 10K
    ->Arg(100 * 1024)          // 100K
    ->Arg(1024 * 1024)         // 1M
    ->Arg(10 * 1024 * 1024)    // 10M
    ->Arg(100 * 1024 * 1024);  // 100M

//-------------------------------------------------------------------------------
// Benchmark for custom CUDA memory manager (CudaMemoryManager).
//-------------------------------------------------------------------------------
static void BM_CudaMemoryManager(benchmark::State& state)
{
    const std::size_t allocationSize = static_cast<std::size_t>(state.range(0));
    // Create a memory manager with a preallocated pool size (e.g., 512 MB)
    const std::size_t poolSize = 512ull * 1024ull * 1024ull;
    CudaMemoryManager memManager(poolSize);

    for (auto _ : state) {
        // Allocate device memory using the memory manager.
        void* ptr = memManager.malloc(allocationSize, 0);
        // (Optional: You can perform operations on ptr here if needed)
        memManager.free(ptr, 0);
    }
    state.SetBytesProcessed(state.iterations() * allocationSize);
}
BENCHMARK(BM_CudaMemoryManager)
    ->Arg(1024)                // 1K
    ->Arg(10 * 1024)           // 10K
    ->Arg(100 * 1024)          // 100K
    ->Arg(1024 * 1024)         // 1M
    ->Arg(10 * 1024 * 1024)    // 10M
    ->Arg(100 * 1024 * 1024);  // 100M

//-------------------------------------------------------------------------------
// Main function for running the benchmarks.
//-------------------------------------------------------------------------------
BENCHMARK_MAIN();