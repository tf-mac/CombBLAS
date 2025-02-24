#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <unordered_map>

// The code is generated from gpt, feel free to further improve it.
// I explictly remove the mutex to protect the map in multi-threaded environment.
// SO, THIS CODE IS NOT THREAD-SAFE!!!

namespace combblas
{
/*
 * CudaMemoryManager:
 * A memory manager built on top of Rapids RMM that preallocates a device memory
 * pool and provides stream-aware allocation (malloc/free) of CUDA memory.
 */
class CudaMemoryManager
{
   public:
    /// Construct a memory manager that preallocates a pool of 'pool_size' bytes.
    explicit CudaMemoryManager(std::size_t pool_size);

    /// Destructor. Warns if there are still outstanding allocations.
    ~CudaMemoryManager();

    /// Allocate 'bytes' bytes on the given CUDA stream.
    /// @param bytes Number of bytes to allocate.
    /// @param stream CUDA stream for asynchronous allocation (default: 0).
    /// @return Pointer to allocated device memory.
    void* malloc(std::size_t bytes, cudaStream_t stream = 0);

    /// Free memory previously allocated on the given CUDA stream.
    /// @param ptr Pointer returned by malloc.
    /// @param stream CUDA stream used when the memory was allocated (default: 0).
    /// @throws std::runtime_error if the pointer was not allocated by this manager.
    void free(void* ptr, cudaStream_t stream = 0);

   private:
    // Upstream asynchronous memory resource (using cudaMallocAsync/cudaFreeAsync).
    std::unique_ptr<rmm::mr::cuda_memory_resource> upstream_;

    // Pool memory resource that preallocates a large block of memory.
    std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_resource_;

    // Map from allocated pointer to allocation size (needed for deallocation).
    std::unordered_map<void*, std::size_t> allocations_;
};

}  // namespace combblas

#endif  // USE_CUDA