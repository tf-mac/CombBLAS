#ifdef USE_CUDA
#include "CombBLAS/CudaMemoryManager.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <stdexcept>
#include <unordered_map>

#include "CombBLAS/cuutils.h"
namespace combblas
{
CudaMemoryManager::CudaMemoryManager(std::size_t pool_size)
{
    // Create an upstream asynchronous memory resource that uses cudaMallocAsync/cudaFreeAsync.
    upstream_ = std::make_unique<rmm::mr::cuda_memory_resource>();

    // Create a pool memory resource that preallocates 'pool_size' bytes from the upstream resource.
    pool_resource_ =
        std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(upstream_.get(), pool_size);
}

CudaMemoryManager::~CudaMemoryManager()
{
    // Optionally warn if there are still outstanding allocations.
    if (!allocations_.empty()) {
        std::cerr << "Warning: There are outstanding allocations!" << std::endl;
    }
}

void* CudaMemoryManager::malloc(std::size_t bytes, cudaStream_t stream)
{
    // Allocate memory from the pool resource on the given stream.
    void* ptr = pool_resource_->allocate(bytes, stream);
    // Record the allocation size for proper deallocation later.
    allocations_[ptr] = bytes;
    return ptr;
}

void CudaMemoryManager::free(void* ptr, cudaStream_t stream)
{
    // Look up the pointer in the allocations map.
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        throw std::runtime_error("Attempting to free memory that was not allocated");
    }
    // Retrieve the allocation size.
    std::size_t bytes = it->second;
    // Deallocate the memory using the pool resource.
    pool_resource_->deallocate(ptr, bytes, stream);
    // Remove the pointer from our map.
    allocations_.erase(it);
}

}  // namespace combblas
#endif // USE_CUDA