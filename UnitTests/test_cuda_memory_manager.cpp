#include <gtest/gtest.h>

#include "CombBLAS/CudaMemoryManager.h"

using namespace combblas;

/*
 * Test Case 1: BasicMallocFree
 *
 * This test allocates 1 KB of device memory, uses cudaMemset to set a known value,
 * synchronizes the device, copies the data to host memory, verifies the value, and
 * finally frees the allocated memory.
 */
TEST(CudaMemoryManagerTest, BasicMallocFree)
{
    const std::size_t poolSize = 512 * 1024 * 1024;  // 512 MB pool.
    CudaMemoryManager memManager(poolSize);

    const std::size_t allocSize = 1024;  // 1 KB.
    void* d_ptr = memManager.malloc(allocSize, 0);

    int value = 42;
    cudaError_t err = cudaMemset(d_ptr, value, allocSize);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess);

    std::vector<unsigned char> hostBuffer(allocSize);
    err = cudaMemcpy(hostBuffer.data(), d_ptr, allocSize, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    for (std::size_t i = 0; i < allocSize; ++i) {
        EXPECT_EQ(hostBuffer[i], static_cast<unsigned char>(value));
    }

    memManager.free(d_ptr, 0);
}

/*
 * Test Case 2: MultiStreamAllocation
 *
 * This test creates two CUDA streams, allocates memory on each stream, initializes the
 * memory asynchronously with distinct values, synchronizes each stream, verifies the data,
 * and then frees the allocated memory.
 */
TEST(CudaMemoryManagerTest, MultiStreamAllocation)
{
    CudaMemoryManager memManager(512 * 1024 * 1024);

    cudaStream_t stream1, stream2;
    cudaError_t err = cudaStreamCreate(&stream1);
    ASSERT_EQ(err, cudaSuccess);
    err = cudaStreamCreate(&stream2);
    ASSERT_EQ(err, cudaSuccess);

    const std::size_t size1 = 2048;  // 2 KB.
    const std::size_t size2 = 4096;  // 4 KB.

    void* ptr1 = memManager.malloc(size1, stream1);
    void* ptr2 = memManager.malloc(size2, stream2);

    int value1 = 55;
    int value2 = 77;
    err = cudaMemsetAsync(ptr1, value1, size1, stream1);
    ASSERT_EQ(err, cudaSuccess);
    err = cudaMemsetAsync(ptr2, value2, size2, stream2);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaStreamSynchronize(stream1);
    ASSERT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(stream2);
    ASSERT_EQ(err, cudaSuccess);

    std::vector<unsigned char> hostBuffer1(size1);
    err = cudaMemcpy(hostBuffer1.data(), ptr1, size1, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
    for (std::size_t i = 0; i < size1; ++i) {
        EXPECT_EQ(hostBuffer1[i], static_cast<unsigned char>(value1));
    }

    std::vector<unsigned char> hostBuffer2(size2);
    err = cudaMemcpy(hostBuffer2.data(), ptr2, size2, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
    for (std::size_t i = 0; i < size2; ++i) {
        EXPECT_EQ(hostBuffer2[i], static_cast<unsigned char>(value2));
    }

    memManager.free(ptr1, stream1);
    memManager.free(ptr2, stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

/*
 * Test Case 3: InvalidFreeThrows
 *
 * This test attempts to free a pointer not allocated by the memory manager. It uses
 * a host allocation to simulate an invalid pointer, and verifies that the expected
 * runtime exception is thrown.
 */
TEST(CudaMemoryManagerTest, InvalidFreeThrows)
{
    CudaMemoryManager memManager(512 * 1024 * 1024);
    void* dummy = malloc(128);
    EXPECT_THROW(memManager.free(dummy, 0), std::runtime_error);
    free(dummy);
}

/*
 * Test Case 4: RepeatedAllocFree
 *
 * This test performs several allocation/free cycles. For each allocation, a unique
 * value is set with cudaMemset, the data is verified, and then the memory is freed.
 */
TEST(CudaMemoryManagerTest, RepeatedAllocFree)
{
    CudaMemoryManager memManager(512 * 1024 * 1024);
    const int iterations = 10;
    const std::size_t allocSize = 1024;  // 1 KB.

    for (int i = 0; i < iterations; ++i) {
        void* ptr = memManager.malloc(allocSize, 0);
        int value = i + 10;
        cudaError_t err = cudaMemset(ptr, value, allocSize);
        ASSERT_EQ(err, cudaSuccess);
        err = cudaDeviceSynchronize();
        ASSERT_EQ(err, cudaSuccess);

        std::vector<unsigned char> hostBuffer(allocSize);
        err = cudaMemcpy(hostBuffer.data(), ptr, allocSize, cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess);
        for (std::size_t j = 0; j < allocSize; ++j) {
            EXPECT_EQ(hostBuffer[j], static_cast<unsigned char>(value));
        }
        memManager.free(ptr, 0);
    }
}

/*
 * Test Case 5: ZeroSizeAllocation
 *
 * This test checks that an allocation request of zero bytes can be made and then
 * freed without throwing an exception.
 */
TEST(CudaMemoryManagerTest, ZeroSizeAllocation)
{
    CudaMemoryManager memManager(512 * 1024 * 1024);
    void* ptr = memManager.malloc(0, 0);
    EXPECT_NO_THROW(memManager.free(ptr, 0));
}

/*
 * Main function for running all GoogleTests.
 */
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}