#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

// Helper macro to check CUDA errors.
#define CUDA_CHECK(err)                                                            \
    do {                                                                           \
        cudaError_t err_ = (err);                                                  \
        if (err_ != cudaSuccess) {                                                 \
            std::cerr << "CUDA error (" << __FILE__ << ":" << __LINE__ << "): "    \
                      << cudaGetErrorString(err_) << std::endl;                    \
            MPI_Abort(MPI_COMM_WORLD, err_);                                       \
        }                                                                          \
    } while(0)


// check the CUDA-aware MPI implementation.
// launch this program with 2 mpi processors, e.g. mpirun -np 2 ./test_cudaawarempi


int main(int argc, char** argv) {
    // Initialize MPI.
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // This test requires at least 2 MPI processes.
    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This test requires at least 2 MPI processes." << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    const int count = 10; // number of float elements

    if (rank == 0) {
        // Rank 0: Allocate memory on the GPU.
        float* d_data = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_data, count * sizeof(float)));

        // Initialize device memory using host data.
        float host_init[count];
        for (int i = 0; i < count; ++i) {
            host_init[i] = static_cast<float>(i);
        }
        CUDA_CHECK(cudaMemcpy(d_data, host_init, count * sizeof(float), cudaMemcpyHostToDevice));

        // Send the GPU memory pointer directly via MPI.
        int mpiErr = MPI_Send(d_data, count, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        if (mpiErr == MPI_SUCCESS) {
            std::cout << "Rank 0: MPI_Send with GPU memory succeeded. "
                      << "Your MPI library is CUDA-awared." << std::endl;
        } else {
            std::cerr << "Rank 0: MPI_Send with GPU memory failed." << std::endl;
        }

        // Free the GPU memory.
        CUDA_CHECK(cudaFree(d_data));
    } else if (rank == 1) {
        // Rank 1: Prepare host memory to receive the data.
        
        float host_recv[count] = {0};

        // Receive the data sent from rank 0.
        int mpiErr = MPI_Recv(host_recv, count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (mpiErr == MPI_SUCCESS) {
            std::cout << "Rank 1: MPI_Recv succeeded. Received data: ";
            for (int i = 0; i < count; ++i) {
                std::cout << host_recv[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Rank 1: MPI_Recv failed." << std::endl;
        }
    }

    // Finalize MPI.
    MPI_Finalize();
    return 0;
}