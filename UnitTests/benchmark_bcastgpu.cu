#include <cuda_runtime.h>
#include <limits.h>
#include <mpi.h>
#include <nccl.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cxxopts.hpp>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

enum class CommBackend { CPU, MPI_GPU, NCCL };

#define MPI_CHECK(cmd)                                                       \
    do {                                                                     \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS) {                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Helper: Check CUDA errors
#define CUDA_CHECK(cmd)                                                                           \
    do {                                                                                          \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

// Helper: Check NCCL errors
#define NCCL_CHECK(cmd)                                                                           \
    do {                                                                                          \
        ncclResult_t r = cmd;                                                                     \
        if (r != ncclSuccess) {                                                                   \
            fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

__global__ void initarray(double* array, size_t arraylength, double initval)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < arraylength) {
        array[idx] = initval + (double)idx;
    }
}

// one gpu per mpi process
void benchmark_bcast(int myrank, int nprocs, CommBackend backend, int min_bytes, int max_bytes, int n_iters,
                     int n_warmup)
{
    double* device_sendbuf = nullptr;
    double* device_recvbuf = nullptr;
    double* host_sendbuf = nullptr;
    double* host_recvbuf = nullptr;
    ncclComm_t nccl_comm;
    cudaStream_t stream;
    ncclUniqueId id;
    // Seed the random number generator with the current time
    std::srand(std::time(0));
    // Generate a random number between 1 and 100
    double initval = (std::rand() % 100) + 1 + 0.1;

    if (myrank == 0) {
        cerr << "init random number is " << initval << endl;
    }
    MPI_CHECK(MPI_Bcast(&initval, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    cerr << "I received initval " << initval << endl;
    if (backend == CommBackend::NCCL) {
        CUDA_CHECK(cudaSetDevice(myrank));
        CUDA_CHECK(cudaStreamCreate(&stream));
        if (myrank == 0) {
            ncclGetUniqueId(&id);
        }
        MPI_CHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
        NCCL_CHECK(ncclCommInitRank(&nccl_comm, nprocs, id, myrank));
    }
    for (int bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
        size_t count = bytes / sizeof(double);
        if (myrank == 0) cerr << "current benchmark bytes " << bytes << " num of double: " << count << endl;
        if (count == 0) count = 1;
        host_sendbuf = new double[count]();
        host_recvbuf = new double[count]();
        if (backend == CommBackend::CPU && myrank == 0) {
            for (int ti = 0; ti < count; ti++) {
                host_sendbuf[ti] = initval + ti;
            }
        }
        if (backend == CommBackend::MPI_GPU || backend == CommBackend::NCCL) {
            // Allocate device memory

            CUDA_CHECK(cudaMalloc(&device_sendbuf, bytes));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMalloc(&device_recvbuf, bytes));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemset(device_recvbuf, 0, bytes));
            CUDA_CHECK(cudaGetLastError());
            // init data on rank 0
            if (myrank == 0) {
                size_t bsize = (count + 127) / 128;
                cerr << "bsize " << bsize << ", count " << count << ", initval " << initval << endl;
                for (int ti = 0; ti < count; ti++) {
                    host_sendbuf[ti] = initval + ti;
                }
                // initarray<<<128, bsize>>>(device_sendbuf, count, initval);
                // CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(device_sendbuf, host_sendbuf, count * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(host_recvbuf, device_sendbuf, count * sizeof(double), cudaMemcpyDeviceToHost));
                for (int ti = 0; ti < count; ti++) {
                    if (host_recvbuf[ti] != initval + ti) {
                        cerr << "rank 0 set error when using gpu/nccl backend" << endl;
                    }
                }
            }
        }
        switch (backend) {
            case CommBackend::CPU:
                MPI_CHECK(MPI_Bcast(host_sendbuf, count, MPI_DOUBLE, 0, MPI_COMM_WORLD));
                break;
            case CommBackend::MPI_GPU:
                MPI_CHECK(MPI_Bcast(device_sendbuf, count, MPI_DOUBLE, 0, MPI_COMM_WORLD));
                break;
            case CommBackend::NCCL:
                NCCL_CHECK(ncclBroadcast(device_sendbuf, device_recvbuf, count, ncclDouble, 0, nccl_comm, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                break;
        }
        // Warmup
        for (int i = 0; i < n_warmup; ++i) {
        }
        // correctness check
        int correctval = 1;
        if (backend == CommBackend::MPI_GPU) {
            CUDA_CHECK(cudaMemcpy(host_recvbuf, device_sendbuf, count * sizeof(double), cudaMemcpyDeviceToHost));
        } else if (backend == CommBackend::NCCL) {
            CUDA_CHECK(cudaMemcpy(host_recvbuf, device_recvbuf, count * sizeof(double), cudaMemcpyDeviceToHost));
        } else if (backend == CommBackend::CPU) {
            memcpy(host_recvbuf, host_sendbuf, count * sizeof(double));
        }
        for (int ti = 0; ti < count; ti++) {
            if (host_recvbuf[ti] != ti + initval) {
                printf("rank %d error idx %d expected %.3f actual %.3f\n", myrank, ti, ti + initval, host_recvbuf[ti]);
                correctval = 0;
                break;
            }
        }
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &correctval, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD));
        if (correctval) {
            if (myrank == 0) cerr << "results are correct and we start benchmark!" << endl;
            // Timed iterations
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_iters; ++i) {
                switch (backend) {
                    case CommBackend::CPU:
                        MPI_Bcast(host_sendbuf, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        break;
                    case CommBackend::MPI_GPU:
                        MPI_Bcast(device_sendbuf, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        break;
                    case CommBackend::NCCL:
                        NCCL_CHECK(
                            ncclBroadcast(device_sendbuf, device_recvbuf, count, ncclDouble, 0, nccl_comm, stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        break;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            double avg_time = elapsed / n_iters;
            double bandwidth = bytes / avg_time / 1024.0 / 1024.0;  // MB/s

            if (myrank == 0) {
                std::string backend_str = (backend == CommBackend::CPU)       ? "CPU"
                                          : (backend == CommBackend::MPI_GPU) ? "MPI_GPU"
                                                                              : "NCCL";
                std::cout << "Backend: " << backend_str << ", Size: " << (double)bytes / 1024.0 << " KB"
                          << ", Time: " << avg_time * 1e3 << " ms"
                          << ", Bandwidth: " << bandwidth << " MB/s" << std::endl;
            }
        }
        if (backend == CommBackend::MPI_GPU || backend == CommBackend::NCCL) {
            CUDA_CHECK(cudaFree(device_sendbuf));
            CUDA_CHECK(cudaFree(device_recvbuf));
        }

        delete[] host_sendbuf;
        delete[] host_recvbuf;
        // break;  // for test
    }

    if (backend == CommBackend::NCCL) {
        ncclCommDestroy(nccl_comm);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (myrank == 0) cerr << "Nprocs: " << nprocs << endl;

    // clang-format off
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
    ("Warmup", "warmup rounds", cxxopts::value<int>()->default_value("5"))
    ("TimingIter", "timing rounds", cxxopts::value<int>()->default_value("20"))
    ("MinBytes", "minimum bytes", cxxopts::value<int>()->default_value("1024")) // 1024  = 1 K Bytes
    ("MaxBytes", "maximum bytes", cxxopts::value<int>()->default_value("67108864")) // 64 * 1024 * 1024 = 80 M Bytes
    ("CommBackend", "comm backend, cpu | gpu | nccl", cxxopts::value<string>()->default_value("nccl"))
    ;
    // clang-format on
    auto result = options.parse(argc, argv);
    if (myrank == 0) {
        // Print all parsed arguments
        std::cerr << "Parsed options:" << std::endl;
        for (const auto& kv : result.arguments()) {
            std::cerr << "  --" << kv.key() << " = " << kv.value() << std::endl;
        }
    }

    // Default params
    int min_bytes = result["MinBytes"].as<int>();
    int max_bytes = result["MaxBytes"].as<int>();
    int n_iters = result["TimingIter"].as<int>();
    int n_warmup = result["Warmup"].as<int>();
    string commbackend = result["CommBackend"].as<string>();
    CommBackend backend = CommBackend::NCCL;
    // Simple backend argument parsing
    if (commbackend == "cpu") {
        backend = CommBackend::CPU;
        if (myrank == 0) cerr << "backend use mpi cpu" << endl;
    } else if (commbackend == "gpu") {
        backend = CommBackend::MPI_GPU;
        if (myrank == 0) cerr << "backend use mpi gpu" << endl;
    } else if (commbackend == "nccl") {
        backend = CommBackend::NCCL;
        if (myrank == 0) cerr << "backend use nccl" << endl;
    }

    benchmark_bcast(myrank, nprocs, backend, min_bytes, max_bytes, n_iters, n_warmup);

    // finalizing NCCL

    MPI_Finalize();
    return 0;
}
