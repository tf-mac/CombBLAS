#include <cuda.h>
#include <cusparse_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <cstdlib>  // for getenv

#include <benchmark/benchmark.h>

#include "nsparse/nsparse.h"
#include "nsparse/utils/cudautils.h"

typedef int IT;
typedef float VT;
using namespace nsparse;

// A benchmark fixture for SpGEMM tests.
class SpGEMMBenchmark : public benchmark::Fixture {
public:
    CSR<IT, VT> a, b, c;
    long long int flop_count;

    // SetUp: called once per benchmark run.
    void SetUp(const ::benchmark::State& state) override {
        // Get matrix file names from environment variables.
        const char* aFile = std::getenv("MATA");
        const char* bFile = std::getenv("MATB");
        if (!aFile || !bFile) {
            std::cerr << "Please set environment variables MATRIX_A and MATRIX_B." << std::endl;
            exit(1);
        }

        std::cout << "Initializing Matrix A from " << aFile << std::endl;
        a.init_data_from_mtx(aFile);
        std::cout << "Initializing Matrix B from " << bFile << std::endl;
        b.init_data_from_mtx(bFile);

        // Copy matrices from host to device.
        a.memcpyHtD();
        b.memcpyHtD();

        // Count the number of floating point operations for SpGEMM.
        get_spgemm_flop(a, b, flop_count);
        std::cout << "Flop count: " << flop_count << std::endl;
    }

    // TearDown: called once per benchmark run.
    void TearDown(const ::benchmark::State& state) override {
        // Release device and host resources.
        a.release_csr();
        b.release_csr();
        c.release_csr();

        a.release_cpu_csr();
        b.release_cpu_csr();
        c.release_cpu_csr();
    }
};

// Benchmark for the full hash-based SpGEMM (both symbolic and numeric phases).
BENCHMARK_F(SpGEMMBenchmark, BM_Hash)(benchmark::State& state) {
    for (auto _ : state) {
        // Release any previous result.
        c.release_csr();

        // Execute full SpGEMM using the hash algorithm.
        SpGEMM_Hash(a, b, c);
        // Ensure GPU computations have completed.
        CUDA_CHECK_CUDART_ERROR(cudaDeviceSynchronize());

        // Tell benchmark how many floating point operations were processed.
        state.SetItemsProcessed(flop_count);
    }
}

// Benchmark for the numeric phase only.
BENCHMARK_F(SpGEMMBenchmark, BM_HashNumeric)(benchmark::State& state) {
    for (auto _ : state) {
        // Release any previous result.
        c.release_csr();

        // Execute only the numeric phase of the hash-based SpGEMM.
        SpGEMM_Hash_Numeric(a, b, c);
        CUDA_CHECK_CUDART_ERROR(cudaDeviceSynchronize());

        // Report the flop count per iteration.
        state.SetItemsProcessed(flop_count);
    }
}

// Main function: initialize and run benchmarks.
int main(int argc, char** argv) {
    // Google Benchmark parses its own command-line flags.
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}