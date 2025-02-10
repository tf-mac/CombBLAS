/******************************************************************************
* Combined SpGEMM Example using cuSPARSE APIs and CUDA event timing.
*
* This file demonstrates three variants of sparse matrix-matrix multiplication:
*
* 1. Basic SpGEMM (one-shot API):
*    - Implements a complete SpGEMM operation by performing both the symbolic 
*      phase (sparsity analysis) and the numeric phase (computing nonzero values)
*      in one call.
*    - Suitable for single-shot computations when no workspace reuse is desired.
*
* 2. Memory Optimized SpGEMM:
*    - Focuses on workspace management by querying and reusing scratch memory.
*    - In this example, it simply calls the basicSpGEMM routine as a placeholder.
*
* 3. Reuse Analysis SpGEMM:
*    - Performs the symbolic phase (analysis) once and then reuses it for 
*      multiple numeric phases.
*    - Useful in iterative solvers or when the sparsity pattern does not change.
*
* Benchmarking:
*   - Each SpGEMM variant is run repeatedly (with the number of iterations provided
*     as a command-line argument), skipping the first (warm-up) iteration.
*   - CUDA events (cudaEventRecord/cudaEventElapsedTime) are used to measure the 
*     execution time of the SpGEMM routines.
*   - The average execution time (in ms) is printed for each method.
*
* Usage:
*   combined_spgemm <matrix A file> <matrix B file> <benchmark iterations>
*
* Note:
*   This code uses your provided CSR class (template <class idType, class valType> CSR)
*   which implements the matrix input function init_data_from_mtx as well as device- and 
*   host-memory management (memcpyHtD, memcpyDtH, release_cpu_csr, release_csr).
*
* Compilation (example):
*   nvcc -arch=sm_70 -o combined_spgemm combined_spgemm.cpp -lcusparse -lcudart
*
******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <string>
#include <sys/time.h>

#include "nsparse/nsparse.h"
#include "nsparse/utils/cudautils.h"

using namespace nsparse;

// Assuming your CSR class is declared in a header (or here directly).
// For this example, we assume the following typedefs:
typedef int IT;
typedef float VT;

// Returns the current time in seconds (including microseconds as fractional part).
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}



double basic_p1_workest = 0.0;
double basic_p2_compute = 0.0;
double basic_p3_genc = 0.0;

//------------------------------------------------------------------------------
// Basic SpGEMM using the one-shot API (similar to spgemm_example.c)
void basicSpGEMM(cusparseHandle_t handle, CSR<IT, VT>& A, CSR<IT, VT>& B) {
    basic_p1_workest = basic_p2_compute = basic_p3_genc = 0.0;
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseStatus_t status;

    VT alpha = 1.0f;
    VT beta  = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // Create descriptors for matrices A and B using their device pointers.
    status = cusparseCreateCsr(&matA, A.nrow, A.ncolumn, A.nnz,
                            A.d_rpt, A.d_colids, A.d_values,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseCreateCsr(&matB, B.nrow, B.ncolumn, B.nnz,
                            B.d_rpt, B.d_colids, B.d_values,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // Create an empty descriptor for C.
    IT C_num_rows = A.nrow;
    IT C_num_cols = B.ncolumn;
    status = cusparseCreateCsr(&matC, C_num_rows, C_num_cols, 0,
                            nullptr, nullptr, nullptr,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // Create the SpGEMM descriptor.
    status = cusparseSpGEMM_createDescr(&spgemmDesc);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    basic_p1_workest -= getCurrentTime();
    // Phase 1: Work estimation.
    size_t bufferSize1 = 0, bufferSize2 = 0;
    void *dBuffer1 = nullptr, *dBuffer2 = nullptr;
    status = cusparseSpGEMM_workEstimation(handle,
                                        opA, opB,
                                        &alpha, matA, matB, &beta,
                                        matC, CUDA_R_32F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize1, nullptr);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    CUDA_CHECK_CUDART_ERROR(cudaMalloc(&dBuffer1, bufferSize1));
    status = cusparseSpGEMM_workEstimation(handle,
                                        opA, opB,
                                        &alpha, matA, matB, &beta,
                                        matC, CUDA_R_32F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize1, dBuffer1);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    basic_p1_workest += getCurrentTime();
    // Phase 2: Compute the SpGEMM product.
    basic_p2_compute -= getCurrentTime();
    status = cusparseSpGEMM_compute(handle,
                                    opA, opB,
                                    &alpha, matA, matB, &beta,
                                    matC, CUDA_R_32F,
                                    CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc,
                                    &bufferSize2, nullptr);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    CUDA_CHECK_CUDART_ERROR(cudaMalloc(&dBuffer2, bufferSize2));
    status = cusparseSpGEMM_compute(handle,
                                    opA, opB,
                                    &alpha, matA, matB, &beta,
                                    matC, CUDA_R_32F,
                                    CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc,
                                    &bufferSize2, dBuffer2);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    basic_p2_compute += getCurrentTime();
    // Query the size of the resulting C.
    basic_p3_genc -= getCurrentTime();
    int64_t numRows, numCols, nnzC;
    status = cusparseSpMatGetSize(matC, &numRows, &numCols, &nnzC);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    // printf("Basic SpGEMM: C has %d nonzeros\n", (int)nnzC);

    // Allocate memory for C.
    int *dC_rpt, *dC_columns;
    VT *dC_values;
    CUDA_CHECK_CUDART_ERROR(cudaMalloc(&dC_rpt, sizeof(IT) * (C_num_rows + 1)));
    CUDA_CHECK_CUDART_ERROR(cudaMalloc(&dC_columns, sizeof(IT) * nnzC));
    CUDA_CHECK_CUDART_ERROR(cudaMalloc(&dC_values, sizeof(VT) * nnzC));
    status = cusparseCsrSetPointers(matC, dC_rpt, dC_columns, dC_values);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // Phase 3: Copy the computed values into C.
    status = cusparseSpGEMM_copy(handle,
                                opA, opB,
                                &alpha, matA, matB, &beta,
                                matC, CUDA_R_32F,
                                CUSPARSE_SPGEMM_DEFAULT,
                                spgemmDesc);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // Clean up.
    status = cusparseSpGEMM_destroyDescr(spgemmDesc);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseDestroySpMat(matA);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseDestroySpMat(matB);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseDestroySpMat(matC);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    CUDA_CHECK_CUDART_ERROR(cudaFree(dBuffer1));
    CUDA_CHECK_CUDART_ERROR(cudaFree(dBuffer2));
    CUDA_CHECK_CUDART_ERROR(cudaFree(dC_rpt)); 
    CUDA_CHECK_CUDART_ERROR(cudaFree(dC_columns)); 
    CUDA_CHECK_CUDART_ERROR(cudaFree(dC_values));
    basic_p3_genc += getCurrentTime();
}

//------------------------------------------------------------------------------
// Memory Optimized SpGEMM: calls basicSpGEMM as a placeholder.
void memOptimizedSpGEMM(cusparseHandle_t handle, CSR<IT, VT>& A, CSR<IT, VT>& B) {
    // std::cout << "Memory Optimized SpGEMM: Using basicSpGEMM implementation." << std::endl;
    // basicSpGEMM(handle, A, B);
}


double reuse_p1_workest = 0.0;
double reuse_p2_compute = 0.0;
double reuse_p3_genc = 0.0;

/*
 * Typical Use Cases for Reusing the Symbolic Analysis Phase in Sparse Matrix-Matrix Multiplication:
 *
 * 1. Iterative Solvers for PDEs:
 *    In many scientific simulations (e.g., finite element or finite difference methods),
 *    the underlying mesh or grid structure remains constant across iterations while the
 *    numerical coefficients (e.g., material properties) vary. Reusing the symbolic analysis
 *    avoids redundant work on the fixed sparsity pattern.
 *
 * 2. Preconditioning and Multigrid Methods:
 *    When constructing or updating preconditioners (such as ILU or multigrid hierarchies),
 *    the matrix structure often remains unchanged. Reusing the symbolic phase here reduces
 *    computational overhead when only numerical values are updated.
 *
 * 3. Graph Analytics and Network Analysis:
 *    Algorithms that operate on graph structures (e.g., adjacency matrices) benefit when the
 *    graph connectivity is constant, even if edge weights or other attributes change. This
 *    allows for efficient updates using the precomputed sparsity pattern.
 *
 * 4. Time-Dependent Simulations:
 *    In transient simulations like fluid flow or heat transfer, the domain connectivity is fixed,
 *    while the associated physical quantities update over time. Reusing the symbolic analysis can
 *    significantly improve performance by focusing only on the numeric updates.
 *
 * In summary, the reuse of the symbolic analysis phase is ideal when the sparsity structure
 * remains constant across multiple multiplications, allowing for efficient numerical updates
 * without the need to recompute the entire analysis.
 */

//------------------------------------------------------------------------------
// Reuse Analysis SpGEMM: reuse the symbolic phase for multiple numeric computations.
double reuseAnalysisSpGEMM(cusparseHandle_t handle, CSR<IT, VT>& A, CSR<IT, VT>& B, int iterations) {
    reuse_p1_workest = 0.0;
    reuse_p1_workest -= getCurrentTime();
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseStatus_t status;

    VT alpha = 1.0f;
    VT beta  = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // Create descriptors for A and B.
    status = cusparseCreateCsr(&matA, A.nrow, A.ncolumn, A.nnz,
                            A.d_rpt, A.d_colids, A.d_values,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseCreateCsr(&matB, B.nrow, B.ncolumn, B.nnz,
                            B.d_rpt, B.d_colids, B.d_values,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    IT C_num_rows = A.nrow;
    IT C_num_cols = B.ncolumn;
    status = cusparseCreateCsr(&matC, C_num_rows, C_num_cols, 0,
                            nullptr, nullptr, nullptr,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // Create the SpGEMM descriptor.
    status = cusparseSpGEMM_createDescr(&spgemmDesc);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // Symbolic phase: work estimation (done once).
    size_t bufferSize1 = 0;
    void *dBuffer1 = nullptr;
    status = cusparseSpGEMM_workEstimation(handle,
                                        opA, opB,
                                        &alpha, matA, matB, &beta,
                                        matC, CUDA_R_32F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize1, nullptr);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    CUDA_CHECK_CUDART_ERROR(cudaMalloc(&dBuffer1, bufferSize1));
    status = cusparseSpGEMM_workEstimation(handle,
                                        opA, opB,
                                        &alpha, matA, matB, &beta,
                                        matC, CUDA_R_32F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize1, dBuffer1);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    reuse_p1_workest += getCurrentTime();
    // Benchmark numeric phase over multiple iterations.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_ms = 0.0f;
    double totalTime = 0.0;

    for (int iter = 0; iter < iterations + 1; iter++) {
        size_t bufferSize2 = 0;
        void *dBuffer2 = nullptr;
        cudaEventRecord(start, 0);
        if(iter > 0) reuse_p2_compute -= getCurrentTime();
        status = cusparseSpGEMM_compute(handle,
                                        opA, opB,
                                        &alpha, matA, matB, &beta,
                                        matC, CUDA_R_32F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize2, nullptr);
        assert(status == CUSPARSE_STATUS_SUCCESS);
        cudaMalloc(&dBuffer2, bufferSize2);
        status = cusparseSpGEMM_compute(handle,
                                        opA, opB,
                                        &alpha, matA, matB, &beta,
                                        matC, CUDA_R_32F,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize2, dBuffer2);
        assert(status == CUSPARSE_STATUS_SUCCESS);
        if(iter > 0) reuse_p2_compute += getCurrentTime();
        if(iter > 0) reuse_p3_genc -= getCurrentTime();
        int64_t numRows, numCols, nnzC;
        status = cusparseSpMatGetSize(matC, &numRows, &numCols, &nnzC);
        assert(status == CUSPARSE_STATUS_SUCCESS);
        int *dC_rpt, *dC_columns;
        VT *dC_values;
        cudaMalloc(&dC_rpt, sizeof(IT) * (C_num_rows + 1));
        cudaMalloc(&dC_columns, sizeof(IT) * nnzC);
        cudaMalloc(&dC_values, sizeof(VT) * nnzC);
        status = cusparseCsrSetPointers(matC, dC_rpt, dC_columns, dC_values);
        assert(status == CUSPARSE_STATUS_SUCCESS);
        status = cusparseSpGEMM_copy(handle,
                                    opA, opB,
                                    &alpha, matA, matB, &beta,
                                    matC, CUDA_R_32F,
                                    CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc);
        assert(status == CUSPARSE_STATUS_SUCCESS);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        if(iter > 0) { // skip warm-up iteration
            totalTime += elapsed_ms;
        }
        if(iter > 0) reuse_p3_genc += getCurrentTime();
        CUDA_CHECK_CUDART_ERROR(cudaFree(dBuffer2));
        CUDA_CHECK_CUDART_ERROR(cudaFree(dC_rpt)); 
        CUDA_CHECK_CUDART_ERROR(cudaFree(dC_columns)); 
        CUDA_CHECK_CUDART_ERROR(cudaFree(dC_values));
    }
    CUDA_CHECK_CUDART_ERROR(cudaEventDestroy(start));
    CUDA_CHECK_CUDART_ERROR(cudaEventDestroy(stop));

    // Clean up.
    status = cusparseSpGEMM_destroyDescr(spgemmDesc);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseDestroySpMat(matA);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseDestroySpMat(matB);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    status = cusparseDestroySpMat(matC);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    CUDA_CHECK_CUDART_ERROR(cudaFree(dBuffer1));
    reuse_p2_compute /= iterations;
    reuse_p3_genc /= iterations;
    return totalTime / iterations;
}

//------------------------------------------------------------------------------
// Benchmark wrapper for Basic and Memory Optimized SpGEMM.
// Runs the given SpGEMM routine (with a warm-up iteration skipped) and returns
// the average execution time (in ms).
template <typename SpGEMMFunc>
double benchmarkSpGEMM(SpGEMMFunc spgemmFunc, cusparseHandle_t handle, CSR<IT, VT>& A, CSR<IT, VT>& B, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_ms = 0.0f;
    double totalTime = 0.0;
    for (int iter = 0; iter < iterations + 1; iter++) {
        cudaEventRecord(start, 0);
        spgemmFunc(handle, A, B);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        if(iter > 0) {
            totalTime += elapsed_ms;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return totalTime / iterations;
}

//------------------------------------------------------------------------------
// Main: Reads matrices using the CSR class, sets up cuSPARSE, and benchmarks each API variant.
int main(int argc, char* argv[]) {
    CSR<IT, VT> A,B;
    std::string dloc = std::getenv("DLOC");
    if(dloc == ""){
        std::cerr << "Please set DLOC environment variable" << std::endl;
        exit(1);
    }else{
        std::cerr << "DLOC: " << dloc << std::endl;
    }
    /* Set CSR reding from MM file or generating random matrix */
    std::cout << "Initialize Matrix A" << std::endl;
    std::cout << "Read matrix data from " << argv[1] << std::endl;
    std::string filename = dloc + "/" + argv[1] + "/" + argv[1] + ".mtx";
    A.init_data_from_mtx(filename);

    std::cout << "Initialize Matrix B" << std::endl;
    std::cout << "Read matrix data from " << argv[2] << std::endl;
    filename = dloc + "/" + argv[2] + "/" + argv[2] + ".mtx";
    B.init_data_from_mtx(filename);
    int iterations = atoi(argv[3]);
    
    // Copy matrices from Host to Device.
    A.memcpyHtD();
    B.memcpyHtD();
    
    // Create cuSPARSE handle.
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    std::cerr << "======================== " << std::endl;
    std::cout << "Benchmarking Basic SpGEMM..." << std::endl;
    double avgTimeBasic = benchmarkSpGEMM(basicSpGEMM, handle, A, B, iterations);
    std::printf("Average time for Basic SpGEMM: %.3f ms\n", avgTimeBasic);
    std::cerr << "basic reuse_p1_workest: " << basic_p1_workest * 1e3 << " ms" << std::endl;
    std::cerr << "basic reuse_p2_compute: " << basic_p2_compute * 1e3 << " ms" << std::endl;
    std::cerr << "basic reuse_p3_genc: " << basic_p3_genc * 1e3 << " ms" << std::endl;

    std::cerr << "======================== " << std::endl;
    
    std::cout << "Benchmarking Memory Optimized SpGEMM..." << std::endl;
    double avgTimeMemOpt = benchmarkSpGEMM(memOptimizedSpGEMM, handle, A, B, iterations);
    std::printf("Average time for Memory Optimized SpGEMM: %.3f ms\n", avgTimeMemOpt);
    
    std::cerr << "======================== " << std::endl;
    std::cout << "Benchmarking Reuse Analysis SpGEMM (numeric phase)..." << std::endl;
    double avgTimeReuse = reuseAnalysisSpGEMM(handle, A, B, iterations);
    std::printf("Average time for Reuse Analysis SpGEMM (numeric phase): %.3f ms\n", avgTimeReuse);
    std::cerr << "reuse_p1_workest: " << reuse_p1_workest * 1e3 << " ms" << std::endl;
    std::cerr << "reuse_p2_compute: " << reuse_p2_compute * 1e3 << " ms" << std::endl;
    std::cerr << "reuse_p3_genc: " << reuse_p3_genc * 1e3 << " ms" << std::endl;
    
    cusparseDestroy(handle);
    
    // Release CSR host memory.
    A.release_cpu_csr();
    B.release_cpu_csr();
    
    return EXIT_SUCCESS;
}