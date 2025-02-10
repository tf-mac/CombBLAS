#include <cuda.h>
#include <cusparse_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <iostream>

#include "nsparse/nsparse.h"
#include "nsparse/utils/cudautils.h"

// typedef int IT;
// #ifdef FLOAT
// typedef float VT;
// #else
// typedef double VT;
// #endif

typedef int IT;
typedef float VT;

using namespace nsparse;

template <class idType, class valType>
void spgemm_hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType>& c, int iternum)
{
    idType i;

    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;

    for (i = 0; i < 2; i++) {
        CUDA_CHECK_CUDART_ERROR(cudaEventCreate(&(event[i])));
    }

    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();

    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);
    std::cerr << "Flop count: " << flop_count * 1e-9 << " GFLOPS" << std::endl;
    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < iternum; i++) {
        if (i > 0) {
            c.release_csr();
        }
        CUDA_CHECK_CUDART_ERROR(cudaEventRecord(event[0], 0));
        SpGEMM_Hash(a, b, c);
        CUDA_CHECK_CUDART_ERROR(cudaEventRecord(event[1], 0));
        CUDA_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
        CUDA_CHECK_CUDART_ERROR(cudaEventElapsedTime(&msec, event[0], event[1]));

        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= iternum - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    /* Numeric Only */
    ave_msec = 0;
    for (i = 0; i < iternum; i++) {
        CUDA_CHECK_CUDART_ERROR(cudaEventRecord(event[0], 0));
        SpGEMM_Hash_Numeric(a, b, c);
        CUDA_CHECK_CUDART_ERROR(cudaEventRecord(event[1], 0));
        CUDA_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
        CUDA_CHECK_CUDART_ERROR(cudaEventElapsedTime(&msec, event[0], event[1]));

        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= iternum - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf(
        "SpGEMM using CSR format (Hash, only numeric phase): %f[GFLOPS], "
        "%f[ms]\n",
        flops, ave_msec);

    c.memcpyDtH();
    c.release_csr();

    // #ifdef sfDEBUG
    //     CSR<IT, VT> cusparse_c;
    //     // SpGEMM_cuSPARSE(a, b, cusparse_c);
    //     if (c == cusparse_c) {
    //         std::cout << "HashSpGEMM is correctly executed" << std::endl;
    //     }
    //     std::cout << "Nnz of A: " << a.nnz << std::endl;
    //     std::cout << "Number of intermediate products: " << flop_count / 2 <<
    //     std::endl; std::cout << "Nnz of C: " << c.nnz << std::endl;
    //     cusparse_c.release_cpu_csr();
    // #endif

    a.release_csr();
    b.release_csr();

    for (i = 0; i < 2; i++) {
        CUDA_CHECK_CUDART_ERROR(cudaEventDestroy(event[i]));
    }
}

/*Main Function*/
int main(int argc, char* argv[])
{
    CSR<IT, VT> a, b, c;
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
    a.init_data_from_mtx(filename);

    std::cout << "Initialize Matrix B" << std::endl;
    std::cout << "Read matrix data from " << argv[2] << std::endl;
    filename = dloc + "/" + argv[2] + "/" + argv[2] + ".mtx";
    b.init_data_from_mtx(filename);
    int iternum = atoi(argv[3]);
    /* Execution of SpGEMM on GPU */
    spgemm_hash(a, b, c, iternum);

    a.release_cpu_csr();
    b.release_cpu_csr();
    c.release_cpu_csr();

    return 0;
}