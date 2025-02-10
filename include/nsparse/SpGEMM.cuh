#ifndef NSPARSE_SPGEMM_CUH
#define NSPARSE_SPGEMM_CUH

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "utils/CSR.h"
#include "utils/Plan.h"
#include "utils/cudautils.h"
#include "utils/nsparse_asm.cuh"

namespace nsparse
{

template <class idType>
__global__ void set_flop_per_row(idType* d_arpt, idType* d_acol, const idType* __restrict__ d_brpt,
                                 long long int* d_flop_per_row, idType nrow)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    idType flop_per_row = 0;
    idType j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        flop_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_flop_per_row[i] = flop_per_row;
}

}  // namespace nsparse

#endif

#endif
