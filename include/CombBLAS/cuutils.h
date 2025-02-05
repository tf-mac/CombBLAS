#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CHECK_CUSPARSE(func)                                                                                         \
    {                                                                                                                \
        cusparseStatus_t status = (func);                                                                            \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                     \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), \
                   status);                                                                                          \
            return EXIT_FAILURE;                                                                                     \
        }                                                                                                            \
    }
