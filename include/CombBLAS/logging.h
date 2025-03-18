#pragma once

// most of the code is from https://github.com/spcl/smat/blob/main/src/cuda_hgemm/src/common/common.h
// thanks!

#include <stdio.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace combblas
{

char *curr_time();
int get_pid();
long int get_tid();

#define HGEMM_LOG_TAG "COMBBLAS"
#define HGEMM_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)

// #define HLOG(format, ...)                                                                                                                   \
// do {                                                                                                                                    \
// fprintf(stderr, "[%s %s %d:%ld %s:%d %s] " format "\n", HGEMM_LOG_TAG, curr_time(), get_pid(), get_tid(), HGEMM_LOG_FILE(__FILE__), \
// __LINE__, __FUNCTION__, ##__VA_ARGS__);                                                                                     \
// } while (0)

#define HLOG(format, ...)                                                                                                              \
    do {                                                                                                                               \
        fprintf(stderr, "[%s %s %s:%d %s] " format "\n", HGEMM_LOG_TAG, curr_time(), HGEMM_LOG_FILE(__FILE__), __LINE__, __FUNCTION__, \
                ##__VA_ARGS__);                                                                                                        \
    } while (0)

#define HGEMM_LIKELY(x) __builtin_expect(!!(x), 1)
#define HGEMM_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define HGEMM_CHECK(x)                    \
    do {                                  \
        if (HGEMM_UNLIKELY(!(x))) {       \
            HLOG("Check failed: %s", #x); \
            exit(EXIT_FAILURE);           \
        }                                 \
    } while (0)

#define HGEMM_CHECK_EQ(x, y) HGEMM_CHECK((x) == (y))
#define HGEMM_CHECK_NE(x, y) HGEMM_CHECK((x) != (y))
#define HGEMM_CHECK_LE(x, y) HGEMM_CHECK((x) <= (y))
#define HGEMM_CHECK_LT(x, y) HGEMM_CHECK((x) < (y))
#define HGEMM_CHECK_GE(x, y) HGEMM_CHECK((x) >= (y))
#define HGEMM_CHECK_GT(x, y) HGEMM_CHECK((x) > (y))

#define HGEMM_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;         \
    void operator=(const TypeName &) = delete

#ifdef USE_CUDA

#define HGEMM_CHECK_CUDART_ERROR(_expr_)                                                                                                            \
    do {                                                                                                                                            \
        cudaError_t _ret_ = _expr_;                                                                                                                 \
        if (HGEMM_UNLIKELY(_ret_ != cudaSuccess)) {                                                                                                 \
            const char *_err_str_ = cudaGetErrorName(_ret_);                                                                                        \
            int _rt_version_ = 0;                                                                                                                   \
            cudaRuntimeGetVersion(&_rt_version_);                                                                                                   \
            int _driver_version_ = 0;                                                                                                               \
            cudaDriverGetVersion(&_driver_version_);                                                                                                \
            HLOG("CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d", static_cast<int>(_ret_), _err_str_, _rt_version_, \
                 _driver_version_);                                                                                                                 \
            exit(EXIT_FAILURE);                                                                                                                     \
        }                                                                                                                                           \
    } while (0)

#define HGEMM_CHECK_CUBLAS_ERROR(_expr_)                                                                  \
    do {                                                                                                  \
        cublasStatus_t _ret_ = _expr_;                                                                    \
        if (HGEMM_UNLIKELY(_ret_ != CUBLAS_STATUS_SUCCESS)) {                                             \
            size_t _rt_version_ = cublasGetCudartVersion();                                               \
            HLOG("CUBLAS API error = %04d, runtime version: %zu", static_cast<int>(_ret_), _rt_version_); \
            exit(EXIT_FAILURE);                                                                           \
        }                                                                                                 \
    } while (0)

#endif

}  // namespace combblas
