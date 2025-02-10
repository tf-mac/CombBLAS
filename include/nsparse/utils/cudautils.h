#ifndef CUDA_UTILS_ALL_IN_ONE
#define CUDA_UTILS_ALL_IN_ONE

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

inline char* curr_time()
{
    time_t raw_time = time(nullptr);
    struct tm* time_info = localtime(&raw_time);
    static char now_time[64];
    now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S", time_info)] = '\0';

    return now_time;
}

inline int get_pid()
{
    static int pid = getpid();

    return pid;
}

inline long int get_tid()
{
    thread_local long int tid = syscall(SYS_gettid);

    return tid;
}

#define CUDA_LOG_TAG "CUDA"
#define CUDA_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define HLOG(format, ...)                                                                  \
    do {                                                                                   \
        fprintf(stderr, "[%s %s %d:%ld %s:%d %s] " format "\n", CUDA_LOG_TAG, curr_time(), \
                get_pid(), get_tid(), CUDA_LOG_FILE(__FILE__), __LINE__, __FUNCTION__,     \
                ##__VA_ARGS__);                                                            \
    } while (0)

// plain C++ version of get time
inline double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

#define CUDA_LIKELY(x) __builtin_expect(!!(x), 1)
#define CUDA_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define CUDA_CHECK(x)                     \
    do {                                  \
        if (CUDA_UNLIKELY(!(x))) {        \
            HLOG("Check failed: %s", #x); \
            exit(EXIT_FAILURE);           \
        }                                 \
    } while (0)

#define CUDA_CHECK_EQ(x, y) CUDA_CHECK((x) == (y))
#define CUDA_CHECK_NE(x, y) CUDA_CHECK((x) != (y))
#define CUDA_CHECK_LE(x, y) CUDA_CHECK((x) <= (y))
#define CUDA_CHECK_LT(x, y) CUDA_CHECK((x) < (y))
#define CUDA_CHECK_GE(x, y) CUDA_CHECK((x) >= (y))
#define CUDA_CHECK_GT(x, y) CUDA_CHECK((x) > (y))

#define CUDA_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&) = delete;         \
    void operator=(const TypeName&) = delete

#define CUDA_CHECK_CUDART_ERROR(_expr_)                                              \
    do {                                                                             \
        cudaError_t _ret_ = _expr_;                                                  \
        if (CUDA_UNLIKELY(_ret_ != cudaSuccess)) {                                   \
            const char* _err_str_ = cudaGetErrorName(_ret_);                         \
            int _rt_version_ = 0;                                                    \
            cudaRuntimeGetVersion(&_rt_version_);                                    \
            int _driver_version_ = 0;                                                \
            cudaDriverGetVersion(&_driver_version_);                                 \
            HLOG(                                                                    \
                "CUDA Runtime API error = %04d \"%s\", runtime version: %d, "        \
                "driver version: %d",                                                \
                static_cast<int>(_ret_), _err_str_, _rt_version_, _driver_version_); \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

#define CUDA_CHECK_CUBLAS_ERROR(_expr_)                                                    \
    do {                                                                                   \
        cublasStatus_t _ret_ = _expr_;                                                     \
        if (CUDA_UNLIKELY(_ret_ != CUBLAS_STATUS_SUCCESS)) {                               \
            size_t _rt_version_ = cublasGetCudartVersion();                                \
            HLOG("CUBLAS API error = %04d, runtime version: %zu", static_cast<int>(_ret_), \
                 _rt_version_);                                                            \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

class CudaTimer
{
   public:
    CudaTimer()
    {
        CUDA_CHECK_CUDART_ERROR(cudaEventCreate(&m_start));
        CUDA_CHECK(m_start);
        CUDA_CHECK_CUDART_ERROR(cudaEventCreate(&m_end));
        CUDA_CHECK(m_end);
    }

    ~CudaTimer()
    {
        if (m_start) {
            CUDA_CHECK_CUDART_ERROR(cudaEventDestroy(m_start));
            m_start = nullptr;
        }

        if (m_end) {
            CUDA_CHECK_CUDART_ERROR(cudaEventDestroy(m_end));
            m_end = nullptr;
        }
    }

    void start() { CUDA_CHECK_CUDART_ERROR(cudaEventRecord(m_start)); }

    float end()
    {
        CUDA_CHECK_CUDART_ERROR(cudaEventRecord(m_end));
        CUDA_CHECK_CUDART_ERROR(cudaEventSynchronize(m_end));
        CUDA_CHECK_CUDART_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_end));

        return m_elapsed_time;
    }

   private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_end = nullptr;
    float m_elapsed_time = 0.0;

    CUDA_DISALLOW_COPY_AND_ASSIGN(CudaTimer);
};

template <typename IT>
inline __device__ __host__ IT div_ceil(IT a, IT b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Beginning of GPU Architecture definitions
inline int convert_SM_to_cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the
    // # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
                 // minor version
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
        {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
        {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run
    // properly
    HLOG("MapSMtoCores for SM %d.%d is undefined. Default to use %d cores/SM", major, minor,
         nGpuArchCoresPerSM[index - 1].cores);

    return nGpuArchCoresPerSM[index - 1].cores;
}

inline const char* convert_SM_to_arch_name(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the
    // GPU Arch name)
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
                 // minor version
        const char* name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {
        {0x30, "Kepler"},  {0x32, "Kepler"},  {0x35, "Kepler"},       {0x37, "Kepler"},
        {0x50, "Maxwell"}, {0x52, "Maxwell"}, {0x53, "Maxwell"},      {0x60, "Pascal"},
        {0x61, "Pascal"},  {0x62, "Pascal"},  {0x70, "Volta"},        {0x72, "Xavier"},
        {0x75, "Turing"},  {0x80, "Ampere"},  {0x86, "Ampere"},       {0x87, "Ampere"},
        {0x89, "Ada"},     {0x90, "Hopper"},  {-1, "Graphics Device"}};

    int index = 0;

    while (nGpuArchNameSM[index].SM != -1) {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchNameSM[index].name;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run
    // properly
    HLOG("MapSMtoArchName for SM %d.%d is undefined. Default to use %s", major, minor,
         nGpuArchNameSM[index - 1].name);

    return nGpuArchNameSM[index - 1].name;
}

#endif  // CUDA_UTILS_ALL_IN_ONE
