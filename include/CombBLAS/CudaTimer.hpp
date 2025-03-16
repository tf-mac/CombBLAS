#pragma once

#include "logging.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace combblas {

class CudaTimer {
public:
    CudaTimer();

    ~CudaTimer();

    void start() ;

    float end() ;

private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_end = nullptr;
    float m_elapsed_time = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(CudaTimer);
};

#endif

}
