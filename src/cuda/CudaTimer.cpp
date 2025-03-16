//
// Created by Yuxi Hong on 2/24/25.
//

#ifdef USE_CUDA
#include "CombBLAS/CudaTimer.hpp"
namespace combblas
{

CudaTimer::CudaTimer()
{
    HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&m_start));
    HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&m_end));
}

CudaTimer::~CudaTimer()
{
    if (m_start) {
        HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(m_start));
        m_start = nullptr;
    }

    if (m_end) {
        HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(m_end));
        m_end = nullptr;
    }
}

void CudaTimer::start() { HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(m_start)); }

float CudaTimer::end()
{
    HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(m_end));
    HGEMM_CHECK_CUDART_ERROR(cudaEventSynchronize(m_end));
    HGEMM_CHECK_CUDART_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_end));
    return m_elapsed_time;
}

}  // namespace combblas
#endif