#ifdef USE_CUDA
#include <thrust/device_vector.h>

#include "nsparse/BIN.cuh"
#include "nsparse/HashSpGEMM_Volta.cuh"
#include "nsparse/SpGEMM.cuh"
#include "nsparse/nsparse.h"
#include "nsparse/utils/CSR.h"
#include "nsparse/utils/cudautils.h"
#include "nsparse/utils/def.h"
namespace nsparse
{

/* Compare the vectors */
template <class idType, class valType>
void check_answer(valType* csr_ans, valType* ans_vec, idType nrow)
{
    idType i;
    int total_fail = 10;
    valType delta, base;
    valType scale;
    if (typeid(valType) == typeid(float)) {
        scale = 1000;
    } else {
        scale = 1000 * 1000;
    }

    for (i = 0; i < nrow; i++) {
        delta = ans_vec[i] - csr_ans[i];
        base = ans_vec[i];

        if (delta < 0) {
            delta *= -1;
        }
        if (base < 0) {
            base *= -1;
        }
        if (delta * 100 * scale > base) {
            printf("i=%d, ans=%e, csr=%e, delta=%e\n", i, ans_vec[i], csr_ans[i], delta);
            total_fail--;
            if (total_fail == 0) {
                break;
            }
        }
    }
    if (total_fail != 10) {
        printf("Calculation Result is Incorrect\n");
    } else {
        printf("Calculation Result is Correct\n");
    }
}

template <class idType, class valType>
void init_vector(valType* x, int row)
{
    int i;

    srand48((unsigned)time(NULL));

    for (i = 0; i < row; i++) {
        x[i] = drand48();
    }
}

template <class idType, class valType>
void get_spgemm_flop(CSR<idType, valType> a, CSR<idType, valType> b, long long int& flop)
{
    int GS, BS;
    long long int* d_flop_per_row;

    BS = MAX_LOCAL_THREAD_NUM;
    CUDA_CHECK_CUDART_ERROR(
        cudaMalloc((void**)&(d_flop_per_row), sizeof(long long int) * (1 + a.nrow)));

    GS = div_ceil((size_t)a.nrow, (size_t)BS);
    set_flop_per_row<<<GS, BS>>>(a.d_rpt, a.d_colids, b.d_rpt, d_flop_per_row, a.nrow);

    long long int* tmp = (long long int*)malloc(sizeof(long long int) * a.nrow);
    cudaMemcpy(tmp, d_flop_per_row, sizeof(long long int) * a.nrow, cudaMemcpyDeviceToHost);
    flop = thrust::reduce(thrust::device, d_flop_per_row, d_flop_per_row + a.nrow);

    flop *= 2;
    cudaFree(d_flop_per_row);
}

template void get_spgemm_flop<int, float>(CSR<int, float> a, CSR<int, float> b,
                                          long long int& flop);
template void get_spgemm_flop<int, double>(CSR<int, double> a, CSR<int, double> b,
                                           long long int& flop);
// template void get_spgemm_flop<int64_t, float>(CSR<int64_t, float> a,
// CSR<int64_t, float> b,
//                                               long long int &flop);
// template void get_spgemm_flop<int64_t, double>(CSR<int64_t, double> a,
// CSR<int64_t, double> b,
//                                                long long int &flop);

template void SpGEMM_Hash<int, float>(CSR<int, float> a, CSR<int, float> b, CSR<int, float>& c);
template void SpGEMM_Hash<int, double>(CSR<int, double> a, CSR<int, double> b, CSR<int, double>& c);
// template void SpGEMM_Hash<int64_t, float>(CSR<int64_t, float> a, CSR<int64_t,
// float> b,
//                                           CSR<int64_t, float> &c);
// template void SpGEMM_Hash<int64_t, double>(CSR<int64_t, double> a,
// CSR<int64_t, double> b,
//                                            CSR<int64_t, double> &c);

template void SpGEMM_Hash_Numeric<int, float>(CSR<int, float> a, CSR<int, float> b,
                                              CSR<int, float>& c);
template void SpGEMM_Hash_Numeric<int, double>(CSR<int, double> a, CSR<int, double> b,
                                               CSR<int, double>& c);
// template void SpGEMM_Hash_Numeric<int64_t, float>(CSR<int64_t, float> a,
// CSR<int64_t, float> b,
//                                                   CSR<int64_t, float> &c);
// template void SpGEMM_Hash_Numeric<int64_t, double>(CSR<int64_t, double> a,
// CSR<int64_t, double> b,
//                                                    CSR<int64_t, double> &c);

}  // namespace nsparse

#endif  // USE_CUDA
