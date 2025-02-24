#ifndef _mtSpGEMM_h
#define _mtSpGEMM_h

#include <omp.h>

#include <tuple>
#include <vector>

namespace combblas
{
/*
 Multithreaded prefix sum
 Inputs:
    in: an input array
    size: the length of the input array "in"
    nthreads: number of threads used to compute the prefix sum

 Output:
    return an array of size "size+1"
    the memory of the output array is allocated internallay

 Example:

    in = [2, 1, 3, 5]
    out = [0, 2, 3, 6, 11]
 */
template <typename T>
T* prefixsum(T* in, int size, int nthreads);

// multithreaded HeapSpGEMM
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO>* LocalSpGEMM(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, bool clearA, bool clearB);

template <typename IT, typename NT>
bool sort_less(const std::pair<IT, NT>& left, const std::pair<IT, NT>& right);

// Hybrid approach of multithreaded HeapSpGEMM and HashSpGEMM
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO>* LocalHybridSpGEMM(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, bool clearA, bool clearB,
                                     IT* aux = nullptr);

#ifdef GPU_ENABLED

template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO>* LocalHybridSpGEMM_CUDA(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, bool clearA,
                                          bool clearB, IT* aux = nullptr)
{
    IT mdim = A.getnrow();
    IT ndim = B.getncol();
    IT nnzA = A.getnnz();
    if (A.isZero() || B.isZero()) {
        return new SpTuples<IT, NTO>(0, mdim, ndim);
    }

    Dcsc<IT, NT1>* Adcsc = A.GetDCSC();
    Dcsc<IT, NT2>* Bdcsc = B.GetDCSC();
    IT nA = A.getncol();
    float cf = static_cast<float>(nA + 1) / static_cast<float>(Adcsc->nzc);
    IT csize = static_cast<IT>(ceil(cf));  // chunk size
    bool deleteAux = false;
    if (aux == nullptr) {
        deleteAux = true;
        Adcsc->ConstructAux(nA, aux);
    }

    int numThreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif

    IT* flopC = estimateFLOP(A, B, aux);

    IT* colnnzC = estimateNNZ_Hash(A, B, flopC, aux);
    IT* flopptr = prefixsum<IT>(flopC, Bdcsc->nzc, numThreads);
    IT flop = flopptr[Bdcsc->nzc];
    IT* colptrC = prefixsum<IT>(colnnzC, Bdcsc->nzc, numThreads);
    delete[] colnnzC;
    delete[] flopC;
    IT nnzc = colptrC[Bdcsc->nzc];

    std::tuple<IT, IT, NTO>* tuplesC =
        static_cast<std::tuple<IT, IT, NTO>*>(::operator new(sizeof(std::tuple<IT, IT, NTO>[nnzc])));

    std::vector<std::vector<std::pair<IT, IT>>> colindsVec(numThreads);

    std::vector<std::vector<std::pair<IT, NTO>>> globalHashVecAll(numThreads);
    std::vector<std::vector<HeapEntry<IT, NT1>>> globalHeapVecAll(numThreads);

    SpDCCols<IT, NT1> A_Tran = A.TransposeConst();
    SpDCCols<IT, NT1> B_Tran = B.TransposeConst();

    Dcsc<IT, NT1>* Adcsc_Tran = A_Tran.GetDCSC();
    IT* A_Tran_CP;
    IT* A_Tran_IR;
    IT* A_Tran_JC;
    NT1* A_Tran_numx;
    IT* B_CP;
    IT* B_IR;
    IT* B_JC;
    NT2* B_numx;
    std::tuple<IT, IT, NTO>* tuplesC_d;
    IT* tuplesC_d_o;
    IT* tuplesC_d_t;
    NTO* tuplesC_d_th;
    uint* colptr_size_d;
    uint* curptr_d;
    IT* colptrC_d;
    cudaMalloc((void**)&curptr_d, sizeof(uint));
    cudaMalloc((void**)&tuplesC_d_o, (sizeof(IT[nnzc])));
    cudaMalloc((void**)&tuplesC_d_t, (sizeof(IT[nnzc])));
    cudaMalloc((void**)&tuplesC_d_th, (sizeof(NTO[nnzc])));
    cudaMalloc((void**)&tuplesC_d, (sizeof(std::tuple<IT, IT, NTO>[nnzc])));
    cudaMalloc((void**)&colptr_size_d, (sizeof(uint[Bdcsc->nzc])));
    cudaMemset(colptr_size_d, 0, sizeof(uint[Bdcsc->nzc]));
    cudaMalloc((void**)&A_Tran_CP, sizeof(IT[Adcsc_Tran->nzc + 1]));
    cudaMalloc((void**)&A_Tran_IR, sizeof(IT[Adcsc_Tran->nz]));
    cudaMalloc((void**)&A_Tran_JC, sizeof(IT[Adcsc_Tran->nzc]));
    cudaMalloc((void**)&A_Tran_numx, sizeof(NT1[Adcsc_Tran->nz]));
    cudaMalloc((void**)&B_CP, sizeof(IT[Bdcsc->nzc + 1]));
    cudaMalloc((void**)&B_IR, sizeof(IT[Bdcsc->nz]));
    cudaMalloc((void**)&B_JC, sizeof(IT[Bdcsc->nzc]));
    cudaMalloc((void**)&B_numx, sizeof(NT2[Bdcsc->nz]));
    cudaMalloc((void**)&colptrC_d, sizeof(IT[Bdcsc->nzc]));
    cudaMemcpy(colptrC_d, colptrC, sizeof(IT[Bdcsc->nzc]), cudaMemcpyHostToDevice);
    cudaMemcpy(A_Tran_CP, Adcsc_Tran->cp, sizeof(IT[Adcsc_Tran->nzc + 1]), cudaMemcpyHostToDevice);
    cudaMemcpy(A_Tran_IR, Adcsc_Tran->ir, sizeof(IT[Adcsc_Tran->nz]), cudaMemcpyHostToDevice);
    cudaMemcpy(A_Tran_JC, Adcsc_Tran->jc, sizeof(IT[Adcsc_Tran->nzc]), cudaMemcpyHostToDevice);
    cudaMemcpy(A_Tran_numx, Adcsc_Tran->numx, sizeof(NT1[Adcsc_Tran->nz]), cudaMemcpyHostToDevice);
    cudaMemcpy(B_CP, Bdcsc->cp, sizeof(IT[Bdcsc->nzc + 1]), cudaMemcpyHostToDevice);
    cudaMemcpy(B_IR, Bdcsc->ir, sizeof(IT[Bdcsc->nz]), cudaMemcpyHostToDevice);
    cudaMemcpy(B_JC, Bdcsc->jc, sizeof(IT[Bdcsc->nzc]), cudaMemcpyHostToDevice);
    cudaMemcpy(B_numx, Bdcsc->numx, sizeof(NT1[Bdcsc->nz]), cudaMemcpyHostToDevice);
    /*#ifdef THREADED
    #pragma omp parallel for
    #endif
        for(size_t i=0; i < Bdcsc->nzc; ++i)
        {
            size_t nnzcolB = Bdcsc->cp[i+1] - Bdcsc->cp[i]; //nnz in the current column of B
            int myThread = 0;

    #ifdef THREADED
            myThread = omp_get_thread_num();
    #endif
                uint* curptr = new uint;
                *curptr = colptrC[i];
                cudaMemcpy(curptr_d, curptr, sizeof(uint), cudaMemcpyHostToDevice);
                delete curptr;
                //uint curptr = colptrC[i];
                /*for(size_t j = 0; j < Adcsc_Tran->nzc; ++j) {
                    bool made = false;
                    size_t r = Adcsc_Tran->cp[j];
                    for (size_t k = 0; k < nnzcolB; ++k) {
                        while (r < Adcsc_Tran->cp[j + 1] && Bdcsc->ir[Bdcsc->cp[i]+k] > Adcsc_Tran->ir[r]) {
                            r++;
                        }
                        if (r >= Adcsc_Tran->cp[j + 1]) {
                                break;
                            }
                        if (Bdcsc->ir[Bdcsc->cp[i]+k] == Adcsc_Tran->ir[r]) {
                            NTO mrhs = Adcsc_Tran->numx[r] * Bdcsc->numx[Bdcsc->cp[i]+k];
                            if(true) {
                                if (made) {
                                    std::get<2>(tuplesC[curptr - 1]) = std::get<2>(tuplesC[curptr - 1]) + mrhs;
                                } else {
                                    made = true;
                                    //tuplesC[curptr++] = std::make_tuple(Adcsc_Tran->jc[j], Bdcsc->jc[i], mrhs);
                                    std::get<0>(tuplesC[curptr]) = Adcsc_Tran->jc[j];
                                    std::get<1>(tuplesC[curptr]) = Bdcsc->jc[i];
                                    std::get<2>(tuplesC[curptr++]) = mrhs;
                                }
                            }
                        }
                    }
                }
                  //cudaDeviceSynchronize();
        }*/

    transformColumn(Adcsc_Tran->nzc, A_Tran_CP, A_Tran_IR, A_Tran_JC, A_Tran_numx, B_CP, B_IR, B_JC, B_numx, tuplesC_d,
                    colptrC_d, Bdcsc->nzc);

    if (clearA) delete const_cast<SpDCCols<IT, NT1>*>(&A);
    if (clearB) delete const_cast<SpDCCols<IT, NT2>*>(&B);

    if (deleteAux) delete[] aux;
    // std::cout << "Made it to receive" << std::endl;
    IT* tuplesC_o = static_cast<IT*>(::operator new(sizeof(IT[nnzc])));
    IT* tuplesC_t = static_cast<IT*>(::operator new(sizeof(IT[nnzc])));
    NTO* tuplesC_th = static_cast<NTO*>(::operator new(sizeof(NTO[nnzc])));

    uint* colptr_size = static_cast<uint*>(::operator new(sizeof(uint[Bdcsc->nzc])));
    cudaMemcpy(tuplesC, tuplesC_d, sizeof(std::tuple<IT, IT, NTO>[nnzc]), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    /*std::cout << "Made it to loop" << std::endl;
   #ifdef THREADED
#pragma omp parallel for
#endif
    for (IT i = 0; i < Bdcsc -> nzc; ++i) {
        //std::cout << "Getting: " << i << std::endl;
        for (IT j = 0; j < colptr_size[i]; ++j) {
            IT in = colptrC[i] + j;
            //std::cout << "Grabbed: " << j << " with " << in << std::endl;
            tuplesC[in] = std::make_tuple(tuplesC_o[in], tuplesC_t[in], tuplesC_th[in]);
            //printf("Made tuple at in %i, with values %i, %i, and %i", in, tuplesC_o[in], tuplesC_t[in],
tuplesC_th[in]);
            //std::cout << "Built!" << std::endl;
            //std::cout << "Done" <<std::endl;
        }
    }*/
    delete[] colptrC;
    delete[] flopptr;
    delete[] tuplesC_o;
    delete[] tuplesC_t;
    delete[] tuplesC_th;
    delete[] colptr_size;
    // std::cout << "Made it to build" << std::endl;
    SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO>(nnzc, mdim, ndim, tuplesC, false, true);

    // std::cout << "Made it to return" << std::endl;
    //  std::cout << "localspgemminfo," << flop << "," << nnzc << "," << compression_ratio << "," << t1-t0 << std::endl;
    //  std::cout << hashSelected << ", " << Bdcsc->nzc << ", " << (float)hashSelected / Bdcsc->nzc << std::endl;
    return spTuplesC;
}
#endif
// Hybrid approach of multithreaded HeapSpGEMM and HashSpGEMM
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO>* LocalSpGEMMHash(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, bool clearA, bool clearB,
                                   bool sort = true);

/*
 *  Estimates total flops necessary to multiply A and B
 *  Then returns the number
 * */
template <typename SR, typename IT, typename NT1, typename NT2>
IT EstimateLocalFLOP(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, bool clearA, bool clearB);

// estimate space for result of SpGEMM
template <typename IT, typename NT1, typename NT2>
IT* estimateNNZ(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, IT* aux = nullptr, bool freeaux = true);

// estimate space for result of SpGEMM with Hash
template <typename IT, typename NT1, typename NT2>
IT* estimateNNZ_Hash(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, IT* flopC, IT* aux = nullptr);

// sampling-based nnz estimation (within SUMMA)
template <typename IT, typename NT1, typename NT2>
int64_t estimateNNZ_sampling(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, int nrounds = 5);

// estimate the number of floating point operations of SpGEMM
template <typename IT, typename NT1, typename NT2>
IT* estimateFLOP(const SpDCCols<IT, NT1>& A, const SpDCCols<IT, NT2>& B, IT* aux = nullptr);

////////////////////////////////////////////////////////////////////////////////
//////////////////////////// CSC-based local SpGEMM	////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO>* LocalHybridSpGEMM(const SpCCols<IT, NT1>& A, const SpCCols<IT, NT2>& B, bool clearA, bool clearB);

template <typename IT, typename NT1, typename NT2>
IT* estimateFLOP(const SpCCols<IT, NT1>& A, const SpCCols<IT, NT2>& B);

template <typename IT, typename NT1, typename NT2>
IT* estimateNNZ_Hash(const SpCCols<IT, NT1>& A, const SpCCols<IT, NT2>& B, const IT* flopC);

}  // namespace combblas

#endif
