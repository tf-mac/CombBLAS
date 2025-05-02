#pragma once

#include <tuple>

#include "GALATIC/include/CSR.cuh"
#include "GALATIC/include/SemiRingInterface.h"
#include "GALATIC/include/default_scheduling_traits.h"
#include "GALATIC/source/device/Multiply.cuh"
#include "SpTuples.h"
#include "cuutils.h"

namespace combblas
{
struct Arith_SR : SemiRing<double, double, double> {
    __host__ __device__ double multiply(const double &a, const double &b) const { return a * b; }
    __host__ __device__ double add(const double &a, const double &b) const { return a + b; }
    __host__ __device__ static double AdditiveIdentity() { return 0; }
};
template <typename NTO, typename IT, typename NT1, typename NT2>
void transformColumn(IT A_nzc, IT *A_Tran_CP, IT *A_Tran_IR, IT *A_Tran_JC, NT1 *A_Tran_numx, IT *B_CP, IT *B_IR,
                     IT *B_JC, NT2 *B_numx, std::tuple<IT, IT, NTO> *tuplesC_d, IT *curptrC, IT B_nzc);

template <typename Arith_SR, typename NTO, typename NT1, typename NT2, typename IT>
CSR<NTO> LocalGalaticSPGEMM(CSR<NT1> input_A_CPU, CSR<NT2> input_B_CPU, bool clearA, bool clearB, Arith_SR semiring,
                            IT *aux = nullptr);

template <typename NT1, typename NT2, typename NT3, typename sr>
struct Wrap_SR : SemiRing<NT1, NT2, NT3> {
    __host__ __device__ NT3 multiply(const NT1 &a, const NT2 &b) const { return sr::multiply(a, b); }
    __host__ __device__ NT3 add(const NT1 &a, const NT2 &b) const { return sr::add(a, b); }
    __host__ __device__ static double AdditiveIdentity() { return 0; }
};

// to tuples and sort by rows and compress
template <typename UDERA, typename NU1>
void convertCSRReal(UDERA *ARecv, dCSR<NU1> &input_GPU, int id)
{
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERA::LocalNT LNA;
    SpTuples<LIA, LNA> sptuples(*ARecv);
    sptuples.SortRowBased();
    // convert to CSR
    LIA *rowptr = new LIA[ARecv->getnrow() + 1];
    LIA *indices = new LIA[ARecv->getnnz()];
    LNA *values = new LNA[ARecv->getnnz()];

    for (size_t nnzidx = 0; nnzidx < ARecv->getnnz(); ++nnzidx) {
        ++rowptr[sptuples.rowindex(nnzidx) + 1];
        indices[nnzidx] = sptuples.colindex(nnzidx);
        values[nnzidx] = sptuples.numvalue(nnzidx);
    }
    LIA nrows(sptuples.getnrow());
    LIA ncols(sptuples.getncol());
    input_GPU.alloc(nrows, ncols, sptuples.getnnz());
    cudaMemcpy(input_GPU.row_offsets, rowptr, sizeof(LIA) * (sptuples.getnrow() + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(input_GPU.col_ids, indices, sizeof(LIA) * (sptuples.getnnz()), cudaMemcpyHostToDevice);
    cudaMemcpy(input_GPU.data, values, sizeof(LIA) * (sptuples.getnnz()), cudaMemcpyHostToDevice);
    delete[] rowptr;
    delete[] indices;
    delete[] values;
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
}

template <typename UDERA, typename NU1>
void convertCSR(UDERA *ARecv, dCSR<NU1> &input_GPU, int id)
{
    typedef typename UDERA::LocalIT LIA;
    LIA j = 0;
    unsigned int *rows;
    cudaMallocHost(&rows, sizeof(unsigned int) * (ARecv->getncol() + 1));
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    for (LIA i = 0; i <= ARecv->getnzc(); ++i) {
        if (i == ARecv->getnzc()) {
            while (j <= ARecv->getncol()) {
                rows[j] = ARecv->getnnz();
                j++;
            }
            break;
        }
        unsigned int val = (unsigned int)ARecv->GetDCSC()->cp[i];
        while (j <= ARecv->GetDCSC()->jc[i] && j <= ARecv->getncol()) {
            rows[j] = val;
            j++;
        }
    }
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    // std::cout << "STARTING ALLOCING in CONV " << id << std::endl;
    if (input_GPU.nnz != 0) dealloc(input_GPU);
    input_GPU.rows = ARecv->getncol();
    input_GPU.cols = ARecv->getnrow();
    input_GPU.nnz = ARecv->getnnz();
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    // std::cout << input_GPU.nnz << std::endl;
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc(&input_GPU.data, sizeof(NU1) * (ARecv->getnnz())));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc(&input_GPU.col_ids, sizeof(unsigned int) * (ARecv->getnnz())));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc(&input_GPU.row_offsets, sizeof(unsigned int) * (ARecv->getncol() + 1)));
    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    // std::cout << "STARTING COPY " << id << std::endl;

    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(input_GPU.row_offsets, rows, (input_GPU.rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    // std::cout << "CPED ROW/COLS " << id << std::endl;
    if (ARecv->getnnz() > 0)
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(input_GPU.data, ARecv->GetDCSC()->numx, (ARecv->getnnz()) * sizeof(NU1),
                                            cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    // std::cout << "CPED NUM " << id << std::endl;
    if (ARecv->getnnz() > 0)
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(input_GPU.col_ids, &(ARecv->GetDCSC()->ir[0]),
                                            (ARecv->getnnz()) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    // std::cout << "DELETING ROWS " << id << std::endl;

    cudaFreeHost(rows);
    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    // free(rows);
}

typedef Arith_SR ringss;
Arith_SR sr;
// double comptime = 0;
extern double convertingtime;

template <typename SR, typename NU1, typename NU2, typename NUO>
CSR<NUO> GPULocalMultiply(dCSR<NU1> &A, dCSR<NU2> &B)
{
    // double t1 = MPI_Wtime();
    const int Threads = 128;
    const int BlocksPerMP = 1;
    const int NNZPerThread = 2;
    const int InputElementsPerThreads = 2;
    const int RetainElementsPerThreads = 1;
    const int MaxChunksToMerge = 16;
    const int MaxChunksGeneralizedMerge = 512;  // MAX: 865
    const int MergePathOptions = 8;
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    cudaDeviceSynchronize();
    SR semiring2;
    if (A.nnz == 0 || B.nnz == 0) {
        CSR<NUO> C;
        C.alloc(A.rows, B.rows, 0);
        return C;
    }
    dCSR<NUO> result_mat_GPU;
    GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread, InputElementsPerThreads,
                                                RetainElementsPerThreads, MaxChunksToMerge, MaxChunksGeneralizedMerge,
                                                MergePathOptions);

    const bool Debug_Mode = true;
    // DefaultTraits.preferLoadBalancing = false;
    ExecutionStats stats;
    // stats.measure_all = false;
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    std::cout << "ENTERED MULT" << std::endl;
    ACSpGEMM::Multiply<ringss>(A, B, result_mat_GPU, DefaultTraits, stats, Debug_Mode, sr);
    std::cout << "EXITED MULT" << std::endl;

    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    // std::cout << "DONE" << std::endl;
    CSR<NUO> result_mat_CPU;
    size_t it = 0;
    // std::unordered_set<LIC> nnzc_set;
    // std::cout << result_mat_GPU.rows << std::endl;
    // double tmp1 = MPI_Wtime();
    convert(result_mat_CPU, result_mat_GPU);
    // double tmp2 = MPI_Wtime();
    // std::cerr << "Convert time: " << tmp2 - tmp1 << std::endl;
    // double convert
    // double convertingtime += tmp2 - tmp1;

    //::cout << sizeof(NUO) * result_mat_GPU.nnz << std::endl;
    // std::cout << sizeof(uint) * result_mat_GPU.rows << std::endl;
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
    // cudaFree(result_mat_GPU.data);
    // cudaFree(result_mat_GPU.col_ids);
    // cudaFree(result_mat_GPU.row_offsets);
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    // result_mat_GPU.reset();
    cudaDeviceSynchronize();
    // double t2 = MPI_Wtime();
    // comptime += (t2 - t1);
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    return result_mat_CPU;
}

template <typename SR, typename NU1, typename NU2, typename NUO>
SpCCols<int32_t, NUO> GPULocalMultiply(SpCCols<int32_t, NU1> &A, SpCCols<int32_t, NU2> &B)
{
    double t1 = MPI_Wtime();
    const int Threads = 128;
    const int BlocksPerMP = 1;
    const int NNZPerThread = 2;
    const int InputElementsPerThreads = 2;
    const int RetainElementsPerThreads = 1;
    const int MaxChunksToMerge = 16;
    const int MaxChunksGeneralizedMerge = 512;  // MAX: 865
    const int MergePathOptions = 8;
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    cudaDeviceSynchronize();
    SR semiring2;
    if (A.nnz == 0 || B.nnz == 0) {
        CSR<NUO> C;
        C.alloc(A.rows, B.rows, 0);
        return C;
    }
    dCSR<NUO> result_mat_GPU;
    GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread, InputElementsPerThreads,
                                                RetainElementsPerThreads, MaxChunksToMerge, MaxChunksGeneralizedMerge,
                                                MergePathOptions);

    const bool Debug_Mode = false;
    // DefaultTraits.preferLoadBalancing = false;
    ExecutionStats stats;
    // stats.measure_all = false;
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    // std::cout << "ENTERED MULT" << std::endl;
    ACSpGEMM::Multiply<ringss>(A, B, result_mat_GPU, DefaultTraits, stats, Debug_Mode, sr);
    // std::cout << "EXITED MULT" << std::endl;
    //  std::cout << "EXITED MULT" << std::endl;

    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    // std::cout << "DONE" << std::endl;
    CSR<NUO> result_mat_CPU;
    size_t it = 0;
    // std::unordered_set<LIC> nnzc_set;
    // std::cout << result_mat_GPU.rows << std::endl;
    double tmp1 = MPI_Wtime();
    convert(result_mat_CPU, result_mat_GPU);
    double tmp2 = MPI_Wtime();
    // std::cerr << "Convert time: " << tmp2 - tmp1 << std::endl;
    // double convert
    // double convertingtime += tmp2 - tmp1;

    //::cout << sizeof(NUO) * result_mat_GPU.nnz << std::endl;
    // std::cout << sizeof(uint) * result_mat_GPU.rows << std::endl;
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
    // cudaFree(result_mat_GPU.data);
    // cudaFree(result_mat_GPU.col_ids);
    // cudaFree(result_mat_GPU.row_offsets);
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    // result_mat_GPU.reset();
    cudaDeviceSynchronize();
    double t2 = MPI_Wtime();
    // comptime += (t2 - t1);
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    return result_mat_CPU;
}

}  // namespace combblas