/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#pragma once

// #ifdef __CUDACC__

#include <mpi.h>
#include <sys/select.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <type_traits>

#include "CombBLAS/PerfRecorder.h"
#include "CombBLAS/SpCRows.h"
#include "Friends.h"
#include "GALATIC/include/CSR.cuh"
#include "GALATIC/include/Multiply.h"
#include "GALATIC/include/SemiRingInterface.h"
#include "GALATIC/include/TestSpGEMM.cuh"
#include "GALATIC/include/dCSR.cuh"
#include "GALATIC/source/device/Multiply.cuh"
#include "MPIType.h"
#include "MultiwayMerge.h"
#include "OptBuf.h"
#include "SpParHelper.h"
#include "SpParMat.h"
#include "SpParMat3D.h"
#include "cudaSpGEMM.h"
#include "cuutils.h"
// #include "logging.h"
#include "mtSpGEMM.h"
extern int GPUTradeoff;

namespace combblas
{
template <class IT, class NT, class DER>
class SpParMat;

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
    HANDLE_ERROR(cudaGetLastError());
}

template <typename UDERA, typename NU1>
void convertCSR(UDERA *ARecv, dCSR<NU1> &input_GPU, int id)
{
    typedef typename UDERA::LocalIT LIA;
    LIA j = 0;
    unsigned int *rows;
    cudaMallocHost(&rows, sizeof(unsigned int) * (ARecv->getncol() + 1));
    HANDLE_ERROR(cudaGetLastError());

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
    HANDLE_ERROR(cudaGetLastError());

    // std::cout << "STARTING ALLOCING in CONV " << id << std::endl;
    if (input_GPU.nnz != 0) dealloc(input_GPU);
    input_GPU.rows = ARecv->getncol();
    input_GPU.cols = ARecv->getnrow();
    input_GPU.nnz = ARecv->getnnz();
    HANDLE_ERROR(cudaGetLastError());

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
    HANDLE_ERROR(cudaGetLastError());

    // free(rows);
}

// Workaround for now

// struct MinPlusSRingGPU : SemiRing<double, double, double> {
//     __host__ __device__ double multiply(const double &a, const double &b) const
//     {
//         if (a == std::numeric_limits<double>::max() || b == std::numeric_limits<double>::max()) {
//             return std::numeric_limits<double>::max();
//         } else
//             return a + b;
//     }
//
//     __host__ __device__ double add(const double &a, const double &b) const { return std::min(a, b); }
//     __host__ __device__ static double AdditiveIdentity() { return std::numeric_limits<double>::max(); }
// };

typedef Arith_SR ringss;
Arith_SR sr;
// double comptime = 0;
extern double convertingtime;

template <typename SR, typename NU1, typename NU2, typename NUO>
CSR<NUO> GPULocalMultiply(dCSR<NU1> &A, dCSR<NU2> &B)
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
    HANDLE_ERROR(cudaGetLastError());

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
    HANDLE_ERROR(cudaGetLastError());

    // std::cout << "ENTERED MULT" << std::endl;
    ACSpGEMM::Multiply<ringss>(A, B, result_mat_GPU, DefaultTraits, stats, Debug_Mode, sr);
    // std::cout << "EXITED MULT" << std::endl;
    //  std::cout << "EXITED MULT" << std::endl;

    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());
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
    HANDLE_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
    // cudaFree(result_mat_GPU.data);
    // cudaFree(result_mat_GPU.col_ids);
    // cudaFree(result_mat_GPU.row_offsets);
    HANDLE_ERROR(cudaGetLastError());
    // result_mat_GPU.reset();
    cudaDeviceSynchronize();
    double t2 = MPI_Wtime();
    comptime += (t2 - t1);
    HANDLE_ERROR(cudaGetLastError());
    return result_mat_CPU;
}

/**
 * Parallel C = A*B routine that uses a double buffered broadcasting scheme, but
 * this time with CUDA
 * @pre { Input matrices, A and B, should not alias }
 * Most memory efficient version available. Total stages: 2*sqrt(p)
 * Memory requirement during first sqrt(p) stages: <= (3/2)*(nnz(A)+nnz(B))+(1/2)*nnz(C)
 * Memory requirement during second sqrt(p) stages: <= nnz(A)+nnz(B)+nnz(C)
 * Final memory requirement: nnz(C) if clearA and clearB are true
 **/

// extern PerformanceRecorder prspgemmdbuffcuda;
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_DoubleBuff_CUDA(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B,
                                                    bool clearA = false, bool clearB = false)
{
    if (!CheckSpGEMMCompliance(A, B)) {
        return SpParMat<IU, NUO, UDERO>();
    }
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;

    static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");
    static_assert(std::is_same<LIA, LIC>::value, "local index types for input and output matrices should be the same");

    int stages, dummy;  // last two parameters of ProductGrid are ignored for
    // Synch multiplication
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    double converttime = 0.0;  // data movement
    double convertingtime = 0.0;
    double transposetime = 0.0;  //
    double commtime = 0.0;
    double comptime = 0.0;
    double mergetime = 0.0;
    double csr2tuplestime = 0.0;
    double memcomm = 0.0;
    double memtuples = 0.0;
    double memC = 0.0;

    ACSpGEMM::id = id;
    int devices;
    cudaGetDeviceCount(&devices);
    int local_rank, local_size;
    cudaSetDevice(id % devices);
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
    LIA C_m = A.spSeq->getnrow();
    LIB C_n = B.spSeq->getncol();
    UDERA *A1seq = new UDERA();
    UDERA *A2seq = new UDERA();
    UDERB *B1seq = new UDERA();
    UDERB *B2seq = new UDERB();
    int Aself = (A.commGrid)->GetRankInProcRow();
    int Bself = (B.commGrid)->GetRankInProcCol();
    transposetime -= MPI_Wtime();
    (A.spSeq)->Split(*A1seq, *A2seq);
    const_cast<UDERB *>(B.spSeq)->Transpose();
    (B.spSeq)->Split(*B1seq, *B2seq);
    const_cast<UDERB *>(B1seq)->Transpose();
    const_cast<UDERB *>(B2seq)->Transpose();
    transposetime += MPI_Wtime();
    prspgemmdbuffcuda.doublemap["transposeT(ms)"] = transposetime;
    dCSR<NU1> input_A_GPU;
    dCSR<NU2> input_B_GPU;
    Wrap_SR<NU1, NU2, NUO, SR> semiring;
    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    // HLOG("Rank %d: transposetime %.6f s", id, transposetime);
    LIA **ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
    LIB **BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);

    SpParHelper::GetSetSizes(*A1seq, ARecvSizes, (A.commGrid)->GetRowWorld());
    SpParHelper::GetSetSizes(*B1seq, BRecvSizes, (B.commGrid)->GetColWorld());

    // Remotely fetched matrices are stored as pointers
    UDERA *ARecv;
    UDERB *BRecv;
    std::vector<SpTuples<LIC, NUO> *> tomerge;
    HANDLE_ERROR(cudaGetLastError());
    for (int i = 0; i < stages; ++i) {
        /////////////////////// format conversion of A /////////////////////////
        converttime -= MPI_Wtime();
        dCSR<NU1> input_A_recv_GPU;
        dCSR<NU2> input_B_recv_GPU;
        std::vector<uint> ess;
        if (i == Aself) {
            convertCSR<UDERA, NU1>(A1seq, input_A_recv_GPU, id);
        } else {
            ARecv = new UDERA();  // first, create the object
        }
        converttime += MPI_Wtime();

        /////////////////////// communication A /////////////////////////
        ess.resize(UDERA::esscount);
        for (int j = 0; j < UDERA::esscount; ++j) {
            ess[j] = ARecvSizes[j][i];  // essentials of the ith
        }
        commtime -= MPI_Wtime();
        SpParHelper::BCastMatrixCUDA<uint, NU1>(GridC->GetRowWorld(), input_A_recv_GPU, ess, i,
                                                GPUTradeoff);  // then, receive its elements
        commtime += MPI_Wtime();
        ess.clear();

        /////////////////////// format conversion of B /////////////////////////
        converttime -= MPI_Wtime();
        if (i == Bself) {
            convertCSR<UDERB, NU2>(B1seq, input_B_recv_GPU, id);  // shallow-copy
        } else {
            BRecv = new UDERB();
        }
        converttime += MPI_Wtime();
        ess.resize(UDERB::esscount);
        for (int j = 0; j < UDERB::esscount; ++j) {
            ess[j] = BRecvSizes[j][i];
        }
        /////////////////////// communication B /////////////////////////
        commtime -= MPI_Wtime();
        SpParHelper::BCastMatrixCUDA(GridC->GetColWorld(), input_B_recv_GPU, ess, i,
                                     GPUTradeoff);  // then, receive its elements
        commtime += MPI_Wtime();

        comptime -= MPI_Wtime();
        HANDLE_ERROR(cudaGetLastError());
        CSR<NUO> result_mat_CPU = GPULocalMultiply<SR, NU1, NU2, NUO>(input_B_recv_GPU, input_A_recv_GPU);
        HANDLE_ERROR(cudaGetLastError());
        comptime += MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        size_t it = 0;
        // std::tuple<LIC, LIC, NUO> *tuplesC =
        //     static_cast<std::tuple<LIC, LIC, NUO> *>(::operator new(sizeof(std::tuple<LIC, LIC,
        //     NUO>[result_mat_CPU.nnz])));
        csr2tuplestime -= MPI_Wtime();
        auto *tuplesC = new std::tuple<LIC, LIC, NUO>[result_mat_CPU.nnz];
        for (LIC ti = 0; ti < result_mat_CPU.rows; ++ti) {
            for (LIC tj = result_mat_CPU.row_offsets[ti]; tj < result_mat_CPU.row_offsets[ti + 1]; ++tj) {
                tuplesC[it++] = std::make_tuple(result_mat_CPU.col_ids[tj], ti, result_mat_CPU.data[tj]);
            }
        }
        SpTuples<LIC, NUO> *C_cont = new SpTuples<LIC, NUO>(result_mat_CPU.nnz, C_m, C_n, tuplesC, false, true);
        csr2tuplestime += MPI_Wtime();
        if (i != Aself) delete ARecv;
        if (i != Bself) delete BRecv;
        if (!C_cont->isZero())
            tomerge.push_back(C_cont);
        else
            delete C_cont;
    }
    HANDLE_ERROR(cudaGetLastError());

    // HLOG("Rank %d: p1: convert %.6f s, comm %.6f s, comp %.6f s, csr2tuples %.6f s", id, converttime, commtime,
    //      comptime, csr2tuplestime);

    if (clearA) delete A1seq;
    if (clearB) delete B1seq;

    dCSR<NU1> input_A2_GPU;
    dCSR<NU2> input_B2_GPU;
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaGetLastError());

    SpParHelper::GetSetSizes(*A2seq, ARecvSizes, (A.commGrid)->GetRowWorld());
    SpParHelper::GetSetSizes(*B2seq, BRecvSizes, (B.commGrid)->GetColWorld());
    converttime = commtime = comptime = csr2tuplestime = 0.0;
    for (int i = 0; i < stages; ++i) {
        dCSR<NU1> input_A_recv_GPU;
        dCSR<NU2> input_B_recv_GPU;
        std::vector<uint> ess;
        converttime -= MPI_Wtime();
        if (i == Aself) {
            convertCSR<UDERA, NU1>(A2seq, input_A_recv_GPU, id);
        } else {
            ARecv = new UDERA();  // first, create the object
        }
        converttime += MPI_Wtime();
        ess.resize(UDERA::esscount);
        for (int j = 0; j < UDERA::esscount; ++j) {
            ess[j] = ARecvSizes[j][i];  // essentials of the ith
            // matrix in this row
        }
        commtime -= MPI_Wtime();
        SpParHelper::BCastMatrixCUDA<uint, NU1>(GridC->GetRowWorld(), input_A_recv_GPU, ess, i,
                                                GPUTradeoff);  // then, receive its elements
        commtime += MPI_Wtime();
        ess.clear();
        converttime -= MPI_Wtime();
        if (i == Bself) {
            convertCSR<UDERB, NU2>(B2seq, input_B_recv_GPU, id);
        } else {
            BRecv = new UDERB();
        }
        converttime += MPI_Wtime();
        ess.resize(UDERB::esscount);
        for (int j = 0; j < UDERB::esscount; ++j) {
            ess[j] = BRecvSizes[j][i];
        }
        commtime -= MPI_Wtime();
        SpParHelper::BCastMatrixCUDA(GridC->GetColWorld(), input_B_recv_GPU, ess, i,
                                     GPUTradeoff);  // then, receive its elements
        commtime += MPI_Wtime();
        HANDLE_ERROR(cudaGetLastError());
        comptime -= MPI_Wtime();
        CSR<NUO> result_mat_CPU = GPULocalMultiply<SR, NU1, NU2, NUO>(input_B_recv_GPU, input_A_recv_GPU);
        comptime += MPI_Wtime();
        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaGetLastError());

        csr2tuplestime -= MPI_Wtime();
        size_t it = 0;
        std::tuple<LIC, LIC, NUO> *tuplesC = static_cast<std::tuple<LIC, LIC, NUO> *>(
            ::operator new(sizeof(std::tuple<LIC, LIC, NUO>[result_mat_CPU.nnz])));
        for (LIC ti = 0; ti < result_mat_CPU.rows; ++ti) {
            for (LIC tj = result_mat_CPU.row_offsets[ti]; tj < result_mat_CPU.row_offsets[ti + 1]; ++tj) {
                tuplesC[it++] = std::make_tuple(result_mat_CPU.col_ids[tj], ti, result_mat_CPU.data[tj]);
            }
        }
        SpTuples<LIC, NUO> *C_cont = new SpTuples<LIC, NUO>(result_mat_CPU.nnz, C_m, C_n, tuplesC, false, true);
        csr2tuplestime += MPI_Wtime();
        if (i != Aself) delete ARecv;
        if (i != Bself) delete BRecv;
        if (!C_cont->isZero())
            tomerge.push_back(C_cont);
        else
            delete C_cont;
    }

    SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
    SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
    if (clearA) {
        delete A2seq;
        delete A.spSeq;
        A.spSeq = NULL;
    } else {
        (A.spSeq)->Merge(*A1seq, *A2seq);
        delete A1seq;
        delete A2seq;
    }
    if (clearB) {
        delete B2seq;
        delete B.spSeq;
        B.spSeq = NULL;
    } else {
        B1seq->Transpose();
        B2seq->Transpose();
        (B.spSeq)->Merge(*B1seq, *B2seq);
        delete B1seq;
        delete B2seq;
        const_cast<UDERB *>(B.spSeq)->Transpose();  // transpose back to original
    }
    mergetime -= MPI_Wtime();
    UDERO *C = new UDERO(MergeAll<SR>(tomerge, C_m, C_n, true), false);
    mergetime += MPI_Wtime();
    // HLOG("Rank %d: p2: convert %.6f s, comm %.6f s, comp %.6f s, csr2tuples %.6f s", id, converttime, commtime,
    //      comptime, csr2tuplestime);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return SpParMat<IU, NUO, UDERO>(C, GridC);  // return the result object	// return the result object
}

// DCSR Format DBUFF
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_DoubleBuff_CUDA_dCSR(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B,
                                                         bool clearA = false, bool clearB = false)
{
    if (!CheckSpGEMMCompliance(A, B)) {
        return SpParMat<IU, NUO, UDERO>();
    }
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;

    static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");
    static_assert(std::is_same<LIA, LIC>::value, "local index types for input and output matrices should be the same");

    int stages, dummy;  // last two parameters of ProductGrid are ignored for
    // Synch multiplication
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    double converttime = 0.0;    // data movement
    double transposetime = 0.0;  //
    double commtime = 0.0;
    double comptime = 0.0;
    double mergetime = 0.0;
    double csr2tuplestime = 0.0;
    double memcomm = 0.0;
    double memtuples = 0.0;
    double memC = 0.0;

    IU Problem_M = A.getnrow();
    IU Problem_N = B.getncol();
    IU Problem_K = A.getncol();

    IU Local_A_M = A.seqptr()->getnrow();
    IU Local_A_N = A.seqptr()->getncol();
    IU local_A_NZ = A.seqptr()->getnnz();
    IU Local_B_M = B.seqptr()->getnrow();
    IU Local_B_N = B.seqptr()->getncol();
    IU local_B_NZ = B.seqptr()->getnnz();
    // ACSpGEMM::id = id;
    int devices;
    cudaGetDeviceCount(&devices);
    int local_rank, local_size;
    cudaSetDevice(id % devices);
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
    LIA C_m = A.spSeq->getnrow();
    LIB C_n = B.spSeq->getncol();

    // UDERA *A1seq = new UDERA();
    // UDERA *A2seq = new UDERA();
    // UDERB *B1seq = new UDERA();
    // UDERB *B2seq = new UDERB();
    // int Aself = (A.commGrid)->GetRankInProcRow();
    // int Bself = (B.commGrid)->GetRankInProcCol();
    // transposetime -= MPI_Wtime();
    // (A.spSeq)->Split(*A1seq, *A2seq);
    // const_cast<UDERB *>(B.spSeq)->Transpose();
    // (B.spSeq)->Split(*B1seq, *B2seq);
    // const_cast<UDERB *>(B1seq)->Transpose();
    // const_cast<UDERB *>(B2seq)->Transpose();
    // transposetime += MPI_Wtime();

    dCSR<NU1> input_A_GPU;
    dCSR<NU2> input_B_GPU;
    convertCSRReal(A.seqptr(), input_A_GPU, id);
    convertCSRReal(B.seqptr(), input_B_GPU, id);

    Wrap_SR<NU1, NU2, NUO, SR> semiring;
    HGEMM_CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    HLOG("Rank %d: finish allocating A and B in devices", id);
    LIA **ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
    LIB **BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);

    // SpParHelper::GetSetSizes(*A1seq, ARecvSizes, (A.commGrid)->GetRowWorld());
    // SpParHelper::GetSetSizes(*B1seq, BRecvSizes, (B.commGrid)->GetColWorld());

    // // Remotely fetched matrices are stored as pointers
    // UDERA *ARecv;
    // UDERB *BRecv;
    // std::vector<SpTuples<LIC, NUO> *> tomerge;
    // HANDLE_ERROR(cudaGetLastError());
    // for (int i = 0; i < stages; ++i) {
    //     /////////////////////// format conversion of A /////////////////////////
    //     converttime -= MPI_Wtime();
    //     dCSR<NU1> input_A_recv_GPU;
    //     dCSR<NU2> input_B_recv_GPU;
    //     std::vector<uint> ess;
    //     if (i == Aself) {
    //         convertCSR<UDERA, NU1>(A1seq, input_A_recv_GPU, id);
    //     } else {
    //         ARecv = new UDERA();  // first, create the object
    //     }
    //     converttime += MPI_Wtime();

    //     /////////////////////// communication A /////////////////////////
    //     ess.resize(UDERA::esscount);
    //     for (int j = 0; j < UDERA::esscount; ++j) {
    //         ess[j] = ARecvSizes[j][i];  // essentials of the ith
    //     }
    //     commtime -= MPI_Wtime();
    //     SpParHelper::BCastMatrixCUDA<uint, NU1>(GridC->GetRowWorld(), input_A_recv_GPU, ess, i,
    //                                             GPUTradeoff);  // then, receive its elements
    //     commtime += MPI_Wtime();
    //     ess.clear();

    //     /////////////////////// format conversion of B /////////////////////////
    //     converttime -= MPI_Wtime();
    //     if (i == Bself) {
    //         convertCSR<UDERB, NU2>(B1seq, input_B_recv_GPU, id);  // shallow-copy
    //     } else {
    //         BRecv = new UDERB();
    //     }
    //     converttime += MPI_Wtime();
    //     ess.resize(UDERB::esscount);
    //     for (int j = 0; j < UDERB::esscount; ++j) {
    //         ess[j] = BRecvSizes[j][i];
    //     }
    //     /////////////////////// communication B /////////////////////////
    //     commtime -= MPI_Wtime();
    //     SpParHelper::BCastMatrixCUDA(GridC->GetColWorld(), input_B_recv_GPU, ess, i,
    //                                  GPUTradeoff);  // then, receive its elements
    //     commtime += MPI_Wtime();

    //     comptime -= MPI_Wtime();
    //     HANDLE_ERROR(cudaGetLastError());
    //     CSR<NUO> result_mat_CPU = GPULocalMultiply<SR, NU1, NU2, NUO>(input_B_recv_GPU, input_A_recv_GPU);
    //     HANDLE_ERROR(cudaGetLastError());
    //     comptime += MPI_Wtime();
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     size_t it = 0;
    //     // std::tuple<LIC, LIC, NUO> *tuplesC =
    //     //     static_cast<std::tuple<LIC, LIC, NUO> *>(::operator new(sizeof(std::tuple<LIC, LIC,
    //     //     NUO>[result_mat_CPU.nnz])));
    //     csr2tuplestime -= MPI_Wtime();
    //     auto *tuplesC = new std::tuple<LIC, LIC, NUO>[result_mat_CPU.nnz];
    //     for (LIC ti = 0; ti < result_mat_CPU.rows; ++ti) {
    //         for (LIC tj = result_mat_CPU.row_offsets[ti]; tj < result_mat_CPU.row_offsets[ti + 1]; ++tj) {
    //             tuplesC[it++] = std::make_tuple(result_mat_CPU.col_ids[tj], ti, result_mat_CPU.data[tj]);
    //         }
    //     }
    //     SpTuples<LIC, NUO> *C_cont = new SpTuples<LIC, NUO>(result_mat_CPU.nnz, C_m, C_n, tuplesC, false, true);
    //     csr2tuplestime += MPI_Wtime();
    //     if (i != Aself) delete ARecv;
    //     if (i != Bself) delete BRecv;
    //     if (!C_cont->isZero())
    //         tomerge.push_back(C_cont);
    //     else
    //         delete C_cont;
    // }
    // HANDLE_ERROR(cudaGetLastError());

    // HLOG("Rank %d: p1: convert %.6f s, comm %.6f s, comp %.6f s, csr2tuples %.6f s", id, converttime, commtime,
    //      comptime, csr2tuplestime);

    // if (clearA) delete A1seq;
    // if (clearB) delete B1seq;

    // dCSR<NU1> input_A2_GPU;
    // dCSR<NU2> input_B2_GPU;
    // HANDLE_ERROR(cudaGetLastError());
    // HANDLE_ERROR(cudaGetLastError());

    // SpParHelper::GetSetSizes(*A2seq, ARecvSizes, (A.commGrid)->GetRowWorld());
    // SpParHelper::GetSetSizes(*B2seq, BRecvSizes, (B.commGrid)->GetColWorld());
    // converttime = commtime = comptime = csr2tuplestime = 0.0;
    // for (int i = 0; i < stages; ++i) {
    //     dCSR<NU1> input_A_recv_GPU;
    //     dCSR<NU2> input_B_recv_GPU;
    //     std::vector<uint> ess;
    //     converttime -= MPI_Wtime();
    //     if (i == Aself) {
    //         convertCSR<UDERA, NU1>(A2seq, input_A_recv_GPU, id);
    //     } else {
    //         ARecv = new UDERA();  // first, create the object
    //     }
    //     converttime += MPI_Wtime();
    //     ess.resize(UDERA::esscount);
    //     for (int j = 0; j < UDERA::esscount; ++j) {
    //         ess[j] = ARecvSizes[j][i];  // essentials of the ith
    //         // matrix in this row
    //     }
    //     commtime -= MPI_Wtime();
    //     SpParHelper::BCastMatrixCUDA<uint, NU1>(GridC->GetRowWorld(), input_A_recv_GPU, ess, i,
    //                                             GPUTradeoff);  // then, receive its elements
    //     commtime += MPI_Wtime();
    //     ess.clear();
    //     converttime -= MPI_Wtime();
    //     if (i == Bself) {
    //         convertCSR<UDERB, NU2>(B2seq, input_B_recv_GPU, id);
    //     } else {
    //         BRecv = new UDERB();
    //     }
    //     converttime += MPI_Wtime();
    //     ess.resize(UDERB::esscount);
    //     for (int j = 0; j < UDERB::esscount; ++j) {
    //         ess[j] = BRecvSizes[j][i];
    //     }
    //     commtime -= MPI_Wtime();
    //     SpParHelper::BCastMatrixCUDA(GridC->GetColWorld(), input_B_recv_GPU, ess, i,
    //                                  GPUTradeoff);  // then, receive its elements
    //     commtime += MPI_Wtime();
    //     HANDLE_ERROR(cudaGetLastError());
    //     comptime -= MPI_Wtime();
    //     CSR<NUO> result_mat_CPU = GPULocalMultiply<SR, NU1, NU2, NUO>(input_B_recv_GPU, input_A_recv_GPU);
    //     comptime += MPI_Wtime();
    //     HANDLE_ERROR(cudaDeviceSynchronize());
    //     HANDLE_ERROR(cudaGetLastError());

    //     csr2tuplestime -= MPI_Wtime();
    //     size_t it = 0;
    //     std::tuple<LIC, LIC, NUO> *tuplesC = static_cast<std::tuple<LIC, LIC, NUO> *>(
    //         ::operator new(sizeof(std::tuple<LIC, LIC, NUO>[result_mat_CPU.nnz])));
    //     for (LIC ti = 0; ti < result_mat_CPU.rows; ++ti) {
    //         for (LIC tj = result_mat_CPU.row_offsets[ti]; tj < result_mat_CPU.row_offsets[ti + 1]; ++tj) {
    //             tuplesC[it++] = std::make_tuple(result_mat_CPU.col_ids[tj], ti, result_mat_CPU.data[tj]);
    //         }
    //     }
    //     SpTuples<LIC, NUO> *C_cont = new SpTuples<LIC, NUO>(result_mat_CPU.nnz, C_m, C_n, tuplesC, false, true);
    //     csr2tuplestime += MPI_Wtime();
    //     if (i != Aself) delete ARecv;
    //     if (i != Bself) delete BRecv;
    //     if (!C_cont->isZero())
    //         tomerge.push_back(C_cont);
    //     else
    //         delete C_cont;
    // }

    // SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
    // SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
    // if (clearA) {
    //     delete A2seq;
    //     delete A.spSeq;
    //     A.spSeq = NULL;
    // } else {
    //     (A.spSeq)->Merge(*A1seq, *A2seq);
    //     delete A1seq;
    //     delete A2seq;
    // }
    // if (clearB) {
    //     delete B2seq;
    //     delete B.spSeq;
    //     B.spSeq = NULL;
    // } else {
    //     B1seq->Transpose();
    //     B2seq->Transpose();
    //     (B.spSeq)->Merge(*B1seq, *B2seq);
    //     delete B1seq;
    //     delete B2seq;
    //     const_cast<UDERB *>(B.spSeq)->Transpose();  // transpose back to original
    // }
    // mergetime -= MPI_Wtime();
    // UDERO *C = new UDERO(MergeAll<SR>(tomerge, C_m, C_n, true), false);
    // mergetime += MPI_Wtime();
    // HLOG("Rank %d: p2: convert %.6f s, comm %.6f s, comp %.6f s, csr2tuples %.6f s", id, converttime, commtime,
    //      comptime, csr2tuplestime);
    // cudaDeviceSynchronize();
    // HANDLE_ERROR(cudaGetLastError());
    // return SpParMat<IU, NUO, UDERO>(C, GridC);  // return the result object	// return the result object
    SpParMat<IU, NUO, UDERO> tmp;
    return tmp;
}

// CUDA implementation for Mult_AnXBn_Synch, use SpCuCRows as local data structure.
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_Synch_CUDA(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B,
                                               bool clearA = false, bool clearB = false)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (!CheckSpGEMMCompliance(A, B)) {
        return SpParMat<IU, NUO, UDERO>();
    }
    int stages, dummy;  // last two parameters of ProductGrid are ignored for Synch multiplication
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
    IU C_m = A.spSeq->getnrow();
    IU C_n = B.spSeq->getncol();

    // const_cast< UDERB* >(B.spSeq)->Transpose(); // do not transpose for colum-by-column
    // multiplication

    IU **ARecvSizes = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
    IU **BRecvSizes = SpHelper::allocate2D<IU>(UDERB::esscount, stages);

    // SpParHelper::GetSetSizes(*(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
    // SpParHelper::GetSetSizes(*(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());

    // // Remotely fetched matrices are stored as pointers
    // UDERA *ARecv;
    // UDERB *BRecv;
    // std::vector<SpTuples<IU, NUO> *> tomerge;

    // int Aself = (A.commGrid)->GetRankInProcRow();
    // int Bself = (B.commGrid)->GetRankInProcCol();

    //     for (int i = 0; i < stages; ++i) {
    //         std::vector<IU> ess;
    //         if (i == Aself) {
    //             ARecv = A.spSeq;  // shallow-copy
    //         } else {
    //             ess.resize(UDERA::esscount);
    //             for (int j = 0; j < UDERA::esscount; ++j) {
    //                 ess[j] = ARecvSizes[j][i];  // essentials of the ith matrix in this row
    //             }
    //             ARecv = new UDERA();  // first, create the object
    //         }
    //         SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);  // then, receive its elements
    //         ess.clear();

    //         if (i == Bself) {
    //             BRecv = B.spSeq;  // shallow-copy
    //         } else {
    //             ess.resize(UDERB::esscount);
    //             for (int j = 0; j < UDERB::esscount; ++j) {
    //                 ess[j] = BRecvSizes[j][i];
    //             }
    //             BRecv = new UDERB();
    //         }
    //         SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);  // then, receive its elements

    //         SpTuples<IU, NUO> *C_cont = LocalHybridSpGEMM<SR, NUO>(*ARecv, *BRecv,  // parameters themselves
    //                                                                false,           // 'delete A' condition
    //                                                                false);          // 'delete B' condition

    //         if (i != Bself && (!BRecv->isZero())) delete BRecv;
    //         if (i != Aself && (!ARecv->isZero())) delete ARecv;

    //         if (!C_cont->isZero()) tomerge.push_back(C_cont);

    // #ifdef COMBBLAS_DEBUG
    //         std::ostringstream outs;
    //         outs << i << "th SUMMA iteration" << std::endl;
    //         SpParHelper::Print(outs.str());
    // #endif
    //     }

    // if (clearA && A.spSeq != NULL) {
    //     delete A.spSeq;
    //     A.spSeq = NULL;
    // }
    // if (clearB && B.spSeq != NULL) {
    //     delete B.spSeq;
    //     B.spSeq = NULL;
    // }

    // SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
    // SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

    // SpTuples<IU, NUO> *C_tuples = MultiwayMerge<SR>(tomerge, C_m, C_n, true);  // Last parameter to delete input
    // tuples UDERO *C = new UDERO(*C_tuples, false);                             // Last parameter to prevent
    // transpose delete C_tuples;

    // if(!clearB)
    //	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original

    // return SpParMat<IU, NUO, UDERO>(C, GridC);  // return the result object
    return SpParMat<IU, NUO, UDERO>(GridC);  // return empty
}
}  // namespace combblas

// #endif  // __CUDACC__
