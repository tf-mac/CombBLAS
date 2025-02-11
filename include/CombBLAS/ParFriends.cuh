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

// only include this file if CUDA is enabled

#include <mpi.h>
#include <unistd.h>

#include <cstdarg>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_set>

#include "Friends.h"
#include "MPIType.h"
#include "MultiwayMerge.h"
#include "OptBuf.h"
#include "SpParHelper.h"
#include "SpParMat.h"
#include "SpParMat3D.h"
#include "mtSpGEMM.h"

#ifdef __CUDACC__

#include "../GALATIC/include/CSR.cuh"
#include "../GALATIC/include/SemiRingInterface.h"
#include "../GALATIC/include/TestSpGEMM.cuh"
#include "../GALATIC/include/dCSR.cuh"
#include "../GALATIC/source/device/Multiply.cuh"
#include "cudaSpGEMM.h"
#include "cuutils.h"

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
    gpuErrchk(cudaMalloc(&input_GPU.data, sizeof(NU1) * (ARecv->getnnz())));
    gpuErrchk(cudaMalloc(&input_GPU.col_ids, sizeof(unsigned int) * (ARecv->getnnz())));
    gpuErrchk(cudaMalloc(&input_GPU.row_offsets, sizeof(unsigned int) * (ARecv->getncol() + 1)));
    gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "STARTING COPY " << id << std::endl;

    cudaMemcpy(input_GPU.row_offsets, rows, (input_GPU.rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

    gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "CPED ROW/COLS " << id << std::endl;
    if (ARecv->getnnz() > 0) gpuErrchk(cudaMemcpy(input_GPU.data, ARecv->GetDCSC()->numx, (ARecv->getnnz()) * sizeof(NU1), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "CPED NUM " << id << std::endl;
    if (ARecv->getnnz() > 0) gpuErrchk(cudaMemcpy(input_GPU.col_ids, &(ARecv->GetDCSC()->ir[0]), (ARecv->getnnz()) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "DELETING ROWS " << id << std::endl;

    cudaFreeHost(rows);
    gpuErrchk(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());

    // free(rows);
}

// Workaround for now

struct MinPlusSRingGPU : SemiRing<double, double, double> {
    __host__ __device__ double multiply(const double &a, const double &b) const
    {
        if (a == std::numeric_limits<double>::max() || b == std::numeric_limits<double>::max()) {
            return std::numeric_limits<double>::max();
        } else
            return a + b;
    }
    __host__ __device__ double add(const double &a, const double &b) const { return std::min(a, b); }
    __host__ __device__ static double AdditiveIdentity() { return std::numeric_limits<double>::max(); }
};

typedef Arith_SR ringss;
Arith_SR sr;
double comptime = 0;
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
    GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread, InputElementsPerThreads, RetainElementsPerThreads, MaxChunksToMerge, MaxChunksGeneralizedMerge, MergePathOptions);

    const bool Debug_Mode = false;
    // DefaultTraits.preferLoadBalancing = false;
    ExecutionStats stats;
    // stats.measure_all = false;
    HANDLE_ERROR(cudaGetLastError());

    // std::cout << "ENTERED MULT" << std::endl;
    ACSpGEMM::Multiply<ringss>(A, B, result_mat_GPU, DefaultTraits, stats, Debug_Mode, sr);
    // std::cout << "EXITED MULT" << std::endl;

    gpuErrchk(cudaDeviceSynchronize());
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
    convertingtime += tmp2 - tmp1;

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
double checkingTime = 0;
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_DoubleBuff_CUDA(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B, bool clearA = false, bool clearB = false)
{
    HANDLE_ERROR(cudaGetLastError());

    if (!CheckSpGEMMCompliance(A, B)) {
        return SpParMat<IU, NUO, UDERO>();
    }
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;

    double over = 0;
    double t1 = MPI_Wtime();
    static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");
    static_assert(std::is_same<LIA, LIC>::value, "local index types for input and output matrices should be the same");

    int stages, dummy;  // last two parameters of ProductGrid are ignored for
    // Synch multiplication
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    ACSpGEMM::id = id;
    int devices;
    HANDLE_ERROR(cudaGetLastError());
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

    checkingTime += MPI_Wtime() - t1;

    (A.spSeq)->Split(*A1seq, *A2seq);
    const_cast<UDERB *>(B.spSeq)->Transpose();
    (B.spSeq)->Split(*B1seq, *B2seq);
    HANDLE_ERROR(cudaGetLastError());

    // std::cout << Aself << " " << Bself << " starting GPU" << std::endl;
    dCSR<NU1> input_A_GPU;
    dCSR<NU2> input_B_GPU;
    Wrap_SR<NU1, NU2, NUO, SR> semiring;
    HANDLE_ERROR(cudaGetLastError());

    // std::cout << Aself << " " << Bself << " ending cpus" << std::endl;

    gpuErrchk(cudaDeviceSynchronize());

    // Transpose back for the column-by-column algorithm
    const_cast<UDERB *>(B1seq)->Transpose();
    const_cast<UDERB *>(B2seq)->Transpose();
    LIA **ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
    LIB **BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);

    SpParHelper::GetSetSizes(*A1seq, ARecvSizes, (A.commGrid)->GetRowWorld());
    SpParHelper::GetSetSizes(*B1seq, BRecvSizes, (B.commGrid)->GetColWorld());

    // Remotely fetched matrices are stored as pointers
    UDERA *ARecv;
    UDERB *BRecv;
    std::vector<SpTuples<LIC, NUO> *> tomerge;
    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaGetLastError());

    over += MPI_Wtime() - t1;

    double mpi_overhead = 0.0;
    convertingtime = 0.0;
    for (int i = 0; i < stages; ++i) {
        HANDLE_ERROR(cudaGetLastError());
        double t2 = MPI_Wtime();
        dCSR<NU1> input_A_recv_GPU;
        dCSR<NU2> input_B_recv_GPU;
        std::vector<LIA> ess;
        if (i == Aself) {
            double tmp1 = MPI_Wtime();
            convertCSR<UDERA, NU1>(A1seq, input_A_recv_GPU, id);
            double tmp2 = MPI_Wtime();
            // std::cerr << "Convert A time: " << tmp2 - tmp1 << std::endl;
            convertingtime += tmp2 - tmp1;

        } else {
            ARecv = new UDERA();  // first, create the object
        }
        ess.resize(UDERA::esscount);
        for (int j = 0; j < UDERA::esscount; ++j) {
            ess[j] = ARecvSizes[j][i];  // essentials of the ith
                                        // matrix in this row
        }
        // std::cout << "STARTING BCAST " << id << std::endl;
        SpParHelper::BCastMatrixCUDA<uint, NU1>(GridC->GetRowWorld(), input_A_recv_GPU, ess, i,
                                                GPUTradeoff);  // then, receive its elements
        // std::cout << "ENDING BCAST " << id << std::endl;
        ess.clear();
        if (i == Bself) {
            double tmp1 = MPI_Wtime();
            convertCSR<UDERB, NU2>(B1seq, input_B_recv_GPU, id);  // shallow-copy
            double tmp2 = MPI_Wtime();
            // std::cerr << "Convert B time: " << tmp2 - tmp1 << std::endl;
            convertingtime += tmp2 - tmp1;
        } else {
            BRecv = new UDERB();
        }
        ess.resize(UDERB::esscount);
        for (int j = 0; j < UDERB::esscount; ++j) {
            ess[j] = BRecvSizes[j][i];
        }
        SpParHelper::BCastMatrixCUDA(GridC->GetColWorld(), input_B_recv_GPU, ess, i,
                                     GPUTradeoff);  // then, receive its elements
        HANDLE_ERROR(cudaGetLastError());
        mpi_overhead += MPI_Wtime() - t2;
        CSR<NUO> result_mat_CPU = GPULocalMultiply<SR, NU1, NU2, NUO>(input_B_recv_GPU, input_A_recv_GPU);
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaGetLastError());
        MPI_Barrier(MPI_COMM_WORLD);
        size_t it = 0;
        std::tuple<LIC, LIC, NUO> *tuplesC = static_cast<std::tuple<LIC, LIC, NUO> *>(::operator new(sizeof(std::tuple<LIC, LIC, NUO>[result_mat_CPU.nnz])));
        for (LIC i = 0; i < result_mat_CPU.rows; ++i) {
            for (LIC j = result_mat_CPU.row_offsets[i]; j < result_mat_CPU.row_offsets[i + 1]; ++j) {
                tuplesC[it++] = std::make_tuple(result_mat_CPU.col_ids[j], i, result_mat_CPU.data[j]);
            }
        }
        SpTuples<LIC, NUO> *C_cont = new SpTuples<LIC, NUO>(result_mat_CPU.nnz, C_m, C_n, tuplesC, false, true);
        if (i != Aself) delete ARecv;
        if (i != Bself) delete BRecv;
        if (!C_cont->isZero())
            tomerge.push_back(C_cont);
        else
            delete C_cont;
    }
    HANDLE_ERROR(cudaGetLastError());

    if (clearA) delete A1seq;
    if (clearB) delete B1seq;

    // Set the new dimensions
    t1 = MPI_Wtime();
    // dealloc(input_A_GPU);
    // dealloc(input_B_GPU);
    cudaDeviceSynchronize();
    dCSR<NU1> input_A2_GPU;
    dCSR<NU2> input_B2_GPU;
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaGetLastError());

    SpParHelper::GetSetSizes(*A2seq, ARecvSizes, (A.commGrid)->GetRowWorld());
    SpParHelper::GetSetSizes(*B2seq, BRecvSizes, (B.commGrid)->GetColWorld());
    over += MPI_Wtime() - t1;

    // std::cout << "S3 " << id << std::endl;
    for (int i = 0; i < stages; ++i) {
        double t2 = MPI_Wtime();
        dCSR<NU1> input_A_recv_GPU;
        dCSR<NU2> input_B_recv_GPU;
        // std::cout << Aself << " " << Bself << " starting stage " << i
        // << std::endl;
        std::vector<LIA> ess;
        if (i == Aself) {
            double tmp1 = MPI_Wtime();
            convertCSR<UDERA, NU1>(A2seq, input_A_recv_GPU, id);
            double tmp2 = MPI_Wtime();
            // std::cerr << "Convert A time: " << tmp2 - tmp1 << std::endl;
            convertingtime += tmp2 - tmp1;
        } else {
            ARecv = new UDERA();  // first, create the object
        }
        ess.resize(UDERA::esscount);
        for (int j = 0; j < UDERA::esscount; ++j) {
            ess[j] = ARecvSizes[j][i];  // essentials of the ith
                                        // matrix in this row
        }
        // std::cout << "STARTING BCAST " << id << std::endl;
        SpParHelper::BCastMatrixCUDA<uint, NU1>(GridC->GetRowWorld(), input_A_recv_GPU, ess, i,
                                                GPUTradeoff);  // then, receive its elements
        // std::cout << "ENDING BCAST " << id << std::endl;
        ess.clear();
        if (i == Bself) {
            double tmp1 = MPI_Wtime();
            convertCSR<UDERB, NU2>(B2seq, input_B_recv_GPU, id);
            double tmp2 = MPI_Wtime();
            // std::cerr << "Convert B time: " << tmp2 - tmp1 << std::endl;
            convertingtime += tmp2 - tmp1;
        } else {
            BRecv = new UDERB();
        }
        ess.resize(UDERB::esscount);
        for (int j = 0; j < UDERB::esscount; ++j) {
            ess[j] = BRecvSizes[j][i];
        }
        SpParHelper::BCastMatrixCUDA(GridC->GetColWorld(), input_B_recv_GPU, ess, i,
                                     GPUTradeoff);  // then, receive its elements

        // before activating this remove transposing B1seq
        /*
        SpTuples<LIC,NUO> * C_cont = MultiplyReturnTuples<SR, NUO>
                                        (*ARecv, *BRecv, // parameters
        themselves false, true,	// transpose information (B is
        transposed) i != Aself, 	// 'delete A' condition i !=
        Bself);	// 'delete B' condition

        */
        /* ARecv->Transpose();
         BRecv->Transpose();
         SpTuples<LIC,NUO> * C_cont = LocalHybridSpGEMM<SR, NUO>
                         (*ARecv, *BRecv, // parameters themselves
                         i != Aself,    // 'delete A' condition
                         i != Bself);   // 'delete B' condition*/
        // const_cast< UDERB* >(B.spSeq)->Transpose();
        HANDLE_ERROR(cudaGetLastError());

        mpi_overhead += MPI_Wtime() - t2;
        CSR<NUO> result_mat_CPU = GPULocalMultiply<SR, NU1, NU2, NUO>(input_B_recv_GPU, input_A_recv_GPU);
        gpuErrchk(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaGetLastError());

        // over += MPI_Wtime() - t1;
        // std::cout << over << std::endl;
        //  std::cout << "ENDING MULT" << std::endl;
        //  mpi_overhead += MPI_Wtime() - start;
        //  double t2 = MPI_Wtime();
        //  printf("Time for actual mult = %.6lf \n", t2 - t1);
        size_t it = 0;
        // std::unordered_set<LIC> nnzc_set;
        // std::cout << result_mat_GPU.nnz << std::endl;
        // std::cout << Aself << " " << Bself << " ending GPU " << i <<
        // std::endl;
        // printf("OC = %i\n", result_mat_CPU.nnz);

        std::tuple<LIC, LIC, NUO> *tuplesC = static_cast<std::tuple<LIC, LIC, NUO> *>(::operator new(sizeof(std::tuple<LIC, LIC, NUO>[result_mat_CPU.nnz])));
        for (LIC i = 0; i < result_mat_CPU.rows; ++i) {
            for (LIC j = result_mat_CPU.row_offsets[i]; j < result_mat_CPU.row_offsets[i + 1]; ++j) {
                // nzc_set.insert(result_mat_CPU.col_ids[j]);
                // std::cout << "IT " << it << " EXCEEDED " <<
                // result_mat_CPU.nnz <<std::endl;
                tuplesC[it++] = std::make_tuple(result_mat_CPU.col_ids[j], i, result_mat_CPU.data[j]);
            }
        }

        // std::cout << Aself << " " << Bself << " ending tupling " << i
        // << std::endl;
        // load results  onto CPU.
        SpTuples<LIC, NUO> *C_cont = new SpTuples<LIC, NUO>(result_mat_CPU.nnz, C_m, C_n, tuplesC, false, true);
        //(*C_cont).PrintInfo();
        if (i != Aself) delete ARecv;
        // dealloc(input_A_recv_GPU);

        if (i != Bself) delete BRecv;
        // dealloc(input_B_recv_GPU);

        if (!C_cont->isZero())
            tomerge.push_back(C_cont);
        else
            delete C_cont;
    }
    t1 = MPI_Wtime();
    // dealloc(input_A2_GPU);
    // dealloc(input_B2_GPU);
    SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
    SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
    // A2seq->Transpose();
    //     B2seq->Transpose();
    if (clearA) {
        delete A2seq;
        delete A.spSeq;
        A.spSeq = NULL;
    } else {
        // A1seq->Transpose();
        // A2seq->Transpose();
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
    // checkingTime += MPI_Wtime() - t1;
    //  printf("%.6lf\n", mpi_overhead);
    UDERO *C = new UDERO(MergeAll<SR>(tomerge, C_m, C_n, true), false);
    // printf("Full output has rows = %i, cols = %i, nnz = %i\n", C->getnrow(),
    // C->getncol(), C->getnnz());
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    over += MPI_Wtime() - t1;
    // std::cout << over << "\n";
    std::cerr << "Converting time: " << convertingtime << std::endl;
    HANDLE_ERROR(cudaGetLastError());
    return SpParMat<IU, NUO, UDERO>(C, GridC);  // return the result object	// return the result object
}

// CUDA implementation for Mult_AnXBn_Synch, use SpCuCRows as local data structure.
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_Synch_CUDA(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B, bool clearA = false, bool clearB = false)
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

    SpParHelper::GetSetSizes(*(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
    SpParHelper::GetSetSizes(*(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());

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

#endif  // __CUDACC__