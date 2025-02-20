/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California

 permission is hereby granted, free of charge, to any person obtaining a copy
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

// #include <cuda.h>

#include <mpi.h>
#include <sys/time.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/ParFriends.h"
// #include "mpi_proto.h"

using namespace std;
using namespace combblas;

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif
double combblas::convertingtime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif
int GPUTradeoff = 1024 * 1024;
#define ElementType double
int iterations = 50;

// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
   public:
    typedef SpDCCols<uint32_t, NT> DCCols;
    typedef SpParMat<uint32_t, NT, DCCols> MPI_DCCols;
};

int main(int argc, char *argv[])
{
    int nprocs, myrank;
    int host_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (true) {
        // wrap it
        // test iterations, how many iteration you want to benchmark the performance
        // i suggest it should be > 1000 to make it stable. but you can start with smaller size
        // in test mode, this param is ignored.
        int iterations = stoi(argv[1]);

        // spgemm test type choice: bench | comm | test
        // if bench: bench the target function,
        // if test: check the correctness
        // if comm: check the communication, without launch local spgemm kernel
        string testtype(argv[2]);
        if (myrank == 0) std::cerr << "parsing paramters: testtype " << testtype << std::endl;
        assert(testtype == "comm" || testtype == "comp" || testtype == "test");

        // A and B matrix name in absolute path
        string Aname(argv[3]);
        string Bname(argv[4]);
        if (myrank == 0) {
            std::cerr << "parsing paramters: input A:" << Aname << std::endl;
            std::cerr << "parsing paramters: input B:" << Bname << std::endl;
        }

        // default no permutation , choice: perm | noperm
        string perm = "noperm";
        perm = string(argv[5]);
        if (myrank == 0) std::cerr << "parsing paramters: perm " << perm << std::endl;
        assert(perm == "perm" || perm == "noperm");

        // default double buffering, choice: sync | dbuff
        string spgemmtype = "dbuff";
        spgemmtype = string(argv[6]);
        if (myrank == 0) std::cerr << "parsing paramters: spgemmtype " << spgemmtype << std::endl;
        assert(spgemmtype == "dbuff" || spgemmtype == "synch");

        // semiring type you want to test
        // e.g. pt -> PlusTimesSRing
        string testsr;
        testsr = string(argv[7]);
        if (myrank == 0) std::cerr << "parsing paramters: testsr " << testsr << std::endl;
        assert(testsr == "pt");

        // numeric data type you want to specify
        // e.g. gdld -> "global double local double",
        // gdlf -> "global double local float"
        // gflf -> "global float local float"
        string dtype;
        dtype = string(argv[8]);
        if (myrank == 0) std::cerr << "parsing paramters: dtype " << dtype << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        // END OF PARSING PARAMS

        typedef PlusTimesSRing<double, double> PTFF;

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        // construct objects
        PSpMat<double>::MPI_DCCols A(fullWorld);
        PSpMat<double>::MPI_DCCols B(fullWorld);
        PSpMat<double>::MPI_DCCols Ccpu(fullWorld);
        PSpMat<double>::MPI_DCCols Cgpu(fullWorld);

        A.ParallelReadMM(Aname, true, maximum<double>());
        B.ParallelReadMM(Bname, true, maximum<double>());
        A.PrintInfo();
        B.PrintInfo();
        // if (perm == "perm") {
        //     if (A.getnrow() == A.getncol()) {
        //         FullyDistVec<int64_t, int64_t> p(A.getcommgrid());
        //         p.iota(A.getnrow(), 0);
        //         p.Randperm();
        //         (A)(p, p, true);  // in-place permute to save memory
        //     } else {
        //         SpParHelper::Print("nrow != ncol. Can not apply symmetric permutation.\n");
        //     }
        //     B = A;
        // }
        // TODO: add semiring and dtype logic
        Ccpu = Mult_AnXBn_Synch<PTFF, ElementType, PSpMat<ElementType>::DCCols>(A, B);
        double gputime = 0.0;
        if (testtype == "test") {
            // correctness check
            if (spgemmtype == "dbuff") {
                Cgpu = Mult_AnXBn_DoubleBuff_CUDA<PTFF, ElementType, PSpMat<double>::DCCols>(A, B);
            } else if (spgemmtype == "synch") {
                Cgpu = Mult_AnXBn_Synch_CUDA<PTFF, ElementType, PSpMat<double>::DCCols>(A, B);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (Ccpu == Cgpu) {
                if (myrank == 0) std::cerr << "Results are correct! " << std::endl;
            } else {
                if (myrank == 0) std::cerr << "Results are wrong! " << std::endl;
            }
        } else if (testtype == "comm") {
            // communication only test
        } else if (testtype == "comp") {
            // frist launch CPU version, just for correctness check
            // TODO: add semiring test
            if (spgemmtype == "dbuff") {
                double cputime = 0.0;
                double gputime = 0.0;
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                float elapsed_ms = 0.0f;
                double totalTime = 0.0;
                for (int iter = 0; iter < iterations + 1; iter++) {
                    cudaEventRecord(start, 0);
                    Mult_AnXBn_DoubleBuff_CUDA<PTFF, ElementType, PSpMat<double>::DCCols>(A, B);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&elapsed_ms, start, stop);
                    if (iter > 0) {
                        totalTime += elapsed_ms;
                    }
                }
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                gputime = totalTime / iterations;
                totalTime = 0.0;
                for (int iter = 0; iter < iterations + 1; iter++) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    double t1 = MPI_Wtime();
                    Mult_AnXBn_DoubleBuff<PTFF, ElementType, PSpMat<double>::DCCols>(A, B);
                    MPI_Barrier(MPI_COMM_WORLD);
                    double t2 = MPI_Wtime();
                    if (iter > 0) {
                        totalTime += t2 - t1;
                    }
                }
                cputime = totalTime / iterations;
                if (myrank == 0) {
                    std::cerr << "CUDA SpGEMM Type: " << spgemmtype << ", cpu time is " << cputime << " ms, gpu time is " << gputime << " ms"
                              << std::endl;
                }
            } else if (spgemmtype == "synch") {
                // test Mult_AnXBn_Synch
                if (myrank == 0) std::cerr << "doing spgemm synch " << std::endl;
            }
        }
    }
    MPI_Finalize();
    return 0;
}
