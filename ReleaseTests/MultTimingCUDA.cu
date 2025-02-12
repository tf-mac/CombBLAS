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
#include "mpi_proto.h"

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

//------------------------------------------------------------------------------
// Benchmark wrapper for Basic and Memory Optimized SpGEMM.
// Runs the given SpGEMM routine (with a warm-up iteration skipped) and returns
// the average execution time (in ms).
template <typename SpGEMMFunc, typename LTYPE>
double benchmarkSpGEMM(SpGEMMFunc spgemmFunc, LTYPE &A, LTYPE &B, int iterations)
{
}

// Outline of debug stages
// stage = 0: LocalHybrid does not run/immediately returns
// stage = 1: LocalHybrid mallocs and transposes as needed, but returns immediately after
// stage = 2: LocalHybrid runs the kernel, but does not perform cleanup
// stage = 3: Full run of LocalHybrid
// stages 1 & 2 may lead to memory leaks, be aware on memory limited systems
int main(int argc, char *argv[])
{
    int * ptr = (int*)malloc(1000);
    int nprocs, myrank;
    int host_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    typedef PlusTimesSRing<ElementType, ElementType> PTDOUBLEDOUBLE;

    if (argc < 5) {
        if (myrank == 0) {
            cout << "Usage: ./MultTest <MatrixA> <MatrixB> <MatrixC>" << endl;
            cout << "<MatrixA>,<MatrixB>,<MatrixC> are absolute addresses, and files should be in "
                    "triples format"
                 << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        // test iterations
        int iterations = stoi(argv[1]);
        // spgemm test type choice: comm | comp , if comm is provided, only do communication test
        string testtype(argv[2]);
        assert(testtype == "comm" || testtype == "comp");
        string Aname(argv[3]);
        string Bname(argv[4]);
        // default no permutation , choice: perm | noperm
        string perm = "noperm";
        perm = string(argv[5]);
        assert(perm == "perm" || perm == "noperm");
        // default double buffering, choice: sync | dbuff
        string spgemmtype = "dbuff";
        spgemmtype = string(argv[6]);
        assert(spgemmtype == "dbuff" || spgemmtype == "sync");

        if (myrank == 0 || nprocs == 1) {
            std::cout << Aname << std::endl;
            std::cout << Bname << std::endl;
            std::cout << nprocs << std::endl;
            std::string filename = Aname + "_output.txt";
            FILE *f = fopen(filename.c_str(), "a");
            if (f == NULL) {
                printf("failed to open file: permission issue ?\n");
                exit(1);
            }
            // cout << "Double buffered CUDA multiplications finished" << endl;
            fprintf(f, "Input A: %s, with NPROCS: %i\n", Aname.c_str(), nprocs);
            fclose(f);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        typedef PlusTimesSRing<double, double> MinPlusSRing;
        typedef SelectMaxSRing<bool, int64_t> SR;

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
        {
            Ccpu = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
        }
        double gputime = 0.0;
        if (testtype == "comm") {
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
                    Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, ElementType, PSpMat<double>::DCCols>(A, B);
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
                    Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<double>::DCCols>(A, B);
                    MPI_Barrier(MPI_COMM_WORLD);
                    double t2 = MPI_Wtime();
                    if (iter > 0) {
                        totalTime += t2 - t1;
                    }
                }
                cputime = totalTime / iterations;
                if (myrank == 0) {
                    std::cerr << "CUDA SpGEMM Type: " << spgemmtype << ", cpu time is " << cputime << " ms, gpu time is " << gputime << " ms" << std::endl;
                }
            } else if (spgemmtype == "sync") {
            }
        }
    }
    MPI_Finalize();
    return 0;
}
