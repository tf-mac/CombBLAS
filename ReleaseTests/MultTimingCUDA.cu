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

// #include <cuda.h>

#include <mpi.h>
#include <omp.h>
#include <sys/time.h>

#include <algorithm>
#include <cxxopts.hpp>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/cuutils.h"
// #include "../include/GALATIC/source/device/Multiply.cuh"

using namespace std;
using namespace combblas;

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

#define ElementType double
int ITERATIONS = 50;
int GPUTradeoff = 1024 * 1024;

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

// Outline of debug stages
// stage = 0: LocalHybrid does not run/immediately returns
// stage = 1: LocalHybrid mallocs and transposes as needed, but returns immediately after
// stage = 2: LocalHybrid runs the kernel, but does not perform cleanup
// stage = 3: Full run of LocalHybrid
// stages 1 & 2 may lead to memory leaks, be aware on memory limited systems
int main(int argc, char *argv[])
{
    int nprocs, myrank;
    int host_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    typedef PlusTimesSRing<ElementType, ElementType> PTDOUBLEDOUBLE;

    // clang-format off
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
    ("Iter", "iteration", cxxopts::value<int>()) // a bool parameter
    ("COMMTEST", "do communication test only", cxxopts::value<bool>())
    ("Aname", "Matrix A path", cxxopts::value<string>())
    ("Bname", "Matrix B path", cxxopts::value<string>());
    // clang-format on
    auto result = options.parse(argc, argv);
    if (myrank == 0) {
        // Print all parsed arguments
        std::cerr << "Parsed options:" << std::endl;
        for (const auto &kv : result.arguments()) {
            std::cerr << "  --" << kv.key() << " = " << kv.value() << std::endl;
        }
    }
    // start mult cuda
    {
        string Aname = result["Aname"].as<string>();
        string Bname = result["Bname"].as<string>();
        int ompthreads = 0;
#ifdef THREADED
        if (myrank == 0) {
            cerr << "we have threaded marco" << endl;
        }
#endif
#pragma omp parallel
        {
            ompthreads = omp_get_num_threads();
        }
        if (myrank == 0) {
            cerr << "Aname:" << Aname << endl;
            cerr << "Bname:" << Bname << endl;
            cerr << "nprocs:" << nprocs << endl;
            cerr << "omp threads:" << ompthreads << endl;
        }
        ITERATIONS = result["Iter"].as<int>();
        bool COMMTEST = result["COMMTEST"].as<bool>();
        // if(!COMMTESTON) GPUTradeoff = 1024 * 100 * 500;
        MPI_Barrier(MPI_COMM_WORLD);
        typedef PlusTimesSRing<double, double> MinPlusSRing;
        typedef SelectMaxSRing<bool, int64_t> SR;

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        // construct objects
        PSpMat<double>::MPI_DCCols A(fullWorld);
        PSpMat<double>::MPI_DCCols B(fullWorld);
        PSpMat<double>::MPI_DCCols C(fullWorld);
        PSpMat<double>::MPI_DCCols Ccpu(fullWorld);
        A.ParallelReadMM(Aname, true, maximum<double>());
        B.ParallelReadMM(Bname, true, maximum<double>());
        bool rescorrect = false;
        {
            C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
            HANDLE_ERROR(cudaGetLastError());
            Ccpu = Mult_AnXBn_DoubleBuff<true, PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
            // Ccpu = Mult_AnXBn_Synch<true, PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
            if (C == Ccpu) {
                if (myrank == 0) cerr << "Mult_AnXBn_DoubleBuff_CUDA is correct" << endl;
                rescorrect = true;
            } else {
                if (myrank == 0) cerr << "Mult_AnXBn_DoubleBuff_CUDA is wrong" << endl;
            }
        }

        if (rescorrect) {
            // benchmark timing
            double t3, t4, tottime;
            vector<double> timevec(ITERATIONS, 0);
            // CPU benchmark
            for (int benchi = 0; benchi < ITERATIONS; ++benchi) {
                MPI_Barrier(MPI_COMM_WORLD);
                t3 = MPI_Wtime();
                Ccpu = Mult_AnXBn_DoubleBuff<false, PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
                MPI_Barrier(MPI_COMM_WORLD);
                t4 = MPI_Wtime();
                tottime += t4 - t3;
                if (tottime > 10.0 && benchi >= 2) break;  // at least do twice, if larger than 10 seconds, break
                timevec[benchi] = t4 - t3;
            }
            double t2s = 0.0;
            if (timevec.size() >= 10) {
                // for now remove first 5
                double t2s = 0.0;
                for (int i = 5; i < timevec.size(); i++) {
                    t2s += timevec[i];
                }
                t2s /= (timevec.size() - 5);
                if (myrank == 0) cerr << "Mult_AnXBn_DoubleBuff time: " << t2s << endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            // GPU Benchmark
            for (int benchi = 0; benchi < ITERATIONS; ++benchi) {
                MPI_Barrier(MPI_COMM_WORLD);
                t3 = MPI_Wtime();
                C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
                MPI_Barrier(MPI_COMM_WORLD);
                t4 = MPI_Wtime();
                tottime += t4 - t3;
                if (tottime > 10.0 && benchi >= 2) break;  // at least do twice, if larger than 10 seconds, break
                timevec[benchi] = t4 - t3;
            }
            if (timevec.size() >= 10) {
                // for now remove first 5
                double t2s = 0.0;
                for (int i = 5; i < timevec.size(); i++) {
                    t2s += timevec[i];
                }
                t2s /= (timevec.size() - 5);
                if (myrank == 0) cerr << "Mult_AnXBn_DoubleBuff_CUDA time: " << t2s << endl;
            }
            // MPI_Barrier(MPI_COMM_WORLD);
            // double t2 = MPI_Wtime();
            // MPI_Pcontrol(-1, "SpGEMM_DoubleBuff");
            // if (myrank == 0 || nprocs == 1) {
            //     std::string filename = "output" + Aname.substr(0, Aname.length() - 4) + ".txt";
            //     // std::cout << filename.c_str() << std::endl;
            //     FILE *f = fopen(filename.c_str(), "a");
            //     if (f == NULL) {
            //         printf("failed to open file: permission issue ?\n");
            //         exit(1);
            //     }
            //     // cout << "Double buffered CUDA multiplications finished" << endl;
            //     fprintf(f, "CPU Time: %.6lf\n", (t2 - t1) / ((double)ITERATIONS));
            //     fclose(f);
            // }
            // int maxhits = 0;
            // for (int j = 0; j < 500; ++j) {
            //     size_t free, total;
            //     int id;
            //     MPI_Comm_rank(MPI_COMM_WORLD, &id);
            //     cudaMemGetInfo(&free, &total);
            //     commtime = 0;
            //     comms = 0;
            //     datahits = 0;
            //     rowshits = 0;
            //     colhits = 0;
            //     cudaDeviceSynchronize();
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
            //     {
            //         C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
            //     }

            //     int svdhits = datahits + rowshits + colhits;
            //     int commper = comms;
            //     comms = 0;
            //     datahits = 0;
            //     rowshits = 0;
            //     colhits = 0;
            //     GPUTradeoff = 1024 * 100 * j;
            //     if (myrank == 0) {
            //         cerr << "GPUTradeoff" << GPUTradeoff << endl;
            //     }
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
            //     {
            //         C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
            //     }

            //     bool allt;
            //     int nnprocs;
            //     MPI_Comm_size(MPI_COMM_WORLD, &nnprocs);
            //     int newhits = datahits + rowshits + colhits;
            //     if (myrank == 0) {
            //         for (int i = 1; i < nnprocs; ++i) {
            //             MPI_Status idc;
            //             int recv;
            //             MPI_Recv(&recv, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &idc);
            //             svdhits += recv;
            //             MPI_Recv(&recv, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &idc);
            //             newhits += recv;
            //         }
            //     } else {
            //         MPI_Send(&svdhits, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            //         MPI_Send(&newhits, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            //     }
            //     allt = j > 0 && svdhits == newhits;
            //     if (j == 0) maxhits = newhits;
            //     MPI_Bcast(&allt, 1, MPI_INT, 0, MPI_COMM_WORLD);
            //     if (allt) {
            //         continue;
            //     }
            //     comms = 0;
            //     datahits = 0;
            //     rowshits = 0;
            //     colhits = 0;
            //     commtime = 0;
            //     comptime = 0;
            //     checkingTime = 0;
            //     // std::cout << "Running with tradeoff of " << 100 * j << "KB" << std::endl;
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
            //     t1 = MPI_Wtime();  // initilize (wall-clock) timer

            //     for (int i = 0; i < ITERATIONS; i++) {
            //         // std::cout << "--------------NEW ITER------------" << std::endl;
            //         C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
            //     }
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     t2 = MPI_Wtime();
            //     MPI_Pcontrol(-1, "SpGEMM_DoubleBuff");
            //     commper = 3 * nnprocs * nnprocs;
            //     if (myrank == 0 || nprocs == 1) {
            //         // std::string filename = "output" + Aname.substr(0, Aname.length() - 4) + ".txt";
            //         // // std::cout << filename.c_str() << std::endl;
            //         // FILE *f = fopen(filename.c_str(), "a");
            //         // if (f == NULL) {
            //         //     printf("failed to open file: permission issue ?\n");
            //         //     exit(1);
            //         // }
            //         // printf("%i,%i,%i,%.6lf,%.6lf,%.6lf,%.6lf\n", GPUTradeoff / 1024, newhits, maxhits,
            //         //        (t2 - t1) / (double)ITERATIONS, (commtime) / (double)ITERATIONS, comptime /
            //         //        (double)ITERATIONS, checkingTime / (double)ITERATIONS);
            //         // fprintf(f, "%i,%i,%i,%.6lf,%.6lf,%.6lf\n", GPUTradeoff / 1024, newhits, maxhits,
            //         //         (t2 - t1) / (double)ITERATIONS, (commtime) / (double)ITERATIONS, comptime /
            //         //         (double)ITERATIONS);
            //         // fclose(f);
            //     }
            //     // if (!COMMTEST) break;
            //     // if (!newhits) break;
            //     // if (nprocs == 1) break;
            //     cerr << "break!" << endl;
            //     break;
            // }
        }
    }
    MPI_Finalize();
    return 0;
}
