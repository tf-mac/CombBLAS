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

#include <mpi.h>
#include <sys/time.h>

#include <cxxopts.hpp>
#include <iostream>
#include <sstream>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpDCCols.h"
using namespace std;
using namespace combblas;

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif
// double combblas::convertingtime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif
int GPUTradeoff = 1024 * 1024;
int iterations = 50;

extern PerformanceRecorder combblas::prspgemmdbuffcuda;

template <class SR, class IT, class NT, class DER>
void Benchmark_SpGEMM(string Aname, string Bname, string testtype, string spgemmtype, int myrank, int nprocs)
{
    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

    SpParMat<IT, NT, DER> Agpu(fullWorld);
    SpParMat<IT, NT, DER> Bgpu(fullWorld);
    SpParMat<IT, NT, DER> Cgpu(fullWorld);
    SpParMat<IT, NT, SpDCCols<IT, NT>> Acpu(fullWorld);
    SpParMat<IT, NT, SpDCCols<IT, NT>> Bcpu(fullWorld);
    SpParMat<IT, NT, SpDCCols<IT, NT>> Ccpu(fullWorld);
    Acpu.ParallelReadMM(Aname, true, maximum<NT>());
    Agpu.ParallelReadMM(Aname, true, maximum<NT>());
    Agpu.PrintInfo();
    if (Aname == Bname) {
        Bgpu = Agpu;
        Bcpu = Bgpu;
        if (myrank == 0) std::cerr << "A and B are the same." << std::endl;
    } else {
        Bcpu.ParallelReadMM(Bname, true, maximum<NT>());
        Bgpu.ParallelReadMM(Bname, true, maximum<NT>());
        Bcpu.PrintInfo();
        Bgpu.PrintInfo();
    }

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

    // cpu version, baseline
    Ccpu = Mult_AnXBn_Synch<SR, NT, SpDCCols<IT, NT>>(Acpu, Bcpu);
    // std::cerr << Cdcsc.getnnz() << std::endl;
    // double gputime = 0.0;
    if (testtype == "test") {
        // correctness check
        if (spgemmtype == "dbuff") {
            // Cgpu = Mult_AnXBn_DoubleBuff_CUDA<SR, NT, SpDCCols<IT, NT> >(Agpu, Bgpu);
            Cgpu = Mult_AnXBn_DoubleBuff_CUDA_dCSR<SR, NT, SpDCCols<IT, NT>>(Acpu, Bcpu);
        } else if (spgemmtype == "synch") {
            // Cgpu = Mult_AnXBn_Synch_CUDA<SR, NT, DER>(Agpu, Bgpu);
        }
        // prspgemmdbuffcuda.OutputRecords("tmp.txt");
        // prspgemmdbuffcuda.PrintInfo();
        // auto cgpunnz = Cgpu.getnnz();
        // if (myrank == 0)std::cerr << "nnz is " << cgpunnz << std::endl;
        // Ccpu = Cdcscgpu;
        auto gpunnz = Cgpu.getnnz();
        auto cpunnz = Ccpu.getnnz();
        if (myrank == 0) std::cerr << "gpu nnz" << gpunnz << "cpu nnz" << cpunnz << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        // then we need to convert Cgpu to Ccpu using SPDCCols<IT,NT> as local
        if (Cgpu == Ccpu) {
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
                // Mult_AnXBn_DoubleBuff_CUDA<SR, NT, SpDCCols<IT, NT>>(Acpu, Bcpu);
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
                // Mult_AnXBn_DoubleBuff<SR, NT, SpDCCols<IT, NT>>(Acpu, Bcpu);
                MPI_Barrier(MPI_COMM_WORLD);
                double t2 = MPI_Wtime();
                if (iter > 0) {
                    totalTime += t2 - t1;
                }
            }
            cputime = totalTime / iterations;
            if (myrank == 0) {
                std::cerr << "CUDA SpGEMM Type: " << spgemmtype << ", cpu time is " << cputime << " ms, gpu time is "
                          << gputime << " ms" << std::endl;
            }
        } else if (spgemmtype == "synch") {
            // test Mult_AnXBn_Synch
            if (myrank == 0) std::cerr << "doing spgemm synch " << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#ifdef GCOMM_NCCL
    int numGPUs = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&numGPUs));
    if (numGPUs < 2) {
        std::cerr << "Need at least 2 GPUs for this example." << std::endl;
        return -1;
    }
#endif
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    // std::string logfilename = "logfile_Rank" + std::to_string(myrank) + ".txt";
    // freopen(logfilename.c_str(), "a", stderr);

    // Iter: test iterations, how many iteration you want to benchmark the performance
    // i suggest it should be > 1000 to make it stable. but you can start with smaller size
    // in test mode, this param is ignored.

    // Testtype: spgemm test type choice: bench | comm | test
    // if bench: bench the target function,
    // if test: check the correctness
    // if comm: check the communication, without launch local spgemm kernel

    // Perm: default no permutation , choice: perm | noperm

    // Func: default double buffering, choice: synch | dbuff

    // SR: semiring type you want to test
    // e.g. pt -> PlusTimesSRing

    // clang-format off
    options.add_options()
    ("Iter", "iteration", cxxopts::value<int>()) // a bool parameter
    ("Testtype", "test type", cxxopts::value<string>())
    ("Aname", "Matrix A path", cxxopts::value<string>())
    ("Bname", "Matrix B path", cxxopts::value<string>())
    ("Perm", "File name", cxxopts::value<std::string>())
    ("Func", "File name", cxxopts::value<std::string>())
    ("SR", "Semiring", cxxopts::value<string>()->default_value("pt"))
    ("Dtype", "Numeric type", cxxopts::value<string>()->default_value("double"))
    ("Ltype", "Local sparse matrix type", cxxopts::value<string>()->default_value("dcsc"));
    // clang-format on
    auto result = options.parse(argc, argv);
    if (myrank == 0) {
        // Print all parsed arguments
        std::cerr << "Parsed options:" << std::endl;
        for (const auto &kv : result.arguments()) {
            std::cerr << "  --" << kv.key() << " = " << kv.value() << std::endl;
        }
    }

    // wrap it

    const int iterations = result["Iter"].as<int>();
    const string testtype = result["Testtype"].as<string>();
    const string spgemmtype = result["Func"].as<string>();
    assert(spgemmtype == "dbuff" || spgemmtype == "synch");

    const string Aname = result["Aname"].as<string>();
    const string Bname = result["Bname"].as<string>();

    const string testsr = result["SR"].as<string>();
    assert(testsr == "pt");

    const string dtype = result["Dtype"].as<string>();
    assert(dtype == "double" || dtype == "float");
    // numeric data type you want to specify
    const string localtype = result["Ltype"].as<string>();
    assert(localtype == "dcsc" || localtype == "csr" || localtype == "cucsr");
    MPI_Barrier(MPI_COMM_WORLD);

    // clang-format off
    if (dtype == "double") {
        if (localtype == "dcsc") {
            if (testsr == "pt") {
                Benchmark_SpGEMM<PlusTimesSRing<double, double>, int32_t, double, SpDCCols<int32_t, double> >
                        (Aname, Bname, testtype, spgemmtype, myrank, nprocs);
            }
        }
        // else if (localtype == "cucsr") {
        //     if (testsr == "pt") {
        //         Benchmark_SpGEMM<PlusTimesSRing<double, double>, int32_t, double, SpCuCRows<int32_t, double> >
        //                 (Aname, Bname, testtype, spgemmtype, myrank, nprocs);
        //     }
        // } else if (localtype == "csr") {
        //     if (testsr == "pt") {
        //         Benchmark_SpGEMM<PlusTimesSRing<double, double>, int32_t, double, SpCRows<int32_t, double> >
        //                 (Aname, Bname, testtype, spgemmtype, myrank, nprocs);
        //     }
        // }
    } else if (dtype == "float") {
        // if (localtype == "dcsc") {
        //     if (testsr == "pt") {
        //         Benchmark_SpGEMM<PlusTimesSRing<float, float>, int32_t, float, SpDCCols<int32_t, float> >
        //                 (Aname, Bname, testtype, spgemmtype, myrank, nprocs);
        //     }
        // } else if (localtype == "cucsr") {
        //     if (testsr == "pt") {
        //         Benchmark_SpGEMM<PlusTimesSRing<float, float>, int32_t, float, SpCuCRows<int32_t, float> >
        //                 (Aname, Bname, testtype, spgemmtype, myrank, nprocs);
        //     }
        // } else if (localtype == "csr") {
        //     if (testsr == "pt") {
        //         Benchmark_SpGEMM<PlusTimesSRing<float, float>, int32_t, float, SpCRows<int32_t, float> >
        //                 (Aname, Bname, testtype, spgemmtype, myrank, nprocs);
        //     }
        // }
    }
    // clang-format on
    fclose(stderr);
    MPI_Finalize();
    return 0;
}