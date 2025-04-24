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
// #include <mpi.h>
#include <sys/time.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "CombBLAS/CombBLAS.h"
// #include "../include/GALATIC/source/device/Multiply.cuh"

#include <cxxopts.hpp>

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
int GPUTradeoff = 0;
int main(int argc, char *argv[])
{
#pragma omp parallel
{
#pragma omp single
    std::cout << "Number of threads in parallel region: " << omp_get_num_threads() << std::endl;
}

    typedef int32_t IT;
    typedef double NT;
    // typedef PlusTimesSRing<double, double> MinPlusSRing;
    // typedef SelectMaxSRing<bool, int64_t> SR;
    typedef PlusTimesSRing<NT, NT> PTFF;
    // shared_ptr<CommGrid> fullWorld;
    // fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
    // clang-format off
    cxxopts::Options options("achalf", "ACSpGEMM Half SpGEMM Test");
    options.add_options()
    ("Aname", "Matrix A path", cxxopts::value<string>())
    ("Bname", "Matrix B path", cxxopts::value<string>())
    ("Cname", "Matrix C path", cxxopts::value<string>());
    // clang-format on
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    // Print all parsed arguments
    std::cerr << "Parsed options:" << std::endl;
    for (const auto &kv : result.arguments()) {
        std::cerr << "  --" << kv.key() << " = " << kv.value() << std::endl;
    }
    string Aname = result["Aname"].as<string>();
    string Bname = result["Bname"].as<string>();
    string Cname = result["Cname"].as<string>();

    SpDCCols<IT, NT> A, B, Cref;
    A.ReadMM(Aname);
    B.ReadMM(Bname);
    Cref.ReadMM(Cname);
    std::cerr << A.getnnz() << std::endl;
    Timer timer;
    timer.start();
    SpTuples<IT, NT> *sptupleptr = LocalHybridSpGEMM<PTFF, NT, IT>(A, B, false, false);
    timer.stop();
    double time = timer.elapsedSeconds();
    std::cerr << "combblas LocalHybridSpGEMM time " << time << std::endl;
    SpDCCols<IT,NT> Ccomb(*sptupleptr, false);
    if (Ccomb == Cref) {
        std::cerr << "results are correct!" << std::endl;
    }


    // std::cerr << sptupleptr->getnnz() << std::endl;
    // COO<NT> combblascpures;
    // combblascpures.alloc(sptupleptr->getnrow(), sptupleptr->getncol(), sptupleptr->getnnz());
    // for (size_t i = 0; i < sptupleptr->getnnz(); ++i) {
    //     combblascpures.row_ids[i] = sptupleptr->rowindex(i);
    //     combblascpures.col_ids[i] = sptupleptr->colindex(i);
    //     combblascpures.data[i] = sptupleptr->numvalue(i);
    // }
    // combblascpures.sorted(true);
    delete sptupleptr;
    // CSR<NT> cpucsr;
    // convert(cpucsr, combblascpures);
    // B.ReadMM(Bname);
    // cerr << "Constructing objects:" << endl;
    // cerr << A.getnnz() << endl;
    // cerr << B.getnnz() << endl;
    // SpTuples<int32_t, double>* Ctp = LocalHybridSpGEMM<PTFF, double, int32_t, double, double>(A,B,false,false);
    // SpCCols<int32_t,double> C(*Ctp,false);
    // delete Ctp;

    // COO<double> cooA = loadMTX<double>(Aname.c_str());
    // cooA.saveMTX("cooAtest.mtx");
    // CSR<double> csrA;
    // convert(csrA, cooA);
    // dCSR<double> dcsrA;
    // HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());

    // convert(dcsrA, csrA);
    // dCSR<double> dcsrB(dcsrA);
    // HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    // CSR<NT> csrC = GPULocalMultiply<Arith_SR, NT, NT, NT>(dcsrA, dcsrB);
    // COO<NT> cooC;
    //
    // convert(cooC, csrC);
    //
    // cooC.sorted(true);
    // cooC.saveMTX("cooC.mtx");
    // std::cerr << csrC.nnz << std::endl;
    // HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
    // bool correct = true;
    // double maxdiff = 0.0;
    //
    // //! compare results
    // for (size_t i = 0; i < combblascpures.nnz; ++i) {
    //     if (combblascpures.row_ids[i] != cooC.row_ids[i]) {
    //         std::cerr << "row offset not correct !" << std::endl;
    //         std::cerr << "combblascpu: " << combblascpures.row_ids[i] << ", cooC: " << cooC.row_ids[i] << std::endl;
    //         break;
    //     }
    //     if (combblascpures.col_ids[i] != cooC.col_ids[i]) {
    //         std::cerr << "col offset not correct !" << std::endl;
    //         std::cerr << "combblascpu: " << combblascpures.col_ids[i] << ", cooC: " << cooC.col_ids[i] << std::endl;
    //         break;
    //     }
    // }
    // for (size_t i = 0; i < csrC.rows + 1; ++i) {
    //     if (cpucsr.row_offsets[i] != csrC.row_offsets[i]) {
    //         std::cerr << "row offset not correct!" << std::endl;
    //         correct = false;
    //         break;
    //     }
    // }
    // if (correct) {
    //     for (size_t i = 0; i < csrC.nnz; ++i) {
    //         if (cpucsr.col_ids[i] != csrC.col_ids[i]) {
    //             std::cerr << "col id not correct!" << std::endl;
    //             correct = false;
    //             break;
    //         }
    //     }
    // }
    // if (correct) {
    //     for (size_t i = 0; i < csrC.nnz; ++i) {
    //         maxdiff = abs(cpucsr.data[i] - csrC.data[i]) / abs(cpucsr.data[i]);
    //         if (maxdiff > 1e-6) {
    //
    //             std::cerr << "data not correct! original value is "<< cpucsr.data[i] <<" max diff is " << maxdiff <<
    //             std::endl; std::cerr << "row idx " << correct = false; break;
    //         }
    //     }
    // }
    // if (correct) {
    //     std::cerr << "results identical, max diff is " << maxdiff << std::endl;
    // }
    return 0;
}