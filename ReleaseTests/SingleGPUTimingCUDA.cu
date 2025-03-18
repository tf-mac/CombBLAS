// a single gpu spgemm test for timing
// to check whether cpu time is slower than gpu.

#include <mpi.h>
#include <sys/time.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "CombBLAS/CombBLAS.h"

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
int GPUTradeoff = 1024 * 1024;
#define ElementType double
int ITERATIONS = 50;
// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"

#define IT uint32_t

template <class NT>
class PSpMat
{
   public:
    typedef SpDCCols<IT, NT> DCCols;
    typedef SpParMat<IT, NT, DCCols> MPI_DCCols;
};

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (nprocs > 1) {
        if (myrank == 0) {
            cout << "This test is for single GPU, please run with 1 process" << endl;
        }
        MPI_Finalize();
        return 0;
    }
    typedef PlusTimesSRing<ElementType, ElementType> PTFF;
    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./MultTest <MatrixA> <MatrixB>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    if(myrank == 0){
        std::cerr << "openmp threads size " << cblas_splits << std::endl;
    }
    {
        string Aname(argv[1]);
        string Bname(argv[2]);
        MPI_Barrier(MPI_COMM_WORLD);

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        // construct objects
        PSpMat<double>::MPI_DCCols A(fullWorld);
        PSpMat<double>::MPI_DCCols B(fullWorld);
        PSpMat<double>::MPI_DCCols C(fullWorld);

        A.ParallelReadMM(Aname, true, maximum<double>());
#ifndef NOGEMM
        B.ParallelReadMM(Bname, true, maximum<double>());
#endif
        A.PrintInfo();
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        SpTuples<IT, ElementType>* sptuples =
            LocalHybridSpGEMM<PTFF, ElementType, IT>(A.seq(), B.seq(), false, false, nullptr);
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        if (myrank == 0) {
            cout << "CPU SpGEMM Time taken: " << t2 - t1 << " seconds" << endl;
        }
        {
            dCSR<ElementType> dA;
            dCSR<ElementType> dB;
            dCSR<ElementType> dC;
            convertCSR(A.seqptr(),dA, 0);
            convertCSR(B.seqptr(),dB, 0);
            gpuErrchk(cudaDeviceSynchronize());
            double t1 = MPI_Wtime();
            CSR<ElementType> hC = GPULocalMultiply<PTFF, ElementType, ElementType, ElementType>(dA, dB);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaGetLastError());
            double t2 = MPI_Wtime();
            std::cerr << "GPU SpGEMM Time taken: " << t2 - t1 << " seconds" << std::endl;
        }

        delete sptuples;
    }
    MPI_Finalize();
    return 0;
}
