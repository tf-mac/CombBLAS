//
// Created by Yuxi Hong on 2/25/25.
//
#include "CombBLAS/PerfRecorder.h"

#include <mpi.h>
#ifdef THREADED
#include <omp.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
namespace combblas
{
/*****************************************************************************
 * Global Performance Recorder Instance
 *****************************************************************************/

PerformanceRecorder pr1dspgemm;
PerformanceRecorder PR_Mult_AnXBn_DoubleBuff;
PerformanceRecorder prspgemmdbuffcuda;
void Init_PR_Mult_AnXBn_DoubleBuff(){
    prspgemmdbuffcuda.GetRankInfo();
    int nprocs = pr1dspgemm.nprocs;
    int nthreads = 1;  // namespace combblas
#ifdef THREADED
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif
    // PR_Mult_AnXBn_DoubleBuff.
}


void InitPR_SpGEMMDBUFFCUDA()
{
    prspgemmdbuffcuda.GetRankInfo();
    int nprocs = pr1dspgemm.nprocs;
    int nthreads = 1;  // namespace combblas
#ifdef THREADED
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif
    prspgemmdbuffcuda.RegisterDouble("transposeT(ms)", 2, 1000.);
}
// initialize performance recorder of spgemm 1d algorithm
void InitPR_SPGEMM1D()
{
    /*init opt*/
    /*register performance counter metrics.*/
    pr1dspgemm.GetRankInfo();
    int nprocs = pr1dspgemm.nprocs;
    int nthreads = 1;  // namespace combblas
#ifdef THREADED
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif
    // these permutation metrics should be seperated.
    // pr1dspgemm.RegisterDouble("Rseq(ms)",2,1000.); // report time in ms.
    // pr1dspgemm.RegisterDouble("gpprep(ms)",2,1000.); // report time in ms.
    // pr1dspgemm.RegisterDouble("metis(ms)",2,1000.); // report time in ms.
    // pr1dspgemm.RegisterDouble("permute(ms)",2,1000.); // report time in ms.
    pr1dspgemm.RegisterDouble("prepT(ms)", 2, 1000.);  // report time in ms.
    pr1dspgemm.RegisterDouble("commT(ms)", 2, 1000.);
    pr1dspgemm.RegisterDouble("commOH(ms)", 2, 1000.);
    pr1dspgemm.RegisterDouble("compT(ms)", 2, 1000.);
    pr1dspgemm.RegisterDouble("constCT(ms)", 2, 1000.);
    pr1dspgemm.RegisterInteger("Alocalnnz");
    pr1dspgemm.RegisterDouble("Alocalmem(MB)", 2, 1.0);
    pr1dspgemm.RegisterInteger("Aneednnz");
    pr1dspgemm.RegisterDouble("Aneedmem(MB)", 2, 1.0);
    pr1dspgemm.RegisterDouble("Aneedmem(Pt)", 2, 1.0);
    pr1dspgemm.RegisterInteger("Bnnz    ");
    pr1dspgemm.RegisterDouble("Bmem(MB)", 2, 1.0);
    pr1dspgemm.RegisterInteger("Cnnz    ");
    pr1dspgemm.RegisterDouble("Flops    ");
    pr1dspgemm.RegisterDouble("FlopsLB    ", 2, 1.0);
    pr1dspgemm.RegisterDouble("Cmem(MB)", 2, 1.0);
    pr1dspgemm.RegisterDouble("Cmem(MB)LB", 2, 1.0);
    pr1dspgemm.RegisterInteger("useheap", 2, 1.0);
    pr1dspgemm.RegisterInteger("usehash", 2, 1.0);
    pr1dspgemm.RegisterDouble("TotTime(s)", 2, 1.0);  // report total time in s. in noanalysis mode.
    pr1dspgemm.RegisterDouble("AnaTotT(s)", 2, 1.0);  // report analysis mode total time in s
    // register vector variables of spgemm1d
    pr1dspgemm.RegisterIntegerVector("rdmanum", nprocs, 2, 1.0);
    pr1dspgemm.RegisterDoubleVector("rdmamem", nprocs, 2, 1.0);
    pr1dspgemm.RegisterDoubleVector("rdmatime", nprocs, 2, 1.0);
    pr1dspgemm.RegisterDoubleVector("ThdHashTime", nthreads, 2, 1.0);
    pr1dspgemm.RegisterDoubleVector("ThdHeapTime", nthreads, 2, 1.0);
}  // namespace combblas

template <typename T>
int64_t getindex(std::vector<T>& vec, T a)
{
    for (int64_t i = 0; i < vec.size(); i++)
        if (vec[i] == a) return i;
    return -1;
}

void PerformanceRecorder::GetRankInfo()
{
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
}

void PerformanceRecorder::RegisterDouble(std::string printinfo, int printspace, double scale)
{
    if (doublemap.find(printinfo) != doublemap.end()) {
        std::cerr << printinfo << " already regiestered! " << std::endl;
        exit(0);
    }
    doublekeys.push_back(printinfo);
    doublemap[printinfo] = 0.0;
    printseq.push_back(printinfo);
    printspacemap[printinfo] = printspace;
    scalemap[printinfo] = scale;
}

void PerformanceRecorder::RegisterInteger(std::string printinfo, int printspace, double scale)
{
    if (integermap.find(printinfo) != integermap.end()) {
        std::cerr << printinfo << " already regiestered! " << std::endl;
        exit(0);
    }
    integerkeys.push_back(printinfo);
    integermap[printinfo] = 0.0;
    printseq.push_back(printinfo);
    printspacemap[printinfo] = printspace;
    scalemap[printinfo] = scale;
}

void PerformanceRecorder::RegisterDoubleVector(std::string printinfo, int length, int printspace, double scale)
{
    if (doublevectormap.find(printinfo) != doublevectormap.end()) {
        std::cerr << printinfo << " already regiestered! " << std::endl;
        exit(0);
    }
    doublevectorkeys.push_back(printinfo);
    doublevectormap[printinfo].resize(length, 0.0);
}

void PerformanceRecorder::RegisterIntegerVector(std::string printinfo, int length, int printspace, double scale)
{
    if (integervectormap.find(printinfo) != integervectormap.end()) {
        std::cerr << printinfo << " already regiestered! " << std::endl;
        exit(0);
    }
    integervectorkeys.push_back(printinfo);
    integervectormap[printinfo].resize(length, 0.0);
}

void PerformanceRecorder::RegisterDoubleMatrix(std::string printinfo, std::vector<int> shape, int printspace,
                                               double scale)
{
    if (doublematrixmap.find(printinfo) != doublematrixmap.end()) {
        std::cerr << printinfo << " already regiestered! " << std::endl;
        exit(0);
    }
    doublevectorkeys.push_back(printinfo);
    doublematrixmap[printinfo].resize(shape[0]);
    for (int i = 0; i < shape[0]; i++) doublematrixmap[printinfo][i].resize(shape[1], 0.0);
}

void PerformanceRecorder::RegisterIntegerMatrix(std::string printinfo, std::vector<int> shape, int printspace,
                                                double scale)
{
    if (integermatrixmap.find(printinfo) != integermatrixmap.end()) {
        std::cerr << printinfo << " already regiestered! " << std::endl;
        exit(0);
    }
    integerkeys.push_back(printinfo);
    integermatrixmap[printinfo].resize(shape[0]);
    for (int i = 0; i < shape[0]; i++) integermatrixmap[printinfo][i].resize(shape[1], 0.0);
}

void PerformanceRecorder::Reset()
{
    // clear scalar
    for (auto& it : doublemap) it.second = 0.0;
    for (auto& it : integermap) it.second = 0.0;
    // clear vector
    for (const auto& x : doublevectorkeys) {
        for (auto& v : doublevectormap[x]) v = 0.0;
    }
    for (const auto& x : integervectorkeys) {
        for (auto& v : integervectormap[x]) v = 0;
    }
    // clear matrix
    for (const auto& x : doublematrixkeys) {
        // iterate 2d vector
        for (auto& v : doublematrixmap[x]) {
            for (auto& y : v) {
                y = 0.0;
            }
        }
    }
    for (const auto& x : integermatrixkeys) {
        // iterate 2d vector
        for (auto& v : integermatrixmap[x]) {
            for (auto& y : v) {
                y = 0;
            }
        }
    }
}

std::string PerformanceRecorder::GatherLocalString(std::string localstr) const
{
    int emptystr = 0;
    if (localstr.empty()) {
        emptystr = 1;
    }

    std::string ret;
    int sendsize = localstr.size();
    std::vector<int> recvsize(nprocs, 0);
    MPI_Allgather(&sendsize, 1, MPI_INT, recvsize.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> recvdisp(nprocs, 0);
    // simple prefix sum
    for (int i = 1; i < recvdisp.size(); i++) {
        recvdisp[i] = recvdisp[i - 1] + recvsize[i - 1];
    }
    int totalelems = std::accumulate(recvsize.begin(), recvsize.end(), 0);
    ret.resize(totalelems);
    char tmpc[ret.size()];
    memcpy(tmpc, ret.data(), ret.size());
    MPI_Allgatherv(localstr.data(), localstr.size(), MPI_CHAR, tmpc, recvsize.data(), recvdisp.data(), MPI_CHAR,
                   MPI_COMM_WORLD);
    return ret;
}

void PerformanceRecorder::PrintInfo(std::string description)
{
    if (myrank == 0) {
        std::cerr << "Description:" << description << std::endl;
        PrintScalarInternal(std::cerr);
        PrintVectorInternal(std::cerr);
        PrintMatrixInternal(std::cerr);
    }
}

void PerformanceRecorder::OutputRecords(std::string filename, std::string description, bool removeexistingfile)
{
    if (description == "") description = filename;
    std::ofstream os;
    if (myrank == 0) {
        if (removeexistingfile) {
            // SpHelper::RemoveExistingFile(filename);
            os = std::ofstream(filename);
        } else {
            os = std::ofstream(filename, std::ios::app);
        }
        if (os.is_open()) {
            os << "\\%\\%Performance Recorder: " << description << std::endl;  // heade
            os << "Rank , ";
            for (auto key : printseq) {
                std::string suffix = "";
                for (int i = 0; i < printspacemap[key]; i++) suffix += " ";
                suffix += " ,";
                os << key << suffix;
            }
            os << std::endl;
        } else {
            std::cerr << "ERROR: PerformanceRecorder: Rank 0 can't open " << filename << " for write records!"
                      << std::endl;
        }
        std::cerr << "write performance reports to " << filename << std::endl;
    }
    PrintScalarInternal(os);
    PrintVectorInternal(os);
    PrintMatrixInternal(os);
    if (myrank == 0) os.close();
}
void PerformanceRecorder::OutputRecords(std::ostream& os)
{
    PrintHeaderInternal(os);
    PrintScalarInternal(os);
    PrintVectorInternal(os);
    PrintMatrixInternal(os);
    if (myrank == 0) os << std::endl;
}
void PerformanceRecorder::PrintHeaderInternal(std::ostream& os)
{
    if (myrank == 0) {
        os << description;
        os << "Scalar performance variables:" << std::endl;
        os << "Rank , ";
        for (auto key : printseq) {
            std::string suffix = "";
            for (int i = 0; i < printspacemap[key]; i++) suffix += " ";
            suffix += " ,";
            os << key << suffix;
        }
        os << std::endl;
    }
}
void PerformanceRecorder::PrintScalarInternal(std::ostream& os)
{
    std::stringstream tmpos;
    tmpos << std::defaultfloat << std::left << std::setw(6) << myrank << ",";
    for (auto key : printseq) {
        if (doublemap.find(key) != doublemap.end()) {
            int64_t j = getindex(doublekeys, key);
            tmpos << std::defaultfloat << std::setprecision(5)
                  << std::setw(doublekeys[j].size() + printspacemap[doublekeys[j]] + 1) << std::left
                  << double(doublemap[key] * scalemap[key]) << ",";  // time is in ms unit.
        }
        if (integermap.find(key) != integermap.end()) {
            int64_t j = getindex(integerkeys, key);
            tmpos << std::setprecision(3) << std::scientific
                  << std::setw(integerkeys[j].size() + printspacemap[integerkeys[j]] + 1) << std::left
                  << int(integermap[key] * scalemap[key]) << ",";
        }
    }
    tmpos << std::endl;
    std::string mergedstring = GatherLocalString(tmpos.str());
    if (myrank == 0) os << mergedstring;
}

void PerformanceRecorder::PrintVectorInternal(std::ostream& os)
{
    for (const auto& x : doublevectorkeys) {
        std::string printperrank;
        std::stringstream tmpstream;
        if (myrank == 0) tmpstream << "Double Vector Performance Variable:" << x << std::endl;
        tmpstream << "Rank";
        int prefix = 5 - std::to_string(myrank).size();
        for (int i = 0; i < prefix; i++) tmpstream << " ";
        tmpstream << myrank << ":  ";
        for (auto v : doublevectormap[x]) tmpstream << std::defaultfloat << std::setprecision(5) << std::setw(12) << v;
        tmpstream << std::endl;
        std::string recvbuff = GatherLocalString(tmpstream.str());
        if (myrank == 0) {
            os << recvbuff;
        }
    }
    for (const auto& x : integervectorkeys) {
        std::string printperrank;
        std::stringstream tmpstream;
        if (myrank == 0) tmpstream << "Integer Vector Performance Variable:" << x << std::endl;
        tmpstream << "Rank";
        int prefix = 5 - std::to_string(myrank).size();
        for (int i = 0; i < prefix; i++) tmpstream << " ";
        tmpstream << myrank << ":  ";
        for (auto v : integervectormap[x]) {
            tmpstream << std::setprecision(3) << std::scientific << std::setw(12) << std::left << v;
        }
        tmpstream << std::endl;
        std::string recvbuff = GatherLocalString(tmpstream.str());
        if (myrank == 0) {
            os << recvbuff;
        }
    }
}

void PerformanceRecorder::PrintMatrixInternal(std::ostream& os)
{
    for (int i = 0; i < doublematrixkeys.size(); i++) {
        std::string printperrank;
        std::stringstream tmpstream;
        auto x = doublematrixkeys[i];
        if (myrank == 0) tmpstream << "Double Matrix Performance Variable:" << x << std::endl;
        tmpstream << "Rank: " << myrank << std::endl;
        // iterate 2d vector
        for (auto v : doublematrixmap[x]) {
            for (auto y : v) {
                tmpstream << std::defaultfloat << std::setprecision(5) << std::setw(12) << y;
            }
            tmpstream << std::endl;
        }
        tmpstream << std::endl;
        std::string recvbuff = GatherLocalString(tmpstream.str());
        if (myrank == 0) {
            os << recvbuff;
        }
    }

    for (int i = 0; i < integermatrixkeys.size(); i++) {
        std::string printperrank;
        std::stringstream tmpstream;
        auto x = integermatrixkeys[i];
        if (myrank == 0) tmpstream << "Integer Matrix Performance Variable:" << x << std::endl;
        tmpstream << "Rank: " << myrank << std::endl;
        // iterate 2d vector
        for (auto v : integermatrixmap[x]) {
            for (auto y : v) {
                tmpstream << std::setprecision(3) << std::scientific << std::setw(12) << std::left << y;
            }
            tmpstream << std::endl;
        }
        tmpstream << std::endl;
        std::string recvbuff = GatherLocalString(tmpstream.str());
        if (myrank == 0) {
            os << recvbuff;
        }
    }
}

}  // namespace combblas