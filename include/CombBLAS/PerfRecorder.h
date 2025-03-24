#pragma once

#include <map>
#include <string>
#include <vector>

namespace combblas
{
class PerformanceRecorder
{
   public:
    int myrank;
    int nprocs;
    static int SpGEMM1DCallCnt;
    static int SpGEMM1DBaselineCallCnt;
    static int SpGEMM2DCallCnt;
    static int SpGEMM3DCallCnt;
    static int SpGEMM1DOPCallCnt;
    // variables about a single (int)double performance value.
    // it can be a collective op time, a memory size and flops.
    std::vector<std::string> printseq;
    std::map<std::string, double> doublemap;
    std::vector<std::string> doublekeys;
    std::map<std::string, int> integermap;
    std::vector<std::string> integerkeys;
    std::map<std::string, int> printspacemap;
    std::map<std::string, double> scalemap;

    // variables about a vector perfomrance value.
    // everyone has such a vector, and we want to write it,
    // into file. When writing, each processor value has a single line.
    std::vector<std::string> doublevectorkeys;
    std::vector<std::string> integervectorkeys;
    std::map<std::string, std::vector<double>> doublevectormap;
    std::map<std::string, std::vector<int>> integervectormap;

    // variables about a matrix perfomrance value.
    // everyone has such a matrix, and we want to write it,
    // into file. When writing, follow the rank order.
    std::vector<std::string> doublematrixkeys;
    std::vector<std::string> integermatrixkeys;
    std::map<std::string, std::vector<std::vector<double>>> doublematrixmap;
    std::map<std::string, std::vector<std::vector<int>>> integermatrixmap;
    std::string description;
    PerformanceRecorder() = default;
    void GetRankInfo();
    void RegisterDouble(std::string printinfo, int printspace = 2, double scale = 1.);
    void RegisterInteger(std::string printinfo, int printspace = 2, double scale = 1.);
    void RegisterDoubleVector(std::string printinfo, int length, int printspace = 2, double scale = 1.);
    void RegisterIntegerVector(std::string printinfo, int length, int printspace = 2, double scale = 1.);
    void RegisterDoubleMatrix(std::string printinfo, std::vector<int> shape, int printspace = 2, double scale = 1.);
    void RegisterIntegerMatrix(std::string printinfo, std::vector<int> shape, int printspace = 2, double scale = 1.);

    void Reset();

    // gather string to rank 0
    std::string GatherLocalString(std::string) const;

    void PrintInfo(std::string description = "");

    void OutputRecords(std::string filename, std::string description = "", bool removeexistingfile = true);
    void OutputRecords(std::ostream& os);

   private:
    void PrintHeaderInternal(std::ostream& os);
    void PrintScalarInternal(std::ostream& os);
    void PrintVectorInternal(std::ostream& os);
    void PrintMatrixInternal(std::ostream& os);
};

extern PerformanceRecorder pr1dspgemm;
extern PerformanceRecorder PR_Mult_AnXBn_DoubleBuff;
extern PerformanceRecorder prspgemmdbuffcuda;
}  // namespace combblas