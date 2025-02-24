#pragma once

#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

#include "StackEntry.h"
#if USE_CUDA
#include "SpCRows.h"
#include "SpCuCRows.h"
#endif

namespace combblas
{

template <class IT, class NT, class DER>
class SpMat;
template <class IT, class NT>
class SpDCCols;
template <class IT, class NT>
class SpCCols;
template <class IT, class NT>
class Dcsc;

template <class IT, class NT>
class SpTuples : public SpMat<IT, NT, SpTuples<IT, NT> >
{
   public:
    std::tuple<IT, IT, NT> *tuples;
    bool tuples_deleted = false;  // This is a temporary patch to avoid memory leak in 3d-memory multiplication

    // Constructors
    SpTuples(int64_t size, IT nRow, IT nCol);
    SpTuples(int64_t size, IT nRow, IT nCol, std::tuple<IT, IT, NT> *mytuples, bool sorted = false,
             bool isOpNew = false);
    SpTuples(int64_t maxnnz, IT nRow, IT nCol, std::vector<IT> &edges, bool removeloops = true);
    SpTuples(int64_t size, IT nRow, IT nCol, StackEntry<NT, std::pair<IT, IT> > *&multstack);
    SpTuples(const SpTuples<IT, NT> &rhs);
    SpTuples(const SpDCCols<IT, NT> &rhs);
    SpTuples(const SpCCols<IT, NT> &rhs);

#ifdef USE_CUDA
    SpTuples(const SpCuCRows<IT, NT> &rhs);
#endif

    ~SpTuples();

    SpTuples<IT, NT> &operator=(const SpTuples<IT, NT> &rhs);

    IT &rowindex(IT i) { return std::get<0>(tuples[i]); }
    IT &colindex(IT i) { return std::get<1>(tuples[i]); }
    NT &numvalue(IT i) { return std::get<2>(tuples[i]); }

    IT rowindex(IT i) const { return std::get<0>(tuples[i]); }
    IT colindex(IT i) const { return std::get<1>(tuples[i]); }
    NT numvalue(IT i) const { return std::get<2>(tuples[i]); }

    void RemoveDuplicates(std::function<NT(NT, NT)> BinOp);

    void SortRowBased();
    void SortColBased();

    IT AddLoops(NT loopval, bool replaceExisting = false);
    IT AddLoops(std::vector<NT> loopvals, bool replaceExisting = false);
    IT RemoveLoops();

    std::pair<IT, IT> RowLimits();
    std::pair<IT, IT> ColLimits();

    std::tuple<IT, IT, NT> front() { return tuples[0]; };
    std::tuple<IT, IT, NT> back() { return tuples[nnz - 1]; };

    // std::ofstream &put(std::ofstream &outfile) const;
    // std::ifstream &get(std::ifstream &infile);

    [[nodiscard]] bool isZero() const { return (nnz == 0); }
    inline IT getnrow() const { return m; }
    inline IT getncol() const { return n; }
    [[nodiscard]] inline int64_t getnnz() const { return nnz; }

    void PrintInfo();
    // Performs a balanced merge of the array of SpTuples
    template <typename SR, typename IU, typename NU>
    friend SpTuples<IU, NU> MergeAll(const std::vector<SpTuples<IU, NU> *> &ArrSpTups, IU mstar, IU nstar,
                                     bool delarrs);

    template <typename SR, typename IU, typename NU>
    friend SpTuples<IU, NU> *MergeAllRec(const std::vector<SpTuples<IU, NU> *> &ArrSpTups, IU mstar, IU nstar);

   private:
    IT m, n;
    int64_t nnz;
    bool isOperatorNew;

    void FillTuples(Dcsc<IT, NT> *mydcsc);

    template <typename IU, typename NU>
    friend class SpDCCols;

    template <typename IU, typename NU>
    friend class SpCCols;
};
}  // namespace combblas
