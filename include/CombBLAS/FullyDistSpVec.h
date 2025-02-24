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

#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include "CommGrid.h"
#include "FullyDist.h"
#include "Operations.h"
#include "OptBuf.h"
#include "SpParMat.h"
#include "promote.h"

namespace combblas
{
template <class IT, class NT, class DER>
class SpParMat;

template <class IT>
class DistEdgeList;

template <class IU, class NU>
class FullyDistVec;

template <class IU, class NU>
class SparseVectorLocalIterator;

/**
 * A sparse vector of length n (with nnz <= n of them being nonzeros) is distributed to
 * "all the processors" in a way that "respects ordering" of the nonzero indices
 * Example: x = [5,1,6,2,9] for nnz(x)=5 and length(x)=12
 *	we use 4 processors P_00, P_01, P_10, P_11
 * 	Then P_00 owns [1,2] (in the range [0,...,2]), P_01 ow`ns [5] (in the range [3,...,5]), and so on.
 * In the case of A(v,w) type sparse matrix indexing, this doesn't matter because n = nnz
 * 	After all, A(v,w) will have dimensions length(v) x length (w)
 * 	v and w will be of numerical type (NT) "int" and their indices (IT) will be consecutive integers
 * It is possibly that nonzero counts are distributed unevenly
 * Example: x=[1,2,3,4,5] and length(x) = 20, then P_00 would own all the nonzeros and the rest will hold empry vectors
 * Just like in SpParMat case, indices are local to processors (they belong to range [0,...,length-1] on each processor)
 * \warning Always create vectors with the right length, setting elements won't increase its length (similar to
 *operator[] on std::vector)
 **/
template <class IT, class NT>
class FullyDistSpVec : public FullyDist<IT, NT>
{
   public:
    FullyDistSpVec();
    FullyDistSpVec(IT glen);
    FullyDistSpVec(std::shared_ptr<CommGrid> grid);
    FullyDistSpVec(std::shared_ptr<CommGrid> grid, IT glen);
    FullyDistSpVec(const FullyDistVec<IT, NT> &rhs, std::function<bool(NT)> unop);
    FullyDistSpVec(const FullyDistVec<IT, NT> &rhs);  // Conversion copy-constructor
    FullyDistSpVec(IT globalsize, const FullyDistVec<IT, IT> &inds, const FullyDistVec<IT, NT> &vals,
                   bool SumDuplicates = false);
    FullyDistSpVec(std::shared_ptr<CommGrid> grid, IT globallen, const std::vector<IT> &indvec,
                   const std::vector<NT> &numvec, bool SumDuplicates = false, bool sorted = false);
    IT NnzUntil() const;
    FullyDistSpVec<IT, NT> Invert(IT globallen);
    FullyDistSpVec<IT, NT> Invert(IT globallen, std::function<IT(IT, IT)> BinOpIdx, std::function<NT(NT, NT)> BinOpVal,
                                  std::function<NT(NT, NT)> BinOpDup);
    FullyDistSpVec<IT, NT> InvertRMA(IT globallen, std::function<IT(IT, IT)> BinOpIdx,
                                     std::function<NT(NT, NT)> BinOpVal);
    template <typename NT1>
    void Select(const FullyDistVec<IT, NT1> &denseVec, std::function<bool(NT1)> unop);
    void FilterByVal(FullyDistSpVec<IT, IT> Selector, std::function<bool(NT)> UnaryOp, bool filterByIndex);
    template <typename NT1>
    void Setminus(const FullyDistSpVec<IT, NT1> &other);
    template <typename NT1>
    void SelectApply(const FullyDistVec<IT, NT1> &denseVec, std::function<bool(NT)> UnaryOp,
                     std::function<bool(NT, NT)> BinOp);

    //! like operator=, but instead of making a deep copy it just steals the contents.
    //! Useful for places where the "victim" will be distroyed immediately after the call.
    void stealFrom(FullyDistSpVec<IT, NT> &victim);
    FullyDistSpVec<IT, NT> &operator=(const FullyDistSpVec<IT, NT> &rhs);
    FullyDistSpVec<IT, NT> &operator=(const FullyDistVec<IT, NT> &rhs);  // convert from dense
    inline FullyDistSpVec<IT, NT> &operator=(NT fixedval)                // assign fixed value
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < ind.size(); ++i) num[i] = fixedval;
        return *this;
    }

    FullyDistSpVec<IT, NT> &operator+=(const FullyDistSpVec<IT, NT> &rhs);
    FullyDistSpVec<IT, NT> &operator-=(const FullyDistSpVec<IT, NT> &rhs);

    template <class HANDLER>
    void ParallelWrite(const std::string &filename, bool onebased, HANDLER handler, bool includeindices = true,
                       bool includeheader = false);

    void ParallelWrite(const std::string &filename, bool onebased, bool includeindices = true);

    void ParallelRead(const std::string &filename, bool onebased, std::function<bool(NT, NT)> BinOp);

    //! Totally obsolete version that only accepts an ifstream object and ascii files
    template <class HANDLER>
    std::ifstream &ReadDistribute(std::ifstream &infile, int master, HANDLER handler);

    std::ifstream &ReadDistribute(std::ifstream &infile, int master);

    template <class HANDLER>
    void SaveGathered(std::ofstream &outfile, int master, HANDLER handler, bool printProcSplits = false);

    void SaveGathered(std::ofstream &outfile, int master);

    inline bool operator==(const FullyDistSpVec<IT, NT> &rhs) const
    {
        FullyDistVec<IT, NT> v = *this;
        FullyDistVec<IT, NT> w = rhs;
        return (v == w);
    }

    void PrintInfo(std::string vecname) const;
    void iota(IT globalsize, NT first);
    void nziota(NT first);
    FullyDistVec<IT, NT> operator()(const FullyDistVec<IT, IT> &ri) const;  //!< SpRef (expects ri to be 0-based)
    void SetElement(IT indx, NT numx);                                      // element-wise assignment
    void DelElement(IT indx);                                               // element-wise deletion
    NT operator[](IT indx);
    inline bool WasFound() const { return wasFound; }

    //! sort the vector itself, return the permutation vector (0-based)
    FullyDistSpVec<IT, IT> sort();

    FullyDistSpVec<IT, NT> Uniq(std::function<NT(NT, NT)> BinOp = minimum<NT>(), MPI_Op mympiop = MPI_MIN);

    // Aydin TODO: parallelize with OpenMP
    inline FullyDistSpVec<IT, NT> Prune(std::function<bool(NT)> UnaryOp, bool inPlace = true)
    //<! Prune any nonzero entries for which the __unary_op evaluates to true (solely based on value)
    {
        FullyDistSpVec<IT, NT> temp(commGrid);
        IT spsize = ind.size();
        for (IT i = 0; i < spsize; ++i) {
            if (!(UnaryOp(num[i])))  // keep this nonzero
            {
                temp.ind.push_back(ind[i]);
                temp.num.push_back(num[i]);
            }
        }

        if (inPlace) {
            ind.swap(temp.ind);
            num.swap(temp.num);

            return FullyDistSpVec<IT, NT>(commGrid);  // return blank to match signature
        } else {
            return temp;
        }
    }

    inline IT getlocnnz() const { return ind.size(); }

    inline IT getnnz() const
    {
        IT totnnz = 0;
        IT locnnz = ind.size();
        MPI_Allreduce(&locnnz, &totnnz, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
        return totnnz;
    }

    using FullyDist<IT, NT>::LengthUntil;
    using FullyDist<IT, NT>::MyLocLength;
    using FullyDist<IT, NT>::MyRowLength;
    using FullyDist<IT, NT>::TotalLength;
    using FullyDist<IT, NT>::Owner;
    using FullyDist<IT, NT>::RowLenUntil;

    inline void setNumToInd()
    {
        IT offset = LengthUntil();
        IT spsize = ind.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (IT i = 0; i < spsize; ++i) num[i] = ind[i] + offset;
    }

    template <typename _Predicate>
    IT Count(_Predicate pred) const;  //!< Return the number of elements for which pred is true

    void Apply(std::function<NT(NT)> UnaryOp)
    {
        IT spsize = num.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (IT i = 0; i < spsize; ++i) num[i] = UnaryOp(num[i]);
    }

    inline void ApplyInd(std::function<NT(NT, NT)> BinOp)
    {
        IT offset = LengthUntil();
        IT spsize = ind.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (IT i = 0; i < spsize; ++i) num[i] = BinOp(num[i], ind[i] + offset);
    }

    template <typename BinaryOp>
    NT Reduce(BinaryOp __binary_op, NT init) const;

    template <typename OUT, typename BinaryOp, typename UnaryOp>
    OUT Reduce(BinaryOp __binary_op, OUT default_val, UnaryOp __unary_op) const;

    void DebugPrint();
    std::shared_ptr<CommGrid> getcommgrid() const { return commGrid; }

    void Reset();
    NT GetLocalElement(IT indx);
    void BulkSet(IT inds[], int count);

    inline std::vector<IT> GetLocalInd()
    {
        std::vector<IT> rind = ind;
        return rind;
    };

    inline std::vector<NT> GetLocalNum()
    {
        std::vector<NT> rnum = num;
        return rnum;
    };

    template <typename _Predicate>
    FullyDistVec<IT, IT> FindInds(_Predicate pred) const;
    template <typename _Predicate>
    FullyDistVec<IT, NT> FindVals(_Predicate pred) const;

   protected:
    using FullyDist<IT, NT>::glen;
    using FullyDist<IT, NT>::commGrid;

   private:
    std::vector<IT> ind;  // ind.size() give the number of nonzeros
    std::vector<NT> num;
    bool wasFound;  // true if the last GetElement operation returned an actual value

    void SparseCommon(std::vector<std::vector<std::pair<IT, NT> > > &data, std::function<NT(NT, NT)> BinOp);

    FullyDistSpVec<IT, NT> UniqAll2All(std::function<NT(NT, NT)> BinOp = minimum<NT>(), MPI_Op mympiop = MPI_MIN);

    template <class IU, class NU>
    friend class FullyDistSpVec;

    template <class IU, class NU>
    friend class FullyDistVec;

    template <class IU, class NU, class UDER>
    friend class SpParMat;

    template <class IU, class NU>
    friend class SparseVectorLocalIterator;

    // clang-format off
    template <typename SR, typename IU, typename NUM, typename NUV, typename UDER>
    friend FullyDistSpVec<IU, typename promote_trait<NUM, NUV>::T_promote>
    SpMV(const SpParMat<IU, NUM, UDER> &A,const FullyDistSpVec<IU, NUV> &x);

    template <typename SR, typename IU, typename NUM, typename UDER>
    friend FullyDistSpVec<IU, typename promote_trait<NUM, IU>::T_promote>
    SpMV(const SpParMat<IU, NUM, UDER> &A,const FullyDistSpVec<IU, IU> &x,bool indexisvalue);

    template <typename VT, typename IU, typename UDER>  // NoSR version (in BFSFriends.h)
    friend FullyDistSpVec<IU, VT>
    SpMV(const SpParMat<IU, bool, UDER> &A, const FullyDistSpVec<IU, VT> &x,OptBuf<int32_t, VT> &optbuf);

    template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
    friend void
    SpMV(const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IVT> &x, FullyDistSpVec<IU, OVT> &y,
        bool indexisvalue, OptBuf<int32_t, OVT> &optbuf);

    template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
    friend void
    SpMV(const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IVT> &x, FullyDistSpVec<IU, OVT> &y,
        bool indexisvalue, OptBuf<int32_t, OVT> &optbuf, PreAllocatedSPA<OVT> &SPA);

    template <typename IU, typename NU1, typename NU2>
    friend FullyDistSpVec<IU, typename promote_trait<NU1, NU2>::T_promote>
    EWiseMult(const FullyDistSpVec<IU, NU1> &V,const FullyDistVec<IU, NU2> &W,bool exclude, NU2 zero);

    template <typename RET, typename IU, typename NU1, typename NU2, typename BinaryOp, typename _BinaryPredicate>
    friend FullyDistSpVec<IU, RET>
    EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W,
        BinaryOp _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero,const bool useExtendedBinOp);

    template <typename RET, typename IU, typename NU1, typename NU2, typename BinaryOp, typename _BinaryPredicate>
    friend FullyDistSpVec<IU, RET>
    EWiseApply_threaded(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W,
        BinaryOp _binary_op, _BinaryPredicate _doOp, bool allowVNulls,
        NU1 Vzero, const bool useExtendedBinOp);

    template <typename RET, typename IU, typename NU1, typename NU2, typename BinaryOp, typename _BinaryPredicate>
    friend FullyDistSpVec<IU, RET>
    EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistSpVec<IU, NU2> &W,
        BinaryOp _binary_op, _BinaryPredicate _doOp, bool allowVNulls,
        bool allowWNulls, NU1 Vzero, NU2 Wzero, const bool allowIntersect,
        const bool useExtendedBinOp);

    template <typename IU>
    friend void RandPerm(FullyDistSpVec<IU, IU> &V);  // called on an existing object, randomly permutes it

    template <typename IU>
    friend void RenameVertices(DistEdgeList<IU> &DEL);

    //! Helper functions for sparse matrix X sparse vector
    // Ariful: I made this an internal function in ParFriends.h
    // template <typename SR, typename IU, typename OVT>
    // friend void MergeContributions(FullyDistSpVec<IU,OVT> & y, int * & recvcnt, int * & rdispls, int32_t * &
    // recvindbuf, OVT * & recvnumbuf, int rowneighs);

    template <typename IU, typename VT>
    friend void
    MergeContributions(FullyDistSpVec<IU, VT> &y, int *&recvcnt, int *&rdispls, int32_t *&recvindbuf,
                                   VT *&recvnumbuf, int rowneighs);

    template <typename IU, typename NV>
    friend void
    TransposeVector(MPI_Comm &World, const FullyDistSpVec<IU, NV> &x, int32_t &trxlocnz, IU &lenuntil,
                                int32_t *&trxinds, NV *&trxnums, bool indexisvalue);

    template <class IU, class NU, class DER, typename UnaryOp>
    friend SpParMat<IU, bool, DER>
    PermMat1(const FullyDistSpVec<IU, NU> &ri, const IU ncol, UnaryOp __unop);
    // clang-format on
};

}  // namespace combblas
