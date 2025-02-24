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

#include <fstream>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

#include "CommGrid.h"
#include "ForwardDecl.h"
#include "FullyDist.h"
#include "MPIOp.h"
#include "promote.h"

namespace combblas
{

// ABAB: As opposed to SpParMat, IT here is used to encode global size and global indices;
// therefore it can not be 32-bits, in general.
template <class IT, class NT>
class FullyDistVec : public FullyDist<IT, NT>
{
   public:
    //////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    //////////////////////////////////////////////////////////////////////////////////////////
    FullyDistVec();
    FullyDistVec(IT globallen, NT initval);
    FullyDistVec(std::shared_ptr<CommGrid> grid);
    FullyDistVec(std::shared_ptr<CommGrid> grid, IT globallen, NT initval);
    FullyDistVec(const FullyDistSpVec<IT, NT> &rhs);  // Sparse -> Dense conversion constructor
    FullyDistVec(const std::vector<NT> &fillarr, std::shared_ptr<CommGrid> grid);
    // initialize a FullyDistVec with a vector of length n/p from each processor

    //!< type converter constructor
    template <class ITRHS, class NTRHS>
    explicit FullyDistVec(const FullyDistVec<ITRHS, NTRHS> &rhs);

    template <class HANDLER>
    void ParallelWrite(const std::string &filename, bool onebased, HANDLER handler, bool includeindices = true);
    void ParallelWrite(const std::string &filename, bool onebased, bool includeindices = true);
    void ParallelRead(const std::string &filename, bool onebased, std::function<bool(NT, NT)> BinOp);

    template <class HANDLER>
    void SaveGathered(std::ofstream &outfile, int master, HANDLER handler, bool printProcSplits = false);
    void SaveGathered(std::ofstream &outfile, int master);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Operator
    //////////////////////////////////////////////////////////////////////////////////////////

    //!< basic assignment operator
    FullyDistVec<IT, NT> &operator=(const FullyDistVec<IT, NT> &rhs);
    //!< assignment operator with type conversion
    template <class ITRHS, class NTRHS>
    FullyDistVec<IT, NT> &operator=(const FullyDistVec<ITRHS, NTRHS> &rhs);
    //!< assignment operator from FullyDistSpVec
    FullyDistVec<IT, NT> &operator=(const FullyDistSpVec<IT, NT> &rhs);
    //!< assignment operator with fixed numeric value
    FullyDistVec<IT, NT> &operator=(NT fixedval);

    FullyDistVec<IT, NT> operator()(const FullyDistVec<IT, IT> &ri) const;  //<! subsref
    //!< assignment operator with fixed numeric value
    FullyDistVec<IT, NT> &operator+=(const FullyDistSpVec<IT, NT> &rhs);
    FullyDistVec<IT, NT> &operator+=(const FullyDistVec<IT, NT> &rhs);
    FullyDistVec<IT, NT> &operator-=(const FullyDistSpVec<IT, NT> &rhs);
    FullyDistVec<IT, NT> &operator-=(const FullyDistVec<IT, NT> &rhs);
    //!< equality operator
    bool operator==(const FullyDistVec<IT, NT> &rhs) const;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Getter, Setter
    //////////////////////////////////////////////////////////////////////////////////////////

    void SetElement(IT indx, NT numx);                                        // element-wise assignment
    inline void SetLocalElement(IT index, NT value) { arr[index] = value; };  // no checks, local index
    NT GetElement(IT indx) const;                                             // element-wise fetch
    inline NT GetLocalElement(IT index) const { return arr[index]; }

    // template <class NT1, typename BinaryOpIdx, typename BinaryOpVal>
    // void GSet(const FullyDistSpVec<IT, NT1> &spVec, BinaryOpIdx BinOpIdx, BinaryOpVal BinOpVal, MPI_Win win);
    // template <class NT1, typename BinaryOpIdx>
    // FullyDistSpVec<IT, NT> GGet(const FullyDistSpVec<IT, NT1> &spVec, BinaryOpIdx BinOpIdx, NT nullValue);

    void iota(IT globalsize, NT first);
    void RandPerm(uint64_t seed = 1383098845);           // randomly permute the vector
    FullyDistVec<IT, IT> sort();                         // sort and return the permutation
    inline IT LocArrSize() const { return arr.size(); }  // = MyLocLength() once arr is resized
    // TODO: we should change this function and return the vector directly
    inline const NT *GetLocArr() const { return arr.data(); }  // = MyLocLength() once arr is resized
    inline const std::vector<NT> &GetLocVec() const { return arr; }

    //!< Return the elements for which pred is true
    FullyDistSpVec<IT, NT> Find(std::function<bool(NT)> pred) const;
    //!< Return the elements val is found
    FullyDistSpVec<IT, NT> Find(NT val) const;
    //!< Return the indices where pred is true
    FullyDistVec<IT, IT> FindInds(std::function<bool(NT)> pred) const;
    //!< Return the number of elements for which pred is true
    IT Count(std::function<bool(NT)> pred) const;

    void Apply(std::function<NT(NT)> UnaryOp);

    template <typename IRRELEVANT_NT>
    void Apply(std::function<NT(NT)> UnaryOp, const FullyDistSpVec<IT, IRRELEVANT_NT> &mask);

    void ApplyInd(std::function<NT(NT, NT)> BinOp);

    // extended callback versions
    template <typename BinaryOp, typename _BinaryPredicate, class NT2>
    void EWiseApply(const FullyDistVec<IT, NT2> &other, BinaryOp __binary_op, _BinaryPredicate _do_op,
                    const bool useExtendedBinOp);
    template <typename BinaryOp, typename _BinaryPredicate, class NT2>
    void EWiseApply(const FullyDistSpVec<IT, NT2> &other, BinaryOp __binary_op, _BinaryPredicate _do_op,
                    bool applyNulls, NT2 nullValue, const bool useExtendedBinOp);

    template <class NT2>
    void EWiseApply(const FullyDistVec<IT, NT2> &other, std::function<bool(NT)> __binary_op)
    {
        auto rettrue = [](auto a, auto b) { return true; };
        this->EWiseApply(other, __binary_op, rettrue);
    }

    template <typename BinaryOp, class NT2>
    void EWiseApply(const FullyDistSpVec<IT, NT2> &other, BinaryOp __binary_op, bool applyNulls, NT2 nullValue)
    {
        auto rettrue = [](auto a, auto b) { return true; };
        this->EWiseApply(other, __binary_op, rettrue, applyNulls, nullValue);
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // Print and Debug
    //////////////////////////////////////////////////////////////////////////////////////////
    void PrintToFile(std::string prefix);
    void PrintInfo(std::string vectorname) const;
    void DebugPrint();
    std::shared_ptr<CommGrid> getcommgrid() const { return commGrid; }

    std::pair<IT, NT> MinElement() const;  // returns <index, value> pair of global minimum

    //! Reduce can be used to implement max_element, for instance
    template <MPIReduceType mpitype>
    NT Reduce(std::function<NT(NT, NT)> BinOp, NT identity) const;
    template <MPIReduceType mpitype>
    NT Reduce(std::function<NT(NT, NT)> Binop, NT outval, std::function<NT(NT)> UnaryOp) const;

    void SelectCandidates(double nver);

    // template <typename BinaryOp, typename OUT = typename std::result_of<BinaryOp &(NT, NT)>::type>
    // void EWiseOut(const FullyDistVec<IT, NT> &rhs, BinaryOp __binary_op, FullyDistVec<IT, OUT> &result);

    using FullyDist<IT, NT>::LengthUntil;
    using FullyDist<IT, NT>::TotalLength;
    using FullyDist<IT, NT>::Owner;
    using FullyDist<IT, NT>::MyLocLength;
    using FullyDist<IT, NT>::glen;
    using FullyDist<IT, NT>::commGrid;

   private:
    std::vector<NT> arr;
    void EWise(const FullyDistVec<IT, NT> &rhs, std::function<NT(NT, NT)> BinOp);

    //////////////////////////////////////////////////////////////////////////////////////////
    // Friends
    //////////////////////////////////////////////////////////////////////////////////////////
    template <class IU, class NU>
    friend class DenseParMat;

    template <class IU, class NU, class UDER>
    friend class SpParMat;

    template <class IU, class NU>
    friend class FullyDistVec;

    template <class IU, class NU>
    friend class FullyDistSpVec;

    template <class IU, class NU>
    friend class DenseVectorLocalIterator;

    // clang-format off
    template <typename SR, typename IU, typename NUM, typename NUV, typename UDER>
    friend FullyDistVec<IU, typename promote_trait<NUM, NUV>::T_promote>
    SpMV(const SpParMat<IU, NUM, UDER> &A, const FullyDistVec<IU, NUV> &x);

    template <typename IU, typename NU1, typename NU2>
    friend FullyDistSpVec<IU, typename promote_trait<NU1, NU2>::T_promote>
    EWiseMult(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W, bool exclude, NU2 zero);

    template <typename IU, typename NU1, typename NU2, typename BinaryOp>
    friend FullyDistSpVec<IU, typename promote_trait<NU1, NU2>::T_promote>
    EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W, BinaryOp _binary_op,
        typename promote_trait<NU1, NU2>::T_promote zero);

    template <typename RET, typename IU, typename NU1, typename NU2, typename BinaryOp, typename _BinaryPredicate>
    friend FullyDistSpVec<IU, RET>
    EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W,
        BinaryOp _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero,
        const bool useExtendedBinOp);

    template <typename RET, typename IU, typename NU1, typename NU2, typename BinaryOp, typename _BinaryPredicate>
    friend FullyDistSpVec<IU, RET>
    EWiseApply_threaded(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W,
        BinaryOp _binary_op, _BinaryPredicate _doOp, bool allowVNulls,
        NU1 Vzero, const bool useExtendedBinOp);

    template <typename IU>
    friend void
    RenameVertices(DistEdgeList<IU> &DEL);

    template <typename IU, typename NU>
    friend FullyDistVec<IU, NU>
    Concatenate(std::vector<FullyDistVec<IU, NU> > &vecs);

    template <typename IU, typename NU>
    friend void
    Augment(FullyDistVec<int64_t, int64_t> &mateRow2Col, FullyDistVec<int64_t, int64_t> &mateCol2Row,
        FullyDistVec<int64_t, int64_t> &parentsRow, FullyDistVec<int64_t, int64_t> &leaves);

    template <class IU, class DER>
    friend SpParMat<IU, bool, DER>
    PermMat(const FullyDistVec<IU, IU> &ri, const IU ncol);

    friend void
    maximumMatching(SpParMat<int64_t, bool, SpDCCols<int64_t, bool> > &A,
        FullyDistVec<int64_t, int64_t> &mateRow2Col,
        FullyDistVec<int64_t, int64_t> &mateCol2Row);
    // clang-format on
};
}  // namespace combblas
