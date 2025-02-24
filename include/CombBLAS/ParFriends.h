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

#include <mpi.h>
#include <unistd.h>

#include <cstdarg>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "Friends.h"
#include "MPIType.h"
#include "MultiwayMerge.h"
#include "OptBuf.h"
#include "SpParHelper.h"
#include "SpParMat.h"
#include "SpParMat3D.h"
#include "mtSpGEMM.h"

// #include "cudaSpGEMM.cu"

namespace combblas
{
template <class IT, class NT, class DER>
class SpParMat;

/*************************************************************************************************/
/**************************** FRIEND FUNCTIONS FOR PARALLEL CLASSES ******************************/
/*************************************************************************************************/

/**
 ** Concatenate all the FullyDistVec<IT,NT> objects into a single one
 **/
template <typename IT, typename NT>
FullyDistVec<IT, NT> Concatenate(std::vector<FullyDistVec<IT, NT>> &vecs);

template <typename MATRIXA, typename MATRIXB>
bool CheckSpGEMMCompliance(const MATRIXA &A, const MATRIXB &B);

// Combined logic for prune, recovery, and select
template <typename IT, typename NT, typename DER>
void MCLPruneRecoverySelect(SpParMat<IT, NT, DER> &A, NT hardThreshold, IT selectNum, IT recoverNum, NT recoverPct,
                            int kselectVersion);

template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
IU EstimateFLOP(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B, bool clearA = false, bool clearB = false);

/**
 * Broadcasts A multiple times (#phases) in order to save storage in the output
 * Only uses 1/phases of C memory if the threshold/max limits are proper
 * Parameters:
 *  - computationKernel: 1 means hash-based, 2 means heap-based
 */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat<IU, NUO, UDERO> MemEfficientSpGEMM(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B, int phases,
                                            NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct,
                                            int kselectVersion, int computationKernel, int64_t perProcessMemory);

template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
int CalculateNumberOfPhases(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B, NUO hardThreshold, IU selectNum,
                            IU recoverNum, NUO recoverPct, int kselectVersion, int64_t perProcessMemory);

/*
 * A^2 with incremental MCL matrix
 * Non-zeroes are heavily skewed on the diagonals, hence SUMMA is suboptimal
 * We seprate diagonal elements from offdiagonals, M = D + A; D is diagonal and A is off-diagonal matrix;
 * D can be thought of sparse vector, but we use dense vector here to avoid technical difficult;
 * M^2 = (D+A)^2 = D^2 + A^2 + DxA + AxD
 * A^2: SUMMA (Verify whether SUMMA or 1D multiplication would be optimal?)
 * D^2: Elementwise squaring of vector
 * DxA: Vector dimapply along row of A
 * AxD: Vector dimapply along column of A
 * */
template <typename SR, typename ITA, typename NTA, typename DERA>
SpParMat<ITA, NTA, DERA> IncrementalMCLSquare(SpParMat<ITA, NTA, DERA> &A, int phases, NTA hardThreshold, ITA selectNum,
                                              ITA recoverNum, NTA recoverPct, int kselectVersion, int computationKernel,
                                              int64_t perProcessMemory);

/**
 * Parallel C = A*B routine that uses a double buffered broadcasting scheme
 * @pre { Input matrices, A and B, should not alias }
 * Most memory efficient version available. Total stages: 2*sqrt(p)
 * Memory requirement during first sqrt(p) stages: <= (3/2)*(nnz(A)+nnz(B))+(1/2)*nnz(C)
 * Memory requirement during second sqrt(p) stages: <= nnz(A)+nnz(B)+nnz(C)
 * Final memory requirement: nnz(C) if clearA and clearB are true
 **/
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_DoubleBuff(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B,
                                               bool clearA = false, bool clearB = false);

/**
 * Parallel A = B*C routine that uses only MPI-1 features
 * Relies on simple blocking broadcast
 * @pre { Input matrices, A and B, should not alias }
 **/
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_Synch(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B, bool clearA = false,
                                          bool clearB = false);

/*
 * Experimental SUMMA implementation with communication and computation overlap.
 * Not stable.
 * */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_Overlap(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B,
                                            bool clearA = false, bool clearB = false);

/**
 * Estimate the maximum nnz needed to store in a process from all stages of SUMMA before reduction
 * @pre { Input matrices, A and B, should not alias }
 **/
template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
int64_t EstPerProcessNnzSUMMA(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B, bool hashEstimate);

template <typename MATRIX, typename VECTOR>
void CheckSpMVCompliance(const MATRIX &A, const VECTOR &x);

template <typename SR, typename IU, typename NUM, typename UDER>
FullyDistSpVec<IU, typename promote_trait<NUM, IU>::T_promote> SpMV(
    const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IU> &x, bool indexisvalue,
    OptBuf<int32_t, typename promote_trait<NUM, IU>::T_promote> &optbuf);

template <typename SR, typename IU, typename NUM, typename UDER>
FullyDistSpVec<IU, typename promote_trait<NUM, IU>::T_promote> SpMV(const SpParMat<IU, NUM, UDER> &A,
                                                                    const FullyDistSpVec<IU, IU> &x, bool indexisvalue);

/**
 * Step 1 of the sparse SpMV algorithm
 * @param[in,out]   trxlocnz, lenuntil,trxinds,trxnums  { set or allocated }
 * @param[in] 	indexisvalue
 **/
template <typename IU, typename NV>
void TransposeVector(MPI_Comm &World, const FullyDistSpVec<IU, NV> &x, int32_t &trxlocnz, IU &lenuntil,
                     int32_t *&trxinds, NV *&trxnums, bool indexisvalue);

/**
 * Step 2 of the sparse SpMV algorithm
 * @param[in,out]   trxinds, trxnums { deallocated }
 * @param[in,out]   indacc, numacc { allocated }
 * @param[in,out]	accnz { set }
 * @param[in] 		trxlocnz, lenuntil, indexisvalue
 **/
template <typename IU, typename NV>
void AllGatherVector(MPI_Comm &ColWorld, int trxlocnz, IU lenuntil, int32_t *&trxinds, NV *&trxnums, int32_t *&indacc,
                     NV *&numacc, int &accnz, bool indexisvalue);

/**
 * Step 3 of the sparse SpMV algorithm, with the semiring
 * @param[in,out] optbuf {scratch space for all-to-all (fold) communication}
 * @param[in,out] indacc, numacc {index and values of the input vector, deleted upon exit}
 * @param[in,out] sendindbuf, sendnumbuf {index and values of the output vector, created}
 **/
template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void LocalSpMV(const SpParMat<IU, NUM, UDER> &A, int rowneighs, OptBuf<int32_t, OVT> &optbuf, int32_t *&indacc,
               IVT *&numacc, int32_t *&sendindbuf, OVT *&sendnumbuf, int *&sdispls, int *sendcnt, int accnz,
               bool indexisvalue, PreAllocatedSPA<OVT> &SPA);

// non threaded
template <typename SR, typename IU, typename OVT>
void MergeContributions(int *listSizes, std::vector<int32_t *> &indsvec, std::vector<OVT *> &numsvec,
                        std::vector<IU> &mergedind, std::vector<OVT> &mergednum);

template <typename SR, typename IU, typename OVT>
void MergeContributions_threaded(int *&listSizes, std::vector<int32_t *> &indsvec, std::vector<OVT *> &numsvec,
                                 std::vector<IU> &mergedind, std::vector<OVT> &mergednum, IU maxindex);

/**
 * This version is the most flexible sparse matrix X sparse vector [Used in KDT]
 * It accepts different types for the matrix (NUM), the input vector (IVT) and the output vector (OVT)
 * without relying on automatic type promotion
 * Input (x) and output (y) vectors can be ALIASED because y is not written until the algorithm is done with x.
 */
template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV(const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IVT> &x, FullyDistSpVec<IU, OVT> &y,
          bool indexisvalue, OptBuf<int32_t, OVT> &optbuf, PreAllocatedSPA<OVT> &SPA);

template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV(const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IVT> &x, FullyDistSpVec<IU, OVT> &y,
          bool indexisvalue, PreAllocatedSPA<OVT> &SPA);

template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV(const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IVT> &x, FullyDistSpVec<IU, OVT> &y,
          bool indexisvalue);

template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV(const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IVT> &x, FullyDistSpVec<IU, OVT> &y,
          bool indexisvalue, OptBuf<int32_t, OVT> &optbuf);

/**
 * Automatic type promotion is ONLY done here, all the callee functions (in Friends.h and below) are initialized with
 *the promoted type If indexisvalues = true, then we do not need to transfer values for x (happens for BFS iterations
 *with boolean matrices and integer rhs vectors)
 **/
template <typename SR, typename IU, typename NUM, typename UDER>
FullyDistSpVec<IU, typename promote_trait<NUM, IU>::T_promote> SpMV(
    const SpParMat<IU, NUM, UDER> &A, const FullyDistSpVec<IU, IU> &x, bool indexisvalue,
    OptBuf<int32_t, typename promote_trait<NUM, IU>::T_promote> &optbuf);

/**
 * Parallel dense SpMV
 **/
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER>
FullyDistVec<IU, typename promote_trait<NUM, NUV>::T_promote> SpMV(const SpParMat<IU, NUM, UDER> &A,
                                                                   const FullyDistVec<IU, NUV> &x);

/**
 * \TODO: Old version that is no longer considered optimal
 * Kept for legacy purposes
 * To be removed when other functionals are fully tested.
 **/
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER>
FullyDistSpVec<IU, typename promote_trait<NUM, NUV>::T_promote> SpMV(const SpParMat<IU, NUM, UDER> &A,
                                                                     const FullyDistSpVec<IU, NUV> &x);

// Aydin (June 2021):
// This currently duplicates the work of EWiseMult with exclude = true
// However, this is the right way of implementing it because it allows set difference when
// the types of two matrices do not have a valid multiplication operator defined
// set difference should not require such an operator so we will move all code
// bases that use EWiseMult(..., exclude=true) to this one
template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat<IU, NU1, UDERA> SetDifference(const SpParMat<IU, NU1, UDERA> &A, const SpParMat<IU, NU2, UDERB> &B);

template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat<IU, typename promote_trait<NU1, NU2>::T_promote, typename promote_trait<UDERA, UDERB>::T_promote> EWiseMult(
    const SpParMat<IU, NU1, UDERA> &A, const SpParMat<IU, NU2, UDERB> &B, bool exclude);

template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB,
          typename _BinaryOperation>
SpParMat<IU, RETT, RETDER> EWiseApply(const SpParMat<IU, NU1, UDERA> &A, const SpParMat<IU, NU2, UDERB> &B,
                                      _BinaryOperation __binary_op, bool notB, const NU2 &defaultBVal);

template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB,
          typename _BinaryOperation, typename _BinaryPredicate>
SpParMat<IU, RETT, RETDER> EWiseApply(const SpParMat<IU, NU1, UDERA> &A, const SpParMat<IU, NU2, UDERB> &B,
                                      _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls,
                                      bool allowBNulls, const NU1 &ANullVal, const NU2 &BNullVal,
                                      const bool allowIntersect, const bool useExtendedBinOp);
// // plain adapter
// template<typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB,
//     typename _BinaryOperation,
//     typename _BinaryPredicate>
// SpParMat<IU, RETT, RETDER> EWiseApply(const SpParMat<IU, NU1, UDERA> &A, const SpParMat<IU, NU2, UDERB> &B,
//                                       _BinaryOperation __binary_op,
//                                       _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1
//                                       &ANullVal, const NU2 &BNullVal, const bool allowIntersect = true) {
//     return EWiseApply<RETT, RETDER>(A, B, EWiseExtToPlainAdapter<RETT, NU1, NU2, _BinaryOperation>(__binary_op),
//                                     EWiseExtToPlainAdapter<bool, NU1, NU2, _BinaryPredicate>(do_op), allowANulls,
//                                     allowBNulls, ANullVal, BNullVal,
//                                     allowIntersect, true);
// }

// end adapter

/**
 * if exclude is true, then we prune all entries W[i] != zero from V
 * if exclude is false, then we perform a proper elementwise multiplication
 **/
template <typename IU, typename NU1, typename NU2>
FullyDistSpVec<IU, typename promote_trait<NU1, NU2>::T_promote> EWiseMult(const FullyDistSpVec<IU, NU1> &V,
                                                                          const FullyDistVec<IU, NU2> &W, bool exclude,
                                                                          NU2 zero);

/**
 Threaded EWiseApply. Only called internally from EWiseApply.
**/
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU, RET> EWiseApply_threaded(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W,
                                            _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls,
                                            NU1 Vzero, const bool useExtendedBinOp);

/**
 * Performs an arbitrary binary operation _binary_op on the corresponding elements of two vectors with the result stored
 *in a return vector ret. The binary operatiation is only performed if the binary predicate _doOp returns true for those
 *elements. Otherwise the binary operation is not performed and ret does not contain an element at that position. More
 *formally the operation is defined as: if (_doOp(V[i], W[i])) ret[i] = _binary_op(V[i], W[i]) else
 *    // ret[i] is not set
 * Hence _doOp can be used to implement a filter on either of the vectors.
 *
 * The above is only defined if both V[i] and W[i] exist (i.e. an intersection). To allow a union operation (ex. when
 *V[i] doesn't exist but W[i] does) the allowVNulls flag is set to true and the Vzero argument is used as the missing
 *V[i] value.
 *
 * The type of each element of ret must not necessarily be related to the types of V or W, so the return type must be
 *explicitly specified as a template parameter: FullyDistSpVec<int, double> r = EWiseApply<double>(V, W, plus, retTrue,
 *false, 0)
 **/
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU, RET> EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W,
                                   _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero,
                                   const bool useExtendedBinOp);

/**
 * Performs an arbitrary binary operation _binary_op on the corresponding elements of two vectors with the result stored
 *in a return vector ret. The binary operatiation is only performed if the binary predicate _doOp returns true for those
 *elements. Otherwise the binary operation is not performed and ret does not contain an element at that position. More
 *formally the operation is defined as: if (_doOp(V[i], W[i])) ret[i] = _binary_op(V[i], W[i]) else
 *    // ret[i] is not set
 * Hence _doOp can be used to implement a filter on either of the vectors.
 *
 * The above is only defined if both V[i] and W[i] exist (i.e. an intersection). To allow a union operation (ex. when
 *V[i] doesn't exist but W[i] does) the allowVNulls flag is set to true and the Vzero argument is used as the missing
 *V[i] value. !allowVNulls && !allowWNulls => intersection !allowVNulls &&  allowWNulls => operate on all elements of V
 *  allowVNulls && !allowWNulls => operate on all elements of W
 *  allowVNulls &&  allowWNulls => union
 *
 * The type of each element of ret must not necessarily be related to the types of V or W, so the return type must be
 *explicitly specified as a template parameter: FullyDistSpVec<int, double> r = EWiseApply<double>(V, W, plus, ...) For
 *intersection, Vzero and Wzero are irrelevant ABAB: \todo: Should allowIntersect be "false" for all SetDifference uses?
 **/
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU, RET> EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistSpVec<IU, NU2> &W,
                                   _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls,
                                   bool allowWNulls, NU1 Vzero, NU2 Wzero, const bool allowIntersect,
                                   const bool useExtendedBinOp);

//
// // plain callback versions
// template<typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename
//     _BinaryPredicate>
// FullyDistSpVec<IU, RET> EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistVec<IU, NU2> &W,
//                                    _BinaryOperation _binary_op,
//                                    _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero) {
//     return EWiseApply<RET>(V, W, EWiseExtToPlainAdapter<RET, NU1, NU2, _BinaryOperation>(_binary_op),
//                            EWiseExtToPlainAdapter<bool, NU1, NU2, _BinaryPredicate>(_doOp), allowVNulls, Vzero,
//                            true);
// }
//
// template<typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename
//     _BinaryPredicate>
// FullyDistSpVec<IU, RET> EWiseApply(const FullyDistSpVec<IU, NU1> &V, const FullyDistSpVec<IU, NU2> &W,
//                                    _BinaryOperation _binary_op,
//                                    _BinaryPredicate _doOp, bool allowVNulls, bool allowWNulls, NU1 Vzero, NU2 Wzero,
//                                    const bool allowIntersect = true) {
//     return EWiseApply<RET>(V, W, EWiseExtToPlainAdapter<RET, NU1, NU2, _BinaryOperation>(_binary_op),
//                            EWiseExtToPlainAdapter<bool, NU1, NU2, _BinaryPredicate>(_doOp), allowVNulls,
//                            allowWNulls,
//                            Vzero, Wzero, allowIntersect,
//                            true);
// }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// sampling-based nnz estimation via SpMV
// @OGUZ-NOTE This is not based on SUMMA, do not use. Estimates the number of
// nonzeros in the final output matrix.

#define NROUNDS 5
typedef std::array<float, NROUNDS> samparr_t;

template <typename NZT>
struct promote_trait<NZT, samparr_t> {
    typedef samparr_t T_promote;
};

class SamplesSaveHandler
{
   public:
    template <typename c, typename t, typename V>
    void save(std::basic_ostream<c, t> &os, std::array<V, NROUNDS> &sample_vec, int64_t index)
    {
        for (auto it = sample_vec.begin(); it != sample_vec.end(); ++it) os << *it << " ";
    }
};

template <typename NZT>
struct SelectMinxSR {
    static samparr_t id()
    {
        samparr_t arr;
        for (auto it = arr.begin(); it != arr.end(); ++it) *it = std::numeric_limits<float>::max();
        return arr;
    }

    static bool returnedSAID() { return false; }

    static samparr_t add(const samparr_t &arg1, const samparr_t &arg2)
    {
        samparr_t out;
        for (int i = 0; i < NROUNDS; ++i) out[i] = std::min(arg1[i], arg2[i]);
        return out;
    }

    static samparr_t multiply(const NZT arg1, const samparr_t &arg2) { return arg2; }

    static void axpy(const NZT a, const samparr_t &x, samparr_t &y) { y = add(y, multiply(a, x)); }

    static MPI_Op mpi_op()
    {
        static MPI_Op mpiop;
        static bool exists = false;
        if (exists)
            return mpiop;
        else {
            MPI_Op_create(MPI_func, true, &mpiop);
            exists = true;
            return mpiop;
        }
    }

    static void MPI_func(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype)
    {
        samparr_t *in = static_cast<samparr_t *>(invec);
        samparr_t *inout = static_cast<samparr_t *>(inoutvec);
        for (int i = 0; i < *len; ++i) inout[i] = add(inout[i], in[i]);
    }
};

template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
int64_t EstPerProcessNnzSpMV(SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B);

template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1,
          typename UDER2>
SpParMat3D<IU, NUO, UDERO> Mult_AnXBn_SUMMA3D(SpParMat3D<IU, NU1, UDER1> &A, SpParMat3D<IU, NU2, UDER2> &B);

/*
 * Parameters:
 *  - computationKernel: 1 for hash-based, 2 for heap-based
 * */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA,
          typename UDERB>
SpParMat3D<IU, NUO, UDERO> MemEfficientSpGEMM3D(SpParMat3D<IU, NU1, UDERA> &A, SpParMat3D<IU, NU2, UDERB> &B,
                                                int phases, NUO hardThreshold, IU selectNum, IU recoverNum,
                                                NUO recoverPct, int kselectVersion, int computationKernel,
                                                int64_t perProcessMemory);

/**
 * Prune all nz of A keeping only those having -
 * either (a) value of isOld corresponding to the two endpoints of the nz is different ( Edge between old and new
 *vertices) or (b) value is higher than the threshold (edge between old-old or new-new but edge weight is high) or (c)
 *nz exists in M (edge exists in the Mask matrix)
 **/
// template <typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB,
// typename FLAGTYPE> SpParMat<IU, NUO, UDERO> SelectivePrune (SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & M,
// FullyDistVec<IU,FLAGTYPE>& isOld, double threshold, bool inPlace ) {
/*
 * Parameter compatibility check
 * */
// if((A.getncol() != M.getncol()) && (A.getnrow() != M.getnrow()) && (A.getnrow() != isOld.TotalLength()) ){
// std::ostringstream outs;
// outs << "Can not perform selective prune, dimensions does not match"<< std::endl;
// SpParHelper::Print(outs.str());
// MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
//}

// int nprocs, myrank, nthreads = 1;
// MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
// MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
// #ifdef THREADED
// #pragma omp parallel
//{
// nthreads = omp_get_num_threads();
//}
// #endif

/*
 * Gather isOld vector along grid row and column
 * */
//// MTH: Potential bug. count can technically be larger than int ( depending on IU type)
//// But later MPI call will need it as int
// int rankInRow = isOld.getcommgrid()->GetRankInProcRow();
// int rankInCol = isOld.getcommgrid()->GetRankInProcCol();
// int nGridRow = isOld.getcommgrid()->GetGridRow();
// int sendcnt = int(isOld.MyLocLength()); // Have to be int because the same number will be used as count in Allgatherv
// call std::vector<int> recvcnt(isOld.getcommgrid()->GetGridRow(), 0); // Can be reused later because of square process
// grid std::vector<int> rdispls(isOld.getcommgrid()->GetGridRow()+1, 0); // Can be reused later because of square
// process grid

// MPI_Allgather(&sendcnt, 1, MPI_INT, recvcnt.data(), 1, MPI_INT, isOld.getcommgrid()->GetRowWorld());
// rdispls[0] = 0; // First element 0 for prefix sum
// std::partial_sum(recvcnt.begin(), recvcnt.end(), rdispls.begin() + 1); // Prefix sum
// std::vector<FLAGTYPE> isOldRow(rdispls[rankInRow]);
// MPI_Allgatherv(isOld.GetLocArr(), sendcnt, MPI_INT, isOldRow.data(), recvcnt.data(), rdispls.data(), MPI_INT,
// isOld.getcommgrid()->GetRowWorld());

//// Do same thing along column
// MPI_Allgather(&sendcnt, 1, MPI_INT, recvcnt.data(), 1, MPI_INT, isOld.getcommgrid()->GetColWorld());
// rdispls[0] = 0; // First element 0 for prefix sum
// std::partial_sum(recvcnt.begin(), recvcnt.end(), rdispls.begin() + 1); // Prefix sum
// std::vector<FLAGTYPE> isOldCol(rdispls[rankInCol]);
// MPI_Allgatherv(isOld.GetLocArr(), sendcnt, MPI_INT, isOldCol.data(), recvcnt.data(), rdispls.data(), MPI_INT,
// isOld.getcommgrid()->GetColWorld());

// if(inPlace){
// A.seqptr()->SelectivePrune(M.seqptr(), isOldRow, isOldCol, threshold, inPlace);
// return SpParMat<IU, NUO, UDERO>(MPI_COMM_WORLD); // Return empty SpParMat to match function signature
//}
// else{
// return SpParMat<IU, NUO, UDERO>(A.seqptr()->SelectivePrune(M.seqptr(), isOldRow, isOldCol, threshold, inPlace),
// A.getcommgrid());
//}
// return PrunedA;
//}
}  // namespace combblas
