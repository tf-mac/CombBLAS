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

#include "ForwardDecl.h"
#include "PreAllocatedSPA.h"
#include "promote.h"

namespace combblas
{

/*************************************************************************************************/
/**************************** SHARED ADDRESS SPACE FRIEND FUNCTIONS ******************************/
/****************************** MULTITHREADED LOGIC ALSO GOES HERE *******************************/
/*************************************************************************************************/

//! SpMV with dense vector
template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
void dcsc_gespmv(const SpDCCols<IU, NU> &A, const RHS *x, LHS *y);

//! SpMV with dense vector (multithreaded version)
template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
void dcsc_gespmv_threaded_nosplit(const SpDCCols<IU, NU> &A, const RHS *x, LHS *y);
/**
 * Multithreaded SpMV with dense vector
 */
template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
void dcsc_gespmv_threaded(const SpDCCols<IU, NU> &A, const RHS *x, LHS *y);

/**
 * Multithreaded SpMV with sparse vector
 * the assembly of outgoing buffers sendindbuf/sendnumbuf are done here
 */
template <typename SR, typename IU, typename NUM, typename DER, typename IVT, typename OVT>
int generic_gespmv_threaded(const SpMat<IU, NUM, DER> &A, const int32_t *indx, const IVT *numx, int32_t nnzx,
                            int32_t *&sendindbuf, OVT *&sendnumbuf, int *&sdispls, int p_c, PreAllocatedSPA<OVT> &SPA);

/**
 * Multithreaded SpMV with sparse vector and preset buffers
 * the assembly of outgoing buffers sendindbuf/sendnumbuf are done here
 * IVT: input vector numerical type
 * OVT: output vector numerical type
 */
template <typename SR, typename IU, typename NUM, typename DER, typename IVT, typename OVT>
void generic_gespmv_threaded_setbuffers(const SpMat<IU, NUM, DER> &A, const int32_t *indx, const IVT *numx,
                                        int32_t nnzx, int32_t *sendindbuf, OVT *sendnumbuf, int *cnts, int *dspls,
                                        int p_c);

//! SpMV with sparse vector
//! MIND: Matrix index type
//! VIND: Vector index type (optimized: int32_t, general: int64_t)
template <typename SR, typename MIND, typename VIND, typename DER, typename NUM, typename IVT, typename OVT>
void generic_gespmv(const SpMat<MIND, NUM, DER> &A, const VIND *indx, const IVT *numx, VIND nnzx,
                    std::vector<VIND> &indy, std::vector<OVT> &numy, PreAllocatedSPA<OVT> &SPA);

/** SpMV with sparse vector
 * @param[in] indexisvalue is only used for BFS-like computations, if true then we can call the optimized version that
 * skips SPA
 */
template <typename SR, typename IU, typename DER, typename NUM, typename IVT, typename OVT>
void generic_gespmv(const SpMat<IU, NUM, DER> &A, const int32_t *indx, const IVT *numx, int32_t nnzx, int32_t *indy,
                    OVT *numy, int *cnts, int *dspls, int p_c, bool indexisvalue);

// NU must be bool
template <typename IU, typename NU>
void BooleanRowSplit(SpDCCols<IU, NU> &A, int numsplits);

/**
 * SpTuples(A*B') (Using OuterProduct Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template <class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> *Tuples_AnXBt(const SpDCCols<IU, NU1> &A, const SpDCCols<IU, NU2> &B, bool clearA = false,
                                bool clearB = false);

/**
 * SpTuples(A*B) (Using ColByCol Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template <class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> *Tuples_AnXBn(const SpDCCols<IU, NU1> &A, const SpDCCols<IU, NU2> &B, bool clearA = false,
                                bool clearB = false);

template <class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> *Tuples_AtXBt(const SpDCCols<IU, NU1> &A, const SpDCCols<IU, NU2> &B, bool clearA = false,
                                bool clearB = false);

template <class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> *Tuples_AtXBn(const SpDCCols<IU, NU1> &A, const SpDCCols<IU, NU2> &B, bool clearA = false,
                                bool clearB = false);

// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template <class SR, class IU, class NU>
SpTuples<IU, NU> MergeAll(const std::vector<SpTuples<IU, NU> *> &ArrSpTups, IU mstar = 0, IU nstar = 0,
                          bool delarrs = false);

/**
 *  operation is A = A .* not(B)
 **/
template <typename IU, typename NU>
Dcsc<IU, NU> SetDifference(const Dcsc<IU, NU> &A, const Dcsc<IU, NU> *B);

template <typename IU, typename NU>
SpDCCols<IU, NU> EWiseMult(const SpDCCols<IU, NU> &A, const SpDCCols<IU, NU> &B, bool exclude);

/**
 * @param[in]   exclude if false,
 *      \n              then operation is A = A .* B
 *      \n              else operation is A = A .* not(B)
 *
 * Aydin (June 2021):  exclude=true case of this function now calls SetDifference above, to remove code duplication
 **/
template <typename IU, typename NU>
Dcsc<IU, NU> EWiseMult(const Dcsc<IU, NU> &A, const Dcsc<IU, NU> *B, bool exclude);

template <typename IU, typename NU>
Dcsc<IU, NU> EWiseApply(const Dcsc<IU, NU> &A, const Dcsc<IU, NU> *B, std::function<NU(NU, NU)> BinOp, bool notB,
                        const NU &defaultBVal);

template <typename IU, typename NU>
SpDCCols<IU, NU> EWiseApply(const SpDCCols<IU, NU> &A, const SpDCCols<IU, NU> &B, std::function<bool(NU, NU)> BinOp,
                            bool notB, const NU &defaultBVal);

/**
 * Implementation based on operator +=
 * Element wise apply with the following constraints
 * The operation to be performed is __binary_op
 * The operation `c = __binary_op(a, b)` is only performed if `do_op(a, b)` returns true
 * If allowANulls is true, then if A is missing an element that B has, then ANullVal is used
 * In that case the operation becomes c[i,j] = __binary_op(ANullVal, b[i,j])
 * If both allowANulls and allowBNulls is false then the function degenerates into intersection
 */
template <typename RETT, typename IU, typename NU>
Dcsc<IU, RETT> EWiseApply(const Dcsc<IU, NU> *Ap, const Dcsc<IU, NU> *Bp, std::function<NU(NU, NU)> BinOp,
                          std::function<bool(NU, NU)> do_op, bool allowANulls, bool allowBNulls, const NU &ANullVal,
                          const NU &BNullVal, bool allowIntersect);

template <typename RETT, typename IU, typename NU>
SpDCCols<IU, RETT> EWiseApply(const SpDCCols<IU, NU> &A, const SpDCCols<IU, NU> &B, std::function<NU(NU, NU)> BinOp,
                              std::function<bool(NU, NU)> UnaryOp, bool allowANulls, bool allowBNulls,
                              const NU &ANullVal, const NU &BNullVal, bool allowIntersect);
}  // namespace combblas
