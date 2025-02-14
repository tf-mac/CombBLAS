/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 04/29/2018 --------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, John Gilbert------------*/
/****************************************************************/
/*

Combinatorial BLAS, Copyright (c) 2018, The Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy) and University of California,
Santa Barbara.  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation &
Partnerships Office at  IPO@lbl.gov.


NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government
consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its
behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the
public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.
 */

#ifndef COMBBLAS_H
#define COMBBLAS_H

// #ifdef _MSC_VER
// #pragma warning( disable : 4244 ) // conversion from 'int64_t' to 'double', possible loss of data
// #endif

extern int cblas_splits;
extern double cblas_alltoalltime;
extern double cblas_allgathertime;
extern double cblas_localspmvtime;
extern double cblas_mergeconttime;
extern double cblas_transvectime;

extern double mcl_Abcasttime;
extern double mcl_Bbcasttime;
extern double mcl_localspgemmtime;
extern double mcl_multiwaymergetime;
extern double mcl_kselecttime;
extern double mcl_prunecolumntime;
extern double mcl_symbolictime;

extern double mcl3d_conversiontime;
extern double mcl3d_symbolictime;
extern double mcl3d_Abcasttime;
extern double mcl3d_Bbcasttime;
extern double mcl3d_SUMMAtime;
extern double mcl3d_localspgemmtime;
extern double mcl3d_SUMMAmergetime;
extern double mcl3d_reductiontime;
extern double mcl3d_3dmergetime;
extern double mcl3d_kselecttime;

// performance variables for CUDA related
extern double convertingtime;

// An adapter function that allows using extended-callback EWiseApply with plain-old binary functions that don't want
// the extra parameters.
template <typename RETT, typename NU1, typename NU2, typename BINOP>
class EWiseExtToPlainAdapter
{
   public:
    BINOP plain_binary_op;

    EWiseExtToPlainAdapter(BINOP op) : plain_binary_op(op) {}

    RETT operator()(const NU1& a, const NU2& b, bool aIsNull, bool bIsNull) { return plain_binary_op(a, b); }
};

#include "BFSFriends.h"
#include "BitMap.h"
#include "BlockSpGEMM.h"
#include "DistEdgeList.h"
#include "FullyDistSpVec.h"
#include "FullyDistVec.h"
#include "MPIOp.h"
#include "MPIType.h"
#include "Operations.h"
#include "ParFriends.h"
#include "PreAllocatedSPA.h"
#include "Semirings.h"
#include "SpCCols.h"
#include "SpCRows.h"
#include "SpDCCols.h"
#include "SpDefs.h"
#include "SpParMat.h"
#include "SpParMat3D.h"
#include "SpTuples.h"
#include "VecIterator.h"

#ifdef USE_CUDA
#include "SpCuCRows.h"
#endif  // USE_CUDA

#ifdef __CUDACC__
#include "ParFriends.cuh"
#endif  // __CUDACC__

#endif
