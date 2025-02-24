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


#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifdef _STDINT_H
#undef _STDINT_H
#endif
#ifdef _GCC_STDINT_H  // for cray
#undef _GCC_STDINT_H  // original stdint does #include_next<"/opt/gcc/4.5.2/snos/lib/gcc/x86_64-suse-linux/4.5.2/include/stdint-gcc.h">
#endif
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "SequenceHeaps/knheap.C"
#include "psort/MersenneTwister.h"
#include "psort/psort.h"
#include "psort/psort_samplesort.h"
// #include "CommGrid.h"

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#if defined(COMBBLAS_BOOST)
#ifdef CRAYCOMP
#include <boost/config/compiler/cray.hpp>
#endif
#include <boost/tr1/memory.hpp>
#include <boost/tr1/tuple.hpp>
#include <boost/tr1/unordered_map.hpp>
#define joker boost  // namespace
#elif defined(COMBBLAS_TR1)
#include <tr1/memory>
#include <tr1/tuple>
#include <tr1/type_traits>
#include <tr1/unordered_map>
#define joker std::tr1
#elif defined(_MSC_VER) && (_MSC_VER < 1600)
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#define joker std::tr1
#else  // C++11
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#define joker std
#endif
// for VC2008

// Just in case the -fopenmp didn't define _OPENMP by itself
#ifdef THREADED
// #ifndef _OPENMP
// #define _OPENMP
// #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern int cblas_splits; // TODO: move this inside namespace
extern int GPUTradeoff;

namespace combblas {
#define ONEMILLION 1000000
#define MAXLINELENGTH 200
#define MINLINELENGTH 2
#define PRINT_LIMIT 50
#define EPSILON 0.01
#define FLOPSPERLOC 0  // always use SPA based merger inside the sequential code
#define HEAPMERGE 1    // use heapmerge for accumulating contributions from row neighbors
#define MEM_EFFICIENT_STAGES 16
#define MAXVERTNAME 64

 // MPI::Abort codes
#define GRIDMISMATCH 3001
#define DIMMISMATCH 3002
#define NOTSQUARE 3003
#define NOFILE 3004
#define MATRIXALIAS 3005
#define UNKNOWNMPITYPE 3006
#define INVALIDPARAMS 3007

 // Enable bebug prints
 //#define SPREFDEBUG
 //#define IODEBUG
 //#define SPGEMMDEBUG

 // MPI Message tags
 // Prefixes denote functions
 //	TR: Transpose
 //	RD: ReadDistribute
 //	RF: Sparse matrix indexing
#define TRTAGNZ 121
#define TRTAGM 122
#define TRTAGN 123
#define TRTAGROWS 124
#define TRTAGCOLS 125
#define TRTAGVALS 126
#define RDTAGINDS 127
#define RDTAGVALS 128
#define RDTAGNNZ 129
#define RFROWIDS 130
#define RFCOLIDS 131
#define TRROWX 132
#define TRCOLX 133
#define TRX 134
#define TRI 135
#define TRNNZ 136
#define TROST 137
#define TRLUT 138
#define SWAPTAG 139
#define ROTATE 140
#define PUPSIZE 141
#define PUPDATA 142

 enum Dim { Column, Row };

 // force 8-bytes alignment in heap allocated memory
#ifndef ALIGNX
#define ALIGNX 8
#endif

#ifndef THRESHOLD
#define THRESHOLD 4  // if range1.size() / range2.size() < threshold, use scanning based indexing
#endif

#ifndef MEMORYINBYTES
#define MEMORYINBYTES (196 * 1048576)  // 196 MB, it is advised to define MEMORYINBYTES to be "at most" (1/4)th of available memory per core



#endif
} // namespace combblas

