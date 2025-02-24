/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version  ----------------------------------------------------*/
/* date: 2/xx/2025 ---------------------------------------------*/
/* authors: Yuxi Hong ------------------------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2016, The Regents of the University of California

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
// wrapper for cuSparse CSR format
// only visible when combblas enables cuda.
#ifdef USE_CUDA

namespace combblas
{
// CSR sparse matrix format
template <typename IT, typename NT>
struct CuCsr {
    typedef NT value_type;
    typedef IT index_type;
    IT *_jc;                                                       // row indices, size nz
    IT *_ir;                                                       // row pointers, size n+1
    NT *_num;                                                      // generic values, size nz
    int64_t _n;                                                    // number of rows
    int64_t _nz;                                                   // number of non-zeroes
    CuCsr();                                                       // default constructor
    CuCsr(const int64_t nnz, const int64_t nRows);                 // size: nnz, nRows: number of rows
    CuCsr(const CuCsr<IT, NT> &rhs);                               // copy constructor
    CuCsr(const CuCsr<IT, NT> &&rhs) noexcept;                     // move constructor
    CuCsr<IT, NT> &operator=(const CuCsr<IT, NT> &&rhs) noexcept;  // assignment
    // convert from CSR
    explicit CuCsr(const Csr<IT, NT> &rhs);                // construct from a host csr object
    ~CuCsr();                                              // deconstructor
    CuCsr<IT, NT> &operator=(const CuCsr<IT, NT> &other);  // assignment operator
};
}  // namespace combblas
#endif  // USE_CUDA
