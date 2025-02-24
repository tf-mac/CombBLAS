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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "CombBLAS/Isect.h"
#include "CombBLAS/SequenceHeaps/knheap.h"
#include "CombBLAS/StackEntry.h"

namespace combblas
{
template <class IT, class NT>
class Dcsc;

class SpHelper
{
   public:
    template <typename T>
    static std::vector<size_t> find_order(const std::vector<T> &values);

    template <typename IT1, typename NT1, typename IT2, typename NT2>
    static void push_to_vectors(std::vector<IT1> &rows, std::vector<IT1> &cols, std::vector<NT1> &vals, IT2 ii, IT2 jj,
                                NT2 vv, int symmetric, bool onebased = true);
    static void ProcessLinesWithStringKeys(std::vector<std::map<std::string, uint64_t> > &allkeys,
                                           std::vector<std::string> &lines, int nprocs);
    template <typename IT1, typename NT1>
    static void ProcessStrLinesNPermute(std::vector<IT1> &rows, std::vector<IT1> &cols, std::vector<NT1> &vals,
                                        std::vector<std::string> &lines, std::map<std::string, uint64_t> &ultperm);
    template <typename IT1, typename NT1>
    static void ProcessLines(std::vector<IT1> &rows, std::vector<IT1> &cols, std::vector<NT1> &vals,
                             std::vector<std::string> &lines, int symmetric, int type, bool onebased = true);
    // pointer to array
    template <typename T>
    static const T *p2a(const std::vector<T> &v);
    // pointer to array
    template <typename T>
    static T *p2a(std::vector<T> &v);
    template <typename _ForwardIterator>
    static bool is_sorted(_ForwardIterator __first, _ForwardIterator __last);
    template <typename _ForwardIterator, typename _StrictWeakOrdering>
    static bool is_sorted(_ForwardIterator __first, _ForwardIterator __last, _StrictWeakOrdering __comp);
    template <typename _ForwardIter, typename T>
    static void iota(_ForwardIter __first, _ForwardIter __last, T __val);
    template <typename In, typename Out, typename UnPred>
    static Out copyIf(In first, In last, Out result, UnPred pred);
    template <typename T, typename I1, typename I2>
    static T **allocate2D(I1 m, I2 n);
    template <typename T, typename I>
    static void deallocate2D(T **array, I m);
    template <typename SR, typename NT1, typename NT2, typename IT, typename OVT>
    static IT Popping(NT1 *numA, NT2 *numB, StackEntry<OVT, std::pair<IT, IT> > *multstack, IT &cnz,
                      KNHeap<std::pair<IT, IT>, IT> &sHeap, Isect<IT> *isect1, Isect<IT> *isect2);
    template <typename IT, typename NT1, typename NT2>
    static void SpIntersect(const Dcsc<IT, NT1> &Adcsc, const Dcsc<IT, NT2> &Bdcsc, Isect<IT> *&cols, Isect<IT> *&rows,
                            Isect<IT> *&isect1, Isect<IT> *&isect2, Isect<IT> *&itr1, Isect<IT> *&itr2);
    template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
    static IT SpCartesian(const Dcsc<IT, NT1> &Adcsc, const Dcsc<IT, NT2> &Bdcsc, IT kisect, Isect<IT> *isect1,
                          Isect<IT> *isect2, StackEntry<OVT, std::pair<IT, IT> > *&multstack);
    template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
    static IT SpColByCol(const Dcsc<IT, NT1> &Adcsc, const Dcsc<IT, NT2> &Bdcsc, IT nA,
                         StackEntry<OVT, std::pair<IT, IT> > *&multstack);
    template <typename NT, typename IT>
    static void ShrinkArray(NT *&array, IT newsize);
    template <typename NT, typename IT>
    static void DoubleStack(StackEntry<NT, std::pair<IT, IT> > *&multstack, IT &cnzmax, IT add);

    template <typename IT>
    static bool first_compare(std::pair<IT, IT> pair1, std::pair<IT, IT> pair2);
};
}  // namespace combblas
