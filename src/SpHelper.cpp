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

#include "CombBLAS/SpHelper.h"

#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "CombBLAS/HeapEntry.h"
#include "CombBLAS/Semirings.h"
#include "CombBLAS/SequenceHeaps/knheap.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/hash.hpp"

namespace combblas
{

template <typename T>
std::vector<size_t> SpHelper::find_order(const std::vector<T> &values)
{
    size_t index = 0;
    std::vector<std::pair<T, size_t> > tosort;
    for (auto &val : values) {
        tosort.push_back(std::make_pair(val, index++));
    }
    sort(tosort.begin(), tosort.end());
    std::vector<size_t> permutation;
    for (auto &sorted : tosort) {
        permutation.push_back(sorted.second);
    }
    return permutation;
}

template <typename IT1, typename NT1, typename IT2, typename NT2>
void SpHelper::push_to_vectors(std::vector<IT1> &rows, std::vector<IT1> &cols, std::vector<NT1> &vals, IT2 ii, IT2 jj,
                               NT2 vv, int symmetric, bool onebased)
{
    if (onebased) {
        ii--; /* adjust from 1-based to 0-based */
        jj--;
    }
    rows.push_back(ii);
    cols.push_back(jj);
    vals.push_back(vv);
    if (symmetric && ii != jj) {
        rows.push_back(jj);
        cols.push_back(ii);
        vals.push_back(vv);
    }
}

void SpHelper::ProcessLinesWithStringKeys(std::vector<std::map<std::string, uint64_t> > &allkeys,
                                          std::vector<std::string> &lines, int nprocs)
{
    std::string frstr, tostr;
    uint64_t frhash, tohash;
    double vv;
    for (auto itr = lines.begin(); itr != lines.end(); ++itr) {
        char fr[MAXVERTNAME];
        char to[MAXVERTNAME];
        sscanf(itr->c_str(), "%s %s %lg", fr, to, &vv);
        frstr = std::string(fr);
        tostr = std::string(to);
        MurmurHash3_x64_64(frstr.c_str(), frstr.size(), 0, &frhash);
        MurmurHash3_x64_64(tostr.c_str(), tostr.size(), 0, &tohash);

        double range_fr = static_cast<double>(frhash) * static_cast<double>(nprocs);
        double range_to = static_cast<double>(tohash) * static_cast<double>(nprocs);
        size_t owner_fr = range_fr / static_cast<double>(std::numeric_limits<uint64_t>::max());
        size_t owner_to = range_to / static_cast<double>(std::numeric_limits<uint64_t>::max());

        // cout << frstr << " with hash " << frhash << " is going to " << owner_fr << endl;
        // cout << tostr << " with hash " << tohash << " is going to " << owner_to << endl;

        allkeys[owner_fr].insert(std::make_pair(frstr, frhash));
        allkeys[owner_to].insert(std::make_pair(tostr, tohash));
    }
    lines.clear();
}

template <typename IT1, typename NT1>
void SpHelper::ProcessStrLinesNPermute(std::vector<IT1> &rows, std::vector<IT1> &cols, std::vector<NT1> &vals,
                                       std::vector<std::string> &lines, std::map<std::string, uint64_t> &ultperm)
{
    char *fr = new char[MAXVERTNAME];
    char *to = new char[MAXVERTNAME];
    std::string frstr, tostr;
    double vv;
    for (auto itr = lines.begin(); itr != lines.end(); ++itr) {
        sscanf(itr->c_str(), "%s %s %lg", fr, to, &vv);
        frstr = std::string(fr);
        tostr = std::string(to);

        rows.emplace_back((IT1)ultperm[frstr]);
        cols.emplace_back((IT1)ultperm[tostr]);
        vals.emplace_back((NT1)vv);
    }
    delete[] fr;
    delete[] to;
    lines.clear();
}

template <typename IT1, typename NT1>
void SpHelper::ProcessLines(std::vector<IT1> &rows, std::vector<IT1> &cols, std::vector<NT1> &vals,
                            std::vector<std::string> &lines, int symmetric, int type, bool onebased)
{
    if (type == 0)  // real
    {
        int64_t ii, jj;
        double vv;
        for (auto itr = lines.begin(); itr != lines.end(); ++itr) {
            // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of
            // characters (i.e., a C-string)
            sscanf(itr->c_str(), "%lld %lld %lg", &ii, &jj, &vv);
            SpHelper::push_to_vectors(rows, cols, vals, ii, jj, vv, symmetric, onebased);
        }
    } else if (type == 1)  // integer
    {
        int64_t ii, jj, vv;
        for (auto itr = lines.begin(); itr != lines.end(); ++itr) {
            sscanf(itr->c_str(), "%lld %lld %lld", &ii, &jj, &vv);
            SpHelper::push_to_vectors(rows, cols, vals, ii, jj, vv, symmetric, onebased);
        }
    } else if (type == 2)  // pattern
    {
        int64_t ii, jj;
        for (auto itr = lines.begin(); itr != lines.end(); ++itr) {
            sscanf(itr->c_str(), "%lld %lld", &ii, &jj);
            SpHelper::push_to_vectors(rows, cols, vals, ii, jj, 1, symmetric, onebased);
        }
    } else {
        std::cout << "COMBBLAS: Unrecognized matrix market scalar type" << std::endl;
    }
    lines.clear();
}

template <typename T>
const T *SpHelper::p2a(const std::vector<T> &v)  // pointer to array
{
    if (v.empty())
        return NULL;
    else
        return (&v[0]);
}
template const int32_t *SpHelper::p2a(const std::vector<int32_t> &);
template const int64_t *SpHelper::p2a(const std::vector<int64_t> &);

template <typename T>
T *SpHelper::p2a(std::vector<T> &v)  // pointer to array
{
    if (v.empty())
        return NULL;
    else
        return (&v[0]);
}
template int32_t *SpHelper::p2a(std::vector<int32_t> &);
template int64_t *SpHelper::p2a(std::vector<int64_t> &);

template <typename _ForwardIterator>
bool SpHelper::is_sorted(_ForwardIterator __first, _ForwardIterator __last)
{
    if (__first == __last) return true;

    _ForwardIterator __next = __first;
    for (++__next; __next != __last; __first = __next, ++__next)
        if (*__next < *__first) return false;
    return true;
}

template <typename _ForwardIterator, typename _StrictWeakOrdering>
bool SpHelper::is_sorted(_ForwardIterator __first, _ForwardIterator __last, _StrictWeakOrdering __comp)
{
    if (__first == __last) return true;

    _ForwardIterator __next = __first;
    for (++__next; __next != __last; __first = __next, ++__next)
        if (__comp(*__next, *__first)) return false;
    return true;
}

// template <typename _ForwardIter, typename T>
// void SpHelper::iota(_ForwardIter __first, _ForwardIter __last, T __val)
// {
//     while (__first != __last) *__first++ = __val++;
// }
// template void SpHelper::iota(int32_t*, int);

template <typename In, typename Out, typename UnPred>
Out SpHelper::copyIf(In first, In last, Out result, UnPred pred)
{
    for (; first != last; ++first)
        if (pred(*first)) *result++ = *first;
    return (result);
}

template <typename T, typename I1, typename I2>
T **SpHelper::allocate2D(I1 m, I2 n)
{
    T **array = new T *[m];
    for (I1 i = 0; i < m; ++i) array[i] = new T[n];
    return array;
}
template std::string **SpHelper::allocate2D<std::string, int32_t, int32_t>(int32_t, int32_t);
template std::string **SpHelper::allocate2D<std::string, int64_t, int64_t>(int64_t, int64_t);
template bool **SpHelper::allocate2D<bool, int32_t, int32_t>(int32_t, int32_t);
template int32_t **SpHelper::allocate2D<int32_t, int32_t, int32_t>(int32_t, int32_t);
template float **SpHelper::allocate2D<float, int32_t, int32_t>(int32_t, int32_t);
template double **SpHelper::allocate2D<double, int32_t, int32_t>(int32_t, int32_t);
template bool **SpHelper::allocate2D<bool, int64_t, int64_t>(int64_t, int64_t);
template int64_t **SpHelper::allocate2D<int64_t, int64_t, int64_t>(int64_t, int64_t);
template float **SpHelper::allocate2D<float, int64_t, int64_t>(int64_t, int64_t);
template double **SpHelper::allocate2D<double, int64_t, int64_t>(int64_t, int64_t);

template <typename T, typename I>
void SpHelper::deallocate2D(T **array, I m)
{
    for (I i = 0; i < m; ++i) delete[] array[i];
    delete[] array;
}
template void SpHelper::deallocate2D<std::string, int32_t>(std::string **, int32_t);
template void SpHelper::deallocate2D<int32_t, int32_t>(int32_t **, int32_t);
template void SpHelper::deallocate2D<bool, int32_t>(bool **, int32_t);
template void SpHelper::deallocate2D<float, int32_t>(float **, int32_t);
template void SpHelper::deallocate2D<double, int32_t>(double **, int32_t);
template void SpHelper::deallocate2D<std::string, int64_t>(std::string **, int64_t);
template void SpHelper::deallocate2D<bool, int64_t>(bool **, int64_t);
template void SpHelper::deallocate2D<int64_t, int64_t>(int64_t **, int64_t);
template void SpHelper::deallocate2D<float, int64_t>(float **, int64_t);
template void SpHelper::deallocate2D<double, int64_t>(double **, int64_t);

template <typename NT, typename IT>
void SpHelper::ShrinkArray(NT *&array, IT newsize)
{
    NT *narray = new NT[newsize];
    memcpy(narray, array, newsize * sizeof(NT));  // copy only a portion of the old elements

    delete[] array;
    array = narray;
}

template <typename NT, typename IT>
void SpHelper::DoubleStack(StackEntry<NT, std::pair<IT, IT> > *&multstack, IT &cnzmax, IT add)
{
    StackEntry<NT, std::pair<IT, IT> > *tmpstack = multstack;
    multstack = new StackEntry<NT, std::pair<IT, IT> >[2 * cnzmax + add];
    memcpy(multstack, tmpstack, sizeof(StackEntry<NT, std::pair<IT, IT> >) * cnzmax);

    cnzmax = 2 * cnzmax + add;
    delete[] tmpstack;
}

template <typename IT>
bool SpHelper::first_compare(std::pair<IT, IT> pair1, std::pair<IT, IT> pair2)
{
    return pair1.first < pair2.first;
}

template bool SpHelper::first_compare(std::pair<int32_t, int32_t>, std::pair<int32_t, int32_t>);
template bool SpHelper::first_compare(std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>);

/**
 * Pop an element, do the numerical semiring multiplication & insert the result into multstack
 */
template <typename SR, typename NT1, typename NT2, typename IT, typename OVT>
IT SpHelper::Popping(NT1 *numA, NT2 *numB, StackEntry<OVT, std::pair<IT, IT> > *multstack, IT &cnz,
                     KNHeap<std::pair<IT, IT>, IT> &sHeap, Isect<IT> *isect1, Isect<IT> *isect2)
{
    std::pair<IT, IT> key;
    IT inc;
    sHeap.deleteMin(&key, &inc);

    OVT value = SR::multiply(numA[isect1[inc].current], numB[isect2[inc].current]);
    if (!SR::returnedSAID()) {
        if (cnz != 0) {
            if (multstack[cnz - 1].key == key)  // already exists
            {
                multstack[cnz - 1].value = SR::add(multstack[cnz - 1].value, value);
            } else {
                multstack[cnz].value = value;
                multstack[cnz].key = key;
                ++cnz;
            }
        } else {
            multstack[cnz].value = value;
            multstack[cnz].key = key;
            ++cnz;
        }
    }
    return inc;
}

/**
 * Finds the intersecting row indices of Adcsc and col indices of Bdcsc
 * @param[IT] Bdcsc {the transpose of the dcsc structure of matrix B}
 * @param[IT] Adcsc {the dcsc structure of matrix A}
 **/
template <typename IT, typename NT1, typename NT2>
void SpHelper::SpIntersect(const Dcsc<IT, NT1> &Adcsc, const Dcsc<IT, NT2> &Bdcsc, Isect<IT> *&cols, Isect<IT> *&rows,
                           Isect<IT> *&isect1, Isect<IT> *&isect2, Isect<IT> *&itr1, Isect<IT> *&itr2)
{
    cols = new Isect<IT>[Adcsc.nzc];
    rows = new Isect<IT>[Bdcsc.nzc];

    for (IT i = 0; i < Adcsc.nzc; ++i) {
        cols[i].index = Adcsc.jc[i];  // column index
        cols[i].size = Adcsc.cp[i + 1] - Adcsc.cp[i];
        cols[i].start = Adcsc.cp[i];    // pointer to row indices
        cols[i].current = Adcsc.cp[i];  // pointer to row indices
    }
    for (IT i = 0; i < Bdcsc.nzc; ++i) {
        rows[i].index = Bdcsc.jc[i];  // column index
        rows[i].size = Bdcsc.cp[i + 1] - Bdcsc.cp[i];
        rows[i].start = Bdcsc.cp[i];    // pointer to row indices
        rows[i].current = Bdcsc.cp[i];  // pointer to row indices
    }

    /* A single set_intersection would only return the elements of one sequence
     * But we also want random access to the other array's elements
     * Thus we do the intersection twice
     */
    IT mink = std::min(Adcsc.nzc, Bdcsc.nzc);
    isect1 = new Isect<IT>[mink];  // at most
    isect2 = new Isect<IT>[mink];  // at most
    itr1 = std::set_intersection(cols, cols + Adcsc.nzc, rows, rows + Bdcsc.nzc, isect1);
    itr2 = std::set_intersection(rows, rows + Bdcsc.nzc, cols, cols + Adcsc.nzc, isect2);
    // itr1 & itr2 are now pointing to one past the end of output sequences
}

/**
 * Performs cartesian product on the dcsc structures.
 * Indices to perform the product are given by isect1 and isect2 arrays
 * Returns the "actual" number of elements in the merged stack
 * Bdcsc is "already transposed" (i.e. Bdcsc->ir gives column indices, and Bdcsc->jc gives row indices)
 **/
template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
IT SpHelper::SpCartesian(const Dcsc<IT, NT1> &Adcsc, const Dcsc<IT, NT2> &Bdcsc, IT kisect, Isect<IT> *isect1,
                         Isect<IT> *isect2, StackEntry<OVT, std::pair<IT, IT> > *&multstack)
{
    std::pair<IT, IT> supremum(std::numeric_limits<IT>::max(), std::numeric_limits<IT>::max());
    std::pair<IT, IT> infimum(std::numeric_limits<IT>::min(), std::numeric_limits<IT>::min());

    KNHeap<std::pair<IT, IT>, IT> sHeapDcsc(supremum, infimum);

    // Create a sequence heap that will eventually construct DCSC of C
    // The key to sort is pair<col_ind, row_ind> so that output is in column-major order
    for (IT i = 0; i < kisect; ++i) {
        std::pair<IT, IT> key(Bdcsc.ir[isect2[i].current], Adcsc.ir[isect1[i].current]);
        sHeapDcsc.insert(key, i);
    }

    IT cnz = 0;
    IT cnzmax = Adcsc.nz + Bdcsc.nz;  // estimate on the size of resulting matrix C
    multstack = new StackEntry<OVT, std::pair<IT, IT> >[cnzmax];

    bool finished = false;
    while (!finished)  // multiplication loop  (complexity O(flops * log (kisect))
    {
        finished = true;
        if (cnz + kisect > cnzmax)  // double the size of multstack
        {
            DoubleStack(multstack, cnzmax, kisect);
        }

        // inc: the list to increment its pointer in the k-list merging
        IT inc = Popping<SR>(Adcsc.numx, Bdcsc.numx, multstack, cnz, sHeapDcsc, isect1, isect2);
        isect1[inc].current++;

        if (isect1[inc].current < isect1[inc].size + isect1[inc].start) {
            std::pair<IT, IT> key(Bdcsc.ir[isect2[inc].current], Adcsc.ir[isect1[inc].current]);
            sHeapDcsc.insert(key, inc);  // push the same element with a different key [increasekey]
            finished = false;
        }
        // No room to go in isect1[], but there is still room to go in isect2[i]
        else if (isect2[inc].current + 1 < isect2[inc].size + isect2[inc].start) {
            isect1[inc].current = isect1[inc].start;  // wrap-around
            isect2[inc].current++;

            std::pair<IT, IT> key(Bdcsc.ir[isect2[inc].current], Adcsc.ir[isect1[inc].current]);
            sHeapDcsc.insert(key, inc);  // push the same element with a different key [increasekey]
            finished = false;
        } else  // don't push, one of the lists has been deplated
        {
            kisect--;
            if (kisect != 0) {
                finished = false;
            }
        }
    }
    return cnz;
}

template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
IT SpHelper::SpColByCol(const Dcsc<IT, NT1> &Adcsc, const Dcsc<IT, NT2> &Bdcsc, IT nA,
                        StackEntry<OVT, std::pair<IT, IT> > *&multstack)
{
    IT cnz = 0;
    IT cnzmax = Adcsc.nz + Bdcsc.nz;  // estimate on the size of resulting matrix C
    multstack = new StackEntry<OVT, std::pair<IT, IT> >[cnzmax];

    float cf = static_cast<float>(nA + 1) / static_cast<float>(Adcsc.nzc);
    IT csize = static_cast<IT>(ceil(cf));  // chunk size
    IT *aux;
    // IT auxsize = Adcsc.ConstructAux(nA, aux);
    Adcsc.ConstructAux(nA, aux);

    for (IT i = 0; i < Bdcsc.nzc; ++i)  // for all the columns of B
    {
        IT prevcnz = cnz;
        IT nnzcol = Bdcsc.cp[i + 1] - Bdcsc.cp[i];
        HeapEntry<IT, NT1> *wset = new HeapEntry<IT, NT1>[nnzcol];
        // heap keys are just row indices (IT)
        // heap values are <numvalue, runrank>
        // heap size is nnz(B(:,i)

        // colnums vector keeps column numbers requested from A
        std::vector<IT> colnums(nnzcol);

        // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
        // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
        std::vector<std::pair<IT, IT> > colinds(nnzcol);
        std::copy(Bdcsc.ir + Bdcsc.cp[i], Bdcsc.ir + Bdcsc.cp[i + 1], colnums.begin());

        Adcsc.FillColInds(&colnums[0], colnums.size(), colinds, aux, csize);
        IT maxnnz = 0;  // max number of nonzeros in C(:,i)
        IT hsize = 0;

        for (IT j = 0; (unsigned)j < colnums.size(); ++j)  // create the initial heap
        {
            if (colinds[j].first != colinds[j].second)  // current != end
            {
                wset[hsize++] = HeapEntry<IT, NT1>(Adcsc.ir[colinds[j].first], j, Adcsc.numx[colinds[j].first]);
                maxnnz += colinds[j].second - colinds[j].first;
            }
        }
        std::make_heap(wset, wset + hsize);

        if (cnz + maxnnz > cnzmax)  // double the size of multstack
        {
            SpHelper::DoubleStack(multstack, cnzmax, maxnnz);
        }

        // No need to keep redefining key and hentry with each iteration of the loop
        while (hsize > 0) {
            std::pop_heap(wset, wset + hsize);  // result is stored in wset[hsize-1]
            IT locb = wset[hsize - 1].runr;     // relative location of the nonzero in B's current column

            // type promotion done here:
            // static T_promote multiply(const T1 & arg1, const T2 & arg2)
            //	return (static_cast<T_promote>(arg1) * static_cast<T_promote>(arg2) );
            OVT mrhs = SR::multiply(wset[hsize - 1].num, Bdcsc.numx[Bdcsc.cp[i] + locb]);
            if (!SR::returnedSAID()) {
                if (cnz != prevcnz && multstack[cnz - 1].key.second == wset[hsize - 1].key)
                // if (cnz == prevcnz) => first nonzero for this column
                {
                    multstack[cnz - 1].value = SR::add(multstack[cnz - 1].value, mrhs);
                } else {
                    multstack[cnz].value = mrhs;
                    multstack[cnz++].key = std::make_pair(Bdcsc.jc[i], wset[hsize - 1].key);
                    // first entry is the column index, as it is in column-major order
                }
            }

            if ((++(colinds[locb].first)) != colinds[locb].second)  // current != end
            {
                // runr stays the same !
                wset[hsize - 1].key = Adcsc.ir[colinds[locb].first];
                wset[hsize - 1].num = Adcsc.numx[colinds[locb].first];
                std::push_heap(wset, wset + hsize);
            } else {
                --hsize;
            }
        }
        delete[] wset;
    }
    delete[] aux;
    return cnz;
}

template int32_t SpHelper::SpColByCol<PlusTimesSRing<float, float>, int32_t, float, float, float>(
    const Dcsc<int32_t, float> &, const Dcsc<int32_t, float> &, int32_t,
    StackEntry<float, std::pair<int32_t, int32_t> > *&);
template int32_t SpHelper::SpColByCol<PlusTimesSRing<double, double>, int32_t, double, double, double>(
    const Dcsc<int32_t, double> &, const Dcsc<int32_t, double> &, int32_t,
    StackEntry<double, std::pair<int32_t, int32_t> > *&);
template int64_t SpHelper::SpColByCol<PlusTimesSRing<float, float>, int64_t, float, float, float>(
    const Dcsc<int64_t, float> &, const Dcsc<int64_t, float> &, int64_t,
    StackEntry<float, std::pair<int64_t, int64_t> > *&);
template int64_t SpHelper::SpColByCol<PlusTimesSRing<double, double>, int64_t, double, double, double>(
    const Dcsc<int64_t, double> &, const Dcsc<int64_t, double> &, int64_t,
    StackEntry<double, std::pair<int64_t, int64_t> > *&);

}  // namespace combblas
