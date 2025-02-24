#pragma once

// #include "CombBLAS.h"
#include <omp.h>

#include <vector>

#include "PBBS/radixSort.h"
#include "SpTuples.h"

namespace combblas
{

/*
 Multithreaded prefix sum
 Inputs:
    in: an input array
    size: the length of the input array "in"
    nthreads: number of threads used to compute the prefix sum

 Output:
    return an array of size "size+1"
    the memory of the output array is allocated internallay

 Example:

    in = [2, 1, 3, 5]
    out = [0, 2, 3, 6, 11]
 */
template <typename T>
T* prefixSum(T* in, int size, int nthreads);
/***************************************************************************
 * Find indices of column splitters in a list of tuple in parallel.
 * Inputs:
 *      tuples: an array of SpTuples each tuple is (rowid, colid, val)
 *      nsplits: number of splits requested
 *  Output:
 *      splitters: An array of size (nsplits+1) storing the starts and ends of split tuples.
 *      different type used for output since we might need int or IT
 ***************************************************************************/

template <typename RT, typename IT, typename NT>
std::vector<RT> findColSplitters(SpTuples<IT, NT>*& spTuples, int nsplits);
// Find ColSplitters using finger search
// Run by one threrad
template <typename RT, typename IT, typename NT>
std::vector<RT> findColSplittersFinger(SpTuples<IT, NT>*& spTuples, int nsplits);
// Symbolic serial merge : only estimates nnz
template <class IT, class NT>
IT SerialMergeNNZ(const std::vector<SpTuples<IT, NT>*>& ArrSpTups);
/*
 "Internal function" called by MultiwayMerge inside threaded region.
 The merged list is stored in a preallocated buffer ntuples
 Never called from outside.
 Assumption1: the input lists are already column sorted
 Assumption2: at least two lists are passed to this function
 Assumption3: the input and output lists are to be deleted by caller
 */

template <class SR, class IT, class NT>
void SerialMerge(const std::vector<SpTuples<IT, NT>*>& ArrSpTups, std::tuple<IT, IT, NT>* ntuples);
// Symbolic serial merge : only estimates nnz
template <class IT, class NT>
IT* SerialMergeNNZHash(const std::vector<SpTuples<IT, NT>*>& ArrSpTups, IT& totnnz, IT& maxnnzPerCol, IT startCol,
                       IT endCol);
// Serially merge a split along the column
// startCol and endCol denote the start and end of the current split
// maxcolnnz: maximum nnz in a merged column (from symbolic)
template <class SR, class IT, class NT>
void SerialMergeHash(const std::vector<SpTuples<IT, NT>*>& ArrSpTups, std::tuple<IT, IT, NT>* ntuples, IT* colnnz,
                     IT maxcolnnz, IT startCol, IT endCol, bool sorted);
// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template <class SR, class IT, class NT>
SpTuples<IT, NT>* MultiwayMerge(std::vector<SpTuples<IT, NT>*>& ArrSpTups, IT mdim = 0, IT ndim = 0,
                                bool delarrs = false);
// --------------------------------------------------------
// Hash-based multiway merge
// Columns of the input matrices may or may not be sorted
//  the hash merging algorithm does not need sorted inputs
// If sorted=true, columns of the output matrix are sorted
// --------------------------------------------------------
template <class SR, class IT, class NT>
SpTuples<IT, NT>* MultiwayMergeHash(std::vector<SpTuples<IT, NT>*>& ArrSpTups, IT mdim = 0, IT ndim = 0,
                                    bool delarrs = false, bool sorted = true);
// --------------------------------------------------------
// Hash-based multiway merge
// Columns of the input matrices may or may not be sorted
//  the hash merging algorithm does not need sorted inputs
// If sorted=true, columns of the output matrix are sorted
// --------------------------------------------------------
template <class SR, class IT, class NT>
SpTuples<IT, NT>* MultiwayMergeHashSliding(std::vector<SpTuples<IT, NT>*>& ArrSpTups, IT mdim = 0, IT ndim = 0,
                                           bool delarrs = false, bool sorted = true, IT maxHashTableSize = 16384);

}  // namespace combblas
