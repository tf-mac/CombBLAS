//
// Created by Yuxi Hong on 2/22/25.
//

#include "CombBLAS/SpTuples.h"

#include <iomanip>

#include "CombBLAS/Compare.h"
#include "CombBLAS/SpCCols.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/csc.h"

namespace combblas
{
template <class IT, class NT>
SpTuples<IT, NT>::SpTuples(int64_t size, IT nRow, IT nCol) : m(nRow), n(nCol), nnz(size)
{
    if (nnz > 0) {
        tuples = new std::tuple<IT, IT, NT>[nnz];
    } else {
        tuples = NULL;
    }
    isOperatorNew = false;
}

template <class IT, class NT>
SpTuples<IT, NT>::SpTuples(int64_t size, IT nRow, IT nCol, std::tuple<IT, IT, NT> *mytuples, bool sorted, bool isOpNew)
    : tuples(mytuples), m(nRow), n(nCol), nnz(size), isOperatorNew(isOpNew)
{
    if (!sorted) {
        SortColBased();
    }
}

/**
 * Generate a SpTuples object from an edge list
 * @param[in,out] edges: edge list that might contain duplicate edges. freed upon return
 * Semantics differ depending on the object created:
 * NT=bool: duplicates are ignored
 * NT='countable' (such as short,int): duplicated as summed to keep count
 **/
template <class IT, class NT>
SpTuples<IT, NT>::SpTuples(int64_t maxnnz, IT nRow, IT nCol, std::vector<IT> &edges, bool removeloops)
    : m(nRow), n(nCol)
{
    if (maxnnz > 0) {
        tuples = new std::tuple<IT, IT, NT>[maxnnz];
    }
    for (int64_t i = 0; i < maxnnz; ++i) {
        rowindex(i) = edges[2 * i + 0];
        colindex(i) = edges[2 * i + 1];
        numvalue(i) = (NT)1;
    }
    std::vector<IT>().swap(edges);  // free memory for edges

    nnz = maxnnz;  // for now (to sort)
    SortColBased();

    int64_t cnz = 0;
    int64_t dup = 0;
    int64_t self = 0;
    nnz = 0;
    while (cnz < maxnnz) {
        int64_t j = cnz + 1;
        while (j < maxnnz && rowindex(cnz) == rowindex(j) && colindex(cnz) == colindex(j)) {
            numvalue(cnz) += numvalue(j);
            numvalue(j++) = 0;  // mark for deletion
            ++dup;
        }
        if (removeloops && rowindex(cnz) == colindex(cnz)) {
            numvalue(cnz) = 0;
            --nnz;
            ++self;
        }
        ++nnz;
        cnz = j;
    }

    std::tuple<IT, IT, NT> *ntuples = new std::tuple<IT, IT, NT>[nnz];
    int64_t j = 0;
    for (int64_t i = 0; i < maxnnz; ++i) {
        if (numvalue(i) != 0) {
            ntuples[j++] = tuples[i];
        }
    }
    assert(j == nnz);

    delete[] tuples;
    tuples = ntuples;
    isOperatorNew = false;
}

/**
 * Generate a SpTuples object from StackEntry array, then delete that array
 * @param[in] multstack {value-key pairs where keys are pair<col_ind, row_ind> sorted lexicographically}
 * \remark Since input is column sorted, the tuples are automatically generated in that way too
 **/
template <class IT, class NT>
SpTuples<IT, NT>::SpTuples(int64_t size, IT nRow, IT nCol, StackEntry<NT, std::pair<IT, IT> > *&multstack)
    : m(nRow), n(nCol), nnz(size)
{
    isOperatorNew = false;
    if (nnz > 0) {
        tuples = new std::tuple<IT, IT, NT>[nnz];
    }
    for (int64_t i = 0; i < nnz; ++i) {
        colindex(i) = multstack[i].key.first;
        rowindex(i) = multstack[i].key.second;
        numvalue(i) = multstack[i].value;
    }
    delete[] multstack;
}
//! Constructor for converting SpDCCols matrix -> SpTuples
template <class IT, class NT>
SpTuples<IT, NT>::SpTuples(const SpDCCols<IT, NT> &rhs) : m(rhs.m), n(rhs.n), nnz(rhs.nnz)
{
    if (nnz > 0) {
        FillTuples(rhs.dcsc);
    }
    isOperatorNew = false;
}

template <class IT, class NT>
SpTuples<IT, NT>::SpTuples(const SpCCols<IT, NT> &rhs) : m(rhs.m), n(rhs.n), nnz(rhs.nnz)
{
    isOperatorNew = false;
    if (nnz > 0) {
        tuples = new std::tuple<IT, IT, NT>[nnz];
        Csc<IT, NT> *csc = rhs.csc;
        IT k = 0;
        for (IT i = 0; i < csc->n; ++i) {
            for (IT j = csc->jc[i]; j < csc->jc[i + 1]; ++j) {
                colindex(k) = i;
                rowindex(k) = csc->ir[j];
                numvalue(k++) = csc->num[j];
            }
        }
    }
}

template <class IT, class NT>
SpTuples<IT, NT>::~SpTuples()
{
    // This tuples_deleted member is a temporary patch to avoid memory leak from MemEfficietnSpGEMM3D
    if ((nnz > 0) && (tuples_deleted != true)) {
        if (isOperatorNew)
            ::operator delete(tuples);
        else
            delete[] tuples;
    }
}

/**
 * Hint1: copy constructor (constructs a new object. i.e. this is NEVER called on an existing object)
 * Hint2: Base's default constructor is called under the covers
 *	  Normally Base's copy constructor should be invoked but it doesn't matter here as Base has no data members
 */
template <class IT, class NT>
SpTuples<IT, NT>::SpTuples(const SpTuples<IT, NT> &rhs) : m(rhs.m), n(rhs.n), nnz(rhs.nnz)
{
    tuples = new std::tuple<IT, IT, NT>[nnz];
    isOperatorNew = false;
    for (IT i = 0; i < nnz; ++i) {
        tuples[i] = rhs.tuples[i];
    }
}

template <class IT, class NT>
inline void SpTuples<IT, NT>::FillTuples(Dcsc<IT, NT> *mydcsc)
{
    tuples = new std::tuple<IT, IT, NT>[nnz];
    IT k = 0;
    for (IT i = 0; i < mydcsc->nzc; ++i) {
        for (IT j = mydcsc->cp[i]; j < mydcsc->cp[i + 1]; ++j) {
            colindex(k) = mydcsc->jc[i];
            rowindex(k) = mydcsc->ir[j];
            numvalue(k++) = mydcsc->numx[j];
        }
    }
}

// Hint1: The assignment operator (operates on an existing object)
// Hint2: The assignment operator is the only operator that is not inherited.
//		  Make sure that base class data are also updated during assignment
template <class IT, class NT>
SpTuples<IT, NT> &SpTuples<IT, NT>::operator=(const SpTuples<IT, NT> &rhs)
{
    if (this != &rhs)  // "this" pointer stores the address of the class instance
    {
        if (nnz > 0) {
            // make empty
            if (isOperatorNew)
                ::operator delete(tuples);
            else
                delete[] tuples;
        }
        m = rhs.m;
        n = rhs.n;
        nnz = rhs.nnz;
        isOperatorNew = false;

        if (nnz > 0) {
            tuples = new std::tuple<IT, IT, NT>[nnz];
            for (IT i = 0; i < nnz; ++i) {
                tuples[i] = rhs.tuples[i];
            }
        }
    }
    return *this;
}

/**
 * \pre {The object is either column-sorted or row-sorted, either way the identical entries will be consecutive}
 **/
template <class IT, class NT>
void SpTuples<IT, NT>::RemoveDuplicates(std::function<NT(NT, NT)> BinOp)
{
    if (nnz > 0) {
        std::vector<std::tuple<IT, IT, NT> > summed;
        summed.push_back(tuples[0]);

        for (IT i = 1; i < nnz; ++i) {
            if ((joker::get<0>(summed.back()) == joker::get<0>(tuples[i])) &&
                (joker::get<1>(summed.back()) == joker::get<1>(tuples[i]))) {
                joker::get<2>(summed.back()) = BinOp(joker::get<2>(summed.back()), joker::get<2>(tuples[i]));
            } else {
                summed.push_back(tuples[i]);
            }
        }
        if (isOperatorNew)
            ::operator delete(tuples);
        else
            delete[] tuples;
        tuples = new std::tuple<IT, IT, NT>[summed.size()];
        isOperatorNew = false;
        std::copy(summed.begin(), summed.end(), tuples);
        nnz = summed.size();
    }
}

template <class IT, class NT>
void SpTuples<IT, NT>::PrintInfo()
{
    std::cout << "This is a SpTuples class" << std::endl;

    std::cout << "m: " << m;
    std::cout << ", n: " << n;
    std::cout << ", nnz: " << nnz << std::endl;

    for (IT i = 0; i < nnz; ++i) {
        if (rowindex(i) < 0 || colindex(i) < 0) {
            std::cout << "Negative index at " << i << std::endl;
            return;
        } else if (rowindex(i) >= m || colindex(i) >= n) {
            std::cout << "Index " << i << " too big with values (" << rowindex(i) << "," << colindex(i) << ")"
                      << std::endl;
        }
    }

    if (m < 8 && n < 8)  // small enough to print
    {
        NT **A = SpHelper::allocate2D<NT>(m, n);
        for (IT i = 0; i < m; ++i)
            for (IT j = 0; j < n; ++j) A[i][j] = 0.0;

        for (IT i = 0; i < nnz; ++i) {
            A[rowindex(i)][colindex(i)] = numvalue(i);
        }
        for (IT i = 0; i < m; ++i) {
            for (IT j = 0; j < n; ++j) {
                std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2) << A[i][j];
                std::cout << " ";
            }
            std::cout << std::endl;
        }
        SpHelper::deallocate2D(A, m);
    }
}

// Sorting functions
template <class IT, class NT>
void SpTuples<IT, NT>::SortRowBased()
{
    std::sort(tuples, tuples + nnz, [](const auto &a, const auto &b) { return std::get<0>(a) < std::get<0>(b); });
}

template <class IT, class NT>
void SpTuples<IT, NT>::SortColBased()
{
    std::sort(tuples, tuples + nnz, [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); });
}

// Loop addition and removal
template <class IT, class NT>
IT SpTuples<IT, NT>::AddLoops(NT loopval, bool replaceExisting)
{
    std::vector<bool> existing(n, false);
    IT loop = 0;
    for (IT i = 0; i < nnz; ++i) {
        if (rowindex(i) == colindex(i)) {
            ++loop;
            existing[rowindex(i)] = true;
            if (replaceExisting) {
                numvalue(i) = loopval;
            }
        }
    }
    return loop;
}
template <class IT, class NT>
IT SpTuples<IT, NT>::AddLoops(std::vector<NT> loopvals, bool replaceExisting)
{
    // expectation n == loopvals.size())

    std::vector<bool> existing(n, false);  // none of the diagonals exist
    IT loop = 0;
    for (IT i = 0; i < nnz; ++i) {
        if (joker::get<0>(tuples[i]) == joker::get<1>(tuples[i])) {
            ++loop;
            existing[joker::get<0>(tuples[i])] = true;
            if (replaceExisting) joker::get<2>(tuples[i]) = loopvals[joker::get<0>(tuples[i])];
        }
    }
    std::vector<IT> missingindices;
    for (IT i = 0; i < n; ++i) {
        if (!existing[i]) missingindices.push_back(i);
    }
    IT toadd = n - loop;  // number of new entries needed (equals missingindices.size())
    auto *ntuples = new std::tuple<IT, IT, NT>[nnz + toadd];

    std::copy(tuples, tuples + nnz, ntuples);

    for (IT i = 0; i < toadd; ++i) {
        ntuples[nnz + i] = std::make_tuple(missingindices[i], missingindices[i], loopvals[missingindices[i]]);
    }
    if (isOperatorNew)
        ::operator delete(tuples);
    else
        delete[] tuples;
    tuples = ntuples;
    isOperatorNew = false;
    nnz = nnz + toadd;
    return loop;
}
template <class IT, class NT>
IT SpTuples<IT, NT>::RemoveLoops()
{
    IT loop = 0;
    for (IT i = 0; i < nnz; ++i) {
        if (rowindex(i) == colindex(i)) {
            ++loop;
        }
    }
    return loop;
}

template <class IT, class NT>
std::pair<IT, IT> SpTuples<IT, NT>::RowLimits()
{
    if (nnz > 0) {
        RowCompare<IT, NT> rowcmp;
        std::tuple<IT, IT, NT> *maxit = std::max_element(tuples, tuples + nnz, rowcmp);
        std::tuple<IT, IT, NT> *minit = std::min_element(tuples, tuples + nnz, rowcmp);
        return std::make_pair(joker::get<0>(*minit), joker::get<0>(*maxit));
    } else
        return std::make_pair(0, 0);
}
template <class IT, class NT>
std::pair<IT, IT> SpTuples<IT, NT>::ColLimits()
{
    if (nnz > 0) {
        ColCompare<IT, NT> colcmp;
        std::tuple<IT, IT, NT> *maxit = std::max_element(tuples, tuples + nnz, colcmp);
        std::tuple<IT, IT, NT> *minit = std::min_element(tuples, tuples + nnz, colcmp);
        return std::make_pair(joker::get<1>(*minit), joker::get<1>(*maxit));
    } else
        return std::make_pair(0, 0);
}
template class SpTuples<int32_t, bool>;
template class SpTuples<int32_t, float>;
template class SpTuples<int32_t, double>;
template class SpTuples<int32_t, int32_t>;
template class SpTuples<int64_t, bool>;
template class SpTuples<int64_t, float>;
template class SpTuples<int64_t, double>;
template class SpTuples<int64_t, int64_t>;

// At this point, complete type of of SpTuples is known, safe to declare these specialization (but macros won't work as
// they are preprocessed)
template <>
struct promote_trait<SpTuples<int, int>, SpTuples<int, int> > {
    typedef SpTuples<int, int> T_promote;
};

template <>
struct promote_trait<SpTuples<int, float>, SpTuples<int, float> > {
    typedef SpTuples<int, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int, double>, SpTuples<int, double> > {
    typedef SpTuples<int, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int, bool>, SpTuples<int, int> > {
    typedef SpTuples<int, int> T_promote;
};

template <>
struct promote_trait<SpTuples<int, int>, SpTuples<int, bool> > {
    typedef SpTuples<int, int> T_promote;
};

template <>
struct promote_trait<SpTuples<int, int>, SpTuples<int, float> > {
    typedef SpTuples<int, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int, float>, SpTuples<int, int> > {
    typedef SpTuples<int, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int, int>, SpTuples<int, double> > {
    typedef SpTuples<int, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int, double>, SpTuples<int, int> > {
    typedef SpTuples<int, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int, unsigned>, SpTuples<int, bool> > {
    typedef SpTuples<int, unsigned> T_promote;
};

template <>
struct promote_trait<SpTuples<int, bool>, SpTuples<int, unsigned> > {
    typedef SpTuples<int, unsigned> T_promote;
};

template <>
struct promote_trait<SpTuples<int, bool>, SpTuples<int, double> > {
    typedef SpTuples<int, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int, bool>, SpTuples<int, float> > {
    typedef SpTuples<int, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int, double>, SpTuples<int, bool> > {
    typedef SpTuples<int, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int, float>, SpTuples<int, bool> > {
    typedef SpTuples<int, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, int>, SpTuples<int64_t, int> > {
    typedef SpTuples<int64_t, int> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, float>, SpTuples<int64_t, float> > {
    typedef SpTuples<int64_t, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, double>, SpTuples<int64_t, double> > {
    typedef SpTuples<int64_t, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, int64_t>, SpTuples<int64_t, int64_t> > {
    typedef SpTuples<int64_t, int64_t> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, bool>, SpTuples<int64_t, int> > {
    typedef SpTuples<int64_t, int> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, int>, SpTuples<int64_t, bool> > {
    typedef SpTuples<int64_t, int> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, int>, SpTuples<int64_t, float> > {
    typedef SpTuples<int64_t, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, float>, SpTuples<int64_t, int> > {
    typedef SpTuples<int64_t, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, int>, SpTuples<int64_t, double> > {
    typedef SpTuples<int64_t, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, double>, SpTuples<int64_t, int> > {
    typedef SpTuples<int64_t, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, unsigned>, SpTuples<int64_t, bool> > {
    typedef SpTuples<int64_t, unsigned> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, bool>, SpTuples<int64_t, unsigned> > {
    typedef SpTuples<int64_t, unsigned> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, bool>, SpTuples<int64_t, double> > {
    typedef SpTuples<int64_t, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, bool>, SpTuples<int64_t, float> > {
    typedef SpTuples<int64_t, float> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, double>, SpTuples<int64_t, bool> > {
    typedef SpTuples<int64_t, double> T_promote;
};

template <>
struct promote_trait<SpTuples<int64_t, float>, SpTuples<int64_t, bool> > {
    typedef SpTuples<int64_t, float> T_promote;
};

}  // namespace combblas
