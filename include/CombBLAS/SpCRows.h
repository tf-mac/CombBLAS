#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>

// #include "Logger.h"
#include "SpTuples.h"
#include "csr.h"

namespace combblas {
// extern std::shared_ptr<spdlog::logger> cblogger;

// compress sparse row matrix with iterator.
// TODO: make it more nice class. Here is just an ugly impl.
template<class IT, class NT>
class SpCRows {
private:
    int64_t _m;
    int64_t _n;
    int64_t _nnz;

    Csr<IT, NT> *_csr;

public:
    typedef IT LocalIT;
    typedef NT LocalNT;

    SpCRows(const SpTuples<IT, NT> &rhs, bool transpose);

    SpCRows(const SpCCols<IT, NT> &rhs);

    SpCRows();

    ~SpCRows();

    SpCRows<IT, NT> &operator=(const SpCRows<IT, NT> &rhs);

    // getter and setter
    bool isZero() const { return (_nnz == 0); }
    const static IT esscount;

    int64_t getnrow() const;

    int64_t getncol() const;

    int64_t getnnz() const;

    const Csr<IT, NT> *csrptr() const;

    // IO
    void ReadMM(const std::string mtxname);

    friend SpDCCols<IT, NT>;
};

template<class IT, class NT>
const IT SpCRows<IT, NT>::esscount = static_cast<IT>(3);

// SpTuples should be sorted
// SpTuples is assumed to be column sorted
template<class IT, class NT>
SpCRows<IT, NT>::SpCRows(const SpTuples<IT, NT> &rhs, bool transpose) {
    // convert SpTuples to _csr
    _m = rhs.getnrow();
    _n = rhs.getncol();
    _nnz = rhs.getnnz();
    if (transpose) {
        std::cerr << "not implemented!" << std::endl;
        _csr = nullptr;
    } else {
        if (_nnz == 0) {
            _csr = nullptr;
        } else {
            _csr = new Csr<IT, NT>(_nnz, _m);
            std::vector<IT> work(_m + 1, (IT) 0);
            for (IT k = 0; k < _nnz; ++k) {
                IT tmp = rhs.rowindex(k);
                work[tmp + 1]++; // row counts
            }
            // prefix sum
            std::partial_sum(work.begin(), work.end(), work.begin());
            std::copy(work.begin(), work.end(), _csr->_ir);
            std::vector<std::pair<IT, NT> > tosort(_nnz);
            for (IT k = 0; k < _nnz; ++k) {
                tosort[work[rhs.rowindex(k)]++] = std::make_pair(rhs.colindex(k), rhs.numvalue(k));
            }
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (IT i = 0; i < _m; ++i) {
                sort(tosort.begin() + _csr->_ir[i], tosort.begin() + _csr->_ir[i + 1]);
                IT ind;
                typename std::vector<std::pair<IT, NT> >::iterator itr; // iterator is a dependent name
                for (itr = tosort.begin() + _csr->_ir[i], ind = _csr->_ir[i]; itr != tosort.begin() + _csr->_ir[i + 1];
                     ++itr, ++ind) {
                    _csr->_jc[ind] = itr->first;
                    _csr->_num[ind] = itr->second;
                }
            }
        }
    }
}

template<class IT, class NT>
SpCRows<IT, NT>::SpCRows(const SpCCols<IT, NT> &rhs) {
    SpTuples<IT, NT> tuples(rhs);
    SpCRows<IT, NT> object(tuples, false);
    // cblogger->info("m {}, n {}, nnz {}", object.getnrow(), object.getncol(), object.getnnz());
    _m = _n = _nnz = 0;
    _csr = nullptr;
}

template<class IT, class NT>
SpCRows<IT, NT>::SpCRows() {
    _m = 0;
    _n = 0;
    _nnz = 0;
    _csr = nullptr;
}

template<class IT, class NT>
SpCRows<IT, NT>::~SpCRows() {
    if (_csr) {
        delete _csr;
    }
}

template<class IT, class NT>
SpCRows<IT, NT> &SpCRows<IT, NT>::operator=(const SpCRows<IT, NT> &rhs) {
    if (this == &rhs) {
        return *this;
    }
    if (_csr) {
        delete _csr;
        _csr = nullptr;
    } else {
        _csr = new Csr<IT, NT>(*rhs._csr);
        _m = rhs._m;
        _n = rhs._n;
        _nnz = rhs._nnz;
    }
}

template<class IT, class NT>
int64_t SpCRows<IT, NT>::getnrow() const {
    return _m;
}

template<class IT, class NT>
int64_t SpCRows<IT, NT>::getncol() const {
    return _n;
}

template<class IT, class NT>
int64_t SpCRows<IT, NT>::getnnz() const {
    return _nnz;
}

template<class IT, class NT>
const Csr<IT, NT> *SpCRows<IT, NT>::csrptr() const {
    return _csr;
}
} // namespace combblas
