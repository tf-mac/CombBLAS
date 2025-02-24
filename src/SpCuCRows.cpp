//
// Created by Yuxi Hong on 2/22/25.
//
#ifdef USE_CUDA

#include "CombBLAS/SpTuples.h"
#include "CombBLAS/SpCuCRows.h"

namespace combblas {
template<class IT, class NT>
const IT SpCuCRows<IT, NT>::esscount = static_cast<IT>(3);

template<class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows() {
    _cucsr = nullptr;
    _m = 0;
    _n = 0;
    _nnz = 0;
}

// SpTuples should be sorted
// SpTuples is assumed to be column sorted
template<class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows(const SpTuples<IT, NT> &rhs, bool transpose) {
    _m = rhs.getnrow();
    _n = rhs.getncol();
    _nnz = rhs.getnnz();
    // convert SpTuples to _csr
    SpCRows<IT, NT> *hostspcrows = new SpCRows<IT, NT>(rhs, transpose);
    _cucsr = new CuCsr<IT, NT>(*hostspcrows->csrptr()); // allocate device buffer
    delete hostspcrows;
}

template<class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows(const SpCuCRows<IT, NT> &rhs) {
    _m = rhs._m;
    _n = rhs._n;
    _nnz = rhs._nnz;
    if (rhs.csrptr() != nullptr) {
        _cucsr = new CuCsr<IT, NT>(*rhs.csrptr()); // allocate device buffer
    } else {
        _cucsr = nullptr;
    }
}

template<class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows(const SpCuCRows<IT, NT> &&rhs) noexcept {
    _m = rhs._m;
    _n = rhs._n;
    _nnz = rhs._nnz;
    _cucsr = rhs._cucsr;
    rhs._cucsr = nullptr;
}

template<class IT, class NT>
SpCuCRows<IT, NT>::~SpCuCRows() {
    if (_cucsr != nullptr) {
        delete _cucsr;
    }
}

template<class IT, class NT>
SpCuCRows<IT, NT> &SpCuCRows<IT, NT>::operator=(const SpCuCRows<IT, NT> &rhs) {
    if (this == &rhs) {
        return *this;
    }
    if (_cucsr != nullptr) {
        delete _cucsr;
    }
    _m = rhs._m;
    _n = rhs._n;
    _nnz = rhs._nnz;
    if (rhs._cucsr != nullptr) {
        _cucsr = new CuCsr<IT, NT>(*rhs._cucsr); // allocate device buffer
    } else {
        _cucsr = nullptr;
    }

    return *this;
}

template<class IT, class NT>
SpCuCRows<IT, NT> &SpCuCRows<IT, NT>::operator=(const SpCuCRows<IT, NT> &&rhs) noexcept {
    if (this == &rhs) {
        return *this;
    }
    if (_cucsr != nullptr) {
        delete _cucsr;
    }
    _m = rhs._m;
    _n = rhs._n;
    _nnz = rhs._nnz;
    _cucsr = rhs._cucsr;
    rhs._cucsr = nullptr;
    return *this;
}

template<class IT, class NT>
const CuCsr<IT, NT> *SpCuCRows<IT, NT>::csrptr() const {
    return _cucsr;
}

template<class IT, class NT>
IT SpCuCRows<IT, NT>::getnrow() const {
    return _m;
}

template<class IT, class NT>
IT SpCuCRows<IT, NT>::getncol() const {
    return _n;
}

template<class IT, class NT>
IT SpCuCRows<IT, NT>::getnnz() const {
    return _nnz;
}

// Explicit instantiations
template class SpCuCRows<int32_t, float>;
template class SpCuCRows<int32_t, double>;
template class SpCuCRows<int64_t, float>;
template class SpCuCRows<int64_t, double>;

// Explicit instantiations for SpMat
template class SpMat<int32_t, float, SpCuCRows<int32_t, float> >;
template class SpMat<int32_t, double, SpCuCRows<int32_t, double> >;
template class SpMat<int64_t, float, SpCuCRows<int64_t, float> >;
template class SpMat<int64_t, double, SpCuCRows<int64_t, double> >;
} // combblas
#endif // USE_CUDA
