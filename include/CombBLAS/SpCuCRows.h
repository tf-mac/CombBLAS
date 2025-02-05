#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>

#include "SpCRows.h"
#include "SpTuples.h"
#include "cucsr.h"

namespace combblas
{

// compress sparse row matrix with iterator in NVIDIA GPU.
// TODO: make it more nice class. Here is just an ugly impl.
template <class IT, class NT>
class SpCuCRows
{
   private:
    int64_t _m;
    int64_t _n;
    int64_t _nnz;

    CuCsr<IT, NT>* _cucsr;

   public:
    typedef IT LocalIT;
    typedef NT LocalNT;
    SpCuCRows(const SpTuples<IT, NT>& rhs, bool transpose);
    SpCuCRows(const SpCuCRows<IT, NT>& rhs);
    SpCuCRows(const SpCuCRows<IT, NT>&& rhs) noexcept;
    SpCuCRows<IT, NT>& operator=(const SpCuCRows<IT, NT>& rhs);
    SpCuCRows<IT, NT>& operator=(const SpCuCRows<IT, NT>&& rhs) noexcept;
    SpCuCRows();
    ~SpCuCRows();
    // getter and setter
    IT getnrows() const;
    IT getncols() const;
    const CuCsr<IT, NT>* csrptr() const;
    // IO
    void ReadMM(const std::string mtxname);
};

template <class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows()
{
    _cucsr = nullptr;
    _m = 0;
    _n = 0;
    _nnz = 0;
}

// SpTuples should be sorted
// SpTuples is assumed to be column sorted
template <class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows(const SpTuples<IT, NT>& rhs, bool transpose)
{
    _m = rhs.getnrow();
    _n = rhs.getncol();
    _nnz = rhs.getnnz();
    // convert SpTuples to _csr
    SpCRows<IT, NT>* hostspcrows = new SpCRows<IT, NT>(rhs, transpose);
    _cucsr = new CuCsr<IT, NT>(*hostspcrows->csrptr());  // allocate device buffer
    delete hostspcrows;
}

template <class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows(const SpCuCRows<IT, NT>& rhs)
{
    _m = rhs._m;
    _n = rhs._n;
    _nnz = rhs._nnz;
    if (rhs.csrptr() != nullptr) {
        _cucsr = new CuCsr<IT, NT>(*rhs.csrptr());  // allocate device buffer
    } else {
        _cucsr = nullptr;
    }
}

template <class IT, class NT>
SpCuCRows<IT, NT>::SpCuCRows(const SpCuCRows<IT, NT>&& rhs) noexcept
{
    _m = rhs._m;
    _n = rhs._n;
    _nnz = rhs._nnz;
    _cucsr = rhs._cucsr;
    rhs._cucsr = nullptr;
}

template <class IT, class NT>
SpCuCRows<IT, NT>::~SpCuCRows()
{
    if (_cucsr != nullptr) {
        delete _cucsr;
    }
}
template <class IT, class NT>
SpCuCRows<IT, NT>& SpCuCRows<IT, NT>::operator=(const SpCuCRows<IT, NT>& rhs)
{
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
        _cucsr = new CuCsr<IT, NT>(*rhs._cucsr);  // allocate device buffer
    } else {
        _cucsr = nullptr;
    }

    return *this;
}

template <class IT, class NT>
SpCuCRows<IT, NT>& SpCuCRows<IT, NT>::operator=(const SpCuCRows<IT, NT>&& rhs) noexcept
{
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

template <class IT, class NT>
const CuCsr<IT, NT>* SpCuCRows<IT, NT>::csrptr() const
{
    return _cucsr;
}
template <class IT, class NT>
IT SpCuCRows<IT, NT>::getnrows() const
{
    return _m;
}

template <class IT, class NT>
IT SpCuCRows<IT, NT>::getncols() const
{
    return _n;
}

}  // namespace combblas