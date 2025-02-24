//
// Created by Yuxi Hong on 2/21/25.
//


#include "CombBLAS/csr.h"
#ifdef USE_CUDA
#include "CombBLAS/cucsr.h"
#include "CombBLAS/cuutils.h"
#endif

#include <iostream>

namespace combblas {
template<typename IT, typename NT>
Csr<IT, NT>::Csr(): _jc(nullptr), _ir(nullptr), _num(nullptr), _n(0), _nz(0) {
}

#ifdef USE_CUDA
template<typename IT, typename NT>
Csr<IT, NT>::Csr(const CuCsr<IT, NT> &rhs): _n(rhs._n), _nz(rhs._nz) {
    if (_n <= 0) {
        std::cerr << "ERROR!!!!: invalid number of rows in Csr<IT, NT>::Csr(const CuCsr<IT, NT> &rhs) constructor: " <<
                _n << std::endl;
        _ir = nullptr;
        _num = nullptr;
        _jc = nullptr;
    } else {
        _ir = new IT[_n + 1];
        gpuErrchk(cudaMemcpy(_ir, rhs._ir, sizeof(IT) * (_n + 1), cudaMemcpyDeviceToHost));
        if (_nz > 0) {
            _num = new NT[_nz];
            _jc = new IT[_nz];
            gpuErrchk(cudaMemcpy(_num, rhs._num, sizeof(NT) * _nz, cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(_jc, rhs._jc, sizeof(NT) * _nz, cudaMemcpyDeviceToHost));
        } else {
            _num = nullptr;
            _jc = nullptr;
        }
    }
}
#endif

template<typename IT, typename NT>
Csr<IT, NT>::Csr(const int64_t nnz, const int64_t nRows) {
    _nz = nnz;
    _n = nRows;
    if (_n <= 0) {
        std::cerr << "ERROR!!!!: invalid number of rows in Csr constructor: " << _n << std::endl;
        _ir = nullptr;
        _num = nullptr;
        _jc = nullptr;
    } else {
        _ir = new IT[_n + 1];
        if (_nz > 0) {
            _num = new NT[_nz];
            _jc = new IT[_nz];
        } else {
            _num = nullptr;
            _jc = nullptr;
        }
    }
}

template<typename IT, typename NT>
Csr<IT, NT>::Csr(const Csr<IT, NT> &rhs) {
    // Copy dimensions and number of non-zeros
    _n = rhs._n;
    _nz = rhs._nz;
    if (_n <= 0) {
        std::cerr << "ERROR!!!!: invalid number of rows in Csr copy constructor: " << std::endl;
        _ir = nullptr;
        _num = nullptr;
        _jc = nullptr;
    } else {
        // Allocate memory and copy row pointers (_ir)
        _ir = new IT[_n + 1];
        std::copy(rhs._ir, rhs._ir + _n + 1, _ir);

        if (_nz > 0) {
            // Allocate memory and copy column indices (_jc)
            _jc = new IT[_nz];
            std::copy(rhs._jc, rhs._jc + _nz, _jc);

            // Allocate memory and copy numeric values (_num)
            _num = new NT[_nz];
            std::copy(rhs._num, rhs._num + _nz, _num);
        } else {
            // Handle edge case where there are no non-zeros
            _jc = nullptr;
            _num = nullptr;
        }
    }
}

template<typename IT, typename NT>
Csr<IT, NT>::~Csr() {
    delete[] _ir; // _ir must be allocated, it's safe to delete it.
    if (_nz > 0) {
        delete[] _num;
        delete[] _jc;
    }
}

// Explicit instantiations
template class Csr<int32_t, float>;
template class Csr<int32_t, double>;
template class Csr<int64_t, float>;
template class Csr<int64_t, double>;
} // namespace combblas
