//
// Created by Yuxi Hong on 2/21/25.
//

#ifdef USE_CUDA

#include <cuda_runtime_api.h>
#include <iostream>

#include "CombBLAS/csr.h"
#include "CombBLAS/cucsr.h"
#include "CombBLAS/cuutils.h"

namespace combblas {
template<typename IT, typename NT>
CuCsr<IT, NT>::CuCsr() : _jc(nullptr), _ir(nullptr), _num(nullptr), _n(0), _nz(0) {
}

template<typename IT, typename NT>
CuCsr<IT, NT>::CuCsr(const int64_t nnz, const int64_t nRows) : _jc(nullptr), _ir(nullptr), _num(nullptr), _n(nRows),
                                                               _nz(nnz) {
    _nz = nnz;
    _n = nRows;
    if (_n > 0) {
        gpuErrchk(cudaMalloc(&_ir, sizeof(IT) * (_n + 1)));
    }
    if (_nz > 0) {
        gpuErrchk(cudaMalloc(&_jc, sizeof(IT) * _nz));
        gpuErrchk(cudaMalloc(&_num, sizeof(NT) * _nz));
    }
}

template<typename IT, typename NT>
CuCsr<IT, NT>::CuCsr(const CuCsr<IT, NT> &rhs)
    : _n(rhs._n), _nz(rhs._nz), _jc(nullptr), _ir(nullptr), _num(nullptr) {
    if (_n > 0) {
        gpuErrchk(cudaMalloc(&_ir, sizeof(IT) * (_n + 1)));
        gpuErrchk(cudaMemcpy(_ir, rhs._ir, sizeof(IT) * (_n + 1), cudaMemcpyDeviceToDevice));
    }

    if (_nz > 0) {
        gpuErrchk(cudaMalloc(&_jc, sizeof(IT) * _nz));
        gpuErrchk(cudaMalloc(&_num, sizeof(NT) * _nz));
        gpuErrchk(cudaMemcpy(_jc, rhs._jc, sizeof(IT) * _nz, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(_num, rhs._num, sizeof(NT) * _nz, cudaMemcpyDeviceToDevice));
    }
}

template<typename IT, typename NT>
CuCsr<IT, NT>::CuCsr(const CuCsr<IT, NT> &&rhs) noexcept
    : _n(rhs._n), _nz(rhs._nz), _jc(rhs._jc), _ir(rhs._ir), _num(rhs._num) {
}

template<typename IT, typename NT>
CuCsr<IT, NT>::CuCsr(const Csr<IT, NT> &rhs) {
    _n = rhs._n;
    _nz = rhs._nz;
    // Allocate new resources and copy data
    if (_n > 0) {
        gpuErrchk(cudaMalloc(&_ir, sizeof(IT) * (_n + 1)));
        gpuErrchk(cudaMemcpy(_ir, rhs._ir, sizeof(IT) * (_n + 1), cudaMemcpyHostToDevice));
    }

    if (_nz > 0) {
        gpuErrchk(cudaMalloc(&_jc, sizeof(IT) * (_nz + 1)));
        gpuErrchk(cudaMalloc(&_num, sizeof(NT) * (_nz + 1)));
        gpuErrchk(cudaMemcpy(_num, rhs._num, sizeof(NT) * _nz, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(_jc, rhs._jc, sizeof(IT) * _nz, cudaMemcpyHostToDevice));
    }
}

template<typename IT, typename NT>
CuCsr<IT, NT>::~CuCsr() {
    if (_ir != nullptr) {
        gpuErrchk(cudaFree(_ir)); // it's safe to free it.
    }
    if (_num != nullptr) {
        gpuErrchk(cudaFree(_num));
    }
    if (_jc != nullptr) {
        gpuErrchk(cudaFree(_jc));
    }
}

template<typename IT, typename NT>
CuCsr<IT, NT> &CuCsr<IT, NT>::operator=(const CuCsr<IT, NT> &other) {
    // Self-assignment check
    if (this == &other) {
        return *this;
    }
    // Free existing resources
    if (_ir != nullptr) {
        gpuErrchk(cudaFree(_ir));
        _ir = nullptr;
    }
    if (_jc != nullptr) {
        gpuErrchk(cudaFree(_jc));
        _jc = nullptr;
    }
    if (_num != nullptr) {
        gpuErrchk(cudaFree(_num));
        _num = nullptr;
    }

    // Copy metadata
    _n = other._n;
    _nz = other._nz;
    // Allocate new resources and copy data
    if (_n > 0) {
        gpuErrchk(cudaMalloc(&_ir, sizeof(IT) * (_n + 1)));
        gpuErrchk(cudaMemcpy(_ir, other._ir, sizeof(IT) * (_n + 1), cudaMemcpyDeviceToDevice));
    }
    if (_nz > 0) {
        gpuErrchk(cudaMalloc(&_jc, sizeof(IT) * _nz));
        gpuErrchk(cudaMalloc(&_num, sizeof(NT) * _nz));
        gpuErrchk(cudaMemcpy(_jc, other._jc, sizeof(IT) * _nz, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(_num, other._num, sizeof(NT) * _nz, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

template<typename IT, typename NT>
CuCsr<IT, NT> &CuCsr<IT, NT>::operator=(const CuCsr<IT, NT> &&rhs) noexcept {
    // Self-assignment check
    if (this == &rhs) {
        return *this;
    }
    // Free existing resources
    if (_ir != nullptr) {
        gpuErrchk(cudaFree(_ir));
        _ir = rhs._ir;
    }
    if (_jc != nullptr) {
        gpuErrchk(cudaFree(_jc));
        _jc = rhs._jc;
    }
    if (_num != nullptr) {
        gpuErrchk(cudaFree(_num));
        _num = rhs._num;
    }

    // Copy metadata
    _n = rhs._n;
    _nz = rhs._nz;
    return *this;
}

template class CuCsr<int32_t, float>;
template class CuCsr<int32_t, double>;
template class CuCsr<int64_t, float>;
template class CuCsr<int64_t, double>;
}
#endif // USE_CUDA
