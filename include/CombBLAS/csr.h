#pragma once

#include <iostream>

namespace combblas
{

// CSR sparse matrix format
template <typename IT, typename NT>
struct Csr {
    IT* _jc;                          // row indices, size nz
    IT* _ir;                          // row pointers, size n+1
    NT* _num;                         // generic values, size nz
    int64_t _n;                       // number of rows
    int64_t _nz;                      // number of non-zeroes
    Csr();                            // default constructor
    Csr(int64_t nnz, int64_t nRows);  // size: nnz, nRows: number of rows
    Csr(const Csr<IT, NT>& other);    // copy constructor
    ~Csr();                           // deconstructor
};

template <typename IT, typename NT>
Csr<IT, NT>::Csr(int64_t nnz, int64_t nRows)
{
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

template <typename IT, typename NT>
Csr<IT, NT>::Csr(const Csr<IT, NT>& other)
{
    // Copy dimensions and number of non-zeros
    _n = other._n;
    _nz = other._nz;
    if (_n <= 0) {
        std::cerr << "ERROR!!!!: invalid number of rows in Csr copy constructor: " << std::endl;
        _ir = nullptr;
        _num = nullptr;
        _jc = nullptr;
    } else {
        // Allocate memory and copy row pointers (_ir)
        _ir = new IT[_n + 1];
        std::copy(other._ir, other._ir + _n + 1, _ir);

        if (_nz > 0) {
            // Allocate memory and copy column indices (_jc)
            _jc = new IT[_nz];
            std::copy(other._jc, other._jc + _nz, _jc);

            // Allocate memory and copy numeric values (_num)
            _num = new NT[_nz];
            std::copy(other._num, other._num + _nz, _num);
        } else {
            // Handle edge case where there are no non-zeros
            _jc = nullptr;
            _num = nullptr;
        }
    }
}
template <typename IT, typename NT>
Csr<IT, NT>::~Csr()
{
    delete[] _ir;  // _ir must be allocated, it's safe to delete it.
    if (_nz > 0) {
        delete[] _num;
        delete[] _jc;
    }
}

}  // namespace combblas