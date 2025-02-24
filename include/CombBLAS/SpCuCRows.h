#pragma once

#ifdef USE_CUDA

#include <cstdint>
#include <string>

#include "ldtypedel.h"
#include "SpMat.h"

namespace combblas {
// compress sparse row matrix with iterator in NVIDIA GPU.
template<class IT, class NT>
class SpCuCRows : public SpMat<IT, NT, SpCuCRows<IT, NT> > {
private:
    int64_t _m;
    int64_t _n;
    int64_t _nnz;
    CuCsr<IT, NT> *_cucsr;

public:
    const static IT esscount;
    typedef IT LocalIT;
    typedef NT LocalNT;
    SpCuCRows(const SpTuples<IT, NT> &rhs, bool transpose);
    SpCuCRows(const SpCuCRows<IT, NT> &rhs);
    SpCuCRows(const SpCuCRows<IT, NT> &&rhs) noexcept;
    SpCuCRows<IT, NT> &operator=(const SpCuCRows<IT, NT> &rhs);
    SpCuCRows<IT, NT> &operator=(const SpCuCRows<IT, NT> &&rhs) noexcept;
    SpCuCRows();
    ~SpCuCRows();
    // getter and setter
    IT getnrow() const;
    IT getncol() const;
    IT getnnz() const;
    const CuCsr<IT, NT> *csrptr() const;
    // IO
    void ReadMM(const std::string mtxname);
    friend SpDCCols<IT, NT>;
};
} // namespace combblas

#endif // USE_CUDA


