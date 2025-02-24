#pragma once

namespace combblas
{

// local sparse matrix format

template <class IT, class NT>
struct Csr;
template <class IT, class NT>
struct Csc;
template <class IT, class NT>
class Dcsc;

#ifdef USE_CUDA
template <class IT, class NT>
struct CuCsr;
#endif

template <class IT, class NT>
class SpTuples;
template <class IT, class NT>
class SpDCCols;
template <class IT, class NT>
class SpCCols;
template <class IT, class NT>
class SpCRows;

template <class IT, class NT, class DER>
class SpMat;

// distributed sparse matrix and vector
template <class IT, class NT>
class FullyDist;

template <class IT, class NT>
class FullyDistVec;

template <class IT, class NT>
class FullyDistSpVec;

template <class IT, class NT, class DER>
class SpParMat;

template <class IT>
class DistEdgeList;

// other utility classes
template <class IU, class NU>
class DenseVectorLocalIterator;

}  // namespace combblas
