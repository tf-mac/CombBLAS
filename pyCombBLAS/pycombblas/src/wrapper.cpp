#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <pybind11/detail/common.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mpi.h>
#include <limits.h>
#include <tuple>
#include <vector>
#include "DistVec/FullyDistVec.hpp"
#include "grid/CommGrid.hpp"
#include "functional/Operations.hpp"
#include "functional/SpParHelper.hpp"
#include "DistMat/SpParMat1D.hpp"
#include "DistFriends/SpParMat1DFriends.hpp"
#include "SpFormat/csc.hpp"
#include "SpFormat/dcsc.hpp"
#include "SpFormat/SpTuples.hpp"
#include "SpFormat/SpDCCols.hpp"
#include "DistMat/SpParMat.hpp"
#include "Friends/mtSpGEMM.hpp"
#include "ConnectedComponent/CC.hpp"
// #include "fast_matrix_market/read_body_threads.hpp"
#include "fast_float/fast_float.h"
namespace py = pybind11;
using namespace combblas;
typedef int64_t IT;
typedef int64_t IT;


template<class IT, class NT>
std::vector<std::tuple<IT,IT,NT>> 
ParallelReadTuples(const std::vector<std::string>& input_list){
    // You can only read from input_list, not modify it.
    std::vector<std::tuple<IT,IT,NT>> tvec(input_list.size());
    int64_t tpcnt = 0;
    #pragma omp parallel for 
    for (size_t i=0; i<input_list.size(); i++) {
        std::string item = input_list[i];
        if(item.size() == 0) {
            std::cerr<<"i " << i << "value "<<std::endl;
        }
        IT ii,jj;
        NT vv;
        auto answer = fast_float::from_chars(item.data(), item.data() + item.size(), ii);
        if(answer.ec != std::errc()) { std::cerr << "parsing failure\n"; }
        answer = fast_float::from_chars(answer.ptr+1, item.data() + item.size(), jj);
        if(answer.ec != std::errc()) { std::cerr << "parsing failure\n"; }
        answer = fast_float::from_chars(answer.ptr+1, item.data() + item.size(), vv);
        if(answer.ec != std::errc()) { std::cerr << "parsing failure\n"; }
        tpcnt++;
        tvec[i] = std::make_tuple(ii,jj,vv);
    }
    // int64_t tprec;
    // MPI_Allreduce(&tpcnt, &tprec, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    // int myrank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // if(myrank==0){
    //     std::cerr<<"total read"<<tprec<<std::endl;
    // }
    return tvec;
}




namespace combblas {




template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> pyLocalSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB)
{
    auto ptr = LocalSpGEMM<SR, NTO>(A,B,clearA,clearB);
    return *ptr;
}

template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> pyLocalHybridSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB)
{
    auto ptr = LocalHybridSpGEMM<SR,NTO,IT,NT1,NT2>(A,B,clearA,clearB,nullptr);
    return *ptr;
}


std::shared_ptr<CommGrid> GetCommGrid(){
    return std::make_shared<CommGrid>();
}



template<class IT, class NT1, class NT2>
std::vector<IT> estimateFLOPDCCols(SpDCCols<IT, NT1> & A, SpDCCols<IT, NT2> & B){
    IT * flopsC = estimateFLOP(A,B);
    std::vector<IT> flopsCvec(A.getncol(),0);
    for(IT i=0; i<B.getnzc(); i++) flopsCvec[i] = flopsC[i];
    delete[] flopsC;
    return flopsCvec;
}

template<class IT, class NT>
SpDCCols<IT, NT> PyMakeSpDCCols(py::list &pytuplelist, int64_t nrow, int64_t ncol){
    std::vector<std::tuple<IT,IT,NT>> tpvec;
    for (auto item : pytuplelist) {
        auto tuple = item.cast<py::tuple>();  // Cast each item to a tuple
        int64_t a = tuple[0].cast<int64_t>();
        int64_t b = tuple[1].cast<int64_t>();
        double c = tuple[2].cast<double>();
        tpvec.push_back(std::make_tuple(a,b,c));
    }
    SpTuples<IT, NT> spt(tpvec.size(), nrow, ncol,tpvec.data());
    spt.tuples_deleted = true;
    SpDCCols<IT, NT> dcsc(spt, false);
    return dcsc;
}

} // combblas namespace ends


py::array_t<int64_t> CSCColNNZ(py::array_t<int64_t> indptr, int64_t ncols) {
    py::buffer_info buf1 = indptr.request();
    auto result = py::array_t<int64_t>(ncols);
    py::buffer_info buf2 = result.request();
    int64_t *ptr1 = static_cast<int64_t *>(buf1.ptr);
    int64_t *ptr2 = static_cast<int64_t *>(buf2.ptr);
    #pragma omp parallel for
    for (int64_t idx = 0; idx < ncols; idx++)
        ptr2[idx] = ptr1[idx+1] - ptr1[idx];
    return result;
}


py::array_t<int64_t> CSCColFlops(
    py::array_t<int64_t> indptr, 
    py::array_t<int64_t> colnnz, 
    py::array_t<int64_t> indices,
    int64_t ncols) {
    py::buffer_info buf1 = indptr.request(), buf2 = colnnz.request(), buf3 = indices.request();

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<int64_t>(ncols);

    py::buffer_info buf4 = result.request();

    int64_t *indptr_ptr1 = static_cast<int64_t *>(buf1.ptr);
    int64_t *colnnz_ptr2 = static_cast<int64_t *>(buf2.ptr);
    int64_t *indices_ptr3 = static_cast<int64_t *>(buf3.ptr);
    int64_t *flops_ptr4 = static_cast<int64_t *>(buf4.ptr);
    for(int64_t i=0; i<ncols; i++) flops_ptr4[i] = 0;
    #pragma omp parallel for
    for (int64_t idx = 0; idx < ncols; idx++){
        for(int64_t r=indptr_ptr1[idx]; r<indptr_ptr1[idx+1]; r++){
            flops_ptr4[idx] += colnnz_ptr2[indices_ptr3[r]];
        }
    }
    return result;
}


#define INST_SpDCCols(IT, NT, NAME)                        \
    py::class_<SpDCCols<IT, NT>>(m, NAME)                  \
    .def(py::init<const std::string>())       \
    .def(py::init<const SpDCCols<IT, NT>&>())       \
    .def(py::init<const SpTuples<IT, NT> &, bool>())       \
    .def("getnnz",&SpDCCols<IT, NT>::getnnz)               \
    .def("getncol",&SpDCCols<IT, NT>::getncol)             \
    .def("getnrow",&SpDCCols<IT, NT>::getnrow)             \
    .def("GetDCSCOBJ",&SpDCCols<IT, NT>::GetDCSCOBJ)       \
    ;

#define INST_SpParMat(IT, NT, NAME)                    \
    py::class_<SpParMat<IT, NT, SpDCCols<IT,NT>>>(m,NAME)               \
    .def(py::init<>())                                      \
    .def(py::init<const SpParMat<IT, NT, SpDCCols<IT,NT>>&>())      \
    .def(py::init<const SpParMat1D<IT, NT, SpDCCols<IT,NT>>&, std::vector<IT>, bool>())  \
    .def(py::self += py::self)                                      \
    .def("ParallelReadMM",&SpParMat<IT,NT,SpDCCols<IT,NT>>::ParallelReadMM<>) \
    .def("getnnz",&SpParMat<IT,NT,SpDCCols<IT,NT>>::getnnz)             \
    .def("getnzc",&SpParMat<IT,NT,SpDCCols<IT,NT>>::getnzc)             \
    .def("getnrow",&SpParMat<IT,NT,SpDCCols<IT,NT>>::getnrow)             \
    .def("getncol",&SpParMat<IT,NT,SpDCCols<IT,NT>>::getncol)             \
    .def("seq",&SpParMat<IT,NT,SpDCCols<IT,NT>>::seq)                 \
    .def("Transpose",&SpParMat<IT,NT,SpDCCols<IT,NT>>::Transpose)                 \
    .def("ParallelWriteMM", &SpParMat<IT,NT,SpDCCols<IT,NT>>::ParallelWriteMM<>) \
    .def("GetLocalRowColStart",&SpParMat<IT,NT,SpDCCols<IT,NT>>::GetLocalRowColStart) \
    ;



PYBIND11_MODULE(combblas_cppext, m) {
    py::class_<CommGrid>(m,
    "CommGrid")
    .def(py::init<>())
    ;
    py::class_<Dcsc<int64_t, double>>(m,
    "Dcsc_i64f64")
    .def(py::init<>())
    ;
    py::class_<Csc<int64_t, double>>(m,
    "Csc_i64f64")
    .def(py::init<>())
    .def("coord",&Csc<int64_t, double>::coord)
    .def("values",&Csc<int64_t, double>::values)
    ;

    INST_SpDCCols(int64_t, double, "SpDCCols_i64f64")

    // SpCCols
    py::class_<SpCCols<int64_t, double>>(m,
    "SpCCols_i64f64")
    .def(py::init<const SpTuples<int64_t, double> &, bool>())
    .def(py::init<const SpDCCols<int64_t, double> &>())
    .def("getnnz",&SpCCols<int64_t, double>::getnnz)
    .def("GetCSCOBJ",&SpCCols<int64_t, double>::GetCSCOBJ) // return value
    ;

    py::class_<SpTuples<int64_t, double>>(m,
    "SpTuples_i64f64")
    .def(py::init<int64_t,int64_t,int64_t>())
    .def("GetValues",&SpTuples<int64_t, double>::GetValues)
    .def("GetRows",&SpTuples<int64_t, double>::GetRows)
    .def("GetCols",&SpTuples<int64_t, double>::GetCols)
    .def("Shape",&SpTuples<int64_t, double>::Shape)
    ;
    
    // SpParMat 2D class
    INST_SpParMat(int64_t,double,"SpParMat_i64f64")
    
    py::class_<SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>>(m, "SpParMat1D_i64f64")
    .def(py::init<const SpParMat<int64_t,double,SpDCCols<int64_t,double>>, const std::vector<int64_t>, const std::vector<int64_t> >())
    .def(py::init<const SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>>())
    .def("getblocksizevec",&SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>::getblocksizevec)
    .def("getblocksizeprefix",&SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>::getblocksizeprefix)
    .def("getrowblocksizevec",&SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>::getrowblocksizevec)
    .def("getrowblocksizeprefix",&SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>::getrowblocksizeprefix)
    .def("Permute",&SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>::Permute)
    .def("ParallelWriteMM",&SpParMat1D<int64_t, double, SpDCCols<int64_t, double>>::pyParallelWriteMM)
    ;

    py::class_<maximum<double>>(m,
    "maximum_f64")
    .def(py::init<>())
    .def("__call__",&maximum<double>::operator())
    ;

    // MCL interface
    m.def("MCLPruneRecoverySelect",&MCLPruneRecoverySelect<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("MakeColStochastic",&MakeColStochastic<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("RandPermute", &RandPermute<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("PermuteSpParMat", &PermuteSpParMat<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("RemoveIsolated",&RemoveIsolated<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("AdjustLoops",&AdjustLoops<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("Inflate",&Inflate<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("Chaos",&Chaos<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("Symmetricize",&Symmetricize<int64_t,double,SpDCCols<int64_t, double>>);
    
    py::class_<SpgemmOpts>(m,"SpgemmOpts")
    .def(py::init<>());

    // gemm interface
    typedef PlusTimesSRing<double, double> PTFF;
    typedef double NT;
    typedef int64_t IT;
    typedef SpDCCols<IT, NT> DER;
    m.def("Mult_AnXBn_Synch_i64f64",&Mult_AnXBn_Synch<PTFF,NT,DER,IT,NT,NT,DER,DER>);
    m.def("Mult_AnXBn_1D_CbC_RDMA_FetchAll_i64f64",&Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF,NT,DER,IT,NT,NT,DER,DER>);
    m.def("pyLocalSpGEMM",&pyLocalSpGEMM<PTFF,NT,IT,NT,NT>);
    m.def("pyLocalHybridSpGEMM",&pyLocalHybridSpGEMM<PTFF,NT,IT,NT,NT>);
    m.def("GetCommGrid",&GetCommGrid);

    m.def("CC",&pyCC<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("EstimateFLOP",&EstimateFLOP<PTFF,int64_t,double,double,DER,DER>);
    m.def("estimateFLOPDCCols",&estimateFLOPDCCols<int64_t,double,double>);
    m.def("ColumnNNZ",&ColumnNNZ<int64_t,double,SpDCCols<int64_t, double>>);
    m.def("PyMakeSpDCCols",&PyMakeSpDCCols<int64_t,double>);
    m.def("CSCColNNZ",&CSCColNNZ);
    m.def("CSCColFlops",&CSCColFlops);

    m.def("ParallelReadTuples",&ParallelReadTuples<int64_t,double>);
}