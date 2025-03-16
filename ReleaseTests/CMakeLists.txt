# Top level directory has the include files

ADD_EXECUTABLE(MultTiming MultTiming.cpp)
ADD_EXECUTABLE(MultTest MultTest.cpp)
ADD_EXECUTABLE(ReduceTest ReduceTest.cpp)
ADD_EXECUTABLE(TransposeTest TransposeTest.cpp)
ADD_EXECUTABLE(IteratorTest IteratorTest.cpp)
ADD_EXECUTABLE(IndexingTest IndexingTest.cpp)
ADD_EXECUTABLE(SpAsgnTest SpAsgnTest.cpp)
ADD_EXECUTABLE(GalerkinNew GalerkinNew.cpp)
ADD_EXECUTABLE(IndexingTiming IndexingTiming.cpp)
ADD_EXECUTABLE(FindSparse FindSparse.cpp)
ADD_EXECUTABLE(ParIOTest ParIOTest.cpp)
ADD_EXECUTABLE(GenWrMat GenWriteMatrix.cpp)
ADD_EXECUTABLE(BlockedSpGEMM BlockedSpGEMM.cpp)
ADD_EXECUTABLE(SpGEMM3DTest SpGEMM3DTest.cpp)
ADD_EXECUTABLE(HashSpGEMMTest HashSpGEMMTest.cpp)
ADD_EXECUTABLE(InducedSubgraphsTest InducedSubgraphsTest.cpp)
ADD_EXECUTABLE(PruneColumn PruneColumn.cpp)
ADD_EXECUTABLE(KTipsTest KTipsTest.cpp)
ADD_EXECUTABLE(mtRCMTest mtrcm.cpp)

TARGET_LINK_LIBRARIES(MultTiming CombBLAS)
TARGET_LINK_LIBRARIES(MultTest CombBLAS)
TARGET_LINK_LIBRARIES(ReduceTest CombBLAS MPI::MPI_CXX)
TARGET_LINK_LIBRARIES(TransposeTest CombBLAS)
TARGET_LINK_LIBRARIES(IteratorTest CombBLAS)
TARGET_LINK_LIBRARIES(IndexingTest CombBLAS)
TARGET_LINK_LIBRARIES(SpAsgnTest CombBLAS)
TARGET_LINK_LIBRARIES(GalerkinNew CombBLAS)
TARGET_LINK_LIBRARIES(IndexingTiming CombBLAS)
TARGET_LINK_LIBRARIES(FindSparse CombBLAS)
TARGET_LINK_LIBRARIES(ParIOTest CombBLAS)
TARGET_LINK_LIBRARIES(GenWrMat CombBLAS)
TARGET_LINK_LIBRARIES(BlockedSpGEMM CombBLAS)
TARGET_LINK_LIBRARIES(SpGEMM3DTest CombBLAS)
TARGET_LINK_LIBRARIES(HashSpGEMMTest CombBLAS)
TARGET_LINK_LIBRARIES(InducedSubgraphsTest CombBLAS)
TARGET_LINK_LIBRARIES(PruneColumn CombBLAS)
TARGET_LINK_LIBRARIES(KTipsTest CombBLAS)
CPMAddPackage(
    NAME Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0  # Adjust to the latest stable version if needed
    OPTIONS "EIGEN_BUILD_DOC OFF"
            "BUILD_TESTING OFF"
            "EIGEN_BUILD_PKGCONFIG OFF"
)
TARGET_LINK_LIBRARIES(mtRCMTest CombBLAS Eigen3::Eigen)

ADD_TEST(NAME GenMMWrite_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:GenWrMat> 20 16 1 scale20_ef16_symmetric.mtx)
ADD_TEST(NAME Multiplication_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:MultTest> ../TESTDATA/rmat_scale16_A.mtx ../TESTDATA/rmat_scale16_B.mtx ../TESTDATA/rmat_scale16_productAB.mtx ../TESTDATA/x_65536_halfdense.txt ../TESTDATA/y_65536_halfdense.txt)
ADD_TEST(NAME SpGEMM3D_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 16 $<TARGET_FILE:SpGEMM3DTest> ../TESTDATA/rmat_scale16_A.mtx ../TESTDATA/rmat_scale16_B.mtx ../TESTDATA/rmat_scale16_productAB.mtx)
ADD_TEST(NAME HashSpGEMMTest COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:HashSpGEMMTest> ../TESTDATA/rmat_scale16_A.mtx ../TESTDATA/rmat_scale16_B.mtx ../TESTDATA/rmat_scale16_productAB.mtx)
ADD_TEST(NAME Reduction_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:ReduceTest> ../TESTDATA/sprand10000 ../TESTDATA/sprand10000_sumcols ../TESTDATA/sprand10000_sumrows)
ADD_TEST(NAME Iterator_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:IteratorTest> ../TESTDATA sprand10000)
ADD_TEST(NAME Transpose_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:TransposeTest> ../TESTDATA betwinput_scale16 betwinput_transposed_scale16)
ADD_TEST(NAME Indexing_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:IndexingTest> ../TESTDATA B_100x100.txt B_10x30_Indexed.txt rand10outta100.txt rand30outta100.txt)
ADD_TEST(NAME SpAsgn_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:SpAsgnTest> ../TESTDATA A_100x100.txt A_with20x30hole.txt dense_20x30matrix.txt A_wdenseblocks.txt 20outta100.txt 30outta100.txt)
ADD_TEST(NAME GalerkinNew_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:GalerkinNew> ../TESTDATA/grid3d_k5.txt ../TESTDATA/offdiag_grid3d_k5.txt ../TESTDATA/diag_grid3d_k5.txt ../TESTDATA/restrict_T_grid3d_k5.txt)
ADD_TEST(NAME FindSparse_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:FindSparse> ../TESTDATA findmatrix.txt)


if (ACCELERATOR STREQUAL "cuda")
    CPMAddPackage("gh:jarro2783/cxxopts@2.2.0")
    add_executable(MultTimingCUDA MultTimingCUDA.cu)
    add_executable(MultAccuracyCUDA MultAccuracyCUDA.cu)
    add_executable(MultTimingCUDACuCSR MultTimingCUDACuCSR.cu)
    add_executable(SingleGPUTimingCUDA SingleGPUTimingCUDA.cu)
    TARGET_LINK_LIBRARIES(MultTimingCUDA CombBLAS cxxopts)
    set_target_properties(MultTimingCUDA PROPERTIES LINKER_LANGUAGE CXX)
    TARGET_LINK_LIBRARIES(MultTimingCUDACuCSR CombBLAS)
    TARGET_LINK_LIBRARIES(MultAccuracyCUDA CombBLAS)
    TARGET_LINK_LIBRARIES(SingleGPUTimingCUDA CombBLAS)
endif ()


