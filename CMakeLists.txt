cmake_minimum_required(VERSION 3.26)
project(
        CombBLAS
        VERSION 2.0.1
        LANGUAGES C CXX)

# combblas requies c++17
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Add CPM
include(cmake/CPM.cmake)


option(USE_CUDA "Enable CUDA support" OFF)
# Turn this on only if you have a CUDA-aware MPI implementation.
# otherwise CombBLAS will use NCCL for GPU communication.
option(MPI_CUDA "Enable MPI CUDA-Aware support" OFF)


set(SOURCE_FILES "")
# basic CombBLAS library sources
file(GLOB SOURCES_C "src/*.c")
list(APPEND SOURCE_FILES ${SOURCES_C})
file(GLOB SOURCES_CPP "src/*.cpp")
list(APPEND SOURCE_FILES ${SOURCES_CPP})
file(GLOB SOURCES_H "include/CombBLAS/*.h")
list(APPEND SOURCE_FILES ${SOURCES_H})
file(GLOB SOURCES_HPP "include/CombBLAS/*.hpp")
list(APPEND SOURCE_FILES ${SOURCES_HPP})

# add cuda file if cuda is enabled
if (USE_CUDA)
    file(GLOB SOURCES_CUDA_CPP "src/cuda/*.cpp")
    file(GLOB SOURCES_CUDA_CU "src/cuda/*.cu")
    list(APPEND SOURCE_FILES ${SOURCES_CUDA_CPP})
    list(APPEND SOURCE_FILES ${SOURCES_CUDA_CU})
    # set(CombBLASSRC "${CombBLASSRC} ${SOURCES_CUDA_CPP} ${SOURCES_CUDA_CU}")
endif (USE_CUDA)

# Main CombBLAS library
add_library(CombBLAS ${SOURCE_FILES})

# set local include directories
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psort-1.0/include> $<INSTALL_INTERFACE:include>)


# find MPI
find_package(MPI REQUIRED)
## if compiler is cray wrapper, find mpi header folder manually.
if (CMAKE_CXX_COMPILER MATCHES ".*cray.*" OR CMAKE_C_COMPILER MATCHES ".*cray.*")
    message(STATUS "Detected Cray compiler: ${CMAKE_CXX_COMPILER}")
    execute_process(
            COMMAND ${MPI_C_COMPILER} --cray-print-opts=all
            OUTPUT_VARIABLE MPI_FLAGS
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX MATCHALL "-I([^ ]+)" MPI_INCLUDE_FLAGS "${MPI_FLAGS}")
    list(GET MPI_INCLUDE_FLAGS 0 MPI_INCLUDE_DIR_RAW)
    string(REPLACE "-I" "" MPI_INCLUDE_PATH "${MPI_INCLUDE_DIR_RAW}")

    message(STATUS "Detected MPI include path: ${MPI_INCLUDE_PATH}")

    # Set MPI include path globally
    set(MPI_INCLUDE_PATH ${MPI_INCLUDE_PATH} CACHE PATH "MPI include path")
endif ()
target_include_directories(CombBLAS PUBLIC ${MPI_INCLUDE_PATH})


# find OpenMP
find_package(OpenMP)

# find CUDA
if (USE_CUDA)
    # This needs to be split based on if CMake >= 3.17, as this is deprecated above that
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 80 90) # For A100
    enable_language(CUDA)
    string(APPEND CMAKE_CUDA_FLAGS
            " -Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr ")
    # string(APPEND CMAKE_CUDA_FLAGS  " -DUSE_CUDA ")
    if (MPI_CUDA)
        string(APPEND CMAKE_CUDA_FLAGS " -DMPI_CUDA ")
        target_compile_definitions(CombBLAS PUBLIC -DMPI_CUDA)
    endif (MPI_CUDA)
    target_compile_definitions(CombBLAS PUBLIC -DUSE_CUDA)
    target_include_directories(CombBLAS PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_options(CombBLAS PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(CombBLAS PUBLIC ${CUDA_LIBRARIES})
    # not sure why, but rmm taget needs fmt and CCCL. In theory, they should be included inside the rmm
    # but we need to add it manually ..
    CPMAddPackage(
            NAME fmt
            GITHUB_REPOSITORY fmtlib/fmt
            GIT_TAG 11.0.2
    )
    CPMAddPackage(
            NAME CCCL
            GITHUB_REPOSITORY NVIDIA/cccl
            GIT_TAG v2.5.0
    )
    # Add RMM
    CPMAddPackage(
            NAME RMM
            GITHUB_REPOSITORY rapidsai/rmm
            GIT_TAG branch-24.10
    )
    find_package(CCCL REQUIRED)
    find_package(rmm REQUIRED)
    target_link_libraries(CombBLAS PUBLIC rmm::rmm)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
    set_property(
            TARGET Threads::Threads
            PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler
            -pthread> "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")
else ()
    message(STATUS "CUDA features disabled")
endif (USE_CUDA)

if (TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
    target_link_libraries(CombBLAS PUBLIC MPI::MPI_CXX)
else ()
    target_compile_options(CombBLAS PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
    target_link_libraries(CombBLAS PUBLIC "${MPI_CXX_LIBRARIES}"
            "${MPI_CXX_LINKFLAGS}")
    target_include_directories(CombBLAS PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif ()

if (TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
    target_compile_definitions(CombBLAS PUBLIC THREADED)
    target_link_libraries(CombBLAS PUBLIC OpenMP::OpenMP_CXX)
elseif (OPENMP_FOUND)
    target_compile_definitions(CombBLAS PUBLIC THREADED)
    target_compile_options(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
    target_link_libraries(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
endif ()

add_subdirectory(usort)
target_link_libraries(CombBLAS PUBLIC Usortlib)

add_subdirectory(graph500-1.2/generator)
target_link_libraries(CombBLAS PUBLIC GraphGenlib)

# Set up exported configuration This allows CombBLAS to be installed in two
# ways: 1. In /usr/local (or whatever prefix is specified) 2. Exporting the
# current build directory. This allows a user to make modifications to CombBLAS
# and have the changes automatically recompiled for dependent projects. Either
# way, we need to create a CombBLASConfig.cmake.
set(ConfigPackageLocation lib/cmake/CombBLAS)

# Generate version number header
include(GenerateExportHeader)
generate_export_header(CombBLAS)
set_target_properties(CombBLAS PROPERTIES VERSION ${CombBLAS_VERSION})

# installation
install(
        DIRECTORY include/
        DESTINATION include)
install(
        DIRECTORY psort-1.0/include/
        DESTINATION include)
install(
        TARGETS CombBLAS
        EXPORT CombBLASTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES
        DESTINATION include)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
        VERSION ${CombBLAS_VERSION}
        COMPATIBILITY AnyNewerVersion)

# The following commands allow for option 2.
export(
        EXPORT CombBLASTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASTargets.cmake"
        NAMESPACE CombBLAS::)
configure_file(
        cmake/CombBLASConfig.cmake
        "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfig.cmake" COPYONLY)

# Allow for option 2
install(
        EXPORT CombBLASTargets
        FILE CombBLASTargets.cmake
        NAMESPACE CombBLAS::
        DESTINATION ${ConfigPackageLocation})
export(PACKAGE CombBLAS)
install(
        FILES cmake/CombBLASConfig.cmake
        "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
        DESTINATION ${ConfigPackageLocation}
        COMPONENT Devel)

# Testing
enable_testing()
include(CTest)

# Warnings cause the compiler to crash, surpress them to prevent that
add_definitions(-w)

add_subdirectory(ReleaseTests)
add_subdirectory(Applications)
add_subdirectory(Applications/Ordering)
add_subdirectory(Applications/BipartiteMatchings)
add_subdirectory(Applications/SpMSpV-IPDPS2017)
add_subdirectory(Applications/Incremental)
add_subdirectory(3DSpGEMM)

# Add the UnitTests directory
add_subdirectory(UnitTests)