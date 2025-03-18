cmake_minimum_required(VERSION 3.26)
project(
        CombBLAS
        VERSION 2.0.1
        LANGUAGES C CXX)

# combblas requies c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Add CPM
include(cmake/CPM.cmake)

# specify GPU communication layer
# default: copy back to cpu and use mpi impl.
# mpicuda: if you have a CUDA-aware MPI implementation.
# nccl: if you has nccl.
set(GPU_COMM "default" CACHE STRING "GPU communication layer choice: mpicuda|nccl|mpi")
set(ACCELERATOR "none" CACHE STRING "ACCELERATOR: cuda|hip")

set(SOURCE_FILES "")
# cpu CombBLAS library sources
file(GLOB TMP "src/*.c")
list(APPEND SOURCE_FILES ${TMP})
file(GLOB TMP "src/*.cpp")
list(APPEND SOURCE_FILES ${TMP})
file(GLOB TMP "include/CombBLAS/*.h")
list(APPEND SOURCE_FILES ${TMP})
file(GLOB TMP "include/CombBLAS/*.hpp")
list(APPEND SOURCE_FILES ${TMP})


if (ACCELERATOR STREQUAL "cuda")
    # cpu file if cuda is enabled
    file(GLOB TMP "src/cuda/*.cpp")
    list(APPEND SOURCE_FILES ${TMP})
    file(GLOB TMP "src/cuda/*.cu")
    list(APPEND SOURCE_FILES ${TMP})
    file(GLOB TMP "include/CombBLAS/*.cuh")
    list(APPEND SOURCE_FILES ${TMP})
    # This needs to be split based on if CMake >= 3.17, as this is deprecated above that
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 80) # For A100
    enable_language(CUDA)
    # Main CombBLAS library
    add_library(CombBLAS ${SOURCE_FILES})
    string(APPEND CMAKE_CUDA_FLAGS
            " -Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr ")
    # string(APPEND CMAKE_CUDA_FLAGS  " -DUSE_CUDA ")

    if (GPU_COMM STREQUAL "mpicuda")
        message(STATUS "GPU_COMM is set to mpicuda")
        string(APPEND CMAKE_CUDA_FLAGS " -DGCOMM_DEFAULT ")
        target_compile_definitions(CombBLAS PUBLIC -DGCOMM_DEFAULT)
    elseif (GPU_COMM STREQUAL "default")
        message(STATUS "GPU_COMM is set to default")
        string(APPEND CMAKE_CUDA_FLAGS " -DGCOMM_MPICUDA ")
        target_compile_definitions(CombBLAS PUBLIC -DGCOMM_MPICUDA)
    else ()
        message(STATUS "GPU_COMM is set to nccl")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(NCCL REQUIRED nccl)
        string(APPEND CMAKE_CUDA_FLAGS " -DGCOMM_NCCL ")
        target_compile_definitions(CombBLAS PUBLIC -DGCOMM_NCCL)
        message(STATUS "NCCL FOLDER: ${NCCL_INCLUDE_DIRS}")
        target_include_directories(CombBLAS PUBLIC ${NCCL_INCLUDE_DIRS})
        target_link_libraries(CombBLAS PUBLIC ${NCCL_LIBRARIES})
    endif ()

    target_compile_definitions(CombBLAS PUBLIC -DUSE_CUDA)
    # Add RMM
    CPMAddPackage(
            NAME RMM
            GITHUB_REPOSITORY rapidsai/rmm
            GIT_TAG branch-24.10
            SYSTEM OFF
    )
    target_link_libraries(CombBLAS PUBLIC rmm::rmm)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
    set_property(
            TARGET Threads::Threads
            PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread>
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")

elseif (ACCELERATOR STREQUAL "hip")
    # hip is not implemented
else ()
    # Main CombBLAS library
    add_library(CombBLAS ${SOURCE_FILES})
endif ()


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

if (TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
    target_link_libraries(CombBLAS PUBLIC MPI::MPI_CXX)
else ()
    target_compile_options(CombBLAS PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
    target_link_libraries(CombBLAS PUBLIC "${MPI_CXX_LIBRARIES}"
            "${MPI_CXX_LINKFLAGS}")
    target_include_directories(CombBLAS PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif ()
# find OpenMP
find_package(OpenMP)
if(TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
  target_compile_definitions(CombBLAS PUBLIC THREADED)
  target_link_libraries(CombBLAS PUBLIC OpenMP::OpenMP_CXX)
elseif(OPENMP_FOUND)
  target_compile_definitions(CombBLAS PUBLIC THREADED)
  target_compile_options(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
endif()


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

if (ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${ASAN_FLAGS}")
endif ()

# set local include directories
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psort-1.0/include> $<INSTALL_INTERFACE:include>)


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
