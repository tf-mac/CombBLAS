find_package (Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
pybind11_add_module(pycombblaslib MODULE pycombblas/src/wrapper.cpp)
target_link_libraries(pycombblaslib PUBLIC CombBLAS)
target_include_directories(pycombblaslib PUBLIC ${CMAKE_SOURCE_DIR}/Applications)

# copy back into package folder
add_custom_command(TARGET pycombblaslib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pycombblaslib>
        ${CMAKE_CURRENT_SOURCE_DIR}/pycombblas/pycombblas)

#pybind11_add_module(pymetis_internal MODULE pymetis/src/wrapper.cpp)
#target_link_libraries(pymetis_internal PUBLIC metis parmetis)
## copy back into package folder
#add_custom_command(TARGET pymetis_internal POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy
#        $<TARGET_FILE:pymetis_internal>
#        ${CMAKE_CURRENT_SOURCE_DIR}/pymetis/pymetis)