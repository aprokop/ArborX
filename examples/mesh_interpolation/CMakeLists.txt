# More components are needed to read the mesh file and write the output file.
list(APPEND ARBORX_TRILINOS_COMPONENT_LIST "STKIO;STKMesh;SEACASExodus")
foreach(component ${ARBORX_TRILINOS_COMPONENT_LIST})
  target_link_libraries(ArborX INTERFACE ${component}::all_libs)
endforeach()

set(ARBORX_BENCHMARK_UTILS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/benchmarks/utils)

add_executable(ArborX_Example_Mesh_Interpolation.exe example_mesh_interpolation.cpp helpers.cpp)
target_include_directories(ArborX_Example_Mesh_Interpolation.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ARBORX_BENCHMARK_UTILS_INCLUDE_DIR})
target_link_libraries(ArborX_Example_Mesh_Interpolation.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_Example_Mesh_Interpolation COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Example_Mesh_Interpolation.exe> ${MPIEXEC_POSTFLAGS} --help)
set_tests_properties(ArborX_Example_Mesh_Interpolation PROPERTIES ENVIRONMENT "IOSS_PROPERTIES=DECOMPOSITION_METHOD=RIB")
