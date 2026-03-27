find_package(Trilinos)

add_executable(ArborX_Example_Wall_Distance.exe example_wall_distance.cpp)
target_link_libraries(ArborX_Example_Wall_Distance.exe ArborX::ArborX STK::all_libs PanzerAdaptersSTK::all_libs)
add_test(NAME ArborX_Example_Wall_Distance COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ArborX_Example_Wall_Distance.exe> ${MPIEXEC_POSTFLAGS})
