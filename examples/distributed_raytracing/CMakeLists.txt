add_executable(ArborX_Example_Distributed_RayTracing.exe example_distributed_raytracing.cpp)
target_link_libraries(ArborX_Example_Distributed_RayTracing.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_Example_Distributed_RayTracing COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_Example_Distributed_RayTracing.exe --nx=${MPIEXEC_MAX_NUMPROCS} --ny=1 --nz=1 --lx=1 --ly=100000 --lz=100000 --nx_mpi=${MPIEXEC_MAX_NUMPROCS} --ny_mpi=1 --nz_mpi=1 --rays=1000 ${MPIEXEC_POSTFLAGS})

