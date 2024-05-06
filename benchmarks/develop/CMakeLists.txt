set(ARBORX_BENCHMARK_UTILS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/benchmarks/utils)

add_executable(ArborX_Benchmark_Develop.exe develop.cpp)
target_link_libraries(ArborX_Benchmark_Develop.exe ArborX::ArborX benchmark::benchmark)
target_include_directories(ArborX_Benchmark_Develop.exe PRIVATE ${ARBORX_BENCHMARK_UTILS_INCLUDE_DIR})
add_test(NAME ArborX_Benchmark_Develop COMMAND ArborX_Benchmark_Develop.exe)
