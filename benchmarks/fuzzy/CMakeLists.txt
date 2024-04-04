set(EXPLICIT_INSTANTIATION_SOURCE_FILES)

add_executable(ArborX_Benchmark_Fuzzy.exe fuzzy.cpp)
target_include_directories(ArborX_Benchmark_Fuzzy.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_Benchmark_Fuzzy.exe ArborX::ArborX Boost::program_options)

set(input_file "input.txt")
add_test(NAME ArborX_Benchmark_Fuzzy COMMAND ArborX_Benchmark_Fuzzy.exe --filename=${input_file} --eps=1.4 --verify)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${input_file} ${CMAKE_CURRENT_BINARY_DIR}/${input_file} COPYONLY)
