add_executable(ArborX_Example_Plasma.exe example_plasma.cpp)
target_link_libraries(ArborX_Example_Plasma.exe ArborX::ArborX)
add_test(NAME ArborX_Example_Plasma COMMAND ./ArborX_Example_Plasma.exe)
