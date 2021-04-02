add_executable(ArborX_npcf.exe npcf.cpp)
target_include_directories(ArborX_npcf.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ArborX_npcf.exe ArborX::ArborX Boost::program_options)
