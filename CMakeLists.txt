cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: Debug Release."
	FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2 -Wextra -Wno-unused-parameter -Wno-unused-but-set-parameter -fsanitize=address -Wno-odr")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native -Wno-odr")

include_directories(./)

project(imgui_sfml)
add_library(imgui_sfml SHARED imgui.cpp imgui_draw.cpp imgui-SFML.cpp)
target_link_libraries(imgui_sfml GL sfml-audio sfml-window sfml-graphics sfml-system nupic_core)

project(HTMPath)
add_executable(HTMPath main.cpp)
target_link_libraries(HTMPath imgui_sfml)

project(bench)
add_executable(bench bench.cpp)
target_link_libraries(bench nupic_core)