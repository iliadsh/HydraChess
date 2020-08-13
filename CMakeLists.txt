cmake_minimum_required(VERSION 3.17)

project(PegaZero VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB PegaZero_SRC
    "*.cpp"
    "*.hpp")

add_subdirectory(libchess)

add_executable(PegaZero ${PegaZero_SRC})

target_link_libraries(PegaZero PUBLIC libchess)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(PegaZero PUBLIC OpenMP::OpenMP_CXX)
endif()