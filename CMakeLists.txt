cmake_minimum_required(VERSION 3.17)

project(HydraChess VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB HydraChess_SRC
    "*.cpp"
    "*.hpp")

add_subdirectory(libchess)

add_executable(HydraChess ${HydraChess_SRC})

target_link_libraries(HydraChess PUBLIC libchess)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(HydraChess PUBLIC OpenMP::OpenMP_CXX)
endif()