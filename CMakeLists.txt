cmake_minimum_required(VERSION 3.17)

project(HydraChess VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB HydraChess_SRC
    "src/*.cpp"
    "src/*.hpp")

add_subdirectory(src/libchess)

add_executable(HydraChess ${HydraChess_SRC})

target_link_libraries(HydraChess PUBLIC libchess)

find_package(Torch REQUIRED)
target_link_libraries(HydraChess PUBLIC ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET HydraChess
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:HydraChess>)
endif (MSVC)