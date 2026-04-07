cmake_minimum_required(VERSION 3.13)
project(MacroAnalyzer)

set(CMAKE_PREFIX_PATH "/home/ubuntu/c_parser/llvm-custom")
set(LLVM_DIR "/home/ubuntu/c_parser/llvm-custom/lib/cmake/llvm")
set(Clang_DIR "/home/ubuntu/c_parser/llvm-custom/lib/cmake/clang")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_executable(analyzer include_finder.cpp)

llvm_map_components_to_libnames(llvm_libs support core mc option)

target_link_libraries(analyzer
  clangTooling
  clangFrontend
  clangParse
  clangSema
  clangAnalysis
  clangAST
  clangBasic
  clangLex
  clangSerialization
  clangAPINotes
  clangEdit
  clangIndex        # ← これを追加！
  clangRewrite
  clangDriver
  ${llvm_libs}
)