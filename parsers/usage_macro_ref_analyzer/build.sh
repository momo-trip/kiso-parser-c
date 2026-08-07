#!/bin/bash

rm -rf build
mkdir build && cd build
# cmake ..
# cmake -DCMAKE_PREFIX_PATH=~/c_parser/llvm-custom \
#       -DCMAKE_C_COMPILER=clang \
#       -DCMAKE_CXX_COMPILER=clang++ \

cmake -DCMAKE_PREFIX_PATH=~/c_parser/llvm-custom ..
make
cd ..
