#!/bin/bash

# ./download_clang.sh

cd parsers/usage_macro_ref_analyzer && ./build.sh
cd ../..

cd usage_analyzer && ./build.sh
cd ..

cd include_finder && ./build.sh
cd ..