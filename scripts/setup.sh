#!/bin/zsh

SCRIPT_DIR="${0:A:h}"

if [[ -z "$SCRIPT_DIR" ]]; then
    SCRIPT_DIR="."
fi

ROOT_DIR="$SCRIPT_DIR/.."

mkdir -p $ROOT_DIR/build
cd $ROOT_DIR/build

# # Option 2: Manual installation
# git clone https://github.com/google/clspv.git
# cd clspv
# python3 utils/fetch_sources.py
# mkdir -p build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc) clspv
# cd ../../


# rm -r *

cmake -G Xcode ../
# cmake ../