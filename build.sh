#!/bin/bash

# Canny Edge Detector — CUDA Build Script
# Run from project root: ./build.sh
#
# Options:
#   GAUSSIAN_OPT=0  naive         (default)
#   GAUSSIAN_OPT=1  shared memory
#   GAUSSIAN_OPT=2  pinned memory
#   GAUSSIAN_OPT=3  warp level
#   SOBEL_OPT=0     naive         (default)
#   SOBEL_OPT=1     optimized

BIN_DIR="bin"
BINARY="$BIN_DIR/canny.out"
GAUSSIAN_OPT="${GAUSSIAN_OPT:-0}"
SOBEL_OPT="${SOBEL_OPT:-0}"
ARCH="sm_75"

mkdir -p "$BIN_DIR"

echo "Compiling with GAUSSIAN_OPT=$GAUSSIAN_OPT and SOBEL_OPT=$SOBEL_OPT..."

# compile each translation unit separately
nvcc -rdc=true -O2 -std=c++20 -arch=$ARCH \
  -g -G -lineinfo \
  -DGAUSSIAN_OPT=$GAUSSIAN_OPT \
  -DSOBEL_OPT=$SOBEL_OPT \
  -c -o "$BIN_DIR/main.o" src/main.cu

if [ $? -ne 0 ]; then echo "Failed: main.cu"; exit 1; fi

nvcc -rdc=true -O2 -std=c++20 -arch=$ARCH \
  -g -G -lineinfo \
  -DGAUSSIAN_OPT=$GAUSSIAN_OPT \
  -DSOBEL_OPT=$SOBEL_OPT \
  -c -o "$BIN_DIR/gaussian.o" src/gaussian.cu

if [ $? -ne 0 ]; then echo "Failed: gaussian.cu"; exit 1; fi

nvcc -rdc=true -O2 -std=c++20 -arch=$ARCH \
  -g -G -lineinfo \
  -DGAUSSIAN_OPT=$GAUSSIAN_OPT \
  -DSOBEL_OPT=$SOBEL_OPT \
  -c -o "$BIN_DIR/sobel.o" src/sobel.cu

if [ $? -ne 0 ]; then echo "Failed: sobel.cu"; exit 1; fi

# device link step (required for rdc=true)
nvcc -dlink -arch=$ARCH \
  -o "$BIN_DIR/device_link.o" \
  "$BIN_DIR/main.o" "$BIN_DIR/gaussian.o" "$BIN_DIR/sobel.o"

if [ $? -ne 0 ]; then echo "Failed: device link"; exit 1; fi

# final host link
nvcc -o "$BINARY" \
  "$BIN_DIR/main.o" \
  "$BIN_DIR/gaussian.o" \
  "$BIN_DIR/device_link.o" \
  "$BIN_DIR/sobel.o"

if [ $? -eq 0 ]; then
  echo "Done. Run with: ./$BINARY <input.jpg> <output.png>"
else
  echo "Compilation failed."
  exit 1
fi
