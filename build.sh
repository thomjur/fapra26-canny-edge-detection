#!/bin/bash

# Canny Edge Detector — CUDA Build Script
# Run from project root: ./build.sh

SOURCE="src/main.cu"
BIN_DIR="bin"
BINARY="$BIN_DIR/canny.out"

# Create bin/ directory if it doesn't exist yet
mkdir -p "$BIN_DIR"

echo "Compiling $SOURCE -> $BINARY ..."

nvcc \
  -O2 \
  -std=c++20 \
  -g -G \
  -lineinfo \
  -o "$BINARY" \
  "$SOURCE"

if [ $? -eq 0 ]; then
  echo "Done. Run with: ./$BINARY <image.png>"
else
  echo "Compilation failed."
  exit 1
fi