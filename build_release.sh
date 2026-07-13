#!/bin/bash

# Canny Edge Detector — CUDA Build Script (RELEASE)
# Run from project root: ./build_release.sh
#
# Options:
#   GAUSSIAN_OPT=0  naive         (default)
#   GAUSSIAN_OPT=1  shared memory
#   GAUSSIAN_OPT=2  pinned memory
#
#   SOBEL_OPT=0     naive         (default)
#   SOBEL_OPT=1     shared memory
#   SOBEL_OPT=2     optimized version
#   SOBEL_OPT=3     optimized version + pinned memory
#   SOBEL_OPT=4     optimized bucket version
#
#   NMS_OPT=0       naive         (default)
#   NMS_OPT=1       shared memory
#
#   WARMUP=1        warm-up run enabled   (default)
#   WARMUP=0        warm-up run disabled

BIN_DIR="bin"
BINARY="$BIN_DIR/canny_release.out"
GAUSSIAN_OPT="${GAUSSIAN_OPT:-2}"
SOBEL_OPT="${SOBEL_OPT:-3}"
NMS_OPT="${NMS_OPT:-1}"
WARMUP="${WARMUP:-0}"
ARCH="sm_75"

mkdir -p "$BIN_DIR"

echo "[release] Compiling with GAUSSIAN_OPT=$GAUSSIAN_OPT, SOBEL_OPT=$SOBEL_OPT, NMS_OPT=$NMS_OPT, WARMUP=$WARMUP ..."

WARMUP_FLAG=""
if [ "$WARMUP" -eq 1 ]; then
  WARMUP_FLAG="-DWARMUP"
fi

# compile each translation unit separately
nvcc -rdc=true -O3 --use_fast_math -std=c++20 -arch=$ARCH \
  --generate-line-info \
  -DGAUSSIAN_OPT=$GAUSSIAN_OPT -DSOBEL_OPT=$SOBEL_OPT -DNMS_OPT=$NMS_OPT $WARMUP_FLAG \
  -c -o "$BIN_DIR/main.o" src/main.cu

if [ $? -ne 0 ]; then echo "Failed: main.cu"; exit 1; fi

nvcc -rdc=true -O3 --use_fast_math -std=c++20 -arch=$ARCH \
  --generate-line-info \
  -DGAUSSIAN_OPT=$GAUSSIAN_OPT \
  -c -o "$BIN_DIR/gaussian.o" src/gaussian.cu

if [ $? -ne 0 ]; then echo "Failed: gaussian.cu"; exit 1; fi

nvcc -rdc=true -O3 --use_fast_math -std=c++20 -arch=$ARCH \
  --generate-line-info \
  -DSOBEL_OPT=$SOBEL_OPT \
  -c -o "$BIN_DIR/sobel.o" src/sobel.cu

if [ $? -ne 0 ]; then echo "Failed: sobel.cu"; exit 1; fi

nvcc -rdc=true -O3 --use_fast_math -std=c++20 -arch=$ARCH \
  --generate-line-info \
  -DNMS_OPT=$NMS_OPT \
  -c -o "$BIN_DIR/nms.o" src/nms.cu

if [ $? -ne 0 ]; then echo "Failed: nms.cu"; exit 1; fi

# device link step (required for rdc=true)
nvcc -dlink -arch=$ARCH \
  -o "$BIN_DIR/device_link.o" \
  "$BIN_DIR/main.o" "$BIN_DIR/gaussian.o" "$BIN_DIR/sobel.o" "$BIN_DIR/nms.o"

if [ $? -ne 0 ]; then echo "Failed: device link"; exit 1; fi

# final host link
nvcc -arch=$ARCH -o "$BINARY" \
  "$BIN_DIR/main.o" \
  "$BIN_DIR/gaussian.o" \
  "$BIN_DIR/sobel.o" \
  "$BIN_DIR/nms.o" \
  "$BIN_DIR/device_link.o"

if [ $? -eq 0 ]; then
  echo "Done. Run with: ./$BINARY <input.jpg> <output.png>"
else
  echo "Compilation failed."
  exit 1
fi
