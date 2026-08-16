#!/bin/bash

# Canny Edge Detector — CUDA Build Script (PROFILE/RELEASE)
# Run from project root: ./build_profile.sh
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
#   SOBEL_OPT=5     split Gx/Gy bucket version
#
#   NMS_OPT=0       naive         (default)
#   NMS_OPT=1       shared memory
#   NMS_OPT=2       pinned memory
#
#   HYSTERESIS_OPT=0       naive         (default)
#   HYSTERESIS_OPT=1       shared memory
#   HYSTERESIS_OPT=2       shared memory + pinned memory
#
#   WARMUP=1        warm-up run enabled   (default)
#   WARMUP=0        warm-up run disabled
#
#   BENCHMARK_RUNS=n  number of measured runs (default: 1)

set -e

BIN_DIR="bin"
BINARY="$BIN_DIR/canny_profile.out"

GAUSSIAN_OPT="${GAUSSIAN_OPT:-2}"
SOBEL_OPT="${SOBEL_OPT:-1}"
NMS_OPT="${NMS_OPT:-1}"
WARMUP="${WARMUP:-1}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-1}"

ARCH="sm_75"

mkdir -p "$BIN_DIR"

echo "[profile] Compiling with GAUSSIAN_OPT=$GAUSSIAN_OPT, SOBEL_OPT=$SOBEL_OPT, NMS_OPT=$NMS_OPT, WARMUP=$WARMUP, BENCHMARK_RUNS=$BENCHMARK_RUNS ..."

WARMUP_FLAG=""
if [ "$WARMUP" -eq 1 ]; then
  WARMUP_FLAG="-DWARMUP"
fi

COMMON_FLAGS="-rdc=true -O3 -std=c++20 -arch=$ARCH -lineinfo"
DEFINES="-DGAUSSIAN_OPT=$GAUSSIAN_OPT -DSOBEL_OPT=$SOBEL_OPT -DNMS_OPT=$NMS_OPT -DBENCHMARK_RUNS=$BENCHMARK_RUNS $WARMUP_FLAG"

nvcc $COMMON_FLAGS $DEFINES \
  -c -o "$BIN_DIR/main.profile.o" src/main.cu

nvcc $COMMON_FLAGS -DGAUSSIAN_OPT=$GAUSSIAN_OPT \
  -c -o "$BIN_DIR/gaussian.profile.o" src/gaussian.cu

nvcc $COMMON_FLAGS -DSOBEL_OPT=$SOBEL_OPT -DNMS_OPT=$NMS_OPT \
  -c -o "$BIN_DIR/sobel.profile.o" src/sobel.cu

nvcc $COMMON_FLAGS -DSOBEL_OPT=$SOBEL_OPT -DNMS_OPT=$NMS_OPT \
  -c -o "$BIN_DIR/nms.profile.o" src/nms.cu

nvcc -dlink -arch=$ARCH \
  -o "$BIN_DIR/device_link.profile.o" \
  "$BIN_DIR/main.profile.o" \
  "$BIN_DIR/gaussian.profile.o" \
  "$BIN_DIR/sobel.profile.o" \
  "$BIN_DIR/nms.profile.o"

nvcc -arch=$ARCH -o "$BINARY" \
  "$BIN_DIR/main.profile.o" \
  "$BIN_DIR/gaussian.profile.o" \
  "$BIN_DIR/sobel.profile.o" \
  "$BIN_DIR/nms.profile.o" \
  "$BIN_DIR/device_link.profile.o"

echo "Done. Run with:"
echo "./$BINARY <input.jpg> <output.png>"
