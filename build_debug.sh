#!/bin/bash

# Canny Edge Detector — CUDA Build Script (DEBUG)
# Run from project root: ./build_debug.sh
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
#   NMS_OPT=2       pinned memory
#
#   HYSTERESIS_OPT=0       naive         (default)
#   HYSTERESIS_OPT=1       shared memory
#   HYSTERESIS_OPT=2       shared memory + pinned memory
#
#   WARMUP=1        warm-up run enabled   (default)
#   WARMUP=0        warm-up run disabled
#
#   PIPELINE_FUSED=1  fused pipeline enabled (default)
#   PIPELINE_FUSED=0  fused pipeline disabled

BIN_DIR="bin"
BINARY="$BIN_DIR/canny_debug.out"
GAUSSIAN_OPT="${GAUSSIAN_OPT:-2}"
SOBEL_OPT="${SOBEL_OPT:-1}"
NMS_OPT="${NMS_OPT:-1}"
HYSTERESIS_OPT="${HYSTERESIS_OPT:-0}"
WARMUP="${WARMUP:-1}"
PIPELINE_FUSED="${PIPELINE_FUSED:-1}"
ARCH="sm_75"

# fused end-to-end pipeline benchmark (single H2D/D2H, no host sync between
# stages) -- must be defined for EVERY translation unit, since each .cu file
# is preprocessed independently and doesn't see main.cu's defines
FUSED_FLAG=""
if [ "$PIPELINE_FUSED" -eq 1 ]; then
  FUSED_FLAG="-DPIPELINE_FUSED"
fi

mkdir -p "$BIN_DIR"

echo "[debug] Compiling with GAUSSIAN_OPT=$GAUSSIAN_OPT, SOBEL_OPT=$SOBEL_OPT, NMS_OPT=$NMS_OPT, HYSTERESIS_OPT=$HYSTERESIS_OPT, WARMUP=$WARMUP, PIPELINE_FUSED=$PIPELINE_FUSED ..."

WARMUP_FLAG=""
if [ "$WARMUP" -eq 1 ]; then
  WARMUP_FLAG="-DWARMUP"
fi

# compile each translation unit separately
nvcc -rdc=true -O0 -std=c++20 -arch=$ARCH \
  -g -G --generate-line-info \
  -DGAUSSIAN_OPT=$GAUSSIAN_OPT -DSOBEL_OPT=$SOBEL_OPT -DNMS_OPT=$NMS_OPT $WARMUP_FLAG $FUSED_FLAG \
  -c -o "$BIN_DIR/main.o" src/main.cu

if [ $? -ne 0 ]; then echo "Failed: main.cu"; exit 1; fi

nvcc -rdc=true -O0 -std=c++20 -arch=$ARCH \
  -g -G --generate-line-info \
  -DGAUSSIAN_OPT=$GAUSSIAN_OPT $FUSED_FLAG \
  -c -o "$BIN_DIR/gaussian.o" src/gaussian.cu

if [ $? -ne 0 ]; then echo "Failed: gaussian.cu"; exit 1; fi

nvcc -rdc=true -O0 -std=c++20 -arch=$ARCH \
  -g -G --generate-line-info \
  -DSOBEL_OPT=$SOBEL_OPT -DNMS_OPT=$NMS_OPT $FUSED_FLAG \
  -c -o "$BIN_DIR/sobel.o" src/sobel.cu

if [ $? -ne 0 ]; then echo "Failed: sobel.cu"; exit 1; fi

nvcc -rdc=true -O0 -std=c++20 -arch=$ARCH \
  -g -G --generate-line-info \
  -DSOBEL_OPT=$SOBEL_OPT -DNMS_OPT=$NMS_OPT $FUSED_FLAG \
  -c -o "$BIN_DIR/nms.o" src/nms.cu

if [ $? -ne 0 ]; then echo "Failed: nms.cu"; exit 1; fi

nvcc -rdc=true -O0 -std=c++20 -arch=$ARCH \
  -g -G --generate-line-info \
  -DHYSTERESIS_OPT=$HYSTERESIS_OPT $FUSED_FLAG \
  -c -o "$BIN_DIR/hysteresis.o" src/hysteresis.cu

if [ $? -ne 0 ]; then echo "Failed: hysteresis.cu"; exit 1; fi

# device link step (required for rdc=true)
nvcc -dlink -arch=$ARCH \
  -o "$BIN_DIR/device_link.o" \
  "$BIN_DIR/main.o" "$BIN_DIR/gaussian.o" "$BIN_DIR/sobel.o" "$BIN_DIR/nms.o" "$BIN_DIR/hysteresis.o"

if [ $? -ne 0 ]; then echo "Failed: device link"; exit 1; fi

# final host link
nvcc -arch=$ARCH -o "$BINARY" \
  "$BIN_DIR/main.o" \
  "$BIN_DIR/gaussian.o" \
  "$BIN_DIR/sobel.o" \
  "$BIN_DIR/nms.o" \
  "$BIN_DIR/hysteresis.o" \
  "$BIN_DIR/device_link.o"

if [ $? -eq 0 ]; then
  echo "Done. Run with: ./$BINARY <input.jpg> <output.png>"
else
  echo "Compilation failed."
  exit 1
fi
