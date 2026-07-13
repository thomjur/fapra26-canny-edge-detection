#pragma once

/// Wraps a CUDA API call and aborts on failure.
/// Prints error string, file, and line to stderr before calling
/// exit(EXIT_FAILURE).
#define CUDA_THROW_IF_FAILED(expr)                                             \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s\n at %s:%d\n", cudaGetErrorString(err),  \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/// Block size for all CUDA kernels (16x16 = 256 threads per block).
#define BLOCK_SIZE 16
/// We use a special block size for sobel filter to make use of coalescened
/// memory
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
