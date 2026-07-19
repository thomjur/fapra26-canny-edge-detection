#include "common.cuh"
#include "hysteresis.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <math_constants.h>

HysteresisResult hysteresis_execute(const uint8_t *host_buffer,
                                    const int32_t width, const int32_t height) {
  const size_t pixel_count =
      static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t img_size = pixel_count * sizeof(uint8_t);

  // Define lower and upper thresholds for the hysteresis algorithm
#ifdef HIGH_THRESHOLD
  uint8_t high_threshold = HIGH_THRESHOLD;
#else
  uint8_t high_threshold = 150;
#endif
#ifdef LOW_THRESHOLD
  uint8_t low_threshold = LOW_THRESHOLD;
#else
  uint8_t low_threshold = 50;
#endif

  // Grid / block layout
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim(
      static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
      static_cast<uint32_t>(
          std::ceil(static_cast<float>(height) / BLOCK_SIZE)));

  // Host output buffer
  uint8_t *host_dst = nullptr;

  // Input buffers are already page-locked by the NMS stage
  // (NMS_HOST_PINNED), so no cudaHostRegister() is needed here.
#ifdef HYSTERESIS_PINNED
  CUDA_THROW_IF_FAILED(cudaMallocHost(&host_dst, img_size));
#else
  host_dst = static_cast<uint8_t *>(malloc(img_size));
#endif

  // Device buffers
  uint8_t *device_buffer = nullptr;
  uint8_t *device_dst = nullptr;
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_buffer, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_dst, img_size));
  CUDA_THROW_IF_FAILED(cudaMemset(device_dst, 0, img_size));

  // Timing events
  cudaEvent_t t0, t1, t2, t3;
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t0));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t1));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t2));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t3));

  // H->D transfer
  cudaEventRecord(t0);
  CUDA_THROW_IF_FAILED(
      cudaMemcpy(device_buffer, host_buffer, img_size, cudaMemcpyHostToDevice));
  cudaEventRecord(t1);

  // Kernel execution
#if defined(HYSTERESIS_NAIVE)
  naive_hysteresis_linker<<<grid_dim, block_dim>>>(
      device_buffer, width, height, device_dst, low_threshold, high_threshold);
#endif
  CUDA_THROW_IF_FAILED(cudaGetLastError());
  cudaEventRecord(t2);

  // D->H transfer
  CUDA_THROW_IF_FAILED(
      cudaMemcpy(host_dst, device_dst, img_size, cudaMemcpyDeviceToHost));
  cudaEventRecord(t3);
  CUDA_THROW_IF_FAILED(cudaEventSynchronize(t3));

  // Collect timing results
  HysteresisResult result{};
  result.hysteresis_buffer = host_dst;
  cudaEventElapsedTime(&result.ms_h2d, t0, t1);
  cudaEventElapsedTime(&result.ms_kernel, t1, t2);
  cudaEventElapsedTime(&result.ms_d2h, t2, t3);

  // Clean up device resources
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaEventDestroy(t2);
  cudaEventDestroy(t3);
  CUDA_THROW_IF_FAILED(cudaFree(device_buffer));
  CUDA_THROW_IF_FAILED(cudaFree(device_dst));

  return result;
}

void hysteresis_cleanup(HysteresisResult &result) {
  if (result.hysteresis_buffer == nullptr)
    return;

#ifdef HYSTERESIS_PINNED
  CUDA_THROW_IF_FAILED(cudaFreeHost(result.hysteresis_buffer));
#else
  free(result.hysteresis_buffer);
#endif
  result.hysteresis_buffer = nullptr;
}

/**
 * @brief CUDA kernel for the hysteresis step to link strong and weak edges.
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_buffer Output buffer for final image with linked edges (device
 * memory, 8-bit).
 * @param low_threshold Lower threshold value for hysteresis.
 * @param high_threshold Higher threshold value for hysteresis.
 */
__global__ void
naive_hysteresis_linker(const uint8_t *src_buffer, const int32_t width,
                        const int32_t height, uint8_t *out_buffer,
                        uint8_t low_threshold, uint8_t high_threshold) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int index = y * width + x;
  float pixel = src_buffer[index];

  // Accept strong edges immediately
  if (pixel >= high_threshold) {
    out_buffer[index] = 255;
  }
  // Check weak edges: accept only if connected to strong edges
  else if (pixel >= low_threshold) {
    // Check 8-neighborhood for strong edges
    bool connected = false;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          int nindex = ny * width + nx;
          if (src_buffer[nindex] >= high_threshold) {
            connected = true;
            break;
          }
        }
      }
      if (connected)
        break;
    }
    out_buffer[index] = connected ? 255 : 0;
  }
  // Discard all other pixels
  else {
    out_buffer[index] = 0;
  }
}
