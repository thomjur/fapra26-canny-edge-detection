#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include "common.cuh"
#include "sobel.cuh"

SobelResult sobel_execute(const uint8_t *host_src, int32_t width,
                          int32_t height) {
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim(
      static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
      static_cast<uint32_t>(
          std::ceil(static_cast<float>(height) / BLOCK_SIZE)));
  const size_t img_size = width * height * sizeof(uint8_t);
  const size_t dir_size = width * height * sizeof(float);
  // device buffers
  uint8_t *device_src;      // Buffer for Gaussian-preprocessed image
  uint8_t *device_gradient; // Buffer for gradient value image
  float *device_direction;  // Buffer for grad direction values
  auto *host_gradient = static_cast<uint8_t *>(malloc(img_size));
  auto *host_direction = static_cast<float *>(malloc(dir_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_src, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_gradient, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_direction, dir_size));
  CUDA_THROW_IF_FAILED(cudaMemset(device_gradient, 0, img_size));

  // timing events
  cudaEvent_t t0, t1, t2, t3;
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t0));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t1));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t2));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t3));

  // H->D
  cudaEventRecord(t0);
  // Copy image H -> D
  CUDA_THROW_IF_FAILED(
      cudaMemcpy(device_src, host_src, img_size, cudaMemcpyHostToDevice));
  cudaEventRecord(t1);
  // Running Kernel
  naive_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);
  cudaEventRecord(t2);
  // Copy image D -> H
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_gradient, device_gradient, img_size,
                                  cudaMemcpyDeviceToHost));
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_direction, device_direction, img_size,
                                  cudaMemcpyDeviceToHost));
  cudaEventRecord(t3);
  CUDA_THROW_IF_FAILED(cudaEventSynchronize(t3));

  // collect timings
  SobelResult result{};
  result.host_grad = host_gradient;
  result.host_dir = host_direction;
  cudaEventElapsedTime(&result.ms_h2d, t0, t1);
  cudaEventElapsedTime(&result.ms_kernel, t1, t2);
  cudaEventElapsedTime(&result.ms_d2h, t2, t3);

  // clean up device resources
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaEventDestroy(t2);
  cudaEventDestroy(t3);
  CUDA_THROW_IF_FAILED(cudaFree(device_src));
  CUDA_THROW_IF_FAILED(cudaFree(device_gradient));
  CUDA_THROW_IF_FAILED(cudaFree(device_direction));

  return result;
}

void sobel_cleanup(SobelResult &result) {
  free(result.host_dir);
  free(result.host_grad);
  result.host_dir = nullptr;
  result.host_grad = nullptr;
}

// Kernel for Sobel edge detection
__global__ void naive_sobel_filter(const uint8_t *src_buffer,
                                   const int32_t width, const int32_t height,
                                   uint8_t *out_grad_buffer,
                                   float *out_dir_buffer) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the thread is within the image boundaries
  // and if the 3x3 kernel fits around the pixel
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    // Sobel kernels
    int32_t sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int32_t sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int32_t gx = 0, gy = 0;

    // Apply the Sobel kernels
    for (int8_t i = -1; i <= 1; i++) {
      for (int8_t j = -1; j <= 1; j++) {
        uint8_t pixel = src_buffer[(y + i) * width + (x + j)];
        gx += pixel * sobelX[i + 1][j + 1];
        gy += pixel * sobelY[i + 1][j + 1];
      }
    }

    // Calculate the gradient magnitude
    float gradient = sqrtf(gx * gx + gy * gy);

    // Clamp the value to 0-255
    gradient = min(255.0f, max(0.0f, gradient));

    // Write the gradient result
    out_grad_buffer[y * width + x] = static_cast<uint8_t>(gradient);

    // Calculate the
    out_dir_buffer[y * width + x] =
        atan2f(static_cast<float>(gy), static_cast<float>(gx));
  }
}
