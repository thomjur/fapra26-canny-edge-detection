#include <cmath> // For std::ceil
#include <cstdint>
#include <cstdio>
#include <cstdlib> // For malloc, free
#include <cuda.h>
#include <math_constants.h>

#include "common.cuh"
#include "sobel.cuh"

/**
 * @struct SobelResult
 * @brief Structure to hold the results of the Sobel edge detection.
 *
 * Contains pointers to the gradient and direction buffers on the host,
 * as well as timing information for H2D, kernel execution, and D2H transfers.
 */

/**
 * @brief Executes the Sobel edge detection algorithm on the GPU.
 *
 * This function allocates device memory, copies the input image from host to
 * device, launches the Sobel kernel, and copies the results back to the host.
 * It also measures the time taken for each major step (H2D, kernel, D2H).
 *
 * @param host_src Pointer to the host-side input image buffer (grayscale,
 * 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @return SobelResult Structure containing the gradient and direction buffers,
 *         and timing information in milliseconds.
 */
SobelResult sobel_execute(const uint8_t *host_src, int32_t width,
                          int32_t height) {
  // Define block and grid dimensions for the CUDA kernel
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim(
      static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
      static_cast<uint32_t>(
          std::ceil(static_cast<float>(height) / BLOCK_SIZE)));

  // Calculate buffer sizes
  const size_t img_size = width * height * sizeof(uint8_t);
  const size_t dir_size = width * height * sizeof(float);

  // Allocate device buffers
  uint8_t *device_src;      // Buffer for input image
  uint8_t *device_gradient; // Buffer for gradient magnitude image
  float *device_direction;  // Buffer for gradient direction values

  // Allocate host buffers for results
  auto *host_gradient = static_cast<uint8_t *>(malloc(img_size));
  auto *host_direction = static_cast<float *>(malloc(dir_size));

  // Allocate device memory
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_src, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_gradient, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_direction, dir_size));
  CUDA_THROW_IF_FAILED(cudaMemset(device_gradient, 0, img_size));

  // Create timing events
  cudaEvent_t t0, t1, t2, t3;
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t0));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t1));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t2));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t3));

  // --- Timed operations ---

  // Start H2D transfer timing
  cudaEventRecord(t0);
  // Copy input image from host to device
  CUDA_THROW_IF_FAILED(
      cudaMemcpy(device_src, host_src, img_size, cudaMemcpyHostToDevice));
  cudaEventRecord(t1);

  // Launch Sobel kernel
#ifdef SOBEL_NAIVE
  printf("Starting naive Sobel kernel...\n");
  naive_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);
#endif
#ifdef SOBEL_OPTIMIZED
  printf("Starting optimized Sobel kernel...\n");
  optimized_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);
#endif
  cudaEventRecord(t2);

  // Copy results from device to host
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_gradient, device_gradient, img_size,
                                  cudaMemcpyDeviceToHost));
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_direction, device_direction, dir_size,
                                  cudaMemcpyDeviceToHost));
  cudaEventRecord(t3);
  CUDA_THROW_IF_FAILED(cudaEventSynchronize(t3));

  // --- Collect timing results ---
  SobelResult result{};
  result.host_grad = host_gradient;
  result.host_dir = host_direction;
  cudaEventElapsedTime(&result.ms_h2d, t0, t1);
  cudaEventElapsedTime(&result.ms_kernel, t1, t2);
  cudaEventElapsedTime(&result.ms_d2h, t2, t3);

  // --- Clean up device resources ---
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaEventDestroy(t2);
  cudaEventDestroy(t3);
  CUDA_THROW_IF_FAILED(cudaFree(device_src));
  CUDA_THROW_IF_FAILED(cudaFree(device_gradient));
  CUDA_THROW_IF_FAILED(cudaFree(device_direction));

  return result;
}

/**
 * @brief Frees the host-side buffers allocated by sobel_execute.
 *
 * @param result Reference to the SobelResult structure containing the buffers
 * to free.
 */
void sobel_cleanup(SobelResult &result) {
  free(result.host_dir);
  free(result.host_grad);
  result.host_dir = nullptr;
  result.host_grad = nullptr;
}

/**
 * @brief CUDA kernel for Sobel edge detection.
 *
 * Each thread computes the Sobel gradient magnitude and direction for one
 * pixel. The Sobel operator uses two 3x3 convolution kernels (Gx and Gy) to
 * detect edges in the horizontal and vertical directions, respectively. The
 * gradient magnitude is calculated as sqrt(Gx^2 + Gy^2) and clamped to [0,
 * 255]. The gradient direction is calculated using atan2(Gy, Gx).
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_grad_buffer Output buffer for gradient magnitudes (device memory,
 * 8-bit).
 * @param out_dir_buffer Output buffer for gradient directions (device memory,
 * float).
 */
__global__ void naive_sobel_filter(const uint8_t *src_buffer,
                                   const int32_t width, const int32_t height,
                                   uint8_t *out_grad_buffer,
                                   float *out_dir_buffer) {
  // Calculate pixel coordinates for this thread
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the thread is within the image boundaries
  // and if the 3x3 kernel fits around the pixel
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    // Sobel kernels for horizontal (Gx) and vertical (Gy) edge detection
    int32_t sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int32_t sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int32_t gx = 0, gy = 0;

    // Apply the Sobel kernels: convolve the 3x3 neighborhood with Gx and Gy
    for (int8_t i = -1; i <= 1; i++) {
      for (int8_t j = -1; j <= 1; j++) {
        uint8_t pixel = src_buffer[(y + i) * width + (x + j)];
        gx += pixel * sobelX[i + 1][j + 1];
        gy += pixel * sobelY[i + 1][j + 1];
      }
    }

    // Calculate the gradient magnitude: sqrt(Gx^2 + Gy^2)
    float gradient = sqrtf(gx * gx + gy * gy);

    // Clamp the gradient magnitude to the valid 8-bit range [0, 255]
    gradient = fminf(255.0f, fmaxf(0.0f, gradient));

    // Write the gradient magnitude to the output buffer
    out_grad_buffer[y * width + x] = static_cast<uint8_t>(gradient);

    // Calculate and write the gradient direction (in radians) using atan2
    out_dir_buffer[y * width + x] =
        atan2f(static_cast<float>(gy), static_cast<float>(gx));
  }
}

/**
 * @brief Optimized CUDA kernel for Sobel edge detection.
 *
 * Each thread computes the Sobel gradient magnitude and direction for one
 * pixel. The Sobel operator uses two 3x3 convolution kernels (Gx and Gy) to
 * detect edges in the horizontal and vertical directions, respectively. The
 * gradient magnitude is calculated as sqrt(Gx^2 + Gy^2) and clamped to [0,
 * 255]. The gradient direction is calculated using atan2(Gy, Gx).
 *
 * As optimizations, we have:
 * 1) Implemented shared memory to avoid loading neighboring pixels from global
 * memory in each thread.
 * 2) Implemented a more simple version of the gradient
 * calculation (TODO).
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_grad_buffer Output buffer for gradient magnitudes (device memory,
 * 8-bit).
 * @param out_dir_buffer Output buffer for gradient directions (device memory,
 * float).
 */

__global__ void optimized_sobel_filter(const uint8_t *src_buffer,
                                       const int32_t width,
                                       const int32_t height,
                                       uint8_t *out_grad_buffer,
                                       float *out_dir_buffer) {
  // We use shared memory in this case
  // We must also allocate memory for the halo pixels
  __shared__ uint8_t
      shared_memory[BLOCK_SIZE * BLOCK_SIZE + 4 * BLOCK_SIZE + 4];

  // Calculate pixel coordinates for this thread
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t STRIDE = BLOCK_SIZE + 2;

  // We first define a helper function to check if a pixel is actually in image
  auto in_bounds = [&](int gx, int gy) {
    return (gx >= 0 && gx < width && gy >= 0 && gy < height);
  };

  // We load pixels from src into shared memory
  // First, we only load the current pixel into correct position in shared
  // memory; Note that we know that the halo radius is always 1 due to the 3x3
  // Sobel kernel
  shared_memory[(threadIdx.y + 1) * STRIDE + (threadIdx.x + 1)] =
      in_bounds(x, y) ? src_buffer[y * width + x] : 0;

  //// 1. Optimization: Loading pixel grid + halo into shared memory
  // Only edge pixels need to load halo into shared memory

  // We start by loading the neighboring cols and rows outside of the actual
  // grid
  // We are in upper row
  if (threadIdx.y == 0)
    shared_memory[0 * STRIDE + (threadIdx.x + 1)] =
        in_bounds(x, y - 1) ? src_buffer[(y - 1) * width + x] : 0;

  // We are in final row
  if (threadIdx.y == BLOCK_SIZE - 1)
    shared_memory[(BLOCK_SIZE + 1) * STRIDE + (threadIdx.x + 1)] =
        in_bounds(x, y + 1) ? src_buffer[(y + 1) * width + x] : 0;

  // We are in first column
  if (threadIdx.x == 0)
    shared_memory[(threadIdx.y + 1) * STRIDE + 0] =
        in_bounds(x - 1, y) ? src_buffer[y * width + x - 1] : 0;

  // We are in last column
  if (threadIdx.x == BLOCK_SIZE - 1)
    shared_memory[(threadIdx.y + 1) * STRIDE + (BLOCK_SIZE + 1)] =
        in_bounds(x + 1, y) ? src_buffer[y * width + x + 1] : 0;

  // Next, we load the four edge pixels
  if (threadIdx.x == 0 && threadIdx.y == 0)
    shared_memory[0 * STRIDE + 0] =
        in_bounds(x - 1, y - 1) ? src_buffer[(y - 1) * width + x - 1] : 0;

  // Loading upper right corner pixel
  if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0)
    shared_memory[0 * STRIDE + (BLOCK_SIZE + 1)] =
        in_bounds(x + 1, y - 1) ? src_buffer[(y - 1) * width + (x + 1)] : 0;

  // Bottom left corner pixel
  if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1)
    shared_memory[(BLOCK_SIZE + 1) * STRIDE + 0] =
        in_bounds(x - 1, y + 1) ? src_buffer[(y + 1) * width + (x - 1)] : 0;

  // Bottom right corner pixel
  if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1)
    shared_memory[(BLOCK_SIZE + 1) * STRIDE + (BLOCK_SIZE + 1)] =
        in_bounds(x + 1, y + 1) ? src_buffer[(y + 1) * width + (x + 1)] : 0;

  __syncthreads();

  // Check if the thread is within the image boundaries
  // and if the 3x3 kernel fits around the pixel
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    // Sobel kernels for horizontal (Gx) and vertical (Gy) edge detection
    int32_t sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int32_t sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int32_t gx = 0, gy = 0;

    // Apply the Sobel kernels: convolve the 3x3 neighborhood with Gx and Gy
    for (int8_t i = -1; i <= 1; i++) {
      for (int8_t j = -1; j <= 1; j++) {
        uint8_t pixel = shared_memory[(threadIdx.y + 1 + i) * STRIDE +
                                      (threadIdx.x + 1 + j)];
        gx += pixel * sobelX[i + 1][j + 1];
        gy += pixel * sobelY[i + 1][j + 1];
      }
    }

    //// 2. Optimization: Instead of using sqrtf, we use the optimized formel
    /// for / gradient calculations in XXX
    // Calculate the gradient magnitude: |Gx| + |Gy|
    float gradient = abs(gx) + abs(gy);

    // Clamp the gradient magnitude to the valid 8-bit range [0, 255]
    gradient = fminf(255.0f, fmaxf(0.0f, gradient));

    // Write the gradient magnitude to the output buffer
    out_grad_buffer[y * width + x] = static_cast<uint8_t>(gradient);

    // Calculate and write the gradient direction (in radians) using atan2
    out_dir_buffer[y * width + x] =
        atan2f(static_cast<float>(gy), static_cast<float>(gx));
  }
}
