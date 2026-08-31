#include <cmath> // For std::ceil
#include <cstdint>
#include <cstdio>
#include <cstdlib> // For malloc, free
#include <cuda.h>
#include <math_constants.h>

#include "common.cuh"
#include "sobel.cuh"

/**
 * @brief This is the central function to start the Sobel algorithm on the GPU.
 *
 * This function allocates device memory, copies the input image from host to
 * device, launches the Sobel kernel, and copies the results back to the host.
 * It also measures the time taken for each major step (H2D, kernel, D2H).
 *
 * @param host_src Pointer to the host-side input image buffer (grayscale,
 * 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @return SobelResult Structure containing pointers to the gradient and
 * direction buffers, and timing information in milliseconds.
 */
SobelResult sobel_execute(const uint8_t *host_src, int32_t width,
                          int32_t height) {
  // grid / block layout: the shared-memory kernel uses a custom blcok format
  // such as 32x8 other functions always use a square BLOCK_SIZE x BLOCK_SIZE
  // block
#ifdef SOBEL_SHARED
  dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid_dim(static_cast<uint32_t>(
                    std::ceil(static_cast<float>(width) / BLOCK_SIZE_X)),
                static_cast<uint32_t>(
                    std::ceil(static_cast<float>(height) / BLOCK_SIZE_Y)));
#else
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim(
      static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
      static_cast<uint32_t>(
          std::ceil(static_cast<float>(height) / BLOCK_SIZE)));
#endif

  // buffer sizes; direction is a bucket index (uint8_t) instead of radians
  // (float) when SOBEL_BUCKET is active -- see sobel_dir_t
  const size_t img_size = static_cast<size_t>(width) *
                          static_cast<size_t>(height) * sizeof(uint8_t);
  const size_t dir_size = static_cast<size_t>(width) *
                          static_cast<size_t>(height) * sizeof(sobel_dir_t);

  // device buffers
  uint8_t *device_src;           // input image
  uint8_t *device_gradient;      // gradient magnitude
  sobel_dir_t *device_direction; // gradient direction (radians or bucket)
#ifdef SOBEL_SPLIT
  int16_t *device_gx = nullptr;
  int16_t *device_gy = nullptr;
#endif

  // host output buffers
  uint8_t *host_gradient = nullptr;
  sobel_dir_t *host_direction = nullptr;

#ifdef SOBEL_HOST_PINNED
  // Page-locked: enables DMA for the D2H copies below, and lets the NMS stage
  // use these buffers directly for its H2D copies without registering them.
  CUDA_THROW_IF_FAILED(cudaMallocHost(&host_gradient, img_size));
  CUDA_THROW_IF_FAILED(cudaMallocHost(&host_direction, dir_size));
#else
  host_gradient = static_cast<uint8_t *>(malloc(img_size));
  host_direction = static_cast<sobel_dir_t *>(malloc(dir_size));
#endif

  CUDA_THROW_IF_FAILED(cudaMalloc(&device_src, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_gradient, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_direction, dir_size));
  CUDA_THROW_IF_FAILED(cudaMemset(device_direction, 0, dir_size));
  CUDA_THROW_IF_FAILED(cudaMemset(device_gradient, 0, img_size));
#ifdef SOBEL_SPLIT
  const size_t component_size = static_cast<size_t>(width) *
                                static_cast<size_t>(height) * sizeof(int16_t);
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_gx, component_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_gy, component_size));
#endif

  cudaEvent_t t0, t1, t2, t3;
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t0));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t1));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t2));
  CUDA_THROW_IF_FAILED(cudaEventCreate(&t3));

  // H->D
  cudaEventRecord(t0);
  CUDA_THROW_IF_FAILED(
      cudaMemcpy(device_src, host_src, img_size, cudaMemcpyHostToDevice));
  cudaEventRecord(t1);

  // kernel
#if defined(SOBEL_SPLIT)
  split_sobel_x_filter<<<grid_dim, block_dim>>>(device_src, width, height,
                                                device_gx);
  split_sobel_y_filter<<<grid_dim, block_dim>>>(device_src, width, height,
                                                device_gy);
  split_sobel_combine_filter<<<grid_dim, block_dim>>>(
      device_gx, device_gy, width, height, device_gradient, device_direction);
#elif defined(SOBEL_NAIVE)
  naive_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);
#elif defined(SOBEL_SHARED)
  sharedm_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);
#elif defined(SOBEL_BUCKET)
  bucket_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);
#elif defined(SOBEL_OPTIMIZED) || defined(SOBEL_PINNED)
  // pinned reuses the optimized kernel, it only differs in host allocation
  optimized_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);
#endif
  CUDA_THROW_IF_FAILED(cudaGetLastError());
  cudaEventRecord(t2);

  // D->H
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_gradient, device_gradient, img_size,
                                  cudaMemcpyDeviceToHost));
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_direction, device_direction, dir_size,
                                  cudaMemcpyDeviceToHost));
  cudaEventRecord(t3);
  CUDA_THROW_IF_FAILED(cudaEventSynchronize(t3));

  SobelResult result{};
  result.host_grad = host_gradient;
#ifdef SOBEL_BUCKET
  result.host_dir_bucket = host_direction;
  result.host_dir = nullptr;
#else
  result.host_dir = host_direction;
  result.host_dir_bucket = nullptr;
#endif
  cudaEventElapsedTime(&result.ms_h2d, t0, t1);
  cudaEventElapsedTime(&result.ms_kernel, t1, t2);
  cudaEventElapsedTime(&result.ms_d2h, t2, t3);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaEventDestroy(t2);
  cudaEventDestroy(t3);
  CUDA_THROW_IF_FAILED(cudaFree(device_src));
  CUDA_THROW_IF_FAILED(cudaFree(device_gradient));
  CUDA_THROW_IF_FAILED(cudaFree(device_direction));
#ifdef SOBEL_SPLIT
  CUDA_THROW_IF_FAILED(cudaFree(device_gx));
  CUDA_THROW_IF_FAILED(cudaFree(device_gy));
#endif

  return result;
}

/**
 * @brief Frees the host-side buffers allocated by sobel_execute.
 *
 * @param result Reference to the SobelResult structure containing the buffers
 * to free.
 */
void sobel_cleanup(SobelResult &result) {
  // exactly one of the two direction pointers is non-null, see SOBEL_BUCKET
#ifdef SOBEL_BUCKET
  void *dir = result.host_dir_bucket;
#else
  void *dir = result.host_dir;
#endif

#ifdef SOBEL_HOST_PINNED
  CUDA_THROW_IF_FAILED(cudaFreeHost(dir));
  CUDA_THROW_IF_FAILED(cudaFreeHost(result.host_grad));
#else
  free(dir);
  free(result.host_grad);
#endif

  result.host_dir = nullptr;
  result.host_dir_bucket = nullptr;
  result.host_grad = nullptr;
}

/**
 * @brief CUDA kernel for Sobel edge detection.
 *
 * Each thread computes the Sobel gradient magnitude and direction for one
 * pixel. The Sobel operator uses two 3x3 convolution kernels (Gx and Gy) to
 * detect edges in the horizontal and vertical directions. The
 * gradient magnitude is calculated as sqrt(Gx^2 + Gy^2) and cast to [0,
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
 * detect edges in the horizontal and vertical directions. The
 * gradient magnitude is calculated as sqrt(Gx^2 + Gy^2) and cast to [0,
 * 255]. The gradient direction is calculated using atan2(Gy, Gx).
 *
 * As optimizations, we have:
 * 1) Implemented shared memory to avoid loading neighboring pixels from global
 * memory in each thread.
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_grad_buffer Output buffer for gradient magnitudes (device memory,
 * 8-bit).
 * @param out_dir_buffer Output buffer for gradient directions (device memory,
 * float).
 */

__global__ void sharedm_sobel_filter(const uint8_t *src_buffer,
                                     const int32_t width, const int32_t height,
                                     uint8_t *out_grad_buffer,
                                     float *out_dir_buffer) {
  // We use shared memory in this case
  // We must also allocate memory for the halo pixels
  // global pixel position in the image
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);

  // local thread position within the block (0..BLOCK_SIZE-1)
  // used to index into the shared memory tile
  const int32_t tx = static_cast<int32_t>(threadIdx.x);
  const int32_t ty = static_cast<int32_t>(threadIdx.y);

  // shared memory tile including halo border of 1
  const uint8_t SOBEL_RADIUS = 1;
  constexpr int32_t tile_size_x = BLOCK_SIZE_X + 2 * SOBEL_RADIUS;
  constexpr int32_t tile_size_y = BLOCK_SIZE_Y + 2 * SOBEL_RADIUS;
  __shared__ uint8_t shared_memory[tile_size_x * tile_size_y];

  // load tile into shared memory (including halo)
  // each thread loads one or more pixels depending on tile size vs block size
  for (int32_t j = ty; j < tile_size_y; j += BLOCK_SIZE_Y) {
    for (int32_t i = tx; i < tile_size_x; i += BLOCK_SIZE_X) {
      // map tile position back to image coordinates (subtract halo offset)
      const int32_t img_x =
          static_cast<int32_t>(blockIdx.x * BLOCK_SIZE_X) + i - SOBEL_RADIUS;
      const int32_t img_y =
          static_cast<int32_t>(blockIdx.y * BLOCK_SIZE_Y) + j - SOBEL_RADIUS;

      // zero-padding for pixels outside image bounds
      shared_memory[j * tile_size_x + i] =
          (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height)
              ? src_buffer[img_y * width + img_x]
              : 0;
    }
  }

  // wait until every thread has finished loading before any thread starts
  // reading
  __syncthreads();

  if (x >= width || y >= height)
    return;

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
        uint8_t pixel = shared_memory[(threadIdx.y + 1 + i) * tile_size_x +
                                      (threadIdx.x + 1 + j)];
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
    // We round the grad dir to the four directions 0°, 45°, 90°, 135°, 180°
    out_dir_buffer[y * width + x] =
        atan2f(static_cast<float>(gy), static_cast<float>(gx));
  }
}

/**
 * @brief Optimized CUDA kernel for Sobel edge detection. Since the shared
 * memory version did not improve the runtime, this version uses other
 * optimizations.
 *
 * We use two main optimization:
 *
 * 1. We use a simpler version to calculate the gradient value.
 * 2. We unroll the Sobel kernel loop
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
  // Calculate pixel coordinates for this thread
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  // OPTIMIZATION 1: We unroll the former Sobel Kernel Arrays
  if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
    return;

  // First, we define the starting positions of the three rows that we need
  // (e.g., r0 is the upper left pixel of the current pixel in the center)
  const uint8_t *r0 = src_buffer + (y - 1) * width + (x - 1);
  const uint8_t *r1 = r0 + width;
  const uint8_t *r2 = r1 + width;

  // Next, we get all neighboring pixels of p11 (which is our current pixel)
  // Since we are using pointers, we can just index the next positions
  int p00 = r0[0], p01 = r0[1], p02 = r0[2];
  int p10 = r1[0], p12 = r1[2];
  int p20 = r2[0], p21 = r2[1], p22 = r2[2];

  // Finally, we directly calculate the values of gx and gy instead of iterating
  // over an array; We also use bit shifts instead of multiplication operators
  int gx = -p00 + p02 - (p10 << 1) + (p12 << 1) - p20 + p22;
  int gy = -p00 - (p01 << 1) - p02 + p20 + (p21 << 1) + p22;

  // OPTIMIZATION 2: We are using a simplified version to calculate the gradient
  // magnitude: abs(Gx) + abs(Gy)
  float gradient = abs(gx) + abs(gy);

  // Clamp the gradient magnitude to the valid 8-bit range [0, 255]
  gradient = fminf(255.0f, fmaxf(0.0f, gradient));

  // Write the gradient magnitude to the output buffer
  out_grad_buffer[y * width + x] = static_cast<uint8_t>(gradient);

  // Calculate and write the gradient direction (in radians) using atan2
  out_dir_buffer[y * width + x] =
      atan2f(static_cast<float>(gy), static_cast<float>(gx));
}

/**
 * @brief Sobel kernel, builds on optimized_sobel_filter but drops atan2f().
 * Direction bucket is derived from gx/gy sign and |gy|/|gx| slope thresholds
 * instead of computing the actual angle.
 *
 * @param src_buffer Input image (device, grayscale, 8-bit).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param out_grad_buffer Output gradient magnitude (device, 8-bit).
 * @param out_dir_buffer Output direction bucket (device, 0-3 = 0/45/90/135°).
 */
__global__ void bucket_sobel_filter(const uint8_t *src_buffer,
                                    const int32_t width, const int32_t height,
                                    uint8_t *out_grad_buffer,
                                    uint8_t *out_dir_buffer) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
    return;

  const uint8_t *r0 = src_buffer + (y - 1) * width + (x - 1);
  const uint8_t *r1 = r0 + width;
  const uint8_t *r2 = r1 + width;
  int p00 = r0[0], p01 = r0[1], p02 = r0[2];
  int p10 = r1[0], p12 = r1[2];
  int p20 = r2[0], p21 = r2[1], p22 = r2[2];

  int gx = -p00 + p02 - (p10 << 1) + (p12 << 1) - p20 + p22;
  int gy = -p00 - (p01 << 1) - p02 + p20 + (p21 << 1) + p22;

  float gradient =
      fabsf(static_cast<float>(gx)) + fabsf(static_cast<float>(gy));
  out_grad_buffer[y * width + x] =
      static_cast<uint8_t>(fminf(255.0f, gradient));

  const float abs_gx = fabsf(static_cast<float>(gx));
  const float abs_gy = fabsf(static_cast<float>(gy));
  constexpr float TAN_22_5 = 0.414213562f;
  constexpr float TAN_67_5 = 2.414213562f;

  uint8_t bucket;
  if (abs_gy < TAN_22_5 * abs_gx)
    bucket = 0; // 0°
  else if (abs_gy > TAN_67_5 * abs_gx)
    bucket = 2; // 90°
  else if ((gx > 0) == (gy > 0))
    bucket = 1; // 45°
  else
    bucket = 3; // 135°

  out_dir_buffer[y * width + x] = bucket;
}

#ifdef SOBEL_SPLIT

/**
 * @brief Computes the horizontal Sobel component into an int16 buffer.
 *
 * This is intentionally a separate kernel from split_sobel_y_filter so the
 * split implementation can be benchmarked against the combined kernel.
 */
__global__ void split_sobel_x_filter(const uint8_t *src_buffer,
                                     const int32_t width, const int32_t height,
                                     int16_t *out_gx_buffer) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);

  if (x >= width || y >= height)
    return;

  const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) +
                       static_cast<size_t>(x);
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
    out_gx_buffer[index] = 0;
    return;
  }

  const uint8_t *r0 = src_buffer + (y - 1) * width + (x - 1);
  const uint8_t *r1 = r0 + width;
  const uint8_t *r2 = r1 + width;

  const int gx = -r0[0] + r0[2] - (r1[0] << 1) + (r1[2] << 1) - r2[0] + r2[2];
  out_gx_buffer[index] = static_cast<int16_t>(gx);
}

/**
 * @brief Computes the vertical Sobel component into an int16 buffer.
 */
__global__ void split_sobel_y_filter(const uint8_t *src_buffer,
                                     const int32_t width, const int32_t height,
                                     int16_t *out_gy_buffer) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);

  if (x >= width || y >= height)
    return;

  const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) +
                       static_cast<size_t>(x);
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
    out_gy_buffer[index] = 0;
    return;
  }

  const uint8_t *r0 = src_buffer + (y - 1) * width + (x - 1);
  const uint8_t *r1 = r0 + width;
  const uint8_t *r2 = r1 + width;

  const int gy = -r0[0] - (r0[1] << 1) - r0[2] + r2[0] + (r2[1] << 1) + r2[2];
  out_gy_buffer[index] = static_cast<int16_t>(gy);
}

/**
 * @brief Combines split Gx/Gy components into the bucket-mode outputs.
 */
__global__ void
split_sobel_combine_filter(const int16_t *gx_buffer, const int16_t *gy_buffer,
                           const int32_t width, const int32_t height,
                           uint8_t *out_grad_buffer, uint8_t *out_dir_buffer) {
  const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);

  if (x >= width || y >= height)
    return;

  const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) +
                       static_cast<size_t>(x);
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
    out_grad_buffer[index] = 0;
    out_dir_buffer[index] = 0;
    return;
  }

  const int gx = static_cast<int>(gx_buffer[index]);
  const int gy = static_cast<int>(gy_buffer[index]);
  const float abs_gx = fabsf(static_cast<float>(gx));
  const float abs_gy = fabsf(static_cast<float>(gy));

  const float gradient = fminf(255.0f, abs_gx + abs_gy);
  out_grad_buffer[index] = static_cast<uint8_t>(gradient);

  constexpr float TAN_22_5 = 0.414213562f;
  constexpr float TAN_67_5 = 2.414213562f;

  uint8_t bucket;
  if (abs_gy < TAN_22_5 * abs_gx)
    bucket = 0; // 0 degrees
  else if (abs_gy > TAN_67_5 * abs_gx)
    bucket = 2; // 90 degrees
  else if ((gx > 0) == (gy > 0))
    bucket = 1; // 45 degrees
  else
    bucket = 3; // 135 degrees

  out_dir_buffer[index] = bucket;
}

#endif // SOBEL_SPLIT

//
// FUSED PIPELINE SUPPORT
//

#ifdef PIPELINE_FUSED

void sobel_launch(const uint8_t *d_src, uint8_t *d_grad, sobel_dir_t *d_dir,
                  int32_t width, int32_t height, int16_t *d_gx, int16_t *d_gy,
                  cudaStream_t stream) {
// same block layout logic as sobel_execute(): SOBEL_SHARED uses a
// 32x8 block, all others a square BLOCK_SIZE x BLOCK_SIZE block
#ifdef SOBEL_SHARED
  dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid_dim(static_cast<uint32_t>(
                    std::ceil(static_cast<float>(width) / BLOCK_SIZE_X)),
                static_cast<uint32_t>(
                    std::ceil(static_cast<float>(height) / BLOCK_SIZE_Y)));
#else
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim(
      static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
      static_cast<uint32_t>(
          std::ceil(static_cast<float>(height) / BLOCK_SIZE)));
#endif

#if defined(SOBEL_SPLIT)
  split_sobel_x_filter<<<grid_dim, block_dim, 0, stream>>>(d_src, width, height,
                                                           d_gx);
  split_sobel_y_filter<<<grid_dim, block_dim, 0, stream>>>(d_src, width, height,
                                                           d_gy);
  split_sobel_combine_filter<<<grid_dim, block_dim, 0, stream>>>(
      d_gx, d_gy, width, height, d_grad, d_dir);
#elif defined(SOBEL_NAIVE)
  naive_sobel_filter<<<grid_dim, block_dim, 0, stream>>>(d_src, width, height,
                                                         d_grad, d_dir);
#elif defined(SOBEL_SHARED)
  sharedm_sobel_filter<<<grid_dim, block_dim, 0, stream>>>(d_src, width, height,
                                                           d_grad, d_dir);
#elif defined(SOBEL_BUCKET)
  bucket_sobel_filter<<<grid_dim, block_dim, 0, stream>>>(d_src, width, height,
                                                          d_grad, d_dir);
#elif defined(SOBEL_OPTIMIZED) || defined(SOBEL_PINNED)
  // pinned reuses the optimized kernel, it only differs in host allocation
  optimized_sobel_filter<<<grid_dim, block_dim, 0, stream>>>(
      d_src, width, height, d_grad, d_dir);
#endif
}

#endif // #ifdef PIPELINE_FUSED
