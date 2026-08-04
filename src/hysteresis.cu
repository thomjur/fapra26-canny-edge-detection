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
#if defined(HYSTERESIS_SHARED)
  hysteresis_sharedm<<<grid_dim, block_dim>>>(
      device_buffer, width, height, device_dst, low_threshold, high_threshold);
#endif
#if defined(HYSTERESIS_OPTIMIZED)
  hysteresis_opt<<<grid_dim, block_dim>>>(
      device_buffer, width, height, device_dst, low_threshold, high_threshold);
#endif
#if defined(HYSTERESIS_PINNED)
  hysteresis_opt<<<grid_dim, block_dim>>>(
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

/**
 * @brief Hysteresis edge linking using shared-memory tiling.
 *
 * This kernel classifies each pixel as edge (255) or non-edge (0):
 * - Strong pixel (p >= high_t) -> edge (255)
 * - Weak pixel   (low_t <= p < high_t) -> edge only if at least one of its
 *   8 neighbors is strong (>= high_t)
 * - Otherwise -> non-edge (0)
 *
 * Performance notes:
 * - Uses shared memory tile with a 1-pixel halo to reduce global memory reads.
 * - Neighbor checks are performed from shared memory (faster than global
 * loads).
 *
 * @param src      Input grayscale image (device pointer, uint8_t)
 * @param width    Image width in pixels
 * @param height   Image height in pixels
 * @param dst      Output binary edge image (device pointer, uint8_t)
 * @param low_t    Low threshold
 * @param high_t   High threshold
 */
__global__ void hysteresis_sharedm(uint8_t *src, const int32_t width,
                                   const int32_t height, uint8_t *dst,
                                   uint8_t low_t, uint8_t high_t) {
  // Shared tile includes a 1-pixel halo on each side:
  // tile size = (BY + 2) x (BX + 2)
  __shared__ uint8_t tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

  // Global coordinates of this thread's pixel
  const int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int gy = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  // Local coordinates inside shared tile (+1 because of halo border)
  const int lx = threadIdx.x + 1;
  const int ly = threadIdx.y + 1;

  // -------------------------------
  // 1) Load center pixel into shared memory
  // -------------------------------
  tile[ly][lx] = (gx < width && gy < height) ? src[gy * width + gx] : 0;

  // -------------------------------
  // 2) Load halo pixels (block borders)
  //    Only border threads do these loads
  // -------------------------------
  if (threadIdx.x == 0) {
    tile[ly][0] = (gx > 0 && gy < height) ? src[gy * width + (gx - 1)] : 0;
  }
  if (threadIdx.x == BLOCK_SIZE - 1) {
    tile[ly][BLOCK_SIZE + 1] =
        (gx + 1 < width && gy < height) ? src[gy * width + (gx + 1)] : 0;
  }
  if (threadIdx.y == 0) {
    tile[0][lx] = (gy > 0 && gx < width) ? src[(gy - 1) * width + gx] : 0;
  }
  if (threadIdx.y == BLOCK_SIZE - 1) {
    tile[BLOCK_SIZE + 1][lx] =
        (gy + 1 < height && gx < width) ? src[(gy + 1) * width + gx] : 0;
  }

  // -------------------------------
  // 3) Load halo corners
  // -------------------------------
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    tile[0][0] = (gx > 0 && gy > 0) ? src[(gy - 1) * width + (gx - 1)] : 0;
  }
  if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
    tile[0][BLOCK_SIZE + 1] =
        (gx + 1 < width && gy > 0) ? src[(gy - 1) * width + (gx + 1)] : 0;
  }
  if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
    tile[BLOCK_SIZE + 1][0] =
        (gx > 0 && gy + 1 < height) ? src[(gy + 1) * width + (gx - 1)] : 0;
  }
  if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
    tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] =
        (gx + 1 < width && gy + 1 < height) ? src[(gy + 1) * width + (gx + 1)]
                                            : 0;
  }

  // Wait until full tile (center + halo) is available
  __syncthreads();

  // Skip out-of-image threads
  if (gx >= width || gy >= height)
    return;

  // -------------------------------
  // 4) Classify pixel
  // -------------------------------
  const uint8_t p = tile[ly][lx];
  uint8_t out = 0;

  // Strong edges are accepted directly
  if (p >= high_t) {
    out = 255;
  }
  // Weak edges: keep only if at least one strong neighbor exists
  else if (p >= low_t) {
    uint8_t max_n = 0;

    // Check 8-neighborhood from shared memory
    max_n = max(max_n, tile[ly - 1][lx - 1]);
    max_n = max(max_n, tile[ly - 1][lx]);
    max_n = max(max_n, tile[ly - 1][lx + 1]);
    max_n = max(max_n, tile[ly][lx - 1]);
    max_n = max(max_n, tile[ly][lx + 1]);
    max_n = max(max_n, tile[ly + 1][lx - 1]);
    max_n = max(max_n, tile[ly + 1][lx]);
    max_n = max(max_n, tile[ly + 1][lx + 1]);

    out = (max_n >= high_t) ? 255 : 0;
  }

  // Write output
  dst[gy * width + gx] = out;
}

/**
 * @brief Hysteresis edge linking optimized WITHOUT shared-memory tiling.
 *
 * This kernel classifies each pixel as edge (255) or non-edge (0):
 * - Strong pixel (p >= high_t) -> edge (255)
 * - Weak pixel   (low_t <= p < high_t) -> edge only if at least one of its
 *   8 neighbors is strong (>= high_t)
 * - Otherwise -> non-edge (0)
 *
 * Performance notes:
 * - Neighbor checks are performed using unrolled neighbor checking.
 *
 * @param src      Input grayscale image (device pointer, uint8_t)
 * @param width    Image width in pixels
 * @param height   Image height in pixels
 * @param dst      Output binary edge image (device pointer, uint8_t)
 * @param low_t    Low threshold
 * @param high_t   High threshold
 */
__global__ void hysteresis_opt(uint8_t *src, const int32_t width,
                               const int32_t height, uint8_t *dst,
                               uint8_t low_t, uint8_t high_t) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int index = y * width + x;
  const uint8_t p = src[index];

  uint8_t out = 0;

  // Strong edges are accepted directly
  if (p >= high_t) {
    out = 255;
  }
  // Weak edges: keep only if at least one strong neighbor exists
  else if (p >= low_t) {
    uint8_t max_n = 0;

    // Check 8-neighborhood from shared memory
    max_n = max(max_n, src[(y - 1) * width + x - 1]);
    max_n = max(max_n, src[(y - 1) * width + x]);
    max_n = max(max_n, src[(y - 1) * width + x + 1]);
    max_n = max(max_n, src[y * width + x - 1]);
    max_n = max(max_n, src[y * width + x + 1]);
    max_n = max(max_n, src[(y + 1) * width + x - 1]);
    max_n = max(max_n, src[(y + 1) * width + x]);
    max_n = max(max_n, src[(y + 1) * width + x + 1]);

    out = (max_n >= high_t) ? 255 : 0;
  }

  // Write output
  dst[index] = out;
}

//
// FUSED PIPELINE SUPPORT
//

#ifdef PIPELINE_FUSED

void hysteresis_launch(
    const uint8_t *d_src,
    uint8_t *d_dst,
    int32_t width,
    int32_t height,
    cudaStream_t stream)
{
    // same block/grid layout as hysteresis_execute()
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
                  static_cast<uint32_t>(std::ceil(static_cast<float>(height) / BLOCK_SIZE)));

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

    // note: d_src is non-const in the kernel signatures (hysteresis_sharedm /
    // hysteresis_opt take uint8_t*, not const uint8_t*), hence the cast
    auto *src = const_cast<uint8_t *>(d_src);

    #if defined(HYSTERESIS_NAIVE)
    naive_hysteresis_linker<<<grid_dim, block_dim, 0, stream>>>(
        src, width, height, d_dst, low_threshold, high_threshold);
    #elif defined(HYSTERESIS_SHARED)
    hysteresis_sharedm<<<grid_dim, block_dim, 0, stream>>>(
        src, width, height, d_dst, low_threshold, high_threshold);
    #elif defined(HYSTERESIS_OPTIMIZED) || defined(HYSTERESIS_PINNED)
    // pinned reuses the optimized kernel, it only differs in host allocation
    hysteresis_opt<<<grid_dim, block_dim, 0, stream>>>(
        src, width, height, d_dst, low_threshold, high_threshold);
    #endif
}

#endif // #ifdef PIPELINE_FUSED