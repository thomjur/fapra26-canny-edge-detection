#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <math_constants.h>

#include "sobel.cuh"
#include "nms.cuh"
#include "common.cuh"

//
// >>> NON-MAXIMUM SUPPRESSION <
//
// Kernel variants are selected at compile time along two independent axes:
//   NMS_NAIVE / NMS_SHARED  -> memory access pattern
//   NMS_BUCKET              -> direction representation (uint8 bucket vs. float radians)
//   NMS_PINNED              -> host-side allocation only, implies NMS_SHARED
//

//
// HOST
//

NmsResult nms_execute(
    const uint8_t *host_magnitude,
    const nms_dir_t *host_direction,
    const int32_t width,
    const int32_t height)
{
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t img_size    = pixel_count * sizeof(uint8_t);
    const size_t dir_size    = pixel_count * sizeof(nms_dir_t);

    // grid / block layout
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
                  static_cast<uint32_t>(std::ceil(static_cast<float>(height) / BLOCK_SIZE)));

    // host output buffer
    uint8_t *host_dst = nullptr;

    // Input buffers are already page-locked by the sobel stage
    // (SOBEL_HOST_PINNED), so no cudaHostRegister() is needed here.
#ifdef NMS_PINNED
    CUDA_THROW_IF_FAILED(cudaMallocHost(&host_dst, img_size));
#else
    host_dst = static_cast<uint8_t *>(malloc(img_size));
#endif

    // device buffers
    uint8_t *device_magnitude   = nullptr;
    nms_dir_t *device_direction = nullptr;
    uint8_t *device_dst         = nullptr;
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_magnitude, img_size));
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_direction, dir_size));
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_dst, img_size));
    CUDA_THROW_IF_FAILED(cudaMemset(device_dst, 0, img_size));

    // timing events
    cudaEvent_t t0, t1, t2, t3;
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t0));
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t1));
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t2));
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t3));

    // H->D
    cudaEventRecord(t0);
    CUDA_THROW_IF_FAILED(cudaMemcpy(device_magnitude, host_magnitude, img_size, cudaMemcpyHostToDevice));
    CUDA_THROW_IF_FAILED(cudaMemcpy(device_direction, host_direction, dir_size, cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    // kernel
#if defined(NMS_NAIVE)
#ifdef NMS_BUCKET
    non_maximum_suppression_bucket<<<grid_dim, block_dim>>>(device_magnitude, device_direction, width, height,
                                                            device_dst);
#else
    non_maximum_suppression<<<grid_dim, block_dim>>>(device_magnitude, device_direction, width, height, device_dst);
#endif
#elif defined(NMS_SHARED) // also covers NMS_PINNED
#ifdef NMS_BUCKET
    non_maximum_suppression_shared_bucket<<<grid_dim, block_dim>>>(device_magnitude, device_direction, width, height,
                                                                   device_dst);
#else
    non_maximum_suppression_shared<<<grid_dim, block_dim>>>(device_magnitude, device_direction, width, height,
                                                            device_dst);
#endif
#endif
    CUDA_THROW_IF_FAILED(cudaGetLastError());
    cudaEventRecord(t2);

    // D->H
    CUDA_THROW_IF_FAILED(cudaMemcpy(host_dst, device_dst, img_size, cudaMemcpyDeviceToHost));
    cudaEventRecord(t3);
    CUDA_THROW_IF_FAILED(cudaEventSynchronize(t3));

    // collect timings
    NmsResult result{};
    result.host_nms = host_dst;
    cudaEventElapsedTime(&result.ms_h2d, t0, t1);
    cudaEventElapsedTime(&result.ms_kernel, t1, t2);
    cudaEventElapsedTime(&result.ms_d2h, t2, t3);

    // clean up device resources
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaEventDestroy(t2);
    cudaEventDestroy(t3);
    CUDA_THROW_IF_FAILED(cudaFree(device_magnitude));
    CUDA_THROW_IF_FAILED(cudaFree(device_direction));
    CUDA_THROW_IF_FAILED(cudaFree(device_dst));

    return result;
}

void nms_cleanup(NmsResult &result)
{
    if (result.host_nms == nullptr)
        return;

#ifdef NMS_PINNED
    CUDA_THROW_IF_FAILED(cudaFreeHost(result.host_nms));
#else
    free(result.host_nms);
#endif
    result.host_nms = nullptr;
}

//
// DEVICE HELPERS
//

/// Halo radius around each block's tile. NMS only ever looks at the immediate
/// 8-neighborhood, so a single-pixel border is enough (unlike the Gaussian
/// blur, which needs BLUR_RADIUS pixels of halo).
constexpr int32_t NMS_RADIUS    = 1;
constexpr int32_t NMS_TILE_SIZE = BLOCK_SIZE + 2 * NMS_RADIUS;

/**
 * @brief Quantizes a gradient direction in radians into one of 4 principal
 * directions and returns the (dx, dy) offset pointing along the gradient.
 */
__device__ __forceinline__ void quantize_direction(const float direction, int32_t &dx, int32_t &dy)
{
    // convert radians to degrees and fold into [0, 180)
    // gradient direction and its opposite (+180 deg) point along the same edge normal
    float deg = direction * (180.0f / CUDART_PI_F);
    deg       = fmodf(deg + 180.0f, 180.0f);

    if (deg < 22.5f)
    {
        dx = 1;
        dy = 0;
    } // 0 deg   (W/E)
    else if (deg < 67.5f)
    {
        dx = 1;
        dy = -1;
    } // 45 deg  (NE/SW)
    else if (deg < 112.5f)
    {
        dx = 0;
        dy = 1;
    } // 90 deg  (N/S)
    else if (deg < 157.5f)
    {
        dx = 1;
        dy = 1;
    } // 135 deg (NW/SE)
    else
    {
        dx = 1;
        dy = 0;
    } // wraps back to 0 deg
}

/// Lookup table: bucket index -> (dx, dy) offset along the gradient direction.
/// 0 = 0 deg, 1 = 45 deg, 2 = 90 deg, 3 = 135 deg -- matches bucket_sobel_filter's encoding.
__constant__ int32_t dir_bucket_dx[4] = {1, 1, 0, 1};
__constant__ int32_t dir_bucket_dy[4] = {0, -1, 1, 1};

/**
 * @brief Samples the magnitude buffer from global memory, returning 0 for
 * out-of-bounds coordinates (zero padding at the image border).
 */
__device__ __forceinline__ uint8_t sample_clamped(
    const uint8_t *buffer,
    const int32_t width,
    const int32_t height,
    const int32_t x,
    const int32_t y)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
        return 0;
    return buffer[y * width + x];
}

/**
 * @brief Cooperatively loads the block's magnitude tile plus its 1px halo into
 * shared memory. Every thread of the block must call this (and hit the
 * __syncthreads() inside), even threads whose own pixel lies outside the image.
 */
__device__ __forceinline__ void load_magnitude_tile(
    const uint8_t *magnitude_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *tile)
{
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);

    // BLOCK_SIZE x BLOCK_SIZE threads fill a (BLOCK_SIZE + 2)^2 tile,
    // so the strided loops run at most twice per axis
    for (int32_t j = ty; j < NMS_TILE_SIZE; j += BLOCK_SIZE)
    {
        for (int32_t i = tx; i < NMS_TILE_SIZE; i += BLOCK_SIZE)
        {
            const int32_t img_x = static_cast<int32_t>(blockIdx.x * BLOCK_SIZE) + i - NMS_RADIUS;
            const int32_t img_y = static_cast<int32_t>(blockIdx.y * BLOCK_SIZE) + j - NMS_RADIUS;

            tile[j * NMS_TILE_SIZE + i] = sample_clamped(magnitude_buffer, width, height, img_x, img_y);
        }
    }

    __syncthreads();
}

/**
 * @brief The suppression test itself: keep the magnitude only if it is a local
 * maximum along the gradient direction, otherwise zero it out.
 */
__device__ __forceinline__ uint8_t suppress(const uint8_t magnitude, const uint8_t neighbor_a, const uint8_t neighbor_b)
{
    return (magnitude >= neighbor_a && magnitude >= neighbor_b) ? magnitude : 0;
}

//
// NAIVE
//

#if defined(NMS_NAIVE) && !defined(NMS_BUCKET)

__global__ void non_maximum_suppression(
    const uint8_t *magnitude_buffer,
    const float *direction_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_nms_buffer)
{
    const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;

    const uint8_t magnitude = magnitude_buffer[y * width + x];
    const float direction   = direction_buffer[y * width + x];

    int32_t dx = 0;
    int32_t dy = 0;
    quantize_direction(direction, dx, dy);

    const uint8_t neighbor_a = sample_clamped(magnitude_buffer, width, height, x + dx, y + dy);
    const uint8_t neighbor_b = sample_clamped(magnitude_buffer, width, height, x - dx, y - dy);

    out_nms_buffer[y * width + x] = suppress(magnitude, neighbor_a, neighbor_b);
}

#endif // NMS_NAIVE && !NMS_BUCKET

//
// NAIVE + BUCKET
//

#if defined(NMS_NAIVE) && defined(NMS_BUCKET)

__global__ void non_maximum_suppression_bucket(
    const uint8_t *magnitude_buffer,
    const uint8_t *dir_bucket_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_nms_buffer)
{
    const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;

    const uint8_t magnitude = magnitude_buffer[y * width + x];
    const uint8_t bucket    = dir_bucket_buffer[y * width + x];

    // no angle computation at all, just a constant-memory lookup
    const int32_t dx = dir_bucket_dx[bucket];
    const int32_t dy = dir_bucket_dy[bucket];

    const uint8_t neighbor_a = sample_clamped(magnitude_buffer, width, height, x + dx, y + dy);
    const uint8_t neighbor_b = sample_clamped(magnitude_buffer, width, height, x - dx, y - dy);

    out_nms_buffer[y * width + x] = suppress(magnitude, neighbor_a, neighbor_b);
}

#endif // NMS_NAIVE && NMS_BUCKET

//
// SHARED
//

#if defined(NMS_SHARED) && !defined(NMS_BUCKET)

__global__ void non_maximum_suppression_shared(
    const uint8_t *magnitude_buffer,
    const float *direction_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_nms_buffer)
{
    const int32_t x  = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y  = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);

    __shared__ uint8_t tile[NMS_TILE_SIZE * NMS_TILE_SIZE];
    load_magnitude_tile(magnitude_buffer, width, height, tile);

    // bail out only AFTER every thread has helped fill shared memory and
    // reached the barrier inside load_magnitude_tile()
    if (x >= width || y >= height)
        return;

    const uint8_t magnitude = tile[(ty + NMS_RADIUS) * NMS_TILE_SIZE + (tx + NMS_RADIUS)];
    const float direction   = direction_buffer[y * width + x];

    int32_t dx = 0;
    int32_t dy = 0;
    quantize_direction(direction, dx, dy);

    const uint8_t neighbor_a = tile[(ty + dy + NMS_RADIUS) * NMS_TILE_SIZE + (tx + dx + NMS_RADIUS)];
    const uint8_t neighbor_b = tile[(ty - dy + NMS_RADIUS) * NMS_TILE_SIZE + (tx - dx + NMS_RADIUS)];

    out_nms_buffer[y * width + x] = suppress(magnitude, neighbor_a, neighbor_b);
}

#endif // NMS_SHARED && !NMS_BUCKET

//
// SHARED + BUCKET
//

#if defined(NMS_SHARED) && defined(NMS_BUCKET)

__global__ void non_maximum_suppression_shared_bucket(
    const uint8_t *magnitude_buffer,
    const uint8_t *dir_bucket_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_nms_buffer)
{
    const int32_t x  = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y  = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);

    __shared__ uint8_t tile[NMS_TILE_SIZE * NMS_TILE_SIZE];
    load_magnitude_tile(magnitude_buffer, width, height, tile);

    if (x >= width || y >= height)
        return;

    const uint8_t magnitude = tile[(ty + NMS_RADIUS) * NMS_TILE_SIZE + (tx + NMS_RADIUS)];
    const uint8_t bucket    = dir_bucket_buffer[y * width + x];

    const int32_t dx = dir_bucket_dx[bucket];
    const int32_t dy = dir_bucket_dy[bucket];

    const uint8_t neighbor_a = tile[(ty + dy + NMS_RADIUS) * NMS_TILE_SIZE + (tx + dx + NMS_RADIUS)];
    const uint8_t neighbor_b = tile[(ty - dy + NMS_RADIUS) * NMS_TILE_SIZE + (tx - dx + NMS_RADIUS)];

    out_nms_buffer[y * width + x] = suppress(magnitude, neighbor_a, neighbor_b);
}

#endif // NMS_SHARED && NMS_BUCKET
