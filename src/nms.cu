#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <math_constants.h>

#include "sobel.cuh"
#include "nms.cuh"
#include "common.cuh"

//
// >>> NON-MAXIMUM SUPPRESSION <
//

//
// HOST
//

NmsResult nms_execute(
    const uint8_t *host_magnitude,
#ifdef NMS_BUCKET
    const uint8_t *host_direction,
#else
    const float *host_direction,
#endif
    const int32_t width,
    const int32_t height)
{
    const size_t img_size = width * height * sizeof(uint8_t);
#ifdef NMS_BUCKET
    const size_t dir_size = width * height * sizeof(uint8_t);
#else
    const size_t dir_size = width * height * sizeof(float);
#endif

    // grid / block layout
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
                  static_cast<uint32_t>(std::ceil(static_cast<float>(height) / BLOCK_SIZE)));

    // host output buffer
    auto *host_dst = static_cast<uint8_t *>(malloc(img_size));

    // device buffers
    uint8_t *device_magnitude;
#ifdef NMS_BUCKET
    uint8_t *device_direction;
#else
    float *device_direction;
#endif
    uint8_t *device_dst;
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
#ifdef NMS_NAIVE
    non_maximum_suppression<<<grid_dim, block_dim>>>(device_magnitude, device_direction, width, height, device_dst);
#endif
#ifdef NMS_SHARED
    non_maximum_suppression<<<grid_dim, block_dim>>>(device_magnitude, device_direction, width, height, device_dst);
#endif
#ifdef NMS_BUCKET
    non_maximum_suppression_bucket<<<grid_dim, block_dim>>>(device_magnitude, device_direction, width, height, device_dst);
#endif
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
    free(result.host_nms);
    result.host_nms = nullptr;
}

//
// shared helper: quantize gradient direction into one of 4 principal
// directions and return the (dx, dy) offset pointing along the gradient
//
__device__ __forceinline__ void quantize_direction(const float direction, int32_t &dx, int32_t &dy)
{
    // convert radians to degrees and fold into [0, 180)
    // gradient direction and its opposite (+180°) point along the same edge normal
    float deg = direction * (180.0f / CUDART_PI_F);
    deg       = fmodf(deg + 180.0f, 180.0f);

    if (deg < 22.5f)
    {
        dx = 1;
        dy = 0;
    } // 0°   (W/E)
    else if (deg < 67.5f)
    {
        dx = 1;
        dy = -1;
    } // 45°  (NE/SW)
    else if (deg < 112.5f)
    {
        dx = 0;
        dy = 1;
    } // 90°  (N/S)
    else if (deg < 157.5f)
    {
        dx = 1;
        dy = 1;
    } // 135° (NW/SE)
    else
    {
        dx = 1;
        dy = 0;
    } // wraps back to 0°
}


//
// NAIVE
//

#ifdef NMS_NAIVE
/**
 * @brief Samples the magnitude buffer, returning 0 for out-of-bounds coordinates.
 */
__device__ inline uint8_t sample_clamped(
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

__global__ void non_maximum_suppression(
    const uint8_t* magnitude_buffer,
    const float*   direction_buffer,
    const int32_t  width,
    const int32_t  height,
    uint8_t*       out_nms_buffer)
{
    const auto x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;

    const uint8_t magnitude = magnitude_buffer[y * width + x];
    const float   direction = direction_buffer[y * width + x];

    int32_t dx = 0;
    int32_t dy = 0;
    quantize_direction(direction, dx, dy);

    const uint8_t neighbor_a = sample_clamped(magnitude_buffer, width, height, x + dx, y + dy);
    const uint8_t neighbor_b = sample_clamped(magnitude_buffer, width, height, x - dx, y - dy);

    out_nms_buffer[y * width + x] = (magnitude >= neighbor_a && magnitude >= neighbor_b) ? magnitude : 0;
}
#endif // #ifdef NMS_NAIVE

//
// SHARED
//

#ifdef NMS_SHARED
/// Halo radius needed around each block's tile, NMS only ever looks at the
/// immediate 8-neighborhood, so a single-pixel border is enough (unlike the
/// Gaussian blur, which needs BLUR_RADIUS pixels of halo).
constexpr int32_t NMS_RADIUS = 1;

__global__ void non_maximum_suppression(
    const uint8_t* magnitude_buffer,
    const float*   direction_buffer,
    const int32_t  width,
    const int32_t  height,
    uint8_t*       out_nms_buffer)
{
    // global pixel position in the image
    const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);

    // local thread position within the block (0..BLOCK_SIZE-1)
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);

    // shared memory tile including 1px halo border on each side
    constexpr int32_t tile_size = BLOCK_SIZE + 2 * NMS_RADIUS;
    __shared__ uint8_t tile[tile_size * tile_size];

    // load tile into shared memory (including halo), every thread in the
    // block participates regardless of whether its own (x, y) is in bounds,
    // since all threads must reach the __syncthreads() below
    for (int32_t j = ty; j < tile_size; j += BLOCK_SIZE)
    {
        for (int32_t i = tx; i < tile_size; i += BLOCK_SIZE)
        {
            const int32_t img_x = static_cast<int32_t>(blockIdx.x * BLOCK_SIZE) + i - NMS_RADIUS;
            const int32_t img_y = static_cast<int32_t>(blockIdx.y * BLOCK_SIZE) + j - NMS_RADIUS;

            // zero-padding for pixels outside image bounds
            tile[j * tile_size + i] = (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height)
                ? magnitude_buffer[img_y * width + img_x]
                : 0;
        }
    }

    __syncthreads();

    // only bail out on the actual output write, AFTER every thread has
    // helped fill shared memory and reached the barrier above
    if (x >= width || y >= height)
        return;

    const uint8_t magnitude = tile[(ty + NMS_RADIUS) * tile_size + (tx + NMS_RADIUS)];
    const float   direction = direction_buffer[y * width + x];

    int32_t dx = 0;
    int32_t dy = 0;
    quantize_direction(direction, dx, dy);

    const uint8_t neighbor_a = tile[(ty + dy + NMS_RADIUS) * tile_size + (tx + dx + NMS_RADIUS)];
    const uint8_t neighbor_b = tile[(ty - dy + NMS_RADIUS) * tile_size + (tx - dx + NMS_RADIUS)];

    out_nms_buffer[y * width + x] = (magnitude >= neighbor_a && magnitude >= neighbor_b) ? magnitude : 0;
}

#endif // #ifdef NMS_SHARED

//
// BUCKET (mit Shared Memory Tiling)
//
#ifdef NMS_BUCKET

// lookup table: bucket index -> (dx, dy) offset along the gradient direction
// 0 = 0°, 1 = 45°, 2 = 90°, 3 = 135°, matches bucket_sobel_filter's encoding
__constant__ int32_t dir_bucket_dx[4] = {1, 1, 0, 1};
__constant__ int32_t dir_bucket_dy[4] = {0, -1, 1, 1};

constexpr int32_t NMS_BUCKET_RADIUS = 1;

__global__ void non_maximum_suppression_bucket(
    const uint8_t* magnitude_buffer,
    const uint8_t* dir_bucket_buffer,
    const int32_t  width,
    const int32_t  height,
    uint8_t*       out_nms_buffer)
{
    // global pixel position in the image
    const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);

    // local thread position within the block (0..BLOCK_SIZE-1)
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);

    // shared memory tile including 1px halo border on each side
    constexpr int32_t tile_size = BLOCK_SIZE + 2 * NMS_BUCKET_RADIUS;
    __shared__ uint8_t tile[tile_size * tile_size];

    // load tile into shared memory (including halo), every thread
    // participates regardless of whether its own (x, y) is in bounds
    for (int32_t j = ty; j < tile_size; j += BLOCK_SIZE)
    {
        for (int32_t i = tx; i < tile_size; i += BLOCK_SIZE)
        {
            const int32_t img_x = static_cast<int32_t>(blockIdx.x * BLOCK_SIZE) + i - NMS_BUCKET_RADIUS;
            const int32_t img_y = static_cast<int32_t>(blockIdx.y * BLOCK_SIZE) + j - NMS_BUCKET_RADIUS;

            tile[j * tile_size + i] = (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height)
                ? magnitude_buffer[img_y * width + img_x]
                : 0;
        }
    }

    __syncthreads();

    // bail out only after every thread has helped fill shared memory
    if (x >= width || y >= height)
        return;

    const uint8_t magnitude = tile[(ty + NMS_BUCKET_RADIUS) * tile_size + (tx + NMS_BUCKET_RADIUS)];
    const uint8_t bucket    = dir_bucket_buffer[y * width + x];

    const int32_t dx = dir_bucket_dx[bucket];
    const int32_t dy = dir_bucket_dy[bucket];

    const uint8_t neighbor_a = tile[(ty + dy + NMS_BUCKET_RADIUS) * tile_size + (tx + dx + NMS_BUCKET_RADIUS)];
    const uint8_t neighbor_b = tile[(ty - dy + NMS_BUCKET_RADIUS) * tile_size + (tx - dx + NMS_BUCKET_RADIUS)];

    out_nms_buffer[y * width + x] = (magnitude >= neighbor_a && magnitude >= neighbor_b) ? magnitude : 0;
}
#endif // #ifdef NMS_BUCKET