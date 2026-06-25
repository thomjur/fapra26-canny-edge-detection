#include <cstdint>

#include "gaussian.cuh"
#include "common.cuh"

//
// >>> GAUSSIAN FILTER <<<
//

#ifndef GAUSSIAN_OPT
    #define GAUSSIAN_OPT 0
#endif

#if GAUSSIAN_OPT == 0
    #define GAUSSIAN_NAIVE
#elif GAUSSIAN_OPT == 1
    #define GAUSSIAN_SHARED
#elif GAUSSIAN_OPT == 2
    #define GAUSSIAN_PINNED
#elif GAUSSIAN_OPT == 3
    #define GAUSSIAN_WARP
#else
    #error "GAUSSIAN_OPT must be 0, 1, 2, or 3"
#endif

//
// NAIVE
//

#ifdef GAUSSIAN_NAIVE
__global__ void gaussian_filter(
    const uint8_t* src_buffer,
    const int32_t width,
    const int32_t height,
    const float* weights,
    const uint32_t blur_radius,
    uint8_t* out_dst_buffer)
{
    // global pixel position in the image
    const auto x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;

    // accumulate weighted sum of neighboring pixels (zero-padding at borders)
    float color = 0.0f;
    const auto radius = static_cast<int32_t>(blur_radius);
    for (int32_t y_ = -radius; y_ <= radius; y_++)
    {
        for (int32_t x_ = -radius; x_ <= radius; x_++)
        {
            // clamp to image border (zero-padding)
            uint8_t pixel = 0;
            if (x + x_ >= 0 && x + x_ < width && y + y_ >= 0 && y + y_ < height)
                pixel = src_buffer[(y + y_) * width + (x + x_)];

            // each neighbor is weighted by the precomputed gaussian weight
            const uint32_t weight_idx = (y_ + radius) * (blur_radius * 2 + 1) + (x_ + radius);
            color += pixel * weights[weight_idx];
        }
    }

    out_dst_buffer[y * width + x] = static_cast<uint8_t>(color);
}
#endif // #ifdef GAUSSIAN_NAIVE

//
// SHARED
//

#ifdef GAUSSIAN_SHARED
__constant__ float device_weights[(2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1)];

__global__ void gaussian_filter(
    const uint8_t* src_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t* out_dst_buffer)
{
    // global pixel position in the image
    const int32_t x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;

    // local thread position within the block (0..BLOCK_SIZE-1)
    // used to index into the shared memory tile
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ty = static_cast<int32_t>(threadIdx.y);

    // shared memory tile including halo border (BLUR_RADIUS pixels on each side)
    constexpr int32_t tile_size = BLOCK_SIZE + 2 * BLUR_RADIUS;
    __shared__ uint8_t tile[tile_size * tile_size];

    // load tile into shared memory (including halo)
    // each thread loads one or more pixels depending on tile size vs block size
    for (int32_t j = ty; j < tile_size; j += BLOCK_SIZE)
    {
        for (int32_t i = tx; i < tile_size; i += BLOCK_SIZE)
        {
            // map tile position back to image coordinates (subtract halo offset)
            const int32_t img_x = static_cast<int32_t>(blockIdx.x * BLOCK_SIZE) + i - BLUR_RADIUS;
            const int32_t img_y = static_cast<int32_t>(blockIdx.y * BLOCK_SIZE) + j - BLUR_RADIUS;

            // zero-padding for pixels outside image bounds
            tile[j * tile_size + i] = (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height)
                ? src_buffer[img_y * width + img_x]
                : 0;
        }
    }

    // wait until every thread has finished loading before any thread starts reading
    __syncthreads();

    // apply gaussian blur using values from shared memory tile
    float color = 0.0f;
    for (int32_t y_ = -BLUR_RADIUS; y_ <= BLUR_RADIUS; y_++)
    {
        for (int32_t x_ = -BLUR_RADIUS; x_ <= BLUR_RADIUS; x_++)
        {
            const int32_t tile_idx = (ty + y_ + BLUR_RADIUS) * tile_size + (tx + x_ + BLUR_RADIUS);
            const uint32_t weight_idx = (y_ + BLUR_RADIUS) * (2 * BLUR_RADIUS + 1) + (x_ + BLUR_RADIUS);
            color += tile[tile_idx] * device_weights[weight_idx];
        }
    }

    out_dst_buffer[y * width + x] = static_cast<uint8_t>(color);
}
#endif // #ifdef GAUSSIAN_SHARED

void calculate_gaussian_weights(
    const uint32_t blur_radius,
    const float sigma,
    float* out_weights)
{
    const uint32_t blur_size = blur_radius * 2 + 1;
    float weight_sum = 0.0f;

    // calculate gaussian weights
    for (int32_t y = -static_cast<int32_t>(blur_radius); y <= static_cast<int32_t>(blur_radius); y++)
    {
        for (int32_t x = -static_cast<int32_t>(blur_radius); x <= static_cast<int32_t>(blur_radius); x++)
        {
            const float weight = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            const uint32_t idx = (y + blur_radius) * blur_size + (x + blur_radius);
            out_weights[idx] = weight;
            weight_sum += weight;
        }
    }

    // normalize
    for (uint32_t i = 0; i < blur_size * blur_size; i++)
        out_weights[i] /= weight_sum;
}