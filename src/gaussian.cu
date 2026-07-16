#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <math_constants.h>

#include "common.cuh"
#include "gaussian.cuh"

//
// >>> GAUSSIAN FILTER <<<
//

//
// HOST
//

GaussianResult gaussian_execute(const uint8_t *host_src, const int32_t width, const int32_t height)
{
    constexpr float sigma        = 1.4f;
    constexpr uint32_t blur_size = BLUR_RADIUS * 2 + 1;
    const size_t img_size        = width * height * sizeof(uint8_t);

    // compute weights on host
    float weights[blur_size * blur_size];
    calculate_gaussian_weights(BLUR_RADIUS, sigma, weights);

    // grid / block layout
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
                  static_cast<uint32_t>(std::ceil(static_cast<float>(height) / BLOCK_SIZE)));

    // host output buffer — pinned for faster D->H transfer if GAUSSIAN_PINNED
#ifdef GAUSSIAN_PINNED
    uint8_t *host_dst = nullptr;
    CUDA_THROW_IF_FAILED(cudaMallocHost(&host_dst, img_size));
#else
    auto *host_dst = static_cast<uint8_t *>(malloc(img_size));
#endif

    // device buffers
    uint8_t *device_src;
    uint8_t *device_dst;
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_src, img_size));
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_dst, img_size));
    CUDA_THROW_IF_FAILED(cudaMemset(device_dst, 0, img_size));

    // upload weights
#ifdef GAUSSIAN_NAIVE
    float *device_weights_buf = nullptr;
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_weights_buf, blur_size * blur_size * sizeof(float))); CUDA_THROW_IF_FAILED(
        cudaMemcpy(device_weights_buf, weights, blur_size * blur_size * sizeof(float), cudaMemcpyHostToDevice));
#endif
#ifdef GAUSSIAN_SHARED
    CUDA_THROW_IF_FAILED(cudaMemcpyToSymbol( device_weights, weights, blur_size * blur_size * sizeof(float)));
#endif

    // timing events
    cudaEvent_t t0, t1, t2, t3;
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t0));
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t1));
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t2));
    CUDA_THROW_IF_FAILED(cudaEventCreate(&t3));

    // H->D
    cudaEventRecord(t0);
    CUDA_THROW_IF_FAILED(cudaMemcpy(device_src, host_src, img_size, cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    // kernel
#ifdef GAUSSIAN_NAIVE
    gaussian_filter<<<grid_dim, block_dim>>>(device_src, width, height, device_weights_buf, BLUR_RADIUS, device_dst);
#endif
#ifdef GAUSSIAN_SHARED
    gaussian_filter<<<grid_dim, block_dim>>>(device_src, width, height, device_dst);
#endif
    cudaEventRecord(t2);

    // D->H
    CUDA_THROW_IF_FAILED(cudaMemcpy(host_dst, device_dst, img_size, cudaMemcpyDeviceToHost));
    cudaEventRecord(t3);
    CUDA_THROW_IF_FAILED(cudaEventSynchronize(t3));

    // collect timings
    GaussianResult result{};
    result.host_buffer = host_dst;
    cudaEventElapsedTime(&result.ms_h2d, t0, t1);
    cudaEventElapsedTime(&result.ms_kernel, t1, t2);
    cudaEventElapsedTime(&result.ms_d2h, t2, t3);

    // clean up device resources
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaEventDestroy(t2);
    cudaEventDestroy(t3);
#ifdef GAUSSIAN_NAIVE
    CUDA_THROW_IF_FAILED(cudaFree(device_weights_buf));
#endif
    CUDA_THROW_IF_FAILED(cudaFree(device_src));
    CUDA_THROW_IF_FAILED(cudaFree(device_dst));

    return result;
}

void gaussian_cleanup(GaussianResult &result)
{
#ifdef GAUSSIAN_PINNED
    cudaFreeHost(result.host_buffer);
#else
    free(result.host_buffer);
#endif
    result.host_buffer = nullptr;
}

void calculate_gaussian_weights(const uint32_t blur_radius, const float sigma, float *out_weights)
{
    const uint32_t blur_size = blur_radius * 2 + 1;
    float weight_sum         = 0.0f;

    // calculate gaussian weights
    for (int32_t y = -static_cast<int32_t>(blur_radius); y <= static_cast<int32_t>(blur_radius); y++)
    {
        for (int32_t x = -static_cast<int32_t>(blur_radius); x <= static_cast<int32_t>(blur_radius); x++)
        {
            const float weight = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            const uint32_t idx = (y + blur_radius) * blur_size + (x + blur_radius);
            out_weights[idx]   = weight;
            weight_sum         += weight;
        }
    }

    // normalize
    for (uint32_t i = 0; i < blur_size * blur_size; i++)
        out_weights[i] /= weight_sum;
}

//
// NAIVE
//

#ifdef GAUSSIAN_NAIVE
__global__ void gaussian_filter(
    const uint8_t *src_buffer,
    const int32_t width,
    const int32_t height,
    const float *weights,
    const uint32_t blur_radius,
    uint8_t *out_dst_buffer)
{
    // global pixel position in the image
    const auto x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;

    // accumulate weighted sum of neighboring pixels (zero-padding at borders)
    float color       = 0.0f;
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
            color                     += pixel * weights[weight_idx];
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
    const uint8_t *src_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_dst_buffer)
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

    // wait until every thread has finished loading before any thread starts
    // reading
    __syncthreads();

    // apply gaussian blur using values from shared memory tile
    float color = 0.0f;
    for (int32_t y_ = -BLUR_RADIUS; y_ <= BLUR_RADIUS; y_++)
    {
        for (int32_t x_ = -BLUR_RADIUS; x_ <= BLUR_RADIUS; x_++)
        {
            const int32_t tile_idx    = (ty + y_ + BLUR_RADIUS) * tile_size + (tx + x_ + BLUR_RADIUS);
            const uint32_t weight_idx = (y_ + BLUR_RADIUS) * (2 * BLUR_RADIUS + 1) + (x_ + BLUR_RADIUS);
            color                     += tile[tile_idx] * device_weights[weight_idx];
        }
    }

    out_dst_buffer[y * width + x] = static_cast<uint8_t>(color);
}
#endif // #ifdef GAUSSIAN_SHARED
