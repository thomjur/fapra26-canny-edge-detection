#include <cstdint>
#include <cstdio>
#include <cuda.h>

#include "common.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//
// Canny Edge Detection Pipeline
//
// Step 1: Gaussian Blur
//         - reduce noise before edge detection
//
// Step 2: Gradient Calculation (Sobel)
//         - compute gradient magnitude and direction for each pixel
//
// Step 3: Non-Maximum Suppression
//         - thin edges to 1 pixel width
//
// Step 4: Hysteresis Thresholding
//         - two thresholds (high/low) to determine real edges
//

/**
 * @brief Applies a gaussian blur to a grayscale image.
 *
 * @param src_buffer    Input image buffer (grayscale, 1 byte per pixel).
 * @param width         Image width in pixels.
 * @param height        Image height in pixels.
 * @param weights       Precomputed gaussian weights on device. See calculate_gaussian_weights().
 * @param blur_radius   Radius of the blur kernel (e.g. 1 = 3x3, 2 = 5x5).
 * @param out_dst_buffer Output image buffer (grayscale, 1 byte per pixel).
 */
__global__ void gaussian_filter(
    const uint8_t* src_buffer,
    const int32_t width,
    const int32_t height,
    const float* weights,
    const uint32_t blur_radius,
    uint8_t* out_dst_buffer)
{
    // TODO: shared memory optimization

    const auto x = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto y = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;

    float color = 0.0f;
    const auto radius = static_cast<int32_t>(blur_radius);
    for (int32_t y_ = -radius; y_ <= radius; y_++)
    {
        for (int32_t x_ = -radius; x_ <= radius; x_++)
        {
            uint8_t pixel = 0;
            if (x + x_ >= 0 && x + x_ < width && y + y_ >= 0 && y + y_ < height)
                pixel = src_buffer[(y + y_) * width + (x + x_)];

            const uint32_t weight_idx = (y_ + radius) * (blur_radius * 2 + 1) + (x_ + radius);
            color += pixel * weights[weight_idx];
        }
    }

    out_dst_buffer[y * width + x] = static_cast<uint8_t>(color);
}

/**
 * @brief Calculates normalized gaussian weights for a given blur radius and sigma.
 *
 * @details Larger sigma = flatter curve = more blur.
 *          Smaller sigma = sharper curve = less blur.
 *
 *          Example for blur_radius=1, blur_size=3, sigma=1.0f:
 *
 *          +--------+--------+--------+
 *          | 0.0894 | 0.1201 | 0.0894 |   y = -1
 *          +--------+--------+--------+
 *          | 0.1201 | 0.1615 | 0.1201 |   y =  0
 *          +--------+--------+--------+
 *          | 0.0894 | 0.1201 | 0.0894 |   y = +1
 *          +--------+--------+--------+
               x=-1     x=0     x=+1
 *
 * @param blur_radius Radius of the blur kernel (e.g. 1 = 3x3, 2 = 5x5)
 * @param sigma       Controls the spread of the gaussian bell curve.
 * @param out_weights Output buffer for the weights. Must be at least
 *                    (blur_radius * 2 + 1)² floats large.
 */
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


int main(int argc, const char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: ./canny.out <input.jpg> <output.png>\n");
        return -1;
    }

    const char* path     = argv[1];
    const char* out_path = argv[2];

    //
    // load image
    //

    int32_t width{};
    int32_t height{};
    int32_t channels_in_file{};
    const uint8_t* host_src_image_buffer = stbi_load(path, &width, &height, &channels_in_file, 1);
    if (host_src_image_buffer == nullptr)
        return -1;

    printf("Image loaded : %s\n", path);
    printf("Size         : %d x %d\n", width, height);
    printf("Channels     : %d (loaded as grayscale)\n", channels_in_file);

    //
    // blur weights
    //

    constexpr uint32_t blur_radius = 9;
    constexpr float sigma = 6.3f;
    constexpr uint32_t blur_size = blur_radius * 2 + 1;
    float weights[blur_size * blur_size];
    calculate_gaussian_weights(blur_radius, sigma, weights);

    //
    // prepare cuda resources
    //

    constexpr uint32_t BLOCK_SIZE = 16;
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
    static_cast<uint32_t>(std::ceil(static_cast<float>(width) / static_cast<float>(BLOCK_SIZE))),
    static_cast<uint32_t>(std::ceil(static_cast<float>(height) / static_cast<float>(BLOCK_SIZE))));

    uint8_t* device_src_buffer;
    uint8_t* device_dst_buffer;
    float* device_weights = nullptr;
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_src_buffer, width * height * sizeof(uint8_t)));
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_dst_buffer, width * height * sizeof(uint8_t)));
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_weights, blur_size * blur_size * sizeof(float)));

    CUDA_THROW_IF_FAILED(cudaMemcpy(device_src_buffer, host_src_image_buffer, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_THROW_IF_FAILED(cudaMemcpy(device_weights, weights, blur_size * blur_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_THROW_IF_FAILED(cudaMemset(device_dst_buffer, 0, width * height * sizeof(uint8_t)));

    //
    // kernels (blur)
    //

    gaussian_filter<<<grid_dim, block_dim>>>(
        device_src_buffer,
        width,
        height,
        device_weights,
        blur_radius,
        device_dst_buffer);
    CUDA_THROW_IF_FAILED(cudaDeviceSynchronize());

    //
    // get output
    //

    auto* host_blurred_buffer = static_cast<uint8_t*>(malloc(width * height * sizeof(uint8_t)));
    CUDA_THROW_IF_FAILED(cudaMemcpy(
        host_blurred_buffer,
        device_dst_buffer,
        width * height * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));
    CUDA_THROW_IF_FAILED(cudaFree(device_weights));

    //
    // write image
    //

    const int32_t write_result = stbi_write_png(
        out_path,
        width,
        height,
        1,
        host_blurred_buffer,
        width * 1);
    if (write_result == 0)
    {
        fprintf(stderr, "Error: could not write image '%s'\n", out_path);
        stbi_image_free(static_cast<void*>(const_cast<uint8_t*>(host_blurred_buffer)));
        return -1;
    }
    printf("Grayscale written : %s\n", out_path);

    //
    // clean up
    //

    free(host_blurred_buffer);
    stbi_image_free(static_cast<void*>(const_cast<uint8_t*>(host_src_image_buffer)));
    CUDA_THROW_IF_FAILED(cudaFree(device_src_buffer));
    CUDA_THROW_IF_FAILED(cudaFree(device_dst_buffer));

    return 0;
}