#include <cstdint>
#include <cstdio>
#include <cuda.h>

#include "common.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "gaussian.cuh"

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
    // LOAD IMAGE
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
    // >>> GAUSSIAN FILTER <<<
    //

    // blur weights
    constexpr float sigma = 1.4f;
    constexpr uint32_t blur_size = BLUR_RADIUS * 2 + 1;
    float weights[blur_size * blur_size];
    calculate_gaussian_weights(BLUR_RADIUS, sigma, weights);

    // prepare cuda resources
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
        static_cast<uint32_t>(std::ceil(static_cast<float>(width) / static_cast<float>(BLOCK_SIZE))),
        static_cast<uint32_t>(std::ceil(static_cast<float>(height) / static_cast<float>(BLOCK_SIZE))));

    uint8_t* device_src_buffer;
    uint8_t* device_dst_buffer;
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_src_buffer, width * height * sizeof(uint8_t)));
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_dst_buffer, width * height * sizeof(uint8_t)));
    CUDA_THROW_IF_FAILED(cudaMemcpy(device_src_buffer, host_src_image_buffer, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_THROW_IF_FAILED(cudaMemset(device_dst_buffer, 0, width * height * sizeof(uint8_t)));

#ifdef GAUSSIAN_NAIVE
    float* device_weights = nullptr;
    CUDA_THROW_IF_FAILED(cudaMalloc(&device_weights, blur_size * blur_size * sizeof(float)));
    CUDA_THROW_IF_FAILED(cudaMemcpy(device_weights, weights, blur_size * blur_size * sizeof(float), cudaMemcpyHostToDevice));
#endif // #ifdef GAUSSIAN_NAIVE

#ifdef GAUSSIAN_SHARED
    CUDA_THROW_IF_FAILED(cudaMemcpyToSymbol(device_weights, weights, blur_size * blur_size * sizeof(float)));
#endif // #ifdef GAUSSIAN_SHARED

    // kernel
#ifdef GAUSSIAN_NAIVE
    gaussian_filter<<<grid_dim, block_dim>>>(
        device_src_buffer,
        width,
        height,
        device_weights,
        BLUR_RADIUS,
        device_dst_buffer);
#endif // #ifdef GAUSSIAN_NAIVE

#ifdef GAUSSIAN_SHARED
    gaussian_filter<<<grid_dim, block_dim>>>(
        device_src_buffer,
        width,
        height,
        device_dst_buffer);
#endif // #ifdef GAUSSIAN_SHARED

    CUDA_THROW_IF_FAILED(cudaDeviceSynchronize());

    //
    //
    // NEXT
    //
    //

    //
    // GET OUTPUT
    //

    auto* host_blurred_buffer = static_cast<uint8_t*>(malloc(width * height * sizeof(uint8_t)));
    CUDA_THROW_IF_FAILED(cudaMemcpy(
        host_blurred_buffer,
        device_dst_buffer,
        width * height * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));
#ifdef GAUSSIAN_NAIVE
    CUDA_THROW_IF_FAILED(cudaFree(device_weights));
#endif

    //
    // WRITE IMAGE
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
    // CLEAN UP
    //

    free(host_blurred_buffer);
    stbi_image_free(static_cast<void*>(const_cast<uint8_t*>(host_src_image_buffer)));
    CUDA_THROW_IF_FAILED(cudaFree(device_src_buffer));
    CUDA_THROW_IF_FAILED(cudaFree(device_dst_buffer));

    return 0;
}