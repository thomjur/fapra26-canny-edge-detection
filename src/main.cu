#include <cstdint>
#include <cstdio>
#include <cuda.h>

#include "common.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "gaussian.cuh"
#include "sobel.cuh"

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

// 200 runs for stable avg — reduce to ~10 when profiling with NCU
#define GAUSSIAN_BENCHMARK_RUNS 1
int main(int argc, const char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: ./canny.out <input.jpg> <output.png>\n");
    return -1;
  }

  const char *path = argv[1];
  const char *out_path = argv[2];

  // load image as grayscale
  int32_t width{}, height{}, channels_in_file{};
  const uint8_t *host_src =
      stbi_load(path, &width, &height, &channels_in_file, 1);
  if (host_src == nullptr)
    return -1;

  printf("Image loaded : %s\n", path);
  printf("Size         : %d x %d\n", width, height);
  printf("Channels     : %d (loaded as grayscale)\n", channels_in_file);

  //
  // >>> GAUSSIAN FILTER <<<
  //

  // Warm-up
  {
    GaussianResult warm_up = gaussian_execute(host_src, width, height);
    printf("--- Gaussian Filter (warm-up run) ---\n");
    printf("H->D:   %.3f ms\n", warm_up.ms_h2d);
    printf("Kernel: %.3f ms\n", warm_up.ms_kernel);
    printf("D->H:   %.3f ms\n", warm_up.ms_d2h);
    printf("Total:  %.3f ms\n",
           warm_up.ms_h2d + warm_up.ms_kernel + warm_up.ms_d2h);
    gaussian_cleanup(warm_up);
  }

  // Gaussian Runs
  float total_h2d = 0, total_kernel = 0, total_d2h = 0;
  GaussianResult gaussian{};
  for (int i = 0; i < GAUSSIAN_BENCHMARK_RUNS; i++) {
    if (i > 0)
      gaussian_cleanup(gaussian);
    gaussian = gaussian_execute(host_src, width, height);
    total_h2d += gaussian.ms_h2d;
    total_kernel += gaussian.ms_kernel;
    total_d2h += gaussian.ms_d2h;
  }

  printf("--- Gaussian Filter (%d runs avg) ---\n", GAUSSIAN_BENCHMARK_RUNS);
  printf("H->D:   %.3f ms\n", total_h2d / GAUSSIAN_BENCHMARK_RUNS);
  printf("Kernel: %.3f ms\n", total_kernel / GAUSSIAN_BENCHMARK_RUNS);
  printf("D->H:   %.3f ms\n", total_d2h / GAUSSIAN_BENCHMARK_RUNS);
  printf("Total:  %.3f ms\n",
         (total_h2d + total_kernel + total_d2h) / GAUSSIAN_BENCHMARK_RUNS);

  //
  // >>> SOBEL FILTER <<<
  //
  // Currently only naive implementation of single run Sobel filter
  // First we manually load preprocessed Gaussian image for testing
  const uint8_t *host_src_processed =
      stbi_load("assets/gaussian.jpg", &width, &height, &channels_in_file, 1);
  if (host_src_processed == nullptr)
    return -1;

  printf("Image loaded : %s\n", "gaussian_img.jpg");
  printf("Size         : %d x %d\n", width, height);
  printf("Channels     : %d (loaded as grayscale)\n", channels_in_file);

  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim(
      static_cast<uint32_t>(std::ceil(static_cast<float>(width) / BLOCK_SIZE)),
      static_cast<uint32_t>(
          std::ceil(static_cast<float>(height) / BLOCK_SIZE)));
  const size_t img_size = width * height * sizeof(uint8_t);
  // device buffers
  uint8_t *device_src;
  uint8_t *device_gradient;
  float *device_direction;
  auto *host_gradient = static_cast<uint8_t *>(malloc(img_size));
  auto *host_direction =
      static_cast<float *>(malloc(width * height * sizeof(float)));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_src, img_size));
  CUDA_THROW_IF_FAILED(cudaMalloc(&device_gradient, img_size));
  CUDA_THROW_IF_FAILED(
      cudaMalloc(&device_direction, width * height * sizeof(float)));
  CUDA_THROW_IF_FAILED(cudaMemset(device_gradient, 0, img_size));
  // Copy image H -> D
  CUDA_THROW_IF_FAILED(cudaMemcpy(device_src, host_src_processed, img_size,
                                  cudaMemcpyHostToDevice));
  // Running Kernel
  naive_sobel_filter<<<grid_dim, block_dim>>>(
      device_src, width, height, device_gradient, device_direction);

  // Copy image D -> H
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_gradient, device_gradient, img_size,
                                  cudaMemcpyDeviceToHost));
  CUDA_THROW_IF_FAILED(cudaMemcpy(host_direction, device_direction, img_size,
                                  cudaMemcpyDeviceToHost));
  // Save image on host
  int32_t write_result = stbi_write_png("assets/sobel.jpg", width, height, 1,
                                        host_gradient, width * 1);
  if (write_result == 0)
    fprintf(stderr, "Error: could not write image '%s'\n", out_path);
  else
    printf("Grayscale Sobel image written : %s\n", out_path);

  // TODO

  //
  // >>> NON-MAXIMUM SUPPRESSION <<<
  //

  // TODO

  //
  // >>> HYSTERESIS THRESHOLDING <<<
  //

  // TODO

  //
  // write output image
  //

  write_result = stbi_write_png(out_path, width, height, 1,
                                gaussian.host_buffer, width * 1);
  if (write_result == 0)
    fprintf(stderr, "Error: could not write image '%s'\n", out_path);
  else
    printf("Grayscale written : %s\n", out_path);

  //
  // clean up
  //

  gaussian_cleanup(gaussian);
  stbi_image_free(static_cast<void *>(const_cast<uint8_t *>(host_src)));

  return write_result == 0 ? -1 : 0;
}
