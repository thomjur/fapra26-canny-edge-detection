#include <cstdint>
#include <cstdio>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "common.cuh"
#include "gaussian.cuh"
#include "nms.cuh"
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
#define BENCHMARK_RUNS 10

int main(int argc, const char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: ./canny.out <input.jpg> <output.png>\n");
        return -1;
    }

    const char *path     = argv[1];
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

#ifdef WARMUP
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
#endif

  // Gaussian Runs
  float total_h2d = 0, total_kernel = 0, total_d2h = 0;
  GaussianResult gaussian{};
  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    if (i > 0)
      gaussian_cleanup(gaussian);
    gaussian = gaussian_execute(host_src, width, height);
    total_h2d += gaussian.ms_h2d;
    total_kernel += gaussian.ms_kernel;
    total_d2h += gaussian.ms_d2h;
  }

  printf("--- Gaussian Filter (%d runs avg) ---\n", BENCHMARK_RUNS);
  printf("H->D:   %.3f ms\n", total_h2d / BENCHMARK_RUNS);
  printf("Kernel: %.3f ms\n", total_kernel / BENCHMARK_RUNS);
  printf("D->H:   %.3f ms\n", total_d2h / BENCHMARK_RUNS);
  printf("Total:  %.3f ms\n",
         (total_h2d + total_kernel + total_d2h) / BENCHMARK_RUNS);

  //
  // >>> SOBEL FILTER <<<
  //
  // First we manually load preprocessed Gaussian image for testing
  const uint8_t *host_src_processed =
      stbi_load("assets/gaussian.jpg", &width, &height, &channels_in_file, 1);
  if (host_src_processed == nullptr)
    return -1;

  printf("Image loaded : %s\n", "gaussian_img.jpg");
  printf("Size         : %d x %d\n", width, height);
  printf("Channels     : %d (loaded as grayscale)\n", channels_in_file);

#ifdef WARMUP
  {
    SobelResult warm_up = sobel_execute(host_src_processed, width, height);
    printf("--- Sobel Filter (warm-up run) ---\n");
    printf("H->D:   %.3f ms\n", warm_up.ms_h2d);
    printf("Kernel: %.3f ms\n", warm_up.ms_kernel);
    printf("D->H:   %.3f ms\n", warm_up.ms_d2h);
    printf("Total:  %.3f ms\n",
           warm_up.ms_h2d + warm_up.ms_kernel + warm_up.ms_d2h);
    sobel_cleanup(warm_up);
  }
#endif

  // Sobel Runs
  total_h2d = 0, total_kernel = 0, total_d2h = 0;
  SobelResult sobel{};
  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    if (i > 0)
      sobel_cleanup(sobel);
    sobel = sobel_execute(host_src_processed, width, height);
    total_h2d += sobel.ms_h2d;
    total_kernel += sobel.ms_kernel;
    total_d2h += sobel.ms_d2h;
  }

  printf("--- Sobel Filter (%d runs avg) ---\n", BENCHMARK_RUNS);
  printf("H->D:   %.3f ms\n", total_h2d / BENCHMARK_RUNS);
  printf("Kernel: %.3f ms\n", total_kernel / BENCHMARK_RUNS);
  printf("D->H:   %.3f ms\n", total_d2h / BENCHMARK_RUNS);
  printf("Total:  %.3f ms\n",
         (total_h2d + total_kernel + total_d2h) / BENCHMARK_RUNS);

  // TODO

  //
  // >>> NON-MAXIMUM SUPPRESSION <<<
  //

#ifdef WARMUP
    {
        NmsResult warm_up = nms_execute(sobel.host_grad, sobel.host_dir, width, height);
        printf("--- Non-Maximum Suppression (warm-up run) ---\n");
        printf("H->D:   %.3f ms\n", warm_up.ms_h2d);
        printf("Kernel: %.3f ms\n", warm_up.ms_kernel);
        printf("D->H:   %.3f ms\n", warm_up.ms_d2h);
        printf("Total:  %.3f ms\n", warm_up.ms_h2d + warm_up.ms_kernel + warm_up.ms_d2h);
        nms_cleanup(warm_up);
    }
#endif

    // NMS Runs
    total_h2d = 0, total_kernel = 0, total_d2h = 0;
    NmsResult nms{};
    for (int i = 0; i < BENCHMARK_RUNS; i++)
    {
        if (i > 0)
            nms_cleanup(nms);

        nms = nms_execute(sobel.host_grad, sobel.host_dir, width, height);
        total_h2d    += nms.ms_h2d;
        total_kernel += nms.ms_kernel;
        total_d2h    += nms.ms_d2h;
    }

    printf("--- Non-Maximum Suppression (%d runs avg) ---\n", BENCHMARK_RUNS);
    printf("H->D:   %.3f ms\n", total_h2d / BENCHMARK_RUNS);
    printf("Kernel: %.3f ms\n", total_kernel / BENCHMARK_RUNS);
    printf("D->H:   %.3f ms\n", total_d2h / BENCHMARK_RUNS);
    printf("Total:  %.3f ms\n", (total_h2d + total_kernel + total_d2h) / BENCHMARK_RUNS);

  //
  // >>> HYSTERESIS THRESHOLDING <<<
  //

  // TODO

  //
  // write output image
  //

    int32_t write_result = 0;

    // Write Gaussian image
    write_result = stbi_write_png("assets/gaussian.png", width, height, 1, gaussian.host_buffer, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image '%s'\n", "assets/gaussian.png");
    else
        printf("Grayscale written : %s\n", out_path);

    // Write Sobel images
    write_result = stbi_write_png("assets/sobel_grad.png", width, height, 1, sobel.host_grad, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image 'assets/sobel_grad.png'\n");
    else
        printf("Grayscale Sobel gadient image written : assets/sobel_grad.png\n");

    write_result = stbi_write_png("assets/sobel_dir.png", width, height, 1, sobel.host_dir, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image 'assets/sobel_dir.png'\n");
    else
        printf("Grayscale Sobel gadient image written : assets/sobel_dir.png\n");

    // Write NMS image
    write_result = stbi_write_png("assets/nms.png", width, height, 1, nms.host_nms, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image 'assets/nms.png'\n");
    else
        printf("Thinned NMS image written : assets/nms.png\n");


    // Write NMS image (current pipeline output)
    write_result = stbi_write_png(out_path, width, height, 1, nms.host_nms, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image '%s'\n", out_path);
    else
        printf("Pipeline output (NMS) written : %s\n", out_path);

    //
    // clean up
    //

    gaussian_cleanup(gaussian);
    sobel_cleanup(sobel);
    nms_cleanup(nms);

    stbi_image_free(static_cast<void *>(const_cast<uint8_t *>(host_src)));
    stbi_image_free(static_cast<void *>(const_cast<uint8_t *>(host_src_processed)));

    return write_result == 0 ? -1 : 0;
}
