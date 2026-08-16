#include <cstdint>
#include <cstdio>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <math_constants.h>

#include "stb_image_write.h"

// PIPELINE_FUSED is set via -DPIPELINE_FUSED in build_release.sh / build_debug.sh,
// not defined here, since gaussian.cu / sobel.cu / nms.cu / hysteresis.cu are
// separate translation units and wouldn't see a local #define anyway
#include "common.cuh"
#include "gaussian.cuh"
#include "hysteresis.cuh"
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
#define BENCHMARK_RUNS 1
//#define WARMUP

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
    const uint8_t *host_src = stbi_load(path, &width, &height, &channels_in_file, 1);
    if (host_src == nullptr)
        return -1;

    printf("Image loaded : %s\n", path);
    printf("Size         : %d x %d\n", width, height);
    printf("Channels     : %d (loaded as grayscale)\n", channels_in_file);

    // per-stage averages, kept around for the "sum of stages" vs. "fused
    // pipeline" comparison further down (PIPELINE_FUSED block)
    float avg_total_gaussian = 0, avg_total_sobel = 0, avg_total_nms = 0, avg_total_hysteresis = 0;

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
        printf("Total:  %.3f ms\n", warm_up.ms_h2d + warm_up.ms_kernel + warm_up.ms_d2h);
        gaussian_cleanup(warm_up);
    }
#endif

    // Gaussian Runs
    float total_h2d = 0, total_kernel = 0, total_d2h = 0;
    GaussianResult gaussian{};
    for (int i = 0; i < BENCHMARK_RUNS; i++)
    {
        if (i > 0)
            gaussian_cleanup(gaussian);
        gaussian     = gaussian_execute(host_src, width, height);
        total_h2d    += gaussian.ms_h2d;
        total_kernel += gaussian.ms_kernel;
        total_d2h    += gaussian.ms_d2h;
    }

    printf("--- Gaussian Filter (%d runs avg) ---\n", BENCHMARK_RUNS);
    printf("H->D:   %.3f ms\n", total_h2d / BENCHMARK_RUNS);
    printf("Kernel: %.3f ms\n", total_kernel / BENCHMARK_RUNS);
    printf("D->H:   %.3f ms\n", total_d2h / BENCHMARK_RUNS);
    avg_total_gaussian = (total_h2d + total_kernel + total_d2h) / BENCHMARK_RUNS;
    printf("Total:  %.3f ms\n", avg_total_gaussian);

    //
    // >>> SOBEL FILTER <<<
    //
#if false
    // First we manually load preprocessed Gaussian image for testing
    const uint8_t *host_src_processed = stbi_load("assets/gaussian.jpg", &width, &height, &channels_in_file, 1); if (
        host_src_processed == nullptr)
        return -1; printf("Image loaded : %s\n", "gaussian_img.jpg"); printf("Size         : %d x %d\n", width, height);
    printf("Channels     : %d (loaded as grayscale)\n", channels_in_file);
#endif

#ifdef WARMUP
    {
        SobelResult warm_up = sobel_execute(gaussian.host_buffer, width, height);
        printf("--- Sobel Filter (warm-up run) ---\n");
        printf("H->D:   %.3f ms\n", warm_up.ms_h2d);
        printf("Kernel: %.3f ms\n", warm_up.ms_kernel);
        printf("D->H:   %.3f ms\n", warm_up.ms_d2h);
        printf("Total:  %.3f ms\n", warm_up.ms_h2d + warm_up.ms_kernel + warm_up.ms_d2h);
        sobel_cleanup(warm_up);
    }
#endif

    // Sobel Runs
    total_h2d = 0, total_kernel = 0, total_d2h = 0;
    SobelResult sobel{};
    for (int i = 0; i < BENCHMARK_RUNS; i++)
    {
        if (i > 0)
            sobel_cleanup(sobel);
        sobel        = sobel_execute(gaussian.host_buffer, width, height);
        total_h2d    += sobel.ms_h2d;
        total_kernel += sobel.ms_kernel;
        total_d2h    += sobel.ms_d2h;
    }

    printf("--- Sobel Filter (%d runs avg) ---\n", BENCHMARK_RUNS);
    printf("H->D:   %.3f ms\n", total_h2d / BENCHMARK_RUNS);
    printf("Kernel: %.3f ms\n", total_kernel / BENCHMARK_RUNS);
    printf("D->H:   %.3f ms\n", total_d2h / BENCHMARK_RUNS);
    avg_total_sobel = (total_h2d + total_kernel + total_d2h) / BENCHMARK_RUNS;
    printf("Total:  %.3f ms\n", avg_total_sobel);

    //
    // >>> NON-MAXIMUM SUPPRESSION <<<
    //

#ifdef WARMUP
    {
#ifdef NMS_BUCKET
        NmsResult warm_up = nms_execute(sobel.host_grad, sobel.host_dir_bucket, width, height);
#else
        NmsResult warm_up = nms_execute(sobel.host_grad, sobel.host_dir, width, height);
#endif
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

#ifdef NMS_BUCKET
        nms = nms_execute(sobel.host_grad, sobel.host_dir_bucket, width, height);
#else
        nms = nms_execute(sobel.host_grad, sobel.host_dir, width, height);
#endif
        total_h2d    += nms.ms_h2d;
        total_kernel += nms.ms_kernel;
        total_d2h    += nms.ms_d2h;
    }

    printf("--- Non-Maximum Suppression (%d runs avg) ---\n", BENCHMARK_RUNS);
    printf("H->D:   %.3f ms\n", total_h2d / BENCHMARK_RUNS);
    printf("Kernel: %.3f ms\n", total_kernel / BENCHMARK_RUNS);
    printf("D->H:   %.3f ms\n", total_d2h / BENCHMARK_RUNS);
    avg_total_nms = (total_h2d + total_kernel + total_d2h) / BENCHMARK_RUNS;
    printf("Total:  %.3f ms\n", avg_total_nms);

    //
    // >>> HYSTERESIS THRESHOLDING <<<
    //

#ifdef WARMUP
    {
        HysteresisResult warm_up = hysteresis_execute(nms.host_nms, width, height);
        printf("--- Hysteresis Edge Linker (warm-up run) ---\n");
        printf("H->D:   %.3f ms\n", warm_up.ms_h2d);
        printf("Kernel: %.3f ms\n", warm_up.ms_kernel);
        printf("D->H:   %.3f ms\n", warm_up.ms_d2h);
        printf("Total:  %.3f ms\n", warm_up.ms_h2d + warm_up.ms_kernel + warm_up.ms_d2h);
        hysteresis_cleanup(warm_up);
    }
#endif

    // Hysteresis Runs
    total_h2d = 0, total_kernel = 0, total_d2h = 0;
    HysteresisResult hysteresis{};
    for (int i = 0; i < BENCHMARK_RUNS; i++)
    {
        if (i > 0)
            hysteresis_cleanup(hysteresis);
        hysteresis   = hysteresis_execute(nms.host_nms, width, height);
        total_h2d    += hysteresis.ms_h2d;
        total_kernel += hysteresis.ms_kernel;
        total_d2h    += hysteresis.ms_d2h;
    }

    printf("--- Hysteresis Edge linker (%d runs avg) ---\n", BENCHMARK_RUNS);
    printf("H->D:   %.3f ms\n", total_h2d / BENCHMARK_RUNS);
    printf("Kernel: %.3f ms\n", total_kernel / BENCHMARK_RUNS);
    printf("D->H:   %.3f ms\n", total_d2h / BENCHMARK_RUNS);
    avg_total_hysteresis = (total_h2d + total_kernel + total_d2h) / BENCHMARK_RUNS;
    printf("Total:  %.3f ms\n", avg_total_hysteresis);

    //
    // FUSED PIPELINE (end-to-end, single H2D/D2H)
    //
    // Single upload, kernels chained on the same stream with no host sync in
    // between, single download. Compared against the sum of the isolated
    // stage measurements above to quantify the effect Frau Oden pointed out.
    //

#ifdef PIPELINE_FUSED
    {
        const size_t img_size = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uint8_t);
        const size_t dir_size = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(sobel_dir_t);

        // device buffers for the whole chain -- allocated once, reused every run
        uint8_t *d_src, *d_gaussian, *d_grad, *d_nms, *d_hysteresis;
        sobel_dir_t *d_dir;
        int16_t *d_sobel_gx = nullptr;
        int16_t *d_sobel_gy = nullptr;
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_src, img_size));
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_gaussian, img_size));
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_grad, img_size));
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_dir, dir_size));
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_nms, img_size));
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_hysteresis, img_size));
#ifdef SOBEL_SPLIT
        const size_t sobel_component_size = img_size * sizeof(int16_t);
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_sobel_gx, sobel_component_size));
        CUDA_THROW_IF_FAILED(cudaMalloc(&d_sobel_gy, sobel_component_size));
#endif

        // pinned host output buffer -- the one spot where pinned memory still
        // pays off per Frau Odens Punkt: this is the single D2H at the end
        uint8_t *host_out = nullptr;
        CUDA_THROW_IF_FAILED(cudaMallocHost(&host_out, img_size));

        // weights upload once, not part of the per-run timing (same sigma as
        // gaussian_execute(), see gaussian.cu)
        gaussian_prepare_weights(BLUR_RADIUS, 1.4f);

        cudaEvent_t f_start, f_stop;
        CUDA_THROW_IF_FAILED(cudaEventCreate(&f_start));
        CUDA_THROW_IF_FAILED(cudaEventCreate(&f_stop));

#ifdef WARMUP
        {
            cudaEventRecord(f_start);
            CUDA_THROW_IF_FAILED(cudaMemcpyAsync(d_src, host_src, img_size, cudaMemcpyHostToDevice));
            gaussian_launch(d_src, d_gaussian, width, height);
            sobel_launch(d_gaussian, d_grad, d_dir, width, height, d_sobel_gx, d_sobel_gy);
            nms_launch(d_grad, d_dir, width, height, d_nms);
            hysteresis_launch(d_nms, d_hysteresis, width, height);
            CUDA_THROW_IF_FAILED(cudaMemcpyAsync(host_out, d_hysteresis, img_size, cudaMemcpyDeviceToHost));
            cudaEventRecord(f_stop);
            CUDA_THROW_IF_FAILED(cudaEventSynchronize(f_stop));

            float ms = 0;
            cudaEventElapsedTime(&ms, f_start, f_stop);
            printf("--- Fused Pipeline (warm-up run) ---\n");
            printf("Total: %.3f ms\n", ms);
        }
#endif

        float total_fused = 0;
        for (int i = 0; i < BENCHMARK_RUNS; i++)
        {
            cudaEventRecord(f_start);

            CUDA_THROW_IF_FAILED(cudaMemcpyAsync(d_src, host_src, img_size, cudaMemcpyHostToDevice));
            gaussian_launch(d_src, d_gaussian, width, height);
            sobel_launch(d_gaussian, d_grad, d_dir, width, height, d_sobel_gx, d_sobel_gy);
            nms_launch(d_grad, d_dir, width, height, d_nms);
            hysteresis_launch(d_nms, d_hysteresis, width, height);
            CUDA_THROW_IF_FAILED(cudaMemcpyAsync(host_out, d_hysteresis, img_size, cudaMemcpyDeviceToHost));

            cudaEventRecord(f_stop);
            CUDA_THROW_IF_FAILED(cudaEventSynchronize(f_stop));

            float ms = 0;
            cudaEventElapsedTime(&ms, f_start, f_stop);
            total_fused += ms;
        }

        const float avg_fused  = total_fused / BENCHMARK_RUNS;
        const float sum_stages = avg_total_gaussian + avg_total_sobel + avg_total_nms + avg_total_hysteresis;

        printf("--- Fused Pipeline (%d runs avg) ---\n", BENCHMARK_RUNS);
        printf("Total:                 %.3f ms\n", avg_fused);
        printf("Sum of isolated stages: %.3f ms\n", sum_stages);
        printf("Overhead removed:       %.3f ms (%.1f%%)\n", sum_stages - avg_fused,
               100.0f * (sum_stages - avg_fused) / sum_stages);

        // write fused pipeline output for a sanity check against assets/hysteresis.png
        stbi_write_png("assets/hysteresis_fused.png", width, height, 1, host_out, width * 1);

        // clean up
        CUDA_THROW_IF_FAILED(cudaFree(d_src));
        CUDA_THROW_IF_FAILED(cudaFree(d_gaussian));
        CUDA_THROW_IF_FAILED(cudaFree(d_grad));
        CUDA_THROW_IF_FAILED(cudaFree(d_dir));
        CUDA_THROW_IF_FAILED(cudaFree(d_nms));
        CUDA_THROW_IF_FAILED(cudaFree(d_hysteresis));
#ifdef SOBEL_SPLIT
        CUDA_THROW_IF_FAILED(cudaFree(d_sobel_gx));
        CUDA_THROW_IF_FAILED(cudaFree(d_sobel_gy));
#endif
        CUDA_THROW_IF_FAILED(cudaFreeHost(host_out));
        cudaEventDestroy(f_start);
        cudaEventDestroy(f_stop);
    }
#endif // #ifdef PIPELINE_FUSED

    //
    // write output image
    //

    int32_t write_result = 0;

    // Write Gaussian image
    write_result = stbi_write_png("assets/gaussian.png", width, height, 1, gaussian.host_buffer, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image '%s'\n", "assets/gaussian.png");
    else
        printf("Grayscale written : %s\n", "assets/gaussian.png");

    // Write Sobel images
    write_result = stbi_write_png("assets/sobel_grad.png", width, height, 1, sobel.host_grad, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image 'assets/sobel_grad.png'\n");
    else
        printf("Grayscale Sobel gadient image written : assets/sobel_grad.png\n");

#ifdef SOBEL_BUCKET
    {
        // scale bucket index (0-3) to visible grayscale steps (0/85/170/255)
        auto *dir_vis = static_cast<uint8_t *>(malloc(width * height));
        for (int32_t p = 0; p < width * height; p++)
            dir_vis[p] = static_cast<uint8_t>(sobel.host_dir_bucket[p] * 85);
        write_result = stbi_write_png("assets/sobel_dir.png", width, height, 1, dir_vis, width * 1);
        free(dir_vis);
    }
#else
    {
        // scale radians [-pi, pi] to [0, 255] for visualization
        auto *dir_vis = static_cast<uint8_t *>(malloc(width * height));
        for (int32_t p = 0; p < width * height; p++)
            dir_vis[p] = static_cast<uint8_t>((sobel.host_dir[p] + CUDART_PI_F) / (2.0f * CUDART_PI_F) * 255.0f);
        write_result = stbi_write_png("assets/sobel_dir.png", width, height, 1, dir_vis, width * 1);
        free(dir_vis);
    }
#endif

    // Write NMS image
    write_result = stbi_write_png("assets/nms.png", width, height, 1, nms.host_nms, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image 'assets/nms.png'\n");
    else
        printf("Thinned NMS image written : assets/nms.png\n");

    // Write Hysteresis image (fixed intermediate debug copy)
    write_result = stbi_write_png("assets/hysteresis.png", width, height, 1, hysteresis.hysteresis_buffer, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image 'assets/hysteresis.png'\n");
    else
        printf("Hysteresis linked edge image written : assets/hysteresis.png\n");

    // Write Hysteresis image (current pipeline output)
    write_result = stbi_write_png(out_path, width, height, 1, hysteresis.hysteresis_buffer, width * 1);
    if (write_result == 0)
        fprintf(stderr, "Error: could not write image '%s'\n", out_path);
    else
        printf("Pipeline output (Hysteresis) written : %s\n", out_path);

    //
    // clean up
    //

    gaussian_cleanup(gaussian);
    sobel_cleanup(sobel);
    nms_cleanup(nms);
    hysteresis_cleanup(hysteresis);

    stbi_image_free(static_cast<void *>(const_cast<uint8_t *>(host_src)));

    return write_result == 0 ? -1 : 0;
}
