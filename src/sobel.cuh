#pragma once

#include <cmath>
#include <cstdint>

#define SOBEL_KERNEL_RADIUS 1

#ifndef SOBEL_OPT
    #define SOBEL_OPT 0
#endif

#if SOBEL_OPT == 0
    #define SOBEL_NAIVE
#elif SOBEL_OPT == 1
    #define SOBEL_SHARED
#elif SOBEL_OPT == 2
    #define SOBEL_OPTIMIZED
#elif SOBEL_OPT == 3
    #define SOBEL_PINNED
#elif SOBEL_OPT == 4
    #define SOBEL_BUCKET
#else
    #error "SOBEL_OPT must be 0, 1, 2, 3 or 4"
#endif

#ifdef SOBEL_BUCKET
    using sobel_dir_t = uint8_t;
#else
    using sobel_dir_t = float;
#endif

#if defined(SOBEL_PINNED) || (defined(NMS_OPT) && NMS_OPT == 2)
    #define SOBEL_HOST_PINNED
#endif

/**
 * @brief CUDA kernel for Sobel edge detection.
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_grad_buffer Output buffer for gradient magnitudes (device memory,
 * 8-bit).
 * @param out_dir_buffer Output buffer for gradient directions (device memory,
 * float).
 */
__global__ void naive_sobel_filter(
    const uint8_t *src_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_grad_buffer,
    float *out_dir_buffer);

/**
 * @brief Optimized CUDA kernel for Sobel edge detection.
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_grad_buffer Output buffer for gradient magnitudes (device memory,
 * 8-bit).
 * @param out_dir_buffer Output buffer for gradient directions (device memory,
 * float).
 */
__global__ void sharedm_sobel_filter(
    const uint8_t *src_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_grad_buffer,
    float *out_dir_buffer);

/**
 * @brief Optimized CUDA kernel for Sobel edge detection. Since the shared
 * memory version did not improve the runtime, this version uses other
 * optimizations.
 *
 * We use two main optimization:
 *
 * 1. We use a simpler version to calculate the gradient value.
 * 2. We unroll the Sobel kernel loop
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_grad_buffer Output buffer for gradient magnitudes (device memory,
 * 8-bit).
 * @param out_dir_buffer Output buffer for gradient directions (device memory,
 * float).
 */
__global__ void optimized_sobel_filter(
    const uint8_t *src_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_grad_buffer,
    float *out_dir_buffer);

/**
 * @brief Sobel kernel, builds on SOBEL_OPTIMIZED but skips atan2f() by
 * bucketing the direction directly from gx/gy via sign and slope checks.
 *
 * @param src_buffer Input image (device, grayscale, 8-bit).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param out_grad_buffer Output gradient magnitude (device, 8-bit).
 * @param out_dir_buffer Output direction bucket (device, 0-3 = 0/45/90/135°).
 */
__global__ void bucket_sobel_filter(
    const uint8_t *src_buffer,
    const int32_t width,
    const int32_t height,
    uint8_t *out_grad_buffer,
    uint8_t *out_dir_buffer);

/**
 * @struct SobelResult
 * @brief Holds the gradient and direction images from sobel_execute().
 *
 * The host_grad and host_dir buffers must be freed using sobel_cleanup()
 * after use to avoid memory leaks.
 */
struct SobelResult
{
    uint8_t *host_grad;       ///< Grayscale image with gradient values (0-255).
    float *host_dir;          ///< Image with gradient direction values in radians (atan2).
    uint8_t *host_dir_bucket; ///< Gradient direction bucket (0-3). Valid only if SOBEL_BUCKET is active.
    float ms_h2d;             ///< Time taken for host-to-device transfer in milliseconds.
    float ms_kernel;          ///< Time taken for kernel execution in milliseconds.
    float ms_d2h;             ///< Time taken for device-to-host transfer in milliseconds.
};

/**
 * @brief Executes the Sobel edge detection on a grayscale image.
 *
 * Allocates device memory, launches the Sobel kernel, and copies results back
 * to the host. Also measures timing for H2D, kernel execution, and D2H
 * transfers.
 *
 * @param host_src Input grayscale image (1 byte per pixel).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @return SobelResult Structure containing gradient/direction buffers and
 * timing info. Use sobel_cleanup() to free the host buffers.
 */
SobelResult sobel_execute(const uint8_t *host_src, int32_t width, int32_t height);

/**
 * @brief Releases the host buffers allocated by sobel_execute().
 *
 * @param result SobelResult returned by sobel_execute().
 *              After cleanup, host_grad and host_dir are set to nullptr.
 */
void sobel_cleanup(SobelResult &result);

//
// >>> FUSED PIPELINE SUPPORT <<<
//

#ifdef PIPELINE_FUSED

/// @brief Launches only the sobel kernel — no cudaMalloc/cudaMemcpy/cudaFree.
///        For the fused pipeline: d_src/d_grad/d_dir must already be on the
///        device (allocated by caller).
///
/// @param d_src   Input image, already on device.
/// @param d_grad  Output gradient magnitude buffer, already on device.
/// @param d_dir   Output direction buffer, already on device.
///                Type is sobel_dir_t (uint8_t for SOBEL_BUCKET, float otherwise).
void sobel_launch(
    const uint8_t *d_src,
    uint8_t *d_grad,
    sobel_dir_t *d_dir,
    int32_t width,
    int32_t height,
    cudaStream_t stream = 0);

#endif // #ifdef PIPELINE_FUSED
