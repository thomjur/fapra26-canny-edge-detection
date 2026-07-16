#pragma once

#include <cstdint>
#include "sobel.cuh"

//
// >>> NON-MAXIMUM SUPPRESSION <<<
//

// Two independent axes:
//
//   NMS_OPT       optimization level of the kernel / memory path
//     0 - naive:  each thread reads magnitude neighbors directly from global memory
//     1 - shared: magnitude tile (with 1px halo) loaded cooperatively into shared memory
//     2 - pinned: shared-memory kernel, but the host buffers are page-locked
//                 (allocated with cudaMallocHost, inputs already pinned by the
//                 sobel stage) so H2D/D2H run via DMA without staging copies
//
//   SOBEL_BUCKET  direction representation, inherited from the sobel stage
//     unset:      direction is a float buffer in radians -> quantize per pixel
//     set:        direction is a pre-quantized uint8_t bucket (0-3) -> constant LUT,
//                 no angle computation in this kernel at all
//
// Every optimization level exists in both a float and a bucket flavor.

#ifndef NMS_OPT
    #define NMS_OPT 0
#endif

#if NMS_OPT == 0
    #define NMS_NAIVE
#elif NMS_OPT == 1
    #define NMS_SHARED
#elif NMS_OPT == 2
    #define NMS_SHARED   // pinned reuses the shared-memory kernel
    #define NMS_PINNED
#else
    #error "NMS_OPT must be 0 (naive), 1 (shared) or 2 (pinned)"
#endif

#ifdef SOBEL_BUCKET
#define NMS_BUCKET
#endif

/// Direction element type: inherited from the sobel stage.
using nms_dir_t = sobel_dir_t;

/**
 * @struct NmsResult
 * @brief Holds the thinned edge image from nms_execute().
 *
 * The host_nms buffer must be freed using nms_cleanup() after use to avoid
 * memory leaks.
 */
struct NmsResult
{
    uint8_t *host_nms; ///< Thinned gradient magnitude image (0-255).
    float ms_h2d;      ///< Time taken for host-to-device transfer in milliseconds.
    float ms_kernel;   ///< Time taken for kernel execution in milliseconds.
    float ms_d2h;      ///< Time taken for device-to-host transfer in milliseconds.
};

/**
 * @brief Executes non-maximum suppression on the Sobel gradient/direction images.
 *
 * Allocates device memory, uploads the magnitude and direction buffers,
 * launches the NMS kernel for the active optimization level, and copies the
 * thinned result back to the host.
 *
 * @param host_magnitude Gradient magnitude image from SobelResult::host_grad.
 * @param host_direction Gradient direction from SobelResult::host_dir (float radians),
 *                       or SobelResult::host_dir_bucket (uint8_t, 0-3) if SOBEL_BUCKET
 *                       is active. See nms_dir_t.
 * @param width          Image width in pixels.
 * @param height         Image height in pixels.
 * @return               NmsResult containing the thinned edge buffer and timing info.
 *                       Use nms_cleanup() to free the host buffer.
 */
NmsResult nms_execute(const uint8_t *host_magnitude, const nms_dir_t *host_direction, int32_t width, int32_t height);

/**
 * @brief Releases the host buffer allocated by nms_execute().
 *
 * @param result NmsResult returned by nms_execute().
 *               After cleanup, host_nms is set to nullptr.
 */
void nms_cleanup(NmsResult &result);

//
// Kernels: {naive, shared} x {float direction, direction bucket}
//

/**
 * @brief Naive NMS. Each thread reads its two interpolation neighbors straight
 * from global memory and quantizes the gradient angle itself.
 *
 * @param magnitude_buffer Gradient magnitude from the sobel filter.
 * @param direction_buffer Gradient direction in radians.
 * @param width            Image width in pixels.
 * @param height           Image height in pixels.
 * @param out_nms_buffer   Output: thinned magnitude image, same size as magnitude_buffer.
 */
__global__ void non_maximum_suppression(
    const uint8_t *magnitude_buffer,
    const float *direction_buffer,
    int32_t width,
    int32_t height,
    uint8_t *out_nms_buffer);

/**
 * @brief Naive NMS using pre-quantized direction buckets (0-3) from
 * bucket_sobel_filter, avoiding any angle computation in this kernel.
 *
 * @param magnitude_buffer  Gradient magnitude from the sobel filter.
 * @param dir_bucket_buffer Direction bucket (0-3) from bucket_sobel_filter.
 * @param width             Image width in pixels.
 * @param height            Image height in pixels.
 * @param out_nms_buffer    Output: thinned magnitude image, same size as magnitude_buffer.
 */
__global__ void non_maximum_suppression_bucket(
    const uint8_t *magnitude_buffer,
    const uint8_t *dir_bucket_buffer,
    int32_t width,
    int32_t height,
    uint8_t *out_nms_buffer);

/**
 * @brief Shared-memory NMS. The magnitude tile plus a 1px halo is loaded
 * cooperatively into shared memory, so each neighbor is fetched from global
 * memory once per tile instead of once per thread.
 *
 * Requires (TILE_W + 2) * (TILE_H + 2) bytes of dynamic shared memory.
 *
 * @param magnitude_buffer Gradient magnitude from the sobel filter.
 * @param direction_buffer Gradient direction in radians.
 * @param width            Image width in pixels.
 * @param height           Image height in pixels.
 * @param out_nms_buffer   Output: thinned magnitude image, same size as magnitude_buffer.
 */
__global__ void non_maximum_suppression_shared(
    const uint8_t *magnitude_buffer,
    const float *direction_buffer,
    int32_t width,
    int32_t height,
    uint8_t *out_nms_buffer);

/**
 * @brief Shared-memory NMS on pre-quantized direction buckets (0-3).
 * Combines the tiled magnitude load with the constant-LUT neighbor offsets.
 *
 * Requires (TILE_W + 2) * (TILE_H + 2) bytes of dynamic shared memory.
 *
 * @param magnitude_buffer  Gradient magnitude from the sobel filter.
 * @param dir_bucket_buffer Direction bucket (0-3) from bucket_sobel_filter.
 * @param width             Image width in pixels.
 * @param height            Image height in pixels.
 * @param out_nms_buffer    Output: thinned magnitude image, same size as magnitude_buffer.
 */
__global__ void non_maximum_suppression_shared_bucket(
    const uint8_t *magnitude_buffer,
    const uint8_t *dir_bucket_buffer,
    int32_t width,
    int32_t height,
    uint8_t *out_nms_buffer);
