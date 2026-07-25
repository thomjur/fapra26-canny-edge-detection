#pragma once

#include <cmath>
#include <cstdint>

#ifndef HYSTERESIS_OPT
#define HYSTERESIS_OPT 0
#endif

#if HYSTERESIS_OPT == 0
#define HYSTERESIS_NAIVE
#elif HYSTERESIS_OPT == 1
#define HYSTERESIS_SHARED
#elif HYSTERESIS_OPT == 2
#define HYSTERESIS_OPTIMIZED
#elif HYSTERESIS_OPT == 3
#define HYSTERESIS_PINNED
#else
#error "HYSTERESIS_OPT must be 0, 1 or 2"
#endif

#define HIGH_THRESHOLD 150
#define LOW_THRESHOLD 50

/**
 * @struct HysteresisResult
 * @brief Holds the hysteresis linked edges image from hysteresis_execute().
 *
 * The hysteresis_buffer must be freed using hysteresis_cleanup() after use to
 * avoid memory leaks.
 */
struct HysteresisResult {
  uint8_t *hysteresis_buffer; ///< Thinned gradient magnitude image (0-255).
  float ms_h2d;    ///< Time taken for host-to-device transfer in milliseconds.
  float ms_kernel; ///< Time taken for kernel execution in milliseconds.
  float ms_d2h;    ///< Time taken for device-to-host transfer in milliseconds.
};

/**
 * @brief Executes the Hysteresis edge linker on a grayscale image processed by
 * nms_execute().
 *
 * Allocates device memory, launches the Hysteresis kernel, and copies results
 * back to the host. Also measures timing for H2D, kernel execution, and D2H
 * transfers.
 *
 * @param host_src Input grayscale image (1 byte per pixel).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @return Hysteresis Structure containing final buffer and
 * timing info. Use hysteresis_cleanup() to free the host buffers.
 */
HysteresisResult hysteresis_execute(const uint8_t *host_src, int32_t width,
                                    int32_t height);

/**
 * @brief CUDA kernel for Hysteresis step to link strong and weak edges.
 *
 * @param src_buffer Input image buffer (device memory, grayscale, 8-bit).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param out_buffer Output buffer for final image with linked edges (device
 * memory, 8-bit).
 * @param low_threshold Lower threshold value for hysteresis.
 * @param high_threshold Higher threshold value for hysteresis.
 */
__global__ void
naive_hysteresis_linker(const uint8_t *src_buffer, const int32_t width,
                        const int32_t height, uint8_t *out_buffer,
                        uint8_t low_threshold, uint8_t high_threshold);

/**
 * @brief Hysteresis edge linking using shared-memory tiling.
 *
 * This kernel classifies each pixel as edge (255) or non-edge (0):
 * - Strong pixel (p >= high_t) -> edge (255)
 * - Weak pixel   (low_t <= p < high_t) -> edge only if at least one of its
 *   8 neighbors is strong (>= high_t)
 * - Otherwise -> non-edge (0)
 *
 * Performance notes:
 * - Uses shared memory tile with a 1-pixel halo to reduce global memory reads.
 * - Neighbor checks are performed from shared memory (faster than global
 * loads).
 *
 * @param src      Input grayscale image (device pointer, uint8_t)
 * @param width    Image width in pixels
 * @param height   Image height in pixels
 * @param dst      Output binary edge image (device pointer, uint8_t)
 * @param low_t    Low threshold
 * @param high_t   High threshold
 */
__global__ void hysteresis_sharedm(uint8_t *src, const int32_t width,
                                   const int32_t height, uint8_t *dst,
                                   uint8_t low_t, uint8_t high_t);

/**
 * @brief Hysteresis edge linking optimized WITHOUT shared-memory tiling.
 *
 * This kernel classifies each pixel as edge (255) or non-edge (0):
 * - Strong pixel (p >= high_t) -> edge (255)
 * - Weak pixel   (low_t <= p < high_t) -> edge only if at least one of its
 *   8 neighbors is strong (>= high_t)
 * - Otherwise -> non-edge (0)
 *
 * Performance notes:
 * - Neighbor checks are performed using unrolled neighbor checking.
 *
 * @param src      Input grayscale image (device pointer, uint8_t)
 * @param width    Image width in pixels
 * @param height   Image height in pixels
 * @param dst      Output binary edge image (device pointer, uint8_t)
 * @param low_t    Low threshold
 * @param high_t   High threshold
 */
__global__ void hysteresis_opt(uint8_t *src, const int32_t width,
                               const int32_t height, uint8_t *dst,
                               uint8_t low_t, uint8_t high_t);
/**
 * @brief Releases the host buffer allocated by hysteresis_execute().
 *
 * @param result HysteresisResult returned by hysteresis_execute().
 *               After cleanup, hysteresis_buffer is set to nullptr.
 */
void hysteresis_cleanup(HysteresisResult &result);
