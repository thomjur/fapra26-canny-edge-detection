#pragma once
#include <cmath>
#include <cstdint>

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
__global__ void naive_sobel_filter(const uint8_t *src_buffer,
                                   const int32_t width, const int32_t height,
                                   uint8_t *out_grad_buffer,
                                   float *out_dir_buffer);

/**
 * @struct SobelResult
 * @brief Holds the gradient and direction images from sobel_execute().
 *
 * The host_grad and host_dir buffers must be freed using sobel_cleanup()
 * after use to avoid memory leaks.
 */
struct SobelResult {
  uint8_t *host_grad; ///< Grayscale image with gradient values (0-255).
  float *host_dir; ///< Image with gradient direction values in radians (atan2).
  float ms_h2d;    ///< Time taken for host-to-device transfer in milliseconds.
  float ms_kernel; ///< Time taken for kernel execution in milliseconds.
  float ms_d2h;    ///< Time taken for device-to-host transfer in milliseconds.
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
SobelResult sobel_execute(const uint8_t *host_src, int32_t width,
                          int32_t height);

/**
 * @brief Releases the host buffers allocated by sobel_execute().
 *
 * @param result SobelResult returned by sobel_execute().
 *              After cleanup, host_grad and host_dir are set to nullptr.
 */
void sobel_cleanup(SobelResult &result);
