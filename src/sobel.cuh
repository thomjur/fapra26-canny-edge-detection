#pragma once
#include <cmath>
#include <cstdint>

__global__ void naive_sobel_filter(const uint8_t *src_buffer,
                                   const int32_t width, const int32_t height,
                                   uint8_t *out_grad_buffer,
                                   float *out_dir_buffer);

/// Holds the gradient and direction images from sobel_execute().
struct SobelResult {
  uint8_t *host_grad; ///< Grayscale image with gradient values— free via
                      ///< sobel_cleanup()
  float *host_dir;    ///< Image with atan values— free via sobel_cleanup()
  float ms_h2d;       ///< host-to-device transfer time (ms)
  float ms_kernel;    ///< kernel execution time (ms)
  float ms_d2h;       ///< device-to-host transfer time (ms)
};

/// @brief Runs the full sobel filter  a grayscale image preprocessed with
/// Gaussian filte.
///
/// @param host_src  Input grayscale image (1 byte per pixel).
/// @param width     Image width in pixels.
/// @param height    Image height in pixels.
/// @return          SobelResult with blurred image and timing. Free via
/// sobel_cleanup().
SobelResult sobel_execute(const uint8_t *host_src, int32_t width,
                          int32_t height);

/// @brief Releases the host buffer allocated by sobel_execute().
///
/// @param result  SobelResult returned by sobel_execute().
///                host_grad and host_dir are set to nullptr after cleanup.
void sobel_cleanup(SobelResult &result);
