#pragma once
#include <cstdint>

//
// >>> GAUSSIAN FILTER <
//

// Gaussian filter optimization levels:
// 0 - naive:         each thread reads directly from global memory
// 1 - shared memory: image tiles loaded cooperatively into shared memory
// 2 - pinned memory: host buffer allocated as page-locked memory

#ifndef GAUSSIAN_OPT
    #define GAUSSIAN_OPT 0
#endif

#if GAUSSIAN_OPT == 0
    #define GAUSSIAN_NAIVE
#elif GAUSSIAN_OPT == 1
    #define GAUSSIAN_SHARED
#elif GAUSSIAN_OPT == 2
    // kernel is identical to GAUSSIAN_SHARED — only host-side allocation differs
    #define GAUSSIAN_PINNED
    #define GAUSSIAN_SHARED
#else
    #error "GAUSSIAN_OPT must be 0, 1, or 2"
#endif

/// Blur radius for the gaussian filter (e.g. 2 = 5x5 kernel).
/// Must be a compile-time constant for shared memory tile sizing.
#define BLUR_RADIUS 9

/// Holds the blurred image and per-stage timing results from gaussian_execute().
struct GaussianResult
{
    uint8_t* host_buffer;  ///< blurred grayscale image — free via gaussian_cleanup()
    float    ms_h2d;       ///< host-to-device transfer time (ms)
    float    ms_kernel;    ///< kernel execution time (ms)
    float    ms_d2h;       ///< device-to-host transfer time (ms)
};

/// @brief Runs the full gaussian blur pipeline for a grayscale image.
///
/// @param host_src  Input grayscale image (1 byte per pixel).
/// @param width     Image width in pixels.
/// @param height    Image height in pixels.
/// @return          GaussianResult with blurred image and timing. Free via gaussian_cleanup().
GaussianResult gaussian_execute(
    const uint8_t* host_src,
    int32_t        width,
    int32_t        height);

/// @brief Releases the host buffer allocated by gaussian_execute().
///
/// @param result  GaussianResult returned by gaussian_execute().
///                host_buffer is set to nullptr after cleanup.
void gaussian_cleanup(GaussianResult& result);

/// @brief Calculates normalized gaussian weights for a given blur radius and sigma.
///
/// @details Larger sigma = flatter curve = more blur.
///          Smaller sigma = sharper curve = less blur.
///
///          Example for blur_radius=1, blur_size=3, sigma=1.0f:
///
///          +--------+--------+--------+
///          | 0.0894 | 0.1201 | 0.0894 |   y = -1
///          +--------+--------+--------+
///          | 0.1201 | 0.1615 | 0.1201 |   y =  0
///          +--------+--------+--------+
///          | 0.0894 | 0.1201 | 0.0894 |   y = +1
///          +--------+--------+--------+
///             x=-1     x=0     x=+1
///
/// @param blur_radius Radius of the blur kernel (e.g. 1 = 3x3, 2 = 5x5).
/// @param sigma       Controls the spread of the gaussian bell curve.
/// @param out_weights Output buffer for the weights. Must be at least
///                    (blur_radius * 2 + 1)^2 floats large.
void calculate_gaussian_weights(
    const uint32_t blur_radius,
    const float    sigma,
    float*         out_weights);

#ifdef GAUSSIAN_NAIVE
/// @brief Applies a gaussian blur to a grayscale image (naive).
///        Each thread reads weights and pixels directly from global memory.
///
/// @param src_buffer     Input image buffer (grayscale, 1 byte per pixel).
/// @param width          Image width in pixels.
/// @param height         Image height in pixels.
/// @param weights        Precomputed gaussian weights on device. See calculate_gaussian_weights().
/// @param blur_radius    Radius of the blur kernel (e.g. 1 = 3x3, 2 = 5x5).
/// @param out_dst_buffer Output image buffer (grayscale, 1 byte per pixel).
__global__ void gaussian_filter(
    const uint8_t* src_buffer,
    const int32_t  width,
    const int32_t  height,
    const float*   weights,
    const uint32_t blur_radius,
    uint8_t*       out_dst_buffer);
#endif // #ifdef GAUSSIAN_NAIVE

#ifdef GAUSSIAN_SHARED
/// Gaussian weights in __constant__ memory, shared across all threads without
/// global memory overhead. Upload via cudaMemcpyToSymbol() before kernel launch.
extern __constant__ float device_weights[(2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1)];

/// @brief Applies a gaussian blur to a grayscale image (shared memory tiling).
///        Threads within a block cooperatively load a tile (incl. halo) into
///        shared memory, then read neighbors from there instead of global memory.
///        Weights are stored in __constant__ memory.
///
/// @param src_buffer     Input image buffer (grayscale, 1 byte per pixel).
/// @param width          Image width in pixels.
/// @param height         Image height in pixels.
/// @param out_dst_buffer Output image buffer (grayscale, 1 byte per pixel).
__global__ void gaussian_filter(
    const uint8_t* src_buffer,
    const int32_t  width,
    const int32_t  height,
    uint8_t*       out_dst_buffer);
#endif // #ifdef GAUSSIAN_SHARED