#pragma once

#include <cstdint>
#include "sobel.cuh"

//
// >>> NON-MAXIMUM SUPPRESSION <<<
//

// Non-maximum suppression optimization levels:
// 0 - naive:  each thread reads magnitude neighbors directly from global memory
// 1 - shared: magnitude tile (with 1px halo) loaded cooperatively into shared memory
// 2 - bucket: takes pre-quantized direction buckets from bucket_sobel_filter,
//             skips quantize_direction() entirely via constant lookup table

#ifdef SOBEL_BUCKET
    #define NMS_BUCKET
#else
    #ifndef NMS_OPT
        #define NMS_OPT 0
    #endif

    #if NMS_OPT == 0
        #define NMS_NAIVE
    #elif NMS_OPT == 1
        #define NMS_SHARED
    #elif NMS_OPT == 2
        #define NMS_BUCKET
    #else
        #error "NMS_OPT must be 0, 1 or 2"
    #endif
#endif

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
 * launches the NMS kernel, and copies the thinned result back to the host.
 *
 * The direction parameter type depends on the active optimization level:
 * a radian float buffer for NMS_NAIVE/NMS_SHARED, or a pre-quantized
 * uint8_t bucket buffer (0-3) for NMS_BUCKET.
 *
 * @param host_magnitude Gradient magnitude image from SobelResult::host_grad.
 * @param host_direction Gradient direction image from SobelResult::host_dir
 *                       (or host_dir_bucket if NMS_BUCKET is active).
 * @param width          Image width in pixels.
 * @param height         Image height in pixels.
 * @return               NmsResult Structure containing the thinned edge buffer and timing info.
 *                       Use nms_cleanup() to free the host buffer.
 */
NmsResult nms_execute(
    const uint8_t* host_magnitude,
#ifdef NMS_BUCKET
    const uint8_t *host_direction,
#else
    const float *host_direction,
#endif
    int32_t        width,
    int32_t        height);

/**
 * @brief Releases the host buffer allocated by nms_execute().
 *
 * @param result NmsResult returned by nms_execute().
 *               After cleanup, host_nms is set to nullptr.
 */
void nms_cleanup(NmsResult &result);

/**
 * @brief CUDA kernel for non-maximum suppression on a Sobel gradient image (naive).
 *
 * @param magnitude_buffer Gradient magnitude from step 2 (sobel filter).
 * @param direction_buffer Gradient direction in radians from step 2.
 * @param width            Image width in pixels.
 * @param height           Image height in pixels.
 * @param out_nms_buffer   Output: thinned magnitude image, same size as magnitude_buffer.
 */
__global__ void non_maximum_suppression(
    const uint8_t* magnitude_buffer,
    const float*   direction_buffer,
    int32_t        width,
    int32_t        height,
    uint8_t*       out_nms_buffer);

/**
 * @brief CUDA kernel for non-maximum suppression using pre-quantized
 * direction buckets (0-3) from bucket_sobel_filter, avoiding any
 * angle computation in this kernel.
 *
 * @param magnitude_buffer Gradient magnitude from step 2 (sobel filter).
 * @param dir_bucket_buffer Direction bucket (0-3) from bucket_sobel_filter.
 * @param width            Image width in pixels.
 * @param height           Image height in pixels.
 * @param out_nms_buffer   Output: thinned magnitude image, same size as magnitude_buffer.
 */
__global__ void non_maximum_suppression_bucket(
    const uint8_t* magnitude_buffer,
    const uint8_t* dir_bucket_buffer,
    int32_t        width,
    int32_t        height,
    uint8_t*       out_nms_buffer);