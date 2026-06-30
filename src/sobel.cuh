#pragma once
#include <cmath>
#include <cstdint>

__global__ void naiveSobelFilter(const uint8_t *src_buffer, const int32_t width,
                                 const int32_t height, uint8_t *out_dst_buffer);
