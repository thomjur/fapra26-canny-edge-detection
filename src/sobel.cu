#include <cmath>
#include <cstdint>

// Kernel for Sobel edge detection
__global__ void naive_sobel_filter(const uint8_t *src_buffer,
                                   const int32_t width, const int32_t height,
                                   uint8_t *out_grad_buffer,
                                   float *out_dir_buffer) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the thread is within the image boundaries
  // and if the 3x3 kernel fits around the pixel
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    // Sobel kernels
    int32_t sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int32_t sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int32_t gx = 0, gy = 0;

    // Apply the Sobel kernels
    for (int8_t i = -1; i <= 1; i++) {
      for (int8_t j = -1; j <= 1; j++) {
        uint8_t pixel = src_buffer[(y + i) * width + (x + j)];
        gx += pixel * sobelX[i + 1][j + 1];
        gy += pixel * sobelY[i + 1][j + 1];
      }
    }

    // Calculate the gradient magnitude
    float gradient = sqrtf(gx * gx + gy * gy);

    // Clamp the value to 0-255
    gradient = min(255.0f, max(0.0f, gradient));

    // Write the gradient result
    out_grad_buffer[y * width + x] = static_cast<uint8_t>(gradient);

    // Calculate the
    out_dir_buffer[y * width + x] =
        atan2f(static_cast<float>(gy), static_cast<float>(gx));
  }
}
