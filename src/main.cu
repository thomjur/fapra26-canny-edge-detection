#include <cstdio>
#include <cuda.h>

#include "common.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(int argc, const char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: ./canny.out <input.jpg> <output.png>\n");
        return -1;
    }

    const char* path     = argv[1];
    const char* out_path = argv[2];

    // load image
    int width{};
    int height{};
    int channels_in_file{};
    const unsigned char* image_buffer = stbi_load(path, &width, &height, &channels_in_file, 1);
    if (image_buffer == nullptr)
        return -1;

    printf("Image loaded : %s\n", path);
    printf("Size         : %d x %d\n", width, height);
    printf("Channels     : %d (loaded as grayscale)\n", channels_in_file);

    // write image
    const int write_result = stbi_write_png(out_path, width, height, 1, image_buffer, width * 1);
    if (write_result == 0)
    {
        fprintf(stderr, "Error: could not write image '%s'\n", out_path);
        stbi_image_free((void*)image_buffer);
        return -1;
    }
    printf("Grayscale written : %s\n", out_path);

    // clean up
    stbi_image_free((void*)image_buffer);
    return 0;
}