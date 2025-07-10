#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <webp/encode.h>
#include <fstream>

// CUDA Kernel that computes Mandelbrot iteration counts for each pixel.
global void mandelbrot_kernel(uint32_t counts, uint32_t maxcount, uint32_t w, uint32_t h, float xmin, float xmax, float ymin, float ymax) {
    //Compute 2D thread coordinates for pixel locations
    int idx = blockIdx.x blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * w + idx;
    //Ensure threads do not access out-of-bounds pixels
    if (idx >= w || idy >= h) return;
    //Map pixel to complex plane
    float x0 = xmin + idx * (xmax - xmin) / w;
    float y0 = ymin + idy * (ymax - ymin) / h;
    float x = 0.0, y = 0.0;
    uint32_t iteration = 0;
    //Mandelbrot iteration loop: z = z^2 + c
    while (xx + yy <= 4 && iteration < maxcount) {
        float xtemp = xx - yy + x0;
        y = 2xy + y0;
        x = xtemp;
        iteration++;
    }
    // Store number of iterations or max in the output array
    counts[index] = (iteration == maxcount) ? maxcount : iteration;
}

//Host function that configures and launches Mandelbrot on the CUDA kernel
void mandelbrot(uint32_t d_counts, uint32_t maxcount, uint32_t w, uint32_t h, float xmin, float xmax, float ymin, float ymax) {
    //Define number of threads per block and blocks per grid
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    //Launch kernel on GPU
    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_counts, maxcount, w, h, xmin, xmax, ymin, ymax);
}

void build_color_table(uint32_t colors[], uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        uint8_t r = (i 5) % 256;
        uint8_t g = (i * 7) % 256;
        uint8_t b = (i * 11) % 256;
        uint8_t a = 0xFF;  
        colors[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

void convert_mandelbrot_count_to_rgb(uint32_t pixels[], uint32_t mandelbrot_count[], uint32_t w, uint32_t h, const uint32_t colors[], uint32_t color_count) {
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t index = y * w + x;
            uint32_t count = mandelbrot_count[index];
            uint32_t color_index = count % color_count;
            pixels[index] = colors[color_index];
        }
    }
}
//Saves image to a WebP file from raw RGBA pixel data
bool save_webp(const char* filename, uint32_t* pixels, uint32_t w, uint32_t h, int quality) {
    uint8_t* webp_data;

    //Encode the image into a WebP format
    size_t webp_size = WebPEncodeRGBA((uint8_t)pixels, w, h, w 4, quality, &webp_data);

    if (webp_size == 0) {
        std::cerr << "Error encoding WebP image!" << std::endl;
        return false;
    }
    //open output file in binary mode
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing!" << std::endl;
        WebPFree(webp_data);
        return false;
    }
    //Write the encoded image data to the file
    file.write(reinterpret_cast<const char>(webp_data), webp_size);
    file.close();
    WebPFree(webp_data);
    return true;
}

int main() {
    const uint32_t w = 1920, h = 1080, maxIterations = 256;
    //Allocate host and device memory
    uint32_th_counts = new uint32_t[w * h];
    uint32_t d_counts;
    uint32_tpixels = new uint32_t[w * h];
    uint32_t colors[256];  
    cudaMalloc(&d_counts, w * h * sizeof(uint32_t));
    //Launch mandelbrot generations on GPU
    mandelbrot(d_counts, maxIterations, w, h, -2.0, 1.0, -1.0, 1.0);
    //Copy results back from GPU to CPU
    cudaMemcpy(h_counts, d_counts, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //Build a color palette and convert iteration counts to RGB pixels
    build_color_table(colors, 256);
    convert_mandelbrot_count_to_rgb(pixels, h_counts, w, h, colors, 256);
    //save final image as a WebP file
    save_webp("mandelbrot_image.webp", pixels, w, h, 100);
    // Clean up
    cudaFree(d_counts);
    delete[] h_counts;
    delete[] pixels;

    return 0;
}
