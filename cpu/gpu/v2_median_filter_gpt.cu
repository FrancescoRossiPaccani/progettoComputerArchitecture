#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_32_intrinsics.h> // Per le istruzioni SIMD Video (__vminu4, __vmaxu4)

using namespace std;

// Usiamo un singolo uint32_t per impacchettare i 4 canali (R, G, B, Padding)
struct Image {
    int width, height;
    vector<uint32_t> pixels;

    Image(int h, int w) : width(w), height(h), pixels(h * w) {}
};

// ---------------- LOAD ----------------
Image loadImage(const string& path) {
    ifstream file(path, ios::binary);
    if (!file) { cerr << "Errore apertura file\n"; exit(1); }
    string format; file >> format;
    int w, h, maxval; file >> w >> h >> maxval;

    Image img(h, w);
    if (format == "P6") {
        while (isspace(file.peek())) file.get();
        vector<uint8_t> buffer(w * h * 3);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        // Impacchettamento manuale in 32-bit: | Padding(255) | Blu | Verde | Rosso |
        for (int i = 0; i < h * w; i++) {
            uint32_t r = buffer[i * 3];
            uint32_t g = buffer[i * 3 + 1];
            uint32_t b = buffer[i * 3 + 2];
            uint32_t a = 255; 
            img.pixels[i] = r | (g << 8) | (b << 16) | (a << 24);
        }
    }
    return img;
}

// ---------------- SAVE ----------------
void saveImage(const string& path, const Image& img) {
    ofstream file(path, ios::binary);
    file << "P6\n" << img.width << " " << img.height << "\n255\n";

    vector<uint8_t> buffer(img.width * img.height * 3);
    for (int i = 0; i < img.width * img.height; i++) {
        uint32_t p = img.pixels[i];
        buffer[i * 3]     = p & 0xFF;         // Estrazione R
        buffer[i * 3 + 1] = (p >> 8) & 0xFF;  // Estrazione G
        buffer[i * 3 + 2] = (p >> 16) & 0xFF; // Estrazione B
    }
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// ---------------- DEVICE MATH SIMD (3x Istruzioni in meno) ----------------
__device__ __forceinline__ uint32_t vmin4_3(uint32_t a, uint32_t b, uint32_t c) {
    return __vminu4(__vminu4(a, b), c);
}
__device__ __forceinline__ uint32_t vmax4_3(uint32_t a, uint32_t b, uint32_t c) {
    return __vmaxu4(__vmaxu4(a, b), c);
}
__device__ __forceinline__ uint32_t vmed4_3(uint32_t a, uint32_t b, uint32_t c) {
    return __vmaxu4(__vminu4(a, b), __vminu4(__vmaxu4(a, b), c));
}

// Trova il mediano di tutti e 3 i canali simultaneamente in registri a 32-bit
__device__ __forceinline__ uint32_t fast_median_9_simd(
    uint32_t p0, uint32_t p1, uint32_t p2,
    uint32_t p3, uint32_t p4, uint32_t p5,
    uint32_t p6, uint32_t p7, uint32_t p8) 
{
    uint32_t r0_min = vmin4_3(p0, p1, p2);
    uint32_t r0_med = vmed4_3(p0, p1, p2);
    uint32_t r0_max = vmax4_3(p0, p1, p2);

    uint32_t r1_min = vmin4_3(p3, p4, p5);
    uint32_t r1_med = vmed4_3(p3, p4, p5);
    uint32_t r1_max = vmax4_3(p3, p4, p5);

    uint32_t r2_min = vmin4_3(p6, p7, p8);
    uint32_t r2_med = vmed4_3(p6, p7, p8);
    uint32_t r2_max = vmax4_3(p6, p7, p8);

    uint32_t max_of_mins = vmax4_3(r0_min, r1_min, r2_min);
    uint32_t med_of_meds = vmed4_3(r0_med, r1_med, r2_med);
    uint32_t min_of_maxs = vmin4_3(r0_max, r1_max, r2_max);

    return vmed4_3(max_of_mins, med_of_meds, min_of_maxs);
}

// ---------------- KERNEL ----------------
__global__ void image_filtering_simd(const uint32_t* __restrict__ img_in, uint32_t* __restrict__ img_out, int img_h, int img_w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h) return;

    int idx = y * img_w + x;

    // --- FAST PATH --- 
    // Nessun if, max() o min() per calcolare le coordinate. Calcoli puri.
    // Viene eseguito per il 99.9% dei pixel dell'immagine.
    if (x >= 1 && x < img_w - 1 && y >= 1 && y < img_h - 1) {
        
        uint32_t p00 = __ldg(&img_in[idx - img_w - 1]);
        uint32_t p01 = __ldg(&img_in[idx - img_w]);
        uint32_t p02 = __ldg(&img_in[idx - img_w + 1]);
        
        uint32_t p10 = __ldg(&img_in[idx - 1]);
        uint32_t p11 = __ldg(&img_in[idx]);
        uint32_t p12 = __ldg(&img_in[idx + 1]);
        
        uint32_t p20 = __ldg(&img_in[idx + img_w - 1]);
        uint32_t p21 = __ldg(&img_in[idx + img_w]);
        uint32_t p22 = __ldg(&img_in[idx + img_w + 1]);

        img_out[idx] = fast_median_9_simd(p00, p01, p02, p10, p11, p12, p20, p21, p22);
    } 
    // --- SLOW PATH --- (Per i pixel sul bordo estremo)
    else {
        int x0 = max(x - 1, 0);
        int x1 = x;
        int x2 = min(x + 1, img_w - 1);

        int y0 = max(y - 1, 0) * img_w;
        int y1 = y * img_w;
        int y2 = min(y + 1, img_h - 1) * img_w;

        uint32_t p00 = __ldg(&img_in[y0 + x0]);
        uint32_t p01 = __ldg(&img_in[y0 + x1]);
        uint32_t p02 = __ldg(&img_in[y0 + x2]);
        
        uint32_t p10 = __ldg(&img_in[y1 + x0]);
        uint32_t p11 = __ldg(&img_in[y1 + x1]);
        uint32_t p12 = __ldg(&img_in[y1 + x2]);
        
        uint32_t p20 = __ldg(&img_in[y2 + x0]);
        uint32_t p21 = __ldg(&img_in[y2 + x1]);
        uint32_t p22 = __ldg(&img_in[y2 + x2]);

        img_out[idx] = fast_median_9_simd(p00, p01, p02, p10, p11, p12, p20, p21, p22);
    }
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[]) {
    if (argc < 1) {
        cout << "Uso: " << argv[0] << " nome_immagine\n";
        return 1;
    }

    string img_name = argv[1];
    
    Image img = loadImage("../error_images/" + img_name + ".ppm");
    Image out(img.height, img.width);
    cout << "Immagine " << img.width << "x" << img.height << "\n";

    int img_dim = img.width * img.height;
    
    uint32_t *d_img_in, *d_img_out; 

    cudaMalloc(&d_img_in, img_dim * sizeof(uint32_t));
    cudaMalloc(&d_img_out, img_dim * sizeof(uint32_t));

    cudaMemcpy(d_img_in, img.pixels.data(), img_dim * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Un blocco quadrato (16x16) solitamente favorisce un hit-rate migliore in cache L1 per i filtri a finestra (stencil 2D)
    dim3 block(16, 16);
    dim3 grid((img.width + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    image_filtering_simd <<< grid, block >>> (d_img_in, d_img_out, img.height, img.width);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Kernel time: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaMemcpy(out.pixels.data(), d_img_out, img_dim * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_img_in);
    cudaFree(d_img_out);

    saveImage("./output_images/" + img_name + ".ppm", out);
    
    return 0;
}