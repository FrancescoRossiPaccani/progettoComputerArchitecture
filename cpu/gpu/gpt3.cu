#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

struct MMM {
    uint32_t v_min, v_med, v_max;
};

struct Image {
    int width, height;
    vector<uint32_t> pixels;
    Image(int h, int w) : width(w), height(h), pixels(h * w) {}
};

// --- Helper funzioni I/O (P3/P6) ---
void skipComments(ifstream& f) {
    char ch;
    while (f.get(ch) && (isspace(ch) || ch == '#')) {
        if (ch == '#') f.ignore(1000, '\n');
    }
    f.putback(ch);
}

Image loadImage(const string& path) {
    ifstream file(path, ios::binary);
    if (!file) { cerr << "Errore apertura: " << path << endl; exit(1); }
    string format; file >> format;
    int w, h, maxval;
    skipComments(file); file >> w;
    skipComments(file); file >> h;
    skipComments(file); file >> maxval;
    Image img(h, w);
    if (format == "P6") {
        file.ignore(1000, '\n');
        vector<uint8_t> buffer(w * h * 3);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        for (int i = 0; i < h * w; i++) 
            img.pixels[i] = buffer[i*3] | (buffer[i*3+1] << 8) | (buffer[i*3+2] << 16) | (255u << 24);
    } else {
        for (int i = 0; i < h * w; i++) {
            int r, g, b; file >> r >> g >> b;
            img.pixels[i] = (uint8_t)r | ((uint8_t)g << 8) | ((uint8_t)b << 16) | (255u << 24);
        }
    }
    return img;
}

void saveImage(const string& path, const Image& img) {
    ofstream file(path, ios::binary);
    file << "P6\n" << img.width << " " << img.height << "\n255\n";
    vector<uint8_t> buffer(img.width * img.height * 3);
    for (int i = 0; i < img.width * img.height; i++) {
        uint32_t p = img.pixels[i];
        buffer[i * 3] = p & 0xFF; buffer[i * 3 + 1] = (p >> 8) & 0xFF; buffer[i * 3 + 2] = (p >> 16) & 0xFF;
    }
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// --- SIMD Intrinsics ---
__device__ __forceinline__ uint32_t vmin4_3(uint32_t a, uint32_t b, uint32_t c) { return __vminu4(__vminu4(a, b), c); }
__device__ __forceinline__ uint32_t vmax4_3(uint32_t a, uint32_t b, uint32_t c) { return __vmaxu4(__vmaxu4(a, b), c); }
__device__ __forceinline__ uint32_t vmed4_3(uint32_t a, uint32_t b, uint32_t c) { return __vmaxu4(__vminu4(a, b), __vminu4(__vmaxu4(a, b), c)); }

// --- CONFIGURAZIONE ---
#define BW 32    // Block Width
#define BH 8     // Block Height
#define N_ROWS 16// Righe per thread (ridotto per bilanciare shared memory e registri)

// --- KERNEL TILED ---
__global__ void image_filtering_tiled(const uint32_t* __restrict__ img_in, uint32_t* __restrict__ img_out, int img_h, int img_w) {
    // Shared memory: carichiamo un'area più grande (BW + 2) x (BH*N_ROWS + 2) per i bordi del kernel 3x3
    __shared__ uint32_t s_tile[BH * N_ROWS + 2][BW + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BW + tx;
    int y_base = blockIdx.y * (BH * N_ROWS) + (ty * N_ROWS);

    // Caricamento dei dati in Shared Memory
    // Ogni thread carica N_ROWS pixel, più alcuni si occupano dei bordi del tile
    #pragma unroll
    for (int i = 0; i < N_ROWS; i++) {
        int cur_y = y_base + i;
        int glb_y = max(0, min(cur_y, img_h - 1));
        int glb_x = max(0, min(x, img_w - 1));
        
        // Caricamento centrale
        s_tile[ty * N_ROWS + i + 1][tx + 1] = img_in[glb_y * img_w + glb_x];

        // Caricamento bordi laterali (solo i thread alle estremità del blocco)
        if (tx == 0) 
            s_tile[ty * N_ROWS + i + 1][0] = img_in[glb_y * img_w + max(0, x - 1)];
        if (tx == BW - 1) 
            s_tile[ty * N_ROWS + i + 1][BW + 1] = img_in[glb_y * img_w + min(img_w - 1, x + 1)];
    }

    // Caricamento bordi superiore e inferiore (solo la prima e l'ultima riga del blocco)
    if (ty == 0) {
        int glb_x = max(0, min(x, img_w - 1));
        s_tile[0][tx + 1] = img_in[max(0, y_base - 1) * img_w + glb_x];
        if (tx == 0) s_tile[0][0] = img_in[max(0, y_base - 1) * img_w + max(0, x - 1)];
        if (tx == BW - 1) s_tile[0][BW + 1] = img_in[max(0, y_base - 1) * img_w + min(img_w - 1, x + 1)];
    }
    if (ty == BH - 1) {
        int last_y = y_base + N_ROWS - 1;
        int glb_x = max(0, min(x, img_w - 1));
        s_tile[BH * N_ROWS + 1][tx + 1] = img_in[min(img_h - 1, last_y + 1) * img_w + glb_x];
        if (tx == 0) s_tile[BH * N_ROWS + 1][0] = img_in[min(img_h - 1, last_y + 1) * img_w + max(0, x - 1)];
        if (tx == BW - 1) s_tile[BH * N_ROWS + 1][BW + 1] = img_in[min(img_h - 1, last_y + 1) * img_w + min(img_w - 1, x + 1)];
    }

    __syncthreads(); // Barriera per assicurarsi che tutta la shared memory sia piena

    // Calcolo del filtro
    if (x < img_w) {
        #pragma unroll
        for (int i = 0; i < N_ROWS; i++) {
            int cur_y = y_base + i;
            if (cur_y < img_h) {
                int sm_y = ty * N_ROWS + i + 1;
                int sm_x = tx + 1;

                // Estrazione righe dalla shared memory
                uint32_t p00 = s_tile[sm_y - 1][sm_x - 1], p01 = s_tile[sm_y - 1][sm_x], p02 = s_tile[sm_y - 1][sm_x + 1];
                uint32_t p10 = s_tile[sm_y][sm_x - 1],     p11 = s_tile[sm_y][sm_x],     p12 = s_tile[sm_y][sm_x + 1];
                uint32_t p20 = s_tile[sm_y + 1][sm_x - 1], p21 = s_tile[sm_y + 1][sm_x], p22 = s_tile[sm_y + 1][sm_x + 1];

                // Median 3x3 usando SIMD
                uint32_t r0_min = vmin4_3(p00, p01, p02), r0_med = vmed4_3(p00, p01, p02), r0_max = vmax4_3(p00, p01, p02);
                uint32_t r1_min = vmin4_3(p10, p11, p12), r1_med = vmed4_3(p10, p11, p12), r1_max = vmax4_3(p10, p11, p12);
                uint32_t r2_min = vmin4_3(p20, p21, p22), r2_med = vmed4_3(p20, p21, p22), r2_max = vmax4_3(p20, p21, p22);

                uint32_t max_min = vmax4_3(r0_min, r1_min, r2_min);
                uint32_t med_med = vmed4_3(r0_med, r1_med, r2_med);
                uint32_t min_max = vmin4_3(r0_max, r1_max, r2_max);

                img_out[cur_y * img_w + x] = vmed4_3(max_min, med_med, min_max);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) { cout << "Uso: " << argv[0] << " nome_immagine\n"; return 1; }
    string img_name = argv[1];
    
    Image img = loadImage("../error_images/" + img_name + ".ppm");
    Image out(img.height, img.width);
    
    uint32_t *d_img_in, *d_img_out;
    cudaMalloc(&d_img_in, img.pixels.size() * sizeof(uint32_t));
    cudaMalloc(&d_img_out, img.pixels.size() * sizeof(uint32_t));
    cudaMemcpy(d_img_in, img.pixels.data(), img.pixels.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    dim3 block(BW, BH);
    dim3 grid((img.width + BW - 1) / BW, (img.height + (BH * N_ROWS) - 1) / (BH * N_ROWS));

    cudaEventRecord(start);
    image_filtering_tiled<<<grid, block>>>(d_img_in, d_img_out, img.height, img.width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Kernel time (Tiled): " << ms << " ms\n";

    cudaMemcpy(out.pixels.data(), d_img_out, img.pixels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    saveImage("./output_images/" + img_name + ".ppm", out);

    cudaFree(d_img_in); cudaFree(d_img_out);
    return 0;
}