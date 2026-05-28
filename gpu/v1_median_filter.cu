#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

struct Image {
    int width, height;
    vector<uint8_t> r, g, b;

    Image(int h, int w) : width(w), height(h), r(h * w), g(h * w), b(h * w) {}
};

Image loadImage(const string& path) {
    ifstream file(path, ios::binary);
    string format;
    file >> format;
    int w, h, maxval;
    file >> w >> h >> maxval;
    Image img(h, w);
    if (format == "P6") {
        while (isspace(file.peek())) file.get();
        vector<uint8_t> buffer(w * h * 3);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        for (int i = 0; i < h * w; i++) {
            img.r[i] = buffer[i * 3];
            img.g[i] = buffer[i * 3 + 1];
            img.b[i] = buffer[i * 3 + 2];
        }
    }
    return img;
}

void saveImage(const string& path, Image& img) {
    ofstream file(path, ios::binary);
    file << "P6\n" << img.width << " " << img.height << "\n255\n";
    vector<uint8_t> buffer(img.width * img.height * 3);
    for (int i = 0; i < img.width * img.height; i++) {
        buffer[i * 3] = img.r[i];
        buffer[i * 3 + 1] = img.g[i];
        buffer[i * 3 + 2] = img.b[i];
    }
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

__device__ __forceinline__ void med3(uint8_t a, uint8_t b, uint8_t c, uint8_t &medVal) {
    uint8_t min_ab = (a < b) ? a : b;
    uint8_t max_ab = (a > b) ? a : b;
    uint8_t min_max_ab_c = (max_ab < c) ? max_ab : c;
    medVal = (min_ab > min_max_ab_c) ? min_ab : min_max_ab_c;
}

__device__ __forceinline__ void fast_median_scalar(uint8_t m0, uint8_t m1, uint8_t m2, 
                                                   uint8_t m3, uint8_t m4, uint8_t m5, 
                                                   uint8_t m6, uint8_t m7, uint8_t m8, 
                                                   uint8_t &median) {
    uint8_t r0_min, r0_med, r0_max;
    uint8_t r1_min, r1_med, r1_max;
    uint8_t r2_min, r2_med, r2_max;
    uint8_t max_of_mins, med_of_meds, min_of_maxs;

    uint8_t min01 = (m0 < m1) ? m0 : m1;
    r0_min = (min01 < m2) ? min01 : m2;
    med3(m0, m1, m2, r0_med);
    uint8_t max01 = (m0 > m1) ? m0 : m1;
    r0_max = (max01 > m2) ? max01 : m2;

    uint8_t min34 = (m3 < m4) ? m3 : m4;
    r1_min = (min34 < m5) ? min34 : m5;
    med3(m3, m4, m5, r1_med);
    uint8_t max34 = (m3 > m4) ? m3 : m4;
    r1_max = (max34 > m5) ? max34 : m5;

    uint8_t min67 = (m6 < m7) ? m6 : m7;
    r2_min = (min67 < m8) ? min67 : m8;
    med3(m6, m7, m8, r2_med);
    uint8_t max67 = (m6 > m7) ? m6 : m7;
    r2_max = (max67 > m8) ? max67 : m8;

    uint8_t max_min01 = (r0_min > r1_min) ? r0_min : r1_min;
    max_of_mins = (max_min01 > r2_min) ? max_min01 : r2_min;
    med3(r0_med, r1_med, r2_med, med_of_meds);
    uint8_t min_max01 = (r0_max < r1_max) ? r0_max : r1_max;
    min_of_maxs = (min_max01 < r2_max) ? min_max01 : r2_max;

    med3(max_of_mins, med_of_meds, min_of_maxs, median);
}

#define TILE_H 96

__global__ void median_shuffle_v10(const uint8_t* __restrict__ r_in, const uint8_t* __restrict__ g_in, const uint8_t* __restrict__ b_in,
                                   uint8_t* __restrict__ r_out, uint8_t* __restrict__ g_out, uint8_t* __restrict__ b_out,
                                   int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_start = blockIdx.y * TILE_H;
    int lane = threadIdx.x % 32;

    if (x >= w) return;

    // Registri per 3 canali x 3 righe x 3 colonne
    uint8_t r0[3], r1[3], r2[3];
    uint8_t g0[3], g1[3], g2[3];
    uint8_t b0[3], b1[3], b2[3];

    auto load_and_shuffle = [&](int row_y, uint8_t* reg, const uint8_t* in_ptr) {
        int target_y = (row_y < 0) ? 0 : (row_y >= h ? h - 1 : row_y);
        uint8_t val = in_ptr[target_y * w + x];
        
        // Shuffle per ottenere i vicini
        reg[1] = val;
        reg[0] = __shfl_up_sync(0xFFFFFFFF, val, 1);
        reg[2] = __shfl_down_sync(0xFFFFFFFF, val, 1);

        // Gestione bordi warp (se x-1 o x+1 è fuori dal warp)
        if (lane == 0)   reg[0] = in_ptr[target_y * w + (x > 0 ? x - 1 : 0)];
        if (lane == 31 || x == w - 1) reg[2] = in_ptr[target_y * w + (x < w - 1 ? x + 1 : w - 1)];
    };

    // Pre-caricamento prime due righe
    load_and_shuffle(y_start - 1, r0, r_in); load_and_shuffle(y_start - 1, g0, g_in); load_and_shuffle(y_start - 1, b0, b_in);
    load_and_shuffle(y_start,     r1, r_in); load_and_shuffle(y_start,     g1, g_in); load_and_shuffle(y_start,     b1, b_in);

    #pragma unroll 2
    for (int iy = 0; iy < TILE_H; iy++) {
        int y_curr = y_start + iy;
        if (y_curr >= h) break;

        load_and_shuffle(y_curr + 1, r2, r_in);
        load_and_shuffle(y_curr + 1, g2, g_in);
        load_and_shuffle(y_curr + 1, b2, b_in);

        uint8_t res_r, res_g, res_b;
        fast_median_scalar(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], res_r);
        fast_median_scalar(g0[0], g0[1], g0[2], g1[0], g1[1], g1[2], g2[0], g2[1], g2[2], res_g);
        fast_median_scalar(b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], res_b);

        r_out[y_curr * w + x] = res_r;
        g_out[y_curr * w + x] = res_g;
        b_out[y_curr * w + x] = res_b;

        for(int i=0; i<3; i++) {
            r0[i] = r1[i]; r1[i] = r2[i];
            g0[i] = g1[i]; g1[i] = g2[i];
            b0[i] = b1[i]; b1[i] = b2[i];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    string img_name = argv[1];
    Image img = loadImage("../error_images/" + img_name + ".ppm");
    int dim = img.width * img.height;

    uint8_t *dr_i, *dg_i, *db_i, *dr_o, *dg_o, *db_o;
    cudaMalloc(&dr_i, dim); cudaMalloc(&dg_i, dim); cudaMalloc(&db_i, dim);
    cudaMalloc(&dr_o, dim); cudaMalloc(&dg_o, dim); cudaMalloc(&db_o, dim);
    cudaMemcpy(dr_i, img.r.data(), dim, cudaMemcpyHostToDevice);
    cudaMemcpy(dg_i, img.g.data(), dim, cudaMemcpyHostToDevice);
    cudaMemcpy(db_i, img.b.data(), dim, cudaMemcpyHostToDevice);

    dim3 block(256, 1);
    dim3 grid((img.width + block.x - 1) / block.x, (img.height + TILE_H - 1) / TILE_H);

    // Warm-up
    median_shuffle_v10<<<grid, block>>>(dr_i, dg_i, db_i, dr_o, dg_o, db_o, img.height, img.width);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    median_shuffle_v10<<<grid, block>>>(dr_i, dg_i, db_i, dr_o, dg_o, db_o, img.height, img.width);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);
    cout << "Tempo totale (3 canali Shuffle): " << ms << " ms" << endl;

    Image out(img.height, img.width);
    cudaMemcpy(out.r.data(), dr_o, dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.g.data(), dg_o, dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.b.data(), db_o, dim, cudaMemcpyDeviceToHost);
    saveImage("./output_images/" + img_name + "_shuffle.ppm", out);
    return 0;
}
