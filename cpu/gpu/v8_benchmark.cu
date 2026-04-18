#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

struct Image{
    int width, height;
    vector<uint8_t> r, g, b;

    Image(int h,int w):width(w),height(h),
        r(h*w), g(h*w), b(h*w){}
};

Image loadImage(const string& path){
    ifstream file(path, ios::binary);
    string format;
    file>>format;
    int w,h,maxval;
    file>>w>>h>>maxval;
    Image img(h,w);
    if (format == "P6") {
        while(isspace(file.peek())) file.get();
        vector<uint8_t> buffer(w * h * 3);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        for(int i=0;i<h*w;i++){
            img.r[i] = buffer[i*3];
            img.g[i] = buffer[i*3+1];
            img.b[i] = buffer[i*3+2];
        }
    } else {
        for(int i=0;i<h*w;i++){
            int R,G,B;
            file>>R>>G>>B;
            img.r[i]=R;
            img.g[i]=G;
            img.b[i]=B;
        }
    }
    return img;
}

void saveImage(const string& path, Image& img){
    ofstream file(path, ios::binary);
    file<<"P6\n"<<img.width<<" "<<img.height<<"\n255\n";
    vector<uint8_t> buffer(img.width * img.height * 3);
    for(int i=0;i<img.width*img.height;i++){
        buffer[i*3]   = img.r[i];
        buffer[i*3+1] = img.g[i];
        buffer[i*3+2] = img.b[i];
    }
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}


__device__ __forceinline__ void min3(uint8_t a, uint8_t b, uint8_t c, uint8_t &minVal){
    minVal = (uint8_t)min(min((int)a, (int)b), (int)c);
}

__device__ __forceinline__ void max3(uint8_t a, uint8_t b, uint8_t c, uint8_t &maxVal){
    maxVal = (uint8_t)max(max((int)a, (int)b), (int)c);
}

__device__ __forceinline__ void med3(uint8_t a, uint8_t b, uint8_t c, uint8_t &medVal){
    int ai = (int)a, bi = (int)b, ci = (int)c;
    medVal = (uint8_t)max(min(ai, bi), min(max(ai, bi), ci));
}

__device__ __forceinline__ void fast_median_scalar(uint8_t m0, uint8_t m1, uint8_t m2, 
                                                   uint8_t m3, uint8_t m4, uint8_t m5, 
                                                   uint8_t m6, uint8_t m7, uint8_t m8, 
                                                   uint8_t &median){
    uint8_t r0_min, r0_med, r0_max;
    uint8_t r1_min, r1_med, r1_max;
    uint8_t r2_min, r2_med, r2_max;
    uint8_t max_of_mins, med_of_meds, min_of_maxs;

    min3(m0, m1, m2, r0_min);
    med3(m0, m1, m2, r0_med);
    max3(m0, m1, m2, r0_max);

    min3(m3, m4, m5, r1_min);
    med3(m3, m4, m5, r1_med);
    max3(m3, m4, m5, r1_max);

    min3(m6, m7, m8, r2_min);
    med3(m6, m7, m8, r2_med);
    max3(m6, m7, m8, r2_max);

    max3(r0_min, r1_min, r2_min, max_of_mins);
    med3(r0_med, r1_med, r2_med, med_of_meds);
    min3(r0_max, r1_max, r2_max, min_of_maxs);

    med3(max_of_mins, med_of_meds, min_of_maxs, median);
}

#define TILE_H 96

__global__ void image_filtering(const uint8_t* __restrict__ img_r_in, 
                                const uint8_t* __restrict__ img_g_in, 
                                const uint8_t* __restrict__ img_b_in, 
                                uint8_t* __restrict__ img_r_out, 
                                uint8_t* __restrict__ img_g_out, 
                                uint8_t* __restrict__ img_b_out, 
                                int img_h, int img_w){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= img_w) return;

    int y_start = blockIdx.y * TILE_H;

    // Allocate 27 registers for sliding window state
    uint8_t r0, r1, r2, r3, r4, r5, r6, r7, r8;
    uint8_t g0, g1, g2, g3, g4, g5, g6, g7, g8;
    uint8_t b0, b1, b2, b3, b4, b5, b6, b7, b8;

    // Calcoliamo x-1, x, x+1
    int nx0 = max(x - 1, 0);
    int nx1 = x;
    int nx2 = min(x + 1, img_w - 1);

    // Initial load: y_start - 1 and y_start
    int ny0 = max(y_start - 1, 0);
    int ny1 = min(y_start, img_h - 1);

    int i00 = ny0 * img_w + nx0;
    int i01 = ny0 * img_w + nx1;
    int i02 = ny0 * img_w + nx2;

    int i10 = ny1 * img_w + nx0;
    int i11 = ny1 * img_w + nx1;
    int i12 = ny1 * img_w + nx2;

    // Load row 0
    r0 = __ldg(&img_r_in[i00]); r1 = __ldg(&img_r_in[i01]); r2 = __ldg(&img_r_in[i02]);
    g0 = __ldg(&img_g_in[i00]); g1 = __ldg(&img_g_in[i01]); g2 = __ldg(&img_g_in[i02]);
    b0 = __ldg(&img_b_in[i00]); b1 = __ldg(&img_b_in[i01]); b2 = __ldg(&img_b_in[i02]);

    // Load row 1
    r3 = __ldg(&img_r_in[i10]); r4 = __ldg(&img_r_in[i11]); r5 = __ldg(&img_r_in[i12]);
    g3 = __ldg(&img_g_in[i10]); g4 = __ldg(&img_g_in[i11]); g5 = __ldg(&img_g_in[i12]);
    b3 = __ldg(&img_b_in[i10]); b4 = __ldg(&img_b_in[i11]); b5 = __ldg(&img_b_in[i12]);

    // Scorriamo verticalmente lungo la colonna TILE_H volte (a salti di 3)
    #pragma unroll 2
    for (int iy = 0; iy < TILE_H; iy += 3) {
        
        // --- STEP 0: rimpiazza riga 2 (r6, r7, r8) ---
        int y_curr = y_start + iy;
        if (y_curr >= img_h) break;

        int ny2 = min(y_curr + 1, img_h - 1);
        int i20 = ny2 * img_w + nx0;
        int i21 = ny2 * img_w + nx1;
        int i22 = ny2 * img_w + nx2;

        r6 = __ldg(&img_r_in[i20]); r7 = __ldg(&img_r_in[i21]); r8 = __ldg(&img_r_in[i22]);
        g6 = __ldg(&img_g_in[i20]); g7 = __ldg(&img_g_in[i21]); g8 = __ldg(&img_g_in[i22]);
        b6 = __ldg(&img_b_in[i20]); b7 = __ldg(&img_b_in[i21]); b8 = __ldg(&img_b_in[i22]);

        int out_idx = y_curr * img_w + x;
        fast_median_scalar(r0, r1, r2, r3, r4, r5, r6, r7, r8, img_r_out[out_idx]);
        fast_median_scalar(g0, g1, g2, g3, g4, g5, g6, g7, g8, img_g_out[out_idx]);
        fast_median_scalar(b0, b1, b2, b3, b4, b5, b6, b7, b8, img_b_out[out_idx]);

        // --- STEP 1: rimpiazza riga 0 (r0, r1, r2) ---
        y_curr++;
        if (y_curr >= img_h) break;

        ny2 = min(y_curr + 1, img_h - 1);
        i20 = ny2 * img_w + nx0;
        i21 = ny2 * img_w + nx1;
        i22 = ny2 * img_w + nx2;

        r0 = __ldg(&img_r_in[i20]); r1 = __ldg(&img_r_in[i21]); r2 = __ldg(&img_r_in[i22]);
        g0 = __ldg(&img_g_in[i20]); g1 = __ldg(&img_g_in[i21]); g2 = __ldg(&img_g_in[i22]);
        b0 = __ldg(&img_b_in[i20]); b1 = __ldg(&img_b_in[i21]); b2 = __ldg(&img_b_in[i22]);

        out_idx = y_curr * img_w + x;
        fast_median_scalar(r0, r1, r2, r3, r4, r5, r6, r7, r8, img_r_out[out_idx]);
        fast_median_scalar(g0, g1, g2, g3, g4, g5, g6, g7, g8, img_g_out[out_idx]);
        fast_median_scalar(b0, b1, b2, b3, b4, b5, b6, b7, b8, img_b_out[out_idx]);

        // --- STEP 2: rimpiazza riga 1 (r3, r4, r5) ---
        y_curr++;
        if (y_curr >= img_h) break;

        ny2 = min(y_curr + 1, img_h - 1);
        i20 = ny2 * img_w + nx0;
        i21 = ny2 * img_w + nx1;
        i22 = ny2 * img_w + nx2;

        r3 = __ldg(&img_r_in[i20]); r4 = __ldg(&img_r_in[i21]); r5 = __ldg(&img_r_in[i22]);
        g3 = __ldg(&img_g_in[i20]); g4 = __ldg(&img_g_in[i21]); g5 = __ldg(&img_g_in[i22]);
        b3 = __ldg(&img_b_in[i20]); b4 = __ldg(&img_b_in[i21]); b5 = __ldg(&img_b_in[i22]);

        out_idx = y_curr * img_w + x;
        fast_median_scalar(r0, r1, r2, r3, r4, r5, r6, r7, r8, img_r_out[out_idx]);
        fast_median_scalar(g0, g1, g2, g3, g4, g5, g6, g7, g8, img_g_out[out_idx]);
        fast_median_scalar(b0, b1, b2, b3, b4, b5, b6, b7, b8, img_b_out[out_idx]);
    }
}

int main(int argc, char* argv[])
{
    if(argc < 1){
        cout<<"Uso: "<<argv[0]<<" nome_immagine\n";
        return 1;
    }

    string img_name = argv[1];
    Image img = loadImage("../error_images/"+img_name+".ppm");
    Image out(img.height,img.width);
    int img_dim = img.width * img.height; 
    uint8_t *img_red_in, *img_green_in, *img_blue_in, *img_red_out, *img_green_out, *img_blue_out; 
    
    cudaMalloc(&img_red_in, img_dim);
    cudaMalloc(&img_green_in, img_dim);
    cudaMalloc(&img_blue_in, img_dim);
    cudaMalloc(&img_red_out, img_dim);
    cudaMalloc(&img_green_out, img_dim);
    cudaMalloc(&img_blue_out, img_dim);

    cudaMemcpy(img_red_in, img.r.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_green_in, img.g.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_blue_in, img.b.data(), img_dim, cudaMemcpyHostToDevice);
    for(int t = 1; t<= 256; t++){
        // Blocco orizzontale 256 thread per il perfect coalescing
        dim3 block(t, 1);
        dim3 grid((img.width  + block.x - 1) / block.x, (img.height + TILE_H - 1) / TILE_H);
        
        for(int i = 0; i<10; i++){
            auto start = chrono::high_resolution_clock::now();
            image_filtering <<< grid, block >>>(img_red_in, img_green_in, img_blue_in, img_red_out, img_green_out, img_blue_out, img.height, img.width);
            cudaDeviceSynchronize();
            auto end = chrono::high_resolution_clock::now();
            
            cout << "n_thread: " << t << " rep: " << i << " tempo: " <<chrono::duration_cast<chrono::milliseconds>(end-start).count() << " ms\n";
        }
    }
    cudaMemcpy(out.r.data(), img_red_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.g.data(), img_green_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.b.data(), img_blue_out, img_dim, cudaMemcpyDeviceToHost);

    saveImage("./output_images/"+img_name+".ppm",out);
}
