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

__global__ void image_filtering(const uint8_t* __restrict__ img_r_in, 
                                const uint8_t* __restrict__ img_g_in, 
                                const uint8_t* __restrict__ img_b_in, 
                                uint8_t* __restrict__ img_r_out, 
                                uint8_t* __restrict__ img_g_out, 
                                uint8_t* __restrict__ img_b_out, 
                                int img_h, int img_w){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= img_w || y >= img_h) 
        return;

    int out_idx = y * img_w + x;
    uint8_t m0, m1, m2, m3, m4, m5, m6, m7, m8;

    if (x > 0 && x < img_w - 1 && y > 0 && y < img_h - 1) {
        int idx = out_idx;
        int idx_up = idx - img_w;
        int idx_dn = idx + img_w;
        
        // RED
        m0 = __ldg(&img_r_in[idx_up - 1]); m1 = __ldg(&img_r_in[idx_up]); m2 = __ldg(&img_r_in[idx_up + 1]);
        m3 = __ldg(&img_r_in[idx - 1]);    m4 = __ldg(&img_r_in[idx]);    m5 = __ldg(&img_r_in[idx + 1]);
        m6 = __ldg(&img_r_in[idx_dn - 1]); m7 = __ldg(&img_r_in[idx_dn]); m8 = __ldg(&img_r_in[idx_dn + 1]);
        fast_median_scalar(m0, m1, m2, m3, m4, m5, m6, m7, m8, img_r_out[out_idx]);
        
        // GREEN
        m0 = __ldg(&img_g_in[idx_up - 1]); m1 = __ldg(&img_g_in[idx_up]); m2 = __ldg(&img_g_in[idx_up + 1]);
        m3 = __ldg(&img_g_in[idx - 1]);    m4 = __ldg(&img_g_in[idx]);    m5 = __ldg(&img_g_in[idx + 1]);
        m6 = __ldg(&img_g_in[idx_dn - 1]); m7 = __ldg(&img_g_in[idx_dn]); m8 = __ldg(&img_g_in[idx_dn + 1]);
        fast_median_scalar(m0, m1, m2, m3, m4, m5, m6, m7, m8, img_g_out[out_idx]);

        // BLUE
        m0 = __ldg(&img_b_in[idx_up - 1]); m1 = __ldg(&img_b_in[idx_up]); m2 = __ldg(&img_b_in[idx_up + 1]);
        m3 = __ldg(&img_b_in[idx - 1]);    m4 = __ldg(&img_b_in[idx]);    m5 = __ldg(&img_b_in[idx + 1]);
        m6 = __ldg(&img_b_in[idx_dn - 1]); m7 = __ldg(&img_b_in[idx_dn]); m8 = __ldg(&img_b_in[idx_dn + 1]);
        fast_median_scalar(m0, m1, m2, m3, m4, m5, m6, m7, m8, img_b_out[out_idx]);
    } else {
        int ny0 = max(y - 1, 0);
        int ny1 = y;
        int ny2 = min(y + 1, img_h - 1);
        int nx0 = max(x - 1, 0);
        int nx1 = x;
        int nx2 = min(x + 1, img_w - 1);

        int idx00 = ny0 * img_w + nx0; int idx01 = ny0 * img_w + nx1; int idx02 = ny0 * img_w + nx2;
        int idx10 = ny1 * img_w + nx0; int idx11 = ny1 * img_w + nx1; int idx12 = ny1 * img_w + nx2;
        int idx20 = ny2 * img_w + nx0; int idx21 = ny2 * img_w + nx1; int idx22 = ny2 * img_w + nx2;

        m0 = __ldg(&img_r_in[idx00]); m1 = __ldg(&img_r_in[idx01]); m2 = __ldg(&img_r_in[idx02]);
        m3 = __ldg(&img_r_in[idx10]); m4 = __ldg(&img_r_in[idx11]); m5 = __ldg(&img_r_in[idx12]);
        m6 = __ldg(&img_r_in[idx20]); m7 = __ldg(&img_r_in[idx21]); m8 = __ldg(&img_r_in[idx22]);
        fast_median_scalar(m0, m1, m2, m3, m4, m5, m6, m7, m8, img_r_out[out_idx]);

        m0 = __ldg(&img_g_in[idx00]); m1 = __ldg(&img_g_in[idx01]); m2 = __ldg(&img_g_in[idx02]);
        m3 = __ldg(&img_g_in[idx10]); m4 = __ldg(&img_g_in[idx11]); m5 = __ldg(&img_g_in[idx12]);
        m6 = __ldg(&img_g_in[idx20]); m7 = __ldg(&img_g_in[idx21]); m8 = __ldg(&img_g_in[idx22]);
        fast_median_scalar(m0, m1, m2, m3, m4, m5, m6, m7, m8, img_g_out[out_idx]);

        m0 = __ldg(&img_b_in[idx00]); m1 = __ldg(&img_b_in[idx01]); m2 = __ldg(&img_b_in[idx02]);
        m3 = __ldg(&img_b_in[idx10]); m4 = __ldg(&img_b_in[idx11]); m5 = __ldg(&img_b_in[idx12]);
        m6 = __ldg(&img_b_in[idx20]); m7 = __ldg(&img_b_in[idx21]); m8 = __ldg(&img_b_in[idx22]);
        fast_median_scalar(m0, m1, m2, m3, m4, m5, m6, m7, m8, img_b_out[out_idx]);
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

    dim3 block(32, 8);
    dim3 grid((img.width  + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);
    
    auto start = chrono::high_resolution_clock::now();
    image_filtering <<< grid, block >>>(img_red_in, img_green_in, img_blue_in, img_red_out, img_green_out, img_blue_out, img.height, img.width);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    
    cout<<"Tempo: " <<chrono::duration_cast<chrono::milliseconds>(end-start).count() <<" ms\n";
    
    cudaMemcpy(out.r.data(), img_red_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.g.data(), img_green_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.b.data(), img_blue_out, img_dim, cudaMemcpyDeviceToHost);

    saveImage("./output_images/"+img_name+".ppm",out);
}
