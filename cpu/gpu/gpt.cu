#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// ---------------- IMAGE ----------------
struct Image{
    int width, height;
    vector<uint8_t> r, g, b;

    Image(int h,int w):width(w),height(h),
        r(h*w), g(h*w), b(h*w){}
};

// ---------------- LOAD ----------------
Image loadImage(const string& path){
    ifstream file(path, ios::binary);
    if(!file){
        cerr<<"Errore apertura file\n";
        exit(1);
    }

    string format;
    file>>format;

    int w,h,maxval;
    file>>w>>h>>maxval;

    Image img(h,w);

    while(isspace(file.peek())) file.get();

    vector<uint8_t> buffer(w*h*3);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for(int i=0;i<h*w;i++){
        img.r[i] = buffer[i*3];
        img.g[i] = buffer[i*3+1];
        img.b[i] = buffer[i*3+2];
    }

    return img;
}

// ---------------- SAVE ----------------
void saveImage(const string& path, Image& img){
    ofstream file(path, ios::binary);

    file<<"P6\n"<<img.width<<" "<<img.height<<"\n255\n";

    vector<uint8_t> buffer(img.width*img.height*3);

    for(int i=0;i<img.width*img.height;i++){
        buffer[i*3]   = img.r[i];
        buffer[i*3+1] = img.g[i];
        buffer[i*3+2] = img.b[i];
    }

    file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
}

// ---------------- DEVICE UTILS ----------------
__device__ __forceinline__ int clamp(int v, int low, int high){
    return v < low ? low : (v > high ? high : v);
}

__device__ __forceinline__ uint8_t my_min(uint8_t a, uint8_t b){
    return a < b ? a : b;
}

__device__ __forceinline__ uint8_t my_max(uint8_t a, uint8_t b){
    return a > b ? a : b;
}

__device__ __forceinline__ void min3(uint8_t a, uint8_t b, uint8_t c, uint8_t &minVal){
    minVal = my_min(my_min(a,b),c);
}

__device__ __forceinline__ void max3(uint8_t a, uint8_t b, uint8_t c, uint8_t &maxVal){
    maxVal = my_max(my_max(a,b),c);
}

__device__ __forceinline__ void med3(uint8_t a, uint8_t b, uint8_t c, uint8_t &medVal){
    uint8_t ab_min = my_min(a,b);
    uint8_t ab_max = my_max(a,b);
    medVal = my_max(ab_min, my_min(ab_max,c));
}

__device__ void fast_median(uint8_t *mat, uint8_t &median){

    uint8_t r0_min, r0_med, r0_max;
    uint8_t r1_min, r1_med, r1_max;
    uint8_t r2_min, r2_med, r2_max;
    uint8_t max_of_mins, med_of_meds, min_of_maxs;

    min3(mat[0],mat[1],mat[2], r0_min);
    med3(mat[0],mat[1],mat[2], r0_med);
    max3(mat[0],mat[1],mat[2], r0_max);

    min3(mat[3],mat[4],mat[5], r1_min);
    med3(mat[3],mat[4],mat[5], r1_med);
    max3(mat[3],mat[4],mat[5], r1_max);

    min3(mat[6],mat[7],mat[8], r2_min);
    med3(mat[6],mat[7],mat[8], r2_med);
    max3(mat[6],mat[7],mat[8], r2_max);

    max3(r0_min, r1_min, r2_min, max_of_mins);
    med3(r0_med, r1_med, r2_med, med_of_meds);
    min3(r0_max, r1_max, r2_max, min_of_maxs);

    med3(max_of_mins, med_of_meds, min_of_maxs, median);
}

// ---------------- KERNEL ----------------
__global__ void image_filtering(
    uint8_t* img_r_in,
    uint8_t* img_g_in,
    uint8_t* img_b_in,
    uint8_t* img_r_out,
    uint8_t* img_g_out,
    uint8_t* img_b_out,
    int img_h, int img_w)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    // shared memory con padding anti bank conflict
    __shared__ uint8_t tile_r[18][18+1];
    __shared__ uint8_t tile_g[18][18+1];
    __shared__ uint8_t tile_b[18][18+1];

    // caricamento cooperativo
    for(int dy = -1; dy <= 1; dy++){
        for(int dx = -1; dx <= 1; dx++){

            int lx = tx + dx + 1;
            int ly = ty + dy + 1;

            if(lx >= 0 && lx < blockDim.x+2 &&
               ly >= 0 && ly < blockDim.y+2){

                int gx = clamp(x + dx, 0, img_w - 1);
                int gy = clamp(y + dy, 0, img_h - 1);

                int idx = gy * img_w + gx;

                tile_r[ly][lx] = img_r_in[idx];
                tile_g[ly][lx] = img_g_in[idx];
                tile_b[ly][lx] = img_b_in[idx];
            }
        }
    }

    __syncthreads();

    if(x >= img_w || y >= img_h) return;

    int index = y * img_w + x;

    uint8_t r_kernel[9];
    uint8_t g_kernel[9];
    uint8_t b_kernel[9];

    int k = 0;
    for(int dy=0; dy<3; dy++){
        for(int dx=0; dx<3; dx++){
            r_kernel[k] = tile_r[ty+dy][tx+dx];
            g_kernel[k] = tile_g[ty+dy][tx+dx];
            b_kernel[k] = tile_b[ty+dy][tx+dx];
            k++;
        }
    }

    fast_median(r_kernel, img_r_out[index]);
    fast_median(g_kernel, img_g_out[index]);
    fast_median(b_kernel, img_b_out[index]);
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[])
{
    if(argc < 2){
        cout<<"Uso: "<<argv[0]<<" nome_immagine\n";
        return 1;
    }

    string img_name = argv[1];

    Image img = loadImage("../cpu/error_images/"+img_name+".ppm");
    Image out(img.height,img.width);

    cout<<"Immagine "<<img.width<<"x"<<img.height<<"\n";

    int img_dim = img.width * img.height;

    uint8_t *d_r_in, *d_g_in, *d_b_in;
    uint8_t *d_r_out, *d_g_out, *d_b_out;

    cudaMalloc(&d_r_in, img_dim);
    cudaMalloc(&d_g_in, img_dim);
    cudaMalloc(&d_b_in, img_dim);

    cudaMalloc(&d_r_out, img_dim);
    cudaMalloc(&d_g_out, img_dim);
    cudaMalloc(&d_b_out, img_dim);

    cudaMemcpy(d_r_in, img.r.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_in, img.g.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_in, img.b.data(), img_dim, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((img.width+15)/16, (img.height+15)/16);

    auto start = chrono::high_resolution_clock::now();

    image_filtering<<<grid, block>>>(d_r_in, d_g_in, d_b_in,
                                    d_r_out, d_g_out, d_b_out,
                                    img.height, img.width);

    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();

    cudaMemcpy(out.r.data(), d_r_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.g.data(), d_g_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.b.data(), d_b_out, img_dim, cudaMemcpyDeviceToHost);

    cout<<"Tempo: "
        <<chrono::duration_cast<chrono::milliseconds>(end-start).count()
        <<" ms\n";

    saveImage("./output_images/"+img_name+".ppm",out);

    cudaFree(d_r_in); cudaFree(d_g_in); cudaFree(d_b_in);
    cudaFree(d_r_out); cudaFree(d_g_out); cudaFree(d_b_out);

    return 0;
}