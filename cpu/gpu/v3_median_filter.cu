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

// ---------------- DEVICE ----------------
__device__ __forceinline__ int clamp(int v, int low, int high){
    return v < low ? low : (v > high ? high : v);
}

// median su uchar4 (RGB insieme)
__device__ __forceinline__ uchar4 median9(
    uchar4 a0, uchar4 a1, uchar4 a2,
    uchar4 b0, uchar4 b1, uchar4 b2,
    uchar4 c0, uchar4 c1, uchar4 c2)
{
    #define MIN(a,b) __vminu4(a,b)
    #define MAX(a,b) __vmaxu4(a,b)

    unsigned int A0 = *(unsigned int*)&a0;
    unsigned int A1 = *(unsigned int*)&a1;
    unsigned int A2 = *(unsigned int*)&a2;
    unsigned int B0 = *(unsigned int*)&b0;
    unsigned int B1 = *(unsigned int*)&b1;
    unsigned int B2 = *(unsigned int*)&b2;
    unsigned int C0 = *(unsigned int*)&c0;
    unsigned int C1 = *(unsigned int*)&c1;
    unsigned int C2 = *(unsigned int*)&c2;

    unsigned int r0min = MIN(MIN(A0,A1),A2);
    unsigned int r0max = MAX(MAX(A0,A1),A2);
    unsigned int r0med = MAX(MIN(A0,A1), MIN(MAX(A0,A1),A2));

    unsigned int r1min = MIN(MIN(B0,B1),B2);
    unsigned int r1max = MAX(MAX(B0,B1),B2);
    unsigned int r1med = MAX(MIN(B0,B1), MIN(MAX(B0,B1),B2));

    unsigned int r2min = MIN(MIN(C0,C1),C2);
    unsigned int r2max = MAX(MAX(C0,C1),C2);
    unsigned int r2med = MAX(MIN(C0,C1), MIN(MAX(C0,C1),C2));

    unsigned int max_of_mins = MAX(MAX(r0min, r1min), r2min);
    unsigned int min_of_maxs = MIN(MIN(r0max, r1max), r2max);
    unsigned int med_of_meds = MAX(MIN(r0med, r1med), MIN(MAX(r0med, r1med), r2med));

    unsigned int med = MAX(MIN(max_of_mins, med_of_meds),
                           MIN(MAX(max_of_mins, med_of_meds), min_of_maxs));

    return *(uchar4*)&med;
}

// ---------------- KERNEL ----------------
__global__ void image_filtering(
    uchar4* img_in,
    uchar4* img_out,
    int img_h, int img_w)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= img_w || y >= img_h) return;

    int idx = y * img_w + x;

    uchar4 k[9];

    int t = 0;
    for(int dy=-1; dy<=1; dy++){
        for(int dx=-1; dx<=1; dx++){

            int nx = clamp(x+dx, 0, img_w-1);
            int ny = clamp(y+dy, 0, img_h-1);

            k[t++] = img_in[ny * img_w + nx];
        }
    }

    img_out[idx] = median9(
        k[0],k[1],k[2],
        k[3],k[4],k[5],
        k[6],k[7],k[8]
    );
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

    // packing uchar4
    vector<uchar4> h_img(img_dim);
    for(int i=0;i<img_dim;i++){
        h_img[i] = make_uchar4(img.r[i], img.g[i], img.b[i], 0);
    }

    uchar4 *d_in, *d_out;

    cudaMalloc(&d_in,  img_dim * sizeof(uchar4));
    cudaMalloc(&d_out, img_dim * sizeof(uchar4));

    cudaMemcpy(d_in, h_img.data(), img_dim * sizeof(uchar4), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((img.width+15)/16, (img.height+15)/16);

    auto start = chrono::high_resolution_clock::now();

    image_filtering<<<grid, block>>>(d_in, d_out, img.height, img.width);
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();

    vector<uchar4> h_out(img_dim);
    cudaMemcpy(h_out.data(), d_out, img_dim * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // unpack
    for(int i=0;i<img_dim;i++){
        out.r[i] = h_out[i].x;
        out.g[i] = h_out[i].y;
        out.b[i] = h_out[i].z;
    }

    cout<<"Tempo: "
        <<chrono::duration_cast<chrono::milliseconds>(end-start).count()
        <<" ms\n";

    saveImage("./output_images/"+img_name+".ppm",out);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}