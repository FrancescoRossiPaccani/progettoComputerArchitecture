#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <atomic>

//lib per cuda 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

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

    if (format == "P6") {
        while(isspace(file.peek())) file.get();
        // Lettura massiva in blocco (Buffer) per saltare l'overhead di file.get()
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

// ---------------- SAVE ----------------
void saveImage(const string& path, Image& img){
    // Salvataggio nel formato P6 (binario) invece di P3 testuale per ridurre drasticamente 
    // l'overhead del filesystem ed evitare i dump I/O lenti degli iostream.
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

__device__ __forceinline__ unsigned int pack_uchar4(const uchar4& v) {
    return reinterpret_cast<const unsigned int&>(v);
}

__device__ __forceinline__ uchar4 unpack_uchar4(unsigned int v) {
    return reinterpret_cast<const uchar4&>(v);
}

__device__ __forceinline__ void uchar_min3(uchar4 a, uchar4 b, uchar4 c, uchar4 &minVal) {
    unsigned int pa = pack_uchar4(a);
    unsigned int pb = pack_uchar4(b);
    unsigned int pc = pack_uchar4(c);

    unsigned int r = __vminu4(__vminu4(pa, pb), pc);
    minVal = unpack_uchar4(r);
}

__device__ __forceinline__ void uchar_max3(uchar4 a, uchar4 b, uchar4 c, uchar4 &maxVal) {
    unsigned int pa = pack_uchar4(a);
    unsigned int pb = pack_uchar4(b);
    unsigned int pc = pack_uchar4(c);

    unsigned int r = __vmaxu4(__vmaxu4(pa, pb), pc);
    maxVal = unpack_uchar4(r);
}

__device__ __forceinline__ void uchar_med3(uchar4 a, uchar4 b, uchar4 c, uchar4 &medVal) {
    unsigned int pa = pack_uchar4(a);
    unsigned int pb = pack_uchar4(b);
    unsigned int pc = pack_uchar4(c);

    unsigned int ab_min = __vminu4(pa, pb);
    unsigned int ab_max = __vmaxu4(pa, pb);
    unsigned int r      = __vmaxu4(ab_min, __vminu4(ab_max, pc));

    medVal = unpack_uchar4(r);
}

__device__ void fast_median_rgb( uchar4 m0, uchar4 m1, uchar4 m2,
                                 uchar4 m3, uchar4 m4, uchar4 m5,
                                 uchar4 m6, uchar4 m7, uchar4 m8,
                                 uint8_t& out_r, uint8_t& out_g, uint8_t& out_b){
                                    
    uchar4 r0_min, r0_med, r0_max;
    uchar4 r1_min, r1_med, r1_max;
    uchar4 r2_min, r2_med, r2_max;
    uchar4 max_of_mins, med_of_meds, min_of_maxs, median;

    //riga 1
    uchar_min3(m0, m1, m2, r0_min);
    uchar_med3(m0, m1, m2, r0_med);
    uchar_max3(m0, m1, m2, r0_max);

    //riga 2
    uchar_min3(m3, m4, m5, r1_min);
    uchar_med3(m3, m4, m5, r1_med);
    uchar_max3(m3, m4, m5, r1_max);

    //riga3
    uchar_min3(m6, m7, m8, r2_min);
    uchar_med3(m6, m7, m8, r2_med);
    uchar_max3(m6, m7, m8, r2_max);

    uchar_max3(r0_min, r1_min, r2_min, max_of_mins);
    uchar_med3(r0_med, r1_med, r2_med, med_of_meds);
    uchar_min3(r0_max, r1_max, r2_max, min_of_maxs);

    uchar_med3(max_of_mins, med_of_meds, min_of_maxs, median);
    
    out_r = median.x;
    out_g = median.y;
    out_b = median.z;

}

__global__ void image_filtering(uint8_t* img_r_in, uint8_t* img_g_in,  uint8_t* img_b_in, uint8_t* img_r_out, uint8_t* img_g_out, uint8_t* img_b_out, int img_h, int img_w){

    // posizione globale
    int x = blockIdx.x * blockDim.x + threadIdx.x; // colonna
    int y = blockIdx.y * blockDim.y + threadIdx.y; // riga

    //posizione locale
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if(x >= img_w || y >= img_h) 
        return;

    int index  = y * img_w + x; //corrente
    int index_sx = index - 1;
    int index_dx = index + 1;
    int index_ax = index - img_w; // alto
    int index_bx = index + img_w; // basso

    int ty_cx = ty+1; //corrente
    int ty_dx = ty+2; //destra (basso)

    int tx_cx = tx+1; //corrente
    int tx_dx = tx+2; //destra

    __shared__ uchar4 kernel[34][35];

    kernel[ty_cx][tx_cx].x = img_r_in[index];
    kernel[ty_cx][tx_cx].y = img_g_in[index];
    kernel[ty_cx][tx_cx].z = img_b_in[index];

    if(tx == 0){
        if(x == 0){
            // Se mi trovo sul bordo sx --> replicazione
            kernel[ty_cx][tx].x = img_r_in[index];
            kernel[ty_cx][tx].y = img_g_in[index];
            kernel[ty_cx][tx].z = img_b_in[index];
        }
        else{
            //sx
            kernel[ty_cx][tx].x = img_r_in[index_sx];
            kernel[ty_cx][tx].y = img_g_in[index_sx];
            kernel[ty_cx][tx].z = img_b_in[index_sx];
        }
    }

    if(tx == blockDim.x - 1){
        if(x == img_w - 1){
            // Se mi trovo sul bordo dx --> replicazione
            kernel[ty_cx][tx_dx].x = img_r_in[index];
            kernel[ty_cx][tx_dx].y = img_g_in[index];
            kernel[ty_cx][tx_dx].z = img_b_in[index];
        }
        else{
            //dx
            kernel[ty_cx][tx_dx].x = img_r_in[index_dx];
            kernel[ty_cx][tx_dx].y = img_g_in[index_dx];
            kernel[ty_cx][tx_dx].z = img_b_in[index_dx];
        }
    }

    if(ty == 0){
        if(y == 0){
            // Se mi trovo sul bordo alto --> replicazione
            kernel[ty][tx_cx].x = img_r_in[index];
            kernel[ty][tx_cx].y = img_g_in[index];
            kernel[ty][tx_cx].z = img_b_in[index];
        }
        else{
            // alto
            kernel[ty][tx_cx].x = img_r_in[index_ax];
            kernel[ty][tx_cx].y = img_g_in[index_ax];
            kernel[ty][tx_cx].z = img_b_in[index_ax];
        }
    }

    if(ty == blockDim.y - 1){
        if(y == img_h - 1){
            // Se mi trovo sul bordo in basso --> replicazione
            kernel[ty_dx][tx_cx].x = img_r_in[index];
            kernel[ty_dx][tx_cx].y = img_g_in[index];
            kernel[ty_dx][tx_cx].z = img_b_in[index];
        }
        else{
            // basso
            kernel[ty_dx][tx_cx].x = img_r_in[index_bx];
            kernel[ty_dx][tx_cx].y = img_g_in[index_bx];
            kernel[ty_dx][tx_cx].z = img_b_in[index_bx];
        }
    }

    //da finire
    kernel[0][0].x = kernel[0][blockDim.x+1].x = kernel[blockDim.y+1][0].x = kernel[blockDim.y+1][blockDim.x+1].x = 0;
    kernel[0][0].y = kernel[0][blockDim.x+1].y = kernel[blockDim.y+1][0].y = kernel[blockDim.y+1][blockDim.x+1].y = 0;
    kernel[0][0].z = kernel[0][blockDim.x+1].z = kernel[blockDim.y+1][0].z = kernel[blockDim.y+1][blockDim.x+1].z = 0;

    __syncthreads();

    fast_median_rgb(kernel[ty][tx],kernel[ty][tx+1],kernel[ty][tx+2],
                    kernel[ty_cx][tx],kernel[ty_cx][tx+1],kernel[ty_cx][tx+2],
                    kernel[ty+2][tx],kernel[ty+2][tx+1],kernel[ty+2][tx+2], 
                    img_r_out[y*img_w+x], img_g_out[y*img_w+x], img_b_out[y*img_w+x]); 
}

int main(int argc, char* argv[])
{
    if(argc < 2){
        cout<<"Uso: "<<argv[0]<<" nome_immagine\n";
        return 1;
    }

    string img_name = argv[1];
    
    Image img = loadImage("../error_images/"+img_name+".ppm");
    Image out(img.height,img.width);

    cout<<"Immagine elaborata: "<<img.width<<"x"<<img.height<<"\n";

    int img_dim = img.width * img.height; 

    uint8_t *img_red_in, *img_green_in, *img_blue_in;
    uint8_t *img_red_out, *img_green_out, *img_blue_out; 

    // ===========================
    // ALLOCAZIONE GPU
    // ===========================
    cudaMalloc(&img_red_in,    img_dim * sizeof(uint8_t));
    cudaMalloc(&img_green_in,  img_dim * sizeof(uint8_t));
    cudaMalloc(&img_blue_in,   img_dim * sizeof(uint8_t));

    cudaMalloc(&img_red_out,   img_dim * sizeof(uint8_t));
    cudaMalloc(&img_green_out, img_dim * sizeof(uint8_t));
    cudaMalloc(&img_blue_out,  img_dim * sizeof(uint8_t));

    cudaMemcpy(img_red_in,   img.r.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_green_in, img.g.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_blue_in,  img.b.data(), img_dim, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ===========================
    // BENCHMARK COMPLETO
    // ===========================
    ofstream dati_v2("dati_v2.txt");
    if(!dati_v2){
        cerr<<"Errore apertura file dati_v2.txt\n";
        return 1;
    }
    
    dati_v2 << "x,y,avg_ms\n";
    cout << "x,y,avg_ms\n";

    for(int bx = 1; bx <= 32; bx++){
        for(int by = 1; by <= 32; by++){

            if(bx * by > 1024) continue;

            dim3 block(bx, by);
            dim3 grid((img.width  + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);

            float total_ms = 0.0f;

            // ---- 10 ITERAZIONI ----
            int iterations = 10;
            for(int i = 0; i < iterations; i++){
                cudaEventRecord(start);

                image_filtering<<<grid, block>>>(
                    img_red_in, img_green_in, img_blue_in,
                    img_red_out, img_green_out, img_blue_out,
                    img.height, img.width
                );

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float ms = 0;
                cudaEventElapsedTime(&ms, start, stop);
                total_ms += ms;
            }

            float avg_ms = total_ms / (float) iterations;

            dati_v2 << bx << "," << by << "," << avg_ms << "\n";
            cout << bx << "," << by << "," << avg_ms << "\n";
        }
    }

    // ===========================
    // COPIA RISULTATO (ultima config)
    // ===========================
    cudaMemcpy(out.r.data(), img_red_out,   img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.g.data(), img_green_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.b.data(), img_blue_out,  img_dim, cudaMemcpyDeviceToHost);

    // ===========================
    // CLEANUP
    // ===========================
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(img_red_in);
    cudaFree(img_green_in);
    cudaFree(img_blue_in);
    cudaFree(img_red_out);
    cudaFree(img_green_out);
    cudaFree(img_blue_out);

    saveImage("./output_images/"+img_name+".ppm", out);

    return 0;
}
