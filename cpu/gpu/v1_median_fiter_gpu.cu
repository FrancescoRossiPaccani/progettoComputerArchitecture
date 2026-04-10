#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>
//#include <immintrin.h>
#include <atomic>

//lib per cuda 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
/*
Manuale:
    Compilazione:
        nvcc fiel.cu -o programma
        (Vedere se posso mettere flag tipo -O3 per ottimizzazione)
        nvcc -Xptxas -O3 file.cu -o 
    Esecuzione
        ./programma
    Profiler:
        ncu --set full --target-processes all ./tuo_programma

*/



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

// ---------------- ERROR MEASUREMENT  ----------------
//long long vectorizedSAD(const uint8_t* a, const uint8_t* b, int size) {
//    long long total_diff = 0;
//    int i = 0;
//    
//    // Accumulatore di differenze assolute
//    __m256i sum_vec = _mm256_setzero_si256();
//    
//    // Processa blocchi di 32 byte (32 pixel) alla volta
//    for(; i <= size - 32; i += 32){
//        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
//        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
//        
//        // Calcola la somma delle differenze assolute per i byte (_mm256_sad_epu8)
//        // produce 4 blocchi da 64-bit con gli accumuli parziali
//        __m256i sad = _mm256_sad_epu8(va, vb);
//        sum_vec = _mm256_add_epi64(sum_vec, sad);
//    }
//    
//    // Estrai i valori finali dall'accumulatore a 256-bit
//    uint64_t sums[4];
//    _mm256_storeu_si256((__m256i*)sums, sum_vec);
//    total_diff += sums[0] + sums[1] + sums[2] + sums[3];
//
//    // Coda: elaborazione scalare dei restanti pixel finali
//    for(; i < size; i++){
//        total_diff += abs((int)a[i] - (int)b[i]);
//    }
//    return total_diff;
//}
//double calculateError(const Image& original, const Image& current) {
//    int size = original.width * original.height;
//    long long total_diff = 0;
//    total_diff += vectorizedSAD(original.r.data(), current.r.data(), size);
//    total_diff += vectorizedSAD(original.g.data(), current.g.data(), size);
//    total_diff += vectorizedSAD(original.b.data(), current.b.data(), size);
//    
//    return (double)total_diff / (size * 3.0);
//}

__device__ inline void min3(uint8_t a, uint8_t b, uint8_t c, uint8_t &minVal){
    minVal = (uint8_t)min(min((int)a, (int)b), (int)c);
}

__device__ inline void max3(uint8_t a, uint8_t b, uint8_t c, uint8_t &maxVal){
    maxVal = (uint8_t)max(max((int)a, (int)b), (int)c);
}

__device__ inline void med3(uint8_t a, uint8_t b, uint8_t c, uint8_t &medVal){
    int ai = (int)a, bi = (int)b, ci = (int)c;
    medVal = (uint8_t)max(min(ai, bi), min(max(ai, bi), ci));
}

__device__ void fast_median(uint8_t *mat, uint8_t &median){

    uint8_t r0_min, r0_med, r0_max;
    uint8_t r1_min, r1_med, r1_max;
    uint8_t r2_min, r2_med, r2_max;
    uint8_t max_of_mins, med_of_meds, min_of_maxs;

    //riga 1
    min3(mat[0],mat[1],mat[2], r0_min);
    med3(mat[0],mat[1],mat[2], r0_med);
    max3(mat[0],mat[1],mat[2], r0_max);

    //riga 2
    min3(mat[3],mat[4],mat[5], r1_min);
    med3(mat[3],mat[4],mat[5], r1_med);
    max3(mat[3],mat[4],mat[5], r1_max);

    //riga3
    min3(mat[6],mat[7],mat[8], r2_min);
    med3(mat[6],mat[7],mat[8], r2_med);
    max3(mat[6],mat[7],mat[8], r2_max);


    max3(r0_min, r1_min, r2_min, max_of_mins);
    med3(r0_med, r1_med, r2_med, med_of_meds);
    min3(r0_max, r1_max, r2_max, min_of_maxs);

    med3(max_of_mins, med_of_meds, min_of_maxs, median);
}



__global__ void image_filtering(uint8_t* img_r_in, uint8_t* img_g_in,  uint8_t* img_b_in, uint8_t* img_r_out, uint8_t* img_g_out, uint8_t* img_b_out, int img_h, int img_w){

    int x = blockIdx.x * blockDim.x + threadIdx.x; // colonna
    int y = blockIdx.y * blockDim.y + threadIdx.y; // riga

    if(x >= img_w || y >= img_h) 
        return;

    int index = y * img_w + x;

    uint8_t r_kernel[3][3];
    uint8_t g_kernel[3][3];
    uint8_t b_kernel[3][3];

    for(int dy=-1; dy<=1; dy++){
        for(int dx=-1; dx<=1; dx++){
            int nx = x + dx;
            int ny = y + dy;

            // Gestione bordi
            if(nx >= 0 && nx < img_w && ny >= 0 && ny < img_h){
                int n_idx = ny * img_w + nx;
                r_kernel[dy+1][dx+1] = img_r_in[n_idx];
                g_kernel[dy+1][dx+1] = img_g_in[n_idx];
                b_kernel[dy+1][dx+1] = img_b_in[n_idx];
            } else {
                r_kernel[dy+1][dx+1] = 0;
                g_kernel[dy+1][dx+1] = 0;
                b_kernel[dy+1][dx+1] = 0;
            }
        }
    }

    fast_median(*r_kernel, img_r_out[index]); //si può usare median 9 
    fast_median(*g_kernel, img_g_out[index]); //si può usare median 9
    fast_median(*b_kernel, img_b_out[index]); //si può usare median 9

}

// ---------------- MAIN ----------------
int main(int argc, char* argv[])
{
    if(argc < 1){
        cout<<"Uso: "<<argv[0]<<" nome_immagine\n";
        return 1;
    }

    string img_name = argv[1];
    //int n_threads = stoi(argv[2]);
    
    int img_dim = 0;
    Image img = loadImage("../error_images/"+img_name+".ppm");
    Image out(img.height,img.width);
    cout<<"Immagine "<<img.width<<"x"<<img.height<<"\n";

    //Calcolo MAE iniziale
    //Image img_original = loadImage("../input_images/"+img_name+".ppm");
    //double initial_error = calculateError(img_original, img);
    //cout<<"Errore medio sui pixel pre-filtro (Rumore): "<<initial_error<<" (MAE su scala 0-255)\n";


    //Copro applicazione
    img_dim = img.width * img.height; //x3 per ogni canale
    uint8_t *img_red_in, *img_green_in, *img_blue_in, *img_red_out, *img_green_out, *img_blue_out; 
    
    // uchar3* img_in; Potrei usarlo ma dovrei costruirlo scorrendo nuovamente l'immagine 
    // per cui forse ha senso al momento non metterlo perchè si dovrebbe ricostruire load e store magari poi si può vedere se è meglio

    //alloco memoria 

    cudaMalloc(&img_red_in,    img_dim * sizeof(uint8_t));
    cudaMalloc(&img_green_in,  img_dim * sizeof(uint8_t));
    cudaMalloc(&img_blue_in,   img_dim * sizeof(uint8_t));

    cudaMalloc(&img_red_out,   img_dim * sizeof(uint8_t));
    cudaMalloc(&img_green_out, img_dim * sizeof(uint8_t));
    cudaMalloc(&img_blue_out,  img_dim * sizeof(uint8_t));

    //copio memoria

    cudaMemcpy(img_red_in,   img.r.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_green_in, img.g.data(), img_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_blue_in,  img.b.data(), img_dim, cudaMemcpyHostToDevice);

    int n_threads = 256;
    //image_filtering<<< img_dim/n_threads, n_threads>>>(img_red_in, img_green_in, img_blue_in, img_red_out, img_green_out, img_blue_out, img_dim);
    //meglio questa che prende parte superiore

    int dim_block = 16;
    dim3 block(dim_block, dim_block);
    dim3 grid((img.width  + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);
    
    auto start = chrono::high_resolution_clock::now();
    image_filtering <<< grid, block >>>(img_red_in, img_green_in, img_blue_in, img_red_out, img_green_out, img_blue_out, img.height, img.width);
    //serve a far aspettare la cpu che la gpu finisca
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    //Prendo dati elaborati da GPU
    
    cudaMemcpy(out.r.data(), img_red_out  , img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.g.data(), img_green_out, img_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.b.data(), img_blue_out , img_dim, cudaMemcpyDeviceToHost);

    //Fine applicazione

    // Statistiche finali
    //cout<<"Tempo: " << chrono::duration_cast<chrono::microseconds>(end-start).count() <<" µs\n";
    cout<<"Tempo: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() <<" ms\n";

    //double final_error = calculateError(img_original, out);
    //cout<<"Errore: "<<final_error<<"\n";

    saveImage("./output_images/"+img_name+".ppm",out);
}
