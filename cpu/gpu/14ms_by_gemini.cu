#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// Struttura per il ritorno dei valori nel kernel (sostituisce std::tuple)
struct MMM {
    uint32_t v_min, v_med, v_max;
};

struct Image {
    int width, height;
    vector<uint32_t> pixels; // Pixel impacchettati come 0xAABBGGRR
    Image(int h, int w) : width(w), height(h), pixels(h * w) {}
};

// Funzione per saltare commenti e spazi bianchi nei file PPM
void skipComments(ifstream& f) {
    char ch;
    while (f.get(ch) && (isspace(ch) || ch == '#')) {
        if (ch == '#') f.ignore(1000, '\n');
    }
    f.putback(ch);
}

// ---------------- LOAD ----------------
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
        for (int i = 0; i < h * w; i++) {
            img.pixels[i] = buffer[i*3] | (buffer[i*3+1] << 8) | (buffer[i*3+2] << 16) | (255u << 24);
        }
    } else if (format == "P3") {
        for (int i = 0; i < h * w; i++) {
            int r, g, b;
            file >> r >> g >> b;
            img.pixels[i] = (uint8_t)r | ((uint8_t)g << 8) | ((uint8_t)b << 16) | (255u << 24);
        }
    } else {
        cerr << "Formato non supportato: " << format << endl; exit(1);
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
        buffer[i * 3]     = p & 0xFF;         // R
        buffer[i * 3 + 1] = (p >> 8) & 0xFF;  // G
        buffer[i * 3 + 2] = (p >> 16) & 0xFF; // B
    }
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// ---------------- KERNEL HELPERS ----------------
__device__ __forceinline__ uint32_t vmin4_3(uint32_t a, uint32_t b, uint32_t c) {
    return __vminu4(__vminu4(a, b), c);
}
__device__ __forceinline__ uint32_t vmax4_3(uint32_t a, uint32_t b, uint32_t c) {
    return __vmaxu4(__vmaxu4(a, b), c);
}
__device__ __forceinline__ uint32_t vmed4_3(uint32_t a, uint32_t b, uint32_t c) {
    return __vmaxu4(__vminu4(a, b), __vminu4(__vmaxu4(a, b), c));
}

#define N_ROWS 64

// ---------------- KERNEL ----------------
__global__ void image_filtering_simd_warp_opt(const uint32_t* __restrict__ img_in, uint32_t* __restrict__ img_out, int img_h, int img_w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x; 
    int y_start = blockIdx.y * blockDim.y * N_ROWS + threadIdx.y * N_ROWS;

    if (x >= img_w || y_start >= img_h) return;

    // Fetch con Warp Shuffle per gestire i vicini orizzontali
    auto fetch_mmm = [&](int row_y) -> MMM {
        int target_y = max(0, min(row_y, img_h - 1));
        uint32_t p_c = __ldg(&img_in[target_y * img_w + x]);
        
        uint32_t p_l = __shfl_up_sync(0xffffffff, p_c, 1);
        uint32_t p_r = __shfl_down_sync(0xffffffff, p_c, 1);
        
        if (lane == 0)  p_l = __ldg(&img_in[target_y * img_w + max(0, x - 1)]);
        if (lane == 31) p_r = __ldg(&img_in[target_y * img_w + min(img_w - 1, x + 1)]);
        
        MMM res;
        res.v_min = vmin4_3(p_l, p_c, p_r);
        res.v_med = vmed4_3(p_l, p_c, p_r);
        res.v_max = vmax4_3(p_l, p_c, p_r);
        return res;
    };

    MMM r0 = fetch_mmm(y_start - 1);
    MMM r1 = fetch_mmm(y_start);

    #pragma unroll
    for (int k = 0; k < N_ROWS; k++) {
        int y = y_start + k;
        if (y >= img_h) break;

        MMM r2 = fetch_mmm(y + 1);

        uint32_t max_min = vmax4_3(r0.v_min, r1.v_min, r2.v_min);
        uint32_t med_med = vmed4_3(r0.v_med, r1.v_med, r2.v_med);
        uint32_t min_max = vmin4_3(r0.v_max, r1.v_max, r2.v_max);

        img_out[y * img_w + x] = vmed4_3(max_min, med_med, min_max);

        r0 = r1;
        r1 = r2;
    }
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[]) {
    if (argc < 2) { cout << "Uso: " << argv[0] << " nome_file_senza_estensione\n"; return 1; }
    string img_name = argv[1];
    
    // Caricamento
    Image img = loadImage("../error_images/" + img_name + ".ppm");
    Image out(img.height, img.width);
    
    uint32_t *d_img_in, *d_img_out;
    cudaMalloc(&d_img_in, img.pixels.size() * sizeof(uint32_t));
    cudaMalloc(&d_img_out, img.pixels.size() * sizeof(uint32_t));
    
    cudaMemcpy(d_img_in, img.pixels.data(), img.pixels.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Configurazione Warp-oriented (32x4 threads)
    dim3 block(256, 1);
    dim3 grid((img.width + block.x - 1) / block.x, (img.height + (block.y * N_ROWS) - 1) / (block.y * N_ROWS));

    //auto start = chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start GPU timer
    cudaEventRecord(start);

    // Esecuzione
    image_filtering_simd_warp_opt<<<grid, block>>>(d_img_in, d_img_out, img.height, img.width);
      // stop GPU timer
    cudaEventRecord(stop);

    // aspetta fine kernel
    cudaEventSynchronize(stop);

    // calcolo tempo
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cout << "Kernel time: " << ms << " ms\n";

    
    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(out.pixels.data(), d_img_out, img.pixels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Salvataggio
    saveImage("./output_images/" + img_name + ".ppm", out);

    cudaFree(d_img_in);
    cudaFree(d_img_out);
    return 0;
}