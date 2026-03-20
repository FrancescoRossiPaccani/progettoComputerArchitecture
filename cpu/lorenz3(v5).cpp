#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>
#include <atomic>

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

// ---------------- CORE ----------------
inline void updateMedian(int& m, int& count, uint8_t* hist, int mid)
{
    // Scendiamo col mediano solo se rimuovendo la frequenza attuale superiamo ancora il mid
    while(m > 0 && count - hist[m] > mid){
        count -= hist[m];
        m--;
    }
    // Saliamo finchè le frequenze dei baricentri precedenti non superano la soglia
    while(m < 255 && count <= mid){
        m++;
        count += hist[m];
    }
}

// 🔥 funzione ottimizzata per un canale
void processChannel(uint8_t* __restrict in,
                    uint8_t* __restrict out,
                    int w, int h,
                    int kernel, int threshold,
                    int start_y, int end_y,
                    int bucket_size)
{
    int radius = kernel/2;
    int mid = (kernel*kernel)/2;
    // Buffer per pre-calcolare gli offset delle righe
    std::vector<int> row_offsets(2 * radius + 1);

    for(int bx=0; bx<w; bx+=bucket_size)
    {
        int end_x = min(bx+bucket_size, w);

        for(int y=start_y; y<end_y; y++)
        {
            uint8_t hist[256] = {0}; // MICRO-OTTIMIZZAZIONE: occupazione in cache L1 dimezzata rispetto a uint16_t

            // PRECALCOLO: calcoliamo y+fy (incluso min/max e moltipicazioni) solo una volta per ogni riga y
            for(int fy=-radius; fy<=radius; fy++)
            {
                row_offsets[fy + radius] = min(max(y+fy,0), h-1) * w;
            }

            // init histogram (start-edge del bucket bx)
            for(int fy=-radius; fy<=radius; fy++)
            {
                int row_offset = row_offsets[fy + radius];
                for(int fx=bx-radius; fx<=bx+radius; fx++)
                {
                    int ix = min(max(fx,0), w-1);
                    hist[in[row_offset + ix]]++;
                }
            }

            // initial median
            int m=0,count=0;
            while(count<=mid){
                count += hist[m];
                m++;
            }
            m--;

            for(int x=bx; x<end_x; x++)
            {
                int idx = y*w+x;
                int center = in[idx];

                out[idx] = (abs(center-m)>threshold)?m:center;

                if(x==end_x-1) break;

                int left  = max(x-radius,0);
                int right = min(x+radius+1,w-1);

                for(int fy=-radius; fy<=radius; fy++)
                {
                    int row_offset = row_offsets[fy + radius];

                    int oldv = in[row_offset + left];
                    int newv = in[row_offset + right];

                    // BRANCHLESS: Evita predizioni errate della CPU che rallentano il ciclo
                    hist[oldv]--; count -= (oldv <= m);
                    hist[newv]++; count += (newv <= m);
                }

                updateMedian(m,count,hist,mid);
            }
        }
    }
}

// ---------------- THREAD WORKER ----------------
void worker(Image& img, Image& out,
            int kernel, int threshold,
            std::atomic<int>& next_row, int end_y, int chunk_size,
            int bucket_size)
{
    while (true)
    {
        int start_y = next_row.fetch_add(chunk_size, std::memory_order_relaxed);
        if (start_y >= end_y) break;

        int current_end_y = std::min(start_y + chunk_size, end_y);

        processChannel(img.r.data(), out.r.data(),
                       img.width, img.height,
                       kernel, threshold,
                       start_y, current_end_y, bucket_size);

        processChannel(img.g.data(), out.g.data(),
                       img.width, img.height,
                       kernel, threshold,
                       start_y, current_end_y, bucket_size);

        processChannel(img.b.data(), out.b.data(),
                       img.width, img.height,
                       kernel, threshold,
                       start_y, current_end_y, bucket_size);
    }
}

// ---------------- ERROR MEASUREMENT (VETTORIZZATO AVX2) ----------------
long long vectorizedSAD(const uint8_t* a, const uint8_t* b, int size) {
    long long total_diff = 0;
    int i = 0;
    
    // Accumulatore di differenze assolute
    __m256i sum_vec = _mm256_setzero_si256();
    
    // Processa blocchi di 32 byte (32 pixel) alla volta
    for(; i <= size - 32; i += 32){
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        
        // Calcola la somma delle differenze assolute per i byte (_mm256_sad_epu8)
        // produce 4 blocchi da 64-bit con gli accumuli parziali
        __m256i sad = _mm256_sad_epu8(va, vb);
        sum_vec = _mm256_add_epi64(sum_vec, sad);
    }
    
    // Estrai i valori finali dall'accumulatore a 256-bit
    uint64_t sums[4];
    _mm256_storeu_si256((__m256i*)sums, sum_vec);
    total_diff += sums[0] + sums[1] + sums[2] + sums[3];

    // Coda: elaborazione scalare dei restanti pixel finali
    for(; i < size; i++){
        total_diff += abs((int)a[i] - (int)b[i]);
    }
    return total_diff;
}

double calculateError(const Image& original, const Image& current) {
    int size = original.width * original.height;
    long long total_diff = 0;
    total_diff += vectorizedSAD(original.r.data(), current.r.data(), size);
    total_diff += vectorizedSAD(original.g.data(), current.g.data(), size);
    total_diff += vectorizedSAD(original.b.data(), current.b.data(), size);
    
    return (double)total_diff / (size * 3.0);
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[])
{
    if(argc < 5){
        cout<<"Uso: "<<argv[0]<<" nome_immagine num_thread dimensione_kernel dimensione_bucket\n";
        return 1;
    }

    string img_name = argv[1];
    int n_threads = stoi(argv[2]);
    int kernel = stoi(argv[3]);
    int bucket_size = stoi(argv[4]);

    Image img = loadImage("./error_images/"+img_name+".ppm");
    Image out(img.height,img.width);

    // Carichiamo l'originale per fare un confronto (non viene misurato nel tempo di elaborazione)
    Image img_original = loadImage("./input_images/"+img_name+".ppm");
    double initial_error = calculateError(img_original, img);
    cout<<"Immagine "<<img.width<<"x"<<img.height<<"\n";
    cout<<"Errore medio sui pixel pre-filtro (Rumore): "<<initial_error<<" (MAE su scala 0-255)\n";

    int threshold = 40;

    auto start = chrono::high_resolution_clock::now();

    vector<thread> threads;
    std::atomic<int> next_row(0);
    // Dinamic Load Balancing: dividiamo il lavoro in un maggior numero di blocchi 
    // così i thread si smistano dinamicamente e non creano squilibri di tempi di esecuzione
    int chunk_size = std::max(1, img.height / (n_threads * 10));

    for(int i=0;i<n_threads;i++)
    {
        threads.emplace_back(worker,
                             ref(img), ref(out),
                             kernel, threshold,
                             ref(next_row), img.height, chunk_size,
                             bucket_size);
    }

    for(auto& t:threads) t.join();

    auto end = chrono::high_resolution_clock::now();

    cout<<"Tempo: "
        <<chrono::duration_cast<chrono::milliseconds>(end-start).count()
        <<" ms\n";

    double final_error = calculateError(img_original, out);
    cout<<"Errore medio sui pixel post-filtro (Migliorata): "<<final_error<<" (MAE su scala 0-255)\n";

    saveImage("./output_images/"+img_name+".ppm",out);
}