#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>
#include <atomic>
#include <cstring>

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
inline void updateMedian(int& m, int& count, uint8_t* hist, int mid){

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

// MACRO AVX2 per Sorting Network 3x3
inline __m256i min3(__m256i a, __m256i b, __m256i c) { return _mm256_min_epu8(a, _mm256_min_epu8(b, c)); }
inline __m256i max3(__m256i a, __m256i b, __m256i c) { return _mm256_max_epu8(a, _mm256_max_epu8(b, c)); }
inline __m256i med3(__m256i a, __m256i b, __m256i c) { return _mm256_max_epu8(_mm256_min_epu8(a, b), _mm256_min_epu8(_mm256_max_epu8(a, b), c)); }


// 🔥 funzione ottimizzata Base (Con Prefetch)
template<int RADIUS>
void processChannel(uint8_t* __restrict in, uint8_t* __restrict out, int w, int h, int threshold, int start_y, int end_y, int bucket_size){

    int kernel = 2 * RADIUS + 1;
    int mid = (kernel*kernel)/2;
    int prefetch_dist = 64; // Proviamo a prefetchare 1-2 cache line in avanti 

    // Buffer per pre-calcolare gli offset delle righe
    std::vector<int> row_offsets(2 * RADIUS + 1);

    for(int bx=0; bx<w; bx+=bucket_size){
        int end_x = min(bx+bucket_size, w);

        for(int y=start_y; y<end_y; y++){
            uint8_t hist[256] = {0}; // MICRO-OTTIMIZZAZIONE: occupazione in cache L1 dimezzata rispetto a uint16_t

            // PRECALCOLO: calcoliamo y+fy (incluso min/max e moltipicazioni) solo una volta per ogni riga y
            #pragma GCC unroll 15
            for(int fy=-RADIUS; fy<=RADIUS; fy++)
                row_offsets[fy + RADIUS] = min(max(y+fy,0), h-1) * w;

            // init histogram (start-edge del bucket bx)
            #pragma GCC unroll 15
            for(int fy=-RADIUS; fy<=RADIUS; fy++){
                int row_offset = row_offsets[fy + RADIUS];
                for(int fx=bx-RADIUS; fx<=bx+RADIUS; fx++)
                    hist[in[row_offset + min(max(fx,0), w-1)]]++;
            }

            // initial median
            int m=0,count=0;
            while(count<=mid){
                count += hist[m];
                m++;
            }
            m--;

            for(int x=bx; x<end_x; x++){
                // *** 3. SOFTWARE PREFETCHING ***
                // Chiediamo asincronamente alla CPU di prelevare in L1 i pixel futuri che il kernel processerà tra qualche ciclo
                #pragma GCC unroll 15
                for(int fy=-RADIUS; fy<=RADIUS; fy++){
                    _mm_prefetch((const char*)&in[row_offsets[fy + RADIUS] + x + prefetch_dist], _MM_HINT_T0);
                }

                int idx = y*w+x;
                int center = in[idx];

                out[idx] = (abs(center-m)>threshold)?m:center;

                if(x==end_x-1) 
                    break;

                int left  = max(x-RADIUS,0);
                int right = min(x+RADIUS+1,w-1);

                #pragma GCC unroll 15
                for(int fy=-RADIUS; fy<=RADIUS; fy++){
                    int row_offset = row_offsets[fy + RADIUS];

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

// 🔥 SPECIALIZZAZIONE EXTREME AVX2 per Kernel 3x3 (RADIUS = 1) - Con Prefetch
template<>
void processChannel<1>(uint8_t* __restrict in, uint8_t* __restrict out, int w, int h, int threshold, int start_y, int end_y, int bucket_size){
    __m256i thresh_vec = _mm256_set1_epi8(threshold);
    __m256i offset128 = _mm256_set1_epi8(128); // trick per usare cmpgt_epi8 unsigned

    // In AVX divoriamo la memoria a bocconi da 32 byte. Effettuiamo un look-ahead molto aggressivo (128-256 byte).
    int prefetch_dist = 256;

    // La versione AVX non divide orizzontalmente in bucket, per massimizzare il troughput lineare (L1 Prefetch) e azzerare i salti!
    for(int y=start_y; y<end_y; y++){
        int r0 = max(y-1, 0) * w;
        int r1 = y * w;
        int r2 = min(y+1, h-1) * w;

        int x = 0;

        // Calcolo scalare nudo e crudo per il primissimo pixel (x=0) che non ha x-1
        {
            uint8_t nh[9] = {
                in[r0], in[r0], in[r0+1],
                in[r1], in[r1], in[r1+1],
                in[r2], in[r2], in[r2+1]
            };
            std::sort(nh, nh+9);
            uint8_t m = nh[4];
            uint8_t c = in[r1];
            out[r1] = (abs(c-m)>threshold)?m:c;
        }

        x = 1;
        // AVX2 CORE: Processiamo massivamente 32 pixel alla volta con la Sorting Network, parallelismo assurdo!
        // Ci fermiamo a w - 32 per evitare di leggere fuori bound
        for(; x <= w - 32; x += 32){
            // *** 3. SOFTWARE PREFETCHING MULTI-LINEA ***
            // Spingiamo esplicitamente nella L1 cache le 3 righe che leggeremo fra "prefetch_dist" pixel
            _mm_prefetch((const char*)&in[r0 + x + prefetch_dist], _MM_HINT_T0);
            _mm_prefetch((const char*)&in[r1 + x + prefetch_dist], _MM_HINT_T0);
            _mm_prefetch((const char*)&in[r2 + x + prefetch_dist], _MM_HINT_T0);

            // Prefetch per preparare anche la scrittura in parallelo
            _mm_prefetch((const char*)&out[r1 + x + prefetch_dist], _MM_HINT_T0);

            __m256i v0 = _mm256_loadu_si256((const __m256i*)&in[r0 + x - 1]);
            __m256i v1 = _mm256_loadu_si256((const __m256i*)&in[r0 + x]);
            __m256i v2 = _mm256_loadu_si256((const __m256i*)&in[r0 + x + 1]);

            __m256i v3 = _mm256_loadu_si256((const __m256i*)&in[r1 + x - 1]);
            __m256i v4 = _mm256_loadu_si256((const __m256i*)&in[r1 + x]);
            __m256i v5 = _mm256_loadu_si256((const __m256i*)&in[r1 + x + 1]);

            __m256i v6 = _mm256_loadu_si256((const __m256i*)&in[r2 + x - 1]);
            __m256i v7 = _mm256_loadu_si256((const __m256i*)&in[r2 + x]);
            __m256i v8 = _mm256_loadu_si256((const __m256i*)&in[r2 + x + 1]);

            // Ordinamento Parziale Matematico (Min/Med/Max) su 32 pixel sovrapposti!
            __m256i r0_min = min3(v0, v1, v2);
            __m256i r0_med = med3(v0, v1, v2);
            __m256i r0_max = max3(v0, v1, v2);

            __m256i r1_min = min3(v3, v4, v5);
            __m256i r1_med = med3(v3, v4, v5);
            __m256i r1_max = max3(v3, v4, v5);

            __m256i r2_min = min3(v6, v7, v8);
            __m256i r2_med = med3(v6, v7, v8);
            __m256i r2_max = max3(v6, v7, v8);

            __m256i max_of_mins = max3(r0_min, r1_min, r2_min);
            __m256i med_of_meds = med3(r0_med, r1_med, r2_med);
            __m256i min_of_maxs = min3(r0_max, r1_max, r2_max);

            __m256i median = med3(max_of_mins, med_of_meds, min_of_maxs);

            // Controllo Threshold Branchless per 32 pixel (Thresholding su Differenze Assolute)
            __m256i diff1 = _mm256_subs_epu8(v4, median);
            __m256i diff2 = _mm256_subs_epu8(median, v4);
            __m256i abs_diff = _mm256_or_si256(diff1, diff2);

            // Shiftiamo i valori di -128 (facendoli diventare interi positivi virtuali) dato che cmpgt su 8bit e' signed
            __m256i biased_diff = _mm256_sub_epi8(abs_diff, offset128);
            __m256i biased_thresh = _mm256_sub_epi8(thresh_vec, offset128);
            __m256i mask = _mm256_cmpgt_epi8(biased_diff, biased_thresh);

            // Blend: se pixel supera la threshold metto la mask (e pesco dalla mediana), altrimenti metto center.
            __m256i res = _mm256_blendv_epi8(v4, median, mask); 

            // Sparo in memoria i 32 valori finali
            _mm256_storeu_si256((__m256i*)&out[r1 + x], res);
        }

        // Coda scalare per gestire le "briciole" finali della larghezza senza andare out-of-bounds (segmentation fault)
        for(; x < w; x++){
            uint8_t nh[9] = {
                in[r0 + x - 1], in[r0 + x], in[r0 + min(x+1, w-1)],
                in[r1 + x - 1], in[r1 + x], in[r1 + min(x+1, w-1)],
                in[r2 + x - 1], in[r2 + x], in[r2 + min(x+1, w-1)]
            };
            std::sort(nh, nh+9);
            uint8_t m = nh[4];
            uint8_t c = in[r1 + x];
            out[r1 + x] = (abs(c-m)>threshold)?m:c;
        }
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


// ---------------- THREAD WORKER ----------------
template<int RADIUS>
void worker(Image& img, Image& out, int threshold, std::atomic<int>& next_row, int end_y, int chunk_size, int bucket_size){
    int c = 0;
    while (true){
        auto start = chrono::high_resolution_clock::now(); 
        int start_y = next_row.fetch_add(chunk_size, std::memory_order_relaxed);
        auto end = chrono::high_resolution_clock::now();
        cout<<"overhead sul atomico di: "<<chrono::duration_cast<chrono::nanoseconds>(end-start).count()<<" n   s\n";
        if (start_y >= end_y){ 
            cout << "thread finito, blocchi consumati: " + to_string(c) + "\n";;
            break;
        }


        c++;
        int current_end_y = std::min(start_y + chunk_size, end_y);

        processChannel<RADIUS>(img.r.data(), out.r.data(), img.width, img.height, threshold, start_y, current_end_y, bucket_size);
        processChannel<RADIUS>(img.g.data(), out.g.data(), img.width, img.height, threshold, start_y, current_end_y, bucket_size);
        processChannel<RADIUS>(img.b.data(), out.b.data(), img.width, img.height, threshold, start_y, current_end_y, bucket_size);
    }
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[])
{
    if(argc < 5){
        cout<<"Uso: "<<argv[0]<<" nome_immagine num_thread dimensione_kernel dimensione_bucket\n";
        return 1;
    }

    // Salvataggio dei parametri in input da terminale
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

    int threshold = 0;

    vector<thread> threads;
    std::atomic<int> next_row(0);

    int chunk_size = 3;
    int radius = kernel / 2;
    auto start = chrono::high_resolution_clock::now();

    if (radius == 1) {
        cout << "[PREFETCH ATTIVO] [OTTIMIZZAZIONE AVX2 SORTING NETWORK (KERNEL = 3)]\n";
        for(int i=0;i<n_threads;i++) threads.emplace_back(worker<1>, ref(img), ref(out), threshold, ref(next_row), img.height, chunk_size, bucket_size);
    } else if (radius == 2) {
        cout << "[PREFETCH ATTIVO] [OTTIMIZZAZIONE TEMPLATE UNROLLING (KERNEL > 3)]\n";
        for(int i=0;i<n_threads;i++) threads.emplace_back(worker<2>, ref(img), ref(out), threshold, ref(next_row), img.height, chunk_size, bucket_size);
    } else if (radius == 3) {
        for(int i=0;i<n_threads;i++) threads.emplace_back(worker<3>, ref(img), ref(out), threshold, ref(next_row), img.height, chunk_size, bucket_size);
    } else if (radius == 4) {
        for(int i=0;i<n_threads;i++) threads.emplace_back(worker<4>, ref(img), ref(out), threshold, ref(next_row), img.height, chunk_size, bucket_size);
    } else if (radius == 5) {
        for(int i=0;i<n_threads;i++) threads.emplace_back(worker<5>, ref(img), ref(out), threshold, ref(next_row), img.height, chunk_size, bucket_size);
    } else {
        cout << "Kernel non supportato per questa ottimizzazione template (usa 3, 5, 7, 9 o 11)." << endl;
        return 1;
    }

    for(auto& t:threads) 
        t.join();

    auto end = chrono::high_resolution_clock::now();

    cout<<"Tempo: " <<chrono::duration_cast<chrono::milliseconds>(end-start).count() <<" ms\n";

    double final_error = calculateError(img_original, out);
    cout<<"Errore medio sui pixel post-filtro (Migliorata): "<<final_error<<" (MAE su scala 0-255)\n";

    cout<<"DEBUG: chunk size "<< chunk_size<<endl;

    saveImage("./output_images/"+img_name+".ppm",out);
}
