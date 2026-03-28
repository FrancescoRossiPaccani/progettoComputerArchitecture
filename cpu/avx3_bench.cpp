#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

using namespace std;

// ============================================================
// CONFIG
// ============================================================

// 1 = abilita software prefetch, 0 = disabilita
#ifndef ENABLE_PREFETCH
#define ENABLE_PREFETCH 0
#endif

// 1 = prova a pinnare i thread su CPU specifiche (Linux)
#ifndef ENABLE_PINNING
#define ENABLE_PINNING 1
#endif

// 1 = usa una piccola guard band tra i blocchi di righe assegnati
#ifndef USE_ROW_GUARD
#define USE_ROW_GUARD 1
#endif

// distanza prefetch per il path AVX2 3x3
#ifndef PREFETCH_DISTANCE_AVX
#define PREFETCH_DISTANCE_AVX 256
#endif

// distanza prefetch per il path generico
#ifndef PREFETCH_DISTANCE_GENERIC
#define PREFETCH_DISTANCE_GENERIC 64
#endif

// ============================================================
// IMAGE
// ============================================================

struct Image {
    int width, height;
    vector<uint8_t> r, g, b;

    Image(int h, int w) : width(w), height(h), r(h * w), g(h * w), b(h * w) {}
};

// ============================================================
// LOAD / SAVE
// ============================================================

Image loadImage(const string& path) {
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Errore apertura file\n";
        exit(1);
    }

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
    } else {
        for (int i = 0; i < h * w; i++) {
            int R, G, B;
            file >> R >> G >> B;
            img.r[i] = static_cast<uint8_t>(R);
            img.g[i] = static_cast<uint8_t>(G);
            img.b[i] = static_cast<uint8_t>(B);
        }
    }

    return img;
}

void saveImage(const string& path, Image& img) {
    ofstream file(path, ios::binary);
    if (!file) {
        cerr << "Errore salvataggio file\n";
        exit(1);
    }

    file << "P6\n" << img.width << " " << img.height << "\n255\n";

    vector<uint8_t> buffer(img.width * img.height * 3);
    for (int i = 0; i < img.width * img.height; i++) {
        buffer[i * 3]     = img.r[i];
        buffer[i * 3 + 1] = img.g[i];
        buffer[i * 3 + 2] = img.b[i];
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// ============================================================
// CORE HELPERS
// ============================================================

inline void updateMedian(int& m, int& count, uint16_t* hist, int mid) {
    while (m > 0 && count - hist[m] > mid) {
        count -= hist[m];
        m--;
    }
    while (m < 255 && count <= mid) {
        m++;
        count += hist[m];
    }
}

// AVX2 helpers 3x3
inline __m256i min3(__m256i a, __m256i b, __m256i c) {
    return _mm256_min_epu8(a, _mm256_min_epu8(b, c));
}
inline __m256i max3(__m256i a, __m256i b, __m256i c) {
    return _mm256_max_epu8(a, _mm256_max_epu8(b, c));
}
inline __m256i med3(__m256i a, __m256i b, __m256i c) {
    return _mm256_max_epu8(_mm256_min_epu8(a, b), _mm256_min_epu8(_mm256_max_epu8(a, b), c));
}

static inline int clampi(int x, int lo, int hi) {
    return (x < lo) ? lo : (x > hi ? hi : x);
}

// ============================================================
// GENERIC MEDIAN FILTER
// ============================================================

template<int RADIUS>
void processChannel(uint8_t* __restrict in,
                    uint8_t* __restrict out,
                    int w, int h,
                    int threshold,
                    int start_y, int end_y,
                    int bucket_size) {
    const int kernel = 2 * RADIUS + 1;
    const int mid = (kernel * kernel) / 2;

    vector<int> row_offsets(2 * RADIUS + 1);

    for (int bx = 0; bx < w; bx += bucket_size) {
        int end_x = min(bx + bucket_size, w);

        for (int y = start_y; y < end_y; y++) {
            alignas(64) uint16_t hist[256] = {0};

            #pragma GCC unroll 15
            for (int fy = -RADIUS; fy <= RADIUS; fy++) {
                row_offsets[fy + RADIUS] = clampi(y + fy, 0, h - 1) * w;
            }

            #pragma GCC unroll 15
            for (int fy = -RADIUS; fy <= RADIUS; fy++) {
                int row_offset = row_offsets[fy + RADIUS];
                for (int fx = bx - RADIUS; fx <= bx + RADIUS; fx++) {
                    int ix = clampi(fx, 0, w - 1);
                    hist[in[row_offset + ix]]++;
                }
            }

            int m = 0, count = 0;
            while (count <= mid) {
                count += hist[m];
                m++;
            }
            m--;

            for (int x = bx; x < end_x; x++) {
#if ENABLE_PREFETCH
                #pragma GCC unroll 15
                for (int fy = -RADIUS; fy <= RADIUS; fy++) {
                    int pf = min(x + PREFETCH_DISTANCE_GENERIC, w - 1);
                    _mm_prefetch((const char*)&in[row_offsets[fy + RADIUS] + pf], _MM_HINT_T0);
                }
#endif

                int idx = y * w + x;
                int center = in[idx];
                out[idx] = (abs(center - m) > threshold) ? static_cast<uint8_t>(m) : static_cast<uint8_t>(center);

                if (x == end_x - 1) break;

                int left  = max(x - RADIUS, 0);
                int right = min(x + RADIUS + 1, w - 1);

                #pragma GCC unroll 15
                for (int fy = -RADIUS; fy <= RADIUS; fy++) {
                    int row_offset = row_offsets[fy + RADIUS];
                    int oldv = in[row_offset + left];
                    int newv = in[row_offset + right];

                    hist[oldv]--;
                    count -= (oldv <= m);

                    hist[newv]++;
                    count += (newv <= m);
                }

                updateMedian(m, count, hist, mid);
            }
        }
    }
}

// ============================================================
// SPECIALIZED AVX2 3x3
// ============================================================

template<>
void processChannel<1>(uint8_t* __restrict in,
                       uint8_t* __restrict out,
                       int w, int h,
                       int threshold,
                       int start_y, int end_y,
                       int bucket_size) {
    (void)bucket_size;

    __m256i thresh_vec = _mm256_set1_epi8((char)threshold);
    __m256i offset128  = _mm256_set1_epi8((char)128);

    for (int y = start_y; y < end_y; y++) {
        int r0 = max(y - 1, 0) * w;
        int r1 = y * w;
        int r2 = min(y + 1, h - 1) * w;

        if (w < 3) {
            for (int x = 0; x < w; x++) {
                int xm1 = max(x - 1, 0);
                int xp1 = min(x + 1, w - 1);

                uint8_t nh[9] = {
                    in[r0 + xm1], in[r0 + x], in[r0 + xp1],
                    in[r1 + xm1], in[r1 + x], in[r1 + xp1],
                    in[r2 + xm1], in[r2 + x], in[r2 + xp1]
                };
                sort(nh, nh + 9);
                uint8_t m = nh[4];
                uint8_t c = in[r1 + x];
                out[r1 + x] = (abs((int)c - (int)m) > threshold) ? m : c;
            }
            continue;
        }

        {
            uint8_t nh[9] = {
                in[r0 + 0], in[r0 + 0], in[r0 + 1],
                in[r1 + 0], in[r1 + 0], in[r1 + 1],
                in[r2 + 0], in[r2 + 0], in[r2 + 1]
            };
            sort(nh, nh + 9);
            uint8_t m = nh[4];
            uint8_t c = in[r1 + 0];
            out[r1 + 0] = (abs((int)c - (int)m) > threshold) ? m : c;
        }

        int x = 1;

        // Fix bordo destro:
        // v2 legge da x+1 a x+32, quindi serve x <= w-33
        for (; x <= w - 33; x += 32) {
#if ENABLE_PREFETCH
            int pf = min(x + PREFETCH_DISTANCE_AVX, w - 1);
            _mm_prefetch((const char*)&in[r0 + pf], _MM_HINT_T0);
            _mm_prefetch((const char*)&in[r1 + pf], _MM_HINT_T0);
            _mm_prefetch((const char*)&in[r2 + pf], _MM_HINT_T0);
#endif

            __m256i v0 = _mm256_loadu_si256((const __m256i*)&in[r0 + x - 1]);
            __m256i v1 = _mm256_loadu_si256((const __m256i*)&in[r0 + x]);
            __m256i v2 = _mm256_loadu_si256((const __m256i*)&in[r0 + x + 1]);

            __m256i v3 = _mm256_loadu_si256((const __m256i*)&in[r1 + x - 1]);
            __m256i v4 = _mm256_loadu_si256((const __m256i*)&in[r1 + x]);
            __m256i v5 = _mm256_loadu_si256((const __m256i*)&in[r1 + x + 1]);

            __m256i v6 = _mm256_loadu_si256((const __m256i*)&in[r2 + x - 1]);
            __m256i v7 = _mm256_loadu_si256((const __m256i*)&in[r2 + x]);
            __m256i v8 = _mm256_loadu_si256((const __m256i*)&in[r2 + x + 1]);

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

            __m256i diff1 = _mm256_subs_epu8(v4, median);
            __m256i diff2 = _mm256_subs_epu8(median, v4);
            __m256i abs_diff = _mm256_or_si256(diff1, diff2);

            __m256i biased_diff   = _mm256_sub_epi8(abs_diff, offset128);
            __m256i biased_thresh = _mm256_sub_epi8(thresh_vec, offset128);
            __m256i mask = _mm256_cmpgt_epi8(biased_diff, biased_thresh);

            __m256i res = _mm256_blendv_epi8(v4, median, mask);
            _mm256_storeu_si256((__m256i*)&out[r1 + x], res);
        }

        for (; x < w; x++) {
            int xm1 = max(x - 1, 0);
            int xp1 = min(x + 1, w - 1);

            uint8_t nh[9] = {
                in[r0 + xm1], in[r0 + x], in[r0 + xp1],
                in[r1 + xm1], in[r1 + x], in[r1 + xp1],
                in[r2 + xm1], in[r2 + x], in[r2 + xp1]
            };
            sort(nh, nh + 9);
            uint8_t m = nh[4];
            uint8_t c = in[r1 + x];
            out[r1 + x] = (abs((int)c - (int)m) > threshold) ? m : c;
        }
    }
}

// ============================================================
// ERROR MEASUREMENT
// ============================================================

long long vectorizedSAD(const uint8_t* a, const uint8_t* b, int size) {
    long long total_diff = 0;
    int i = 0;

    __m256i sum_vec = _mm256_setzero_si256();

    for (; i <= size - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i sad = _mm256_sad_epu8(va, vb);
        sum_vec = _mm256_add_epi64(sum_vec, sad);
    }

    uint64_t sums[4];
    _mm256_storeu_si256((__m256i*)sums, sum_vec);
    total_diff += (long long)sums[0] + (long long)sums[1] + (long long)sums[2] + (long long)sums[3];

    for (; i < size; i++) {
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

// ============================================================
// THREAD PINNING
// ============================================================

#ifdef __linux__
bool pinThreadToCpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_t current = pthread_self();
    return pthread_setaffinity_np(current, sizeof(cpu_set_t), &cpuset) == 0;
}
#else
bool pinThreadToCpu(int) { return false; }
#endif

// ============================================================
// STATIC WORKER
// ============================================================

template<int RADIUS>
void workerStatic(Image& img,
                  Image& out,
                  int threshold,
                  int start_y,
                  int end_y,
                  int bucket_size,
                  int pinned_cpu) {
#if ENABLE_PINNING
    if (pinned_cpu >= 0) pinThreadToCpu(pinned_cpu);
#endif

    processChannel<RADIUS>(img.r.data(), out.r.data(), img.width, img.height,
                           threshold, start_y, end_y, bucket_size);
    processChannel<RADIUS>(img.g.data(), out.g.data(), img.width, img.height,
                           threshold, start_y, end_y, bucket_size);
    processChannel<RADIUS>(img.b.data(), out.b.data(), img.width, img.height,
                           threshold, start_y, end_y, bucket_size);
}

// ============================================================
// CPU LIST HELPERS
// ============================================================

vector<int> buildCpuList(int n_threads) {
    vector<int> cpus;

#ifdef __linux__
    unsigned int hw = thread::hardware_concurrency();
    if (hw == 0) hw = 20;

    // Strategia semplice:
    // prima usa CPU 0..min(n_threads, hw)-1
    // l’utente può poi confrontare con taskset se vuole separare P/E core
    for (unsigned int i = 0; i < hw && (int)cpus.size() < n_threads; i++) {
        cpus.push_back((int)i);
    }
#else
    (void)n_threads;
#endif

    return cpus;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cout << "Uso: " << argv[0] << " nome_immagine num_thread dimensione_kernel dimensione_bucket\n";
        return 1;
    }

    string img_name = argv[1];
    int n_threads = stoi(argv[2]);
    int kernel = stoi(argv[3]);
    int bucket_size = stoi(argv[4]);

    Image img = loadImage("./error_images/" + img_name + ".ppm");
    Image out(img.height, img.width);
    Image img_original = loadImage("./input_images/" + img_name + ".ppm");

    double initial_error = calculateError(img_original, img);
    cout << "Immagine " << img.width << "x" << img.height << "\n";
    cout << "Errore medio sui pixel pre-filtro (Rumore): " << initial_error << " (MAE su scala 0-255)\n";

    int threshold = 0;
    int radius = kernel / 2;
    vector<string> image_names = {"img1", "img2", "img3"};

    for(int k=0; k<=2; k++){
        img_name = image_names[k];
        Image img = loadImage("./error_images/"+img_name+".ppm");
        for(int t = 20; t>=1; t--){
            int n_threads = t;
            for(int j = 0; j<10; j++){
                vector<thread> threads;
                vector<int> cpu_list = buildCpuList(n_threads);

                auto start = chrono::high_resolution_clock::now();

                // static partition
                int rows_per_thread = (img.height + n_threads - 1) / n_threads;

//                //cout << "[STATIC SCHEDULING]";
//            #if ENABLE_PINNING
//                //cout << " [PINNING ATTIVO]";
//            #else
//                //cout << " [PINNING DISATTIVO]";
//            #endif
//
//            #if ENABLE_PREFETCH
//                cout << " [PREFETCH ATTIVO]";
//            #else
//                cout << " [PREFETCH DISATTIVO]";
//            #endif
//
//                if (radius == 1) cout << " [AVX2 SORTING NETWORK 3x3]\n";
//                else cout << " [GENERIC HISTOGRAM]\n";

                for (int i = 0; i < n_threads; i++) {
                    int start_y = i * rows_per_thread;
                    int end_y   = min(start_y + rows_per_thread, img.height);
                
                    if (start_y >= img.height) break;
                
            #if USE_ROW_GUARD
                    // La guard band non cambia i pixel scritti:
                    // è solo un modo per evitare bande microscopiche troppo vicine quando il partizionamento è stretto.
                    // Qui manteniamo l'intervallo logico semplice, senza overlap in scrittura.
            #endif
                
                    int pinned_cpu = (i < (int)cpu_list.size()) ? cpu_list[i] : -1;
                
                    if (radius == 1) {
                        threads.emplace_back(workerStatic<1>, ref(img), ref(out), threshold, start_y, end_y, bucket_size, pinned_cpu);
                    } else if (radius == 2) {
                        threads.emplace_back(workerStatic<2>, ref(img), ref(out), threshold, start_y, end_y, bucket_size, pinned_cpu);
                    } else if (radius == 3) {
                        threads.emplace_back(workerStatic<3>, ref(img), ref(out), threshold, start_y, end_y, bucket_size, pinned_cpu);
                    } else if (radius == 4) {
                        threads.emplace_back(workerStatic<4>, ref(img), ref(out), threshold, start_y, end_y, bucket_size, pinned_cpu);
                    } else if (radius == 5) {
                        threads.emplace_back(workerStatic<5>, ref(img), ref(out), threshold, start_y, end_y, bucket_size, pinned_cpu);
                    } else {
                        cout << "Kernel non supportato per questa ottimizzazione template (usa 3, 5, 7, 9 o 11)." << endl;
                        return 1;
                    }
                }
            
                for (auto& t : threads) t.join();
            
                auto end = chrono::high_resolution_clock::now();
                cout<<"Thread: " << n_threads<<" \t|| Run: "<< j <<" \t|| " << "Tempo: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl; 
                //cout << "Tempo: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";
            }
        }
    }

//    double final_error = calculateError(img_original, out);
//    cout << "Errore medio sui pixel post-filtro (Migliorata): " << final_error << " (MAE su scala 0-255)\n";
//
//    cout << "DEBUG: rows_per_thread " << rows_per_thread << "\n";
//
//    saveImage("./output_images/" + img_name + ".ppm", out);
    return 0;
}