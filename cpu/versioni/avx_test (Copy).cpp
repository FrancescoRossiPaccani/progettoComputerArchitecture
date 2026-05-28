#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>
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

#ifndef ENABLE_PREFETCH
#define ENABLE_PREFETCH 1
#endif

#ifndef ENABLE_PINNING
#define ENABLE_PINNING 0
#endif

#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 8192
#endif

#if defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#define HOT_FN __attribute__((hot))
#define RESTRICT __restrict__
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define FORCE_INLINE inline
#define HOT_FN
#define RESTRICT
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
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
// AVX2 HELPERS
// ============================================================

FORCE_INLINE __m256i min3(__m256i a, __m256i b, __m256i c) {
    return _mm256_min_epu8(a, _mm256_min_epu8(b, c));
}

FORCE_INLINE __m256i max3(__m256i a, __m256i b, __m256i c) {
    return _mm256_max_epu8(a, _mm256_max_epu8(b, c));
}

FORCE_INLINE __m256i med3(__m256i a, __m256i b, __m256i c) {
    return _mm256_max_epu8(_mm256_min_epu8(a, b),
                           _mm256_min_epu8(_mm256_max_epu8(a, b), c));
}

// ============================================================
// SCALAR MEDIAN 9
// ============================================================

FORCE_INLINE void sort2(uint8_t& a, uint8_t& b) {
    uint8_t mn = (a < b) ? a : b;
    uint8_t mx = (a < b) ? b : a;
    a = mn;
    b = mx;
}

FORCE_INLINE uint8_t median9_scalar(uint8_t a0, uint8_t a1, uint8_t a2,
                                    uint8_t b0, uint8_t b1, uint8_t b2,
                                    uint8_t c0, uint8_t c1, uint8_t c2) {
    sort2(a1, a2); sort2(a0, a1); sort2(a1, a2);
    sort2(b1, b2); sort2(b0, b1); sort2(b1, b2);
    sort2(c1, c2); sort2(c0, c1); sort2(c1, c2);

    sort2(a0, b0); sort2(a1, b1); sort2(a2, b2);
    sort2(b0, c0); sort2(b1, c1); sort2(b2, c2);
    sort2(a0, b0); sort2(a1, b1); sort2(a2, b2);

    sort2(a2, b1); sort2(b1, c0);
    sort2(a1, b0); sort2(b2, c1);
    sort2(b0, b1); sort2(c0, c1);

    return b1;
}

FORCE_INLINE uint8_t applyThreshold(uint8_t center, uint8_t med, int threshold) {
    return (abs((int)center - (int)med) > threshold) ? med : center;
}

// ============================================================
// BORDER SCALAR
// ============================================================

FORCE_INLINE void processBorderPixel(const uint8_t* RESTRICT in,
                                     uint8_t* RESTRICT out,
                                     int w, int h, int x, int y, int threshold) {
    const int xm1 = (x > 0) ? x - 1 : 0;
    const int xp1 = (x + 1 < w) ? x + 1 : (w - 1);
    const int ym1 = (y > 0) ? y - 1 : 0;
    const int yp1 = (y + 1 < h) ? y + 1 : (h - 1);

    const int r0 = ym1 * w;
    const int r1 = y   * w;
    const int r2 = yp1 * w;

    const uint8_t med = median9_scalar(
        in[r0 + xm1], in[r0 + x],   in[r0 + xp1],
        in[r1 + xm1], in[r1 + x],   in[r1 + xp1],
        in[r2 + xm1], in[r2 + x],   in[r2 + xp1]
    );

    out[r1 + x] = applyThreshold(in[r1 + x], med, threshold);
}

FORCE_INLINE uint8_t processInnerScalar(const uint8_t* RESTRICT in,
                                        int r0, int r1, int r2, int x,
                                        int threshold) {
    const uint8_t med = median9_scalar(
        in[r0 + x - 1], in[r0 + x], in[r0 + x + 1],
        in[r1 + x - 1], in[r1 + x], in[r1 + x + 1],
        in[r2 + x - 1], in[r2 + x], in[r2 + x + 1]
    );

    return applyThreshold(in[r1 + x], med, threshold);
}

// ============================================================
// AVX2 INTERIOR
// ============================================================

HOT_FN
FORCE_INLINE void processInteriorAVX2(const uint8_t* RESTRICT in,
                                      uint8_t* RESTRICT out,
                                      int w,
                                      int start_y,
                                      int end_y,
                                      int threshold) {
    const __m256i thresh_vec = _mm256_set1_epi8((char)threshold);
    const __m256i offset128  = _mm256_set1_epi8((char)128);

    for (int y = start_y; y < end_y; ++y) {
        const int r0 = (y - 1) * w;
        const int r1 = y * w;
        const int r2 = (y + 1) * w;

        int x = 1;

        for (; x <= w - 33; x += 32) {
#if ENABLE_PREFETCH
            __builtin_prefetch(in + r0 + x + PREFETCH_DISTANCE, 0, 1);
            __builtin_prefetch(in + r1 + x + PREFETCH_DISTANCE, 0, 1);
            __builtin_prefetch(in + r2 + x + PREFETCH_DISTANCE, 0, 1);
#endif

            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(in + r0 + x - 1));
            const __m256i v1 = _mm256_loadu_si256((const __m256i*)(in + r0 + x));
            const __m256i v2 = _mm256_loadu_si256((const __m256i*)(in + r0 + x + 1));

            const __m256i v3 = _mm256_loadu_si256((const __m256i*)(in + r1 + x - 1));
            const __m256i v4 = _mm256_loadu_si256((const __m256i*)(in + r1 + x));
            const __m256i v5 = _mm256_loadu_si256((const __m256i*)(in + r1 + x + 1));

            const __m256i v6 = _mm256_loadu_si256((const __m256i*)(in + r2 + x - 1));
            const __m256i v7 = _mm256_loadu_si256((const __m256i*)(in + r2 + x));
            const __m256i v8 = _mm256_loadu_si256((const __m256i*)(in + r2 + x + 1));

            const __m256i r0_min = min3(v0, v1, v2);
            const __m256i r0_med = med3(v0, v1, v2);
            const __m256i r0_max = max3(v0, v1, v2);

            const __m256i r1_min = min3(v3, v4, v5);
            const __m256i r1_med = med3(v3, v4, v5);
            const __m256i r1_max = max3(v3, v4, v5);

            const __m256i r2_min = min3(v6, v7, v8);
            const __m256i r2_med = med3(v6, v7, v8);
            const __m256i r2_max = max3(v6, v7, v8);

            const __m256i max_of_mins = max3(r0_min, r1_min, r2_min);
            const __m256i med_of_meds = med3(r0_med, r1_med, r2_med);
            const __m256i min_of_maxs = min3(r0_max, r1_max, r2_max);

            const __m256i median = med3(max_of_mins, med_of_meds, min_of_maxs);

            const __m256i diff1 = _mm256_subs_epu8(v4, median);
            const __m256i diff2 = _mm256_subs_epu8(median, v4);
            const __m256i abs_diff = _mm256_or_si256(diff1, diff2);

            const __m256i biased_diff   = _mm256_sub_epi8(abs_diff, offset128);
            const __m256i biased_thresh = _mm256_sub_epi8(thresh_vec, offset128);
            const __m256i mask = _mm256_cmpgt_epi8(biased_diff, biased_thresh);

            const __m256i res = _mm256_blendv_epi8(v4, median, mask);
            _mm256_storeu_si256((__m256i*)(out + r1 + x), res);
        }

        for (; x < w - 1; ++x) {
            out[r1 + x] = processInnerScalar(in, r0, r1, r2, x, threshold);
        }
    }
}

// ============================================================
// CHANNEL DRIVER
// ============================================================

HOT_FN
void processChannel3x3(const uint8_t* RESTRICT in,
                       uint8_t* RESTRICT out,
                       int w, int h,
                       int threshold,
                       int start_y, int end_y) {
    if (UNLIKELY(w <= 0 || h <= 0 || start_y >= end_y)) return;

    if (UNLIKELY(w < 3 || h < 3)) {
        for (int y = start_y; y < end_y; ++y)
            for (int x = 0; x < w; ++x)
                processBorderPixel(in, out, w, h, x, y, threshold);
        return;
    }

    int inner_start = start_y;
    int inner_end   = end_y;

    if (start_y == 0) {
        for (int x = 0; x < w; ++x)
            processBorderPixel(in, out, w, h, x, 0, threshold);
        inner_start = 1;
    }

    if (end_y == h) {
        --inner_end;
    }

    for (int y = inner_start; y < inner_end; ++y) {
        processBorderPixel(in, out, w, h, 0, y, threshold);
        processBorderPixel(in, out, w, h, w - 1, y, threshold);
    }

    if (inner_start < inner_end) {
        processInteriorAVX2(in, out, w, inner_start, inner_end, threshold);
    }

    if (end_y == h) {
        for (int x = 0; x < w; ++x)
            processBorderPixel(in, out, w, h, x, h - 1, threshold);
    }
}

// ============================================================
// THREAD PINNING
// ============================================================

#ifdef __linux__
bool pinThreadToCpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
}
#else
bool pinThreadToCpu(int) { return false; }
#endif

vector<int> buildCpuMap(int n_threads) {
    vector<int> cpus;
    unsigned int hw = thread::hardware_concurrency();
    if (hw == 0) hw = n_threads;

    for (unsigned int i = 0; i < hw && (int)cpus.size() < n_threads; ++i) {
        cpus.push_back((int)i);
    }
    return cpus;
}

// ============================================================
// THREAD WORKER
// ============================================================

void worker3x3Pinned(Image& img,
                     Image& out,
                     int threshold,
                     int start_y,
                     int end_y,
                     int cpu_id) {
#if ENABLE_PINNING
    if (cpu_id >= 0) pinThreadToCpu(cpu_id);
#else
    (void)cpu_id;
#endif

    processChannel3x3(img.r.data(), out.r.data(), img.width, img.height, threshold, start_y, end_y);
    processChannel3x3(img.g.data(), out.g.data(), img.width, img.height, threshold, start_y, end_y);
    processChannel3x3(img.b.data(), out.b.data(), img.width, img.height, threshold, start_y, end_y);
}

// ============================================================
// ERROR
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
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Uso: " << argv[0]
             << " nome_immagine num_thread [iters] [warmup] [save_output]\n";
        return 1;
    }

    string img_name = argv[1];
    int n_threads   = stoi(argv[2]);
    int iters       = (argc > 3) ? stoi(argv[3]) : 30;
    int warmup      = (argc > 4) ? stoi(argv[4]) : 5;
    int save_output = (argc > 5) ? stoi(argv[5]) : 0;

    const int threshold = 1;

    Image img = loadImage("../error_images/" + img_name + ".ppm");
    Image img_original = loadImage("../input_images/" + img_name + ".ppm");
    Image out(img.height, img.width);

    vector<int> cpu_map = buildCpuMap(n_threads);

    double initial_error = calculateError(img_original, img);

    //cout << "Immagine " << img.width << "x" << img.height << "\n";
    //cout << "Errore medio sui pixel pre-filtro: " << initial_error << "\n";
    //cout << "[CORE 3x3 AVX2 + MT + PINNING]"
    //     << " [threads=" << n_threads << "]"
    //     << " [iters=" << iters << "]"
    //     << " [warmup=" << warmup << "]";
//#if ENABLE_PINNING
//    cout << " [pinning=on]\n";
//#else
//    cout << " [pinning=off]\n";
//#endif

    auto run_once = [&]() {
        vector<thread> threads;
        int rows_per_thread = (img.height + n_threads - 1) / n_threads;

        for (int t = 0; t < n_threads; ++t) {
            int start_y = t * rows_per_thread;
            int end_y   = min(start_y + rows_per_thread, img.height);
            if (start_y >= img.height) break;

            int cpu_id = (t < (int)cpu_map.size()) ? cpu_map[t] : -1;

            threads.emplace_back(worker3x3Pinned,
                                 ref(img), ref(out),
                                 threshold,
                                 start_y, end_y,
                                 cpu_id);
        }

        for (auto& th : threads) th.join();
    };

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    vector<double> times_ms;
    times_ms.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        auto start = chrono::high_resolution_clock::now();
        run_once();
        auto end = chrono::high_resolution_clock::now();

        double ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
        times_ms.push_back(ms);
    }

    double sum = 0.0;
    double best = times_ms[0];
    double worst = times_ms[0];

    for (double t : times_ms) {
        sum += t;
        if (t < best) best = t;
        if (t > worst) worst = t;
    }

    double avg = sum / times_ms.size();

    vector<double> sorted = times_ms;
    sort(sorted.begin(), sorted.end());

    double median;
    if (sorted.size() % 2 == 0) {
        median = (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) * 0.5;
    } else {
        median = sorted[sorted.size() / 2];
    }

    double final_error = calculateError(img_original, out);
    cout << "Threads: " << n_threads<<endl;
    cout << "Tempo kernel BEST   : " << best   << " ms\n";
    cout << "Tempo kernel AVG    : " << avg    << " ms\n";
    cout << "Tempo kernel MEDIAN : " << median << " ms\n";
    cout << "Tempo kernel WORST  : " << worst  << " ms\n";
    cout << "Errore: " << final_error << "\n";

    if (save_output) {
        saveImage("../output_images/" + img_name + ".ppm", out);
    }

    return 0;
}