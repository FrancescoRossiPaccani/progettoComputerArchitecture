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

using namespace std;

struct Image {
    int width, height;
    vector<uint8_t> r, g, b;

    Image(int h, int w) : width(w), height(h), r(h * w), g(h * w), b(h * w) {}
};

// ---------------- LOAD ----------------
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

// ---------------- SAVE ----------------
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

// ---------------- AVX2 HELPERS ----------------
inline __m256i min3(__m256i a, __m256i b, __m256i c) {
    return _mm256_min_epu8(a, _mm256_min_epu8(b, c));
}

inline __m256i max3(__m256i a, __m256i b, __m256i c) {
    return _mm256_max_epu8(a, _mm256_max_epu8(b, c));
}

inline __m256i med3(__m256i a, __m256i b, __m256i c) {
    return _mm256_max_epu8(_mm256_min_epu8(a, b),
                           _mm256_min_epu8(_mm256_max_epu8(a, b), c));
}

// ---------------- SCALAR BORDER PIXEL ----------------
inline uint8_t median9_scalar(uint8_t a0, uint8_t a1, uint8_t a2,
                              uint8_t b0, uint8_t b1, uint8_t b2,
                              uint8_t c0, uint8_t c1, uint8_t c2) {
    uint8_t v[9] = {a0,a1,a2,b0,b1,b2,c0,c1,c2};
    sort(v, v + 9);
    return v[4];
}

inline void processBorderPixel(const uint8_t* in, uint8_t* out, int w, int h, int x, int y, int threshold) {
    int xm1 = max(x - 1, 0);
    int xp1 = min(x + 1, w - 1);
    int ym1 = max(y - 1, 0);
    int yp1 = min(y + 1, h - 1);

    uint8_t m = median9_scalar(
        in[ym1 * w + xm1], in[ym1 * w + x],   in[ym1 * w + xp1],
        in[y   * w + xm1], in[y   * w + x],   in[y   * w + xp1],
        in[yp1 * w + xm1], in[yp1 * w + x],   in[yp1 * w + xp1]
    );

    uint8_t c = in[y * w + x];
    out[y * w + x] = (abs((int)c - (int)m) > threshold) ? m : c;
}

// ---------------- AVX2 3x3 CHANNEL ----------------
void processChannel3x3_AVX2(const uint8_t* __restrict in,
                            uint8_t* __restrict out,
                            int w, int h,
                            int threshold,
                            int start_y, int end_y) {
    __m256i thresh_vec = _mm256_set1_epi8((char)threshold);
    __m256i offset128  = _mm256_set1_epi8((char)128);

    if (w <= 0 || h <= 0) return;

    if (w < 3 || h < 3) {
        for (int y = start_y; y < end_y; y++) {
            for (int x = 0; x < w; x++) {
                processBorderPixel(in, out, w, h, x, y, threshold);
            }
        }
        return;
    }

    for (int y = start_y; y < end_y; y++) {
        // righe bordo: tutto scalare
        if (y == 0 || y == h - 1) {
            for (int x = 0; x < w; x++) {
                processBorderPixel(in, out, w, h, x, y, threshold);
            }
            continue;
        }

        const int r0 = (y - 1) * w;
        const int r1 = y * w;
        const int r2 = (y + 1) * w;

        // bordo sinistro
        processBorderPixel(in, out, w, h, 0, y, threshold);

        int x = 1;

        // interno vettoriale: x in [1, w-2]
        // servono letture:
        // x-1 .. x+30
        // x   .. x+31
        // x+1 .. x+32
        // quindi x <= w-33
        for (; x <= w - 33; x += 32) {
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

        // coda interna scalare fino a w-2
        for (; x < w - 1; x++) {
            uint8_t m = median9_scalar(
                in[r0 + x - 1], in[r0 + x], in[r0 + x + 1],
                in[r1 + x - 1], in[r1 + x], in[r1 + x + 1],
                in[r2 + x - 1], in[r2 + x], in[r2 + x + 1]
            );

            uint8_t c = in[r1 + x];
            out[r1 + x] = (abs((int)c - (int)m) > threshold) ? m : c;
        }

        // bordo destro
        processBorderPixel(in, out, w, h, w - 1, y, threshold);
    }
}

// ---------------- THREAD WORKER ----------------
void worker3x3(Image& img, Image& out, int threshold, int start_y, int end_y) {
    processChannel3x3_AVX2(img.r.data(), out.r.data(), img.width, img.height, threshold, start_y, end_y);
    processChannel3x3_AVX2(img.g.data(), out.g.data(), img.width, img.height, threshold, start_y, end_y);
    processChannel3x3_AVX2(img.b.data(), out.b.data(), img.width, img.height, threshold, start_y, end_y);
}

// ---------------- ERROR MEASUREMENT ----------------
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

// ---------------- MAIN ----------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Uso: " << argv[0] << " nome_immagine num_thread\n";
        return 1;
    }

    string img_name = argv[1];
    int n_threads = stoi(argv[2]);

    Image img = loadImage("../error_images/" + img_name + ".ppm");
    Image out(img.height, img.width);

    Image img_original = loadImage("../input_images/" + img_name + ".ppm");

    double initial_error = calculateError(img_original, img);
    cout << "Immagine " << img.width << "x" << img.height << "\n";
    cout << "Errore medio sui pixel pre-filtro: " << initial_error << "\n";

    const int threshold = 1;

    vector<thread> threads;
    int rows_per_thread = (img.height + n_threads - 1) / n_threads;

    auto start = chrono::high_resolution_clock::now();

    for (int t = 0; t < n_threads; t++) {
        int start_y = t * rows_per_thread;
        int end_y = min(start_y + rows_per_thread, img.height);
        if (start_y >= img.height) break;

        threads.emplace_back(worker3x3, ref(img), ref(out), threshold, start_y, end_y);
    }

    for (auto& th : threads) th.join();

    auto end = chrono::high_resolution_clock::now();

    cout << "Thread: " << n_threads << " || ";
    cout << "Tempo: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    double final_error = calculateError(img_original, out);
    cout << "Errore: " << final_error << "\n";

    saveImage("../output_images/" + img_name + ".ppm", out);
    return 0;
}