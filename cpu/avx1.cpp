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
    string format;
    file>>format;

    int w,h,maxval;
    file>>w>>h>>maxval;

    Image img(h,w);

    while(isspace(file.peek())) file.get();

    vector<uint8_t> buffer(w*h*3);
    file.read((char*)buffer.data(), buffer.size());

    for(int i=0;i<h*w;i++){
        img.r[i]=buffer[i*3];
        img.g[i]=buffer[i*3+1];
        img.b[i]=buffer[i*3+2];
    }

    return img;
}

// ---------------- SAVE ----------------
void saveImage(const string& path, Image& img){
    ofstream file(path, ios::binary);
    file<<"P6\n"<<img.width<<" "<<img.height<<"\n255\n";

    vector<uint8_t> buffer(img.width*img.height*3);
    for(int i=0;i<img.width*img.height;i++){
        buffer[i*3]=img.r[i];
        buffer[i*3+1]=img.g[i];
        buffer[i*3+2]=img.b[i];
    }
    file.write((char*)buffer.data(), buffer.size());
}
// dichiarazione template GENERICO (necessaria!)
template<int R>
void processChannel(uint8_t* in, uint8_t* out, int w, int h,
                    int threshold, int start_y, int end_y, int bucket_size);


// ---------------- AVX2 MEDIAN 3x3 ----------------
template<>
void processChannel<1>(uint8_t* in, uint8_t* out, int w, int h,
                       int threshold, int start_y, int end_y, int){

    __m256i thresh_vec = _mm256_set1_epi8(threshold);
    __m256i offset128  = _mm256_set1_epi8(128);

    for(int y=start_y; y<end_y; y++){

        int r0 = max(y-1,0)*w;
        int r1 = y*w;
        int r2 = min(y+1,h-1)*w;

        int x = 1;

        __m256i v0_0 = _mm256_loadu_si256((__m256i*)&in[r0 + x - 1]);
        __m256i v0_1 = _mm256_loadu_si256((__m256i*)&in[r0 + x]);
        __m256i v0_2 = _mm256_loadu_si256((__m256i*)&in[r0 + x + 1]);

        __m256i v1_0 = _mm256_loadu_si256((__m256i*)&in[r1 + x - 1]);
        __m256i v1_1 = _mm256_loadu_si256((__m256i*)&in[r1 + x]);
        __m256i v1_2 = _mm256_loadu_si256((__m256i*)&in[r1 + x + 1]);

        __m256i v2_0 = _mm256_loadu_si256((__m256i*)&in[r2 + x - 1]);
        __m256i v2_1 = _mm256_loadu_si256((__m256i*)&in[r2 + x]);
        __m256i v2_2 = _mm256_loadu_si256((__m256i*)&in[r2 + x + 1]);

        for(; x <= w-32; x+=32){

            auto min3 = [](__m256i a,__m256i b,__m256i c){
                return _mm256_min_epu8(a,_mm256_min_epu8(b,c));
            };
            auto max3 = [](__m256i a,__m256i b,__m256i c){
                return _mm256_max_epu8(a,_mm256_max_epu8(b,c));
            };
            auto med3 = [](__m256i a,__m256i b,__m256i c){
                return _mm256_max_epu8(_mm256_min_epu8(a,b),
                       _mm256_min_epu8(_mm256_max_epu8(a,b),c));
            };

            __m256i r0_min = min3(v0_0,v0_1,v0_2);
            __m256i r0_med = med3(v0_0,v0_1,v0_2);
            __m256i r0_max = max3(v0_0,v0_1,v0_2);

            __m256i r1_min = min3(v1_0,v1_1,v1_2);
            __m256i r1_med = med3(v1_0,v1_1,v1_2);
            __m256i r1_max = max3(v1_0,v1_1,v1_2);

            __m256i r2_min = min3(v2_0,v2_1,v2_2);
            __m256i r2_med = med3(v2_0,v2_1,v2_2);
            __m256i r2_max = max3(v2_0,v2_1,v2_2);

            __m256i max_of_mins = max3(r0_min,r1_min,r2_min);
            __m256i med_of_meds = med3(r0_med,r1_med,r2_med);
            __m256i min_of_maxs = min3(r0_max,r1_max,r2_max);

            __m256i median = med3(max_of_mins,med_of_meds,min_of_maxs);

            __m256i center = v1_1;

            __m256i diff1 = _mm256_subs_epu8(center,median);
            __m256i diff2 = _mm256_subs_epu8(median,center);
            __m256i abs_diff = _mm256_or_si256(diff1,diff2);

            __m256i bd = _mm256_sub_epi8(abs_diff,offset128);
            __m256i bt = _mm256_sub_epi8(thresh_vec,offset128);

            __m256i mask = _mm256_cmpgt_epi8(bd,bt);

            __m256i res = _mm256_blendv_epi8(center,median,mask);

            _mm256_storeu_si256((__m256i*)&out[r1 + x],res);

            // sliding window
            v0_0=v0_1; v0_1=v0_2;
            v1_0=v1_1; v1_1=v1_2;
            v2_0=v2_1; v2_1=v2_2;

            v0_2 = _mm256_loadu_si256((__m256i*)&in[r0 + x + 32]);
            v1_2 = _mm256_loadu_si256((__m256i*)&in[r1 + x + 32]);
            v2_2 = _mm256_loadu_si256((__m256i*)&in[r2 + x + 32]);
        }
    }
}       
// ---------------- THREAD ----------------
template<int R>
void worker(Image& img, Image& out, int threshold,
            atomic<int>& next_row, int h, int chunk){

    while(true){
        int y = next_row.fetch_add(chunk);
        if(y>=h) break;

        int end = min(y+chunk,h);

        processChannel<R>(img.r.data(),out.r.data(),img.width,img.height,threshold,y,end,0);
        processChannel<R>(img.g.data(),out.g.data(),img.width,img.height,threshold,y,end,0);
        processChannel<R>(img.b.data(),out.b.data(),img.width,img.height,threshold,y,end,0);
    }
}

// ---------------- MAIN ----------------
int main(int argc,char* argv[]){

    string name=argv[1];
    int n_threads=stoi(argv[2]);

    Image img = loadImage("./error_images/"+name+".ppm");
    Image out(img.height,img.width);

    int chunk_size = 64;   // 🔥 importante

    atomic<int> next_row(0);

    vector<thread> threads;

    auto start=chrono::high_resolution_clock::now();

    for(int i=0;i<n_threads;i++)
        threads.emplace_back(worker<1>,ref(img),ref(out),0,ref(next_row),img.height,chunk_size);

    for(auto& t:threads) t.join();

    auto end=chrono::high_resolution_clock::now();

    cout<<"Tempo: "<<chrono::duration_cast<chrono::milliseconds>(end-start).count()<<" ms\n";

    saveImage("./output_images/"+name+".ppm",out);
}