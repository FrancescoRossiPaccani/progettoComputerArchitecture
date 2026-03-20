#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <immintrin.h>

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
        
        for(int i=0;i<h*w;i++){
            img.r[i] = file.get();
            img.g[i] = file.get();
            img.b[i] = file.get();
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
    ofstream file(path);

    file<<"P3\n"<<img.width<<" "<<img.height<<"\n255\n";

    for(int i=0;i<img.width*img.height;i++){
        file<<(int)img.r[i]<<" "
            <<(int)img.g[i]<<" "
            <<(int)img.b[i]<<" ";
    }
}

// ---------------- MEDIAN ----------------
inline void updateMedian(int& m, int& count, uint16_t* hist, int mid)
{
    while(count > mid){
        count -= hist[m];
        m--;
    }
    while(count + hist[m] <= mid){
        m++;
        count += hist[m];
    }
}

// ---------------- CORE OTTIMIZZATO ----------------
void processChannel(uint8_t* __restrict in,
                    uint8_t* __restrict out,
                    int w, int h,
                    int kernel, int threshold,
                    int start_y, int end_y)
{
    int radius = kernel/2;
    int mid = (kernel*kernel)/2;

    vector<int> row_offsets(2 * radius + 1);

    for(int y=start_y; y<end_y; y++)
    {
        alignas(64) uint16_t hist[256] = {0};

        // PRECALCOLO righe
        for(int fy=-radius; fy<=radius; fy++)
            row_offsets[fy + radius] = min(max(y+fy,0), h-1) * w;

        // INIT HIST
        for(int fy=-radius; fy<=radius; fy++){
            int ro = row_offsets[fy + radius];
            for(int fx=-radius; fx<=radius; fx++){
                int ix = min(max(fx,0), w-1);
                hist[in[ro + ix]]++;
            }
        }

        int m=0,count=0;
        while(count<=mid){
            count += hist[m];
            m++;
        }
        m--;

        for(int x=0; x<w; x++)
        {
            int idx = y*w+x;
            int center = in[idx];

            out[idx] = (abs(center-m)>threshold)?m:center;

            if(x==w-1) break;

            int left  = max(x-radius,0);
            int right = min(x+radius+1,w-1);

            // PREFETCH (aiuta su Alder Lake)
            _mm_prefetch((char*)(in + idx + 64), _MM_HINT_T0);

            for(int fy=-radius; fy<=radius; fy++){
                int ro = row_offsets[fy + radius];

                int oldv = in[ro + left];
                int newv = in[ro + right];

                hist[oldv]--; count -= (oldv <= m);
                hist[newv]++; count += (newv <= m);
            }

            updateMedian(m,count,hist,mid);
        }
    }
}

// ---------------- THREAD ----------------
void worker(Image& img, Image& out,
            int kernel, int threshold,
            atomic<int>& next_row)
{
    int h = img.height;

    while(true){
        int y = next_row.fetch_add(1);
        if(y >= h) break;

        processChannel(img.r.data(), out.r.data(), img.width, img.height, kernel, threshold, y, y+1);
        processChannel(img.g.data(), out.g.data(), img.width, img.height, kernel, threshold, y, y+1);
        processChannel(img.b.data(), out.b.data(), img.width, img.height, kernel, threshold, y, y+1);
    }
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[])
{
    if(argc < 5){
        cout<<"Uso: "<<argv[0]<<" nome_immagine num_thread kernel bucket\n";
        return 1;
    }

    string img_name = argv[1];
    int n_threads = stoi(argv[2]);
    int kernel = stoi(argv[3]);

    Image img = loadImage("./error_images/"+img_name+".ppm");
    Image out(img.height,img.width);

    int threshold = 40;

    atomic<int> next_row(0);

    auto start = chrono::high_resolution_clock::now();

    vector<thread> threads;

    for(int i=0;i<n_threads;i++){
        threads.emplace_back(worker,
                             ref(img), ref(out),
                             kernel, threshold,
                             ref(next_row));
    }

    for(auto& t:threads) t.join();

    auto end = chrono::high_resolution_clock::now();

    cout<<"Tempo: "
        <<chrono::duration_cast<chrono::milliseconds>(end-start).count()
        <<" ms\n";

    saveImage("./output_images/"+img_name+".ppm",out);
}