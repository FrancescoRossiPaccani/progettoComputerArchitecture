#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>

using namespace std;

struct Image{
    int width, height;
    vector<uint8_t> r, g, b;

    Image(int h,int w):width(w),height(h),
        r(h*w), g(h*w), b(h*w){}
};

// ---------------- LOAD ----------------
Image loadImage(const string& path){
    ifstream file(path);
    if(!file){
        cerr<<"Errore apertura file\n";
        exit(1);
    }

    string format;
    file>>format;

    int w,h,maxval;
    file>>w>>h>>maxval;

    Image img(h,w);

    for(int i=0;i<h*w;i++){
        int R,G,B;
        file>>R>>G>>B;
        img.r[i]=R;
        img.g[i]=G;
        img.b[i]=B;
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

// ---------------- CORE ----------------
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

// 🔥 funzione ottimizzata per un canale
void processChannel(uint8_t* __restrict in,
                    uint8_t* __restrict out,
                    int w, int h,
                    int kernel, int threshold,
                    int start_y, int end_y)
{
    int radius = kernel/2;
    int mid = (kernel*kernel)/2;

    for(int y=start_y; y<end_y; y++)
    {
        uint16_t hist[256] = {0};

        // init histogram
        for(int fy=-radius; fy<=radius; fy++)
        {
            int iy = min(max(y+fy,0), h-1);

            for(int fx=-radius; fx<=radius; fx++)
            {
                int ix = min(max(fx,0), w-1);
                hist[in[iy*w+ix]]++;
            }
        }

        // initial median
        int m=0,count=0;
        while(count<=mid){
            count += hist[m];
            m++;
        }
        m--;

        for(int x=0;x<w;x++)
        {
            int idx = y*w+x;
            int center = in[idx];

            out[idx] = (abs(center-m)>threshold)?m:center;

            if(x==w-1) break;

            int left  = max(x-radius,0);
            int right = min(x+radius+1,w-1);

            for(int fy=-radius; fy<=radius; fy++)
            {
                int iy = min(max(y+fy,0), h-1);

                int oldv = in[iy*w+left];
                int newv = in[iy*w+right];

                hist[oldv]--; if(oldv<=m) count--;
                hist[newv]++; if(newv<=m) count++;
            }

            updateMedian(m,count,hist,mid);
        }
    }
}

// ---------------- THREAD WORKER ----------------
void worker(Image& img, Image& out,
            int kernel, int threshold,
            int start_y, int end_y)
{
    processChannel(img.r.data(), out.r.data(),
                   img.width, img.height,
                   kernel, threshold,
                   start_y, end_y);

    processChannel(img.g.data(), out.g.data(),
                   img.width, img.height,
                   kernel, threshold,
                   start_y, end_y);

    processChannel(img.b.data(), out.b.data(),
                   img.width, img.height,
                   kernel, threshold,
                   start_y, end_y);
}

// ---------------- MAIN ----------------
int main(int argc, char* argv[])
{
    if(argc < 3){
        cout<<"Uso: "<<argv[0]<<" nome_immagine num_thread\n";
        return 1;
    }

    string img_name = argv[1];
    int n_threads = stoi(argv[2]);

    Image img = loadImage("./error_images/"+img_name+".ppm");
    Image out(img.height,img.width);

    cout<<"Immagine "<<img.width<<"x"<<img.height<<"\n";

    int kernel = 7;
    int threshold = 40;

    auto start = chrono::high_resolution_clock::now();

    vector<thread> threads;

    for(int i=0;i<n_threads;i++)
    {
        int s = i*img.height/n_threads;
        int e = (i+1)*img.height/n_threads;

        threads.emplace_back(worker,
                             ref(img), ref(out),
                             kernel, threshold,
                             s, e);
    }

    for(auto& t:threads) t.join();

    auto end = chrono::high_resolution_clock::now();

    cout<<"Tempo: "
        <<chrono::duration_cast<chrono::milliseconds>(end-start).count()
        <<" ms\n";

    saveImage("./output_images/"+img_name+".ppm",out);
}