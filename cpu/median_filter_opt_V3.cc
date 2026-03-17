#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <algorithm>
#include <chrono>

using namespace std;

struct Image{
    int width;
    int height;
    vector<int> data;

    Image(int h,int w):width(w),height(h),data(h*w*3){}

    int& operator()(int y,int x,int c){
        return data[(y*width+x)*3+c];
    }

    int operator()(int y,int x,int c) const{
        return data[(y*width+x)*3+c];
    }
};

Image loadImage(const string& path){
    ifstream file(path);

    if(!file){
        cerr<<"Errore apertura file\n";
        exit(1);
    }

    string format;
    file>>format;

    if(format!="P3"){
        cerr<<"Formato non supportato\n";
        exit(1);
    }

    int w,h,maxval;
    file>>w>>h>>maxval;

    Image img(h,w);

    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++)
    {
        int r,g,b;
        file>>r>>g>>b;

        img(y,x,0)=r;
        img(y,x,1)=g;
        img(y,x,2)=b;
    }

    return img;
}

void saveImage(const string& path, Image& img){
    ofstream file(path);

    if(!file){
        cerr<<"Errore scrittura file\n";
        exit(1);
    }

    file<<"P3\n"<<img.width<<" "<<img.height<<"\n255\n";

    for(int y=0;y<img.height;y++)
    for(int x=0;x<img.width;x++)
    {
        file<<img(y,x,0)<<" "
            <<img(y,x,1)<<" "
            <<img(y,x,2)<<" ";
    }
}

void logPerformance(const string& filename, const string& img_name, int width, int height, int n_threads, long long time_ms){
    ofstream logFile(filename, ios::app);

    if(!logFile){
        cerr << "Errore apertura file log\n";
        return;
    }

    logFile << img_name << "," << width << "x" << height << "," << n_threads << "," << time_ms << "ms" << endl;
}

#include <algorithm>
#include <cmath>

using std::min;
using std::max;

inline void histogramMedian3(int hist[3][256], int windowSize,
                             int &m0, int &m1, int &m2)
{
    int mid = windowSize / 2;

    int c0 = 0, c1 = 0, c2 = 0;
    bool f0 = false, f1 = false, f2 = false;

    for(int i = 0; i < 256; i++)
    {
        if(!f0) { c0 += hist[0][i]; if(c0 > mid) { m0 = i; f0 = true; } }
        if(!f1) { c1 += hist[1][i]; if(c1 > mid) { m1 = i; f1 = true; } }
        if(!f2) { c2 += hist[2][i]; if(c2 > mid) { m2 = i; f2 = true; } }

        if(f0 && f1 && f2) break;
    }
}

void selectiveMedianWorker(Image& img, Image& out,
                           int kernel_size,
                           int threshold,
                           int start_row,
                           int end_row)
{
    int radius = kernel_size / 2;
    int windowSize = kernel_size * kernel_size;

    for(int y = start_row; y < end_row; y++)
    {
        int hist[3][256] = {0};

        for(int fy = -radius; fy <= radius; fy++)
        {
            int iy = min(max(y + fy, 0), img.height - 1);

            for(int fx = -radius; fx <= radius; fx++)
            {
                int ix = min(max(fx, 0), img.width - 1);

                int v0 = img(iy, ix, 0);
                int v1 = img(iy, ix, 1);
                int v2 = img(iy, ix, 2);

                hist[0][v0]++;
                hist[1][v1]++;
                hist[2][v2]++;
            }
        }

        for(int x = 0; x < img.width; x++)
        {
            int m0 = 0, m1 = 0, m2 = 0;
            histogramMedian3(hist, windowSize, m0, m1, m2);

            int c0 = img(y, x, 0);
            int c1 = img(y, x, 1);
            int c2 = img(y, x, 2);

            out(y, x, 0) = (abs(c0 - m0) > threshold) ? m0 : c0;
            out(y, x, 1) = (abs(c1 - m1) > threshold) ? m1 : c1;
            out(y, x, 2) = (abs(c2 - m2) > threshold) ? m2 : c2;

            if(x == img.width - 1)
                continue;

            for(int fy = -radius; fy <= radius; fy++)
            {
                int iy = min(max(y + fy, 0), img.height - 1);
                int ix = min(max(x - radius, 0), img.width - 1);

                int v0 = img(iy, ix, 0);
                int v1 = img(iy, ix, 1);
                int v2 = img(iy, ix, 2);

                hist[0][v0]--;
                hist[1][v1]--;
                hist[2][v2]--;
            }

            for(int fy = -radius; fy <= radius; fy++)
            {
                int iy = min(max(y + fy, 0), img.height - 1);
                int ix = min(max(x + radius + 1, 0), img.width - 1);

                int v0 = img(iy, ix, 0);
                int v1 = img(iy, ix, 1);
                int v2 = img(iy, ix, 2);

                hist[0][v0]++;
                hist[1][v1]++;
                hist[2][v2]++;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        cout << "Uso: " << argv[0] << " nome_immagine num_thread\n";
        return 1;
    }

    string img_name = argv[1];
    string img_ext  = ".ppm";
    string input    = "./error_images/";
    string output   = "./output_images/";

    Image img = loadImage(input + img_name + img_ext);

    cout << "Immagine caricata "
         << img.height << "x"
         << img.width << endl;

    int kernel_size = 7;
    int threshold = 40;

    int n_threads = stoi(argv[2]);
    cout<<"Inizio a calcolare utilizzando " << n_threads << " threads\n";

    Image out(img.height,img.width);

    auto start = chrono::high_resolution_clock::now();

    vector<thread> threads;

    for(int i=0;i<n_threads;i++)
    {
        int start_row = i * img.height / n_threads;
        int end_row   = (i+1) * img.height / n_threads;

        threads.emplace_back(
            selectiveMedianWorker,
            ref(img),
            ref(out),
            kernel_size,
            threshold,
            start_row,
            end_row
        );
    }

    for(auto& t:threads)
        t.join();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    cout << "Tempo impiegato: "
         << duration.count()
         << " ms\n";

    logPerformance(
        "log.txt",
        img_name + img_ext,
        img.width,
        img.height,
        n_threads,
        duration.count()
    );

    saveImage(output + img_name + img_ext,out);

    cout << "Immagine salvata in "
         << output << endl;

    return 0;
}