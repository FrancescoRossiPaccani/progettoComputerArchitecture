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

    int& operator()(int y,int x,int c)
    {
        return data[(y*width+x)*3+c];
    }

    int operator()(int y,int x,int c) const
    {
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

inline int histogramMedian(int hist[], int windowSize)
{
    int count = 0;
    int mid = windowSize/2;

    for(int i=0;i<256;i++)
    {
        count += hist[i];
        if(count > mid)
            return i;
    }

    return 255;
}

void selectiveMedianWorker(Image& img, Image& out,
                           int kernel_size,
                           int threshold,
                           int start_row,
                           int end_row)
{
    int radius = kernel_size/2;
    int windowSize = kernel_size * kernel_size;

    for(int y=start_row; y<end_row; y++)
    {
        int histR[256]={0};
        int histG[256]={0};
        int histB[256]={0};

        // inizializzazione finestra
        for(int fy=-radius; fy<=radius; fy++)
        for(int fx=-radius; fx<=radius; fx++)
        {
            int iy = min(max(y+fy,0), img.height-1);
            int ix = min(max(fx,0), img.width-1);

            histR[ img(iy,ix,0) ]++;
            histG[ img(iy,ix,1) ]++;
            histB[ img(iy,ix,2) ]++;
        }

        for(int x=0;x<img.width;x++)
        {
            int medR = histogramMedian(histR,windowSize);
            int medG = histogramMedian(histG,windowSize);
            int medB = histogramMedian(histB,windowSize);

            int r = img(y,x,0);
            int g = img(y,x,1);
            int b = img(y,x,2);

            out(y,x,0) = (abs(r-medR)>threshold)?medR:r;
            out(y,x,1) = (abs(g-medG)>threshold)?medG:g;
            out(y,x,2) = (abs(b-medB)>threshold)?medB:b;

            if(x == img.width-1)
                continue;

            // rimuovi colonna sinistra
            for(int fy=-radius; fy<=radius; fy++)
            {
                int iy = min(max(y+fy,0), img.height-1);
                int ix = min(max(x-radius,0), img.width-1);

                histR[ img(iy,ix,0) ]--;
                histG[ img(iy,ix,1) ]--;
                histB[ img(iy,ix,2) ]--;
            }

            // aggiungi colonna destra
            for(int fy=-radius; fy<=radius; fy++)
            {
                int iy = min(max(y+fy,0), img.height-1);
                int ix = min(max(x+radius+1,0), img.width-1);

                histR[ img(iy,ix,0) ]++;
                histG[ img(iy,ix,1) ]++;
                histB[ img(iy,ix,2) ]++;
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