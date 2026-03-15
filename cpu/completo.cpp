#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cmath> // per exp()

using namespace std;

struct Image
{
    int width;
    int height;
    vector<int> data;

    Image(int h, int w) : width(w), height(h), data(h*w*3) {}

    int& operator()(int y, int x, int c){
        return data[(y * width + x)*3 + c];
    }
};

Image loadImage(const string& path)
{
    ifstream file(path);
    if(!file){ cerr << "Errore apertura file\n"; exit(1); }

    string format;
    file >> format;
    if(format != "P3"){ cerr << "Formato non supportato\n"; exit(1); }

    int w,h,maxval;
    file >> w >> h >> maxval;

    Image img(h,w);
    for(int y=0;y<h;y++)
        for(int x=0;x<w;x++){
            int r,g,b; file >> r >> g >> b;
            img(y,x,0)=r; img(y,x,1)=g; img(y,x,2)=b;
        }
    return img;
}

void saveImage(const string& path, Image& img)
{
    ofstream file(path);
    if(!file){ cerr << "Errore scrittura " << path << endl; exit(1); }
    file << "P3\n" << img.width << " " << img.height << "\n255\n";
    for(int y=0;y<img.height;y++)
        for(int x=0;x<img.width;x++)
            file << img(y,x,0) << " " << img(y,x,1) << " " << img(y,x,2) << " ";
}

// ==========================
// Median filter worker
// ==========================
void medianWorker(Image& img, Image& out, int M, int& counter, mutex& mtx)
{
    int radius = M/2;
    while(true){
        mtx.lock();
        int y = counter++;
        mtx.unlock();
        if(y >= img.height) break;

        for(int x=0;x<img.width;x++)
            for(int c=0;c<3;c++){
                vector<int> block;
                for(int fy=-radius; fy<=radius; fy++)
                    for(int fx=-radius; fx<=radius; fx++){
                        int iy = min(max(y+fy,0),img.height-1);
                        int ix = min(max(x+fx,0),img.width-1);
                        block.push_back(img(iy,ix,c));
                    }
                sort(block.begin(), block.end());
                out(y,x,c) = block[block.size()/2];
            }
    }
}

// ==========================
// Bilateral filter worker
// ==========================
void bilateralWorker(Image& img, Image& out, int M, float sigma_s, float sigma_r, int& counter, mutex& mtx)
{
    int radius = M/2;
    while(true){
        mtx.lock();
        int y = counter++;
        mtx.unlock();
        if(y >= img.height) break;

        for(int x=0;x<img.width;x++)
            for(int c=0;c<3;c++){
                float sum=0, wsum=0;
                for(int fy=-radius; fy<=radius; fy++)
                    for(int fx=-radius; fx<=radius; fx++){
                        int iy = min(max(y+fy,0), img.height-1);
                        int ix = min(max(x+fx,0), img.width-1);
                        float gs = exp(-(fx*fx+fy*fy)/(2*sigma_s*sigma_s));
                        float gr = exp(-((img(iy,ix,c)-img(y,x,c))*(img(iy,ix,c)-img(y,x,c)))/(2*sigma_r*sigma_r));
                        float w = gs*gr;
                        sum += w*img(iy,ix,c);
                        wsum += w;
                    }
                out(y,x,c) = min(max(int(sum/wsum),0),255);
            }
    }
}

// ==========================
// Sharpen worker
// ==========================
void sharpenWorker(Image& img, Image& out, int* filter, int M, int& counter, mutex& mtx)
{
    int radius = M/2;
    while(true){
        mtx.lock();
        int y = counter++;
        mtx.unlock();
        if(y >= img.height) break;

        for(int x=0;x<img.width;x++)
            for(int c=0;c<3;c++){
                float sum=0;
                for(int fy=-radius; fy<=radius; fy++)
                    for(int fx=-radius; fx<=radius; fx++){
                        int iy = min(max(y+fy,0), img.height-1);
                        int ix = min(max(x+fx,0), img.width-1);
                        int k = (fy+radius)*M + (fx+radius);
                        sum += filter[k]*img(iy,ix,c);
                    }
                out(y,x,c) = min(max(int(sum),0),255);
            }
    }
}


void detailWorker(Image& img, Image& out, int* filter, int M, float norm, float alpha, int& counter, mutex& mtx)
{
    int radius = M/2;
    while(true){
        mtx.lock();
        int y = counter++;
        mtx.unlock();
        if(y>=img.height) break;

        for(int x=0;x<img.width;x++){
            for(int c=0;c<3;c++){
                float sum=0;
                // Gaussian blur leggera
                for(int fy=-radius; fy<=radius; fy++)
                    for(int fx=-radius; fx<=radius; fx++){
                        int iy = min(max(y+fy,0), img.height-1);
                        int ix = min(max(x+fx,0), img.width-1);
                        int k = (fy+radius)*M + (fx+radius);
                        sum += filter[k]*img(iy,ix,c);
                    }
                float blurred = sum*norm;
                float detail = img(y,x,c) - blurred;
                int val = min(max(int(img(y,x,c) + alpha*detail),0),255);
                out(y,x,c) = val;
            }
        }
    }
}
int main()
{
    string input_path  = "./img/";
    string output_path = "./output/";
    string img_name    = "img1.ppm";

    Image img = loadImage(input_path + img_name);
    cout<<"Immagine caricata, dimensione: "<< img.height<<" x "<< img.width << endl;

    int n_thread = 3;
    mutex mtx;
    int counter;
    vector<thread> threads;

    // =========================
    // Median filter 7x7
    // =========================
    Image temp1(img.height,img.width);
    counter=0;
    for(int i=0;i<n_thread;i++) threads.push_back(thread(medianWorker,ref(img),ref(temp1),7,ref(counter),ref(mtx)));
    for(auto &t: threads) t.join();
    threads.clear();
    cout << "Median 7x7 completato" << endl;

    // =========================
    // Median filter 5x5
    // =========================
    Image temp2(img.height,img.width);
    counter=0;
    for(int i=0;i<n_thread;i++) threads.push_back(thread(medianWorker,ref(temp1),ref(temp2),5,ref(counter),ref(mtx)));
    for(auto &t: threads) t.join();
    threads.clear();
    cout << "Median 5x5 completato" << endl;

    // =========================
    // Bilateral filter 5x5
    // =========================
    Image temp3(img.height,img.width);
    counter=0;
    for(int i=0;i<n_thread;i++) threads.push_back(thread(bilateralWorker,ref(temp2),ref(temp3),5,5.0,50.0,ref(counter),ref(mtx)));
    for(auto &t: threads) t.join();
    threads.clear();
    cout << "Bilateral 5x5 completato" << endl;

    // =========================
    // Sharpen 3x3
    // =========================
    int sharpen3[9]={0,-1,0,-1,5,-1,0,-1,0};
    Image out(img.height,img.width);
    counter=0;
    for(int i=0;i<n_thread;i++) threads.push_back(thread(sharpenWorker,ref(temp3),ref(out),sharpen3,3,ref(counter),ref(mtx)));
    for(auto &t: threads) t.join();
    threads.clear();
    cout << "Sharpen 3x3 completato" << endl;

    saveImage(output_path + img_name, out);
    cout << "Immagine finale salvata in " << output_path + img_name << endl;

    // =========================
    // Detail Enhancement
    // =========================
    int gauss3[9]={1,2,1,2,4,2,1,2,1};
    float norm3 = 1.0/16.0;
    Image temp4(img.height,img.width);
    counter=0;
    float alpha = 1.2; // controlla forza dei dettagli
    for(int i=0;i<n_thread;i++)
        threads.push_back(thread(detailWorker, ref(out), ref(temp4), gauss3, 3, norm3, alpha, ref(counter), ref(mtx)));
    for(auto &t: threads) t.join();
    threads.clear();
    out = temp4; // ora out contiene immagine finale con dettagli enfatizzati

    return 0;
}