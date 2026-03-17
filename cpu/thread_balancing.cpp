#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdint>

using namespace std;

struct Image{
    int width, height;
    vector<uint8_t> data;

    Image(int h,int w):width(w),height(h),data(h*w*3){}

    inline uint8_t& operator()(int y,int x,int c){
        return data[(y*width+x)*3+c];
    }
};

Image loadImage(const string& path){
    ifstream file(path);
    string format; file>>format;

    int w,h,maxval;
    file>>w>>h>>maxval;

    Image img(h,w);

    for(int i=0;i<h*w;i++){
        int r,g,b;
        file>>r>>g>>b;
        img.data[i*3+0]=r;
        img.data[i*3+1]=g;
        img.data[i*3+2]=b;
    }
    return img;
}

void saveImage(const string& path, Image& img){
    ofstream file(path);
    file<<"P3\n"<<img.width<<" "<<img.height<<"\n255\n";

    for(int i=0;i<img.width*img.height;i++){
        file<<(int)img.data[i*3]<<" "
            <<(int)img.data[i*3+1]<<" "
            <<(int)img.data[i*3+2]<<" ";
    }
}

// 🔥 aggiorna mediana in O(1)
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

void worker(Image& img, Image& out,
            int kernel, int threshold,
            int start_y, int end_y)
{
    int radius = kernel/2;
    int mid = (kernel*kernel)/2;

    int w = img.width;
    int h = img.height;

    uint8_t* in = img.data.data();
    uint8_t* outp = out.data.data();

    for(int y=start_y; y<end_y; y++)
    {
        for(int c=0;c<3;c++)
        {
            uint16_t hist[256] = {0};

            // init
            for(int fy=-radius; fy<=radius; fy++)
            {
                int iy = min(max(y+fy,0), h-1);

                for(int fx=-radius; fx<=radius; fx++)
                {
                    int ix = min(max(fx,0), w-1);
                    hist[in[(iy*w+ix)*3+c]]++;
                }
            }

            // trova mediana iniziale
            int m = 0, count = 0;
            while(count <= mid){
                count += hist[m];
                m++;
            }
            m--;

            for(int x=0;x<w;x++)
            {
                int idx = (y*w+x)*3+c;
                int center = in[idx];

                outp[idx] = (abs(center-m) > threshold) ? m : center;

                if(x == w-1) break;

                int left = max(x-radius,0);
                int right = min(x+radius+1,w-1);

                for(int fy=-radius; fy<=radius; fy++)
                {
                    int iy = min(max(y+fy,0), h-1);

                    int oldv = in[(iy*w+left)*3+c];
                    int newv = in[(iy*w+right)*3+c];

                    hist[oldv]--;
                    if(oldv <= m) count--;

                    hist[newv]++;
                    if(newv <= m) count++;
                }

                updateMedian(m, count, hist, mid);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    string img_name = argv[1];
    int n_threads = stoi(argv[2]);

    Image img = loadImage("./error_images/"+img_name+".ppm");
    Image out(img.height,img.width);

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