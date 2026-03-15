#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>
#include <algorithm>

using namespace std;

struct Image
{
    int width;
    int height;
    vector<int> data;

    Image(int h,int w):width(w),height(h),data(h*w*3){}

    int& operator()(int y,int x,int c)
    {
        return data[(y*width+x)*3+c];
    }
};

////////////////////////////////////////////////////////////
// Load image
////////////////////////////////////////////////////////////

Image loadImage(const string& path)
{
    ifstream file(path);

    if(!file)
    {
        cerr<<"Errore apertura file\n";
        exit(1);
    }

    string format;
    file>>format;

    if(format!="P3")
    {
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

////////////////////////////////////////////////////////////
// Save image
////////////////////////////////////////////////////////////

void saveImage(const string& path,Image& img)
{
    ofstream file(path);

    if(!file)
    {
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

////////////////////////////////////////////////////////////
// Selective Median Worker
////////////////////////////////////////////////////////////

void selectiveMedianWorker(Image& img,Image& out,
                           int M,int threshold,
                           int& counter,mutex& mtx)
{
    int radius=M/2;

    while(true)
    {
        mtx.lock();
        int y=counter++;
        mtx.unlock();

        if(y>=img.height)
            break;

        for(int x=0;x<img.width;x++)
        for(int c=0;c<3;c++)
        {
            vector<int> block;
            block.reserve(M*M);

            for(int fy=-radius;fy<=radius;fy++)
            for(int fx=-radius;fx<=radius;fx++)
            {
                int iy=min(max(y+fy,0),img.height-1);
                int ix=min(max(x+fx,0),img.width-1);

                block.push_back(img(iy,ix,c));
            }

            sort(block.begin(),block.end());

            int median=block[block.size()/2];
            int center=img(y,x,c);

            if(abs(center-median)>threshold)
                out(y,x,c)=median;
            else
                out(y,x,c)=center;
        }
    }
}

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        cout << "Uso: " << argv[0] << " nome_immagine\n";
        return 1;
    }

    string img_name = argv[1];
    string img_ext = ".ppm";
    string input="./error_images/";
    string output="./output_images/";

    Image img=loadImage(input + img_name + img_ext);

    cout<<"Immagine caricata "<<img.height<<"x"<<img.width<<endl;

    ////////////////////////////////////////////////////////

    int kernel_size=7;
    int threshold=40;
    int n_threads=12;

    ////////////////////////////////////////////////////////

    Image out(img.height,img.width);

    mutex mtx;
    int counter=0;

    vector<thread> threads;

    for(int i=0;i<n_threads;i++)
    {
        threads.emplace_back(
            selectiveMedianWorker,
            ref(img),
            ref(out),
            kernel_size,
            threshold,
            ref(counter),
            ref(mtx)
        );
    }

    for(auto& t:threads)
        t.join();

    ////////////////////////////////////////////////////////

    saveImage(output + img_name + img_ext,out);

    cout<<"Immagine salvata in "<<output<<endl;

    return 0;
}