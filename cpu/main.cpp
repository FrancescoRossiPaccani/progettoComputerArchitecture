#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

/*
    Per convertire l'immagine png (o altro) in ppm da shell
    1) 
        convert img.png -compress none img.ppm
    2)
        magick img.png -compress none img.ppm

    N.B. dipende dalla versione di ImageMagick
*/


struct Image
{
    int width;
    int height;
    vector<int> data;

    Image(int h, int w) : width(w), height(h), data(h*w) {}

    int& operator()(int y, int x)
    {
        return data[y * width + x];
    }
};

Image loadImageWithPadding(const string& path, int M)
{
    ifstream file(path);

    if(!file)
    {
        cerr << "Errore apertura file\n";
        exit(1);
    }

    string format;
    file >> format;

    if(format != "P3")
    {
        cerr << "Formato non supportato (usa P3 PPM)\n";
        exit(1);
    }

    int w,h,maxval;
    file >> w >> h >> maxval;

    int pad = M / 2;

    int newH = h + 2*pad;
    int newW = w + 2*pad;

    Image out(newH,newW);

    // lettura pixel
    for(int y=0;y<h;y++)
    {
        for(int x=0;x<w;x++)
        {
            int r,g,b;
            file >> r >> g >> b;

            int gray = (r+g+b)/3;

            out(y+pad,x+pad) = gray;
        }
    }

    // padding sinistra/destra
    for(int y=pad;y<h+pad;y++)
    {
        for(int x=0;x<pad;x++)
        {
            out(y,x) = out(y,pad);
            out(y,newW-x-1) = out(y,newW-pad-1);
        }
    }

    // padding sopra/sotto
    for(int x=0;x<newW;x++)
    {
        for(int y=0;y<pad;y++)
        {
            out(y,x) = out(pad,x);
            out(newH-y-1,x) = out(newH-pad-1,x);
        }
    }

    return out;
}

int main()
{
    string path = "./img/img1.ppm";
    int M = 5; //dimensione filtro

    Image img = loadImageWithPadding(path,M);

    cout<<"dimensione matrice: "
        <<img.height<<" x "
        <<img.width<<endl;

    return 0;
}
