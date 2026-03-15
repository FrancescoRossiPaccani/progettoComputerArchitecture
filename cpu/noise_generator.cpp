#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>

using namespace std;

struct Image {
    int width, height;
    vector<int> data;

    Image() : width(0), height(0) {}
    Image(int h, int w) : width(w), height(h), data(h * w * 3, 0) {}

    int& operator()(int y, int x, int c) {
        return data[(y * width + x) * 3 + c];
    }

    const int& operator()(int y, int x, int c) const {
        return data[(y * width + x) * 3 + c];
    }
};

static inline int clampi(int v, int lo = 0, int hi = 255) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

Image loadImage(const string& path)
{
    ifstream file(path, ios::binary);

    if(!file){
        cerr << "Errore apertura file: " << path << endl;
        exit(1);
    }

    string format;
    file >> format;

    int w,h,maxval;
    file >> w >> h >> maxval;

    file.get(); // consuma newline dopo header

    Image img(h,w);

    //////////////////////////////////////////////////////
    // P3 (ASCII)
    //////////////////////////////////////////////////////

    if(format == "P3")
    {
        for(int y=0;y<h;y++)
        for(int x=0;x<w;x++)
        {
            int r,g,b;

            file >> r >> g >> b;

            img(y,x,0) = clampi(r);
            img(y,x,1) = clampi(g);
            img(y,x,2) = clampi(b);
        }
    }

    //////////////////////////////////////////////////////
    // P6 (Binary)
    //////////////////////////////////////////////////////

    else if(format == "P6")
    {
        for(int y=0;y<h;y++)
        for(int x=0;x<w;x++)
        {
            unsigned char rgb[3];

            file.read((char*)rgb,3);

            img(y,x,0) = rgb[0];
            img(y,x,1) = rgb[1];
            img(y,x,2) = rgb[2];
        }
    }

    else
    {
        cerr << "Formato PPM non supportato: " << format << endl;
        exit(1);
    }

    return img;
}


void saveImage(const string& path, const Image& img) {
    ofstream file(path);
    if (!file) {
        cerr << "Errore scrittura file: " << path << endl;
        exit(1);
    }

    file << "P3\n";
    file << img.width << " " << img.height << "\n255\n";
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            file << clampi(img(y, x, 0)) << " "
                 << clampi(img(y, x, 1)) << " "
                 << clampi(img(y, x, 2)) << " ";
        }
        file << "\n";
    }
}

struct NoiseParams {
    double saltPepperPct = 20.0;   // percentuale pixel colpiti da S&P
    double saltRatio     = 0.5;    // 0.5 => metà sale, metà pepe

    double gaussianPct   = 100.0;  // percentuale pixel colpiti da gaussiano
    double gaussianSigma = 8.0;    // deviazione standard

    double specklePct    = 0.0;    // percentuale pixel colpiti da speckle
    double speckleSigma  = 0.05;   // intensità speckle

    double rvImpulsePct  = 0.0;    // random valued impulse noise %

    double poissonPct    = 0.0;    // percentuale pixel colpiti da Poisson
};

void addNoise(Image& img, const NoiseParams& p, unsigned int seed = 12345) {
    mt19937 rng(seed);

    uniform_real_distribution<double> prob(0.0, 100.0);
    uniform_real_distribution<double> saltChoice(0.0, 1.0);
    normal_distribution<double> gaussDist(0.0, p.gaussianSigma);
    normal_distribution<double> speckleDist(0.0, p.speckleSigma);
    uniform_int_distribution<int> randVal(0, 255);

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            for (int c = 0; c < 3; c++) {
                int v = img(y, x, c);

                // Salt & Pepper
                if (prob(rng) < p.saltPepperPct) {
                    v = (saltChoice(rng) < p.saltRatio) ? 255 : 0;
                }

                // Random-valued impulse noise
                if (prob(rng) < p.rvImpulsePct) {
                    v = randVal(rng);
                }

                // Additive Gaussian noise
                if (prob(rng) < p.gaussianPct) {
                    v = clampi(int(v + gaussDist(rng) + 0.5));
                }

                // Multiplicative speckle noise
                if (prob(rng) < p.specklePct) {
                    double n = speckleDist(rng);
                    v = clampi(int(v + v * n + 0.5));
                }

                // Poisson-like noise
                if (prob(rng) < p.poissonPct) {
                    double lambda = max(0.0, double(v));
                    poisson_distribution<int> pois(lambda);
                    v = clampi(pois(rng));
                }

                img(y, x, c) = v;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if(argc < 2)
        {
            cout << "Uso: " << argv[0] << " nome_immagine\n";
            return 1;
        }

    string img_name    = argv[1];
    string input_path  = "./input_images/";
    string output_path = "./error_images/";
    string img_ext     = ".ppm";

    Image img = loadImage(input_path + img_name + img_ext);
    cout << "Immagine caricata: " << img.height << " x " << img.width << endl;

    // ============================
    // PARAMETRI RUMORE DA CODICE
    // ============================
    NoiseParams p;

    p.saltPepperPct = 30.0;   // 30% salt & pepper
    p.saltRatio     = 0.5;    // 50% sale, 50% pepe

    p.gaussianPct   = 100.0;  // gaussiano su tutti i pixel
    p.gaussianSigma = 6.0;    // sigma gaussiana

    p.specklePct    = 0.0;    // disattivato
    p.speckleSigma  = 0.03;

    p.rvImpulsePct  = 0.0;    // disattivato
    p.poissonPct    = 0.0;    // disattivato

    unsigned int seed = 12345; // per risultati ripetibili

    Image noisy = img;
    addNoise(noisy, p, seed);

    saveImage(output_path + img_name + img_ext, noisy);

    cout << "Immagine rumorosa salvata in: " << output_path + img_name << endl;
    return 0;
}
