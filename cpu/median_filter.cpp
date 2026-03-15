#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>
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
    //overload per lettura su const Image
    int operator()(int y,int x,int c) const
    {
        return data[(y*width+x)*3+c];
    }
};

/**
 * @brief Carica un immagine
 * 
 * @param path path dell'immagine da caricare
 * @return immgagine in formato Image
 */
Image loadImage(const string& path){
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
/**
 * @brief Carica un immagine
 * 
 * @param path path dove salvare l'immgine
 * @param img  immagine da salvare
 */
void saveImage(const string& path, Image& img){
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

/**
 * @brief Salva le statisctiche in un file
 * 
 * @param filename file contenente le statistiche
 * @param img_name nome immagine
 * @param width larghezza immagine
 * @param height altezza immagine
 * @param n_thread numero di thread 
 * @param time_ms tempo di esecuzione [ms]
 */
void logPerformance(const string& filename, const string& img_name, int width, int height, int n_threads, long long time_ms){
    ofstream logFile(filename, ios::app); // apertura in append
    if(!logFile){
        cerr << "Errore apertura file log\n";
        return;
    }

    // Scrive una riga con: immagine, dimensioni, thread, tempo
    logFile << img_name << "," << width << "x" << height << "," << n_threads << "," << time_ms << "ms" << endl;
    logFile.close();
}

/**
 * @brief Limita un valore tra 0 e 255
 * 
 * @param v valore da limitare
 * @return int valore clampato
 */
inline int clamp(int v){
    return min(max(v,0),255);
}

/**
 * @brief Estrae un blocco MxM di pixel intorno a (y,x) per il canale c
 * 
 * @param img immagine originale
 * @param y coordinata verticale
 * @param x coordinata orizzontale
 * @param c canale (0=R,1=G,2=B)
 * @param M dimensione del kernel
 * @return vector<int> blocco di valori
 */
vector<int> getNeighborhood(const Image& img, int y, int x, int c, int M){
    int radius = M/2;
    vector<int> block;
    block.reserve(M*M);

    for(int fy=-radius; fy<=radius; fy++)
        for(int fx=-radius; fx<=radius; fx++)
        {
            int iy = min(max(y+fy,0), img.height-1);
            int ix = min(max(x+fx,0), img.width-1);
            block.push_back(img(iy,ix,c));
        }

    return block;
}

/**
 * @brief Calcola la mediana di un blocco di valori
 * 
 * @param block vettore di interi
 * @return int valore mediano
 */
int medianOfBlock(vector<int>& block){
    sort(block.begin(), block.end());
    if (block.size()% 2 == 0)
        return block[(block.size()/2)-1] + block[block.size()/2];
    else
        return block[block.size()/2];
}

/**
 * @brief Applica il filtro selettivo a un singolo pixel
 * 
 * @param img immagine originale
 * @param y coordinata verticale
 * @param x coordinata orizzontale
 * @param c canale (0=R,1=G,2=B)
 * @param M dimensione kernel
 * @param threshold soglia per sostituire il pixel con la mediana
 * @return int valore filtrato
 */
int selectiveMedianPixel(const Image& img, int y, int x, int c, int M, int threshold){
    vector<int> block = getNeighborhood(img,y,x,c,M);
    int med = medianOfBlock(block);
    int center = img(y,x,c);

    if(abs(center - med) > threshold)
        return med;
    else
        return center;
}

/**
 * @brief Worker del filtro selettivo, eseguito da un thread
 * 
 * @param img immagine originale
 * @param out immagine filtrata (output)
 * @param M dimensione del kernel
 * @param threshold soglia per sostituire pixel
 * @param counter contatore condiviso delle righe processate
 * @param mtx mutex per proteggere il contatore
 */
void selectiveMedianWorker(Image& img, Image& out, int M, int threshold, int& counter, mutex& mtx){
    while(true){
        mtx.lock();
        int y = counter++;
        mtx.unlock();

        if(y >= img.height) break;

        for(int x=0;x<img.width;x++)
            for(int c=0;c<3;c++)
                out(y,x,c) = selectiveMedianPixel(img,y,x,c,M,threshold);
    }
}

// ============================================
// =              Main                        =
// ============================================

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        cout << "Uso: " << argv[0] << " nome_immagine "<< "num_thread\n";
        return 1;
    }

    string img_name = argv[1];
    string img_ext  = ".ppm";
    string input    = "./error_images/";
    string output   = "./output_images/";

    Image img = loadImage(input + img_name + img_ext);
    cout << "Immagine caricata " << img.height << "x" << img.width << endl;

    //Setup filtro 
    int kernel_size=7;
    int threshold=40;

    //N threads
    int n_threads = stoi(argv[2]);

    Image out(img.height,img.width);

    mutex mtx;
    int counter=0;

    //inizio misurazione
    auto start = chrono::high_resolution_clock::now();

    vector<thread> threads;

    for(int i=0;i<n_threads;i++)
    {
        threads.emplace_back(selectiveMedianWorker, ref(img), ref(out), kernel_size, threshold, ref(counter), ref(mtx));
    }

    for(auto& t:threads)
        t.join();

    //fine misurazione
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    logPerformance("log.txt", img_name + img_ext, img.width, img.height, n_threads, duration.count());


    saveImage(output + img_name + img_ext,out);
    cout << "Immagine salvata in " << output << endl;
    cout << "Tempo impiegato: " << duration.count() << " ms" << endl << endl; 
    return 0;
}