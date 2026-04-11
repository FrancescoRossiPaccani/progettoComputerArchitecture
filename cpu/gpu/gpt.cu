#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// ================= IMAGE =================
struct Image {
    int w, h;
    vector<uint8_t> r, g, b;
    Image(int h_, int w_) : w(w_), h(h_), r(h_*w_), g(h_*w_), b(h_*w_) {}
};

// ================= LOAD =================
Image loadImage(const string& path) {
    ifstream f(path, ios::binary);

    string fmt;
    int w,h,maxv;
    f >> fmt >> w >> h >> maxv;
    f.get();

    Image img(h,w);

    vector<uint8_t> buf(w*h*3);
    f.read((char*)buf.data(), buf.size());

    for(int i=0;i<w*h;i++){
        img.r[i]=buf[i*3];
        img.g[i]=buf[i*3+1];
        img.b[i]=buf[i*3+2];
    }

    return img;
}

// ================= SAVE =================
void saveImage(const string& p, Image& img){
    ofstream f(p, ios::binary);
    f << "P6\n" << img.w << " " << img.h << "\n255\n";

    vector<uint8_t> buf(img.w*img.h*3);

    for(int i=0;i<img.w*img.h;i++){
        buf[i*3]=img.r[i];
        buf[i*3+1]=img.g[i];
        buf[i*3+2]=img.b[i];
    }

    f.write((char*)buf.data(), buf.size());
}

// ================= FAST DEVICE MEDIAN =================
__device__ __forceinline__ uint8_t median9(uint8_t v[9]){

    #define SWAP(a,b) { if(a>b){uint8_t t=a;a=b;b=t;} }

    // sorting network (fixed, branchless layout)
    SWAP(v[0],v[1]); SWAP(v[3],v[4]); SWAP(v[6],v[7]);
    SWAP(v[1],v[2]); SWAP(v[4],v[5]); SWAP(v[7],v[8]);
    SWAP(v[0],v[1]); SWAP(v[3],v[4]); SWAP(v[6],v[7]);

    SWAP(v[0],v[3]); SWAP(v[1],v[4]); SWAP(v[2],v[5]);
    SWAP(v[3],v[6]); SWAP(v[4],v[7]); SWAP(v[5],v[8]);

    SWAP(v[1],v[3]); SWAP(v[2],v[4]); SWAP(v[5],v[7]);
    SWAP(v[2],v[3]); SWAP(v[5],v[6]);

    SWAP(v[2],v[5]);

    return v[4]; // median
}

// ================= KERNEL (SHARED MEMORY TILE) =================
__global__ void median3x3(
    const uchar3* __restrict__ in,
    uchar3* __restrict__ out,
    int w, int h)
{
    __shared__ uchar3 tile[18][18]; // 16 + halo

    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // load center
    if(x < w && y < h)
        tile[ly][lx] = in[y*w + x];

    // halo
    if(threadIdx.x == 0 && x > 0)
        tile[ly][0] = in[y*w + x - 1];

    if(threadIdx.x == 15 && x < w-1)
        tile[ly][17] = in[y*w + x + 1];

    if(threadIdx.y == 0 && y > 0)
        tile[0][lx] = in[(y-1)*w + x];

    if(threadIdx.y == 15 && y < h-1)
        tile[17][lx] = in[(y+1)*w + x];

    __syncthreads();

    if(x >= w || y >= h) return;

    uint8_t r[9], g[9], b[9];
    int k=0;

    #pragma unroll
    for(int dy=-1;dy<=1;dy++){
        #pragma unroll
        for(int dx=-1;dx<=1;dx++){
            uchar3 p = tile[ly+dy][lx+dx];
            r[k]=p.x;
            g[k]=p.y;
            b[k]=p.z;
            k++;
        }
    }

    out[y*w + x] = make_uchar3(
        median9(r),
        median9(g),
        median9(b)
    );
}

// ================= MAIN =================
int main(int argc, char** argv){

    if(argc < 2){
        cout << "use: ./prog img\n";
        return 0;
    }

    string name = argv[1];

    Image img = loadImage("../error_images/"+name+".ppm");
    Image out(img.h, img.w);

    int N = img.w * img.h;

    // PACK RGB -> uchar3
    vector<uchar3> in(N);

    for(int i=0;i<N;i++)
        in[i] = make_uchar3(img.r[i], img.g[i], img.b[i]);

    uchar3 *d_in, *d_out;

    cudaMalloc(&d_in, N*sizeof(uchar3));
    cudaMalloc(&d_out, N*sizeof(uchar3));

    cudaMemcpy(d_in, in.data(), N*sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((img.w+15)/16, (img.h+15)/16);


    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start GPU timer
    cudaEventRecord(start);

    // KERNEL
    median3x3<<<grid, block>>>(d_in, d_out, w, h);

    // stop GPU timer
    cudaEventRecord(stop);

    // aspetta fine kernel
    cudaEventSynchronize(stop);

    // calcolo tempo
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cout << "Kernel time: " << ms << " ms\n";

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    vector<uchar3> outbuf(N);
    cudaMemcpy(outbuf.data(), d_out, N*sizeof(uchar3), cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++){
        out.r[i]=outbuf[i].x;
        out.g[i]=outbuf[i].y;
        out.b[i]=outbuf[i].z;
    }

    cout << "Time: "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
         << " ms\n";

    saveImage("./out.ppm", out);

    cudaFree(d_in);
    cudaFree(d_out);
}