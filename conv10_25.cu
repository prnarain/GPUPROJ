#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>

using namespace std;
using namespace cv;

#define FILTER_WIDTH 5
#define FILTER_RADIUS FILTER_WIDTH / 2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + FILTER_WIDTH - 1)


// Conv2D_GPU applies a 2D convolution on an RGB image and runs on the GPU
__global__ void Conv2D_GPU(unsigned char* outImg, unsigned char* inImg, const float* __restrict__ filter, int numRows, int numCols, int numChans) {
    int filterRow, filterCol;
    int cornerRow, cornerCol;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int bdx = blockDim.x; int bdy = blockDim.y;

    // compute global thread coordinates in output image
    int row = by * (bdy - 2*FILTER_RADIUS) + ty;
    int col = bx * (bdx - 2*FILTER_RADIUS) + tx;

    // make sure thread is within augmented boundaries
    if ((row < numRows + FILTER_RADIUS) && (col < numCols + FILTER_RADIUS)) {
        // allocate a 2D chunk of shared memory
        __shared__ unsigned char chunk[BLOCK_WIDTH][BLOCK_WIDTH];

        // loop through the channels
        for (int c = 0; c < numChans; c++) {
            // load into shared memory
            int relativeRow = row - FILTER_RADIUS;
            int relativeCol = col - FILTER_RADIUS;
            if ((relativeRow < numRows) && (relativeCol < numCols) && (relativeRow >= 0) && (relativeCol >= 0)) {
                chunk[ty][tx] = inImg[(relativeRow*numCols + relativeCol)*numChans + c];
            }
            else {
                chunk[ty][tx] = 0;
            }

            // ensure all threads have loaded to SM
            __syncthreads();

            // instantiate accumulator
            float cumSum = 0;

            // only a subset of threads in block need to do computation
            if ((tx >= FILTER_RADIUS) && (ty >= FILTER_RADIUS) && (ty < bdy - FILTER_RADIUS) && (tx < bdx - FILTER_RADIUS)) {
                // top-left corner coordinates
                cornerRow = ty - FILTER_RADIUS;
                cornerCol = tx - FILTER_RADIUS;

                for (int i = 0; i < FILTER_WIDTH; i++) {
                    for (int j = 0; j < FILTER_WIDTH; j++) {
                        // filter coordinates
                        filterRow = cornerRow + i;
                        filterCol = cornerCol + j;

                        // accumulate sum
                        if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
                            cumSum += chunk[filterRow][filterCol] * filter[i*FILTER_WIDTH + j];
                        }
                    }
                }
                // write to global memory
                outImg[(relativeRow*numCols + relativeCol)*numChans + c] = (unsigned char)cumSum;
            }
        }
    }
}

int main() {
 
cv::Mat img;
vector<cv::String> fn;
glob("jpg/*.jpg", fn, false);
size_t count = fn.size();
cout << "\nTotal images : " << count << endl ;
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
    
float milliseconds = 0;
float total =0 ; int tempcount = 1200 ; 

for (int r=0; r<tempcount; r = r+100){
/////////////////////////////////////////////////////////

    for (size_t i=0; i<r; i++){

        // for testing taken the image barbara
        //img = imread("barbara.jpg", CV_LOAD_IMAGE_COLOR);

        img = imread(fn[i], CV_LOAD_IMAGE_COLOR);

        unsigned char* h_inImg = img.data;

        // grab image dimensions
        int imgChans = img.channels();
        int imgWidth = img.cols;
        int imgHeight = img.rows;

        // useful params
        size_t imgSize = sizeof(unsigned char)*imgWidth*imgHeight*imgChans;
        size_t filterSize = sizeof(float)*FILTER_WIDTH*FILTER_WIDTH;
        //GpuTimer timer0, timer1, timer2, timer3;

        // allocate host memory
        float* h_filter = (float*)malloc(filterSize);
        unsigned char* h_outImg = (unsigned char*)malloc(imgSize);
        unsigned char* h_outImg_CPU = (unsigned char*)malloc(imgSize);

        // hardcoded filter values
        float filter[FILTER_WIDTH*FILTER_WIDTH] = {
            1/273.0, 4/273.0, 7/273.0, 4/273.0, 1/273.0,
            4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0,
            7/273.0, 26/273.0, 41/273.0, 26/273.0, 7/273.0,
            4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0,
            1/273.0, 4/273.0, 7/273.0, 4/273.0, 1/273.0
        };
        h_filter = filter;

        // allocate device memory
        float* d_filter;
        unsigned char* d_inImg;
        unsigned char* d_outImg;
        //timer0.Start();
        cudaMalloc((void**)&d_filter, filterSize);
        cudaMalloc((void**)&d_inImg, imgSize);
        cudaMalloc((void**)&d_outImg, imgSize);
        
        // host2device transfer
        cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_inImg, h_inImg, imgSize, cudaMemcpyHostToDevice);
        
        // kernel launch
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
        dim3 dimGrid(ceil(imgWidth/(float)O_TILE_WIDTH), ceil(imgHeight/(float)O_TILE_WIDTH), 1);
        
        cudaEventRecord(start);
        Conv2D_GPU<<<dimGrid, dimBlock>>>(d_outImg, d_inImg, d_filter, imgWidth, imgHeight, imgChans);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        total = total + milliseconds; 

        // device2host transfer
        cudaMemcpy(h_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost);
        
        // display images
        Mat img1(imgHeight, imgWidth, CV_8UC3, h_outImg);
        Mat img2(imgHeight, imgWidth, CV_8UC3, h_outImg_CPU);
        
        
        //cout<< "\n" << fn[i];

    }
        printf("\ntotal count  %d ",r+100);
        printf("\nTotal time taken in -- GPU --  for images convolution :  %lf\n", total/1000);
        total = 0 ; milliseconds = 0 ;
}
    return 0;
}

/*  

nvcc conv10_25.cu -o conv10_25cu.out `pkg-config --cflags --libs opencv`
./conv10_25cu.out

*/