#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

using namespace std;
using namespace cv;

#define FILTER_WIDTH 5
#define FILTER_RADIUS FILTER_WIDTH / 2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + FILTER_WIDTH - 1)

// Conv2D_CPU applies a 2D convolution on an RGB image and runs on the CPU
void Conv2D_CPU(unsigned char* outImg, unsigned char* inImg, float* filter, int numRows, int numCols, int numChans) {
    float cumSum;
    int cornerRow, cornerCol;
    int filterRow, filterCol;

    // loop through the pixels in the output image
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            // compute coordinates of top-left corner
            cornerRow = row - FILTER_RADIUS;
            cornerCol = col - FILTER_RADIUS;

            // loop through the channels
            for (int c = 0; c < numChans; c++) {
                // reset accumulator
                cumSum = 0;

                // accumulate values inside filter
                for (int i = 0; i < FILTER_WIDTH; i++) {
                    for (int j = 0; j < FILTER_WIDTH; j++) {
                        // compute pixel coordinates inside filter
                        filterRow = cornerRow + i;
                        filterCol = cornerCol + j;

                        // make sure we are within image boundaries
                        if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
                            cumSum += inImg[(filterRow*numCols + filterCol)*numChans + c] * filter[i*FILTER_WIDTH + j];
                        }
                    }
                }
                outImg[(row*numCols + col)*numChans + c] = (unsigned char)cumSum;
            }
        }
    }
}


int main() {
 
Mat img;
vector<String> fn;
glob("jpg/*.jpg", fn, false);
size_t count = fn.size();
cout << "\nTotal images : " << count << endl ;
clock_t start,stop;
int tempcount = 1200 ; 

for (int r=0; r<tempcount; r = r+100){

/////////////////////////////////////////////////////////

    start=clock();
    for (size_t i=0; i<r; i++){

        // for testing taken the image barbara
        //Mat img = imread("barbara.jpg",CV_LOAD_IMAGE_COLOR) ;
        
        Mat img = imread(fn[i],CV_LOAD_IMAGE_COLOR) ;

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

        Conv2D_CPU(h_outImg_CPU, h_inImg, h_filter, imgWidth, imgHeight, imgChans);
        
        // display images
        Mat img2(imgHeight, imgWidth, CV_8UC3, h_outImg_CPU);
        
        //cout<< "\n" << fn[i];
        //imwrite("cpuimg.jpg", img2);

    }

    stop=clock();
    printf("\ntotal count  %d ",r+100);

    printf("\n\nTime taken in -- CPU Sequential --  for images convolution %lf\n", (double)(stop-start)/CLOCKS_PER_SEC);
}
    return 0;
}

/*  

g++ conv10_25.cpp -o conv10_25cpp.out `pkg-config --cflags --libs opencv`
./conv10_25cpp.out

*/
