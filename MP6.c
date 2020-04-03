// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

//Cast the image from float to unsigned char
__global__ void floatToUChar(float* input, unsigned char* output, int iw, int ih, int ic) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  if (i < iw * ih * ic) {
    output[i] = (unsigned char) (255 * input[i]);
  }
}

//Convert the image from RGB to GrayScale
__global__ void RGBtoGray(unsigned char* input, unsigned char* output, int iw, int ih) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned char r, g, b;
  if (i < iw * ih) {
    r = input[3 * i];
    g = input[3 * i + 1];
    b = input[3 * i + 2];
    
    output[i] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  //Define the variables on the device
  float* deviceInputImageData;
  float* deviceOutputImageData;
  unsigned char* deviceUCharImageData;
  
  //Define the dimension of the block and the grid
  dim3 dimGrid((ceil(imageWidth * imageHeight * inmageChannels) / HISTOGRAM_LENGTH), 1, 1);
  dim3 dimBlock(HISTOGRAM_LENGTH);
  
  //Memory allocation in the device
  cudaMalloc((void**) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceUCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  //Process the image
  floatToUChar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceUCharImageData, imageWidth, imageHeight, imageChannels);

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
