// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

/*Cast the image from float to unsigned char*/
__global__ void floatToUChar(float* input, unsigned char* output, int iw, int ih, int ic) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  if (i < iw * ih * ic) {
    output[i] = (unsigned char) ((HISTOGRAM_LENGTH - 1) * input[i]);
  }
}

/*Convert the image from RGB to GrayScale*/
__global__ void RGBtoGray(unsigned char* input, unsigned char* output, int iw, int ih) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
    
  if (i < iw * ih) {
    unsigned char r, g, b;
    r = input[3 * i];
    g = input[3 * i + 1];
    b = input[3 * i + 2];
    
    output[i] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

/*Compute the Cumulative Distribution Function of histogram*/
__global__ void grayToHisto(unsigned char* input, unsigned int* output, int iw, int ih) {
  __shared__ unsigned int histoPrivate[HISTOGRAM_LENGTH];
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histoPrivate[threadIdx.x] = 0;
  }
  __syncthreads();
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  while (i < iw * ih) {
    atomicAdd(&(histoPrivate[input[i]]), 1);
    i += stride;
  }
  __syncthreads();
  
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[threadIdx.x]), histoPrivate[threadIdx.x]);
  }
}

/*Compute the Cumulative Distribution Function of histogram*/
__global__ void histoToCDF(unsigned int* input, float* output, int iw, int ih) {
  __shared__ float T[HISTOGRAM_LENGTH];
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  //**Perform the scan**//
  if (i < HISTOGRAM_LENGTH) {
    T[threadIdx.x] = (float) input[i];
  }
  
  /*Reduction kernel*/
  int stride = 1;
  while (stride < HISTOGRAM_LENGTH) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    
    if (index < HISTOGRAM_LENGTH && (index - stride) >= 0) {
      T[index] += T[index - stride];
    }
    
    stride *= 2;
  }
  
  /*Post scan*/
  stride = HISTOGRAM_LENGTH / 4;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < HISTOGRAM_LENGTH) {
      T[index + stride] += T[index];
    }
    
    stride /= 2;
  }
  
  __syncthreads();
  
  if (i < HISTOGRAM_LENGTH) {
    output[i] = T[threadIdx.x] / (1.0 * iw * ih);
  }
}

/*Compute from the CDF to the output image*/
__global__ void CDFtoToOut(float* CDF, unsigned char* input, float* output, int iw, int ih, int ic) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < iw * ih * ic) {
    output[i] = (float) min(max(255 * (CDF[input[i]] - CDF[0]) / (1.0 - CDF[0]) / 255.0, 0.0), 255.0);
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
  /*Define the variables on the device*/
  float* deviceInputImageData;
  float* deviceOutputImageData;
  unsigned char* deviceUCharImageData;
  unsigned char* deviceGrayData;
  unsigned int* deviceHisto;
  float* deviceCDF;

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
  /*Get the input and the output images*/
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  
  /*Define the dimension of the block and the grid*/
  dim3 dimGrid(ceil(imageWidth * imageHeight * imageChannels * 1.0 / HISTOGRAM_LENGTH), 1, 1);
  dim3 dimBlock(HISTOGRAM_LENGTH, 1, 1);
  
  /*Memory allocation in the device*/
  cudaMalloc((void**) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceUCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &deviceGrayData, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  
  /*Memory copy from the host to the device*/
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  
  /*Memory set in the device*/
  cudaMemset((void*) deviceHisto, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void*) deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float));
  
  /*Process the image*/
  floatToUChar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceUCharImageData, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  RGBtoGray<<<dimGrid, dimBlock>>>(deviceUCharImageData, deviceGrayData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  grayToHisto<<<dimGrid, dimBlock>>>(deviceGrayData, deviceHisto, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  histoToCDF<<<dimGrid, dimBlock>>>(deviceHisto, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  CDFtoToOut<<<dimGrid, dimBlock>>>(deviceCDF, deviceUCharImageData, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
  /*Memory copy from device to host*/
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  /*Free the memory in the device*/
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceUCharImageData);
  cudaFree(deviceGrayData);
  cudaFree(deviceHisto);
  cudaFree(deviceCDF);
  
  /*Free the memory in the host*/
  free(hostInputImageData);
  free(hostOutputImageData);
  //free(inputImageFile);

  return 0;
}
