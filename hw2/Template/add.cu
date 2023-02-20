#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add_serial(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

__global__
void add_parallel(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int run_in_parallel = 1;
  int N = 1<<30;
  float *x, *y;
  int blockSize;
  int numBlocks;
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU

  // simple serial with 1 thread block

  if( run_in_parallel == 0 )
   add_serial<<<1, 1>>>(N, x, y);

  else{
  // add_parallel code with multiple threads and blocks
  blockSize = 2048;
  numBlocks = (N+ blockSize - 1) / blockSize;
  printf("Invoking add_parallel with numBlocks = %d and blockSize = %d\n", numBlocks, blockSize);
  add_parallel<<<numBlocks, blockSize>>>(N, x, y);
  }
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}

