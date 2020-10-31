import pycuda.driver as cuda # import PyCuda library
import pycuda.autoinit # instruct PyCuda library to implicitly perform the initialization of CUDA runtime
import scipy
from pycuda.compiler import SourceModule # this class is needed to compile CUDA kernels
import numpy as np # import Python module for working with numerical arrays
from time import perf_counter # import routines for doing time measurements
import cv2
import matplotlib.pyplot as plt # import pyplot to load and save the image
from scipy.ndimage import filters

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

# defining image file name
imageFileName = 'lena'

# loading image
image = cv2.imread(imageFileName + ".png")
height,width,channels = image.shape

# allocating memory on host for filtered results
image_filtered_gpu = np.zeros_like(image)

# Defining a template for CUDA kernel as a multiline string.
kernelCode = """
#define Z 2
#define Y 3
#define X 3
#define xBound X / 2
#define yBound Y / 2
#define SCALE 6  // strength of filter


#define T_WIDTH 16 // Tile width
#define M_WIDTH 3  // Mask width
#define M_RADIUS 1 // Mask radius = (M_WIDTH - 1) / 2
#define SM_WIDTH 18 // Shared Memory width = T_WIDTH + MASK_WIDTH - 1 
#define C_NUM 3 // Number of channels 

__constant__ int filter[Z][Y][X] = { { { 1, 0, -1 },
                                       { 2, 0, -2 },
                                       { 1, 0, -1 } },
                                     { { 1, 2, 1 },
                                       { 0, 0, 0 },
                                       { -1, -2, -1 } } }; // constant memory for sobel operators

inline __device__ int boundary_check(int val, int lower, int upper) {
  if (val >= lower && val < upper)
    return 1;
  else
    return 0;
} // To check if the pixel is in bounds of the actual image

__global__ void sobel_filter(unsigned char *s, unsigned char *t, unsigned height,
                      unsigned width, unsigned channels) {
  __shared__ unsigned char sm
      [C_NUM * SM_WIDTH * SM_WIDTH];

  for (int c = 0; c < C_NUM; ++c) {
  
    // ceil(SM_WIDTH^2 / T_WIDTH^2): number of times you need to load = 2(here)
    
    // First batch load
    int ID = threadIdx.y * T_WIDTH + threadIdx.x;
    int dest_y = ID / SM_WIDTH;
    int dest_x = ID % SM_WIDTH;
    
    int src_y = blockIdx.y * T_WIDTH + dest_y - M_RADIUS;
    int src_x = blockIdx.x * T_WIDTH + dest_x - M_RADIUS;
    
    int dest_index = (dest_y * SM_WIDTH + dest_x) * C_NUM + c;
    int src_index = (src_y * width + src_x) * C_NUM + c;

    if (dest_y < SM_WIDTH) {
      if (boundary_check(src_y, 0, height) && boundary_check(src_x, 0, width)) {
        sm[dest_index] = s[src_index];
      } else {
        sm[dest_index] = 0;
      }
    }

    // second batch load
    ID = threadIdx.y * T_WIDTH + threadIdx.x + T_WIDTH * T_WIDTH;
    dest_y = ID / SM_WIDTH;
    dest_x = ID % SM_WIDTH;
    
    src_y = blockIdx.y * T_WIDTH + dest_y - M_RADIUS;
    src_x = blockIdx.x * T_WIDTH + dest_x - M_RADIUS;
    
    dest_index = (dest_y * SM_WIDTH + dest_x) * C_NUM + c;
    src_index = (src_y * width + src_x) * C_NUM + c;

    if (dest_y < SM_WIDTH) {
      if (boundary_check(src_y, 0, height) && boundary_check(src_x, 0, width)) {
        sm[dest_index] = s[src_index];
      } else {
        sm[dest_index] = 0;
      }
    }
        
  }
  __syncthreads();

  float val[Z][3];

  int src_y = blockIdx.y * T_WIDTH + threadIdx.y;
  int src_x = blockIdx.x * T_WIDTH + threadIdx.x;
  int y = threadIdx.y;
  int x = threadIdx.x;

  {
    if (boundary_check(src_y, 0, height) && boundary_check(src_x, 0, width)) {
      /* Z axis of filter */
      for (int i = 0; i < Z; ++i) {
        val[i][2] = 0.;
        val[i][1] = 0.;
        val[i][0] = 0.;


        /* Y and X axis of filter */
        for (int v = 0; v < M_WIDTH; ++v) {
          for (int u = 0; u < M_WIDTH; ++u) {          
            {
              const unsigned char R =
                  sm[channels * (SM_WIDTH * (y + v) + (x + u)) + 2];
              const unsigned char G =
                  sm[channels * (SM_WIDTH * (y + v) + (x + u)) + 1];
              const unsigned char B =
                  sm[channels * (SM_WIDTH * (y + v) + (x + u)) + 0];
              val[i][2] += R * filter[i][u][v];
              val[i][1] += G * filter[i][u][v];
              val[i][0] += B * filter[i][u][v];
            }
          }
        }
      }

      float totalR = 0.;
      float totalG = 0.;
      float totalB = 0.;

      for (int i = 0; i < Z; ++i) {
        totalR += val[i][2] * val[i][2];
        totalG += val[i][1] * val[i][1];
        totalB += val[i][0] * val[i][0];
      }

      totalR = sqrt(totalR) / SCALE;
      totalG = sqrt(totalG) / SCALE;
      totalB = sqrt(totalB) / SCALE;
      
      // Thresholding the Sobel Gradients

      const unsigned char cR = (totalR > 255.) ? 255 : totalR;
      const unsigned char cG = (totalG > 255.) ? 255 : totalG;
      const unsigned char cB = (totalB > 255.) ? 255 : totalB;

      t[channels * (width * src_y + src_x) + 2] = cR;
      t[channels * (width * src_y + src_x) + 1] = cG;
      t[channels * (width * src_y + src_x) + 0] = cB;
    }
  }

}
"""

# compiling the CUDA code
mod = SourceModule(kernelCode)

# extracting the reference to kernel function
func = mod.get_function("sobel_filter")

# defining grid (we create 2D grid consisting of 2D blocks)
blockSize = 16 # T_WIDTH
block = (blockSize, blockSize, 1)
grid = (width // 16 + 1, height // 16 + 1, 1) #T_WIDTH = 16

# allocating memory and transfer data
gpu_t1_start = perf_counter()
image_gpu = cuda.mem_alloc(image.nbytes)
output_gpu = cuda.mem_alloc(image.nbytes)
cuda.memcpy_htod(image_gpu, image)
gpu_t1_end = perf_counter()

# filtering image on GPU
gpu_t2_start = perf_counter()
func(
    image_gpu,
    output_gpu,
    np.int32(height),
    np.int32(width),
    np.int32(channels),
    block=block,
    grid=grid)
cuda.Context.synchronize()
gpu_t2_end = perf_counter()

# transferring result back to host
gpu_t3_start = perf_counter()
cuda.memcpy_dtoh(image_filtered_gpu, output_gpu)
gpu_t3_end = perf_counter()

# saving resulting image (GPU) to file
plt.imsave(imageFileName + '_filtered_gpu.png', image_filtered_gpu)

# CPU Computation
cpu_t_start = perf_counter()
imx = np.zeros(image.shape)
imy = np.zeros(image.shape)
filters.sobel(image,1,imx,cval=0.0)  # axis 1 is x
filters.sobel(image,0,imy, cval=0.0) # axis 0 is y
magnitude = np.sqrt(imx**2+imy**2)
magnitude = scale(magnitude,0,1)
cpu_t_end = perf_counter()
plt.imsave(imageFileName + '_filtered_cpu.png', magnitude)

# Results
print('Data allocation and transfer time (milliseconds) GPU = ', (gpu_t1_end - gpu_t1_start) * 1000)
print('Computation time (milliseconds) GPU = ', (gpu_t2_end - gpu_t2_start) * 1000)
print('Result transfer time (milliseconds) GPU = ', (gpu_t3_end - gpu_t3_start) * 1000)
print('Total time (milliseconds) GPU = ', (gpu_t3_end - gpu_t1_start) * 1000)
print('Total time (milliseconds) CPU = ', (cpu_t_end - cpu_t_start) * 1000)



