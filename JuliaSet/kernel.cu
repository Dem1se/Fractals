
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int DIM = 4096 * 2;

struct Complex {
    float r, i;
};

__device__ float magnitude(Complex a)
{
    return ((a.r * a.r) + (a.i * a.i));
}

__device__ void add(Complex a, Complex b, Complex* res)
{
    res->r = a.r + b.r;
    res->i = a.i + b.i;
}

__device__ void mul(Complex a, Complex b, Complex* res)
{
    res->r = (a.r * b.r) - (a.i * b.i);
    res->i = (a.r * b.i) + (a.i * b.r);
}

// Calculates the given point's membership in the julia set. 1 or 0
__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    Complex r1, r2;
    Complex c = { -0.8, 0.154 };
    Complex a = { jx, jy };
    for (int i = 0; i < 200; i++)
    {
        mul(a, a, &r1);
        add(r1, c, &r2);
        if (magnitude(r2) > 1000)
            return 0;
        a.r = r2.r;
        a.i = r2.i;
    }
    return 1;
}

__global__ void kernel(unsigned char* devImg)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int index = x + (y * DIM);
    
    int juliaVal = julia(x, y);
    devImg[index * 3 + 0] = 255 * juliaVal;
    devImg[index * 3 + 1] = 0;
    devImg[index * 3 + 2] = 0;
}


int main()
{
    cudaError_t cuErr;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t dataSize = sizeof(unsigned char) * 3 * DIM * DIM;
    unsigned char* img = nullptr;
    unsigned char* devImg = nullptr;
    cudaMallocHost((unsigned char**)&img, dataSize);

    cuErr = cudaSetDevice(0);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cuErr = cudaMalloc((void**)&devImg, dataSize);
    if (cuErr != cudaSuccess)
    {
        fprintf(stderr, "Error allocating device memory.\n");
        goto Error;
    }

    printf("Allocated device buffer\n");
    cudaEventRecord(start);
    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(devImg);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel completed, %.2f ms\n", ms);

    cuErr = cudaGetLastError();
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cuErr));
        goto Error;
    }

    cuErr = cudaDeviceSynchronize();
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cuErr);
        goto Error;
    }

    cudaEventRecord(start);
    cuErr = cudaMemcpy(img, devImg, dataSize, cudaMemcpyDeviceToHost);
    if (cuErr != cudaSuccess)
    {
        fprintf(stderr, "Error copying data from device to host.\n");
        goto Error;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Copied back from device buffer. %.2f ms\n", ms);

    int stbErr = stbi_write_bmp("julia.bmp", DIM, DIM, 3, img);
    if (stbErr == 0)
    {
        fprintf(stderr, "Error writing BMP with STB library.\n");
        goto Error;
    }

    printf("Wrote image data to file\n");

    cuErr = cudaDeviceReset();
    if (cuErr != cudaSuccess)
    {
        fprintf(stderr, "Error resetting device!\n");
        goto Error;
    }

    cudaFree(devImg);
    cudaFreeHost(img);
    return 0;
Error:
    cudaFree(devImg);
    cudaFreeHost(img);
    return -3;
}
