#include "kernel.h"
#include <curand.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include <cstdio>

__host__ void cudaErrorCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
}

__device__ float4 pickRandomFloat4(curandState* randState)
{
    return make_float4(curand_uniform(randState),
        curand_uniform(randState),
        curand_uniform(randState), 1);
}

surface<void, cudaSurfaceType2D> screen_surface;

__global__ void visualizeDeviceBlocks(dim3 screen_size)
{
    // Picks a random color for each kernel block on the screen

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= screen_size.x || j >= screen_size.y)
        return; // Out of texture bounds

    const auto threadId = blockIdx.x * blockDim.y + blockIdx.y;

    static int count = 0;

    curandState randState;
    curand_init(threadId + count, 0, 0, &randState);
    if (i == 0 && j == 0) { count++; }

    surf2Dwrite(pickRandomFloat4(&randState), screen_surface,
                i * sizeof(float4), j);
}

void kernelCall(const unsigned int width, const unsigned int height,
                cudaArray * graphics_array)
{
    dim3 blockDim(16, 16, 1);
    dim3  gridDim((width  - 1)/blockDim.x + 1,
                  (height - 1)/blockDim.y + 1, 1);

    cudaErrorCheck(cudaBindSurfaceToArray(
        screen_surface, graphics_array)
    );

    visualizeDeviceBlocks<<<gridDim, blockDim>>>(dim3(width, height));

    // Checking failures on kernel
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "CUDA kernel failed: %s\n",
                cudaGetErrorString(cudaStatus));

    cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void FillVBOKernel(float* vbo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 2; ++j) {
        vbo[3 * i + j] *= 0.8;
    }
}

void FillVBO(float *vbo) {
    FillVBOKernel << <dim3(1,1,1), dim3(6,1,1) >> > (vbo);
}
