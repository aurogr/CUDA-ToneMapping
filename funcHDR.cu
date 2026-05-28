#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include "timer.h"

const int BLOCK_SIZE = 1024;

__global__ void reduce(float* input, float* output, int len, bool isMin) {
    // shared memory
    extern __shared__ float sdata[];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load values into shared memory
    if (global_idx >= len) {
        sdata[threadIdx.x] = input[0]; // dummy innit
    }
    else {
        sdata[threadIdx.x] = input[global_idx];
    }
    __syncthreads();

    // reduce inside shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = isMin ? fminf(sdata[threadIdx.x], sdata[threadIdx.x + s]) : fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // return the block min/max
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void histo(const float* input, unsigned int* histo, int len, float lumMin, float lumRange, int numBins)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < len) {
        int bin = (input[global_idx] - lumMin) / lumRange * numBins;
        bin = fminf(numBins - 1, fmaxf(0, bin)); // clamp
        atomicAdd(&(histo[bin]), 1);  // Varios threads podrían intentar incrementar el mismo valor a  la vez
    }
}

__global__ void exclusive_scan(unsigned int* output, const unsigned int* input, int texSize)
{
    extern __shared__ unsigned int tempArray[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;
    int offset = 1, temp;
    int ai = threadId;
    int bi = threadId + texSize / 2;
    int i;
    //assign the shared memory
    tempArray[ai] = input[id];
    tempArray[bi] = input[id + texSize / 2];
    //up tree
    for (i = texSize >> 1; i > 0; i >>= 1)
    {
        __syncthreads();
        if (threadId < i)
        {
            ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            tempArray[bi] += tempArray[ai];
        }
        offset <<= 1;
    }
    //put the last one 0
    if (threadId == 0)
        tempArray[texSize - 1] = 0;
    //down tree
    for (i = 1; i < texSize; i <<= 1) // traverse down tree & build scan  
    {
        offset >>= 1;
        __syncthreads();
        if (threadId < i)
        {
            ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            temp = tempArray[ai];
            tempArray[ai] = tempArray[bi];
            tempArray[bi] += temp;
        }
    }
    __syncthreads();
    output[id] = tempArray[threadId];
    output[id + texSize / 2] = tempArray[threadId + texSize / 2];
}



void calculate_cdf(const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float& min_logLum,
    float& max_logLum,
    const size_t numRows,
    const size_t numCols,
    const size_t numBins)
{
    /* TODO
      1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance
      2) Obtener el rango a representar
      3) Generar un histograma de todos los valores del canal logLuminance usando la formula
      bin = (Lum [i] - lumMin) / lumRange * numBins
      4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
      de los valores de luminancia. Se debe almacenar en el puntero c_cdf
    */
    size_t numPixels = numRows * numCols;

    GpuTimer timer;

    // 0) configure threads and blocks
    int threads = BLOCK_SIZE;
    int blocks = (numPixels + threads - 1) / threads;

    // 1) Find max and min luminance values from const float* const d_logLuminance (reduce) 
    timer.Start();
    // 1.1. declare and allocate GPU memory
    float* d_blockMax;
    float* d_blockMin;

    cudaMalloc(&d_blockMax, blocks * sizeof(float));
    cudaMalloc(&d_blockMin, blocks * sizeof(float));

    // 1.2. get maximum and minimum of every block
    reduce << <blocks, threads, threads * sizeof(float)>> > (
        (float*)d_logLuminance, d_blockMax, numPixels, false);

    reduce << <blocks, threads, threads * sizeof(float)>> > (
        (float*)d_logLuminance, d_blockMin, numPixels, true);

    // 1.3. run again for the blocks, so it gets global maximum and minimum
    int remaining = blocks;

    while (remaining > 1) {
        int newBlocks = (remaining + threads - 1) / threads;

        reduce << <newBlocks, threads, threads * sizeof(float) >> > (
            d_blockMax, d_blockMax, remaining, false);

        reduce << <newBlocks, threads, threads * sizeof(float)>> > (
            d_blockMin, d_blockMin, remaining, true);

        remaining = newBlocks;
    }

    // 1.4. store in float& min_logLum, float& max_logLum,
    cudaMemcpy(&max_logLum, d_blockMax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_logLum, d_blockMin, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Min: " << min_logLum << "\n";
    std::cout << "Max: " << max_logLum << "\n";

    timer.Stop();
    printf("Reduce ran in: %f msecs.\n", timer.Elapsed());

    // 2) Obtain range to represent (histogram)
    float lumRange = max_logLum - min_logLum;

    // 3) Obtain histogram using bin = (Lum [i] - lumMin) / lumRange * numBins
    timer.Start();
    // 3.1 declare and allocate GPU memory
    unsigned int* d_histo;
    cudaMalloc(&d_histo, numBins * sizeof(unsigned int));
    cudaMemset(d_histo, 0, numBins * sizeof(unsigned int));

    // 3.2 launch kernel
    histo << <blocks, threads>> > ((float*)d_logLuminance, d_histo, numPixels, min_logLum, lumRange, numBins);


    timer.Stop();
    printf("Histogram ran in: %f msecs.\n", timer.Elapsed());

    // 4) Exclusive scan to obtain cdf
    timer.Start();
    exclusive_scan << <1, numBins, 2 * numBins * sizeof(unsigned int) >> > (d_cdf, d_histo, numBins);

    timer.Stop();
    printf("Exclusive scan ran in: %f msecs.\n", timer.Elapsed());
}

