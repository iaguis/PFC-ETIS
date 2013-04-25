#ifndef _REDUCTION_GPU_H
#define	_REDUCTION_GPU_H

#include "Cuda_runtime_error.h"

#include "common.h"
#include "Reduction_op.h"
#include "Timer.h"

#include <cuda_runtime.h>

#include <iostream>
#include <math.h>

/*
 * Each instance of this class executes the code in transform in its own thread.
 * */
template <typename T> class Reduction_GPU {
public:
    Reduction_GPU(T* const result, const T *const vector, const int size, Reduction_op<T> *kernel, int nGPU)
        : m_Thread(boost::thread(&Reduction_GPU::reduction, this, result, vector, size, kernel, nGPU)), clean_(true),err_(cudaSuccess)
    {

    }

    ~Reduction_GPU()
    {
        m_Thread.join();
        if (!clean_)
            throw err_;
    }
    void reduction(T* const result, const T* const vector, const int size, Reduction_op<T> *kernel, int nGPU) throw (Cuda_runtime_error);
private:
    void reduce(const int size, const int threads, const int blocks, const T* const in, T* const out, const char *kernel) throw (Cuda_runtime_error);
    int nextPow2(int n);
    boost::thread m_Thread;
    bool clean_;
    Cuda_runtime_error err_;
};

template <typename T> int Reduction_GPU<T>::nextPow2(int n)
{
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;

    return ++n;
}

template <typename T> void Reduction_GPU<T>::reduce (const int size, const int threads, const int blocks, const T* const in, T* const out, const char *kernel)
throw (Cuda_runtime_error)
{
    int blocks1, blocks2;
    if (blocks > 65535) {
        blocks1 = 65535;
        blocks2 = ceil((double)blocks/65535);
    }
    else {
        blocks1 = blocks;
        blocks2 = 1;
    }

    dim3 dimGrid(blocks1,blocks2);

    if (cudaConfigureCall(dimGrid,threads,(size_t) (threads * sizeof(T)),0) != cudaSuccess)
        throw Cuda_runtime_error(cudaGetLastError());

    int offset = 0;

    // Alignment
    void *ptr;

    ptr = (void *)(size_t) in;
    ALIGN_UP(offset, __alignof(ptr));

    if (cudaSetupArgument(&in, sizeof(ptr), offset) != cudaSuccess)
        throw Cuda_runtime_error(cudaGetLastError());

    offset += sizeof(ptr);

    ptr = (void *)(size_t) out;
    ALIGN_UP(offset, __alignof(ptr));

    if (cudaSetupArgument(&out, sizeof(ptr), offset) != cudaSuccess)
        throw Cuda_runtime_error(cudaGetLastError());

    offset += sizeof(ptr);

    if (cudaSetupArgument(&size, sizeof(T), offset) != cudaSuccess)
        throw Cuda_runtime_error(cudaGetLastError());

    // Checks type to select appropriate kernel
    char final_gpu_kernel[strlen(kernel) + 2];

    strcpy(final_gpu_kernel,kernel);

    strcat(final_gpu_kernel,"_");
    strcat(final_gpu_kernel,typeid(T).name());

    if (cudaLaunch(final_gpu_kernel) != cudaSuccess)
        throw Cuda_runtime_error(cudaGetLastError());
}

template <typename T> void Reduction_GPU<T>::reduction(T* const result, const T* const vector, const int size, Reduction_op<T> *kernel, int nGPU)
throw (Cuda_runtime_error)
{
    try {
        struct timeval start, stop;

        Timer::timestamp(&start);

        T *g_vector, *g_res;
        unsigned int numBytes = size * sizeof(T);
        int threads, blocks;

        if (cudaSetDevice(nGPU) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        threads = threadsPerBlock;
        blocks = ceil((double) size / threadsPerBlock);

        if (cudaMalloc((void **)&g_vector, numBytes) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        if (cudaMalloc((void **)&g_res, blocks * sizeof(T)) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        if (cudaMemcpy(g_vector, vector, numBytes, cudaMemcpyHostToDevice) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        reduce(size,threads,blocks,g_vector,g_res,kernel->getName());

        int size2 = blocks;

         while (size2 > 1) {
            int threads = (size2 < threadsPerBlock) ? nextPow2(size2) : threadsPerBlock;
            int blocks = (size2 + threads - 1) / threads;

            reduce(size2,threads,blocks,g_res,g_res,kernel->getName());

            size2 = (size2 + threads - 1) / threads;
        }

        cudaThreadSynchronize();

        if (cudaMemcpy(result, g_res, sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        if (cudaFree(g_vector) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        if (cudaFree(g_res) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        Timer::timestamp(&stop);

        std::cout << "Reduction: Tiempo GPU " << nGPU << ": " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;
    }
    catch (Cuda_runtime_error err) {
        clean_ = false;
        err_ = err;
    }
}

#endif

