#ifndef _TRANSFORM_GPU_H
#define	_TRANSFORM_GPU_H

#include <boost/thread.hpp>

#include <math.h>

#include <cuda_runtime.h>

#include "common.h"
#include "Transform_op.h"
#include "Cuda_runtime_error.h"
#include "Timer.h"

#include <iostream>

/*
 * Each instance of this class executes the code in transform in its own thread.
 * */
template <typename T> class Transform_GPU {
public:
    Transform_GPU(T *const out, const T *const in, const int size, Transform_op<T> *kernel, int nGPU)
        : clean_(true), m_Thread(boost::thread(&Transform_GPU::transform, this, out, in, size, kernel, nGPU))
    {
    }
    
    ~Transform_GPU()
    {
        m_Thread.join();
        if (!clean_)
            throw err_;
    }
    void transform(T *const out, const T *const in, const int size, Transform_op<T> *kernel, int nGPU);
private:
    bool clean_;
    Cuda_runtime_error err_;
    boost::thread m_Thread;
};

/*
 * Transforms in according to the operation defined in kernel and saves it to out
 * */
template <typename T> void Transform_GPU<T>::transform(T *const out, const T *const in, const int size, Transform_op<T> *kernel, int nGPU)
{
    try {
        int *g_vector;
        unsigned int numBytes = size * sizeof(T);
        int threads,blocks, blocks1, blocks2;

        if (cudaSetDevice(nGPU) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        struct timeval start, stop;

        Timer::timestamp(&start);


        // Check if vector fits in a block
        if (size > threadsPerBlock) {
            threads = threadsPerBlock;
            blocks = ceil((double)size/threadsPerBlock);

            if (blocks > 65535) {
                blocks1 = 65535;
                blocks2 = ceil((double)blocks/65535);
            }
            else {
                blocks1 = blocks;
                blocks2 = 1;
            }

        }
        else {
            threads = size;
            blocks = 1;
        }

        if (cudaMalloc((void **)&g_vector, numBytes) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        if (cudaMemcpy(g_vector, in, numBytes, cudaMemcpyHostToDevice) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        dim3 dimGrid(blocks1,blocks2);

        if (cudaConfigureCall(dimGrid,threads,0,0) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        int offset = 0;

        // Alignment
        void *ptr;

        ptr = (void *)(size_t) g_vector;
        ALIGN_UP(offset, __alignof(ptr));

        if (cudaSetupArgument(&g_vector, sizeof(ptr), offset) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        offset += sizeof(ptr);

        if (cudaSetupArgument(&size, sizeof(T), offset) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        // Checks type to select appropriate kernel.
        char final_gpu_kernel[strlen(kernel->getName()) + 2];

        strcpy(final_gpu_kernel,kernel->getName());

        strcat(final_gpu_kernel,"_");
        strcat(final_gpu_kernel,typeid(T).name());
        
        if (cudaLaunch(final_gpu_kernel) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());
        
        if (cudaMemcpy(out, g_vector, numBytes, cudaMemcpyDeviceToHost) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        if (cudaFree(g_vector) != cudaSuccess)
            throw Cuda_runtime_error(cudaGetLastError());

        Timer::timestamp(&stop);
        
        std::cout << "Transform: Tiempo GPU " << nGPU << ": " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;
    }
    catch (Cuda_runtime_error err) {
        clean_ = false;
        err_ = err;
    }
}

#endif

