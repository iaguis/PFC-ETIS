#ifndef _RUNTIME_H
#define	_RUNTIME_H

#include <typeinfo>
#include <cstring>
#include <math.h>
#include <cuda_runtime.h>

#include "Transform_CPU.h"
#include "Transform_GPU.h"
#include "Reduction_CPU.h"
#include "Reduction_GPU.h"

#include "Cuda_runtime_error.h"
#include "No_device.h"

template <typename T> class Runtime {
public:
    Runtime(T * const out, const T* const in, const int size) throw (No_device): nGPU(gpusAvailable()), nThreads(threadsAvailable()),
            _size(size), _out(out), _in(in)
    {
        if (nGPU == 0)
            throw No_device();
    }
    ~Runtime()
    {}
    
    void transform(Transform_op<T>& kernel, const double gpu_proportion) throw (Cuda_runtime_error);
    void reduction(T* const result, Reduction_op<T>& kernel) throw (Cuda_runtime_error);

    
private:
    int threadsAvailable() const;
    int gpusAvailable() const;

private:
    int nGPU;
    int nThreads;
    const int _size;
    T* const _out;
    const T* const _in;

};

/*
 * Transform _vector in GPU and CPU.
 *
 * First splits _vector in two parts: one for the GPU and other for the CPU according to
 * gpu_proportion and then splits the two parts to fit the number of GPU's and threads.
 * */
template <typename T> void Runtime<T>::transform(Transform_op<T>& kernel, const double gpu_proportion)
throw (Cuda_runtime_error)
{
    int sizeGPU = (gpu_proportion * _size);
    int sizeCPU = _size - sizeGPU;
    
    int chunkSizeGPU = ceil((double) sizeGPU/nGPU);
    int chunkSizeCPU = ceil((double) sizeCPU/nThreads);

    Transform_GPU<T> *temp_gpu[nGPU];
    Transform_CPU<T> *temp_cpu[nGPU];

    if (sizeGPU) {
        for (int i=0,offset=0; i<nGPU; ++i, offset+=chunkSizeGPU) {
            // Reduce chunk size if we exceed GPU size
            if (offset+chunkSizeGPU > sizeGPU)
                chunkSizeGPU -= (offset+chunkSizeGPU - sizeGPU);

            if (chunkSizeGPU == 0)
                break;
            temp_gpu[i] = new Transform_GPU<T>(_out + offset, _in + offset, chunkSizeGPU, &kernel, i);
//            Transform_GPU<T> temp(_out +  offset, _in + offset, chunkSizeGPU, &kernel, i);
        }
    }

    if (sizeCPU) {
        for (int i=0,offset = sizeGPU; i<nThreads; ++i, offset+=chunkSizeCPU) {
            if (offset+chunkSizeCPU > _size)
                chunkSizeCPU -= (offset+chunkSizeCPU - _size);

//            Transform_CPU<T> temp(_out +  offset, _in + offset, chunkSizeCPU, &kernel);
            temp_cpu[i] = new Transform_CPU<T>(_out + offset, _in + offset, chunkSizeCPU, &kernel);
        }

   }


    if (sizeGPU) {
        for (int i=0; i < nGPU; i++) {
                delete temp_gpu[i];
        }
    }

    if (sizeCPU) {
        for (int i=0; i < nThreads; i++) {
                delete temp_cpu[i];
        }
    }
    
}

template <typename T> void Runtime<T>::reduction(T* const result, Reduction_op<T>& kernel)
throw (Cuda_runtime_error)
{
    T partial_result[nGPU];
    int offset=0, chunkSize=_size/nGPU;

    Reduction_GPU<T> *temp_gpu[nGPU];

    for (int i=0; i<nGPU; ++i) {
        temp_gpu[i] = new Reduction_GPU<T>(&partial_result[i],_out + offset, chunkSize, &kernel, i);
//        Reduction_GPU<T> temp(&partial_result[i],_out + offset,chunkSize,&kernel,i);
        offset += chunkSize;
    }

    for (int i=0; i<nGPU; i++) {
        delete temp_gpu[i];
    }

    Reduction_CPU<T> temp(result,partial_result,nGPU,&kernel);

    return;
}

/*
 * Returns the number of CPU threads available
 * */
template <typename T> int Runtime<T>::threadsAvailable() const
{
    return (boost::thread::hardware_concurrency())? boost::thread::hardware_concurrency() - (gpusAvailable() - 1)   : -1;
}

/*
 * Returns the number of GPU's available
 * */
template <typename T> int Runtime<T>::gpusAvailable() const
{
    int count;
    cudaGetDeviceCount(&count);

    if (count == 1) {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,0);
        if (prop.major == 9999)
            return -1;
        else
            return 1;
    }

    return count;
}


#endif

