#ifndef _TRANSFORM_CPU_H
#define	_TRANSFORM_CPU_H

#include <boost/thread.hpp>

#include "Transform_op.h"
#include "Timer.h"

/*
 * Each instance of this class executes the code in transform in its own thread.
 * */
template <typename T> class Transform_CPU {
public:
    Transform_CPU(T *const out, const T *const in, const int size, Transform_op<T> *kernel):
        m_Thread(boost::thread(&Transform_CPU::transform, this, out, in, size, kernel))
    {
    }

    ~Transform_CPU()
    {
        m_Thread.join();
    }
    void transform(T *const out, const T* const in, const int size, Transform_op<T>* kernel);
private:
    boost::thread m_Thread;
};

/*
 * Transforms every element of in with the operation defined in kernel and saves it to out
 * */
template <typename T> void Transform_CPU<T>::transform(T *const out, const T* const in, const int size, Transform_op<T> *kernel)
{
    struct timeval start, stop;

    Timer::timestamp(&start);
    
    for (int i=0; i<size; ++i) {
        out[i] = (*kernel)(in[i]);
    }

    Timer::timestamp(&stop);
    
    std::cout << "Transform: Tiempo CPU: " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;

}

#endif

