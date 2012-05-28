#ifndef _REDUCTION_CPU_H
#define	_REDUCTION_CPU_H

#include <boost/thread.hpp>

#include "Reduction_op.h"
#include "Timer.h"

/*
 * Each instance of this class executes the code in transform in its own thread.
 * */
template <typename T> class Reduction_CPU {
public:
    Reduction_CPU(T* const result, const T *const vector, const int size, Reduction_op<T> *kernel)
        : m_Thread(boost::thread(&Reduction_CPU::reduction, this, result, vector, size, kernel))
    {
    }

    ~Reduction_CPU()
    {
        m_Thread.join();
    }
    void reduction(T* const result, const T* const vector, const int size, Reduction_op<T> *kernel);
private:
    boost::thread m_Thread;
};

template <typename T> void Reduction_CPU<T>::reduction(T* const result, const T* const vector, const int size, Reduction_op<T> *kernel)
{
    struct timeval start,stop;
    Timer::timestamp(&start);
    T acc = kernel->identity();
    for (int i=0; i<size; ++i) {
        acc = (*kernel)(acc,vector[i]);
    }
    *result = acc;
    Timer::timestamp(&stop);
    std::cout << "Reduction: Tiempo CPU: " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;
}

#endif

