#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <cuda_runtime.h>

#include "Runtime.h"
#include "Transform_op.h"
#include "Reduction_op.h"
#include "algorithm.h"
#include "Timer.h"

template <typename T> class t_sqrt: public Transform_op<T> {
public:
    t_sqrt():Transform_op<T>("sqrt"){}
    T operator()(const T& n) const {
        return sqrt(n);
    }
};

int main(int argc, char **argv) {
    double prop = atof(argv[1]);
    const float epsilon = 1e-6;
    const int size = atoi(argv[2]);
    std::vector<int> v_reduction(size);
    std::vector<float> v_transform(size),v_transform_stl(size);

    srand((unsigned int) time(0));
    
    for (int i=0; i<size; i++) {
        float value_transform = rand();
        int value_reduction = rand()%1500-750;
        v_transform[i] = value_transform;
        v_transform_stl[i] = value_transform;
        v_reduction[i] = value_reduction;
    }


    struct timeval start, stop;

    int overhead;

    Timer::timestamp(&start);
    cudaGetDeviceCount(&overhead);
    Timer::timestamp(&stop);

    std::cout << "Overhead CUDA: " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;

    Timer::timestamp(&start);
    std::transform(v_transform_stl.begin(), v_transform_stl.end(), v_transform_stl.begin(), t_sqrt<float>());
    Timer::timestamp(&stop);

    std::cout << "Transform: Tiempo STL: " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;

    t_sqrt<float> f;
    Timer::timestamp(&start);
    libreria::transform(v_transform.begin(), v_transform.end(), v_transform.begin(), f, prop);
    Timer::timestamp(&stop);

    std::cout << "Transform: Tiempo libreria: " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;

    Timer::timestamp(&start);
    int reduction_stl = std::accumulate(v_reduction.begin(), v_reduction.end(), 0);
    Timer::timestamp(&stop);

    std::cout << "Reduction: Tiempo STL: " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;

    Timer::timestamp(&start);
    int reduction_libreria = libreria::accumulate(v_reduction.begin(), v_reduction.end(), 0);
    Timer::timestamp(&stop);

    std::cout << "Reduction: Tiempo libreria: " << Timer::elapsed(start,stop)*1000 << "ms" << std::endl;

    for (int i=	0; i<size;i++) {
        if (abs(v_transform.at(i) - v_transform_stl.at(i)) > epsilon) {
            std::cout << i << ": " << "ERROR: transform: ";
            std::cout << "Expected: " << v_transform_stl.at(i) << " Received: " << v_transform.at(i) << " ERROR: ";
            std::cout << abs(v_transform.at(i) - v_transform_stl.at(i)) << std::endl;
        }
    }

    if (reduction_libreria != reduction_stl) {
        std::cout << "ERROR: reduction" << std::endl;
        std::cout << "Reduction: Esperado: " << reduction_stl << "   Devuelto: " << reduction_libreria << std::endl;
    }

    return 0;
}
