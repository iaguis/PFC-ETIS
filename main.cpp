#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "Runtime.h"
#include "Transform_op.h"
#include "Reduction_op.h"
#include "algorithm.h"

template <typename T> class t_sqrt: public Transform_op<T> {
public:
    t_sqrt():Transform_op<T>("sqrt"){}
    int operator()(const T& n) const {
        return sqrt(n);
    }
};

template <typename T> class add1: public Transform_op<T> {
public:
    add1():Transform_op<T>("add1"){}
    int operator()(const T& n) const {
        return ;
    }
};

template <typename T> class multiply: public Reduction_op<T> {
public:
    multiply():Reduction_op<T>("reduction_multiply"){}
    T operator()(const T& acc, const T &n) const {
        return acc*n;
    }
    
    T identity() const {
        return 1;
    }
};

template <typename T> class add: public Reduction_op<T> {
public:
    add():Reduction_op<T>("reduction_add"){}
    T operator()(const T& acc, const T& n) const {
        return acc+n;
    }

    T identity() const {
        return 0;
    }
};

int main(int argc, char **argv) {
    std::vector<int> v(1000000),v1(1000000);
    std::vector<unsigned int> v2(10);
    std::vector<int>::iterator it;

    srand((unsigned int) time(0));
    
    for (int i=0; i<1000000; i++) {
        int l = rand()%150+1;
        v[i] = l;
        v1[i] = l+1;
    }

    for (int i=0; i<10; i++) {
        v2[i] = i+1;

//        std::cout << i << ": " << v2[i] << std::endl;
    }


//    for (it=v.begin(); it!=v.end(); ++it)
//        std::cout << " " << *it;
//    std::cout << std::endl;


    std::transform(v.begin(), v.end(), v.begin(), t_sqrt<float>());
    //libreria::transform(v.begin(), v.end(), v.begin(), t_sqrt<float>());

    unsigned int referencia1, res1, referencia2, res2;

    referencia1 = std::accumulate(v2.begin(), v2.end(), (unsigned int)0);
    res1 = libreria::accumulate(v2.begin(), v2.end(), (unsigned int)0);

    referencia2 = std::accumulate(v2.begin(), v2.end(), (unsigned int)1, multiply<unsigned int>());
    res2 = libreria::accumulate(v2.begin(), v2.end(), (unsigned int)1, multiply<unsigned int>());

/*    for (int i=0; i<1000000;i++) {
        if (v.at(i) != v1.at(i))
            std::cout << "ERROR: transform" << std::endl;
    }*/

    std::cout << "std::acumulate: " << referencia1 << std::endl;
    std::cout << "libreria::acumulate: " << res1 << std::endl;

    std::cout << "std::acumulate (2): " << referencia2 << std::endl;
    std::cout << "libreria::acumulate (2): " << res2 << std::endl;

    if (referencia1 != res1)
        std::cout << "ERROR: accumulate 1" << std::endl;

    if (referencia2 != res2)
        std::cout << "ERROR: accumulate 2" << std::endl;

//    for (it=v.begin(); it!=v.end(); ++it)
//        std::cout << " " << *it;
//    std::cout << std::endl;

    return 0;
}

/*using namespace std;

int main(int argc, char** argv) {
    try {
        unsigned int size = atoi(argv[1]);
        double prop = atof(argv[2]);

        int v1[size], v1_out[size], transform_result[size], expected_result_transform=0, result_transform=0;
        unsigned int v2[size], expected_result_reduction=0,
                result_reduction=0;

        srand((unsigned int) time(0));

        for (unsigned int i=0;i<size;++i) {
            v1[i] = rand()%150+1;
            v2[i] = rand()%4+1;
            transform_result[i] = v1[i]+1;
        }

        for (unsigned int i=0;i<size;++i) {
            expected_result_transform += transform_result[i];
            expected_result_reduction += v2[i];
        }

        Runtime<int> r1(v1_out,v1,size);
        Runtime<unsigned int> r2(v2,v2,size);

        add1<int> temp1;
        add<unsigned int> temp2;

        r1.transform(temp1,prop);

        r2.reduction(&result_reduction, temp2);

        cout << "Test:" << endl;
        for (unsigned int i=0;i<size;++i) {
            cout << transform_result[i] << ": " << v1_out[i] << endl;
            result_transform += v1_out[i];
        }
        cout << endl;

        cout << "TRANSFORM:" << endl;
        (expected_result_transform == result_transform)? cout << "Success" << endl : cout << "FAIL" << endl << endl;
        cout << "REDUCTION:" << endl;
        (expected_result_reduction == result_reduction)? cout << "Success" << endl : cout << "FAIL" << endl;

        return 0;
    }
    catch (No_device &err) {
        cerr << "No CUDA-capable device!" << endl;
    }
    catch(Cuda_runtime_error &err) {
        cerr << "CUDA Runtime Error: " << err.what() << endl;
    }
    catch (exception &err) {
        cerr << "Exception: " << err.what() << endl;
    }
}*/
