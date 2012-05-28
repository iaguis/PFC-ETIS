#ifndef _ALGORITHM_H
#define	_ALGORITHM_H

#include "Runtime.h"
#include "Transform_op.h"
#include "Reduction_op.h"

namespace libreria {
    template <typename T> class accumulate_default: public Reduction_op<T> {
    public:
        accumulate_default():Reduction_op<T>("reduction_add"){}
        T operator()(const T& acc, const T& n) const {
            return acc+n;
        }

        T identity() const {
            return 0;
        }
    };

    template < typename InputIterator, typename OutputIterator, typename T>
      OutputIterator transform ( InputIterator first1, InputIterator last1,
                                 OutputIterator result, Transform_op<T> & op, double prop ) {

        int size = last1-first1;
        Runtime<T> r1(&*result,&*first1,size);
        r1.transform(op,prop);

        return result;
    }
    
    template <class InputIterator, class T, class BinaryFunction>
    T accumulate(InputIterator first, InputIterator last, T init,
                BinaryFunction binary_op) {
        int size = last - first;
        if (size <= 0)
            return init;
        T result = 0;
        Runtime<T> r1(&*first,&*last,size);
        r1.reduction(&result,binary_op);

        return (binary_op(result,init));
    }

    template <class InputIterator, class T>
    T accumulate(InputIterator first, InputIterator last, T init) {
        return libreria::accumulate(first, last, init, accumulate_default<T>());
    }


}


#endif	/* _ALGORITHM_H */

