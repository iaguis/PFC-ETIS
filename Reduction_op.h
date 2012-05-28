#ifndef _REDUCTION_OP_H
#define	_REDUCTION_OP_H

template <typename T> class Reduction_op {
protected:
    const char *_name;
public:
    Reduction_op(const char* name):_name(name){}
    virtual T operator()(T const& acc, T const& n) const = 0;
    virtual T identity() const = 0;
    const char* getName()
    {
        return _name;
    }
};

#endif

