#ifndef _TRANSFORM_OP_H
#define _TRANSFORM_OP_H

template <typename T> class Transform_op {
protected:
    const char *_name;
public:
    Transform_op(const char* name):_name(name){}
    virtual T operator()(T const& n) const = 0;
    const char* getName()
    {
        return _name;
    }
};

#endif
