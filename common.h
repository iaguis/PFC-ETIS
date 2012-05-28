#ifndef _COMMON_H
#define _COMMON_H

const int threadsPerBlock = 512;

template<typename T, typename A> inline void ALIGN_UP(T offset, A alignment) {
    offset = (offset + alignment - 1) & ~(alignment - 1);
}

#endif
