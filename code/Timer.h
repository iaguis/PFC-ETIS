#ifndef _TIMER_H_
#define _TIMER_H_

#include <ctime>

namespace Timer {
        void timestamp(struct timeval *time) {
            gettimeofday(time,0);
        }

        double elapsed(struct timeval start, struct timeval stop) {
            return (stop.tv_sec + (stop.tv_usec * 1e-6)) - (start.tv_sec + (start.tv_usec * 1e-6));
        }
}

#endif
