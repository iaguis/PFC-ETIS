#ifndef _NO_DEVICE_H
#define	_NO_DEVICE_H

#include <exception>

class No_device: public std::exception {
	public:
		char const* what() const throw() {
			return "No CUDA device found.";
		}
};

#endif	/* _NO_DEVICE_H */

