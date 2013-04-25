#ifndef _CUDA_RUNTIME_ERROR_H
#define	_CUDA_RUNTIME_ERROR_H

#include <cuda_runtime.h>
#include <exception>

class Cuda_runtime_error: public std::exception {
	private:
		cudaError_t error_;
	public:                
		Cuda_runtime_error(cudaError_t error): error_(error) {}
                Cuda_runtime_error(): error_(static_cast<cudaError_t>(0)) {}

		const char *what() const throw() {
			return cudaGetErrorString(error_);
		}
};

#endif

