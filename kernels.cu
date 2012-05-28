extern "C" __global__ void add1_i(int * const vector, const int size)
{
    int i = (blockIdx.y * gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        vector[i] += 1;
}

extern "C" __global__ void add1_f(float * const vector, const int size)
{
    int i = (blockIdx.y * gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        vector[i] += 1;
}

extern "C" __global__ void sqrt_f(float * const vector, const int size)
{
    int i = (blockIdx.y * gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        vector[i] = sqrt(vector[i]);
}

extern "C" __global__ void reduction_add_i(const int * const d_in, int * const res, const int size)
{
    extern __shared__ int s_data[];

    int idxOffset = (blockIdx.y * gridDim.x * blockDim.x ) + blockDim.x * blockIdx.x;
    int idx = idxOffset + threadIdx.x;
    int tx = threadIdx.x;

    s_data[tx] = (idx < size)? d_in[idx]: 0;

    __syncthreads();

    int stride;

    for (stride=blockDim.x/2; stride > 0; stride>>=1) {
	if (tx < stride) {
	    s_data[tx] += s_data[tx + stride];
	}

        __syncthreads();
    }

    if (tx == 0)
        res[blockIdx.y * gridDim.x + blockIdx.x] = s_data[0];
}

extern "C" __global__ void reduction_add_j(const unsigned int * const d_in, unsigned int * const res, const int size)
{
    extern __shared__ int s_data[];

    int idxOffset = (blockIdx.y * gridDim.x * blockDim.x ) + blockDim.x * blockIdx.x;
    int idx = idxOffset + threadIdx.x;
    int tx = threadIdx.x;

    s_data[tx] = (idx < size)? d_in[idx]: 0;

    __syncthreads();

    int stride;

    for (stride=blockDim.x/2; stride > 0; stride>>=1) {
	if (tx < stride) {
	    s_data[tx] += s_data[tx + stride];
	}

        __syncthreads();
    }

    if (tx == 0)
        res[blockIdx.y * gridDim.x + blockIdx.x] = s_data[0];
}

extern "C" __global__ void reduction_multiply_j(const unsigned int * const d_in, unsigned int * const res, const int size)
{
    extern __shared__ int s_data[];

    int idxOffset = (blockIdx.y * gridDim.x * blockDim.x ) + blockDim.x * blockIdx.x;
    int idx = idxOffset + threadIdx.x;
    int tx = threadIdx.x;

    s_data[tx] = (idx < size)? d_in[idx]: 1;

    __syncthreads();

    int stride;

    for (stride=blockDim.x/2; stride > 0; stride>>=1) {
	if (tx < stride) {
	    s_data[tx] *= s_data[tx + stride];
	}

        __syncthreads();
    }

    if (tx == 0)
        res[blockIdx.y * gridDim.x + blockIdx.x] = s_data[0];
        res[blockIdx.x] = s_data[0];
}
