#include <cstdio>

#include <cuda_runtime.h>

#include "ray.cuh"


__global__
void cudaRayKernel(float *vel, float *thick, float *rho, float *c_rho,
    int n_layers, int l_rho, float start, float end, float threshold) {

	/* get current thread's id */
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	/* create float to hold error */
	float err;

	/* while this thread is dealing with a valid index */
	while (index < l_rho) {
		/* initialize error */
		err = start;

		/* sum over all layers with this rho value */
		for(int k = 0; k < n_layers * 2; k++){
			err += (rho[index] * vel[k] * thick[k]) / sqrt(1 - (rho[index] * rho[index] * vel[k] * vel[k]));
		}

		/* calculate and save error */
		err = err - end;

		/* check if error is within threshold */
		if(abs(err) < threshold){
			* c_rho = rho[index];
		}

		/* advance thread id */
		index += blockDim.x * gridDim.x;
	}
}


void cudaCallRayKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        float *vel,
        float *thick,
        float *rho,
        float *c_rho,
        int n_layers,
        int l_rho,
        float start,
        float end,
        float threshold){
        
    // Call the kernel above this function.
	cudaRayKernel<<<blocks, threadsPerBlock>>>(vel, thick, rho, c_rho,
	n_layers, l_rho, start, end, threshold);
}
