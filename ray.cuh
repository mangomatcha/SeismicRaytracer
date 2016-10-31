#ifndef RAY_CUH
#define RAY_CUH
#define TRUE 1

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
        float threshold);

#endif