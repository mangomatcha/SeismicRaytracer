#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <istream>
#include <windows.h>


#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "ray.cuh"


using namespace std;

const float PI = 3.14159265358979;

/*
 * NOTE: You can use this macro to easily check cuda error codes 
 * and get more information. 
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *   what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
        exit(code);
    }
}

/* Checks the passed-in arguments for validity. */
void check_args(int argc, char **argv) {
    if (argc != 8) {
        cerr << "Incorrect number of arguments.\n";
        cerr << "Arguments: <threads per block> <max number of blocks> " <<
            "<start point> <end point> <layer velocities file> " <<
            "<layer thicknesses file> <number of layers> \n";
        exit(EXIT_FAILURE);
    }
}

// Global variables to assist in timing
double PCFreq = 0.0;
__int64 CounterStart = 0;

// Initialize Windows-specific precise timing 
void initTiming()
{
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed! Timing routines won't work. \n");
    
    PCFreq = double(li.QuadPart)/1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}

// Get precise time
double preciseClock()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart)/PCFreq;
}


/* 
 * Generates random layer velocity and thickness data and computes travel
 * time for seismic wave traced from starting point to ending point.
 * 
 * Uses both CPU and GPU implementations, and compares the results.
 */
int raytracer_test(int argc, char **argv) {
    check_args(argc, argv);

    // Initialize timing
    initTiming();
    double time_initial, time_final, elapsed_ms;

    // Set the number of layers
    int n_layers = atoi(argv[7]);

    // Input velocity and thickness data
    float *input_vel = (float *) malloc(sizeof (float) * n_layers * 2);
    float *input_thick = (float *)malloc(sizeof(float) * n_layers * 2);

    // Allocate memory on the GPU here for velocities
    float *dev_input_vel;
	cudaMalloc((void**) &dev_input_vel, n_layers * 2 * sizeof(float));


    // Allocate memory on the GPU here for thicknesses
    float *dev_input_thick;
    cudaMalloc((void**) &dev_input_thick, n_layers * 2 * sizeof(float));

    // Read in layer velocities
    ifstream vfile(argv[5]);
    string temp;
    int index = 0;

    while(getline(vfile, temp)){
        input_vel[index] = stof(temp);
        index++;
    }

    // Read in layer thicknesses
    ifstream tfile(argv[6]);
    index = 0;

    while(getline(tfile, temp)){
        input_thick[index] = stof(temp);
        index++;
    }

    // Copy over data from first half of velocities and thicknesses to second
    // half to simulate reflect at the bottom of this chunk of layers. Also
    // calculate maximum velocity to set boundary rho values
    float vel_max = 0;
    for(int l = 0; l < n_layers; l++){
        input_vel[l + n_layers] = input_vel[n_layers - 1 - l];
        input_thick[l + n_layers] = input_thick[n_layers - 1 - l];

        if(input_vel[l] > vel_max){
            vel_max = input_vel[l];
        }
    }

    // Copy input velocity and thickness data from host memory to the GPU
    cudaMemcpy(dev_input_vel, input_vel, n_layers * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_input_thick, input_thick, n_layers * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate min and max rho and create array of rho values using max_vel
    float max_rho = (1 / vel_max) - 0.000001;
    int l_rho = (int) (max_rho / 0.000001);
    float * rho = (float *) malloc(l_rho * sizeof(float));

    printf("l_rho: %d\n", l_rho);

    for(int i = 0; i < l_rho; i++){
        rho[i] = i * 0.000001;
        //printf("rho[%d]: %f\n", i, rho[i]);
    }

    // Create rho array for GPU. Allocate memory & memcpy data over
    float *dev_rho;
    cudaMalloc((void **) &dev_rho, l_rho * sizeof(float));
    cudaMemcpy(dev_rho, rho, l_rho * sizeof(float), cudaMemcpyHostToDevice);

    // Create float to hold correct rho. Allocate memory & memset to 0. Repeat for GPU.
    float * c_rho = (float *) malloc(sizeof(float));
    memset(c_rho, 0, sizeof(float));
    float * dev_c_rho;
    cudaMalloc((void **) &dev_c_rho, sizeof(float));
    cudaMemcpy(dev_c_rho, c_rho, sizeof(float), cudaMemcpyHostToDevice);

    // Calculate threshold
    float start = atof(argv[3]);
    float end = atof(argv[4]);
    float threshold = (end - start)/100000;

    // CPU raytracing
    cout << "CPU raytracing..." << endl;

    // Start timer
    time_initial = preciseClock();
    
    // Set the number of rays the CPU implementation will check at each iteration
    int n_rays = 3;

    // Set the minimum rho to 0
    float min_rho = 0;

    // Set up variables to hold the spacing between each ray in the fan, the
    // minimum error at each iteration, the rho value corresponding to this
    // error, and arrays to hold the rho values and errors for each ray in the
    // fan
    float spacing, min_err, min_err_rho, * cpu_rho, * cpu_err;

    // Allocate memory for arrays
    cpu_rho = (float *) malloc(n_rays * sizeof(float));
    cpu_err = (float *) malloc(n_rays * sizeof(float));

    // Initialize minimum error
    min_err = 100000;

    // Initialize minimum positive and negative errors
    float min_err_pos = 100000;
    float min_err_neg = -100000;

    // While the minimum error between the calculated endpoint and the passed
    // endpoint is larger than the threshold, continue sending out fans of rays
    while(TRUE){
        // Calculate spacing between each ray in the fan
        spacing = (max_rho - min_rho) / (n_rays - 1);

        // Populate array of rho values for the rays in this fan, perform
        // distance equation summation for each ray, and calculate errors
        for(int i = 0; i < n_rays; i++){
            cpu_rho[i] = min_rho + (i * spacing);

            cpu_err[i] = start;

            // sum over all layers
            for(int k = 0; k < n_layers * 2; k++){
                cpu_err[i] += (cpu_rho[i] * input_vel[k] * input_thick[k]) / sqrt(1 - (cpu_rho[i] * cpu_rho[i] * input_vel[k] * input_vel[k]));   
            }

            // calculate error
            cpu_err[i] = cpu_err[i] - end;

            // check if error is larger than the currently saved minimum error
            if(min_err > abs(cpu_err[i])){
                min_err = abs(cpu_err[i]);
                min_err_rho = cpu_rho[i];
            }
        }

        // break out of the loop if the minimum error < threshold
        if(min_err < threshold){
            break;
        }

        for(int i = 0; i < n_rays; i++){
            // find the minimum negative error and set min_rho for next fan
            if(cpu_err[i] < 0 & min_err_neg < cpu_err[i]){
                min_err_neg = cpu_err[i];
                min_rho = cpu_rho[i];
            }

            // find the minimum positive error and set max_rho for next fan
            else if(cpu_err[i] > 0 & min_err_pos > cpu_err[i]){
                min_err_pos = cpu_err[i];
                max_rho = cpu_rho[i];
            }
        }

    }

    // use min_err_rho to calculate travel time
    float tr_time = 0;
    for(int k = 0; k < n_layers * 2; k++){
        tr_time += input_thick[k] / (input_vel[k] * sqrt(1 - (min_err_rho * min_err_rho * input_vel[k] * input_vel[k])));
    }

    // Stop timer
    time_final = preciseClock();
    elapsed_ms = time_final - time_initial;

    // Calculate distance error
    float e = start;
    for(int k = 0; k < n_layers * 2; k++){
        e += (min_err_rho * input_vel[k] * input_thick[k]) / sqrt(1 - (min_err_rho * min_err_rho * input_vel[k] * input_vel[k]));
    }
    e = e - end;
    cout << "Travel time: " << tr_time << " seconds" << endl;
    cout << "Error: " << e << " meters" << endl;

    cout << endl;
    cout << "CPU time: " << elapsed_ms << " milliseconds" << endl;

    free(cpu_rho);
    free(cpu_err);

    // GPU raytracing
    cout << endl;
    cout << "GPU raytracing..." << endl;

    // Set the number of threads per block and the number of blocks
    const unsigned int local_size = atoi(argv[1]);
    const unsigned int blocks = atoi(argv[2]);

    // Start timer
    time_initial = preciseClock();

    // Call the kernel
    cudaCallRayKernel(blocks, local_size, dev_input_vel, dev_input_thick, dev_rho, 
        dev_c_rho, n_layers, l_rho, start, end, threshold);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        cerr << "Error " << cudaGetErrorString(err) << endl;
    else
        cerr << "No kernel error detected" << endl;

    // Copy the correct rho from the GPU to host memory
    cudaMemcpy(c_rho, dev_c_rho, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate travel time with c_rho
    tr_time = 0;
    float r = * c_rho;
    for(int k = 0; k < n_layers * 2; k++){
        tr_time += input_thick[k] / (input_vel[k] * sqrt(1 - (r * r * input_vel[k] * input_vel[k])));
    }

    // Stop timer
    time_final = preciseClock();
    elapsed_ms = time_final - time_initial;

    // Calculate distance error
    e = start;
    for(int k = 0; k < n_layers * 2; k++){
        e += (r * input_vel[k] * input_thick[k]) / sqrt(1 - (r * r * input_vel[k] * input_vel[k]));
    }
    //e = e - end;
    cout << "Travel time: " << tr_time << " seconds" << endl;

    cout << endl;
    cout << "GPU time: " << elapsed_ms << " milliseconds" << endl;

    // Free all allocated memory on the GPU
	cudaFree(dev_rho);
    cudaFree(dev_c_rho);
    cudaFree(dev_input_thick);
    cudaFree(dev_input_vel);


    // Free memory on host
    free(rho);
    free(c_rho);
    free(input_thick);
    free(input_vel);

    return EXIT_SUCCESS;
}


int main(int argc, char **argv) {
    return raytracer_test(argc, argv);
}
