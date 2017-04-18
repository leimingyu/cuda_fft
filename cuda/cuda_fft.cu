#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>	// srand
//#include <math.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>



int main(int argc, char **argv)
{
	int fftlen = 1024;

	if(argc == 2) {
		fftlen = atoi(argv[1]);
	}

	assert(argc <= 2 && "Wrong input. Just specify the fft len!");

	printf("Running fft length %d on GPU.\n", fftlen);

	//------------------------------------------------------------------------//
	// host memory
	//------------------------------------------------------------------------//
	cufftComplex *h_sig;
	cudaMallocHost((void **) &h_sig, sizeof(cufftComplex) * fftlen);

	srand(time(NULL)); // initialize random seed
	for (int i = 0; i < fftlen; i++) {
		h_sig[i].x = (float)rand() / RAND_MAX;
		h_sig[i].y = 0.0;
	}

	//------------------------------------------------------------------------//
	// device memory
	//------------------------------------------------------------------------//
	cufftComplex *d_sig, *d_result;
	cudaMalloc((void **) &d_sig,    sizeof(cufftComplex) * fftlen);
	cudaMalloc((void **) &d_result, sizeof(cufftComplex) * fftlen);

	//------------------------------------------------------------------------//
	// copy data from host to device 
	//------------------------------------------------------------------------//
	cudaMemcpyAsync(d_sig, h_sig, sizeof(cufftComplex) * fftlen, cudaMemcpyHostToDevice);

	//------------------------------------------------------------------------//
	// set up cuda fft env 
	//------------------------------------------------------------------------//
	cufftHandle cufft_plan;
	cufftPlan1d(&cufft_plan, fftlen, CUFFT_C2C, 1); // batch = 1

	//------------------------------------------------------------------------//
	// gpu timer 
	//------------------------------------------------------------------------//
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int total_iters = 100;
	//------------------------------------------------------------------------//
	// run fft on gpu 
	//------------------------------------------------------------------------//
	cudaEventRecord(start, 0);

	for(int i = 0; i<total_iters; i++)
		cufftExecC2C(cufft_plan, d_sig, d_result, CUFFT_FORWARD);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float gputime_ms;
	cudaEventElapsedTime(&gputime_ms, start, stop);

	float avg_time_ms = gputime_ms / float(total_iters);

	printf("Avg %d fft on GPU : %f ms\n", fftlen, avg_time_ms);

	//------------------------------------------------------------------------//
	// free 
	//------------------------------------------------------------------------//
	checkCudaErrors(cufftDestroy(cufft_plan)); 	// cuda fft context

	checkCudaErrors(cudaFreeHost(h_sig));
	checkCudaErrors(cudaFree(d_sig));
	checkCudaErrors(cudaFree(d_result));
}
