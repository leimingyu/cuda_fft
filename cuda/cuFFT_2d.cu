#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "common.h"

extern const int SigLen;
extern const int FFTRun;
extern const int Trials; 


int main(int argc, char **argv)
{
	int sig_len = SigLen;
	int fft_run = FFTRun;
	int trials  = Trials;

	//-------------------//
	// read cmd options
	//-------------------//
	int i = 0;
	while(i < argc)
	{
		if(argv[i][0]=='-') 
		{
			if(argv[i][1]=='-'){
				// read long options
				if(moreopt(argv[i]))
					fprintf(stderr,"unknown verbose option : %s\n", argv[i]);
			}	

			// read short options
			switch(argv[i][1])
			{
				case 'u':
					usage(argv[0]);
					exit(EXIT_FAILURE);

				case 'l':
					i=read_opt(argc, argv, i, &sig_len, "int");
					break;

				case 'r':
					i=read_opt(argc, argv, i, &fft_run, "int");
					break;

				case 't':
					i=read_opt(argc, argv, i, &trials, "int");
					break;
			}
		}
		i++;
	}

	printf("[LOG] Signal Length: %d\n", sig_len);
	printf("[LOG] FFT run: %d\n", fft_run);
	printf("[LOG] Trials: %d\n", trials);


	printf("[LOG] Start 2d-fft on GPU.\n");

	//------------------------------------------------------------------------//
	// host memory
	//------------------------------------------------------------------------//
	cufftComplex *h_sig;
	cudaMallocHost((void **) &h_sig, sizeof(cufftComplex) * sig_len * sig_len);

	srand(time(NULL));
	for (int i = 0; i < sig_len * sig_len; i++) {
			h_sig[i].x = (float)rand() / RAND_MAX;
			h_sig[i].y = (float)rand() / RAND_MAX;
	}

	//------------------------------------------------------------------------//
	// device memory
	//------------------------------------------------------------------------//
	cufftComplex *d_sig, *d_result;
	cudaMalloc((void **) &d_sig,    sizeof(cufftComplex) * sig_len * sig_len);
	cudaMalloc((void **) &d_result, sizeof(cufftComplex) * sig_len * sig_len);

	//------------------------------------------------------------------------//
	// copy data from host to device 
	//------------------------------------------------------------------------//
	cudaMemcpyAsync(d_sig, h_sig, sizeof(cufftComplex) * sig_len * sig_len, 
			cudaMemcpyHostToDevice);

	//------------------------------------------------------------------------//
	// set up cuda fft env 
	//------------------------------------------------------------------------//
	printf("[LOG] Create C2C plan for %i x %i on GPU.\n", sig_len, sig_len);

	cufftHandle cufft_plan2d;

	cufftPlan2d(&cufft_plan2d, sig_len, sig_len, CUFFT_C2C);

	//------------------------------------------------------------------------//
	// gpu timer 
	//------------------------------------------------------------------------//
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//------------------------------------------------------------------------//
	// run fft on gpu 
	//------------------------------------------------------------------------//
	double sum_gputime_ms = 0.0;

	printf("[LOG] Benchmarking cufft ...\n");

	for (int i = 0; i < trials; i++) {

		cudaEventRecord(start, 0);

		for (int j = 0; j < fft_run; j++) {
			cufftExecC2C(cufft_plan2d, d_sig, d_result, CUFFT_FORWARD);
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float gputime_ms = 0.0;
		cudaEventElapsedTime(&gputime_ms, start, stop);

		sum_gputime_ms += (double) gputime_ms;

		printf("%lf ms (%d iters)\n", gputime_ms, fft_run);
	}

	printf("[LOG] Finished!\n");
	printf("[LOG] Average: %lf sec (per %d iters)\n", sum_gputime_ms * 1e-3 / (double)trials, fft_run);

	//------------------------------------------------------------------------//
	// free 
	//------------------------------------------------------------------------//
	checkCudaErrors(cufftDestroy(cufft_plan2d)); 	// cuda fft context

	checkCudaErrors(cudaFreeHost(h_sig));
	checkCudaErrors(cudaFree(d_sig));
	checkCudaErrors(cudaFree(d_result));
}
