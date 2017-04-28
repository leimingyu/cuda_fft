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


	printf("[LOG] Start 1d-fft GPU.\n");

	//------------------------------------------------------------------------//
	// host memory
	//------------------------------------------------------------------------//
	// ToDo : allocate host memory using cufftComplex (h_sig) 
	


	// ToDo: init host memory ( hint : float2(float x, float y)) 


	//------------------------------------------------------------------------//
	// device memory
	//------------------------------------------------------------------------//
	// ToDo: allocate device memory for host data (d_sig) and  for output results (d_result)

	//------------------------------------------------------------------------//
	// copy data from host to device 
	//------------------------------------------------------------------------//
	// ToDo: copy data to device , hsig -> d_sig

	//------------------------------------------------------------------------//
	// set up cuda fft env 
	//------------------------------------------------------------------------//
	printf("[LOG] Create C2C plan for %i on GPU.\n", sig_len);

	cufftHandle cufft_plan;

	// ToDo: fill in the plan info 
	// http://docs.nvidia.com/cuda/cufft/index.html#function-cufftplan1d
	cufftPlan1d( );

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
			// ToDo: run c2c cufft
			// http://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecc2c-cufftexecz2z 
			cufftExecC2C( );
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float gputime_ms = 0.0;
		cudaEventElapsedTime(&gputime_ms, start, stop);

		sum_gputime_ms += (double) gputime_ms;

		printf("%lf ms (%d iters)\n", gputime_ms, fft_run);
	}

	printf("[LOG] Finished!\n");
	printf("[LOG] Average: %lf sec (per %d iters)\n", 
			sum_gputime_ms * 1e-3 / (double)trials, fft_run);

	//------------------------------------------------------------------------//
	// free 
	//------------------------------------------------------------------------//
	checkCudaErrors(cufftDestroy(cufft_plan)); 	// cuda fft context

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	// ToDo: deallocate h_sig, d_sig, d_result
}
