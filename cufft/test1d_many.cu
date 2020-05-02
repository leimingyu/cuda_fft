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

int err = -1;

int main(int argc, char **argv)
{
	/*
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

	printf("[LOG] FFT Length: %d\n", sig_len);
	printf("[LOG] Runs: %d\n", fft_run);
	printf("[LOG] Trials: %d\n", trials);

	*/

	printf("[LOG] Start 1d-fft GPU.\n");

	//------------------------------------------------------------------------//
	// host memory
	//------------------------------------------------------------------------//
	float *h1 =  (float*) malloc(sizeof(float) * 2048 * 64);
	float *h2 =  (float*) malloc(sizeof(float) * 4096 * 64);
	float *h3 =  (float*) malloc(sizeof(float) * 4098 * 64);

	for(int i=0;i<64;i++){
		for(int j=0;j<2048;j++){
			h1[i*2048 + j] = (float)(j+1);
		}
	}

	for(int i=0;i<64;i++){
		for(int j=0;j<4096;j++){
			h2[i*4096 + j] = (float)(j+1);
		}
	}

	for(int i=0;i<64;i++){
		for(int j=0;j<4098;j++){
			h3[i*4098 + j] = (float)(j+1);
		}
	}

	//------------------------------------------------------------------------//
	// gpu timer 
	//------------------------------------------------------------------------//
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//------------------------------------------------------------------------//
	// device memory
	//------------------------------------------------------------------------//
	// ToDo: allocate device memory for host data (d_sig) and  for output results (d_result)
	float *d1,*d2,*d3;
	checkCuda( cudaMalloc((void**)&d1, sizeof(float) * 2048 * 64) );
	checkCuda( cudaMalloc((void**)&d2, sizeof(float) * 4096 * 64) );

	//checkCuda( cudaMalloc((void**)&d3, sizeof(float) * 4098 * 64) );
	checkCuda( cudaMalloc((void**)&d3, sizeof(float) * 4096 * 64) );

	cufftComplex *d1_complex, *d2_complex, *d3_complex;
	checkCuda( cudaMalloc((void**)&d1_complex, sizeof(cufftComplex) * 2048 * 64) );
	checkCuda( cudaMalloc((void**)&d2_complex, sizeof(cufftComplex) * 4096 * 64) );

	//checkCuda( cudaMalloc((void**)&d3_complex, sizeof(cufftComplex) * 4098 * 64) );
	checkCuda( cudaMalloc((void**)&d3_complex, sizeof(cufftComplex) * 4096 * 64) );

	//------------------------------------------------------------------------//
	// copy data from host to device 
	//------------------------------------------------------------------------//
	checkCuda( cudaMemcpy(d1, h1, sizeof(float)*64*2048, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(d2, h2, sizeof(float)*64*4096, cudaMemcpyHostToDevice) );
	//checkCuda( cudaMemcpy(d3, h3, sizeof(float)*64*4098, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(d3, h3, sizeof(float)*64*4096, cudaMemcpyHostToDevice) );

	//------------------------------------------------------------------------//
	// Create FFT plan 
	//------------------------------------------------------------------------//

	// 2k fft
	printf("[planmany] 2k fft.\n");
	cufftHandle fwd_plan_2k;

	int fftsize =2048;
	int batch = 64;
	int n[] = {fftsize};

	if (cufftPlanMany(&fwd_plan_2k,
					  1, // 1d transform
					  n, // fft size 
					  NULL, 1, fftsize,
					  NULL, 1, fftsize/2 + 1,
					  CUFFT_R2C,
					  batch) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create cufftPlanMany for 2k\n");
		int myErr = err;
		err = err -1;
		return myErr;	
	}


	// 4k fft
	printf("[LOG] 4k fft plan.\n");
	cufftHandle fwd_plan_4k;

	fftsize = 4096;
	int n1[] = {fftsize};

	if (cufftPlanMany(&fwd_plan_4k,
					  1, // 1d transform
					  n1, // fft size 
					  NULL, 1, fftsize,
					  NULL, 1, fftsize/2 + 1,
					  CUFFT_R2C,
					  batch) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create cufftPlanMany for 4k\n");
		int myErr = err;
		err = err -1;
		return myErr;	
	}

	// 2k fft on 4k data size
	printf("[planmany] 2k fft on 4k data.\n");
	cufftHandle fwd_plan_2k_4k;

	fftsize =2048;
	batch = 64;
	int n2[] = {fftsize};

	if (cufftPlanMany(&fwd_plan_2k_4k,
					  1, // 1d transform
					  n2, // fft size 
					  NULL, 1, fftsize,
					  NULL, 1, fftsize/2 + 1,
					  CUFFT_R2C,
					  batch) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create cufftPlanMany for 2k on 4k\n");
		int myErr = err;
		err = err -1;
		return myErr;	
	}


	//------------------------------------------------------------------------//
	// run forward FFT
	//------------------------------------------------------------------------//
	float gputime_ms;

	//--------//
	// 2k fft
	//--------//
	printf("[LOG] planmany 2k fft.\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecR2C(fwd_plan_2k, (cufftReal*)d1, (cufftComplex*)d1_complex) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 2K Forward failed");
			int myErr = err;
			err = err -1;
			return myErr;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);


	//--------//
	// 4k fft
	//--------//
	printf("[LOG] planmany 4k fft.\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecR2C(fwd_plan_4k, (cufftReal*)d2, (cufftComplex*)d2_complex) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 4K Forward failed");
			int myErr = err;
			err = err -1;
			return myErr;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);

	//--------//
	// 2k fft on 4k data
	//--------//
	printf("[LOG] planmany 2k fft on 4k data.\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecR2C(fwd_plan_2k_4k, (cufftReal*)d3, (cufftComplex*)d3_complex) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 2K Forward on 4K failed");
			int myErr = err;
			err = err -1;
			return myErr;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);

	//------------------------------------------------------------------------//
	// free 
	//------------------------------------------------------------------------//
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	cufftDestroy(fwd_plan_2k);
	cufftDestroy(fwd_plan_4k);
	cufftDestroy(fwd_plan_2k_4k);

	cudaFree(d1);
	cudaFree(d2);
	cudaFree(d3);

	cudaFree(d1_complex);
	cudaFree(d2_complex);
	cudaFree(d3_complex);

	free(h1);
	free(h2);
	free(h3);

	return 0;
}
