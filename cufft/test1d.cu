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
	int targetDev = 0;
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, targetDev);
	printf("Device name: %s\n", prop.name);

	cudaSetDevice(targetDev);

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
	checkCuda( cudaMalloc((void**)&d3, sizeof(float) * 4098 * 64) );

	cufftComplex *d1_complex, *d2_complex, *d3_complex;
	checkCuda( cudaMalloc((void**)&d1_complex, sizeof(cufftComplex) * 2048 * 64) );
	checkCuda( cudaMalloc((void**)&d2_complex, sizeof(cufftComplex) * 4096 * 64) );
	checkCuda( cudaMalloc((void**)&d3_complex, sizeof(cufftComplex) * 4098 * 64) );

	//------------------------------------------------------------------------//
	// copy data from host to device 
	//------------------------------------------------------------------------//
	checkCuda( cudaMemcpy(d1, h1, sizeof(float)*64*2048, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(d2, h2, sizeof(float)*64*4096, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(d3, h3, sizeof(float)*64*4098, cudaMemcpyHostToDevice) );

	//------------------------------------------------------------------------//
	// Create FFT plan 
	//------------------------------------------------------------------------//
	printf("[LOG] 2k fft plan.\n");
	cufftHandle plan2k;
	if (cufftPlan1d(&plan2k, 2048, CUFFT_R2C, 64) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: 2k Plan creation failed");
		return -1;	
	}	

	printf("[LOG] 2k inverse fft plan.\n");
	cufftHandle plan2ki;
	if (cufftPlan1d(&plan2ki, 2048, CUFFT_C2R, 64) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ifft 2k Plan creation failed");
		return -10;	
	}

	printf("[LOG] 4k fft plan.\n");
	cufftHandle plan4k;
	if (cufftPlan1d(&plan4k, 4096, CUFFT_R2C, 64) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: 4k Plan creation failed");
		return -2;	
	}	

	printf("[LOG] 4k inverse fft plan.\n");
	cufftHandle plan4ki;
	if (cufftPlan1d(&plan4ki, 4096, CUFFT_C2R, 64) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ifft 4k Plan creation failed");
		return -20;	
	}	

	printf("[LOG] 4k_a fft plan.\n");
	cufftHandle plan4k_a;
	if (cufftPlan1d(&plan4k_a, 4096, CUFFT_R2C, 64) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: 4k_a Plan creation failed");
		return -3;	
	}	

	printf("[LOG] 4k_a_i fft plan.\n");
	cufftHandle plan4k_a_i;
	if (cufftPlan1d(&plan4k_a_i, 4096, CUFFT_C2R, 64) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ifft 4k_a Plan creation failed");
		return -30;	
	}	


	//------------------------------------------------------------------------//
	// run forward FFT
	//------------------------------------------------------------------------//
	float gputime_ms;

	//--------//
	// 2k fft
	//--------//
	printf("[LOG] Test R2C 2048 fft.\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecR2C(plan2k, (cufftReal*)d1, (cufftComplex*)d1_complex) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 2K Forward failed");
			return -4;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);

	//--------//
	// 2k ifft
	//--------//
	printf("[ifft] 2048.\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecC2R(plan2ki, (cufftComplex*)d1_complex, (cufftReal*)d1) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 2K Inverset failed");
			return -40;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);

	//--------//
	// 4k fft
	//--------//
	printf("[LOG] Test R2C 4096 fft.\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecR2C(plan4k, (cufftReal*)d2, (cufftComplex*)d2_complex) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 4K Forward failed");
			return -5;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);

	//--------//
	// 4k ifft
	//--------//
	printf("[ifft] 4096\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecC2R(plan4ki, (cufftComplex*)d2_complex, (cufftReal*)d2) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 4K Inverse failed");
			return -50;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);

	//--------//
	// 4k fft on 4098 
	//--------//
	printf("[LOG] Test R2C 4096 fft on 4098.\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecR2C(plan4k_a, (cufftReal*)d3, (cufftComplex*)d3_complex) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 4K_a Forward failed");
			return -6;	
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime_ms, start, stop);
	printf("runtime = %lf (ms)\n", gputime_ms * 0.01);

	//--------//
	// 4k ifft on 4098 
	//--------//
	printf("[ifft] 4096 on 4098\n");

	gputime_ms = 0.f;
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (cufftExecC2R(plan4k_a_i, (cufftComplex*)d3_complex, (cufftReal*)d3) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecR2C 4K_a Inverse failed");
			return -60;	
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

	cufftDestroy(plan2k);
	cufftDestroy(plan2ki);

	cufftDestroy(plan4k);
	cufftDestroy(plan4ki);

	cufftDestroy(plan4k_a);
	cufftDestroy(plan4k_a_i);

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
