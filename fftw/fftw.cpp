#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fftw3.h>
#include <omp.h>

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

	/*
	 * Parameters
	 */
	fftwf_complex *input, *output; // fp32

	input  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * sig_len);
	output = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * sig_len);

	/*
	 * Init 
	 */
	srand(time(NULL));
	for (int i = 0; i < sig_len; i++) {
		input[i][0] = (float)rand() / RAND_MAX;
		input[i][1] = 0.f;
	}


	fftwf_plan plan_fp32;

	plan_fp32 = fftwf_plan_dft_1d(sig_len, input, output, FFTW_FORWARD, FFTW_ESTIMATE);


	/*
	 * Benchmarking
	 */
	double sum_of_elapsed_times = 0.0;
	double start, end;

	printf("[LOG] Run FFTW ...\n");
	for (int i = 0; i < trials; i++) {
		start = omp_get_wtime(); // timing : not multi-threaded

		for (int j = 0; j < fft_run; j++) {
			fftwf_execute(plan_fp32);
		}

		end = omp_get_wtime();

		double elapsed_time_sec = end - start;
		sum_of_elapsed_times += elapsed_time_sec;
		printf("%lf sec\n", elapsed_time_sec);
	}
	printf("[LOG] Finished!\n");
	printf("[LOG] Average: %lf sec\n", sum_of_elapsed_times / trials);



	/*
	 * Free 
	 */
	fftwf_destroy_plan(plan_fp32);

	fftwf_free(input);
	fftwf_free(output);

	return 0;
}
