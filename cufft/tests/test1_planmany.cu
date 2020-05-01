#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define FFTSIZE 8
#define BATCH 2

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/********/
/* MAIN */
/********/
int main (int argc, char* argv[])
{
	if(argc != 2) {
		printf("Please specify 1 (plan1d) or 2 (planmany).\n");	
		return -1;
	};

	int N = FFTSIZE;
	N = N + 2;

	// --- Host side input data allocation and initialization
	cufftReal *hostInputData = (cufftReal*)malloc(N*BATCH*sizeof(cufftReal));
	for (int i=0; i<BATCH; i++) {
		for (int j=0; j<N; j++){
			//hostInputData[i*FFTSIZE + j] = (cufftReal)(j + 1);
			//hostInputData[i*FFTSIZE + j] = 100.f; 
			hostInputData[i*N+ j] = (float)(j % 10); 
			printf("%f ", hostInputData[i*N + j]);
		}
		printf("\n");
	}
	printf("\n");

	// --- Device side input data allocation and initialization
	cufftReal *deviceInputData;
	gpuErrchk(cudaMalloc((void**)&deviceInputData, N * BATCH * sizeof(cufftReal)));
	cudaMemcpy(deviceInputData, hostInputData, N * BATCH * sizeof(cufftReal), cudaMemcpyHostToDevice);

	// --- Host side output data allocation
	cufftComplex *hostOutputData = (cufftComplex*)malloc((FFTSIZE / 2 + 1) * BATCH * sizeof(cufftComplex));

	// --- Device side output data allocation
	cufftComplex *deviceOutputData; gpuErrchk(cudaMalloc((void**)&deviceOutputData, (FFTSIZE / 2 + 1) * BATCH * sizeof(cufftComplex)));

	// --- Batched 1D FFTs
	cufftHandle handle;
	int rank = 1;                           // --- 1D FFTs
	int n[] = { FFTSIZE };                 // --- Size of the Fourier transform
	int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
	int idist = FFTSIZE, odist = (FFTSIZE / 2 + 1); // --- Distance between batches
	int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
	int batch = BATCH;                      // --- Number of batched executions


	int plantype = atoi(argv[1]);

	if(plantype == 1) {
		printf("cufftplan1d\n");
		cufftPlan1d(&handle, FFTSIZE, CUFFT_R2C, BATCH);
	}

	if(plantype == 2) {
		printf("cufftplan2d\n");
		cufftPlanMany(&handle, rank, n, 
				inembed, istride, idist,
				onembed, ostride, odist, CUFFT_R2C, batch);
	}


	cufftExecR2C(handle,  deviceInputData, deviceOutputData);

	// --- Device->Host copy of the results
	gpuErrchk(cudaMemcpy(hostOutputData, deviceOutputData, (FFTSIZE / 2 + 1) * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	for (int i=0; i<BATCH; i++)
		for (int j=0; j<(FFTSIZE / 2 + 1); j++)
			printf("%i %i %f %f\n", i, j, hostOutputData[i*(FFTSIZE / 2 + 1) + j].x, hostOutputData[i*(FFTSIZE / 2 + 1) + j].y);

	cufftDestroy(handle);
	gpuErrchk(cudaFree(deviceOutputData));
	gpuErrchk(cudaFree(deviceInputData));

}
