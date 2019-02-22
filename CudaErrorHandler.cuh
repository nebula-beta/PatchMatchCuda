#ifndef CUDA_ERROR_HANDLER_H
#define CUDA_ERROR_HANDLER_H
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>



inline cudaError_t __checkCudaError(cudaError_t error_code, const char* const file, const int line)
{
#if defined(DEBUG) || defined(_DEBUG)
	if(error_code != cudaSuccess)
	{
		fprintf(stderr, "[file : %s line : %d] CUDA Runtime Error : %s\n", file, line, cudaGetErrorString(error_code));
		cudaDeviceReset();
		exit(EXIT_FAILURE);

	}
#endif
	return error_code;
}


inline void __checkForLastCudaError(const char* const file, const int line)
{
#if defined(DEBUG) || defined(_DEBUG)
	cudaError_t error_code = cudaGetLastError();
	if(error_code != cudaSuccess)
	{
		fprintf(stderr, "[file : %s, line : %d] CUDA Error : %s\n", file, line, cudaGetErrorString(error_code));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
	error_code = cudaDeviceSynchronize();
	if(error_code != cudaSuccess)
	{
		fprintf(stderr, "[file %s, line : %d] CUDA Sync Error : %s\n", file, line, cudaGetErrorString(error_code));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
#endif
}


#define checkCudaError(val) __checkCudaError((val), __FILE__, __LINE__)
#define checkForLastCudaError() __checkForLastCudaError(__FILE__, __LINE__)





#endif //CUDA_ERROR_HANDLER_H
