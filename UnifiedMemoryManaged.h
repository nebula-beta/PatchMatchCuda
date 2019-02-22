#ifndef UNIFIED_MEMORY_MANAGED_H
#define UNIFIED_MEMORY_MANAGED_H


#include "CudaErrorHandler.cuh"


class UnifiedMemoryManaged
{
public:
	void* operator new(size_t num_of_byte)
	{
		void* ptr;
		checkCudaError( cudaMallocManaged(&ptr, num_of_byte) );
		return ptr;
	}

	void operator delete(void* ptr)
	{
		cudaFree(ptr);
	}
};



#endif //UNIFIED_MEMORY_MANAGED_H
