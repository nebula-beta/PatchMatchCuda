#ifndef GLOBAL_STATE_H
#define GLOBAL_STATE_H


#include "UnifiedMemoryManaged.h"
#include "AlgorithmParameters.h"
#include "LineState.h"


// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

class GlobalState : public UnifiedMemoryManaged
{
public:
	LineState* lines;

	curandState* cs;
	AlgorithmParameters* params;

	cudaTextureObject_t imgs[2];
	cudaArray* cuArray[2];


	GlobalState()
	{
		lines = new LineState;
	}

	~GlobalState()
	{
		delete lines;
	}
};




#endif //GLOBAL_STATE_H
