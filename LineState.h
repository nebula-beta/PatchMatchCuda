#ifndef LINE_STATE_H
#define LINE_STATE_H


#include "UnifiedMemoryManaged.h"


class __align__(128) LineState : public UnifiedMemoryManaged
{
public:
	float4* norm4;
	float* cost;
	bool* validity;
	int rows;
	int cols;

	void resize(int rows_, int cols_)
	{
		rows = rows_;
		cols = cols_;

		int n = rows * cols;
		cudaMallocManaged(&norm4, sizeof(float4) * n);
		cudaMallocManaged(&cost, sizeof(float) * n);
		cudaMallocManaged(&validity, sizeof(bool) * n);
		
		memset(norm4, 0, sizeof(float4) * n);
		memset(cost, 0, sizeof(float) * n);
		memset(validity, 0, sizeof(validity));
	}

	~LineState()
	{
		cudaFree(norm4);
		cudaFree(cost);
	}
};

#endif //LINE_STATE
