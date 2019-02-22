#include "PatchMatch.h"

#include <thrust/device_vector.h>



#define pow2(x) (x) * (x)



/* static __device__ float4 operator*(float4 a, float4 b) */
/* { */
/*     return make_float4(a.x * b.x, */
/*                        a.y * b.y, */
/*                        a.z * b.z, */
/*                        0); */
/* } */
static __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x,
                       a.y - b.y,
                       a.z - b.z,
                       0);
}
/* static __device__ float4 operator-(float4 a) */
/* { */
/*     return make_float4(-a.x, */
/*                        -a.y, */
/*                        -a.z, */
/*                        0); */
/* } */
static __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x,
                       a.y + b.y,
                       a.z + b.z,
                       a.w + b.w);
}
/* static __device__ float4 operator/(float4 a, float k) */
/* { */
/*     return make_float4(a.x / k, */
/*                        a.y / k, */
/*                        a.z / k, */
/*                        0); */
/* } */
/* static __device__ float l1_float4 (float4 a) */
/* { */
/*     return ( fabsf (a.x) + */
/*              fabsf (a.y) + */
/*              fabsf (a.z)) * 0.3333333f; */

/* } */


__device__ float curand_between(curandState* cs, const float& min, const float& max)
{
	return (curand_uniform(cs) * (max - min) + min);
}
__device__ void Normalize(float4* __restrict__ v)
{
	const float norm_squared = pow2(v->x) + pow2(v->y) + pow2(v->z);
	const float inverse_sqrt = rsqrtf(norm_squared);
	v->x *= inverse_sqrt;
	v->y *= inverse_sqrt;
	v->z *= inverse_sqrt;
}

__device__ float l1_norm(float f)
{
	return fabsf(f);
}

__device__ float l1_norm(float4 f)
{
	return (fabsf(f.x) + fabsf(f.y) + fabsf(f.z)) * 0.3333333f;
}


template<typename T>
__device__ float Weight_cu(const T& c1, const T& c2, const float& gamma)
{
	const float color_dis = l1_norm(c1 - c2);
	return expf( -color_dis / gamma );
}


template<typename T>
__device__ float  CostComputation_cu(const cudaTextureObject_t& l, 
								  const cudaTextureObject_t& r,
								  const float4& pt_l,
								  const float4& pt_r,
								  const float& tau_color,
								  const float& tau_gradient,
								  const float& alpha,
								  const float& w)
{

	const float col_diff = l1_norm(tex2D<T>(l, pt_l.x + 0.5f, pt_l.y + 0.5f) - tex2D<T>(r, pt_r.x + 0.5f, pt_r.y + 0.5f));
	/* const float col_dis = fminf(col_diff, tau_color); */
	const float col_dis = fmin(col_diff, tau_color);

	const T gx1 = tex2D<T>(l, pt_l.x + 1 + 0.5f, pt_l.y +     0.5f) - tex2D<T>(l, pt_l.x - 1 + 0.5f, pt_l.y     + 0.5f);
	const T gy1 = tex2D<T>(l, pt_l.x     + 0.5f, pt_l.y + 1 + 0.5f) - tex2D<T>(l, pt_l.x     + 0.5f, pt_l.y - 1 + 0.5f);

	const T gx2 = tex2D<T>(r, pt_r.x + 1 + 0.5f, pt_r.y +     0.5f) - tex2D<T>(r, pt_r.x - 1 + 0.5f, pt_r.y     + 0.5f);
	const T gy2 = tex2D<T>(r, pt_r.x     + 0.5f, pt_r.y + 1 + 0.5f) - tex2D<T>(r, pt_r.x     + 0.5f, pt_r.y - 1 + 0.5f);

	const T grad_x_diff = (gx1 - gx2);
	const T grad_y_diff = (gy1 - gy2);

	const float grad_dis = fmin( (l1_norm(grad_x_diff) + l1_norm(grad_y_diff)) * 0.0625, tau_gradient);
	/* const float grad_dis = fminf( (l1_norm(grad_x_diff) + l1_norm(grad_y_diff)) * 0.0625, tau_gradient); */

	const float dis = (1.f - alpha) * col_dis + alpha * grad_dis;

	/* return dis; */
	return w * dis;
}


template<typename T>
__device__ float EvaluatePlaneCost_cu(const int2 p, float4 normal, GlobalState& gs)
{

	const float a = -normal.x / normal.z;
	const float b = -normal.y / normal.z;
	const float c = (normal.x * p.x + normal.y * p.y + normal.z * normal.w) / normal.z;

	const int h_rad = gs.params->box_width / 2;
	const int v_rad = gs.params->box_height / 2;

	const int sign = gs.params->sign;

	float cost = 0;
	for(int i = -v_rad; i <= v_rad; ++i)
	{
		for(int j = -h_rad; j <= h_rad; ++j)
		{
			const int win_x = j + p.x;
			const int win_y = i + p.y;
			
			float4 win_pt;
			win_pt.x = __int2float_rn(win_x);
			win_pt.y = __int2float_rn(win_y);

			float disp = a * win_pt.x + b * win_pt.y + c;
			float4 corr_win_pt;
			corr_win_pt.x = win_pt.x + -1 * disp;
			corr_win_pt.y = win_pt.y;

			float w = Weight_cu<T>(tex2D<T>(gs.imgs[0], win_pt.x + 0.5f, win_pt.y + 0.5f), tex2D<T>(gs.imgs[0], p.x + 0.5f, p.y + 0.5f), gs.params->gamma);
			
			cost += CostComputation_cu<T>(gs.imgs[0], gs.imgs[1], win_pt, corr_win_pt, gs.params->tau_color, gs.params->tau_gradient, gs.params->alpha, w);
		}
	}

	return cost;
}


template<typename T>
__global__ void InitializeRandomPlane_cu(GlobalState& gs)
{
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int rows = gs.lines->rows;
	const int cols = gs.lines->cols;
	const float min_disparity = gs.params->min_disparity;
	const float max_disparity = gs.params->max_disparity;

	if(p.x >= cols)
		return;
	if(p.y >= rows)
		return;
	const int center = p.y * cols + p.x;
	curandState local_state = gs.cs[center];
	curand_init(clock64(), p.y, p.x, &local_state);
	float4 norm;
	/* norm.x = curand_between(&local_state, -1.0f, 1.0f); */
	/* norm.y = curand_between(&local_state, -1.0f, 1.0f); */
	/* norm.z = curand_between(&local_state, -1.0f, 1.0f); */
	norm.x = curand_between(&local_state, 0.0f, 1.0f);
	norm.y = curand_between(&local_state, 0.0f, 1.0f);
	norm.z = curand_between(&local_state, 0.0f, 1.0f);
	Normalize(&norm);
	norm.w = curand_between(&local_state, min_disparity, max_disparity);

	gs.lines->cost[center] = EvaluatePlaneCost_cu<T>(p, norm, gs);
	gs.lines->norm4[center] = norm;
}

template<typename T>
__device__ void SpatialPropagation_cu(GlobalState& gs, const int2& pt1, const int2& pt2)
{
	const int center1 = pt1.y * gs.lines->cols + pt1.x;
	const int center2 = pt2.y * gs.lines->cols + pt2.x;
	const float4 new_norm = gs.lines->norm4[center2];
	const float new_cost = EvaluatePlaneCost_cu<T>(pt1, new_norm, gs);
	const float old_cost = gs.lines->cost[center1];
	if(new_cost < old_cost)
	{
		gs.lines->cost[center1] = new_cost;
		gs.lines->norm4[center1] = new_norm;
	}
}

template<typename T>
__device__ void CheckboardSpatialPropClose_cu(GlobalState& gs, const int2& pt)
{

	if(pt.x >= gs.lines->cols)
		return;
	if(pt.y >= gs.lines->rows)
		return;

	// up
	if(pt.y > 0)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x, pt.y - 1));
	}
	if(pt.y - 5 >= 0)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x, pt.y - 5));
	}

	// down
	if(pt.y < gs.lines->rows - 1)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x, pt.y + 1));
	}
	if(pt.y + 5< gs.lines->rows)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x, pt.y + 5));
	}

	// left
	if(pt.x > 0)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x - 1, pt.y));
	}
	if(pt.x  - 5 >= 0)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x - 5, pt.y));
	}
	// right
	if(pt.x < gs.lines->cols - 1)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x + 1, pt.y));
	}
	if(pt.x + 5 < gs.lines->cols)
	{
		SpatialPropagation_cu<T>(gs, pt, make_int2(pt.x + 5, pt.y));
	}
}

template<typename T>
__global__ void RedSpatialPropClose_cu(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(threadIdx.x % 2 == 0)
		pt.y = pt.y * 2 + 1;
	else
		pt.y = pt.y * 2;
	CheckboardSpatialPropClose_cu<T>(gs, pt);
}

template<typename T>
__global__ void BlackSpatialPropClose_cu(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(threadIdx.x % 2 == 0)
		pt.y = pt.y * 2;
	else
		pt.y = pt.y * 2 + 1;
	CheckboardSpatialPropClose_cu<T>(gs, pt);
}

template<typename T>
__device__ void CheckboardPlaneRefinement_cu(GlobalState& gs, const int2& pt)
{
	if(pt.x >= gs.lines->cols)
		return;
	if(pt.y >= gs.lines->rows)
		return;

	const int center = pt.y * gs.lines->cols + pt.x;
	const float4 old_norm = gs.lines->norm4[center];

	float max_delta_z = gs.params->max_disparity;
	float max_delta_n = 1.0f;
	float end_z = 0.1f;

	while(max_delta_z >= end_z)
	{
		curandState local_state = gs.cs[center];
		curand_init(clock64(), pt.y, pt.x, &local_state);

		float4 delta_norm;
		delta_norm.x = curand_between(&local_state, -max_delta_n, max_delta_n);
		delta_norm.y = curand_between(&local_state, -max_delta_n, max_delta_n);
		delta_norm.z = curand_between(&local_state, -max_delta_n, max_delta_n);
		delta_norm.w = curand_between(&local_state, -max_delta_z, max_delta_z);
		
		float4 new_norm = old_norm + delta_norm;
		Normalize(&new_norm);
		float new_cost = EvaluatePlaneCost_cu<T>(pt, new_norm, gs);
		float old_cost = gs.lines->cost[center];

		if(new_cost < old_cost)
		{
			gs.lines->cost[center] = new_cost;
			gs.lines->norm4[center] = new_norm;
		}

		max_delta_z /= 2.0f;
		max_delta_n /= 2.0f;

	}
	
}

template<typename T>
__global__ void RedPlaneRefinement_cu(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(threadIdx.x % 2 == 0)
		pt.y = pt.y * 2 + 1;
	else
		pt.y = pt.y * 2;
	CheckboardPlaneRefinement_cu<T>(gs, pt);
}

template<typename T>
__global__ void BlackPlaneRefinement_cu(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(threadIdx.x % 2 == 0)
		pt.y = pt.y * 2;
	else
		pt.y = pt.y * 2 + 1;
	CheckboardPlaneRefinement_cu<T>(gs, pt);
}


__global__ void ComputeDisparityMat_cu(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(pt.x >= gs.lines->cols)
		return;
	if(pt.y >= gs.lines->rows)
		return;

	const int center = pt.y * gs.lines->cols + pt.x;
	float4 normal = gs.lines->norm4[center];

	const float a = -normal.x / normal.z;
	const float b = -normal.y / normal.z;
	const float c = (normal.x * pt.x + normal.y * pt.y + normal.z * normal.w) / normal.z;

	float disp = a * pt.x + b * pt.y + c;

	gs.lines->norm4[center].w = disp;
}


__device__ float VecAverage(const float& x, const float& y, float wm)
{
	return wm * x + (1 - wm ) * y;
}

__global__ void CheckValidity_cu(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(pt.x >= gs.lines->cols)
		return;
	if(pt.y >= gs.lines->rows)
		return;

	const int center = pt.y * gs.lines->cols + pt.x;
	const float4 normal = gs.lines->norm4[center];
	const float disp = normal.w;
	float corr_x = pt.x - disp;
	int corr_x_i = (int)corr_x;
	float wm = 1 - (corr_x - corr_x_i);

	if(corr_x_i > gs.lines->cols - 2)
		corr_x_i = gs.lines->cols - 2;
	if(corr_x_i < 0)
		corr_x_i = 0;

	float corr_disp1 = gs.lines->norm4[pt.y * gs.lines->cols + corr_x_i].w;
	float corr_disp2 = gs.lines->norm4[pt.y * gs.lines->cols + corr_x_i + 1].w;
	float corr_disp = VecAverage(corr_disp1, corr_disp2, wm);
	/* float corr_disp = VecAverage(gs.lines->norm4[pt.y * gs.lines->cols + corr_x_i].w, gs.lines->norm4[pt.y * gs.lines.cols + corr_x_i + 1].w, wm); */
	gs.lines->validity[center] = fabsf(disp - corr_disp) <= 1;
}

__global__ void FillInvalidPixel(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(pt.x >= gs.lines->cols)
		return;
	if(pt.y >= gs.lines->rows)
		return;
	int rows = gs.lines->rows;
	int cols = gs.lines->cols;
	if(gs.lines->validity[pt.y * cols + pt.x])
		return;

	int x_left = pt.x - 1;
	int x_right = pt.x + 1;
	int y_up = pt.y - 1;
	int y_down = pt.y + 1;

	while(x_left >= 0 && !gs.lines->validity[pt.y * cols + x_left])
		--x_left;

	while(x_right <= cols && !gs.lines->validity[pt.y * cols + x_right])
		++x_right;

	while(y_up >= 0 && !gs.lines->validity[y_up * cols + pt.x])
		--y_up;

	while(y_down <= rows && !gs.lines->validity[y_down * cols + pt.x])
		++y_down;

	float disp_left = (x_left >= 0) ? gs.lines->norm4[pt.y * cols + x_left].w : (1 << 30);
	float disp_right = (x_right < cols) ? gs.lines->norm4[pt.y * cols + x_right].w : (1 << 30);
	float disp_up = (y_up >= 0) ? gs.lines->norm4[y_up * cols + pt.x].w : (1 << 30);
	float disp_down = (y_down < rows) ? gs.lines->norm4[y_down * cols + pt.x].x : (1<<30);

	gs.lines->norm4[pt.y * cols + pt.x].w = fminf(fminf(disp_left, disp_right), fminf(disp_up, disp_down));

}


template<typename T>
__global__ void WeightedMedianFilter(GlobalState& gs)
{
	int2 pt = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if(pt.x >= gs.lines->cols)
		return;
	if(pt.y >= gs.lines->rows)
		return;

	const int h_rad = gs.params->box_width / 2;
	const int v_rad = gs.params->box_height / 2;

	float total_w = 0;
	/* thrust::device_vector<float> keys; */
	thrust::device_vector<float> values;
	int keys[5] = {1,2,3,4,5};
	thrust::sort(thrust::device, keys, keys + 5);
	/* for(int i = -v_rad; i <= v_rad; ++i) */
	/* { */
	/* 	for(int j = -h_rad; j <= h_rad; ++j) */
	/* 	{ */
	/* 		const int win_x = j + pt.x; */
	/* 		const int win_y = i + pt.y; */

	/* 		if(win_x < 0 || win_y < 0 || win_x >= gs.lines->cols || win_y >= gs.lines->rows) */
	/* 			continue; */
			
	/* 		float4 win_pt; */
	/* 		win_pt.x = __int2float_rn(win_x); */
	/* 		win_pt.y = __int2float_rn(win_y); */

	/* 		float w = Weight_cu<T>(tex2D<T>(gs.imgs[0], win_pt.x + 0.5f, win_pt.y + 0.5f), tex2D<T>(gs.imgs[0], pt.x + 0.5f, pt.y + 0.5f), gs.params->gamma); */

	/* 		total_w += w; */

	/* 		keys.push_back(w); */
	/* 		values.push_back(gs.lines->norm4[win_y * gs.lines->cols + win_x].w); */
	/* 	} */
	/* } */
	/* thrust::sort(keys.begin(), keys.end(), values.begin()); */
	/* float w = 0; */
	/* float median_w = total_w / 2.0f; */
	/* for(int i = 0; i < keys.size(); ++i) */
	/* { */
	/* 	w += keys[i]; */
	/* 	if(w >= median_w) */
	/* 	{ */
	/* 		if(i == 0) */
	/* 		{ */
	/* 			ts.lines->norm4[pt.y * gs.lines->cols + pt.x].w = values[i]; */
	/* 		} */
	/* 		else */
	/* 		{ */
	/* 			ts.lines->norm4[pt.y * gs.lines->cols + pt.x].w = (values[i - 1] + values[i]) / 2.0f; */
	/* 		} */
	/* 	} */
	/* } */

}

template<typename T>
void PatchMatch(GlobalState& gs)
{
	int rows = gs.lines->rows;
	int cols = gs.lines->cols;
	

	cudaMalloc(&gs.cs, rows * cols * sizeof(curandState));

	dim3 init_rand_block(32, 16, 1);
	dim3 init_rand_grid((cols + init_rand_block.x - 1) / init_rand_block.x, (rows + init_rand_block.y - 1) / init_rand_block.y, 1);
	InitializeRandomPlane_cu<T><<<init_rand_grid, init_rand_block>>>(gs);



	const int BLOCK_W = 32;
	const int BLOCK_H  = (BLOCK_W / 2);

	dim3 block(BLOCK_W, BLOCK_H, 1);
	dim3 grid((cols + block.x - 1) / block.x, (rows / 2 + block.y - 1) / block.y, 1);

	
	for(int it = 0; it < gs.params->iterations; ++it)
	{
		RedSpatialPropClose_cu<T><<<grid, block>>>(gs);
		cudaDeviceSynchronize();

		RedPlaneRefinement_cu<T><<<grid, block>>>(gs);
		cudaDeviceSynchronize();

		BlackSpatialPropClose_cu<T><<<grid, block>>>(gs);
		cudaDeviceSynchronize();

		BlackPlaneRefinement_cu<T><<<grid, block>>>(gs);
		cudaDeviceSynchronize();
	}
	/* ComputeDisparityMat_cu<<<init_rand_grid, init_rand_block>>>(gs); */
	/* CheckValidity_cu<<<init_rand_grid, init_rand_block>>>(gs); */
	/* FillInvalidPixel<<<init_rand_grid, init_rand_block>>>(gs); */

	/* WeightedMedianFilter<T><<<init_rand_grid, init_rand_block>>>(gs); */
	cudaDeviceSynchronize();


	cudaFree(&gs.cs);
}

void runCuda(GlobalState& gs)
{
	if(gs.params->color_processing)
	{
		PatchMatch<float4>(gs);
	}
	else
	{
		PatchMatch<float>(gs);
	}
}
