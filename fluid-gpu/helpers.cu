#include "helpers.cuh"

__forceinline__ __device__ float squaredNorm(float2 f)
{
	return f.x*f.x + f.y*f.y;
}

__forceinline__ __device__ float2 normalized(float2 f)
{
	float normF = norm(f);
	return make_float2(f.x / normF, f.y / normF);
}

__forceinline__ __device__ float2 norm(float2 f)
{
	return sqrtf(squaredNorm(f));
}
