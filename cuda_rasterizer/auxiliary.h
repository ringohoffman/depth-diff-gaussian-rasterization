/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include "bfloat16.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ bfloat16 ndc2Pix(bfloat16 v, int S)
{
	return ((v + __float2bfloat16(1.0f)) * __int2bfloat16_rn(S) - __float2bfloat16(1.0f)) * __float2bfloat16(0.5f);
}

__forceinline__ __device__ void getRect(const bfloat162 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((__bfloat162int_rn(p.x) - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((__bfloat162int_rn(p.y) - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((__bfloat162int_rn(p.x) + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((__bfloat162int_rn(p.y) + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ bfloat163 transformPoint4x3(const bfloat163& p, const bfloat16* matrix)
{
	bfloat163 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ bfloat164 transformPoint4x4(const bfloat163& p, const bfloat16* matrix)
{
	bfloat164 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ bfloat163 transformVec4x3(const bfloat163& p, const bfloat16* matrix)
{
	bfloat163 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ bfloat163 transformVec4x3Transpose(const bfloat163& p, const bfloat16* matrix)
{
	bfloat163 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ bfloat16 dnormvdz(bfloat163 v, bfloat163 dv)
{
	bfloat16 sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	bfloat16 invsum32 = hrsqrt(sum2 * sum2 * sum2);
	bfloat16 dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ bfloat163 dnormvdv(bfloat163 v, bfloat163 dv)
{
	bfloat16 sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	bfloat16 invsum32 = hrsqrt(sum2 * sum2 * sum2);

	bfloat163 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ bfloat164 dnormvdv(bfloat164 v, bfloat164 dv)
{
	bfloat16 sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	bfloat16 invsum32 = hrsqrt(sum2 * sum2 * sum2);

	bfloat164 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	bfloat16 vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	bfloat164 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ bfloat16 sigmoid(bfloat16 x)
{
	return hrcp(__float2bfloat16(1.0f) + hexp(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const bfloat16* orig_points,
	const bfloat16* viewmatrix,
	const bfloat16* projmatrix,
	bool prefiltered,
	bfloat163& p_view)
{
	bfloat163 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	bfloat164 p_hom = transformPoint4x4(p_orig, projmatrix);
	bfloat16 p_w = hrcp(p_hom.w + __float2bfloat16(0.0000001f));
	bfloat163 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= __float2bfloat16(0.2f))// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif