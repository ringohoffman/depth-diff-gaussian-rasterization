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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "bfloat16.h"
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::bfvec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::bfvec3* means, glm::bfvec3 campos, const bfloat16* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::bfvec3 pos = means[idx];
	glm::bfvec3 dir = pos - campos;
	dir = dir / hsqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);

	glm::bfvec3* sh = ((glm::bfvec3*)shs) + idx * max_coeffs;
	glm::bfvec3 result = __float2bfloat16(SH_C0) * sh[0];
	const bfloat16 sh_c1 = __float2bfloat16(SH_C1);
	const bfloat16 sh_c2[] = {
		__float2bfloat16(SH_C2[0]),
		__float2bfloat16(SH_C2[1]),
		__float2bfloat16(SH_C2[2]),
		__float2bfloat16(SH_C2[3]),
		__float2bfloat16(SH_C2[4]),
	};
	const bfloat16 sh_c3[] = {
		__float2bfloat16(SH_C3[0]),
		__float2bfloat16(SH_C3[1]),
		__float2bfloat16(SH_C3[2]),
		__float2bfloat16(SH_C3[3]),
		__float2bfloat16(SH_C3[4]),
		__float2bfloat16(SH_C3[5]),
		__float2bfloat16(SH_C3[6]),
	};
	if (deg > 0)
	{
		bfloat16 x = dir.x;
		bfloat16 y = dir.y;
		bfloat16 z = dir.z;
		result = result - sh_c1 * y * sh[1] + sh_c1 * z * sh[2] - sh_c1 * x * sh[3];

		if (deg > 1)
		{
			bfloat16 xx = x * x, yy = y * y, zz = z * z;
			bfloat16 xy = x * y, yz = y * z, xz = x * z;
			result = result +
				sh_c2[0] * xy * sh[4] +
				sh_c2[1] * yz * sh[5] +
				sh_c2[2] * (__float2bfloat16(2.0f) * zz - xx - yy) * sh[6] +
				sh_c2[3] * xz * sh[7] +
				sh_c2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					sh_c3[0] * y * (__float2bfloat16(3.0f) * xx - yy) * sh[9] +
					sh_c3[1] * xy * z * sh[10] +
					sh_c3[2] * y * (__float2bfloat16(4.0f) * zz - xx - yy) * sh[11] +
					sh_c3[3] * z * (__float2bfloat16(2.0f) * zz - __float2bfloat16(3.0f) * xx - __float2bfloat16(3.0f) * yy) * sh[12] +
					sh_c3[4] * x * (__float2bfloat16(4.0f) * zz - xx - yy) * sh[13] +
					sh_c3[5] * z * (xx - yy) * sh[14] +
					sh_c3[6] * x * (xx - __float2bfloat16(3.0f)* yy) * sh[15];
			}
		}
	}
	result += __float2bfloat16(0.5f);

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < __float2bfloat16(0.0f));
	clamped[3 * idx + 1] = (result.y < __float2bfloat16(0.0f));
	clamped[3 * idx + 2] = (result.z < __float2bfloat16(0.0f));
	// TODO: is this what glm::max does?
	result[0] = __hmax(result[0], __float2bfloat16(0.0f));
	result[1] = __hmax(result[1], __float2bfloat16(0.0f));
	result[2] = __hmax(result[2], __float2bfloat16(0.0f));
	return result;
}

// Forward version of 2D covariance matrix computation
__device__ bfloat163 computeCov2D(const bfloat163& mean, bfloat16 focal_x, bfloat16 focal_y, bfloat16 tan_fovx, bfloat16 tan_fovy, const bfloat16* cov3D, const bfloat16* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	bfloat163 t = transformPoint4x3(mean, viewmatrix);

	const bfloat16 limx = __float2bfloat16(1.3f) * tan_fovx;
	const bfloat16 limy = __float2bfloat16(1.3f) * tan_fovy;
	const bfloat16 txtz = t.x / t.z;
	const bfloat16 tytz = t.y / t.z;
	t.x = __hmin(limx, __hmax(-limx, txtz)) * t.z;
	t.y = __hmin(limy, __hmax(-limy, tytz)) * t.z;

	glm::bfmat3 J = glm::bfmat3(
		focal_x / t.z, __float2bfloat16(0.0f), -(focal_x * t.x) / (t.z * t.z),
		__float2bfloat16(0.0f), focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		__float2bfloat16(0.0f), __float2bfloat16(0.0f), __float2bfloat16(0.0f));

	glm::bfmat3 W = glm::bfmat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::bfmat3 T = W * J;

	glm::bfmat3 Vrk = glm::bfmat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::bfmat3 cov = glm::bfmat3(
		T[0][0], T[1][0], T[2][0],
		T[0][1], T[1][1], T[2][1],
		T[0][2], T[1][2], T[2][2]
	) * glm::bfmat3(
		Vrk[0][0], Vrk[1][0], Vrk[2][0],
		Vrk[0][1], Vrk[1][1], Vrk[2][1],
		Vrk[0][2], Vrk[1][2], Vrk[2][2]
	) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += __float2bfloat16(0.3f);
	cov[1][1] += __float2bfloat16(0.3f);
	return { bfloat16(cov[0][0]), bfloat16(cov[0][1]), bfloat16(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::bfvec3 scale, bfloat16 mod, const glm::bfvec4 rot, bfloat16* cov3D)
{
	// Create scaling matrix
	glm::bfmat3 S = glm::bfmat3(
		mod * scale.x, __float2bfloat16(0.0f), __float2bfloat16(0.0f),
		__float2bfloat16(0.0f), mod * scale.y, __float2bfloat16(0.0f),
		__float2bfloat16(0.0f), __float2bfloat16(0.0f), mod * scale.z
	);

	// Normalize quaternion to get valid rotation
	glm::bfvec4 q = rot;// / glm::length(rot);
	bfloat16 r = q.x;
	bfloat16 x = q.y;
	bfloat16 y = q.z;
	bfloat16 z = q.w;

	// Compute rotation matrix from quaternion
	glm::bfmat3 R = glm::bfmat3(
		__float2bfloat16(1.0f) - __float2bfloat16(2.0f) * (y * y + z * z), __float2bfloat16(2.0f) * (x * y - r * z), __float2bfloat16(2.0f) * (x * z + r * y),
		__float2bfloat16(2.0f) * (x * y + r * z), __float2bfloat16(1.0f) - __float2bfloat16(2.0f) * (x * x + z * z), __float2bfloat16(2.0f) * (y * z - r * x),
		__float2bfloat16(2.0f) * (x * z - r * y), __float2bfloat16(2.0f) * (y * z + r * x), __float2bfloat16(1.0f) - __float2bfloat16(2.0f) * (x * x + y * y)
	);

	glm::bfmat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::bfmat3 Sigma = glm::bfmat3(
		M[0][0], M[1][0], M[2][0],
		M[0][1], M[1][1], M[2][1],
		M[0][2], M[1][2], M[2][2]
	) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const bfloat16* orig_points,
	const glm::bfvec3* scales,
	const bfloat16 scale_modifier,
	const glm::bfvec4* rotations,
	const bfloat16* opacities,
	const bfloat16* shs,
	bool* clamped,
	const bfloat16* cov3D_precomp,
	const bfloat16* colors_precomp,
	const bfloat16* viewmatrix,
	const bfloat16* projmatrix,
	const glm::bfvec3* cam_pos,
	const int W, int H,
	const bfloat16 tan_fovx, bfloat16 tan_fovy,
	const bfloat16 focal_x, bfloat16 focal_y,
	int* radii,
	bfloat162* points_xy_image,
	bfloat16* depths,
	bfloat16* cov3Ds,
	bfloat16* rgb,
	bfloat164* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	bfloat163 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	bfloat163 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	bfloat164 p_hom = transformPoint4x4(p_orig, projmatrix);
	bfloat16 p_w = hrcp(p_hom.w + __float2bfloat16(0.0000001f));
	bfloat163 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const bfloat16* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	bfloat163 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	bfloat16 det = (cov.x * cov.z - cov.y * cov.y);
	if (det == __float2bfloat16(0.0f))
		return;
	bfloat16 det_inv = hrcp(det);
	bfloat163 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	bfloat16 mid = __float2bfloat16(0.5f) * (cov.x + cov.z);
	bfloat16 lambda1 = mid + hsqrt(__hmax(__float2bfloat16(0.1f), mid * mid - det));
	bfloat16 lambda2 = mid - hsqrt(__hmax(__float2bfloat16(0.1f), mid * mid - det));
	bfloat16 my_radius = hceil(__float2bfloat16(3.0f) * hsqrt(__hmax(lambda1, lambda2)));
	bfloat162 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, __bfloat162int_rn(my_radius), rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::bfvec3 result = computeColorFromSH(idx, D, M, (glm::bfvec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = __bfloat162int_rn(my_radius);
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one bfloat164
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const bfloat162* __restrict__ points_xy_image,
	const bfloat16* __restrict__ features,
	const bfloat16* __restrict__ depths,
	const bfloat164* __restrict__ conic_opacity,
	bfloat16* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const bfloat16* __restrict__ bg_color,
	bfloat16* __restrict__ out_color,
	bfloat16* __restrict__ out_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	bfloat162 pixf = { __uint2bfloat16_rn(pix.x), __uint2bfloat16_rn(pix.y) };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ bfloat162 collected_xy[BLOCK_SIZE];
	__shared__ bfloat164 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	bfloat16 T = __float2bfloat16(1.0f);
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	bfloat16 C[CHANNELS] = { __float2bfloat16(0.0f) };
	bfloat16 D = { __float2bfloat16(0.0f) };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			bfloat162 xy = collected_xy[j];
			bfloat162 d = { xy.x - pixf.x, xy.y - pixf.y };
			bfloat164 con_o = collected_conic_opacity[j];
			bfloat16 power = -__float2bfloat16(0.5f) * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > __float2bfloat16(0.0f))
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			bfloat16 alpha = __hmin(__float2bfloat16(0.99f), con_o.w * hexp(power));
			if (alpha < __float2bfloat16(1.0f) / __float2bfloat16(255.0f))
				continue;
			bfloat16 test_T = T * (__float2bfloat16(1.0f) - alpha);
			if (test_T < __float2bfloat16(0.0001f))
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			D += depths[collected_id[j]] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const bfloat162* means2D,
	const bfloat16* colors,
	const bfloat16* depths,
	const bfloat164* conic_opacity,
	bfloat16* final_T,
	uint32_t* n_contrib,
	const bfloat16* bg_color,
	bfloat16* out_color,
	bfloat16* out_depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const bfloat16* means3D,
	const glm::bfvec3* scales,
	const bfloat16 scale_modifier,
	const glm::bfvec4* rotations,
	const bfloat16* opacities,
	const bfloat16* shs,
	bool* clamped,
	const bfloat16* cov3D_precomp,
	const bfloat16* colors_precomp,
	const bfloat16* viewmatrix,
	const bfloat16* projmatrix,
	const glm::bfvec3* cam_pos,
	const int W, int H,
	const bfloat16 focal_x, bfloat16 focal_y,
	const bfloat16 tan_fovx, bfloat16 tan_fovy,
	int* radii,
	bfloat162* means2D,
	bfloat16* depths,
	bfloat16* cov3Ds,
	bfloat16* rgb,
	bfloat164* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
