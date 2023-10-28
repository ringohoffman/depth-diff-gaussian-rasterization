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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "bfloat16.h"

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
		const bfloat16 focal_x, bfloat16 focal_y,
		const bfloat16 tan_fovx, bfloat16 tan_fovy,
		int* radii,
		bfloat162* points_xy_image,
		bfloat16* depths,
		bfloat16* cov3Ds,
		bfloat16* colors,
		bfloat164* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const bfloat162* points_xy_image,
		const bfloat16* features,
		const bfloat16* depths,
		const bfloat164* conic_opacity,
		bfloat16* final_T,
		uint32_t* n_contrib,
		const bfloat16* bg_color,
		bfloat16* out_color,
		bfloat16* out_depth);
}


#endif
