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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "bfloat16.h"

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const bfloat16* bg_color,
		const bfloat162* means2D,
		const bfloat164* conic_opacity,
		const bfloat16* colors,
		const bfloat16* final_Ts,
		const uint32_t* n_contrib,
		const bfloat16* dL_dpixels,
		bfloat163* dL_dmean2D,
		bfloat164* dL_dconic2D,
		bfloat16* dL_dopacity,
		bfloat16* dL_dcolors);

	void preprocess(
		int P, int D, int M,
		const bfloat163* means,
		const int* radii,
		const bfloat16* shs,
		const bool* clamped,
		const glm::bfvec3* scales,
		const glm::bfvec4* rotations,
		const bfloat16 scale_modifier,
		const bfloat16* cov3Ds,
		const bfloat16* view,
		const bfloat16* proj,
		const bfloat16 focal_x, bfloat16 focal_y,
		const bfloat16 tan_fovx, bfloat16 tan_fovy,
		const glm::bfvec3* campos,
		const bfloat163* dL_dmean2D,
		const bfloat16* dL_dconics,
		glm::bfvec3* dL_dmeans,
		bfloat16* dL_dcolor,
		bfloat16* dL_dcov3D,
		bfloat16* dL_dsh,
		glm::bfvec3* dL_dscale,
		glm::bfvec4* dL_drot);
}

#endif