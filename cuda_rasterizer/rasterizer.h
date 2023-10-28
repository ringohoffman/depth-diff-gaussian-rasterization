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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include "bfloat16.h"

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			bfloat16* means3D,
			bfloat16* viewmatrix,
			bfloat16* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const bfloat16* background,
			const int width, int height,
			const bfloat16* means3D,
			const bfloat16* shs,
			const bfloat16* colors_precomp,
			const bfloat16* opacities,
			const bfloat16* scales,
			const bfloat16 scale_modifier,
			const bfloat16* rotations,
			const bfloat16* cov3D_precomp,
			const bfloat16* viewmatrix,
			const bfloat16* projmatrix,
			const bfloat16* cam_pos,
			const bfloat16 tan_fovx, bfloat16 tan_fovy,
			const bool prefiltered,
			bfloat16* out_color,
			bfloat16* out_depth,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const bfloat16* background,
			const int width, int height,
			const bfloat16* means3D,
			const bfloat16* shs,
			const bfloat16* colors_precomp,
			const bfloat16* scales,
			const bfloat16 scale_modifier,
			const bfloat16* rotations,
			const bfloat16* cov3D_precomp,
			const bfloat16* viewmatrix,
			const bfloat16* projmatrix,
			const bfloat16* campos,
			const bfloat16 tan_fovx, bfloat16 tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const bfloat16* dL_dpix,
			bfloat16* dL_dmean2D,
			bfloat16* dL_dconic,
			bfloat16* dL_dopacity,
			bfloat16* dL_dcolor,
			bfloat16* dL_dmean3D,
			bfloat16* dL_dcov3D,
			bfloat16* dL_dsh,
			bfloat16* dL_dscale,
			bfloat16* dL_drot,
			bool debug);
	};
};

#endif
