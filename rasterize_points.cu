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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/bfloat16.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kBFloat16);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, int_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		(bfloat16 *)background.contiguous().data<at::BFloat16>(),
		W, H,
		(bfloat16 *)means3D.contiguous().data<at::BFloat16>(),
		(bfloat16 *)sh.contiguous().data_ptr<at::BFloat16>(),
		(bfloat16 *)colors.contiguous().data<at::BFloat16>(), 
		(bfloat16 *)opacity.contiguous().data<at::BFloat16>(), 
		(bfloat16 *)scales.contiguous().data_ptr<at::BFloat16>(),
		__float2bfloat16(scale_modifier),
		(bfloat16 *)rotations.contiguous().data_ptr<at::BFloat16>(),
		(bfloat16 *)cov3D_precomp.contiguous().data<at::BFloat16>(), 
		(bfloat16 *)viewmatrix.contiguous().data<at::BFloat16>(), 
		(bfloat16 *)projmatrix.contiguous().data<at::BFloat16>(),
		(bfloat16 *)campos.contiguous().data<at::BFloat16>(),
		__float2bfloat16(tan_fovx),
		__float2bfloat16(tan_fovy),
		prefiltered,
		(bfloat16 *)out_color.contiguous().data<at::BFloat16>(),
		(bfloat16 *)out_depth.contiguous().data<at::BFloat16>(),
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  (bfloat16 *)background.contiguous().data<at::BFloat16>(),
	  W, H, 
	  (bfloat16 *)means3D.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)sh.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)colors.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)scales.data_ptr<at::BFloat16>(),
	  __float2bfloat16(scale_modifier),
	  (bfloat16 *)rotations.data_ptr<at::BFloat16>(),
	  (bfloat16 *)cov3D_precomp.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)viewmatrix.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)projmatrix.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)campos.contiguous().data<at::BFloat16>(),
	  __float2bfloat16(tan_fovx),
	  __float2bfloat16(tan_fovy),
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  (bfloat16 *)dL_dout_color.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_dmeans2D.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_dconic.contiguous().data<at::BFloat16>(),  
	  (bfloat16 *)dL_dopacity.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_dcolors.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_dmeans3D.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_dcov3D.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_dsh.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_dscales.contiguous().data<at::BFloat16>(),
	  (bfloat16 *)dL_drotations.contiguous().data<at::BFloat16>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		(bfloat16 *)means3D.contiguous().data<at::BFloat16>(),
		(bfloat16 *)viewmatrix.contiguous().data<at::BFloat16>(),
		(bfloat16 *)projmatrix.contiguous().data<at::BFloat16>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
