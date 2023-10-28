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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "bfloat16.h"
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::bfvec3* means, glm::bfvec3 campos, const bfloat16* shs, const bool* clamped, const glm::bfvec3* dL_dcolor, glm::bfvec3* dL_dmeans, glm::bfvec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::bfvec3 pos = means[idx];
	glm::bfvec3 dir_orig = pos - campos;
	glm::bfvec3 dir = dir_orig / hsqrt(dir_orig[0] * dir_orig[0] + dir_orig[1] * dir_orig[1] + dir_orig[2] * dir_orig[2]);

	glm::bfvec3* sh = ((glm::bfvec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::bfvec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? __float2bfloat16(0.0f) : __float2bfloat16(1.0f);
	dL_dRGB.y *= clamped[3 * idx + 1] ? __float2bfloat16(0.0f) : __float2bfloat16(1.0f);
	dL_dRGB.z *= clamped[3 * idx + 2] ? __float2bfloat16(0.0f) : __float2bfloat16(1.0f);

	glm::bfvec3 dRGBdx(__float2bfloat16(0.0f), __float2bfloat16(0.0f), __float2bfloat16(0.0f));
	glm::bfvec3 dRGBdy(__float2bfloat16(0.0f), __float2bfloat16(0.0f), __float2bfloat16(0.0f));
	glm::bfvec3 dRGBdz(__float2bfloat16(0.0f), __float2bfloat16(0.0f), __float2bfloat16(0.0f));
	bfloat16 x = dir.x;
	bfloat16 y = dir.y;
	bfloat16 z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::bfvec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	const bfloat16 dRGBdsh0 = __float2bfloat16(SH_C0);
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
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		bfloat16 dRGBdsh1 = -sh_c1 * y;
		bfloat16 dRGBdsh2 = sh_c1 * z;
		bfloat16 dRGBdsh3 = -sh_c1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -sh_c1 * sh[3];
		dRGBdy = -sh_c1 * sh[1];
		dRGBdz = sh_c1 * sh[2];

		if (deg > 1)
		{
			bfloat16 xx = x * x, yy = y * y, zz = z * z;
			bfloat16 xy = x * y, yz = y * z, xz = x * z;

			bfloat16 dRGBdsh4 = sh_c2[0] * xy;
			bfloat16 dRGBdsh5 = sh_c2[1] * yz;
			bfloat16 dRGBdsh6 = sh_c2[2] * (__float2bfloat16(2.0f) * zz - xx - yy);
			bfloat16 dRGBdsh7 = sh_c2[3] * xz;
			bfloat16 dRGBdsh8 = sh_c2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += sh_c2[0] * y * sh[4] + sh_c2[2] * __float2bfloat16(2.0f) * -x * sh[6] + sh_c2[3] * z * sh[7] + sh_c2[4] * __float2bfloat16(2.0f) * x * sh[8];
			dRGBdy += sh_c2[0] * x * sh[4] + sh_c2[1] * z * sh[5] + sh_c2[2] * __float2bfloat16(2.0f) * -y * sh[6] + sh_c2[4] * __float2bfloat16(2.0f) * -y * sh[8];
			dRGBdz += sh_c2[1] * y * sh[5] + sh_c2[2] * __float2bfloat16(2.0f) * __float2bfloat16(2.0f) * z * sh[6] + sh_c2[3] * x * sh[7];

			if (deg > 2)
			{
				bfloat16 dRGBdsh9 = sh_c3[0] * y * (__float2bfloat16(3.0f) * xx - yy);
				bfloat16 dRGBdsh10 = sh_c3[1] * xy * z;
				bfloat16 dRGBdsh11 = sh_c3[2] * y * (__float2bfloat16(4.0f) * zz - xx - yy);
				bfloat16 dRGBdsh12 = sh_c3[3] * z * (__float2bfloat16(2.0f) * zz - __float2bfloat16(3.0f) * xx - __float2bfloat16(3.0f) * yy);
				bfloat16 dRGBdsh13 = sh_c3[4] * x * (__float2bfloat16(4.0f) * zz - xx - yy);
				bfloat16 dRGBdsh14 = sh_c3[5] * z * (xx - yy);
				bfloat16 dRGBdsh15 = sh_c3[6] * x * (xx - __float2bfloat16(3.0f) * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					sh_c3[0] * sh[9] * __float2bfloat16(3.0f) * __float2bfloat16(2.0f) * xy +
					sh_c3[1] * sh[10] * yz +
					sh_c3[2] * sh[11] * -__float2bfloat16(2.0f) * xy +
					sh_c3[3] * sh[12] * -__float2bfloat16(3.0f) * __float2bfloat16(2.0f) * xz +
					sh_c3[4] * sh[13] * (-__float2bfloat16(3.0f) * xx + __float2bfloat16(4.0f) * zz - yy) +
					sh_c3[5] * sh[14] * __float2bfloat16(2.0f) * xz +
					sh_c3[6] * sh[15] * __float2bfloat16(3.0f) * (xx - yy));

				dRGBdy += (
					sh_c3[0] * sh[9] * __float2bfloat16(3.0f) * (xx - yy) +
					sh_c3[1] * sh[10] * xz +
					sh_c3[2] * sh[11] * (-__float2bfloat16(3.0f) * yy + __float2bfloat16(4.0f) * zz - xx) +
					sh_c3[3] * sh[12] * -__float2bfloat16(3.0f) * __float2bfloat16(2.0f) * yz +
					sh_c3[4] * sh[13] * -__float2bfloat16(2.0f) * xy +
					sh_c3[5] * sh[14] * -__float2bfloat16(2.0f) * yz +
					sh_c3[6] * sh[15] * -__float2bfloat16(3.0f) * __float2bfloat16(2.0f) * xy);

				dRGBdz += (
					sh_c3[1] * sh[10] * xy +
					sh_c3[2] * sh[11] * __float2bfloat16(4.0f) * __float2bfloat16(2.0f) * yz +
					sh_c3[3] * sh[12] * __float2bfloat16(3.0f) * (__float2bfloat16(2.0f) * zz - xx - yy) +
					sh_c3[4] * sh[13] * __float2bfloat16(4.0f) * __float2bfloat16(2.0f) * xz +
					sh_c3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::bfvec3 dL_ddir(
		dRGBdx[0] * dL_dRGB[0] + dRGBdx[1] * dL_dRGB[1] + dRGBdx[2] * dL_dRGB[2],
		dRGBdy[0] * dL_dRGB[0] + dRGBdy[1] * dL_dRGB[1] + dRGBdy[2] * dL_dRGB[2],
		dRGBdz[0] * dL_dRGB[0] + dRGBdz[1] * dL_dRGB[1] + dRGBdz[2] * dL_dRGB[2]
	);

	// Account for normalization of direction
	bfloat163 dL_dmean = dnormvdv(bfloat163{ dir_orig.x, dir_orig.y, dir_orig.z }, bfloat163{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::bfvec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const bfloat163* means,
	const int* radii,
	const bfloat16* cov3Ds,
	const bfloat16 h_x, bfloat16 h_y,
	const bfloat16 tan_fovx, bfloat16 tan_fovy,
	const bfloat16* view_matrix,
	const bfloat16* dL_dconics,
	bfloat163* dL_dmeans,
	bfloat16* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const bfloat16* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	bfloat163 mean = means[idx];
	bfloat163 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	bfloat163 t = transformPoint4x3(mean, view_matrix);
	
	const bfloat16 limx = __float2bfloat16(1.3f) * tan_fovx;
	const bfloat16 limy = __float2bfloat16(1.3f) * tan_fovy;
	const bfloat16 txtz = t.x / t.z;
	const bfloat16 tytz = t.y / t.z;
	t.x = __hmin(limx, __hmax(-limx, txtz)) * t.z;
	t.y = __hmin(limy, __hmax(-limy, tytz)) * t.z;
	
	const bfloat16 x_grad_mul = txtz < -limx || txtz > limx ? __float2bfloat16(0.0f) : __float2bfloat16(1.0f);
	const bfloat16 y_grad_mul = tytz < -limy || tytz > limy ? __float2bfloat16(0.0f) : __float2bfloat16(1.0f);

	glm::bfmat3 J = glm::bfmat3(h_x / t.z, __float2bfloat16(0.0f), -(h_x * t.x) / (t.z * t.z),
		__float2bfloat16(0.0f), h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		__float2bfloat16(0.0f), __float2bfloat16(0.0f), __float2bfloat16(0.0f));

	glm::bfmat3 W = glm::bfmat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::bfmat3 Vrk = glm::bfmat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::bfmat3 T = W * J;

	glm::bfmat3 cov2D = glm::bfmat3(
		T[0][0], T[1][0], T[2][0],
		T[0][1], T[1][1], T[2][1],
		T[0][2], T[1][2], T[2][2]
	) * glm::bfmat3(
		Vrk[0][0], Vrk[1][0], Vrk[2][0],
		Vrk[0][1], Vrk[1][1], Vrk[2][1],
		Vrk[0][2], Vrk[1][2], Vrk[2][2]
	) * T;

	// Use helper variables for 2D covariance entries. More compact.
	bfloat16 a = cov2D[0][0] += __float2bfloat16(0.3f);
	bfloat16 b = cov2D[0][1];
	bfloat16 c = cov2D[1][1] += __float2bfloat16(0.3f);

	bfloat16 denom = a * c - b * b;
	bfloat16 dL_da = __float2bfloat16(0.0f), dL_db = __float2bfloat16(0.0f), dL_dc = __float2bfloat16(0.0f);
	bfloat16 denom2inv = hrcp((denom * denom) + __float2bfloat16(0.0000001f));

	if (denom2inv != __float2bfloat16(0.0f))
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + __float2bfloat16(2.0f) * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + __float2bfloat16(2.0f) * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * __float2bfloat16(2.0f) * (b * c * dL_dconic.x - (denom + __float2bfloat16(2.0f) * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = __float2bfloat16(2.0f) * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + __float2bfloat16(2.0f) * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = __float2bfloat16(2.0f) * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + __float2bfloat16(2.0f) * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = __float2bfloat16(2.0f) * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + __float2bfloat16(2.0f) * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = __float2bfloat16(0.0f);
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	bfloat16 dL_dT00 = __float2bfloat16(2.0f) * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	bfloat16 dL_dT01 = __float2bfloat16(2.0f) * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	bfloat16 dL_dT02 = __float2bfloat16(2.0f) * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	bfloat16 dL_dT10 = __float2bfloat16(2.0f) * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	bfloat16 dL_dT11 = __float2bfloat16(2.0f) * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	bfloat16 dL_dT12 = __float2bfloat16(2.0f) * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	bfloat16 dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	bfloat16 dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	bfloat16 dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	bfloat16 dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	bfloat16 tz = hrcp(t.z);
	bfloat16 tz2 = tz * tz;
	bfloat16 tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	bfloat16 dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	bfloat16 dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	bfloat16 dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (__float2bfloat16(2.0f) * h_x * t.x) * tz3 * dL_dJ02 + (__float2bfloat16(2.0f) * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	bfloat163 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::bfvec3 scale, bfloat16 mod, const glm::bfvec4 rot, const bfloat16* dL_dcov3Ds, glm::bfvec3* dL_dscales, glm::bfvec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::bfvec4 q = rot;// / glm::length(rot);
	bfloat16 r = q.x;
	bfloat16 x = q.y;
	bfloat16 y = q.z;
	bfloat16 z = q.w;

	glm::bfmat3 R = glm::bfmat3(
		__float2bfloat16(1.0f) - __float2bfloat16(2.0f) * (y * y + z * z), __float2bfloat16(2.0f) * (x * y - r * z), __float2bfloat16(2.0f) * (x * z + r * y),
		__float2bfloat16(2.0f) * (x * y + r * z), __float2bfloat16(1.0f) - __float2bfloat16(2.0f) * (x * x + z * z), __float2bfloat16(2.0f) * (y * z - r * x),
		__float2bfloat16(2.0f) * (x * z - r * y), __float2bfloat16(2.0f) * (y * z + r * x), __float2bfloat16(1.0f) - __float2bfloat16(2.0f) * (x * x + y * y)
	);

	// we can't use the single float constructor since bfloat16 isn't automatically convertiblef from an int
	glm::bfmat3 S = glm::bfmat3(
		__float2bfloat16(1.0f), __float2bfloat16(0.0f), __float2bfloat16(0.0f),
		__float2bfloat16(0.0f), __float2bfloat16(1.0f), __float2bfloat16(0.0f),
		__float2bfloat16(0.0f), __float2bfloat16(0.0f), __float2bfloat16(1.0f)
	);

	glm::bfvec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::bfmat3 M = S * R;

	const bfloat16* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::bfvec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::bfvec3 ounc = __float2bfloat16(0.5f) * glm::bfvec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::bfmat3 dL_dSigma = glm::bfmat3(
		dL_dcov3D[0], __float2bfloat16(0.5f) * dL_dcov3D[1], __float2bfloat16(0.5f) * dL_dcov3D[2],
		__float2bfloat16(0.5f) * dL_dcov3D[1], dL_dcov3D[3], __float2bfloat16(0.5f) * dL_dcov3D[4],
		__float2bfloat16(0.5f) * dL_dcov3D[2], __float2bfloat16(0.5f) * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = __float2bfloat16(2.0f) * M
	glm::bfmat3 dL_dM = __float2bfloat16(2.0f) * M * dL_dSigma;

	glm::bfmat3 Rt = glm::bfmat3(
		R[0][0], R[1][0], R[2][0],
		R[0][1], R[1][1], R[2][1],
		R[0][2], R[1][2], R[2][2]
	);
	glm::bfmat3 dL_dMt = glm::bfmat3(
		dL_dM[0][0], dL_dM[1][0], dL_dM[2][0],
		dL_dM[0][1], dL_dM[1][1], dL_dM[2][1],
		dL_dM[0][2], dL_dM[1][2], dL_dM[2][2]
	);

	// Gradients of loss w.r.t. scale
	glm::bfvec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = Rt[0][0] * dL_dMt[0][0] + Rt[0][1] * dL_dMt[0][1] + Rt[0][2] * dL_dMt[0][2];
	dL_dscale->y = Rt[1][0] * dL_dMt[1][0] + Rt[1][1] * dL_dMt[1][1] + Rt[1][2] * dL_dMt[1][2];
	dL_dscale->z = Rt[2][0] * dL_dMt[2][0] + Rt[2][1] * dL_dMt[2][1] + Rt[2][2] * dL_dMt[2][2];

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::bfvec4 dL_dq;
	dL_dq.x = __float2bfloat16(2.0f) * z * (dL_dMt[0][1] - dL_dMt[1][0]) + __float2bfloat16(2.0f) * y * (dL_dMt[2][0] - dL_dMt[0][2]) + __float2bfloat16(2.0f) * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = __float2bfloat16(2.0f) * y * (dL_dMt[1][0] + dL_dMt[0][1]) + __float2bfloat16(2.0f) * z * (dL_dMt[2][0] + dL_dMt[0][2]) + __float2bfloat16(2.0f) * r * (dL_dMt[1][2] - dL_dMt[2][1]) - __float2bfloat16(4.0f) * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = __float2bfloat16(2.0f) * x * (dL_dMt[1][0] + dL_dMt[0][1]) + __float2bfloat16(2.0f) * r * (dL_dMt[2][0] - dL_dMt[0][2]) + __float2bfloat16(2.0f) * z * (dL_dMt[1][2] + dL_dMt[2][1]) - __float2bfloat16(4.0f) * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = __float2bfloat16(2.0f) * r * (dL_dMt[0][1] - dL_dMt[1][0]) + __float2bfloat16(2.0f) * x * (dL_dMt[2][0] + dL_dMt[0][2]) + __float2bfloat16(2.0f) * y * (dL_dMt[1][2] + dL_dMt[2][1]) - __float2bfloat16(4.0f) * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	bfloat164* dL_drot = (bfloat164*)(dL_drots + idx);
	*dL_drot = bfloat164{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(bfloat164{ rot.x, rot.y, rot.z, rot.w }, bfloat164{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const bfloat163* means,
	const int* radii,
	const bfloat16* shs,
	const bool* clamped,
	const glm::bfvec3* scales,
	const glm::bfvec4* rotations,
	const bfloat16 scale_modifier,
	const bfloat16* proj,
	const glm::bfvec3* campos,
	const bfloat163* dL_dmean2D,
	glm::bfvec3* dL_dmeans,
	bfloat16* dL_dcolor,
	bfloat16* dL_dcov3D,
	bfloat16* dL_dsh,
	glm::bfvec3* dL_dscale,
	glm::bfvec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	bfloat163 m = means[idx];

	// Taking care of gradients from the screenspace points
	bfloat164 m_hom = transformPoint4x4(m, proj);
	bfloat16 m_w = hrcp(m_hom.w + __float2bfloat16(0.0000001f));

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::bfvec3 dL_dmean;
	bfloat16 mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	bfloat16 mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::bfvec3*)means, *campos, shs, clamped, (glm::bfvec3*)dL_dcolor, (glm::bfvec3*)dL_dmeans, (glm::bfvec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const bfloat16* __restrict__ bg_color,
	const bfloat162* __restrict__ points_xy_image,
	const bfloat164* __restrict__ conic_opacity,
	const bfloat16* __restrict__ colors,
	const bfloat16* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const bfloat16* __restrict__ dL_dpixels,
	bfloat163* __restrict__ dL_dmean2D,
	bfloat164* __restrict__ dL_dconic2D,
	bfloat16* __restrict__ dL_dopacity,
	bfloat16* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const bfloat162 pixf = { __uint2bfloat16_rn(pix.x), __uint2bfloat16_rn(pix.y) };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ bfloat162 collected_xy[BLOCK_SIZE];
	__shared__ bfloat164 collected_conic_opacity[BLOCK_SIZE];
	__shared__ bfloat16 collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const bfloat16 T_final = inside ? final_Ts[pix_id] : __float2bfloat16(0.0f);
	bfloat16 T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	bfloat16 accum_rec[C] = { __float2bfloat16(0.0f) };
	bfloat16 dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	bfloat16 last_alpha = __float2bfloat16(0.0f);
	bfloat16 last_color[C] = { __float2bfloat16(0.0f) };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const bfloat16 ddelx_dx = __float2bfloat16(0.5f * W);
	const bfloat16 ddely_dy = __float2bfloat16(0.5f * H);

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const bfloat162 xy = collected_xy[j];
			const bfloat162 d = { xy.x - pixf.x, xy.y - pixf.y };
			const bfloat164 con_o = collected_conic_opacity[j];
			const bfloat16 power = -__float2bfloat16(0.5f) * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > __float2bfloat16(0.0f))
				continue;

			const bfloat16 G = hexp(power);
			const bfloat16 alpha = __hmin(__float2bfloat16(0.99f), con_o.w * G);
			if (alpha < __float2bfloat16(1.0f) / __float2bfloat16(255.0f))
				continue;

			T = T / (__float2bfloat16(1.0f) - alpha);
			const bfloat16 dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			bfloat16 dL_dalpha = __float2bfloat16(0.0f);
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const bfloat16 c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (__float2bfloat16(1.0f) - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const bfloat16 dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			bfloat16 bg_dot_dpixel = __float2bfloat16(0.0f);
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (__float2bfloat16(1.0f) - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const bfloat16 dL_dG = con_o.w * dL_dalpha;
			const bfloat16 gdx = G * d.x;
			const bfloat16 gdy = G * d.y;
			const bfloat16 dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const bfloat16 dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, - __float2bfloat16(0.5f) * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, - __float2bfloat16(0.5f) * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, - __float2bfloat16(0.5f) * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd((__nv_bfloat16 *)&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const bfloat163* means3D,
	const int* radii,
	const bfloat16* shs,
	const bool* clamped,
	const glm::bfvec3* scales,
	const glm::bfvec4* rotations,
	const bfloat16 scale_modifier,
	const bfloat16* cov3Ds,
	const bfloat16* viewmatrix,
	const bfloat16* projmatrix,
	const bfloat16 focal_x, bfloat16 focal_y,
	const bfloat16 tan_fovx, bfloat16 tan_fovy,
	const glm::bfvec3* campos,
	const bfloat163* dL_dmean2D,
	const bfloat16* dL_dconic,
	glm::bfvec3* dL_dmean3D,
	bfloat16* dL_dcolor,
	bfloat16* dL_dcov3D,
	bfloat16* dL_dsh,
	glm::bfvec3* dL_dscale,
	glm::bfvec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(bfloat163*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(bfloat163*)means3D,
		radii,
		shs,
		clamped,
		(glm::bfvec3*)scales,
		(glm::bfvec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(bfloat163*)dL_dmean2D,
		(glm::bfvec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
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
	bfloat16* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}