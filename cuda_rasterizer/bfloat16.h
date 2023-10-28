// #include <cuda.h>
// #include "cuda_runtime.h"
// #include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <glm/glm.hpp>

#ifndef CUDA_RASTERIZER_BFLOAT16_H_INCLUDED
#define CUDA_RASTERIZER_BFLOAT16_H_INCLUDED

typedef __nv_bfloat16 bfloat16;
typedef __nv_bfloat162 bfloat162;

struct bfloat163
{
    __nv_bfloat16 x, y, z;
};

struct __builtin_align__(8) bfloat164
{
    __nv_bfloat16 x, y, z, w;
};

#if !((defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA))
__nv_bfloat16 atomicAdd(__nv_bfloat16 *const address, const __nv_bfloat16 val)
{
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __bfloat16_as_ushort(__hadd(val, __ushort_as_bfloat16(assumed))));
    } while (assumed != old);
    return __ushort_as_bfloat16(old);
}
#endif /* !((defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)) */

namespace glm
{
    typedef vec<1, bfloat16, qualifier::defaultp>		bfvec1;
    typedef vec<2, bfloat16, qualifier::defaultp>		bfvec2;
    typedef vec<3, bfloat16, qualifier::defaultp>		bfvec3;
    typedef vec<4, bfloat16, qualifier::defaultp>		bfvec4;

    typedef mat<2, 2, bfloat16, qualifier::defaultp>	bfmat2;
    typedef mat<3, 3, bfloat16, qualifier::defaultp>	bfmat3;
    typedef mat<4, 4, bfloat16, qualifier::defaultp>	bfmat4;
}

#endif // CUDA_RASTERIZER_BFLOAT16_H_INCLUDED
