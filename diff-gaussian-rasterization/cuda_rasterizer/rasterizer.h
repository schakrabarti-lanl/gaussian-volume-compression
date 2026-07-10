#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cuBQL/bvh.h>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void forward(
			const int P, const int S,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* values,
			const float* weights,
			const float3 volume_mins,
			const float3 volume_maxes,
			const float* samples,
			const cuBQL::bvh3f& bvh,
			cuBQL::bvh3f& gaussian_bvh,
			float* conics,
			float* out_test,
			float* out_testw,
			const bool use_gaussian_bvh,
			bool debug = false
		);

		static void backward(
			const int P, const int S,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float3 volume_mins, const float3 volume_maxes,
			const float* rotations,
			const float* conics,
			const float* values,
			const float* weights,
			const float* samples,
			const cuBQL::bvh3f& bvh,
			const cuBQL::bvh3f& gaussian_bvh,
			const float* out_cells,
			const float* out_weights,
			const float* dL_dsamples,
			const float* dL_dsample_weights,
			float* dL_dmean3D,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dvalue,
			float* dL_dweights,
			const bool use_gaussian_bvh,
			bool debug
		);
	};
};

#endif
