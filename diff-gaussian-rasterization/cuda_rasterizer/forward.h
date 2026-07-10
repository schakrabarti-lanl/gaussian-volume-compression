#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
		const int P,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* weights,
		float* conics,
		cuBQL::box3f* aabbs);

	void render(
		const int P, const int S,
		const float* means3D,
		const float* values,
		const float* weights,
		const float* samples,
		const float* conics,
		const cuBQL::box3f* aabbs,
		const cuBQL::bvh3f& bvh,
		float* out_test,
		float* out_testw,
		int* count_intersections,
		const bool use_gaussian_bvh);
}


#endif
