#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
		int P,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* values,
		const float* weights,
		bool* clamped,
		const float3 volume_mins,
		const float3 volume_maxes,
		const uint3 num_cells,
		const float3 cell_size,
		int* radii,
		float3* means,
		float* values_out, 
		float* weights_out,
		float* volumes,
		float* conic,
		uint* aabbs,
		const dim3 grid,
		uint32_t* blocks_touched);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const float3 volume_mins,
		const uint3 num_cells,
		const float3 cell_size,
		const float* jitter,
		const float3* means,
		const float* values,
		const float* weights,
		const float* volumes,
		const float* conic,
		float* accumulated_weights,
		uint32_t* n_contrib,
		float* out_cells);
}


#endif
