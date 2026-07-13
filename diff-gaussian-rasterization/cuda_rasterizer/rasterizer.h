#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* values,
			const float* weights,
			const float* jitter,
			const float3 volume_mins,
			const float3 volume_maxes,
			const uint3 num_cells,
			float* out_cells,
			float* out_weights,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int R,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const uint3 num_cells,
			const float3 volume_mins, const float3 volume_maxes,
			const float* rotations,
			const float* values,
			const float* weights,
			const float* jitter,
			const float* out_cells,
			const float* out_weights,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const float* dL_dcells,
			const float* dL_dcell_weights,
			float* dL_dconic,
			float* dL_dmean3D,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dvalue,
			float* dL_dweights,
			bool debug);
	};
};

#endif
