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
	#include "cuda_rasterizer/utils.h"
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
		const torch::Tensor& means3D,
		const torch::Tensor& scales,
		const torch::Tensor& rotations,
		const torch::Tensor& values,
		const torch::Tensor& weights,
		const torch::Tensor& jitter,
		const float scale_modifier,
		const float min_x, const float min_y, const float min_z, 
		const float max_x, const float max_y, const float max_z,
		const uint cell_count,
		const float background,
		const bool debug)
	{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	const float3 volume_mins = make_float3(min_x, min_y, min_z);
	const float3 volume_maxes = make_float3(max_x, max_y, max_z);
	
	const int P = means3D.size(0);

	const uint3 num_cells = make_uint3(
		cell_count,
		cell_count,
		cell_count
	);  
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_cells = torch::full({num_cells.x, num_cells.y, num_cells.z}, background, float_opts);
	torch::Tensor out_weights = torch::full({num_cells.x, num_cells.y, num_cells.z}, background, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
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
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P,
			means3D.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			values.contiguous().data<float>(),
			weights.contiguous().data<float>(),
			jitter.contiguous().data<float>(),
			volume_mins,
			volume_maxes,
			num_cells,
			out_cells.contiguous().data<float>(),
			out_weights.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug);
	}
	return std::make_tuple(rendered, out_cells, out_weights, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& values,
	const torch::Tensor& weights,
	const torch::Tensor& jitter,
	const torch::Tensor& out_cells,
	const torch::Tensor& out_weights,
	const float scale_modifier,
	const float min_x, const float min_y, const float min_z, 
	const float max_x, const float max_y, const float max_z,
	const uint cell_count,
	const float background,
	const torch::Tensor& dL_dout_cells,
	const torch::Tensor& dL_dout_cell_weights,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
	const int P = means3D.size(0);
	const float3 volume_mins = make_float3(min_x, min_y, min_z);
	const float3 volume_maxes = make_float3(max_x, max_y, max_z);
	const uint3 num_cells = make_uint3(
		cell_count,
		cell_count,
		cell_count
	);  
	int M = 0;

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dweights = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dvalues = torch::zeros({P, 1}, means3D.options());

	if(P != 0)
	{  
		CudaRasterizer::Rasterizer::backward(P, R,
		means3D.contiguous().data<float>(),
		scales.data_ptr<float>(),
		scale_modifier,
		num_cells,
		volume_mins,
		volume_maxes,
		rotations.data_ptr<float>(),
		values.contiguous().data<float>(),
		weights.contiguous().data<float>(),
		jitter.contiguous().data<float>(),
		out_cells.contiguous().data<float>(),
		out_weights.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		dL_dout_cells.contiguous().data<float>(),
		dL_dout_cell_weights.contiguous().data<float>(),
		dL_dconic.contiguous().data<float>(),  
		dL_dmeans3D.contiguous().data<float>(),
		dL_dscales.contiguous().data<float>(),
		dL_drotations.contiguous().data<float>(),
		dL_dvalues.contiguous().data<float>(),
		dL_dweights.contiguous().data<float>(),
		debug);
	}

	return std::make_tuple(dL_dmeans3D, dL_dscales, dL_drotations, dL_dvalues, dL_dweights);
}


std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}
