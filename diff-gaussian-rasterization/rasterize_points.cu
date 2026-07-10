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
#include <fstream>
#include <string>
#include <functional>
#include <cuBQL/bvh.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& values,
	const torch::Tensor& weights,
	const float scale_modifier,
	const float min_x, const float min_y, const float min_z, 
	const float max_x, const float max_y, const float max_z,
	const float background,
	const bool use_gaussian_bvh,
	const bool debug,
	const torch::Tensor& samples,
	const cuBQL::bvh3f& bvh,
	cuBQL::bvh3f& gaussian_bvh

) {
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	const float3 volume_mins = make_float3(min_x, min_y, min_z);
	const float3 volume_maxes = make_float3(max_x, max_y, max_z);
	
	const int P = means3D.size(0);
	const int S = samples.size(0);

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_test = torch::zeros({S}, float_opts);
	torch::Tensor out_testw = torch::zeros({S}, float_opts);
	torch::Tensor conics = torch::zeros({P * 6}, float_opts);
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	
	if(P != 0)
	{
		CudaRasterizer::Rasterizer::forward(
			P, S,
			means3D.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			values.contiguous().data<float>(),
			weights.contiguous().data<float>(),
			volume_mins,
			volume_maxes,
			samples.contiguous().data<float>(),
			bvh,
			gaussian_bvh,
			conics.contiguous().data<float>(),
			out_test.contiguous().data<float>(),
			out_testw.contiguous().data<float>(),
			use_gaussian_bvh,
			debug);
	}
	return std::make_tuple(out_test, out_testw, conics);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& conics,
	const torch::Tensor& values,
	const torch::Tensor& weights,
	const torch::Tensor& out_cells,
	const torch::Tensor& out_weights,
	const float scale_modifier,
	const float min_x, const float min_y, const float min_z, 
	const float max_x, const float max_y, const float max_z,
	const float background,
	const torch::Tensor& dL_dout_cells,
	const torch::Tensor& dL_dout_cell_weights,
	const bool use_gaussian_bvh,
	const bool debug,
	const torch::Tensor& samples,
	const cuBQL::bvh3f& bvh,
	const cuBQL::bvh3f& gaussian_bvh
) {
	const int P = means3D.size(0);
	const int S = samples.size(0);
	const float3 volume_mins = make_float3(min_x, min_y, min_z);
	const float3 volume_maxes = make_float3(max_x, max_y, max_z);

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dweights = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dvalues = torch::zeros({P, 1}, means3D.options());

	if(P != 0)
	{  
		CudaRasterizer::Rasterizer::backward(P, S,
			means3D.contiguous().data<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			volume_mins,
			volume_maxes,
			rotations.data_ptr<float>(),
			conics.contiguous().data<float>(),
			values.contiguous().data<float>(),
			weights.contiguous().data<float>(),
			samples.contiguous().data<float>(),
			bvh,
			gaussian_bvh,
			out_cells.contiguous().data<float>(),
			out_weights.contiguous().data<float>(),
			dL_dout_cells.contiguous().data<float>(),
			dL_dout_cell_weights.contiguous().data<float>(),
			dL_dmeans3D.contiguous().data<float>(),
			dL_dscales.contiguous().data<float>(),
			dL_drotations.contiguous().data<float>(),
			dL_dvalues.contiguous().data<float>(),
			dL_dweights.contiguous().data<float>(),
			use_gaussian_bvh,
			debug);
	}

	return std::make_tuple(dL_dmeans3D, dL_dscales, dL_drotations, dL_dvalues, dL_dweights);
}
