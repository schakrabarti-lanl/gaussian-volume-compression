#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include <vector_types.h>
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
);

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
);