#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include "cuBQL/builder/cuda.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
void CudaRasterizer::Rasterizer::forward(
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
	bool debug)
{
	// Create CUDA events for timing (only when debug is enabled)
	cudaEvent_t events[6]; // 8 pairs of start/stop events
	if (debug) {
		for (int i = 0; i < 6; i++) {
			cudaEventCreate(&events[i]);
		}
	}

	int* d_count_intersections = nullptr;
	if (use_gaussian_bvh) {
		CHECK_CUDA(cudaMalloc(&d_count_intersections, sizeof(int) * S), debug);
	} else {
		CHECK_CUDA(cudaMalloc(&d_count_intersections, sizeof(int) * P), debug);
	}
	cuBQL::box3f* aabbs = nullptr;
	CHECK_CUDA(cudaMalloc(&aabbs, sizeof(cuBQL::box3f) * P), debug);

	// Preprocessing
	if (debug) cudaEventRecord(events[0]);
	CHECK_CUDA(FORWARD::preprocess(
		P,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		weights,
		conics,
		aabbs
	), debug)
	if (debug) cudaEventRecord(events[1]);

	if (debug) cudaEventRecord(events[2]);
	if (use_gaussian_bvh) {
		cuBQL::BuildConfig cfg;
    	// cfg.makeLeafThreshold = 33;
		// cfg.maxAllowedLeafSize = 32;
    	cuBQL::cuda::radixBuilder(gaussian_bvh, aabbs, P, cfg);
	}
	if (debug) cudaEventRecord(events[3]);

	// Rendering
	if (debug) cudaEventRecord(events[4]);
	if (use_gaussian_bvh) {
		CHECK_CUDA(FORWARD::render(
			P, S,
			means3D,
			values,
			weights,
			samples,
			conics,
			aabbs,
			gaussian_bvh,
			out_test,
			out_testw,
			d_count_intersections,
			use_gaussian_bvh
		), debug)
	} else {
		CHECK_CUDA(FORWARD::render(
			P, S,
			means3D,
			values,
			weights,
			samples,
			conics,
			aabbs,
			bvh,
			out_test,
			out_testw,
			d_count_intersections,
			use_gaussian_bvh
		), debug)
		}
		if (debug) cudaEventRecord(events[5]);

		const int num_count_entries = use_gaussian_bvh ? S : P;
		float inclusive_scan_time_ms = 0.0f;
		int total_intersections = 0;
		if (num_count_entries > 0) {
			int* d_prefix_intersections = nullptr;
			void* d_scan_temp_storage = nullptr;
			size_t scan_temp_storage_bytes = 0;
			CHECK_CUDA(cudaMalloc(&d_prefix_intersections, sizeof(int) * num_count_entries), debug);
			CHECK_CUDA(cub::DeviceScan::InclusiveSum(
				d_scan_temp_storage,
				scan_temp_storage_bytes,
				d_count_intersections,
				d_prefix_intersections,
				num_count_entries
			), debug);
			CHECK_CUDA(cudaMalloc(&d_scan_temp_storage, scan_temp_storage_bytes), debug);
			if (debug) {
				cudaEvent_t scan_start, scan_stop;
				cudaEventCreate(&scan_start);
				cudaEventCreate(&scan_stop);
				cudaEventRecord(scan_start);
				CHECK_CUDA(cub::DeviceScan::InclusiveSum(
					d_scan_temp_storage,
					scan_temp_storage_bytes,
					d_count_intersections,
					d_prefix_intersections,
					num_count_entries
				), debug);
				cudaEventRecord(scan_stop);
				cudaEventSynchronize(scan_stop);
				cudaEventElapsedTime(&inclusive_scan_time_ms, scan_start, scan_stop);
				cudaEventDestroy(scan_start);
				cudaEventDestroy(scan_stop);
			} else {
				CHECK_CUDA(cub::DeviceScan::InclusiveSum(
					d_scan_temp_storage,
					scan_temp_storage_bytes,
					d_count_intersections,
					d_prefix_intersections,
					num_count_entries
				), debug);
			}
			CHECK_CUDA(cudaMemcpy(&total_intersections,
				d_prefix_intersections + (num_count_entries - 1),
				sizeof(int),
				cudaMemcpyDeviceToHost), debug);
			CHECK_CUDA(cudaFree(d_scan_temp_storage), debug);
			CHECK_CUDA(cudaFree(d_prefix_intersections), debug);
		}
		if (debug) {
			std::printf("Inclusive scan: %.3f ms, total intersections=%d\n",
				inclusive_scan_time_ms, total_intersections);
		}

		if (debug) {
			if (use_gaussian_bvh) {
				std::vector<int> h_counts(S, 0);
				CHECK_CUDA(cudaMemcpy(h_counts.data(), d_count_intersections, sizeof(int) * S, cudaMemcpyDeviceToHost), debug);

			// Compute max and average
			long long sum = 0;
			int max_val = 0;
			int max_idx = -1;
			for (int i = 0; i < S; ++i) {
				sum += h_counts[i];
				if (h_counts[i] > max_val) {
					max_val = h_counts[i];
					max_idx = i;
				}
			}
			const double avg = (S > 0) ? static_cast<double>(sum) / static_cast<double>(S) : 0.0;
			std::printf("Intersections: max=%d (sample %d), avg=%.3f over %d samples\n",
					max_val, max_idx, avg, S);
		} else {
			std::vector<int> h_counts(P, 0);
			CHECK_CUDA(cudaMemcpy(h_counts.data(), d_count_intersections, sizeof(int) * P, cudaMemcpyDeviceToHost), debug);

			// Compute max and average
			long long sum = 0;
			int max_val = 0;
			int max_idx = -1;
			for (int i = 0; i < P; ++i) {
				sum += h_counts[i];
				if (h_counts[i] > max_val) {
					max_val = h_counts[i];
					max_idx = i;
				}
			}
			const double avg = (P > 0) ? static_cast<double>(sum) / static_cast<double>(P) : 0.0;

			std::printf("Intersections: max=%d (gaussian %d), avg=%.3f over %d gaussians\n",
						max_val, max_idx, avg, P);
		} 


	}
	CHECK_CUDA(cudaFree(d_count_intersections), debug);
	CHECK_CUDA(cudaFree(aabbs), debug);

	// Calculate and print timing (only when debug is enabled)
	if (debug) {
		cudaDeviceSynchronize();
		
		float elapsed_time;
		const char* operation_names[] = {
			"Preprocess", "BVH", "Render"
		};
		
		for (int i = 0; i < 3; i++) {
			cudaEventElapsedTime(&elapsed_time, events[i*2], events[i*2+1]);
			std::cout << operation_names[i] << " time: " << elapsed_time << " ms" << std::endl;
		}
		
		// Clean up events
		for (int i = 0; i < 6; i++) {
			cudaEventDestroy(events[i]);
		}
	}
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	bool debug)
{
	// Create CUDA events for timing (only when debug is enabled)
	cudaEvent_t events[2]; // 2 pairs of start/stop events
	if (debug) {
		for (int i = 0; i < 2; i++) {
			cudaEventCreate(&events[i]);
		}
	}

	int* d_count_intersections = nullptr;
	float* dL_dconics = nullptr;
	CHECK_CUDA(cudaMalloc(&dL_dconics, sizeof(float) * P * 6), debug);
	if (use_gaussian_bvh) {
		CHECK_CUDA(cudaMalloc(&d_count_intersections, sizeof(int) * S), debug);
	} else {
		CHECK_CUDA(cudaMalloc(&d_count_intersections, sizeof(int) * P), debug);
	}

	if (debug) cudaEventRecord(events[0]);
	// compute loss w.r.t gradients.
	if (use_gaussian_bvh) {
		CHECK_CUDA(BACKWARD::render(P, S,
			means3D,
			(glm::vec3*)scales,
			scale_modifier,
			(glm::vec4*)rotations,
			conics,
			values,
			weights,
			volume_mins, volume_maxes,
			samples,
			gaussian_bvh,
			out_cells,
			out_weights,
			dL_dsamples,
			dL_dsample_weights,
			dL_dmean3D,
			dL_dvalue,
			dL_dweights,
			(glm::vec3*)dL_dscale,
			(glm::vec4*)dL_drot,
			dL_dconics,
			d_count_intersections,
			true), debug);
	} else {
		CHECK_CUDA(BACKWARD::render(P, S,
			means3D,
			(glm::vec3*)scales,
			scale_modifier,
			(glm::vec4*)rotations,
			conics,
			values,
			weights,
			volume_mins, volume_maxes,
			samples,
			bvh,
			out_cells,
			out_weights,
			dL_dsamples,
			dL_dsample_weights,
			dL_dmean3D,
			dL_dvalue,
			dL_dweights,
			(glm::vec3*)dL_dscale,
			(glm::vec4*)dL_drot,
			dL_dconics,
			d_count_intersections,
			false), debug);
		}
		if (debug) cudaEventRecord(events[1]);

		const int num_count_entries = use_gaussian_bvh ? S : P;
		float inclusive_scan_time_ms = 0.0f;
		int total_intersections = 0;
		if (num_count_entries > 0) {
			int* d_prefix_intersections = nullptr;
			void* d_scan_temp_storage = nullptr;
			size_t scan_temp_storage_bytes = 0;
			CHECK_CUDA(cudaMalloc(&d_prefix_intersections, sizeof(int) * num_count_entries), debug);
			CHECK_CUDA(cub::DeviceScan::InclusiveSum(
				d_scan_temp_storage,
				scan_temp_storage_bytes,
				d_count_intersections,
				d_prefix_intersections,
				num_count_entries
			), debug);
			CHECK_CUDA(cudaMalloc(&d_scan_temp_storage, scan_temp_storage_bytes), debug);
			if (debug) {
				cudaEvent_t scan_start, scan_stop;
				cudaEventCreate(&scan_start);
				cudaEventCreate(&scan_stop);
				cudaEventRecord(scan_start);
				CHECK_CUDA(cub::DeviceScan::InclusiveSum(
					d_scan_temp_storage,
					scan_temp_storage_bytes,
					d_count_intersections,
					d_prefix_intersections,
					num_count_entries
				), debug);
				cudaEventRecord(scan_stop);
				cudaEventSynchronize(scan_stop);
				cudaEventElapsedTime(&inclusive_scan_time_ms, scan_start, scan_stop);
				cudaEventDestroy(scan_start);
				cudaEventDestroy(scan_stop);
			} else {
				CHECK_CUDA(cub::DeviceScan::InclusiveSum(
					d_scan_temp_storage,
					scan_temp_storage_bytes,
					d_count_intersections,
					d_prefix_intersections,
					num_count_entries
				), debug);
			}
			CHECK_CUDA(cudaMemcpy(&total_intersections,
				d_prefix_intersections + (num_count_entries - 1),
				sizeof(int),
				cudaMemcpyDeviceToHost), debug);
			CHECK_CUDA(cudaFree(d_scan_temp_storage), debug);
			CHECK_CUDA(cudaFree(d_prefix_intersections), debug);
		}
		if (debug) {
			std::printf("Backward inclusive scan: %.3f ms, total intersections=%d\n",
				inclusive_scan_time_ms, total_intersections);
		}

		if (debug) {
			if (use_gaussian_bvh) {
				std::vector<int> h_counts(S, 0);
				CHECK_CUDA(cudaMemcpy(h_counts.data(), d_count_intersections, sizeof(int) * S, cudaMemcpyDeviceToHost), debug);

			// Compute max and average
			long long sum = 0;
			int max_val = 0;
			int max_idx = -1;
			for (int i = 0; i < S; ++i) {
				sum += h_counts[i];
				if (h_counts[i] > max_val) {
					max_val = h_counts[i];
					max_idx = i;
				}
			}
			const double avg = (S > 0) ? static_cast<double>(sum) / static_cast<double>(S) : 0.0;
			std::printf("Backward intersections: max=%d (sample %d), avg=%.3f over %d samples\n",
					max_val, max_idx, avg, S);
		} else {
			std::vector<int> h_counts(P, 0);
			CHECK_CUDA(cudaMemcpy(h_counts.data(), d_count_intersections, sizeof(int) * P, cudaMemcpyDeviceToHost), debug);

			// Compute max and average
			long long sum = 0;
			int max_val = 0;
			int max_idx = -1;
			for (int i = 0; i < P; ++i) {
				sum += h_counts[i];
				if (h_counts[i] > max_val) {
					max_val = h_counts[i];
					max_idx = i;
				}
			}
			const double avg = (P > 0) ? static_cast<double>(sum) / static_cast<double>(P) : 0.0;

			std::printf("Backward intersections: max=%d (gaussian %d), avg=%.3f over %d gaussians\n",
						max_val, max_idx, avg, P);
		} 


	}

	CHECK_CUDA(cudaFree(d_count_intersections), debug);
	CHECK_CUDA(cudaFree(dL_dconics), debug);

	if (debug) {
		cudaDeviceSynchronize(); // ensure all events are completed
		float elapsed_time;
		const char* operation_names[] = { "Backward Render" };

		for (int i = 0; i < 1; ++i) {
			cudaEventElapsedTime(&elapsed_time, events[i * 2], events[i * 2 + 1]);
			std::cout << operation_names[i] << " time: " << elapsed_time << " ms" << std::endl;
		}

		for (int i = 0; i < 2; ++i) {
			cudaEventDestroy(events[i]);
		}
	}
}
