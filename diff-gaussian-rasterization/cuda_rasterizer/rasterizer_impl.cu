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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// // Generates one key/value pair for all Gaussian / tile overlaps. 
// // Run once per Gaussian (1:N mapping).
// __global__ void duplicateWithKeys(
// 	int P,
// 	const uint* aabbs,
// 	const uint32_t* offsets,
// 	uint32_t* gaussian_keys_unsorted,
// 	uint32_t* gaussian_values_unsorted,
// 	dim3 grid)
// {
// 	auto idx = cg::this_grid().thread_rank();
// 	if (idx >= P)
// 		return;

// 	// Find this Gaussian's offset in buffer for writing keys/values.
// 	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
// 	for (int z = aabbs[idx * 6 + 2]; z < aabbs[idx * 6 + 5]; z++) {
// 		for (int y = aabbs[idx * 6 + 1]; y < aabbs[idx * 6 + 4]; y++) {
// 			for (int x = aabbs[idx * 6]; x < aabbs[idx * 6 + 3]; x++) {
// 				gaussian_keys_unsorted[off] =  z * grid.x * grid.y + y * grid.x + x;
// 				gaussian_values_unsorted[off] = idx;
// 				off++;
// 			}
// 		}
// 	}
// }

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian using one warp (32 threads) per Gaussian.
__global__ void duplicateWithKeys(
	int P,
	const uint* aabbs,
	const uint32_t* offsets,
	uint32_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	dim3 grid)
{
	unsigned int idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
	if (idx >= P)
		return;

	unsigned int lane = threadIdx.x & 31;

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

	uint x0 = aabbs[idx * 6 + 0];
	uint y0 = aabbs[idx * 6 + 1];
	uint z0 = aabbs[idx * 6 + 2];
	uint x1 = aabbs[idx * 6 + 3];
	uint y1 = aabbs[idx * 6 + 4];
	uint z1 = aabbs[idx * 6 + 5];

	uint nx = x1 - x0;
	uint ny = y1 - y0;
	uint nz = z1 - z0;
	uint total = nx * ny * nz;

	for (uint i = lane; i < total; i += 32) {
		uint rem = i;
		uint iz = rem / (nx * ny);
		rem -= iz * (nx * ny);
		uint iy = rem / nx;
		uint ix = rem - iy * nx;

		uint x = x0 + ix;
		uint y = y0 + iy;
		uint z = z0 + iz;

		gaussian_keys_unsorted[off + i] = z * grid.x * grid.y + y * grid.x + x;
		gaussian_values_unsorted[off + i] = idx;
	}
}

__global__ void duplicateWithKeysParallel(
    int P,
    const uint* aabbs,
    const uint32_t* offsets,
    const uint32_t* blocks_touched,
    uint32_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int total_intersections,
    dim3 grid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_intersections) return;
    
    // Binary search to find which Gaussian this output belongs to
    int left = 0, right = P - 1;
    int gaussian_id = 0;
    while (left <= right) {
        int mid = (left + right) / 2;
        uint32_t mid_offset = (mid == 0) ? 0 : offsets[mid - 1];
        if (tid < mid_offset) {
            right = mid - 1;
        } else if (tid >= offsets[mid]) {
            left = mid + 1;
        } else {
            gaussian_id = mid;
            break;
        }
    }
    
    // Calculate the local index within this Gaussian's outputs
    int local_idx = (gaussian_id == 0) ? tid : tid - offsets[gaussian_id - 1];
    
    // Decode which cell this local_idx corresponds to
    int cells_per_gaussian = blocks_touched[gaussian_id];
    uint3 aabb_min = make_uint3(aabbs[gaussian_id * 6], 
                                 aabbs[gaussian_id * 6 + 1], 
                                 aabbs[gaussian_id * 6 + 2]);
    uint3 aabb_size = make_uint3(aabbs[gaussian_id * 6 + 3] - aabb_min.x,
                                  aabbs[gaussian_id * 6 + 4] - aabb_min.y,
                                  aabbs[gaussian_id * 6 + 5] - aabb_min.z);
    
    // Convert linear index to 3D position within the AABB
    int z = local_idx / (aabb_size.x * aabb_size.y);
    int y = (local_idx % (aabb_size.x * aabb_size.y)) / aabb_size.x;
    int x = local_idx % aabb_size.x;
    
    // Calculate actual grid position
    uint3 cell_pos = make_uint3(
		aabb_min.x + x,
		aabb_min.y + y,
		aabb_min.z + z
	);
    
    // Write the key-value pair
    gaussian_keys_unsorted[tid] = cell_pos.z * grid.x * grid.y + 
                                  cell_pos.y * grid.x + cell_pos.x;
    gaussian_values_unsorted[tid] = gaussian_id;
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint32_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint32_t currcell = point_list_keys[idx];
	if (idx == 0)
		ranges[currcell].x = 0;
	else
	{
		uint32_t prevcell = point_list_keys[idx - 1];
		if (currcell != prevcell)
		{
			ranges[prevcell].y = idx;
			ranges[currcell].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currcell].y = L;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.clamped, P, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.values, P, 128);
	obtain(chunk, geom.weights, P, 128);
	obtain(chunk, geom.volumes, P, 128);
	obtain(chunk, geom.means, P, 128);
	obtain(chunk, geom.conic, P * 6, 128);
	obtain(chunk, geom.aabbs, P * 6, 128);
	obtain(chunk, geom.blocks_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.blocks_touched, geom.blocks_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	int* radii,
	bool debug)
{
	const float3 cell_size = make_float3(
		(volume_maxes.x - volume_mins.x) / float(num_cells.x - 1),
		(volume_maxes.y - volume_mins.y) / float(num_cells.y - 1),
		(volume_maxes.z - volume_mins.z) / float(num_cells.z - 1)
	);
	
	// Create CUDA events for timing (only when debug is enabled)
	cudaEvent_t events[14]; // 7 pairs of start/stop events
	if (debug) {
		for (int i = 0; i < 14; i++) {
			cudaEventCreate(&events[i]);
		}
	}

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 block_grid((num_cells.x + BLOCK_X - 1) / BLOCK_X, (num_cells.y + BLOCK_Y - 1) / BLOCK_Y, (num_cells.z + BLOCK_Z - 1) / BLOCK_Z);
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(num_cells.x * num_cells.y * num_cells.z);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, num_cells.x * num_cells.y * num_cells.z);

	// Preprocessing
	if (debug) cudaEventRecord(events[0]);
	CHECK_CUDA(FORWARD::preprocess(
		P,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		values,
		weights,
		geomState.clamped,
		volume_mins, volume_maxes,
		num_cells,
		cell_size,
		radii,
		geomState.means,
		geomState.values, 
		geomState.weights,
		geomState.volumes,
		geomState.conic,
		geomState.aabbs,
		block_grid,
		geomState.blocks_touched
	), debug)
	if (debug) cudaEventRecord(events[1]);

	cuBQL::bvh3f bvh;

	// Prefix sum computation
	if (debug) cudaEventRecord(events[2]);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.blocks_touched, geomState.point_offsets, P), debug)
	if (debug) cudaEventRecord(events[3]);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_intersections;
	CHECK_CUDA(cudaMemcpy(&num_intersections, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	if (debug) {
		std::cout << "Total Num Intersections: " << num_intersections << "\n";
		std::cout << "BLOCK_SIZE compiled as: " << BLOCK_SIZE << std::endl;
		// int* host_blocks_touched = new int[P];
		// CHECK_CUDA(cudaMemcpy(host_blocks_touched,
		// 					geomState.blocks_touched,
		// 					P * sizeof(int),
		// 					cudaMemcpyDeviceToHost),
		// 		debug);
		// int min_inter = *std::min_element(host_blocks_touched, host_blocks_touched + P);
		// int max_inter = *std::max_element(host_blocks_touched, host_blocks_touched + P);
		// double avg_inter = double(num_intersections) / P;

		// std::cout << "Intersections per Gaussian statistics:\n"
		// 		<< "  Min:     " << min_inter << "\n"
		// 		<< "  Max:     " << max_inter << "\n"
		// 		<< "  Average: " << avg_inter << "\n";

		// int max_idx = std::distance(
		// 	host_blocks_touched,
		// 	std::max_element(host_blocks_touched, host_blocks_touched + P)
		// );
		// glm::vec3* host_scales = new glm::vec3[P];
		// CHECK_CUDA(cudaMemcpy(host_scales,
		// 					scales,
		// 					P * sizeof(glm::vec3),
		// 					cudaMemcpyDeviceToHost),debug);

		// glm::vec3  orig_scale = host_scales[max_idx];
		// glm::vec3  mod_scale  = orig_scale * scale_modifier;

		// std::cout << "Gaussian #" << max_idx
		// 		<< " had the most intersections.\n"
		// 		<< "  Original scale: ("
		// 		<< orig_scale.x << ", "
		// 		<< orig_scale.y << ", "
		// 		<< orig_scale.z << ")\n";

		// cleanup
		// delete[] host_blocks_touched;
		// delete[] host_scales;
	}

	size_t binning_chunk_size = required<BinningState>(num_intersections);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_intersections);

	// Key duplication
	if (debug) cudaEventRecord(events[4]);
	// duplicateWithKeys << <(P + 3) / 4, 256 >> > (
	// 	P,
	// 	geomState.aabbs,
	// 	geomState.point_offsets,
	// 	binningState.point_list_keys_unsorted,
	// 	binningState.point_list_unsorted,
	// 	block_grid);
	int threads = 256;
	int blocks = (num_intersections + threads - 1) / threads;
	duplicateWithKeysParallel<<<blocks, threads>>>(
		P,
		geomState.aabbs,
		geomState.point_offsets,
		geomState.blocks_touched,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		num_intersections,
		block_grid
	);
	CHECK_CUDA(, debug)
	if (debug) cudaEventRecord(events[5]);

	int bit = getHigherMsb(block_grid.x * block_grid.y * block_grid.z);

	// Sorting
	if (debug) cudaEventRecord(events[6]);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_intersections, 0, bit), debug)
	if (debug) cudaEventRecord(events[7]);

	// Number of blocks in each dimension
	if (debug) cudaEventRecord(events[8]);
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, block_grid.x * block_grid.y * block_grid.z * sizeof(uint2)), debug);
	if (debug) cudaEventRecord(events[9]);

	// Tile range identification
	if (num_intersections > 0) {
		if (debug) cudaEventRecord(events[10]);
		identifyTileRanges << <(num_intersections + 255) / 256, 256 >> > (
			num_intersections,
			binningState.point_list_keys,
			imgState.ranges);
		CHECK_CUDA(, debug)
		if (debug) cudaEventRecord(events[11]);
	}

	// Rendering
	const float* feature_ptr = geomState.values;
	if (debug) cudaEventRecord(events[12]);
	CHECK_CUDA(FORWARD::render(
		block_grid, block,
		imgState.ranges,
		binningState.point_list,
		volume_mins,
		num_cells,
		cell_size,
		jitter,
		geomState.means,
		feature_ptr,
		geomState.weights,
		geomState.volumes,
		geomState.conic,
		out_weights,
		imgState.n_contrib,
		out_cells), 
		debug)
	if (debug) cudaEventRecord(events[13]);

	if (debug) {
		// allocate host array and copy back the per‐cell counts
		int* host_n_contrib = new int[num_cells.x * num_cells.y * num_cells.z];
		CHECK_CUDA(cudaMemcpy(
			host_n_contrib,
			imgState.n_contrib,
			num_cells.x * num_cells.y * num_cells.z * sizeof(int),
			cudaMemcpyDeviceToHost
		), debug);

		// compute statistics
		int min_contrib = std::numeric_limits<int>::max();
		int max_contrib = std::numeric_limits<int>::min();
		int max_idx = 0;
		int64_t sum_contrib = 0;
		for (int64_t i = 0; i < num_cells.x * num_cells.y * num_cells.z; ++i) {
			int c = host_n_contrib[i];
			min_contrib = std::min(min_contrib, c);
			if (c > max_contrib) {
				max_contrib = c;
				max_idx = int(i);
			}
			sum_contrib += c;
		}
		double avg_contrib = double(sum_contrib) / double(num_cells.x * num_cells.y * num_cells.z);

		// decode (x,y,z) of the busiest cell for extra insight
		int cx =  max_idx % num_cells.x;
		int cy = (max_idx / num_cells.x) % num_cells.y;
		int cz =  max_idx / (num_cells.x * num_cells.y);

		// print out
		std::cout << "Gaussians per Cell statistics:\n"
				<< "  Min:     " << min_contrib << "\n"
				<< "  Max:     " << max_contrib << "\n"
				<< "  Average: " << avg_contrib << "\n"
				<< "  Busiest cell: (" 
					<< cx << ", " 
					<< cy << ", " 
					<< cz << ") with " 
					<< max_contrib 
					<< " contributions\n";

		delete[] host_n_contrib;
	}

	// Calculate and print timing (only when debug is enabled)
	if (debug) {
		cudaDeviceSynchronize(); // Single sync at the end
		
		float elapsed_time;
		const char* operation_names[] = {
			"Preprocessing", "Prefix sum", "Key duplication", 
			"Sorting", "Memory set", "Tile range identification", "Rendering"
		};
		
		float total = 0.0;
		for (int i = 0; i < 7; i++) {
			if (i == 5 && num_intersections == 0) continue; // Skip tile range if no intersections
			cudaEventElapsedTime(&elapsed_time, events[i*2], events[i*2+1]);
			std::cout << operation_names[i] << " time: " << elapsed_time << " ms" << std::endl;
			total += elapsed_time;
		}
		std::cout << "Total time: " << total << " ms" << std::endl;

		
		// Clean up events
		for (int i = 0; i < 14; i++) {
			cudaEventDestroy(events[i]);
		}
	}

	return num_intersections;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	bool debug)
{
	const float3 cell_size = make_float3(
		(volume_maxes.x - volume_mins.x) / float(num_cells.x - 1),
		(volume_maxes.y - volume_mins.y) / float(num_cells.y - 1),
		(volume_maxes.z - volume_mins.z) / float(num_cells.z - 1)
	);

	// Create CUDA events for timing (only when debug is enabled)
	cudaEvent_t events[4]; // 2 pairs of start/stop events
	if (debug) {
		for (int i = 0; i < 4; i++) {
			cudaEventCreate(&events[i]);
		}
	}

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, num_cells.x * num_cells.y * num_cells.z);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 block_grid((num_cells.x + BLOCK_X - 1) / BLOCK_X, (num_cells.y + BLOCK_Y - 1) / BLOCK_Y, (num_cells.z + BLOCK_Z - 1) / BLOCK_Z);
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

	if (debug) cudaEventRecord(events[0]);
	// Compute loss gradients w.r.t. mean position, conic matrix,
	// opacity and value of Gaussians from per-cell loss gradients.
	CHECK_CUDA(BACKWARD::render(
		block_grid, block,
		imgState.ranges,
		binningState.point_list,
		volume_mins,
		num_cells,
		cell_size,
		jitter,
		geomState.clamped,
		geomState.means,
		geomState.values,
		geomState.weights,
		out_cells,
		geomState.volumes,
		geomState.conic,
		out_weights,
		imgState.n_contrib,
		dL_dcells,
		dL_dcell_weights,
		(float3*)dL_dmean3D,
		dL_dconic,
		dL_dvalue,
		dL_dweights), debug);
	if (debug) cudaEventRecord(events[1]);

	if (debug) cudaEventRecord(events[2]);
	// Take care of the rest of preprocessing, compute loss w.r.t
	// scales and rotation from conic gradients.
	CHECK_CUDA(BACKWARD::preprocess(P,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		geomState.conic,
		scale_modifier,
		dL_dconic,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug);
	if (debug) cudaEventRecord(events[3]);

	if (debug) {
		cudaDeviceSynchronize(); // ensure all events are completed
		float elapsed_time;
		const char* operation_names[] = { "Backward Render", "Backward Preprocess" };

		for (int i = 0; i < 2; ++i) {
			cudaEventElapsedTime(&elapsed_time, events[i * 2], events[i * 2 + 1]);
			std::cout << operation_names[i] << " time: " << elapsed_time << " ms" << std::endl;
		}

		for (int i = 0; i < 4; ++i) {
			cudaEventDestroy(events[i]);
		}
	}
}
