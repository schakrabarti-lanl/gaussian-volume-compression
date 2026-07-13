#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
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
	uint32_t* blocks_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize touched blocks to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	blocks_touched[idx] = 0;
	radii[idx] = 0;

	auto scale = scales[idx];
	auto rot = rotations[idx];
	
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale_modifier * scale.x;
	S[1][1] = scale_modifier * scale.y;
	S[2][2] = scale_modifier * scale.z;

	// Normalize quaternion to get valid rotation (commented out for some reason?)
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Normalize by epsilon to prevent numerical issues
	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	const float cov[6] = {
		Sigma[0][0] + epsilon,
        Sigma[0][1],
        Sigma[0][2],
		Sigma[1][1] + epsilon,
        Sigma[1][2],
        Sigma[2][2] + epsilon,
	};

	// Use 3D covariance to compute and store 3D conic
	const float a = cov[0]; // Sigma[0][0]
    const float b = cov[1]; // Sigma[0][1]
    const float c = cov[2]; // Sigma[0][2]
    const float d = cov[3]; // Sigma[1][1]
    const float e = cov[4]; // Sigma[1][2]
    const float f = cov[5]; // Sigma[2][2]
    const float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    const float det_inv = 1.0 / det;
    conic[idx * 6] = (d * f - e * e) * det_inv;
    conic[idx * 6 + 1] = (c * e - b * f) * det_inv;
    conic[idx * 6 + 2] = (b * e - c * d) * det_inv;
    conic[idx * 6 + 3] = (a * f - c * c) * det_inv;
    conic[idx * 6 + 4] = (b * c - a * e) * det_inv;
    conic[idx * 6 + 5] = (a * d - b * b) * det_inv;

	// Scale S by 3 to include up to three std from Gaussian position
	// const float m = 3.0;
	float m = sqrtf(-2 * logf((0.1 * WEIGHT_CUTOFF) / weights[idx]));
	const float3 scaled_S = { S[0][0] * m, S[1][1] * m, S[2][2] * m };

 	// Create array for corner computations
    const float n[2] = {-1.0f, 1.0f};
    
    // Initialize mins and maxes with gaussian position
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
	means[idx] = position;
    float3 mins = position;
    float3 maxes = position;

	// Compute corners using vector operations
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float3 corner = make_float3(
					position.x + n[i] * R[0].x * scaled_S.x + n[j] * R[1].x * scaled_S.y +  n[k] * R[2].x * scaled_S.z,
					position.y + n[i] * R[0].y * scaled_S.x + n[j] * R[1].y * scaled_S.y +  n[k] * R[2].y * scaled_S.z,
					position.z + n[i] * R[0].z * scaled_S.x + n[j] * R[1].z * scaled_S.y +  n[k] * R[2].z * scaled_S.z
				);
                    
                mins = make_float3(
					min(mins.x, corner.x),
					min(mins.y, corner.y),
					min(mins.z, corner.z)
				);
                maxes = make_float3(
					max(maxes.x, corner.x),
					max(maxes.y, corner.y),
					max(maxes.z, corner.z)
				);
            }
        }
    }

	// Calculate block size in world coordinates
	const float block_size_x = cell_size.x * BLOCK_X;
	const float block_size_y = cell_size.y * BLOCK_Y;
	const float block_size_z = cell_size.z * BLOCK_Z;

	// Find which blocks the Gaussian intersects
	uint3 start_block = make_uint3(
		max(0, min(static_cast<int>(grid.x), static_cast<int>(floor((mins.x - volume_mins.x) / block_size_x)))),
		max(0, min(static_cast<int>(grid.y), static_cast<int>(floor((mins.y - volume_mins.y) / block_size_y)))),
		max(0, min(static_cast<int>(grid.z), static_cast<int>(floor((mins.z - volume_mins.z) / block_size_z))))
	);    
	uint3 end_block = make_uint3(
		max(0, min(static_cast<int>(grid.x), static_cast<int>(ceil((maxes.x - volume_mins.x) / block_size_x)))),
		max(0, min(static_cast<int>(grid.y), static_cast<int>(ceil((maxes.y - volume_mins.y) / block_size_y)))),
		max(0, min(static_cast<int>(grid.z), static_cast<int>(ceil((maxes.z - volume_mins.z) / block_size_z))))
	);
    uint3 block_dims = make_uint3(
		end_block.x - start_block.x,
		end_block.y - start_block.y,
		end_block.z - start_block.z
	);

    // Store results
    blocks_touched[idx] = static_cast<int>(block_dims.x * block_dims.y * block_dims.z);
	radii[idx] = 1;
    aabbs[idx * 6] = start_block.x;
	aabbs[idx * 6 + 1] = start_block.y;
    aabbs[idx * 6 + 2] = start_block.z;
    aabbs[idx * 6 + 3] = end_block.x;
    aabbs[idx * 6 + 4] = end_block.y;
	aabbs[idx * 6 + 5] = end_block.z;
	// Clamping may not be necessary since these are stored with sigmoid activation?
	// clamped[idx] = (values[idx] < 0.0f) || (values[idx] > 1.0f);
    // values_out[idx] = glm::clamp(values[idx], 0.0f, 1.0f); 
	// weights_out[idx] = glm::clamp(weights[idx], 0.0f, 1.0f); 
	clamped[idx] = false;
    values_out[idx] = values[idx]; 
	weights_out[idx] = weights[idx]; 
    volumes[idx] = static_cast<float>(block_dims.x * block_dims.y * block_dims.z);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const dim3 grid,
	const float3 volume_mins,
	const uint3 num_cells,
	const float3 cell_size,
	const float* __restrict__ jitter,
	const float3* __restrict__ means,
	const float* __restrict__ values,
	const float* __restrict__ weights,
	const float* __restrict__ conic,
	float* __restrict__ accumulated_weights,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_cells)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint3 cell_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y, block.group_index().z * BLOCK_Z};
	uint3 cell_max = { min(cell_min.x + BLOCK_X, num_cells.x), min(cell_min.y + BLOCK_Y , num_cells.y), min(cell_min.z + BLOCK_Z , num_cells.z) };
	uint3 cell = { cell_min.x + block.thread_index().x, cell_min.y + block.thread_index().y, cell_min.z + block.thread_index().z  };
	uint32_t cell_id = cell.z * num_cells.x * num_cells.y + cell.y * num_cells.x + cell.x;
	float3 cell_pos =  make_float3(
		static_cast<float>(cell.x) * cell_size.x + volume_mins.x + jitter[cell_id * 3], 
		static_cast<float>(cell.y) * cell_size.y + volume_mins.y + jitter[cell_id * 3 + 1], 
		static_cast<float>(cell.z) * cell_size.z + volume_mins.z + jitter[cell_id * 3 + 2]
	);

	// Check if this thread is associated with a valid cell or outside.
	bool inside = cell.x < num_cells.x && cell.y < num_cells.y && cell.z < num_cells.z;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().z * grid.y * grid.x + block.group_index().y * grid.x + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ float3 collected_means[BLOCK_SIZE];
	__shared__ float collected_values[BLOCK_SIZE];
	__shared__ float collected_weights[BLOCK_SIZE];
	__shared__ float collected_conic[BLOCK_SIZE * 6];

	// Initialize helper variables
	float accumulated_weight = 0;
	float accumulated_value = 0;
	uint32_t n_contributor = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) // TODO: try using float4s, align to 128 bit for bank conflict
		{
			int coll_id = point_list[range.x + progress];
			collected_means[block.thread_rank()] = means[coll_id];
			collected_values[block.thread_rank()] = values[coll_id];
			collected_weights[block.thread_rank()] = weights[coll_id];
			for (int k = 0; k < 6; k++)
                collected_conic[block.thread_rank() * 6 + k] = conic[coll_id * 6 + k];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			// n_contributor++;

			float3 d = make_float3(cell_pos.x - collected_means[j].x, cell_pos.y - collected_means[j].y, cell_pos.z - collected_means[j].z);
			float quad_form = (
				d.x * (collected_conic[j * 6] * d.x + collected_conic[j * 6 + 1] * d.y + collected_conic[j * 6 + 2] * d.z) +
				d.y * (collected_conic[j * 6 + 1] * d.x + collected_conic[j * 6 + 3] * d.y + collected_conic[j * 6 + 4] * d.z) +
				d.z * (collected_conic[j * 6 + 2] * d.x + collected_conic[j * 6 + 4] * d.y + collected_conic[j * 6 + 5] * d.z)
			);
			float power = -0.5 * quad_form;
			if (power < -14.0 || power > 0.0) continue;
			float weight = collected_weights[j] * __expf(power);
			// float dx = cell_pos.x - collected_means[j].x;
			// float dy = cell_pos.y - collected_means[j].y;
			// float dz = cell_pos.z - collected_means[j].z;
			// int base = j * 6;
			// float t;
			// t = collected_conic[base] * dx + collected_conic[base + 1] * dy + collected_conic[base + 2] * dz;
			// float qf = dx * t;
			// t = collected_conic[base + 1] * dx + collected_conic[base + 3] * dy + collected_conic[base + 4] * dz;
			// qf += dy * t;
			// t = collected_conic[base + 2] * dx + collected_conic[base + 4] * dy + collected_conic[base + 5] * dz;
			// qf += dz * t;
			// if (qf > 28.0f || qf < 0.0f) continue;  // equivalent to power check
			// float weight = collected_weights[j] * __expf(-0.5f * qf);

			accumulated_value += collected_values[j] * weight;
			accumulated_weight += weight;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// This both gives a dropoff where we have to have a certain weight to set a value
		// and prevents numerical issues of dividing by something close to 0
		if (accumulated_weight > WEIGHT_CUTOFF) {
			out_cells[cell_id] = accumulated_value / accumulated_weight;
			accumulated_weights[cell_id] = accumulated_weight;
			n_contrib[cell_id] = 0;

		} else {
			out_cells[cell_id] = -1.0;
			accumulated_weights[cell_id] = 0.0;
			n_contrib[cell_id] = 0;
		}
	}
}

void FORWARD::render(
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
	float* out_cells)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		grid,
		volume_mins,
		num_cells,
		cell_size,
		jitter,
		means,
		values,
		weights,
		conic,
		accumulated_weights,
		n_contrib,
		out_cells);
}

void FORWARD::preprocess(int P,
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
	uint32_t* blocks_touched)
{
	preprocessCUDA<NUM_CHANNELS> <<<(P + 255) / 256, 256>>> (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		values,
		weights,
		clamped,
		volume_mins,
		volume_maxes,
		num_cells,
		cell_size,
		radii,
		means,
		values_out, 
		weights_out,
		volumes,
		conic,
		aabbs,
		grid,
		blocks_touched
	);
}
