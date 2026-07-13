#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* conics,
	const float scale_modifier,
	const float* dL_dconics,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

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

	float conic[6] = {
		conics[6 * idx],
		conics[6 * idx + 1],
		conics[6 * idx + 2],
		conics[6 * idx + 3],
		conics[6 * idx + 4],
		conics[6 * idx + 5],
	};
	float dL_dconic[6] = {
		dL_dconics[6 * idx],
		dL_dconics[6 * idx + 1],
		dL_dconics[6 * idx + 2],
		dL_dconics[6 * idx + 3],
		dL_dconics[6 * idx + 4],
		dL_dconics[6 * idx + 5],
	};

	// // Compute dL_dcov as -conic * dL_dconic * conic
	// // since conic is inverse of cov
	// const float dL_dconic_conic[6] = {
	// 	dL_dconic[0]*conic[0] + dL_dconic[1]*conic[1] + dL_dconic[2]*conic[2],
    // 	dL_dconic[0]*conic[1] + dL_dconic[1]*conic[3] + dL_dconic[2]*conic[4],
    // 	dL_dconic[0]*conic[2] + dL_dconic[1]*conic[4] + dL_dconic[2]*conic[5],
    // 	dL_dconic[1]*conic[1] + dL_dconic[3]*conic[3] + dL_dconic[4]*conic[4],
    // 	dL_dconic[1]*conic[2] + dL_dconic[3]*conic[4] + dL_dconic[4]*conic[5],
    // 	dL_dconic[2]*conic[2] + dL_dconic[4]*conic[4] + dL_dconic[5]*conic[5]
	// };
	// const float dL_dcov[6] = {
	// 	-1.0f * (conic[0]*dL_dconic_conic[0] + conic[1]*dL_dconic_conic[1] + conic[2]*dL_dconic_conic[2]),
    // 	-1.0f * (conic[0]*dL_dconic_conic[1] + conic[1]*dL_dconic_conic[3] + conic[2]*dL_dconic_conic[4]),
    // 	-1.0f * (conic[0]*dL_dconic_conic[2] + conic[1]*dL_dconic_conic[4] + conic[2]*dL_dconic_conic[5]),
    // 	-1.0f * (conic[1]*dL_dconic_conic[1] + conic[3]*dL_dconic_conic[3] + conic[4]*dL_dconic_conic[4]),
    // 	-1.0f * (conic[1]*dL_dconic_conic[2] + conic[3]*dL_dconic_conic[4] + conic[4]*dL_dconic_conic[5]),
    // 	-1.0f * (conic[2]*dL_dconic_conic[2] + conic[4]*dL_dconic_conic[4] + conic[5]*dL_dconic_conic[5])
	// };
	// Unpack 'conic' and dL_dconic into full 3x3 matrices
	glm::mat3 Co(
		conic[0], conic[1], conic[2],
		conic[1], conic[3], conic[4],
		conic[2], conic[4], conic[5]
	);

	glm::mat3 G(
		dL_dconic[0], dL_dconic[1], dL_dconic[2],
		dL_dconic[1], dL_dconic[3], dL_dconic[4],
		dL_dconic[2], dL_dconic[4], dL_dconic[5]
	);

	// dL/dSigma = -C * G * C
	glm::mat3 dL_dSigma_mat = -Co * G * Co;

	// Repack to 6 parameters (xx, xy, xz, yy, yz, zz)
	float dL_dcov[6] = {
		dL_dSigma_mat[0][0],
		dL_dSigma_mat[0][1],
		dL_dSigma_mat[0][2],
		dL_dSigma_mat[1][1],
		dL_dSigma_mat[1][2],
		dL_dSigma_mat[2][2]
	};

	float abs_Sigma00 = abs(Sigma[0][0]);
	float abs_Sigma11 = abs(Sigma[1][1]);
	float abs_Sigma22 = abs(Sigma[2][2]);
	float depsilon_dSigma00 = (abs_Sigma00 >= abs_Sigma11 && abs_Sigma00 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[0][0]) : 0.0f;
	float depsilon_dSigma11 = (abs_Sigma11 >= abs_Sigma00 && abs_Sigma11 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[1][1]) : 0.0f;
	float depsilon_dSigma22 = (abs_Sigma22 >= abs_Sigma00 && abs_Sigma22 >= abs_Sigma11) ? 1e-5 * glm::sign(Sigma[2][2]) : 0.0f;
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov[0] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma00, dL_dcov[1], dL_dcov[2],
		dL_dcov[1], dL_dcov[3] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma11, dL_dcov[4],
		dL_dcov[2], dL_dcov[4], dL_dcov[5] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma22
	);
	glm::mat3 dL_dM = 2.f * dL_dSigma * M;
	glm::mat3 dL_dS = dL_dM * glm::transpose(R);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = dL_dS[0][0] * scale_modifier;
	dL_dscale->y = dL_dS[1][1] * scale_modifier;
	dL_dscale->z = dL_dS[2][2] * scale_modifier;

	dL_dM[0] *= scale_modifier * scale.x;
	dL_dM[1] *= scale_modifier * scale.y;
	dL_dM[2] *= scale_modifier * scale.z;
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dM[1][0] - dL_dM[0][1]) + 2 * y * (dL_dM[0][2] - dL_dM[2][0]) + 2 * x * (dL_dM[2][1] - dL_dM[1][2]);
	dL_dq.y = 2 * y * (dL_dM[0][1] + dL_dM[1][0]) + 2 * z * (dL_dM[0][2] + dL_dM[2][0]) + 2 * r * (dL_dM[2][1] - dL_dM[1][2]) - 4 * x * (dL_dM[2][2] + dL_dM[1][1]);
	dL_dq.z = 2 * x * (dL_dM[0][1] + dL_dM[1][0]) + 2 * r * (dL_dM[0][2] - dL_dM[2][0]) + 2 * z * (dL_dM[2][1] + dL_dM[1][2]) - 4 * y * (dL_dM[2][2] + dL_dM[0][0]);
	dL_dq.w = 2 * r * (dL_dM[1][0] - dL_dM[0][1]) + 2 * x * (dL_dM[0][2] + dL_dM[2][0]) + 2 * y * (dL_dM[2][1] + dL_dM[1][2]) - 4 * z * (dL_dM[1][1] + dL_dM[0][0]);
	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward version of the rendering procedure.
// Each thread handles one Gaussian that intersects this block,
// accumulating gradient contributions from all cells in the block
// before writing to global memory with a single set of atomicAdds.
template <uint32_t C>
__global__ void renderCUDA(
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float3 volume_mins,
	const uint3 num_cells,
	const float3 cell_size,
	const float* __restrict__ jitter,
	const float3* __restrict__ means3D,
	const float* __restrict__ values,
	const float* __restrict__ weights,
	const float* __restrict__ out_cells,
	const float* __restrict__ conic,
	const float* __restrict__ accumulated_weights,
	const float* __restrict__ dL_dcells,
	const float* __restrict__ dL_dcell_weights,
	float3* __restrict__ dL_dmeans,
	float* __restrict__ dL_dconic,
	float* __restrict__ dL_dvalues,
	float* __restrict__ dL_dweights
)
{
	// Block tile boundaries in cell space
	uint3 cell_min = { blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y, blockIdx.z * BLOCK_Z };
	uint3 cell_max = { min(cell_min.x + BLOCK_X, num_cells.x), min(cell_min.y + BLOCK_Y, num_cells.y), min(cell_min.z + BLOCK_Z, num_cells.z) };
	uint32_t bx = cell_max.x - cell_min.x;
	uint32_t by = cell_max.y - cell_min.y;
	uint32_t bz = cell_max.z - cell_min.z;
	uint32_t num_cells_in_block = bx * by * bz;

	// Load all cell data for this block into shared memory
	__shared__ float3 s_cell_pos[BLOCK_X * BLOCK_Y * BLOCK_Z];
	__shared__ float s_acc_weight[BLOCK_X * BLOCK_Y * BLOCK_Z];
	__shared__ float s_inv_acc_weight[BLOCK_X * BLOCK_Y * BLOCK_Z];
	__shared__ float s_dL_doutv_over_accw[BLOCK_X * BLOCK_Y * BLOCK_Z];
	__shared__ float s_dL_doutw[BLOCK_X * BLOCK_Y * BLOCK_Z];
	__shared__ float s_out_cell_val[BLOCK_X * BLOCK_Y * BLOCK_Z];

	for (uint32_t idx = threadIdx.x; idx < num_cells_in_block; idx += blockDim.x)
	{
		uint32_t lz = idx / (bx * by);
		uint32_t ly = (idx % (bx * by)) / bx;
		uint32_t lx = idx % bx;

		uint3 cell = { cell_min.x + lx, cell_min.y + ly, cell_min.z + lz };
		uint32_t cell_id = cell.z * num_cells.x * num_cells.y + cell.y * num_cells.x + cell.x;

		s_cell_pos[idx] = make_float3(
			static_cast<float>(cell.x) * cell_size.x + volume_mins.x + jitter[cell_id * 3],
			static_cast<float>(cell.y) * cell_size.y + volume_mins.y + jitter[cell_id * 3 + 1],
			static_cast<float>(cell.z) * cell_size.z + volume_mins.z + jitter[cell_id * 3 + 2]
		);

		float aw = accumulated_weights[cell_id];
		s_acc_weight[idx] = aw;
		float inv_aw = (aw > WEIGHT_CUTOFF) ? 1.0f / aw : 0.0f;
		s_inv_acc_weight[idx] = inv_aw;
		s_dL_doutv_over_accw[idx] = dL_dcells[cell_id] * inv_aw;
		s_dL_doutw[idx] = dL_dcell_weights[cell_id];
		s_out_cell_val[idx] = out_cells[cell_id];
	}
	__syncthreads();

	// Load range of Gaussians for this block
	uint2 range = ranges[blockIdx.z * grid.y * grid.x + blockIdx.y * grid.x + blockIdx.x];
	uint32_t num_gaussians = range.y - range.x;

	// Each thread processes one (or more) Gaussians, striding by blockDim.x
	for (uint32_t g = threadIdx.x; g < num_gaussians; g += blockDim.x)
	{
		int point_idx = point_list[range.x + g];

		// Load Gaussian data into registers
		float3 mean = means3D[point_idx];
		float value = values[point_idx];
		float w = weights[point_idx];
		float con_xx = conic[point_idx * 6 + 0];
		float con_xy = conic[point_idx * 6 + 1];
		float con_xz = conic[point_idx * 6 + 2];
		float con_yy = conic[point_idx * 6 + 3];
		float con_yz = conic[point_idx * 6 + 4];
		float con_zz = conic[point_idx * 6 + 5];

		// Accumulate gradients across all cells in the block
		float acc_dL_dvalue = 0.0f;
		float acc_dL_dxx = 0.0f;
		float acc_dL_dxy = 0.0f;
		float acc_dL_dxz = 0.0f;
		float acc_dL_dyy = 0.0f;
		float acc_dL_dyz = 0.0f;
		float acc_dL_dzz = 0.0f;
		// Replace the 3 mean accumulators + weight accumulator with:
		float acc_Gx = 0.0f, acc_Gy = 0.0f, acc_Gz = 0.0f;
		float acc_sum_dLdq = 0.0f;
		// acc_dL_dvalue stays the same

		for (uint32_t c = 0; c < num_cells_in_block; c++)
		{
			if (s_acc_weight[c] <= WEIGHT_CUTOFF) continue;

			float3 d = make_float3(s_cell_pos[c].x - mean.x, s_cell_pos[c].y - mean.y, s_cell_pos[c].z - mean.z);
			float quad_form = d.x*(con_xx*d.x + con_xy*d.y + con_xz*d.z)
							+ d.y*(con_xy*d.x + con_yy*d.y + con_yz*d.z)
							+ d.z*(con_xz*d.x + con_yz*d.y + con_zz*d.z);
			float power = -0.5f * quad_form;
			if (power < -14.0f || power > 0.0f) continue;

			float e = __expf(power);
			float weight_val = w * e;
			float dL_doutv_over_accw = s_dL_doutv_over_accw[c];

			acc_dL_dvalue += dL_doutv_over_accw * weight_val;

			float F = dL_doutv_over_accw * (value - s_out_cell_val[c]) + s_dL_doutw[c];
			float dL_dquad = F * (-0.5f * weight_val);

			acc_sum_dLdq += dL_dquad;          // replaces acc_dL_dw computation
			acc_Gx += dL_dquad * d.x;          // replaces the con-multiply mean grad lines
			acc_Gy += dL_dquad * d.y;
			acc_Gz += dL_dquad * d.z;

			acc_dL_dxx += dL_dquad * d.x * d.x;
			acc_dL_dxy += dL_dquad * d.x * d.y;
			acc_dL_dxz += dL_dquad * d.x * d.z;
			acc_dL_dyy += dL_dquad * d.y * d.y;
			acc_dL_dyz += dL_dquad * d.y * d.z;
			acc_dL_dzz += dL_dquad * d.z * d.z;
		}

		// Recover mean grads post-loop (con is still in registers)
		float dL_dmean_x = -2.0f * (con_xx*acc_Gx + con_xy*acc_Gy + con_xz*acc_Gz);
		float dL_dmean_y = -2.0f * (con_xy*acc_Gx + con_yy*acc_Gy + con_yz*acc_Gz);
		float dL_dmean_z = -2.0f * (con_xz*acc_Gx + con_yz*acc_Gy + con_zz*acc_Gz);

		// Recover weight grad: since dL_dquad = F * (-0.5*w*e), and acc_dL_dw = sum(F*e) = sum(-2*dL_dquad/w)
		float acc_dL_dw = -2.0f * acc_sum_dLdq / w;

		// Single atomic write per Gaussian — all cell contributions already accumulated
		atomicAdd(&dL_dvalues[point_idx], acc_dL_dvalue);
		atomicAdd(&dL_dweights[point_idx], acc_dL_dw);

		atomicAdd(&dL_dmeans[point_idx].x, dL_dmean_x);
		atomicAdd(&dL_dmeans[point_idx].y, dL_dmean_y);
		atomicAdd(&dL_dmeans[point_idx].z, dL_dmean_z);

		atomicAdd(&dL_dconic[point_idx * 6 + 0], acc_dL_dxx);
		atomicAdd(&dL_dconic[point_idx * 6 + 1], acc_dL_dxy);
		atomicAdd(&dL_dconic[point_idx * 6 + 2], acc_dL_dxz);
		atomicAdd(&dL_dconic[point_idx * 6 + 3], acc_dL_dyy);
		atomicAdd(&dL_dconic[point_idx * 6 + 4], acc_dL_dyz);
		atomicAdd(&dL_dconic[point_idx * 6 + 5], acc_dL_dzz);
	}
}

void BACKWARD::preprocess(
	int P,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* conics,
	const float scale_modifier,
	const float* dL_dconic,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for remaining steps: using dL_dconics
	// propagate back to scales and rotations
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		conics,
		scale_modifier,
		dL_dconic,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float3 volume_mins,
	const uint3 num_cells,
	const float3 cell_size,
	const float* jitter,
	const bool* clamped,
	const float3* means3D,
	const float* values,
	const float* weights,
	const float* out_cells,
	const float* volumes,
	const float* conic,
	const float* accumulated_weights,
	const uint32_t* n_contrib,
	const float* dL_dcells,
	const float* dL_dcell_weights,
	float3* dL_dmean3D,
	float* dL_dconic,
	float* dL_dvalue,
	float* dL_dweights
)
{
	renderCUDA<NUM_CHANNELS> << <grid, 128 >> >(
		grid,
		ranges,
		point_list,
		volume_mins,
		num_cells,
		cell_size,
		jitter,
		means3D,
		values,
		weights,
		out_cells,
		conic,
		accumulated_weights,
		dL_dcells,
		dL_dcell_weights,
		dL_dmean3D,
		dL_dconic,
		dL_dvalue,
		dL_dweights
	);
}
