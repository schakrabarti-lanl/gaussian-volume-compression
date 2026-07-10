#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <cuBQL/bvh.h>
#include <cuBQL/traversal/fixedBoxQuery.h>

__global__ void preprocessCUDA(int P,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* conics,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots,
	float* dL_dconics
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
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

	const float dL_dconic[6] = {
		dL_dconics[idx * 6 + 0],
		dL_dconics[idx * 6 + 1],
		dL_dconics[idx * 6 + 2],
		dL_dconics[idx * 6 + 3],
		dL_dconics[idx * 6 + 4],
		dL_dconics[idx * 6 + 5]
	};

	const float conic[6] = {
		conics[idx * 6 + 0],
		conics[idx * 6 + 1],
		conics[idx * 6 + 2],
		conics[idx * 6 + 3],
		conics[idx * 6 + 4],
		conics[idx * 6 + 5]
	};

	// Unpack 'conic' and dL_dconic into full 3x3 matrices
	glm::mat3 C(
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
	glm::mat3 dL_dSigma_mat = -C * G * C;

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

__global__ void renderCUDA(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* conics,
	const float* values,
	const float* weights,
	const float3 volume_mins,
	const float3 volume_maxes,
	const float* samples,
	const cuBQL::bvh3f bvh,
	const float* out_cells,
	const float* out_weights,
	const float* dL_dsamples,
	const float* dL_dsample_weights,
	float* dL_dmeans,
	float* dL_dvalues,
	float* dL_dweights,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots,
	float* dL_dconics,
	int* count_intersections)
{
	const int THREADS_PER_GAUSSIAN = 32; // One warp per Gaussian
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<THREADS_PER_GAUSSIAN>(block);
	int idx = blockIdx.x * (blockDim.x / THREADS_PER_GAUSSIAN) + (threadIdx.x / THREADS_PER_GAUSSIAN);
	int thread_in_warp = warp.thread_rank();	
	if (idx >= P)
		return;

	auto scale = scales[idx];
	auto rot = rotations[idx];
	const float conic[6] = {
		conics[idx * 6 + 0],
		conics[idx * 6 + 1],
		conics[idx * 6 + 2],
		conics[idx * 6 + 3],
		conics[idx * 6 + 4],
		conics[idx * 6 + 5]
	};
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };

	float3 mins, maxes;
	{
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


		// Scale S by 3 to include up to where the weight is a tenth the cutoff
		float m = sqrtf(-2 * logf((0.1 * WEIGHT_CUTOFF) / weights[idx]));
		const float3 scaled_S = { S[0][0] * m, S[1][1] * m, S[2][2] * m };

		// Create array for corner computations
		const float n[2] = {-1.0f, 1.0f};
		
		// Initialize mins and maxes with gaussian position
		mins = position;
		maxes = position;

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
	}

	float dL_dvalue = 0.0;
	float dL_dw = 0.0;
	float dL_dmean_x = 0.0;
	float dL_dmean_y = 0.0;
	float dL_dmean_z = 0.0;
	float dL_dxx = 0.0;
	float dL_dxy = 0.0;
	float dL_dxz = 0.0;
	float dL_dyy = 0.0;
	float dL_dyz = 0.0;
	float dL_dzz = 0.0;
	int count = 0;
	int numLeaves = 0;
	cuBQL::fixedBoxQuery::forEachLeaf<float,3>(
	[&](uint32_t* primIDs, int count2) {
		for (int i = thread_in_warp; i < count2; i += THREADS_PER_GAUSSIAN) {
			int primID = primIDs[i];
			float3 d = make_float3(
				samples[primID * 3] - position.x, 
				samples[primID * 3 + 1] - position.y, 
				samples[primID * 3 + 2] - position.z
			);
			float quad_form = (
				d.x * (conic[0] * d.x + conic[1] * d.y + conic[2] * d.z) +
				d.y * (conic[1] * d.x + conic[3] * d.y + conic[4] * d.z) +
				d.z * (conic[2] * d.x + conic[4] * d.y + conic[5] * d.z)
			);
			float power = -0.5 * quad_form;
			if (power < -14.0 || power > 0.0) {continue;};

			float dL_doutv = dL_dsamples[primID];
			float dL_doutw = dL_dsample_weights[primID];
			float acc_weight = out_weights[primID];
			if (acc_weight <= WEIGHT_CUTOFF) {continue;};

			float e = exp(power);
			float weight = weights[idx] * e;

			dL_dvalue += dL_doutv * weight / acc_weight;

			float dLv_dweight = dL_doutv * (values[idx] / acc_weight - out_cells[primID] / acc_weight);
			float dLv_dw = dLv_dweight * e;

			float dweight_dquad = -0.5f * weight;
			float dLv_dquad = dLv_dweight * dweight_dquad;
			
			// Test not having gradients for weight from weight loss
			float dLw_dw = dL_doutw * e;
			// float dLw_dw = 0;
			float dLw_dquad = dL_doutw * dweight_dquad;

			dL_dw += dLv_dw + dLw_dw;
			float dL_dquad = dLv_dquad + dLw_dquad;
			
			// Gradients for means
			dL_dmean_x += dL_dquad * 2 * -1 *
				(conic[0] * d.x + conic[1] * d.y + conic[2] * d.z);
			dL_dmean_y += dL_dquad * 2 * -1 *
				(conic[1] * d.x + conic[3] * d.y + conic[4] * d.z);
			dL_dmean_z += dL_dquad * 2 * -1 *
				(conic[2] * d.x + conic[4] * d.y + conic[5] * d.z);

			// Gradients for conic
			dL_dxx += dL_dquad * d.x * d.x;
			dL_dxy += dL_dquad * 2.f * d.x * d.y;
			dL_dxz += dL_dquad * 2.f * d.x * d.z;
			dL_dyy += dL_dquad * d.y * d.y;
			dL_dyz += dL_dquad * 2.f * d.y * d.z;
			dL_dzz += dL_dquad * d.z * d.z;
			count++;
		}
		numLeaves++;
		return 0;
    },
		bvh,
		cuBQL::box3f(cuBQL::vec3f(mins.x, mins.y, mins.z), cuBQL::vec3f(maxes.x, maxes.y, maxes.z))
	);

	dL_dvalue = cg::reduce(warp, dL_dvalue, cg::plus<float>());
	dL_dw = cg::reduce(warp, dL_dw, cg::plus<float>());
	dL_dmean_x = cg::reduce(warp, dL_dmean_x, cg::plus<float>());
	dL_dmean_y = cg::reduce(warp, dL_dmean_y, cg::plus<float>());
	dL_dmean_z = cg::reduce(warp, dL_dmean_z, cg::plus<float>());
	dL_dxx = cg::reduce(warp, dL_dxx, cg::plus<float>());
	dL_dxy = cg::reduce(warp, dL_dxy, cg::plus<float>());
	dL_dxz = cg::reduce(warp, dL_dxz, cg::plus<float>());
	dL_dyy = cg::reduce(warp, dL_dyy, cg::plus<float>());
	dL_dyz = cg::reduce(warp, dL_dyz, cg::plus<float>());
	dL_dzz = cg::reduce(warp, dL_dzz, cg::plus<float>());
	count = cg::reduce(warp, count, cg::plus<int>());
	if (thread_in_warp == 0) {
		count_intersections[idx] = count;
	}

	if (thread_in_warp == 0) { 
		dL_dvalues[idx] = dL_dvalue;
		dL_dweights[idx] = dL_dw;
		dL_dmeans[idx * 3] = dL_dmean_x;
		dL_dmeans[idx * 3 + 1] = dL_dmean_y;	
		dL_dmeans[idx * 3 + 2] = dL_dmean_z;
		dL_dconics[idx * 6 + 0] = dL_dxx;
		dL_dconics[idx * 6 + 1] = dL_dxy;
		dL_dconics[idx * 6 + 2] = dL_dxz;
		dL_dconics[idx * 6 + 3] = dL_dyy;
		dL_dconics[idx * 6 + 4] = dL_dyz;
		dL_dconics[idx * 6 + 5] = dL_dzz;
	}
}

__global__ void sampleRenderCUDA(const int S,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* conics,
	const float* values,
	const float* weights,
	const float3 volume_mins,
	const float3 volume_maxes,
	const float* samples,
	const cuBQL::bvh3f bvh,
	const float* out_cells,
	const float* out_weights,
	const float* dL_dsamples,
	const float* dL_dsample_weights,
	float* dL_dmeans,
	float* dL_dvalues,
	float* dL_dweights,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots,
	float* dL_dconics,
	int* count_intersections)
{
	const int THREADS_PER_SAMPLE = 1; // One warp per sample
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<THREADS_PER_SAMPLE>(block);
	int idx = blockIdx.x * (blockDim.x / THREADS_PER_SAMPLE) + (threadIdx.x / THREADS_PER_SAMPLE);
	int thread_in_warp = warp.thread_rank();
	// auto idx = cg::this_grid().thread_rank();
	if (idx >= S)
		return;

	const float3 sample = { samples[3 * idx], samples[3 * idx + 1], samples[3 * idx + 2] };
	float dL_doutv = dL_dsamples[idx];
	float dL_doutw = dL_dsample_weights[idx];
	float acc_weight = out_weights[idx];
	if (acc_weight <= WEIGHT_CUTOFF) return;

	cuBQL::fixedBoxQuery::forEachPrim<float,3>(
	[&](int primID) {
		auto scale = scales[primID];
		auto rot = rotations[primID];
		
		// Create scaling matrix
		glm::mat3 Smat = glm::mat3(1.0f);
		Smat[0][0] = scale_modifier * scale.x;
		Smat[1][1] = scale_modifier * scale.y;
		Smat[2][2] = scale_modifier * scale.z;

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

		glm::mat3 M = Smat * R;

		// Compute 3D world covariance matrix Sigma
		glm::mat3 Sigma = glm::transpose(M) * M;

		const float conic[6] = {
			conics[primID * 6 + 0],
			conics[primID * 6 + 1],
			conics[primID * 6 + 2],
			conics[primID * 6 + 3],
			conics[primID * 6 + 4],
			conics[primID * 6 + 5]
		};

		// Scale S by 3 to include up to where the weight is a tenth the cutoff
		float m = sqrtf(-2 * logf((0.1 * WEIGHT_CUTOFF) / weights[primID]));
		const float3 scaled_S = { Smat[0][0] * m, Smat[1][1] * m, Smat[2][2] * m };

		// Create array for corner computations
		const float n[2] = {-1.0f, 1.0f};
		
		const float3 position = { means3D[3 * primID], means3D[3 * primID + 1], means3D[3 * primID + 2] };

		float dL_dvalue = 0.0;
		float dL_dw = 0.0;
		float dL_dmean_x = 0.0;
		float dL_dmean_y = 0.0;
		float dL_dmean_z = 0.0;
		float dL_dxx = 0.0;
		float dL_dxy = 0.0;
		float dL_dxz = 0.0;
		float dL_dyy = 0.0;
		float dL_dyz = 0.0;
		float dL_dzz = 0.0;
		float3 d = make_float3(
			sample.x - position.x, 
			sample.y - position.y, 
			sample.z - position.z
		);
		float quad_form = (
			d.x * (conic[0] * d.x + conic[1] * d.y + conic[2] * d.z) +
			d.y * (conic[1] * d.x + conic[3] * d.y + conic[4] * d.z) +
			d.z * (conic[2] * d.x + conic[4] * d.y + conic[5] * d.z)
		);
		float power = -0.5 * quad_form;
		if (power < -14.0 || power > 0.0) {return 0;};

		float e = exp(power);
		float weight = weights[primID] * e;

		dL_dvalue += dL_doutv * weight / acc_weight;

		float dLv_dweight = dL_doutv * (values[primID] / acc_weight - out_cells[idx] / acc_weight);
		float dLv_dw = dLv_dweight * e;

		float dweight_dquad = -0.5f * weight;
		float dLv_dquad = dLv_dweight * dweight_dquad;

		float dLw_dw = dL_doutw * e;
		float dLw_dquad = dL_doutw * dweight_dquad;

		dL_dw += dLv_dw + dLw_dw;
		float dL_dquad = dLv_dquad + dLw_dquad;
		
		// Gradients for means
		dL_dmean_x += dL_dquad * 2 * -1 *
			(conic[0] * d.x + conic[1] * d.y + conic[2] * d.z);
		dL_dmean_y += dL_dquad * 2 * -1 *
			(conic[1] * d.x + conic[3] * d.y + conic[4] * d.z);
		dL_dmean_z += dL_dquad * 2 * -1 *
			(conic[2] * d.x + conic[4] * d.y + conic[5] * d.z);

		// Gradients for conic
		dL_dxx += dL_dquad * d.x * d.x;
		dL_dxy += dL_dquad * 2.f * d.x * d.y;
		dL_dxz += dL_dquad * 2.f * d.x * d.z;
		dL_dyy += dL_dquad * d.y * d.y;
		dL_dyz += dL_dquad * 2.f * d.y * d.z;
		dL_dzz += dL_dquad * d.z * d.z;

		atomicAdd(&dL_dvalues[primID], dL_dvalue);
		atomicAdd(&dL_dweights[primID], dL_dw);
		atomicAdd(&dL_dmeans[primID * 3], dL_dmean_x);
		atomicAdd(&dL_dmeans[primID * 3 + 1], dL_dmean_y);
		atomicAdd(&dL_dmeans[primID * 3 + 2], dL_dmean_z);

		atomicAdd(&dL_dconics[primID * 6 + 0], dL_dxx);
		atomicAdd(&dL_dconics[primID * 6 + 1], dL_dxy);
		atomicAdd(&dL_dconics[primID * 6 + 2], dL_dxz);
		atomicAdd(&dL_dconics[primID * 6 + 3], dL_dyy);
		atomicAdd(&dL_dconics[primID * 6 + 4], dL_dyz);
		atomicAdd(&dL_dconics[primID * 6 + 5], dL_dzz);

		// float dL_dconic[6] = {
		// 	dL_dxx,
		// 	dL_dxy,
		// 	dL_dxz,
		// 	dL_dyy,
		// 	dL_dyz,
		// 	dL_dzz
		// };
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
		// float abs_Sigma00 = abs(Sigma[0][0]);
		// float abs_Sigma11 = abs(Sigma[1][1]);
		// float abs_Sigma22 = abs(Sigma[2][2]);
		// float depsilon_dSigma00 = (abs_Sigma00 >= abs_Sigma11 && abs_Sigma00 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[0][0]) : 0.0f;
		// float depsilon_dSigma11 = (abs_Sigma11 >= abs_Sigma00 && abs_Sigma11 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[1][1]) : 0.0f;
		// float depsilon_dSigma22 = (abs_Sigma22 >= abs_Sigma00 && abs_Sigma22 >= abs_Sigma11) ? 1e-5 * glm::sign(Sigma[2][2]) : 0.0f;
		// glm::mat3 dL_dSigma = glm::mat3(
		// 	dL_dcov[0] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma00, dL_dcov[1], dL_dcov[2],
		// 	dL_dcov[1], dL_dcov[3] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma11, dL_dcov[4],
		// 	dL_dcov[2], dL_dcov[4], dL_dcov[5] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma22
		// );
		// glm::mat3 dL_dM = 2.f * dL_dSigma * M;
		// glm::mat3 dL_dS = dL_dM * glm::transpose(R);

		// // Gradients of loss w.r.t. scale
		// atomicAdd(&dL_dscales[primID].x, dL_dS[0][0] * scale_modifier);
		// atomicAdd(&dL_dscales[primID].y, dL_dS[1][1] * scale_modifier);
		// atomicAdd(&dL_dscales[primID].z, dL_dS[2][2] * scale_modifier);

		// dL_dM[0] *= scale_modifier * scale.x;
		// dL_dM[1] *= scale_modifier * scale.y;
		// dL_dM[2] *= scale_modifier * scale.z;
		// glm::vec4 dL_dq;
		// dL_dq.x = 2 * z * (dL_dM[1][0] - dL_dM[0][1]) + 2 * y * (dL_dM[0][2] - dL_dM[2][0]) + 2 * x * (dL_dM[2][1] - dL_dM[1][2]);
		// dL_dq.y = 2 * y * (dL_dM[0][1] + dL_dM[1][0]) + 2 * z * (dL_dM[0][2] + dL_dM[2][0]) + 2 * r * (dL_dM[2][1] - dL_dM[1][2]) - 4 * x * (dL_dM[2][2] + dL_dM[1][1]);
		// dL_dq.z = 2 * x * (dL_dM[0][1] + dL_dM[1][0]) + 2 * r * (dL_dM[0][2] - dL_dM[2][0]) + 2 * z * (dL_dM[2][1] + dL_dM[1][2]) - 4 * y * (dL_dM[2][2] + dL_dM[0][0]);
		// dL_dq.w = 2 * r * (dL_dM[1][0] - dL_dM[0][1]) + 2 * x * (dL_dM[0][2] + dL_dM[2][0]) + 2 * y * (dL_dM[2][1] + dL_dM[1][2]) - 4 * z * (dL_dM[1][1] + dL_dM[0][0]);
		// // Gradients of loss w.r.t. unnormalized quaternion
		// atomicAdd(&dL_drots[primID].x, dL_dq.x);
		// atomicAdd(&dL_drots[primID].y, dL_dq.y);
		// atomicAdd(&dL_drots[primID].z, dL_dq.z);
		// atomicAdd(&dL_drots[primID].w, dL_dq.w);
		return 0;
	},
		bvh,
		cuBQL::box3f(cuBQL::vec3f(sample.x, sample.y, sample.z))
	);
}

void BACKWARD::render(
	const int P, const int S,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* conics,
	const float* values,
	const float* weights,
	const float3 volume_mins,
	const float3 volume_maxes,
	const float* samples,
	const cuBQL::bvh3f& bvh,
	const float* out_cells,
	const float* out_weights,
	const float* dL_dsamples,
	const float* dL_dsample_weights,
	float* dL_dmean3D,
	float* dL_dvalue,
	float* dL_dweights,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dconics,
	int* count_intersections,
	const bool use_gaussian_bvh)
{
	if (use_gaussian_bvh) {
		dim3 block(256);
		dim3 grid((S + block.x - 1) / block.x); // 32 threads per sample
		sampleRenderCUDA << < grid, block >> > (
			S,
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
			dL_dscale,
			dL_drot,
			dL_dconics,
			count_intersections);
		preprocessCUDA<<<(P + block.x - 1) / block.x, block.x>>>(
			P,
			(glm::vec3*)scales,
			scale_modifier,
			(glm::vec4*)rotations,
			conics,
			dL_dscale,
			dL_drot,
			dL_dconics
		);
	} else {
		dim3 block(256);
		dim3 grid((P * 32 + block.x - 1) / block.x); // 32 threads per Gaussian
		renderCUDA << < grid, block >> > (
			P,
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
			dL_dscale,
			dL_drot,
			dL_dconics,
			count_intersections);
		preprocessCUDA<<<(P + block.x - 1) / block.x, block.x>>>(
			P,
			(glm::vec3*)scales,
			scale_modifier,
			(glm::vec4*)rotations,
			conics,
			dL_dscale,
			dL_drot,
			dL_dconics
		);
	}
}