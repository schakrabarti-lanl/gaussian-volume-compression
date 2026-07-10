#include <torch/extension.h>
#include <cuda_runtime.h>
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include "cuBQL/builder/cuda.h"
#include "rasterize_points.h"

static cuBQL::bvh3f samples_bvh;
static cuBQL::bvh3f gaussian_bvh;
static torch::Tensor stored_samples;

// buildBoxes: one thread per sample; packs (x,y,z) into a box3f at that point
__global__ void buildBoxes(
    cuBQL::box3f*  aabbs,
    const float*   samples,
    int            num_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float x = samples[3*idx + 0];
    float y = samples[3*idx + 1];
    float z = samples[3*idx + 2];
    aabbs[idx] = cuBQL::box3f(cuBQL::vec3f(x, y, z));
}

// Host entrypoint: alloc → kernel → build BVH → free temp buffer
void BuildBVH(const torch::Tensor& samples, const bool debug, const bool use_gaussian_bvh) {
    stored_samples = samples.contiguous();
    if (!use_gaussian_bvh) {
        cudaEvent_t gpuStart, gpuStop;
        cudaEventCreate(&gpuStart);
        cudaEventCreate(&gpuStop);
        cudaEventRecord(gpuStart, 0);
        int  N = stored_samples.size(0);
        auto ptr = stored_samples.data_ptr<float>();

        cuBQL::cuda::free(samples_bvh);
        samples_bvh.nodes    = nullptr;
        samples_bvh.primIDs  = nullptr;
        samples_bvh.numNodes = 0;
        samples_bvh.numPrims = 0;
        samples_bvh = cuBQL::bvh3f();
        cuBQL::box3f* d_boxes;
        cudaMalloc(&d_boxes, N * sizeof(cuBQL::box3f));

        const int threads = 256;
        const int blocks  = (N + threads - 1) / threads;
        buildBoxes<<<blocks, threads>>>(d_boxes, ptr, N);
        cuBQL::BuildConfig cfg;
        cfg.makeLeafThreshold = 257;
        // cfg.maxAllowedLeafSize = 256;
        cuBQL::cuda::radixBuilder(samples_bvh, d_boxes, N, cfg);
        // cuBQL::gpuBuilder(samples_bvh, d_boxes, N, cfg);
        cudaFree(d_boxes);

        cudaEventRecord(gpuStop, 0);
        cudaEventSynchronize(gpuStop);  
        float msBoxes = 0.f;
        cudaEventElapsedTime(&msBoxes, gpuStart, gpuStop);
        if (debug) {
            std::cout << "Sample BVH time: " << msBoxes << " ms\n";
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDAWrapper(
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
	const bool debug
) {
    cuBQL::cuda::free(gaussian_bvh);
    gaussian_bvh.nodes    = nullptr;
    gaussian_bvh.primIDs  = nullptr;
    gaussian_bvh.numNodes = 0;
    gaussian_bvh.numPrims = 0;
    gaussian_bvh = cuBQL::bvh3f();

    return RasterizeGaussiansCUDA(
        means3D,
        scales,
        rotations,
        values,
        weights,
        scale_modifier,
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        background,
        use_gaussian_bvh,
        debug,
        stored_samples,
        samples_bvh,
        gaussian_bvh
    );

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDAWrapper(
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
	const bool debug
) {
    return RasterizeGaussiansBackwardCUDA(
        means3D,
        scales,
        rotations,
        conics,
        values,
        weights,
        out_cells,
        out_weights,
        scale_modifier,
        min_x, min_y, min_z, 
        max_x, max_y, max_z,
        background,
        dL_dout_cells,
        dL_dout_cell_weights,
        use_gaussian_bvh,
        debug,
        stored_samples,
        samples_bvh,
        gaussian_bvh
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_bvh", &BuildBVH, "Build BVH from samples");
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDAWrapper, "Forward pass");
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDAWrapper, "Backward pass");
}
