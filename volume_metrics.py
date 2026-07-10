import sys
from argparse import ArgumentParser, Namespace
from random import randint
import numpy as np

import torch
from tqdm import tqdm
import pyvista as pv

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import init_rasterizer, render, build_bvh
from scene import GaussianModel, Scene
from gpu_mesh_sampling import gpu_sample, gpu_sampleu
from utils.debug_utils import tensor_to_vtk
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import bounding_box_regularization, create_window, l1_loss, l2_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DEBUG = True


def compute_bhattacharyya_sums(means: torch.Tensor, 
                                     covariances: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized computation of sum of overlap penalties from each 
    Gaussian to all other Gaussians using CUDA acceleration.
    
    Computes exp(-D_B) for each pair, where D_B is the Bhattacharyya distance:
    D_B = (1/8) * (μ₁-μ₂)ᵀ * Σ⁻¹ * (μ₁-μ₂) + (1/2) * ln(|Σ| / √(|Σ₁||Σ₂|))
    
    where Σ = (Σ₁ + Σ₂) / 2
    
    Returns higher values for overlapping Gaussians (small distances).
    
    Args:
        means: Tensor of shape (N, 3) containing mean vectors for N Gaussians
        covariances: Tensor of shape (N, 3, 3) containing covariance matrices
        
    Returns:
        Tensor of shape (N,) containing sum of overlap penalties 
        from each Gaussian to all others
    """
    device = means.device
    N = means.shape[0]
    
    # Ensure input tensors are on the same device and have correct dtype
    means = means.to(device=device, dtype=torch.float32)
    covariances = covariances.to(device=device, dtype=torch.float32)
    
    # Add small regularization to covariances for numerical stability
    eps = 1e-6
    eye = torch.eye(3, device=device, dtype=torch.float32)
    covariances = covariances + eps * eye
    
    # Precompute log determinants for all covariances
    log_dets = torch.logdet(covariances)  # Shape: (N,)
    
    # Create all pairwise combinations using broadcasting
    # means_i: (N, 1, 3), means_j: (1, N, 3) -> mu_diff: (N, N, 3)
    means_i = means.unsqueeze(1)  # Shape: (N, 1, 3)
    means_j = means.unsqueeze(0)  # Shape: (1, N, 3)
    mu_diff = means_i - means_j   # Shape: (N, N, 3)
    
    # Create all pairwise average covariances
    # covs_i: (N, 1, 3, 3), covs_j: (1, N, 3, 3) -> avg_cov: (N, N, 3, 3)
    covs_i = covariances.unsqueeze(1)  # Shape: (N, 1, 3, 3)
    covs_j = covariances.unsqueeze(0)  # Shape: (1, N, 3, 3)
    avg_cov = 0.5 * (covs_i + covs_j)  # Shape: (N, N, 3, 3)
    
    # Flatten the N×N matrices for batch operations
    mu_diff_flat = mu_diff.view(N * N, 3)           # Shape: (N*N, 3)
    avg_cov_flat = avg_cov.view(N * N, 3, 3)       # Shape: (N*N, 3, 3)
    
    # Compute determinants and inverses for all pairwise average covariances
    log_avg_det_flat = torch.logdet(avg_cov_flat)   # Shape: (N*N,)
    
    # Compute inverse covariances using Cholesky decomposition for stability
    try:
        L_avg = torch.linalg.cholesky(avg_cov_flat)  # Shape: (N*N, 3, 3)
        inv_avg_cov_flat = torch.cholesky_inverse(L_avg)  # Shape: (N*N, 3, 3)
    except RuntimeError:
        # Fallback to standard inverse if Cholesky fails
        inv_avg_cov_flat = torch.inverse(avg_cov_flat)
    
    # Compute Mahalanobis term: (1/8) * (μ₁-μ₂)ᵀ * Σ⁻¹ * (μ₁-μ₂)
    # Using batch matrix multiplication: mu_diff_flat @ inv_avg_cov_flat @ mu_diff_flat.T
    mahal_term_flat = 0.125 * torch.sum(
        mu_diff_flat.unsqueeze(-2) @ inv_avg_cov_flat @ mu_diff_flat.unsqueeze(-1),
        dim=(-2, -1)
    )  # Shape: (N*N,)
    
    # Reshape back to (N, N) matrix
    mahal_term = mahal_term_flat.view(N, N)         # Shape: (N, N)
    log_avg_det = log_avg_det_flat.view(N, N)       # Shape: (N, N)
    
    # Compute determinant term using broadcasting
    log_dets_i = log_dets.unsqueeze(1)  # Shape: (N, 1)
    log_dets_j = log_dets.unsqueeze(0)  # Shape: (1, N)
    det_term = 0.5 * (log_avg_det - 0.5 * (log_dets_i + log_dets_j))  # Shape: (N, N)
    
    # Compute full Bhattacharyya distance matrix
    bhatt_distances = mahal_term + det_term  # Shape: (N, N)
    
    # Create mask to exclude diagonal (self-distances)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    
    # Apply negative exponential to get overlap penalties (higher for smaller distances)
    overlap_penalties = torch.exp(-bhatt_distances)  # Shape: (N, N)
    
    # Zero out diagonal elements and sum across rows
    overlap_penalties = overlap_penalties * mask.float()
    overlap_sums = overlap_penalties.sum(dim=1)  # Shape: (N,)
    
    return overlap_sums


def sample_mesh_points(mesh, num_batches, batch_size, device="cuda"):
    points = torch.tensor(mesh.points, dtype=torch.float32, device=device)
    # tet_mesh = mesh.triangulate()
    tet_mesh = pv.read("impacttet.vtu")

    # Extract cell connectivity — assumes tets (4 verts per cell)
    cells = torch.tensor(
        tet_mesh.cells.reshape(-1, 5)[:, 1:], dtype=torch.long, device=device
    )  # (C, 4)

    total = num_batches * batch_size

    # Pick random cells (uniform = denser where cells are smaller = where points are denser)
    cell_idx = torch.randint(0, cells.shape[0], (total,), device=device)
    verts = points[cells[cell_idx]]  # (total, 4, 3)

    # Random barycentric coordinates inside a tetrahedron
    # Uniformly sample a tet: take 3 random values, sort, then differences give bary coords
    u = torch.rand(total, 3, device=device).sort(dim=1).values
    bary = torch.zeros(total, 4, device=device)
    bary[:, 0] = u[:, 0]
    bary[:, 1] = u[:, 1] - u[:, 0]
    bary[:, 2] = u[:, 2] - u[:, 1]
    bary[:, 3] = 1.0 - u[:, 2]

    # Interpolate: (total, 4, 1) * (total, 4, 3) summed over verts
    samples = (bary.unsqueeze(2) * verts).sum(dim=1)

    return samples.reshape(num_batches, batch_size, 3)


def sample_exterior_points(mesh, num_batches, batch_size, offset=0.01, device="cuda"):
    """Sample points just outside the mesh surface using properly oriented normals."""
    surf = mesh.extract_surface().triangulate().compute_normals(cell_normals=True, point_normals=False)

    face_normals = torch.tensor(surf.cell_data["Normals"], dtype=torch.float32, device=device)
    points = torch.tensor(surf.points, dtype=torch.float32, device=device)
    faces = torch.tensor(
        surf.faces.reshape(-1, 4)[:, 1:], dtype=torch.long, device=device
    )

    total = num_batches * batch_size

    face_idx = torch.randint(0, faces.shape[0], (total,), device=device)
    v = points[faces[face_idx]]

    # Uniform barycentric coords on triangle
    u = torch.rand(total, 2, device=device)
    sqrt_u0 = u[:, 0].sqrt()
    bary = torch.stack([1 - sqrt_u0, sqrt_u0 * (1 - u[:, 1]), sqrt_u0 * u[:, 1]], dim=1)

    surf_pts = (bary.unsqueeze(2) * v).sum(dim=1)

    # Use PyVista's properly oriented outward normals
    normals = face_normals[face_idx]

    dist = torch.rand(total, 1, device=device) * offset
    samples = surf_pts + normals * dist

    return samples.reshape(num_batches, batch_size, 3)


def training(
    dataset,
    opt,
    pipe,
    is_scaled,
    test_mesh
):
    
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians, load_iteration=-1, normalized=is_scaled, fraction=-1)
    # gaussians.cull_exterior_gaussians()
    # gaussians.save_ply("./test.ply")
    struct = dataset.source_path.lower().endswith('.vtk')
    pipe.debug = True
    init_rasterizer(
        gaussians,
        pipe,
    )
    # bsums = compute_bhattacharyya_sums(gaussians.get_xyz, gaussians.get_covariance(scaling_modifier=1, stripped=False))
    # print(bsums.mean())
    if not test_mesh:
        cell_count = 128
        spacing = [
            (gaussians.maxes[0] - gaussians.mins[0]) / (cell_count - 1),
            (gaussians.maxes[1] - gaussians.mins[1]) / (cell_count - 1),
            (gaussians.maxes[2] - gaussians.mins[2]) / (cell_count - 1)
        ]
        x = np.linspace(gaussians.mins[0], gaussians.maxes[0], cell_count)
        y = np.linspace(gaussians.mins[1], gaussians.maxes[1], cell_count)
        z = np.linspace(gaussians.mins[2], gaussians.maxes[2], cell_count)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)
        rot = np.rot90(samples_3d, k=1, axes=(2,0))

        samples_tf = np.flip(rot, axis=2)
        samples_tf_flat = samples_tf.reshape(-1, 3)
        jitter = np.random.uniform(-0.5, 0.5, samples_tf_flat.shape)
        jitter *= np.array(spacing)[None, :]
        samples_tf_flat = np.clip(
            samples_tf_flat + jitter,
            np.array(gaussians.mins)[None, :], 
            np.array(gaussians.maxes)[None, :]
        )
        if struct:
            gt_cells = gpu_sample(
                gaussians.mesh.dimensions,
                gaussians.mesh.origin,
                gaussians.mesh.spacing,
                gaussians.mesh.point_data['value'],
                samples_tf_flat
            )
        else:
            gt_cells = gpu_sampleu(
                gaussians.mesh.points, 
                gaussians.mesh.cell_connectivity.astype(np.int64),
                gaussians.mesh.celltypes.astype(np.int64),
                gaussians.mesh.offset.astype(np.int64),
                gaussians.mesh.point_data[gaussians.mesh.array_names[0]],
                samples_tf_flat
            )
        # tensor_to_vtk(gt_cells.reshape(cell_count, cell_count, cell_count), "test_gt.vtk", spacing)
        build_bvh(
            torch.tensor(samples_tf_flat, dtype=torch.float, device="cuda")
        )
    else:
        # idx = np.random.choice(gaussians.mesh.n_points, size=(1000000), replace=True)
        # start = np.random.randint(0, gaussians.mesh.n_points - 1000000 + 1)
        # idx = np.arange(start, start + 1000000)
        # if struct:
        #     nx, ny, nz = gaussians.mesh.dimensions
        #     ox, oy, oz = gaussians.mesh.origin
        #     sx, sy, sz = gaussians.mesh.spacing
        #     nxny = nx * ny
        #     k, r = np.divmod(idx, nxny)
        #     j, i = np.divmod(r, nx)
        #     x = ox + i * sx
        #     y = oy + j * sy
        #     z = oz + k * sz
        #     mesh_samples = np.stack((x, y, z), axis=-1)
        # idx = np.random.choice(gaussians.mesh.n_points, size=(1000000), replace=True)
        # if struct:
        #     nx, ny, nz = gaussians.mesh.dimensions
        #     ox, oy, oz = gaussians.mesh.origin
        #     sx, sy, sz = gaussians.mesh.spacing
        #     nxny = nx * ny
        #     k, r = np.divmod(idx, nxny)
        #     j, i = np.divmod(r, nx)

        #     # Sort spatially via Morton code (Z-order curve)
        #     def part1by2(n):
        #         n = n.astype(np.uint64) & 0x1fffff
        #         n = (n | (n << 32)) & 0x1f00000000ffff
        #         n = (n | (n << 16)) & 0x1f0000ff0000ff
        #         n = (n | (n << 8))  & 0x100f00f00f00f00f
        #         n = (n | (n << 4))  & 0x10c30c30c30c30c3
        #         n = (n | (n << 2))  & 0x1249249249249249
        #         return n

        #     scale = (1 << 21) - 1
        #     order = np.argsort(
        #         part1by2((i * scale) // (nx - 1))
        #         | (part1by2((j * scale) // (ny - 1)) << 1)
        #         | (part1by2((k * scale) // (nz - 1)) << 2)
        #     )
        #     idx = idx[order]
        #     i, j, k = i[order], j[order], k[order]

        #     x = ox + i * sx
        #     y = oy + j * sy
        #     z = oz + k * sz
        #     mesh_samples = np.stack((x, y, z), axis=-1)
        # else:
        #     mesh_samples = gaussians.mesh.points[idx]

        # gt_cells = gaussians.mesh.point_data[gaussians.mesh.array_names[0]][idx]
        # mesh_samples = sample_exterior_points(gaussians.mesh, 1, 1000000).reshape(1000000, 3).cpu().numpy()
        mesh_samples = sample_mesh_points(gaussians.mesh, 1, 1000000).reshape(1000000, 3).cpu().numpy()
        probe_mesh = pv.PolyData(mesh_samples)
        probed = probe_mesh.sample(gaussians.mesh)
        gt_cells = probed[gaussians.mesh.array_names[0]]
        valid_mask = probed['vtkValidPointMask'].astype(bool)
        gt_cells[~valid_mask] = -1
        # gt_cells = gpu_sampleu(
        #     gaussians.mesh.points, 
        #     gaussians.mesh.cell_connectivity.astype(np.int64),
        #     gaussians.mesh.celltypes.astype(np.int64),
        #     gaussians.mesh.offset.astype(np.int64),
        #     gaussians.mesh.point_data[gaussians.mesh.array_names[0]],
        #     mesh_samples.reshape(1000000, 3)
        # )
        gt_cells = gt_cells.reshape(1000000)
        mesh_samples = mesh_samples.reshape(1000000, 3)
        build_bvh(
            torch.tensor(mesh_samples, dtype=torch.float, device="cuda")
        )
    render_pkg = render(
        gaussians,
    )
    cells, weights = (
        render_pkg["cells"],
        render_pkg["weights"]
    )
    gt = torch.tensor(gt_cells).cuda()
    # intersect_pkg = intersect(
    #     gaussians,
    # )
    # intersections, intersection_weights = (
    #     intersect_pkg["intersections"],
    #     intersect_pkg["intersection_weight"]
    # )
    # print(torch.mean(intersections))
    # print(torch.mean(intersection_weights))

    cells2 = cells.clone()
    cells2[cells2 == -1] = 0
    l1_l = l1_loss(cells, gt)
    # l1_l.backward()
    
    mse = torch.mean((cells - gt) ** 2)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
    mse2 = torch.mean((cells[torch.logical_and(gt != -1, cells != -1)] - gt[torch.logical_and(gt != -1, cells != -1)]) ** 2)
    psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
    mse3 = torch.mean((gt) ** 2)
    psnr3 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse3 + 1e-8)
    mse4 = torch.mean((cells2 - gt) ** 2)
    psnr4 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse4 + 1e-8)
    mse5 = torch.mean((cells2[gt != -1] - gt[gt != -1]) ** 2)
    psnr5 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse5 + 1e-8)
    print(f"L1 loss: {l1_l.item()}")
    print(f"L2 loss: {mse}")
    print(f"PSNR: {psnr}")
    print(f"PSNR without false positives/negatives: {psnr2}")
    print(f"PSNR of original: {psnr3}")
    print(f"PSNR with 0s for fn: {psnr4}")
    print(f"PSNR with 0s for fn and no fp: {psnr5}")
    if not test_mesh:
        print(f"Percent invalid samples: {np.count_nonzero(gt_cells == -1) / cell_count ** 3}")
        print(f"False negative percent: {torch.count_nonzero(torch.logical_and(cells == -1, gt != -1)) / cell_count ** 3}")
        print(f"false positive percent: {torch.count_nonzero(torch.logical_and(cells != -1, gt == -1)) / cell_count ** 3}")
        tensor_to_vtk(cells.detach().cpu().numpy().reshape(cell_count, cell_count, cell_count), f"test.vtk", spacing)
    else:
        print(f"False negative: {torch.count_nonzero(torch.logical_and(cells == -1, gt != -1))}")
        print(f"false positive: {torch.count_nonzero(torch.logical_and(cells != -1, gt == -1))}")
        print(f"false positive percent: {torch.count_nonzero(torch.logical_and(cells != -1, gt == -1)) / torch.count_nonzero(gt == -1)}")
    # gaussians.save_ply_activated('apoint_cloud.ply')

if __name__ == "__main__":
    window = create_window()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--is_scaled", action="store_true")
    parser.add_argument("--test_mesh", action="store_true")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print(args.model_path)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.is_scaled,
        args.test_mesh
    )