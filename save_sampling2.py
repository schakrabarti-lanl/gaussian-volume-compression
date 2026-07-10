"""
Sample random points from a VTU mesh and interpolate field values using gpu_sampleu.

Outputs:
  - big_samples.npy: (num_batches, size, 3) sample coordinates
  - big_gt.npy: (num_batches, size) interpolated field values

Usage:
  python sample_vtu.py <path.vtu> --num_batches 100 --size 2097152
"""

import argparse
import math
import numpy as np
import torch
import pyvista as pv
import trimesh

from gpu_mesh_sampling import gpu_sample, gpu_sampleu


def sample_mesh_points(mesh, num_batches, batch_size, device="cuda"):
    """Sample random points uniformly inside the tetrahedra of a mesh."""
    points = torch.tensor(mesh.points, dtype=torch.float32, device=device)
    # tet_mesh = mesh.triangulate()
    # tet_mesh.save("impacttet.vtu")
    tet_mesh = pv.read("impacttet.vtu")
    cells = torch.tensor(
        tet_mesh.cells.reshape(-1, 5)[:, 1:], dtype=torch.long, device=device
    )

    all_samples = np.empty((num_batches, batch_size, 3), dtype=np.float32)

    for i in range(num_batches):
        cell_idx = torch.randint(0, cells.shape[0], (batch_size,), device=device)
        verts = points[cells[cell_idx]]

        u = torch.rand(batch_size, 3, device=device).sort(dim=1).values
        bary = torch.zeros(batch_size, 4, device=device)
        bary[:, 0] = u[:, 0]
        bary[:, 1] = u[:, 1] - u[:, 0]
        bary[:, 2] = u[:, 2] - u[:, 1]
        bary[:, 3] = 1.0 - u[:, 2]

        samples = (bary.unsqueeze(2) * verts).sum(dim=1)
        all_samples[i] = samples.cpu().numpy()

    return all_samples

def sample_exterior_points(mesh, num_batches, batch_size, offset=0.01, device="cuda"):
    """Sample points just outside the mesh surface using properly oriented normals."""
    # tet_mesh = pv.read("impacttet.vtu")
    s = mesh.extract_surface().triangulate()
    print("Manifold", s.is_manifold)
    print("Open edges", s.n_open_edges)
    print("Faces", s.n_cells)
    surf = s.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=False,
    )
    print("Normals computed")

    face_normals = torch.tensor(surf.cell_data["Normals"], dtype=torch.float32, device=device)
    points = torch.tensor(surf.points, dtype=torch.float32, device=device)
    faces = torch.tensor(
        surf.faces.reshape(-1, 4)[:, 1:], dtype=torch.long, device=device
    )

    total = num_batches * batch_size

    face_idx = torch.randint(0, faces.shape[0], (total,), device=device)
    v = points[faces[face_idx]]
    print("Points grabbed")

    # Uniform barycentric coords on triangle
    u = torch.rand(total, 2, device=device)
    sqrt_u0 = u[:, 0].sqrt()
    bary = torch.stack([1 - sqrt_u0, sqrt_u0 * (1 - u[:, 1]), sqrt_u0 * u[:, 1]], dim=1)

    surf_pts = (bary.unsqueeze(2) * v).sum(dim=1)
    print("Barycentric grabbed")

    # Use PyVista's properly oriented outward normals
    normals = face_normals[face_idx]

    dist = torch.rand(total, 1, device=device) * offset
    samples = surf_pts + normals * dist
    print("Normals grabbed")

    return samples.reshape(num_batches, batch_size, 3).cpu().numpy()


def sample_verified_exterior(mesh, surface, num_batches, batch_size,
                             oversample_factor=0.5, max_iters=10, offset=0.01, device="cuda"):
    """
    Wrapper around sample_exterior_points that verifies points are truly outside
    using select_enclosed_points. Oversamples and retries until we have enough.
    Returns (num_batches, batch_size, 3).
    """
    total_needed = num_batches * batch_size
    outside_pool = np.empty((0, 3), dtype=np.float32)

    for iteration in range(max_iters):
        still_need = total_needed - len(outside_pool)
        if still_need <= 0:
            break

        n_gen = int(still_need * oversample_factor)
        candidates = sample_exterior_points(mesh, 1, n_gen, offset=offset, device=device)
        candidates = candidates.reshape(-1, 3)

        cloud = pv.PolyData(candidates)
        selected = cloud.select_enclosed_points(surface, check_surface=False)
        is_inside = selected["SelectedPoints"].astype(bool)

        outside_pool = np.concatenate([outside_pool, candidates[~is_inside]])
        pct_outside = (~is_inside).sum() / len(is_inside) * 100
        print(f"    exterior iter {iteration}: {pct_outside:.0f}% actually outside, "
              f"pool {len(outside_pool)}/{total_needed}")

    if len(outside_pool) < total_needed:
        raise RuntimeError(
            f"Could not collect enough exterior points after {max_iters} iterations. "
            f"Got {len(outside_pool)}/{total_needed}"
        )

    return outside_pool[:total_needed].reshape(num_batches, batch_size, 3)


def main():
    parser = argparse.ArgumentParser(description="Sample a VTU mesh and save numpy arrays.")
    parser.add_argument("vtu", help="Path to .vtu file")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches")
    parser.add_argument("--size", type=int, default=128**3, help="Samples per batch")
    parser.add_argument("--array_name", type=str, default=None,
                        help="Point data array to interpolate (default: first array)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device for sampling (cuda or cpu)")
    parser.add_argument("--output_prefix", type=str, default="",
                        help="Prefix for output filenames")
    args = parser.parse_args()

    print(f"Loading mesh: {args.vtu}")
    mesh = pv.read(args.vtu)
    print(f"  Points: {mesh.n_points}, Cells: {mesh.n_cells}")
    print(f"  Arrays: {mesh.array_names}")

    mesh = mesh.cell_data_to_point_data()
    values = mesh.get_array(mesh.array_names[0]).reshape(-1, 1)

    # Rescale the values to the range [0, 1]
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)
    mesh.get_array(mesh.array_names[0])[:] = values.ravel()

    # Scale mesh to the unit cube
    global_min = mesh.points.min()
    global_max = mesh.points.max()
    mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
    mesh.scale(1/(global_max - global_min), inplace=True)

    prefix = args.output_prefix
    # mesh.save(f"{prefix}norm.vtu")

    array_name = args.array_name or mesh.array_names[0]
    print(f"  Using array: '{array_name}'")

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print("  CUDA not available, falling back to CPU")

    # Extract closed surface once for inside/outside classification
    surface = mesh.extract_surface().triangulate()

    print(f"Sampling {args.num_batches} batches x {args.size} points...")
    size1 = int(math.ceil(args.size * 0.0))
    size2 = int(math.floor(args.size * 1.0))
    big_samples = sample_mesh_points(mesh, args.num_batches, size1)
    big_samples2 = sample_exterior_points(mesh, args.num_batches, size2, device=device)
    # print(big_samples.shape, big_samples2.shape)
    big_samples = np.concatenate([big_samples, big_samples2], axis=1).reshape(args.num_batches * args.size, 3)

    print("Interpolating field values with mesh...")
    probe_mesh = pv.PolyData(big_samples)
    probed = probe_mesh.sample(mesh)
    # big_samples = probed.points
    big_gt = probed[mesh.array_names[0]]
    valid_mask = probed['vtkValidPointMask'].astype(bool)
    big_gt[~valid_mask] = -1

    print("Interpolating field values with gpu_sampleu...")
    # big_gt = gpu_sampleu(
    #     mesh.points,
    #     mesh.cell_connectivity.astype(np.int64),
    #     mesh.celltypes.astype(np.int64),
    #     mesh.offset.astype(np.int64),
    #     mesh.point_data[array_name],
    #     big_samples,
    # )

    big_gt = big_gt.reshape(args.num_batches, args.size)
    big_samples = big_samples.reshape(args.num_batches, args.size, 3)
    print(f"Number of invalid samples: {np.count_nonzero(big_gt[0] == -1)}")

    samples_path = f"{prefix}big_samples.npy"
    gt_path = f"{prefix}big_gt.npy"

    np.save(samples_path, big_samples)
    np.save(gt_path, big_gt)
    print(f"Saved: {samples_path}  shape={big_samples.shape}")
    print(f"Saved: {gt_path}  shape={big_gt.shape}")


if __name__ == "__main__":
    main()