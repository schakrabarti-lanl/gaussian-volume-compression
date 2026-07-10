import numpy as np
import pyvista as pv
import torch 

from scene.gaussian_model import BasicPointCloud
from gpu_mesh_sampling import gpu_sample, gpu_sampleu

def readData(path, fraction, normalized=False):
    mesh = pv.read(path)
    print("Mesh read")

    if not normalized:
        # Rescale the values to the range [0, 1]
        values = mesh.get_array("value").reshape(-1, 1)
        values_min = values.min()
        values_max = values.max()
        values = (values - values_min) / (values_max - values_min)
        mesh.get_array("value")[:] = values.ravel()

        # Scale mesh to the unit cube
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0/(global_max - global_min), inplace=True)
        # mesh.translate(np.array([0.01,0.01,0.01]), inplace=True)
        print("Mesh scaled")

    if fraction != -1:
        nx, ny, nz = mesh.dimensions
        ox, oy, oz = mesh.origin
        sx, sy, sz = mesh.spacing
        n_pts = mesh.n_points

        mask = (torch.rand(n_pts) < fraction)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        print("After rand")

        nxny = nx * ny
        k = idx // nxny
        r = idx % nxny
        j = r // nx
        i = r % nx
        pts_sampled = torch.stack([
            i.float() * sx + ox,
            j.float() * sy + oy,
            k.float() * sz + oz,
        ], dim=1)
        print("Points gathered")

        vals = mesh.point_data["value"][mask].reshape(-1, 1)  
        print("Mesh dropout")
        return mesh, BasicPointCloud(points=pts_sampled, values=vals)
        # nx, ny, nz = mesh.dimensions
        # ox, oy, oz = mesh.origin
        # sx, sy, sz = mesh.spacing
        # n_pts = mesh.n_points

        # # Values & eligibility (> 0.001)
        # vals_full = torch.as_tensor(mesh.point_data["value"]).float()
        # eligible_idx = torch.nonzero(vals_full > 1e-3, as_tuple=False).squeeze(1)

        # # Desired number of samples = same fraction of total points
        # desired = int(round(float(fraction) * n_pts))
        # desired = max(0, min(desired, n_pts))  # clamp

        # if desired == 0 or eligible_idx.numel() == 0:
        #     # Nothing to sample (either fraction==0 or no eligible points)
        #     idx = eligible_idx.new_empty((0,), dtype=torch.long)
        # else:
        #     if eligible_idx.numel() >= desired:
        #         # Enough eligible points: sample without replacement
        #         perm = torch.randperm(eligible_idx.numel())
        #         idx = eligible_idx[perm[:desired]]
        #     else:
        #         # Not enough eligible: take all eligible, then sample extras WITH replacement
        #         extra = desired - eligible_idx.numel()
        #         fill = eligible_idx[torch.randint(high=eligible_idx.numel(), size=(extra,))]
        #         idx = torch.cat([eligible_idx, fill], dim=0)

        # print("After eligibility + exact-count sampling")

        # # Convert flat indices -> (i, j, k)
        # nxny = nx * ny
        # k = idx // nxny
        # r = idx % nxny
        # j = r // nx
        # i = r % nx

        # pts_sampled = torch.stack([
        #     i.float() * sx + ox,
        #     j.float() * sy + oy,
        #     k.float() * sz + oz,
        # ], dim=1)
        # print("Points gathered")

        # # Values for the chosen points
        # vals = vals_full[idx].unsqueeze(1)
        # print("Mesh dropout")

        # return mesh, BasicPointCloud(points=pts_sampled, values=vals)
    else:
        return mesh, BasicPointCloud(points=None, values=None)


def readDatau(path, fraction, normalized=False):
    mesh = pv.read(path)
    mesh = mesh.cell_data_to_point_data()
    values = mesh.get_array(mesh.array_names[0]).reshape(-1, 1)

    if not normalized:
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
        # mesh.translate(np.array([0.01,0.01,0.01]), inplace=True)

    if fraction != -1:
        num_points = mesh.points.shape[0]
        print(num_points)
        indices = np.random.choice(num_points, size=int(num_points * fraction), replace=False)
        points_sampled = mesh.points[indices]
        values_sampled = values[indices]

        # cell_count = 50
        # xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        # x = np.linspace(xmin, xmax, cell_count)
        # y = np.linspace(ymin, ymax, cell_count)
        # z = np.linspace(zmin, zmax, cell_count)
        # x, y, z = np.meshgrid(x, y, z, indexing='ij')
        # samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        # samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)
        # rot = np.rot90(samples_3d, k=1, axes=(2,0))
        # samples_tf = np.flip(rot, axis=2)
        # save_cell = samples_tf.reshape(-1, 3)
        # save_gt = gpu_sampleu(
        #     mesh.points, 
        #     mesh.cell_connectivity.astype(np.int64),
        #     mesh.point_data['scalar'],
        #     save_cell
        # )
        # print(points_sampled.shape, values_sampled.shape)
        # print(save_cell.shape, save_gt.reshape(-1, 1).shape)

        return mesh, BasicPointCloud(points=points_sampled, values=values_sampled)
    else:
        return mesh, BasicPointCloud(points=None, values=None)
