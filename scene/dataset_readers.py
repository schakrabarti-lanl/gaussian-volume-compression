import numpy as np
import pyvista as pv
import torch

from scene.gaussian_model import BasicPointCloud
# from gpu_mesh_sampling import gpu_sample

def readData(path, fraction, normalized):
    mesh = pv.read(path)

    # Rescale the values to the range [0, 1]
    # values = mesh.get_array("value").reshape(-1, 1)
    # values_min = values.min()
    # values_max = values.max()
    # values = (values - values_min) / (values_max - values_min)
    # mesh.get_array("value")[:] = values.ravel()

    # # Scale mesh to the unit cube
    # global_min = mesh.points.min()
    # global_max = mesh.points.max()
    # mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
    # mesh.scale(1.0/(global_max - global_min), inplace=True)
    # # mesh.translate(np.array([0.01,0.01,0.01]), inplace=True)

    # print("before dropout")
    # num_points = mesh.points.shape[0]
    # indices = np.random.choice(num_points, size=int(num_points * fraction), replace=False)
    # points_sampled = mesh.points[indices]
    # values_sampled = values[indices]
    # print("dropout")
    scalar_name = mesh.point_data.keys()[0]

    if not normalized:
        # Rescale the values to the range [0, 1]
        values = mesh.get_array(scalar_name).reshape(-1, 1)
        values_min = values.min()
        values_max = values.max()
        values = (values - values_min) / (values_max - values_min)
        mesh.get_array(scalar_name)[:] = values.ravel()
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

        vals = mesh.point_data[scalar_name][mask].reshape(-1, 1)  
        print("Mesh dropout")
        return mesh, BasicPointCloud(points=pts_sampled, values=vals)


    # cell_count = 25
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
    # save_gt = gpu_sample(
    #     mesh.points, 
    #     mesh.cell_connectivity.astype(np.int64),
    #     mesh.point_data['value'],
    #     save_cell
    # )

    # return mesh, BasicPointCloud(points=points_sampled, values=values_sampled)

