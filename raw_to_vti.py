#!/usr/bin/env python3
"""
Convert a 2048×2048×1920 uint8 raw volume (NRRD-style) into a
float32, normalized [0,1] voxel dataset on the unit cube, and save
as a structured VTK file using PyVista (ImageData).
"""

import numpy as np
import pyvista as pv
import os

def read_raw_volume(filename, shape, dtype=np.uint8, order='C'):
    """
    Reads a raw binary file into a NumPy array.
    """
    count = np.prod(shape)
    data = np.fromfile(filename, dtype=dtype, count=count)
    if data.size != count:
        raise IOError(f"Expected {count} elements, got {data.size}")
    return data.reshape(shape, order=order)

def normalize_volume(vol):
    """
    Linearly scales vol so its values lie in [0,1].
    """
    vmin, vmax = vol.min(), vol.max()
    if vmax == vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)

def build_image_data(volume):
    """
    Builds a PyVista ImageData whose points span [0,1]^3 and whose
    point-data is 'volume'.
    """
    nx, ny, nz = volume.shape
    m = max(nx, ny, nz)
    spacing = (1.0/(m), 1.0/(m), 1.0/(m))
    origin  = (0.0, 0.0, 0.0)
    
    grid = pv.ImageData(dimensions=(nx, ny, nz),
                        spacing=spacing,
                        origin=origin)
    print(grid.bounds)
    grid.point_data["value"] = volume.ravel(order='C')
    return grid

def main():
    # --- Update these to match your header ---
    raw_file = 'miranda_1024x1024x1024_float32.raw'
    vtk_file = 'miran.vtk'
    shape    = (1024, 1024, 1024)  # as per sizes: X, Y, Z
    dtype    = np.float32           # input type is uint8
    order    = 'C'                # use 'F' if needed for Fortran-order
    
    if not os.path.isfile(raw_file):
        raise FileNotFoundError(f"Could not locate raw file: {raw_file}")
    
    print("1) Reading raw uint8 volume…")
    vol_u8 = read_raw_volume(raw_file, shape, dtype=dtype, order=order)
    
    print("2) Converting to float32…")
    vol_f32 = vol_u8.astype(np.float32)
    
    print("3) Normalizing to [0,1]…")
    vol_norm = normalize_volume(vol_f32)
    
    print("4) Building ImageData in [0,1]^3…")
    grid = build_image_data(vol_norm)
    
    print(f"5) Saving to VTK: {vtk_file} …")
    grid.save(vtk_file)
    
    print("Done!")
    mesh = pv.read(vtk_file)
    print(mesh.bounds)
    # cell_count = 128
    # xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    # mins = [xmin, ymin, zmin]
    # maxes = [xmax, ymax, zmax]
    # spacing = [
    #     (maxes[0] - mins[0]) / (cell_count - 1),
    #     (maxes[1] - mins[1]) / (cell_count - 1),
    #     (maxes[2] - mins[2]) / (cell_count - 1)
    # ]
    # x = np.linspace(mins[0], maxes[0], cell_count)
    # y = np.linspace(mins[1], maxes[1], cell_count)
    # z = np.linspace(mins[2], maxes[2], cell_count)
    # x, y, z = np.meshgrid(x, y, z, indexing='ij')
    # samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    # samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)
    # rot = np.rot90(samples_3d, k=1, axes=(2,0))
    # samples_tf = np.flip(rot, axis=2)
    # save_cell = samples_tf.reshape(-1, 3)
    # print("Save cell made")
    # num_batches = 100
    # size = cell_count ** 3
    # big_samples = np.tile(save_cell, (num_batches, 1))
    # big_gt = gpu_sample(
    #     mesh.dimensions,
    #     mesh.origin,
    #     mesh.spacing,
    #     mesh.point_data['value'],
    #     big_samples
    # )
if __name__ == "__main__":
    main()
