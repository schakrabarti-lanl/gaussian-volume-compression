#!/usr/bin/env python3
import sys
import time
import numpy as np
import pyvista as pv
from gpu_mesh_sampling import gpu_sample

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.vtk>")
        sys.exit(1)

    filename = sys.argv[1]

    # ————————————————————————————————
    # 1) Load the input VTK dataset
    t0 = time.perf_counter()
    mesh = pv.read(filename)
    t1 = time.perf_counter()
    print(f"1) Load dataset: {t1 - t0:.6f} s")

    global_min = mesh.points.min()
    global_max = mesh.points.max()
    mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
    mesh.scale(1.0/(global_max - global_min), inplace=True)

    # ————————————————————————————————
    # 2) Build a 10×10×10 grid of points in [0,1]^3
    t2_start = time.perf_counter()
    N = 5
    coords = np.linspace(0.0, 1.0, N)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    samps = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    print(samps.shape)
    probe = pv.PolyData(samps)
    t2_end = time.perf_counter()
    print(f"2) Build grid & polydata: {t2_end - t2_start:.6f} s")

    pts = mesh.points
    conn = mesh.cell_connectivity.astype(np.int64)
    values = mesh.point_data['value']
    vals = gpu_sample(pts, conn, values, samps)
    print(np.count_nonzero(vals == -1) / vals.shape[0])

    # ————————————————————————————————
    # 3) Probe (sample) the mesh at our grid points
    t3_start = time.perf_counter()
    sampled = probe.sample(mesh)
    t3_end = time.perf_counter()
    print(f"3) Probe/sample mesh: {t3_end - t3_start:.6f} s")

    # ————————————————————————————————
    # 4) Extract the first point‐data array and print its values
    t4_start = time.perf_counter()
    print(np.count_nonzero(sampled.point_data['vtkValidPointMask'] == 0) / sampled.point_data['value'].shape[0])
    values = sampled.point_data['value']
    values[sampled.point_data['vtkValidPointMask'] == 0] = -1
    # Optionally separate extraction and printing:
    t_extract = time.perf_counter()
    print(f"4a) Extract field data: {t_extract - t4_start:.6f} s")
    assert(np.all(np.abs(values - vals) < 0.0001))



if __name__ == "__main__":
    main()
