import pyvista as pv
import numpy as np

mesh = pv.read("../../gaussian-volume/vtu/impact.vtu")

def nbytes(a): return np.asarray(a).nbytes

points_bytes = nbytes(mesh.points)

# modern topology (VTK 9 style)
conn_bytes   = nbytes(mesh.cell_connectivity)
offset_bytes = nbytes(mesh.offset)
types_bytes  = nbytes(mesh.celltypes)

point_data_bytes = sum(nbytes(mesh.point_data[k]) for k in mesh.point_data.keys())
cell_data_bytes  = sum(nbytes(mesh.cell_data[k])  for k in mesh.cell_data.keys())

points_total = points_bytes + point_data_bytes
cells_total  = conn_bytes + offset_bytes + types_bytes + cell_data_bytes
total = points_total + cells_total
print(f"Num points: {mesh.n_points}, num cells: {mesh.n_cells}")
print(f"Point data size: {point_data_bytes/1e6:.3f} MB")
print(f"Cell data size: {cell_data_bytes/1e6:.3f} MB")

print(f"Points: {points_total/1e6:.3f} MB")
print(f"Cells: {cells_total/1e6:.3f} MB")

print(f"Total MB (1e6): {total/1e6:.3f}")
print(f"Outer surface size: {mesh.extract_surface().GetActualMemorySize() * 1024 / 1e6:.3f} MB")