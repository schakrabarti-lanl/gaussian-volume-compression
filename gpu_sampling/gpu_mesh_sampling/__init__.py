from ._gpu_mesh_sampling import sample_mesh, sample_meshu
import numpy as np

def gpu_sample(
    dims: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray
):
    return sample_mesh(dims, origin, spacing, values, samples)

def gpu_sampleu(
    pts: np.ndarray,
    conn: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray
):
    return sample_meshu(pts, conn, values, samples)