#!/usr/bin/env python3
"""
Read a raw volume file and generate big_gt (sampled ground truth values)
and big_samples (sample positions with jitter) for training data.
"""

import numpy as np
from gpu_mesh_sampling import gpu_sample
import os
import re
import sys


def parse_filename(filepath):
    """
    Parse shape and dtype from a raw filename.
    Expected pattern: *_<NX>x<NY>x<NZ>_<dtype>.raw
    e.g. richtmyer_meshkov_2048x2048x1920_uint8.raw
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]

    # Match dimensions like 2048x2048x1920
    shape_match = re.search(r'(\d+)x(\d+)x(\d+)', basename)
    if not shape_match:
        raise ValueError(
            f"Could not parse shape from filename '{basename}'. "
            "Expected pattern like *_2048x2048x1920_uint8.raw"
        )
    shape = (int(shape_match.group(1)), int(shape_match.group(2)), int(shape_match.group(3)))

    # Match dtype after the last underscore (e.g. uint8, float32, uint16)
    dtype_match = re.search(r'_(u?int(?:8|16|32|64)|float(?:16|32|64))(?:\.|$)', basename)
    if not dtype_match:
        raise ValueError(
            f"Could not parse dtype from filename '{basename}'. "
            "Expected a numpy dtype like uint8, float32, etc."
        )
    dtype = np.dtype(dtype_match.group(1))

    return shape, dtype


def read_raw_volume(filename, shape, dtype=np.uint8, order='C'):
    """
    Reads a raw binary file into a NumPy array.
    """
    count = np.prod(shape)
    data = np.fromfile(filename, dtype=dtype, count=count)
    if data.size != count:
        raise IOError(f"Expected {count} elements, got {data.size}")
    return data.reshape(shape, order=order)


def generate_samples(cell_count, num_batches, mins, maxes):
    """
    Generate sample positions with jittered batches within given bounds.

    Args:
        cell_count: number of cells per dimension
        num_batches: number of batches to generate
        mins: tuple/array of (min_x, min_y, min_z)
        maxes: tuple/array of (max_x, max_y, max_z)

    Returns:
        big_samples: array of shape (num_batches, cell_count**3, 3)
        big_jitter: array of shape (num_batches, cell_count**3, 3)
        spacing: tuple of spacing values per dimension
    """
    mins = np.array(mins)
    maxes = np.array(maxes)

    spacing = [
        (maxes[0] - mins[0]) / (cell_count - 1),
        (maxes[1] - mins[1]) / (cell_count - 1),
        (maxes[2] - mins[2]) / (cell_count - 1)
    ]

    # Create base grid
    x = np.linspace(mins[0], maxes[0], cell_count)
    y = np.linspace(mins[1], maxes[1], cell_count)
    z = np.linspace(mins[2], maxes[2], cell_count)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)

    # Apply transformations (rot90 and flip)
    rot = np.rot90(samples_3d, k=1, axes=(2, 0))
    samples_tf = np.flip(rot, axis=2)
    save_cell = samples_tf.reshape(-1, 3)

    # Create batches with jitter
    size = cell_count ** 3
    big_samples = np.tile(save_cell, (num_batches, 1))

    big_jitter = np.random.uniform(-0.5, 0.5, big_samples.shape)
    big_jitter *= np.array(spacing)[None, :]
    big_jitter[:size, :] = 0  # First batch has no jitter

    big_samples = np.clip(
        big_samples + big_jitter,
        mins,
        maxes
    )

    return big_samples.reshape(num_batches, size, 3), big_jitter.reshape(num_batches, size, 3), spacing


def compute_mesh_params(shape):
    """
    Compute mesh parameters (dimensions, origin, spacing) for gpu_sample,
    with aspect-preserving normalization centered in [0,1]^3.

    Args:
        shape: tuple of (nx, ny, nz) volume dimensions

    Returns:
        dimensions: tuple of volume dimensions
        origin: tuple of origin per axis (centered)
        spacing: tuple of uniform spacing per axis
        maxes: tuple of max bounds per axis
    """
    nx, ny, nz = shape
    max_dim = max(nx, ny, nz)

    # Uniform spacing based on largest dimension
    spacing_val = 1.0 / np.float32(max_dim)
    spacing = (np.float32(spacing_val), np.float32(spacing_val), np.float32(spacing_val))

    # Calculate extent for each axis
    extent_x = (nx - 1) * spacing_val
    extent_y = (ny - 1) * spacing_val
    extent_z = (nz - 1) * spacing_val
    origin  = (0.0, 0.0, 0.0)
    maxes = (
        np.float32(origin[0] + extent_x - 1e-6),
        np.float32(origin[1] + extent_y - 1e-6),
        np.float32(origin[2] + extent_z - 1e-6)
    )

    return shape, origin, spacing, maxes


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_raw_file>")
        print(f"  Filename must contain shape and dtype, e.g.:")
        print(f"  richtmyer_meshkov_2048x2048x1920_uint8.raw")
        sys.exit(1)

    raw_file = sys.argv[1]

    if not os.path.isfile(raw_file):
        raise FileNotFoundError(f"Could not locate raw file: {raw_file}")

    shape, dtype = parse_filename(raw_file)
    print(f"   Parsed from filename: shape={shape}, dtype={dtype}")

    # Extract name prefix (everything before the dimensions pattern)
    basename = os.path.splitext(os.path.basename(raw_file))[0]
    prefix = re.split(r'_?\d+x\d+x\d+', basename)[0].rstrip('_')

    # --- Configuration ---
    output_samples = f'{prefix}_big_samples.npy'
    output_gt = f'{prefix}_big_gt.npy'
    output_jitter = f'{prefix}_big_jitter.npy'
    order = 'C'

    # Sampling parameters
    cell_count = 128
    num_batches = 100

    # --- Processing ---
    print(f"1) Reading raw volume ({dtype})...")
    volume = read_raw_volume(raw_file, shape, dtype=dtype, order=order)
    print(f"   Shape: {volume.shape}, dtype: {volume.dtype}")
    vol_min, vol_max = volume.min(), volume.max()
    print(f"   Value range: [{vol_min}, {vol_max}]")

    print("2) Computing mesh parameters (aspect-preserving, centered)...")
    dimensions, origin, spacing, maxes = compute_mesh_params(shape)
    print(f"   Dimensions: {dimensions}")
    print(f"   Origin: {origin}")
    print(f"   Spacing: {spacing}")
    print(f"   Bounds: {origin} to {maxes}")

    print(f"3) Generating {num_batches} batches of {cell_count}^3 samples...")
    big_samples, big_jitter, sample_spacing = generate_samples(cell_count, num_batches, origin, maxes)
    print(f"   big_samples shape: {big_samples.shape}")
    print(f"   big_jitter shape: {big_jitter.shape}")
    print(f"   Sample spacing: {sample_spacing}")

    print("4) Sampling volume at all positions using gpu_sample...")
    flat_samples = big_samples.reshape(-1, 3)
    big_gt = gpu_sample(
        dimensions,
        origin,
        spacing,
        volume.ravel(order='C'),
        flat_samples
    )
    big_gt = big_gt.reshape(num_batches, cell_count**3)
    print(f"   big_gt shape: {big_gt.shape}")
    print(f"   big_gt value range: [{big_gt.min():.4f}, {big_gt.max():.4f}]")

    print("5) Normalizing big_gt to [0,1] and casting to float32...")
    big_gt = ((big_gt.astype(np.float32) - vol_min) / (vol_max - vol_min))
    print(f"   big_gt normalized range: [{big_gt.min():.4f}, {big_gt.max():.4f}]")
    print(f"   big_gt dtype: {big_gt.dtype}")

    print(f"6) Saving outputs...")
    np.save(output_samples, big_samples)
    np.save(output_gt, big_gt)
    np.save(output_jitter, big_jitter)
    print(f"   Saved: {output_samples}")
    print(f"   Saved: {output_gt}")
    print(f"   Saved: {output_jitter}")

    print("Done!")


if __name__ == "__main__":
    main()