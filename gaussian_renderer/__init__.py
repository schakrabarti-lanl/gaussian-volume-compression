from bvh_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel
import torch

_rasterizer: GaussianRasterizer | None = None

def init_rasterizer(
    pc: GaussianModel,
    pipe,
    cell_count=100,
    bg=-1.0,
    scaling_modifier=1.0,
    use_gaussian_bvh=False
) -> None:
    """
    Initialize the GaussianRasterizer once with these settings.
    Must be called before any render(...) calls.
    """
    global _rasterizer

    raster_settings = GaussianRasterizationSettings(
        volume_mins=pc.mins,
        volume_maxes=pc.maxes,
        cell_count=cell_count,
        bg=bg,
        scale_modifier=scaling_modifier,
        use_gaussian_bvh=use_gaussian_bvh,
        debug=pipe.debug,
    )

    # store the rasterizer; we'll patch the settings per-call
    _rasterizer = GaussianRasterizer(
        raster_settings=raster_settings,
    )


def build_bvh(samples, debug=False, use_gaussian_bvh=False):
    _rasterizer.build_bvh(samples, debug, use_gaussian_bvh)


def morton3d_unit(xyz: torch.Tensor) -> torch.Tensor:
    """Morton codes for points already in [0, 1]^3."""
    norm = (xyz * ((1 << 21) - 1)).long()

    def part1by2(n):
        n = n & 0x1fffff
        n = (n | (n << 32)) & 0x1f00000000ffff
        n = (n | (n << 16)) & 0x1f0000ff0000ff
        n = (n | (n << 8))  & 0x100f00f00f00f00f
        n = (n | (n << 4))  & 0x10c30c30c30c30c3
        n = (n | (n << 2))  & 0x1249249249249249
        return n

    return part1by2(norm[:, 0]) | (part1by2(norm[:, 1]) << 1) | (part1by2(norm[:, 2]) << 2)


def render(
    pc: GaussianModel,
    debug = False
):
    """
    Render the scene.

    """

    means3D = pc.get_xyz
    scales = pc.get_scaling
    rotations = pc.get_rotation
    values = pc.get_values
    weights = pc.get_weight

    # Sort Gaussians by Morton code for spatial coherence
    order = torch.argsort(morton3d_unit(means3D))
    means3D = means3D[order]
    scales = scales[order]
    rotations = rotations[order]
    values = values[order]
    weights = weights[order]

    out_cells, out_weights = _rasterizer(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        values=values,
        weights=weights,
        debug=debug
    )

    out = {
        "cells": out_cells,
        "weights": out_weights
    }

    return out