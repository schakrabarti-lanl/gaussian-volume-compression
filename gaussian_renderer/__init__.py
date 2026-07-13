#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel


def render(
    pc: GaussianModel,
    pipe,
    jitter,
    cell_count=100,
    bg=-1.0,
    scaling_modifier=1.0,
    debug=False
):
    """
    Render the scene.

    """

    raster_settings = GaussianRasterizationSettings(
        volume_mins=pc.mins,
        volume_maxes=pc.maxes,
        cell_count=cell_count,
        bg=bg,
        scale_modifier=scaling_modifier,
        debug=pipe.debug if not debug else True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    scales = pc.get_scaling
    rotations = pc.get_rotation

    values = pc.get_values
    weights = pc.get_weight

    # Rasterize visible Gaussians to cells, obtain their radii
    out_cells, out_weights, radii = rasterizer(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        values=values,
        weights=weights,
        jitter=jitter,
        debug=debug
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "cells": out_cells,
        "weights": out_weights,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
    }

    return out
