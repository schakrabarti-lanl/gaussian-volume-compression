from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    scales,
    rotations,
    values,
    weights,
    jitter,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        scales,
        rotations,
        values,
        weights,
        jitter,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        scales,
        rotations,
        values,
        weights,
        jitter,
        raster_settings,
    ):
        volume_mins_x, volume_mins_y, volume_mins_z = raster_settings.volume_mins
        volume_maxes_x, volume_maxes_y, volume_maxes_z = raster_settings.volume_maxes
        
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            scales,
            rotations,
            values,
            weights,
            jitter,
            raster_settings.scale_modifier,
            volume_mins_x, volume_mins_y, volume_mins_z,
            volume_maxes_x, volume_maxes_y, volume_maxes_z,
            raster_settings.cell_count,
            raster_settings.bg,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, cells, cell_weights, radii, geomBuffer, binningBuffer, imgBuffer = (
            _C.rasterize_gaussians(*args)
        )
        # print(f"Num Rendered: {num_rendered}")

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            values,
            weights,
            jitter,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            cells,
            cell_weights
        )
        return cells, cell_weights, radii

    @staticmethod
    def backward(ctx, grad_out_cells, grad_out_cell_weights, __):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            means3D,
            scales,
            rotations,
            values,
            weights,
            jitter,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            cells,
            cell_weights
        ) = ctx.saved_tensors

        volume_mins_x, volume_mins_y, volume_mins_z = raster_settings.volume_mins
        volume_maxes_x, volume_maxes_y, volume_maxes_z = raster_settings.volume_maxes

        # Restructure args as C++ method expects them
        args = (
            means3D,
            radii,
            scales,
            rotations,
            values,
            weights,
            jitter,
            cells,
            cell_weights,
            raster_settings.scale_modifier,
            volume_mins_x, volume_mins_y, volume_mins_z,
            volume_maxes_x, volume_maxes_y, volume_maxes_z,
            raster_settings.cell_count,
            raster_settings.bg,
            grad_out_cells,
            grad_out_cell_weights,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        (
            grad_means3D,
            grad_scales,
            grad_rotations,
            grad_values,
            grad_weights
        ) = _C.rasterize_gaussians_backward(*args)

        # print(f"Grads computed.")

        grads = (
            grad_means3D,
            grad_scales,
            grad_rotations,
            grad_values,
            grad_weights,
            None,
            None
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    volume_mins: tuple[float, float, float]
    volume_maxes: tuple[float, float, float]
    cell_count: int
    bg: float
    scale_modifier: float
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D,
        scales=None,
        rotations=None,
        values=None,
        weights=None,
        jitter=None,
        debug=False
    ):
        if debug:
            raster_settings = self.raster_settings._replace(debug=debug)
        else:
            raster_settings = self.raster_settings

        if (scales is None or rotations is None):
            raise Exception(
                "Please provide scale/rotation pair!"
            )
        
        if (values is None):
            raise Exception(
                "Please provide scalar values for each Gaussian!"
            )
        
        if (weights is None):
            raise Exception(
                "Please provide scalar weights for each Gaussian!"
            )

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if jitter is None:
            jitter = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            scales,
            rotations,
            values,
            weights,
            jitter,
            raster_settings,
        )

def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale 