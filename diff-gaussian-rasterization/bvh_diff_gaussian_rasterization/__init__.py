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
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        scales,
        rotations,
        values,
        weights,
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
            raster_settings.scale_modifier,
            volume_mins_x, volume_mins_y, volume_mins_z,
            volume_maxes_x, volume_maxes_y, volume_maxes_z,
            raster_settings.bg,
            raster_settings.use_gaussian_bvh,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        cells, cell_weights, conics = (
            _C.rasterize_gaussians(*args)
        )

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            values,
            weights,
            cells,
            cell_weights,
            conics
        )
        return cells, cell_weights

    @staticmethod
    def backward(ctx, grad_out_cells, grad_out_cell_weights):

        # Restore necessary values from context
        raster_settings = ctx.raster_settings
        (
            means3D,
            scales,
            rotations,
            values,
            weights,
            cells,
            cell_weights,
            conics
        ) = ctx.saved_tensors

        volume_mins_x, volume_mins_y, volume_mins_z = raster_settings.volume_mins
        volume_maxes_x, volume_maxes_y, volume_maxes_z = raster_settings.volume_maxes

        # Restructure args as C++ method expects them
        args = (
            means3D,
            scales,
            rotations,
            conics,
            values,
            weights,
            cells,
            cell_weights,
            raster_settings.scale_modifier,
            volume_mins_x, volume_mins_y, volume_mins_z,
            volume_maxes_x, volume_maxes_y, volume_maxes_z,
            raster_settings.bg,
            grad_out_cells,
            grad_out_cell_weights,
            raster_settings.use_gaussian_bvh,
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
            None
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    volume_mins: tuple[float, float, float]
    volume_maxes: tuple[float, float, float]
    cell_count: int
    bg: float
    scale_modifier: float
    use_gaussian_bvh: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def build_bvh(self, samples, force_debug=False, use_gaussian_bvh=False):
        if force_debug:
            _C.build_bvh(samples, True, use_gaussian_bvh)
        else:
            _C.build_bvh(samples, self.raster_settings.debug, use_gaussian_bvh)
    def forward(
        self,
        means3D,
        scales=None,
        rotations=None,
        values=None,
        weights=None,
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

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            scales,
            rotations,
            values,
            weights,
            raster_settings,
        )

    def intersect(
        self,
        means3D,
        scales=None,
        rotations=None,
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

        # Invoke C++/CUDA intersection routine
        return intersect_gaussians(
            means3D,
            scales,
            rotations,
            raster_settings,
        )