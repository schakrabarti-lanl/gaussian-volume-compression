import sys
from argparse import ArgumentParser, Namespace
from random import randint
import numpy as np

import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm import tqdm
import pyvista as pv

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from gpu_mesh_sampling import gpu_sample
from utils.debug_utils import tensor_to_vtk
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import bounding_box_regularization, create_window, l1_loss, l2_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DEBUG = True


def training(
    dataset,
    opt,
    pipe,
    is_scaled
):
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians, load_iteration=-1, normalized=is_scaled)
    # Make ground truth
    cell_count = 128
    spacing = [
        (gaussians.maxes[0] - gaussians.mins[0]) / (cell_count - 1),
        (gaussians.maxes[1] - gaussians.mins[1]) / (cell_count - 1),
        (gaussians.maxes[2] - gaussians.mins[2]) / (cell_count - 1)
    ]
    x = np.linspace(gaussians.mins[0], gaussians.maxes[0], cell_count)
    y = np.linspace(gaussians.mins[1], gaussians.maxes[1], cell_count)
    z = np.linspace(gaussians.mins[2], gaussians.maxes[2], cell_count)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)
    rot = np.rot90(samples_3d, k=1, axes=(2,0))
    print(f"Mean scaling: {torch.mean(gaussians.get_scaling)}")
    print(f"Mean position: {torch.mean(gaussians.get_xyz, dim=0)}")

    samples_tf = np.flip(rot, axis=2)
    samples_tf_flat = samples_tf.reshape(-1, 3)
    jitter = np.random.uniform(-0.5, 0.5, samples_tf_flat.shape)
    jitter *= np.array(spacing)[None, :]
    jitter = np.zeros_like(jitter)
    samples_tf_flat = np.clip(
        samples_tf_flat + jitter,
        np.array(gaussians.mins),
        np.array(gaussians.maxes)
    )
    gt_cells = gpu_sample(
        gaussians.mesh.dimensions,
        gaussians.mesh.origin,
        gaussians.mesh.spacing,
        gaussians.mesh.point_data[gaussians.mesh.point_data.keys()[0]],
        samples_tf_flat
    )
    gt_cells = gt_cells.reshape(cell_count, cell_count, cell_count)
    gt = torch.tensor(gt_cells).cuda()
    # tensor_to_vtk(gt_cells, "test_gt.vtk", spacing)
    # tensor_to_vtk(gt_weights, "test_gt_weight.vtk", spacing)

    pipe.debug = True
    render_pkg = render(
        gaussians,
        pipe,
        torch.tensor(jitter.ravel(), dtype=torch.float, device="cuda"),
        cell_count,
        # scaling_modifier=2.0
    )
    cells, weights, visibility_filter, radii = (
        render_pkg["cells"],
        render_pkg["weights"],
        render_pkg["visibility_filter"],
        render_pkg["radii"],
    )
    # cells[cells == -1.0] = 0.0
    print(gt.mean())
    l1_l = l1_loss(cells, torch.zeros_like(cells))
    # l1_l.backward()
    mse = torch.mean((cells - gt) ** 2)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
    print(f"Percent invalid samples: {np.count_nonzero(gt_cells == -1) / cell_count ** 3}")
    print(f"False negative percent: {torch.count_nonzero(torch.logical_and(cells == -1, gt != -1)) / cell_count ** 3}")
    print(f"false positive percent: {torch.count_nonzero(torch.logical_and(cells != -1, gt == -1)) / cell_count ** 3}")
    mse2 = torch.mean((cells[torch.logical_and(gt != -1, cells != -1)] - gt[torch.logical_and(gt != -1, cells != -1)]) ** 2)
    psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
    mse3 = torch.mean((gt) ** 2)
    psnr3 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse3 + 1e-8)
    print(f"L2 loss: {mse}")
    print(f"PSNR: {psnr}")
    print(f"PSNR without false positives/negatives: {psnr2}")
    print(f"PSNR of ground truth: {psnr3}")
    # cells_5d = cells.unsqueeze(0).unsqueeze(0).float()
    # gt_5d = gt.unsqueeze(0).unsqueeze(0).float()
    # ssim = structural_similarity_index_measure(
    #     cells_5d, gt_5d,
    #     data_range=1.0,
    #     kernel_size=(11, 11, 11),
    #     sigma=(1.5, 1.5, 1.5)
    # )
    # print(f"SSIM: {ssim}")
    tensor_to_vtk(torch.abs((cells - gt)).detach().cpu().numpy(), f"test_loss.vtk", spacing)
    tensor_to_vtk(cells.detach().cpu().numpy(), f"test.vtk", spacing)

if __name__ == "__main__":
    window = create_window()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--is_scaled", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.is_scaled
    )