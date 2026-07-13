import os
import sys
import uuid
import json
import time
from argparse import ArgumentParser, Namespace
from random import randint
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pyvista as pv

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from gpu_mesh_sampling import gpu_sample, gpu_sampleu
from scene import GaussianModel, Scene
from utils.debug_utils import tensor_to_vtk, analyze_array
from utils.general_utils import get_expon_lr_func, safe_state, build_scaling_rotation
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
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    log_to_file,
    fraction,
    min_weight,
    is_scaled,
    precomputed_samples,
    use_mcmc
):
    print(use_mcmc)
    vtk_files = []
    vtk_files_loss = []
    log_data = []
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel()
    scene = Scene(
        dataset, 
        gaussians, 
        load_iteration=0 if args.model_path else None, 
        normalized=is_scaled, 
        fraction=fraction)
    gaussians.training_setup(opt)
    scene.save(0)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_lv_for_log = 0.0
    ema_lfp_for_log = 0.0
    ema_lfn_for_log = 0.0
    ema_lpsnr_for_log = 0.0
    error_thresh = 0.05
    new_scale = 0.006 # 4 * 100^3 cell?
    densifies = 0

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
    samples_tf = np.flip(rot, axis=2)
    samples_tf_flat = samples_tf.reshape(-1, 3)
    start = time.time()
    if precomputed_samples:
        big_gt = np.load("richtmyer_meshkov_big_gt.npy")
        num_jitters = big_gt.shape[0]
        size = big_gt.shape[1]
        big_samples = np.load("richtmyer_meshkov_big_samples.npy")
        big_jitter = np.load("richtmyer_meshkov_big_jitter.npy")
    else:
        num_jitters = 100
        big_samples = np.tile(samples_tf_flat, (num_jitters, 1))
        big_jitter = np.random.uniform(-0.5, 0.5, big_samples.shape)
        big_jitter *= np.array(spacing)[None, :]
        big_jitter[: cell_count**3, :] = 0
        big_samples = np.clip(
            big_samples + big_jitter,
            np.array(gaussians.mins),
            np.array(gaussians.maxes)
        )
        # probe = pv.PolyData(big_samples)
        # sampled = probe.sample(gaussians.mesh)
        # big_gt = sampled.point_data['value']
        big_gt = gpu_sample(
            gaussians.mesh.dimensions,
            gaussians.mesh.origin,
            gaussians.mesh.spacing,
            gaussians.mesh.point_data['value'],
            big_samples
        )
        # big_gt = gpu_sampleu(
        #     gaussians.mesh.points, 
        #     gaussians.mesh.cell_connectivity.astype(np.int64),
        #     gaussians.mesh.point_data['value'],
        #     big_samples
        # )
        big_gt = big_gt.reshape(num_jitters, cell_count**3)
        # big_gt_weights = big_gt.copy()
        # big_gt_weights[big_gt_weights != -1] = 1
        # big_gt_weights[big_gt_weights == -1] = 0
        big_samples = big_samples.reshape(num_jitters, cell_count**3, 3)
        big_jitter = big_jitter.reshape(num_jitters, cell_count**3, 3)
        end = time.time()
        print(f"Time to sample gt: {end - start}")
    big_jitter_cuda = torch.tensor(big_jitter, dtype=torch.float, device="cuda")
    big_gt_cuda = torch.tensor(big_gt, dtype=torch.float, device="cuda")
    big_samples_cuda = torch.tensor(big_samples, dtype=torch.float, device="cuda")
    gt = big_gt_cuda[0].reshape(cell_count, cell_count, cell_count)
    print(f"Number of invalid samples: {torch.count_nonzero(gt == -1)}")
    # tensor_to_vtk(gt.cpu().numpy(), "test_gt.vtk", spacing)
    # gt_weights = big_gt_weights[0].reshape(cell_count, cell_count, cell_count)
    # gt_weights = torch.tensor(gt_weights).cuda()
    jitter_cuda = big_jitter_cuda[0].ravel()
    loss_samples = np.empty((0, 3))
    loss_vals = np.empty((0, 1))

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    done = 0
    n = False
    for iteration in range(first_iter, opt.iterations + 1):
        deb = False
        iter_start.record()
        jit_idx = 0
        if iteration not in saving_iterations and iteration not in testing_iterations and done != 1:
            jit_idx = np.random.randint(0, num_jitters)
        jitter_cuda = big_jitter_cuda[jit_idx].ravel()
        gt = big_gt_cuda[jit_idx].reshape(cell_count, cell_count, cell_count)
        # gt_weights = big_gt_weights[jit_idx].reshape(cell_count, cell_count, cell_count)
        # gt_weights = torch.tensor(gt_weights).cuda()
        samples_cuda = big_samples_cuda[jit_idx]

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration % 2000 == 0:
            deb = True

        render_pkg = render(
            gaussians,
            pipe,
            jitter_cuda,
            cell_count,
            debug=deb
        )
        cells, weights, visibility_filter, radii = (
            render_pkg["cells"],
            render_pkg["weights"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        # l1_lv = l1_loss(cells, gt)
        l1_lv = torch.abs(cells - gt).mean()
        # l1_lv = torch.mean((cells - gt) ** 2)
        # delta = 1.0
        # residual = cells - gt
        # loss = torch.where(
        #     residual.abs() <= delta,
        #     0.5 * residual ** 2,
        #     delta * (residual.abs() - 0.5 * delta)
        # )
        # l1_lv = loss.mean()
        # TODO: FIX FP AND FN FOR CHANGING CELL COUNTS
        k = 600  # Adjust this to control decay rate
        # fn_mask = torch.logical_and(gt != -1, weights < 0.02)
        # false_negative = torch.exp(-k * weights[fn_mask])
        # false_negative = false_negative[false_negative > 0].mean()
            # false_negative = torch.exp(-k * torch.clamp(weights[fn_mask] - 0.01, 0.0))
        fn_mask = torch.logical_and(gt != -1, weights > 0.0)
        fn_vals = torch.clamp(0.011 - weights, min=0)
        false_negative = args.fn_reg * fn_vals.sum() / ((fn_vals > 0).sum().float() + 1e-8)    
        # overlap_mask = torch.logical_and(gt != -1, weights > 1.0)
        # if overlap_mask.any():
            # overlap_loss = (torch.exp(weights[overlap_mask] - 1) - 1).mean()
        # else:
        #     overlap_loss = torch.tensor(0., device="cuda")
        # fp_mask = torch.logical_and(gt == -1, weights > 0.0)
        # fp_vals = weights[fp_mask]        
        # false_positive = args.fp_reg * fp_vals.sum() / ((fp_vals > 0).sum().float() + 1e-8)
        # if mask.any():
        #     false_positive = (2 * (1 - torch.exp(-k * weights[mask]))).mean()
        # else:
        #     false_positive = torch.tensor(0., device="cuda")
        # min_allowed_scale = min(spacing) / 6.0  # One cell worth of extent
        # mask = gaussians.get_scaling < min_allowed_scale
        # if mask.any():
        #     false_positive = torch.relu(min_allowed_scale - gaussians.get_scaling[mask]).mean()
        # else:
        #     false_positive = torch.tensor(0., device="cuda")
        # loss = l1_lv + false_negative + 0.0000 * overlap_loss
        loss = l1_lv + false_negative
        if gaussians.get_values.shape[0] > args.cap_max:
            n = True
        if use_mcmc:
            loss = loss + args.weight_reg * torch.abs(gaussians.get_weight).mean()
            loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if iteration > opt.densify_from_iter:
                args.fn_reg = args.fn_reg2
            # Logging
            if log_to_file and iteration % 100 == 0:
                mse = torch.mean((cells - gt) ** 2)
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                mse2 = torch.mean((cells[torch.logical_and(weights > 0, gt != -1)] - gt[torch.logical_and(weights > 0, gt != -1)]) ** 2)
                psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
                num_gaussians = gaussians.get_values.shape[0]
                log_data.append({
                    "iteration": iteration,
                    "loss": loss.item(),
                    "l_v": l1_lv.item(),
                    # "false_positive": false_positive.item(),
                    "psnr": psnr.item(),
                    # "psnr2": psnr2.item(),
                    "num_gaussians": num_gaussians
                })
            
            # Progress bar
            if iteration % 500 == 0:
                ema_loss_for_log = 0.9 * loss.item() + 0.1 * ema_loss_for_log
                mse = torch.mean((cells - gt) ** 2)
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                mse2 = torch.mean((cells[torch.logical_and(weights > 0, gt != -1)] - gt[torch.logical_and(weights > 0, gt != -1)]) ** 2)
                psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
                # ema_lv_for_log = 0.4 * l1_lv + 0.6 * ema_lv_for_log
                # ema_lfp_for_log = 0.4 * false_positive + 0.6 * ema_lfp_for_log
                # ema_lfn_for_log = 0.4 * false_negative + 0.6 * ema_lfn_for_log
                # ema_lpsnr_for_log = 0.4 * psnr + 0.6 * ema_lpsnr_for_log
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.{5}f}",
                        # "L_v": f"{ema_lv_for_log:.{5}f}",
                        # "L_fp": f"{ema_lfp_for_log:.{5}f}",
                        # "L_fn": f"{ema_lfn_for_log:.{5}f}",
                        # "PSNR": f"{ema_lpsnr_for_log:.{5}f}"
                    }
                )
                progress_bar.update(500)
                print(f"Num Gaussians: {gaussians.get_values.shape[0]}, psnr: {psnr}, psnr2: {psnr2}, l_v: {l1_lv.item()}")
                # print(f"Num Gaussians: {gaussians.get_values.shape[0]}")
                print(f"w= {(args.weight_reg * torch.abs(gaussians.get_weight).mean()).item()}, {(args.scale_reg * torch.abs(gaussians.get_scaling).mean()).item()}, fn: {false_negative}")
                print(f"Gaussian weight: {torch.mean(gaussians.get_weight)}, gaussian scale: {torch.mean(gaussians.get_scaling)}, scale var: {torch.std(gaussians.get_scaling)}")
                print(f"False negative: {torch.count_nonzero(torch.logical_and(cells == -1, gt != -1))}, fn_reg: {args.fn_reg}")
                # print(f"Overlaps: {torch.count_nonzero(torch.logical_and(gt != -1, weights > 1.0))}")
                print(f"Number of Gaussians to prune: {torch.count_nonzero((gaussians.get_weight < min_weight))}")
               # print(f"Loss samples: {loss_samples.shape}")
                # x = cells * weights
                # low = (x) / (weights + mean_weight)
                # high = (x + mean_weight) / (weights + mean_weight)
                # mm = torch.logical_and(
                #     gt >= low,
                #     gt <= high
                # )
                # print(f"Num between range1: {torch.count_nonzero(gt < low)}, range2: {torch.count_nonzero(gt > high)}")
                # print(f"Num between range1: {torch.count_nonzero(mm)}")
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            if (iteration <= opt.prune_until_iter and
                iteration >= opt.densify_from_iter and
                iteration % opt.densification_interval == 0 and
                iteration not in testing_iterations
            ):
                # if gaussians.get_values.shape[0] > args.cap_max:
                if use_mcmc:
                    # pass
                    dead_mask = (gaussians.get_weight <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask, cells=cells, gt=gt)
                    gaussians.add_new_gs(cap_max=args.cap_max)
                else:
                    loss_idx = torch.topk(
                        torch.abs(cells.ravel() - gt.ravel()),
                        # (cells.ravel() - gt.ravel()) ** 2,
                        # 20 * int((args.cap_max - gaussians.get_values.shape[0]) // (1 + (opt.iterations - iteration) / opt.densification_interval)),
                        min(500000, args.cap_max - gaussians.get_values.shape[0] + torch.count_nonzero(gaussians.get_weight <= min_weight) + 1000)
                    ).indices
                    # loss_idx = (torch.abs(cells.ravel() - gt.ravel()) > 0.01)
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        min_weight,
                        torch.mean(gaussians.get_scaling) / 6.0,
                        0.01,
                        # torch.mean(gaussians.get_weight),
                        samples_cuda[loss_idx],
                        # np.clip(new_vals.cpu().ravel()[loss_idx].reshape(-1, 1), 0.01, 0.99),
                        gt.ravel()[loss_idx].reshape(-1, 1),
                        iteration > opt.densify_until_iter,
                        min(80000, args.cap_max - gaussians.get_values.shape[0] + torch.count_nonzero(gaussians.get_weight <= min_weight) + 1000)
                    )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # if gaussians.get_values.shape[0] > args.cap_max and iteration % 10 == 0:
                if use_mcmc:
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_weight)) * args.noise_lr * xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            # Save
            if iteration in saving_iterations or done == 1:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                cpu_cells = cells.cpu().numpy()
                # tensor_to_vtk(cpu_cells, f"out_vtk/test_{iteration}.vtk", spacing)
                # tensor_to_vtk(torch.abs((cells - gt)).cpu().numpy(), f"out_vtk/test_{iteration}_loss.vtk", spacing)
                vtk_files.append({
                    "name": f"test_{iteration}.vtk",
                    "time": float(iteration)
                })                
                vtk_files_loss.append({
                    "name": f"test_{iteration}_loss.vtk",
                    "time": float(iteration)
                })

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, "/chkpnt{iteration}.pth"),
                )

    series = {
        "file-series-version": "1.0",
        "files": vtk_files
    }
    with open("out_vtk/test.vtk.series", "w") as jf:
        json.dump(series, jf, indent=2)

    series_loss = {
        "file-series-version": "1.0",
        "files": vtk_files_loss
    }
    with open("out_vtk/test_loss.vtk.series", "w") as jf:
        json.dump(series_loss, jf, indent=2)

    if log_to_file:
        log_file_path = os.path.join(scene.model_path, 'training_log.json')
        with open(log_file_path, 'w') as log_file:
            json.dump(log_data, log_file, indent=4)



def prepare_output(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    window = create_window()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--min_weight", type=float, default=0.005)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--is_scaled", action="store_true")
    # parser.add_argument(
    #     "--test_iterations", nargs="+", type=int, default=[i * 1000 for i in range(20)]
    # )
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[]
    )
    # parser.add_argument(
    #     "--save_iterations", nargs="+", type=int, default=[1, 16, 32, 64, 125, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
    # )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--precomputed_samples", action="store_true")
    parser.add_argument("--use_mcmc", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.log_to_file,
        args.fraction,
        args.min_weight,
        args.is_scaled,
        args.precomputed_samples,
        args.use_mcmc
    )

    # All done
    print("\nTraining complete.")
