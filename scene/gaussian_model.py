from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
import os

import numpy as np
import pyvista as pv
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import torch.nn.functional as F


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, stripped=True):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if not stripped:
                return actual_covariance
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.inverse_scaling_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.weight_activation = torch.sigmoid
        self.inverse_weight_activation = inverse_sigmoid

        self.values_activation = torch.sigmoid
        self.inverse_value_activation = inverse_sigmoid

        # self.values_activation = torch.tanh
        # self.inverse_value_activation = torch.atanh

        # self.values_activation = lambda x: x
        # self.inverse_value_activation = lambda x: x

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._weight = torch.empty(0)
        self._values = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.interpolator = None
        self.interpolation_threshold = 0.05
        self.interpolation_mask = None
        self.last_interpolated_xyz = None
        self.should_interpolate = False
        self.mesh = None
        self.max_scale = 0.02
        self.setup_functions()

    def _apply_cap(self, s):
        # r = torch.linalg.norm(s, dim=1, keepdim=True) + 1e-8
        # factor = torch.clamp(self.max_scale / r, max=1.0)
        # return s * factor
        r = torch.linalg.norm(s, dim=1, keepdim=True) + 1e-8
        r_soft = self.max_scale * torch.tanh(r / self.max_scale)
        return s * (r_soft / r)

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._weight,
            self._values,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._weight,
            self._values,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.last_interpolated_xyz = self._xyz.clone()
        self.interpolation_mask = np.full(self._xyz.shape[0], True)
        self.should_interpolate = True

    @property
    def get_scaling(self):
        return self._apply_cap(self.scaling_activation(self._scaling))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_weight(self):
        return self.weight_activation(self._weight)

    @property
    def get_values(self):
        return self.values_activation(self._values)

    def get_covariance(self, scaling_modifier=1, stripped=True):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation, stripped
        )

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        mesh: pv.PolyData
    ):
        values = pcd.values
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        # self.mins = [
        #     xmin - 0.01,
        #     ymin - 0.01,
        #     zmin - 0.01
        # ]
        # self.maxes = [
        #     xmax + 0.01,
        #     ymax + 0.01,
        #     zmax + 0.01
        # ]
        self.mins = [
            xmin,
            ymin,
            zmin
        ]
        self.maxes = [
            xmax,
            ymax,
            zmax
        ]
        print(self.mins)
        print(self.maxes)

        print(
            f"Number of points at initialisation : {fused_point_cloud.shape[0]}"
        )

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = self.inverse_scaling_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        weights = self.inverse_weight_activation(
            (0.1)
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        values = self.inverse_value_activation(
            torch.tensor(values, dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._weight = nn.Parameter(weights.requires_grad_(True))
        self._values = nn.Parameter(values.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.mesh = mesh
        self.last_interpolated_xyz = self._xyz.clone()
        self.interpolation_mask = np.full(self._xyz.shape[0], True)
        self.should_interpolate = True
        print("Done initializing")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        optimizer_params = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init,
                "name": "xyz",
            },
            {
                "params": [self._weight],
                "lr": training_args.weight_lr,
                "name": "weight",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._values],
                "lr": training_args.values_lr,
                "name": "value",
            },
        ]

        self.optimizer = torch.optim.Adam(optimizer_params, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        attributes = ["x", "y", "z", "value", "weight"]
        for i in range(self._scaling.shape[1]):
            attributes.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            attributes.append("rot_{}".format(i))
        return attributes

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        weights = self._weight.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        values = self._values.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, values, weights, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        # Also produce an ascii version of the .ply file
        self.convert_ply_to_ascii(path)

    def save_ply_activated(self, path):
        # Apply activation functions to get the actual values
        xyz = self.get_xyz.detach().cpu().numpy()  # _xyz (no activation)
        weights = self.get_weight.detach().cpu().numpy()  # sigmoid activation
        scaling = self.get_scaling.detach().cpu().numpy()  # exp activation
        rotation = self.get_rotation.detach().cpu().numpy()  # normalize activation
        values = self.get_values.detach().cpu().numpy()  # sigmoid activation

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, values, weights, scaling, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_weight(self):
        weights_new = self.inverse_weight_activation(
            torch.min(self.get_weight, torch.ones_like(self.get_weight) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(weights_new, "weight")
        self._weight = optimizable_tensors["weight"]

    def load_ply(self, path, mesh):
        plydata = PlyData.read(path)
        print(
            f"Number of points at initialisation : {plydata.elements[0]['x'].shape[0]}"
        )
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        # if normalize:
        #     xyz[:,0] = (xyz[:,0] + 1) / 2
        #     xyz[:,1] = (xyz[:,1] + 1) / 2
        #     xyz[:,2] = (xyz[:,2] - 2) / 2
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        # self.mins = [
        #     xmin - 0.01,
        #     ymin - 0.01,
        #     zmin - 0.01
        # ]
        # self.maxes = [
        #     xmax + 0.01,
        #     ymax + 0.01,
        #     zmax + 0.01
        # ]
        self.mins = [
            xmin,
            ymin,
            zmin
        ]
        self.maxes = [
            xmax,
            ymax,
            zmax
        ]
        print(self.mins)
        print(self.maxes)
        weights = np.asarray(plydata.elements[0]["weight"])[..., np.newaxis]

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        values = np.asarray(plydata.elements[0]["value"])[..., np.newaxis]

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._weight = nn.Parameter(
            torch.tensor(weights, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._values = nn.Parameter(
            torch.tensor(values, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.mesh = mesh

        self.last_interpolated_xyz = self._xyz.clone()
        self.interpolation_mask = np.full(len(self._values), True)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "scaling" : self._scaling,
            "rotation" : self._rotation,
            "weight": self._weight,
            "value": self._values
            }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._weight = optimizable_tensors["weight"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"] 

    def _update_params(self, idxs, ratio):
        N = (ratio[idxs, 0] + 1).float()
        new_weight = self.get_weight[idxs, 0] / N
        new_weight = torch.clamp(new_weight.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_weight = self.inverse_weight_activation(new_weight)
        new_scaling = self._scaling[idxs]  # unchanged in internal space

        scaling = self.get_scaling[idxs]                          # (M, 3)
        rotation = build_rotation(self._rotation[idxs])           # (M, 3, 3)
        noise = torch.randn_like(scaling) * scaling * 0.5         # scale-proportional noise
        perturbed_xyz = self._xyz[idxs] + torch.bmm(rotation, noise.unsqueeze(-1)).squeeze(-1)

        return perturbed_xyz, new_weight, new_scaling, self._rotation[idxs], self._values[idxs]
    
    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None, cells=None, gt=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on weight
        probs = (self.get_weight[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._weight[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
            self._values[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._weight[reinit_idx] = self._weight[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx) 

    def add_new_gs(self, cap_max):
        current_num_points = self._weight.shape[0]
        target_num = min(cap_max, int(1.5 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_weight.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_weight,
            new_scaling,
            new_rotation,
            new_values
        ) = self._update_params(add_idx, ratio=ratio)

        self._weight[add_idx] = new_weight
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz, new_weight, new_scaling, new_rotation, new_values)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._weight = optimizable_tensors["weight"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_weights,
        new_scaling,
        new_rotation,
        new_values
    ):
        d = {
            "xyz": new_xyz,
            "weight": new_weights,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "value": new_values
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._weight = optimizable_tensors["weight"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, max_new_points=None):
        n_points = self.get_xyz.shape[0]

        grad_mag = torch.zeros(n_points, device=self.get_xyz.device)
        grad_mag[:grads.shape[0]] = torch.linalg.norm(grads, dim=-1)

        selected_pts_mask = grad_mag >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        selected_idx = selected_pts_mask.nonzero(as_tuple=True)[0]

        if max_new_points is not None:
            max_new_points = int(max_new_points)
            max_parents = max_new_points // N   # use (N - 1) here if you mean net growth instead
            if max_parents <= 0 or selected_idx.numel() == 0:
                return 0
            if selected_idx.numel() > max_parents:
                keep = torch.topk(
                    grad_mag[selected_idx], k=max_parents, sorted=False
                ).indices
                selected_idx = selected_idx[keep]

            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[selected_idx] = True

        num_added = N * selected_pts_mask.sum().item()
        if num_added == 0:
            return 0

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.get_xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )
        new_scaling = self.inverse_scaling_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_weight = self._weight[selected_pts_mask].repeat(N, 1)
        new_values = self._values[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_weight,
            new_scaling,
            new_rotation,
            new_values,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.get_xyz.device, dtype=torch.bool),
            )
        )
        self.prune_points(prune_filter)
        return num_added

    def densify_and_clone(self, grads, grad_threshold, scene_extent, max_new_points=None):
        n_points = self.get_xyz.shape[0]

        grad_mag = torch.zeros(n_points, device=self.get_xyz.device)
        grad_mag[:grads.shape[0]] = torch.linalg.norm(grads, dim=-1)

        selected_pts_mask = grad_mag >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        selected_idx = selected_pts_mask.nonzero(as_tuple=True)[0]

        if max_new_points is not None:
            max_new_points = int(max_new_points)
            if max_new_points <= 0 or selected_idx.numel() == 0:
                return 0
            if selected_idx.numel() > max_new_points:
                keep = torch.topk(
                    grad_mag[selected_idx], k=max_new_points, sorted=False
                ).indices
                selected_idx = selected_idx[keep]

            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[selected_idx] = True

        num_added = selected_pts_mask.sum().item()
        if num_added == 0:
            return 0

        new_xyz = self._xyz[selected_pts_mask]
        new_weights = self._weight[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_values = self._values[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_weights,
            new_scaling,
            new_rotation,
            new_values,
        )
        return num_added

    def densify_in_empty(self, empty_points, empty_values, new_scale):
        # Concatenate existing and new points for distance computation
        all_points = torch.cat([self.get_xyz, empty_points], dim=0)
        all_dist2 = torch.clamp_min(
            distCUDA2(all_points) * 0.1,
            0.0000001,
        )
        dist2 = all_dist2[len(self.get_xyz):]
        new_scaling = self.inverse_scaling_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # new_scaling = self.inverse_scaling_activation(
        #     (new_scale) # Slightly bigger than 1 cell in 100^3 grid
        #     * torch.ones(
        #         (empty_points.shape[0], 3), dtype=torch.float, device="cuda"
        #     )
        # )
        new_rotation = torch.zeros((empty_points.shape[0], 4), device="cuda")
        new_rotation[:, 0] = 1

        new_weights = self.inverse_weight_activation(
            # torch.mean(self.get_weight)
            0.01
            * torch.ones(
                (empty_points.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        new_values = self.inverse_value_activation(
            empty_values
        )

        self.densification_postfix(
            empty_points,
            new_weights,
            new_scaling,
            new_rotation,
            new_values,
        )

    def densify_and_prune(self, max_grad, min_weight, new_scale, empty_points, empty_values, num_densify=None):
        # xyz_grads = None
        # if self._xyz.grad is not None:
        #     xyz_grads = self._xyz.grad   # (N, 3)

        # if xyz_grads is not None:
        #     remaining = num_densify

        #     added = self.densify_and_clone(
        #         xyz_grads, max_grad, 1.0, max_new_points=remaining
        #     )
        #     if remaining is not None:
        #         remaining -= added

        #     if remaining is None or remaining > 0:
        #         added = self.densify_and_split(
        #             xyz_grads, max_grad, 1.0, max_new_points=remaining
        #         )
        #         if remaining is not None:
        #             remaining -= added


        self.densify_in_empty(empty_points, empty_values, new_scale)

        prune_mask = (self.get_weight < min_weight).squeeze()

        # scales = self.get_scaling  # (N, 3)
        # max_axis_scale = torch.min(scales, dim=1).values  # (N,)
        # prune_mask_scale = max_axis_scale < 0.000001
        # prune_mask = torch.logical_or(prune_mask, prune_mask_scale)

        # print(f"Number of Gaussians pruned: {torch.count_nonzero(prune_mask)}")
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def convert_ply_to_ascii(self, binary_ply_file_path):
        ascii_ply_file_path = binary_ply_file_path.replace(".ply", "_ascii.ply")

        ply_data = PlyData.read(binary_ply_file_path)

        with open(ascii_ply_file_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")

            for element in ply_data.elements:
                f.write(f"element {element.name} {element.count}\n")
                for prop in element.properties:
                    f.write(f"property float {prop.name}\n")

            f.write("end_header\n")

            for element in ply_data.elements:
                for row in element.data:
                    f.write(" ".join(str(val) for val in row) + "\n")

    def cull_exterior_gaussians(self, tolerance=1e-1, inplace=True):
        """
        Remove all gaussians whose centers do not probe to a valid mesh location.

        Args:
            tolerance (float): Unused here, kept for API compatibility.
            inplace (bool): If True, update this model in-place and return the keep mask.
                            If False, only return the keep mask.

        Returns:
            torch.BoolTensor: Mask of gaussians to keep (True = valid sampled position).

        Notes:
            - Uses PyVista sampling against self.mesh instead of enclosed-point checks.
            - A point is kept only if vtkValidPointMask is True and sampled cell data exists.
        """
        if self.mesh is None:
            raise ValueError("self.mesh is None. Load or assign a mesh before culling gaussians.")

        if self._xyz.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self._xyz.device)

        # PyVista sampling is CPU-side
        mesh_samples = self._xyz.detach().cpu().numpy()

        # Clean mesh before probing
        mesh = self.mesh

        probe_mesh = pv.PolyData(mesh_samples)
        probed = probe_mesh.sample(mesh)

        if not mesh.array_names:
            raise ValueError(
                "self.mesh has no data arrays to sample. Attach at least one point/cell array before probing."
            )

        gt_cells = probed[mesh.array_names[0]]
        valid_mask_np = probed["vtkValidPointMask"].astype(bool)

        # Require both a valid probe hit and non-null sampled data
        if gt_cells.ndim == 1:
            sampled_ok = np.isfinite(gt_cells)
        else:
            sampled_ok = np.all(np.isfinite(gt_cells), axis=1)

        keep_np = valid_mask_np & sampled_ok
        keep_mask = torch.from_numpy(keep_np).to(device=self._xyz.device, dtype=torch.bool)

        if not inplace:
            return keep_mask

        # prune_points expects a mask of points to REMOVE
        prune_mask = ~keep_mask

        if torch.any(prune_mask):
            if self.optimizer is not None:
                self.prune_points(prune_mask)
            else:
                self._xyz = nn.Parameter(self._xyz[keep_mask].detach().requires_grad_(True))
                self._scaling = nn.Parameter(self._scaling[keep_mask].detach().requires_grad_(True))
                self._rotation = nn.Parameter(self._rotation[keep_mask].detach().requires_grad_(True))
                self._weight = nn.Parameter(self._weight[keep_mask].detach().requires_grad_(True))
                self._values = nn.Parameter(self._values[keep_mask].detach().requires_grad_(True))

        print(f"Culled {int(prune_mask.sum().item())} exterior gaussians, kept {self._xyz.shape[0]}.")
        return keep_mask