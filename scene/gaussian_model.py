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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, build_rotation, quaternion_multiply
from utils.debug_utils import save_cal_graph, save_tensor_img
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, rotate_sh_by_matrix, rotate_sh_by_quaternion
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from mem import Motion_Estimation_Module
import commentjson as ctjs
from scene.Motion_Grid import Motion_Grid
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from scene.entropy_models import EntropyBottleneck

class GaussianModel:

    def setup_functions(self):
        
        # @torch.compile
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
                
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.rotation_compose = quaternion_multiply
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, q : int):
        self.active_sh_degree = 0
        self.q = q
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self._xyz_bound_min = None
        self._xyz_bound_max = None
        
        self._d_xyz = None
        self._d_rot = None
        
        self._new_xyz = None
        self._new_rot = None
        
        self._added_xyz = None
        self._added_features_dc = None
        self._added_features_rest = None
        self._added_opacity = None
        self._added_scaling = None
        self._added_rotation = None
        self._added_mask = None
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.color_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self._added_scaling is not None:
            return self.scaling_activation(torch.cat((self._scaling, self._added_scaling), dim=0))
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self._new_rot is not None:
            return self.rotation_activation(self._new_rot)
        elif self._added_rotation is not None:
            return self.rotation_activation(torch.cat((self._rotation, self._added_rotation), dim=0))
        else:
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self._new_xyz is not None:
            return self._new_xyz
        elif self._added_xyz is not None:
            return torch.cat((self._xyz, self._added_xyz), dim=0)
        else:
            return self._xyz
    
    @property
    def get_features(self):
        if self._added_features_dc is not None and self._added_features_rest is not None:
            features_dc = torch.cat((self._features_dc, self._added_features_dc), dim=0)
            features_rest = torch.cat((self._features_rest, self._added_features_rest), dim=0)
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)  
          
    @property
    def get_opacity(self):
        if self._added_opacity is not None:
            return self.opacity_activation(torch.cat((self._opacity, self._added_opacity), dim=0))
        else:
            return self.opacity_activation(self._opacity)
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self,save_type):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        if save_type == 'added':
            for i in range(self._added_features_rest.shape[1]*self._added_features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        else:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_attributes_ex_sh(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, save_type='all'):
        mkdir_p(os.path.dirname(path))
        if save_type=='added':
            xyz = self._added_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._added_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._added_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._added_opacity.detach().cpu().numpy()
            scale = self._added_scaling.detach().cpu().numpy()
            rotation = self._added_rotation.detach().cpu().numpy()       
        elif save_type=='origin':  
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
        elif save_type=='all':
            xyz = self.get_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self.get_features[:,0:1,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self.get_features[:,1:,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self.inverse_opacity_activation(self.get_opacity).detach().cpu().numpy()
            scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
            rotation = self.get_rotation.detach().cpu().numpy()
   
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(save_type = save_type)]  
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if save_type == 'added' and xyz.shape[0]>=4: 
            dtype_ex_sh = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_ex_sh()]
            elements = np.empty(xyz.shape[0], dtype=dtype_ex_sh)
            attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
            f_dc = self._added_features_dc.detach()
            f_rest = self._added_features_rest.detach()
            f = torch.cat((f_dc, f_rest), dim=1).view((attributes.shape[0],(self.max_sh_degree + 1) ** 2,3,1)).permute(3,1,2,0)*self.q
            elements[:] = list(map(tuple,attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,:attributes.shape[0]//4], path=path.replace('point_cloud.ply', 'feature0'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,(attributes.shape[0]//4):2*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature1'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,(attributes.shape[0]//4)*2:3*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature2'))
            self.entropy_bottleneck_added.compress_range(f[:,:,:,(attributes.shape[0]//4)*3:], path=path.replace('point_cloud.ply', 'feature3'))



    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, spatial_lr_scale=0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def load_added_ply(self, path, spatial_lr_scale=0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._added_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._added_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._added_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def load_added_ply_decompress(self, path, spatial_lr_scale=0):
        if not os.path.exists(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply')).exists():
            print(f"no compressed file, maybe no added gaussian")
            return 
        plydata = PlyData.read(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f0 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature0'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f1 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature1'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f2 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature2'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f3 = np.array(self.entropy_bottleneck_added.decompress_range(path = path.replace('point_cloud.ply','feature3'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)

        f = np.concatenate((f0, f1, f2, f3), axis=0)
        features_dc = f[:,:1,:]/self.q
        features_extra = f[:,1:,:]/self.q 
        

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._added_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._added_features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._added_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._added_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "entropy_model":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # print(f"mask shape: {mask.shape}")
                # print(f"stored_state['exp_avg'] shape: {stored_state['exp_avg'].shape}")
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                print(group["name"])
                print(f"mask shape: {mask.shape}, group['params'][0] shape: {group['params'][0].shape}")
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.color_gradient_accum = self.color_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == 'entropy_model':
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask=(self.denom==0).squeeze()
        self.prune_points(prune_mask)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def adding_postfix(self, added_xyz, added_features_dc, added_features_rest, added_opacities, added_scaling, added_rotation):
        d = {"added_xyz": added_xyz,
        "added_f_dc": added_features_dc,
        "added_f_rest": added_features_rest,
        "added_opacity": added_opacities,
        "added_scaling" : added_scaling,
        "added_rotation" : added_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        
    def adding_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.adding_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def adding_and_split(self, grads, grad_threshold, std_scale, num_of_split=1):
        # Extract points that satisfy the gradient condition
        contracted_xyz=self.get_contracted_xyz()                          
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        num_of_split=num_of_split
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, mask)
        stds = std_scale*self.get_scaling[selected_pts_mask].repeat(num_of_split,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(num_of_split,1,1)
        
        added_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(num_of_split, 1)).detach().requires_grad_(True)
        added_scaling = (self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(num_of_split,1) / (0.8*num_of_split))).detach().requires_grad_(True)
        added_rotation = (self.get_rotation[selected_pts_mask].repeat(num_of_split,1)).detach().requires_grad_(True)
        added_features_dc = (self.get_features[:,0:1,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
        added_features_rest = (self.get_features[:,1:,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
        added_opacity = (self.inverse_opacity_activation(self.get_opacity[selected_pts_mask]).repeat(num_of_split,1)).detach().requires_grad_(True)

        self.adding_postfix(added_xyz, added_features_dc, added_features_rest, added_opacity, added_scaling, added_rotation)

    def adding_and_prune(self, training_args, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        if training_args.s2_adding:
            self.adding_and_split(grads, training_args.densify_grad_threshold, training_args.std_scale, training_args.num_of_split)
        if self._added_xyz.shape[0]>0:
            self.prune_added_points(training_args.min_opacity, extent)
        torch.cuda.empty_cache()

    def prune_added_points(self, min_opacity, extent):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(prune_mask, big_points_ws)[-self._added_xyz.shape[0]:]
        valid_points_mask = ~prune_mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        torch.cuda.empty_cache()
        
    def training_one_frame_s2_setup(self, training_args):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        contracted_xyz=self.get_contracted_xyz()                          
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)

        # Spawn
        num_of_spawn=training_args.num_of_spawn
        selected_pts_mask_spawn = torch.where(torch.norm(grads, dim=-1) >= training_args.densify_grad_threshold, True, False)
        selected_pts_mask_spawn = torch.logical_and(selected_pts_mask_spawn, mask)
        N=selected_pts_mask_spawn.sum()
        stds = training_args.std_scale*self.get_scaling[selected_pts_mask_spawn].repeat(num_of_spawn,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask_spawn]).repeat(num_of_spawn,1,1)
        added_xyz_spawn = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask_spawn].repeat(num_of_spawn, 1)).detach().requires_grad_(True)
        added_rotation_spawn = torch.tensor([1.,0.,0.,0.],device='cuda').repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
        added_opacity_spawn = self.inverse_opacity_activation(torch.tensor([0.1],device='cuda')).repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
        added_scaling_spawn = (self.scaling_inverse_activation(self.get_scaling[selected_pts_mask_spawn].repeat(num_of_spawn,1) / (0.8*num_of_spawn))).detach().requires_grad_(True)
        added_features_dc_spawn = (self.get_features[:,0:1,:][selected_pts_mask_spawn].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
        added_features_rest_spawn = (self.get_features[:,1:,:][selected_pts_mask_spawn].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)

        # Clone
        rotation = 2.0 * torch.acos(torch.clamp(self._d_rot[:,0].abs(), min=0.0, max=1.0))
        xyz_mask = torch.norm(self._d_xyz, dim=-1) >= training_args.xyz_threshold
        rot_mask = torch.norm(rotation, dim=-1) >= training_args.rot_threshold
        scale_mask = torch.norm(self._scaling, dim=-1) >= training_args.scale_threshold
        selected_pts_mask_clone = torch.logical_and(xyz_mask, rot_mask)
        selected_pts_mask_clone = torch.logical_and(selected_pts_mask_clone, scale_mask)
        selected_pts_mask_clone = torch.logical_and(selected_pts_mask_clone, mask)

        added_xyz_clone = self.get_xyz[selected_pts_mask_clone].clone().detach().requires_grad_(True)
        added_features_dc_clone = self.get_features[:, 0:1, :][selected_pts_mask_clone].clone().detach().requires_grad_(True)
        added_features_rest_clone = self.get_features[:, 1:, :][selected_pts_mask_clone].clone().detach().requires_grad_(True)
        added_opacity_clone = (self._opacity[selected_pts_mask_clone]/10).clone().detach().requires_grad_(True)
        added_scaling_clone = (self._scaling[selected_pts_mask_clone] - 2).clone().detach().requires_grad_(True)
        added_rotation_clone = self.get_rotation[selected_pts_mask_clone].clone().detach().requires_grad_(True)

        # Combine spawned and cloned Gaussians
        total_spawn = added_xyz_spawn.shape[0]
        total_clone = added_xyz_clone.shape[0]

        self._added_xyz = torch.zeros((total_spawn + total_clone, 3), device="cuda", requires_grad=True)
        self._added_features_dc = torch.zeros((total_spawn + total_clone, self._features_dc.shape[1], self._features_dc.shape[2]), device="cuda", requires_grad=True)
        self._added_features_rest = torch.zeros((total_spawn + total_clone, self._features_rest.shape[1], self._features_rest.shape[2]), device="cuda", requires_grad=True)
        self._added_opacity = torch.zeros((total_spawn + total_clone, 1), device="cuda", requires_grad=True)
        self._added_scaling = torch.zeros((total_spawn + total_clone, 3), device="cuda", requires_grad=True)
        self._added_rotation = torch.zeros((total_spawn + total_clone, 4), device="cuda", requires_grad=True)

        if total_spawn > 0:
            self._added_xyz[:total_spawn].data.copy_(added_xyz_spawn)
            self._added_features_dc[:total_spawn].data.copy_(added_features_dc_spawn)
            self._added_features_rest[:total_spawn].data.copy_(added_features_rest_spawn)
            self._added_opacity[:total_spawn].data.copy_(added_opacity_spawn)
            self._added_scaling[:total_spawn].data.copy_(added_scaling_spawn)
            self._added_rotation[:total_spawn].data.copy_(added_rotation_spawn)
    
        if total_clone > 0:
            self._added_xyz[total_spawn:].data.copy_(added_xyz_clone)
            self._added_features_dc[total_spawn:].data.copy_(added_features_dc_clone)
            self._added_features_rest[total_spawn:].data.copy_(added_features_rest_clone)
            self._added_opacity[total_spawn:].data.copy_(added_opacity_clone)
            self._added_scaling[total_spawn:].data.copy_(added_scaling_clone)
            self._added_rotation[total_spawn:].data.copy_(added_rotation_clone)

        # Optimizer
        l = [
            {'params': [self._added_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "added_xyz"},
            {'params': [self._added_features_dc], 'lr': training_args.feature_lr, "name": "added_f_dc"},
            {'params': [self._added_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "added_f_rest"},
            {'params': [self._added_opacity], 'lr': training_args.opacity_lr, "name": "added_opacity"},
            {'params': [self._added_scaling], 'lr': training_args.scaling_lr, "name": "added_scaling"},
            {'params': [self._added_rotation], 'lr': training_args.rotation_lr, "name": "added_rotation"}
        ]

        self.entropy_bottleneck_added = EntropyBottleneck(channels=(1+self.max_sh_degree)**2,entropy_coder='rangecoder').to('cuda')
        for param in self.entropy_bottleneck_added.parameters():
            l.append({'params': [param], 'lr': 1e-3,  "name": "entropy_model"})
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
               
        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.color_gradient_accum[update_filter] += torch.norm(self._features_dc.grad[update_filter].squeeze(), dim=-1, keepdim=True)

        self.denom[update_filter] += 1

    def query_mem(self):
        mask, self._d_xyz, self._d_rot = self.mem(self._xyz)
        self._new_xyz = self._d_xyz + self._xyz
        self._new_rot = self.rotation_compose(self._rotation, self._d_rot)

    def update_by_mem(self):
        self._xyz = self._new_xyz.clone().detach()
        self._rotation = self._new_rot.clone().detach()
        self._new_xyz = None
        self._new_rot = None
        
    def get_contracted_xyz(self):
        with torch.no_grad():
            xyz = self.get_xyz
            xyz_bound_min, xyz_bound_max = self.get_xyz_bound(90)
            normalzied_xyz=(xyz-xyz_bound_min)/(xyz_bound_max-xyz_bound_min)
            return normalzied_xyz
    
    def get_xyz_bound(self, percentile=90):
        with torch.no_grad():
            if self._xyz_bound_min is None:
                half_percentile = (100 - percentile) / 200
                self._xyz_bound_min = torch.quantile(self._xyz,half_percentile,dim=0)
                self._xyz_bound_max = torch.quantile(self._xyz,1 - half_percentile,dim=0)
            return self._xyz_bound_min, self._xyz_bound_max       
                     
    def training_one_frame_setup(self,training_args):
        print('training_one_frame_setup')
        model = Motion_Grid(q = self.q).to(torch.device("cuda"))
        self.mem=Motion_Estimation_Module(model,self.get_xyz_bound()[0],self.get_xyz_bound()[1])
        self.mem.load_state_dict(torch.load(training_args.mem_path),strict = False)
        
        self._xyz_bound_min = self.mem.xyz_bound_min
        self._xyz_bound_max = self.mem.xyz_bound_max
        self.mem_optimizer = torch.optim.Adam(self.mem.model.get_optparam_groups())  
        self.scheduler = lr_scheduler.StepLR(self.mem_optimizer, step_size=10, gamma=0.1)
                 
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
    def get_masked_gaussian(self, mask):        
        new_gaussian = GaussianModel(self.max_sh_degree)
        new_gaussian._xyz = self.get_xyz[mask].detach()
        new_gaussian._features_dc = self.get_features[:,0:1,:][mask].detach()
        new_gaussian._features_rest = self.get_features[:,1:,:][mask].detach()
        new_gaussian._scaling = self.scaling_inverse_activation(self.get_scaling)[mask].detach()
        new_gaussian._rotation = self.get_rotation[mask].detach()
        new_gaussian._opacity = self.inverse_opacity_activation(self.get_opacity)[mask].detach()
        new_gaussian.xyz_gradient_accum = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.color_gradient_accum = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.denom = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.max_radii2D = torch.zeros((new_gaussian._xyz.shape[0]), device="cuda")
        return new_gaussian
    
    def query_mem_eval(self):
        with torch.no_grad():
            self.mem.model.is_train = False
            mask, self._d_xyz, self._d_rot = self.mem(self.get_xyz)
            self._new_xyz = self._d_xyz + self._xyz
            self._new_rot = self.rotation_compose(self._rotation, self._d_rot)


class GaussianModel_base:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, is_train: bool, q: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.is_train = is_train
        self.q = q
        self.entropy_bottleneck = EntropyBottleneck(channels=(1+sh_degree)**2,entropy_coder='rangecoder').to('cuda')
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.is_train:
            noise_scale = (torch.max(self._scaling) - torch.min(self._scaling)) / 255.0
            noise = torch.rand_like(self._scaling) * noise_scale - noise_scale / 2.0
            return self.scaling_activation(self._scaling+noise)
        else:    
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.is_train:
            noise_scale = (torch.max(self._rotation) - torch.min(self._rotation)) / 255.0
            noise = torch.rand_like(self._rotation) * noise_scale - noise_scale / 2.0
            return self.rotation_activation(self._rotation + noise)
        else:    
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self.is_train:
            noise_scale = (torch.max(self._xyz) - torch.min(self._xyz)) / 65535.0
            noise = torch.rand_like(self._xyz) * noise_scale - noise_scale / 2.0
            return self._xyz + noise
        else:    
            return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        feature = torch.cat((features_dc, features_rest), dim=1)
        if self.is_train:
            # return feature
            half = 0.5/self.q
            noise = torch.rand(feature.shape,device = 'cuda')/self.q-half
            return feature+noise
        else:
            return feature
    
    @property
    def get_opacity(self):
        if self.is_train:
            noise_scale = (torch.max(self._opacity) - torch.min(self._opacity)) / 255.0
            noise = torch.rand_like(self._opacity) * noise_scale - noise_scale / 2.0
            return self.opacity_activation(self._opacity+noise)
        else:
            return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        for param in self.entropy_bottleneck.parameters():
            l.append({'params': [param], 'lr': 1e-3,  "name": "entropy_model"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_attributes_full(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc_full = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_full = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self._features_dc.detach()
        f_rest = self._features_rest.detach()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_ex_sh = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_full()]

        elements = np.empty(xyz.shape[0], dtype=dtype_ex_sh)
        elements_full = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes_full = np.concatenate((xyz, normals, f_dc_full, f_rest_full, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        f = torch.cat((f_dc, f_rest), dim=1).view((attributes.shape[0],(self.max_sh_degree + 1) ** 2,3,1)).permute(3,1,2,0)*self.q
        elements[:] = list(map(tuple, attributes))
        elements_full[:] = list(map(tuple, attributes_full))
        el_full = PlyElement.describe(elements_full, 'vertex')
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
        # PlyData([el_full]).write(path)
        PlyData([el_full]).write(path)

        self.entropy_bottleneck.compress_range(f[:,:,:,:attributes.shape[0]//4], path=path.replace('point_cloud.ply', 'feature0'))
        self.entropy_bottleneck.compress_range(f[:,:,:,(attributes.shape[0]//4):2*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature1'))
        self.entropy_bottleneck.compress_range(f[:,:,:,(attributes.shape[0]//4)*2:3*(attributes.shape[0]//4)], path=path.replace('point_cloud.ply', 'feature2'))
        self.entropy_bottleneck.compress_range(f[:,:,:,(attributes.shape[0]//4)*3:], path=path.replace('point_cloud.ply', 'feature3'))

        

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs exceps DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_decompress_ply(self, path):
        plydata = PlyData.read(path.replace('point_cloud.ply', 'point_cloud_exp_sh.ply'))
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f0 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature0'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f1 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature1'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f2 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature2'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)
        f3 = np.array(self.entropy_bottleneck.decompress_range(path = path.replace('point_cloud.ply','feature3'))).reshape((self.max_sh_degree + 1) ** 2,3,-1).transpose(2,0,1)

        f = np.concatenate((f0, f1, f2, f3), axis=0)
        features_dc = f[:,:1,:]/self.q
        features_extra = f[:,1:,:]/self.q 

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "entropy_model":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == 'entropy_model':
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1