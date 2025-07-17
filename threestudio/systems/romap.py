from dataclasses import dataclass, field
import torch
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render, render_id, render_certainty
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
import numpy as np
from gaussiansplatting.utils.loss_utils import  loss_3d_certainty, ssim

from gaussiansplatting.utils.sh_utils import eval_sh

import io  
from PIL import Image  
import open3d as o3d

from torchvision import transforms

import gc
import cv2



from threestudio.utils.pamr import PAMR

import segmentation_refinement as refine


import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
from kiui.typing import *

from lgm.core.options import AllConfigs, Options
from lgm.core.models import LGM
from lgm.mvdream.pipeline_mvdream import MVDreamPipeline

from threestudio.data.uncond import RandomCameraDataset, RandomCameraDataModuleConfig

from threestudio.data.multiview import MultiviewsDataModuleConfig, MultiviewIterableDataset, MultiviewDataset
from threestudio.data.uncond import LossCameraDataset
from threestudio.data.ge import GSLoadDataModuleConfig, GSLoadIterableDataset, GSLoadDataset

from omegaconf import OmegaConf
import random

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import matplotlib.pyplot as plt

from diffusers import StableDiffusion3Pipeline

import trimesh

from gaussiansplatting.utils.graphics_utils import focal2fov, fov2focal

import open3d as o3d


from typing import Optional, Union, List, Dict, Any
from jaxtyping import Float
from torch import Tensor


# Constants for spherical harmonics transformations
C0 = 0.28209479177387814

def RGB2SH(rgb):
    """Convert RGB colors to spherical harmonics coefficients (order 0)."""
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    """Convert order-0 spherical harmonics coefficients to RGB colors."""
    return sh * C0 + 0.5

def seed_everything(seed):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_90_ply(path):
    """
    Load a PLY point cloud, rotate it 90 degrees about the X-axis, and center it.
    Colors are extracted from DC SH features and converted to RGB.
    Returns Open3D PointCloud, coordinates, and color array.
    """
    plydata = PlyData.read(path)

    # Extract XYZ coordinates as a numpy array
    vertices = plydata.elements[0]
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    coords = np.vstack((x, y, z)).T

    # 90-degree rotation matrix around X-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # Apply rotation and recenter
    rotated_coords = coords.dot(rotation_matrix.T)
    centroid = rotated_coords.mean(axis=0)
    rotated_coords = rotated_coords - centroid

    # Update PLY data with rotated coordinates
    plydata.elements[0]['x'] = rotated_coords[:, 0]
    plydata.elements[0]['y'] = rotated_coords[:, 1]
    plydata.elements[0]['z'] = rotated_coords[:, 2]

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:, :, 0])

    # Build Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    return point_cloud, xyz, color

def load_ply(path, save_path):
    """
    Load a PLY point cloud, extract coordinates and SH features, convert to RGB, save as Open3D .ply file.
    Returns Open3D PointCloud, coordinates, color array, and object DC features.
    """
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:, :, 0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)

    objects_dc = np.stack((np.asarray(plydata.elements[0]["obj_dc_0"]),
                           np.asarray(plydata.elements[0]["obj_dc_1"]),
                           np.asarray(plydata.elements[0]["obj_dc_2"])), axis=1)
    return point_cloud, xyz, color, objects_dc

@threestudio.register("romap-system")
class RF(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        load_type: int = 0
        load_path: str = "./load/shapes/stand.obj"

        # global guidance model names
        use_global_attn: bool = False
        global_model_name: str = "stable-diffusion-3-medium-diffusers"
        attention_guidance_start_step: int = 1000
        attention_guidance_timestep_start: int = 850
        attention_guidance_timestep_end: int = 400
        attention_guidance_free_style_timestep_start: int = 500
        record_attention_interval: int = 10
        attntion_nerf_train_interval: int = 2000

        cross_attention_scale: float = 1.0
        self_attention_scale: float = 1.0

        visualize: bool = False
        visualize_save_dir: str = ""

    cfg: Config

    def configure(self) -> None:
        seed_everything(self.cfg.seed)
        self.radius = self.cfg.radius
        self.sh_degree = self.cfg.sh_degree
        self.load_type = self.cfg.load_type
        self.load_path = self.cfg.load_path

        self.ply_path = self.cfg.ply_path                    
        self.if_gen = self.cfg.if_gen                        
        self.if_recon = self.cfg.if_recon
        self.cam_scale = 1                                   
        self.data_config = self.cfg.data
        self.spo_path = self.cfg.spo_path                    

        self.seg_list = self.cfg.seg_list
        self.seg_softmax_list = self.cfg.seg_softmax_list    
        self.test_sh_degree = self.cfg.test_sh_degree        

        self.orig_prompt = self.cfg.prompt_processor.prompt

        # Build segmentation prompt from original prompt and segmentation list
        self.obj_list = []
        self.seg_prompt = self.orig_prompt
        orig_prompt_len = len(self.orig_prompt.split(' '))
        i = orig_prompt_len
        for each_parts in self.seg_list:
            temp_list = []
            self.seg_prompt += " with " if i == orig_prompt_len else "and "
            i += 1
            for part in each_parts:
                self.seg_prompt += part + " "
                i += 1
                temp_list.append(i)
            self.obj_list.append(temp_list)
        self.cfg.seg_prompt_processor.prompt = self.seg_prompt

        self.recon_path = self.cfg.recon_path                

        if self.if_recon:
            # When reconstruction is enabled, set up validation data loader and determine image dimensions
            test_cfg = OmegaConf.structured(
                MultiviewsDataModuleConfig(
                    dataroot=self.data_config.dataroot,
                    rot_name=self.data_config.rot_name,
                    eval_interpolation=self.data_config.eval_interpolation,
                    fov=self.data_config.fov,
                )
            )
            test_batch = MultiviewDataset(test_cfg, "val")
            self.h = test_batch.frame_h
            self.w = test_batch.frame_w
        else:
            # Default image dimensions
            self.h = 512
            self.w = 512
        self.diff_h = 512
        self.diff_w = 512

        self.edit_prompt = self.cfg.edit_prompt
        self.cfg.edit_prompt_processor.prompt = self.edit_prompt
        if self.cfg.edit_prompt is not None:
            self.orig_prompt = self.orig_prompt + " " + self.cfg.edit_prompt
            self.cfg.prompt_processor.prompt = self.orig_prompt

        self.obj_num = len(self.seg_list)

        # Assign a unique color for each object, deterministic for first 6, random for others
        self.each_color = []
        one_hot = torch.tensor([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]
        ])
        for i in range(self.obj_num):
            if i < 6:
                self.each_color.append(one_hot[i])
            else:
                self.each_color.append(torch.rand(3))
        self.each_color = torch.stack(self.each_color, dim=0)
        self.colors = self.each_color.to(dtype=torch.float32, device="cuda").view(self.obj_num, 3, 1, 1).repeat(1, 1, self.diff_h, self.diff_w)

        self.gaussian = GaussianModel(sh_degree=self.sh_degree)
        bg_color = [0, 0, 0]                 # Background color in RGB [black]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Classifier - the precise application may need clarification
        self.channel = 3
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='none').to("cuda")
        self.l1_loss = torch.nn.L1Loss()
        self.edit_angle = self.cfg.edit_angle                  

        self.pamr = PAMR().to("cuda")
        self.refiner = refine.Refiner(device='cuda')
        self.if_segment = self.cfg.if_segment
        self.edit_part_index = self.cfg.edit_part_index
        self.cameras_extent = 4.0

        self.if_fine_edit = self.cfg.if_fine_edit

        self.edit_obj_list = self.cfg.edit_obj_list                   

        # If editing specific parts, maintain a copy of the original Gaussian model
        if len(self.edit_part_index) > 0:
            self.original_gaussian = GaussianModel(sh_degree=self.sh_degree)
        else:
            self.original_gaussian = None

        self.original_image = None                                   
        self.mv_images = None                                        
        self.im23D_mask = None                                       
        self.model = None                                            
        self.selected_gaussian_mask = None                           

        self.im23D = self.cfg.im23D
        if self.im23D:
            self.img_path = self.cfg.img_path
        else:
            self.img_path = None

        self.seg_3d_threshold = 1.0      # 3D segmentation confidence threshold
        self.seg_2d_threshold = 0.34     # 2D segmentation confidence threshold
        self.mse_loss = torch.nn.MSELoss()

    def save_gif_to_file(self, images, output_file):
        """Save a list of PIL images as a GIF file."""
        with io.BytesIO() as writer:
            images[0].save(
                writer, format="GIF", save_all=True, append_images=images[1:],
                duration=100, loop=0
            )
            writer.seek(0)
            with open(output_file, 'wb') as file:
                file.write(writer.read())

    def add_points(self, coords, rgb):
        """
        Add random points within the point cloud bounding box if they are near existing points,
        and slightly perturb colors for visualization/augmentation.
        """
        # Create point cloud from given coordinates
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))

        # Axis-aligned bounding box
        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = 1000000
        points = np.random.uniform(
            low=np.asarray(bbox.min_bound),
            high=np.asarray(bbox.max_bound),
            size=(num_points, 3)
        )

        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

        points_inside = []
        color_inside = []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            # If random point is near the surface (within 1cm), add it with slight color noise
            if np.linalg.norm(point - nearest_point) < 0.01:
                points_inside.append(point)
                color_inside.append(rgb[idx[0]] + 0.2 * np.random.random(3))
                # Or, add without color jitter:
                # color_inside.append(rgb[idx[0]])

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords, coords], axis=0)
        all_rgb = np.concatenate([all_rgb, rgb], axis=0)
        return all_coords, all_rgb
    
    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        
        self.guidance.set_path(self.get_save_path)
        self.guidance.set_seed(self.cfg.seed)
        
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        
        self.seg_prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.seg_prompt_processor
        )
        
        self.edit_prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.edit_prompt_processor
        )
        
        
        self.guidance.set_obj_list(self.obj_list)

    def pcb(self, load_ply):
        """
        Execute point cloud building:
        1. Save the current LGM as a .ply file.
        2. Load it back using the load_ply function.
        3. Add random points within the bounding box for a denser cloud.
        4. Return the resulting BasicPointCloud and additional DC features.
        """
        objects_dc = None  # Not currently used but kept for interface

        save_path = self.get_save_path(f"it{self.true_global_step}-lgm.ply")
        self.lgm(save_path)
        point_cloud, coords, rgb = load_ply(save_path)

        scale = 1  # Adjust the cloud scale as required
        self.point_cloud = point_cloud

        bound = self.radius * scale

        all_coords, all_rgb = self.add_points(coords, rgb)
        # Build dense cloud; normals set to zero
        pcd = BasicPointCloud(
            points=all_coords * bound,
            colors=all_rgb,
            normals=np.zeros((all_coords.shape[0], 3))
        )

        return pcd, objects_dc


    def normalize_tensor(self, input_tensor, input_dim=-1, temp=0.1):
        """
        Normalize a tensor to [0,1], apply a softmax transformation, and re-normalize.

        Args:
            input_tensor (torch.Tensor): Input tensor to normalize.
            input_dim (int): Dimension along which to perform the softmax.
            temp (float): Temperature scaling for the softmax.

        Returns:
            torch.Tensor: Softmax-normalized tensor in [0,1] range.
        """
        # Normalize to [0,1]
        input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())

        # Apply softmax with temperature scaling[1][3][5][6]
        input_tensor = torch.nn.functional.softmax(input_tensor / temp, dim=input_dim)

        # Re-scale to [0,1] after softmax for consistent range
        input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())

        return input_tensor


    def forward(self, batch: Dict[str, Any], renderbackground=None, is_attention=False, is_render=False) -> Dict[str, Any]:
        """
        Render a batch of images, depths, and object masks from the current Gaussian model.
        Stores the most visible radii for visibility filtering.
        """
        if renderbackground is None:
            renderbackground = self.background_tensor
        images, depths, objects = [], [], []
        render_h = self.h if is_render else self.diff_h
        render_w = self.w if is_render else self.diff_w
        self.viewspace_point_list = []

        for id in range(batch['c2w_3dgs'].shape[0]):
            # Compute field-of-view in X and Y directions
            FoVx = batch['fovx'][id]
            FoVy = focal2fov(fov2focal(FoVx, batch['width']), batch['height'])

            viewpoint_cam = Camera(
                c2w=batch['c2w_3dgs'][id],
                FoVx=FoVx,
                FoVy=FoVy,
                image=None,
                gt_alpha_masks=None,
                image_name=None,
                uid=id,
                data_device="cuda",
                height=render_h,
                width=render_w,
                scale=self.cam_scale
            )

            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],  # Not used here, but preserved
                render_pkg["radii"]
            )

            self.viewspace_point_list.append(viewspace_point_tensor)

            # Track largest radii for visibility
            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            # Depth and image post-processing
            depth = render_pkg["depth_3dgs"].permute(1, 2, 0)
            image = F.interpolate(image.unsqueeze(0), size=(render_h, render_w), mode='bilinear', align_corners=False)
            image = image.squeeze().permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            object_mask = F.interpolate(render_pkg["render_object"].unsqueeze(0), size=(render_h, render_w), mode='bilinear', align_corners=False)
            objects.append(object_mask.squeeze())

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        objects = torch.stack(objects, 0)
        self.visibility_filter = (self.radii > 0.0).to("cuda")

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg["render_object"] = objects
        return {**render_pkg}


    def forward_original(self, batch: Dict[str, Any], renderbackground=None, is_attention=False, is_render=False) -> Dict[str, Any]:
        """
        Render a batch of images, depths, and object masks from the saved original Gaussian model.
        """
        render_h = self.h if is_render else self.diff_h
        render_w = self.w if is_render else self.diff_w

        if renderbackground is None:
            renderbackground = self.background_tensor
        images, depths, objects = [], [], []

        for id in range(batch['c2w_3dgs'].shape[0]):
            FoVx = batch['fovx'][id]
            FoVy = focal2fov(fov2focal(FoVx, batch['width']), batch['height'])

            viewpoint_cam = Camera(
                c2w=batch['c2w_3dgs'][id],
                FoVx=FoVx,
                FoVy=FoVy,
                image=None,
                gt_alpha_masks=None,
                image_name=None,
                uid=id,
                data_device="cuda",
                height=batch['height'],
                width=batch['width'],
                scale=self.cam_scale
            )

            render_pkg = render(viewpoint_cam, self.original_gaussian, self.pipe, renderbackground)
            image = render_pkg["render"]
            depth = render_pkg["depth_3dgs"].permute(1, 2, 0)
            image = F.interpolate(image.unsqueeze(0), size=(render_h, render_w), mode='bilinear', align_corners=False)
            image = image.squeeze().permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            object_mask = F.interpolate(render_pkg["render_object"].unsqueeze(0), size=(render_h, render_w), mode='bilinear', align_corners=False)
            objects.append(object_mask.squeeze())

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        objects = torch.stack(objects, 0)
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg["render_object"] = objects
        return {**render_pkg}


    def forward_id(self, batch: Dict[str, Any], renderbackground=None, is_attention=False, is_render=False) -> Dict[str, Any]:
        """
        Render identification maps for each view by assigning a black background and producing masks/images with IDs.
        """
        render_h = self.h if is_render else self.diff_h
        render_w = self.w if is_render else self.diff_w

        # Use solid black as the background irrespective of current background setting
        renderbackground = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        images, depths = [], []
        self.viewspace_point_list = []

        for id in range(batch['c2w_3dgs'].shape[0]):
            FoVx = batch['fovx'][id]
            FoVy = focal2fov(fov2focal(FoVx, batch['width']), batch['height'])
            viewpoint_cam = Camera(
                c2w=batch['c2w_3dgs'][id],
                FoVx=FoVx,
                FoVy=FoVy,
                image=None,
                gt_alpha_masks=None,
                image_name=None,
                uid=id,
                data_device="cuda",
                height=batch['height'],
                width=batch['width'],
                scale=self.cam_scale
            )

            render_pkg = render_id(viewpoint_cam, self.gaussian, self.pipe, renderbackground, classifier=None)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],  # Not used here
                render_pkg["radii"]
            )
            self.viewspace_point_list.append(viewspace_point_tensor)

            # Track largest radii for visibility
            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"].permute(1, 2, 0)
            image = F.interpolate(image.unsqueeze(0), size=(render_h, render_w), mode='bilinear', align_corners=False)
            image = image.squeeze().permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = (self.radii > 0.0).to("cuda")
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {**render_pkg}
    def forward_certainty(self, batch: Dict[str, Any], renderbackground=None, is_attention=False, is_render=False) -> Dict[str, Any]:
        """
        Render certainty maps for a batch using render_certainty.
        The background color is always set to black.
        Returns a dictionary with rendered images, depths, and opacity masks.
        """
        render_h = self.h if is_render else self.diff_h
        render_w = self.w if is_render else self.diff_w

        # Always use black background for certainty map
        renderbackground = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        images, depths = [], []
        self.viewspace_point_list = []

        for id in range(batch["c2w_3dgs"].shape[0]):
            FoVx = batch['fovx'][id]
            FoVy = focal2fov(fov2focal(FoVx, batch['width']), batch['height'])

            # Set up camera for current view
            viewpoint_cam = Camera(
                c2w=batch["c2w_3dgs"][id],
                FoVx=FoVx,
                FoVy=FoVy,
                image=None,
                gt_alpha_masks=None,
                image_name=None,
                uid=id,
                data_device="cuda",
                height=batch["height"],
                width=batch["width"],
                scale=self.cam_scale,
            )

            # Render using the certainty-enabled renderer
            render_pkg = render_certainty(viewpoint_cam, self.gaussian, self.pipe, renderbackground, classifier=None)

            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            radii = render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            # Update max radii for visibility mask
            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"].permute(1, 2, 0)

            # Bilinear interpolation for image resizing
            image = F.interpolate(image.unsqueeze(0), size=(render_h, render_w), mode='bilinear', align_corners=False)
            image = image.squeeze().permute(1, 2, 0)

            images.append(image)
            depths.append(depth)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = (self.radii > 0.0).to("cuda")
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {**render_pkg}

    def find_uncertainty(self):
        """
        Measure SH consistency (variance) and entropy-based color uncertainty for Gaussian point features.
        The results are used to update the ._certainty attribute of the gaussian model.
        """
        with torch.no_grad():
            # Get a validation/test batch of views
            if self.if_recon:
                test_cfg = OmegaConf.structured(
                    MultiviewsDataModuleConfig(
                        dataroot=self.data_config.dataroot,
                        rot_name=self.data_config.rot_name,
                        eval_interpolation=self.data_config.eval_interpolation,
                        fov=self.data_config.fov,
                    )
                )
                test_batch = MultiviewDataset(test_cfg, "val")
            else:
                test_cfg = OmegaConf.structured(RandomCameraDataModuleConfig())
                test_batch = RandomCameraDataset(test_cfg, "test")

            total_sh2objs = []
            # Prepare shape (N_view, 3, SH_dim)
            sh_objs_view = self.gaussian.get_objects.transpose(1, 2).view(-1, 3, (self.gaussian.max_obj_sh_degree + 1) ** 2)

            for i in range(len(test_batch)):
                batch = test_batch[i]
                FoVx = batch['fovx']
                FoVy = focal2fov(fov2focal(FoVx, batch['width']), batch['height'])
                viewpoint_cam = Camera(
                    c2w=batch['c2w_3dgs'],
                    FoVx=FoVx,
                    FoVy=FoVy,
                    image=None,
                    gt_alpha_masks=None,
                    image_name=None,
                    uid=0,
                    data_device="cuda",
                    height=batch["height"],
                    width=batch["width"],
                    scale=self.cam_scale,
                )
                # Compute per-object direction vectors for all points to the camera
                dir_obj_pp = self.gaussian.get_xyz - viewpoint_cam.camera_center.repeat(self.gaussian.get_objects.shape[0], 1)
                dir_obj_pp_normalized = dir_obj_pp / dir_obj_pp.norm(dim=1, keepdim=True)

                sh2obj = eval_sh(self.gaussian.max_obj_sh_degree, sh_objs_view, dir_obj_pp_normalized)
                total_sh2objs.append(sh2obj)
            total_sh2objs = torch.stack(total_sh2objs, dim=0)
            # Variance across all views, averaged over color channels
            variance_per_point = total_sh2objs.var(dim=0, unbiased=False).mean(-1)

            # Compute mean color SH value across views
            total_sh2objs = total_sh2objs.mean(0)
            object_precomp = self.gaussian._objects_dc.detach().squeeze()

            # Compute cosine similarities between each object's predicted color and its label prototype
            seg_labels = torch.cat([self.each_color, torch.zeros(1, 3)], dim=0)
            seg_objects_num = self.obj_num + 1
            points_num = len(object_precomp)
            object_precomp = object_precomp.unsqueeze(0).repeat(seg_objects_num, 1, 1, 1)
            seg_labels = seg_labels.view(seg_objects_num, 1, 1, 3).repeat(1, points_num, 1, 1).to("cuda").to(torch.float32)

            epsilon = 1e-8
            object_norm = object_precomp.norm(dim=-1, keepdim=True).clamp(min=epsilon)
            seg_norm = seg_labels.norm(dim=-1, keepdim=True).clamp(min=epsilon)
            object_precomp = (object_precomp / object_norm).squeeze()
            seg_labels = (seg_labels / seg_norm).squeeze()
            similarities = torch.sum(object_precomp * seg_labels, dim=-1).squeeze()

            # Convert cosine similarities to softmax probabilities[1][3][5][6][9]
            similarities = F.softmax(similarities.view(seg_objects_num, points_num), dim=0)

            # Compute entropy per point from probability (uncertainty metric)
            entropy_per_point = -torch.sum(similarities * torch.log(similarities + 1e-8), dim=0)

            # Clamp high-uncertainty scores (using 10th percentile as upper bound)
            percentile_95 = torch.quantile(entropy_per_point, 0.1)
            entropy_per_point = torch.clamp(entropy_per_point, min=0, max=percentile_95)

            # Combine entropy and SH variance as final certainty
            adjusted_variance = variance_per_point
            certainty = entropy_per_point * adjusted_variance
            # Re-normalize to [0,1]
            certainty = (certainty - certainty.min()) / (certainty.max() - certainty.min() + 1e-6)

            # Store as a torch.nn.Parameter in Gaussian
            self.gaussian._certainty = nn.Parameter(certainty.view(points_num, 1).detach().clone().requires_grad_(True))

    def training_step(self, batch, batch_idx):
        """
        Perform one training iteration for the Gaussian 3D segmentation system.
        This includes guidance, mask-based attention, and 2D/3D segmentation losses.
        """
        # Update learning rate for the Gaussian model
        self.gaussian.update_learning_rate(self.true_global_step)
        
        loss = 0.0

        # Use a black background for all renders
        bg_color = [0, 0, 0]
        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        out = self(batch, testbackground_tensor)
        
        edit_mask = None  # reserved for future use

        # ----- Segmentation-guided Label Loss -----
        reg3d_interval = 2         # not used
        reg3d_k = 50
        reg3d_lambda_val = 2
        reg3d_max_points = 300000
        reg3d_sample_size = 1000

        seg_3d_loss = 0
        if self.if_segment:
            seg_threhold = 200     # not used
            seg_interval = 3       # not used

            seg_prompt_utils = self.seg_prompt_processor()
            images = out["comp_rgb"].to("cuda")
            guidance_eval = self.true_global_step % 400 == 0
            
            # Get segmentation guidance masks & attention
            guidance_out = self.guidance(
                images, seg_prompt_utils, **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
                is_attn=True,
                attn_mask=None,
                global_step=self.true_global_step,
            )

            for id in range(batch['c2w_3dgs'].shape[0]):
                with torch.no_grad():
                    FoVx = batch['fovx'][id]
                    FoVy = focal2fov(fov2focal(FoVx, batch['width']), batch['height'])
                    detached_image = images[id].detach()
                    attention_masks = self.calculate_attention_mask(
                        torch.tensor(guidance_out["attention"]).detach(),
                        detached_image.unsqueeze(0),
                        batch_id=id
                    ).detach()
                    # Set up camera for mask-guided rendering
                    viewpoint_cam = Camera(
                        c2w=batch['c2w_3dgs'][id],
                        FoVx=FoVx, FoVy=FoVy,
                        image=detached_image.permute(2, 0, 1),
                        gt_alpha_masks=attention_masks,
                        image_name="----", uid=id, data_device="cuda",
                        objects=attention_masks, scale=self.cam_scale
                    )
                render_pkg = render_id(viewpoint_cam, self.gaussian, self.pipe, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
                objects = render_pkg["render"].to(torch.float32).requires_grad_(True)
                _, H, W = viewpoint_cam.objects.shape

                # Reshape object masks and targets for pixel-wise loss
                reshaped_object = objects.reshape(self.channel, -1).permute(1, 0)  # (H*W, C)
                full_gt_tensor = attention_masks.to("cuda").to(torch.float32).view(self.channel, -1).permute(1, 0)
                background_mask = (full_gt_tensor.sum(1) >= 0.5).reshape(-1)
                background_tensor_mask = torch.stack([background_mask] * self.channel, dim=1).to("cuda")
                coeff = 30

                if self.if_recon:
                    gt_tensor = full_gt_tensor
                else:
                    reshaped_object = reshaped_object[background_tensor_mask]
                    gt_tensor = full_gt_tensor[background_tensor_mask]
                # L1 loss for segmentation error; could replace with Dice/CrossEntropy for classification, see below [1][3][5][6]
                loss_obj = torch.abs((reshaped_object - gt_tensor))
                seg_3d_loss += loss_obj.mean() * coeff

                # Free attention mask memory
                del attention_masks

            loss += seg_3d_loss

            # Uncertainty step
            if self.global_step % 50 == 0:
                self.find_uncertainty()
            if self.global_step > 50:
                # Use certainty regularization on 3D features
                certainty = self.gaussian._certainty.squeeze()
                seg_labels = self.gaussian._objects_dc.squeeze()
                loss_obj_3d = loss_3d_certainty(
                    self.gaussian._xyz.squeeze().detach(),
                    certainty, seg_labels,
                    reg3d_k, reg3d_lambda_val, reg3d_max_points, reg3d_sample_size,
                    self.global_step
                )
                loss += loss_obj_3d * coeff

        # -------- Visualization / Test Hooks --------
        save_video = 20 if self.if_fine_edit else 50
        if self.true_global_step % save_video == 0 or self.true_global_step == 399:
            with torch.no_grad():
                if self.if_recon:
                    test_cfg = OmegaConf.structured(MultiviewsDataModuleConfig(
                        dataroot=self.data_config.dataroot,
                        rot_name=self.data_config.rot_name,
                        eval_interpolation=self.data_config.eval_interpolation,
                        fov=self.data_config.fov))
                    test_batch = MultiviewDataset(test_cfg, "val")
                else:
                    test_cfg = OmegaConf.structured(RandomCameraDataModuleConfig())
                    test_batch = RandomCameraDataset(test_cfg, "test")
                for i in range(len(test_batch)):
                    tb = test_batch[i]
                    for key in tb.keys():
                        out = tb[key]
                        if isinstance(out, torch.Tensor):
                            out = out.to("cuda").unsqueeze(0)
                        else:
                            out = torch.tensor(out, device="cuda").unsqueeze(0)
                        tb[key] = out
                    self.test_step(tb, 0)
                    self.test_attn_step(tb, 0)
                self.on_test()
                self.on_attn()
                if self.true_global_step > 200:
                    self.on_test_epoch_end()
                del test_cfg
                del test_batch

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        if self.true_global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if self.if_fine_edit and self.true_global_step == 300:
            self.on_test_epoch_end()
        
        return {"loss": loss}


    def on_before_optimizer_step(self, optimizer):
        """
        Apply per-point, per-channel gradient masking.
        - Prevents parts of the Gaussian from being updated during region-specific edits.
        - Also handles pruning/densification on a periodic schedule.
        """
        with torch.no_grad():
            if len(self.edit_part_index) > 0:
                # Prevent editing of non-selected 3D regions
                N, _, _ = self.gaussian._objects_dc.shape
                selected_color = self.each_color[self.edit_part_index].view(1, 3, 1).repeat(1, 1, N).permute(2, 0, 1).to("cuda")
                self.selected_gaussian_mask = ((self.gaussian._objects_dc - selected_color).abs().sum(dim=-1) < 0.3)[:, 0]
                mask_3d = ~self.selected_gaussian_mask.to(torch.bool)
                for idx in range(len(self.viewspace_point_list)):
                    if self.viewspace_point_list[idx].grad is not None:
                        self.viewspace_point_list[idx].grad[mask_3d.view(-1, 1).repeat(1, 3)] = 0
                zeros = torch.zeros_like(self.gaussian.xyz_gradient_accum)
                self.gaussian.xyz_gradient_accum = nn.Parameter(
                    torch.where(self.selected_gaussian_mask.unsqueeze(-1) == 0, zeros, self.gaussian.xyz_gradient_accum).detach().clone().requires_grad_(True))
                self.visibility_filter = self.visibility_filter.to("cuda")
                self.gaussian._features_dc.grad[mask_3d.view(-1, 1, 1).repeat(1, 1, 3)] = 0
                self.gaussian._opacity.grad[mask_3d.view(-1, 1)] = 0
                self.gaussian._xyz.grad[mask_3d.view(-1, 1).repeat(1, 3)] = 0
                self.gaussian._scaling.grad[mask_3d.view(-1, 1).repeat(1, 3)] = 0
                self.gaussian._rotation.grad[mask_3d.view(-1, 1).repeat(1, 4)] = 0

            # Densify & prune periodically
            if self.true_global_step < 900:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    if self.viewspace_point_list[idx].grad is not None:
                        viewspace_point_tensor_grad += self.viewspace_point_list[idx].grad
                self.visibility_filter = self.visibility_filter.to("cuda")
                self.gaussian.max_radii2D = self.gaussian.max_radii2D.to("cuda")
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0:
                    size_threshold = 20 if self.true_global_step > 900 else None
                    self.gaussian.densify_and_prune(0.0005, 0.005, self.cameras_extent, size_threshold) 

                if self.if_fine_edit:
                    if self.true_global_step > 200 and self.true_global_step % 50 == 0:
                        size_threshold = 20 if self.true_global_step > 900 else None
                        self.gaussian.densify_and_prune(0.0005, 0.005, self.cameras_extent, size_threshold)
                else:
                    if self.true_global_step > 200 and len(self.edit_part_index) > 0:
                        if self.true_global_step % 50 == 0:
                            size_threshold = 20 if self.true_global_step > 900 else None
                            self.gaussian.densify_and_prune(0.0005, 0.005, self.cameras_extent, size_threshold)
        
                
    def validation_step(self, batch, batch_idx):
        """
        Runs once per validation batch. Saves a visualization grid comparing input, model output,
        and predicted normals if available.
        """
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                # If ground truth RGB is present, include it
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ] if "rgb" in batch else []
            )
            + [
                # Always add model's composite RGB output
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                # Add predicted normals if available
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ] if "comp_normal" in out else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        """
        This method runs after all validation batches are processed. Use it to aggregate or log epoch-level results.
        """
        pass

    def test_step(self, batch, batch_idx, title='test'):
        """
        Runs once per test batch. Saves a grid showing various outputs for visual and quantitative inspection.
        """
        only_rgb = True
        # Set the background color to black for testing
        bg_color = [0, 0, 0]
        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Run different rendering modes for comparison
        out_1 = self.forward_id(batch, testbackground_tensor, is_render=True)
        out_2 = self(batch, testbackground_tensor, is_render=True)
        render_output = out_2["render_object"]

        # New: visualize uncertainty using forward_certainty
        out_3 = self.forward_certainty(batch, testbackground_tensor, is_render=True)

        # Reshape output images as needed for consistent visualization
        render_output = render_output.to(torch.float32).reshape(self.channel, -1).permute(1, 0)
        render_output = render_output.reshape(self.h, self.w, -1)

        render_certainty = out_3["render_object"].to(torch.float32).reshape(self.channel, -1).permute(1, 0).reshape(self.h, self.w, -1)

        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-{title}/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out_2["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    + [
                        {
                            # Output from forward_id, based on object-precomputed mask
                            "type": "rgb",
                            "img": out_1["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    + (
                        [
                            # Output from forward with object rendering
                            {
                                "type": "rgb",
                                "img": render_output,
                                "kwargs": {"data_format": "HWC"},
                            }
                        ] if "render_object" in out_2 else []
                    )
                    + [
                        {
                            # Output from forward_certainty, showing uncertainty rendering
                            "type": "rgb",
                            "img": render_certainty,
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )

        # Release memory for the outputs after visualization
        del out_1
        del out_2

    def calculate_attention_mask(self, attention_maps, img, batch_id):
        """
        Calculates multi-channel attention masks for segmentation from provided attention maps.
        Refines each attention mask and normalizes them for consistent usage.
        """
        # Resize attention maps to match model dimensions
        attention_maps = F.interpolate(attention_maps, size=(self.diff_h, self.diff_w), mode='bilinear', align_corners=False)

        # Initialize full attention mask tensor
        attention_masks = torch.zeros((self.channel, self.diff_h, self.diff_w)).to("cuda")
        img = img.clip(min=0, max=1) * 255.0

        for i in range(self.obj_num):
            # Normalize each object's attention map
            if self.seg_softmax_list is not None:
                mask_result = self.normalize_tensor(attention_maps[i, batch_id, ...].reshape(-1), -1, self.seg_softmax_list[i])
            else:
                mask_result = self.normalize_tensor(attention_maps[i, batch_id, ...].reshape(-1), -1, 0.2)
            mask_result = mask_result.reshape(self.diff_h, self.diff_w).to("cuda") * 255.0
            
            # Refine the mask with the specified refinement strategy
            out_mask_result = self.refiner.refine(
                img[0].detach().cpu().numpy().astype(np.uint8), 
                mask_result.detach().cpu().numpy(), 
                fast=False, L=900
            )
            out_mask_result = torch.tensor(out_mask_result).squeeze().to("cuda")
            out_mask_result = (out_mask_result - out_mask_result.min()) / (out_mask_result.max() - out_mask_result.min())

            # Apply mask exclusivity: if that pixel is already assigned, override to zero
            out_mask_result = torch.where(attention_masks.sum(0) > 0.5, 0, out_mask_result).to(dtype=torch.float32)
            out_mask_result = torch.stack([out_mask_result] * self.channel, dim=0) * self.colors[i]
            attention_masks += out_mask_result

        # Normalize the full mask for consistent visualization/scaling
        attention_masks = (attention_masks - attention_masks.min()) / (attention_masks.max() - attention_masks.min())
        # Optionally, process further with PAMR if needed
        # attention_masks = self.pamr(img, attention_masks.to(img.dtype).to("cuda").unsqueeze(0)).to(attention_maps.dtype).squeeze()

        return attention_masks.to(dtype=torch.float32)

    def test_attn_step(self, batch, batch_idx, title='test-attn'):
        """
        Run test visualization for attention localization and segmentation attention overlays.
        If fine-edit mode is active, temporarily modify object DC for visualizing edited objects only.
        """
        # If editing specific parts, replace gaussian object dc with mask
        if len(self.edit_part_index) > 0:
            temp_object_dc = self.gaussian._objects_dc.detach()
            self.gaussian._objects_dc = nn.Parameter(
                torch.where(
                    self.selected_gaussian_mask, 
                    1.0, 
                    0.0
                ).view(-1, 1, 1).repeat(1, 1, 3).detach().clone().requires_grad_(True)
            )
        
        # Set black background for test rendering
        bg_color = [0, 0, 0]
        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Forward pass to obtain rendered outputs
        out = self(batch, testbackground_tensor)

        images = out["comp_rgb"].to("cuda")
        prompt_utils = self.seg_prompt_processor()
        guidance_eval = (self.true_global_step % 400 == 0)
        render_object = None

        # If editing specific object indices, extract render_object and restore original DC
        if len(self.edit_part_index) > 0:
            render_object = out["render_object"]
            render_object = render_object.to(torch.float32).reshape(self.channel, -1).permute(1, 0)
            render_object = render_object.reshape(self.h, self.w, -1)
            # Restore original object DC
            self.gaussian._objects_dc = nn.Parameter(temp_object_dc.detach().clone().requires_grad_(True)) 

        # Run the guidance (usually segmentation prompt, such as SAM or LGM attention)
        guidance_out = self.guidance(
            rgb=images,
            prompt_utils=prompt_utils,
            **batch,
            rgb_as_latents=False,
            is_attn=True
        )

        # Compute attention mask using guidance output
        attention_masks = self.calculate_attention_mask(
            torch.tensor(guidance_out["attention"]),
            images,
            batch_id=0
        ).to(torch.float32)

        # Format attention mask for visualization (make [H, W, C], scale to 0-255, uint8)
        attention_masks = attention_masks.view(3, self.diff_h, self.diff_w).permute(1, 2, 0) * 255.
        attention_masks = attention_masks.detach().cpu().numpy().astype(np.uint8)

        # Save a grid containing the attention mask, the RGB image, and (optionally) the object mask
        self.save_image_grid(
            f"it{self.true_global_step}-{title}/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": attention_masks,
                        "kwargs": {},
                    }
                ]
                + [
                    {
                        "type": "rgb",
                        "img": images[0],
                        "kwargs": {},
                    }
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": render_object,
                            "kwargs": {},
                        } 
                    ] if render_object is not None else []
                )
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def save_inner_ply(self):
        """
        Save the inner ply representation for current training step.
        """
        save_inner_path = self.get_save_path(f"inner_{self.true_global_step}.ply")
        self.gaussian.save_inner_ply(save_inner_path)
            
    def on_test_epoch_end(self):
        """
        Called at the end of a testing epoch. Saves final full and inner PLY meshes,
        and produces a colored test PLY by re-loading the mesh and saving colors.
        """
        save_path = self.get_save_path(f"last_3dgs_{self.true_global_step}.ply")
        self.gaussian.save_ply(save_path)
        
        see_inner = True
        if see_inner:
            save_inner_path = self.get_save_path(f"inner_{self.true_global_step}.ply")

        load_ply(
            save_path,
            self.get_save_path(f"it{self.true_global_step}-test-color.ply")
        )

    def on_test(self, title='test'):
        """
        Compile all PNGs from test_step into a video with the specified title.
        """
        self.save_img_sequence(
            f"it{self.true_global_step}-{title}",
            f"it{self.true_global_step}-{title}",
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="test",
            step=self.true_global_step,
        )

    def on_attn(self, title='test-attn'):
        """
        Compile all PNGs from test_attn_step into a video named 'test-attn'.
        """
        self.save_img_sequence(
            f"it{self.true_global_step}-{title}",
            f"it{self.true_global_step}-{title}",
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="test",
            step=self.true_global_step,
        )
            
    def on_seg(self):
        """
        Compile all PNGs from segmentation outputs into a video named 'test-seg'.
        """
        self.save_img_sequence(
            f"it{self.true_global_step}-test-seg",
            f"it{self.true_global_step}-test-seg",
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="test",
            step=self.true_global_step,
        )


    def configure_optimizers(self):
        """
        Sets up optimizers and initializes data, reconstruction, or existing point clouds.
        Also optionally prunes or freezes Gaussians for edit-part indices.
        Initializes testing and evaluation dataloaders, then runs initial test steps.
        """
        self.parser = ArgumentParser(description="Training script parameters")
        self.opt = OptimizationParams(self.parser)
        self.pipe = PipelineParams(self.parser)
        
        # Initialize Gaussian from reconstruction or ply if available, otherwise create from pcd
        if self.recon_path:
            self.gaussian.load_recon_ply(self.recon_path, self.test_sh_degree)
            if len(self.edit_part_index) > 0:
                self.original_gaussian.load_recon_ply(self.recon_path)
        elif self.ply_path:
            self.gaussian.load_ply(self.ply_path)
            if len(self.edit_part_index) > 0:
                self.original_gaussian.load_ply(self.ply_path)
        else:
            point_cloud, objects_dc = self.pcb(load_90_ply)
            self.gaussian.create_from_pcd(point_cloud, self.cameras_extent, objects_dc)
        
        # Remove or freeze initial points if in a specific editing mode
        if len(self.edit_part_index) > 0 and not self.if_fine_edit:
            N, _, _ = self.gaussian._objects_dc.shape
            selected_color = self.each_color[self.edit_part_index].view(1, 3, 1).repeat(1, 1, N).permute(2, 0, 1).to("cuda")
            selected_gaussian_mask = ((self.gaussian._objects_dc.clip(min=0, max=1) - selected_color).abs().sum(dim=1) < 0.3)[:, 0]
            objects_dc = SH2RGB(self.gaussian._objects_dc).clip(0, 1)
            objects_dc = (self.gaussian._objects_dc - self.gaussian._objects_dc.min()) / (self.gaussian._objects_dc.max() - self.gaussian._objects_dc.min())
            selected_gaussian_mask = ((self.gaussian._objects_dc - selected_color).abs().sum(dim=-1) < 0.5)[:, 0]
            
            # Zero out features_dc for selected mask (effectively freezing or removing them)
            zeros = torch.zeros_like(self.gaussian._objects_dc)
            self.gaussian._features_dc = nn.Parameter(
                torch.where(
                    selected_gaussian_mask.view(-1, 1, 1).repeat(1, 1, 3) == 1,
                    zeros,
                    self.gaussian._features_dc
                ).detach().clone().requires_grad_(True)
            )

            self.on_fit_start()
            # Remove points not fitting the current images
            self.remove_gaussians()
    
        self.gaussian.training_setup(self.opt)
        ret = {"optimizer": self.gaussian.optimizer}

        # Setup test batches for evaluation, either from recon or random cam datamodules
        if self.if_recon:
            test_cfg = OmegaConf.structured(
                MultiviewsDataModuleConfig(
                    dataroot=self.data_config.dataroot,
                    rot_name=self.data_config.rot_name,
                    eval_interpolation=self.data_config.eval_interpolation,
                    fov=self.data_config.fov
                )
            )
            test_batch = MultiviewIterableDataset(test_cfg)
        else:
            test_cfg = OmegaConf.structured(RandomCameraDataModuleConfig())
            test_batch = RandomCameraDataset(test_cfg, "test")
            
        for i in range(len(test_batch)):
            tb = test_batch[i]
            for key in tb.keys():
                out = tb[key]
                if isinstance(out, torch.Tensor):
                    out = out.to(device="cuda").unsqueeze(0)
                else:
                    out = torch.tensor(out, device="cuda").unsqueeze(0)
                tb[key] = out

            self.test_step(tb, 0, title='recon')
        self.on_test(title='recon')
        del test_cfg
        del test_batch

        return ret

    def remove_gaussians(self):
        """
        Refines the Gaussian mixture by masking/removing selected parts using both 3D semantic segmentation
        and 2D/3D mask-based filtering, and updates model masks and images accordingly.
        """
        # Recalculate gaussian colors for selection
        N, _, _ = self.gaussian._objects_dc.shape
        selected_color = self.each_color[self.edit_part_index].view(1, 3, 1).repeat(1, 1, N).permute(2, 0, 1).to("cuda")  # [N, 3]
        
        with torch.no_grad():
            # Build test batches depending on reconstruction mode
            if self.if_recon:
                test_cfg = OmegaConf.structured(
                    MultiviewsDataModuleConfig(
                        dataroot=self.data_config.dataroot, 
                        rot_name=self.data_config.rot_name, 
                        eval_interpolation=self.data_config.eval_interpolation, 
                        fov=self.data_config.fov
                    )
                )
                test_batch = MultiviewDataset(test_cfg, "val")
            else:
                test_cfg = OmegaConf.structured(RandomCameraDataModuleConfig())
                test_batch = RandomCameraDataset(test_cfg, "test")

            # Take two views from the chosen angle for foreground mask computation
            front_batch = test_batch[self.edit_angle:self.edit_angle+2]
            bg_color = [0, 0, 0]

            # Compute 3D mask based on precomputed object features and camera direction
            object_test_batch = test_batch[0]
            FoVx = object_test_batch['fovx']
            FoVy = focal2fov(fov2focal(FoVx, object_test_batch['width']), object_test_batch['height'])
            viewpoint_cam = Camera(
                c2w=object_test_batch['c2w_3dgs'], 
                FoVx=FoVx,
                FoVy=FoVy,
                image=None,
                gt_alpha_masks=None,
                image_name=None,
                uid=0,
                data_device="cuda",
                height=object_test_batch['height'],
                width=object_test_batch['width'],
                scale=self.cam_scale
            )
            dir_obj_pp = (self.gaussian.get_xyz - viewpoint_cam.camera_center.repeat(self.gaussian.get_objects.shape[0], 1))
            dir_obj_pp_normalized = dir_obj_pp / dir_obj_pp.norm(dim=1, keepdim=True)
            sh_objs_view = self.gaussian.get_objects.transpose(1, 2).view(-1, 3, (self.gaussian.max_obj_sh_degree + 1) ** 2)
            sh2obj = eval_sh(self.gaussian.max_obj_sh_degree, sh_objs_view, dir_obj_pp_normalized)
            object_precomp = torch.clamp_min(sh2obj + 0.5, 0.0)
            self.selected_gaussian_mask = ((object_precomp.unsqueeze(1) - selected_color).abs().sum(dim=-1) < self.seg_3d_threshold)[:, 0]
    
        # Render images and object maps for the selected batch and background
        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        front_out = self(front_batch, testbackground_tensor)
        front_out_object = self.forward_id(front_batch, testbackground_tensor, is_render=True)

        # Extract and process segmentation mask
        front_seg_map = front_out_object["comp_rgb"].squeeze().detach().clone().requires_grad_(False).permute(0, 3, 1, 2)
        B, c, H, W = front_seg_map.shape
        front_seg_map = front_seg_map.clip(min=0, max=1)

        # Build 2D/3D mask that combines selected object indices
        front_edit_mask = []
        for b in range(B):
            masks = []
            for c in self.edit_part_index:
                color_diff = torch.abs(front_seg_map[b] - self.colors[c])  # [3, H, W]
                mask = torch.where(color_diff.mean(0) < self.seg_2d_threshold, 1, 0)  # [H, W]
                masks.append(mask)
            front_edit_mask.append(torch.stack(masks, dim=0).squeeze())
        front_edit_mask = torch.stack(front_edit_mask, dim=0).to("cuda")  # {B, c, H, W}
        front_images = front_out["comp_rgb"].to("cuda")

        # Use the first image/mask only (previous filtering ensures only relevant objects/mask remain)
        front_edit_mask = front_edit_mask[0].unsqueeze(0)  # [1, H, W]
        front_images = front_images[0].unsqueeze(0)  # [1, H, W, 3]
        front_images = (front_images - front_images.min()) / (front_images.max() - front_images.min())

        gray_value = torch.tensor([0.5, 0.5, 0.5], device=front_images.device).view(1, 1, 1, 3)
        # Set masked regions to gray for editing
        masked_rgb = front_images * (1 - front_edit_mask).unsqueeze(-1) + gray_value * front_edit_mask.unsqueeze(-1)

        # Generate edited segment prompt and mask using guidance model
        prompt_utils = self.prompt_processor()
        edit_prompt_utils = self.edit_prompt_processor()
        guidance_eval = (self.true_global_step % 400 == 0)
        front_batch = test_batch[self.edit_angle:self.edit_angle+1]
        elevation = torch.zeros([1], device="cuda")
        azimuth = torch.zeros([1], device="cuda")
        distance = torch.zeros([1], device="cuda")
        mvp_mtx = torch.zeros([1, 4, 4], device="cuda")
        c2w = torch.zeros([1, 4, 4], device="cuda")
        front_guidance_out = self.guidance(
            rgb=masked_rgb,
            prompt_utils=prompt_utils,
            elevation=elevation, 
            azimuth=azimuth,
            camera_distances=distance,
            mvp_mtx=mvp_mtx.clone(),
            c2w=c2w,
            rgb_as_latents=False,
            is_attn=False,
            edit_obj_list=self.edit_obj_list,
            global_step=0,
            edit_prompt_utils=edit_prompt_utils,
            attn_mask=front_edit_mask,
            is_spo=True
        )

        # Load and process a secondary image (e.g., for coarse editing display)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        image = Image.open(self.spo_path).convert("RGB")
        self.coarse_edit_img = transform(image).to('cuda').permute(1, 2, 0)
        self.front_edit_mask = front_guidance_out["new_attn_mask"].squeeze()

        # Save current edit and mask images
        save_path = self.get_save_path(f"initial_edit_img.png")
        save_img = (self.coarse_edit_img.clip(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        Image.fromarray(save_img).save(save_path)

        save_path = self.get_save_path(f"initial_mask_img.png")
        save_img = (self.front_edit_mask.squeeze().clip(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        Image.fromarray(save_img).save(save_path)

        # Set feature DC for selected points to zero—effectively removing their colors/features for further use
        zero = torch.zeros_like(self.gaussian._features_dc)
        self.gaussian._features_dc = nn.Parameter(
            torch.where(
                self.selected_gaussian_mask.view(-1, 1, 1).repeat(1, 1, 3) == 1,
                zero,
                self.gaussian._features_dc
            ).detach().clone().requires_grad_(True)
        )

        # Mask out point cloud based on the updated image mask—project points, check mask, update
        point_cloud = self.generation_o3d_point_cloud()
        camera_intrinsics = np.array([[600, 0, 256], [0, 600, 256], [0, 0, 1]])
        camera_extrinsics = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, -3]])

        front_edit_mask = front_edit_mask[0]
        input_mask = torch.zeros_like(front_edit_mask, dtype=torch.bool).to("cuda")
        mask_condition = (front_edit_mask > 0) & torch.all(self.coarse_edit_img == torch.tensor([0, 0, 0]).to("cuda"), dim=-1)
        input_mask[mask_condition] = True
        input_mask = input_mask.squeeze().detach().cpu().numpy()

        image_height, image_width = input_mask.shape
        points = np.asarray(point_cloud.points)
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        projection_matrix = camera_intrinsics @ camera_extrinsics
        points_2d = points_homogeneous @ projection_matrix.T
        points_2d /= points_2d[:, 2].reshape(-1, 1)
        uvs = points_2d[:, :2].astype(int)

        valid_uvs = (uvs[:, 0] >= 0) & (uvs[:, 0] < image_width) & (uvs[:, 1] >= 0) & (uvs[:, 1] < image_height)
        valid_points = uvs[valid_uvs]
        mask_values = input_mask[valid_points[:, 1], valid_points[:, 0]] > 0
        final_mask = valid_uvs.copy()
        final_mask[valid_uvs] = mask_values
        final_mask = torch.tensor(final_mask, dtype=torch.bool).to("cuda")

        # Use the final mask to apply hard removal/setup in the Gaussian model 
        self.gaussian.removal_setup(final_mask)
    def generation_o3d_point_cloud(self):
        """
        Generate an Open3D PointCloud object from the current Gaussian model state.
        The method extracts 3D coordinates, sets normals to zeros, processes SH colors,
        and packs them into an Open3D point cloud for export or visualization.
        """
        # Convert xyz and normals to Open3D format
        xyz = self.gaussian._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # Normals initialized to zero; update if you have actual normal data

        # Convert SH (spherical harmonics) features to RGB and normalize
        colors = SH2RGB(self.gaussian._features_dc)
        colors = (colors - colors.min()) / (colors.max() - colors.min())
        colors = colors.squeeze().detach().cpu().numpy()

        # Create the Open3D PointCloud object
        point_cloud = self.convert_to_o3d_point_cloud(xyz, normals, colors)
        return point_cloud

    def generate_image(self):
        """
        Generate an image using Stable Diffusion 3, seeded for reproducibility.
        Saves and returns the first image generated for the configured prompt.
        """
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float16
        )
        seed = 0  # Specify a seed for reproducibility
        generator = torch.Generator().manual_seed(seed)
        pipe.to("cuda")
        prompt = str(self.cfg.prompt_processor.prompt) + " in full view"
        image = pipe(prompt, generator=generator).images[0]
        path = self.get_save_path(f"saved_image.png")
        image.save(path)
        return image

    def convert_to_o3d_point_cloud(self, xyz, normals=None, colors=None):
        """
        Helper function to turn coordinate arrays into an Open3D PointCloud.
        Sets points, normals, and colors if available.
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        if normals is not None:
            point_cloud.normals = o3d.utility.Vector3dVector(normals)
        if colors is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    def process(self, opt: Options, path, model, device, rays_embeddings, proj_matrix, pipe, bg_remover):
        """
        Main pipeline for creating, editing, and exporting point clouds with image guidance.
        1. Loads or generates image (with background removal and recentering).
        2. Applies model to extract multi-view images and generate gaussian point clouds.
        3. Saves both raw and pretty-printed outputs (e.g., 360 video, PNGs) for visualization or evaluation.
        """
        name = os.path.splitext(os.path.basename(path))[0]
        print(f'[INFO] Processing {path} --> {name}')

        # Load input image or generate with diffusion model
        if self.im23D:
            input_image = plt.imread(self.img_path)
            input_image_pil = Image.fromarray(input_image)
            input_image = input_image_pil.resize((512,512))
            input_image = np.array(input_image)
        else:
            input_image = self.generate_image()

        # Remove background from the input image
        carved_image = rembg.remove(input_image, session=bg_remover)  # produces [H, W, 4] output
        carved_image = np.array(carved_image)
        mask = carved_image[..., -1] > 0

        # Recenter the image based on the mask
        image = recenter(carved_image, mask, border_ratio=0.2)
        self.original_image = image[..., :3]
        image = image.astype(np.float32) / 255.0

        # If RGBA, merge with white background
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

        # Generate multi-view images using the pipe
        mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
        mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0)  # [4, H, W, 3]

        # Prepare for gaussian generation: normalize and transform
        input_image_tensor = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device)  # [4, 3, H, W]
        input_size = 256
        input_image_tensor = F.interpolate(input_image_tensor, size=(input_size, input_size), mode='bilinear', align_corners=False)
        input_image_tensor = TF.normalize(input_image_tensor, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        input_image_tensor = torch.cat([input_image_tensor, rays_embeddings], dim=1).unsqueeze(0)  # [1, 4, 9, H, W]

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Generate gaussians
                gaussians = model.forward_gaussians(input_image_tensor)
            model.gs.save_ply(gaussians, path)

            # Render a 360º video from orbiting camera views
            images = []
            elevation = 0
            cam_radius = 1.5
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                cam_poses = torch.from_numpy(
                    orbit_camera(elevation, azi, radius=cam_radius, opengl=True)
                ).unsqueeze(0).to(device)
                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction
                cam_view = torch.inverse(cam_poses).transpose(1, 2)
                cam_view_proj = cam_view @ proj_matrix
                cam_pos = -cam_poses[:, :3, 3]
                scale = min(azi / 360, 1)
                image = model.gs.render(
                    gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale
                )['image']
                images.append(
                    (image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                )
            images = np.concatenate(images, axis=0)
            imageio.mimwrite(self.get_save_path(f"saved_lgm.mp4"), images, fps=30)

            # Save the multi-view images for reference
            self.mv_images = mv_image
            mv_image_uint8 = (mv_image * 255).astype(np.uint8)
            for i in range(len(mv_image_uint8)):
                Image.fromarray(mv_image_uint8[i]).save(self.get_save_path(f"saved_image_{i}.png"))

def lgm(self, result_path):
    """
    Runs the LGM (Large Gaussian Mixture) pipeline using pretrained weights,
    sets up the camera and projection matrix, loads background removal and image 
    generation pipelines, and processes the output with the given configuration.
    """
    # Load all configuration options (assume AllConfigs is class or function containing them)
    opt = AllConfigs

    # Initialize the LGM model
    model = LGM(opt)
    
    # Load pretrained checkpoint
    resume = './lgm/pretrained/model_fp16.safetensors'
    if resume.endswith('safetensors'):
        ckpt = load_file(resume, device='cpu')
    else:
        ckpt = torch.load(resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {resume}')
    
    # Set up device and put model in eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.half().to(device)
    model.eval()

    # Prepare default ray embeddings for the model
    rays_embeddings = model.prepare_default_rays(device)
    
    # Setup camera parameters
    fovy = 49.1  # Field of view (Y axis) in degrees
    znear = 0.5  # Near plane
    zfar = 2.5   # Far plane

    # Compute projection matrix
    tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))
    self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    self.proj_matrix[0, 0] = 1 / tan_half_fov
    self.proj_matrix[1, 1] = 1 / tan_half_fov
    self.proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
    self.proj_matrix[3, 2] = - (zfar * znear) / (zfar - znear)
    self.proj_matrix[2, 3] = 1

    # Load the multi-view image generation pipeline 
    pipe = MVDreamPipeline.from_pretrained(
        "ashawkey/imagedream-ipmv-diffusers",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    pipe = pipe.to(device)

    # Load background remover
    bg_remover = rembg.new_session()
    
    # Run the main process method (handles image generation, Gaussian conversion, and I/O)
    self.process(opt, result_path, model, device, rays_embeddings, self.proj_matrix, pipe, bg_remover)
