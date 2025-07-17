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
import math
from gaussiansplatting.submodules.diff_gaussian_rasterization.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.utils.sh_utils import eval_sh


C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb-0.5) / C0 

def SH2RGB(sh):
    return sh * C0 + 0.5
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform.to(torch.float32),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    sh_objs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python:
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
        #     shs = pc.get_features
        #     sh_objs = pc.get_objects
        
        
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        sh_objs = pc._objects_dc
        
        
        
        # sh_objs_view = pc.get_objects.transpose(1, 2).view(-1, 3, (pc.max_obj_sh_degree+1)**2)
        # dir_obj_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_objects.shape[0], 1))
        # dir_obj_pp_normalized = dir_obj_pp/dir_obj_pp.norm(dim=1, keepdim=True)
        
        # sh2obj = eval_sh(pc.active_sh_degree, sh_objs_view, dir_obj_pp_normalized)
        # sh_objs = torch.clamp_min(sh2obj + 0.5, 0.0).unsqueeze(1)
        
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_objects = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sh_objs = sh_objs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
   
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # objects =rendered_objects
    # cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # #! L2d and L3d loss here! 
    # try: 
    #     gt_obj = viewpoint_camera.objects.to("cuda").long()
    #     logits = gt_obj
    #     loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
    #     loss_obj = loss_obj / torch.log(torch.tensor(3))  # normalize to (0,1)
    # except:
    #     breakpoint()
    
    
    #! convert renderd_objects to RGB
    # rendered_objects = (rendered_objects - rendered_objects.min()) / (rendered_objects.max() - rendered_objects.min())
        
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_object": rendered_objects, 
            "depth_3dgs":rendered_objects}
    
    

def render_id(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, classifier=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    screenspace_points = torch.zeros_like(pc._objects_dc.squeeze(), dtype=pc._objects_dc.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform.to(torch.float32),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.detach().clone().requires_grad_(False)
    means2D = screenspace_points
    opacity = pc.get_opacity.detach().clone().requires_grad_(False)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier).detach().clone().requires_grad_(False)
    else:
        scales = pc.get_scaling.detach().clone().requires_grad_(False)
        rotations = pc.get_rotation.detach().clone().requires_grad_(False)

    sh_objs = pc._objects_dc.detach()
    
    
    sh_objs_view = pc.get_objects.transpose(1, 2).view(-1, 3, (pc.max_obj_sh_degree+1)**2)
    dir_obj_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_objects.shape[0], 1))
    dir_obj_pp_normalized = dir_obj_pp/dir_obj_pp.norm(dim=1, keepdim=True)
    
    sh2obj = eval_sh(pc.max_obj_sh_degree, sh_objs_view, dir_obj_pp_normalized)
    object_precomp = torch.clamp_min(sh2obj + 0.5, 0.0)
    
    # object_precomp = pc._objects_dc.detach().squeeze()
    # color_shs = pc._objects_dc.permute(2,0,1).squeeze().permute(1,0) 
        
    rendered_image, radii, rendered_objects = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        sh_objs = sh_objs,
        colors_precomp = object_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )


        
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_object": rendered_objects, 
            "depth_3dgs":rendered_objects}
    
    

def render_certainty(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, classifier=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    screenspace_points = torch.zeros_like(pc._objects_dc.squeeze(), dtype=pc._objects_dc.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform.to(torch.float32),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.detach().clone().requires_grad_(False)
    means2D = screenspace_points
    opacity = pc.get_opacity.detach().clone().requires_grad_(False)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier).detach().clone().requires_grad_(False)
    else:
        scales = pc.get_scaling.detach().clone().requires_grad_(False)
        rotations = pc.get_rotation.detach().clone().requires_grad_(False)

    sh_certainty = pc._certainty.detach().squeeze()
    # sh_certainty = sh_certainty.view(-1, 1, 1).repeat(1,1,3) # [N, 1, 3]
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 무지개 색상 매핑 (matplotlib의 rainbow colormap 사용)
    cmap = plt.get_cmap("rainbow")
    colors = cmap(sh_certainty.cpu().numpy())[:, :3]  # RGB 값 추출

    # Torch Tensor로 변환
    sh_certainty = torch.tensor(colors, dtype=torch.float32).unsqueeze(1).to("cuda")
        
    
    # sh_certainty  = RGB2SH(sh_certainty)
    # color_shs = pc._objects_dc.permute(2,0,1).squeeze().permute(1,0) 
        
    rendered_image, radii, rendered_objects = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = sh_certainty,
        sh_objs = sh_certainty,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )


        
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_object": rendered_objects, 
            "depth_3dgs":rendered_objects}
    
    
