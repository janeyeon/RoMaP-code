import json
import math
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays
from threestudio.utils.typing import *

from kiui.cam import convert 






fov = 0.6
rotation_dict = {

    "yanan": torch.tensor([ 
    [1.0, 0.0, 0.0, 0.0],  
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
    ]),
   
    "3d_ovs":torch.tensor([ 
        [1.0, 0.0, 0.0, 0.0],  
        [0.0, 1.0, 0.0, 1.5],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]), 

    "default":torch.tensor([ 
        [1.0, 0.0, 0.0, 0.0],  
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]), 


}



def get_center_and_diag(cam_centers):
    # Convert list of tensors to a single tensor (concatenated along the last axis)
    cam_centers = torch.cat(cam_centers, dim=1)
    
    # Compute the average camera center
    avg_cam_center = torch.mean(cam_centers, dim=1, keepdim=True)
    center = avg_cam_center
    
    # Compute distances from the center and find the maximum distance
    dist = torch.norm(cam_centers - center, dim=0, keepdim=True)
    diagonal = torch.max(dist)
    
    return center.flatten(), diagonal


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    # Initialize Rt as a 4x4 identity matrix
    Rt = torch.zeros((4, 4), dtype=torch.float32)
    Rt[:3, :3] = R.transpose(0, 1)  # Transpose of the rotation matrix
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # Compute C2W (camera-to-world) as the inverse of Rt
    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]  # Extract camera center

    # Apply translation and scaling
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    # Compute the new Rt (world-to-camera) as the inverse of updated C2W
    Rt = torch.linalg.inv(C2W)
    
    return Rt.float()



def convert_pose(C2W):
    flip_yz = torch.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]


def inter_pose(pose_0, pose_1, ratio):
    pose_0 = pose_0.detach().cpu().numpy()
    pose_1 = pose_1.detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return pose


def interpolate_pose(pose_0, pose_1, ratio):
    """
    Interpolates between two 4x4 camera poses based on the given ratio.

    Parameters:
        pose_0 (numpy.ndarray): The first 4x4 camera matrix.
        pose_1 (numpy.ndarray): The second 4x4 camera matrix.
        ratio (float): The interpolation ratio. 0.0 corresponds to pose_0, 1.0 corresponds to pose_1.

    Returns:
        numpy.ndarray: The interpolated 4x4 camera pose.
    """
    # Ensure the input poses are numpy arrays
    pose_0 = pose_0.detach().cpu().numpy()
    pose_1 = pose_1.detach().cpu().numpy()

    # Extract the translation components (last column of the matrix)
    translation_0 = pose_0[:3, 3]
    translation_1 = pose_1[:3, 3]

    # Interpolate the translation
    interpolated_translation = (1 - ratio) * translation_0 + ratio * translation_1

    # Extract the rotation components (top-left 3x3 submatrix)
    rotation_0 = pose_0[:3, :3]
    rotation_1 = pose_1[:3, :3]

    # Perform spherical linear interpolation (slerp) for the rotation
    # Convert rotation matrices to quaternions
    def rotation_matrix_to_quaternion(matrix):
        q = np.empty(4)
        m = matrix
        t = np.trace(m)
        if t > 0.0:
            t = np.sqrt(t + 1.0)
            q[0] = 0.5 * t
            t = 0.5 / t
            q[1] = (m[2, 1] - m[1, 2]) * t
            q[2] = (m[0, 2] - m[2, 0]) * t
            q[3] = (m[1, 0] - m[0, 1]) * t
        else:
            i = 0
            if m[1, 1] > m[0, 0]:
                i = 1
            if m[2, 2] > m[i, i]:
                i = 2
            j = (i + 1) % 3
            k = (j + 1) % 3
            t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0)
            q[i + 1] = 0.5 * t
            t = 0.5 / t
            q[0] = (m[k, j] - m[j, k]) * t
            q[j + 1] = (m[j, i] + m[i, j]) * t
            q[k + 1] = (m[k, i] + m[i, k]) * t
        return q

    def quaternion_to_rotation_matrix(q):
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
        ])

    def slerp(q0, q1, t):
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return q0 + t * (q1 - q0)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        q2 = q1 - q0 * dot
        q2 /= np.linalg.norm(q2)
        return q0 * np.cos(theta) + q2 * np.sin(theta)

    quaternion_0 = rotation_matrix_to_quaternion(rotation_0)
    quaternion_1 = rotation_matrix_to_quaternion(rotation_1)

    interpolated_quaternion = slerp(quaternion_0, quaternion_1, ratio)

    # Convert the interpolated quaternion back to a rotation matrix
    interpolated_rotation = quaternion_to_rotation_matrix(interpolated_quaternion)

    # Construct the interpolated pose
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = interpolated_rotation
    interpolated_pose[:3, 3] = interpolated_translation

    return interpolated_pose





@dataclass
class MultiviewsDataModuleConfig:
    dataroot: str = ""
    train_downsample_resolution: int = 1
    eval_downsample_resolution: int = 1
    train_data_interval: int = 1
    eval_data_interval: int = 1
    batch_size: int = 1
    eval_batch_size: int = 1
    camera_layout: str = "front"

    camera_distance: float = 1

    eval_interpolation: Optional[Any] = (0,1,2,40) # (0, 1, 30)

    rot_name: str = "default"
    fov: float = 0.6

class MultiviewIterableDataset(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        assert self.cfg.batch_size == 1
        scale = 1
        
        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, "transforms.json"), "r")
        )
        assert camera_dict["camera_model"] == "OPENCV"
        frames = camera_dict["frames"]
        frames = frames[:: self.cfg.train_data_interval]
        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []

        self.frame_w = camera_dict["w"] 
        self.frame_h = camera_dict["h"] 
        # self.n_frames = len(frames)
        self.rotation_z_90 = rotation_dict[self.cfg.rot_name]
        c2w_list = []
        for frame in frames:
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            c2w = extrinsic
            #! convert?? 
            c2w_list.append(c2w)
        c2w_list = torch.stack(c2w_list, dim=0)

        if self.cfg.camera_layout == "around":
            pass
        elif self.cfg.camera_layout == "front":
            pass
        else:
            raise ValueError(
                f"Unknown camera layout {self.cfg.camera_layout}. Now support only around and front."
            )
            
        if not (self.cfg.eval_interpolation is None) and self.cfg.eval_interpolation != []:
            
            idn_list = self.cfg.eval_interpolation[:-1]  # ex: [0, 1, 2, 3]
            eval_nums = self.cfg.eval_interpolation[-1]  # last element is the number of intervals
            self.n_frames = (len(idn_list) - 1) * eval_nums

            for idx_pair in zip(idn_list[:-1], idn_list[1:]):
                idx0, idx1 = idx_pair
                frame = frames[idx0]
                intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
                intrinsic[0, 0] = camera_dict["fl_x"] / scale
                intrinsic[1, 1] = camera_dict["fl_y"] / scale
                intrinsic[0, 2] = camera_dict["cx"] / scale
                intrinsic[1, 2] = camera_dict["cy"] / scale

                for ratio in np.linspace(0, 1, eval_nums):
                    img: Float[Tensor, "H W 3"] = torch.zeros(
                        (self.frame_h, self.frame_w, 3)
                    )
                    frames_img.append(img)

                    direction: Float[Tensor, "H W 3"] = get_ray_directions(
                        self.frame_h,
                        self.frame_w,
                        (intrinsic[0, 0], intrinsic[1, 1]),
                        (intrinsic[0, 2], intrinsic[1, 2]),
                        use_pixel_centers=False,
                    )

                    c2w = torch.FloatTensor(
                        inter_pose(c2w_list[idx0], c2w_list[idx1], ratio)
                    )
                    
                    #! convert?? 
                    c2w[:3, 1:3] *= -1
                    c2w = torch.inverse(c2w)

                    near = 0.1
                    far = 1000.0
                    proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
                    proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                    frames_proj.append(proj)

                    rotation_c2w = self.rotation_z_90
                    c2w_rotated = torch.matmul(rotation_c2w, c2w)
                    
                    R = np.transpose(c2w_rotated[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    T = c2w_rotated[:3, 3]
                    
                    
                    W2C = getWorld2View2(R, T)
                    C2W = torch.linalg.inv(W2C)
                    
                    camera_position: Float[Tensor, "3"] =  C2W[:3, 3:4]
                    

                    frames_c2w.append(c2w_rotated)
                    frames_position.append(camera_position)
                    frames_direction.append(direction)
        else:
            for idx, frame in enumerate(frames):
                intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
                intrinsic[0, 0] = camera_dict["fl_x"] / scale
                intrinsic[1, 1] = camera_dict["fl_y"] / scale
                intrinsic[0, 2] = camera_dict["cx"] / scale
                intrinsic[1, 2] = camera_dict["cy"] / scale

                frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
                img = cv2.imread(frame_path)[:, :, ::-1].copy()
                img = cv2.resize(img, (self.frame_w, self.frame_h))
                img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
                frames_img.append(img)
                self.n_frames = len(frames)

                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.frame_h,
                    self.frame_w,
                    (intrinsic[0, 0], intrinsic[1, 1]),
                    (intrinsic[0, 2], intrinsic[1, 2]),
                    use_pixel_centers=False,
                )

                c2w = c2w_list[idx]
                
                #! convert?? 
                c2w[:3, 1:3] *= -1
                c2w = torch.inverse(c2w)


                near = 0.1
                far = 1000.0
                proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                frames_proj.append(proj)
                #! Rotation 
                                
                rotation_c2w =  self.rotation_z_90
                
                c2w_rotated = torch.matmul(rotation_c2w, c2w)

                R = np.transpose(c2w_rotated[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = c2w_rotated[:3, 3]
                
                
                W2C = getWorld2View2(R, T)
                C2W = torch.linalg.inv(W2C)
                
                camera_position: Float[Tensor, "3"] =  c2w_rotated[:3, 3:4]


                # 나머지 작업 계속 수행
                frames_c2w.append(c2w_rotated)
                
                # frames_c2w.append(c2w)
                frames_position.append(camera_position)
                frames_direction.append(direction)

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(
            self.frames_direction, self.frames_c2w, keepdim=True
        )
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )
        # self.fovy = torch.tensor([70.0 * math.pi / 180] * len(self.frames_c2w)).unsqueeze(1)
        self.fovx = torch.tensor([self.cfg.fov] * len(self.frames_c2w)).unsqueeze(1)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        index = torch.randint(0, self.n_frames, (1,)).item()
        return {
            "index": index,
            "rays_o": self.rays_o[index : index + 1],
            "rays_d": self.rays_d[index : index + 1],
            "mvp_mtx": self.mvp_mtx[index : index + 1],
            "c2w": self.frames_c2w[index : index + 1],
            "c2w_3dgs": self.frames_c2w[index : index + 1],
            "camera_positions": self.frames_position[index : index + 1],
            "light_positions": self.light_positions[index : index + 1],
            "gt_rgb": self.frames_img[index : index + 1],
            "height": self.frame_h,
            "width": self.frame_w,
            "fovx":  self.fovx[index : index + 1],
            "elevation" : torch.tensor([0.0]),
            "azimuth": torch.tensor([0.0]),
            "camera_distances": torch.tensor([1]),
            "for_test": self.frames_c2w
        }
        
    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.frames_c2w[index],
            "c2w_3dgs": self.frames_c2w[index],
            "camera_positions": self.frames_position[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index],
            "fovx":  self.fovx[index],
            "elevation" : torch.tensor([0.0]),
            "azimuth": torch.tensor([0.0]),
            "camera_distances": torch.tensor([1]),
            "height": self.frame_h,
            "width": self.frame_w,
        }
    def __len__(self):
        return self.frames_proj.shape[0]


class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        assert self.cfg.eval_batch_size == 1
        # scale = 2**self.cfg.eval_downsample_resolution
        scale = 1

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, "transforms.json"), "r")
        )
        assert camera_dict["camera_model"] == "OPENCV"
        frames = camera_dict["frames"]
        frames = frames[:: self.cfg.eval_data_interval]
        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []
        cam_centers = []
        
        self.frame_w = camera_dict["w"]
        self.frame_h = camera_dict["h"] 
        self.n_frames = len(frames)
        
        
        self.rotation_z_90 = rotation_dict[self.cfg.rot_name]
        c2w_list = []
        for frame in frames:
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            c2w = extrinsic
                        
            c2w_list.append(c2w)
        c2w_list = torch.stack(c2w_list, dim=0)
        
        if self.cfg.camera_layout == "around":
            pass
        elif self.cfg.camera_layout == "front":
            pass
        else:
            raise ValueError(
                f"Unknown camera layout {self.cfg.camera_layout}. Now support only around and front."
            )
            
        if not (self.cfg.eval_interpolation is None) and self.cfg.eval_interpolation != []:
            
            #! 새롭게 커스텀한 eval_interpolation
            idn_list = self.cfg.eval_interpolation[:-1]  # 예: [0, 1, 2, 3]
            eval_nums = self.cfg.eval_interpolation[-1]  # 마지막 요소를 eval_nums로 사용
            
            # 입력된 idn_list를 순회하며 idx0, idx1 쌍을 생성
            for idx_pair in zip(idn_list[:-1], idn_list[1:]):
                idx0, idx1 = idx_pair
                frame = frames[idx0]
                intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
                intrinsic[0, 0] = camera_dict["fl_x"] / scale
                intrinsic[1, 1] = camera_dict["fl_y"] / scale
                intrinsic[0, 2] = camera_dict["cx"] / scale
                intrinsic[1, 2] = camera_dict["cy"] / scale
                
                for ratio in np.linspace(0, 1, eval_nums):
                    img: Float[Tensor, "H W 3"] = torch.zeros(
                        (self.frame_h, self.frame_w, 3)
                    )
                    frames_img.append(img)
                    
                    direction: Float[Tensor, "H W 3"] = get_ray_directions(
                        self.frame_h,
                        self.frame_w,
                        (intrinsic[0, 0], intrinsic[1, 1]),
                        (intrinsic[0, 2], intrinsic[1, 2]),
                        use_pixel_centers=False,
                    )
                    c2w = torch.FloatTensor(
                        inter_pose(c2w_list[idx0], c2w_list[idx1], ratio)
                    )
                    
                    #! convert?? 
                    c2w[:3, 1:3] *= -1
                    c2w = torch.inverse(c2w)
                    # camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)
                    near = 0.1
                    far = 1000.0
                    proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
                    proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                    frames_proj.append(proj)
                    rotation_c2w = self.rotation_z_90
                    # # Z축으로 90도 회전 적용
                    c2w_rotated = torch.matmul(rotation_c2w, c2w)
                    
                    R = np.transpose(c2w_rotated[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    T = c2w_rotated[:3, 3]
                    
                    
                    W2C = getWorld2View2(R, T)
                    C2W = torch.linalg.inv(W2C)
                    cam_centers.append(C2W[:3, 3:4])
                    
                    camera_position: Float[Tensor, "3"] =  C2W[:3, 3:4]
                    
                    # # 새로운 c2w를 사용
                    # camera_position: Float[Tensor, "3"] = c2w_rotated[:3, 3:].reshape(-1)
                    frames_c2w.append(c2w_rotated)
                    frames_position.append(camera_position)
                    frames_direction.append(direction)
        else:
            
            for idx, frame in enumerate(frames):
                intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
                intrinsic[0, 0] = camera_dict["fl_x"] / scale
                intrinsic[1, 1] = camera_dict["fl_y"] / scale
                intrinsic[0, 2] = camera_dict["cx"] / scale
                intrinsic[1, 2] = camera_dict["cy"] / scale

                frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
                # print(frame_path)
                img = cv2.imread(frame_path)[:, :, ::-1].copy()
                img = cv2.resize(img, (self.frame_w, self.frame_h))
                img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
                frames_img.append(img)
                

                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.frame_h,
                    self.frame_w,
                    (intrinsic[0, 0], intrinsic[1, 1]),
                    (intrinsic[0, 2], intrinsic[1, 2]),
                    use_pixel_centers=False,
                )

                c2w = c2w_list[idx]
                
                #! convert?? 
                c2w[:3, 1:3] *= -1
                c2w = torch.inverse(c2w)



                near = 0.1
                far = 1000.0
                K = intrinsic
                proj = [
                    [
                        2 * K[0, 0] / self.frame_w,
                        -2 * K[0, 1] / self.frame_w,
                        (self.frame_w - 2 * K[0, 2]) / self.frame_w,
                        0,
                    ],
                    [
                        0,
                        -2 * K[1, 1] / self.frame_h,
                        (self.frame_h - 2 * K[1, 2]) / self.frame_h,
                        0,
                    ],
                    [
                        0,
                        0,
                        (-far - near) / (far - near),
                        -2 * far * near / (far - near),
                    ],
                    [0, 0, -1, 0],
                ]
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                frames_proj.append(proj)
                
                rotation_c2w =  self.rotation_z_90

                # Z축으로 90도 회전 적용
                c2w_rotated = torch.matmul(rotation_c2w, c2w)
                
                
                R = np.transpose(c2w_rotated[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = c2w_rotated[:3, 3]
                
                
                W2C = getWorld2View2(R, T)
                C2W = torch.linalg.inv(W2C)
                cam_centers.append(C2W[:3, 3:4])
                

                camera_position: Float[Tensor, "3"] =  c2w_rotated[:3, 3:4]
                # 나머지 작업 계속 수행
                frames_c2w.append(c2w_rotated)
                frames_position.append(camera_position)
                frames_direction.append(direction)
                

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(
            self.frames_direction, self.frames_c2w, keepdim=True
        )
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )
        
        self.fovx = torch.tensor([self.cfg.fov] * len(self.frames_c2w)).unsqueeze(1)

    def __len__(self):
        return self.frames_proj.shape[0]

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.frames_c2w[index],
            "c2w_3dgs": self.frames_c2w[index],
            "camera_positions": self.frames_position[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index],
            "fovx":  self.fovx[index],
            # "fovx":  self.fovx[index],
            "elevation" : torch.tensor([0.0]),
            "azimuth": torch.tensor([0.0]),
            "camera_distances": torch.tensor([1]),
            "height": self.frame_h,
            "width": self.frame_w,
            # "radius": self.radius, 
            # "translate": self.translate
            "for_test": self.frames_c2w
        }

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        # batch.update({"height": self.frame_h, "width": self.frame_w})
        batch.update(
            {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "c2w": self.frames_c2w,
            "c2w_3dgs": self.frames_c2w,
            "camera_positions": self.frames_position,
            "light_positions": self.light_positions,
            "gt_rgb": self.frames_img,
            # "fovy":  self.fovy,
            "fovx":  self.fovx,
            "elevation" : torch.stack([torch.tensor([0.0])]*self.frames_proj.shape[0],dim=0),
            "azimuth": torch.stack([torch.tensor([0.0])]*self.frames_proj.shape[0],dim=0),
            "camera_distances": torch.stack([torch.tensor([1])]*self.frames_proj.shape[0],dim=0),
            "height": self.frame_h,
            "width": self.frame_w,
        }
        )
        return batch

@register("multiview-camera-datamodule")
class MultiviewDataModule(pl.LightningDataModule):
    cfg: MultiviewsDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewsDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiviewIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiviewDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            # self.test_dataset = MultiviewDataset(self.cfg, "test")
            self.test_dataset = MultiviewDataset(self.cfg, "val")


    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=1,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

