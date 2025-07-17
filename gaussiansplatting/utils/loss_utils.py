# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
import numpy as np

def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]
    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]
    
    
    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]
    
    # #! find nearest neighbors in frustrum
    # fov = 30
    
    # # Convert FOV from degrees to radians
    # fov_radians = torch.deg2rad(torch.tensor(fov / 2))
    
    # # Step 1: Calculate opposite direction for each sample
    # directions = -sample_features  # shape: (n, 3)
    # directions = directions / torch.norm(directions, dim=1, keepdim=True)  # shape: (n, 3)
    
    # # Step 2: Calculate vectors from each sample to all points in the point cloud
    # # Using broadcasting: shape of vectors becomes (n, N, 3)
    # vectors = features.unsqueeze(0) - sample_features.unsqueeze(1)  # shape: (n, N, 3)
    # norms = torch.norm(vectors, dim=2)  # shape: (n, N)
    
    # # Avoid division by zero
    # norms[norms == 0] = float('inf')
    
    # # Normalize vectors
    # unit_vectors = vectors / norms.unsqueeze(2)  # shape: (n, N, 3)
    
    # # Step 3: Calculate dot products with direction vectors to filter within FOV
    # dot_products = torch.einsum("nij,nj->ni", unit_vectors, directions)  # shape: (n, N)
    # cone_mask = dot_products >= torch.cos(fov_radians)  # shape: (n, N)
    # # Step 4: Filter points and predictions within the cone
    # # distances_in_cone = torch.where(cone_mask, norms, torch.full_like(norms, float('inf')))
    
    # distances_in_cone = torch.where(cone_mask, norms, torch.full_like(norms, float('inf')))
    
    
    # nearest_distances, nearest_indices = torch.topk(distances_in_cone, k, largest=False, dim=1)
    
    # neighbor_preds = predictions[nearest_indices]
    
    # #! Compute KL divergence
    # kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    # loss = kl.sum(dim=-1).mean()
    # # Normalize loss into [0, 1]
    # num_classes = predictions.size(1)
    # normalized_loss = loss / num_classes
            
    #! L1 loss 
    # 걍 reference랑 L1 distance를 재자 ㅋㅋㅋㅋ
    sample_preds = sample_preds.unsqueeze(1).repeat(1, k, 1) # [sample_size, k, 3]
    loss = torch.abs(sample_preds - neighbor_preds).sum(dim=-1).mean()
    normalized_loss = loss
    
    # #! MAD loss 
    # # no need to be probability
    # neighbor_mean = neighbor_preds.mean(dim=1).unsqueeze(1).repeat(1, k, 1)
    # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6) 
    # mad = cos(neighbor_mean, neighbor_preds).sum(dim=-1) / k
    # loss = mad.mean()
    # normalized_loss = loss
    
    # #! Total variance loss
    # # 데이터의 평균을 계산 (1 x 3)
    # neighbor_mean = neighbor_preds.mean(dim=1).unsqueeze(1).repeat(1, k, 1)
    
    # # 평균을 기준으로 데이터를 중앙정렬
    # centered_data = neighbor_mean - neighbor_preds # (sample_size, k, 3)
    
    # # 공분산 행렬 계산 (3 x 3)
    # # covariance_matrix = centered_data.permute(0, 2, 1) @ centered_data / (k - 1)
    
    # covariance_matrix  = torch.bmm(centered_data, centered_data.permute(0, 2, 1)) / (k - 1) # {sample_size, k, k}
    
    # # 공분산 행렬의 Trace를 계산하여 총 변동성 구하기
    # total_variance = covariance_matrix.diagonal(dim1=1, dim2=2).sum(dim=1) # (sample_size)
    # loss = total_variance.mean()
    # normalized_loss = loss

    return lambda_val * normalized_loss





# def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
#     """
#     Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
#     and the KL divergence.
    
#     :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
#     :param predictions: Tensor of shape (N, C), where C is the number of classes.
#     :param k: Number of neighbors to consider.
#     :param lambda_val: Weighting factor for the loss.
#     :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
#     :param sample_size: Number of points to randomly sample for computing the loss.
    
#     :return: Computed loss value.
#     # """
#     # # Conditionally downsample if points exceed max_points
#     # if features.size(0) > max_points:
#     #     indices = torch.randperm(features.size(0))[:max_points]
#     #     features = features[indices]
#     #     predictions = predictions[indices]
#     # # Randomly sample points for which we'll compute the loss
#     # indices = torch.randperm(features.size(0))[:sample_size]
#     # sample_features = features[indices]
#     # sample_preds = predictions[indices]
#     N = features.size(0)
    
#     # 원점에서 부터 norm이 1인 vector sample_size개를 뽑는다
#     num_directions = 200
#     directions = torch.randn((num_directions, 3)).to("cuda")
#     directions = directions / directions.norm(dim=1, keepdim=True)# normalize
    
#     # FOV 설정 (cosine으로 변환)
#     fov_deg = 5
#     fov_cos = torch.cos(torch.deg2rad(torch.tensor(fov_deg)))
   
#     # 이 direction기준으로 fov 를 가진 frustrum안에 속한 features를 뽑는다 
#     mapping = torch.zeros((num_directions, N), dtype=torch.int32)

#     # 각 방향에 대해 포인트가 원뿔 내부에 있는지 계산
#     cos_angles = torch.einsum("ij,kj->ik", features, directions) / (features.norm(dim=1, keepdim=True) + 1e-8)

#     mapping = (cos_angles >= fov_cos).int()  # (N, 200) 매핑 행렬
    
#     mapping = mapping.T
    
    
#     # norm이 큰 k개의 인덱스 추출
#     reshaped_features = features.unsqueeze(0).repeat(num_directions, 1, 1) # (200, N, 3)
    
#     reshaped_predictions = predictions.unsqueeze(0).repeat(num_directions, 1, 1) #[200, N, 3]
    
#     # norm이 큰 k개의 인덱스 추출
#     norms = reshaped_features.norm(dim=-1) # (200, N)
    
    
#     # 원뿔 내 포인트의 norm을 활용하여 상위 k개 선택
#     masked_norms = norms * mapping  # 원뿔 내부에 있는 포인트의 norm만 유지하고 나머지는 0
#     topk_values, topk_indices = torch.topk(masked_norms, k, dim=-1)  # 각 방향마다 상위 k개의 norm 및 인덱스 추출 #[200, k]
#     # top k 인덱스에 해당하는 위치에서 `reshape`과 `gather` 사용
#     top1_indices = topk_indices.max(-1).values.view(num_directions, 1, 1).repeat(1, 1, 3)  # (200, k, 3)
#     selected_predictions = torch.gather(reshaped_predictions, 1, top1_indices).squeeze()  # (200, 3)
    
#     selected_predictions = selected_predictions.unsqueeze(1).repeat(1, N, 1)
    
#     # 상위 k개의 인덱스에 해당하는 위치를 0으로 설정
#     row_indices = torch.arange(num_directions).view(-1, 1).repeat(1, k).to("cuda")  # (200, k)
#     mapping[row_indices, topk_indices] = 0  # 각 방향마다 상위 k 인덱스를 0으로 설정
    
#     # L1 loss 계산
#     reshaped_predictions_filtered = reshaped_predictions[mapping > 0]
#     selected_predictions_filtered = selected_predictions[mapping > 0]

#     normalized_loss = torch.abs(reshaped_predictions_filtered - selected_predictions_filtered).sum(dim=-1).mean()


#     return lambda_val * normalized_loss


def find_k_nearest_neighbors(orig, directions, tree, d, r):


    # Convert tensors to numpy arrays for KDTree
    orig_np = orig.cpu().numpy() if orig.is_cuda else orig.numpy()
    directions_np = directions.cpu().numpy() if directions.is_cuda else directions.numpy()

    # Create the ray points by extending the origin in the given direction
    ray_points = orig_np + directions_np * np.linspace(0, 0.3, num=d).reshape(-1, 1)
   
   
    # distances, indices = tree.query(ray_points, k=k)
    
    radius = 0.2  # 원하는 거리
    indices = tree.query_ball_point(ray_points, r=radius)

    # Flatten the indices to obtain unique k nearest neighbors
    unique_indices = np.unique(indices.flatten())
    

    nearest_neighbors = []
    # Convert back to tensor if needed
    for i in range(len(unique_indices)):
        nearest_neighbors.append(torch.tensor(unique_indices[i][:r], dtype=torch.long).to("cuda"))
    # nearest_neighbors = torch.tensor(unique_indices[0][:k], dtype=torch.long).to("cuda")
    nearest_neighbors = torch.cat(nearest_neighbors)
    return nearest_neighbors

def visualize_ray_and_neighbors(features, orig, directions, tree, d, r):
    import matplotlib.pyplot as plt
    # Convert tensors to numpy arrays
    features_np = features.cpu().numpy() if features.is_cuda else features.numpy()
    orig_np = orig.cpu().numpy() if orig.is_cuda else orig.numpy()
    directions_np = directions.cpu().numpy() if directions.is_cuda else directions.numpy()

    # Create the ray points
    ray_points = orig_np + directions_np * np.linspace(0, 0.3, num=d).reshape(-1, 1)

    # Find k nearest neighbors
    nearest_neighbors = find_k_nearest_neighbors(orig, directions, tree, d, r)
    nearest_points = features_np[nearest_neighbors.cpu().numpy()]

    # Plot the feature points, ray, and nearest neighbors
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the feature points in blue
    ax.scatter(features_np[:, 0], features_np[:, 1], features_np[:, 2], c='blue', s=0.005, label='Feature Points')

    # Plot the ray points in green
    ax.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], c='green', label='Ray')

    # Plot the nearest neighbors in red
    ax.scatter(nearest_points[:, 0], nearest_points[:, 1], nearest_points[:, 2], c='red', s=20, label='Nearest Neighbors')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

# def loss_cls_3d(features, predictions, selected_shell, selected_normals):
#     """
#     Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
#     and the KL divergence.
    
#     :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
#     :param predictions: Tensor of shape (N, C), where C is the number of classes.
#     :param k: Number of neighbors to consider.
#     :param lambda_val: Weighting factor for the loss.
#     :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
#     :param sample_size: Number of points to randomly sample for computing the loss.
    
#     :return: Computed loss value.
#     # """
    
#     from scipy.spatial import KDTree
#     sample_point = 1000
#     d = 5 
#     r = 8
#     indices = torch.randperm(selected_shell.size(0))[:sample_point]
#     sample_shell = selected_shell[indices]
#     sample_normal = selected_normals[indices]
    
#     features_np = features.cpu().numpy() if features.is_cuda else features.numpy()
#     tree = KDTree(features_np)
    
    
#     # find k nearest neighbors
#     # Compute total loss without using a loop
#     directions = -sample_normal
#     # directions = sample_normal
#     directions = directions / torch.norm(directions, dim=1, keepdim=True)
    
#     target_size = d * r
#     nearest_neighbors = []
    
#     for shell, direction in zip(sample_shell, directions): 
#         neighbors = find_k_nearest_neighbors(features[shell], direction, tree, d, r)
#         padding_size = target_size - neighbors.size(0)
#         if padding_size > 0:
#             neighbors = F.pad(neighbors, (0, padding_size), value=0)
#         nearest_neighbors.append(neighbors)
        
#     #! For visualize
#     # visualize_ray_and_neighbors(features, features[sample_shell], directions, tree, d, r)

#     nearest_neighbors = torch.stack(nearest_neighbors)
#     pivot_pred = predictions[sample_shell].view(-1, 1, 3).repeat(1, d * r, 1)
#     nearest_pred = predictions[nearest_neighbors]
    
#     total_loss = torch.sum(F.l1_loss(pivot_pred, nearest_pred, reduction='none'))
#     # # original_grad = predictions.grad
#     # predictions = predictions.detach().clone()
#     # predictions[nearest_neighbors] = pivot_pred.detach()
#     # new_object_id = predictions.detach().clone().requires_grad_(False)
#     # if original_grad is not None:
#     #     new_object_id.grad = original_grad.clone()
    
    
#     #! Visualize the selected point cloud
#     # import matplotlib.pyplot as plt
#     # # Plot the feature points, ray, and nearest neighbors
#     # fig = plt.figure(figsize=(10, 7))
#     # ax = fig.add_subplot(111, projection='3d')
    
#     # # np_object_id = new_object_id.cpu().detach().numpy()
    
#     # np_features_dc = features[nearest_neighbors].detach().cpu().numpy()
#     # np_object_id = predictions[nearest_neighbors].cpu().detach().numpy()


#     # # Plot the feature points with colors based on np_object_id
#     # np_object_id = (np_object_id - np_object_id.min()) / (np_object_id.max() - np_object_id.min())
#     # np_features_dc = np_features_dc.reshape(-1, 3)
#     # np_object_id = np_object_id.reshape(-1, 3)
    
#     # import open3d as o3d
#     # # Point Cloud 객체 생성
#     # point_cloud = o3d.geometry.PointCloud()
#     # point_cloud.points = o3d.utility.Vector3dVector(np_features_dc)
#     # point_cloud.colors = o3d.utility.Vector3dVector(np_object_id)

#     # # Point Cloud를 PLY 파일로 저장
#     # o3d.io.write_point_cloud("selected_point_cloud.ply", point_cloud)
#     # # ax.scatter(np_features_dc[:, 0], np_features_dc[:, 1], np_features_dc[:, 2], c=np_object_id, s=0.005, label='Feature Points')
    
    

#     # # # Set labels
#     # # ax.set_xlabel('X')
#     # # ax.set_ylabel('Y')
#     # # ax.set_zlabel('Z')
#     # # ax.legend()

#     # plt.show()
    

#     return total_loss




#! ununcertainty based anchoring loss 


def loss_3d_certainty(position, uncertainty, seg_label, k=5, lambda_val=2.0, max_points=200000, sample_size=800, iteration=0):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if position.size(0) > max_points:
        indices = torch.randperm(uncertainty.size(0))[:max_points]
        
        # indices = torch.topk(uncertainty, max_points, largest=True).indices
        position = position[indices]
        seg_label = seg_label[indices]
        uncertainty = uncertainty[indices]
    
    half_sample_size = sample_size // 2
    
    # _, indices = torch.sort(uncertainty)
    # alpha = 10
    # total_iterations= 400
    # center = total_iterations/3 # 중심점 
    # decay_factor = 0.5 * (1 - torch.sigmoid(alpha * ((iteration - center) / torch.tensor(total_iterations, dtype=torch.float32))))


    # max_ratio = sample_size/(2*max_points)
    
    # index_range_size = int(len(indices) * (max(decay_factor, max_ratio)))
    
    # selected_index = torch.randperm(index_range_size)[:half_sample_size]
    
    # half_indices = indices[selected_index]
    # sample_position = position[half_indices]
    # sample_label = seg_label[half_indices]
    # # supression 
    # # indices = torch.topk(uncertainty, half_sample_size, largest=False).indices
    
    # half_indices = indices[-selected_index]
    # sup_sample_position = position[half_indices]
    # sup_sample_label = seg_label[half_indices]
    
    
    
    
    
  
    # # uncertainty size 
    # total_iterations= 400
    # certainty_sample_size = int((1 + iteration/total_iterations) * sample_size / 2)
    
    # selected_index = torch.topk(uncertainty, certainty_sample_size, largest=False).indices
    
    # sup_sample_position = position[selected_index]
    # sup_sample_label = seg_label[selected_index]
    
    
    # uncertainty_sample_size = sample_size - certainty_sample_size
    # selected_index = torch.topk(uncertainty, uncertainty_sample_size, largest=True).indices
    
    # sample_position = position[selected_index]
    # sample_label = seg_label[selected_index]
    
    
    
     # uncertainty size 
    
    selected_index = torch.topk(uncertainty, half_sample_size, largest=False).indices
    
    sup_sample_position = position[selected_index]
    sup_sample_label = seg_label[selected_index]
    
    selected_index = torch.topk(uncertainty, half_sample_size, largest=True).indices
    
    sample_position = position[selected_index]
    sample_label = seg_label[selected_index]
    
    
    
    #! 여기부터 바꾸면됨
    sample_position = torch.cat((sample_position, sup_sample_position), dim=0)
    sample_label = torch.cat((sample_label, sup_sample_label), dim=0)
    
    
    # #! ------------- 여기에서부터 주변 neighbor값들을 다시 뽑기 
    # dists = torch.cdist(sample_position, position)  # Compute pairwise distances
    # _, neighbor_indices_tensor = dists.topk(10, largest=False)  # Get top-k nearest neighbors
    
    # # Flatten and sample again
    # neighbor_indices_flat = neighbor_indices_tensor.view(-1)
    # sampled_indices = torch.randperm(neighbor_indices_flat.size(0))[:sample_size]
    
    # sample_position = position[neighbor_indices_flat[sampled_indices]]
    # sample_label = seg_label[neighbor_indices_flat[sampled_indices]]
    # #! end -------------
    
    
    
    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_position, position)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_seg_label = seg_label[neighbor_indices_tensor]
            
    #! L1 loss 
    # 걍 reference랑 L1 distance를 재자 ㅋㅋㅋㅋ
    sample_label = sample_label.unsqueeze(1).repeat(1, k, 1) # [sample_size, k, 3]
    loss = torch.abs(sample_label - neighbor_seg_label).sum(dim=-1).mean()
    normalized_loss = loss

    return lambda_val * normalized_loss



# # #! ununcertainty based anchoring loss 


# def loss_3d_certainty(position, uncertainty, seg_label, k=5, lambda_val=2.0, max_points=200000, sample_size=800, iteration=0):
#     """
#     Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
#     and the KL divergence.
    
#     :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
#     :param predictions: Tensor of shape (N, C), where C is the number of classes.
#     :param k: Number of neighbors to consider.
#     :param lambda_val: Weighting factor for the loss.
#     :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
#     :param sample_size: Number of points to randomly sample for computing the loss.
    
#     :return: Computed loss value.
#     """
#     # Conditionally downsample if points exceed max_points
#     if position.size(0) > max_points:
#         indices = torch.randperm(uncertainty.size(0))[:max_points]
        
#         # indices = torch.topk(uncertainty, max_points, largest=True).indices
#         position = position[indices]
#         seg_label = seg_label[indices]
#         uncertainty = uncertainty[indices]

#     #! 원래는 이렇습니다 
#     indices = torch.randperm(position.size(0))[:sample_size]
#     sample_position = position[indices]
#     sample_label = seg_label[indices]
    
#     # Compute top-k nearest neighbors directly in PyTorch
#     dists = torch.cdist(sample_position, position)  # Compute pairwise distances
#     _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

#     # Fetch neighbor predictions using indexing
#     neighbor_seg_label = seg_label[neighbor_indices_tensor]
            
#     #! L1 loss 
#     # 걍 reference랑 L1 distance를 재자 ㅋㅋㅋㅋ
#     sample_label = sample_label.unsqueeze(1).repeat(1, k, 1) # [sample_size, k, 3]
#     loss = torch.abs(sample_label - neighbor_seg_label).sum(dim=-1).mean()
#     normalized_loss = loss

#     return lambda_val * normalized_loss



