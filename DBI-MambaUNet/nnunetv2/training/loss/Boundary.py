from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def soft_distance_transform(boundary: torch.Tensor, gamma: float = 10.0, max_distance: float = 100.0,
                            chunk_size: int = 10000) -> torch.Tensor:
    """
    可微分的 soft distance transform，利用 log-sum-exp 近似 min 操作，
    並通過分塊計算降低內存消耗。

    支持 2D ([B,1,H,W]) 與 3D ([B,1,D,H,W]) 邊界圖的計算。

    Args:
        boundary: tensor, 邊界圖 (通常值為 0 或 1)，形狀 [B,1,H,W] 或 [B,1,D,H,W]
        gamma: 控制 softmin 的鋒利度，gamma 越大，近似效果越接近硬 min
        max_distance: 結果中距離的最大值，超過此值 clip 掉
        chunk_size: 分塊計算時一次處理的像素數，降低此值可減少內存占用 (預設 10000)

    Returns:
        soft_distance: tensor, 形狀與 boundary 相同，代表每個像素的軟距離
    """
    if boundary.dim() == 4:
        # 2D case
        B, _, H, W = boundary.shape
        device = boundary.device
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2).float()  # shape [H*W, 2]
        soft_dists = []
        for b in range(B):
            b_boundary = boundary[b, 0].view(-1)  # [H*W]
            if b_boundary.sum() == 0:
                soft_dists.append(torch.zeros(H * W, device=device))
                continue
            idx = (b_boundary > 0.5).nonzero(as_tuple=False).view(-1)
            boundary_coords = grid[idx]  # [num_boundary, 2]
            num_points = grid.shape[0]
            soft_min_chunks = []
            for i in range(0, num_points, chunk_size):
                chunk = grid[i: i + chunk_size]  # [chunk_size, 2]
                dists = torch.cdist(chunk, boundary_coords, p=2)  # [chunk_size, num_boundary]
                soft_min_chunk = -1.0 / gamma * torch.log(torch.sum(torch.exp(-gamma * dists), dim=1) + 1e-8)
                soft_min_chunks.append(soft_min_chunk)
            soft_min_full = torch.cat(soft_min_chunks, dim=0)
            soft_dists.append(soft_min_full)
        soft_dists = torch.stack(soft_dists, dim=0).view(B, 1, H, W)
        soft_dists = torch.clamp(soft_dists, 0, max_distance)
        return soft_dists

    elif boundary.dim() == 5:
        # 3D case
        B, _, D, H, W = boundary.shape
        device = boundary.device
        grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(D, device=device),
                                                torch.arange(H, device=device),
                                                torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([grid_z, grid_y, grid_x], dim=-1).view(-1, 3).float()  # shape [D*H*W, 3]
        soft_dists = []
        for b in range(B):
            b_boundary = boundary[b, 0].view(-1)  # [D*H*W]
            if b_boundary.sum() == 0:
                soft_dists.append(torch.zeros(D * H * W, device=device))
                continue
            idx = (b_boundary > 0.5).nonzero(as_tuple=False).view(-1)
            boundary_coords = grid[idx]  # [num_boundary, 3]
            num_points = grid.shape[0]
            soft_min_chunks = []
            for i in range(0, num_points, chunk_size):
                chunk = grid[i: i + chunk_size]  # [chunk_size, 3]
                dists = torch.cdist(chunk, boundary_coords, p=2)  # [chunk_size, num_boundary]
                soft_min_chunk = -1.0 / gamma * torch.log(torch.sum(torch.exp(-gamma * dists), dim=1) + 1e-8)
                soft_min_chunks.append(soft_min_chunk)
            soft_min_full = torch.cat(soft_min_chunks, dim=0)
            soft_dists.append(soft_min_full)
        soft_dists = torch.stack(soft_dists, dim=0).view(B, 1, D, H, W)
        soft_dists = torch.clamp(soft_dists, 0, max_distance)
        return soft_dists
    else:
        raise ValueError("Boundary tensor must be 4D or 5D.")


class BoundaryLoss(nn.Module):
    """
    Differentiable Boundary Loss：
    使用可微分的 soft distance transform 作為邊界權重，
    loss = lambda_boundary * (1/N) * Σ |pred - target| * (soft_distance + smooth)

    參數:
      - apply_nonlin: (callable) 若模型輸出 logits，建議傳入 torch.sigmoid 轉換為概率。
      - lambda_boundary: (float) 邊界 loss 的權重，預設 1.0。
      - smooth: (float) 平滑項，預設 1e-6。
      - size_average: (bool) 是否對批次取平均，預設 True。
      - gamma: (float) 控制 soft distance transform 的鋒利度，預設 10.0。
      - max_distance: (float) soft distance transform 的最大值 (clipping)，預設 100.0。
      - chunk_size: (int) 分塊大小，用於限制內存占用，預設 10000。
    """

    def __init__(self, apply_nonlin: Callable = None, lambda_boundary: float = 1.0, smooth: float = 1e-6,
                 size_average: bool = True, gamma: float = 10.0, max_distance: float = 100.0, chunk_size: int = 10000):
        super(BoundaryLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.lambda_boundary = lambda_boundary
        self.smooth = smooth
        self.size_average = size_average
        self.gamma = gamma
        self.max_distance = max_distance
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 預測結果 tensor，形狀為 [B, 1, H, W] 或 [B, 1, D, H, W]，可以是 logits 或概率。
            y: ground truth 二值 mask，形狀與 x 相同，值為 0 (背景) 或 1 (Tumor)。
            loss_mask: (optional) 用於遮罩計算的 mask，形狀與 x 相同。
        Returns:
            經加權後的 Differentiable Boundary Loss。
        """
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # 根據 y 的維度選擇 2D 或 3D 邊界提取
        if y.dim() == 4:
            kernel = torch.tensor([[1, 1, 1],
                                   [1, -8, 1],
                                   [1, 1, 1]], dtype=torch.float32, device=y.device).unsqueeze(0).unsqueeze(0)
            y_boundary = torch.abs(F.conv2d(y, kernel, padding=1))
        elif y.dim() == 5:
            kernel_3d = torch.tensor(
                [[[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]],
                 [[1, 1, 1],
                  [1, -26, 1],
                  [1, 1, 1]],
                 [[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]]],
                dtype=torch.float32, device=y.device).unsqueeze(0).unsqueeze(0)
            y_boundary = torch.abs(F.conv3d(y, kernel_3d, padding=1))
        else:
            raise ValueError("Ground truth y must be 4D or 5D.")

        # 將邊界圖乘以一個放大因子後用 sigmoid 做 soft threshold，提取軟邊界
        y_boundary = torch.sigmoid(y_boundary * 10)

        # 計算可微分的 soft distance transform（採用分塊處理）
        soft_dt = soft_distance_transform(y_boundary, gamma=self.gamma, max_distance=self.max_distance,
                                          chunk_size=self.chunk_size)

        B = x.shape[0]
        losses = []
        for b in range(B):
            pred = x[b, 0]
            target_sample = y[b, 0]
            error = torch.abs(pred - target_sample)
            weighted_error = error * (soft_dt[b, 0] + self.smooth)
            if loss_mask is not None:
                current_mask = loss_mask[b, 0].float()
                sample_loss = torch.sum(weighted_error * current_mask) / (torch.sum(current_mask) + self.smooth)
            else:
                sample_loss = torch.mean(weighted_error)
            losses.append(sample_loss)
        if self.size_average:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.stack(losses).sum()
        return self.lambda_boundary * loss
