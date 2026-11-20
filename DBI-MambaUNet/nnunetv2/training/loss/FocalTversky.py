import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss 參考：
      - Salehi et al., "Tversky loss function for image segmentation using 3D FCNs" (Tversky)
      - Abraham & Khan, "A novel Focal Tversky loss function..." (Focal Tversky)

    定義：
      TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
      FTL = (1 - TI)^gamma

    參數
    -------
    alpha, beta : float
        Tversky 的 FP/FN 權重（常見：alpha=0.7, beta=0.3；若更重視召回可改 alpha=0.3, beta=0.7）
    gamma : float
        Focal 指數（論文常用 4/3 ≈ 1.333）
    smooth / eps : float
        平滑避免除 0，任一個填就好（兩者等價，保留 smooth 兼容）
    from_logits : bool
        True 則自動做 sigmoid/softmax；False 表示輸入已是機率
    include_background : bool
        多類時是否把背景通道 (class 0) 也納入；一般建議 False
    ignore_index : int or None
        在 target（整數標籤）中要忽略的索引值
    reduction : str
        'mean' | 'sum' | 'none'
    per_image : bool
        True → 先對每張圖平均，再跨 batch 做 reduction（較常見於 segmentation）
    class_weights : Tensor[used_C] or None
        類別權重（不含被排除的背景），shape = (使用到的通道數,)
    """

    def __init__(self,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 gamma: float = 4/3,
                 smooth: float = 1e-6,
                 eps: float | None = None,
                 from_logits: bool = True,
                 include_background: bool = False,
                 ignore_index: int | None = None,
                 reduction: str = "mean",
                 per_image: bool = True,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        assert 0 <= alpha <= 1 and 0 <= beta <= 1, "alpha/beta must be in [0,1]"
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(smooth if eps is None else eps)
        self.from_logits = bool(from_logits)
        self.include_background = bool(include_background)
        self.ignore_index = ignore_index
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.per_image = bool(per_image)
        self.register_buffer("class_weights",
                             class_weights if class_weights is not None else None,
                             persistent=False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input : [B, C, *spatial]
            - C=1 → binary；C>1 → multi-class
            - 若 from_logits=True，input 是 logits；否則為機率
        target:
            - 若為整數標籤 → [B, 1, *] 或 [B, *]
            - 若為 one-hot   → [B, C, *]
        """
        B, C = input.shape[:2]
        spatial_dims = tuple(range(2, input.ndim))

        # ------ 轉機率 ------
        if self.from_logits:
            if C == 1:
                prob = torch.sigmoid(input)
            else:
                prob = F.softmax(input, dim=1)
        else:
            prob = input

        # ------ 建立 one-hot target（同時支援整數或 one-hot 輸入） ------
        if target.ndim == input.ndim and target.shape[1] == C:
            # 已是 one-hot
            target_1h = target.float()
            target_idx = None
        else:
            # 整數標籤
            tgt = target.long()
            if tgt.ndim == input.ndim:  # [B,1,*] → squeeze 掉通道
                tgt = tgt.squeeze(1)
            # one_hot: [B, *spatial, C] → [B, C, *spatial]
            target_1h = F.one_hot(tgt.clamp_min(0), num_classes=max(C, int(tgt.max().item()+1)))
            # 若 max label < C，補齊到 C
            if target_1h.shape[-1] != C:
                pad_channels = C - target_1h.shape[-1]
                if pad_channels < 0:
                    raise ValueError(f"Target labels exceed input channels: got max label {target_1h.shape[-1]-1}, C={C}")
                pad = [0, 0] * (target_1h.ndim - 2) + [0, pad_channels]
                target_1h = F.pad(target_1h, pad)
            target_1h = target_1h.movedim(-1, 1).to(prob.dtype)
            target_idx = tgt  # for ignore_index

        # ------ 忽略背景（多類） ------
        if C > 1 and not self.include_background:
            prob = prob[:, 1:, ...]
            target_1h = target_1h[:, 1:, ...]
            used_C = C - 1
        else:
            used_C = C

        # ------ ignore_index → valid mask ------
        if self.ignore_index is not None:
            if target_idx is None:
                # 若給的是 one-hot target，則無法可靠推回整數標籤；改以「所有通道和==1」作近似 valid
                valid = (target_1h.sum(dim=1, keepdim=True) > 0).to(prob.dtype)
            else:
                valid = (target_idx != self.ignore_index).to(prob.dtype).unsqueeze(1)
        else:
            valid = torch.ones((B, 1, *input.shape[2:]), dtype=prob.dtype, device=prob.device)

        # ------ per-image 或全局的歸約維度 ------
        # 統計維度（空間）：2..N
        red_dims = spatial_dims
        # 如果要 per-image，就保留 batch 維，否則可合併 batch
        # 我們統一先保留 batch，最後再做 reduction
        # TP/FP/FN 形狀： [B, used_C]
        TP = (prob * target_1h * valid).sum(dim=red_dims)
        FP = (prob * (1.0 - target_1h) * valid).sum(dim=red_dims)
        FN = ((1.0 - prob) * target_1h * valid).sum(dim=red_dims)

        # ------ Tversky Index & Focal ------
        TI = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        loss_per_bc = (1.0 - TI).pow(self.gamma)  # [B, used_C]

        # ------ 類別權重（可選） ------
        if self.class_weights is not None:
            cw = self.class_weights.to(loss_per_bc.dtype).to(loss_per_bc.device)
            if cw.numel() != used_C:
                raise ValueError(f"class_weights length ({cw.numel()}) must equal used classes ({used_C})")
            loss_per_bc = loss_per_bc * cw.unsqueeze(0)  # [B, used_C]

        # ------ per-image 聚合 → [B] ------
        if self.per_image:
            loss_per_b = loss_per_bc.mean(dim=1)  # 每張圖對類別取平均
        else:
            # 不做 per-image，保留 [B, used_C]，交由 reduction
            loss_per_b = loss_per_bc.reshape(B, -1).mean(dim=1)

        # ------ reduction ------
        if self.reduction == "mean":
            return loss_per_b.mean()
        elif self.reduction == "sum":
            return loss_per_b.sum()
        else:
            # 'none' → 回傳 [B]
            return loss_per_b
