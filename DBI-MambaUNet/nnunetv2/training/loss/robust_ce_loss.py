import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Ensure the input and target shapes are the same after one-hot encoding
        if target.shape[1] == 1:
            target = F.one_hot(target.squeeze(1).long(), num_classes=input.shape[1])
            target = target.permute(0, 4, 1, 2, 3).contiguous()

        assert input.shape == target.shape, f"Shape mismatch: input shape {input.shape}, target shape {target.shape}"

        # Convert logits to probabilities
        input = torch.sigmoid(input)

        # Flatten the tensors
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (input_flat * target_flat).sum()
        FP = ((1 - target_flat) * input_flat).sum()
        FN = (target_flat * (1 - input_flat)).sum()

        # Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Tversky loss
        return 1 - tversky_index



import torch
import torch.nn as nn
import torch.nn.functional as F

class SizeAwareTverskyLoss(nn.Module):
    """
    尺寸自適應 Tversky / Asymmetric Focal Tversky（3D）
    - 對每個病灶（connected component）按體素數 s 給權重 w = (s + eps)^(-size_gamma)，並在正類範圍內做歸一化
    - focal_gamma = 0 -> Tversky；>0 -> Asymmetric Focal Tversky
    參數：
      alpha, beta     : Tversky 的 FP/FN 權重（不對稱 ⇒ 非常適合控制小病灶 FN）
      focal_gamma     : Focal 指數（常用 0~2；0 即關閉 focal）
      size_gamma      : 病灶尺寸權重指數（0.5~1.0 推薦；越大越偏好小病灶）
      bg_weight       : 背景權重縮放，<1 可降低 FP 懲罰、>1 可提高
      conn            : 3D 連通性（1=6-connectivity；2=18；3=26，僅在有 skimage/scipy 時生效）
      normalize_pos   : 是否將正類權重歸一，使正類平均權重≈1（穩定訓練）
    輸入：
      input : logits, shape [B,C,D,H,W]（C=1: binary；C>1: multi-class）
      target: [B,1,D,H,W] (整數標籤) 或 [B,C,D,H,W] (one-hot)
    """
    def __init__(self,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 focal_gamma: float = 0.0,
                 size_gamma: float = 0.7,
                 bg_weight: float = 1.0,
                 conn: int = 1,
                 normalize_pos: bool = True,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_gamma = focal_gamma
        self.size_gamma = size_gamma
        self.bg_weight = bg_weight
        self.conn = conn
        self.normalize_pos = normalize_pos
        self.smooth = smooth

        # optional deps
        self._has_skimage = False
        self._has_scipy = False
        try:
            from skimage.measure import label as _label  # noqa: F401
            self._has_skimage = True
        except Exception:
            pass
        try:
            from scipy.ndimage import label as _slabel  # noqa: F401
            self._has_scipy = True
        except Exception:
            pass

    def _to_one_hot(self, input, target):
        # 將 target 轉 one-hot；保持 input/target 同形狀
        if target.shape[1] == 1:
            num_classes = input.shape[1] if input.shape[1] > 1 else 2  # binary 預設背景/前景兩類
            tgt = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
            tgt = tgt.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]
        else:
            tgt = target
        return tgt

    @torch.no_grad()
    def _build_size_weight_map_cc(self, tgt_fg: torch.Tensor) -> torch.Tensor:
        """
        基於連通元件（CC）建立尺寸權重圖。
        tgt_fg: [B,1,D,H,W] 的 0/1 張量（前景 mask）
        回傳 W: [B,1,D,H,W]
        """
        B, _, D, H, W = tgt_fg.shape
        device = tgt_fg.device
        dtype = tgt_fg.dtype
        W_map = torch.ones_like(tgt_fg, dtype=dtype, device=device)  # 先給背景=1，稍後再混和 bg_weight

        # 依序處理 batch，每個 case 單獨 CPU 標記（可靠、簡單）
        for b in range(B):
            m = tgt_fg[b, 0].detach().to('cpu', torch.uint8).numpy()
            if m.sum() == 0:
                # 無前景，維持全 1（背景權重稍後處理）
                continue

            label_arr = None
            if self._has_skimage:
                from skimage.measure import label as sk_label
                # skimage: connectivity=1 對應 6-connectivity（3D）
                label_arr = sk_label(m, connectivity=self.conn)
            elif self._has_scipy:
                from scipy.ndimage import label as sp_label
                label_arr, _ = sp_label(m)  # scipy 只支援 6-connectivity 預設
            else:
                # 無外部庫：退化策略——以局部體素密度近似大小（3D 平均池化 * kernel）
                # 這不是嚴格的 CC，但能提供合理的大小趨勢
                k = 7  # 感受野 ~ 病灶直徑粗估
                pad = k // 2
                density = F.avg_pool3d(torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0),
                                       kernel_size=k, stride=1, padding=pad).squeeze().numpy()
                # 根據局部密度反比做權重
                eps = 1e-6
                w = (density + eps) ** (-self.size_gamma)
                w[m == 0] = 0.0
                W_case = torch.from_numpy(w).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
                # 正類內歸一（使平均≈1）
                if self.normalize_pos and W_case.sum() > 0:
                    pos_cnt = float(m.sum())
                    scale = pos_cnt / (W_case.sum().item() + 1e-6)
                    W_case = W_case * scale
                W_map[b:b+1] = W_case
                continue

            # 有連通標記 label_arr：用每個 component 的體素數做權重
            import numpy as np
            labels, counts = np.unique(label_arr, return_counts=True)
            # label 0 是背景，忽略
            if labels[0] == 0:
                labels = labels[1:]
                counts = counts[1:]
            if len(labels) == 0:
                continue

            eps = 1e-6
            comp_weights = (counts.astype(np.float64) + eps) ** (-self.size_gamma)  # 小病灶權重大
            # 正類平均歸一：使 sum(w) ≈ #pos_voxels
            if self.normalize_pos:
                scale = label_arr.astype(bool).sum() / (comp_weights @ counts + eps)
                comp_weights = comp_weights * scale

            # 建立查表，背景=0
            lookup = np.zeros(label_arr.max() + 1, dtype=np.float32)
            lookup[labels] = comp_weights
            w = lookup[label_arr]  # shape (D,H,W)，前景位置有權重，背景=0

            W_case = torch.from_numpy(w).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            W_map[b:b+1] = W_case

        # 把背景設為 bg_weight，正類為 W_map（>0）
        # 這一步在外面用混合的方式處理，這裡只回傳正類權重（背景=0）
        return W_map  # 正類權重；背景位置=0

    def forward(self, input: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        B, C = input.shape[:2]
        target_1h = self._to_one_hot(input, target)  # [B,C,D,H,W]

        # probabilities
        if C == 1:
            prob = torch.sigmoid(input)
            tgt_fg = target_1h if target_1h.shape[1] == 1 else target_1h[:, 1:2]
        else:
            prob = F.softmax(input, dim=1)
            tgt_fg = target_1h[:, 1:, ...]

        # build size-aware positive weight
        if C == 1:
            W_pos = self._build_size_weight_map_cc((tgt_fg > 0.5).float())
        else:
            Ws = [self._build_size_weight_map_cc((tgt_fg[:, i:i + 1] > 0.5).float()) for i in range(tgt_fg.shape[1])]
            W_pos = torch.max(torch.stack(Ws, dim=1), dim=1).values  # [B,1,D,H,W]

        # background weight
        if C == 1:
            tgt_bg = 1.0 - tgt_fg
        else:
            tgt_bg = 1.0 - tgt_fg.sum(dim=1, keepdim=True).clamp(0, 1)

        W = W_pos + self.bg_weight * tgt_bg  # [B,1,D,H,W]

        # <- 新增：把忽略區域（例如 ignore_label）從計分中排除
        if valid_mask is not None:
            W = W * valid_mask.float()

        losses = []
        if C == 1:
            p = prob
            t = target_1h if target_1h.shape[1] == 1 else target_1h[:, 1:2]
            TP = (W * p * t).sum()
            FP = (W * (1 - t) * p).sum()
            FN = (W * t * (1 - p)).sum()
            TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            loss = (1.0 - TI).pow(self.focal_gamma) if self.focal_gamma > 0 else (1.0 - TI)
            losses.append(loss)
        else:
            for c in range(1, C):
                p = prob[:, c:c + 1, ...]
                t = target_1h[:, c:c + 1, ...]
                TP = (W * p * t).sum()
                FP = (W * (1 - t) * p).sum()
                FN = (W * t * (1 - p)).sum()
                TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
                loss = (1.0 - TI).pow(self.focal_gamma) if self.focal_gamma > 0 else (1.0 - TI)
                losses.append(loss)

        return torch.stack(losses).mean()


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SizeAwareHaloTverskyLoss(nn.Module):
    """
    改進版：Size-Aware Halo Focal Tversky Loss
    創新點：
    1. Halo Mechanism: 將小病灶的高權重擴散到周圍背景，強力抑制邊緣 False Positive。
    2. Log-Cosh: 使用 Log-Cosh 平滑 Loss Surface，提升收斂穩定性。
    """

    def __init__(self,
                 alpha: float = 0.5,  # 建議調回 0.5 或 0.6，因為我們現在有 Halo 機制來懲罰 FP
                 beta: float = 0.5,
                 focal_gamma: float = 4 / 3,  # 推薦值，比 2.0 溫和
                 size_gamma: float = 0.8,
                 bg_weight: float = 1.0,
                 halo_radius: int = 1,  # 新增：光環半徑 (voxels)，對小病灶周圍多少範圍進行高壓懲罰
                 conn: int = 1,
                 normalize_pos: bool = True,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_gamma = focal_gamma
        self.size_gamma = size_gamma
        self.bg_weight = bg_weight
        self.halo_radius = halo_radius
        self.conn = conn
        self.normalize_pos = normalize_pos
        self.smooth = smooth

        self._has_scipy = False
        try:
            from scipy.ndimage import label as _slabel, binary_dilation
            self.binary_dilation = binary_dilation  # 用於生成 Halo
            self._has_scipy = True
        except Exception:
            print("Warning: scipy not found. Halo mechanism will be disabled.")
            pass

    def _to_one_hot(self, input, target):
        if target.shape[1] == 1:
            num_classes = input.shape[1] if input.shape[1] > 1 else 2
            tgt = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
            tgt = tgt.permute(0, 4, 1, 2, 3).contiguous()
        else:
            tgt = target
        return tgt

    @torch.no_grad()
    def _build_size_weight_map_with_halo(self, tgt_fg: torch.Tensor) -> torch.Tensor:
        """
        生成帶有 Halo 懲罰的權重圖
        """
        B, _, D, H, W = tgt_fg.shape
        device = tgt_fg.device
        dtype = tgt_fg.dtype
        # 初始化權重圖，默認為背景權重
        W_map = torch.ones_like(tgt_fg, dtype=dtype, device=device) * self.bg_weight

        for b in range(B):
            m = tgt_fg[b, 0].detach().to('cpu', torch.uint8).numpy()
            if m.sum() == 0: continue

            if self._has_scipy:
                from scipy.ndimage import label as sp_label
                label_arr, _ = sp_label(m)  # Default connectivity

                labels, counts = np.unique(label_arr, return_counts=True)
                if labels[0] == 0:
                    labels = labels[1:]
                    counts = counts[1:]
                if len(labels) == 0: continue

                eps = 1e-6
                # 計算每個 component 的權重
                comp_weights = (counts.astype(np.float64) + eps) ** (-self.size_gamma)

                # 歸一化邏輯 (保持你的邏輯)
                if self.normalize_pos:
                    scale = label_arr.astype(bool).sum() / (comp_weights @ counts + eps)
                    comp_weights = comp_weights * scale

                # 建立查表
                lookup = np.zeros(label_arr.max() + 1, dtype=np.float32)
                lookup[labels] = comp_weights  # 這裡存的是純粹的前景權重

                # --- Halo Magic Happens Here ---
                # 我們不直接查表，而是針對每個 component 做處理 (為了把高權重擴散出去)
                # 為了效率，我們可以用一個比較 tricky 的方法：
                # 生成一個全圖權重矩陣
                w_case_numpy = lookup[label_arr]  # 這是前景的權重

                # 如果啟用了 Halo，我們需要把前景的高權重 "傳染" 給鄰近的背景
                if self.halo_radius > 0:
                    # 1. 找出所有高權重的區域 (比平均權重大的區域)
                    # 簡單起見，我們對整個前景做 dilation，將前景的權重擴張一圈
                    # 這樣邊緣的背景也會獲得跟核心病灶一樣高的權重 -> 強力抑制 FP

                    # 這裡使用簡單的 dilation 擴張「權重圖」。
                    # 實際上，我們希望 dilation 取 max (如果兩個病灶靠近，取大的權重)
                    # 但 scipy.ndimage.grey_dilation 可以做到
                    from scipy.ndimage import grey_dilation

                    # 使用 grey_dilation 將高權重值擴散到周圍
                    # structure 定義了擴散的形狀 (3x3x3 block)
                    struct = np.ones((2 * self.halo_radius + 1,) * 3, dtype=bool)
                    w_dilated = grey_dilation(w_case_numpy, structure=struct)

                    # 更新 w_case_numpy:
                    # 原本是前景的地方保持原值 (或者取 max，但在這裡是一樣的因為 w_case 已經有值)
                    # 原本是背景的地方，現在變成了鄰近病灶的權重
                    w_case_numpy = np.maximum(w_case_numpy, w_dilated)

                # 將背景部分的權重設回 bg_weight (如果沒有被 Halo 覆蓋到)
                # 注意：上面的 w_case_numpy 在純背景處是 0 (因為 label 0 查表是 0)
                # dilation 後，遠離病灶的背景還是 0

                # 轉回 Tensor
                W_case = torch.from_numpy(w_case_numpy).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

                # W_case 現在包含了：前景(高權重) + Halo背景(高權重) + 遠處背景(0)
                # 我們需要把它疊加到基底 W_map 上
                # 邏輯：取 max (保留最高的懲罰)
                W_map[b:b + 1] = torch.max(W_map[b:b + 1], W_case)

            else:
                # Fallback for no scipy (original density logic)
                pass

        return W_map

    def forward(self, input: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        B, C = input.shape[:2]
        target_1h = self._to_one_hot(input, target)

        if C == 1:
            prob = torch.sigmoid(input)
            tgt_fg = target_1h if target_1h.shape[1] == 1 else target_1h[:, 1:2]
        else:
            prob = F.softmax(input, dim=1)
            tgt_fg = target_1h[:, 1:, ...]

        # 構建帶有 Halo 的權重圖
        if C == 1:
            W = self._build_size_weight_map_with_halo((tgt_fg > 0.5).float())
        else:
            Ws = [self._build_size_weight_map_with_halo((tgt_fg[:, i:i + 1] > 0.5).float()) for i in
                  range(tgt_fg.shape[1])]
            W = torch.max(torch.stack(Ws, dim=1), dim=1).values

        if valid_mask is not None:
            W = W * valid_mask.float()

        losses = []
        # Tversky logic
        for c in range(0 if C == 1 else 1, C if C == 1 else C + 1):  # Handle binary/multi-class loop simpler
            p = prob if C == 1 else prob[:, c:c + 1]
            t = tgt_fg if C == 1 else tgt_fg[:, c - 1:c]  # adjust index

            # 重點：現在計算 FP 時，如果落在 Halo 區域，W 會很大！
            TP = (W * p * t).sum()
            FP = (W * (1 - t) * p).sum()
            FN = (W * t * (1 - p)).sum()

            Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

            # Log-Cosh Tversky Implementation
            # Log-Cosh(x) = log(cosh(x)) approx x^2/2 for small x, abs(x)-log2 for large x.
            # 這裡 x = (1 - Tversky)
            loss_val = 1.0 - Tversky

            if self.focal_gamma > 0:
                # Combine Focal with Log-Cosh: log(cosh((1-TI)^gamma))
                # 這比單純的 power 更平滑且 robust
                x = loss_val.pow(self.focal_gamma)
                loss = torch.log(torch.cosh(x))
            else:
                loss = torch.log(torch.cosh(loss_val))

            losses.append(loss)

        return torch.stack(losses).mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class SizeAwareTverskyLossV2(nn.Module):
    """
    Size-aware Tversky / Focal Tversky with per-class instance weighting + optional Boundary(SDF) term.
    目標：在提升小病灶召回的同時穩定地拉升 Dice。

    主要差異 vs 原版：
      (A) per-class 權重 W_c（不再用 max 聚合），避免跨類加權干擾
      (B) 權重裁剪 w_min/w_max + 可選的 'pos' 或 'global' 歸一
      (C) 可選的 Boundary/SDF 子項 (lambda_boundary>0)，改善輪廓/HD，帶動 Dice
      (D) 參數排程 support：set_progress(p∈[0,1]) 動態調整 alpha/beta/focal_gamma/size_gamma/bg_weight/lambda_boundary

    文獻依據：
      - Focal Tversky 提升小結構召回/精度取捨 (Abraham & Khan, ISBI 2019)  [focal_gamma]
      - Boundary loss / SDF 對高度不平衡分割穩定與 HD/Dice 改善 (Kervadec+ 2019; Karimi+ 2019)
      - 連通元件尺寸反比 reweighting 提升小病灶偵測 (Shirokikh+ 2020)
    """

    def __init__(self,
                 alpha: float = 0.3,            # 建議：FN 懲罰較重 ⇒ beta > alpha
                 beta: float = 0.7,
                 focal_gamma: float = 0.5,      # 0=關; 0.5~1.0 前期抓小灶
                 size_gamma: float = 0.7,       # 0.5~1.0：越大越偏小灶
                 bg_weight: float = 0.8,        # <1 降低 FP 懲罰，利於早期召回
                 conn: int = 1,                 # 6/18/26-conn，視庫支持
                 normalize_mode: str = "pos",   # {"pos","global","none"}
                 w_min: float = 0.1,            # 避免極端爆炸/消失
                 w_max: float = 1e4,
                 smooth: float = 1e-6,
                 lambda_boundary: float = 0.0,  # >0 啟用 SDF 邊界項
                 sdf_truncate: float = 20.0,    # SDF 截斷，穩定梯度
                 use_cc3d_first: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_gamma = focal_gamma
        self.size_gamma = size_gamma
        self.bg_weight = bg_weight
        self.conn = conn
        self.normalize_mode = normalize_mode
        self.w_min = w_min
        self.w_max = w_max
        self.smooth = smooth
        self.lambda_boundary = lambda_boundary
        self.sdf_truncate = sdf_truncate
        self.use_cc3d_first = use_cc3d_first

        # deps flags
        self._has_cc3d = False
        self._has_skimage = False
        self._has_scipy = False
        try:
            import cc3d  # noqa: F401
            self._has_cc3d = True
        except Exception:
            pass
        try:
            from skimage.measure import label as _label  # noqa: F401
            self._has_skimage = True
        except Exception:
            pass
        try:
            from scipy.ndimage import label as _slabel, distance_transform_edt as _dt  # noqa: F401
            self._has_scipy = True
        except Exception:
            pass

        self._progress = None  # for curriculum scheduling

    # ===== utilities =====
    def set_progress(self, p: float):
        """
        p in [0,1]: 建議訓練時每個 epoch 更新，用於課程式排程。
        這裡給一個合理的預設策略；你也可以在外部自訂並直接 set。
        """
        self._progress = float(max(0.0, min(1.0, p)))
        # 前期（p≈0）偏召回：FN 懲罰↑、focal↑、size_gamma↑、bg_weight↓、boundary弱
        # 後期（p≈1）回到平衡：FN 懲罰回中、focal關、size_gamma略降、bg_weight=1、boundary↑
        self.beta = 0.7 * (1 - p) + 0.5 * p
        self.alpha = 1.0 - self.beta
        self.focal_gamma = 0.8 * (1 - p) + 0.0 * p
        self.size_gamma = 0.8 * (1 - p) + 0.5 * p
        self.bg_weight = 0.8 * (1 - p) + 1.0 * p
        self.lambda_boundary = (0.1 * (1 - p) + 0.5 * p) if self.lambda_boundary is not None else 0.0

    def _to_one_hot(self, input, target):
        if target.shape[1] == 1:
            num_classes = input.shape[1] if input.shape[1] > 1 else 2
            tgt = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
            tgt = tgt.permute(0, 4, 1, 2, 3).contiguous()
        else:
            tgt = target
        return tgt

    @torch.no_grad()
    def _cc_weight_map_one(self, mask_np, size_gamma, conn):
        """
        對單一 0/1 mask (numpy, D*H*W) 計算 CC 權重圖（正類內：s^{-size_gamma}）。
        依序嘗試 cc3d -> skimage -> scipy，皆不可用時回退密度估計。
        """
        import numpy as np
        if mask_np.sum() == 0:
            return np.zeros_like(mask_np, dtype=np.float32)

        labels = None
        if self.use_cc3d_first and self._has_cc3d:
            import cc3d
            # cc3d connectivity: 6/18/26
            labels = cc3d.connected_components(mask_np.astype(np.uint8), connectivity={1:6,2:18,3:26}.get(conn,6))
        elif self._has_skimage:
            from skimage.measure import label as sk_label
            labels = sk_label(mask_np, connectivity=conn)
        elif self._has_scipy:
            from scipy.ndimage import label as sp_label
            labels, _ = sp_label(mask_np)
        else:
            # fallback: 局部密度近似
            import torch
            k = 7
            pad = k // 2
            density = F.avg_pool3d(torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0),
                                   kernel_size=k, stride=1, padding=pad).squeeze().numpy()
            eps = 1e-6
            w = (density + eps) ** (-size_gamma)
            w[mask_np == 0] = 0.0
            return w.astype(np.float32)

        import numpy as np
        uniq, counts = np.unique(labels, return_counts=True)
        if uniq[0] == 0:
            uniq, counts = uniq[1:], counts[1:]
        if len(uniq) == 0:
            return np.zeros_like(mask_np, dtype=np.float32)

        eps = 1e-6
        comp_w = (counts.astype(np.float64) + eps) ** (-size_gamma)
        # 查表
        lut = np.zeros(labels.max() + 1, dtype=np.float64)
        lut[uniq] = comp_w
        w = lut[labels]
        w[labels == 0] = 0.0
        return w.astype(np.float32)

    @torch.no_grad()
    def _build_W_per_class(self, tgt_1h: torch.Tensor):
        """
        為每個前景類別 c=1..C-1 計算 W_c；背景不在此處計算。
        回傳 W_pos: [B,C,D,H,W]，其中 C 與 tgt_1h 相同；背景通道=0。
        """
        B, C, D, H, W = tgt_1h.shape
        device, dtype = tgt_1h.device, tgt_1h.dtype
        W_pos = torch.zeros_like(tgt_1h, dtype=dtype, device=device)

        # 逐 batch 逐類別（實務上可 cache/預算）
        for b in range(B):
            for c in range(1, C):
                m = (tgt_1h[b, c] > 0.5).detach().to('cpu', torch.uint8).numpy()
                w_np = self._cc_weight_map_one(m, self.size_gamma, self.conn)
                # 歸一 + 裁剪
                if self.normalize_mode in ("pos", "global"):
                    pos_cnt = float(m.sum())
                    if pos_cnt > 0 and w_np.sum() > 0:
                        scale = pos_cnt / (w_np.sum() + 1e-6) if self.normalize_mode == "pos" else 1.0
                        w_np = w_np * scale
                w_np = w_np.clip(self.w_min, self.w_max, out=w_np, where=(w_np>0))
                W_pos[b, c] = torch.from_numpy(w_np).to(device=device, dtype=dtype)

        # 若 global 歸一：使整張圖平均權重 ≈ 1（含背景稍後混入再校正）
        if self.normalize_mode == "global":
            # 先只對正類做一次簡單 global normalize
            mean_pos = W_pos[:, 1:].mean()
            if mean_pos > 0:
                W_pos = W_pos / (mean_pos + 1e-6)
        return W_pos

    @torch.no_grad()
    def _signed_distance_map(self, bin_mask: torch.Tensor) -> torch.Tensor:
        """
        快速 CPU SDF：外部為 +，內部為 -；支援 3D。
        使用 scipy distance_transform_edt；若無 scipy，回退為 0（關閉 boundary 項）。
        """
        if not self._has_scipy:
            return torch.zeros_like(bin_mask, dtype=torch.float32)
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt

        sdf_list = []
        bin_np = bin_mask.detach().to('cpu', torch.uint8).numpy()
        for b in range(bin_np.shape[0]):
            m = bin_np[b, 0]
            if m.sum() == 0:
                sdf_list.append(np.zeros_like(m, dtype=np.float32))
                continue
            # 外部距離
            outside = edt(1 - m)
            # 內部距離
            inside = edt(m)
            sdf = outside.astype(np.float32)
            sdf[m > 0] = -inside[m > 0].astype(np.float32)
            # 截斷避免遠距離梯度過大
            if self.sdf_truncate is not None:
                sdf = np.clip(sdf, -self.sdf_truncate, self.sdf_truncate)
            sdf_list.append(sdf)
        sdf = torch.from_numpy(np.stack(sdf_list)).unsqueeze(1)  # [B,1,D,H,W]
        return sdf.to(bin_mask.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input: logits [B,C,D,H,W] (C=1 => binary; C>1 => multi-class with background at 0)
        target: [B,1,D,H,W] (int) or [B,C,D,H,W] (one-hot)
        """
        B, C = input.shape[:2]
        tgt_1h = self._to_one_hot(input, target)  # [B,C,D,H,W]

        # probabilities & foreground/background masks
        if C == 1:
            prob = torch.sigmoid(input)
            fg_1h = tgt_1h if tgt_1h.shape[1] == 1 else tgt_1h[:, 1:2]
            bg_1h = 1.0 - fg_1h
        else:
            prob = F.softmax(input, dim=1)
            fg_1h = tgt_1h[:, 1:, ...]                  # [B,C-1,D,H,W]
            bg_1h = 1.0 - fg_1h.sum(dim=1, keepdim=True)
            bg_1h = bg_1h.clamp(0, 1)

        # per-class positive weights
        if C == 1:
            W_pos = self._build_W_per_class(torch.cat([1.0 - fg_1h, fg_1h], dim=1))  # 兩通道: bg,fg
        else:
            W_pos = self._build_W_per_class(tgt_1h)  # [B,C,D,H,W]; 背景通道=0

        # 混入背景權重（常數）
        W = W_pos.clone()
        W[:, 0:1, ...] = self.bg_weight * bg_1h

        # valid_mask（忽略區）處理：所有通道同樣遮罩
        if valid_mask is not None:
            W = W * valid_mask.float()

        # ===== Tversky / Focal-Tversky 主項 =====
        losses = []
        if C == 1:
            p = prob
            t = tgt_1h if tgt_1h.shape[1] == 1 else tgt_1h[:, 1:2]
            # 僅使用對應通道的權重：前景用 W_pos 的 fg 通道，背景用 bg 通道
            W_fg = W[:, -1:, ...]
            W_bg = W[:, 0:1, ...]
            TP = (W_fg * p * t).sum()
            FP = (W_bg * p * (1 - t)).sum()
            FN = (W_fg * (1 - p) * t).sum()
            TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            main_loss = (1.0 - TI)
            if self.focal_gamma > 0:
                main_loss = main_loss.pow(self.focal_gamma)
            losses.append(main_loss)
        else:
            # 多類：對每個前景類別 c 分開計算（使用各自 W_c 與 bg 通道）
            W_bg = W[:, 0:1, ...]
            for c in range(1, C):
                p = prob[:, c:c+1, ...]
                t = tgt_1h[:, c:c+1, ...]
                W_fg = W[:, c:c+1, ...]
                TP = (W_fg * p * t).sum()
                FP = (W_bg * p * (1 - t)).sum()
                FN = (W_fg * (1 - p) * t).sum()
                TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
                loss_c = (1.0 - TI)
                if self.focal_gamma > 0:
                    loss_c = loss_c.pow(self.focal_gamma)
                losses.append(loss_c)

        total_loss = torch.stack(losses).mean()

        # ===== Optional: Boundary/SDF 項 =====
        if self.lambda_boundary and self.lambda_boundary > 0:
            # 依 Kervadec：用 GT 的 signed distance map φ_G 與 soft prob 的內積（多類求和）
            # 內部(φ<0)鼓勵 p→1，外部(φ>0)鼓勵 p→0
            if C == 1:
                sdf = self._signed_distance_map((tgt_1h if tgt_1h.shape[1]==1 else tgt_1h[:,1:2]).float())
                boundary_term = (prob * sdf).mean()
            else:
                boundary_list = []
                for c in range(1, C):
                    sdf_c = self._signed_distance_map(tgt_1h[:, c:c+1, ...].float())
                    boundary_list.append((prob[:, c:c+1, ...] * sdf_c).mean())
                boundary_term = torch.stack(boundary_list).mean()
            # 最小化時希望 inside(-)×p + outside(+)×p 變小 ⇒ 加負號
            total_loss = total_loss + self.lambda_boundary * boundary_term

        return total_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class SizeAwareTverskyLossV3(nn.Module):
    """
    Size-aware (per-class) Tversky / Focal Tversky with optional Boundary(SDF).
    強化點：
      - 逐樣本 Tversky（先在 D/H/W 上求和，再對 batch 取平均）
      - 逐樣本 normalize（避免 batch 間耦合）
      - 前景↔前景 的 FP 可降權：fp_in_other_fg_scale
      - SDF 正規化（截斷後再除以截斷值），更穩定
      - AMP/bfloat16 安全：累加時升至 float32
      - 可傳入 precomputed_W_pos，或要求回傳 weight_map 供外部 CE 共用
      - reduction 支援: {"mean","batch","none"}
    """
    def __init__(self,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 focal_gamma: float = 0.5,
                 size_gamma: float = 0.8,
                 bg_weight: float = 0.8,
                 conn: int = 1,
                 normalize_mode: str = "pos",   # {"pos","none"}
                 w_min: float = 0.1,
                 w_max: float = 1e4,
                 smooth: float = 1e-6,
                 lambda_boundary: float = 0.0,
                 sdf_truncate: float = 20.0,
                 use_cc3d_first: bool = True,
                 fp_in_other_fg_scale: float = 0.5,  # 多類時，其他前景上的 FP 降權（0.3~0.7）
                 reduction: str = "mean",            # {"mean","batch","none"}
                 return_weight_for_ce: bool = False  # True 時 forward 回傳 (loss, W_ce)
                 ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0
        self.alpha = alpha
        self.beta = beta
        self.focal_gamma = focal_gamma
        self.size_gamma = size_gamma
        self.bg_weight = bg_weight
        self.conn = conn
        self.normalize_mode = normalize_mode
        self.w_min = w_min
        self.w_max = w_max
        self.smooth = smooth
        self.lambda_boundary = lambda_boundary
        self.sdf_truncate = sdf_truncate
        self.use_cc3d_first = use_cc3d_first
        self.fp_in_other_fg_scale = fp_in_other_fg_scale
        self.reduction = reduction
        self.return_weight_for_ce = return_weight_for_ce

        # deps flags
        self._has_cc3d = False
        self._has_skimage = False
        self._has_scipy = False
        try:
            import cc3d  # noqa
            self._has_cc3d = True
        except Exception:
            pass
        try:
            from skimage.measure import label as _label  # noqa
            self._has_skimage = True
        except Exception:
            pass
        try:
            from scipy.ndimage import label as _slabel, distance_transform_edt as _dt  # noqa
            self._has_scipy = True
        except Exception:
            pass

        self._progress = None

    # ---------- public helpers ----------
    def set_progress(self, p: float):
        p = float(max(0.0, min(1.0, p)))
        self._progress = p
        # 課程式：前期抓小灶、後期貼邊界
        self.beta = 0.7 * (1 - p) + 0.5 * p
        self.alpha = 1.0 - self.beta
        self.focal_gamma = 0.8 * (1 - p) + 0.0 * p
        self.size_gamma = 0.8 * (1 - p) + 0.5 * p
        self.bg_weight = 0.8 * (1 - p) + 1.0 * p
        self.lambda_boundary = (0.1 * (1 - p) + 0.5 * p) if self.lambda_boundary is not None else 0.0

    def _to_one_hot(self, input, target):
        if target.shape[1] == 1:
            num_classes = input.shape[1] if input.shape[1] > 1 else 2
            tgt = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
            tgt = tgt.permute(0, 4, 1, 2, 3).contiguous()
        else:
            tgt = target
        return tgt

    @torch.no_grad()
    def _cc_weight_map_one(self, mask_np, size_gamma, conn):
        import numpy as np
        if mask_np.sum() == 0:
            return np.zeros_like(mask_np, dtype=np.float32)
        labels = None
        if self.use_cc3d_first and self._has_cc3d:
            import cc3d
            labels = cc3d.connected_components(mask_np.astype(np.uint8), connectivity={1:6,2:18,3:26}.get(conn,6))
        elif self._has_skimage:
            from skimage.measure import label as sk_label
            labels = sk_label(mask_np, connectivity=conn)
        elif self._has_scipy:
            from scipy.ndimage import label as sp_label
            labels, _ = sp_label(mask_np)
        else:
            # fallback: 局部密度近似
            import torch
            k = 7; pad = k // 2
            density = F.avg_pool3d(torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0),
                                   kernel_size=k, stride=1, padding=pad).squeeze().numpy()
            eps = 1e-6
            w = (density + eps) ** (-size_gamma)
            w[mask_np == 0] = 0.0
            return w.astype(np.float32)

        import numpy as np
        uniq, counts = np.unique(labels, return_counts=True)
        if uniq[0] == 0:
            uniq, counts = uniq[1:], counts[1:]
        if len(uniq) == 0:
            return np.zeros_like(mask_np, dtype=np.float32)
        eps = 1e-6
        comp_w = (counts.astype(np.float64) + eps) ** (-size_gamma)
        lut = np.zeros(labels.max() + 1, dtype=np.float64)
        lut[uniq] = comp_w
        w = lut[labels]
        w[labels == 0] = 0.0
        return w.astype(np.float32)

    @torch.no_grad()
    def _build_W_per_class(self, tgt_1h: torch.Tensor):
        """
        逐樣本、逐類別建立正類權重；逐樣本 normalize（pos 模式）。
        回傳 W_pos: [B,C,D,H,W]，背景通道=0。
        """
        B, C, D, H, W = tgt_1h.shape
        device, dtype = tgt_1h.device, tgt_1h.dtype
        W_pos = torch.zeros_like(tgt_1h, dtype=dtype, device=device)
        for b in range(B):
            for c in range(1, C):
                m = (tgt_1h[b, c] > 0.5).detach().to('cpu', torch.uint8).numpy()
                w_np = self._cc_weight_map_one(m, self.size_gamma, self.conn)
                # 逐樣本正類 normalize
                if self.normalize_mode == "pos":
                    pos_cnt = float(m.sum())
                    if pos_cnt > 0 and w_np.sum() > 0:
                        w_np *= (pos_cnt / (w_np.sum() + 1e-6))
                # 只裁剪 >0 的正類
                w_np = w_np.clip(self.w_min, self.w_max, out=w_np, where=(w_np > 0))
                W_pos[b, c] = torch.from_numpy(w_np).to(device=device, dtype=dtype)
        return W_pos

    @torch.no_grad()
    def _signed_distance_map(self, bin_mask: torch.Tensor) -> torch.Tensor:
        # phi < 0 inside, phi > 0 outside；截斷後再 /截斷值 做標準化到 [-1,1]
        if not self._has_scipy:
            return torch.zeros_like(bin_mask, dtype=torch.float32)
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt
        sdf_list = []
        bin_np = bin_mask.detach().to('cpu', torch.uint8).numpy()
        for b in range(bin_np.shape[0]):
            m = bin_np[b, 0]
            if m.sum() == 0:
                sdf_list.append(np.zeros_like(m, dtype=np.float32)); continue
            outside = edt(1 - m).astype(np.float32)
            inside  = edt(m).astype(np.float32)
            sdf = outside
            sdf[m > 0] = -inside[m > 0]
            T = float(self.sdf_truncate) if self.sdf_truncate is not None else None
            if T is not None and T > 0:
                sdf = np.clip(sdf, -T, T) / T
            sdf_list.append(sdf.astype(np.float32))
        sdf = torch.from_numpy(np.stack(sdf_list)).unsqueeze(1)
        return sdf.to(bin_mask.device)

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                valid_mask: torch.Tensor = None,
                precomputed_W_pos: torch.Tensor = None  # 可傳 [B,C,D,H,W]，背景通道=0
                ):
        B, C = input.shape[:2]
        tgt_1h = self._to_one_hot(input, target)  # [B,C,D,H,W]

        # probs
        if C == 1:
            prob = torch.sigmoid(input)
            fg_1h = tgt_1h if tgt_1h.shape[1] == 1 else tgt_1h[:, 1:2]
            bg_1h = 1.0 - fg_1h
        else:
            prob = F.softmax(input, dim=1)
            fg_1h = tgt_1h[:, 1:, ...]
            bg_1h = (1.0 - fg_1h.sum(dim=1, keepdim=True)).clamp(0, 1)

        # weights
        if precomputed_W_pos is not None:
            W_pos = precomputed_W_pos.to(input.dtype).to(input.device)
        else:
            if C == 1:
                W_pos = self._build_W_per_class(torch.cat([1.0 - fg_1h, fg_1h], dim=1))
            else:
                W_pos = self._build_W_per_class(tgt_1h)  # 背景通道=0

        W = W_pos.clone()
        W[:, 0:1, ...] = self.bg_weight * bg_1h
        if valid_mask is not None:
            W = W * valid_mask.float()

        # ==== 逐樣本 Tversky（float32 累加）====
        reduce_dims = tuple(range(2, input.dim()))  # D,H,W
        dtype_sum = torch.float32
        losses = []

        if C == 1:
            p = prob.to(dtype_sum); t = (tgt_1h if tgt_1h.shape[1]==1 else tgt_1h[:,1:2]).to(dtype_sum)
            W_fg = W[:, -1:, ...].to(dtype_sum)
            W_bg = W[:,  0:1, ...].to(dtype_sum)

            TP = (W_fg * p * t).sum(dim=reduce_dims)
            FP = (W_bg * p * (1 - t)).sum(dim=reduce_dims)
            FN = (W_fg * (1 - p) * t).sum(dim=reduce_dims)

            TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            loss = (1.0 - TI)
            if self.focal_gamma > 0:
                loss = loss.pow(self.focal_gamma)
            losses.append(loss)  # [B,1] or [B]
        else:
            W_bg = W[:, 0:1, ...].to(dtype_sum)
            # 其他前景區域（非該類）用於 FP 降權
            other_fg_all = fg_1h.sum(dim=1, keepdim=True).to(dtype_sum)

            for c in range(1, C):
                p = prob[:, c:c+1, ...].to(dtype_sum)
                t = tgt_1h[:, c:c+1, ...].to(dtype_sum)
                W_fg = W[:, c:c+1, ...].to(dtype_sum)

                # FP 分解：背景上的 FP 與「其他前景上的 FP」
                other_fg = (other_fg_all - t).clamp(min=0.0)  # 該類以外的前景
                fp_on_bg   = (1 - t) * (1 - other_fg) * p
                fp_on_ofg  = (1 - t) * (other_fg) * p

                TP = (W_fg * p * t).sum(dim=reduce_dims)
                FP = (W_bg * fp_on_bg).sum(dim=reduce_dims) \
                   + (self.fp_in_other_fg_scale * W_bg * fp_on_ofg).sum(dim=reduce_dims)
                FN = (W_fg * (1 - p) * t).sum(dim=reduce_dims)

                TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
                loss_c = (1.0 - TI)
                if self.focal_gamma > 0:
                    loss_c = loss_c.pow(self.focal_gamma)
                losses.append(loss_c)  # [B,1]

        # [num_terms,B,1] -> [B] 平均每樣本，再依 reduction
        loss_per_term = torch.stack(losses, dim=0).squeeze(-1)  # [T,B]
        loss_per_sample = loss_per_term.mean(dim=0)             # [B]
        main_loss = (loss_per_sample.mean() if self.reduction == "mean"
                     else loss_per_sample if self.reduction == "batch"
                     else loss_per_term)

        # ==== Boundary/SDF 項（同樣逐樣本）====
        if self.lambda_boundary and self.lambda_boundary > 0:
            if C == 1:
                sdf = self._signed_distance_map((tgt_1h if tgt_1h.shape[1]==1 else tgt_1h[:,1:2]).float())
                boundary = (prob[:, -1:, ...].to(dtype_sum) * sdf.to(dtype_sum)).mean(dim=reduce_dims).squeeze(-1)
            else:
                blist = []
                for c in range(1, C):
                    sdf_c = self._signed_distance_map(tgt_1h[:, c:c+1, ...].float())
                    blist.append((prob[:, c:c+1, ...].to(dtype_sum) * sdf_c.to(dtype_sum)).mean(dim=reduce_dims).squeeze(-1))
                boundary = torch.stack(blist, dim=0).mean(dim=0)  # [B]
            # 注意：這裡「加號」是正確的號誌（最小化 p×phi）
            boundary_loss = boundary.mean() if self.reduction == "mean" else boundary
            main_loss = main_loss + self.lambda_boundary * boundary_loss

        if self.return_weight_for_ce:
            # 回傳一張供 CE 使用的 scalar 權重圖（把 per-class 正類權重與背景權重疊回到單通道）
            # 做法：取每位置對應 GT 類別的 W_fg，否則用 W_bg（多類亦可）
            with torch.no_grad():
                if C == 1:
                    W_ce = torch.where((tgt_1h if tgt_1h.shape[1]==1 else tgt_1h[:,1:2]).bool(),
                                       W[:, -1:, ...], W[:, 0:1, ...]).squeeze(1)
                else:
                    # one-hot argmax 避免浮點誤差
                    y = tgt_1h.argmax(dim=1, keepdim=True)  # [B,1,D,H,W]
                    gatherW = torch.gather(W, 1, y)        # [B,1,D,H,W]
                    W_ce = gatherW.squeeze(1)
            return main_loss, W_ce
        return main_loss
