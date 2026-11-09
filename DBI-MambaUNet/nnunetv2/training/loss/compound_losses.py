import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss, FocalDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss, TverskyLoss
from nnunetv2.training.loss.Focal import FocalLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import torch.nn.functional as F
from nnunetv2.training.loss.FocalTversky import FocalTverskyLoss
from nnunetv2.training.loss.robust_ce_loss import SizeAwareTverskyLoss, SizeAwareTverskyLossV2, SizeAwareTverskyLossV3, SizeAwareHaloTverskyLoss

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result



class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=SoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_B_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label: bool = False,
                 ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = FocalDiceLoss(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class FocalDC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label: bool = False,
                 ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = FocalDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result



class DC_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs):
        super(DC_and_Focal_loss, self).__init__()
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.focal = FocalLoss(apply_nonlin=softmax_helper_dim1, **focal_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        focal_loss = self.focal(net_output, target)

        result = dc_loss + focal_loss
        return result


class DC_and_SizeAwareTversky_and_CE_loss(nn.Module):
    """
    結合 Dice Loss (DC)、Focal Tversky Loss 與 Robust Cross Entropy Loss 的損失函數，
    同時兼顧分割區域的邊界與病灶級別表現。

    :param soft_dice_kwargs: 傳入 Dice Loss (例如 SoftDiceLoss) 的關鍵字參數
    :param ce_kwargs: 傳入 RobustCrossEntropyLoss 的關鍵字參數
    :param focal_tversky_kwargs: 傳入 FocalTverskyLoss 的關鍵字參數
    :param weight_ce: Cross Entropy 的權重 (預設 1)
    :param weight_dice: Dice Loss 的權重 (預設 1)
    :param weight_ft: Focal Tversky Loss 的權重 (預設 1)
    :param ignore_label: 忽略的標籤 (若有)
    :param dice_class: 使用的 Dice Loss 類別 (預設 SoftDiceLoss)
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, focal_tversky_kwargs,
                 weight_ce=1, weight_dice=1, weight_ft=1, ignore_label=None, dice_class=None):
        super(DC_and_SizeAwareTversky_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        # 若未指定，預設使用 SoftDiceLoss
        if dice_class is None:
            from nnunetv2.training.loss.dice import SoftDiceLoss
            dice_class = SoftDiceLoss

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_ft = weight_ft
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # 轉換 kwargs：兼容舊的 'gamma' 命名；忽略 apply_nonlin
        ft_kwargs = dict(focal_tversky_kwargs) if focal_tversky_kwargs is not None else {}
        if 'gamma' in ft_kwargs and 'focal_gamma' not in ft_kwargs:
            ft_kwargs['focal_gamma'] = ft_kwargs.pop('gamma')
        ft_kwargs.pop('apply_nonlin', None)

        self.ft = SizeAwareTverskyLoss(**ft_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        :param net_output: 預測結果，形狀為 [B, C, ...]
        :param target: ground truth，形狀為 [B, 1, ...]
        :return: 加權後的總損失
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "ignore label 不支援 one hot encoded target (DC_and_FocalTversky_and_CE_loss)"
            mask = (target != self.ignore_label).bool()
            target_mod = torch.clone(target)
            target_mod[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_mod = target
            mask = None

        dc_loss = self.dc(net_output, target_mod) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # FT：忽略標籤區域不計分
        if self.weight_ft != 0:
            if self.ignore_label is not None:
                valid_mask = (target != self.ignore_label)  # [B,1,D,H,W] bool
                target_ft = target.clone()
                target_ft[~valid_mask] = 0  # 忽略區設為背景，以便建立 CC；真正忽略靠 valid_mask
                ft_loss = self.ft(net_output, target_ft, valid_mask=valid_mask)
            else:
                ft_loss = self.ft(net_output, target, valid_mask=None)
        else:
            ft_loss = 0.0
        result = self.weight_dice * dc_loss + self.weight_ce * ce_loss + self.weight_ft * ft_loss
        return result



class SizeAwareTversky_and_CE_loss(nn.Module):
    """
    用尺寸自適應 Tversky/Asymmetric Focal Tversky 取代原本的 FocalTverskyLoss
    - focal_tversky_kwargs 支援 {alpha, beta, gamma(→focal_gamma), size_gamma, bg_weight, conn, normalize_pos, smooth}
    - ce_kwargs 傳給 RobustCrossEntropyLoss（可含 ignore_index）
    """
    def __init__(self, focal_tversky_kwargs, ce_kwargs, weight_ce=1, weight_ft=1, ignore_label=None):
        super(SizeAwareTversky_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_ce = weight_ce
        self.weight_ft = weight_ft
        self.ignore_label = ignore_label

        # 這兩個需由你現有專案提供
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        # 轉換 kwargs：兼容舊的 'gamma' 命名；忽略 apply_nonlin
        ft_kwargs = dict(focal_tversky_kwargs) if focal_tversky_kwargs is not None else {}
        if 'gamma' in ft_kwargs and 'focal_gamma' not in ft_kwargs:
            ft_kwargs['focal_gamma'] = ft_kwargs.pop('gamma')
        ft_kwargs.pop('apply_nonlin', None)

        self.ft = SizeAwareTverskyLoss(**ft_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        net_output: [B, C, D, H, W] logits
        target    : [B, 1, D, H, W] 整數標籤
        """
        # CE （支援 ignore_index）
        if self.weight_ce != 0:
            ce_loss = self.ce(net_output, target[:, 0].long())
        else:
            ce_loss = 0.0

        # FT：忽略標籤區域不計分
        if self.weight_ft != 0:
            if self.ignore_label is not None:
                valid_mask = (target != self.ignore_label)  # [B,1,D,H,W] bool
                target_ft = target.clone()
                target_ft[~valid_mask] = 0  # 忽略區設為背景，以便建立 CC；真正忽略靠 valid_mask
                ft_loss = self.ft(net_output, target_ft, valid_mask=valid_mask)
            else:
                ft_loss = self.ft(net_output, target, valid_mask=None)
        else:
            ft_loss = 0.0

        return self.weight_ce * ce_loss + self.weight_ft * ft_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

# 假設你專案內已有：
# from your_project.losses import RobustCrossEntropyLoss
# from your_project.losses import SizeAwareTverskyLossV2, SizeAwareTverskyLossV3

class SizeAwareTverskyV2_and_CE_loss(nn.Module):
    """
    CE + SizeAwareTverskyLossV2 的組合包裝
    - focal_tversky_kwargs 會自動映射舊參數：gamma→focal_gamma；normalize_pos→normalize_mode
      * 若 normalize_pos=True → normalize_mode="pos"
      * 若 normalize_pos=False → normalize_mode="none"
    - ce_kwargs 直接傳給 RobustCrossEntropyLoss（會補上 ignore_index）
    """
    def __init__(self, focal_tversky_kwargs, ce_kwargs,
                 weight_ce: float = 1.0,
                 weight_ft: float = 1.0,
                 ignore_label: int | None = None):
        super().__init__()
        self.weight_ce = float(weight_ce)
        self.weight_ft = float(weight_ft)
        self.ignore_label = ignore_label

        # ---- CE ----
        ce_kwargs = dict(ce_kwargs or {})
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        # ---- FT-V2 ----
        ft_kwargs = dict(focal_tversky_kwargs or {})
        # 兼容舊命名
        if 'gamma' in ft_kwargs and 'focal_gamma' not in ft_kwargs:
            ft_kwargs['focal_gamma'] = ft_kwargs.pop('gamma')
        # normalize_pos (bool) -> normalize_mode
        if 'normalize_pos' in ft_kwargs and 'normalize_mode' not in ft_kwargs:
            normalize_pos = bool(ft_kwargs.pop('normalize_pos'))
            ft_kwargs['normalize_mode'] = 'pos' if normalize_pos else 'none'
        # 移除不相干的鍵
        ft_kwargs.pop('apply_nonlin', None)

        self.ft = SizeAwareTverskyLossV2(**ft_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        net_output: [B, C, D, H, W] logits
        target    : [B, 1, D, H, W] 整數標籤
        """
        total = 0.0

        # ---- Cross-Entropy ----
        if self.weight_ce != 0.0:
            ce_loss = self.ce(net_output, target[:, 0].long())
            total = total + self.weight_ce * ce_loss

        # ---- SizeAwareTversky V2 ----
        if self.weight_ft != 0.0:
            if self.ignore_label is not None:
                valid_mask = (target != self.ignore_label)  # [B,1,D,H,W] bool
                target_ft = target.clone()
                target_ft[~valid_mask] = 0  # 忽略區設背景，真正忽略靠 valid_mask
                ft_loss = self.ft(net_output, target_ft, valid_mask=valid_mask)
            else:
                ft_loss = self.ft(net_output, target, valid_mask=None)
            total = total + self.weight_ft * ft_loss

        return total


import torch
import torch.nn as nn

# 假設你已把新版 FocalTverskyLoss 放在可匯入的位置
# from your_loss_lib import FocalTverskyLoss
# from your_loss_lib import RobustCrossEntropyLoss

class FocalTversky_and_CE_loss(nn.Module):
    """
    結合 Focal Tversky Loss 與 Cross Entropy 的損失

    Args:
        focal_tversky_kwargs (dict): 傳給新版 FocalTverskyLoss 的參數（不含 apply_nonlin）
            常用鍵：alpha, beta, gamma, smooth/eps, include_background(False), reduction('mean'), per_image(True)
        ce_kwargs (dict): 傳給 RobustCrossEntropyLoss 的參數（例如 reduction='mean'）
        weight_ce (float): CE 權重
        weight_ft (float): FT 權重
        ignore_label (int|None): 要忽略的標籤值（同時套用在 CE 與 FT）
        include_background (bool): 多類時 FTL 是否納入背景（預設 False 較穩）
    """
    def __init__(self,
                 focal_tversky_kwargs: dict | None = None,
                 ce_kwargs: dict | None = None,
                 weight_ce: float = 1.0,
                 weight_ft: float = 1.0,
                 ignore_label: int | None = None,
                 include_background: bool = False):
        super().__init__()
        focal_tversky_kwargs = dict(focal_tversky_kwargs or {})
        ce_kwargs = dict(ce_kwargs or {})

        self.weight_ce = float(weight_ce)
        self.weight_ft = float(weight_ft)
        self.ignore_label = ignore_label

        # ---- FTL：新版 API，用 from_logits=True，不需要 apply_nonlin ----
        ft_defaults = dict(
            from_logits=True,
            include_background=include_background,
            ignore_index=ignore_label,
            reduction="mean",
            per_image=True,
        )
        ft_defaults.update(focal_tversky_kwargs)
        self.ft = FocalTverskyLoss(**ft_defaults)

        # ---- CE：把 ignore_index 套進去，reduction 預設 mean ----
        if ignore_label is not None:
            ce_kwargs.setdefault("ignore_index", ignore_label)
        ce_kwargs.setdefault("reduction", "mean")
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def _prepare_ce_target(self, target: torch.Tensor, C: int) -> torch.Tensor:
        """
        CE 需要整數標籤 [B, *]。容忍以下 target 形式：
        - [B, 1, *]  -> squeeze 成 [B, *]
        - [B, C, *] (one-hot) -> argmax 成 [B, *]
        - [B, *] -> 直接使用
        """
        if target.ndim >= 2 and target.shape[1] == 1:
            return target[:, 0].long()
        if target.ndim >= 2 and target.shape[1] == C:  # one-hot
            return target.argmax(dim=1).long()
        return target.long()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        net_output: [B, C, ...] logits
        target:     [B, 1, ...] (int) 或 [B, ...] (int) 或 [B, C, ...] (one-hot)
        """
        B, C = net_output.shape[:2]

        # ---- FTL：直接把原 target 丟進去（新版 FTL 會自己處理 one-hot/整數/ignore）----
        if self.weight_ft != 0:
            ft_loss = self.ft(net_output, target)
        else:
            ft_loss = net_output.new_tensor(0.0)

        # ---- CE：整理成整數標籤，處理整個 batch 皆 ignore 的情況 ----
        if self.weight_ce != 0:
            tgt_ce = self._prepare_ce_target(target, C)  # [B, *]
            if self.ignore_label is not None:
                valid_any = (tgt_ce != self.ignore_label).any()
                if not valid_any:
                    ce_loss = net_output.new_tensor(0.0)  # 全忽略就給 0，避免 NaN
                else:
                    ce_loss = self.ce(net_output, tgt_ce)
            else:
                ce_loss = self.ce(net_output, tgt_ce)
        else:
            ce_loss = net_output.new_tensor(0.0)

        return self.weight_ce * ce_loss + self.weight_ft * ft_loss

class DC_and_FocalTversky_and_CE_loss(nn.Module):
    """
    結合 Dice Loss (DC)、Focal Tversky Loss 與 Robust Cross Entropy Loss 的損失函數，
    同時兼顧分割區域的邊界與病灶級別表現。

    :param soft_dice_kwargs: 傳入 Dice Loss (例如 SoftDiceLoss) 的關鍵字參數
    :param ce_kwargs: 傳入 RobustCrossEntropyLoss 的關鍵字參數
    :param focal_tversky_kwargs: 傳入 FocalTverskyLoss 的關鍵字參數
    :param weight_ce: Cross Entropy 的權重 (預設 1)
    :param weight_dice: Dice Loss 的權重 (預設 1)
    :param weight_ft: Focal Tversky Loss 的權重 (預設 1)
    :param ignore_label: 忽略的標籤 (若有)
    :param dice_class: 使用的 Dice Loss 類別 (預設 SoftDiceLoss)
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, focal_tversky_kwargs,
                 weight_ce=1, weight_dice=1, weight_ft=1, ignore_label=None, dice_class=None):
        super(DC_and_FocalTversky_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        # 若未指定，預設使用 SoftDiceLoss
        if dice_class is None:
            from nnunetv2.training.loss.dice import SoftDiceLoss
            dice_class = SoftDiceLoss

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_ft = weight_ft
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ft = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **focal_tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        :param net_output: 預測結果，形狀為 [B, C, ...]
        :param target: ground truth，形狀為 [B, 1, ...]
        :return: 加權後的總損失
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "ignore label 不支援 one hot encoded target (DC_and_FocalTversky_and_CE_loss)"
            mask = (target != self.ignore_label).bool()
            target_mod = torch.clone(target)
            target_mod[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_mod = target
            mask = None

        dc_loss = self.dc(net_output, target_mod) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        ft_loss = self.ft(net_output, target_mod) if self.weight_ft != 0 else 0

        result = self.weight_dice * dc_loss + self.weight_ce * ce_loss + self.weight_ft * ft_loss
        return result
